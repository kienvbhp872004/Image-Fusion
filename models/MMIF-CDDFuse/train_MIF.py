"""
Paper-faithful training cho Medical Image Fusion (CDDFuse sect 5.2).

Bám train.py gốc:
- 120 epochs (40 Phase I + 80 Phase II)
- Adam LR 1e-4, StepLR γ=0.5 mỗi 20 ep
- 4 optimizers riêng: Encoder, Decoder, BaseFuseLayer, DetailFuseLayer
  (+ optional GatedB/GatedD optimizers cho Combined variant)
- Phase I: train Encoder + Decoder với recon + decomp CC + SSIM + TV gradient
- Phase II: train tất cả với fusion loss + decomp regularizer

Variants:
- CDDFuse (default): baseline paper (sum + max-target Fusionloss)
- Combined-Gated-Saliency: Gated fusion + Saliency target FusionLossB

Save:
- Checkpoint với keys paper convention + GatedB/GatedD nếu có
- train_history.json (loss per epoch, phase, lr, dt) cho thesis figure

Usage:
    python train_MIF.py                                    # baseline
    python train_MIF.py --variant Combined-Gated-Saliency  # combined
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import h5py
import kornia
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from net import (BaseFeatureExtraction, DetailFeatureExtraction,
                 Restormer_Decoder, Restormer_Encoder)
from utils.loss import Fusionloss, cc
from variants.losses import FusionLossB
from variants.modules import GatedFuseLayer


def linear_cka(x, y, eps=1e-6):
    """Linear CKA between feature tensors [N, C, H, W].

    CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y)), giá trị trong [0, 1].
    Captures *nonlinear* statistical dependence (paper's `cc()` chỉ linear).
    Module C variant: thay Pearson CC bằng CKA giữ nguyên decomp loss shape.

    AMP-safe: force fp32 + row-normalize để tránh K_x overflow ở epoch đầu.
    """
    with torch.amp.autocast('cuda', enabled=False):
        x = x.float()
        y = y.float()
        N = x.shape[0]
        x = x.reshape(N, -1)
        y = y.reshape(N, -1)
        # Center along batch dim
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        # Row-normalize → mỗi row có norm 1 → K_x bounded trong [-1, 1]
        x = x / (x.norm(dim=-1, keepdim=True) + eps)
        y = y / (y.norm(dim=-1, keepdim=True) + eps)
        K_x = x @ x.T          # [N, N], values ∈ [-1, 1]
        K_y = y @ y.T
        hsic_xy = (K_x * K_y).sum()
        hsic_xx = (K_x * K_x).sum()
        hsic_yy = (K_y * K_y).sum()
        return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy + eps) + eps)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ---------- dataset
class MIFH5Dataset(Dataset):
    def __init__(self, h5_path):
        self.path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.keys = list(f['mri_patchs'].keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as f:
            k = self.keys[idx]
            mri = np.array(f['mri_patchs'][k])
            src = np.array(f['src_patchs'][k])
        return torch.Tensor(src), torch.Tensor(mri)


# (use_gated, pixel_select, decomp_fn) — decomp_fn = "cc" (paper) or "cka" (Module C: nonlinear)
VARIANT_REGISTRY = {
    "CDDFuse":                       (False, "max",       "cc"),
    "Combined-Gated-Saliency":       (True,  "saliency",  "cc"),
    "Combined-Gated-Learnable":      (True,  "learnable", "cc"),
    "Combined-Gated-Saliency-CKA":   (True,  "saliency",  "cka"),  # Module C: CKA decomp
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="CDDFuse", choices=list(VARIANT_REGISTRY.keys()))
    parser.add_argument("--h5", default="data/MIF_train_imgsize_128_stride_64.h5")
    parser.add_argument("--output", default="models/")
    parser.add_argument("--num_epochs", type=int, default=120)
    parser.add_argument("--epoch_gap", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)   # paper text §5.1 (was 8 in code release)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    # Loss coefficients theo paper text §5.1: α1=1, α2=2, α3=10, α4=2
    parser.add_argument("--coeff_mse_VF", type=float, default=1.0)   # α1
    parser.add_argument("--coeff_mse_IF", type=float, default=1.0)
    parser.add_argument("--coeff_decomp", type=float, default=2.0)   # α2, α4
    parser.add_argument("--coeff_tv", type=float, default=10.0)      # α3 (paper text = 10, code release = 5)
    parser.add_argument("--clip_grad_norm", type=float, default=0.01)
    parser.add_argument("--optim_step", type=int, default=20)
    parser.add_argument("--optim_gamma", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gated, pixel_select, decomp_fn_name = VARIANT_REGISTRY[args.variant]
    decomp_fn = linear_cka if decomp_fn_name == "cka" else cc
    print(f"[train_MIF] device={device} variant={args.variant} "
          f"(use_gated={use_gated}, pixel_select={pixel_select}, decomp_fn={decomp_fn_name}) "
          f"epochs={args.num_epochs} (P1={args.epoch_gap}, P2={args.num_epochs - args.epoch_gap}) "
          f"batch={args.batch} amp={args.amp}")

    # Model (paper architecture; gated_b/d only for Combined)
    encoder = Restormer_Encoder().to(device)
    decoder = Restormer_Decoder().to(device)
    base_fuse = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)
    gated_b = GatedFuseLayer(64).to(device) if use_gated else None
    gated_d = GatedFuseLayer(64).to(device) if use_gated else None

    # Optimizers (4 paper + 2 if gated)
    opt_enc = torch.optim.Adam(encoder.parameters(),     lr=args.lr)
    opt_dec = torch.optim.Adam(decoder.parameters(),     lr=args.lr)
    opt_bf  = torch.optim.Adam(base_fuse.parameters(),   lr=args.lr)
    opt_df  = torch.optim.Adam(detail_fuse.parameters(), lr=args.lr)
    opts_p1 = [opt_enc, opt_dec]
    opts_p2 = [opt_enc, opt_dec, opt_bf, opt_df]
    if use_gated:
        opt_gb = torch.optim.Adam(gated_b.parameters(), lr=args.lr)
        opt_gd = torch.optim.Adam(gated_d.parameters(), lr=args.lr)
        opts_p2 += [opt_gb, opt_gd]
    all_opts = list({id(o): o for o in opts_p1 + opts_p2}.values())
    scheds = [torch.optim.lr_scheduler.StepLR(o, step_size=args.optim_step, gamma=args.optim_gamma)
              for o in all_opts]

    # Losses
    mse = nn.MSELoss(); l1 = nn.L1Loss()
    ssim_loss = (kornia.losses.SSIMLoss(11, reduction='mean')
                 if hasattr(kornia.losses, 'SSIMLoss')
                 else kornia.losses.SSIM(11, reduction='mean'))
    criteria_fusion = Fusionloss() if pixel_select == "max" else FusionLossB(pixel_select=pixel_select).to(device)

    # Data
    loader = DataLoader(MIFH5Dataset(args.h5), batch_size=args.batch, shuffle=True,
                        num_workers=0, pin_memory=(device == "cuda"))
    print(f"[data ] {len(loader.dataset)} patches, {len(loader)} batches/epoch")

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    ckpt_path = out_dir / f"CDDFuse-{args.variant}_MIF_{timestamp}.pth"
    history_path = out_dir / f"CDDFuse-{args.variant}_MIF_{timestamp}_train_history.json"

    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))
    if args.amp: print(f"[amp  ] fp16 enabled")

    history = []
    for epoch in range(args.num_epochs):
        encoder.train(); decoder.train(); base_fuse.train(); detail_fuse.train()
        if use_gated: gated_b.train(); gated_d.train()
        phase = 1 if epoch < args.epoch_gap else 2
        t0 = time.time()
        pbar = tqdm(loader, desc=f"ep{epoch+1:03d}/{args.num_epochs} P{phase}",
                    dynamic_ncols=True, leave=False)
        ep_total = ep_int = ep_grad = 0.0
        n_seen = 0

        for src, mri in pbar:
            src, mri = src.to(device), mri.to(device)
            for o in all_opts:
                o.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                if phase == 1:
                    f_V_B, f_V_D, _ = encoder(src)
                    f_I_B, f_I_D, _ = encoder(mri)
                    src_hat, _ = decoder(src, f_V_B, f_V_D)
                    mri_hat, _ = decoder(mri, f_I_B, f_I_D)

                    cc_B = decomp_fn(f_V_B, f_I_B); cc_D = decomp_fn(f_V_D, f_I_D)
                    mse_V = 5 * ssim_loss(src, src_hat) + mse(src, src_hat)
                    mse_I = 5 * ssim_loss(mri, mri_hat) + mse(mri, mri_hat)
                    grad_loss = l1(kornia.filters.SpatialGradient()(src),
                                   kornia.filters.SpatialGradient()(src_hat))
                    loss_decomp = (cc_D ** 2) / (1.01 + cc_B)
                    loss = (args.coeff_mse_VF * mse_V + args.coeff_mse_IF * mse_I
                            + args.coeff_decomp * loss_decomp + args.coeff_tv * grad_loss)
                    fusion_loss = torch.tensor(0.0, device=device)
                else:
                    f_V_B, f_V_D, _ = encoder(src)
                    f_I_B, f_I_D, _ = encoder(mri)
                    if use_gated:
                        f_F_B = base_fuse(gated_b(f_V_B, f_I_B))
                        f_F_D = detail_fuse(gated_d(f_V_D, f_I_D))
                    else:
                        f_F_B = base_fuse(f_V_B + f_I_B)
                        f_F_D = detail_fuse(f_V_D + f_I_D)
                    fused, _ = decoder(src, f_F_B, f_F_D)
                    cc_B = decomp_fn(f_V_B, f_I_B); cc_D = decomp_fn(f_V_D, f_I_D)
                    loss_decomp = (cc_D ** 2) / (1.01 + cc_B)
                    fusion_loss, l_int, l_grad = criteria_fusion(src, mri, fused)
                    loss = fusion_loss + args.coeff_decomp * loss_decomp

            scaler.scale(loss).backward()
            active_opts = opts_p1 if phase == 1 else opts_p2
            for o in active_opts: scaler.unscale_(o)
            mods_p1 = [encoder, decoder]
            mods_p2 = [encoder, decoder, base_fuse, detail_fuse] + ([gated_b, gated_d] if use_gated else [])
            for m in (mods_p1 if phase == 1 else mods_p2):
                nn.utils.clip_grad_norm_(m.parameters(), args.clip_grad_norm)
            for o in active_opts: scaler.step(o)
            scaler.update()

            ep_total += float(loss); n_seen += 1
            if phase == 2: ep_int += float(l_int); ep_grad += float(l_grad)
            pbar.set_postfix(loss=f"{ep_total/n_seen:.4f}", phase=phase)

        pbar.close()
        for s in scheds: s.step()
        for o in all_opts:
            for pg in o.param_groups:
                if pg['lr'] < 1e-6: pg['lr'] = 1e-6

        dt = time.time() - t0
        avg_loss = ep_total / max(1, n_seen)
        lr = opt_enc.param_groups[0]['lr']
        hist_entry = {"epoch": epoch + 1, "phase": phase, "loss": avg_loss,
                      "int_loss": ep_int/max(1,n_seen), "grad_loss": ep_grad/max(1,n_seen),
                      "lr": lr, "dt_sec": round(dt, 1)}
        history.append(hist_entry)
        print(f"[ep {epoch+1:03d}] P{phase} loss={avg_loss:.4f} lr={lr:.2e} ({dt:.1f}s)")
        # Save history incrementally (crash-safe)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    # Save checkpoint (paper convention + variant keys)
    state = {
        "DIDF_Encoder":    encoder.state_dict(),
        "DIDF_Decoder":    decoder.state_dict(),
        "BaseFuseLayer":   base_fuse.state_dict(),
        "DetailFuseLayer": detail_fuse.state_dict(),
        "variant":         args.variant,
        "args":            vars(args),
    }
    if use_gated:
        state["GatedB"] = gated_b.state_dict()
        state["GatedD"] = gated_d.state_dict()
    torch.save(state, ckpt_path)
    print(f"[save ] {ckpt_path}")
    print(f"[save ] {history_path}")


if __name__ == "__main__":
    main()
