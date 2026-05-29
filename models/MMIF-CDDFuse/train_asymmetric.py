"""Module D — Asymmetric Fusion Rule training.

Protocol:
- Load pretrained `CDDFuse_MIF.pth` từ paper repo (đã train Phase I trên dataset gốc)
- FREEZE Encoder + Decoder (requires_grad=False)
- Train Phase II ONLY trên 738 cặp Harvard
- Optimizer chỉ chứa params của:
    * BaseFuseLayer (Transformer block)
    * DetailFuseLayer (INN block)
    * base_rule (1 trong DB.0-DB.6)
    * detail_rule (1 trong DD.0-DD.8)

Variant chọn từ `variants/registry_asymmetric.py` (`VARIANT_REGISTRY_ASYM`).

Usage:
    # Stage 1 example
    python train_asymmetric.py --variant AsymD-DB3-BaseVSM \
        --pretrained models/CDDFuse_MIF.pth \
        --num_epochs 30 --batch 8 --amp

    # Stage 2 example
    python train_asymmetric.py --variant AsymD-DD3-DetailSaliency \
        --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

Save:
- Checkpoint với keys paper convention + base_rule_state, detail_rule_state nếu có params
- train_history.json (Phase II only)
"""
from __future__ import annotations

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
from variants.registry_asymmetric import build_asym_variant, list_asym_variants

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ---------- dataset (giống train_MIF.py)
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


def fix_state_dict(state_dict: dict) -> dict:
    """Strip 'module.' prefix khỏi keys (do CDDFuse_MIF.pth save từ DataParallel)."""
    out = {}
    for k, v in state_dict.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def freeze_module(m: nn.Module) -> int:
    """Đặt requires_grad=False cho mọi param. Trả về số param đã freeze."""
    n = 0
    for p in m.parameters():
        p.requires_grad = False
        n += p.numel()
    m.eval()  # tắt dropout / BN running stats
    return n


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list_asym_variants(),
                        help="Asymmetric variant name từ VARIANT_REGISTRY_ASYM")
    parser.add_argument("--pretrained", default="models/CDDFuse_MIF.pth",
                        help="Path tới pretrained CDDFuse checkpoint")
    parser.add_argument("--h5", default="data/MIF_train_imgsize_128_stride_64.h5")
    parser.add_argument("--output", default="models/")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Phase II epochs")
    parser.add_argument("--num_phase1_epochs", type=int, default=0,
                        help="Phase I epochs (default 0 = skip; >0 = full training mode)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coeff_mse_VF", type=float, default=1.0)
    parser.add_argument("--coeff_mse_IF", type=float, default=1.0)
    parser.add_argument("--coeff_decomp", type=float, default=2.0)  # α2/α4
    parser.add_argument("--coeff_tv", type=float, default=10.0)     # α3 (paper text)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="1.0 cho Phase II-only frozen mode. Full training nên dùng 0.01 (paper).")
    parser.add_argument("--optim_step", type=int, default=20)
    parser.add_argument("--optim_gamma", type=float, default=0.5)
    parser.add_argument("--freeze_ed", action="store_true", default=True,
                        help="Freeze Encoder + Decoder (default True cho Phase II-only mode)")
    parser.add_argument("--unfreeze_ed", action="store_true",
                        help="Override: cho phép fine-tune E/D (use cho full training mode)")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Không load pretrained — init random E/D (cho full from-scratch mode)")
    args = parser.parse_args()

    # Khi có Phase 1 → bắt buộc unfreeze E/D (Phase 1 train E/D)
    if args.num_phase1_epochs > 0:
        args.freeze_ed = False
        # Phase 1 thường dùng clip nhỏ hơn cho full retrain
        if args.clip_grad_norm == 1.0:  # user chưa override
            args.clip_grad_norm = 0.01
            print(f"[auto] num_phase1_epochs > 0 → clip_grad_norm=0.01 (paper full retrain)")
    if args.unfreeze_ed:
        args.freeze_ed = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_rule, detail_rule, pixel_select = build_asym_variant(args.variant)
    print(f"[asym] variant={args.variant}")
    print(f"[asym] base_rule={type(base_rule).__name__}  detail_rule={type(detail_rule).__name__}  pixel_select={pixel_select}")
    print(f"[asym] device={device}  freeze_ed={args.freeze_ed}  epochs={args.num_epochs}  batch={args.batch}  amp={args.amp}")

    # ----------------- Build model -----------------
    encoder     = Restormer_Encoder().to(device)
    decoder     = Restormer_Decoder().to(device)
    base_fuse   = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)
    base_rule   = base_rule.to(device)
    detail_rule = detail_rule.to(device)

    # ----------------- Load pretrained (optional) -----------------
    if args.from_scratch:
        print(f"[init ] from scratch — random E/D (no pretrained)")
    else:
        ckpt_path = Path(args.pretrained)
        assert ckpt_path.exists(), f"Pretrained ckpt không tồn tại: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(fix_state_dict(ckpt['DIDF_Encoder']), strict=True)
        decoder.load_state_dict(fix_state_dict(ckpt['DIDF_Decoder']), strict=True)
        if 'BaseFuseLayer' in ckpt:
            try:
                base_fuse.load_state_dict(fix_state_dict(ckpt['BaseFuseLayer']), strict=True)
                print(f"[load ] BaseFuseLayer loaded from pretrained")
            except Exception as e:
                print(f"[load ] BaseFuseLayer skip: {e}")
        if 'DetailFuseLayer' in ckpt:
            try:
                detail_fuse.load_state_dict(fix_state_dict(ckpt['DetailFuseLayer']), strict=True)
                print(f"[load ] DetailFuseLayer loaded from pretrained")
            except Exception as e:
                print(f"[load ] DetailFuseLayer skip: {e}")
        print(f"[load ] Encoder + Decoder loaded from {ckpt_path}")

    # ----------------- Freeze E/D nếu cần -----------------
    if args.freeze_ed:
        n_e = freeze_module(encoder)
        n_d = freeze_module(decoder)
        print(f"[froze] Encoder ({n_e:,} params) + Decoder ({n_d:,} params)")

    # ----------------- Optimizer chỉ chứa params trainable -----------------
    trainable_params = []
    trainable_modules = []

    # base_fuse + detail_fuse luôn train (kể cả khi load pretrained, vì rule mới có thể đổi distribution)
    base_fuse.train(); detail_fuse.train()
    trainable_params += list(base_fuse.parameters()) + list(detail_fuse.parameters())
    trainable_modules += [base_fuse, detail_fuse]

    # base_rule / detail_rule — chỉ train nếu có params
    has_base_params = count_trainable(base_rule) > 0 or any(True for _ in base_rule.parameters())
    has_detail_params = count_trainable(detail_rule) > 0 or any(True for _ in detail_rule.parameters())
    if has_base_params:
        base_rule.train()
        trainable_params += list(base_rule.parameters())
        trainable_modules.append(base_rule)
    else:
        base_rule.eval()
    if has_detail_params:
        detail_rule.train()
        trainable_params += list(detail_rule.parameters())
        trainable_modules.append(detail_rule)
    else:
        detail_rule.eval()

    # Nếu E/D không freeze (override): cũng train
    if not args.freeze_ed:
        encoder.train(); decoder.train()
        trainable_params += list(encoder.parameters()) + list(decoder.parameters())
        trainable_modules += [encoder, decoder]

    n_train = sum(p.numel() for p in trainable_params)
    print(f"[opt  ] trainable params: {n_train:,}  modules: {[type(m).__name__ for m in trainable_modules]}")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.optim_step, gamma=args.optim_gamma)

    # ----------------- Losses -----------------
    criteria_fusion = (Fusionloss() if pixel_select == "max"
                       else FusionLossB(pixel_select=pixel_select).to(device))
    mse = nn.MSELoss(); l1 = nn.L1Loss()
    ssim_loss = (kornia.losses.SSIMLoss(11, reduction='mean')
                 if hasattr(kornia.losses, 'SSIMLoss')
                 else kornia.losses.SSIM(11, reduction='mean'))

    # ----------------- Data -----------------
    loader = DataLoader(MIFH5Dataset(args.h5), batch_size=args.batch, shuffle=True,
                        num_workers=0, pin_memory=(device == "cuda"))
    print(f"[data ] {len(loader.dataset)} patches, {len(loader)} batches/epoch")

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    ckpt_out = out_dir / f"CDDFuse-{args.variant}_MIF_{timestamp}.pth"
    history_path = out_dir / f"CDDFuse-{args.variant}_MIF_{timestamp}_train_history.json"

    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))
    if args.amp: print(f"[amp  ] fp16 enabled")

    history = []

    # ===================================================================
    # PHASE I — pretrain Encoder + Decoder (recon + decomp + TV)
    # Chỉ chạy khi num_phase1_epochs > 0 (full training mode)
    # Optimizer riêng cho E + D (không train fusion modules ở Phase I)
    # ===================================================================
    if args.num_phase1_epochs > 0:
        assert not args.freeze_ed, "Phase I yêu cầu E/D unfreeze"
        opt_p1 = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
        sched_p1 = torch.optim.lr_scheduler.StepLR(
            opt_p1, step_size=args.optim_step, gamma=args.optim_gamma)
        scaler_p1 = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))
        print(f"\n[Phase I] train E+D for {args.num_phase1_epochs} epochs (recon + decomp + TV)")
        for epoch in range(args.num_phase1_epochs):
            encoder.train(); decoder.train()
            t0 = time.time()
            pbar = tqdm(loader, desc=f"ep{epoch+1:03d}/{args.num_phase1_epochs} P1",
                        dynamic_ncols=True, leave=False)
            ep_total = 0.0; n_seen = 0
            for src, mri in pbar:
                src, mri = src.to(device), mri.to(device)
                opt_p1.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                    f_V_B, f_V_D, _ = encoder(src)
                    f_I_B, f_I_D, _ = encoder(mri)
                    src_hat, _ = decoder(src, f_V_B, f_V_D)
                    mri_hat, _ = decoder(mri, f_I_B, f_I_D)
                    cc_B = cc(f_V_B, f_I_B); cc_D = cc(f_V_D, f_I_D)
                    mse_V = 5 * ssim_loss(src, src_hat) + mse(src, src_hat)
                    mse_I = 5 * ssim_loss(mri, mri_hat) + mse(mri, mri_hat)
                    grad_loss = l1(kornia.filters.SpatialGradient()(src),
                                   kornia.filters.SpatialGradient()(src_hat))
                    loss_decomp = (cc_D ** 2) / (1.01 + cc_B)
                    loss = (args.coeff_mse_VF * mse_V + args.coeff_mse_IF * mse_I
                            + args.coeff_decomp * loss_decomp + args.coeff_tv * grad_loss)
                if not torch.isfinite(loss):
                    opt_p1.zero_grad(set_to_none=True)
                    pbar.set_postfix(loss=f"{ep_total/max(1,n_seen):.4f}", skip="NaN")
                    continue
                scaler_p1.scale(loss).backward()
                scaler_p1.unscale_(opt_p1)
                for m in [encoder, decoder]:
                    nn.utils.clip_grad_norm_(m.parameters(), args.clip_grad_norm)
                scaler_p1.step(opt_p1); scaler_p1.update()
                ep_total += float(loss); n_seen += 1
                pbar.set_postfix(loss=f"{ep_total/n_seen:.4f}")
            pbar.close(); sched_p1.step()
            dt = time.time() - t0
            avg_loss = ep_total / max(1, n_seen)
            lr = opt_p1.param_groups[0]['lr']
            history.append({"epoch": epoch + 1, "phase": 1, "loss": avg_loss,
                            "lr": lr, "dt_sec": round(dt, 1)})
            print(f"[ep {epoch+1:03d}] P1 loss={avg_loss:.4f} lr={lr:.2e} ({dt:.1f}s)")
            with open(history_path, 'w') as f: json.dump(history, f, indent=2)
        print(f"[Phase I done] E+D pretrained on {args.num_phase1_epochs} epochs\n")

    # ===================================================================
    # PHASE II — train fusion + (optional finetune E/D)
    # ===================================================================
    for epoch in range(args.num_epochs):
        # Set train mode cho các module trainable
        for m in trainable_modules:
            m.train()
        # Set eval mode cho frozen E/D
        if args.freeze_ed:
            encoder.eval(); decoder.eval()

        t0 = time.time()
        pbar = tqdm(loader, desc=f"ep{epoch+1:03d}/{args.num_epochs} P2",
                    dynamic_ncols=True, leave=False)
        ep_total = ep_int = ep_grad = 0.0
        n_seen = 0

        for src, mri in pbar:
            src, mri = src.to(device), mri.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                # Forward encoder (no grad nếu frozen)
                if args.freeze_ed:
                    with torch.no_grad():
                        f_V_B, f_V_D, _ = encoder(src)
                        f_I_B, f_I_D, _ = encoder(mri)
                else:
                    f_V_B, f_V_D, _ = encoder(src)
                    f_I_B, f_I_D, _ = encoder(mri)

                # Apply asymmetric fusion rules
                f_F_B = base_fuse(base_rule(f_V_B, f_I_B))
                f_F_D = detail_fuse(detail_rule(f_V_D, f_I_D))

                # Decoder
                if args.freeze_ed:
                    with torch.no_grad():
                        # Forward decoder cần grad path qua features (base/detail)
                        # → tạm bỏ no_grad cho decoder để gradient flow ngược về fusion
                        pass
                # Always run decoder WITH grad enabled vì cần gradient cho fusion
                fused, _ = decoder(src, f_F_B, f_F_D)

                # Loss
                cc_B = cc(f_V_B, f_I_B); cc_D = cc(f_V_D, f_I_D)
                loss_decomp = (cc_D ** 2) / (1.01 + cc_B)
                fusion_loss, l_int, l_grad = criteria_fusion(src, mri, fused)
                loss = fusion_loss + args.coeff_decomp * loss_decomp

            # NaN guard — skip batch nếu loss NaN/Inf để tránh corrupt optimizer state
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss=f"{ep_total/max(1,n_seen):.4f}", skip="NaN")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for m in trainable_modules:
                nn.utils.clip_grad_norm_(m.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            ep_total += float(loss); ep_int += float(l_int); ep_grad += float(l_grad)
            n_seen += 1
            pbar.set_postfix(loss=f"{ep_total/n_seen:.4f}")

        pbar.close()
        scheduler.step()
        for pg in optimizer.param_groups:
            if pg['lr'] < 1e-6: pg['lr'] = 1e-6

        dt = time.time() - t0
        avg_loss = ep_total / max(1, n_seen)
        lr = optimizer.param_groups[0]['lr']
        history.append({
            "epoch": epoch + 1, "phase": 2, "loss": avg_loss,
            "int_loss": ep_int / max(1, n_seen),
            "grad_loss": ep_grad / max(1, n_seen),
            "lr": lr, "dt_sec": round(dt, 1),
        })
        print(f"[ep {epoch+1:03d}] P2 loss={avg_loss:.4f} lr={lr:.2e} ({dt:.1f}s)")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    # ----------------- Save -----------------
    state = {
        "DIDF_Encoder":    encoder.state_dict(),
        "DIDF_Decoder":    decoder.state_dict(),
        "BaseFuseLayer":   base_fuse.state_dict(),
        "DetailFuseLayer": detail_fuse.state_dict(),
        "variant":         args.variant,
        "args":            vars(args),
    }
    # Save rule states nếu có params
    if any(True for _ in base_rule.parameters()):
        state["BaseRule"] = base_rule.state_dict()
    if any(True for _ in detail_rule.parameters()):
        state["DetailRule"] = detail_rule.state_dict()
    torch.save(state, ckpt_out)
    print(f"[save ] {ckpt_out}")
    print(f"[save ] {history_path}")


if __name__ == "__main__":
    main()
