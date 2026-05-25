# -*- coding: utf-8 -*-
"""
CDDFuse Evaluation Script
=========================
Runs CDDFuse on the Harvard Medical Image Fusion Dataset (PET/SPECT/CT-MRI)
then computes all available metrics and saves results to CSV + JSON.

Usage (from models/MMIF-CDDFuse directory):
    python evaluate_cddfuse.py --modal PET --max_images 10

Supported modals: PET, SPECT, CT
"""

import os
import sys
import json
import argparse
import warnings
import datetime
import csv
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

warnings.filterwarnings('ignore')

# Deterministic inference: tránh NaN output do cudnn non-determinism trên GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False, warn_only=True)

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # models/MMIF-CDDFuse
REPO_ROOT  = SCRIPT_DIR.parent.parent                 # Image-Fusion/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import metric as M
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from variants.registry import build_variant, VARIANT_REGISTRY
from variants.registry_asymmetric import build_asym_variant, VARIANT_REGISTRY_ASYM


def _is_asym(variant_name: str) -> bool:
    """Check if variant thuộc registry_asymmetric (Module D)."""
    if variant_name is None:
        return False
    name = variant_name[len("CDDFuse-"):] if variant_name.startswith("CDDFuse-") else variant_name
    return name in VARIANT_REGISTRY_ASYM


def _build_any_variant(variant_name: str):
    """Build từ registry phù hợp (Module A/B/C hoặc Module D asym)."""
    if _is_asym(variant_name):
        base_rule, detail_rule, pixel_select = build_asym_variant(variant_name)
        return base_rule, detail_rule, pixel_select, "asym"
    return build_variant(variant_name)

# ═══════════════════════════════════════════════════════════════════════
#  Load Model
# ═══════════════════════════════════════════════════════════════════════
class CDDFuseModel(nn.Module):
    def __init__(self, device, variant: str = None):
        """variant: tên variant (ví dụ 'FuseRule-Gated'). None = baseline (sum)."""
        super().__init__()
        self.encoder = Restormer_Encoder().to(device)
        self.decoder = Restormer_Decoder().to(device)
        self.base_fuse = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
        self.detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)
        self.variant_name = variant
        self.is_asym = _is_asym(variant)
        if variant:
            module_b, module_d, _, _ = _build_any_variant(variant)
            self.gated_b = module_b.to(device)
            self.gated_d = module_d.to(device)
        else:
            self.gated_b = None
            self.gated_d = None
        self.device = device

    def load_weights(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Keys in the PTH might have 'module.' if saved with DataParallel
        def fix_state_dict(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            return new_state_dict

        self.encoder.load_state_dict(fix_state_dict(checkpoint['DIDF_Encoder']))
        self.decoder.load_state_dict(fix_state_dict(checkpoint['DIDF_Decoder']))
        self.base_fuse.load_state_dict(fix_state_dict(checkpoint['BaseFuseLayer']))
        self.detail_fuse.load_state_dict(fix_state_dict(checkpoint['DetailFuseLayer']))
        if self.gated_b is not None:
            if self.is_asym:
                # Module D — asymmetric: keys 'BaseRule' / 'DetailRule' (skip nếu rule không có params)
                has_base_params = any(True for _ in self.gated_b.parameters())
                has_detail_params = any(True for _ in self.gated_d.parameters())
                if has_base_params:
                    if 'BaseRule' not in checkpoint:
                        raise KeyError(f"Variant '{self.variant_name}' yêu cầu 'BaseRule' trong ckpt {ckpt_path}")
                    self.gated_b.load_state_dict(fix_state_dict(checkpoint['BaseRule']))
                if has_detail_params:
                    if 'DetailRule' not in checkpoint:
                        raise KeyError(f"Variant '{self.variant_name}' yêu cầu 'DetailRule' trong ckpt {ckpt_path}")
                    self.gated_d.load_state_dict(fix_state_dict(checkpoint['DetailRule']))
            else:
                # Module A/B/C — symmetric: keys 'GatedB' / 'GatedD'
                if 'GatedB' not in checkpoint or 'GatedD' not in checkpoint:
                    raise KeyError(f"Variant '{self.variant_name}' requires GatedB/GatedD in ckpt {ckpt_path}")
                self.gated_b.load_state_dict(fix_state_dict(checkpoint['GatedB']))
                self.gated_d.load_state_dict(fix_state_dict(checkpoint['GatedD']))

        self.encoder.eval()
        self.decoder.eval()
        self.base_fuse.eval()
        self.detail_fuse.eval()
        if self.gated_b is not None:
            self.gated_b.eval()
            self.gated_d.eval()
        tag = f"variant={self.variant_name}" if self.variant_name else "baseline"
        print(f"  [CDDFuse] Loaded weights from {ckpt_path} ({tag})")

    def fuse(self, t_ir, t_vis, use_mif_logic=True):
        with torch.no_grad():
            f_v_b, f_v_d, _ = self.encoder(t_vis)
            f_i_b, f_i_d, _ = self.encoder(t_ir)

            if self.gated_b is not None:
                f_f_b = self.base_fuse(self.gated_b(f_v_b, f_i_b))
                f_f_d = self.detail_fuse(self.gated_d(f_v_d, f_i_d))
            else:
                f_f_b = self.base_fuse(f_v_b + f_i_b)
                f_f_d = self.detail_fuse(f_v_d + f_i_d)
            
            if use_mif_logic:
                # Based on test_MIF.py Line 56
                fused, _ = self.decoder(None, f_f_b, f_f_d)
            else:
                # Based on test_MIF.py Line 54 (IVF logic)
                fused, _ = self.decoder(t_ir + t_vis, f_f_b, f_f_d)
                
            # Normalization like in test_MIF.py
            fused = (fused - torch.min(fused)) / (torch.max(fused) - torch.min(fused) + 1e-8)
            return fused

# ═══════════════════════════════════════════════════════════════════════
#  Metric Wrapper
# ═══════════════════════════════════════════════════════════════════════
def compute_metrics(img_a: np.ndarray,
                    img_b: np.ndarray,
                    img_f: np.ndarray) -> dict:
    """Inputs: uint8 grayscale HxW numpy (0-255). Outputs: dict of metrics."""
    results = {}
    # No-reference metrics
    for name, fn in [('EN', M.entropy), ('VAR', M.variance), ('AG', M.average_gradient), 
                     ('SF', M.spatial_frequency), ('EI', M.edge_intensity)]:
        try: results[name] = float(fn(img_f))
        except: results[name] = None
    
    try: results['MI'] = float(M.mean_intensity(img_f))
    except: results['MI'] = None

    # Reference metrics
    ref_list = [('NCIE', M.ncie), ('MI_mutual', M.mutual_information), ('NABF', M.nabf), 
                ('FMI', M.fmi), ('CE', M.cross_entropy), ('SSIM', M.ssim), ('PSNR', M.psnr), 
                ('RMSE', M.rmse), ('QG', M.qg_petrovic), ('QM', M.wavelet_qm), ('QC', M.piella_qc), 
                ('QS', M.piella_qs), ('QCB', M.chen_blum), ('QCV',   M.chen_varshney), 
                ('QY', M.yang_ssim), ('QMI', M.mi_normalized), ('QSF', M.sf_relative), 
                ('QNCIE', M.ncc_entropy), ('QTE', M.tsallis_entropy)]
    
    for name, fn in ref_list:
        try: results[name] = float(fn(img_a, img_b, img_f))
        except: results[name] = None
    return results

# ═══════════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════════
def run_inference(model, mri_path: Path, src_path: Path, modal: str, device: torch.device):
    # Load
    img_vi_raw = Image.open(str(src_path))
    img_ir_raw = Image.open(str(mri_path))
    
    w, h = img_ir_raw.size
    
    # CDDFuse expects [1, 1, H, W] grayscale (0-1)
    t_ir = torch.from_numpy(np.array(img_ir_raw.convert('L'), dtype=np.float32) / 255.0).reshape(1, 1, h, w).to(device)
    t_vi = torch.from_numpy(np.array(img_vi_raw.convert('L'), dtype=np.float32) / 255.0).reshape(1, 1, h, w).to(device)
    
    fused_tensor = model.fuse(t_ir, t_vi)

    # NaN fallback: nếu GPU output có NaN (cudnn instability với INN exp()), retry CPU.
    if torch.isnan(fused_tensor).any() and device.type == 'cuda':
        warnings.warn(f"NaN detected on GPU for {mri_path.name}, retrying on CPU")
        model.to('cpu')
        fused_tensor = model.fuse(t_ir.cpu(), t_vi.cpu())
        model.to(device)  # move back to GPU for next iteration

    # Squeeze and convert back to 0-255
    out_np = fused_tensor.squeeze().detach().cpu().numpy()
    out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=0.0)
    out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)

    fused_y_pil = Image.fromarray(out_np)
    
    # Color Restoration for PET/SPECT
    if modal != 'CT':
        ycbcr_src = img_vi_raw.convert('YCbCr')
        _, cb, cr = ycbcr_src.split()
        cb = cb.resize((w, h), Image.BICUBIC)
        cr = cr.resize((w, h), Image.BICUBIC)
        rgb_out = Image.merge('YCbCr', [fused_y_pil, cb, cr]).convert('RGB')
        rgb_np = np.array(rgb_out)
    else:
        rgb_np = out_np
        
    return out_np, rgb_np

# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='CDDFuse Evaluation')
    parser.add_argument('--modal', type=str, default='PET', choices=['PET', 'SPECT', 'CT'])
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=str(SCRIPT_DIR / 'models' / 'CDDFuse_MIF.pth'))
    parser.add_argument('--harvard_root', type=str,
                        default=str(REPO_ROOT / 'Havard-Medical-Image-Fusion-Datasets-main' / 'Havard-Medical-Image-Fusion-Datasets-main'))
    parser.add_argument('--out_dir', type=str, default=str(REPO_ROOT / 'results' / 'CDDFuse'))
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--variant', type=str, default=None,
                        help="Variant name (e.g. 'FuseRule-Gated'). Omit for baseline CDDFuse.")
    parser.add_argument('--save_perimage', action='store_true',
                        help="Dump per-image metrics to <out_dir>/perimage/<model>_<modal>_perimage.csv")
    args = parser.parse_args()
    model_tag = f"CDDFuse-{args.variant}" if args.variant else "CDDFuse"

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    out_dir = Path(args.out_dir)
    fusion_dir = out_dir / 'Fusion'
    fusion_dir.mkdir(parents=True, exist_ok=True)

    # Dataset setup
    h_root = Path(args.harvard_root)
    modal_dir = h_root / (f'{args.modal}-MRI' if args.modal != 'CT' else 'CT-MRI')
    mri_dir = modal_dir / 'MRI'
    src_dir = modal_dir / args.modal
    
    if not mri_dir.exists() or not src_dir.exists():
        print(f"[ERROR] Path missing: {mri_dir} or {src_dir}")
        sys.exit(1)

    mri_files = sorted([f for f in mri_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.bmp')])
    src_map = {f.stem: f for f in src_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.bmp')}
    pairs = [(mf, src_map[mf.stem]) for mf in mri_files if mf.stem in src_map]
    
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"[INFO] Evaluating {len(pairs)} pairs for {args.modal}")

    # Load Model
    model = CDDFuseModel(device, variant=args.variant)
    model.load_weights(args.ckpt)

    all_rows = []
    for idx, (mri_path, src_path) in enumerate(pairs):
        print(f"  [{idx+1}/{len(pairs)}] {mri_path.name} ... ", end='', flush=True)
        try:
            fused_y, rgb_out = run_inference(model, mri_path, src_path, args.modal, device)
            # Save
            Image.fromarray(rgb_out).save(fusion_dir / mri_path.name)
            
            # Grayscale sources for metrics
            mri_gray = np.array(Image.open(str(mri_path)).convert('L'))
            src_gray = np.array(Image.open(str(src_path)).convert('L'))
            
            metrics = compute_metrics(mri_gray, src_gray, fused_y)
            metrics['image'] = mri_path.name
            all_rows.append(metrics)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    # Aggregation
    if not all_rows: return
    metric_keys = [k for k in all_rows[0].keys() if k != 'image']
    avg = {k: float(np.mean([r[k] for r in all_rows if r[k] is not None])) for k in metric_keys}
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_data = {'model': model_tag, 'modal': args.modal, 'n_images': len(all_rows), 'timestamp': timestamp}
    summary_data.update(avg)

    # Save per-image CSV (REQUIRED for skill `fusion-stats`)
    if args.save_perimage:
        perimage_dir = out_dir / 'perimage'
        perimage_dir.mkdir(parents=True, exist_ok=True)
        perimage_csv = perimage_dir / f'{model_tag}_{args.modal}_perimage.csv'
        fieldnames = ['image'] + metric_keys
        with open(perimage_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"[INFO] Per-image metrics saved -> {perimage_csv}")

    # Save CSV & JSON
    with open(out_dir / f'{model_tag}_{args.modal}_summary_{timestamp}.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

    csv_path = out_dir / f'{model_tag}_summary.csv'
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_data.keys()))
        if not file_exists: writer.writeheader()
        writer.writerow(summary_data)

    print('\n' + '=' * 60)
    print(f'  {model_tag} ({args.modal}) Metric Summary')
    print('=' * 60)
    for k, v in avg.items(): print(f"  {k:<12s} {v:>10.4f}")
    print('=' * 60)

if __name__ == '__main__':
    main()
