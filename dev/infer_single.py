"""
Inference single image pair with CDDFuse-AG (AsymD-Comb-WAvg-SML).
Usage: python dev/infer_single.py --modal SPECT --id 40031
"""
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF

REPO  = Path(__file__).resolve().parent.parent
MMIF  = REPO / "models" / "MMIF-CDDFuse"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MMIF))

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from variants.registry_asymmetric import build_asym_variant

CKPT = REPO / "kaggle_run" / "_runs_asym" / "AsymD-Comb-WAvg-SML" / "output" / \
       "CDDFuse-AsymD-Comb-WAvg-SML_MIF_06-03-07-38.pth"

DSET_ROOT = REPO / "Havard-Medical-Image-Fusion-Datasets-main" / \
            "Havard-Medical-Image-Fusion-Datasets-main" / "MyDatasets"

MODAL_DIRS = {
    "CT":    ("CT-MRI/test",   "MRI", "CT"),
    "PET":   ("PET-MRI/test",  "MRI", "PET"),
    "SPECT": ("SPECT-MRI/test","MRI", "SPECT"),
}

OUT_ROOT = REPO / "results_v2" / "AsymD-Comb-WAvg-SML" / "Fusion"

def load_gray_tensor(path, device):
    img = Image.open(path).convert("L")
    t = TF.to_tensor(img).unsqueeze(0).to(device)
    return t, np.array(img)

def fix_sd(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modal", default="SPECT")
    parser.add_argument("--id",    default="40031")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    base_rule, detail_rule, _ = build_asym_variant("AsymD-Comb-WAvg-SML")
    encoder     = Restormer_Encoder().to(device)
    decoder     = Restormer_Decoder().to(device)
    base_fuse   = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)
    base_rule   = base_rule.to(device)
    detail_rule = detail_rule.to(device)

    ckpt = torch.load(CKPT, map_location=device)
    encoder.load_state_dict(fix_sd(ckpt["DIDF_Encoder"]))
    decoder.load_state_dict(fix_sd(ckpt["DIDF_Decoder"]))
    base_fuse.load_state_dict(fix_sd(ckpt["BaseFuseLayer"]))
    detail_fuse.load_state_dict(fix_sd(ckpt["DetailFuseLayer"]))
    has_base   = any(True for _ in base_rule.parameters())
    has_detail = any(True for _ in detail_rule.parameters())
    if has_base:   base_rule.load_state_dict(fix_sd(ckpt["BaseRule"]))
    if has_detail: detail_rule.load_state_dict(fix_sd(ckpt["DetailRule"]))

    for m in [encoder, decoder, base_fuse, detail_fuse, base_rule, detail_rule]:
        m.eval()

    subdir, mri_d, modal_d = MODAL_DIRS[args.modal.upper()]
    base_path = DSET_ROOT / subdir
    mri_path    = base_path / mri_d   / f"{args.id}.png"
    modal_path  = base_path / modal_d / f"{args.id}.png"

    print(f"MRI:   {mri_path}")
    print(f"Modal: {modal_path}")

    t_vis, _ = load_gray_tensor(mri_path,   device)
    t_ir,  _ = load_gray_tensor(modal_path, device)

    with torch.no_grad():
        f_v_b, f_v_d, _ = encoder(t_vis)
        f_i_b, f_i_d, _ = encoder(t_ir)
        f_f_b = base_fuse(base_rule(f_v_b, f_i_b))
        f_f_d = detail_fuse(detail_rule(f_v_d, f_i_d))
        fused, _ = decoder(None, f_f_b, f_f_d)
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    out_arr = (fused.squeeze().cpu().numpy() * 255).astype(np.uint8)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{args.id}.png"

    # Color restoration for PET/SPECT (same as evaluate_cddfuse.py)
    if args.modal.upper() != "CT":
        spect_raw = Image.open(modal_path)  # original color SPECT/PET
        fused_y   = Image.fromarray(out_arr)
        w, h = spect_raw.size
        ycbcr = spect_raw.convert("YCbCr")
        _, cb, cr = ycbcr.split()
        cb = cb.resize((w, h), Image.BICUBIC)
        cr = cr.resize((w, h), Image.BICUBIC)
        rgb_out = Image.merge("YCbCr", [fused_y, cb, cr]).convert("RGB")
        rgb_out.save(out_path)
        print(f"Saved (color): {out_path}")
    else:
        Image.fromarray(out_arr).save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
