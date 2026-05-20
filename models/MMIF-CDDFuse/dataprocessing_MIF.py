"""
Pre-processing cho Medical Image Fusion — dùng split CÓ SẴN của dataset.

Cấu trúc input (dataset đã chia train/test sẵn):
    Havard-Medical-Image-Fusion-Datasets-main/Havard-Medical-Image-Fusion-Datasets-main/MyDatasets/
        CT-MRI/
            train/{CT,MRI}/*.png   (160 cặp)
            test/{CT,MRI}/*.png    (24 cặp)
        PET-MRI/
            train/{PET,MRI}/*.png  (245 cặp)
            test/{PET,MRI}/*.png   (24 cặp)
        SPECT-MRI/
            train/{SPECT,MRI}/*.png  (333 cặp)
            test/{SPECT,MRI}/*.png   (24 cặp)

Tổng: 738 cặp train + 72 cặp test (24 × 3 modality).

Output:
    data/MIF_train_imgsize_128_stride_64.h5    --- patches h5
    data/MIF_split.json                          --- danh sách file train/test

Convention: MRI = "ir" role (anatomical), src = "vi" role (functional).
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------- config
PATCH_SIZE = 128
STRIDE = 64
LOW_CONTRAST_FRACTION = 0.1

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATASET_ROOT = ROOT / "Havard-Medical-Image-Fusion-Datasets-main" / "Havard-Medical-Image-Fusion-Datasets-main" / "MyDatasets"
OUT_DIR = HERE / "data"
OUT_H5 = OUT_DIR / f"MIF_train_imgsize_{PATCH_SIZE}_stride_{STRIDE}.h5"
OUT_SPLIT = OUT_DIR / "MIF_split.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def rgb2y(img_rgb):
    """[H,W,3] -> [H,W] Y channel."""
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def load_pair(src_path, mri_path):
    src = np.array(Image.open(src_path), dtype=np.float32)
    mri = np.array(Image.open(mri_path), dtype=np.float32)
    if src.ndim == 3:
        src = rgb2y(src)
    src = src / 255.0
    mri = mri / 255.0
    return mri, src


def extract_patches(img, patch=PATCH_SIZE, stride=STRIDE):
    H, W = img.shape
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    out = np.zeros((len(ys) * len(xs), patch, patch), dtype=np.float32)
    k = 0
    for y in ys:
        for x in xs:
            out[k] = img[y:y + patch, x:x + patch]
            k += 1
    return out


def is_low_contrast(patch, frac_threshold=LOW_CONTRAST_FRACTION):
    lo, hi = np.percentile(patch, [10, 90])
    if hi < 1e-6:
        return True
    return (hi - lo) / hi < frac_threshold


def collect_pairs(modal_folder, src_subdir, split_name):
    """Return list of (src_path, mri_path) tuples."""
    src_dir = DATASET_ROOT / modal_folder / split_name / src_subdir
    mri_dir = DATASET_ROOT / modal_folder / split_name / "MRI"
    if not src_dir.exists() or not mri_dir.exists():
        raise FileNotFoundError(f"Missing: {src_dir} or {mri_dir}")
    out = []
    for f in sorted(os.listdir(src_dir)):
        if not f.lower().endswith(".png"):
            continue
        mri_path = mri_dir / f
        if mri_path.exists():
            out.append((str(src_dir / f), str(mri_path)))
    return out


def main():
    split = {
        "train":      [],
        "test_CT":    [],
        "test_PET":   [],
        "test_SPECT": [],
    }

    # Train: gộp cả 3 modality (738 cặp)
    for modal, sub in [("CT-MRI", "CT"), ("PET-MRI", "PET"), ("SPECT-MRI", "SPECT")]:
        pairs = collect_pairs(modal, sub, "train")
        split["train"].extend([{"src": s, "mri": m, "modal": sub} for s, m in pairs])
        print(f"[train] {modal}: {len(pairs)} cặp")
    print(f"[train] TOTAL: {len(split['train'])} cặp")

    # Test: riêng từng modality
    for modal, sub, key in [("CT-MRI", "CT", "test_CT"),
                            ("PET-MRI", "PET", "test_PET"),
                            ("SPECT-MRI", "SPECT", "test_SPECT")]:
        pairs = collect_pairs(modal, sub, "test")
        split[key] = [{"src": s, "mri": m} for s, m in pairs]
        print(f"[test ] {key}: {len(pairs)} cặp")

    OUT_SPLIT.write_text(json.dumps(split, indent=2))
    print(f"[split] saved -> {OUT_SPLIT}")

    # Patch h5 cho train set
    h5f = h5py.File(OUT_H5, "w")
    g_mri = h5f.create_group("mri_patchs")
    g_src = h5f.create_group("src_patchs")
    train_num = 0
    for pair in tqdm(split["train"], desc="extract patches"):
        mri, src = load_pair(pair["src"], pair["mri"])
        p_mri = extract_patches(mri)
        p_src = extract_patches(src)
        for j in range(p_mri.shape[0]):
            if is_low_contrast(p_mri[j]) or is_low_contrast(p_src[j]):
                continue
            g_mri.create_dataset(str(train_num), data=p_mri[j:j + 1], dtype=p_mri.dtype)
            g_src.create_dataset(str(train_num), data=p_src[j:j + 1], dtype=p_src.dtype)
            train_num += 1
    h5f.close()
    print(f"[h5  ] {train_num} patches saved -> {OUT_H5}")

    with h5py.File(OUT_H5, "r") as f:
        print(f"[h5  ] verify: mri={len(f['mri_patchs'])} src={len(f['src_patchs'])}")


if __name__ == "__main__":
    main()
