#!/bin/bash
# Đóng gói code + data thành 1 file zip để upload Drive.
# Usage from Image-Fusion root:  bash remote_train/make_package.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="/tmp/cddfuse_ag_package"
ZIP_OUT="$REPO_ROOT/remote_train/cddfuse_ag_package.zip"

echo "[1/4] Clean staging area"
rm -rf "$STAGE"
mkdir -p "$STAGE"

echo "[2/4] Copy code (chỉ phần cần cho train)"
copy_items=(
    "models/MMIF-CDDFuse/net.py"
    "models/MMIF-CDDFuse/train_MIF.py"
    "models/MMIF-CDDFuse/dataprocessing_MIF.py"
    "models/MMIF-CDDFuse/evaluate_cddfuse.py"
    "models/MMIF-CDDFuse/utils"
    "models/MMIF-CDDFuse/variants"
    "models/MMIF-CDDFuse/models/CDDFuse_MIF.pth"
    "metric"
    "remote_train/Dockerfile"
    "remote_train/run_train.sh"
    "remote_train/README.md"
)
for item in "${copy_items[@]}"; do
    src="$REPO_ROOT/$item"
    dst="$STAGE/$item"
    if [ -e "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp -r "$src" "$dst"
        echo "  OK: $item"
    else
        echo "  MISSING: $item"
    fi
done

echo "[3/4] Copy dataset Harvard medical"
ds_src="$REPO_ROOT/Havard-Medical-Image-Fusion-Datasets-main"
ds_dst="$STAGE/Havard-Medical-Image-Fusion-Datasets-main"
if [ -d "$ds_src" ]; then
    cp -r "$ds_src" "$ds_dst"
    echo "  OK: Havard-Medical-Image-Fusion-Datasets-main"
else
    echo "  MISSING: Havard-Medical-Image-Fusion-Datasets-main"
fi

ref_src="$REPO_ROOT/data/reference"
ref_dst="$STAGE/data/reference"
if [ -d "$ref_src" ]; then
    mkdir -p "$(dirname "$ref_dst")"
    cp -r "$ref_src" "$ref_dst"
    echo "  OK: data/reference (72 test pairs)"
fi

echo "[4/4] Tạo zip"
rm -f "$ZIP_OUT"
(cd "$STAGE" && zip -rq "$ZIP_OUT" .)

size_mb=$(du -m "$ZIP_OUT" | cut -f1)
echo ""
echo "=== HOÀN TẤT ==="
echo "Package:  $ZIP_OUT"
echo "Size:     ${size_mb} MB"
echo "Staging:  $STAGE (có thể xóa)"
echo ""
echo "Bước tiếp theo:"
echo "  1. Upload $ZIP_OUT lên Google Drive"
echo "  2. Gửi link Drive + remote_train/README.md cho bạn"
