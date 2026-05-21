#!/bin/bash
# Train CDDFuse-AG full paper text §5.1 hyperparam.
# GPU hỗ trợ: A100, A30, L40, L40s (≥24GB VRAM cho batch 16).
# Usage trong container:  bash /workspace/remote_train/run_train.sh
set -e

cd /workspace/models/MMIF-CDDFuse

echo "[1/3] Preprocess: extract patches → HDF5"
python dataprocessing_MIF.py

echo "[2/3] Train CDDFuse-AG (120 ep, batch 16, α₃=10, AMP fp16) — paper text §5.1"
python train_MIF.py \
    --variant     Combined-Gated-Saliency \
    --num_epochs  120 \
    --epoch_gap   40 \
    --batch       16 \
    --coeff_tv    10.0 \
    --coeff_decomp 2.0 \
    --seed        42 \
    --output      /workspace/output/ \
    --amp

echo "[3/3] Evaluate trên 72 cặp test"
mkdir -p /workspace/output/eval
CKPT=$(ls -t /workspace/output/CDDFuse-Combined-Gated-Saliency_MIF_*.pth | head -1)
echo "Using ckpt: $CKPT"
for modal in CT PET SPECT; do
    python evaluate_cddfuse.py \
        --variant Combined-Gated-Saliency \
        --modal $modal \
        --ckpt $CKPT \
        --harvard_root /workspace/data/reference \
        --out_dir /workspace/output/eval \
        --save_perimage
done

echo ""
echo "=== HOÀN TẤT ==="
echo "Checkpoint: $CKPT"
echo "Metrics:    /workspace/output/eval/"
echo "Train log:  ${CKPT%.pth}_train_history.json"
