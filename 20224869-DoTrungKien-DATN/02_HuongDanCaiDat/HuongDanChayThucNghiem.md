# Hướng dẫn chạy thực nghiệm Ablation Study (Module D)

> **Mục tiêu:** Tái tạo toàn bộ kết quả ablation 3 giai đoạn (13 variant) và đánh giá CDDFuse-AG cuối cùng trên Harvard MIF 72 cặp test.

---

## Chuẩn bị môi trường

Đảm bảo môi trường đã cài đặt theo `README.md` trong cùng thư mục. Sau đó:

```bash
cd 05_Code/Image-Fusion/models/MMIF-CDDFuse
```

Tất cả lệnh dưới đây chạy từ thư mục này.

---

## Bước 1 — Tiền xử lý dữ liệu huấn luyện

Chạy một lần duy nhất để tạo file `.h5` patch từ 738 cặp ảnh Harvard:

```bash
python dataprocessing_MIF.py
```

Đầu ra: `../../data/train_data.h5` (~6000–8000 patch 128×128).

---

## Bước 2 — Huấn luyện CDDFuse baseline (tham chiếu)

Train lại CDDFuse gốc từ pretrained checkpoint để đảm bảo so sánh công bằng (cùng dữ liệu, cùng seed):

```bash
python train_MIF.py \
    --pretrained models/CDDFuse_MIF.pth \
    --num_epochs 120 \
    --batch 8 \
    --amp \
    --seed 42
```

Checkpoint lưu tại: `models/CDDFuse_MIF_retrain.pth`

---

## Bước 3 — Stage 1: Base Rule Ablation

Giữ DetailFuseLayer = Sum, thay thế BaseFuseLayer bằng từng rule.  
Mỗi variant train **30 epoch Phase II** từ cùng pretrained checkpoint.

```bash
# DB.2 — L1Norm (nguồn: DenseFuse [Li et al., 2019])
python train_asymmetric.py --variant AsymD-DB2-BaseL1Norm \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DB.3 — Visual Saliency Map (nguồn: Wang et al., 2008)
python train_asymmetric.py --variant AsymD-DB3-BaseVSM \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DB.4 — Local Energy (nguồn: Burt & Kolczynski, 1993)
python train_asymmetric.py --variant AsymD-DB4-BaseLocalEnergy \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DB.5 — Local Entropy (nguồn: Shannon, 1948)
python train_asymmetric.py --variant AsymD-DB5-BaseLocalEntropy \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DB.6 — Weighted Average Scalar (nguồn: DenseFuse [Li et al., 2019])
python train_asymmetric.py --variant AsymD-DB6-BaseWeightedAvg \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp
```

**Winner Stage 1: DB.6 WeightedAvg** (tốt nhất trên CT, ổn định nhất tổng thể).

---

## Bước 4 — Stage 2: Detail Rule Ablation

Giữ BaseFuseLayer = Sum, thay thế DetailFuseLayer bằng từng rule.

```bash
# DD.1 — Adaptive Gating (nguồn: NestFuse [Li et al., 2020])
python train_asymmetric.py --variant AsymD-DD1-DetailGated \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DD.2 — MaxAbs selection (nguồn: DenseFuse [Li et al., 2019])
python train_asymmetric.py --variant AsymD-DD2-DetailMaxAbs \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DD.3 — Saliency Gate (nguồn: Liu et al., 2016)
python train_asymmetric.py --variant AsymD-DD3-DetailSaliency \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DD.4 — Spatial Frequency (nguồn: Eskicioglu & Fisher, 1995)
python train_asymmetric.py --variant AsymD-DD4-DetailSF \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DD.5 — Local Energy (nguồn: Burt & Kolczynski, 1993)
python train_asymmetric.py --variant AsymD-DD5-DetailLocalEnergy \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DD.6 — SML: Sum-Modified-Laplacian (nguồn: Huang & Jing, 2007)
python train_asymmetric.py --variant AsymD-DD6-DetailSML \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DD.7 — L1Norm (nguồn: DenseFuse [Li et al., 2019])
python train_asymmetric.py --variant AsymD-DD7-DetailL1Norm \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp

# DD.8 — PCNN soft (nguồn: PA-PCNN [Yin et al., 2019])
python train_asymmetric.py --variant AsymD-DD8-DetailPCNNsoft \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 30 --batch 8 --amp
```

**Winner Stage 2: DD.6 SML** (ổn định nhất trên cả 3 modality CT/PET/SPECT).

---

## Bước 5 — Stage 3: Kết hợp Asymmetric

Ghép winner Stage 1 (DB.6 WeightedAvg) với các Detail rule tốt nhất:

```bash
# Comb-WAvg-SML — CDDFuse-AG cuối cùng
python train_asymmetric.py --variant AsymD-Comb-WAvg-SML \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 45 --batch 8 --amp

# Comb-WAvg-L1Norm
python train_asymmetric.py --variant AsymD-Comb-WAvg-L1Norm \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 45 --batch 8 --amp

# Comb-WAvg-Gated
python train_asymmetric.py --variant AsymD-Comb-WAvg-Gated \
    --pretrained models/CDDFuse_MIF.pth --num_epochs 45 --batch 8 --amp
```

**Lựa chọn cuối: `AsymD-Comb-WAvg-SML`** — đặt tên là **CDDFuse-AG**.

> Lý do chọn WAvg+SML thay vì WAvg+L1Norm (dù L1Norm có z-score Stage 3 cao hơn):  
> WAvg+SML thắng **10/18** lần trên 6 chỉ số paper × 3 modality,  
> đứng #1 mean rank toàn bộ ablation, và ổn định hơn trên PET-MRI.

---

## Bước 6 — Đánh giá từng variant

Sau khi train xong, đánh giá từng variant trên 72 cặp test (24 mỗi modality):

```bash
# Đánh giá CDDFuse baseline
for modal in CT PET SPECT; do
    python evaluate_cddfuse.py \
        --modal $modal \
        --ckpt models/CDDFuse_MIF_retrain.pth \
        --out_dir ../../results_v2/CDDFuse
done

# Đánh giá một variant (ví dụ DB.6)
for modal in CT PET SPECT; do
    python evaluate_cddfuse.py \
        --modal $modal \
        --variant AsymD-DB6-BaseWeightedAvg \
        --ckpt ../../results_v2/stage1_base/AsymD-DB6-BaseWeightedAvg/best.pth \
        --out_dir ../../results_v2/stage1_base/AsymD-DB6-BaseWeightedAvg
done

# Đánh giá CDDFuse-AG (Ours)
for modal in CT PET SPECT; do
    python evaluate_cddfuse.py \
        --modal $modal \
        --variant AsymD-Comb-WAvg-SML \
        --ckpt ../../results_v2/stage3_combined/AsymD-Comb-WAvg-SML/best.pth \
        --out_dir ../../results_v2/CDDFuse-AG-45ep
done
```

---

## Bước 7 — Tổng hợp kết quả

Tính Composite Z-score và so sánh tất cả variant:

```bash
cd ../../   # về root Image-Fusion/

# Tính z-score tổng hợp
python dev/compute_zscore_sota.py

# Tạo bảng so sánh vs SOTA
python dev/build_vs_sota_table.py

# Tạo bảng ablation summary
python dev/build_ablation_summary.py
```

Kết quả CSV lưu tại `results_v2/_compare_vs_sota/` và `results_v2/zscore_ranking.csv`.

---

## Tóm tắt thứ tự chạy

```
dataprocessing_MIF.py           # 1 lần, ~10 phút
train_MIF.py (baseline)         # ~3–5h trên P100
train_asymmetric.py × 13        # ~1–2h mỗi variant
evaluate_cddfuse.py × (13+1)×3  # ~5 phút mỗi run
compute_zscore_sota.py          # < 1 phút
```

**Tổng thời gian ước tính:** 20–30h trên Kaggle Tesla P100 (chạy tuần tự).  
Kết quả đã chạy sẵn trong `07_KetQuaThucNghiem/`.

---

## Lưu ý khi chạy trên Kaggle

- Dùng GPU P100 (không cần chờ T4 — sm_60 được hỗ trợ với PyTorch 2.5.1+cu121)
- Bật **Persistent Storage** để checkpoint không mất khi session hết
- Xem script Kaggle mẫu: `kaggle_run/push_asym_variant.py`
