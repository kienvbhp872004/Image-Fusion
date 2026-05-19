# CDDFuse-AG — Cải tiến CDDFuse cho Tổng hợp ảnh Y tế Đa phương thức

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2%2Bcu118-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Academic-orange.svg)](#)

> **Đề tài ĐATN — Đại học Bách khoa Hà Nội (HUST)**
> **Đề xuất mô hình CDDFuse-AG tổng hợp ảnh y tế đa phương thức kết hợp Adaptive Gating với Saliency-guided Pixel**

---

## 1. Tóm tắt

Tổng hợp ảnh y tế đa phương thức (Medical Image Fusion --- MIF) là quá trình kết hợp thông tin từ nhiều loại ảnh chụp khác nhau (CT, MRI, PET, SPECT) thành một ảnh duy nhất giúp bác sĩ vừa quan sát được cấu trúc giải phẫu, vừa thấy được thông tin chức năng, hỗ trợ chẩn đoán các bệnh lý phức tạp như ung thư, đột quỵ và bệnh thần kinh.

Trên cơ sở phương pháp **CDDFuse** (CVPR 2023, rank 2/22 SOTA theo composite z-score trên test set Harvard Medical), đồ án đề xuất mô hình **CDDFuse-AG** với hai cải tiến:

1. **Adaptive Gating (AG)** — thay phép cộng đơn giản `f_I + f_V` ở Fusion Layer bằng cơ chế gated mềm `g·f_V + (1-g)·f_I` với `g = σ(W·[f_V; f_I] + b)`. Init zero để epoch 0 ≈ baseline.
2. **Saliency-guided Pixel** — thay quy tắc `max(I_V, I_I)` trong loss bằng tổ hợp lồi có trọng số gradient `w·I_V + (1-w)·I_I` với `w = |∇I_V| / (|∇I_V| + |∇I_I|)` để giảm ảo ảnh ở vùng biên.

### Kết quả chính

Trên 72 cặp ảnh test (24 mỗi modality CT, PET, SPECT), so với baseline CDDFuse huấn luyện lại cùng quy trình (120 epoch, 2 pha, batch 8, AMP fp16 trên Tesla P100):

| Modality | Kết quả |
|---|---|
| **CT** | **10/22 chỉ số có ý nghĩa thống kê** sau Holm correction; SSIM +3.3%, QM +9.5%, NABF giảm 7.1% |
| **PET** | **10/22 chỉ số có ý nghĩa**; NABF giảm 26.6%, QG +2.9% |
| **SPECT** | 2/22; SSIM +3.5%, QSF +7.5%, QMI +4.1% |
| **Pooled (72 cặp)** | NABF giảm 4.2%, SSIM +1.0%, QG +0.9%, QMI +1.2% |

---

## 2. Đặt vấn đề

Mỗi loại ảnh y khoa cung cấp một góc nhìn khác nhau về cơ thể:
- **CT**: cấu trúc xương, mô đặc.
- **MRI**: mô mềm, bệnh lý thần kinh.
- **PET / SPECT**: chức năng chuyển hóa, vùng tế bào hoạt động mạnh.

Bác sĩ phải xem nhiều loại cùng lúc — dễ bỏ sót thông tin. **MIF** kết hợp thành một ảnh tổng hợp giữ cả cấu trúc và chức năng.

### Thách thức của bài toán
- Mỗi modality có đặc tính cường độ / độ tương phản rất khác nhau.
- Khó giữ thông tin đầy đủ của cả hai mà không tạo ảo ảnh ở biên.
- Dữ liệu y tế công khai có kích thước nhỏ (Harvard Medical: 286 cặp).
- Không có một chỉ số duy nhất phản ánh chất lượng — cần đánh giá đồng thời 22+ metric.

### Điểm yếu của CDDFuse gốc
Sau phân tích kiến trúc, đồ án nhận thấy hai điểm có thể cải tiến trong **Bước 2 (Tổng hợp thành phần)** theo paradigm 3 bước của image fusion:
- Phép cộng đơn giản `f_I + f_V` ở Fusion Layer.
- Quy tắc max-pixel trong hàm mất mát `L_int^II`.

→ Đây là động lực cho mô hình CDDFuse-AG. Chi tiết phân tích xem [`docs/CDDFuse_3_thanh_phan.md`](docs/CDDFuse_3_thanh_phan.md).

---

## 3. Mô hình CDDFuse-AG

### 3.1 Định vị cải tiến trên paradigm 3 bước

| Bước | Mô tả | Cải tiến CDDFuse-AG |
|---|---|---|
| **1. Phân rã** | Encoder (Restormer + INN) tách $I_V, I_I$ thành Base + Detail | ❌ Giữ nguyên |
| **2. Tổng hợp thành phần** | BaseFuseLayer + DetailFuseLayer hợp Base, Detail của 2 modality | ⭐⭐ **Adaptive Gating** + **Saliency-guided Pixel** |
| **3. Biến đổi ngược** | Decoder tái tạo ảnh tổng hợp từ Base + Detail | ❌ Giữ nguyên |

→ Cải tiến tập trung 100% ở **Bước 2**, không thay đổi kiến trúc nền tảng (Encoder + Decoder).

### 3.2 Adaptive Gating

```
g^B = σ(Conv1×1([f_V^B; f_I^B]))      ∈ (0, 1) per-pixel, per-channel
f_F^B = g^B ⊙ f_V^B + (1 - g^B) ⊙ f_I^B
```

- Init zero $W_g, b_g \Rightarrow g = 0.5$ tại epoch 0 → bắt đầu từ baseline, học tinh chỉnh dần.
- Thêm ~16K params (1.4% tổng) → rất nhẹ.
- Áp dụng riêng cho Base path và Detail path.

### 3.3 Saliency-guided Pixel

```
S_V = |∇I_V|,  S_I = |∇I_I|          (Sobel gradient magnitude)
w = (S_V + ε) / (S_V + S_I + 2ε)
target = w·I_V + (1-w)·I_I            (convex combination, liên tục)
L_int = ||fused - target||²
```

- Vùng biên/texture (gradient cao) → ưu tiên modality nào sắc hơn.
- Vùng phẳng → trộn cân bằng.
- Không có tham số học, cost $O(HW)$.

---

## 4. Cài đặt môi trường

### 4.1 Yêu cầu

| | Min | Khuyến nghị |
|---|---|---|
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| Python | 3.8 | 3.8.10 |
| GPU | RTX 2060 (6GB) | RTX 3050+ (CUDA 11.8+) |
| RAM | 8GB | 16GB |
| Disk | 15GB | 30GB (gồm checkpoint + Harvard dataset) |

> **Lưu ý**: Đồ án dùng Python 3.8 vì paper CDDFuse gốc test trên version này (kornia 0.6+, einops 0.4+). Với Python 3.10+ có thể gặp issue tương thích kornia/timm.

### 4.2 Setup từng bước (Windows + PowerShell)

#### Bước 1: Clone repo

```powershell
cd D:\Workspace
git clone https://github.com/kienvbhp872004/Image-Fusion.git
cd Image-Fusion
```

#### Bước 2: Tạo virtual environment Python 3.8

```powershell
# Kiểm tra Python 3.8 đã cài
py -3.8 --version    # Phải hiện Python 3.8.10

# Tạo venv
py -3.8 -m venv .venv38

# Activate (PowerShell)
.\.venv38\Scripts\Activate.ps1
```

> **Nếu lỗi execution policy**: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force`

#### Bước 3: Cài dependencies cơ bản

```powershell
pip install --upgrade pip
pip install einops==0.4.1 kornia==0.6.12 h5py tqdm scikit-image scikit-learn scipy opencv-python pandas matplotlib seaborn openpyxl tensorboardX timm
```

#### Bước 4: Cài PyTorch + CUDA 11.8 cho RTX 30xx/40xx

```powershell
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

> **Lưu ý**: Bản torch 2.1.2 là last version support Python 3.8. Torch 2.5+ drop Python 3.8.

#### Bước 5: Verify GPU

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"
```

Output mong đợi:
```
torch: 2.1.2+cu118
cuda: True
device: NVIDIA GeForce RTX 3050 ...
```

### 4.3 Tải checkpoint

Tải pretrained `CDDFuse_MIF.pth` (paper gốc) và `CDDFuse-Combined-Paper-MIF.pth` (CDDFuse-AG của đồ án) từ Kaggle:

```powershell
# Cài kaggle CLI (1 lần)
pip install kaggle
# Đặt kaggle.json vào %USERPROFILE%\.kaggle\

# Pull checkpoint từ Kaggle output
kaggle kernels output kienvbhp1234/cddfuse-combined-paper-mif -p kaggle_run/_runs/Combined-Paper-MIF
```

Hoặc download trực tiếp từ GitHub release (nếu có).

### 4.4 Tải dataset

```powershell
# Harvard Medical (~50MB)
# Đã có sẵn trong Havard-Medical-Image-Fusion-Datasets-main/
# Nếu chưa: download từ http://www.med.harvard.edu/AANLIB/
```

---

## 5. Hướng dẫn sử dụng

### 5.1 Đánh giá pretrained checkpoint

```powershell
cd models\MMIF-CDDFuse

# Eval CDDFuse baseline (paper pretrained) trên 3 modality
foreach ($m in 'CT','PET','SPECT') {
    python evaluate_cddfuse.py `
        --modal $m `
        --ckpt models/CDDFuse_MIF.pth `
        --harvard_root ..\..\data\reference `
        --out_dir ..\..\results_v2\CDDFuse `
        --save_perimage
}

# Eval CDDFuse-AG
foreach ($m in 'CT','PET','SPECT') {
    python evaluate_cddfuse.py `
        --variant Combined-Gated-Saliency `
        --modal $m `
        --ckpt models/CDDFuse-Combined-Paper-MIF.pth `
        --harvard_root ..\..\data\reference `
        --out_dir ..\..\results_v2\CDDFuse-Combined-Paper-MIF `
        --save_perimage
}
```

### 5.2 So sánh thống kê (Wilcoxon + Cliff's δ + Holm)

```powershell
cd ..\..    # back to repo root
python dev\fusion_stats.py --variant Combined-Paper-MIF
```

Output ở `results_v2\_stats\<timestamp>_Combined-Paper-MIF_vs_CDDFuse\REPORT.md`.

### 5.3 Huấn luyện CDDFuse-AG từ đầu (paper-faithful)

```powershell
cd models\MMIF-CDDFuse

# 1. Pre-process (tạo h5 patches)
python dataprocessing_MIF.py

# 2. Train 120 epoch, 2-phase, AMP fp16
python train_MIF.py --variant Combined-Gated-Saliency --amp --num_epochs 120 --batch 8
```

Wall time:
- Local RTX 3050 4GB: ~12-15 giờ
- Kaggle Tesla P100: ~3-5 giờ

> **Để train trên Kaggle**: xem [`kaggle_run/train_mif_paper.ipynb`](kaggle_run/train_mif_paper.ipynb)

---

## 6. Cấu trúc mã nguồn

```text
Image-Fusion/
├── models/
│   ├── MMIF-CDDFuse/              # Mã nguồn CDDFuse + CDDFuse-AG
│   │   ├── net.py                 # Encoder, Decoder, Restormer blocks
│   │   ├── variants/
│   │   │   ├── modules.py         # GatedFuseLayer (Adaptive Gating)
│   │   │   ├── losses.py          # FusionLossB (Saliency-guided Pixel)
│   │   │   └── registry.py        # Đăng ký 9 variants ablation
│   │   ├── train_MIF.py           # Paper-faithful training script (120 ep, 2-phase)
│   │   ├── dataprocessing_MIF.py  # Pre-process Harvard medical → h5
│   │   ├── evaluate_cddfuse.py    # Eval + per-image metrics
│   │   └── models/                # Pretrained checkpoints
│   ├── DAF-Net/, PSFusion/, ...   # 22 SOTA models để so sánh
│
├── data/reference/                # 72 cặp ảnh test (24 × 3 modality)
├── Havard-Medical-Image-Fusion-Datasets-main/  # Full 810 pairs dataset
│
├── dev/                           # Tooling
│   ├── fusion_stats.py            # Wilcoxon + Cliff's δ + Holm pipeline
│   └── run_all_v2.py              # Batch runner cho 22 SOTA models
│
├── metric/                        # 22+ chỉ số chất lượng ảnh
│
├── results_v2/                    # Output evaluation
│   ├── CDDFuse/                   # Baseline paper pretrained
│   ├── CDDFuse-Paper-MIF/         # Baseline retrain của đồ án
│   ├── CDDFuse-Combined-Paper-MIF/# CDDFuse-AG (model đề xuất)
│   ├── <22 SOTA models>/
│   ├── _stats/                    # Stats reports per variant
│   ├── all_models_summary.csv     # Tổng hợp metric 22 models
│   ├── zscore_ranking.csv         # Xếp hạng theo composite z-score
│   └── PROGRESS.md                # Nhật ký thí nghiệm (single source of truth)
│
├── kaggle_run/                    # Kaggle notebooks
│   └── train_mif_paper.ipynb      # Train CDDFuse-AG trên Kaggle P100
│
├── docs/                          # Tài liệu
│   ├── CDDFuse_architecture.md    # Chi tiết kiến trúc CDDFuse
│   └── CDDFuse_3_thanh_phan.md    # Phân tích 3 bước của CDDFuse
│
├── report_latex/                  # Báo cáo ĐATN LaTeX
│   ├── main.tex
│   ├── chapters/                  # 7 chương + phụ lục
│   ├── bibliography.bib           # 19 references
│   └── README.md                  # Hướng dẫn build PDF
│
├── reports_excel/                 # Báo cáo tiến độ Excel
├── Paper/                         # 12 reference papers (PDF, gitignored)
└── archive/                       # File cũ không dùng nữa
```

---

## 7. Đóng góp chính của đồ án

1. **Mô hình CDDFuse-AG**: kết hợp Adaptive Gating + Saliency-guided Pixel, được triển khai và huấn luyện đầy đủ theo quy trình paper gốc 120 epoch / 2-phase.
2. **Tái thực hiện baseline CDDFuse-MIF**: huấn luyện lại từ đầu trên Harvard medical với cùng cấu hình, làm cơ sở so sánh công bằng.
3. **Pipeline đánh giá thống kê tự động**: Wilcoxon signed-rank + Cliff's $\delta$ + Holm--Bonferroni correction trên 25 chỉ số × 3 modality.
4. **Phân tích modal-specific**: chỉ ra CDDFuse-AG có hiệu quả khác nhau trên CT/PET/SPECT, gợi ý hướng tune theo modality cho thực hành lâm sàng.

---

## 8. Tài liệu tham khảo

| Paper | Tác giả | Venue | Vai trò |
|---|---|---|---|
| **CDDFuse** | Zhao et al. | CVPR 2023 | Base model |
| Restormer | Zamir et al. | CVPR 2022 | Backbone Encoder/Decoder |
| GLU | Dauphin et al. | ICML 2017 | Cơ chế gating |
| Highway Networks | Srivastava et al. | NIPS 2015 | Soft interpolation |
| Itti–Koch Saliency | Itti, Koch, Niebur | PAMI 1998 | Saliency map |
| DenseFuse | Li, Wu | TIP 2019 | Weighted fusion |
| U2Fusion | Xu et al. | PAMI 2020 | Adaptive weights |

Bộ paper đầy đủ (12 papers) trong [`Paper/INDEX.md`](Paper/INDEX.md). BibTeX trong [`report_latex/bibliography.bib`](report_latex/bibliography.bib).

---

## 9. Tác giả

- **Họ tên**: Đỗ Trung Kiên
- **Mã số sinh viên**: 20224869
- **Trường**: Đại học Bách khoa Hà Nội (HUST) — Trường Công nghệ Thông tin và Truyền thông
- **Email**: kien.dt224869@sis.hust.edu.vn
- **GitHub**: [@kienvbhp872004](https://github.com/kienvbhp872004)

---

## 10. License

Đồ án thực hiện trong khuôn khổ học tập tại Đại học Bách khoa Hà Nội. Mã nguồn được công khai dưới dạng **academic license** — sử dụng cho nghiên cứu và giáo dục, không cho thương mại không có sự cho phép.

Paper gốc CDDFuse: bản quyền thuộc về Zhao et al., CVPR 2023.
Dataset Harvard Medical: bản quyền thuộc Harvard Medical School.

---

> [!IMPORTANT]
> Đây là dự án ĐATN, một số thí nghiệm và kết quả có thể còn đang trong quá trình hoàn thiện. Mọi liên hệ về mã nguồn vui lòng email trực tiếp tác giả.
