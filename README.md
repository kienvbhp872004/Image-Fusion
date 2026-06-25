# CDDFuse-AG — Tổng hợp ảnh y tế đa phương thức với Asymmetric Fusion

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Academic-orange.svg)](#)

> **Đồ án Tốt nghiệp — Đại học Bách Khoa Hà Nội (HUST)**
> Đề xuất mô hình CDDFuse-AG cho tổng hợp ảnh y tế đa phương thức
>
> **Sinh viên:** Đỗ Trung Kiên — MSSV 20224869
> **GVHD:** TS. Phạm Đăng Hải · PGS. TS. Phạm Văn Hải

---

## 1. Tóm tắt

Tổng hợp ảnh y tế đa phương thức (Medical Image Fusion — MIF) kết hợp thông tin từ hai phương thức chụp ảnh (CT/PET/SPECT + MRI) thành một ảnh duy nhất, hỗ trợ chẩn đoán lâm sàng. Đồ án cải tiến **CDDFuse** (CVPR 2023) bằng cách thay thế phép cộng đơn giản ở Fusion Layer bằng hai quy tắc chuyên biệt và bất đối xứng:

| Nhánh | Quy tắc cũ (CDDFuse) | Quy tắc mới (CDDFuse-AG) |
|---|---|---|
| **Base** (tần số thấp) | $f_V + f_I$ | **WAvg** — trung bình có trọng số 1 scalar $\theta$ học được |
| **Detail** (tần số cao) | $f_V + f_I$ | **SML** — lựa chọn cục bộ theo Sum-Modified-Laplacian |

Chỉ thêm **4.161 tham số** (~0.35% so với CDDFuse). Huấn luyện 2 pha trên Harvard MIF (738 cặp train, 72 cặp test).

### Kết quả nổi bật

Trên 72 cặp ảnh test (24 mỗi modality), nhóm 6 chỉ số edge/texture/information (SF · Qabf · AG · EI · QM · QMI):

| Modality | CDDFuse-AG #1 | Cải thiện tiêu biểu |
|---|---|---|
| **MRI-CT** | **6/6 chỉ số** | SF +1.3, QM gần gấp đôi CDDFuse |
| **MRI-PET** | **5/6 chỉ số** | QM +21.9% |
| **MRI-SPECT** | **3/6 chỉ số** | QM +10.4% |
| **Tổng** | **14/18 chỉ số** | — |

Z-score tổng hợp (8 chỉ số, so sánh 6 SOTA): z avg **+0.344** (cao nhất nhóm, CDDFuse: +0.286).

---

## 2. Kiến trúc CDDFuse-AG

### 2.1 Quy tắc Base: Weighted Average Scalar (WAvg)

```
α = σ(θ),   θ ∈ ℝ  (1 tham số học duy nhất)
R_WAvg(a, b) = 2(α·a + (1−α)·b)
```

- Khởi tạo `θ=0` → `α=0.5` → tương đương trung bình đều ở epoch 0
- Học tỉ lệ đóng góp tối ưu giữa hai modality end-to-end

### 2.2 Quy tắc Detail: Sum-Modified-Laplacian (SML)

```
ML_ij(f) = |2f_ij − f_{i-1,j} − f_{i+1,j}| + |2f_ij − f_{i,j-1} − f_{i,j+1}|
SML_ij(f) = Σ_{N(i,j)} ML_pq(f)          # tổng vùng 3×3

w_a = SML(a) / (SML(a) + SML(b) + ε)
R_SML(a, b) = 2(w_a ⊙ a + (1−w_a) ⊙ b)
```

- Không có tham số học — tổng quát hóa tốt qua mọi modality
- Lựa chọn spatially adaptive: mỗi vị trí ưu tiên modality có cạnh sắc nét hơn

### 2.3 Phát hiện Modality-Specificity

Không có quy tắc duy nhất tối ưu cho mọi modality — fusion strategy nên điều chỉnh theo đặc tính từng cặp modality đầu vào.

---

## 3. Cài đặt

### Yêu cầu

| | Giá trị |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| GPU | Tesla P100 (Kaggle) / RTX 30xx local |
| RAM | 16GB |

### Setup

```bash
git clone https://github.com/kienvbhp872004/MMIF-CDDFuse-AG.git
cd MMIF-CDDFuse-AG
pip install -r requirements.txt
```

PyTorch (CUDA 12.1):
```bash
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Sử dụng

### Đánh giá checkpoint

```bash
cd models/MMIF-CDDFuse

# Eval CDDFuse baseline
python evaluate_cddfuse.py --modal CT --ckpt models/CDDFuse_MIF.pth \
    --harvard_root ../../data/reference --out_dir ../../results_v2/CDDFuse

# Eval CDDFuse-AG (Comb-WAvg-SML)
python evaluate_cddfuse.py --variant Comb-WAvg-SML --modal CT \
    --ckpt models/CDDFuse-AG.pth \
    --harvard_root ../../data/reference --out_dir ../../results_v2/CDDFuse-AG
```

### Huấn luyện

```bash
# Tiền xử lý dữ liệu
python dataprocessing_MIF.py

# Train 120 epoch, 2-phase, AMP fp16
python train_MIF.py --variant Comb-WAvg-SML --amp --num_epochs 120 --batch 8
```

Wall time: ~3-5 giờ trên Kaggle Tesla P100.

---

## 5. Cấu trúc repo

```
MMIF-CDDFuse-AG/
├── models/
│   ├── MMIF-CDDFuse/           # CDDFuse + CDDFuse-AG source code
│   │   ├── net.py              # Encoder, Decoder, Restormer blocks
│   │   ├── fusion_rules.py     # WAvg, SML và các quy tắc ablation
│   │   ├── train_MIF.py        # Training script (2-phase, 120 epoch)
│   │   ├── evaluate_cddfuse.py # Eval + per-image metrics
│   │   └── models/             # Pretrained checkpoints
│   └── <22 SOTA models>/       # NestFuse, GeSeNet, WaveFusion, ...
│
├── data/reference/             # 72 cặp ảnh test (24 × CT/PET/SPECT)
│
├── results_v2/
│   ├── CDDFuse/                # Baseline retrain
│   ├── CDDFuse-AG/             # Mô hình đề xuất (Comb-WAvg-SML)
│   ├── <22 SOTA>/
│   ├── zscore_ranking.csv      # Xếp hạng composite z-score 22 methods
│   └── PROGRESS.md             # Nhật ký thí nghiệm
│
├── report_latex/               # Báo cáo ĐATN (LaTeX)
│   ├── main.tex
│   ├── main.pdf                # PDF xuất bản
│   ├── chapters/               # 7 chương
│   └── bibliography.bib
│
├── paper_nckh/                 # Paper NCKH (single-column, A4)
│   ├── main.tex
│   └── main.pdf
│
├── presentation_latex/         # Slide báo cáo (Beamer, HUST RED 16:9)
│   ├── CDDFuse_AG_Slide.tex
│   └── CDDFuse_AG_Slide.pdf
│
├── 20224869-DoTrungKien-DATN/  # Thư mục nộp đồ án
│
├── kaggle_run/                 # Kaggle notebook training
└── dev/                        # Tooling phân tích
```

---

## 6. Ablation Study

Khảo sát hệ thống **13 quy tắc tổng hợp** theo chiến lược one-factor-at-a-time:

| Stage | Khảo sát | Winner |
|---|---|---|
| **1** | 5 Base rule (giữ Detail=Sum) | **WAvg** (#1 CT-MRI) |
| **2** | 8 Detail rule (giữ Base=Sum) | **SML** (ổn định nhất 3 modality) |
| **3** | Kết hợp bất đối xứng | **Comb-WAvg-SML** (10/18 chỉ số paper) |

Thiết kế **đối xứng** (Sym-AG) yếu nhất Stage 3 — xác nhận bất đối xứng hiệu quả hơn.

---

## 7. So sánh SOTA

Xếp hạng Composite Z-score (22 chỉ số, 72 cặp test, 22 phương pháp):

| Rank | Phương pháp | Z avg |
|---|---|---|
| 1 | MM-Net-Fusion | +1.028 |
| **2** | **CDDFuse (pretrained paper)** | **+0.926** |
| 3 | MFS-Fusion | +0.688 |
| ... | ... | ... |

CDDFuse (pretrained) xếp **#2/22** — xác nhận backbone đủ mạnh.
CDDFuse-AG retrain đạt z avg **+0.344** trong nhóm 6 SOTA so sánh trực tiếp (CDDFuse retrain: +0.286).

---

## 8. Tài liệu

| File | Mô tả |
|---|---|
| [`report_latex/main.pdf`](report_latex/main.pdf) | Báo cáo đồ án đầy đủ |
| [`paper_nckh/main.pdf`](paper_nckh/main.pdf) | Paper NCKH (A4, 1 cột) |
| [`presentation_latex/CDDFuse_AG_Slide.pdf`](presentation_latex/CDDFuse_AG_Slide.pdf) | Slide bảo vệ ĐATN |
| [`results_v2/PROGRESS.md`](results_v2/PROGRESS.md) | Nhật ký thực nghiệm |
| [`report_latex/bibliography.bib`](report_latex/bibliography.bib) | BibTeX references |

---

## 9. Tác giả

- **Sinh viên:** Đỗ Trung Kiên — MSSV 20224869
- **Trường:** Đại học Bách Khoa Hà Nội — Khoa học máy tính, Trường CNTT & Truyền thông
- **GVHD:** TS. Phạm Đăng Hải · PGS. TS. Phạm Văn Hải
- **Email:** kien.dt224869@sis.hust.edu.vn
- **GitHub:** [kienvbhp872004/MMIF-CDDFuse-AG](https://github.com/kienvbhp872004/MMIF-CDDFuse-AG)

---

## 10. License

Mã nguồn công khai phục vụ mục đích học thuật và nghiên cứu. Không sử dụng cho mục đích thương mại.

- CDDFuse gốc: Zhao et al., CVPR 2023
- Dataset Harvard MIF: Harvard Medical School
