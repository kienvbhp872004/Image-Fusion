# Danh sách hình ảnh cần vẽ cho báo cáo CDDFuse-AG

> Tổ chức theo chương, ưu tiên (★ cao nhất → ☆ thấp nhất), độ khó (E=easy, M=medium, H=hard).

---

## Chương 2 — Cơ sở lý thuyết

### 2.1 Bài toán Medical Image Fusion (MIF)

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| **F1** | Sơ đồ paradigm 3 bước của Image Fusion | ★★★ | E | 3 khối: **Phân rã** (decomposition) $\to$ **Tổng hợp thành phần** (Base + Detail fusion) $\to$ **Biến đổi ngược** (reconstruction). 2 ảnh đầu vào, 1 ảnh đầu ra. |
| F2 | Ví dụ ảnh y tế 4 modality | ★★ | E | Lưới $2\times 2$: CT, MRI, PET, SPECT (lấy mẫu thật từ Harvard). |

### 2.2 Kiến trúc CDDFuse gốc

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| **F3** | Tổng quan CDDFuse end-to-end | ★★★ | M | Input $(I_V, I_I)$ $\to$ **Encoder** (shared) $\to$ tách thành $(f^B, f^D)$ cho mỗi modality $\to$ **Fusion** (BaseFuseLayer + DetailFuseLayer) $\to$ **Decoder** $\to$ Output $\hat{I}_F$. Highlight: phép cộng `f_I + f_V`, max-pixel rule trong loss. |
| **F4** | Cấu trúc Restormer Encoder | ★★ | M | PatchEmbed Conv 3×3 $\to$ SFE (4 Transformer block) $\to$ split branch: BTE (1 Transformer) cho Base, DCE (3 INN block) cho Detail. |
| F5 | MDTA Attention chi tiết | ★ | M | Q, K, V tính qua $1\times 1$ conv + DWConv $3\times 3$ $\to$ transpose attention theo channel $\to$ softmax theo $C$ dimension. |
| F6 | DetailNode (INN affine coupling) | ★ | M | $z_1, z_2$ split $\to$ shuffleconv $\to$ $z_2 = z_2 + \theta_\phi(z_1)$, $z_1 = z_1 \cdot \exp(\theta_\rho(z_2)) + \theta_\eta(z_2)$. |
| **F7** | Pipeline 2-phase training | ★★★ | M | Phase I (ep 0–39): train Encoder + Decoder, loss = recon + decomp + TV. Phase II (ep 40–119): train all + fuse layers, loss = fusion + decomp. |

### 2.3 SOTA comparison (Bảng z-score)

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| F8 | Biểu đồ cột composite z-score 22 SOTA | ★★ | E | Bar chart sorted, highlight CDDFuse (rank 2) bằng màu khác. |
| F9 | Heatmap so sánh metric × model | ★ | E | Ma trận model × 7 metric chính (SSIM, PSNR, QG, NABF, ...), color = z-score. |

### 2.4 Statistical evaluation

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| F10 | Sơ đồ pipeline đánh giá thống kê | ★ | E | Perimage CSV $\to$ Wilcoxon paired test $\to$ Cliff's $\delta$ effect size $\to$ Holm correction $\to$ Verdict. |

---

## Chương 3 — Phương pháp đề xuất CDDFuse-AG

### 3.1 Định vị cải tiến trên paradigm 3 bước

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| **F11** | CDDFuse-AG vs CDDFuse — so sánh side-by-side | ★★★ | M | Hai sơ đồ song song, highlight 2 chỗ thay đổi: (1) Adaptive Gating ở Fusion Layer, (2) Saliency-guided Pixel ở Loss. Phần Encoder + Decoder giữ nguyên (vẽ mờ). |

### 3.2 Adaptive Gating (Module A — đóng góp 1)

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| **F12** | Sơ đồ module Adaptive Gating | ★★★ | M | Input $f_V^B, f_I^B$ $\to$ Concat $\to$ Conv $1\times 1$ $\to$ Sigmoid $\to$ Gate $g^B$ $\to$ blend $g \odot f_V + (1-g) \odot f_I$. Có annotation per-pixel, per-channel. |
| F13 | So sánh trực quan Phép cộng vs Adaptive Gating | ★★ | M | Hai sub-figure: (a) `f_I + f_V` (cứng), (b) `g·f_V + (1-g)·f_I` (mềm). Minh họa gate value thay đổi theo vị trí. |
| F14 | Heatmap gate value $g$ thực tế học được | ★ | M | Lấy 1 cặp ảnh test, plot heatmap $g$ của một channel sample (chứng minh gate "smart"). |

### 3.3 Saliency-guided Pixel (Module B — đóng góp 2)

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| **F15** | Sơ đồ Saliency-guided Pixel target | ★★★ | M | $I_V, I_I$ $\to$ Sobel gradient $\to$ $\|\nabla I_V\|, \|\nabla I_I\|$ $\to$ softmax weight $w$ $\to$ target = $w \cdot I_V + (1-w) \cdot I_I$. |
| F16 | So sánh Max-rule vs Saliency-guided | ★★ | M | (a) Max-rule: target = pixel sáng nhất (gây discontinuity). (b) Saliency: target = weighted by gradient (liên tục). Có ví dụ 1D plot. |
| F17 | Visualization saliency map $w$ | ★ | E | Heatmap $w$ trên 1 cặp ảnh test, cho thấy $w$ cao ở vùng biên/texture. |

---

## Chương 4 — Thí nghiệm & kết quả

### 4.1 Data flow

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| **F18** | Data flow huấn luyện (7 bước) | ★★★ | M | PNG $\to$ Y channel $\to$ normalize $\to$ patch $\to$ low-contrast filter $\to$ HDF5 $\to$ DataLoader. |
| **F19** | Data flow suy luận (8 bước) | ★★★ | M | PNG cặp test $\to$ Encoder $\to$ AG $\to$ Fuse $\to$ Decoder $\to$ post-process $\to$ Save. |
| F20 | Patch sampling stride visualization | ☆ | E | Hình $256\times 256$ chia thành $3\times 3$ patch $128\times 128$ stride 64, overlap màu khác. |

### 4.2 Training curves

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| **F21** | Loss curves CDDFuse-AG | ★★★ | E | Plot từ `train_history.json`: total loss + int loss + grad loss theo epoch. Mark Phase I/II boundary. |
| F22 | Learning rate schedule | ★ | E | Plot LR theo epoch (StepLR γ=0.5 mỗi 20 ep). |

### 4.3 Qualitative comparison

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| **F23** | So sánh ảnh fused: CT-MRI | ★★★ | E | Lưới $1\times 4$: $I_V$ (CT) | $I_I$ (MRI) | CDDFuse-Paper-MIF | CDDFuse-AG. Phóng to vùng biên có artifact. |
| **F24** | So sánh ảnh fused: PET-MRI | ★★★ | E | Tương tự F23 cho PET-MRI. |
| **F25** | So sánh ảnh fused: SPECT-MRI | ★★★ | E | Tương tự F23 cho SPECT-MRI. |
| F26 | Zoom-in artifact biên | ★★ | E | Crop $64\times 64$ vùng biên có artifact rõ, so sánh baseline vs AG. |

### 4.4 Statistical results

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| **F27** | Bảng số liệu metric (đã có dạng bảng LaTeX) | ★★★ | E | Có sẵn ở Chapter 4. Đảm bảo highlight rõ giá trị bold. |
| F28 | Forest plot Cliff's $\delta$ per metric | ★★ | M | 22 metric × 3 modality (CT/PET/SPECT), point estimate + CI 95\%. Vertical line at 0 và ngưỡng $\pm 0.147$. |
| F29 | CD diagram (Critical Difference) Module A/B alternatives | ★ | H | Nemenyi post-hoc test cho ablation light retrain. (Optional, chỉ làm nếu có thời gian.) |
| F30 | Bar chart so sánh số SIG metric per modality | ★★ | E | Bar chart 4 cột (CT, PET, SPECT, Pooled), 2 series (Pretrained, AG). |

### 4.5 Module-wise analysis (heatmap)

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| F31 | NABF artifact heatmap | ★★ | M | Heatmap NABF "ở từng pixel" cho baseline vs AG, ảnh CT-MRI. Highlight vùng giảm artifact. |
| F32 | Feature visualization Base & Detail | ★ | M | Plot $f_V^B, f_V^D, f_I^B, f_I^D$ (1 channel sample) cho 1 cặp ảnh. |

---

## Chương 5 — Thảo luận

| # | Tên hình | Ưu tiên | Độ khó | Nội dung |
|---|---|---|---|---|
| F33 | Trade-off summary chart | ★★ | E | Bảng/diagram tóm tắt: cột "Cải thiện" (NABF, MI, QMI) vs cột "Đánh đổi" (QSF, QM, MI). Với arrows. |
| F34 | Modal-specific results bar chart | ★★ | E | 3 nhóm modality, mỗi nhóm 4 metric chính. So sánh CDDFuse-AG vs baseline. |

---

## Chương 6 — Kết luận

(Không cần hình mới, có thể reuse F1 + F11 + F23.)

---

## Tổng hợp ưu tiên

### MUST DO (★★★) — 11 hình cốt lõi
F1, F3, F7, F11, F12, F15, F18, F19, F21, F23, F24, F25, F27

### SHOULD DO (★★) — 9 hình bổ sung
F2, F4, F8, F13, F16, F26, F28, F30, F31, F33, F34

### NICE TO HAVE (★) — phần còn lại nếu có thời gian

---

## Công cụ vẽ đề xuất

| Loại | Công cụ | Lý do |
|---|---|---|
| Sơ đồ kiến trúc (F1, F3, F11, F12, F15) | **TikZ / PowerPoint / draw.io** | Linh hoạt, vector, embed LaTeX dễ |
| Loss curves (F21, F22) | **matplotlib** | Code Python đọc từ `train_history.json` |
| Heatmap (F14, F17, F31) | **matplotlib (imshow + colorbar)** | Standard cho visualize ảnh |
| Qualitative comparison (F23-25) | **PIL/matplotlib subplot** | Concat 4 ảnh thành 1 figure |
| Bar/Forest (F8, F28, F30) | **seaborn / matplotlib** | Bar chart with error bars |
| CD diagram (F29) | **`scikit-posthocs.critical_difference_diagram`** | Sẵn có |

---

## Kế hoạch vẽ thực tế

### Tuần 1 — Architecture diagrams (sơ đồ tĩnh)
- F1, F3, F7, F11, F12, F15 (6 hình kiến trúc cốt lõi)
- Dùng draw.io hoặc PowerPoint, export PDF/PNG.

### Tuần 2 — Visualization tự động (Python)
- F21 (training curves) — script đọc JSON, plot.
- F23–25 (qualitative comparison) — script load fused images, subplot.
- F14, F17 (gate + saliency heatmap) — load model, dump intermediate.
- F30 (bar chart SIG count) — pandas + seaborn.

### Tuần 3 — Optional decorations
- F8 (z-score chart), F28 (forest plot), F31 (NABF heatmap) — nếu có thời gian.

### Tuần 4 — Polish + integration vào LaTeX
- Tất cả figures save vào `report_latex/figures/`
- `\includegraphics[width=0.8\textwidth]{figures/Fxx_name.pdf}` trong TEX.
- Captions rõ ràng, label, cross-ref.

---

## Naming convention đề xuất

```
figures/
├── F01_paradigm_3steps.pdf
├── F03_cddfuse_overview.pdf
├── F07_two_phase_training.pdf
├── F11_cddfuse_ag_vs_cddfuse.pdf
├── F12_adaptive_gating_module.pdf
├── F15_saliency_pixel_target.pdf
├── F18_dataflow_train.pdf
├── F19_dataflow_inference.pdf
├── F21_training_curves.pdf
├── F23_qualitative_CT.pdf
├── F24_qualitative_PET.pdf
├── F25_qualitative_SPECT.pdf
└── ...
```

→ Khớp `\label{fig:Fxx}` trong LaTeX → dễ tham chiếu chéo.
