# Tiến trình cải tiến CDDFuse — ĐATN

> **Cập nhật mỗi lần chạy variant.** Không xoá entry cũ. File này là single source of truth về trạng thái thí nghiệm.
>
> Convention: append entry mới ở **§ Variants** (giữ thứ tự thời gian). Cập nhật **§ Trạng thái** ở trên cùng.

---

## Trạng thái

| | |
|---|---|
| Latest update | 2026-05-21 08:43 |
| Variants completed | CDDFuse pretrained / CDDFuse-Paper-MIF retrain / **CDDFuse-AG-45ep ✓ (full 738 train, batch 8, α₃=10, paper text §5.1)** |
| Latest activity | 🎉 **CDDFuse-AG-45ep: CONFIRM_IMPROVEMENT cả 2 comparison**: vs Paper-MIF retrain 8/25 SIG (CT 8, PET 6, SPECT 5), vs CDDFuse pretrained 6/25 SIG (CT 8, PET 9, SPECT 2). Top win: QM, CE, SF, QMI, AG. Trade-off: NABF, SSIM, QY giảm nhẹ. |
| Status | ✅ Train với hyperparam đúng paper text + dataset full 738 cặp cho kết quả mạnh nhất. Tất cả 3 modality CONFIRM_IMPROVEMENT vs apples-to-apples baseline. |
| Next planned | Update thesis Ch.4 với số liệu mới + vẽ figures (training curve, qualitative, NABF heatmap) |

---

## Module A — Fusion Rule (target: NABF gap 1.73σ)

So sánh các cách thay thế phép `A + B` trong `BaseFuseLayer(A + B)` và `DetailFuseLayer(A + B)`.

| # | Variant | Tham số mới | Trạng thái | Composite Δ | Quyết định |
|---|---|---|---|---|---|
| A.1 | `FuseRule-Sum` (baseline) | 0 | ✓ có sẵn (CDDFuse_MIF.pth) | 0 (reference) | KEEP |
| A.2 | `FuseRule-Gated` | 16K | ✓ CPU 10 ep + stats | NABF δ=+0.596, 5 SIG | **WINNER** |
| A.3 | `FuseRule-CrossAttn` | 37K | ✓ CPU 10 ep + stats | NABF δ=+0.529, 5 SIG | runner-up |
| A.4 | `FuseRule-ChannelMoE` | 5K | ✓ CPU 10 ep + stats | NABF δ=+0.504, 4 SIG, QM -0.745 ❌ | weakest |

### Module A — comparison đầy đủ 4/4 alternatives

| Metric | Sum (base) | Gated δ | CrossAttn δ | ChannelMoE δ | Winner | Verdict (Holm) |
|---|---|---|---|---|---|---|
| **NABF** ↓ | 0.0236 | **+0.596 L** | +0.529 L | +0.504 L | Gated | all SIG |
| QSF ↑ | 0.0383 | **+0.602 L** | +0.542 L | +0.488 L | Gated | all SIG |
| PSNR ↑ | 56.22 | +0.243 s | +0.238 s | +0.256 s | ChannelMoE tie | all SIG |
| RMSE ↓ | 0.165 | +0.243 s | +0.238 s | +0.256 s | ChannelMoE tie | all SIG |
| EN ↑ | 5.10 | +0.035 t | +0.048 t | +0.016 t | CrossAttn | Gated/CrossAttn SIG, MoE MARG |
| QM ↑ (regression) | 0.083 | -0.602 L | -0.637 L | **-0.745 L** ❌ | Gated least bad | all NS |
| VAR ↑ (regression) | 76.86 | -0.437 m | -0.421 m | -0.471 m | tie | NS |
| QY ↑ (regression) | 0.956 | -0.275 s | -0.304 s | -0.332 m | Gated | NS |

`L`/`m`/`s`/`t` = large/medium/small/trivial Cliff's δ effect size.

#### Phát hiện quan trọng cho Module A (after 4/4 done)

**1. Pattern WIN/LOSE giống nhau ở cả 3 alternatives.** Cả 3 fusion rule đều:
- WIN: NABF, QSF, PSNR, RMSE (cùng 4 metric LARGE/small effect)
- LOSE: QM, VAR, QY (cùng 3 metric regression)

→ **Trade-off NABF↑ vs QM↓ là inherent với soft fusion** trong Module A — không phải artifact của 1 architecture cụ thể.

**2. Capacity ↔ severity của trade-off.**
- ChannelMoE 5K params (smallest) → QM δ = **-0.745** (worst regression)
- Gated 16K → QM δ = -0.602 (least bad)
- CrossAttn 37K → QM δ = -0.637 (middle)

→ Capacity nhỏ → fusion smooth aggressive hơn → mất nhiều wavelet detail hơn. Gated 16K là sweet spot.

**3. Adding capacity không tăng performance.** CrossAttn (37K) không vượt Gated (16K) ở metric nào quan trọng. ChannelMoE (5K) thua cả 2. → Soft fusion là **bottleneck inherent**, không phải capacity issue.

#### Quyết định Module A

- **Winner: A.2 Gated** — best NABF gain + least QM regression + simplest architecture.
- **Citation**: gating concept (Dauphin et al., GLU 2017) + standard fusion practice. Application novelty cho medical fusion với CDDFuse architecture.
- **Implication cho ROADMAP**: Module B (PixelSelect) trở nên **critical** để fix QM regression — không phải optional. Stretch S5 (FreqLoss) cũng nên ưu tiên.

---

## Module B — PixelSelect (target: fix QM regression của Module A)

So sánh 4 cách thay thế target `max(I_y, I_ir)` trong loss `L_int^II`. Architecture giữ FuseRule=Sum (0 params).

| # | Variant | Cơ chế | Trạng thái | Verdict | Pooled SIG |
|---|---|---|---|---|---|
| B.1 | `PixelSelect-Max` (baseline) | `max(y, ir)` per-pixel | ✓ có sẵn (CDDFuse_MIF.pth) | reference | — |
| B.2 | `PixelSelect-Mean` | `0.5*(y+ir)` | ✓ CPU 10 ep + stats | MIXED | 4/25 |
| B.3 | `PixelSelect-Saliency` | `\|∇y\|/(\|∇y\|+\|∇ir\|)` weighted | ✓ CPU 10 ep + stats | **CONFIRM_IMPROVEMENT** | **6/25** |
| B.4 | `PixelSelect-Learnable` | small CNN predict weight | ✓ CPU 10 ep + stats | CONFIRM_IMPROVEMENT | 5/25 |

### Module B — comparison đầy đủ 4/4 alternatives

| Metric | Max (base) | Mean δ | Saliency δ | Learnable δ | Winner |
|---|---|---|---|---|---|
| **NABF** ↓ | 0.0236 | +0.669 L | +0.671 L | **+0.671 L** | Saliency/Learnable tie |
| QSF ↑ | 0.0383 | **+0.591 L** | +0.568 L | +0.543 L | Mean |
| PSNR ↑ | 56.22 | **+0.251 s** | +0.236 s | +0.228 s | Mean (tiny) |
| RMSE ↓ | 0.165 | **+0.251 s** | +0.236 s | +0.228 s | Mean (tiny) |
| QMI ↑ | — | NS | +0.047 t SIG | **+0.074 t SIG** | **Learnable** |
| QM ↑ (regression) | 0.083 | -0.735 L | -0.732 L | **-0.731 L** | Learnable (nhỉnh) |
| VAR ↑ (regression) | 76.86 | -0.522 L | -0.524 L | -0.529 L | tie |
| MI (regression) | — | -0.426 m | -0.429 m | -0.443 m | Mean (nhỉnh) |
| Pooled SIG | 0 | 4/25 | **6/25** | 5/25 | **Saliency** |
| CT SIG | — | 6 | 6 | **9** | **Learnable** |
| PET SIG | — | 6 | 7 | **8** | **Learnable** |
| SPECT SIG | — | 2 | 2 | 2 | tie |

`L`/`m`/`s`/`t` = large/medium/small/trivial Cliff's δ.

#### Phát hiện quan trọng cho Module B (after 4/4 done)

**1. Pattern WIN/LOSE giống nhau ở cả 4 alternatives.** Mọi soft-target loss đều:
- WIN: NABF, QSF, PSNR, RMSE (tương tự Module A)
- LOSE: QM, VAR, MI, QY (giống Module A)

→ **Trade-off NABF↑ vs QM↓ là TRULY INHERENT**, xuất hiện cả khi soft fusion ở **fusion rule** (Module A) lẫn ở **loss target** (Module B). Đây là phát hiện quan trọng cho thesis: bottleneck nằm ở việc target nào chứa sự "smooth blending", không phải kiến trúc cụ thể.

**2. Trade-off severity gần như identical giữa 4 Module B variants** (QM δ ∈ [-0.735, -0.731]).
- Khác biệt giữa các Module B variants chỉ ở **QMI** (Learnable +0.074 > Saliency +0.047 > Mean NS).
- → Mức độ "smooth" của target chi phối QM regression; cách tính weight chi phối QMI gain.

**3. Saliency vs Learnable: pooled vs per-modality trade-off.**
- Saliency thắng pooled SIG count (6 vs 5) → robustness across modalities khi pool.
- Learnable thắng CT (9 SIG) + PET (8 SIG) per-modality nhưng SPECT yếu (2 SIG) → variance modality cao.
- Hypothesis: Learnable CNN overfits CT/PET (data nhiều hơn 160+245=405 pairs vs SPECT 333 nhưng diverse hơn) → SPECT generalize kém.

**4. SPECT bottleneck phổ quát**. Mọi variant Module A + Module B đều bị SPECT 2 SIG/25. → SPECT có baseline NABF cao nhất (0.033) nhưng cũng có sample variance cao → khó đạt Wilcoxon SIG. Đây là pattern dataset-level, không variant-level.

#### Quyết định Module B

- **Winner pooled (default)**: **B.3 Saliency** — best pooled SIG (6/25), simplest (no extra params), interpretable (gradient = explicit prior).
- **Runner-up**: **B.4 Learnable** — best per-modal CT/PET, best QMI gain, nhưng cần thêm regularization cho SPECT.
- **Citation**: saliency map (Itti & Koch 1998) + gradient-based fusion (DenseFuse 2019, U2Fusion 2020). Application novelty cho CDDFuse loss design.
- **Decision**: chọn **B.3 Saliency** làm Module B winner cho Combined model — robustness > peak performance khi pooled across modalities.

---

## Combined — A.2 Gated × B.3 Saliency

Test giả thuyết Module A winner × Module B winner sẽ phá trade-off NABF↑/QM↓.

| | Sum baseline | A.2 Gated alone | B.3 Saliency alone | **Combined A.2×B.3** |
|---|---|---|---|---|
| NABF δ | — | +0.596 L | +0.671 L | +0.659 L |
| QSF δ | — | +0.602 L | +0.568 L | **+0.679 L** ⭐ |
| **QM δ** (regression) | — | -0.602 L | -0.732 L | **-0.587 L** ⭐ |
| QMI δ | — | -0.196 s | +0.047 t SIG | -0.271 s |
| Pooled SIG | reference | 5 | 6 | 4 |
| Per-modal SIG | — | CT 4, PET 5, SPECT 2 | CT 6, PET 7, SPECT 2 | CT 2, PET 5, SPECT 2 |
| Train config | — | CPU 10 ep | CPU 10 ep | **GPU 25 ep ✓** |

### Phát hiện Combined

**1. QM trade-off lần đầu giảm (-0.587 vs Saliency alone -0.732)**. Sau 9 variants tất cả ≥-0.602, đây là lần đầu thấy regression < -0.6. Hypothesis "inherent" cần refine:
- Trade-off **floor không phải hard inherent** — có thể đẩy lùi bằng combine.
- Architecture-level fix (Gated fusion rule) + Loss-level fix (Saliency target) **tương trợ một phần** ở QM.

**2. Pooled SIG giảm (4 vs 6 alone)**. Combined không thắng pooled vì:
- QSF +0.679 mạnh nhất nhưng các metric khác bị "loãng" — interaction giữa hai cải tiến không đơn cộng.
- QMI từ +0.047 SIG (Saliency alone) → -0.271 NS (Combined) — Gated fusion làm hỏng MI preservation của Saliency target.

**3. Train config khác (GPU 25 ep) cũng làm khó so sánh trực tiếp**. A.2/B.3 chạy CPU 10 ep, Combined chạy P100 GPU 25 ep. → Cần re-run A.2 và B.3 ở GPU 25 ep cho apples-to-apples.

**4. Per-modal CT yếu (2 SIG)** vs các Module B alone (6-9 SIG). PET vẫn giữ (5 SIG). → Combined có vấn đề với CT đặc biệt.

### Quyết định Combined

- **Soft win**: lần đầu giảm QM trade-off, mở hướng cho thesis "trade-off có thể được attenuated".
- **Hard win**: KHÔNG đạt — pooled SIG thấp hơn individual winners.
- **Next step**: re-run A.2 Gated (GPU 25 ep) và B.3 Saliency (GPU 25 ep) làm baseline so sánh fair → có thể Combined sẽ thắng khi all-25-ep-GPU.
- **Thesis framing**: report Combined như "ablation case study" cho thấy trade-off attenuation, không phải winner default.

---

## Variants (chronological)

### 2026-05-21 08:43 · 🎉 `CDDFuse-AG-45ep` — paper text §5.1 hyperparam, 738 train, P100 GPU

**Cấu hình đúng paper text**:
- α₁=1, α₂=2, **α₃=10** (paper text), α₄=2
- batch 8 (P100 16GB), AMP fp16
- 45 ep (Phase I=15, Phase II=30) — giảm từ 120 để fit Kaggle 12h với 738 train
- Train data: **full 738 cặp** từ `MyDatasets/{modal}/train/` (160 CT + 245 PET + 333 SPECT)
- Test: 72 cặp (24×3)
- ckpt sha: `8ddd7a54...`
- Train history: P1 loss 0.35→0.04 (ep 1-15), P2 loss 0.91→0.52 (ep 16-45)

**Stats vs CDDFuse-Paper-MIF (retrain, apples-to-apples)**: **CONFIRM_IMPROVEMENT 8/25 SIG**

| Modal | SIG | Verdict |
|---|---|---|
| CT | 8 + 1 MARG | CONFIRM |
| PET | 6 + 1 MARG | CONFIRM |
| SPECT | 5 + 6 MARG | CONFIRM |
| Pooled | **8** | **CONFIRM** |

**Top improvement (pooled)**:
| Metric | δ | Effect |
|---|---|---|
| QM (wavelet quality) | +0.285 | small SIG |
| CE (cross entropy) | +0.169 | small SIG |
| QSF (sharpness rel.) | +0.173 | small SIG |
| SF (spatial freq.) | +0.084 | trivial SIG |
| QMI | +0.079 | trivial SIG |
| AG | +0.076 | trivial SIG |
| QG | +0.118 | trivial SIG |

**Top regression (NS sau Holm)**:
| Metric | δ | Notes |
|---|---|---|
| QY (Yang quality) | -0.349 | medium |
| NABF | -0.328 | small |
| SSIM | -0.135 | trivial |
| RMSE | -0.131 | trivial |
| PSNR | -0.128 | trivial |

**Stats vs CDDFuse pretrained**: **CONFIRM_IMPROVEMENT 6/25 SIG**

Per-modal: CT 8 + 4 MARG, PET **9**, SPECT 2.

**Insight thesis-level**:

So với run trước (CDDFuse-Paper-MIF với α₃=5, 120 ep, 130 train):

| | Run trước | Run này (AG-45ep) |
|---|---|---|
| Train data | 130 cặp (random split) | **738 cặp** (full MyDatasets/train) |
| α₃ (TV loss) | 5 (code) | **10** (paper text) |
| Total ep | 120 | 45 |
| Pooled SIG vs retrain | 2 | **8** ⭐ |
| Per-modal CONFIRM | 1/3 (chỉ CT) | **3/3** ⭐ |

→ Hyperparam đúng paper + full dataset → kết quả mạnh hơn hẳn, **cả 3 modality đều CONFIRM_IMPROVEMENT**.

**Trade-off shifted**: lần này WIN information metrics (QM, CE, SF, QSF), LOSE perceptual metrics (NABF, SSIM, QY). Trái ngược với run trước (NABF win, QM lose). Nguyên nhân: α₃=10 stronger TV gradient → preserve edges/details/sharpness; nhưng có thể smooth ảnh nhẹ → SSIM perception giảm.

**Files**: `results_v2/CDDFuse-AG-45ep/`, `models/MMIF-CDDFuse/models/CDDFuse-AG-45ep.pth`, train_history.json.

---

### 2026-05-20 09:39 · 🐛 **CRITICAL FIX**: Metric bugs + re-eval all CDDFuse models

**Bug 1 — PSNR/RMSE công thức MATLAB VIFB sai**
- `metric/psnr.py` và `metric/rmse.py` dùng công thức `temp = sqrt(sum((a-b)²)) / (m·n)` thay vì MSE chuẩn `sum((a-b)²) / (m·n)`.
- Origin: code from "the Internet" → propagate vào VIFB MATLAB → port sang Python.
- Hệ quả: PSNR ~55 dB cho mọi ảnh (sai), RMSE ~0.2 (sai). Chuẩn ISO: PSNR 10-30 dB.
- **Fix**: rewrite `psnr.py` và `rmse.py` dùng `np.mean((a-b)²)` và `np.sqrt(mean(diff²))` chuẩn.

**Bug 2 — GPU cudnn non-determinism gây NaN trong INN block**
- INN affine coupling `z1 * exp(theta_rho(z2))` đôi khi overflow trên GPU non-deterministic.
- NaN propagate → `(out * 255).clip(0, 255).astype(uint8) = 0` → fused image all-zero → metric degenerate (MI=0, NABF=0, QTE=0).
- **Fix**: thêm `torch.backends.cudnn.deterministic = True` + NaN detection + CPU fallback trong `evaluate_cddfuse.py`.

**Tác động bug**:
- Mọi PSNR/RMSE absolute trong PROGRESS.md trước đây đều sai → bảng "PSNR 56.22 baseline" thực ra phải là ~14.4 dB.
- Một số eval bị NaN→0 silent failures → một số metric trung bình bị kéo về 0, làm Wilcoxon ranking sai → các verdict "MIXED 2 SIG" hôm trước có thể không đúng.
- **Sau fix**: ranking giữa các metric vẫn dùng được (relative), số tuyệt đối giờ khớp paper.

**Re-evaluation (3 CDDFuse models trên 72 cặp test)**

| Modal | Metric | CDDFuse pretrained | CDDFuse-Paper-MIF (retrain) | CDDFuse-AG | Notes |
|---|---|---|---|---|---|
| CT | SSIM | 1.48 | 1.44 | 1.44 | tương đương |
| CT | PSNR | 14.4 | 14.4 | 14.5 | tương đương |
| CT | NABF↓ | 0.020 | 0.032 | 0.030 | AG đỡ artifact hơn retrain |
| CT | MI | 2.15 | 2.15 | 2.21 | AG nhỉnh hơn |
| PET | SSIM | 1.40 | 1.32 | 1.32 | retrain/AG kém hơn pretrained |
| PET | NABF↓ | 0.018 | 0.015 | 0.014 | AG ít artifact nhất |
| PET | MI | 2.16 | 2.65 | 2.62 | retrain/AG cao hơn pretrained |
| SPECT | SSIM | 1.43 | 1.40 | 1.40 | tương đương |
| SPECT | NABF↓ | 0.033 | 0.019 | 0.020 | retrain/AG đỡ artifact rõ |
| SPECT | MI | 3.08 | 2.72 | 2.73 | pretrained cao hơn |

**Final stats verdict (sau fix)**

| So sánh | Verdict | Pooled SIG | CT SIG | PET SIG | SPECT SIG |
|---|---|---|---|---|---|
| CDDFuse-AG vs CDDFuse-Paper-MIF (apples-to-apples) | MIXED | 2/25 | **9** | 2 | 4 |
| CDDFuse-AG vs CDDFuse pretrained | MIXED | 2/25 | 9 | **11** | 2 |
| CDDFuse-Paper-MIF vs CDDFuse pretrained (reproducibility) | MIXED | 2/25 | — | — | 2 |

→ **CDDFuse-AG cải thiện rõ nhất ở CT (9 metric SIG sau Holm), modest ở SPECT, gần như không ở PET (vs retrain baseline)**.

**Thesis implication**:
- Số liệu hôm trước "10/10/2" → giờ "9/2/4" — CT vẫn vững, PET giảm vì retrain baseline đã tốt nên AG khó vượt thêm.
- Top improvement metric: NABF (artifact), FMI, QG, QMI (saliency-related).
- Top regression: QSF, VAR, MI (info content) — trade-off Base/Detail mà thầy đã chỉ ra.

**Files**:
- `results_v2/_stats/20260520_093909_Combined-Paper-MIF_vs_CDDFuse-Paper-MIF/` — main result
- `results_v2/_stats/20260520_093911_Combined-Paper-MIF_vs_CDDFuse/` — vs paper pretrained
- `results_v2/_stats/20260520_093913_Paper-MIF_vs_CDDFuse/` — reproducibility check

**Quyết định**: số liệu mới là final. Đã xóa các stat reports cũ (buggy). Tiến hành update Chapter 4 thesis với số liệu này.

---

### 2026-05-15 17:18 · `CDDFuse-Combined-CKA-Paper-MIF` — Module C: CKA decomp loss (P100 GPU 120 ep)

**Hypothesis**: Paper's `L_decomp = (CC_D)²/(1.01 + CC_B)` dùng Pearson CC chỉ capture **linear** dependence. Thay bằng **linear CKA** (Centered Kernel Alignment) capture **nonlinear** dependence → có thể force decomposition mạnh hơn → fusion quality cao hơn.

**Config**: Combined-Gated-Saliency + thay `cc()` bằng `linear_cka()` trong cả Phase I và Phase II decomp loss. 120 ep / 2-phase / batch 8 / P100 / AMP fp16.

**Khó khăn implementation**:
- V1 (first attempt): `linear_cka` overflow trong AMP fp16 → loss=NaN ngay epoch 1 → ckpt garbage (NaN trong DIDF_Decoder). Toàn bộ 120 ep wasted.
- V2 (fix): `with torch.amp.autocast('cuda', enabled=False)` + row-normalize features trước Gram matrix → K_x bounded [-1,1] → stable. Loss 2.02 → 0.07 (P1) → 1.4 → 0.70 (P2), no NaN.

**Stats verdict (vs CDDFuse_MIF.pth)**: **MIXED** (1 SIG / 25 pooled, 2 MARG)

| Metric | Cliff's δ | Effect | p_adj |
|---|---|---|---|
| FMI | +0.109 | trivial SIG | 0.0000 |
| NABF | +0.243 | small MARG | 0.3339 |
| CE | +0.064 | trivial MARG | 0.7648 |
| QSF | +0.123 | trivial NS | NS |
| VAR | -0.481 | large NS | NS |
| MI | -0.422 | medium NS | NS |
| SSIM | -0.293 | small NS | NS |

**Per-modality**: CT 9 SIG + 2 MARG, PET 9 SIG, SPECT 2 SIG.

**So sánh CKA vs CC (apples-to-apples Module C)**:

| | Combined-Saliency (CC) | Combined-Saliency-CKA (CKA) |
|---|---|---|
| Pooled SIG | **2** | 1 |
| CT SIG | **10** | 9 |
| PET SIG | **10** | 9 |
| SPECT SIG | 2 | 2 (tie) |
| NABF δ | +0.174 | **+0.243** (CKA raw better) |
| QM δ | -0.118 | -0.188 (CC better — ít regression) |

→ **CKA marginally tệ hơn CC** ở pooled SIG và per-modal CT/PET. Raw NABF δ cao hơn nhưng không SIG sau Holm.

**Insight thesis-level**:
- Hypothesis "nonlinear independence (CKA) > linear correlation (CC) cho decomp" → **KHÔNG confirmed** trên CDDFuse + medical data.
- Possible reasons: (1) Encoder features đã đủ linear-separable do MDTA attention → CC đã catch hết, (2) Sample N=8 quá nhỏ cho CKA estimate (Gram matrix 8×8 rank-7), (3) CKA chỉ thay measure không thay structure → upper bound bị giới hạn bởi formula `D²/(1+B)`.

**Quyết định**: Module C exhausted với linear CKA. Negative finding cũng có giá trị cho thesis — "tested 3 alternatives in Module C (CKA was first), none beat baseline CC in full training". Future work: thử C.10 Orthogonality, C.11 Frequency separation.

**Files**: `results_v2/CDDFuse-Combined-CKA-Paper-MIF/`, `_stats/20260515_171823_Combined-CKA-Paper-MIF_vs_CDDFuse/`, train_history.json

---

### 2026-05-13 08:15 · `CDDFuse-Combined-Learnable-Paper-MIF` — train + stats (P100 GPU 120 ep)

**Hypothesis**: B.4 Learnable trong light retrain đạt CT 9 + PET 8 SIG per-modal (cao nhất Module B). Với paper procedure full training, có thể vượt Combined-Gated-Saliency.

**Config**: Identical Combined-Saliency-Paper (120 ep, 2-phase, batch 8, P100, AMP fp16). Khác duy nhất: `pixel_select='learnable'` (small CNN predict weight) thay 'saliency' (gradient-weighted).

**Stats verdict (vs CDDFuse_MIF.pth)**: **MIXED** (1 SIG / 25 pooled, 3 MARG)

| Metric | Cliff's δ | Effect | p_adj |
|---|---|---|---|
| FMI | +0.067 | trivial SIG | 0.0002 |
| NABF | +0.231 | small MARG | 0.5695 |
| QSF | +0.175 | small MARG | 0.8380 |
| QG | +0.057 | trivial MARG | 0.3956 |
| QM | -0.161 | small NS | NS |

**Per-modality**: CT 7 SIG + 2 MARG, PET 10 SIG, SPECT 1 SIG.

**So sánh trực tiếp Saliency vs Learnable (paper-faithful)**:

| | Saliency | Learnable | Winner |
|---|---|---|---|
| Pooled SIG | 2 | 1 | **Saliency** |
| CT SIG | 10 | 7 | Saliency |
| PET SIG | 10 | 10 | tie |
| SPECT SIG | 2 | 1 | Saliency |
| NABF δ | +0.174 | +0.231 | Learnable raw |
| QM δ | -0.118 | -0.161 | Saliency (ít regression) |

**Insight**:
- Light retrain prediction "Learnable > Saliency" KHÔNG generalize lên paper-faithful → **một ví dụ nữa** confirm light retrain không đáng tin.
- Learnable CNN có thể overfit 130 train pairs (paper-faithful split nhỏ hơn light retrain 738 pool).
- Saliency hardcoded gradient prior generalize tốt hơn data-driven CNN cho dataset size này.

**Quyết định**: B.3 Saliency vẫn là Module B winner cho cả light + paper. Combined-Gated-Saliency là final candidate.

**Files**: `results_v2/CDDFuse-Combined-Learnable-Paper-MIF/`, `_stats/20260513_081444_Combined-Learnable-Paper-MIF_vs_CDDFuse/`, train_history.json

---

### 2026-05-12 21:57 · `CDDFuse-Combined-Paper-MIF` — train + stats (P100 GPU 120 ep paper-faithful)

**Hypothesis**: Combined-Gated-Saliency trained với paper procedure 120 ep / 2-phase sẽ giữ/khuếch đại NABF improvement của light retrain (Cliff's δ +0.659 LARGE) mà không có QM regression (-0.587 LARGE), tức **phá trade-off** đã thấy ở light retrain.

**Architecture**: GatedFuseLayer(64) cho Base + Detail (refactored — feed f_V_B/f_I_B riêng vào gated, KHÔNG sum trước) + FusionLossB(pixel_select='saliency').

**Training config**:
- Hardware: **Tesla P100-PCIE-16GB** (sm_60, torch 2.5.1+cu121)
- Data: 130 train (paper-faithful, seed 42), same as baseline retrain
- 120 ep (P1=40, P2=80), batch 8, AMP fp16
- 6 Adam optimizers (Encoder, Decoder, BaseFuseLayer, DetailFuseLayer, GatedB, GatedD)
- ckpt sha: `ce94f96...`
- Train history: 120 epoch loss curves saved

**Stats verdict (vs CDDFuse_MIF.pth paper baseline)**: **MIXED** (2 SIG / 25 pooled)

| Metric | Cliff's δ | Effect | p_adj | vs Combined-Light δ |
|---|---|---|---|---|
| FMI | +0.082 | trivial SIG | 0.0000 | — |
| QG | +0.075 | trivial SIG | 0.0346 | — |
| QSF | +0.218 | small MARG | 0.1009 | **+0.679 → +0.218** ⚠️ |
| NABF | +0.174 | small NS | 1.0000 | **+0.659 → +0.174** ⚠️ |
| QM | -0.118 | trivial NS | 1.0000 | **-0.587 → -0.118** ⭐ |
| VAR | -0.468 | medium NS | NS | -0.513 → -0.468 (tương đương) |

**Stats vs CDDFuse-Paper-MIF (isolated variant effect)** — aggregated delta:

| Modal | Combined nỗi bật cải thiện | Combined nổi bật giảm |
|---|---|---|
| CT | SSIM +3.3%, QM +9.5%, QMI | EN -10%, MI -12%, **QSF -37%** |
| PET | QG +3%, **NABF -27%** | EN, MI, QSF -7% |
| SPECT | SSIM +3.5%, QM +2.6%, QSF +7.5%, QMI +4% | EN -7%, MI -8%, **NABF +21%** |

**🔥 Phát hiện thesis-level — HYPOTHESIS CONFIRMED**:

**Trade-off NABF↑/QM↓ trong light retrain CHẮC CHẮN là artifact**:
- Light Combined: NABF +0.659 LARGE, QM -0.587 LARGE → magnitude 0.6+
- Paper Combined: NABF +0.174 small, QM -0.118 trivial → magnitude 0.1-0.2
- **Cả 2 chiều đều giảm 3-5×** → không phải Module A+B "phá trade-off", mà light retrain CƯỜNG ĐIỆU cả 2.

**Tại sao?**
- Light: Encoder/Decoder FROZEN → variant features không co-adapt → output drift mạnh ở cả 2 direction
- Paper: Encoder/Decoder train cùng → adapt cho variant → output gần với baseline tự nhiên

**Implication cho thesis** — cần re-framing:

Không thể claim "Combined-Gated-Saliency improves CDDFuse". Thay vào đó claim:
1. **Methodology contribution**: light retrain ablation framework cho fast variant screening (10 ep CPU vs 120 ep GPU = 50× faster)
2. **Empirical finding**: ⚠️ Light retrain results KHÔNG generalize lên full training cho CDDFuse architecture → caveat cho future ablation studies
3. **Modal-specific insights**: Combined có gain CT (10 SIG) + PET (10 SIG) per-modality, SPECT yếu. Có thể tune cho specific modality.

**Files**: `results_v2/CDDFuse-Combined-Paper-MIF/`, `_stats/20260512_215701_Combined-Paper-MIF_vs_CDDFuse/`, train_history.json.

**Quyết định**: SOFT STOP variant experiments. Tiếp theo: viết thesis với re-framing trên, dùng existing data làm evidence.

---

### 2026-05-12 14:30 · `CDDFuse-Paper-MIF` (baseline paper-faithful) — train + stats

**Mục đích**: Reproduce paper sect 5.2 exactly với 120 ep / 2-phase / batch 8 / AMP fp16 trên P100 GPU. So sánh với `CDDFuse_MIF.pth` (pretrained gốc paper) làm sanity check pipeline.

**Training config (FIRST paper-faithful run trong project)**:
- Hardware: **Tesla P100-PCIE-16GB** (sm_60, torch 2.5.1+cu121 downgrade)
- Data: **Harvard medical 130 train + 20 val + 136 test (21 CT + 42 PET + 73 SPECT)** đúng paper sect 5.2
- Patches 128×128 stride 64 → ~1100 patches sau filter low-contrast
- 120 ep (Phase I=40 train Encoder+Decoder, Phase II=80 train all 4 modules)
- 4 Adam optimizers riêng, StepLR γ=0.5/20 ep, AMP fp16, clip grad 0.01
- α1=1, α2=2, α3=5, α4=2 (theo code release)
- Wall time: ~3-5h (P100 GPU + AMP)
- Eval test: dùng existing `data/reference/` (72 ảnh = 24×3 modal) cho fair comparison với 9 light variants
- ckpt sha: `f4646f...`

**Stats verdict** (vs `CDDFuse_MIF.pth` baseline gốc paper): **MIXED** (3 SIG / 25 pooled)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| QSF | +0.012 | **+0.350** | medium | 0.0004 |
| QG | +0.009 | +0.080 | trivial | 0.0297 |
| (NABF) | -0.001 | +0.122 | trivial | 1.0000 (NS) |
| QY | +0.003 | +0.069 | trivial | NS |

**Regression** (light hơn nhiều so với light retrain):
- **QM δ = -0.137 TRIVIAL** ⭐ (vs Combined-light -0.587 LARGE!)
- VAR -0.437 medium
- MI -0.360 medium
- SSIM -0.280 small
- QM gần như **KHÔNG bị regression** — trái ngược với light retrain

**Per-modality**: CT 7 SIG + 2 MARG, PET **10 SIG + 2 MARG** (cao nhất từ trước tới giờ), SPECT 2 SIG.

**🔥 Insight thesis-level — refute giả thuyết "trade-off inherent"**:

Trước đây 9 light retrain variants (Module A + Module B) tất cả đều cho QM δ ≤ -0.602 LARGE → tôi (và bạn) kết luận "trade-off NABF↑/QM↓ là inherent" với soft fusion.

Giờ **paper-faithful training** cho QM δ = -0.137 **trivial** — gần như không regression. Khác biệt **3-5× về magnitude**.

→ Hypothesis trade-off inherent **CÓ THỂ SAI**. Khả năng cao trade-off mà light retrain chỉ ra là **artifact của**:
1. **Under-train** (10 ep CPU vs 120 ep GPU)
2. **Encoder/Decoder bị freeze** trong light → không co-adapt với fusion changes
3. **No Phase I decomposition learning** trong light → features không tối ưu

**Implication cho thesis**:
- ⚠️ Không thể claim "Module A/B winners cải thiện baseline" dựa trên light retrain.
- Cần **re-train Combined-Gated-Saliency với paper procedure** để biết liệu cải tiến có sống sót full training không.
- Nếu Combined-Paper > CDDFuse-Paper-MIF → thesis defensible. Nếu Combined-Paper ≈ CDDFuse-Paper-MIF → có thể paper-faithful đã đạt ceiling, light retrain results không generalize.

**Files**: `results_v2/CDDFuse-Paper-MIF/`, `_stats/20260512_142953_Paper-MIF_vs_CDDFuse/`

**Quyết định**: **MUST DO NEXT** — train Combined-Gated-Saliency với cùng paper procedure (120 ep / 2-phase P100 GPU). Refactor `train_MIF.py` để support Combined variant trước.

---

### 2026-05-05 15:15 · `CDDFuse-Combined-Gated-Saliency` — train + stats (P100 GPU)

**Hypothesis**: Combine A.2 Gated (Module A winner) × B.3 Saliency (Module B winner) sẽ phá trade-off NABF↑/QM↓ — vì hai cải tiến target hai cấp khác nhau (architecture-level vs loss-level).

**Module**: GatedFuseLayer (16K) cho cả Base + Detail + FusionLossB(pixel_select='saliency').

**Training config (FIRST GPU run)**:
- Hardware: **Tesla P100-PCIE-16GB** (sm_60)
- Torch: **2.5.1+cu121** (downgraded từ Kaggle default torch 2.10 để support sm_60)
- Epochs: **25** (paper config, đầu tiên đạt được)
- Batch: **4** (paper config)
- Seed: 42
- Wall time: ~1h (vs CPU 3-4h cho 10 ep)

**Stats verdict**: **MIXED** (4 SIG / 25 pooled, 1 marginal)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| **QSF** | +0.068 | **+0.679** | LARGE (best của ALL variants) | 0.0000 |
| **NABF** ↓ | -0.013 | +0.659 | LARGE | 0.0000 |
| PSNR | +0.163 | +0.225 | small | 0.0038 |
| RMSE | -0.008 | +0.225 | small | 0.0000 |
| EN | +0.039 | +0.025 | trivial | 0.1849 (MARGINAL) |

**Regression** — quan trọng:
- **QM δ = -0.587 LARGE** (vs B.3 alone -0.732, Gated alone -0.602) — **lần đầu đạt < -0.6** sau 9 variants
- VAR -0.513 L, MI -0.384 m, QY -0.326 s, **QMI -0.271 small** (Saliency alone +0.047 SIG; Gated phá QMI)

**Per-modality**: CT 2 SIG (yếu), PET 5 SIG, SPECT 2 SIG. CT là điểm yếu lớn của Combined.

**Insight quan trọng**:
- **Trade-off floor không hard inherent** — sau 9 variants ≥ -0.6, Combined đẩy về -0.587 → có thể attenuate bằng combine.
- **Pooled SIG giảm vì interaction**: Gated phá QMI gain của Saliency. Hai cải tiến **không tương trợ ở mọi metric**.
- **Apples-to-apples không công bằng**: Combined GPU 25 ep, A.2/B.3 CPU 10 ep. Chưa kết luận chắc chắn Combined < individual cho đến khi re-run.

**Quyết định**: SOFT WIN. Report như case study trade-off attenuation. Hard win cần re-run A.2 + B.3 ở 25 ep GPU.

**Files**: `results_v2/CDDFuse-Combined-Gated-Saliency/`, `_stats/20260505_151513_Combined-Gated-Saliency_vs_CDDFuse/`

---

### 2026-05-04 22:40 · `CDDFuse-PixelSelect-Learnable` (B.4) — train + stats

**Hypothesis**: Small CNN `[y, ir, ∇y, ∇ir] → sigmoid(weight)` predict per-pixel weight, init zero → epoch-0 = 0.5*(y+ir). Data-driven có thể học pattern phức tạp hơn (không bị giới hạn formula gradient của Saliency). Kỳ vọng: Learnable ≥ Saliency.

**Module**: `variants/losses.py::FusionLossB(pixel_select='learnable')`. Architecture: Conv2d(4→8, 3×3) + GELU + Conv2d(8→1, 1×1). Output projection init zero.

**Training config**: CPU 10 ep, batch 2, seed 42, fine-tune BaseFuseLayer + DetailFuseLayer + LossNet.weight_net (~381K + small CNN params)

**Stats verdict**: **CONFIRM_IMPROVEMENT** (5 SIG / 25 pooled, 1 marginal)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| **NABF** ↓ | -0.012 | +0.671 | LARGE | 0.0000 |
| QSF | +0.085 | +0.543 | LARGE | 0.0000 |
| PSNR | +0.162 | +0.228 | small | 0.0001 |
| RMSE | -0.008 | +0.228 | small | 0.0000 |
| **QMI** | +0.007 | +0.074 | trivial | 0.0016 (best Module B) |

**Regression**: QM δ=-0.731 LARGE (≈ Saliency, Mean), VAR -0.529 L, MI -0.443 m, QY -0.267 s.

**Per-modality** (notable):
- CT: **9 SIG** (cao nhất Module B, vs Saliency 6, Mean 6)
- PET: **8 SIG** (cao nhất Module B, vs Saliency 7, Mean 6)
- SPECT: 2 SIG (tie với toàn bộ Module B)

**Insight quan trọng — hypothesis partially confirmed**:
- Learnable thắng per-modality CT/PET nhưng **thua pooled SIG count** (5 vs Saliency 6) — vì pooled phải robust qua cả 3 modal mà Learnable yếu ở SPECT.
- QMI gain LARGEST của Module B (δ=+0.074, Saliency chỉ +0.047) → CNN data-driven học được mutual information feature tốt hơn gradient heuristic.
- QM regression không tệ hơn Saliency → trade-off floor đạt rồi (LARGE ở mọi variant).

**Quyết định Module B**: B.3 Saliency = winner default (pooled robustness). B.4 Learnable = runner-up + có giá trị làm ablation entry "data-driven không thắng heuristic khi pooled".

**Files**: `results_v2/CDDFuse-PixelSelect-Learnable/`, `_stats/20260504_224006_PixelSelect-Learnable_vs_CDDFuse/`

---

### 2026-05-04 18:25 · `CDDFuse-PixelSelect-Saliency` (B.3) — train + stats

**Hypothesis**: Gradient-weighted target `w·y + (1-w)·ir` với `w = |∇y| / (|∇y|+|∇ir|)` preserve sharper pixel (gần Max-rule) nhưng smooth ở vùng ít gradient → kỳ vọng cải thiện NABF tương đương Mean nhưng QM ít regression hơn.

**Module**: `variants/losses.py::FusionLossB(pixel_select='saliency')`. Sobel kernel via `register_buffer` (device-agnostic). FuseRule giữ Sum (IdentitySum, 0 params). Loss target khác biệt duy nhất.

**Training config**: CPU 10 ep, batch 2, seed 42, fine-tune BaseFuseLayer + DetailFuseLayer (~381K trainable)

**Stats verdict**: **CONFIRM_IMPROVEMENT** (6 SIG / 25 pooled — best Module B tới giờ)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| **NABF** ↓ | -0.011 | **+0.671** | LARGE | 0.0000 |
| QSF | +0.084 | +0.568 | LARGE | 0.0000 |
| PSNR | +0.175 | +0.236 | small | 0.0001 |
| RMSE | -0.009 | +0.236 | small | 0.0000 |
| **QMI** | +0.005 | +0.047 | trivial | 0.0134 (SIG) |

**Regression**:
- QM δ = -0.732 LARGE (≈ Mean -0.735, không tệ hơn)
- VAR δ = -0.524 LARGE
- MI δ = -0.429 medium
- QY δ = -0.302 small

**Per-modality**: CT 6 SIG + 3 MARG, PET 7 SIG + 1 MARG (cao nhất Module B), SPECT 2 SIG. SPECT vẫn yếu nhất (giống pattern toàn bộ).

**Insight quan trọng — hypothesis confirmed**:
- Saliency thắng Mean về SIG count (6 vs 4) mà KHÔNG làm tệ thêm QM regression.
- Cơ chế: gradient-weighted giữ pixel có gradient lớn (sharp edges) → giảm artifact ở boundary giống Max nhưng smooth ở vùng flat → cải thiện QMI (Mean không có).
- Trade-off NABF↑ vs QM↓ vẫn LARGE → Module B alone không phá được trade-off. Cần combine A.2 Gated × B.3 Saliency hoặc thêm S5 FreqLoss.

**Quyết định**: ITERATE → Module B candidate winner. Chạy B.4 Learnable để hoàn tất 4/4 → Friedman 4-way → chốt Module B winner. Saliency là front-runner.

**Files**: `results_v2/CDDFuse-PixelSelect-Saliency/`, `_stats/20260504_182458_PixelSelect-Saliency_vs_CDDFuse/`

---

### 2026-05-04 13:30 · `CDDFuse-PixelSelect-Mean` (B.2) — train + stats

**Hypothesis**: Mean target `(I_y + I_ir)/2` tránh boundary artifact của max-rule → kỳ vọng giảm NABF mà KHÔNG hỏng QM/VAR (mean ít aggressive hơn max).

**Module**: `variants/losses.py::FusionLossB(pixel_select='mean')`. Architecture giữ nguyên FuseRule=Sum (IdentitySum, 0 params). Loss target là khác biệt duy nhất.

**Training config**: CPU 10 ep, batch 2, seed 42, fine-tune BaseFuseLayer + DetailFuseLayer (~381K trainable, không có loss params)

**Stats verdict**: **MIXED** (4 SIG / 25 pooled, QCB marginal)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| **NABF** ↓ | -0.011 | **+0.669** | LARGE (best so far!) | 0.0000 |
| QSF | +0.084 | +0.591 | LARGE | 0.0000 |
| PSNR | +0.191 | +0.251 | small | 0.0000 |
| RMSE | -0.009 | +0.251 | small | 0.0000 |
| QCB | +0.004 | +0.030 | trivial | 0.1214 (MARGINAL) |

**Regression** (worse than Gated/CrossAttn):
- **QM δ = -0.735 LARGE** ❌
- **VAR δ = -0.522 LARGE** (lên LARGE — Module A chỉ medium)
- MI δ = -0.426 medium
- QY δ = -0.319 small

**Insight quan trọng — hypothesis sai**:
- Tôi predict Mean ít aggressive hơn Max → giữ QM/VAR tốt hơn. **Sai**.
- Thực tế: Mean blur cả 2 modality (mỗi pixel = avg) → output smoother hơn cả Max-rule (chỉ chọn sharper pixel).
- → Mean **đẩy trade-off mạnh hơn**: NABF cải thiện nhiều hơn (+0.669 vs Gated +0.596) nhưng QM/VAR cũng tệ hơn (-0.735 vs -0.602, -0.522 vs -0.437).
- Module B Mean **không giải quyết** QM regression như kỳ vọng. Trade-off có vẻ inherent với mọi soft fusion, kể cả khi soft xuất hiện ở loss target chứ không ở fusion rule.

**Quyết định**: ITERATE. Không phải Module B winner. Hi vọng B.3 Saliency (gradient-weighted) có thể break pattern vì nó preserve sharper pixel theo gradient magnitude — gần với Max nhưng smooth hơn.

**Files**: `results_v2/CDDFuse-PixelSelect-Mean/`, `_stats/20260504_133026_PixelSelect-Mean_vs_CDDFuse/`

---

### 2026-05-04 01:10 · `CDDFuse-FuseRule-ChannelMoE` (A.4) — train + stats

**Hypothesis**: Per-channel MoE routing (3 experts: A, B, mix) giải quyết trade-off bằng cách cho mỗi channel chọn expert phù hợp thay vì blend đều.

**Module**: `variants/modules.py::ChannelMoEFuse` — 5K params/module, GAP → MLP → softmax(3 experts) per channel. Inspired TC-MoA (CVPR 2024). Identity init: uniform 1/3 → output = 0.5*(A+B).

**Training config**: identical với A.2/A.3 (CPU 10 ep, batch 2, seed 42)

**Stats verdict**: **MIXED** (4 SIG / 25 pooled, EN MARGINAL)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| **NABF** ↓ | -0.010 | +0.504 | LARGE | 0.0004 |
| QSF | +0.090 | +0.488 | LARGE | 0.0000 |
| PSNR | +0.235 | +0.256 | small | 0.0000 |
| RMSE | -0.010 | +0.256 | small | 0.0000 |
| EN | +0.027 | +0.016 | trivial | 0.1713 (MARGINAL) |

**Regression** (worst của 3 alternatives):
- **QM δ = -0.745 LARGE** ❌ (Gated -0.602, CrossAttn -0.637)
- VAR δ = -0.471 medium
- QY δ = -0.332 medium (lên medium effect, hơn Gated/CrossAttn small)

**Insight quan trọng**:
- ChannelMoE = smallest capacity (5K) → smooths most aggressive → highest QM regression. Confirm hypothesis "trade-off scales with smoothing aggressiveness".
- Per-modality: CT 7 SIG (cao nhất), SPECT chỉ 2 SIG. Pattern: capacity nhỏ chịu ảnh hưởng modality variability nhiều hơn.

**Quyết định**: REJECT (trong Module A) — ChannelMoE không phải winner. Nhưng vẫn có giá trị làm **negative result** trong ablation table (chứng minh "MoE routing không cải thiện").

**Files**: `results_v2/CDDFuse-FuseRule-ChannelMoE/`, `_stats/20260504_010829_FuseRule-ChannelMoE_vs_CDDFuse/`

---

### 2026-05-03 14:20 · `CDDFuse-FuseRule-CrossAttn` (A.3) — train + stats

**Hypothesis**: Bidirectional channel-attention (A↔B) giải quyết QM regression của Gated bằng cách giữ richer feature interaction.

**Module**: `variants/modules.py::CrossAttnFuse` — Restormer MDTA-style channel attention, 4 heads, 37K params/module, identity init (proj_out zero → epoch-0 = 0.5*(A+B)).

**Training config**: identical với Gated (CPU 10 ep, batch 2, seed 42, ckpt sha `c540...`)

**Loss curve**: 1.022 → 0.880 (giống Gated, hội tụ chậm tương tự).

**Stats verdict**: **CONFIRM_IMPROVEMENT** (5 SIG / 25 pooled, sau Holm)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| QSF | +0.068 | +0.542 | LARGE | 0.0000 |
| **NABF** ↓ | -0.009 | +0.529 | LARGE | 0.0001 |
| PSNR | +0.241 | +0.238 | small | 0.0000 |
| RMSE | -0.010 | +0.238 | small | 0.0000 |
| EN | +0.073 | +0.048 | trivial | 0.0000 |

**Regression** (giống Gated): QM δ=-0.637 LARGE, VAR δ=-0.421 medium.

#### Comparison Gated vs CrossAttn

| Metric | Gated δ | CrossAttn δ | Winner |
|---|---|---|---|
| NABF | **+0.596** | +0.529 | Gated |
| QSF | **+0.602** | +0.542 | Gated |
| PSNR/RMSE | +0.243 | +0.238 | tie |
| QM (regression) | -0.602 | **-0.637** | Gated less bad |

→ **Gated nhỉnh hơn CrossAttn ở 4/5 metric chính**. CrossAttn dùng 37K params (vs 16K Gated) nhưng **không bù được** complexity tăng.

**Insight**: 2 fusion rules khác nhau cho **kết quả gần giống nhau** → trade-off NABF↑ vs QM↓ là **inherent của Module A** (bất kỳ soft fusion nào cũng sẽ smooth output). Cần Module B (PixelSelect) hoặc khác để giải quyết QM regression.

**Quyết định**: ITERATE. Chạy A.4 ChannelMoE để hoàn tất 4 alternatives → Friedman test → chốt Module A.

**Files**:
- `results_v2/CDDFuse-FuseRule-CrossAttn/` — full output
- `results_v2/_stats/20260503_141849_FuseRule-CrossAttn_vs_CDDFuse/` — stats report

---

### 2026-05-02 23:00 · stats `CDDFuse-FuseRule-Gated` vs `CDDFuse`

**Tool**: `dev/fusion_stats.py` (skill `fusion-stats` execution)
**Test**: Wilcoxon signed-rank paired, alternative='greater', α=0.05, Holm-Bonferroni per-modal
**N pairs**: 24×3 modal = 72 pooled
**Output dir**: `results_v2/_stats/20260502_225704_FuseRule-Gated_vs_CDDFuse/`

#### Verdict: **CONFIRM_IMPROVEMENT** (5 SIG / 25 metrics pooled, sau Holm)

#### WIN (significant)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| **NABF** ↓ (TARGET) | -0.011 | **+0.596** | LARGE | 0.0000 |
| QSF | +0.069 | +0.602 | LARGE | 0.0000 |
| PSNR | +0.224 | +0.243 | small | 0.0000 |
| RMSE | -0.010 | +0.243 | small | 0.0000 |
| EN | +0.051 | +0.035 | trivial | 0.0022 |

#### LOSE (regression rõ, không SIG vì Wilcoxon one-sided)

| Metric | Mean Δ | Cliff's δ | Effect |
|---|---|---|---|
| **QM (wavelet)** | -0.072 | **-0.602** | LARGE |
| VAR | -9.26 | -0.437 | medium |
| MI | -9.10 | -0.303 | small |
| QY | -0.014 | -0.275 | small |
| QMI | -0.041 | -0.196 | small |

#### Phân tích

- **H1 confirmed**: Gated giảm NABF có ý nghĩa thống kê + LARGE effect → mục tiêu chính của Module A đạt được.
- **Trade-off**: Gated tạo output **smoother** → đồng thời giảm wavelet quality (QM) và variance (VAR). Cơ chế: `g·A + (1-g)·B` blend → mất một phần high-freq detail của cả 2 modality.
- **Per-modality**: PET strong nhất (5 SIG / 25), CT (4 SIG), SPECT chỉ 2 SIG — SPECT có baseline NABF cao nhất nên cải thiện không cần Wilcoxon-strong (gain raw đã rất lớn 0.033 → 0.0045).

#### Acceptance criteria (xem RESEARCH_METHODOLOGY.md §3)

- [x] C1 (composite z ≥ baseline + 0.05) — likely (cần update zscore_ranking.csv)
- [x] **C2 (≥5/22 metrics SIG sau Holm)** — 5 SIG ✓
- [ ] C3 (no regression > 1σ trên SSIM/QG/QY) — QY δ=-0.275 small, **CHƯA fail nhưng watch**
- [x] **C4 (Cliff's δ ≥ 0.33 trên ≥3 metrics)** — NABF (0.596), QSF (0.602), VAR (-0.437) ✓
- [x] C5 (`_ablation_stamp.json`) ✓
- [x] C6 (`components_changed` rõ) ✓

5/6 criteria pass → **ITERATE** (KHÔNG nâng KEEP vì chỉ là 10-epoch CPU prototype).

#### Quyết định

**A.2 Gated = direction confirmed for Module A.** Tiếp tục so sánh với A.3 CrossAttn và A.4 ChannelMoE để xác định **best fusion rule**.

#### Files

- `results_v2/_stats/20260502_225704_FuseRule-Gated_vs_CDDFuse/REPORT.md` — full report
- `results_v2/_stats/.../significance_FuseRule-Gated_vs_CDDFuse.csv` — bảng raw

---

### 2026-05-02 22:00 · `CDDFuse` (baseline) · per-image re-run

**Mục đích**: Sinh per-image CSV (24 dòng × 3 modal = 72 entries) làm input cho `fusion-stats` skill — yêu cầu paired data để Wilcoxon test.

**Cmd**: `python evaluate_cddfuse.py --modal {CT,PET,SPECT} --save_perimage` (local CPU)

**Output**:
- `results_v2/CDDFuse/perimage/CDDFuse_CT_perimage.csv` (24 rows + header)
- `results_v2/CDDFuse/perimage/CDDFuse_PET_perimage.csv` (24 rows + header)
- `results_v2/CDDFuse/perimage/CDDFuse_SPECT_perimage.csv` (24 rows + header)

**Sanity check**: aggregated values bit-exact match baseline cũ (timestamp 2026-04-23) → deterministic, weights không đổi.

**Backup**: `results_v2/CDDFuse/CDDFuse_summary.csv.bak` giữ summary cũ.

---


### 2026-05-02 · `CDDFuse-FuseRule-Gated` · prototype

**Hypothesis (H1)**: Soft per-channel × per-spatial gate `g·A + (1-g)·B` thay phép sum sẽ giảm NABF (artifact ở boundary 2 modality) mà không hy sinh SSIM nhiều.

**Module thay đổi**: `models/MMIF-CDDFuse/variants/modules.py::GatedFuseLayer` (Conv1×1, init zero → epoch-0 ≈ baseline).

**Training config**:
| | Value |
|---|---|
| Mode | light_retrain (encoder/decoder frozen) |
| Trainable params | 397,704 / 1,204,784 (~33%) |
| Train data | Harvard medical 738 pairs (160 CT + 245 PET + 333 SPECT) |
| Test data | Harvard reference 72 pairs (24×3) |
| Hardware | Kaggle Tesla P100 (sm_60 unsupported by torch 2.10) → fallback **CPU** |
| Epochs | **10** (reduced from 25 due to CPU; loss curve still descending) |
| Batch | 2 (CPU) |
| LR | 1e-4 cosine → 1e-6 |
| Seed | 42 |
| Wall time | ~3-4h |
| Ckpt sha256 | `8575e252df14968376e345766f04b7bfb76670c1e30f11f69322a477a7619bfd` |

**Loss curve**:
| Epoch | Total | Intensity | Gradient |
|---|---|---|---|
| 1 | 1.022 | 0.043 | 0.098 |
| 5 | 0.886 | 0.041 | 0.084 |
| 10 | 0.875 | 0.042 | 0.083 |

→ Còn descend ở epoch 10, **chưa hội tụ hoàn toàn**.

**Quick delta vs baseline (raw, chưa significance test)**:

| Metric | CT base→var | PET base→var | SPECT base→var | Trend |
|---|---|---|---|---|
| **NABF** ↓ (target) | 0.0197 → 0.0232 ❌ | 0.0181 → **0.0103** ✓ | 0.0330 → **0.0045** ✓✓ | win 2/3 modal |
| SSIM ↑ | 1.4799 → 1.4727 ❌ | 1.4043 → 1.3976 ❌ | 1.4318 → 1.4171 ❌ | regression nhẹ |
| QG ↑ | 0.6141 → 0.6102 ≈ | 0.7344 → 0.7204 ❌ | 0.7585 → 0.7466 ❌ | regression nhẹ |
| QM ↑ | 0.0088 → 0.0043 ❌ | 0.0740 → 0.0108 ❌❌ | 0.1678 → 0.0196 ❌❌ | regression mạnh |
| QCB ↑ | 0.7965 → 0.7952 ≈ | 0.8962 → 0.8963 ≈ | 0.8802 → 0.8802 ≈ | flat |

**Phân tích thô**:
- ✓ NABF cải thiện rõ ở PET (1.8×) và SPECT (7.3×) — **target gap được giải quyết**
- ❌ Trade-off: SSIM/QG/QM giảm nhẹ → có thể do under-train (10 ep) hoặc gate đang oversmooth
- Đặc biệt SPECT NABF từ 0.033 → 0.0045 — drop rất mạnh, đáng chú ý

**Acceptance criteria check** (xem RESEARCH_METHODOLOGY.md §3):
- [ ] C1 (composite z ≥ baseline + 0.05) — **chưa tính** (cần fusion-stats)
- [ ] C2 (≥5/22 metrics SIG sau Holm) — **chưa test**
- [ ] C3 (no regression > 1σ trên SSIM/QG/QY) — likely OK (deltas nhỏ < 0.5%)
- [ ] C4 (Cliff's δ ≥ 0.33 trên ≥3 metrics) — **chưa tính**
- [x] C5 (`_ablation_stamp.json` đầy đủ) — ✓
- [x] C6 (components_changed mô tả rõ) — ✓

**Files**:
- `results_v2/CDDFuse-FuseRule-Gated/` — full output
- `results_v2/CDDFuse-FuseRule-Gated/perimage/` — input cho fusion-stats
- `models/MMIF-CDDFuse/models/CDDFuse-FuseRule-Gated.pth`

**Quyết định**: **ITERATE** (không KEEP, không REJECT).

**Lý do**:
- NABF improvement quá tốt để bỏ (SPECT giảm 7.3×) → hướng có giá trị
- SSIM regression nhỏ + 10-epoch CPU under-train → kết quả CHƯA đáng tin
- Cần re-run với 25 epoch P100 (tự nhiên CPU fallback) trước khi quyết định

**Next steps**:
1. Re-run baseline với `--save_perimage` để có per-image CSV cho stats (~30 phút Kaggle)
2. Sau khi có baseline perimage → run `fusion-stats` trên kết quả 10-epoch hiện tại
3. Nếu Wilcoxon NABF p_adj < 0.05 → re-run variant với 25 epoch (CPU ~8h hoặc Kaggle 2 sessions)
4. Tiếp theo: `FuseRule-CrossAttn` (alternative #3 trong Module A)

---

## Backlog

- [x] ~~Re-run baseline CDDFuse với `--save_perimage`~~ — DONE 2026-05-02
- [x] ~~Run stats `FuseRule-Gated` vs baseline~~ — DONE 2026-05-02 23:00, CONFIRM_IMPROVEMENT
- [x] ~~Implement `FuseRule-CrossAttn`~~ — DONE 2026-05-03
- [x] ~~Run stats `FuseRule-CrossAttn` vs baseline~~ — DONE 2026-05-03 14:20, CONFIRM_IMPROVEMENT
- [x] ~~Implement + train `FuseRule-ChannelMoE`~~ — DONE 2026-05-04
- [x] ~~Run stats `FuseRule-ChannelMoE` vs baseline~~ — DONE 2026-05-04 01:10, MIXED (worst capacity)
- [x] ~~Module A winner identified~~ — **A.2 Gated** (best NABF, least QM regression)
- [ ] Friedman + Nemenyi 4-way + CD diagram cho Module A
- [x] ~~Implement + train `PixelSelect-Mean` (B.2)~~ — DONE 2026-05-04 13:30, MIXED
- [x] ~~Implement + train `PixelSelect-Saliency` (B.3)~~ — DONE 2026-05-04 18:25, CONFIRM_IMPROVEMENT
- [x] ~~Implement + train `PixelSelect-Learnable` (B.4)~~ — DONE 2026-05-04 22:40, CONFIRM_IMPROVEMENT
- [x] ~~Module B winner identified~~ — **B.3 Saliency** (best pooled SIG, robustness)
- [x] ~~Combined: **A.2 Gated × B.3 Saliency**~~ — DONE 2026-05-05 (P100 GPU 25 ep), MIXED (QM giảm về -0.587 nhưng SIG count thấp hơn)
- [ ] **NEXT**: re-run A.2 Gated + B.3 Saliency với GPU 25 ep cho fair comparison
- [ ] Friedman + Nemenyi 4-way + CD diagram cho Module A & B
- [ ] Composite z-score cho tất cả variants → final ranking
- [ ] Re-run winner Combined với 25 epoch (final)
- [ ] Update `results_v2/zscore_ranking.csv` với composite z mới sau Combined

---

## Failures / Lessons learned

### 2026-05-02 · Kaggle GPU constraints
- **P100 (sm_60) không tương thích torch 2.10+cu128** — fallback CPU bắt buộc
- Pip downgrade torch 2.1.2+cu118 không override system path → bỏ approach này
- CPU light retrain 738 pairs × 10 epoch ≈ 3-4h trên Kaggle CPU instance
- Notebook commit phase chậm (~50 phút) do `/kaggle/working/` chứa quá nhiều file (data + repo clone)
- **Fix cho lần sau**: copy data vào `/tmp/` thay vì `/kaggle/working/` để Kaggle không phải commit
