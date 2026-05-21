# Statistical comparison: `CDDFuse-AG-45ep` vs `CDDFuse-Paper-MIF` baseline

**Run ID**: 20260521_084314
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260521_084314_AG-45ep_vs_CDDFuse-Paper-MIF\significance_AG-45ep_vs_CDDFuse-Paper-MIF.csv`

## Bottom line

- Verdict: **CONFIRM_IMPROVEMENT**
- Pooled (ALL): 8/25 metrics SIG, 0 marginal, 17 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QM | +0.0437 | +0.285 | small | 0.0000 | 0.0000 | **SIG** |
| CE | -0.0955 | +0.169 | small | 0.0000 | 0.0000 | **SIG** |
| SF | +1.0229 | +0.084 | trivial | 0.0000 | 0.0000 | **SIG** |
| QMI | +0.0109 | +0.079 | trivial | 0.0002 | 0.0039 | **SIG** |
| AG | +0.3276 | +0.076 | trivial | 0.0000 | 0.0000 | **SIG** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QY | -0.0141 | -0.349 | medium | 1.0000 | 1.0000 | NS |
| NABF | +0.0054 | -0.328 | small | 1.0000 | 1.0000 | NS |
| SSIM | -0.0243 | -0.135 | trivial | 1.0000 | 1.0000 | NS |
| RMSE | +1.0951 | -0.131 | trivial | 1.0000 | 1.0000 | NS |
| PSNR | -0.4377 | -0.128 | trivial | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 8 | 1 | 16 | 25 |
| PET | 6 | 1 | 18 | 25 |
| SPECT | 5 | 6 | 14 | 25 |
| ALL | 8 | 0 | 17 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
