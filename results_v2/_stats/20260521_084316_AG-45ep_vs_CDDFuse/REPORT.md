# Statistical comparison: `CDDFuse-AG-45ep` vs `CDDFuse` baseline

**Run ID**: 20260521_084316
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260521_084316_AG-45ep_vs_CDDFuse\significance_AG-45ep_vs_CDDFuse.csv`

## Bottom line

- Verdict: **CONFIRM_IMPROVEMENT**
- Pooled (ALL): 6/25 metrics SIG, 2 marginal, 17 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| CE | -0.1491 | +0.220 | small | 0.0000 | 0.0001 | **SIG** |
| QM | +0.0289 | +0.186 | small | 0.0050 | 0.0950 | **MARGINAL** |
| QSF | +0.0133 | +0.173 | small | 0.0011 | 0.0223 | **SIG** |
| SF | +1.8795 | +0.158 | small | 0.0000 | 0.0001 | **SIG** |
| QG | +0.0180 | +0.118 | trivial | 0.0000 | 0.0000 | **SIG** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| VAR | -9.5069 | -0.424 | medium | 1.0000 | 1.0000 | NS |
| SSIM | -0.0730 | -0.409 | medium | 1.0000 | 1.0000 | NS |
| MI | -10.9803 | -0.382 | medium | 1.0000 | 1.0000 | NS |
| QCV | +98.4463 | -0.273 | small | 1.0000 | 1.0000 | NS |
| NABF | +0.0042 | -0.255 | small | 0.9911 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 8 | 4 | 13 | 25 |
| PET | 9 | 1 | 15 | 25 |
| SPECT | 2 | 1 | 22 | 25 |
| ALL | 6 | 2 | 17 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
