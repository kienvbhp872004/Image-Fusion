# Statistical comparison: `CDDFuse-Combined-Paper-MIF` vs `CDDFuse` baseline

**Run ID**: 20260520_093911
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260520_093911_Combined-Paper-MIF_vs_CDDFuse\significance_Combined-Paper-MIF_vs_CDDFuse.csv`

## Bottom line

- Verdict: **MIXED**
- Pooled (ALL): 2/25 metrics SIG, 1 marginal, 22 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QSF | +0.0075 | +0.217 | small | 0.0044 | 0.1009 | **MARGINAL** |
| NABF | -0.0023 | +0.174 | small | 0.0754 | 1.0000 | **NS** |
| FMI | +0.0073 | +0.081 | trivial | 0.0000 | 0.0000 | **SIG** |
| QG | +0.0080 | +0.075 | trivial | 0.0013 | 0.0322 | **SIG** |
| QMI | -0.0203 | +0.061 | trivial | 0.7937 | 1.0000 | **NS** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| VAR | -10.7076 | -0.468 | medium | 1.0000 | 1.0000 | NS |
| MI | -11.4657 | -0.403 | medium | 1.0000 | 1.0000 | NS |
| SSIM | -0.0507 | -0.291 | small | 1.0000 | 1.0000 | NS |
| QCV | +69.9086 | -0.244 | small | 1.0000 | 1.0000 | NS |
| QS | -0.0330 | -0.169 | small | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 9 | 0 | 16 | 25 |
| PET | 11 | 0 | 14 | 25 |
| SPECT | 2 | 1 | 22 | 25 |
| ALL | 2 | 1 | 22 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
