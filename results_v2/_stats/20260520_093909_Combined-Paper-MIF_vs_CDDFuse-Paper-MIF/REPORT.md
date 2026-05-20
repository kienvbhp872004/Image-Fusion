# Statistical comparison: `CDDFuse-Combined-Paper-MIF` vs `CDDFuse-Paper-MIF` baseline

**Run ID**: 20260520_093909
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260520_093909_Combined-Paper-MIF_vs_CDDFuse-Paper-MIF\significance_Combined-Paper-MIF_vs_CDDFuse-Paper-MIF.csv`

## Bottom line

- Verdict: **MIXED**
- Pooled (ALL): 2/25 metrics SIG, 1 marginal, 22 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| NABF | -0.0011 | +0.073 | trivial | 0.0000 | 0.0000 | **SIG** |
| FMI | +0.0028 | +0.042 | trivial | 0.0003 | 0.0081 | **SIG** |
| QMI | +0.0046 | +0.031 | trivial | 0.0170 | 0.3899 | **MARGINAL** |
| NCIE | +0.0000 | +0.011 | trivial | 0.4464 | 1.0000 | **NS** |
| QNCIE | +0.0000 | +0.011 | trivial | 0.4464 | 1.0000 | **NS** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QSF | -0.0040 | -0.131 | trivial | 0.9989 | 1.0000 | NS |
| VAR | -1.1286 | -0.097 | trivial | 1.0000 | 1.0000 | NS |
| QCB | -0.0090 | -0.061 | trivial | 1.0000 | 1.0000 | NS |
| MI | -1.0427 | -0.058 | trivial | 1.0000 | 1.0000 | NS |
| QY | -0.0008 | -0.057 | trivial | 0.9988 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 9 | 1 | 15 | 25 |
| PET | 2 | 0 | 23 | 25 |
| SPECT | 4 | 3 | 18 | 25 |
| ALL | 2 | 1 | 22 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
