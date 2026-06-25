"""Generate README assets: comparison bar charts + z-score ranking chart."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# ── 1. Load per-modal data ───────────────────────────────────────────────────
CT  = pd.read_csv(os.path.join(ROOT, "results_v2/_compare_vs_sota/per_modal_CT.csv"))
PET = pd.read_csv(os.path.join(ROOT, "results_v2/_compare_vs_sota/per_modal_PET.csv"))
SPC = pd.read_csv(os.path.join(ROOT, "results_v2/_compare_vs_sota/per_modal_SPECT.csv"))

# Map internal col names → display names
# QG = Qabf (gradient-based quality metric)
COL_MAP = {"AG": "AG", "SF": "SF", "EI": "EI", "QG": "Qabf", "QM": "QM", "QMI": "QMI"}
METRICS = list(COL_MAP.keys())
DISP    = list(COL_MAP.values())

# Methods to show (short name + row key in CSV)
SHOW = {
    "CDDFuse-AG\n(ours)": "CDDFuse-AG-45ep",
    "CDDFuse": "CDDFuse",
    "MM-Net\nFusion": "MM-Net-Fusion",
    "MFS\nFusion": "MFS-Fusion",
    "CM-CSAM\nFNet": "CM-CSAMFNet",
    "MMIF\nINet": "MMIF-INet",
}

COLORS = ["#D62728", "#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD", "#8C564B"]

def get_row(df, key):
    r = df[df["model"] == key]
    return r.iloc[0] if len(r) else None

def make_grouped_bar(modal_label, df, ax_row, axes):
    x = np.arange(len(METRICS))
    n = len(SHOW)
    w = 0.13
    for i, (label, key) in enumerate(SHOW.items()):
        row = get_row(df, key)
        if row is None:
            continue
        vals = [float(row[m]) for m in METRICS]
        offset = (i - n/2 + 0.5) * w
        bars = axes.bar(x + offset, vals, w, label=label, color=COLORS[i], alpha=0.88, edgecolor="white", linewidth=0.4)
        # bold the "ours" bar
        if i == 0:
            for b in bars:
                b.set_edgecolor("#8B0000")
                b.set_linewidth(1.2)

    axes.set_xticks(x)
    axes.set_xticklabels(DISP, fontsize=9)
    axes.set_title(f"MRI-{modal_label}", fontsize=11, fontweight="bold", pad=4)
    axes.tick_params(axis="y", labelsize=8)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
fig.suptitle("CDDFuse-AG vs SOTA — 6 Best Metrics (SF · Qabf · AG · EI · QM · QMI)", fontsize=12, fontweight="bold", y=1.01)

make_grouped_bar("CT",    CT,  0, axes[0])
make_grouped_bar("PET",   PET, 1, axes[1])
make_grouped_bar("SPECT", SPC, 2, axes[2])

handles = [mpatches.Patch(color=COLORS[i], label=label) for i, label in enumerate(SHOW.keys())]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.08), frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "comparison_6metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved comparison_6metrics.png")

# ── 2. Z-score ranking bar chart ─────────────────────────────────────────────
zdf = pd.read_csv(os.path.join(ROOT, "results_v2/zscore_ranking.csv"))
zdf = zdf.sort_values("composite_z", ascending=True).reset_index(drop=True)

bar_colors = ["#D62728" if m in ("CDDFuse-AG-45ep",) else
              "#FF9999" if m == "CDDFuse" else "#AAAAAA"
              for m in zdf["model"]]

# Replace internal names with display names
NAME_MAP = {
    "CDDFuse-AG-45ep": "CDDFuse-AG (ours)",
    "CDDFuse": "CDDFuse (paper)",
}
labels = [NAME_MAP.get(m, m) for m in zdf["model"]]

fig2, ax2 = plt.subplots(figsize=(8, 7))
bars = ax2.barh(range(len(zdf)), zdf["composite_z"], color=bar_colors, edgecolor="white", linewidth=0.4)
ax2.set_yticks(range(len(zdf)))
ax2.set_yticklabels(labels, fontsize=8.5)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Composite Z-score (22 metrics, 72 test pairs)", fontsize=9)
ax2.set_title("SOTA Ranking — Composite Z-score\n(↑ higher is better)", fontsize=11, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.6)

# Annotate values
for i, v in enumerate(zdf["composite_z"]):
    ax2.text(v + 0.01 if v >= 0 else v - 0.01, i, f"{v:.3f}",
             va="center", ha="left" if v >= 0 else "right", fontsize=7)

ours  = mpatches.Patch(color="#D62728", label="CDDFuse-AG (ours)")
base  = mpatches.Patch(color="#FF9999", label="CDDFuse (paper)")
other = mpatches.Patch(color="#AAAAAA", label="Other SOTA")
ax2.legend(handles=[ours, base, other], fontsize=8, loc="lower right")

plt.tight_layout()
fig2.savefig(os.path.join(OUT, "zscore_ranking.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved zscore_ranking.png")

# ── 3. Print markdown tables ──────────────────────────────────────────────────
def print_table(modal, df):
    rows_want = list(SHOW.values())
    sub = df[df["model"].isin(rows_want)].copy()
    sub["_order"] = sub["model"].apply(lambda m: rows_want.index(m) if m in rows_want else 99)
    sub = sub.sort_values("_order").reset_index(drop=True)
    cols = ["model"] + METRICS
    sub = sub[cols]
    sub.columns = ["Method"] + DISP
    print(f"\n### MRI-{modal}")
    print(sub.to_markdown(index=False, floatfmt=".3f"))

print_table("CT",    CT)
print_table("PET",   PET)
print_table("SPECT", SPC)
