"""
Visual comparison CT-MRI: 4 rows layout.
Row 1: full images  — MRI, CT, NestFuse, TUFusion, WaveFusion
Row 2: crops        — same 5
Row 3: full images  — DDBFusion, GeSeNet, CDDFuse, CDDFuse-AG
Row 4: crops        — same 4
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RES  = ROOT / "results_v2"
DSET = ROOT / "Havard-Medical-Image-Fusion-Datasets-main" / \
       "Havard-Medical-Image-Fusion-Datasets-main" / "CT-MRI"
OUT  = ROOT / "report_latex" / "figures"

ID = "16010"

mri_arr = np.array(Image.open(DSET / "MRI" / f"{ID}.png").convert("L"))
ct_arr  = np.array(Image.open(DSET / "CT"  / f"{ID}.png").convert("L"))

MODEL_PATHS = {
    "NestFuse":          RES / "NestFuse"          / "Fusion" / f"{ID}.png",
    "TUFusion":          RES / "TUFusion"          / "Fusion" / f"{ID}.png",
    "WaveFusion":        RES / "WaveFusion"         / "Fusion" / f"{ID}.png",
    "DDBFusion":         RES / "DDBFusion"          / "Fusion" / f"{ID}.png",
    "GeSeNet":           RES / "GeSeNet"            / "Fusion" / f"{ID}.png",
    "CDDFuse":           RES / "CDDFuse-Paper-MIF"  / "Fusion" / f"{ID}.png",
    "CDDFuse-AG\n(Ours)":RES / "AsymD-Comb-WAvg-SML"/"Fusion" / f"{ID}.png",
}

IMGS = {"MRI": mri_arr, "CT": ct_arr}
for k, p in MODEL_PATHS.items():
    IMGS[k] = np.array(Image.open(p).convert("L"))

# Crop region: right orbit (rich edge detail)
CR, CC, CS = 60, 155, 70

BORDER = {
    "MRI":               "#2E86AB",
    "CT":                "#C0392B",
    "CDDFuse-AG\n(Ours)":"#27AE60",
}
TITLE_COLOR = {
    "MRI":               "#2E86AB",
    "CT":                "#C0392B",
    "CDDFuse-AG\n(Ours)":"#27AE60",
}

# Two groups
G1 = ["MRI", "CT", "NestFuse", "TUFusion", "WaveFusion"]
G2 = ["DDBFusion", "GeSeNet", "CDDFuse", "CDDFuse-AG\n(Ours)"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   9,
})

n_cols = max(len(G1), len(G2))
fig, axes = plt.subplots(4, n_cols, figsize=(14, 7.8),
    gridspec_kw={"height_ratios": [3, 1.4, 3, 1.4],
                 "hspace": 0.10, "wspace": 0.04})
fig.patch.set_facecolor("white")

def render_group(group, row_full, row_crop, col_offset=0):
    for ci, label in enumerate(group):
        img = IMGS[label]
        af  = axes[row_full, ci + col_offset]
        ac  = axes[row_crop, ci + col_offset]

        af.imshow(img, cmap="gray", vmin=0, vmax=255)
        af.set_xticks([]); af.set_yticks([])
        rect = patches.Rectangle((CC-CS//2, CR-CS//2), CS, CS,
                                  lw=1.5, edgecolor="#FFD700",
                                  facecolor="none", zorder=5)
        af.add_patch(rect)

        bc = BORDER.get(label, "#555555")
        lw = 3.0 if label in BORDER else 1.5
        for sp in af.spines.values(): sp.set_color(bc); sp.set_linewidth(lw)

        tc = TITLE_COLOR.get(label, "#333333")
        fw = "bold" if label in BORDER else "normal"
        clean = label.replace("\n(Ours)", "")
        if "Ours" in label:
            af.set_title("CDDFuse-AG\n(Ours)", fontsize=8.5,
                         color="#27AE60", pad=3, fontweight="bold")
        else:
            af.set_title(clean, fontsize=8.5, color=tc, pad=3, fontweight=fw)

        crop = img[CR-CS//2: CR+CS//2, CC-CS//2: CC+CS//2]
        ac.imshow(crop, cmap="gray", vmin=img.min(), vmax=img.max())
        ac.set_xticks([]); ac.set_yticks([])
        for sp in ac.spines.values(): sp.set_color("#FFD700"); sp.set_linewidth(1.8)

    # Hide unused columns
    for ci in range(len(group), n_cols):
        axes[row_full, ci + col_offset].set_visible(False)
        axes[row_crop, ci + col_offset].set_visible(False)

render_group(G1, row_full=0, row_crop=1)
render_group(G2, row_full=2, row_crop=3)

# Row labels
axes[0, 0].set_ylabel("Ảnh đầy đủ", fontsize=9, color="#555", labelpad=4)
axes[1, 0].set_ylabel("Vùng\nphóng to", fontsize=9, color="#DAA520", labelpad=4)
axes[2, 0].set_ylabel("Ảnh đầy đủ", fontsize=9, color="#555", labelpad=4)
axes[3, 0].set_ylabel("Vùng\nphóng to", fontsize=9, color="#DAA520", labelpad=4)

# Separator line between group 1 and group 2
fig.add_artist(matplotlib.lines.Line2D(
    [0.02, 0.98], [0.505, 0.505],
    transform=fig.transFigure, color="#cccccc", lw=1.2, ls="--"))

import matplotlib.lines
fig.savefig(OUT / "fig_visual_comparison.png", dpi=160,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved fig_visual_comparison.png")
