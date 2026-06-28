"""
Color visual comparison PET-MRI: 4 rows, white background, no stretch.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RES  = ROOT / "results_v2"
DSET = ROOT / "Havard-Medical-Image-Fusion-Datasets-main" / \
       "Havard-Medical-Image-Fusion-Datasets-main" / "MyDatasets" / "SPECT-MRI"
OUT  = ROOT / "report_latex" / "figures"

ID = "40031"

def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))

mri_arr  = load_rgb(DSET / "test" / "MRI"   / f"{ID}.png")
pet_arr  = load_rgb(DSET / "test" / "SPECT" / f"{ID}.png")

MODEL_PATHS = {
    "NestFuse":           RES / "NestFuse"           / "Fusion" / f"{ID}.png",
    "TUFusion":           RES / "TUFusion"           / "Fusion" / f"{ID}.png",
    "WaveFusion":         RES / "WaveFusion"          / "Fusion" / f"{ID}.png",
    "DDBFusion":          RES / "DDBFusion"           / "Fusion" / f"{ID}.png",
    "GeSeNet":            RES / "GeSeNet"             / "Fusion" / f"{ID}.png",
    "CDDFuse":            RES / "CDDFuse-Paper-MIF"   / "Fusion" / f"{ID}.png",
    "CDDFuse-AG\n(Ours)": RES / "AsymD-Comb-WAvg-SML" / "Fusion" / f"{ID}.png",
}

IMGS = {"MRI": mri_arr, "SPECT": pet_arr}
for k, p in MODEL_PATHS.items():
    IMGS[k] = load_rgb(p)

# Crop: thalamus/basal ganglia region (center, high SPECT activity)
CR, CC, CS = 128, 128, 72

BORDER = {"MRI": "#2E86AB", "SPECT": "#C0392B", "CDDFuse-AG\n(Ours)": "#27AE60"}
TITLE_COLOR = {"MRI": "#2E86AB", "SPECT": "#C0392B", "CDDFuse-AG\n(Ours)": "#27AE60"}

G1 = ["MRI", "SPECT", "NestFuse", "TUFusion", "WaveFusion"]
G2 = ["DDBFusion", "GeSeNet", "CDDFuse", "CDDFuse-AG\n(Ours)"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   18,
})

n_cols = max(len(G1), len(G2))
fig, axes = plt.subplots(4, n_cols, figsize=(14, 9.5),
    gridspec_kw={"height_ratios": [3, 1.2, 3, 1.2],
                 "hspace": 0.18, "wspace": 0.04})
fig.patch.set_facecolor("white")

def render_group(group, row_full, row_crop):
    for ci, label in enumerate(group):
        img = IMGS[label]
        af = axes[row_full, ci]
        ac = axes[row_crop, ci]

        af.set_facecolor("white")
        ac.set_facecolor("white")

        af.imshow(img, aspect="equal")
        af.set_xticks([]); af.set_yticks([])

        rect = patches.Rectangle((CC-CS//2, CR-CS//2), CS, CS,
                                  lw=1.5, edgecolor="#FFD700",
                                  facecolor="none", zorder=5)
        af.add_patch(rect)

        bc = BORDER.get(label, "#666666")
        lw = 3.0 if label in BORDER else 1.5
        for sp in af.spines.values(): sp.set_color(bc); sp.set_linewidth(lw)

        tc = TITLE_COLOR.get(label, "#222222")
        fw = "bold" if label in BORDER else "normal"
        if "Ours" in label:
            af.set_title("CDDFuse-AG\n(Ours)", fontsize=20,
                         color="#27AE60", pad=4, fontweight="bold")
        else:
            af.set_title(label, fontsize=20, color=tc, pad=4, fontweight=fw)

        crop = img[CR-CS//2: CR+CS//2, CC-CS//2: CC+CS//2]
        ac.imshow(crop, aspect="equal")
        ac.set_xticks([]); ac.set_yticks([])
        for sp in ac.spines.values(): sp.set_color("#FFD700"); sp.set_linewidth(1.8)

    for ci in range(len(group), n_cols):
        axes[row_full, ci].set_visible(False)
        axes[row_crop, ci].set_visible(False)

render_group(G1, row_full=0, row_crop=1)
render_group(G2, row_full=2, row_crop=3)

axes[0, 0].set_ylabel("Ảnh đầy đủ", fontsize=16, color="#555", labelpad=4)
axes[1, 0].set_ylabel("Vùng\nphóng to", fontsize=16, color="#DAA520", labelpad=4)
axes[2, 0].set_ylabel("Ảnh đầy đủ", fontsize=16, color="#555", labelpad=4)
axes[3, 0].set_ylabel("Vùng\nphóng to", fontsize=16, color="#DAA520", labelpad=4)

fig.add_artist(mlines.Line2D(
    [0.02, 0.98], [0.505, 0.505],
    transform=fig.transFigure, color="#cccccc", lw=1.2, ls="--"))

fig.savefig(OUT / "fig_visual_comparison_spect.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved fig_visual_comparison_pet.png")
