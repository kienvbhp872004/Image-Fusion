"""
Overlay loss function annotation onto existing phase_2.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "report_latex" / "figures" / "phase_2.png"
OUT  = ROOT / "report_latex" / "figures" / "phase_2.png"

img = np.array(Image.open(SRC).convert("RGB"))
h, w = img.shape[:2]

fig, ax = plt.subplots(figsize=(w/100, h/100 + 1.8), dpi=100)
fig.patch.set_facecolor("white")

# Draw original image in top portion
ax_img = fig.add_axes([0, 0.38, 1.0, 0.62])
ax_img.imshow(img)
ax_img.axis("off")

# Bottom area: loss function
ax_loss = fig.add_axes([0.0, 0.0, 1.0, 0.40])
ax_loss.set_xlim(0, 1)
ax_loss.set_ylim(0, 1)
ax_loss.axis("off")

# Title
ax_loss.text(0.5, 0.93, "Hàm mất mát (Phase II — end-to-end)",
             ha="center", va="top", fontsize=10, fontweight="bold", color="#333333",
             fontfamily="serif")

# Separator line
ax_loss.axhline(0.83, color="#cccccc", lw=1.2)

# Loss formula box
FORMULA_BG = "#FFF8E7"
BORDER_C   = "#E8A020"

loss_box = mpatches.FancyBboxPatch((0.03, 0.30), 0.94, 0.46,
    boxstyle="round,pad=0.015", linewidth=1.6,
    edgecolor=BORDER_C, facecolor=FORMULA_BG, zorder=2)
ax_loss.add_patch(loss_box)

# Main formula
ax_loss.text(0.50, 0.72,
    r"$\mathcal{L} = \mathcal{L}_{\mathrm{int}} + \mathcal{L}_{\mathrm{grad}} + \alpha_4\,\mathcal{L}_{\mathrm{decomp}}$",
    ha="center", va="center", fontsize=13, color="#1a1a1a",
    fontfamily="serif")

# Sub-terms
COL = ["#1565C0", "#B71C1C", "#2E7D32"]
TERMS = [
    (r"$\mathcal{L}_{\mathrm{int}} = \|F - \max(I,V)\|_1$",     "#1565C0", 0.17),
    (r"$\mathcal{L}_{\mathrm{grad}} = \|\nabla F - \max(\nabla I,\nabla V)\|_1$",  "#B71C1C", 0.50),
    (r"$\mathcal{L}_{\mathrm{decomp}}$: ràng buộc Base/Detail",  "#2E7D32", 0.83),
]

for txt, col, x in TERMS:
    ax_loss.text(x, 0.40, txt, ha="center", va="center",
                 fontsize=9.5, color=col, fontfamily="serif")

# Arrow from F label down to loss box — drawn in ax_loss coords (approximate)
ax_loss.annotate("", xy=(0.91, 0.78), xytext=(0.91, 0.98),
                 arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3),
                 xycoords="axes fraction", textcoords="axes fraction")
ax_loss.text(0.905, 0.91, r"$F$", fontsize=9, color="#555", ha="center",
             fontfamily="serif", style="italic")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {OUT}")
