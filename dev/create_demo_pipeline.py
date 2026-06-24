"""
Demo pipeline figure: MRI + CT → CDDFuse-AG → Fused
Real Harvard MIF test images, pair 32010 (CT-MRI).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DSET = ROOT / "Havard-Medical-Image-Fusion-Datasets-main" / \
       "Havard-Medical-Image-Fusion-Datasets-main" / "CT-MRI"
RES  = ROOT / "results_v2" / "AsymD-Comb-WAvg-SML" / "Fusion"
OUT  = ROOT / "report_latex" / "figures"

ID    = "32010"
mri   = np.array(Image.open(DSET / "MRI" / f"{ID}.png").convert("L"))
ct    = np.array(Image.open(DSET / "CT"  / f"{ID}.png").convert("L"))
fused = np.array(Image.open(RES  / f"{ID}.png").convert("L"))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   10,
    "mathtext.fontset": "stix",
})

C_BLUE   = "#2E86AB"
C_RED    = "#C0392B"
C_DARK   = "#2C3E50"
C_GREEN  = "#27AE60"
C_YELLOW = "#F39C12"
C_PURPLE = "#8E44AD"
C_ORANGE = "#E67E22"
C_TEAL   = "#16A085"

# ─── Canvas ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5.0), facecolor="white")

# Three image axes + center diagram
ax_mri   = fig.add_axes([0.01, 0.55, 0.13, 0.40])   # MRI top-left
ax_ct    = fig.add_axes([0.01, 0.08, 0.13, 0.40])   # CT  bottom-left
ax_diag  = fig.add_axes([0.17, 0.02, 0.63, 0.96])   # center diagram
ax_fused = fig.add_axes([0.83, 0.28, 0.15, 0.44])   # fused right

# ─── Source images ─────────────────────────────────────────────────────────────
for ax, img, title, col in [
    (ax_mri,   mri,   "MRI",  C_BLUE),
    (ax_ct,    ct,    "CT",   C_RED),
    (ax_fused, fused, "Fused (CDDFuse-AG)", C_GREEN),
]:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(col); sp.set_linewidth(2.8)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=col, pad=4)

ax_mri.text(0.5, -0.08, "mô mềm não", ha="center", va="top",
            transform=ax_mri.transAxes, fontsize=8.5, color="#555", style="italic")
ax_ct.text(0.5, -0.08, "xương & mô cứng", ha="center", va="top",
           transform=ax_ct.transAxes, fontsize=8.5, color="#555", style="italic")

# ─── Center diagram ────────────────────────────────────────────────────────────
ax = ax_diag
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis("off")

def box(x, y, w, h, txt, fc, ec="#ffffff", fs=9, tc="white", lw=1.5, alpha=1.0):
    bb = FancyBboxPatch((x-w/2, y-h/2), w, h,
                        boxstyle="round,pad=0.04,rounding_size=0.15",
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        zorder=3, alpha=alpha)
    ax.add_patch(bb)
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold", zorder=4, multialignment="center")

def arrow(x0, y0, x1, y1, col="#666", lw=1.8, double=False):
    style = "<|-|>" if double else "-|>"
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle=style, color=col, lw=lw), zorder=2)

# Background panel
bg = FancyBboxPatch((0.15, 0.3), 9.7, 9.4,
                    boxstyle="round,pad=0.05,rounding_size=0.2",
                    facecolor="#f4f6f8", edgecolor="#bdc3c7",
                    linewidth=1.5, zorder=0)
ax.add_patch(bg)

# Title
ax.text(5.0, 9.45, "CDDFuse-AG — Luồng xử lý tổng hợp ảnh",
        ha="center", fontsize=12, fontweight="bold", color=C_DARK, zorder=5)

# ── Encoder row ───────────────────────────────────────────────────────────────
box(5.0, 8.2, 5.5, 0.85, "Encoder  (Restormer + BaseFeatureExtraction + DetailFeatureExtraction)",
    C_DARK, fs=8.5)

# Input labels
ax.text(2.0, 9.0, "MRI", ha="center", fontsize=9.5, color=C_BLUE, fontweight="bold")
ax.text(8.0, 9.0, "CT", ha="center", fontsize=9.5, color=C_RED, fontweight="bold")
arrow(2.0, 8.8, 3.2, 8.6, col=C_BLUE, lw=2.0)
arrow(8.0, 8.8, 6.8, 8.6, col=C_RED, lw=2.0)

# ── Feature split ─────────────────────────────────────────────────────────────
box(2.5, 6.75, 3.5, 0.75, "$f^B_{MRI},\\;f^B_{CT}$  —  Base (tần số thấp)", "#2980b9", fs=8.5)
box(7.5, 6.75, 3.5, 0.75, "$f^D_{MRI},\\;f^D_{CT}$  —  Detail (tần số cao)", C_ORANGE, tc=C_DARK, fs=8.5)

arrow(3.5, 7.77, 2.9, 7.12, col="#555")
arrow(6.5, 7.77, 7.1, 7.12, col="#555")

# Asymmetric label
ax.text(5.0, 6.92, "Fusion Layer bất đối xứng", ha="center", fontsize=8,
        color="#888", style="italic")

# ── Fusion rules ──────────────────────────────────────────────────────────────
box(2.5, 5.35, 3.8, 0.9,
    "WAvg  (1 tham số)\n$2(\\alpha f^B_{MRI} + (1-\\alpha)f^B_{CT})$",
    C_YELLOW, tc=C_DARK, fs=8)
box(7.5, 5.35, 3.8, 0.9,
    "SML  (không tham số học)\n$2(w_a \\odot f^D_{MRI} + (1-w_a)\\odot f^D_{CT})$",
    C_PURPLE, fs=8)

arrow(2.5, 6.37, 2.5, 5.80, col="#555")
arrow(7.5, 6.37, 7.5, 5.80, col="#555")

ax.text(2.5, 4.50, "(1)", ha="center", fontsize=8, color="#999", style="italic")
ax.text(7.5, 4.50, "(2)", ha="center", fontsize=8, color="#999", style="italic")

# ── FuseLayers ────────────────────────────────────────────────────────────────
box(2.5, 4.0, 3.5, 0.72, "BaseFuseLayer\n(BaseFeatureExtraction)", C_DARK, fs=8)
box(7.5, 4.0, 3.5, 0.72, "DetailFuseLayer\n(DetailFeatureExtraction)", C_DARK, fs=8)

arrow(2.5, 4.90, 2.5, 4.36, col="#555")
arrow(7.5, 4.90, 7.5, 4.36, col="#555")

# ── $F_B$ and $F_D$ labels ────────────────────────────────────────────────────
ax.text(2.5, 3.40, "$F_B$", ha="center", fontsize=11, color="#2980b9", fontweight="bold")
ax.text(7.5, 3.40, "$F_D$", ha="center", fontsize=11, color=C_ORANGE, fontweight="bold")
arrow(2.5, 3.64, 2.5, 3.50, col="#555", lw=1.2)
arrow(7.5, 3.64, 7.5, 3.50, col="#555", lw=1.2)

# ── Channel reduce + Decoder ──────────────────────────────────────────────────
box(5.0, 2.5, 6.5, 0.85,
    "cat$(F_B, F_D)$ → reduce\\_channel Conv$_{1\\times1}$ → Decoder (Restormer)",
    C_TEAL, fs=8.5)

arrow(2.5, 3.2, 3.6, 2.85, col="#555")
arrow(7.5, 3.2, 6.4, 2.85, col="#555")

# ── Output ────────────────────────────────────────────────────────────────────
box(5.0, 1.35, 3.2, 0.72, "Ảnh tổng hợp  $F$", C_GREEN, fs=9.5)
arrow(5.0, 2.07, 5.0, 1.71, col=C_GREEN, lw=2.2)

# Sidebar annotations
ax.annotate("", xy=(0.3, 6.75), xytext=(0.3, 3.65),
            arrowprops=dict(arrowstyle="<->", color="#bbb", lw=1.2))
ax.text(0.15, 5.2, "Fusion\nLayer", ha="center", fontsize=7.5,
        color="#aaa", style="italic", rotation=90)

# ─── Figure-level arrows (diagram ↔ images) ────────────────────────────────────
kw = dict(transform=fig.transFigure, color="#999", lw=2,
          arrowstyle="-|>", mutation_scale=13)
fig.add_artist(FancyArrowPatch((0.145, 0.75), (0.20, 0.78), **kw))   # MRI→pipe
fig.add_artist(FancyArrowPatch((0.145, 0.28), (0.20, 0.46), **kw))   # CT→pipe
fig.add_artist(FancyArrowPatch((0.800, 0.50), (0.828, 0.50), **kw))  # pipe→fused

fig.savefig(OUT / "fig_demo_pipeline.png", dpi=160,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved fig_demo_pipeline.png")
