"""
Generate method chapter figures for CDDFuse-AG thesis.
Outputs to report_latex/figures/
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy.ndimage import gaussian_filter

OUT = Path(__file__).resolve().parent.parent / "report_latex" / "figures"

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        10,
    "axes.titlesize":   11,
    "mathtext.fontset": "stix",
})

C_BLUE   = "#2E86AB"
C_GREEN  = "#4CAF50"
C_ORANGE = "#F4A261"
C_RED    = "#E76F51"
C_GRAY   = "#CCCCCC"
C_DARK   = "#264653"
C_YELLOW = "#E9C46A"
C_PURPLE = "#9B72CF"

def box(ax, x, y, w, h, text, color, fontsize=9, textcolor="white", radius=0.03):
    bb = FancyBboxPatch((x-w/2, y-h/2), w, h,
                        boxstyle=f"round,pad=0.01,rounding_size={radius}",
                        facecolor=color, edgecolor="white", linewidth=1.2, zorder=3)
    ax.add_patch(bb)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight="bold", zorder=4, multialignment="center")

def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.5):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=2)

# ==============================================================================
# Figure 1: CDDFuse vs CDDFuse-AG architecture comparison
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.patch.set_facecolor("white")

for ax, subtitle, base_rule, detail_rule, hl in zip(
    axes,
    ["(a) CDDFuse goc", "(b) CDDFuse-AG (De xuat)"],
    ["Sum", "WAvg (hoc duoc)"],
    ["Sum", "SML (sac net cuc bo)"],
    [False, True],
):
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(5, 7.7, subtitle, ha="center", va="center", fontsize=11,
            fontweight="bold", color=C_BLUE if hl else C_DARK)

    box(ax, 1.2, 6.5, 1.6, 0.7, "MRI\n(Source A)", C_BLUE, 8)
    box(ax, 1.2, 5.4, 1.6, 0.7, "CT/PET/SPECT\n(Source B)", C_RED, 8)
    box(ax, 3.3, 6.5, 1.5, 0.65, "Encoder\n(dong bang)", C_DARK, 8)
    box(ax, 3.3, 5.4, 1.5, 0.65, "Encoder\n(dong bang)", C_DARK, 8)
    arrow(ax, 2.0, 6.5, 2.55, 6.5)
    arrow(ax, 2.0, 5.4, 2.55, 5.4)

    box(ax, 5.1, 7.1, 1.3, 0.55, "Base A", C_GREEN, 8, C_DARK)
    box(ax, 5.1, 5.9, 1.3, 0.55, "Detail A", C_ORANGE, 8, C_DARK)
    box(ax, 5.1, 4.8, 1.3, 0.55, "Base B", C_GREEN, 8, C_DARK)
    box(ax, 5.1, 3.6, 1.3, 0.55, "Detail B", C_ORANGE, 8, C_DARK)
    arrow(ax, 4.05, 6.5, 4.45, 7.1)
    arrow(ax, 4.05, 6.5, 4.45, 5.9)
    arrow(ax, 4.05, 5.4, 4.45, 4.8)
    arrow(ax, 4.05, 5.4, 4.45, 3.6)

    bc = C_YELLOW if hl else C_GRAY
    dc = C_PURPLE if hl else C_GRAY
    box(ax, 6.9, 6.0, 1.5, 0.65, f"Base Rule:\n{base_rule}", bc, 8, C_DARK)
    box(ax, 6.9, 4.3, 1.5, 0.65, f"Detail Rule:\n{detail_rule}", dc, 8, C_DARK)
    arrow(ax, 5.75, 7.1, 6.15, 6.2)
    arrow(ax, 5.75, 4.8, 6.15, 6.0)
    arrow(ax, 5.75, 5.9, 6.15, 4.3)
    arrow(ax, 5.75, 3.6, 6.15, 4.1)

    if hl:
        ax.text(6.9, 6.38, "(1) 1 tham so", ha="center", fontsize=7.5,
                color=C_DARK, style="italic")
        ax.text(6.9, 4.68, "(2) Khong tham so", ha="center", fontsize=7.5,
                color=C_DARK, style="italic")

    box(ax, 8.3, 6.0, 1.2, 0.55, "Base\nFuseLayer", C_DARK, 8)
    box(ax, 8.3, 4.3, 1.2, 0.55, "Detail\nFuseLayer", C_DARK, 8)
    arrow(ax, 7.65, 6.0, 7.7, 6.0)
    arrow(ax, 7.65, 4.3, 7.7, 4.3)

    merge_lbl = "Pixel\nSelect" if hl else "Channel\nReduce"
    merge_c   = C_BLUE if hl else C_GRAY
    box(ax, 8.3, 2.9, 1.2, 0.55, merge_lbl, merge_c, 8)
    arrow(ax, 8.3, 5.72, 8.3, 3.18)
    arrow(ax, 8.3, 4.02, 8.3, 3.18)
    box(ax, 8.3, 1.7, 1.2, 0.55, "Decoder\n(dong bang)", C_DARK, 8)
    arrow(ax, 8.3, 2.62, 8.3, 1.98)
    box(ax, 8.3, 0.7, 1.4, 0.6, "Anh fused F", C_BLUE if hl else C_DARK, 9)
    arrow(ax, 8.3, 1.42, 8.3, 1.0)

axes[1].text(0.3, 0.38, "Vang: WAvg - 1 scalar alpha hoc duoc",
             fontsize=8, color=C_DARK, style="italic")
axes[1].text(0.3, 0.12, "Tim: SML - spatial selection, khong tham so",
             fontsize=8, color=C_DARK, style="italic")

fig.tight_layout(pad=0.5)
out = OUT / "fig_cddfuse_ag_arch.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out.name}")

# ==============================================================================
# Figure 2: WAvg rule illustration
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)
ax.axis("off"); fig.patch.set_facecolor("white")

box(ax, 1.0, 3.2, 1.4, 0.65, "Base A\n$f_V^B$", C_BLUE, 10)
box(ax, 1.0, 1.5, 1.4, 0.65, "Base B\n$f_I^B$", C_RED, 10)

box(ax, 3.6, 2.35, 2.0, 0.85,
    "Scalar $\\alpha = \\sigma(\\theta)$\n$\\theta \\in \\mathbb{R}$: 1 tham so hoc",
    C_YELLOW, 9, C_DARK)

arrow(ax, 1.7, 3.2, 2.6, 2.7)
arrow(ax, 1.7, 1.5, 2.6, 2.0)

box(ax, 6.4, 2.35, 2.4, 0.85,
    "$2(\\alpha\\cdot f_V^B + (1-\\alpha)\\cdot f_I^B)$",
    C_GREEN, 9.5, "white")
arrow(ax, 4.6, 2.35, 5.2, 2.35)

box(ax, 9.0, 2.35, 1.4, 0.65, "Base fused\n$f_F^B$", C_DARK, 10)
arrow(ax, 7.6, 2.35, 8.3, 2.35)

ax.text(5.0, 1.2,
        "$\\alpha = 0.5$ khi khoi tao ($\\theta=0$), dieu chinh qua Phase II",
        ha="center", fontsize=9, color="#555555", style="italic")
ax.text(5.0, 0.65,
        "Trung binh hoa toan cuc - phu hop nhanh Base (tan so thap, dong thuan)",
        ha="center", fontsize=9, color=C_BLUE)
ax.annotate("", xy=(4.6, 0.2), xytext=(2.6, 0.2),
            arrowprops=dict(arrowstyle="<->", color=C_ORANGE, lw=2))
ax.text(3.6, 0.03, "$\\alpha \\in (0,1)$", ha="center", fontsize=9, color=C_ORANGE)

fig.tight_layout(pad=0.8)
out = OUT / "fig_rule_wavg.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out.name}")

# ==============================================================================
# Figure 3: SML rule illustration
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                         gridspec_kw={"width_ratios": [1, 1.2]})
fig.patch.set_facecolor("white")

ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 5.2)
ax.axis("off")

box(ax, 1.1, 4.3, 1.6, 0.7, "Detail A\n$f_V^D$", C_BLUE, 9)
box(ax, 1.1, 2.3, 1.6, 0.7, "Detail B\n$f_I^D$", C_RED, 9)
box(ax, 3.6, 4.3, 1.8, 0.7, "$\\mathrm{SML}(f_V^D)$", C_ORANGE, 9, C_DARK)
box(ax, 3.6, 2.3, 1.8, 0.7, "$\\mathrm{SML}(f_I^D)$", C_ORANGE, 9, C_DARK)
arrow(ax, 1.9, 4.3, 2.7, 4.3)
arrow(ax, 1.9, 2.3, 2.7, 2.3)

box(ax, 6.2, 3.3, 2.2, 1.0,
    "$w_a = \\frac{\\mathrm{SML}(a)}{\\mathrm{SML}(a)+\\mathrm{SML}(b)+\\epsilon}$",
    C_YELLOW, 8.5, C_DARK)
arrow(ax, 4.5, 4.3, 5.1, 3.7)
arrow(ax, 4.5, 2.3, 5.1, 2.9)

box(ax, 8.8, 3.3, 1.6, 0.7, "Detail fused\n$f_F^D$", C_DARK, 9)
arrow(ax, 7.3, 3.3, 8.0, 3.3)

# Use plain text (not mathtext) for the formula with \bigl \odot etc.
ax.text(5.0, 1.5,
        r"$R_{\mathrm{SML}}(a,b) = 2(w_a \odot a + (1-w_a) \odot b)$",
        ha="center", fontsize=9, color="#333333")
ax.text(5.0, 0.75,
        "Moi pixel chon modality sac net hon tai vi tri do",
        ha="center", fontsize=9, color="#666666", style="italic")
ax.text(5.0, 0.2,
        r"$\mathrm{SML}_{ij}=\sum_{\mathcal{N}}|2f_{pq}-f_{p-1,q}-f_{p+1,q}|+|\cdots|$",
        ha="center", fontsize=8, color="#888888")

ax2 = axes[1]
np.random.seed(42)
sml_a = np.zeros((64, 64)); sml_b = np.zeros((64, 64))
sml_a[:32, :32] += 3.0; sml_a[32:, 32:] += 0.3
sml_b[:32, :32] += 0.3; sml_b[32:, 32:] += 3.0
sml_a = gaussian_filter(sml_a + np.random.rand(64, 64)*0.3, sigma=3)
sml_b = gaussian_filter(sml_b + np.random.rand(64, 64)*0.3, sigma=3)
w_a = sml_a / (sml_a + sml_b + 1e-6)

im = ax2.imshow(w_a, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cb.set_label("$w_a$ (1 = uu tien A, 0 = uu tien B)", fontsize=8.5)
ax2.set_xlabel("Cot (pixels)", fontsize=9)
ax2.set_ylabel("Hang (pixels)", fontsize=9)
ax2.set_title("Vi du weight map $w_a$", fontsize=10, pad=4)
ax2.text(6, 7, "A sac net\nhon", fontsize=8, color="white", fontweight="bold",
         bbox=dict(facecolor=C_GREEN, alpha=0.75, edgecolor="none", pad=2))
ax2.text(36, 50, "B sac net\nhon", fontsize=8, color="white", fontweight="bold",
         bbox=dict(facecolor=C_RED, alpha=0.75, edgecolor="none", pad=2))

fig.tight_layout(pad=0.8)
out = OUT / "fig_rule_sml.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out.name}")

# ==============================================================================
# Figure 4: Asymmetric principle — Base vs Detail
# Layout: top half = images, bottom = arrows + result, no overlaps
# ==============================================================================
fig = plt.figure(figsize=(13, 5.8))
fig.patch.set_facecolor("white")

# All axes using add_axes([left, bottom, width, height]) in figure coords
# Top row images: Base A, Base B | Detail A, Detail B
aBA  = fig.add_axes([0.03, 0.50, 0.20, 0.38])   # Base A
aBB  = fig.add_axes([0.25, 0.50, 0.20, 0.38])   # Base B
aDA  = fig.add_axes([0.55, 0.50, 0.20, 0.38])   # Detail A
aDB  = fig.add_axes([0.77, 0.50, 0.20, 0.38])   # Detail B
# Bottom row: result images
aBR  = fig.add_axes([0.14, 0.06, 0.20, 0.30])   # Base result
aDR  = fig.add_axes([0.66, 0.06, 0.20, 0.30])   # Detail result

np.random.seed(0)
x = np.linspace(0, 4, 60); y = np.linspace(0, 3, 45)
X, Y = np.meshgrid(x, y)
feat_a = np.exp(-((X-1.5)**2+(Y-1.5)**2)/1.2) + 0.3*np.exp(-((X-3)**2+(Y-0.8)**2)/0.8)
feat_b = np.exp(-((X-1.3)**2+(Y-1.7)**2)/1.3) + 0.25*np.exp(-((X-3.2)**2+(Y-0.9)**2)/0.7)

feat_da = np.zeros((45, 60)); feat_db = np.zeros((45, 60))
feat_da[:, 15:17] = 2.0; feat_da[:, 30:32] = 1.5; feat_da[20:22, :] = 1.0
feat_db[10:12, :] = 1.8; feat_db[30:32, :] = 1.5; feat_db[:, 45:47] = 1.0
feat_da = gaussian_filter(feat_da, 1.2)
feat_db = gaussian_filter(feat_db, 1.2)

for ax_img, feat, cmap, lbl in [
    (aBA, feat_a,  "Blues", "Base A"),
    (aBB, feat_b,  "Reds",  "Base B"),
    (aDA, feat_da, "Blues", "Detail A"),
    (aDB, feat_db, "Reds",  "Detail B"),
]:
    ax_img.imshow(feat, cmap=cmap, aspect="auto")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(lbl, fontsize=9, pad=3)
    for sp in ax_img.spines.values(): sp.set_linewidth(1.2)

aBR.imshow((feat_a + feat_b) / 2, cmap="Purples", aspect="auto")
aBR.set_xticks([]); aBR.set_yticks([])
aBR.set_title("WAvg output", fontsize=9, pad=3)

aDR.imshow(np.maximum(feat_da, feat_db), cmap="Greens", aspect="auto")
aDR.set_xticks([]); aDR.set_yticks([])
aDR.set_title("SML output", fontsize=9, pad=3)

# ── Text labels via fig.text (no overlap with images) ─────────────────────────
# Left panel header
fig.text(0.24, 0.96, "Nhanh Base (Tan so thap)", ha="center",
         fontsize=12, fontweight="bold", color=C_BLUE)
fig.text(0.24, 0.91, "Hai modality dong thuan o cau truc tong the", ha="center",
         fontsize=9.5, color="#444444", style="italic")

# Arrow label between image row and result row (left)
fig.text(0.24, 0.44, "WAvg: trung binh hoa toan cuc", ha="center",
         fontsize=9.5, color=C_BLUE, fontweight="bold")

# Right panel header
fig.text(0.76, 0.96, "Nhanh Detail (Tan so cao)", ha="center",
         fontsize=12, fontweight="bold", color=C_RED)
fig.text(0.76, 0.91, "Hai modality bo tro nhau o chi tiet dac thu", ha="center",
         fontsize=9.5, color="#444444", style="italic")

# Arrow label between image row and result row (right)
fig.text(0.76, 0.44, "SML: lua chon theo do sac net cuc bo", ha="center",
         fontsize=9.5, color=C_RED, fontweight="bold")

# Down arrows (figure coords)
for x_pos in [0.24, 0.76]:
    fig.add_artist(mlines.Line2D(
        [x_pos, x_pos], [0.48, 0.38],
        transform=fig.transFigure, color="#888888", lw=1.5,
        marker="v", markersize=7, markevery=[1]
    ))

# Vertical divider
fig.add_artist(mlines.Line2D(
    [0.5, 0.5], [0.03, 0.98],
    transform=fig.transFigure, color="#CCCCCC", lw=1.5, ls="--"
))

# Bottom key insight
fig.text(0.5, 0.005,
         "Nguyen ly: Base -> averaging  |  Detail -> activity-based selection",
         ha="center", fontsize=10, color=C_DARK, style="italic",
         bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=4))

out = OUT / "fig_asymmetric_principle.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out.name}")

print("\nAll figures generated successfully.")
