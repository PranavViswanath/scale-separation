# Figures for the write-up. Every number is from R_SWEEP.md / PREREG R186-R190 (measured, 5-10 nets per cell).
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OURS, ARC, MC = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 220})

def grid(ax):
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---- Fig 1: 64x16, error vs compute (ARC's own figure format) ----
fig, ax = plt.subplots(figsize=(6.6, 4.3)); grid(ax)
F = np.logspace(5, 11.2, 100); ax.plot(F, 2 * 64**2 * 17 / F, "--", color=MC, lw=1.6)
ax.text(2.2e10, 2 * 64**2 * 17 / 2.2e10 * 0.45, "naive sampling", color=INK2, ha="center", rotation=-33)
arcF = [2.2e5, 1.4e7, 8.2e8, 6.9e10]; arcE = [6.0e-2, 4.62e-3, 3.53e-3, 1.01e-2]
ax.plot(arcF, arcE, "-o", color=ARC, lw=2, ms=7)
for f, e, k in zip(arcF, arcE, ["K1", "K2", "K3", "K4"]):
    ax.annotate(k, (f, e), textcoords="offset points", xytext=(0, 8), ha="center", color=INK2)
ax.text(0.40, 0.97, "ARC cumulant propagation\n(K4 worse than K3 on 10/10 nets;\nAUGMENT K4 diverges)", transform=ax.transAxes,
        va="top", color=INK, fontsize=9.5)
oF = [3.96e7, 1.26e9, 4.9e10]; oE = [1.52e-3, 7.7e-4, 2.76e-4]
ax.plot(oF, oE, "-o", color=OURS, lw=2, ms=7)
ax.plot([3.96e7], [1.52e-3], "o", color=OURS, ms=11, mec="white", mew=2)
ax.annotate("scale separation\n2.3x below sampling", (3.96e7, 1.52e-3), textcoords="offset points",
            xytext=(-30, -34), color=INK, fontsize=9.5)
ax.annotate("your ladder, scale as a law", (4.9e10, 2.76e-4), textcoords="offset points",
            xytext=(6, -18), ha="right", color=OURS, fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e5, 2e11); ax.set_ylim(1e-6, 1.5e-1)
ax.set_xlabel("FLOPs (ARC's counter)"); ax.set_ylabel("normalized MSE, mean of 10 random nets")
ax.set_title("Width 64, depth 16", loc="left", color=INK, fontsize=11.5)
fig.tight_layout(); fig.savefig("fig1_64x16.png")

# ---- Fig 2: the depth = width/4 diagonal, matched compute (new V rule) ----
# H (512x128) is filled in when its truth lands; until then it is omitted.
DIAG = [("64x16", 4.32e-3, 3.51e-3, 1.52e-3), ("128x32", 6.24e-3, 2.37e-3, 1.51e-3), ("256x64", 2.91e-3, 1.46e-3, 9.53e-4), ("512x128", 7.69e-4, 8.35e-4, 5.88e-4)]
cells = [c for c, *_ in DIAG]; arc = [a for _, a, _, _ in DIAG]; mc = [m for _, _, m, _ in DIAG]; us = [u for *_, u in DIAG]
x = np.arange(len(cells)); w = 0.26
fig, ax = plt.subplots(figsize=(7.0, 3.9)); grid(ax)
for i, (vals, c, lab) in enumerate([(arc, ARC, "ARC best at <= our FLOPs"), (mc, MC, "naive sampling at our FLOPs"),
                                    (us, OURS, "ours (scale separation)")]):
    ax.bar(x + (i - 1) * w, vals, w - 0.03, color=c, label=lab)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(cells); ax.set_ylim(3e-4, 1.5e-2)
ax.set_ylabel("normalized MSE (lower is better)"); ax.set_xlabel("width x depth, depth = width / 4 (10 random nets each)")
ax.text(0.99, 0.97, "bold: times better than sampling", transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3, bbox_to_anchor=(-0.02, 1.14), handlelength=1.2, columnspacing=1.0)
for i, u in enumerate(us):
    ax.annotate(f"{mc[i]/u:.1f}x", (x[i] + w, u), textcoords="offset points", xytext=(0, 4),
                ha="center", color=INK, fontsize=9.5, fontweight="bold")
fig.tight_layout(); fig.savefig("fig2_matched.png")

# ---- Fig 4: margin over sampling vs width on the diagonal, old vs exact V rule ----
wid = np.array([64, 128, 256])
old = (np.array([2.83, 1.05, 1.02]), np.array([2.01, 0.85, 0.70]), np.array([4.19, 1.34, 1.79]))
new = (np.array([2.80, 1.77, 1.55]), np.array([1.98, 1.21, 1.01]), np.array([4.34, 3.04, 2.70]))
fig, ax = plt.subplots(figsize=(6.0, 3.7)); grid(ax)
ax.axhline(1, color=MC, ls="--", lw=1.6); ax.text(70, 1.04, "naive sampling", color=INK2, fontsize=9)
for (m, lo, hi), c, lab, dx in [(old, "#a9b8c9", "leading-order scale variance", 0.97), (new, OURS, "exact scale variance", 1.03)]:
    ax.errorbar(wid * dx, m, yerr=[m - lo, hi - m], fmt="-o", color=c, lw=2, ms=7, capsize=3, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks(wid); ax.set_xticklabels(["64x16", "128x32", "256x64"])
ax.set_yscale("log"); ax.set_ylim(0.5, 6); ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5x", "1x", "2x", "4x"])
ax.set_ylabel("times better than sampling, 95% CI"); ax.set_xlabel("width x depth, depth = width / 4")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig("fig4_width.png")

# ---- Fig 3: ladders, ARC vs the same ladder with the scale carried as a law ----
data = {"64x16": ([4.62e-3, 3.53e-3, 1.01e-2], [1.54e-3, 7.7e-4, 2.76e-4]),
        "128x32": ([6.5e-3, 4.9e-3, 1.5e-1], [2.62e-3, 9.19e-4]),
        "256x64": ([3.0e-3, 4.5e-3, None], [1.43e-3, 5.42e-4])}
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True)
for ax, (cell, (a, o)) in zip(axs, data.items()):
    grid(ax); ka = [k for k, v in zip([2, 3, 4], a) if v is not None]; va = [v for v in a if v is not None]
    ax.plot(ka, va, "-o", color=ARC, lw=2, ms=7); ax.plot([2, 3, 4][:len(o)], o, "-o", color=OURS, lw=2, ms=7)
    if a[2] is None:
        ax.annotate("K4 diverges\n(3e16)", (3.75, 1.5e-2), ha="center", color=ARC, fontsize=9)
    if a[2] is not None and a[2] > 0.05:
        ax.annotate("K4 = 0.15\n(off scale)", (2.55, 1.4e-2), ha="center", color=ARC, fontsize=9)
    ax.set_xticks([2, 3, 4]); ax.set_xticklabels(["K2", "K3", "K4"]); ax.set_title(cell, loc="left", color=INK)
    ax.set_yscale("log"); ax.set_ylim(1.5e-4, 3e-2)
axs[0].set_ylabel("normalized MSE")
axs[0].annotate("ARC", (3, 3.53e-3), textcoords="offset points", xytext=(-12, 12), color=ARC, fontsize=9.5)
axs[0].annotate("scale as a law", (3, 7.7e-4), textcoords="offset points", xytext=(-60, -22), color=OURS, fontsize=9.5)
fig.tight_layout(); fig.savefig("fig3_ladders.png")
print("ok")

# ---- Money figure: the diagonal, times better than sampling at equal compute ----
# MC/x = naive-sampling error at the method's own FLOPs / method error. >1 beats sampling. Fill H (512x128) when it lands.
wid = np.array([64, 128, 256, 512]); lab = ["64x16", "128x32", "256x64", "512x128"]
ours = (np.array([2.31, 1.57, 1.53, 1.42]), np.array([1.66, 1.05, 0.96, 1.12]), np.array([3.57, 2.76, 2.89, 1.87]))
old = np.array([2.33, 0.92, 0.98, 1.96])
arc2 = np.array([2.18, 0.78, 0.89, 1.69])            # ARC SIMPLE K2, the level at our cost
arc3w = np.array([64, 128, 256]); arc3 = np.array([0.05, 0.0085, 0.0023])   # ARC SIMPLE K3 (not run at 512)
fig, ax = plt.subplots(figsize=(7.0, 4.4)); grid(ax)
ax.axhspan(1, 20, color="#eaf2fb", zorder=0); ax.axhline(1, color=MC, ls="--", lw=1.6)
ax.text(64 * 0.93, 0.80, "naive sampling at equal FLOPs", color=INK2, fontsize=9, ha="left", va="top")
ax.text(256 * 1.06, 12, "beats\nsampling", color=OURS, fontsize=9, ha="right", va="top")
m, lo, hi = ours
ax.errorbar(wid, m, yerr=[m - lo, hi - m], fmt="-o", color=OURS, lw=2.6, ms=9, capsize=4, zorder=5)
ax.plot(wid, old, "--o", color="#a9b8c9", lw=1.6, ms=6, zorder=4)
ax.plot(wid, arc2, "-o", color=ARC, lw=2, ms=7, zorder=4)
ax.plot(arc3w, arc3, "-o", color=ARC, lw=1.4, ms=6, alpha=0.55, zorder=4)
ax.annotate("ours: scale separation", (256, 1.53), textcoords="offset points", xytext=(-150, 26), color=OURS, fontsize=10.5, fontweight="bold")
ax.annotate("ours, leading-order scale variance", (128, 0.92), textcoords="offset points", xytext=(-40, -22), color="#7d8ea3", fontsize=9)
ax.annotate("ARC K2 (same cost class)", (256, 0.89), textcoords="offset points", xytext=(-30, -30), color=ARC, fontsize=9.5)
ax.annotate("ARC K3\n(K4 inverts or diverges)", (128, 0.0085), textcoords="offset points", xytext=(10, 6), color=ARC, fontsize=9, alpha=0.85)
ax.set_xscale("log", base=2); ax.set_xticks(wid); ax.set_xticklabels(lab); ax.set_xlim(52, 600)
ax.set_yscale("log"); ax.set_ylim(1.2e-3, 12); ax.set_yticks([0.001, 0.01, 0.1, 1, 10]); ax.set_yticklabels(["0.001x", "0.01x", "0.1x", "1x", "10x"])
ax.set_ylabel("times better than naive sampling\n(equal FLOPs, mean of 10 random nets)"); ax.set_xlabel("width x depth, with depth = width / 4")
ax.set_title("Depth growing with width: the regime ARC lists as open", loc="left", color=INK, fontsize=11.5)
fig.tight_layout(); fig.savefig("fig0_diagonal.png")
