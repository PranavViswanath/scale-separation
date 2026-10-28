"""expo.py cells...: width exponent of MC/x on the diagonal; bootstrap over nets within each cell (4000 resamples, seed 0)."""
import sys, numpy as np
import os; RQ = int(os.environ.get("RQ", "16")); cells = sys.argv[1:]; arms = ["LVT", "LVTV2", "LVTV2f1"]; rng = np.random.default_rng(0); B = 4000
data = {}
for c in cells:
    mc = np.load(f"mc_{c}.npy", allow_pickle=True).item(); lv = np.load(f"lv_{c}_V2_R{RQ}.npy", allow_pickle=True).item()
    W = np.load(f"nets_{c}.npz", mmap_mode="r"); n = int(W["n"]); L = int(W["L"]); T = sorted(k for k in mc if (k, "LVT") in lv)
    for a in arms:
        x = np.array([np.mean((lv[(t, a)]["h"] - mc[t]["hm"][0]) * (lv[(t, a)]["h"] - mc[t]["hm"][1])) / mc[t]["vh"].mean() for t in T])
        F = np.mean([lv[(t, a)]["flops"] for t in T]); naive = 2 * n * n * (L + 1) / F
        data[(c, a)] = (n, x, naive)
for a in arms:
    ns = np.log([data[(c, a)][0] for c in cells]); r0 = np.log([data[(c, a)][2] / data[(c, a)][1].mean() for c in cells])
    sl0 = np.polyfit(ns, r0, 1)[0]; sl = []
    for _ in range(B):
        r = [np.log(data[(c, a)][2] / data[(c, a)][1][rng.integers(0, len(data[(c, a)][1]), len(data[(c, a)][1]))].mean()) for c in cells]
        sl.append(np.polyfit(ns, r, 1)[0])
    lo, hi = np.percentile(sl, [2.5, 97.5])
    cellr = []
    for c in cells:
        n, x, nv = data[(c, a)]; bs = [nv / x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        cellr.append(f"{c}: {nv/x.mean():.2f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] ({(x < nv).sum()}/{len(x)})")
    print(f"{a:8s} exponent {sl0:+.2f} CI [{lo:+.2f}, {hi:+.2f}] | " + " | ".join(cellr))
