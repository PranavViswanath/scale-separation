"""score_k2.py: final-layer MSE against the challenge truth for ARC's K2 (arc_P.npy) and my estimator (lv_P_V2_R24.npy), with FLOPs."""
import numpy as np
gt = np.load("truth_P.npy")[:, -1]; T = len(gt); B = 2.0 ** 41
a = np.load("arc_P.npy", allow_pickle=True).item(); l = np.load("lv_P_V2_R24.npy", allow_pickle=True).item()
rows = {f"ARC {k} K{K}": ([a[(t, k, K)]["h"] for t in range(T)], a[(0, k, K)]["flops"]) for k, K in (("SIMPLE", 2), ("AUGMENT", 2))}
rows["mine (LVTV2)"] = ([l[(t, "LVTV2")]["h"] for t in range(T)], l[(0, "LVTV2")]["flops"])
for k, (h, f) in rows.items():
    m = np.array([((h[t] - gt[t]) ** 2).mean() for t in range(T)])
    print(f"{k:16s} mean MSE {m.mean():.3e}  FLOPs {f:.2e} ({f / B:.3f} of the 2^41 budget)")
