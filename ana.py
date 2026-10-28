"""ana.py cell [lvfile]: per-method MSE over nets. Cross-half debiased MSE: mean_j (e - h0_j)(e - h1_j); floor-subtracted: mean_j (e - hbar)^2 - Var_j/N.
Normalized by mean_j Var(h_Lj) per net, then averaged over nets; paired bootstrap CI over nets (2000 resamples, seed 0).
Samplers at matched flops: naive Var/N, N = F / (2 n^2 (L+1)); radial-exact Var_r/N; antithetic(+radial) Var_a / (N/2) per pair."""
import sys, glob, numpy as np
cell = sys.argv[1]; tgt = sys.argv[3] if len(sys.argv) > 3 else "h"
mc = np.load(f"mc_{cell}.npy", allow_pickle=True).item(); d = np.load(f"nets_{cell}.npz"); n = int(d["n"]); L = int(d["L"])
T = sorted(mc.keys()); fpass = 2 * n * n * (L + 1)
meth = {}
try:
    arc = np.load(f"arc_{cell}.npy", allow_pickle=True).item()
    for (t, kind, K), r in arc.items(): meth.setdefault(f"ARC-{kind[:3]}{K}", {})[t] = r
except FileNotFoundError: pass
for lvf in sorted(glob.glob(f"lv_{cell}_R*_Q*.npy") + glob.glob(f"lv_{cell}_F_R*.npy") + glob.glob(f"lv_{cell}_T_R*.npy") + glob.glob(f"lv_{cell}_Q_R*.npy") + glob.glob(f"lv_{cell}_L1_R*.npy") + glob.glob(f"lv_{cell}_V2_R*.npy")):
    tag = lvf.split("_", 2)[2][:-4]
    lv = np.load(lvf, allow_pickle=True).item()
    for (t, arm), r in lv.items(): meth.setdefault(f"{arm}@{tag}", {})[t] = r
rng = np.random.default_rng(0)
def boot(x):
    x = np.asarray(x); b = rng.integers(0, len(x), (2000, len(x))); m = x[b].mean(1); return np.percentile(m, [2.5, 97.5])
vk = "vh" if tgt == "h" else "vy"; mk = "hm" if tgt == "h" else "ym"
rows = []
for name, R in meth.items():
    ts = [t for t in T if t in R]
    if len(ts) < len(T): continue
    xs, fl, raw = [], [], []
    for t in ts:
        h0, h1 = mc[t][mk]; e = R[t][tgt]; V = mc[t][vk]; N = mc[t]["N"]
        xh = np.mean((e - h0) * (e - h1)); xs.append(xh / V.mean()); fl.append(R[t]["flops"]); raw.append(xh)
    xs = np.array(xs); F = np.mean(fl); Nm = F / fpass
    nv = np.array([mc[t][vk].mean() for t in ts])
    naive = 1 / Nm  # normalized
    rad = np.mean([mc[t]["vr"].mean() / mc[t]["vh"].mean() for t in ts]) / Nm if tgt == "h" else np.nan
    ant = np.mean([mc[t]["va"].mean() / mc[t]["vh"].mean() for t in ts]) * 2 / Nm if tgt == "h" else np.nan
    ci = boot(xs)
    rows.append((F, name, xs.mean(), ci, naive, rad, ant, np.mean(raw), (xs < naive).mean()))
rows.sort()
floor = np.mean([1 / mc[t]["N"] for t in T])
print(f"cell {cell} n={n} L={L} nets={len(T)} target={tgt}  normalized label floor V/N = {floor:.2e}  Var_eff(MC) = {np.mean([mc[t]['veff'] for t in T]):.4f} +- {np.std([mc[t]['veff'] for t in T]):.4f}")
print(f"{'method':12s} {'flops':>9s} {'nMSE':>9s} {'95% CI':>21s} {'naiveMC':>9s} {'radial':>9s} {'antith':>9s} {'ratio MC/m':>10s} {'win%':>5s}")
for F, name, m, ci, nv, rd, an, raw, win in rows:
    print(f"{name:12s} {F:9.2e} {m:9.2e} [{ci[0]:9.2e},{ci[1]:9.2e}] {nv:9.2e} {rd:9.2e} {an:9.2e} {nv/m:10.2f} {100*win:5.0f}")

# T2: ARC-best front (Pareto over ARC variants), log-log interpolated at each LV level's flops
arcs = sorted([(F, m, name) for F, name, m, *_ in rows if name.startswith("ARC")])
front = []; best = np.inf
for F, m, name in arcs:
    if m < best: front.append((F, m, name)); best = m
print("ARC-best front:", ", ".join(f"{nm} ({F:.1e}, {m:.2e})" for F, m, nm in front))
def arcbest(F):
    fF = np.array([f[0] for f in front]); fm = np.array([f[1] for f in front])
    if F <= fF[0]: return fm[0], "below-range: cheapest variant"
    if F >= fF[-1]: return fm[-1], "above-range: best variant"
    return float(np.exp(np.interp(np.log(F), np.log(fF), np.log(fm)))), "interp"
for F, name, m, ci, nv, *_ in rows:
    if name.startswith("LV") and front:
        ab, how = arcbest(F); print(f"T2 {name:14s} ARC-best/LV = {ab/m:6.2f} ({how}); MC/LV = {nv/m:6.2f}; LV CI [{ci[0]:.2e},{ci[1]:.2e}]")
tau2 = np.mean([ (mc[t]["hm"].mean(0)**2).sum() / mc[t]["vh"].sum() for t in T])
print(f"T1 inputs: Var_eff(MC) {np.mean([mc[t]['veff'] for t in T]):.4f}, tau2 {tau2:.3f}")
# T1 observed: SIMPLE ladder paired differences (normalized MSE), bootstrap CI over nets
def per_net(name):
    R = meth[name]; out = []
    for t in T:
        h0, h1 = mc[t][mk]; e = R[t][tgt]; out.append(np.mean((e - h0) * (e - h1)) / mc[t][vk].mean())
    return np.array(out)
for kind in ("SIM", "AUG"):
    for K in (1, 2, 3):
        a, b = f"ARC-{kind}{K}", f"ARC-{kind}{K+1}"
        if a in meth and b in meth and all(t in meth[b] for t in T):
            dlt = per_net(b) - per_net(a); ci = boot(dlt)
            print(f"T1 ladder {b} - {a}: {dlt.mean():+.2e} CI [{ci[0]:+.2e},{ci[1]:+.2e}]  ({'INVERTS' if ci[0] > 0 else ('improves' if ci[1] < 0 else 'n.s.')}; nets worse {int((dlt>0).sum())}/{len(dlt)})")
for a, b in (("LV0", "LV1"), ("LV1", "LV"), ("LV0", "LV"), ("LV0", "LVR"), ("LVF0", "LVF1"), ("LVF1", "LVF"), ("LVF0", "LVF"), ("LVT0", "LVT1"), ("LVT1", "LVT"), ("LVT0", "LVT")):
    for tag in ("R10_Q6", "R16_Q16", "F_R8", "F_R10", "F_R16", "T_R16"):
        A_, B_ = f"{a}@{tag}", f"{b}@{tag}"
        if A_ in meth and B_ in meth:
            dlt = per_net(B_) - per_net(A_); ci = boot(dlt)
            print(f"T5/T6 {B_} - {A_}: {dlt.mean():+.2e} CI [{ci[0]:+.2e},{ci[1]:+.2e}] nets better {int((dlt<0).sum())}/{len(dlt)}")
import glob as _g
def _dsc_report(_f):
    dsc = np.load(_f, allow_pickle=True).item(); names = sorted({k[1] for k in dsc})
    for nm in names:
        if all((t, nm) in dsc for t in T):
            x = np.array([np.mean((dsc[(t, nm)][tgt] - mc[t][mk][0]) * (dsc[(t, nm)][tgt] - mc[t][mk][1])) / mc[t][vk].mean() for t in T])
            F = np.mean([dsc[(t, nm)]["flops"] for t in T]); ci = boot(x); nv = fpass / F
            ab = arcbest(F)[0] if front else np.nan
            print(f"DSC {nm:22s} flops {F:.2e} nMSE {x.mean():.2e} CI [{ci[0]:.2e},{ci[1]:.2e}] naiveMC {nv:.2e} MC/DSC {nv/x.mean():.2f} ARCbest/DSC {ab/x.mean():.2f}")
for _f in _g.glob(f"dsc_{cell}.npy") + _g.glob(f"sl3_{cell}_*.npy"):
    _dsc_report(_f)
