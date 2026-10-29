"""dsc.py cell variants: the de-scaling hook (functions in descale_src.py) on ARC's DENSE chain
(kprop_harmonic linear_kprop + nonlin_kprop factor=False, use_avg_metric as mlp_kprop: set_metric = init_scale[l]) on the saved nets.
Between layers (not after the last): h = S'.G, S' lognormal (E 1, Var v), v from the chain's own state (k22 or energy), Leonov-Shiryaev inversion,
chain continues on G; output = shape mean at the last layer (positive homogeneity: E S' = 1). Nothing fitted. Flops: ARC NamedFlopCounter on everything.
GPU allocator capped at 40 % of memory (in-process OOM instead of host thrash)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "estimator"))
import sys, math, time, itertools, logging, numpy as np, torch
logging.disable(logging.WARNING)
torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(0.4)
from mlp_kprop import kprop_harmonic as kh
from mlp_kprop.flop_utils import NamedFlopCounter, flop_name
cell = sys.argv[1]; variants = sys.argv[2].split(",")   # e.g. 2S:none,3S:k22,4S:energy
d = np.load(f"nets_{cell}.npz"); Wall = d["W"]; n = int(d["n"]); L = int(d["L"]); scale = d["init_scale"]
dev = torch.device("cuda"); dt = torch.float64; torch.set_default_dtype(dt)
src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "descale_src.py")).read(); exec(src)   # set_partitions, join_is_full, PARTS, LETTERS, scaled_cumulants, descale, lognormal_cums, choose_v, dense_tower
import lvlib
def gamma_cums(V):
    s1 = lvlib.gmom(V, .5); E = {k: lvlib.gmom(V, k / 2) / s1**k for k in range(1, 5)}
    return {1: 1.0, 2: E[2] - 1, 3: E[3] - 3 * E[2] + 2, 4: E[4] - 4 * E[3] - 3 * E[2]**2 + 12 * E[2] - 6}
def run(Ws, k, kind, vmode):
    kind_e = getattr(kh.Kind, kind)
    K = kh.coerce_input({1: torch.zeros(n, device=dev), 2: torch.eye(n, device=dev)}, k_max=k, kind=kind_e)
    wick = kh.WICK_COEF_D["relu"]; vs = []
    for l in range(L):
        K = kh.linear_kprop(K, Ws[l], k_max=k, set_metric=float(scale[l]))
        K = kh.nonlin_kprop(K, nonlin_wick_coef=wick, k_max=k, kind=kind_e, use_pK=True, factor=False)
        if vmode != "none" and l < L - 1:
            with flop_name("descale"):
                h = dense_tower(K)
                if 3 not in h: h[3] = torch.zeros((n,) * 3, device=dev, dtype=dt)
                if 4 not in h: h[4] = torch.zeros((n,) * 4, device=dev, dtype=dt)
                if vmode == "gam":
                    # SL: exact Gamma scale law. h = sqrt(S) G, S ~ Gamma(1, V), normalized S' = sqrt(S)/E sqrt(S) (E S' = 1);
                    # V from the carried kappa4 (2,2) content by the R186a rule; S' cumulants from exact Gamma moments.
                    Sg = h[2]; k4 = h[4]
                    num = float(torch.einsum("aabb->", k4)); den = float(torch.trace(Sg))**2 + 2 * float((Sg * Sg).sum())
                    v = min(max(num / den, 0.0), 0.999); vs.append(v)
                    G = descale(h, gamma_cums(v))
                else:
                    v = choose_v(h, vmode); vs.append(v)
                    G = descale(h, lognormal_cums(v))
                K = kh.coerce_input({dd: G[dd] for dd in G if dd in K}, k_max=k, kind=kind_e)
    out = K[1].to_tensor().double()
    return out, vs
res = {}
for t in range(Wall.shape[0]):
    Ws = [torch.tensor(Wall[t, i], device=dev, dtype=dt) for i in range(L + 1)]
    for var in variants:
        ks, vmode = var.split(":"); k = int(ks[0]); kind = {"S": "SIMPLE", "A": "AUGMENT"}[ks[1]]
        t0 = time.time(); fc = NamedFlopCounter()
        with fc:
            with flop_name("chain"):
                h, vs = run(Ws, k, kind, vmode)
        res[(t, f"DSC-{kind[:3]}{k}-{vmode}")] = dict(h=h.cpu().numpy(), y=(Ws[L] @ h).cpu().numpy(), flops=fc.total(), vs=np.array(vs), wall=time.time() - t0)
        print(cell, t, var, f"flops {fc.total():.3e} v_last {vs[-1] if vs else 0:.4f} wall {time.time()-t0:.1f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
        np.save(f"dsc_{cell}.npy", res, allow_pickle=True)
