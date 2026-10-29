"""sl_fac.py cell [kind]: SL3 = ARC's FACTORED K3 chain (factor=True, unmodified) + the factored de-scaling hook (descale_fac_src.py,
descale_tower, verbatim) with the EXACT Gamma scale law (h = sqrt(S) G, S ~ Gamma(1, V), S' = sqrt(S)/E sqrt S), V from the carried
kappa4 (2,2) trace by the R186a rule. Output = shape mean. Flops: ARC NamedFlopCounter over everything. Also runs V = 0 (plain ARC K3) as the gate."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "estimator"))
import sys, math, time, logging, numpy as np, torch
logging.disable(logging.WARNING)
torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.set_per_process_memory_fraction(0.35)
from mlp_kprop import kprop_harmonic as kh
from mlp_kprop.harmonic import HTensor
from mlp_kprop.factor_k3 import FactoredTensor
from mlp_kprop.flop_utils import NamedFlopCounter, flop_name
import lvlib
dev = torch.device("cuda"); dt = torch.float64; torch.set_default_dtype(dt)
exec(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "descale_fac_src.py")).read())
def gamma_law(V):
    s1 = lvlib.gmom(V, .5); E = {k: lvlib.gmom(V, k / 2) / s1**k for k in range(1, 5)}
    return {1: 1.0, 2: E[2] - 1, 3: E[3] - 3 * E[2] + 2, 4: E[4] - 4 * E[3] - 3 * E[2]**2 + 12 * E[2] - 6}, E
lognormal_cums = gamma_law   # descale_tower calls lognormal_cums(v): route it to the exact Gamma law
cell = sys.argv[1]; kind = sys.argv[2] if len(sys.argv) > 2 else "SIMPLE"
d = np.load(f"nets_{cell}.npz"); Wall = d["W"]; n = int(d["n"]); L = int(d["L"]); scale = d["init_scale"]
res = {}
for t in range(Wall.shape[0]):
    Ws = [torch.tensor(Wall[t, i], device=dev, dtype=dt) for i in range(L + 1)]
    for gate in (False, True):
        t0 = time.time(); fc = NamedFlopCounter(); kind_e = getattr(kh.Kind, kind); wick = kh.WICK_COEF_D["relu"]; Vs = []
        with fc:
            with flop_name("sl3"):
                K = kh.coerce_input({1: torch.zeros(n, device=dev), 2: torch.eye(n, device=dev)}, k_max=3, kind=kind_e)
                for l in range(L):
                    K = kh.linear_kprop(K, Ws[l], k_max=3, set_metric=float(scale[l]))
                    K = kh.nonlin_kprop(K, nonlin_wick_coef=wick, k_max=3, kind=kind_e, use_pK=True, factor=True)
                    if (not gate) and l < L - 1:
                        Sg = K[2].core; tau = K[4].core
                        num = float(tau) * (n * n + 2 * n) / 3.0; den = float(torch.trace(Sg))**2 + 2 * float((Sg * Sg).sum())
                        V = min(max(num / den, 0.0), 0.999); Vs.append(V)
                        if V > 0: K = descale_tower(K, V, n)
                h = K[1].core.double()
        nm = "ARC-K3-gate" if gate else f"SL3-{kind[:3]}"
        res[(t, nm)] = dict(h=h.cpu().numpy(), y=(Ws[L] @ h).cpu().numpy(), flops=fc.total(), vs=np.array(Vs), wall=time.time() - t0)
        print(cell, t, nm, f"flops {fc.total():.3e} V_last {Vs[-1] if Vs else 0:.4f} wall {time.time()-t0:.1f}s", flush=True)
    np.save(f"sl3_{cell}_{kind}.npy", res, allow_pickle=True)
