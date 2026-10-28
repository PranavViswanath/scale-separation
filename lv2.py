"""lv2.py cell: arms LVT (old leading-order V rule, = lvt_chain), LVTV2 (V2 rule + feasibility cap, Gamma s_k), LVTV2f1 (V2 rule, first-order s_k).
R from argv (24 = the converged setting chosen by the truth-free rule, PREREG R194a; 16 was the earlier frozen value), b = 6.5. Flops: ARC NamedFlopCounter + ndtr elements (1 each) + scalar special functions (hand count)."""
import sys, time, numpy as np, torch
torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32 = False
from mlp_kprop.flop_utils import NamedFlopCounter, flop_name
import lvlib
cell = sys.argv[1]; RQ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
d = np.load(f"nets_{cell}.npz", mmap_mode="r"); Wall = d["W"]; n = int(d["n"]); L = int(d["L"])
NDTR = [0]; _nd = torch.special.ndtr
def nd(x): NDTR[0] += x.numel(); return _nd(x)
lvlib.ndtr = nd
res = {}
for t in range(Wall.shape[0]):
    Ws = [torch.tensor(np.asarray(w), dtype=torch.float64, device="cuda") for w in Wall[t]]
    for arm, mode, rule in (("LVT", "full", "lead"), ("LVTV2", "full", "v2f"), ("LVTV2f1", "first", "v2f")):
        NDTR[0] = 0; lvlib.SCAL[0] = 0; lvlib.V2FLAGS.clear(); t0 = time.time(); fc = NamedFlopCounter()
        with fc:
            with flop_name("lv2"):
                h, y, Vs = lvlib.lvt2_chain(Ws, L, RQ, mode, 6.5, rule)
        fl = fc.total() + NDTR[0] + lvlib.SCAL[0]
        fg = list(lvlib.V2FLAGS)
        res[(t, arm)] = dict(h=h.cpu().numpy(), y=y.cpu().numpy(), Vs=np.array(Vs), flops=fl, wall=time.time() - t0, flags=fg)
        print(cell, t, arm, f"V_L {Vs[-1]:.4f} flops {fl:.3e} (ndtr {NDTR[0]:.2e}, scalar {lvlib.SCAL[0]}) wall {time.time()-t0:.2f}s flags root {fg.count('root')} cap {fg.count('cap')} feas {fg.count('feas')} neg {fg.count('neg')} vmax {fg.count('vmax')}", flush=True)
    np.save(f"lv_{cell}_V2_R{RQ}.npy", res, allow_pickle=True)
