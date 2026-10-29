"""arc.py cell: ARC's mlp_cumulant_propagation (commit 93d091a) UNMODIFIED, f64, on the saved nets. Variants: kind SIMPLE/AUGMENT x K=1..4,
factor=True (K>=3; identical to dense for K<=2), use_avg_metric True (their default). Records E h_L = act{L-1} K[1], output pre{L} K[1], metered flops
(their NamedFlopCounter .total(), symmetric-adjustment convention) and wall time."""
import sys, time, numpy as np, torch
torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(0.35)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlp_kprop.flop_utils import NamedFlopCounter
cell = sys.argv[1]; dt = torch.float64; torch.set_default_dtype(dt); dev = "cuda"
d = np.load(f"nets_{cell}.npz"); Wall = d["W"]; n = int(d["n"]); L = int(d["L"])
variants = [(k, K) for K in (1, 2, 3, 4) for k in ("SIMPLE", "AUGMENT")]
if n >= 128: variants = [v for v in variants if v != ("AUGMENT", 4)]  # AUG4 at 256 needs a 34 GB single tensor: not run (host OOM 28 Oct)
if len(sys.argv) > 2: variants = [v for v in variants if f"{v[0]}{v[1]}" in sys.argv[2].split(",")]
import os
res = np.load(f"arc_{cell}.npy", allow_pickle=True).item() if os.path.exists(f"arc_{cell}.npy") else {}
for t in range(Wall.shape[0]):
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=L + 1).to(dev, dtype=dt)
    for i in range(L + 1): mlp.Ws[i].weight.data = torch.tensor(Wall[t, i], dtype=dt, device=dev)
    for kind, K in variants:
        if (t, kind, K) in res: continue
        t0 = time.time(); fc = NamedFlopCounter()
        with fc:
            KB = mlp_kprop(mlp, {1: torch.zeros(n, device=dev), 2: torch.eye(n, device=dev)}, k_max=K, kind=Kind[kind],
                           output_all=True, output_d_max=1, factor=(K >= 3))
        res[(t, kind, K)] = dict(h=KB[f"act{L-1}"][1].to_tensor().double().cpu().numpy(), y=KB[f"pre{L}"][1].to_tensor().double().cpu().numpy(),
                                 flops=fc.total(), wall=time.time() - t0)
        print(cell, t, kind, K, f"flops {fc.total():.3e} wall {time.time()-t0:.1f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
        del KB; torch.cuda.empty_cache()
    np.save(f"arc_{cell}.npy", res, allow_pickle=True)
