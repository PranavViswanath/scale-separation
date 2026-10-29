"""nets.py cell n L T: build T ARC-MLPs (He init, ARC's MLP class, num_layers=L+1) with seeds 10**6*cid + 1000*n + L*10 + t (cid=7 for this sweep), save weights f64."""
import sys, numpy as np, torch
from mlp_kprop.mlp import MLP
cell, n, L, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
torch.set_default_dtype(torch.float64)
Ws, seeds = [], []
for t in range(T):
    sd = (9 if cell.startswith("dev") else 7) * 10**6 + 1000 * n + 10 * L + t
    torch.manual_seed(sd)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=L + 1)
    Ws.append(np.stack([w.weight.detach().numpy() for w in mlp.Ws])); seeds.append(sd)
np.savez(f"nets_{cell}.npz", W=np.array(Ws), seeds=np.array(seeds), n=n, L=L, init_scale=np.array(mlp.init_scale))
print(cell, np.array(Ws).shape, seeds[:2], mlp.init_scale[:3])
