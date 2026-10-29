"""get_nets.py DATASET_DIR [count]: the challenge's own nets (Phase 2 public dataset, mini split, mlp_id 0..count-1) with their 1e9-sample truth.
Writes nets/mlp_XX.npz (weights, all_layer_means, mlp_seed) for k3_scale_law.py, and nets_P.npz + truth_P.npy (ARC weight convention,
an identity output layer) for ../arc_ladder/arc.py and ../estimator/lv2.py."""
import sys, os, glob, numpy as np, pyarrow.parquet as pq
src, T = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12
rows = {}
for f in sorted(glob.glob(os.path.join(src, "data", "mini-*.parquet"))):
    t = pq.read_table(f).to_pydict()
    for i, mid in enumerate(t["mlp_id"]):
        if mid < T: rows[mid] = {k: t[k][i] for k in ("weights", "all_layer_means", "mlp_seed")}
os.makedirs("nets", exist_ok=True); Ws, gt = [], []
for i in range(T):
    W = np.asarray(rows[i]["weights"], dtype=np.float32); m = np.asarray(rows[i]["all_layer_means"], dtype=np.float32)
    np.savez(f"nets/mlp_{i:02d}.npz", weights=W, all_layer_means=m, mlp_seed=rows[i]["mlp_seed"])
    Ws.append(np.concatenate([np.transpose(W.astype(np.float64), (0, 2, 1)), np.eye(W.shape[1])[None]], 0)); gt.append(m.astype(np.float64))
np.savez("nets_P.npz", W=np.array(Ws), n=W.shape[1], L=W.shape[0], seeds=np.arange(T)); np.save("truth_P.npy", np.array(gt))
print(T, "nets", W.shape)
