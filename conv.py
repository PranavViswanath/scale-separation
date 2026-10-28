"""conv.py (R194): truth-free numerics convergence of the pair vertex: final E h of LVT-V2 at (R, b) vs the (128, 10) reference on dev nets."""
import sys, numpy as np, torch, lvlib
torch.set_grad_enabled(False); dev = "cuda"; dt = torch.float64
for cell in sys.argv[1:]:
    d = np.load(f"nets_{cell}.npz", mmap_mode="r"); L = int(d["L"])
    Ws = [torch.tensor(np.asarray(w), device=dev, dtype=dt) for w in d["W"][0]]
    ref = lvlib.lvt2_chain(Ws, L, 128, "full", 10.0, "v2f")[0]
    out = []
    for R in (16, 24, 32, 48):
        for b in (6.5, 8.5, 10.0):
            e = lvlib.lvt2_chain(Ws, L, R, "full", b, "v2f")[0]; out.append(f"({R},{b}) {float((e - ref).norm() / ref.norm()):.1e}")
    print(cell, " ".join(out), flush=True)
