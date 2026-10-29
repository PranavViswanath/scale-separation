"""convI.py (R195): truth-free convergence of the LVT-V2 pair vertex at cell I (1024x256) on devI nets. Same rule as conv.py/R194a:
final E h of LVT-V2 (rule v2f) at (R, b) vs a fine reference; reference checked against a finer one. Also per-layer max divergence and
|rho|, |t| statistics of the carried state along the reference chain. Chains via lvlib.lvt2_chain (the scored code path)."""
import sys, time, numpy as np, torch, lvlib
torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(0.30)
dev = "cuda"; dt = torch.float64
cell = sys.argv[1] if len(sys.argv) > 1 else "devI"
d = np.load(f"nets_{cell}.npz", mmap_mode="r"); L = int(d["L"]); n = int(d["n"]); T = d["W"].shape[0]
LST = [0, 31, 63, 127, 191, 254]
def run(Ws, R, b):
    tr = []; t0 = time.time(); e = lvlib.lvt2_chain(Ws, L, R, "full", b, "v2f", trace=tr)[0]
    return e, torch.stack([x["Eh"] for x in tr]), time.time() - t0
def stats(Ws, R, b):   # mirror of lvt2_chain (v2f) recording the carried state at LST; final E h must equal lvt2_chain's
    mu = torch.zeros(n, device=dev, dtype=dt); Sig = Ws[0] @ Ws[0].T; V = 0.0; out = {}
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    for l in range(L):
        last = l == L - 1
        if l in LST:
            s1 = lvlib.s_mom(V)[1]; m = mu / s1; C = Sig - (1 - s1 * s1) * torch.outer(m, m); sd = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-12))
            rho = (C / torch.outer(sd, sd))[off].abs(); q = torch.quantile(rho[torch.randperm(rho.numel(), device=dev)[:4000000]].float(), torch.tensor([0.5, 0.99], device=dev))
            out[l] = (float(q[0]), float(q[1]), float(rho.max()), float((m / sd).abs().max()), V)
        o = lvlib.vertex_tri(mu, Sig, V, R, "full", not last, b, ret_t4=not last); Eh = o[0]
        if last: break
        _, Cov, k4, T4 = o; Vn = lvlib.solve_V2(T4, Eh, Cov); W = Ws[l + 1]; mu = W @ Eh; Sig = W @ Cov @ W.T
        Vn = min(Vn, lvlib.V_feasible(mu, Sig)); V = Vn
    return Eh, out
for t in range(T):
    Ws = [torch.tensor(np.asarray(w), device=dev, dtype=dt) for w in d["W"][t]]
    ref, reft, w = run(Ws, 128, 10.0); ref2, ref2t, w2 = run(Ws, 192, 12.0)
    rr = float((ref - ref2).norm() / ref2.norm())
    print(f"{cell} net {t}: ref (128,10) vs (192,12) final rel {rr:.2e}, per-layer max {float(((reft - ref2t).norm(dim=1) / ref2t.norm(dim=1)).max()):.2e} (walls {w:.1f}s, {w2:.1f}s)", flush=True)
    if rr > 1e-7: ref, reft = ref2, ref2t; print(f"{cell} net {t}: reference -> (192, 12)", flush=True)
    for b in (6.5, 8.5, 10.0):
        row = []
        for R in (16, 24, 32, 48, 64):
            e, et, wl = run(Ws, R, b); rl = (et - reft).norm(dim=1) / reft.norm(dim=1)
            row.append(f"({R},{b}) final {float((e - ref).norm() / ref.norm()):.1e} lmax {float(rl.max()):.1e}@l{int(rl.argmax())+1} [{wl:.1f}s]")
        print(f"{cell} net {t}: " + " | ".join(row), flush=True)
    Es, st = stats(Ws, 128, 10.0)
    print(f"{cell} net {t}: stats-mirror vs lvt2_chain (128,10) final rel {float((Es - reft[-1]).norm() / reft[-1].norm()):.1e}", flush=True)
    for l, (med, q99, mx, tm, V) in st.items():
        print(f"{cell} net {t}: z{l+1:3d} V {V:.4f} |rho| median {med:.3f} 99% {q99:.4f} max {mx:.6f} | |t| max {tm:.1f}", flush=True)
