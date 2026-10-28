"""verify_v2.py: step-2(a) verification of the V2 rule WITHOUT truth: sample the law at layer l exactly
(S ~ Gamma(1, V), g ~ N(0, C), h = sqrt(S) relu(m + g)), compute the V2 statistic from the samples, compare with the rule's closed form.
A: random small states (n = 8) at V in {0, 0.1, 0.3}; B: a devB-like mean-dominated state (n = 64, devB net 0 chain state at layer 8,
C eigen-clipped to PSD so that the law is a distribution; rule and sampler use the same clipped law); C: the W-average reduction:
E_W E_x |W h|^4 (sampled W, sampled h) vs E_W of the law-at-l+1 closed form (1 + V') E|m' + g'|^4 with (mu', Sigma') = (W Eh, W Cov W^T)."""
import math, sys, numpy as np, torch
import lvlib
torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"; dt = torch.float64

def law_from(m, C, V):
    s1 = lvlib.s_mom(V)[1]; mu = s1 * m; Sig = C + (1 - s1 * s1) * torch.outer(m, m); return mu, Sig

def sample_stats(m, C, V, N=2**24, B=2**20, nb_split=16, seed=0):
    n = m.shape[0]; g = torch.Generator(device=dev); g.manual_seed(seed)
    Lc = torch.linalg.cholesky(C + 1e-14 * torch.eye(n, device=dev, dtype=dt))
    gam = torch.distributions.Gamma(torch.tensor(1.0 / V if V > 0 else 1.0, device=dev, dtype=dt), torch.tensor(1.0 / V if V > 0 else 1.0, device=dev, dtype=dt))
    stats = []
    torch.manual_seed(seed)
    for b in range(N // B):
        x = torch.randn(B, n, device=dev, dtype=dt, generator=g) @ Lc.T + m
        S = gam.sample((B,)) if V > 0 else torch.ones(B, device=dev, dtype=dt)
        h = torch.sqrt(S)[:, None] * torch.relu(x)
        q = (h * h).sum(1)
        stats.append(dict(s1=h.sum(0), s2=h.T @ h, q2=(q * q).sum(), N=B))
    def agg(st):
        Nn = sum(s["N"] for s in st); a = sum(s["s1"] for s in st) / Nn; M2 = sum(s["s2"] for s in st) / Nn
        K = M2 - torch.outer(a, a); T4 = float(sum(s["q2"] for s in st) / Nn); return a, K, T4
    a, K, T4 = agg(stats); full = lvlib.solve_V2(T4, a, K)
    per = len(stats) // nb_split; sub = []
    for k in range(nb_split):
        aa, KK, TT = agg(stats[k * per:(k + 1) * per]); sub.append(lvlib.solve_V2(TT, aa, KK))
    return a, K, T4, full, float(np.std(sub) / math.sqrt(nb_split)), np.array(sub)

def rule(m, C, V, b_up=6.5):
    mu, Sig = law_from(m, C, V)
    Eh, Cov, k4, T4 = lvlib.vertex_tri(mu, Sig, V, 16, "full", True, b_up, ret_t4=True)
    return Eh, Cov, k4, T4, lvlib.solve_V2(T4, Eh, Cov)

def report(tag, m, C, V, N=2**24):
    Eh, Cov, k4, T4, Vr = rule(m, C, V)
    a, K, T4s, Vs, se, sub = sample_stats(m, C, V, N)
    # identity gate: T4 - N0 == k4sum + 4 a.b (b_i = sum_j kappa3(h_i, h_j, h_j)), from the same tables (checked via N0 algebra)
    print(f"{tag:28s} V {V:.2f} | rule V' {Vr:.5f}  sampled V' {Vs:.5f} +- {se:.5f}  z {(Vr - Vs) / max(se, 1e-12):+.2f} | "
          f"T4 rule {T4:.6e} samp {T4s:.6e} rel {(T4 - T4s) / T4s:+.2e} | |Eh-a|/|a| {float((Eh - a).norm() / a.norm()):.1e} "
          f"|Cov-K|/|K| {float((Cov - K).norm() / K.norm()):.1e} | flags {lvlib.V2FLAGS[-2:]}", flush=True)
    return Vr, Vs, se

if __name__ == "__main__":
    out = {}
    rng = np.random.default_rng(5)
    # A: random small states
    for k in range(3):
        n = 8
        A = rng.normal(size=(n, n)); C = torch.tensor(A @ A.T / n * 0.5 + 0.1 * np.eye(n), device=dev, dtype=dt)
        m = torch.tensor(rng.normal(0.3, 0.8, n), device=dev, dtype=dt)
        for V in (0.0, 0.1, 0.3):
            out[("A", k, V)] = report(f"A{k} n=8", m, C, V)
    # B: devB-like mean-dominated state from the old LVT chain on devB net 0 (dev seed, not scored)
    d = np.load("nets_devB.npz"); Wall = d["W"]; n = int(d["n"]); L = int(d["L"])
    Ws = [torch.tensor(w, device=dev, dtype=dt) for w in Wall[0]]
    mu = torch.zeros(n, device=dev, dtype=dt); Sig = Ws[0] @ Ws[0].T; V = 0.0
    for l in range(8):
        Eh, Cov, k4 = lvlib.vertex_tri(mu, Sig, V, 16, "full", True, 6.5)
        Vn = lvlib.solve_V(k4, Eh, Cov); W = Ws[l + 1]; mu = W @ Eh; Sig = W @ Cov @ W.T; V = Vn
    for Vt in (V, 0.25, 0.4):
        s1 = lvlib.s_mom(Vt)[1]; m = mu / s1; C = Sig - (1 - s1 * s1) * torch.outer(m, m)
        ev, U = torch.linalg.eigh(C); nneg = int((ev < 0).sum()); C = (U * torch.clamp(ev, min=1e-10)) @ U.T
        print(f"B: devB net0 layer-8 state, |m|^2 {float(m @ m):.2f} tr C {float(torch.trace(C)):.3f} (clipped {nneg} negative eigenvalues)")
        out[("B", Vt)] = report(f"B devB-l8 n=64", m, C, Vt)
    # C: W-average reduction at the B state (V = 0.25): sampled W_{l+1}, sampled h
    Vt = 0.25; s1 = lvlib.s_mom(Vt)[1]; m = mu / s1; C = Sig - (1 - s1 * s1) * torch.outer(m, m)
    ev, U = torch.linalg.eigh(C); C = (U * torch.clamp(ev, min=1e-10)) @ U.T
    Eh, Cov, k4, T4, Vr = rule(m, C, Vt)
    s1n = lvlib.s_mom(Vr)[1]
    g = torch.Generator(device=dev); g.manual_seed(11); gam = torch.distributions.Gamma(torch.tensor(1 / Vt, device=dev, dtype=dt), torch.tensor(1 / Vt, device=dev, dtype=dt))
    Lc = torch.linalg.cholesky(C + 1e-12 * torch.eye(n, device=dev, dtype=dt))
    nW = 512; Bx = 2**16; tr_s, law_s = [], []
    torch.manual_seed(12)
    for w in range(nW):
        W = torch.randn(n, n, device=dev, dtype=dt, generator=g) * math.sqrt(2.0 / n)
        x = torch.randn(Bx, n, device=dev, dtype=dt, generator=g) @ Lc.T + m
        h = torch.sqrt(gam.sample((Bx,)))[:, None] * torch.relu(x)
        z = h @ W.T; tr_s.append(float(((z * z).sum(1) ** 2).mean()))
        mup = W @ Eh; Sp = W @ Cov @ W.T; mp = mup / s1n; Cp = Sp - (1 - s1n * s1n) * torch.outer(mp, mp)
        law_s.append((1 + Vr) * float((mp @ mp + torch.trace(Cp)) ** 2 + 2 * (Cp * Cp).sum() + 4 * mp @ Cp @ mp))
    tr_s, law_s = np.array(tr_s), np.array(law_s); dlt = tr_s - law_s
    print(f"C W-average reduction (devB-l8, V {Vt}, rule V' {Vr:.4f}, {nW} W x {Bx} x): E_W E|z'|^4 sampled {tr_s.mean():.5e}  law(V') {law_s.mean():.5e}  "
          f"paired diff {dlt.mean():+.3e} +- {dlt.std() / math.sqrt(nW):.3e} (rel {dlt.mean() / tr_s.mean():+.2e}); same with V' = old rule {lvlib.solve_V(k4, Eh, Cov):.4f}: "
          f"law {np.mean([0]):.0f} (see next line)")
    s1o = lvlib.s_mom(lvlib.solve_V(k4, Eh, Cov))[1]; Vo = lvlib.solve_V(k4, Eh, Cov)
    print(f"   identity check (n^2+2n) w2^2 E|h|^4 = {(n * n + 2 * n) * (2.0 / n) ** 2 * T4:.5e} (rule T4)")
    np.save("verify_v2.npy", out, allow_pickle=True)
