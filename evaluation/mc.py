"""mc.py cell [log2N=30]: MC truth per net, two independent halves (disjoint generator seeds), f32 forward with TF32 OFF, f64 accumulation.
Saves per net: half sums of h_L (=act{L-1}) and of the output y = W_out h_L; per-sample second moments (naive), radial-exact estimator r_bar/|x| * h,
antithetic (h(x)+h(-x))/2 variances (on 2^22 pairs), and Var_eff = LS (2,2) coefficient of kappa4(z_L) over a != b (2^23 samples)."""
import sys, os, math, time, numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
cell = sys.argv[1]; LG = int(sys.argv[2]) if len(sys.argv) > 2 else 30
d = np.load(f"nets_{cell}.npz"); Wall = d["W"]; n = int(d["n"]); L = int(d["L"])
torch.cuda.set_per_process_memory_fraction(0.7 if n >= 1024 else 0.25)  # n = 1024: 0.25 and 0.45 OOMed in the antithetic stage (R196 run; prior-net temporaries still alive)
dev = "cuda"; B = 2**20 if n <= 128 else (2**19 if n < 1024 else 2**17)   # n = 1024 on the Spark (unified 121 GB): 2**17 (R197)
ANT = int(os.environ.get("ANT", "23"))   # log2 antithetic pairs / Var_eff draws; R197: 20 at cell I on the Spark
rbar = math.sqrt(2) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))  # E|x|, x ~ N(0, I_n)
import os
out = np.load(f"mc_{cell}.npy", allow_pickle=True).item() if os.path.exists(f"mc_{cell}.npy") else {}
for t in range(Wall.shape[0]):
    if t in out: continue
    t0 = time.time()
    W = [torch.tensor(w, dtype=torch.float32, device=dev) for w in Wall[t]]
    def fwd(x):
        h = x
        for l in range(L):
            z = h @ W[l].T; h = torch.relu(z)
        return z, h
    acc = {k: torch.zeros(2, n, dtype=torch.float64, device=dev) for k in ["h", "y"]}
    acc2 = {k: torch.zeros(n, dtype=torch.float64, device=dev) for k in ["h2", "y2", "r", "r2"]}
    nb = 2**LG // B // 2
    for half in range(2):
        g = torch.Generator(device=dev); g.manual_seed(10**9 + 7919 * t + 104729 * half + 31 * n + L)
        for b in range(nb):
            x = torch.randn(B, n, device=dev, generator=g)
            z, h = fwd(x); y = h @ W[L].T
            hd = h.double(); yd = y.double()
            acc["h"][half] += hd.sum(0); acc["y"][half] += yd.sum(0)
            acc2["h2"] += (hd * hd).sum(0); acc2["y2"] += (yd * yd).sum(0)
            rr = hd * (rbar / x.double().norm(dim=1, keepdim=True))
            acc2["r"] += rr.sum(0); acc2["r2"] += (rr * rr).sum(0)
    N = nb * B * 2
    hm = (acc["h"] / (N / 2)).cpu().numpy(); ym = (acc["y"] / (N / 2)).cpu().numpy()
    mu = hm.mean(0); vh = (acc2["h2"].cpu().numpy() / N - mu**2)
    vy = acc2["y2"].cpu().numpy() / N - ym.mean(0)**2
    rm = acc2["r"].cpu().numpy() / N; vr = acc2["r2"].cpu().numpy() / N - rm**2
    # antithetic + Var_eff on separate draws
    g = torch.Generator(device=dev); g.manual_seed(2 * 10**9 + 13 * t + n + L)
    s_a = torch.zeros(n, dtype=torch.float64, device=dev); s_a2 = torch.zeros_like(s_a); s_ra = torch.zeros_like(s_a); s_ra2 = torch.zeros_like(s_a)
    m1 = torch.zeros(n, dtype=torch.float64, device=dev); m2 = torch.zeros(n, n, dtype=torch.float64, device=dev); m22 = torch.zeros(n, n, dtype=torch.float64, device=dev)
    NA = 0; zc = None
    for b in range(max(1, 2**ANT // B)):
        x = torch.randn(B, n, device=dev, generator=g)
        z, h = fwd(x); _, hn = fwd(-x)
        a = 0.5 * (h.double() + hn.double()); s_a += a.sum(0); s_a2 += (a * a).sum(0)
        ra = a * (rbar / x.double().norm(dim=1, keepdim=True)); s_ra += ra.sum(0); s_ra2 += (ra * ra).sum(0)
        if zc is None: zc = z.double().mean(0)
        yv = z.double() - zc; m1 += yv.sum(0); m2 += yv.T @ yv; q = yv * yv; m22 += q.T @ q; NA += B
    va = (s_a2 / NA - (s_a / NA)**2).cpu().numpy(); vra = (s_ra2 / NA - (s_ra / NA)**2).cpu().numpy()
    e = m1 / NA; S2 = m2 / NA - torch.outer(e, e); E22 = m22 / NA  # residual mean e ~ 1e-3 sigma: kappa4 corrections O(e^2) dropped (stated)
    k22 = E22 - torch.outer(S2.diag(), S2.diag()) - 2 * S2**2
    base = torch.outer(S2.diag(), S2.diag()) + 2 * S2**2
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    veff = float((k22[off] * base[off]).sum() / (base[off]**2).sum())
    out[t] = dict(hm=hm, ym=ym, vh=vh, vy=vy, vr=vr, va=va, vra=vra, veff=veff, N=N)
    print(cell, t, f"{time.time()-t0:.0f}s veff {veff:.4f} mean|h| {np.abs(mu).mean():.4f} var h {vh.mean():.4f} vr/vh {vr.mean()/vh.mean():.3f} va/vh {va.mean()/vh.mean():.3f} halfdiff2 {((hm[0]-hm[1])**2).mean():.3e} floor {vh.mean()/N:.3e}", flush=True)
    np.save(f"mc_{cell}.npy", out, allow_pickle=True)
