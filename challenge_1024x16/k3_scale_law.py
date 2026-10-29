"""De-scaled ARC *factored* K3 SIMPLE chain (runs at 1024x16).  Per layer after the ReLU step: re-represent h = S'.G
(S' lognormal, E S' = 1, Var S' = v, independent of the shape G) and continue on G.  kappa3 is a FactoredTensor (columns
appended for the subtracted collective blocks), kappa4 is ARC's scalar full-trace piece (r=2), Sigma dense.
Modes: --parity (n=24 L=4: factored de-scaling vs dense descale_test machinery), --mlps (real 1024x16 npz dumps)."""
import sys, os as _os, math, time, logging, argparse, itertools
logging.disable(logging.WARNING)
import numpy as np, torch
from mlp_kprop.mlp import MLP
from mlp_kprop import kprop_harmonic as kh
from mlp_kprop.harmonic import HTensor
from mlp_kprop.factor_k3 import FactoredTensor

ap = argparse.ArgumentParser()
ap.add_argument("--parity", action="store_true"); ap.add_argument("--mlps", default="0")
ap.add_argument("--vmode", default="energy,none"); ap.add_argument("--n", type=int, default=24); ap.add_argument("--L", type=int, default=4)
ap.add_argument("--dtype", default="float64"); ap.add_argument("--nets", default="nets"); ap.add_argument("--kind", default="SIMPLE")
args = ap.parse_args()
dev = torch.device("cuda"); dt = getattr(torch, args.dtype); torch.set_grad_enabled(False)

# ---------- partition machinery (same as descale_test) ----------
def set_partitions(items):
    if len(items) == 1:
        yield [items]; return
    first = items[0]
    for smaller in set_partitions(items[1:]):
        for i, sub in enumerate(smaller):
            yield smaller[:i] + [[first] + sub] + smaller[i + 1:]
        yield [[first]] + smaller
def join_is_full(sig, tau, r):
    parent = list(range(r))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for blocks in (sig, tau):
        for b in blocks:
            for x in b[1:]: parent[find(x)] = find(b[0])
    return len({find(x) for x in range(r)}) == 1
PARTS = {r: list(set_partitions(list(range(r)))) for r in range(1, 5)}
def cS_of(tau, kS, r): return sum(math.prod(kS[len(b)] for b in sig) for sig in PARTS[r] if join_is_full(sig, tau, r))
def lognormal_cums(v):
    E = {k: (1 + v) ** (k * (k - 1) / 2) for k in range(1, 5)}
    return {1: 1.0, 2: E[2] - 1, 3: E[3] - 3 * E[2] + 2, 4: E[4] - 4 * E[3] - 3 * E[2] ** 2 + 12 * E[2] - 6}, E

def k3_entries(F, D3, D21, pattern, n):
    """n x n matrix M[i,j] = kappa3 at index positions given by pattern in {'iij','ijj','iii','jjj','iji',...} (i first index)."""
    s = pattern
    if s.count("i") == 3: return D3[:, None].expand(n, n)
    if s.count("j") == 3: return D3[None, :].expand(n, n)
    if s.count("i") == 2: return D21 + torch.diag(D3)           # kappa3[i,i,j] = D21[i,j] (zero diag) + delta_ij D3
    return D21.T + torch.diag(D3)                                 # kappa3[i,j,j]

def trace_aabb(tau, m, Sig, D3, D21, n):
    """sum_{i,j} of prod over blocks of tau of the block's cumulant evaluated at positions (0,1)->i, (2,3)->j."""
    pos = ["i", "i", "j", "j"]
    M = torch.ones(n, n, device=dev, dtype=dt)
    for b in tau:
        pat = "".join(pos[p] for p in b)
        if len(b) == 1: M = M * (m[:, None] if pat == "i" else m[None, :])
        elif len(b) == 2:
            if pat == "ii": M = M * Sig.diagonal()[:, None]
            elif pat == "jj": M = M * Sig.diagonal()[None, :]
            else: M = M * Sig
        elif len(b) == 3: M = M * k3_entries(None, D3, D21, pat, n)
        else: raise ValueError
    return M.sum()

def k3_contract_m(F, m):
    """sum_c T_abc m_c for T = Sym(sum_r A_r x B_r x C_r) (FactoredTensor, purely factored: repeated entries included)."""
    A, B, C = F._factors
    def t(X, Y, Z):   # ordering (X, Y, Z) on positions (a, b, c): X diag(Z^T m) Y^T
        return (X * (Z.T @ m)[None, :]) @ Y.T
    return (t(A, B, C) + t(B, A, C) + t(A, C, B) + t(C, A, B) + t(B, C, A) + t(C, B, A)) / 6.0

def minv_pt(P, n):
    """Inverse of X -> PT(Sym(X x I)) = ((n+4) X + I tr X)/6."""
    trP = P.diagonal().sum()
    return (6.0 * P - torch.eye(n, device=P.device, dtype=P.dtype) * (6.0 * trP / (2 * n + 4))) / (n + 4)

def pt_abcc(tau, m, Sig, D3, D21, K3m, k3acc, n):
    """n x n matrix P_ab = sum_c prod_{B in tau} kappa_B at positions (0,1,2,3) -> (a,b,c,c)."""
    pos = ["a", "b", "c", "c"]
    pieces, cb = [], []
    for B in tau:
        pat = "".join(sorted(pos[p] for p in B))
        if "c" not in pat:
            if len(B) == 1: pieces.append((pat, m))
            elif pat == "ab": pieces.append(("ab", Sig))
            else: raise ValueError(pat)
        else:
            cb.append(pat)
    if len(cb) == 1:
        p = cb[0]
        if p == "cc": pieces.append(("s", Sig.diagonal().sum()))
        elif p == "acc": pieces.append(("a", k3acc))
        elif p == "bcc": pieces.append(("b", k3acc))
        else: raise ValueError(p)
    else:
        ps = sorted(cb)
        if ps == ["c", "c"]: pieces.append(("s", (m * m).sum()))
        elif ps == ["ac", "c"]: pieces.append(("a", Sig @ m))
        elif ps == ["bc", "c"]: pieces.append(("b", Sig @ m))
        elif ps == ["ac", "bc"]: pieces.append(("ab", Sig @ Sig))
        elif ps == ["abc", "c"]: pieces.append(("ab", K3m))
        else: raise ValueError(ps)
    M = torch.ones(n, n, device=dev, dtype=dt)
    for kind, t in pieces:
        if kind == "s": M = M * t
        elif kind == "a": M = M * t[:, None]
        elif kind == "b": M = M * t[None, :]
        else: M = M * t
    return M

def descale_tower(K, v, n):
    """K: ARC tower after nonlin (K[1],K[2] HTensor r=0; K[3] FactoredTensor; K[4] HTensor r=2 scalar). Returns new tower."""
    kS, E = lognormal_cums(v)
    m = K[1].core; Sig = K[2].core
    assert K[2].r == 0 and K[1].r == 0
    Sig_t = (Sig - kS[2] * torch.outer(m, m)) / E[2]
    F = K[3]; A, B, C = F._factors
    c_pair = cS_of([[0, 1], [2]], kS, 3)   # coefficient of m_c Sig_ab (each of the 3 placements)
    c_sing = cS_of([[0], [1], [2]], kS, 3)
    newF = FactoredTensor(n=n, d=3, factors=(A / E[3], B, C), device=A.device, dtype=A.dtype)
    eye = torch.eye(n, device=dev, dtype=dt)
    # Sym(sum_r A_r (x) B_r (x) C_r) with A = m 1^T, B = I, C = Sig_t gives Sym(m (x) Sig_t) = (1/3) sum of 3 placements
    newF.add_factors_((-(3 * c_pair / E[3]) * m[:, None].expand(n, n), eye, Sig_t))
    newF.add_factors_((-(c_sing / E[3]) * m[:, None], m[:, None], m[:, None]))
    out = {1: HTensor(m.clone(), r=0), 2: HTensor(Sig_t, r=0), 3: newF}
    if 4 in K:
        H4 = K[4]
        D3 = newF.get_dslice((3,)); D21 = newF.get_dslice((2, 1))
        if H4.r == 2 and H4.core.ndim == 0:
            tau = H4.core
            trace_full = tau * (n * n + 2 * n) / 3.0                      # sum_ab Sym(I x I)_aabb = (n^2+2n)/3 (verified in parity)
            T_coll = 0.0
            for t4 in PARTS[4]:
                if len(t4) == 1: continue
                c = cS_of(t4, kS, 4)
                if c == 0.0: continue
                T_coll = T_coll + c * trace_aabb(t4, m, Sig_t, D3, D21, n)
            tau_new = (trace_full - T_coll) / E[4] * 3.0 / (n * n + 2 * n)
            out[4] = HTensor(tau_new, r=2, n=n)
        elif H4.r == 1 and H4.core.ndim == 2:
            # kappa4 = Sym(X x I) = (1/6)(X_ab d_cd + d_ab X_cd + X_ac d_bd + d_ac X_bd + X_ad d_bc + d_ad X_bc).
            # The radial (r>=1) projection of any symmetric 4-tensor is fixed by its Laplacian, i.e. the partial
            # trace PT_ab = sum_c T_abcc; for Sym(X x I): PT = ((n+4) X + I tr X)/6.
            # De-scale by matching PT: X_new = (X - Minv(PT(T_coll))) / E4, Minv(P) = (6P - I 6trP/(2n+4))/(n+4).
            X = H4.core
            K3m = k3_contract_m(newF, m)                                # sum_c kappa3_{abc} m_c from the factors
            k3acc = D21.sum(0) + D3                                     # sum_c kappa3_{acc}
            PT = torch.zeros(n, n, device=dev, dtype=dt)
            for t4 in PARTS[4]:
                if len(t4) == 1: continue
                c = cS_of(t4, kS, 4)
                if c == 0.0: continue
                PT = PT + c * pt_abcc(t4, m, Sig_t, D3, D21, K3m, k3acc, n)
            X_new = (X - minv_pt(PT, n)) / E[4]
            X_new = 0.5 * (X_new + X_new.T)
            out[4] = HTensor(X_new, r=1, n=n)
        else:
            raise AssertionError((H4.r, H4.core.shape))
    return out

def choose_v_fac(K, n, mode):
    if mode == "none": return 0.0
    m = K[1].core; Sig = K[2].core; F = K[3]
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    if 4 in K and K[4].r == 1:
        Xd = K[4].core.diagonal()
        K22 = (Xd[:, None] + Xd[None, :]) / 6.0                             # (aabb) off entries of Sym(X x I)
        tau = None
    else:
        tau = K[4].core if 4 in K else torch.zeros((), device=dev, dtype=dt)
        K22 = tau / 3.0 * (1 + 2 * torch.eye(n, device=dev, dtype=dt))     # (aabb) entries of Sym(tau I x I)
    if mode == "k22":
        v2 = float(K22[off].sum() / (torch.outer(Sig.diagonal(), Sig.diagonal()) + 2 * Sig * Sig)[off].sum())
        return max((1 + v2) ** 0.25 - 1, 0.0) if v2 > -1 else 0.0
    if mode == "radial":   # input radial gauge only: Var(r)/E[r]^2 = 1/(2n) at the first layer, nothing after
        return 1.0 / (2 * n) if K.get("_layer", 0) == 0 else 0.0
    if mode == "const":    # constant per-layer scale variance 1/(2n)
        return 1.0 / (2 * n)
    if mode == "k22f":     # kappa4-trace estimate with a floor of 3e-4
        K22_ = K22
        v2 = float(K22_[off].sum() / (torch.outer(Sig.diagonal(), Sig.diagonal()) + 2 * Sig * Sig)[off].sum())
        return max((1 + v2) ** 0.25 - 1, 3e-4) if v2 > -1 else 3e-4
    D3 = F.get_dslice((3,)); D21 = F.get_dslice((2, 1))
    k3_abb = D21.T + torch.diag(D3); k3_aab = D21 + torch.diag(D3)
    E2 = float((m * m + Sig.diagonal()).sum()) ** 2
    if mode in ("pair", "pair0"):   # collective = pair part of the energy variance, sum over a != b only
        K22p = K22 if mode == "pair" else torch.zeros_like(K22)
        var = (K22p + 2 * Sig * Sig + 4 * torch.outer(m, m) * Sig + 2 * k3_abb * m[:, None] + 2 * k3_aab * m[None, :])[off].sum()
        v2 = float(var / E2); return max((1 + v2) ** 0.25 - 1, 0.0)
    var = (K22 + 2 * Sig * Sig + 4 * torch.outer(m, m) * Sig + 2 * k3_abb * m[:, None] + 2 * k3_aab * m[None, :]).sum()
    v2 = float(var / E2)
    return max((1 + v2) ** 0.25 - 1, 0.0)

def build_mlp(Ws, n, L):
    torch.set_default_dtype(dt); mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu")
    for l in range(L): mlp.Ws[l].weight.data = Ws[l].T.contiguous()
    mlp.Ws[L].weight.data = torch.eye(n, dtype=dt); mlp.init_scale = [2.0] * (L + 1); return mlp.to(dev, dtype=dt)

def run_fac(Ws, n, L, vmode, kind="SIMPLE", k=3):
    mlp = build_mlp(Ws, n, L); kind_e = getattr(kh.Kind, kind); wick = kh.WICK_COEF_D["relu"]
    K = kh.coerce_input({1: torch.zeros(n, device=dev, dtype=dt), 2: torch.eye(n, device=dev, dtype=dt)}, k_max=k, kind=kind_e)
    means, vs, towers = [], [], []
    H = int(_os.environ.get("HORIZON", "0")); ncols = []; births = []
    for l in range(L):
        if H > 0 and 3 in K and l - H >= 0:
            cut = births[l - H]          # drop the cohort born at layer l-H (oldest alive; columns are in birth order)
            F = K[3]
            if cut > 0 and cut < F._factors[0].shape[1]:
                K[3] = FactoredTensor(n=n, d=3, factors=tuple(f[:, cut:] for f in F._factors), device=F.device, dtype=F.dtype)
        nbefore = K[3]._factors[0].shape[1] if 3 in K else 0
        K = kh.linear_kprop(K, mlp.Ws[l].weight, k_max=k, set_metric=2.0)
        XD = _os.environ.get("XDUMP", "")
        if XD and 3 in K:
            _xd = globals().setdefault("_XDUMP", {})
            if 4 in K and K[4].r == 1:
                _xd["L%d_X" % l] = K[4].core.float().cpu().numpy()  # pre-activation harmonic core of layer l (0-based li)
            _xd["L%d_D3" % l] = K[3].get_dslice((3,)).float().cpu().numpy()
            _k21 = K[3].get_dslice((2, 1)).clone(); _k21.fill_diagonal_(0.0)
            _xd["L%d_K21" % l] = _k21.float().cpu().numpy()        # pre-activation (2,1) slice of kappa3 at layer l
            _xd["L%d_mu" % l] = K[1].core.float().cpu().numpy(); _xd["L%d_var" % l] = K[2].core.diagonal().float().cpu().numpy()
            if l == L - 1: np.savez(XD, **_xd)
        tpre = (K[1].core / torch.sqrt(torch.clamp(K[2].core.diagonal(), min=1e-12))).detach()
        K = kh.nonlin_kprop(K, nonlin_wick_coef=wick, k_max=k, kind=kind_e, use_pK=True, factor=True)
        TAU = float(_os.environ.get("HUBTAU", "0"))
        if TAU > 0 and 3 in K:
            F = K[3]; R = F._factors[0].shape[1]; braw = R - nbefore
            assert braw % n == 0, (braw, n)
            nfam = braw // n                      # born families this layer; the last one is from_dstensor (kept whole)
            keep = (tpre.abs() <= TAU)
            mask = torch.ones(R, dtype=torch.bool, device=keep.device)
            for f in range(nfam - 1):
                mask[nbefore + f * n: nbefore + (f + 1) * n] = keep
            K[3] = FactoredTensor(n=n, d=3, factors=tuple(fc[:, mask] for fc in F._factors), device=F.device, dtype=F.dtype)
            if l < 3 or l % 4 == 3: print(f"   layer {l+1}: hub prune |t|>{TAU}: kept {int(keep.sum())}/{n} hubs in {nfam-1} families; columns {R} -> {int(mask.sum())}", flush=True)
        if vmode != "none" and l < L - 1:
            K["_layer"] = l; v = choose_v_fac(K, n, vmode); del K["_layer"]; vs.append(v); K = descale_tower(K, v, n)
        means.append(K[1].core.double().clone()); towers.append(K)
        ncols.append(K[3]._factors[0].shape[1] if 3 in K else 0); births.append(ncols[-1] - nbefore)
        if H > 0: print(f"   layer {l+1}: kappa3 columns {ncols[-1]} (born {births[-1]})", flush=True)
    return torch.stack(means), vs, towers

if args.parity:
    raise SystemExit("use --parity2")

if args.mlps == "parity":
    # dense vs factored parity at small n: same v sequence; compare dense cumulants after de-scaling
    n, L = args.n, args.L; torch.manual_seed(0)
    Ws = [torch.randn(n, n, device=dev, dtype=dt) * math.sqrt(2.0 / n) for _ in range(L)]
    mlp = build_mlp(Ws, n, L); wick = kh.WICK_COEF_D["relu"]; kind_e = kh.Kind.SIMPLE
    Kf = kh.coerce_input({1: torch.zeros(n, device=dev, dtype=dt), 2: torch.eye(n, device=dev, dtype=dt)}, k_max=3, kind=kind_e)
    Kd = kh.coerce_input({1: torch.zeros(n, device=dev, dtype=dt), 2: torch.eye(n, device=dev, dtype=dt)}, k_max=3, kind=kind_e)
    # dense de-scaling via the generic Leonov-Shiryaev inversion on dense tensors (copy of descale_test.descale)
    LET = "abcd"
    def dense_descale(h, kS):
        G = {}
        for r in range(1, 5):
            ES_r = sum(math.prod(kS[len(b)] for b in sig) for sig in PARTS[r])
            rest = torch.zeros((n,) * r, device=dev, dtype=dt)
            for tau in PARTS[r]:
                if len(tau) == 1: continue
                c = cS_of(tau, kS, r)
                if c == 0.0: continue
                ops = [G[len(b)] for b in tau]; subs = ["".join(LET[i] for i in b) for b in tau]
                rest += c * torch.einsum(",".join(subs) + "->" + LET[:r], *ops)
            G[r] = (h[r] - rest) / ES_r
        return G
    for l in range(L):
        W = mlp.Ws[l].weight
        Kf = kh.linear_kprop(Kf, W, k_max=3, set_metric=2.0); Kf = kh.nonlin_kprop(Kf, nonlin_wick_coef=wick, k_max=3, kind=kind_e, use_pK=True, factor=True)
        Kd = kh.linear_kprop(Kd, W, k_max=3, set_metric=2.0); Kd = kh.nonlin_kprop(Kd, nonlin_wick_coef=wick, k_max=3, kind=kind_e, use_pK=True, factor=False)
        # compare towers before de-scaling
        for d in (1, 2, 3, 4):
            a = Kf[d].to_tensor() if d != 1 else Kf[d].core; b = Kd[d].to_tensor() if d != 1 else Kd[d].core
            print(f"L{l} pre-descale d={d} rel diff fac vs dense {float((a-b).norm()/(b.norm()+1e-300)):.1e}")
        # trace check of the scalar piece
        tau = Kf[4].core; T4 = Kf[4].to_tensor(); print(f"   trace check: {float(torch.einsum('aabb->', T4)):.6e} vs tau(n^2+2n)/3 = {float(tau*(n*n+2*n)/3):.6e}")
        v = choose_v_fac(Kf, n, "energy"); print(f"   v = {v:.4e}")
        kS, E = lognormal_cums(v)
        h = {1: Kd[1].core, 2: Kd[2].to_tensor(), 3: Kd[3].to_tensor(), 4: Kd[4].to_tensor()}
        G = dense_descale(h, kS)
        Kf = descale_tower(Kf, v, n)
        Kd = kh.coerce_input({d: G[d] for d in G}, k_max=3, kind=kind_e)
        for d in (1, 2, 3, 4):
            a = Kf[d].to_tensor() if d != 1 else Kf[d].core; b = Kd[d].to_tensor() if d != 1 else Kd[d].core
            print(f"   post-descale d={d} rel diff fac vs dense {float((a-b).norm()/(b.norm()+1e-300)):.1e}")
    print("PARITY_DONE"); raise SystemExit

if args.mlps == "synth":
    n, L = args.n, args.L; torch.manual_seed(int(_os.environ.get("SEED", "0")))
    Ws = [torch.randn(n, n, device=dev, dtype=torch.float64) * math.sqrt(2.0 / n) for _ in range(L)]
    N = int(_os.environ.get("NTRUTH", str(1 << 28))); bs = 1 << 16
    acc = torch.zeros(L, n, device=dev, dtype=torch.float64); g = torch.Generator(device=dev).manual_seed(12345)
    Wf = [W.float() for W in Ws]
    for _ in range(N // bs):
        x = torch.randn(bs, n, device=dev, generator=g, dtype=torch.float32)
        for l, W in enumerate(Wf):
            x = torch.relu(x @ W); acc[l] += x.sum(0, dtype=torch.float64)
    gt = acc / N
    Ws = [W.to(dt) for W in Ws]
    for vmode in args.vmode.split(","):
        t0 = time.time(); means, vs, _ = run_fac(Ws, n, L, vmode, kind=args.kind)
        per = ((means - gt) ** 2).mean(1).cpu().numpy()
        print(f"SYNTH n={n} L={L} K3 {args.kind} fac v={vmode:7s} final {per[-1]:.3e} ({time.time()-t0:.0f}s) | " + " ".join(f"{x:.2e}" for x in per), flush=True)
        print("   v:", " ".join(f"{v:.2e}" for v in vs), flush=True)
    raise SystemExit

if args.mlps == "unit":
    # convention check: PT(Sym(X x I)) == ((n+4) X + I tr X)/6 and minv_pt inverts it
    n = 8; X = torch.randn(n, n, device=dev, dtype=dt); X = X + X.T
    T = HTensor(X, r=1, n=n).to_tensor()
    PT = torch.einsum("abcc->ab", T)
    ref = ((n + 4) * X + torch.eye(n, device=dev, dtype=dt) * X.diagonal().sum()) / 6.0
    print("PT convention rel err", float((PT - ref).norm() / ref.norm()), " minv rel err", float((minv_pt(PT, n) - X).norm() / X.norm()))
    raise SystemExit

# ---------- real MLPs ----------
for mi in [int(s) for s in args.mlps.split(",")]:
    z = np.load(_os.path.join(args.nets, "mlp_%02d.npz" % mi))
    W = z["weights"]; gt = torch.from_numpy(z["all_layer_means"].astype(np.float64)).to(dev)
    Ws = [torch.from_numpy(np.ascontiguousarray(W[l])).to(dev, dt) for l in range(len(W))]
    L = len(Ws); n = Ws[0].shape[0]
    for vmode in args.vmode.split(","):
        t0 = time.time(); means, vs, _ = run_fac(Ws, n, L, vmode, kind=args.kind)
        per = ((means - gt) ** 2).mean(1).cpu().numpy()
        print(f"MLP {mi} K3 {args.kind} fac v={vmode:7s} final {per[-1]:.3e} ({time.time()-t0:.0f}s) | " + " ".join(f"{x:.2e}" for x in per), flush=True)
        if vs: print("    v per layer:", " ".join(f"{v:.2e}" for v in vs), " cum E[S^2]-1 =", f"{math.prod(1+v for v in vs)-1:.4f}", flush=True)
print("FAC_DONE")
