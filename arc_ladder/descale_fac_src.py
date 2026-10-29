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
        H4 = K[4]; assert H4.r == 2 and H4.core.ndim == 0, (H4.r, H4.core.shape)
        tau = H4.core
        trace_full = tau * (n * n + 2 * n) / 3.0                      # sum_ab Sym(I x I)_aabb = (n^2+2n)/3 (verified in parity)
        D3 = newF.get_dslice((3,)); D21 = newF.get_dslice((2, 1))
        T_coll = 0.0
        for t4 in PARTS[4]:
            if len(t4) == 1: continue
            c = cS_of(t4, kS, 4)
            if c == 0.0: continue
            T_coll = T_coll + c * trace_aabb(t4, m, Sig_t, D3, D21, n)
        tau_new = (trace_full - T_coll) / E[4] * 3.0 / (n * n + 2 * n)
        out[4] = HTensor(tau_new, r=2, n=n)
    return out

