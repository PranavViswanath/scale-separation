# ---------- cumulants of a product S*G, S independent scalar (Leonov-Shiryaev: sum over partition pairs with join = full) ----------
def set_partitions(items):
    if len(items) == 1:
        yield [items]; return
    first = items[0]
    for smaller in set_partitions(items[1:]):
        for i, sub in enumerate(smaller):
            yield smaller[:i] + [[first] + sub] + smaller[i + 1:]
        yield [[first]] + smaller
def join_is_full(sig, tau, r):
    # union-find over blocks
    parent = list(range(r))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for blocks in (sig, tau):
        for b in blocks:
            for x in b[1:]: parent[find(x)] = find(b[0])
    return len({find(x) for x in range(r)}) == 1
PARTS = {r: list(set_partitions(list(range(r)))) for r in range(1, 5)}
LETTERS = "abcd"

def scaled_cumulants(kS, G, r):
    """kS: dict order->cumulant of S (kS[1]=1); G: dict order->dense cumulant tensors of the shape. Returns dense k_r(S*G)."""
    out = torch.zeros((n,) * r, device=dev, dtype=dt)
    for tau in PARTS[r]:
        # kappa_tau(G): product over blocks of G cumulant tensors, placed on the block's indices
        cS = 0.0
        for sig in PARTS[r]:
            if join_is_full(sig, tau, r):
                cS += math.prod(kS[len(b)] for b in sig)
        if cS == 0.0: continue
        subs = []; ops = []
        for b in tau:
            ops.append(G[len(b)]); subs.append("".join(LETTERS[i] for i in b))
        T = torch.einsum(",".join(subs) + "->" + LETTERS[:r], *ops)
        out += cS * T
    return out

def descale(h, kS):
    """Invert scaled_cumulants: given cumulants h (orders 1..4) and S cumulants kS, return shape cumulants G."""
    G = {}
    for r in range(1, 5):
        # k_r(h) = E[S^r] k_r(G) + sum_{tau != single block} c_tau k_tau(G)
        ES_r = sum(math.prod(kS[len(b)] for b in sig) for sig in PARTS[r])
        rest = torch.zeros((n,) * r, device=dev, dtype=dt)
        for tau in PARTS[r]:
            if len(tau) == 1: continue
            cS = sum(math.prod(kS[len(b)] for b in sig) for sig in PARTS[r] if join_is_full(sig, tau, r))
            if cS == 0.0: continue
            subs = []; ops = []
            for b in tau:
                ops.append(G[len(b)]); subs.append("".join(LETTERS[i] for i in b))
            rest += cS * torch.einsum(",".join(subs) + "->" + LETTERS[:r], *ops)
        G[r] = (h[r] - rest) / ES_r
    return G

def lognormal_cums(v):
    E = {k: (1 + v) ** (k * (k - 1) / 2) for k in range(1, 5)}   # E S^k, E S = 1, E S^2 = 1+v
    k1 = 1.0; k2 = E[2] - 1; k3 = E[3] - 3 * E[2] + 2; k4 = E[4] - 4 * E[3] - 3 * E[2] ** 2 + 12 * E[2] - 6
    return {1: k1, 2: k2, 3: k3, 4: k4}

def choose_v(h, mode):
    m, S, k3, k4 = h[1], h[2], h[3], h[4]
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    if mode == "k22":
        K22 = torch.einsum("aabb->ab", k4)
        num = K22[off].sum(); den = (torch.outer(S.diagonal(), S.diagonal()) + 2 * S * S)[off].sum()
        v2 = float(num / den)                       # Var(S^2)/E[S^2]^2 ~ (1+v)^4 - 1
        return max((1 + v2) ** 0.25 - 1, 0.0) if v2 > -1 else 0.0
    if mode == "energy":
        # Var(|h|^2) = sum_ab [k4_aabb + 2 S_ab^2 + 4 m_a m_b S_ab + 2 m_a k3_abb + 2 m_b k3_aab]; collective share: (1+v)^4-1 times E^2
        K22 = torch.einsum("aabb->ab", k4); D21 = torch.einsum("aab->ab", k3)
        var = (K22 + 2 * S * S + 4 * torch.outer(m, m) * S + 2 * D21.T * m[:, None] + 2 * D21 * m[None, :])[off].sum()
        E2 = float((m * m + S.diagonal()).sum()) ** 2
        v2 = float(var / E2)
        return max((1 + v2) ** 0.25 - 1, 0.0)
    return 0.0


def dense_tower(K):
    return {d: K[d].to_tensor() if hasattr(K[d], "to_tensor") else K[d].core for d in sorted(K.keys()) if d <= 4}


