"""Law-vertex (LV) estimator library. State of z_l: (mu, Sigma, V). Law z = sqrt(S)(m + g), g ~ N(0, C), S ~ Gamma(mean 1, var V).
Nothing fitted: m, C from (mu, Sigma, V) exactly; V_{l+1} from the law-implied (2,2) kappa4 coefficient under the exact random-W row average
(derive_k4law.py). Vertex exact per neuron (truncated-normal moments) and Mehler-to-order-R per pair, at Q_S S-nodes."""
import math, numpy as np, torch
from scipy.special import digamma, polygamma, gammaln
from numpy.polynomial.hermite import hermgauss
SQ2PI = math.sqrt(2 * math.pi)

def s_nodes(V, QS, mode="full"):
    """nodes S_q, weights P_q; mode full = Gamma(1/V, V) by Gauss-Hermite in log S with exact reweighting; first = 1 +- sqrt V (first order in V)."""
    if V < 1e-12: return np.array([1.0]), np.array([1.0])
    if mode == "first":
        r = math.sqrt(min(V, 0.999)); return np.array([1 - r, 1 + r]), np.array([0.5, 0.5])
    k, th = 1.0 / V, V
    a = digamma(k) + math.log(th); b = math.sqrt(polygamma(1, k))
    t, w = hermgauss(QS); y = a + math.sqrt(2) * b * t
    logp = k * y - np.exp(y) / th - gammaln(k) - k * math.log(th)
    P = w * np.exp(t**2 + logp) * math.sqrt(2) * b; P = P / P.sum()
    return np.exp(y), P

SCAL = [0]   # hand count of scalar special-function work (gammaln/exp/sqrt), 1 flop each, as ndtr

def s_mom(V, mode="full"):
    """s_p = E S^{p/2}, p = 1..4."""
    SCAL[0] += 12
    if V < 1e-12: return {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    if mode == "first":
        S, P = s_nodes(V, 2, "first"); return {p: float((P * S**(p / 2)).sum()) for p in (1, 2, 3, 4)}
    k, th = 1.0 / V, V
    return {p: math.exp(gammaln(k + p / 2) - gammaln(k)) * th**(p / 2) for p in (1, 2, 3, 4)}

def ndtr(x): return torch.special.ndtr(x)

def trunc_moms(mu, sig, kmax=4):
    """E[X^k 1{X>0}], X ~ N(mu, sig^2), k = 0..kmax (recursion M_k = mu M_{k-1} + (k-1) sig^2 M_{k-2})."""
    t = mu / sig; Phi = ndtr(t); phi = torch.exp(-0.5 * t * t) / SQ2PI
    M = [Phi, mu * Phi + sig * phi]
    for k in range(2, kmax + 1): M.append(mu * M[-1] + (k - 1) * sig * sig * M[-2])
    return M, t, Phi, phi

def deriv_tables(mu, sig, R):
    """D1[r] = E[relu^{(r)}(X)], D2[r] = E[(relu^2)^{(r)}(X)], r = 0..R (each shape like mu)."""
    M, t, Phi, phi = trunc_moms(mu, sig, 2)
    He = [torch.ones_like(t), t.clone()]
    for k in range(1, R): He.append(t * He[k] - k * He[k - 1])
    D1 = [M[1], Phi] + [((-1)**r) * He[r - 2] * phi / sig**(r - 1) for r in range(2, R + 1)]
    D2 = [M[2], 2 * M[1], 2 * Phi] + [2 * ((-1)**(r - 3)) * He[r - 3] * phi / sig**(r - 2) for r in range(3, R + 1)]
    return D1, D2

def law_k4coef(V, u, Ch, mode="full"):
    s = s_mom(V, mode); s1, s3, s4 = s[1], s[3], s[4]
    A4 = -6 * s1**4 + 12 * s1**2 - 4 * s1 * s3 + s4 - 3; A2 = 2 * s1**2 - 2 * s1 * s3 + s4 - 1
    uu = float(u @ u); tr = float(torch.trace(Ch)); uCu = float(u @ Ch @ u); tr2 = float((Ch * Ch).sum())
    return A4 * uu**2 + A2 * (2 * uu * tr + 4 * uCu) + (s4 - 1) * (tr**2 + 2 * tr2)

def vertex(mu, Sig, V, R=16, QS=16, mode="full", need_pairs=True):
    """Moments of h = relu(z) under the law with carried (mu, Sig, V). Returns Eh, Cov h, kappa4 (2,2)-sum."""
    n = mu.shape[0]; dev, dt = mu.device, mu.dtype
    S, P = s_nodes(V, QS, mode); s1 = float((P * np.sqrt(S)).sum())
    S_t = torch.tensor(S, device=dev, dtype=dt); P_t = torch.tensor(P, device=dev, dtype=dt)
    m = mu / s1; C = Sig - (1 - s1 * s1) * torch.outer(m, m)
    cd = torch.clamp(torch.diagonal(C), min=1e-12)
    sq = torch.sqrt(S_t)                                 # (Q,)
    MU = m[:, None] * sq[None, :]; SG = torch.sqrt(cd)[:, None] * sq[None, :]   # (n, Q)
    M, _, _, _ = trunc_moms(MU, SG, 4)
    Em = [(Mk * P_t[None, :]).sum(1) for Mk in M]       # E_S of E[relu^k], k = 0..4
    Eh = Em[1]; Eh2 = Em[2]
    if not need_pairs: return Eh, None, None
    # pair moments by exact conditioning: X_j = mu + sig Z on the active half-line (Gauss-Legendre, R points), Y_l | Z closed form
    xg, wg = np.polynomial.legendre.leggauss(R); xg = torch.tensor(xg, device=dev, dtype=dt); wg = torch.tensor(wg, device=dev, dtype=dt)
    M11 = torch.zeros(n, n, device=dev, dtype=dt); M12 = torch.zeros_like(M11); M22 = torch.zeros_like(M11)
    sd = torch.sqrt(cd)
    for q in range(len(S)):
        Sq = float(S[q]); muq = m * math.sqrt(Sq); sgq = sd * math.sqrt(Sq); Cq = C * Sq
        a = torch.clamp(-muq / sgq, min=-8.5, max=8.5); b = 8.5
        half = 0.5 * (b - a)                                        # (n,)
        Z = a[:, None] + half[:, None] * (xg[None, :] + 1)           # (n, R) nodes per j
        om = half[:, None] * wg[None, :] * torch.exp(-0.5 * Z * Z) / SQ2PI
        X = muq[:, None] + sgq[:, None] * Z                          # (n, R)
        beta = Cq / sgq[:, None]                                     # (j, l): cov / sig_j
        s2 = torch.clamp(sgq[None, :]**2 - beta**2, min=1e-18 * float(sgq.max())**2)
        sc = torch.sqrt(s2)                                          # (j, l)
        mc = muq[None, :, None] + beta[:, :, None] * Z[:, None, :]   # (j, l, R)
        tt = mc / sc[:, :, None]
        Ph = ndtr(tt); ph = torch.exp(-0.5 * tt * tt) / SQ2PI
        G1 = mc * Ph + sc[:, :, None] * ph
        G2 = (mc * mc + s2[:, :, None]) * Ph + mc * sc[:, :, None] * ph
        w1 = (om * X)[:, None, :]; w2 = (om * X * X)[:, None, :]
        M11 += float(P[q]) * (w1 * G1).sum(2)
        M12 += float(P[q]) * (w1 * G2).sum(2)
        M22 += float(P[q]) * (w2 * G2).sum(2)
    eye = torch.eye(n, device=dev, dtype=dt)
    M11 = M11 * (1 - eye) + torch.diag(Eh2)
    Cov = M11 - torch.outer(Eh, Eh)
    var = torch.diagonal(Cov)
    # sum_jl E[dh_j^2 dh_l^2]: off-diagonal from pair table, diagonal exact
    # exact expansion with first moments: E[(h_j^2 - 2a h_j + a^2)(h_l^2 - 2b h_l + b^2)], a = Eh_j, b = Eh_l
    a = Eh[:, None]; b = Eh[None, :]; E1j = Eh[:, None]; E1l = Eh[None, :]
    off = (M22 - 2 * b * M12.T + b * b * Eh2[:, None]
           - 2 * a * M12 + 4 * a * b * M11 - 2 * a * b * b * E1j
           + a * a * Eh2[None, :] - 2 * a * a * b * E1l + a * a * b * b)
    d4 = Em[4] - 4 * Eh * Em[3] + 6 * Eh**2 * Eh2 - 3 * Eh**4      # E (h - Eh)^4
    off = off * (1 - eye) + torch.diag(d4)
    k4sum = float((off - torch.outer(var, var) - 2 * Cov * Cov).sum())
    return Eh, Cov, k4sum

def solve_V(k4sum, Eh, Cov, mode="full", Vmax=0.999, rule="lead"):
    if k4sum <= 0: return 0.0
    if rule == "lead":   # R185 rule: leading order in V, c-part only (the m-coupled A2/A4 terms are O(V^2))
        return min(Vmax, k4sum / (float(torch.trace(Cov))**2 + 2 * float((Cov * Cov).sum())))
    def F(V):
        s1 = s_mom(V, mode)[1]; u = Eh / s1; Ch = Cov - (1 - s1 * s1) * torch.outer(u, u)
        return law_k4coef(V, u, Ch, mode)
    lo, hi = 0.0, Vmax
    if F(hi) < k4sum: return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if F(mid) < k4sum: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def lv_chain(Ws, L, R=16, QS=16, mode="full", Vfix=None, Vread=None):
    """Ws: list of torch weights (out, in), layers 0..L (L = output). Vfix: force V (0 for LV0). Vread: V applied at last vertex only (LV-R).
    Returns Eh_L, y estimate, list of V per layer."""
    n = Ws[0].shape[1]; dev, dt = Ws[0].device, Ws[0].dtype
    mu = torch.zeros(n, device=dev, dtype=dt); Sig = Ws[0] @ Ws[0].T; V = 0.0; Vs = [0.0]
    for l in range(L):
        Vuse = V if Vfix is None else Vfix
        if l == L - 1 and Vread is not None: Vuse = Vread
        last = (l == L - 1)
        Eh, Cov, k4 = vertex(mu, Sig, Vuse, R, QS, mode, need_pairs=not last)
        if last: break
        Vn = solve_V(k4, Eh, Cov, mode) if Vfix is None else Vfix
        W = Ws[l + 1]; mu = W @ Eh; Sig = W @ Cov @ W.T; V = Vn; Vs.append(V)
    return Eh, Ws[L] @ Eh, Vs

def vertex_fast(mu, Sig, V, R=16, mode="full", need_pairs=True):
    """Exact factorization (R186c): z | S = sqrt(S)(m + g) and relu homogeneous -> E[h_j^a h_l^b] = s_{a+b} x Gaussian moment at (m, C).
    One Gaussian vertex (QS = 1 code path) with moments rescaled by closed-form s_k."""
    s = s_mom(V, mode); s1 = s[1]
    m = mu / s1; C = Sig - (1 - s1 * s1) * torch.outer(m, m)
    Eh, Cov0, _ = None, None, None
    # Gaussian vertex at (m, C): call vertex with V = 0 on (m, C) but we need raw pair tables -> recompute here
    n = mu.shape[0]; dev, dt = mu.device, mu.dtype
    cd = torch.clamp(torch.diagonal(C), min=1e-12); sd = torch.sqrt(cd)
    M, _, _, _ = trunc_moms(m, sd, 4)
    Em = [M[0], s[1] * M[1], s[2] * M[2], s[3] * M[3], s[4] * M[4]]
    Eh = Em[1]; Eh2 = Em[2]
    if not need_pairs: return Eh, None, None
    xg, wg = np.polynomial.legendre.leggauss(R); xg = torch.tensor(xg, device=dev, dtype=dt); wg = torch.tensor(wg, device=dev, dtype=dt)
    a = torch.clamp(-m / sd, min=-8.5, max=8.5); b = 8.5; half = 0.5 * (b - a)
    Z = a[:, None] + half[:, None] * (xg[None, :] + 1)
    om = half[:, None] * wg[None, :] * torch.exp(-0.5 * Z * Z) / SQ2PI
    X = m[:, None] + sd[:, None] * Z
    beta = C / sd[:, None]
    s2 = torch.clamp(sd[None, :]**2 - beta**2, min=1e-18 * float(sd.max())**2); sc = torch.sqrt(s2)
    mc = m[None, :, None] + beta[:, :, None] * Z[:, None, :]
    tt = mc / sc[:, :, None]; Ph = ndtr(tt); ph = torch.exp(-0.5 * tt * tt) / SQ2PI
    G1 = mc * Ph + sc[:, :, None] * ph
    G2 = (mc * mc + s2[:, :, None]) * Ph + mc * sc[:, :, None] * ph
    w1 = (om * X)[:, None, :]; w2 = (om * X * X)[:, None, :]
    M11 = s[2] * (w1 * G1).sum(2); M12 = s[3] * (w1 * G2).sum(2); M22 = s[4] * (w2 * G2).sum(2)
    eye = torch.eye(n, device=dev, dtype=dt)
    M11 = M11 * (1 - eye) + torch.diag(Eh2)
    Cov = M11 - torch.outer(Eh, Eh); var = torch.diagonal(Cov)
    A = Eh[:, None]; B = Eh[None, :]
    off = (M22 - 2 * B * M12.T + B * B * Eh2[:, None] - 2 * A * M12 + 4 * A * B * M11 - 2 * A * B * B * A
           + A * A * Eh2[None, :] - 2 * A * A * B * B + A * A * B * B)
    d4 = Em[4] - 4 * Eh * Em[3] + 6 * Eh**2 * Eh2 - 3 * Eh**4
    off = off * (1 - eye) + torch.diag(d4)
    k4sum = float((off - torch.outer(var, var) - 2 * Cov * Cov).sum())
    return Eh, Cov, k4sum

def lvf_chain(Ws, L, R=16, mode="full", Vfix=None):
    n = Ws[0].shape[1]; dev, dt = Ws[0].device, Ws[0].dtype
    mu = torch.zeros(n, device=dev, dtype=dt); Sig = Ws[0] @ Ws[0].T; V = 0.0; Vs = [0.0]
    for l in range(L):
        Vuse = V if Vfix is None else Vfix; last = (l == L - 1)
        Eh, Cov, k4 = vertex_fast(mu, Sig, Vuse, R, mode, need_pairs=not last)
        if last: break
        Vn = solve_V(k4, Eh, Cov, mode) if Vfix is None else Vfix
        W = Ws[l + 1]; mu = W @ Eh; Sig = W @ Cov @ W.T; V = Vn; Vs.append(V)
    return Eh, Ws[L] @ Eh, Vs

def vertex_tri(mu, Sig, V, R=16, mode="full", need_pairs=True, b_up=8.5, ret_t4=False):
    """vertex_fast with each unordered pair evaluated ONCE (condition on X_j for j < l; all four tables E[h_j^p h_l^q], p, q in {1, 2},
    from the same conditioning), halving the pair work. Same estimator up to quadrature error."""
    s = s_mom(V, mode); s1 = s[1]
    m = mu / s1; C = Sig - (1 - s1 * s1) * torch.outer(m, m)
    n = mu.shape[0]; dev, dt = mu.device, mu.dtype
    cd = torch.clamp(torch.diagonal(C), min=1e-12); sd = torch.sqrt(cd)
    M, _, _, _ = trunc_moms(m, sd, 4)
    Em = [M[0], s[1] * M[1], s[2] * M[2], s[3] * M[3], s[4] * M[4]]
    Eh = Em[1]; Eh2 = Em[2]
    if not need_pairs: return Eh, None, None
    iu, ju = torch.triu_indices(n, n, 1, device=dev)
    xg, wg = np.polynomial.legendre.leggauss(R); xg = torch.tensor(xg, device=dev, dtype=dt); wg = torch.tensor(wg, device=dev, dtype=dt)
    a = torch.clamp(-m / sd, min=-b_up, max=b_up); half = 0.5 * (b_up - a)
    Z = a[:, None] + half[:, None] * (xg[None, :] + 1)
    om = half[:, None] * wg[None, :] * torch.exp(-0.5 * Z * Z) / SQ2PI
    X = m[:, None] + sd[:, None] * Z
    w1 = (om * X)[iu]; w2 = (om * X * X)[iu]; Zp = Z[iu]                 # (P, R)
    beta = C[iu, ju] / sd[iu]                                            # (P,)
    s2 = torch.clamp(sd[ju]**2 - beta**2, min=1e-18 * float(sd.max())**2); sc = torch.sqrt(s2)
    mc = m[ju][:, None] + beta[:, None] * Zp
    tt = mc / sc[:, None]; Ph = ndtr(tt); ph = torch.exp(-0.5 * tt * tt) / SQ2PI
    G1 = mc * Ph + sc[:, None] * ph
    G2 = (mc * mc + s2[:, None]) * Ph + mc * sc[:, None] * ph
    v11 = s[2] * (w1 * G1).sum(1); v12 = s[3] * (w1 * G2).sum(1); v21 = s[3] * (w2 * G1).sum(1); v22 = s[4] * (w2 * G2).sum(1)
    M11 = torch.diag(Eh2); M11[iu, ju] = v11; M11[ju, iu] = v11
    M22 = torch.zeros(n, n, device=dev, dtype=dt); M22[iu, ju] = v22; M22[ju, iu] = v22
    M12 = torch.zeros(n, n, device=dev, dtype=dt); M12[iu, ju] = v12; M12[ju, iu] = v21     # M12[j,l] = E[h_j h_l^2]
    eye = torch.eye(n, device=dev, dtype=dt)
    Cov = M11 - torch.outer(Eh, Eh); var = torch.diagonal(Cov)
    A = Eh[:, None]; B = Eh[None, :]
    off = (M22 - 2 * B * M12.T + B * B * Eh2[:, None] - 2 * A * M12 + 4 * A * B * M11 - 3 * A * A * B * B + A * A * Eh2[None, :])
    d4 = Em[4] - 4 * Eh * Em[3] + 6 * Eh**2 * Eh2 - 3 * Eh**4
    off = off * (1 - eye) + torch.diag(d4)
    k4sum = float((off - torch.outer(var, var) - 2 * Cov * Cov).sum())
    if ret_t4:   # V2 rule input: T4 = E|h|^4 = sum_j E h_j^4 + sum_{j != l} E[h_j^2 h_l^2] (exact law moments: s4 x Gaussian table)
        return Eh, Cov, k4sum, float(Em[4].sum() + 2 * v22.sum())
    return Eh, Cov, k4sum

def lvt_chain(Ws, L, R=16, mode="full", Vfix=None, b_up=8.5):
    n = Ws[0].shape[1]; dev, dt = Ws[0].device, Ws[0].dtype
    mu = torch.zeros(n, device=dev, dtype=dt); Sig = Ws[0] @ Ws[0].T; V = 0.0; Vs = [0.0]
    for l in range(L):
        Vuse = V if Vfix is None else Vfix; last = (l == L - 1)
        Eh, Cov, k4 = vertex_tri(mu, Sig, Vuse, R, mode, need_pairs=not last, b_up=b_up)
        if last: break
        Vn = solve_V(k4, Eh, Cov, mode) if Vfix is None else Vfix
        W = Ws[l + 1]; mu = W @ Eh; Sig = W @ Cov @ W.T; V = Vn; Vs.append(V)
    return Eh, Ws[L] @ Eh, Vs

def gmom(V, p):
    """E S^p for Gamma(1/V, V) (p may be half-integer); V = 0 -> 1."""
    if V < 1e-12: return 1.0
    k, th = 1.0 / V, V
    return math.exp(gammaln(k + p) - gammaln(k)) * th**p

RETURN_K3 = [False]

def vertex_A(mu, Sig, V, v, R=16, need_pairs=True, b_up=6.5):
    """LVT-A vertex: law z = sqrt(S)(m + (S-1) v + g), exact Gamma s_k on the scale part, v (the AUGMENT-type kappa3 trace / mean-scale
    coupling) at FIRST order with exact Gaussian derivative tables (Stein: v . d/dm). Returns Eh, Cov h, k4sum, b (kappa3 trace vector, v-free part)."""
    n = mu.shape[0]; dev, dt = mu.device, mu.dtype
    s1, s2, s3, s4 = gmom(V, .5), gmom(V, 1.), gmom(V, 1.5), gmom(V, 2.)
    e1 = s3 - s1                                       # E sqrt(S)(S-1)
    m = (mu - e1 * v) / s1
    c3 = gmom(V, 3.) - 2 * gmom(V, 2.) + 1             # E S (S-1)^2
    Ez = mu
    C = Sig + torch.outer(Ez, Ez) - (torch.outer(m, m) + V * (torch.outer(m, v) + torch.outer(v, m)))   # first order in v (the c3 v v^T term is O(v^2) = O(u^2 / V): dropped)
    cd = torch.clamp(torch.diagonal(C), min=1e-12); sd = torch.sqrt(cd)
    M, _, _, _ = trunc_moms(m, sd, 4)
    t = {k: gmom(V, k / 2 + 1) - gmom(V, k / 2) for k in (1, 2, 3, 4)}   # E S^{k/2}(S-1)
    sk = {1: s1, 2: s2, 3: s3, 4: s4}
    Em = [M[0]] + [sk[k] * M[k] + t[k] * v * k * M[k - 1] for k in (1, 2, 3, 4)]
    Eh = Em[1]; Eh2 = Em[2]
    if not need_pairs: return Eh, None, None, None
    iu, ju = torch.triu_indices(n, n, 1, device=dev)
    xg, wg = np.polynomial.legendre.leggauss(R); xg = torch.tensor(xg, device=dev, dtype=dt); wg = torch.tensor(wg, device=dev, dtype=dt)
    a = torch.clamp(-m / sd, min=-b_up, max=b_up); half = 0.5 * (b_up - a)
    Z = a[:, None] + half[:, None] * (xg[None, :] + 1)
    om = half[:, None] * wg[None, :] * torch.exp(-0.5 * Z * Z) / SQ2PI
    X = m[:, None] + sd[:, None] * Z
    w0 = om[iu]; w1 = (om * X)[iu]; w2 = (om * X * X)[iu]; Zp = Z[iu]
    beta = C[iu, ju] / sd[iu]
    s2c = torch.clamp(sd[ju]**2 - beta**2, min=1e-18 * float(sd.max())**2); sc = torch.sqrt(s2c)
    mc = m[ju][:, None] + beta[:, None] * Zp
    tt = mc / sc[:, None]; Ph = ndtr(tt); ph = torch.exp(-0.5 * tt * tt) / SQ2PI
    G1 = mc * Ph + sc[:, None] * ph
    G2 = (mc * mc + s2c[:, None]) * Ph + mc * sc[:, None] * ph
    g11 = (w1 * G1).sum(1); g12 = (w1 * G2).sum(1); g21 = (w2 * G1).sum(1); g22 = (w2 * G2).sum(1)
    p01 = (w0 * G1).sum(1); p10 = (w1 * Ph).sum(1)                 # E[H_j h_l], E[h_j H_l] (Gaussian at (m, C))
    P = torch.diag(M[1]); P[iu, ju] = p01; P[ju, iu] = p10            # P[j,l] = E[H_j h_l] = d/dm_j E[h_j h_l]
    G11 = torch.diag(M[2]); G11[iu, ju] = g11; G11[ju, iu] = g11
    M11 = s2 * G11 + t[2] * (v[:, None] * P + v[None, :] * P.T)
    M11 = M11 - torch.diag(torch.diagonal(M11)) + torch.diag(Eh2)
    M22 = torch.zeros(n, n, device=dev, dtype=dt); M22[iu, ju] = s4 * g22; M22[ju, iu] = s4 * g22
    M12 = torch.zeros(n, n, device=dev, dtype=dt); M12[iu, ju] = s3 * g12; M12[ju, iu] = s3 * g21
    eye = torch.eye(n, device=dev, dtype=dt)
    Cov = M11 - torch.outer(Eh, Eh); var = torch.diagonal(Cov)
    A = Eh[:, None]; B = Eh[None, :]
    M11k = s2 * G11; M11k = M11k - torch.diag(torch.diagonal(M11k)) + torch.diag(Eh2)
    off = (M22 - 2 * B * M12.T + B * B * Eh2[:, None] - 2 * A * M12 + 4 * A * B * M11k - 3 * A * A * B * B + A * A * Eh2[None, :])
    d4 = Em[4] - 4 * Eh * Em[3] + 6 * Eh**2 * Eh2 - 3 * Eh**4
    off = off * (1 - eye) + torch.diag(d4)
    CovK = M11k - torch.outer(Eh, Eh)
    k4sum = float((off - torch.outer(torch.diagonal(CovK), torch.diagonal(CovK)) - 2 * CovK * CovK).sum())
    # kappa3 trace vector b_i = sum_{j != i} kappa3(h_i, h_j, h_j) (v-free tables): E[XY^2] - E X E Y^2 - 2 E Y E[XY] + 2 E X (E Y)^2
    K3 = M12 - Eh[:, None] * Eh2[None, :] - 2 * Eh[None, :] * M11k + 2 * Eh[:, None] * (Eh**2)[None, :]
    b = (K3 * (1 - eye)).sum(1)
    if RETURN_K3[0]:
        k3d = Em[3] - 3 * Eh2 * Eh + 2 * Eh**3
        return Eh, Cov, k4sum, b, K3 * (1 - eye), k3d, M[0]
    return Eh, Cov, k4sum, b

def lva_chain(Ws, L, R=16, use_v=True, Vfix=None, b_up=6.5, trace=None):
    n = Ws[0].shape[1]; dev, dt = Ws[0].device, Ws[0].dtype
    mu = torch.zeros(n, device=dev, dtype=dt); Sig = Ws[0] @ Ws[0].T; V = 0.0; v = torch.zeros(n, device=dev, dtype=dt); Vs = [0.0]
    for l in range(L):
        Vuse = V if Vfix is None else Vfix; last = (l == L - 1)
        Eh, Cov, k4, b = vertex_A(mu, Sig, Vuse, v, R, need_pairs=not last, b_up=b_up)
        if trace is not None: trace.append(Eh.clone())
        if last: break
        Vn = solve_V(k4, Eh, Cov) if Vfix is None else Vfix
        W = Ws[l + 1]; mu = W @ Eh; Sig = W @ Cov @ W.T
        if use_v and Vn > 1e-6:
            # W-row-averaged kappa3(z')_{abb} = (W b)_a * mean_j W_bj^2 ; law: V' (v'_a + m'_a / 2) mean_b C'_bb (leading order)
            wb2 = float((W * W).sum(1).mean()) / n
            u = (W @ b) * wb2 / float(torch.diagonal(Sig).mean())
            v = u / Vn - 0.5 * mu / gmom(Vn, .5)
        else:
            v = torch.zeros(n, device=dev, dtype=dt)
        V = Vn; Vs.append(V)
    return Eh, Ws[L] @ Eh, Vs

def law_k3diag(m, cdiag, V):
    """per-neuron kappa3 of z = sqrt(S)(m + g), g_a ~ N(0, c_aa): exact from E z^k = E S^{k/2} E (m+g)^k."""
    s1, s2, s3 = gmom(V, .5), gmom(V, 1.), gmom(V, 1.5)
    r1 = s1 * m; r2 = s2 * (m * m + cdiag); r3 = s3 * (m**3 + 3 * m * cdiag)
    return r3 - 3 * r2 * r1 + 2 * r1**3

def lvk3_chain(Ws, L, R=16, b_up=6.5, trace=None, k3mode="pair"):
    """LVT + carried per-neuron kappa3 RESIDUAL (AUGMENT-type kappa3 trace, W-row-averaged), first-order Edgeworth on the per-neuron
    moments of the law vertex (exact Gaussian derivative expectations, homogeneity in S). k3mode: 'mean' corrects E h only."""
    n = Ws[0].shape[1]; dev, dt = Ws[0].device, Ws[0].dtype
    mu = torch.zeros(n, device=dev, dtype=dt); Sig = Ws[0] @ Ws[0].T; V = 0.0; k3 = torch.zeros(n, device=dev, dtype=dt); Vs = [0.0]
    for l in range(L):
        last = (l == L - 1)
        Eh, Cov, k4, b = vertex_A(mu, Sig, V, torch.zeros(n, device=dev, dtype=dt), R, need_pairs=not last, b_up=b_up)
        Vuse_for_tables = V
        # residual kappa3 at z (carried minus law-implied), first-order correction to E h: (res / 6) E[delta'(z)]
        s1 = gmom(V, .5); m = mu / s1; cdg = torch.clamp(torch.diagonal(Sig) - (1 - s1 * s1) * m * m, min=1e-12)
        res = k3 - law_k3diag(m, cdg, V)
        sd = torch.sqrt(cdg); tt = m / sd; phi = torch.exp(-0.5 * tt * tt) / SQ2PI
        Ed1 = gmom(V, -1.0) * (-tt * phi / cdg)             # E_law[delta'(z)] = E S^{-1} x Gaussian E[delta'] at (m, c)
        Eh = Eh + res / 6 * Ed1
        if trace is not None: trace.append(Eh.clone())
        if last: break
        Vn = solve_V(k4, Eh, Cov)
        W = Ws[l + 1]; wb2 = float((W * W).sum(1).mean()) / n
        # carried kappa3(z')_aaa = sum_ijk W_ai W_aj W_ak kappa3(h)_ijk ~ 3 (W b)_a wb2 + sum_i W_ai^3 kappa3(h)_iii   (trace part, as AUGMENT)
        if k3mode == "wavg":
            k3 = 3 * (W @ b) * wb2
        else:   # pair-exact per net: kappa3(z')_aaa = 3 sum_{i != j} W_ai W_aj^2 K3[i, j] + sum_i W_ai^3 kappa3(h_i)   (3-label part omitted)
            K3, k3d = k3_tables(mu, Sig, Vuse_for_tables, R, b_up)
            eye = torch.eye(n, device=dev, dtype=dt)
            k3 = 3 * (W * ((W * W) @ (K3 * (1 - eye)).T)).sum(1) + (W**3) @ k3d
        mu = W @ Eh; Sig = W @ Cov @ W.T; V = Vn; Vs.append(V)
    return Eh, Ws[L] @ Eh, Vs

def k3_tables(mu, Sig, V, R=16, b_up=6.5):
    """kappa3(h_i, h_j, h_j) table (off-diagonal from the pair vertex) and per-neuron kappa3(h_i) at the law vertex (v = 0)."""
    n = mu.shape[0]; dev, dt = mu.device, mu.dtype
    s1 = gmom(V, .5); m = mu / s1; C = Sig - (1 - s1 * s1) * torch.outer(m, m)
    cd = torch.clamp(torch.diagonal(C), min=1e-12); sd = torch.sqrt(cd)
    M, _, _, _ = trunc_moms(m, sd, 4); sk = [1, s1, gmom(V, 1.), gmom(V, 1.5), gmom(V, 2.)]
    Em = [M[0]] + [sk[k] * M[k] for k in (1, 2, 3, 4)]
    Eh, Cov, k4, b = vertex_A(mu, Sig, V, torch.zeros(n, device=dev, dtype=dt), R, b_up=b_up)
    # recompute M12 table
    iu, ju = torch.triu_indices(n, n, 1, device=dev)
    xg, wg = np.polynomial.legendre.leggauss(R); xg = torch.tensor(xg, device=dev, dtype=dt); wg = torch.tensor(wg, device=dev, dtype=dt)
    a = torch.clamp(-m / sd, min=-b_up, max=b_up); half = 0.5 * (b_up - a)
    Z = a[:, None] + half[:, None] * (xg[None, :] + 1); om = half[:, None] * wg[None, :] * torch.exp(-0.5 * Z * Z) / SQ2PI
    X = m[:, None] + sd[:, None] * Z; w1 = (om * X)[iu]; w2 = (om * X * X)[iu]
    beta = C[iu, ju] / sd[iu]; s2c = torch.clamp(sd[ju]**2 - beta**2, min=1e-30); sc = torch.sqrt(s2c)
    mc = m[ju][:, None] + beta[:, None] * Z[iu]; tt = mc / sc[:, None]; Ph = ndtr(tt); ph = torch.exp(-0.5 * tt * tt) / SQ2PI
    G1 = mc * Ph + sc[:, None] * ph; G2 = (mc * mc + s2c[:, None]) * Ph + mc * sc[:, None] * ph
    M12 = torch.zeros(n, n, device=dev, dtype=dt); M12[iu, ju] = sk[3] * (w1 * G2).sum(1); M12[ju, iu] = sk[3] * (w2 * G1).sum(1)
    M11 = Cov + torch.outer(Eh, Eh); Eh2 = Em[2]
    K3 = M12 - Eh[:, None] * Eh2[None, :] - 2 * Eh[None, :] * M11 + 2 * Eh[:, None] * (Eh**2)[None, :]
    k3d = Em[3] - 3 * Eh2 * Eh + 2 * Eh**3
    return K3, k3d


def law_k3_aab(m, C, V):
    """law kappa3(z_a, z_a, z_b) matrix for z = sqrt(S)(m + g)."""
    s1, s2, s3 = gmom(V, .5), gmom(V, 1.), gmom(V, 1.5)
    cd = torch.diagonal(C)
    Ex2x = (m * m + cd)[:, None] * m[None, :] + 2 * C * m[:, None]
    Ex2 = (m * m + cd)[:, None]; Exx = torch.outer(m, m) + C
    return s3 * Ex2x - s2 * s1 * (Ex2 * m[None, :] + 2 * Exx * m[:, None]) + 2 * s1**3 * (m * m)[:, None] * m[None, :]

def lvq_chain(Ws, L, R=16, b_up=6.5, trace=None, cov_corr=True, prop_res=True):
    """LVT-Q: law vertex + carried kappa3 residual (diag q and (a,a,b) matrix Q, residual vs the law), first-order Edgeworth on
    E h and on the pair moments (closed forms by conditioning at 0), residual pushed through the vertex into kappa3(h) tables,
    next-layer kappa3(z) by the exact pair-level contraction (3-label part omitted)."""
    n = Ws[0].shape[1]; dev, dt = Ws[0].device, Ws[0].dtype
    eye = torch.eye(n, device=dev, dtype=dt); zero = torch.zeros(n, device=dev, dtype=dt)
    mu = zero.clone(); Sig = Ws[0] @ Ws[0].T; V = 0.0; q = zero.clone(); Q = torch.zeros(n, n, device=dev, dtype=dt); Vs = [0.0]
    for l in range(L):
        last = (l == L - 1)
        RETURN_K3[0] = not last
        out = vertex_A(mu, Sig, V, zero, R, need_pairs=not last, b_up=b_up)
        RETURN_K3[0] = False
        Eh = out[0]
        s1 = gmom(V, .5); m = mu / s1; C = Sig - (1 - s1 * s1) * torch.outer(m, m)
        cd = torch.clamp(torch.diagonal(C), min=1e-12); sd = torch.sqrt(cd); ta = m / sd
        pa = torch.exp(-0.5 * ta * ta) / SQ2PI / sd                      # density of z_a at 0 (S = 1 Gaussian)
        pda = (m / cd) * pa                                              # its derivative at 0
        Edd = -pda                                                       # E[delta'(X)] = -p'(0)
        Eh = Eh + q / 6 * gmom(V, -1.0) * Edd
        if trace is not None: trace.append(Eh.clone())
        if last: break
        _, Cov, k4, b, K3, k3d, Phi = out
        if cov_corr:
            # pair closed forms at the S = 1 Gaussian (m, C); row a conditions on z_a = 0
            beta = C / cd[:, None]                                       # (a, b): regression coefficient of z_b on z_a
            mc0 = m[None, :] - beta * m[:, None]
            s2c = torch.clamp(cd[None, :] - beta * C, min=1e-18); sc = torch.sqrt(s2c); tau = mc0 / sc
            Pt = ndtr(tau); pt = torch.exp(-0.5 * tau * tau) / SQ2PI
            dH = pa[:, None] * Pt                                        # E[delta(z_a) H(z_b)]
            G0 = mc0 * Pt + sc * pt; dprelu = -pda[:, None] * G0 - pa[:, None] * beta * Pt   # E[delta'(z_a) relu(z_b)]
            f = gmom(V, -0.5)
            dM = f * (q[:, None] / 6 * dprelu + Q / 2 * dH + Q.T / 2 * dH.T + q[None, :] / 6 * dprelu.T)
            dM = dM * (1 - eye)
            dEh = q / 6 * gmom(V, -1.0) * Edd
            Cov = Cov + dM - (torch.outer(dEh, out[0]) + torch.outer(out[0], dEh))   # Cov = M11 - Eh Eh^T, first order
            Cov = Cov - torch.diag(torch.diagonal(Cov)) + torch.diag(torch.diagonal(out[1]) - 2 * out[0] * dEh)
        if prop_res:
            K3 = K3 + (Phi[:, None] * (Phi**2)[None, :]) * Q.T * (1 - eye)      # kappa3(h_i,h_j,h_j) += Phi_i Phi_j^2 kappa3res(z_j,z_j,z_i)
            k3d = k3d + Phi**3 * q
        Vn = solve_V(k4, Eh, Cov)
        W = Ws[l + 1]; WK = W @ K3; W2 = W * W
        Qt = W2 @ WK.T + 2 * (W * WK) @ W.T + (W2 * k3d[None, :]) @ W.T
        mu = W @ Eh; Sig = W @ Cov @ W.T; V = Vn; Vs.append(V)
        s1n = gmom(V, .5); mn = mu / s1n; Cn = Sig - (1 - s1n * s1n) * torch.outer(mn, mn)
        Q = Qt - law_k3_aab(mn, Cn, V); q = torch.diagonal(Q).clone(); Q = Q * (1 - eye)
    return Eh, Ws[L] @ Eh, Vs


# ---------------------------------------------------------------------------------------------------------------------------
# V2 rule (R_VRULE.md, PREREG R193): V_{l+1} from the NORM statistic of z_{l+1} = W h_l, averaged over the fresh row-iid Gaussian
# W_{l+1} (independent of h_l), all terms kept. Law at l+1 with (mu', Sigma') = (W Eh, W Cov W^T) gives z' = sqrt(R) W (u + xi),
# u = Eh / s1(V'), xi ~ N(0, Ch), Ch = Cov - beta u... ; E_W E|z'|^4 matching reduces exactly to the h-level equation
#     (1 + V') [N0 - |a|^4 (4 beta + 2 beta^2)] = T4,   beta = 1 / s1(V')^2 - 1,  a = E h,  K = Cov h,
#     N0 = (|a|^2 + tr K)^2 + 2 tr K^2 + 4 a^T K a  (Gaussian surrogate), T4 = E|h|^4 (law-exact from the vertex).
# Well-posedness: T4 <= N0 -> V' = 0 (flag 'neg'); smallest root on (0, Vmax] by bracketing + bisection; no root -> argmax of the
# left side (flag 'cap': the law's capacity at this (a, K) is below T4). Nothing fitted.
V2FLAGS = []

def solve_V2(T4, Eh, Cov, mode="full", Vmax=0.999):
    a2 = float(Eh @ Eh); trK = float(torch.trace(Cov)); trK2 = float((Cov * Cov).sum()); aKa = float(Eh @ (Cov @ Eh))
    E2 = a2 + trK; N0 = E2 * E2 + 2 * trK2 + 4 * aKa; a4 = a2 * a2
    def f(V):
        if V <= 0: return N0 - T4
        b = 1.0 / s_mom(V, mode)[1] ** 2 - 1.0
        return (1.0 + V) * (N0 - a4 * (4 * b + 2 * b * b)) - T4
    if f(0.0) >= 0: V2FLAGS.append("neg"); return 0.0
    grid = np.concatenate([[0.0], np.geomspace(1e-6, Vmax, 200)])
    fv = [f(v) for v in grid]
    for k in range(1, len(grid)):
        if fv[k] >= 0:
            lo, hi = grid[k - 1], grid[k]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if f(mid) < 0: lo = mid
                else: hi = mid
            V2FLAGS.append("root"); return 0.5 * (lo + hi)
    k = int(np.argmax(fv))
    if k == len(grid) - 1: V2FLAGS.append("vmax"); return float(grid[k])
    lo, hi = grid[max(k - 1, 0)], grid[k + 1]; gr = (math.sqrt(5) - 1) / 2       # golden-section refinement of the argmax (numerics only)
    for _ in range(60):
        c1 = hi - gr * (hi - lo); c2 = lo + gr * (hi - lo)
        if f(c1) > f(c2): hi = c2
        else: lo = c1
    V2FLAGS.append("cap"); return 0.5 * (lo + hi)

def lvt2_chain(Ws, L, R=16, mode="full", b_up=6.5, rule="v2", trace=None):
    """LVT with the V2 rule (rule='v2') or the old leading-order rule (rule='lead', identical to lvt_chain). trace: list to append
    per-layer dicts (V used at the vertex of z_{l+1}, T4, N0 pieces) for diagnostics."""
    n = Ws[0].shape[1]; dev, dt = Ws[0].device, Ws[0].dtype
    mu = torch.zeros(n, device=dev, dtype=dt); Sig = Ws[0] @ Ws[0].T; V = 0.0; Vs = [0.0]
    for l in range(L):
        last = (l == L - 1)
        out = vertex_tri(mu, Sig, V, R, mode, need_pairs=not last, b_up=b_up, ret_t4=not last)
        Eh = out[0]
        if trace is not None: trace.append(dict(Eh=Eh.clone()))
        if last: break
        _, Cov, k4, T4 = out
        Vn = 0.0 if rule == "zero" else (solve_V2(T4, Eh, Cov, mode) if rule in ("v2", "v2f") else solve_V(k4, Eh, Cov, mode))   # zero: V = 0 control (R196)
        W = Ws[l + 1]; mu = W @ Eh; Sig = W @ Cov @ W.T
        Vf = V_feasible(mu, Sig, mode) if rule == "v2f" else np.inf
        if Vf < Vn: Vn = Vf; V2FLAGS.append("feas")
        if trace is not None: trace[-1].update(T4=T4, k4=k4, Vnext=Vn, Vfeas=Vf)
        V = Vn; Vs.append(V)
    return Eh, Ws[L] @ Eh, Vs


def V_feasible(mu, Sig, mode="full"):
    """Largest V for which every per-neuron and every pair marginal of the law at (mu, Sig, V) is a valid Gaussian:
    C = Sig - beta mu mu^T, beta = 1/s1(V)^2 - 1; diag beta < Sig_aa / mu_a^2; pair det linear in beta:
    beta <= (S_aa S_bb - S_ab^2) / (mu_b^2 S_aa + mu_a^2 S_bb - 2 mu_a mu_b S_ab) when the denominator is > 0. O(n^2)."""
    d = torch.diagonal(Sig); m2 = mu * mu
    bd = float((d / torch.clamp(m2, min=1e-300)).min())
    det = torch.outer(d, d) - Sig * Sig; den = torch.outer(d, m2) + torch.outer(m2, d) - 2 * torch.outer(mu, mu) * Sig
    n = mu.shape[0]; off = ~torch.eye(n, dtype=torch.bool, device=mu.device); ok = off & (den > 0)
    bp = float((torch.clamp(det, min=0)[ok] / den[ok]).min()) if bool(ok.any()) else np.inf
    bmax = min(bd, bp) * (1 - 1e-9)
    if bmax <= 0: return 0.0
    def beta(V): return 1.0 / s_mom(V, mode)[1] ** 2 - 1.0 if V > 0 else 0.0
    if beta(0.999) <= bmax: return 0.999
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if beta(mid) <= bmax: lo = mid
        else: hi = mid
    return lo
