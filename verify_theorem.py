"""verify_theorem.py: numerical checks of every statement in THEOREM.md. CPU only, about a minute, no truth or ARC code needed."""
import math
from fractions import Fraction as Fr
import numpy as np

rng = np.random.default_rng(0)


def gamma_s1(V):
    """E sqrt(S) for S ~ Gamma(mean 1, variance V)."""
    a = 1.0 / V
    return math.exp(math.lgamma(a + 0.5) - math.lgamma(a)) / math.sqrt(a)


def gamma_moments(V, K):
    """Exact E S^j, j = 0..K, for S ~ Gamma(mean 1, variance V): prod_{i<j} (1 + i V)."""
    ES = [Fr(1)]
    for j in range(1, K + 1):
        ES.append(ES[-1] * (1 + (j - 1) * V))
    return ES


def binom_half(k):
    c = Fr(1)
    for i in range(k):
        c *= (Fr(1, 2) - i) / (i + 1)
    return c


def cumulants_from_moments(m):
    """Cumulants kappa_1..kappa_N from raw moments m[0..N] (m[0] = 1)."""
    N = len(m) - 1
    k = [Fr(0)] * (N + 1)
    for n in range(1, N + 1):
        k[n] = m[n] - sum(math.comb(n - 1, j - 1) * k[j] * m[n - j] for j in range(1, n))
    return k


print("== Theorem 1: homogeneity factorization ==")
V, M = 0.3, 1 << 22
m = np.array([0.4, -0.2]); C = np.array([[1.0, 0.6], [0.6, 0.8]])
y = m + rng.standard_normal((M, 2)) @ np.linalg.cholesky(C).T
S = rng.gamma(1 / V, V, M)
for (p, q) in [(1, 0), (2, 1), (2, 2), (3, 1)]:
    lhs = np.maximum(np.sqrt(S) * y[:, 0], 0) ** p * np.maximum(np.sqrt(S) * y[:, 1], 0) ** q
    gpart = np.maximum(y[:, 0], 0) ** p * np.maximum(y[:, 1], 0) ** q
    ES = math.exp(math.lgamma(1 / V + (p + q) / 2) - math.lgamma(1 / V)) * V ** ((p + q) / 2)
    z = (lhs.mean() - ES * gpart.mean()) / (lhs - ES * gpart).std() * math.sqrt(M)
    print(f"  E[h1^{p} h2^{q}]  sampled {lhs.mean():.6f}  factorized {ES * gpart.mean():.6f}  z = {z:+.2f}")

print("== Proposition 2: input radial scale, fixed weights ==")
n, L = 32, 8
Ws = [rng.standard_normal((n, n)) * math.sqrt(2 / n) for _ in range(L + 1)]
def f(X):
    h = X
    for W in Ws[:-1]:
        h = np.maximum(h @ W.T, 0)
    return h @ Ws[-1].T
X = rng.standard_normal((4096, n)); r = np.linalg.norm(X, axis=1, keepdims=True)
print(f"  max |f(x) - |x|/sqrt(n) f(sqrt(n) x/|x|)| = {np.abs(f(X) - r / math.sqrt(n) * f(math.sqrt(n) * X / r)).max():.2e}")
S0 = (rng.standard_normal((1 << 20, n)) ** 2).mean(1)
print(f"  E sqrt(S0): sampled {np.sqrt(S0).mean():.6f}  Gamma(mean 1, var 2/n) formula {gamma_s1(2 / n):.6f}  var(S0) {S0.var():.5f} vs 2/n = {2 / n:.5f}")

print("== Theorem 3: cumulants of the scale mode (exact arithmetic) ==")
for V in [Fr(1, 10), Fr(1, 4)]:
    ES = gamma_moments(V, 6)
    mom = [ES[k // 2] * math.prod(range(k - 1, 0, -2)) if k % 2 == 0 else Fr(0) for k in range(13)]
    kap = cumulants_from_moments(mom)
    for r in range(2, 7):
        pred = math.factorial(r - 1) * math.prod(range(2 * r - 1, 0, -2)) * V ** (r - 1)
        print(f"  V = {V}  r = {r}: kappa_{2*r}/kappa_2^{r} = {kap[2*r]}  predicted {pred}  odd kappa_{2*r-1} = {kap[2*r-1]}")
print("  excess kurtosis of a unit readout w.z (scale mixture, V = 0.1) vs w.h (independent ReLU coordinates):")
for n in [64, 1024]:
    w = rng.standard_normal(n); w /= np.linalg.norm(w)
    T = 1 << 16
    u_mix = np.concatenate([(np.sqrt(rng.gamma(10, 0.1, (T, 1))) * rng.standard_normal((T, n))) @ w for _ in range(8)])  # z = sqrt(S) y
    u_ind = np.concatenate([np.maximum(rng.standard_normal((T, n)), 0) @ w for _ in range(8)])
    k = lambda u: ((u - u.mean()) ** 4).mean() / u.var() ** 2 - 3
    mu1, m2, m3, m4 = 1 / math.sqrt(2 * math.pi), 0.5, 2 / math.sqrt(2 * math.pi), 1.5   # moments of relu(g)
    k4 = m4 - 4 * m3 * mu1 + 6 * m2 * mu1**2 - 3 * mu1**4 - 3 * (m2 - mu1**2) ** 2
    pred = float(np.sum(w**4)) * k4 / (m2 - mu1**2) ** 2
    print(f"    n = {n:5d}: mixture {k(u_mix):.4f} (predicted 3V = 0.3000)   independent {k(u_ind):+.4f} (predicted {pred:+.4f}, O(1/n)); sampling SE about {math.sqrt(24 / (8 * T)):.3f}")

print("== Theorem 4: the moment expansion over the scale diverges ==")
from numpy.polynomial.hermite_e import hermeval
from scipy.integrate import quad
relu_hat = lambda k: quad(lambda x: x * hermeval(x, [0] * k + [1]) * math.exp(-x * x / 2) / math.sqrt(2 * math.pi), 0, np.inf, epsabs=1e-13)[0]
for k in range(1, 6):
    gc = relu_hat(2 * k) * math.prod(range(2 * k - 1, 0, -2)) / math.factorial(2 * k)
    print(f"  k = {k}: Gram-Charlier coefficient {gc:+.8f}   binom(1/2,k)/sqrt(2 pi) {float(binom_half(k)) / math.sqrt(2 * math.pi):+.8f}")
for V in [Fr(1, 50), Fr(1, 20), Fr(1, 10), Fr(1, 5)]:
    ES = gamma_moments(V, 60)
    mu = [sum(math.comb(k, j) * ES[j] * (-1) ** (k - j) for j in range(k + 1)) for k in range(61)]
    terms = [binom_half(k) * mu[k] for k in range(61)]
    exact, ps, errs = gamma_s1(float(V)), 0.0, []
    for t in terms:
        ps += float(t); errs.append(abs(ps - exact))
    kstar = min(range(2, 61), key=lambda k: abs(terms[k]))
    print(f"  V = {float(V):.2f}: smallest term at k = {kstar:2d} (0.6/V = {0.6 / float(V):4.1f}); |partial sum - E sqrt S| at k = 1..4: "
          + ", ".join(f"{e:.1e}" for e in errs[1:5]) + f"; at k = 60: {errs[60]:.1e}")

print("== Theorem 5: annealed scale variance (1 + 5/n)^L ==")
for n in [16, 64]:
    N = rng.binomial(n, 0.5, 1 << 22); rho = 2 / n * rng.chisquare(np.maximum(N, 1)) * (N > 0)
    print(f"  n = {n}: E rho {rho.mean():.5f} (1), E rho^2 {np.mean(rho**2):.5f} (1 + 5/n = {1 + 5 / n:.5f})")
n, L, T = 16, 6, 1 << 15
x = rng.standard_normal(n); h = np.tile(x, (T, 1))
for _ in range(L):
    W = rng.standard_normal((T, n, n)) * math.sqrt(2 / n); h = np.maximum(np.einsum("tij,tj->ti", W, h), 0)
q = (h ** 2).sum(1)
print(f"  random nets n = {n}, L = {L}, x fixed: E|h_L|^4/(E|h_L|^2)^2 = {np.mean(q**2) / np.mean(q)**2:.4f} +- {np.std(q**2) / np.mean(q)**2 / math.sqrt(T):.4f}   (1 + 5/n)^L = {(1 + 5 / n) ** L:.4f}")
print(f"  depth = width/4: (1 + 5/n)^(n/4) - 1 = " + ", ".join(f"{(1 + 5 / n) ** (n / 4) - 1:.3f} (n={n})" for n in [64, 256, 1024]) + f"; limit e^(5/4) - 1 = {math.exp(1.25) - 1:.3f}")

print("== Proposition 6: the variance recursion is exact W-averaged matching ==")
d = 6
A = rng.standard_normal((d, d)); K = A @ A.T / d + np.eye(d); a = 0.3 * rng.standard_normal(d)
Vp = 0.15; s1 = gamma_s1(Vp); b = 1 / s1 ** 2 - 1
u = a / s1; Cm = K - b * np.outer(a, a)
N0 = (a @ a + np.trace(K)) ** 2 + 2 * np.trace(K @ K) + 4 * a @ K @ a
closed = N0 - (a @ a) ** 2 * (4 * b + 2 * b * b)
v = u + rng.standard_normal((1 << 22, d)) @ np.linalg.cholesky(Cm).T
q4 = np.sum(v**2, 1)**2
print(f"  E|u + xi|^4: sampled {q4.mean():.4f} +- {q4.std() / math.sqrt(len(q4)):.4f}   N0 - |a|^4 (4b + 2b^2) = {closed:.4f}")
print(f"  moment match: s1 u = a {np.allclose(s1 * u, a)}, C + (1 - s1^2) u u^T = K {np.allclose(Cm + (1 - s1**2) * np.outer(u, u), K)}")
m_out, w2 = 24, 0.37; vv = rng.standard_normal(d)
W = rng.standard_normal((1 << 16, m_out, d)) * math.sqrt(w2); Wv = W @ vv
print(f"  E_W |Wv|^4 / (w2^2 |v|^4): sampled {np.mean(np.sum(Wv**2, 1)**2) / (w2**2 * np.linalg.norm(vv)**4):.2f}   m^2 + 2m = {m_out**2 + 2*m_out}")
