# The theory behind scale separation

This is post-submission work (see `README.md` in this folder). It states and proves the results the method rests on. Every statement is checked numerically by `verify_theorem.py`. The script runs in about 15 seconds on a CPU, and its output is in `verify_theorem.log`.

The proofs give four things:

- why the scale is the object to separate,
- why separating it is exact,
- why a cumulant hierarchy that represents the scale order by order cannot converge on it,
- why the scale grows with depth over width.

The final section lists what is not proved.

## Setting and notation

ARC's network is a bias-free ReLU MLP of width $n$. Its input is $x \sim \mathcal{N}(0, I_n)$, and its layers are $z_l = W_l h_{l-1}$, $h_l = \mathrm{ReLU}(z_l)$, with $h_0 = x$. The weights are fixed, and the task is to estimate $\mathbb{E}_x$ of the output.

The scale law at a layer is

$$z = \sqrt{S}\,y, \qquad y \sim \mathcal{N}(m, C), \qquad S \perp y, \qquad S \ge 0,\ \ \mathbb{E}S = 1,\ \ \operatorname{Var}S = V.$$

I write $\mathrm{Gamma}(1, V)$ for the Gamma law with mean 1 and variance $V$ (shape $1/V$, scale $V$). Its moments are $\mathbb{E}S^p = \Gamma(1/V + p)\,V^p/\Gamma(1/V)$. In particular $\mathbb{E}S^2 = 1 + V$ and $s_1(V) := \mathbb{E}\sqrt{S}$.

---

## Theorem 1 (exact factorization)

Let $\varphi$ be positively homogeneous of degree 1, meaning $\varphi(cy) = c\,\varphi(y)$ for every $c \ge 0$. ReLU and leaky ReLU qualify. Under the scale law, for every multi-index $\alpha$,

$$\mathbb{E}\Big[\prod_j \varphi(z_j)^{\alpha_j}\Big] \;=\; \mathbb{E}\big[S^{|\alpha|/2}\big]\;\mathbb{E}_{y \sim \mathcal{N}(m, C)}\Big[\prod_j \varphi(y_j)^{\alpha_j}\Big].$$

**Proof.** By homogeneity $\varphi(\sqrt{S}\,y_j) = \sqrt{S}\,\varphi(y_j)$, so the product equals $S^{|\alpha|/2}\prod_j \varphi(y_j)^{\alpha_j}$. Independence of $S$ and $y$ splits the expectation. $\square$

**Consequence.** Every moment the estimator needs is a Gaussian vertex times a closed-form Gamma moment. The scale costs one scalar per layer and involves no expansion and no quadrature.

---

## Proposition 2 (the input scale is exactly Gamma, for fixed weights)

Fix the weights. Let $f$ be the network's output, $S_0 = \lVert x\rVert^2/n$, and $u = x/\lVert x\rVert$. Then

$$f(x) = \sqrt{S_0}\; f(\sqrt{n}\,u), \qquad S_0 \sim \mathrm{Gamma}(1, 2/n), \qquad S_0 \perp u,$$

and therefore $\mathbb{E}f(x) = s_1(2/n)\,\mathbb{E}f(\sqrt{n}\,u)$ exactly.

**Proof.** Every layer is linear or ReLU with no bias, so $f(cx) = c f(x)$ for $c \ge 0$. Write $x = \sqrt{S_0}\cdot\sqrt{n}\,u$. For a standard Gaussian, $\lVert x\rVert^2 \sim \chi^2_n$ is independent of the uniform direction $u$, and $\chi^2_n/n$ is Gamma with shape $n/2$ and scale $2/n$, that is $\mathrm{Gamma}(1, 2/n)$. $\square$

**Consequence.** A random overall scale is not an approximation layered onto the problem. It is present exactly in the problem, in the Gamma family, before the first layer. The deeper layers add direction-dependent gains to it (Theorem 5).

---

## Theorem 3 (the scale mode's cumulants: coherent, factorial, undiluted)

Take the scale law with $m = 0$, and assume $S$ has a moment generating function near 0.

**(a)** All odd cumulants of $z$ vanish. For $r \ge 1$,

$$\kappa_{2r}(z)_{i_1 \cdots i_{2r}} \;=\; \kappa_r(S)\sum_{\pi \in \mathcal{P}_2(2r)}\ \prod_{\{a,b\} \in \pi} C_{i_a i_b},$$

where $\mathcal{P}_2(2r)$ is the set of perfect pairings of $\{1, \dots, 2r\}$. For $S \sim \mathrm{Gamma}(1, V)$, $\kappa_r(S) = (r-1)!\,V^{r-1}$.

**(b)** For any readout $w$ with $w^\top C w > 0$, the standardized cumulants of $w \cdot z$ are

$$\frac{\kappa_{2r}(w\cdot z)}{\kappa_2(w\cdot z)^r} = (r-1)!\,(2r-1)!!\,V^{r-1},$$

independent of $n$ and of $w$.

**(c)** Compare a vector $h$ with independent coordinates whose standardized cumulants are bounded by $\gamma$. Suppose the readout is balanced: $w_i^2\sigma_i^2 \le (c/n)\sum_j w_j^2\sigma_j^2$, where $\sigma_i^2 = \operatorname{Var}h_i$. Then

$$\Big|\frac{\kappa_{2r}(w\cdot h)}{\kappa_2(w\cdot h)^r}\Big| \;\le\; \gamma\,c^{r-1}\,n^{1-r}.$$

**Proof.** (a) Let $q = t^\top C t/2$. Then $\mathbb{E}\,e^{t\cdot z} = \mathbb{E}\,e^{Sq} = M_S(q)$, so $\log \mathbb{E}\,e^{t\cdot z} = \sum_{r \ge 1}\kappa_r(S)\,(t^\top C t)^r/(2^r r!)$. This is a series in even powers of $t$, so the odd cumulants are zero. The degree-$2r$ term $(t^\top C t)^r/(2^r r!)$ is the degree-$2r$ term of a centered Gaussian's moment generating function. Its coefficient tensor is therefore the Isserlis pairing sum, multiplied by $\kappa_r(S)$. The Gamma cumulant generating function is $-\tfrac{1}{V}\log(1 - Vt)$, which gives $\kappa_r = (r-1)!\,V^{r-1}$.

(b) $w\cdot z = \sqrt{S}\,(w\cdot y)$ is a one-dimensional instance of (a). There are $(2r-1)!!$ pairings, and $\kappa_2(w\cdot z) = w^\top C w$ because $\mathbb{E}S = 1$.

(c) For independent coordinates, $\kappa_{2r}(w\cdot h) = \sum_i w_i^{2r}\kappa_{2r}(h_i)$. Write $x_i = w_i^2\sigma_i^2$. Then $|\kappa_{2r}(w\cdot h)| \le \gamma\sum_i x_i^r \le \gamma\,(\max_i x_i)^{r-1}\sum_i x_i \le \gamma\,(c/n)^{r-1}\big(\sum_i x_i\big)^r$. $\square$

**Consequence.** A wide linear layer dilutes each neuron's own non-Gaussianity at order $2r$ by $n^{1-r}$, but it carries the shared scale through unchanged. After many layers the shared scale is what remains in the high cumulants. A truncated cumulant list has to store it as dense tensors of every even order across every tuple of neurons. Their entries grow factorially in the order, and the list keeps only the first few. The law stores the same object exactly in one scalar.

---

## Theorem 4 (a moment hierarchy over the scale diverges)

Let $S$ be any scale law with $\mathbb{P}(S > 2) > 0$. Gamma and log-normal qualify for every $V > 0$, and so does the input scale $S_0$ of Proposition 2. Let $\xi \sim \mathcal{N}(0, 1)$ and $\sigma > 0$.

**(a)** The Gram–Charlier series of $\mathbb{E}\,\mathrm{ReLU}(\sigma\sqrt{S}\,\xi)$ around $\mathcal{N}(0, \sigma^2)$, organized by moment order $2k$, is, term by term,

$$\frac{\sigma}{\sqrt{2\pi}}\sum_{k \ge 0}\binom{1/2}{k}\,\mathbb{E}\big[(S-1)^k\big],$$

the termwise Taylor expansion of $\sigma\,\mathbb{E}\sqrt{S}/\sqrt{2\pi}$ about $S = 1$.

**(b)** This series diverges: its terms do not tend to zero.

**(c)** For $\mathrm{Gamma}(1, V)$, its terms decrease only up to an order $k^\ast$ that falls as $V$ grows ($k^\ast$ is about $0.6/V$ for $V \le 0.2$, numerically). Beyond $k^\ast$ they grow factorially. Its partial sums are not monotone in the order: the partial sum through $k = 3$ is worse than through $k = 2$ for every $V$ tested.

**Proof.** (a) For even $n$, $\mathbb{E}\,\mathrm{He}_n(a\xi) = (n-1)!!\,(a^2 - 1)^{n/2}$, and the odd-order terms vanish by symmetry. So with $a^2 = S$, the order-$2k$ Gram–Charlier term is $c_{2k}\,(2k-1)!!\,\mathbb{E}(S-1)^k$, where $c_{2k}$ is ReLU's normalized Hermite coefficient. For a deterministic $a^2 = s \in (0, 2)$, the same series is a power series in $t = s - 1$ that sums to $\mathbb{E}\,\mathrm{ReLU}(\sqrt{s}\,\xi) = \sqrt{1+t}/\sqrt{2\pi}$. By uniqueness of power series, $c_{2k}(2k-1)!! = \binom{1/2}{k}/\sqrt{2\pi}$. The script confirms these coefficients to 8 digits.

(b) $\big|\binom{1/2}{k}\big| \ge c\,k^{-3/2}$. Pick $\varepsilon > 0$ with $p := \mathbb{P}(S > 2 + \varepsilon) > 0$. For even $k$, $\mathbb{E}(S-1)^k \ge (1+\varepsilon)^k\,p$. The even-order terms are therefore at least $c\,p\,k^{-3/2}(1+\varepsilon)^k \to \infty$.

(c) The central moments of $\mathrm{Gamma}(1, V)$ are exact polynomials in $V$. The script evaluates the series in exact rational arithmetic and reports $k^\ast$ and the partial-sum errors. $\square$

**Consequence.** Any scheme that represents the scale through the moments or cumulants of $z$ and truncates is summing partial sums of a divergent series. Its useful orders shrink as $V$ grows, and adding an order can make it worse. Theorem 1 sums the series exactly, in closed form, with one scalar. The theorem is about the scale law. Its link to ARC's code is the 42x de-scaling experiment in the write-up (Section 2), which removes exactly this object and restores a monotone ladder.

---

## Theorem 5 (the scale variance is set by depth over width)

Here the weights are random: iid $\mathcal{N}(0, 2/n)$ entries (He initialization), independent of $x$. Then

$$\lVert h_L\rVert^2 = \lVert x\rVert^2\prod_{l=1}^{L}\rho_l, \qquad \rho_l \overset{\text{iid}}{\sim} \tfrac{2}{n}\,\chi^2_{N_l}, \quad N_l \sim \mathrm{Bin}(n, \tfrac12),$$

with the $\rho_l$ independent of $x$. In particular $\mathbb{E}\rho = 1$ and $\mathbb{E}\rho^2 = 1 + 5/n$, so

$$1 + V_L^{\mathrm{ann}} := \frac{\mathbb{E}\lVert h_L\rVert^4}{\big(\mathbb{E}\lVert h_L\rVert^2\big)^2} = \frac{\mathbb{E}\lVert x\rVert^4}{\big(\mathbb{E}\lVert x\rVert^2\big)^2}\Big(1 + \frac{5}{n}\Big)^{L}.$$

At fixed $L$, $V_L^{\mathrm{ann}} = 5L/n + O(L^2/n^2) + O(1/n)$. On the diagonal $L = n/4$, $(1 + 5/n)^{n/4} - 1 \to e^{5/4} - 1 \approx 2.49$, which does not vanish as width grows.

**Proof.** Condition on $h_{l-1}$ and all earlier weights. The rows of $W_l$ are iid Gaussian and independent of $h_{l-1}$, so by rotational invariance $z_l = \lVert h_{l-1}\rVert\sqrt{2/n}\,\xi_l$ with $\xi_l \sim \mathcal{N}(0, I_n)$ independent of the past. If $h_{l-1} = 0$, take $\xi_l$ fresh; the identity still holds. Then $\lVert h_l\rVert^2 = \lVert h_{l-1}\rVert^2\,\rho_l$ with $\rho_l = \tfrac{2}{n}\lVert\mathrm{ReLU}(\xi_l)\rVert^2$, a function of $\xi_l$ alone. Its positive coordinates number $N_l \sim \mathrm{Bin}(n, 1/2)$, and given $N_l$ their squares sum to $\chi^2_{N_l}$. For the moments, $\mathbb{E}\,\mathrm{ReLU}(\xi)^2 = 1/2$ and $\mathbb{E}\,\mathrm{ReLU}(\xi)^4 = 3/2$, so

$$\mathbb{E}\rho^2 = \frac{4}{n^2}\Big[\tfrac{3}{2}n + \tfrac{1}{4}n(n-1)\Big] = 1 + \frac{5}{n}.$$

Independence across layers gives the product. $\square$

This is the norm part of Hanin and Nica's log-normal result [5]. I include it because it pins down the depth problem's parameter: the scale variance is a function of $L/n$, small when the network is wide and shallow and of order one on the diagonal.

**Quenched versus annealed.** ARC's problem fixes the weights and averages over $x$ only. The scale variance that matters there is the input-conditional part, which is smaller. Measured on development nets at depth it is 0.12 to 0.22 (`../docs/VRULE.md`, Section 4). Theorem 5 gives the annealed law and the scaling variable, not the quenched value. The estimator does not use Theorem 5; it tracks the quenched $V$ with Proposition 6.

---

## Proposition 6 (the variance recursion is exact matching)

Let $h$ have mean $a$, covariance $K$ and $T_4 = \mathbb{E}\lVert h\rVert^4$. Let $W$ be $m \times n$ with iid entries of variance $w_2$, independent of $h$. Model $z' = Wh$, given $W$, by the scale law $z' = \sqrt{R}\,(u + \xi)$ with $R \sim \mathrm{Gamma}(1, V')$ and $\xi \sim \mathcal{N}(0, C)$, with mean and covariance matched:

$$u = \frac{Wa}{s_1(V')}, \qquad C = W\big(K - b\,aa^\top\big)W^\top, \qquad b = \frac{1}{s_1(V')^2} - 1.$$

Then the weight-averaged fourth norm moments agree, $\mathbb{E}_W\mathbb{E}_{\mathrm{law}}\lVert z'\rVert^4 = \mathbb{E}_W\mathbb{E}\lVert Wh\rVert^4$, if and only if

$$(1 + V')\Big[N_0 - \lVert a\rVert^4\,(4b + 2b^2)\Big] = T_4, \qquad N_0 = \big(\lVert a\rVert^2 + \operatorname{tr}K\big)^2 + 2\operatorname{tr}K^2 + 4\,a^\top K a.$$

**Proof.**

1. *Matched moments.* Since $\mathbb{E}\sqrt{R} = s_1$ and $\mathbb{E}R = 1$, the law has mean $s_1 u = Wa$ and covariance $C + (1 - s_1^2)\,uu^\top = W K W^\top$. So the first two moments match.

2. *The weight average.* For a fixed vector $v$, $Wv \sim \mathcal{N}(0, w_2\lVert v\rVert^2 I_m)$, so $\mathbb{E}_W\lVert Wv\rVert^4 = w_2^2(m^2 + 2m)\lVert v\rVert^4$. On the truth side this gives $w_2^2(m^2 + 2m)\,T_4$.

3. *The law side.* Write $\tilde u = a/s_1$ and $\tilde\xi \sim \mathcal{N}(0, K - b\,aa^\top)$. Then $u + \xi \overset{d}{=} W(\tilde u + \tilde\xi)$, so the law side is $\mathbb{E}R^2\cdot w_2^2(m^2 + 2m)\,\mathbb{E}\lVert\tilde u + \tilde\xi\rVert^4$. This needs $K - b\,aa^\top \succeq 0$. The final expression is a polynomial in the covariance, so it holds on all symmetric matrices once it holds on the open positive-definite cone.

4. *The Gaussian quartic.* For $v \sim \mathcal{N}(\mu, \Sigma)$, $\mathbb{E}\lVert v\rVert^4 = (\lVert\mu\rVert^2 + \operatorname{tr}\Sigma)^2 + 2\operatorname{tr}\Sigma^2 + 4\mu^\top\Sigma\mu$. Substitute $\lVert\tilde u\rVert^2 = (1 + b)\lVert a\rVert^2$ and $\Sigma = K - b\,aa^\top$:
   - $\lVert\tilde u\rVert^2 + \operatorname{tr}\Sigma = \lVert a\rVert^2 + \operatorname{tr}K$,
   - $\operatorname{tr}\Sigma^2 = \operatorname{tr}K^2 - 2b\,a^\top K a + b^2\lVert a\rVert^4$,
   - $\tilde u^\top\Sigma\tilde u = (1 + b)\big(a^\top K a - b\lVert a\rVert^4\big)$.

   The $a^\top K a$ terms collect to $4\,a^\top K a$, and the $\lVert a\rVert^4$ terms collect to $-(4b + 2b^2)\lVert a\rVert^4$.

5. *Conclusion.* $\mathbb{E}R^2 = 1 + V'$, and the common factor $w_2^2(m^2 + 2m)$ cancels. $\square$

**Consequences.**

- The rule matches the inherited scale exactly: if $h$ itself carries a scale $S$ independent of its shape, both $T_4$ and the law side carry the factor $\mathbb{E}S^2$.
- The leading-order rule drops the mean-coupled terms $4\,a^\top K a$ and $\lVert a\rVert^4(4b + 2b^2)$. On a mean-dominated (rank-collapsed) state these terms are not small, which is why the leading-order $V$ drifts toward zero at large width (`../docs/VRULE.md`).

---

## What is proved and what is not

**Proved.**

- Separating the scale is exact for any positively homogeneous activation (Theorem 1).
- The scale is present exactly, and Gamma-distributed, at the input of ARC's problem (Proposition 2).
- Linear layers dilute every per-neuron non-Gaussianity but not the shared scale, which therefore fills every even cumulant order coherently with factorially growing size (Theorem 3).
- Representing the scale order by order is a divergent expansion whose useful orders shrink as the scale variance grows (Theorem 4).
- The scale variance is governed by $L/n$ and is of order one on the diagonal (Theorem 5, annealed).
- The variance recursion is exact fourth-moment matching under a weight average (Proposition 6).

**Not proved.**

- An error bound for the estimator on fixed-weight networks. The law is exact for the scale but Gaussian for the shape. The shape's own non-Gaussianity, per Theorem 3(c), is diluted by width but not removed. The margins in the write-up are measured, not derived.
- That the Gamma family is the exact scale law beyond the input. It is exact at the input (Proposition 2), and the matched $V$ agrees with independent truth statistics to about 3 % (`../docs/VRULE.md`), but deeper layers are not proved to be Gamma.
- That Theorem 4 is the specific mechanism of ARC's K3 to K4 inversion inside their code. Theorem 4 concerns the scale law. The causal link to ARC's ladder is the de-scaling experiment, which is measured.
- The quenched scale variance. Theorem 5 is annealed over weights.

## References

[5] B. Hanin, M. Nica. Products of many large random matrices and gradients in deep neural networks. arXiv:1812.05994.
