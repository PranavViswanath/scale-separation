# R_VRULE: the scale-variance recursion of the law vertex, derived with all terms (2026-10-28, 23:10-23:30 UTC Spark host)

Labels: [M] measured, [D] derived, [C] conjecture. Stamp: PREREG R193.

Code is in sweep/:
- lvlib.py: `vertex_tri(..., ret_t4=True)`, `solve_V2`, `V_feasible`, `lvt2_chain`. The pre-V2 copy is lvlib_preV2.py.
- Checks: verify_v2.py, vtraj_mc.py, vtraj_cmp.py (log vtraj_cmp3.log), feas_true.py, trace_f1.py.

## 1. Matching condition and why
Under the law at layer l, z = sqrt(S)(m + g), with g ~ N(0, C) and S ~ Gamma(1, V). Homogeneity gives h = sqrt(S) G, with G = relu(m + g). So z' = W h = sqrt(S) W G exactly.

The statistic that defines V' is the norm fluctuation of z': E|z'|^4 at fixed E|z'|^2, averaged over the fresh row-iid W_{l+1}. W_{l+1} is independent of h_l, so this average is an exact expectation [D].

Reasons for this choice:
1. **It is the scale variable.** By homogeneity, |h_L(x)| = |x| x (product of gains).
2. **It composes exactly.** E|z'|^4 = E S^2 E|WG|^4. The inherited S, including every mean-coupled term, is carried with no matching. Only the born part is matched, and (1+V') ~ (1+V)(1+V_born) [D].
3. **It costs only pair work.** It reduces to pair-level quantities, O(n^2), with no 4-index cumulant (section 2).
4. **It agrees with the truth's other statistic.** On true dev states, the law at the measured LS Var_eff reproduces the norm-type statistics rV_P and rV_T to about 3 % [M, R_POLAR logs]. So this statistic and the (2,2)-LS statistic agree on the truth.

The per-neuron (2,2) sum (the old rule's statistic, matched with all its terms) has no root once the mean dominates (R186a), and was not used.

## 2. Closed form [D]
For any fixed v, E_W (v^T W^T W v)^2 = w2^2 (n^2 + 2n)|v|^4.

- **Truth side:** E_W E|z'|^4 = w2^2 (n^2+2n) E|h|^4.
- **Law side:** the law at l+1 with (mu', Sigma') = (W a, W K W^T) is z' = sqrt(R) W(u + xi), with u = a/s1(V'), xi ~ N(0, Ch), and Ch = K - b a a^T, b = 1/s1(V')^2 - 1.

The W factors cancel, which leaves

  (1+V') [N0 - |a|^4 (4b + 2b^2)] = T4,
  N0 = (|a|^2 + tr K)^2 + 2 tr K^2 + 4 a^T K a,
  T4 = E|h|^4 = s4 x (Gaussian pair table), exact from the vertex.

Here a = E h and K = Cov h. The b-terms collapse exactly.

**Leading order.** Using T4 - N0 = sum_jl kappa4(h)_jjll + 4 a.b, with b_i = sum_j kappa3(h)_ijj:

  V' = (k4sum + 4 a.b) / ((tr K)^2 + 2 tr K^2 + 2|a|^2 tr K + 4 aKa).

The old rule is k4sum / ((tr K)^2 + 2 tr K^2). It drops the mean-coupled terms: 4 a.b in the numerator and 2|a|^2 tr K + 4 aKa in the denominator. Those terms dominate once the state is mean-dominated (rank collapse). This is why the old V decays toward 0 at large width.

## 3. Well-posedness [D]
1. **Sub-Gaussian norm (T4 <= N0).** The implied V is <= 0 and is set to 0, because a Gamma law cannot carry a sub-Gaussian scale. This never occurred in any chain [M].
2. **Root exists.** V' is the smallest root on (0, 0.999], found by grid bracketing and bisection.
3. **No root ('cap').** T4 exceeds the law's maximum norm fluctuation at this (a, K). V' is then the argmax of the left side, refined by golden section: the closest the law can reach. This is the common case at depth (cap:root about 3:1 at G and H) [M].
4. **Feasibility cap.** V' is capped at the largest V for which every per-neuron and every pair marginal of C' = Sigma' - b mu' mu'^T is a valid Gaussian. The pair determinant is linear in b, so this is closed form and O(n^2).
   - Without this cap, devF net 1 ran away. A neuron with c_aa <= 0 gave a garbage T4, then V hit 0.999 and the chain diverged [M].
   - Full PSD of C' is unusable, because the true Sigma' is numerically rank-deficient at depth [M].
   - The true Var_eff sits at the pairwise-feasibility edge at depth [M]: devF 0.19-0.22 against a bound of 0.21-0.23; devG 0.12-0.13 against 0.14.
5. **Parameters and cost.** The rule has no parameters and adds about 1-2 % FLOPs [M].

## 4. Verification [M]
**(a) Exact law sampling** (2^24 draws; SE from 16 sub-batches; no truth used):

| state | rule V' vs sampled V' |
|---|---|
| n = 8, three random states, V in {0, 0.1, 0.3} | \|z\| <= 1.93 |
| devB layer-8 state, n = 64, V in {0.17, 0.25, 0.40} | \|z\| <= 0.53 (the 0.40 case is 'cap' on both sides) |

- The W-average reduction was checked with 512 sampled W x 65536 x: E_W E|z'|^4 3.65893e3 against law(V') 3.66016e3. The paired difference is -3.4e-4 relative (1.5 sigma).

**(b) Dev nets** (seeds 9e6+, 2^22 per-layer MC; diagnostic only). In the deep half, chain V minus the truth of the rule's own statistic:

| rule | 9 nets | at devG |
|---|---|---|
| old | -0.10 to -0.19 on 8/9 | V_L 0.000-0.019 against true LS Var_eff 0.12-0.15 |
| V2 | within 0.01 on 7/9; -0.02 to -0.04 against LS Var_eff at F/G | — |

- On devB net 1, V2 overshoots (0.54 against 0.09). V2 cannot decrease when the born excess is positive.

## 5. Caveat found later (R194)
The frozen pair quadrature (R = 16, b = 6.5) is unconverged at n = 512. Its relative error in the final E h is:

| devB | devF | devG | devH |
|---|---|---|---|
| 4e-5 | 2e-4 | 3e-4 | 4e-3 |

(24, 6.5) is converged to <= 7e-8 relative on all four dev widths. It was chosen by a truth-free rule and is used for all final scores (R_DIAGONAL.md).
