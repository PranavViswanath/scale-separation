# Scale separation for cumulant propagation

Code for "Solving cumulant propagation's depth problem with scale separation" (WRITEUP.md), a submission to ARC's Algorithmic Contribution Prize.

ARC's cumulant propagation estimates the mean output of a random ReLU MLP without running it, and its paper leaves one regime open: depth growing in proportion to width. ARC states it in footnote 7 of their post on the paper, [Mechanistic estimation for wide random MLPs](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/): "It remains an open problem to match the performance of random sampling for random MLPs in other regimes, for example, with the depth growing linearly in the width." In that regime ARC's ladder of approximations stops improving and K4 comes out worse than K3. I trace that to one object, the random overall scale of each layer's activations, separate it exactly using ReLU's positive homogeneity, and carry its variance with an exact recursion.

## Results

Depth = width/4, 10 random networks per cell, normalized MSE against Monte Carlo truth in two independent halves, FLOPs counted with ARC's own counter.

| width x depth | ARC's ladder | mine vs naive sampling at equal FLOPs | mine vs ARC's best at my FLOPs |
|---|---|---|---|
| 64x16 | K4 worse than K3 on 10/10 nets | 2.31x | 2.8x |
| 128x32 | K4 worse than K3; AUGMENT K3 diverges | 1.57x | 4.1x |
| 256x64 | K4 and AUGMENT diverge | 1.53x | 3.1x |
| 512x128 | AUGMENT K2 diverges | 1.42x | 1.3x |
| 1024x256 | K2 below sampling; AUGMENT K2 diverges | 1.21x | 2.1x |

- Removing the scale mode from ARC's dense K4 chain at 64x16 cuts its error 42x and makes the ladder monotone again.
- Carrying the scale variance with its mean-coupled terms is what keeps the estimator ahead of sampling as width grows, through 1024x256.
- ARC's own ladder, run with the scale carried as an exact law, gives a K4 37x better than ARC's K4 at 64x16 and a K3 about 5x better than the best of ARC's ladder at 128x32 and 256x64.
- The estimator costs about what ARC's K2 costs. Nothing in it is fitted.

## The idea in three lines

Model each pre-activation layer as z = sqrt(S) (m + g), with g Gaussian and S a Gamma-distributed scale of variance V.
ReLU(sqrt(S) y) = sqrt(S) ReLU(y), so every moment is E[S^(k/2)] times the matching Gaussian moment, exactly.
Carry the mean, the covariance and V; the next layer's V solves an exact equation for its norm statistic, built from the law's own moments.

## Phase 2 submission and later work

Everything in this repository outside `extensions/`, through the 1024x256 cell, is my Phase 2 Algorithmic Contribution to ARC. That state is tagged `phase2-submission`.

`extensions/` holds work I did afterwards, on my own, extending the theory. It is not part of the submission. I share it in case it is useful to ARC.

## Files

| file | what it does |
|---|---|
| `lvlib.py` | the estimator: law vertex, both variance rules (`lvt2_chain`, `solve_V2`) |
| `lv2.py` | runs the estimator with the exact variance rule, plus the leading-order rule and first-order controls |
| `verify_v2.py` | checks the variance recursion against exact samples of the law, without truth |
| `VRULE.md` | derivation of the exact scale-variance recursion |
| `conv.py` | the truth-free convergence rule that sets the pair-vertex quadrature |
| `expo.py` | the width exponent of the margin over sampling, with bootstrap intervals |
| `nets.py` | builds the random networks with fixed seeds, using ARC's MLP class |
| `mc.py` | Monte Carlo truth in two independent halves, plus antithetic and radial-exact sampler variances |
| `arc.py` | ARC's ladder, SIMPLE and AUGMENT, K1 to K4, unmodified |
| `dsc.py`, `descale_src.py` | de-scaling ARC's dense chain (the 42x test and the dense K4 ladder) |
| `sl_fac.py`, `descale_fac_src.py` | ARC's factored K3 with the exact scale law |
| `ana.py` | scores every method: errors, bootstrap intervals, ratios against sampling and ARC's best |
| `figures/figs.py` | the figures in the write-up |
| `PREREG.md` | the timestamped predictions, written before each result, with the outcomes |

## Running a cell

Install ARC's `mlp_cumulant_propagation` (commit 93d091a) so that `mlp_kprop` imports, then:

```
./run_cell.sh F 128 32 10
```

That builds 10 nets at width 128 and depth 32, computes the truth, runs ARC's ladder and the estimator, and prints the comparison table. A cell at this size takes minutes on one GPU, apart from the Monte Carlo truth.

The de-scaled ladders run separately on a cell built the same way. `python dsc.py B 3S:k22,4S:k22` is the 42x test (log-normal scale), `python dsc.py B 3S:gam,4S:gam` uses the exact Gamma law, and `python sl_fac.py B` is the factored K3 version.
