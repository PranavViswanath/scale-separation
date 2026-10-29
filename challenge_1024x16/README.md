# The challenge cell, 1024x16

The challenge's own networks (Phase 2 public dataset, mini split, nets 0 to 11) and their 1e9-sample truth, scored the way the grader scores. Measured on 29 Oct 2026, after the write-up was first posted; the predictions are in `../docs/PREREG.md`, section C1.

| method | without scale separation | with scale separation | gain | nets improved |
|---|---|---|---|---|
| graded entry 336823, adjusted score (official local runner) | 1.17e-8 | 6.79e-9 | 1.73x | 12/12 |
| graded entry 336823, final-layer MSE | 4.59e-8 | 2.65e-8 | 1.73x | 12/12 |
| ARC factored K3 SIMPLE, final-layer MSE | 3.89e-8 | 3.11e-8 | 1.25x | 12/12 |
| ARC factored K3 AUGMENT, final-layer MSE | 1.93e-8 | 1.34e-8 | 1.44x | 6/6 |

Compute is unchanged by the separation (graded entry: 0.256 of the 2^41 budget in both runs).

For reference, the K2-cost methods at this shape (final-layer MSE, mean of 12 nets): ARC SIMPLE K2 4.47e-6, ARC AUGMENT K2 1.08e-6, my standalone estimator 1.37e-6 (3.3x better than SIMPLE K2, 1.3x worse than AUGMENT K2).

## Files

| file | what it does |
|---|---|
| `run.sh` | everything below, in order |
| `get_nets.py` | reads the challenge nets and truth from the public dataset |
| `score_k2.py` | scores ARC's K2 (`../arc_ladder/arc.py`) and my estimator (`../estimator/lv2.py`) on those nets |
| `k3_scale_law.py` | ARC's factored K3, unmodified (`--vmode none`) and with the scale carried as a log-normal law (`--vmode k22`) |
| `graded_entry_336823.py` | the estimator of graded entry 336823, as submitted. It is built on 504aldo's public V29 estimator (MIT license; full notice in `LICENSE-504aldo-V29`); the scale separation is the `DESCALE` blocks, switched by `V30_DESCALE` |
| `graded_ablation.sh` | runs the graded entry through the official local runner with the switch on and off |

AUGMENT K3 at 1024 needs more than 110 GB in float64; it was run in float32, which reproduces net 1's float64 result to four digits (net 0 is the float64 run).
