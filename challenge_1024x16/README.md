# The challenge cell, 1024x16

The challenge's own networks (Phase 2 public dataset, mini split, nets 0 to 11) and their 1e9-sample truth, scored the way the grader scores. Measured on 29 Oct 2026, after the write-up was first posted; the predictions are in `../docs/PREREG.md`, section C1.

| ARC factored K3 | without scale separation | with scale separation | gain | nets improved |
|---|---|---|---|---|
| SIMPLE, final-layer MSE | 3.89e-8 | 3.11e-8 | 1.25x | 12/12 |
| AUGMENT, final-layer MSE | 1.93e-8 | 1.34e-8 | 1.44x | 6/6 |

In graded entry 336823 the separation improves the leaderboard score 1.73x (1.17e-8 to 6.79e-9, 12/12 nets, same compute).

## Files

| file | what it does |
|---|---|
| `run.sh` | everything below, in order |
| `get_nets.py` | reads the challenge nets and truth from the public dataset |
| `score_k2.py` | scores ARC's K2 (`../arc_ladder/arc.py`) and my estimator (`../estimator/lv2.py`) on those nets |
| `k3_scale_law.py` | ARC's factored K3, unmodified (`--vmode none`) and with the scale carried as a log-normal law (`--vmode k22`) |
| `graded_entry_336823.py` | graded entry 336823 as submitted; the separation is switched by `V30_DESCALE` (MIT notice for its base in `LICENSE-504aldo-V29`) |
| `graded_ablation.sh` | runs the graded entry through the official local runner with the switch on and off |

On one GPU, ARC's K2 and my estimator take seconds per net, factored K3 SIMPLE about 20 s and AUGMENT about 2 minutes. AUGMENT K3 at 1024 needs more than 110 GB in float64; it was run in float32, which reproduces net 1's float64 result to four digits (net 0 is the float64 run).
