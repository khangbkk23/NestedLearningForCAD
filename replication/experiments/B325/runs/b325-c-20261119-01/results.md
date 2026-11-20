# B325 job C, seed 2711: code-path check for arm C (b325-c-20261119-01)

Exploratory (exposed MVTec test). Decision rule fixed before the run (card section 11). Label:
**`C_PATH_FAITHFUL`**. Full numbers: [metrics.json](metrics.json); environment, gates and hashes:
[execution.json](execution.json).

## Execution checks

- torch 2.11.0+cu130 on a T4.
- Gate: 23 driver tests and 30 of Khang's tests passed; 3,629 train images byte-identical to Khang's manifests;
  Khang's `BASE.pt` has SHA `dfe6689e…`.
- Training: C kept γ = 0 and λ = 0 for all 15 tasks, 688 s of update time. The initialisation digest equals that of
  F and H in job Q.
- Fidelity: bf16 passed (0.0186), so scored in bf16. Whole job 4,348 s, no traceback in the log.

## Comparison with Khang's C (seed 2711, same BASE, readout A1)

| | Ours (`acquire`, γ = 0) | Khang (`fit_task`) | Δ |
| --- | ---: | ---: | ---: |
| Final Pixel AP | 0.5860 | 0.5898 | −0.0038 |
| Image AUROC | 0.9649 | 0.9649 | +0.0000 |
| Pixel AP at acquisition | 0.5809 | 0.5814 | −0.0006 |
| Pixel FM | 0.0072 | 0.0057 | +0.0015 |

The largest per-category difference is 0.021 (tile). This is a cross-run difference; no cause is assigned.

## Reading

1. **C trained through `acquire(γ=0)` matches Khang's C within the pre-set tolerance, at one seed (2711).** BASE
   identified by its sidecar. Limitation 2 of Q-HP (C on an unchecked code path) is lifted at this level. The Pixel AP
   difference −0.0038 is larger than those of F (+0.0011) and H (−0.0018) in job Q, but within 0.005.
2. **Direction of the difference.** Our C is 0.0038 below Khang's while our F is 0.0011 above his, so at seed 2711
   our F − C is +0.0164 versus Khang's +0.0115.
   - *Illustrative sensitivity, not a correction.* C (job C) and F (job Q) at seed 2711 ran in different jobs, so the
     −0.0038 is not isolated to the code path. If our C were systematically about 0.004 low, F − C in R (+0.0164,
     +0.0207) would be inflated by about that much; subtracting it leaves about +0.012 to +0.017, still above the
     `HARDPATCH_SUPPORTED` threshold. This is arithmetic on point estimates, not a test; one seed cannot separate a
     systematic offset from noise.
3. R's labels are unchanged, as the card specifies.
