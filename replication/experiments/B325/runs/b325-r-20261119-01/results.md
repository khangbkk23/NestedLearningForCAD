# B325 job R, seeds 2712 and 2713 (b325-r-20261119-01)

Exploratory (exposed MVTec test). Decision rules fixed before the run (card section 10); the driver computed the
labels from the `RESULT.json` files.

| Question | Label |
| --- | --- |
| Q-NL | **`NL_NULL_REPLICATED`** |
| Q-HP | **`HARDPATCH_SUPPORTED`** |

Full numbers: [metrics.json](metrics.json); environment, gates and hashes: [execution.json](execution.json).

## Execution checks

- torch 2.11.0+cu130, 2 × T4; 21 driver tests and 30 of Khang's tests passed; 3,629 train images byte-identical to
  Khang's manifests.
- **BASE:** one BASE per seed, generated on the T4 (2712: `eebf94ce…`; 2713: `d3f30604…`); all three arms of a seed
  used exactly that hash.
- **Matching across arms:** within a seed, C, F and H start from the same LoRA initialisation digest; the digests
  differ between the seeds.
- **Fixed policies:** C kept γ = 0, F kept γ = 2.1, and λ = 0 for all 15 tasks.
- **Precision:** bf16 fidelity passed for all 6 arms (0.0168–0.0184, threshold 0.05), so both seeds were scored in
  bf16.
- **Time:** whole job 12,085 s (limit 18,000 s); 707–798 s of update time per arm; no traceback in the log.

## Main numbers (readout A1, macro over 15 categories, final Pixel AP)

| Seed | C (γ = 0) | F (γ = 2.1) | H (HOPE-S) | F − C [interval] | H − F [interval] |
| --- | ---: | ---: | ---: | --- | --- |
| 2712 | 0.5912 | 0.6076 | 0.6063 | +0.0164 [+0.0009, +0.0341] | −0.0014 [−0.0071, +0.0059] |
| 2713 | 0.5849 | 0.6055 | 0.5982 | +0.0207 [+0.0074, +0.0333] | −0.0074 [−0.0139, −0.0008] |
| Mean | | | | +0.0186 | −0.0044 |

The intervals are category bootstraps **within one seed**, not over seeds, and there are only n = 2 seeds.

**Reported alongside, not used by the labels:**

| Difference | Seed | Pixel AP raw | Image AUROC | Pixel AP at acquisition | Pixel FM |
| --- | --- | ---: | ---: | ---: | ---: |
| F − C | 2712 | +0.0183 | +0.0097 | +0.0146 | −0.0036 |
| F − C | 2713 | +0.0228 | +0.0112 | +0.0169 | −0.0051 |
| H − F | 2712 | −0.0020 | −0.0014 | −0.0020 | +0.0012 |
| H − F | 2713 | −0.0082 | −0.0027 | −0.0078 | +0.0021 |

γ of H at tasks 1 / 4 / 8 / 15 has the same shape as Khang's (0.36 / 1.16 / 2.05 / 3.44): seed 2712 0.48 / 1.20 /
2.13 / 3.57; seed 2713 0.32 / 1.14 / 2.04 / 3.54.

## Reading

### Q-NL: the learned slow level adds nothing over the fixed γ, and is slightly worse

- Label `NL_NULL_REPLICATED`: mean −0.0044, inside the 0.005 band with a margin of only 0.0006; |Δ| < 0.010 at each
  seed.
- The accurate description is **"no gain, a small and consistent shortfall"**, not "equivalent": H − F is negative at
  all three seeds run (−0.0036 at 2711 in job Q, −0.0014, −0.0074); the interval at seed 2713 excludes 0; H is also
  lower at acquisition and forgets more at both new seeds.
- H − F compares learned (γ, λ) with fixed (γ = 2.1, λ = 0), not γ alone.
- No experiment explains why H is slightly worse, so no mechanism is proposed.
- The prior written before the runs (Q-NL null at both seeds) is confirmed.

### Q-HP: the fixed hard-patch loss beats the plain patch loss at both new seeds

- Label `HARDPATCH_SUPPORTED`: F − C = +0.0164 and +0.0207; the category intervals of both seeds exclude 0.
- The direction holds for the raw readout, Image AUROC, acquisition (+0.015 to +0.017) and FM (lower), so the gain
  comes mainly from learning the current task better, together with less forgetting.
- The prior (positive but possibly below the threshold) is exceeded: both seeds pass.
- **Limits that go with this label:**
  - **γ = 2.1 is post hoc.** Khang chose it from H's trajectory at seed 2711. The new seeds reproduce the result with
    a new seed and a new BASE, but on the same test set and task order: a replication, not a generalisation.
  - **Our C uses a different code path from Khang's, checked after R.** Job C (seed 2711, Khang's BASE) gives
    `C_PATH_FAITHFUL`: Pixel AP −0.0038, Image AUROC +0.0000 ([results](../b325-c-20261119-01/results.md)). If our C
    is systematically about 0.004 low, F − C here is inflated by about that much, and still above the threshold.
  - **The intervals are not over seeds**, and n = 2.
  - **The gain is not uniform over categories.** Categories below the −0.01 floor: F − C at 2712: carpet −0.031,
    leather −0.027, wood −0.013, grid −0.011; at 2713: tile −0.026, carpet −0.020. Object categories gain (10/15 and
    13/15 categories improve). Carpet loses at both seeds and most of the losses are textures. This is a description,
    not a mechanism; two seeds cannot estimate per-category noise. The 0.038 cross-run difference seen in job Q is not
    a usable noise margin for within-job comparisons.

### F over three seeds (descriptive)

Pixel AP 0.6024 (2711), 0.6076 (2712), 0.6055 (2713): mean 0.6052, standard deviation 0.0026. Seed 2711 uses
Khang's BASE; the other two use BASEs generated on the T4.

## Consequences for the method (scope: two seeds, two T4-generated BASEs, one task order, exposed MVTec test)

1. The learned slow level of HOPE-S **does not improve** over a fixed γ. This repeated with new seeds, new BASEs and
   another GPU. It does not rule out the class of nested-learning methods.
2. Fixed hard-patch weighting (fixed γ, no slow level) **improves** Pixel AP by about +0.019 over the plain patch loss
   at the two new seeds. F is **not** a nested-learning improvement (card section 8).

## Open points

- The T4-generated BASE files of seeds 2712 and 2713 are kept with the run outputs, not in git. They were generated
  on CUDA and cannot be regenerated bit-identically on other hardware.
