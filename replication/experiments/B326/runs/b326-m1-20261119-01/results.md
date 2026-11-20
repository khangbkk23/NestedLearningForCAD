# B326 job M1: BASE from a seeded CPU generator (b326-m1-20261119-01)

Exploratory (exposed MVTec test). Decision rules in the card, sections 2 and 5, fixed before the run.

**Labels:** M1a **`BASE_SAME_MACHINE_ONLY`**; M1b **`BASE_FROM_SEED_OK`**. Full numbers:
[metrics.json](metrics.json); environment, gates and hashes: [execution.json](execution.json).

## Execution checks

- torch 2.11.0+cu130, 2 × T4, Linux x86_64. Pytest gate: 48 driver tests and 30 of Khang's tests passed.
- Inputs: `locate` found the checkpoints and T4 BASEs of B325 job R; hashes matched the B325 records.
- Whole job 8,305 s (limit 12,600 s); 682–719 s of update time per arm (limit 2,400 s); no traceback in the log.
- Fidelity passed for all four arms (relative error 0.0170–0.0172), the same level as M2 and B325.

### Retraining F on the T4 reproduces job R's checkpoint bit for bit

The "T4 BASE" arm reuses the T4 BASE saved by job R, retrains F from scratch in this job and re-scores it.
`stage15.pt` has the same SHA-256 as in job R at both seeds (2712 `e527b449…`, 2713 `d8bedce7…`), and Pixel AP and
Image AUROC are bit-identical to job R: the "rerun minus job R" difference is 0.0 at both seeds.

On these two repeated T4 controls the observed difference is zero for the compared quantities (final checkpoint
SHA, macro Pixel AP and Image AUROC). This supports the within-job design of M1b: with BASE fixed, repeating training
and scoring does not change these quantities. It is two repetitions on one PyTorch build and GPU type, **not an
estimate of execution variance**, and does not imply that run-to-run noise is always zero.

## M1a: reproducibility of the CPU BASE

| Seed | Canonical SHA-256, process 1 | Canonical SHA-256, process 2 | Equal |
| --- | --- | --- | --- |
| 2712 | `50ccb680…8be` | `50ccb680…8be` | yes |
| 2713 | `fa995858…f63` | `fa995858…f63` | yes |

- SHA over fp32 tensor bytes in key order (card). The `torch.save` file bytes are also equal.
- Conditions: `ATEN_CPU_CAPABILITY=default` (torch reports capability `DEFAULT`), GPU hidden, CPU `torch.Generator`
  (mt19937).
- Both processes ran on **the same machine**, hence only `BASE_SAME_MACHINE_ONLY`. `BASE_PORTABLE` needs a second
  machine type (different CPU vendor/ISA or OS). That check was done the same day on Windows/Intel and the SHA-256
  differed, so the label stays: see [b326-m1a-local-20261119](../b326-m1a-local-20261119/results.md).

## M1b: quality (F, γ = 2.1, readout A1, task 15, Δ = F(CPU BASE) − F(T4 BASE), same job)

| Seed | Pixel AP, T4 BASE | Pixel AP, CPU BASE | Δ AP | 95% interval of Δ (category bootstrap) | Δ AUROC |
| --- | ---: | ---: | ---: | --- | ---: |
| 2712 | 0.60764 | 0.60496 | −0.0027 | [−0.0086, +0.0031] | −0.0003 |
| 2713 | 0.60551 | 0.60381 | −0.0017 | [−0.0056, +0.0030] | +0.0003 |
| Mean | | | **−0.0022** | | |

`BASE_FROM_SEED_OK` rule: |mean Δ| < 0.005 and |Δ| < 0.010 at each seed; both hold. The most cautious reading: both
seeds decrease slightly but stay inside the pre-set band; both intervals contain 0; between-seed uncertainty is not
estimated (n = 2).

## Reading

1. **Switching BASE's random source to a CPU generator causes no detectable change at n = 2.** Both Δ are negative,
   about −0.002, and each interval contains 0. For scale, F's Pixel AP over seeds 2711–2713 spans 0.005
   (0.6024–0.6076), so Δ is within the seed-to-seed range. Two seeds of the same sign cannot decide whether a small
   loss of about 0.002 exists; more seeds would be needed.
2. **The CPU BASE is reproducible on one machine but not proven across machines.** The BASE column of the memory table
   **stays at 122.9 MB** (≈ 0 MB would need both `BASE_PORTABLE` and `BASE_FROM_SEED_OK`), and the method-specific
   total stays **135.6 MB** (int8 column 1 + column 2), as after M2.
3. The combination "int8 archive + checkpoint trained on the CPU BASE" was never scored; with the cross-machine check
   failing, that projection does not apply.
4. **Scope:** a change of BASE initialisation (same distribution, different random source), not of the learning
   method. F is not nested learning.
