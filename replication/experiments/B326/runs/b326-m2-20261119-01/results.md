# B326 job M2: int8 archive (b326-m2-20261119-01)

Exploratory (exposed MVTec test). Decision rules in the card, sections 2 and 5, fixed before the run. Label:
**`INT8_LOSSLESS`**. Full numbers: [metrics.json](metrics.json); environment, gates and hashes:
[execution.json](execution.json).

## Execution checks

- torch 2.11.0+cu130, 2 × T4.
- Pytest gate: 48 driver tests (including 3 torch tests: quantiser, `canonical_sha`, `base_from_seed`) and 30 of
  Khang's tests passed.
- Inputs: `locate` found the 3 F checkpoints and 3 BASEs from the B325 runs Q and R and Khang's reference BASE; all
  hashes matched the B325 records.
- Whole job 9,214 s (limit 12,600 s); no traceback in the log.
- **bf16 re-scoring reproduces B325 exactly:** the macro Pixel AP and Image AUROC of the bf16 variant, re-scored in
  this job, are bit-identical to job Q (seed 2711) and job R (seeds 2712, 2713). Scoring is deterministic for the same
  checkpoint, PyTorch build and GPU type, so the cross-run differences against Khang seen in B325 do not come from the
  scoring step.

## Results (F, readout A1, macro Pixel AP and Image AUROC, task 15)

| Seed | bf16 Pixel AP | int8 Pixel AP | Δ AP | Δ AUROC | Max \|Δ AP\| per category | Gate (int8 vs fp32, in fp32) | Same, in bf16 | bf16-vs-fp32 noise floor (whole pipeline) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2711 | 0.60242 | 0.60244 | +0.00002 | +0.00008 | 0.0007 | 0.00110 | 0.00145 | 0.0172 |
| 2712 | 0.60764 | 0.60757 | −0.00008 | −0.00000 | 0.0012 | 0.00115 | 0.00147 | 0.0170 |
| 2713 | 0.60551 | 0.60549 | −0.00002 | −0.00022 | 0.0006 | 0.00114 | 0.00147 | 0.0171 |

Label threshold |Δ| ≤ 0.002; gate threshold 0.01. The category-bootstrap interval of Δ AP lies within ±0.0003 at all
three seeds.

**Measured memory** (identical at all three seeds): 12,734,400 bytes of int8 plus fp16 scales, exactly the
precomputed number; bf16 is 25,436,160 bytes, so int8 is 0.501 × bf16. No rank component is zero and the scale guard
never fired; the smallest component amax is 0.031. The largest absolute dequantisation error is 0.0024–0.0027.

## Reading

1. **Storing the archive in int8 does not change measured detector quality.** Δ Pixel AP within ±0.0001 and
   Δ Image AUROC within ±0.0003 at all three seeds, about 10× below the `INT8_LOSSLESS` threshold. The quantisation
   error on the normal maps (0.0011) is about 15× smaller than the error bf16 itself already introduces against fp32
   (0.017).
2. **The CL state of F drops from 25.4 MB (bf16) to 12.7 MB (int8)**, measured. Against the literature at the same
   dtype (FP32 inferred, 50.9 MB for F) that is a 4× reduction. 12.7 MB is below UCAD (24.4 MB) and CADIC 10k
   (30.7 MB) but above ReplayCAD (3.0 MB, generator not counted), CADIC 2.5k (7.7 MB) and GCR (9.0 MB). The archive
   still grows linearly, 0.85 MB/task.
3. **Method-specific state:** column (1) + column (2) = 12.7 + 122.9 = **135.6 MB**; BASE is still the largest part.
4. **Scope.** Post-training quantisation of a trained checkpoint; not int8 training, and the learning method is
   unchanged. Scored on an in-RAM dequantised copy; reloading the int8 file and resuming training from it were not
   tested. F is not nested learning.
