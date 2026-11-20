# B326 M1a: regenerating the CPU BASE on a different machine type (b326-m1a-local-20261119)

19 Nov 2026, on a Windows laptop with a temporary Python environment (removed afterwards). This checks
`BASE_PORTABLE` (card section 2, M1a). CPU only: no MVTec data read, no training.

**Result: the canonical SHA-256 does NOT match the Linux one.** The M1a label stays **`BASE_SAME_MACHINE_ONLY`**,
now with positive evidence of a mismatch on a different machine type rather than an untested case.

## Environments

| | Linux (job M1) | Windows (this check) |
| --- | --- | --- |
| OS | Linux 6.18, glibc 2.39, x86_64 | Windows 11 (10.0.26300), AMD64 |
| CPU | vendor not recorded | Intel, Family 6 Model 154 |
| torch | 2.11.0+cu130 | 2.11.0+cpu (official CPU wheel) |
| Python | 3.13.16 | 3.13.18 (conda-forge, via pixi) |
| Generator code (`b326_memory.py` as executed) | `592aaea6…` | `592aaea6…` (equal) |
| Khang's code | 5217cb1 | 5217cb1 (equal) |

Environment spec: [pixi.toml](pixi.toml); wrapper: [win_cpu_base.py](win_cpu_base.py). SHA-256 of the lock file was
`617f9876…a146`.

**Shim.** Khang's `exps/nar_pilot_00/full.py` imports the Unix-only `resource` module at top level (used only for
RAM accounting). This run went through a wrapper that installs an inert `resource` stub before loading the driver;
the stub raises if anything calls it. Khang's code and the driver were unchanged (code hashes in the table). BASE
generation does not call `resource`.

The scripts in this directory are kept exactly as run. They refer to the layout of that time, where Khang's code sat
in a vendor directory next to the driver; to rerun them here, point their path arguments at the repository root and
`replication/scripts/`. The `BASE_SEED.json` files record `vendored_commit`, the field name used then for
`khang_code_commit`.

## Canonical SHA-256 (fp32 tensor bytes in key order)

| Seed | Linux, 2 processes (`DEFAULT`) | Windows, `ATEN_CPU_CAPABILITY=default` | Windows, native capability (`AVX2`) |
| --- | --- | --- | --- |
| 2712 | `50ccb680…8be` | `0892af95…07d` | `0892af95…07d` |
| 2713 | `fa995858…f63` | `29229dfb…67c` | not run |

## Reading

1. **Each machine reproduces itself, but the two machines give different BASEs.** On Windows, `default` and `AVX2`
   give the same SHA (and the same file bytes), so the difference does not come from the SIMD kernel choice:
   `ATEN_CPU_CAPABILITY=default` is not the missing piece. A possible cause is the torch build or the platform math
   library (MSVC versus glibc) in the `erfinv` / `log` / `sqrt` operations of `trunc_normal_`. This is a hypothesis,
   not tested.
2. **Consequence for the memory table:** the BASE column **stays at 122.9 MB** and the method-specific total stays
   **135.6 MB**. "Store only the seed" works only if the scorer uses exactly the platform (OS and torch build) that
   generated BASE, which does not replace storing the file.
3. **A portable seed-generated BASE** would need a different generation procedure, e.g. computing in integers or
   float64 and rounding to fp32, to remove the dependence on the platform math library. That is a method change and
   needs a new card, retraining and rescoring. The rule "identical SHA" is not relaxed to "close" after seeing the
   result.

## Size of the difference (seed 2712, measured the same day)

**Procedure.** The CPU BASE of seed 2712 from job M1 (`cpu_base/BASE.pt`, 245,725,241 bytes, file SHA-256
`9e5a727c…`, matching the job's `BASE_SEED.json`) was compared element by element with the same seed regenerated on
Windows (`ATEN_CPU_CAPABILITY=default`, same [pixi.toml](pixi.toml)) using
[measure_base_diff.py](measure_base_diff.py). Full numbers: [base_diff_2712.json](base_diff_2712.json). The
`bf16_cast` entry comes from the short probe [bf16_probe.py](bf16_probe.py), run in the same environment. The
environment, the downloaded tensor and the regenerated BASE were deleted afterwards.

| Quantity | Value |
| --- | ---: |
| Tensors that differ | 34/100: exactly the 34 Linear weight matrices (biases and LayerNorm are constants and match) |
| Elements that differ | 402,455 / 61,425,408 (0.66%) |
| Distribution (fp32 ULP) | 1 ULP: 374,348; 2 ULP: 28,107; nothing larger |
| Largest absolute difference | 3.7·10⁻⁹ (weights have std 0.01) |
| Largest relative difference | 1.3·10⁻⁷ |
| Elements still different after casting to bf16 | 8 / 61,425,408 |

**Reading.**
- **The pattern fits floating-point rounding, not two different random draws:** differences only in the last 1–2
  bits. The comparison does not identify which operation or library causes it (`erfinv`, SIMD, libm).
- **Validity of the ULP count.** ULPs are computed as the difference of the two `view(int32)` values, valid only
  when both numbers are finite and of the same sign. The script does not check this, but the data satisfy it: all
  values lie within ±0.03 (largest `scale_abs_max` among differing tensors is 0.03), hence finite; a sign-flipped pair
  (including +0/−0) would give an int32 difference near 2³¹, while the largest observed is 2.
- **Compared with a CUDA-generated BASE:** T4 versus RTX 4050 differed by up to 0.0596 (B325 job Q), about 10⁷ times
  more; those are two different draws (that comparison also mixes PyTorch 2.11 and 2.10).
- **After casting to bf16 the two are almost identical** (8 elements). Scoring runs in bf16, but the dtype BASE has
  during training was not checked, so "identical detector" cannot be concluded without rescoring.
- **The label does not change.** The pre-set rule was an identical SHA, and the SHA differs. A tolerance criterion
  (e.g. ≤ 2 ULP, or identical after bf16 cast) would have to be fixed in a new card **before** running and tested on
  a new seed or machine; it is not applied retroactively.
- Seed 2713 was not measured (only one ≈ 245 MB file was transferred for this check).

**PyTorch version note.** Both sides of this comparison are torch 2.11.0 (Linux `+cu130`, Windows `+cpu`), so the
1–2 ULP difference comes from the build or platform within one torch version. A check on Khang's machine with torch
2.10 would be a different BASE identity under the card's definition: a match would be strong evidence, a mismatch
could not separate torch version from platform. A clean check needs torch 2.11.0.
