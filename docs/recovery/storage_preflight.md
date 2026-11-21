# NANM-RECOVERY-01 — Storage preflight and recovery record

Date: 2026-11-21 (21:14–21:24 +07:00). Agent: Claude Code under AGENTS.md. Task: `agents/tasks/task_nanm_recovery_01.md`.

## Verdict: **STORAGE_SAFE**

The verdict rests on Windows-side measurements, not on `df /` inside WSL.
- D: has 294.79 GiB physically free, against the 30 GiB reserve.
- Measured growth on D: over the whole session was **0 bytes**: D: free bytes and the VHDX length are byte-identical before and after.
- All new caches and temp files resolve to K:.
- New source files live on E:, beside the existing working tree.

## 1. Environment (Phase 0)

| Item | Measured value | Method |
|---|---|---|
| Execution environment | WSL2 (kernel 6.6.114.1-microsoft-standard-WSL2), distro `Ubuntu` (default, Running, v2); `docker-desktop` Stopped | `uname`, `wsl.exe -l -v` |
| User lookup | `getpwuid(1000)` → `dikhang_hcmut:/home/dikhang_hcmut` (the historical getpwuid failure does not reproduce now) | `getent passwd 1000`, `id` |
| Repository root | `/mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD` (E:, DrvFS 9p) | `pwd`, `mount` |
| Git | branch `dikhang_space`, HEAD `e176d7e` (= `origin/dikhang_space`); working tree clean at start | `git status`, `git ls-remote` |
| WSL VHDX (Ubuntu) | `D:\wsl\Ubuntu_new\ext4.vhdx`, **178,122,653,696 B (165.89 GiB)**, last write 2026-11-21 21:10 | registry `Lxss\BasePath` + `Get-Item` |
| Docker VHDX | `C:\Users\Lenovo\AppData\Local\Docker\wsl\main\ext4.vhdx`, 0.10 GiB | same |
| Physical disk | D:, E:, K: are **partitions of one physical disk** (disk #1, KINGSTON SNV2S2000G NVMe, 1,863 GiB) | `Get-Partition` / `Get-Disk` |
| WSL root as seen inside | `/dev/sdd` ext4 1007G, 162G used, "795G avail". That figure is **virtual capacity, not D: free space** | `df` |

### Physical free capacity (Windows `Get-PSDrive`, bytes)

| Drive | Before (21:17:37) | After (21:23:35) | Δ |
|---|---|---|---|
| D: (VHDX host) | 316,524,720,128 (294.79 GiB) | 316,524,720,128 | **0** |
| K: | 268,433,248,256 (250.00 GiB) | 268,433,244,160 | −4,096 (directory entries) |
| E: (repo) | 211,826,024,448 (197.28 GiB) | 211,825,971,200 | −53,248 (new sources + reports) |
| VHDX length | 178,122,653,696 | 178,122,653,696 | **0** |
| WSL `/` used (inside) | 172,940,947,456 | 172,941,471,744 | +524,288 (scratchpad test files, since deleted; absorbed by already-allocated VHDX blocks) |

D: reserve check: 294.79 GiB free − 30 GiB reserve = 264.79 GiB headroom. The reserve threshold was not changed.

### Discrepancies with the task's historical facts (not resolved; reported)

| Historical fact (task) | Observed now |
|---|---|
| `D:\wsl` ≈ 460.7 GB | `D:\wsl\Ubuntu_new` = 166 GiB; `D:\wsl\Ubuntu` empty |
| Ubuntu ext4.vhdx ≈ 141.54 GB | 165.89 GiB (178.12 GB). Today's `agents/reports/wsl_cache_audit_2026_11_21.md` (14:30) recorded it as 320 GB. So it was **compacted between 14:38 and 21:10 today**. No compaction command appears in the agent transcripts checked (Claude Code session 87983952…, Codex sessions of 11-20/11-21), so the cause is unrecorded (presumably manual). |
| Backup made on K:, SHA-256 matched | **Not found.** K: holds 6.1 GB in total. No `*.vhdx`, `*.tar`, `*.tar.gz` or `*.sha256` within depth 3. K: root contains what look like Ubuntu installer files (`casper/`, `pool/`, `dists/`, `EFI/`, `boot.catalog`, `md5sum.txt`). `D:\backup` is empty. Also note that K: is on the **same physical NVMe** as D:, so even a K: copy would not protect against that drive failing. |
| — | **`~/hope_cad_cache` no longer exists** (79 GB after this morning's step 1–3 cleanup). It is not on D:, E:, K: or `C:\Users\Lenovo` within depth 4. The audit's §G "do not delete" list (G3 SHA-locked feature roots, `bench_g3/normal`, `bench_g3/scores` including the **MPDD sealed scores recorded as existing nowhere else**) is therefore gone from this laptop. There is no record of who removed it. Consequence: none of the SHA-locked G3/BENCH-G3 loaders can run locally, and the MPDD confirmation cannot be re-audited from here. |

## 2. Write locations (Phases 1–2)

| Component | Location | Physical drive | Action |
|---|---|---|---|
| Repository and new sources | E: (existing working tree) | E: | small source files only |
| Caches (`XDG_CACHE_HOME`, `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`, `CONDA_PKGS_DIRS`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`) | `/mnt/k/NLCAD/cache/*` | K: | created (empty) |
| Temp (`TMPDIR`, `TMP`, `TEMP`) | `/mnt/k/NLCAD/tmp` | K: | created (empty) |
| Python env | `~/miniconda3/envs/nested_learning` (existing; Python 3.11.17, torch 2.10.0+cu126, numpy 2.4.6) | D: (VHDX) | **used read-only; nothing installed** |
| `.pyc` | disabled (`PYTHONDONTWRITEBYTECODE=1`, `python -B`) | — | — |
| ext4 checkpoint tests | session scratchpad `/tmp/claude-1000/...` (root ext4) | D: (VHDX) | ~KB, removed afterwards |

Usage: `source docs/recovery/recovery_env.sh`. It refuses to run unless `/mnt/k` is the K:\ DrvFS mount. It also sets `CUDA_VISIBLE_DEVICES=""` because recovery is CPU-only.

**Deviation from the preferred root, with reason.** The project stays on E: rather than being copied to `K:\NLCAD\NestedLearningForCAD`. The policy's goal (keep project writes off D:) is already met on E:. A second copy would create a duplicate, divergent working tree, which AGENTS.md §2 forbids. `results/` (260 GB) would not fit on K: (250 GiB). K: is used for caches and temp, as specified.

### Existing D:-resident (VHDX) consumers, audited, not modified

`~/miniconda3` 86 GiB (13 envs), `~/.local` 22 GiB, `~/.cache` 16 GiB (uv 9.4, huggingface 9.8, overlapping via hardlinks with conda pkgs), `~/.ollama` 12 GiB, `~/.vscode-server` 6 GiB, `~/.npm` 4.6 GiB, `~/tensorflow_datasets` 2.7 GiB, `/tmp` 1.6 GiB.

**Note:** the Claude Code session scratchpad (`/tmp/claude-1000/...`) is also on the VHDX. Any future large scratch output must go to `/mnt/k/NLCAD/tmp` instead.

### Existing data copies (not duplicated)

- Datasets in repo `data/` (E:, 9.6 GB): `dtd`, `mpdd`, `mvtec` (+ `.tar.xz`), `visa`.
- Encoders in `checkpoints/` (E:, 658 MB): `dinov2`, `cadic`.
- `results/` (E:, 260 GB).
- `export/2026-11-20/HOPE-CAD-SHIP.zip` (E:, 10.28 GB, sha256 `eb95d619…c48777`).

No dataset, checkpoint, feature tensor or score grid was copied, downloaded or regenerated.

### Blocked or avoided operations

None of the forbidden operations was performed: no unregister, VHDX delete, compact or resize, distro move, Docker/WSL volume deletion, reset/clean, recursive deletion, clone, dataset download, DYN rerun, GPU use or installation.

### DrvFS caveat found during testing

With `TMPDIR` on DrvFS, pytest's default fd capture crashes with `FileNotFoundError` in `capture.py snap → tmpfile.truncate()`. DrvFS (9p) does not support the open-then-unlink temp-file pattern. Run tests with `--capture=sys` (or `-s`).

## 3. Files created (all small text; nothing else written)

- `agents/tasks/task_nanm_recovery_01.md` (owner task, verbatim)
- `docs/recovery/recovery_env.sh`
- `nanm/__init__.py`, `nanm/g3.py`, `nanm/checkpoint.py`, `nanm/test_g3.py`, `nanm/test_checkpoint.py`
- `docs/recovery/storage_preflight.md` (this file)
- `agents/reports/nanm_dyn_01_archive_2026_11_21.md`, `agents/reports/nanm_recovery_01_2026_11_21.md`
- Directories `/mnt/k/NLCAD/{cache/{xdg,hf/hub,torch,pip,uv,conda_pkgs,triton,cuda},tmp}`: empty (49,152 B of directory entries)

Full hashes at 21:23:

```
a2ad8379dd0abd44c12ca94061fe50eec6ce5160771bc5aa32d228214f889858  nanm/__init__.py
d892b0e09caebb51ce92dd3427c4a77570cc6d9d5e64d33daf19066093cfc501  nanm/checkpoint.py
093e3333f153373dd00b70d4113d95cc5e1b3964c1f6cefeb0d5dbf8c540e3f8  nanm/g3.py
8caf76a930496bdcc94a72d6b49b68ffd211b181dec22ca7a3a64c6f7ef04828  nanm/test_checkpoint.py
0f21b671017152074c4e4792252d34c6dc8155b4d47a6b53806bc3000b81ee68  nanm/test_g3.py
7e95352ccea007b3d2e894948a32944a36efd1c64404ad135e202175ac7dd3db  docs/recovery/recovery_env.sh
ee455a462569b08016b2ca564273574e708b6bad2ba5c34d960732a3a2ee17a5  agents/tasks/task_nanm_recovery_01.md
```

### Oracle source hashes (unchanged; each byte-identical to its `HOPE-CAD-SHIP.manifest.json` entry, i.e. to the code shipped to the server on 2026-11-20)

```
65796a6026c4ad5ce61153cfc237ee2c78f2104d056df6475186264ebe7774c2  exps/nanm_r0/audit.py
bdc996c7010c9f8baf5e0e30cf3e94e5ddb6caf3f27e263316d251c8b2e003cf  exps/nanm_r1/run.py
f0abb92bc27f40e103397382a34e2a25e0341f6b6ae0e90b17ad283b84890376  exps/nanm_r2/run.py
830e945f2effebe0b098b1a12eb16622918c4a84f398580ea449c77347b420f6  exps/nanm_r2b/run.py
2792d40be921968b080eb481cf3dde37d3ad0b12ce60652596cba8eea157fa94  exps/bench_g3/run.py
```

## 4. Tests executed (CPU, synthetic only; `source docs/recovery/recovery_env.sh` first)

1. **Surviving oracle suite.** Command:
   `~/miniconda3/envs/nested_learning/bin/python -B -m pytest -p no:cacheprovider --capture=sys -q exps/nanm_r0 exps/nanm_r1 exps/nanm_r2 exps/nanm_r2b exps/nanm_write_01 exps/bench_g3`
   Result: **42 passed in 9.33 s**. The first attempt without `--capture=sys` aborted ("no tests ran") because of the DrvFS caveat above.
2. **nanm suite.** Command:
   `NANM_EXT4_TESTDIR=<scratchpad>/ckpt_ext4 NANM_DRVFS_TESTDIR=/mnt/k/NLCAD/tmp/ckpt_drvfs <python> -B -m pytest -p no:cacheprovider --capture=sys -v nanm`
   Result: **25 passed in 4.07 s**. Roots: `stat -f` reports `ext2/ext3` (WSL root ext4) and `v9fs` (K: DrvFS).
3. **Negative controls and filesystem probe** (ad-hoc script, same environment):
   - Bit-parity breaks under W+1e-6 and under dropping one key (`torch.equal` False).
   - The fp64 whitening identity fails under shrink α = 0.4 instead of 0.5.
   - A concurrent reader during 25 publishes observed **all 26 versions, only complete payloads**: 20,311 reads on ext4 and 111 on DrvFS.
   - `dir_fsynced` was True on both filesystems. On DrvFS this means only that the call was accepted, not that it was honoured.

## 5. Phase 4 — checkpoint safety, by property

| Property | ext4 (WSL root) | DrvFS (K:, NTFS over 9p) |
|---|---|---|
| Serialization correctness | PASS | PASS |
| Round-trip (overwrite, sidecar digest = file digest, no temp leftovers) | PASS | PASS |
| Checksum validation (bit flip, malformed sidecar, missing sidecar) | PASS | PASS |
| Publication atomicity: failure injected before rename leaves previous version intact | PASS | PASS |
| Publication atomicity: concurrent reader never sees partial or missing payload | PASS (observational, 20,311 reads). Consistent with POSIX `rename(2)` on ext4 | **FAIL** — see correction below |

**Correction (2026-11-21 ~22:00).** The DrvFS row was first recorded as "observational PASS". That was wrong: the ad-hoc probe silently skipped `FileNotFoundError`. A rerun of `test_concurrent_reader_never_sees_partial_payload[drvfs]` during NANM-REPRO-01 Gate 1 failed. The reader got `FileNotFoundError` 3 times: on DrvFS, `os.replace` lets the target path briefly vanish. No partial payload was ever read, but the target is not continuously present. **Do not rely on `nanm.checkpoint` atomic publication on DrvFS.**
| Crash/power-loss durability (fsync file + dir) | UNVERIFIED (not testable here) | UNVERIFIED |
| **Parity with the lost server publisher ("P5")** | **UNVERIFIED.** The original source is lost; `nanm/checkpoint.py` is new code | **UNVERIFIED** |

**No P5 recovery success is claimed.**
