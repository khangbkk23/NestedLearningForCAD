# MVTec AD 2 upload guide for the owner (Claude never uploads)

Two archives were built with the official MVTec AD 2 code utils:
- `check_and_prepare_data_for_upload.check_submission`, which reported "All checks successful";
- then `compress_submission`.

Each archive was read in full afterwards (gzip CRC). Its members (8,180 files) match the checked staging copy one for one, with the same names and sizes.

| Upload order | File | Submission (method name suggestion) | Bytes | sha256 |
|---|---|---|---|---|
| **1 (first)** | `results/hope_cad/nl03_multirate_01/ad2/UPLOAD/E.tar.gz` | **G3-E (proposed), 8 MiB fixed budget**: arm E | 9,828,331,531 | `1294efacd30408417cef2c2c63409f5a2d349d223b38b9f9ed39a96367df0626` |
| 2 | `results/hope_cad/nl03_multirate_01/ad2/UPLOAD/U.tar.gz` | **G3 unbounded (reference)**: arm U | 10,530,362,832 | `e8ba60f49d89c845998af8bec6ee155db16585300cb5b51b05831121f209f5ef` |

**One archive = one submission.** Do not merge them. Each archive's root folder is named after its arm (`E/` or `U/`) and contains:
- `anomaly_images/<scenario>/{test_private,test_private_mixed}/*.tiff`: float16, final-stage continuous maps;
- `anomaly_images_thresholded/…/*.png`: binary 0/255.

**What the maps are.**
- **Stream.** The 8 scenarios form one continual stream in alphabetical order (can → walnuts), seed 2711. Fitting uses train/ only. The method and hyperparameters were locked at FINAL_LOCK (918999763101ef12) and are unchanged.
- **Threshold (deviation D-THR, declared before any SegF1 or server result).** θ_s = mean + 3·std of the per-pixel scores over all validation/ images of scenario s. The scores are taken with the arm's state at that scenario's acquisition stage. The binary map is the final-stage float32 map > θ_s. The θ_s values are in `AD2_SCORE.json → theta_dthr`.
- **Old thresholded maps.** These used θ = max validation image score. They were moved, not deleted, to `AD2_SUBMISSION_OLDTHR/<arm>/`.
- **Not part of this upload:** P, CB and F (folders under `AD2_SUBMISSION/`).

**Before uploading:**
1. Check the hash: `sha256sum E.tar.gz U.tar.gz`. It must match the table above.
2. The archives are about 10 GB each. Make sure your connection and the server's limit allow that.

**After the server returns scores:** fill in the server table in `agents/reports/nl03_multirate_01.md` §10.5 (rows E and U, splits test_private and test_private_mixed).

Local test_public results (HELD_OUT_CONFIRMATION, not server numbers) are in `AD2_TEST_PUBLIC.json` and `AD2_AUPRO.json`.
