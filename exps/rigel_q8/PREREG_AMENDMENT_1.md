# RIGEL-Q8-01, amendment 1 to S2 (owner request 2026-11-26, after the S2 DEV outcome)

**Disclosure.**
- The registered S2 DEV selection (PREREG.md) chose σ\* = 8. That is the largest registered value, at the edge of the candidate grid: Δ pixel AP +0.0193, CI95 [0.0144, 0.0241].
- After seeing this, the owner asked to extend σ step by step up to 24 and to pick the best value ("thử tăng lên lần lượt đến 24 đi và chọn cái tốt nhất").
- This is therefore a post-outcome extension of the DEV candidate set, made on the owner's explicit instruction. It is reported as such.
- The registered FULL confirmation for σ = 8 had not started. It was cancelled before running, so no FULL outcome has been seen.

**Changed: S2 candidate σ only.** σ ∈ {2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}, with kernel size 2·int(4σ + 0.5) + 1.

**Unchanged:**
- the image-score candidates;
- the DEV data;
- the metrics and bootstrap;
- the selection rule:
  - step a: argmax DEV macro pixel AP, adopted if Δ vs σ 4 ≥ +0.003 and the CI95 lower bound > 0;
  - step b: argmax image AUROC at σ\*, adopted if Δ vs top 1 % ≥ +0.005 and the CI95 lower bound > 0;
- the one-shot FULL confirmation rule and its datasets.

**Outputs:**
- `S2_DEV_A1.json` (the original `S2_DEV.json` is kept);
- FULL uses the A1 selection: `S2_FULL.json`.

**Cost note (measured 2026-11-26, per image, read-out smoothing only):**
- GPU: 0.28 / 0.43 / 0.60 / 0.76 / 1.05 ms for σ 4 / 8 / 12 / 16 / 24.
- CPU: grows about linearly with the kernel. σ 24 is about 3.7× σ 4.
- σ adds no persistent state and no training cost.
