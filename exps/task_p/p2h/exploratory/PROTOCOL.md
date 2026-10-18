# P2h-X — runtime-waiver exploratory protocol

The user explicitly authorized this separately named phase after the original
P2h G2 failed17.054x E. Original verdict remains RESOURCE_INFEASIBLE. Only10x
runtime is waived; no mechanism, hyperparameter, generator or endpoint search.
Normal-only synthetic scope,4.5GiB reserved CUDA cap and6GPU-hour total extension
budget remain. This is exploratory because engineering outcomes were inspected.

Inherit original corrected PROTOCOL.md and its Design Lock, including positive
reference-minus-candidate signs, D16L14/W14, actual/zero diagnostics, no anomaly
claim, all memory equations and seeds. Root implementation files stay unchanged.
All extensions live in this package and separate output subdirectory.

## Deterministic implementation completion before execution lock

Candidate math equals original exactly. Control corrections live on the control's
own causal state. Immediate protection computes mean old-anchor query gradient
at base E state; r=-relu(g dot b)g/(||g||²+1e-12), perblock norm capped.25||b||.
Extra replay performs five additional exact E cycles with the same arrived
current/replay images and parity, starting from base state; extra moments are
discarded, base moments committed. Guard on resulting displacement. This matches
12 ordinary gradient calls/cycle, not higher-order/anchor/time cost. No new draws.

Single16 has one n-by16 map,15-bucket+bias key; at four-cycle boundaries short
writes, plus at16 a long gradient of a shared additive perturbation applied to
all four retained versions. Calculate terminal short/long derivatives before
adding gradients and writing; clip/cap sqrt2. M4-only16 has short writes only;
M16-only16 has long writes only. Dummy unused derivative work is executed and
reported. Effective persistent capacity is16 columns in each arm, plus transient
versions. Candidate is two8-column maps. Collapsed-clock updates both8-column
maps every4 from the short objective; full long graph/gradient computed/discarded.
Derivative of the joint short objective is taken with respect to both maps;
each family retains clip/cap1. This is not an alias of candidate.

Short-credit detaches fast state/moments between cycles in each shadow, including
resume reconstruction. Reset zeroes both maps at window entry. Post-onset-only
detaches only direct M16 reads before onset in the onset-containing16-cycle
window. Its harness-only mask never reaches candidate, leaves forward values
unchanged, and is empty for D32 (onset at window start). Long updates after that
window unmodified. It tests within-window credit, not every persistent effect.

Random arm trains the same learned shadow on its own live-window entry, replacing
only live proposed directions with independent standard-normal block directions,
scaled to the exact realized learned norms. Logged learned/random norms must
agree to floating tolerance. Disabled guard changes only live acceptance.
Deployed ablation is a separate zero-correction causal rollout using candidate's
version history as it arrives; maps are not retrained and no future version is
read. It therefore equals E numerically, an explicit redundancy/invariant, not
an independent memory-retention test. Report this limitation.

## Matrix, endpoints and inference

Positive D8,D16,R: all15 named original arms including E and deployed ablation;
diagnostic C,D32: E,zero_memory,candidate,reset_memory,norm_matched_random,
immediate_protection,post_onset_only. Ten seeds3101..3110, one complete matrix.
R continues to160 cycles and additionally reports A recurrence loss at160;
primary old endpoint128 and B acquisition128 unchanged. Other regimes128cycles.
All witnesses/endpoint states archived for independent recomputation; no learner
access. All runs archive trace, causal replay, writes, versions, memory and costs.

C1 D8: benefit_E>=.05 mean E excess, paired95% lower>0 and mean per-seed B
acquisition regression<=.01. Also report paired95% acquisition CI; no equivalence
claim based on nonsignificance. A conservative exploratory mechanism verdict
requires its upper<=.01 as well. This additional conservative reporting rule
cannot rescue an original failure.

C2 D8 compares candidate to immediate_protection,extra_replay,short_credit,
reset_memory,collapsed_clock,single_memory_16,M4_only_16,M16_only_16,zero_memory.
Each benefit=control excess-candidate excess, must>=.5 C1 benefit, positive
paired95% lower and Holm-adjusted p<.05. Include all, never select favorable ones.
C2 pre-onset uses paired per-seed average of D16/R contrasts: post_onset_only
minus candidate, >=.5 mean D16/R benefit_E, positive95% lower, same Holm family.
D8 candidate/post-onset equality expected at endpoint80 before long write deploys.

C3 random-minus-candidate and deployed-minus-candidate benefit>=.5 C1 benefit,
positive95% lower and Holm p<.05 in the same fixed secondary family. Guard
contrast disabled-minus-candidate reported separately, no guard-necessity gate.
C4 D32 post_onset_only equality must be bitwise for states at all milestones;
reset-minus-candidate measures persistence, not pure pre-onset credit. Report
all D16/R control contrasts and diagnostic C/D32 intervals descriptively without
claiming secondary confirmatory discoveries. Paired two-sided Student-t95% df9,
Holm for12 fixed mechanism contrasts above. Constant zero differences p1;
nonzero constant differences p0; positive reference-minus-candidate always.

Primary verdict MECHANISM_SUPPORTED only if all specified effect/attribution/
acquisition/invariant conditions pass. Otherwise MECHANISM_NOT_SUPPORTED if
implementation/power/resources valid. No negative seed/control removal, no
non-significance-as-equivalence, no post-outcome tuning. Any earlier invalidity,
insufficient engineering power or resource breach stops with the applicable
IMPLEMENTATION_INVALID/INCONCLUSIVE_UNDERPOWERED/RESOURCE_INFEASIBLE verdict.

## Locks, resources and lifecycle

Create extension Design Lock before its new engineering runs. Retain G1/G1b
original evidence; require new control/sign/causal/resume tests. Three full16-cycle
released windows on engineering1901 test memory growth (slope <=1MiB/window),
peak reserved<=4.5GiB and600s profiling limit, no speed gate. Explain retained
tensor inventory; no growth claim from a single window. Engineering dry-run
prepare/train/witness_evaluate/analyze/verify across every arm before Execution
Lock. No G3 seed reaches a PRNG before validated Execution Lock. Exact sources,
tolerances, matrix, seed streams, CI implementation, resource projection and
new protocol hash frozen there. No matrix reduction.

One-shot matrix records started ledger before first sample. Check consumed
GPU elapsed budget before/after every run, stop if projected >6hours or actual
>6hours including extension engineering/profile/dryrun. No overwrite/retry of a
failed evaluation run; preserve failure and stop. Use current RTX4050 environment,
deterministic algorithms, TF32 off, existing float32/FP64 test dtypes.
