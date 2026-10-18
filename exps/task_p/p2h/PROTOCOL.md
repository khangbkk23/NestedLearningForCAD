# P2h pre-execution corrigenda and implementation contract

Primary specification: immutable `task_p_p2g_revision_final_v3.md`, with the
current user's explicit corrections. This is a synthetic normal-regression
study, not an anomaly experiment or a novelty claim. No G3 seed has been used.

## Scientific corrections before Design Lock

1. Every benefit is `reference_excess - candidate_excess`. Positive means less
   forgetting. C1 requires mean benefit >= .05 mean E excess, paired 95% lower
   >0 and mean new-normal acquisition regression <=1%. Known-value tests must
   give positive benefit when candidate excess is smaller.
2. User approved **D16 L=14,W=14**. The old L=15 splits a two-cycle image; no
   phase can change halfway through the same image. D16 now has two post-onset
   cycles in the onset window, and two pre-onset cycles in its four-cycle chunk.
3. Actual M4/M16 at active retention are persistent learned versions, not assumed
   zero. Report actual-state decomposition at the first active milestone for
   each derivative contract, and a separately labeled zero-memory replay of the
   same window. Apply the ratio gate to actual active headroom only. Ratios <.05
   stop as underpowered; [.05,.10) are weak diagnostics, not automatic failures.
4. Random-direction control computes its own causal learned proposal, then replaces
   each block's direction with an independent seeded standard-normal direction
   of **exactly that learned proposal's norm**, not the maximum allowed norm.
   Its memory learns on its own causal trajectory; both norms are logged.
5. Phase B has a compatible A-distribution prefix, then B distribution from onset
   until its end. P=8 is the initial conflict probe segment, not the total B
   persistence: total conflict duration is 64-L, explicitly reported. This fills
   v3's unspecified tail without returning to A before the recurrence endpoint.
6. Distinguish gradient-credit availability from deployment latency. D8's endpoint
   at cycle80 is reached before the newly computed M16 write at80 can affect a
   subsequent fast update. The pure onset-window pre-credit ablation must be
   identical there. Test its causal effect on D16/R (after the write), and report
   this D8 invariant rather than falsely attributing an earlier effect to a later
   write. No retrospective reapplication of learned memory to scored states.
7. Reset-memory measures cross-window persistence, not pure pre-onset credit.
   D32 raw benefit can be persistence, post-onset reaction or protection. Never
   call it leakage from that observation alone. A disabled-guard gain does not
   invalidate optimizer memory; the guard contrast is a separate causal question.

## Fixed state, equations and derivatives

Use the pinned P2e E numerical operator, P2 Predictor, loss and clips: two Adam
microsteps, rate .001, betas .9/.999, global gradient clip1, tensor RMS step
cap .02, sqrt variance epsilon1e-12 and denominator epsilon1e-8; global inner
clock 2k-1/2k. Preserve raw forward operation order. No slow pull. A stripped
operator must match reference states/moments/losses bitwise for sixteen cycles.

Each six-block map is n_b by8. Sketch seed is SHA256('p2h/sketch/'+block) first
eight bytes little-endian. h=(seed+(i+1)*2654435761) modulo2^64; bucket h mod7;
sign bit32, + if0 else-. Each bucket sum is divided by sqrt(n_b). Use explicit
bucket reductions to avoid nondeterministic CUDA atomic scatter. Key is
[1;sketch(pre-rate second Adam direction)]/sqrt(1+sum(sketch^2)).
u=(M4+M16)key. r=.25*norm(base_block)*u/sqrt(1+sum(u^2)). Zero base gives zero
correction and derivative. Commit reference Adam moments, not fictitious moments
for r. Memory writes are SGD .01, joint Frobenius gradient clip1 and norm cap1
across all blocks **separately per memory family**, producing detached new clones.

Pass A uses detached entries for each four-cycle chunk, M4 leaf, M16 constant;
retain all pre-write M4 versions. Pass B reconstructs sixteen cycles from window
entry, M16 leaf, four retained M4 versions constant; differentiate (J4+J8+J16)/3.
At cycle16 calculate both derivatives before committing either write. All state,
key, gradient and Adam dependencies inside a pass remain connected. No derivative
through the meta-SGD rule. Forward milestone states must be bitwise identical.

J=A+mean(e)+max(e)+relu(A-1)^2. A averages last four arrived current-image query
losses divided by detached max(E-shadow loss,1e-6)+1e-8. e_i is positive
(trial anchor loss-min(raw window-entry loss,E-shadow loss))/(max(raw entry,1e-6)+1e-8).
Thresholds/reference values are detached; exclude raw entry<=1e-8; hinge derivative
at0=0; max tie first. Training anchors are not evaluation witnesses.

Guard checks unique current/replay/anchor identities on their frozen query
partitions. Current/replay use current query; stored anchor query takes precedence
only for anchor-only identities. Largest feasible lambda in [1,.5,.25,0], loss <=
reference +eps_abs+eps_rel*abs(reference). Zero returns reference tensors exactly.
Nonfinite values stop. Never use a straight-through guard. Select up to four
old reservoir anchors at window entry by arrival-age quartiles, RNG seed+11000.

## Synthetic generator and access contract

d=768, rank8, side28, sigma=.1; reduced Gaussian QR with positive R-diagonal
sign convention; muA Gaussian std1/sqrt(d). Draw muB then project off muA; draw
VB then project off VA and QR. Draw fixed a,b iid integers1..4 and phi uniform
[0,2pi) for each factor. u,v linspace(-1,1); waves cos(pi*(a*u+b*v)+phi).
Every fresh image is mu + sum(V_r*z_r*wave_r)+.1*normal noise; z iid N(0,1).
Compatible prefix uses exactly muA,VA,waves, with fresh image randomness.

Windows start at global zero-based multiples of16; A=[0,64), B=[64,128).
C,D8: (L,P,M,W)=(8,8,8,8); D16=(14,8,8,14); D32=(32,8,8,0).
Only R adds return A=[128,160), with (8,8,56,8) and pre-return endpoint128.
C stays A-distributed throughout. Other B streams stay A before64+L and
B-distributed thereafter, including every cycle after64+L+P. Endpoints count
completed cycles: C/D8=80,D16=86,D32=104,R=128. Two cycles use the same image;
new images arrive on even cycles. Generate no future image in learner APIs.

Separate RNGs: parameters seed+1000, images +2000, replay +9000, anchors +11000,
random directions +13000, evaluator witnesses +30000. Algorithm-R capacity24;
sample before insert, insert after both cycles; no self/future replay. Masks
use parity=(zero-based event+local cycle)%2. No phase reset or learner phase IDs.
Evaluator generates sixteen independent normal witnesses at actual phase arrival
(A0, B onset; C's B-compatible witness at64). Keep witnesses entirely outside
learner state/API; A acquisition loss measured after64, B acquisition after128.
Excess=max(old witness loss-own A acquisition loss,0), reported also unrectified.

## Complete arm contracts

All proposing arms get the same anchor/data draws, guard and calibrated tolerances.
Candidate: n*8 M4 and n*8 M16. E: exact no-pull. Zero-memory: all bookkeeping
and backward work, discard writes and corrections. Immediate protection: project
base displacement against mean old-anchor gradient, r=-relu(dot(g_old,b))*g_old/
(norm(g_old)^2+1e-12), block-cap to .25 norm(base), same line search. Extra replay:
four extra E cycles/eight extra gradients per cycle; r=phi_extra-phi_bar; guard;
commit base moments. Report gradient/HVP/access/time asymmetries explicitly.

Short-credit: detach shadow state between every cycle, retain clocks/all forwards.
Reset: zero memories at each window boundary, keep Adam state. Collapsed-clock:
both memories update every4 from short objective; perform/discard long write,
report useful versus dummy work. Disabled guard: same learning, lambda1 live.
Norm-matched random: replace own learned r directions as specified above.

Capacity-matched single map: n*16, key bias plus15 sketch buckets, same hash
rule with mod15. One map receives four short writes plus a long common-perturbation
gradient across retained versions at16. Compute both terminal gradients first;
sum them for final write. Clip/cap sqrt(2), matching combined two-map norm budget.
M4-only: n*16/key16, short writes only, long work discarded. M16-only: n*16/key16,
long write only, short work discarded. Both are required, not an unspecified OR.

Post-onset-only: in the onset-containing window only, detach direct M16 reads
before onset while retaining identical forward values and state graph. Prior
windows and M4 learning stay intact. This ablation alone receives a harness
derivative mask; candidate receives no onset information. At D32 the mask is empty.
Deployed-ablation: forward causal rollout with candidate's already-written memory
versions but zero deployed corrections; no retraining/reapplication from the future.
Relaxed-cap: D8 engineering only, cap10,rho1, diagnostic and claim-ineligible.

Positive regimes D8/D16/R run all training arms above; deployed-ablation is a
paired causal rollout. C/D32 diagnostic arms: E,zero,candidate,reset,random,
immediate-protection,post-onset-only. No matrix reduction or unfavorable omission.
Prior-art full optimizer controls are deferred; no novelty claim is authorized.

## Statistics, gates and stops

Paired Student t 95% CI across ten seeds, df9. Positive benefit always means
reference excess minus candidate excess. C1 D8 as above. C2 compares D8 candidate
to all conventional/structure controls, requiring benefit >=50% of C1 and positive
interval; pure pre-credit contrast uses pooled D16/R and >=50% of their mean
benefit over E. Report D8 pre-credit's expected zero. C2 and D16/R secondary tests
are Holm-adjusted; select no favorable control/seed. C3 random-minus-candidate
tests learned direction; disabled-minus-candidate tests guard utility separately.
C4 D32 post-onset-minus-candidate tests the empty-mask invariant; reset contrast
measures persistence. No nonsignificance-as-equivalence inference. Acquisition
regression=(candidate B-end loss-E B-end loss)/max(E B-end loss,1e-8); margin1%.
Deployed benefit versus ablation must retain >=50% benefit_E with positive CI.

G1 CPU exact invariants and FP64 finite differences2/4/16, rel<=1e-3 if gradient
norm>1e-8, abs<=1e-6 otherwise. G1b uses engineering seeds1901..1910 only;
active actual retention ratio>=.05, proposed correction median r/base>1e-6,
at least one accepted non-rounded correction after first window. Calibrate guard
from100 engineering states (repeat/FP64 reduction); eps_abs=max(10*p95_abs,1e-7),
eps_rel=max(10*p95_rel,1e-5); stop if p95_rel>1e-4. MDE=(t.975,9+t.8,9)*sd(paired
engineering benefit)/sqrt10 must <=.05*mean E excess; E excess<=1e-8 is no headroom.
No criterion can be rescued by changing caps/objectives/regimes/seeds after outcomes.

G2 full16 CUDA, four short chunks/one long graph, both references, guard, memory
versions, resume/cleanup, <=3 measured windows, reserved<=4.5GiB, <=10x stripped
E, no retained growth, <=600s. G1 timeout600s; G1b bounded1800s; G3 total<=6GPUh
or stop before G3 if budget projection fails. G2 reserved/resource failures give
RESOURCE_INFEASIBLE; G1 failures IMPLEMENTATION_INVALID; signal/MDE failures
INCONCLUSIVE_UNDERPOWERED. Only an active,powered,valid experiment can yield
MECHANISM_SUPPORTED/MECHANISM_NOT_SUPPORTED. Record first failed gate and all skips.

Design Lock freezes this document, v3/review pins, canonical source pins, HEAD,
generator/seeds/arms/statistics and preservation metadata. Engineering code may be
developed after DL; actual source hashes/tolerances/resources/analysis freeze at
Execution Lock after G1/G1b/G2 and engineering-only prepare/train/witness/analyze/
verify dry-run. G3 seeds3101..3110 are forbidden before that lock. No real data,
anomaly scoring, sealed access, architecture search, P3 or automatic continuation.
