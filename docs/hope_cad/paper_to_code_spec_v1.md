# HOPE-CAD v1 paper-to-code specification

**Task 1 revision status: PASS.** Labels are **PAPER-DEFINED**, **PAPER-INFERRED**, **CAD-ADAPTATION**, or **EXPERIMENTAL-HYPOTHESIS**. The standalone SMT, generic CMS, and thin SMT-to-CMS HOPE composition now implement the locked core; anomaly scoring remains deferred.

Primary sources read directly: Nested Learning/HOPE arXiv:2512.24695v1 (especially Eq. 70–74, 79, 83–97), Titans arXiv:2501.00663v1 (neural-memory objective, surprise, momentum and decay), and CADIC arXiv:2511.08634v1 (control protocol and patch coreset).

## Equation-to-module map

| Paper equation/mechanism | Semantic meaning | State/input/output/update | Candidate module |
|---|---|---|---|
| NL Definition 2 | frequency is updates per unit time; one data update is the time unit | v1 makes the event unit explicit for each schedule | `models/hope_cad/state.py` |
| HOPE Eq. 94–97 | Self-Modifying Titans precedes sequential CMS | `o_t=M_memory,t-1(q_t)`; `y_t=MLP^(f_k)(...MLP^(f_1)(o_t))` | `models/hope_cad/hope_block.py` |
| Eq. 76 | projections produce key/value/query, learning rate and retention | `k=xW_k,v=xW_v,q=xW_q,eta=xW_eta,alpha=xW_alpha` | `self_modifying_titans.py` |
| Eq. 79–82 | preceding fully adaptive associative-memory variant | `k=M_k(x),v=M_v(x),q=M_q(x),eta=M_eta(x),alpha=M_alpha(x); o=M_memory(q)` | `self_modifying_titans.py` |
| Eq. 83–85 | self-modifying memories generate their own targets | `vhat_box=M_box(v)` and optimize each memory on `(k,vhat_box)`; Eq. 83 writes `q=xW_q` while Eq. 85 still includes `q` in the update set | `self_modifying_titans.py` |
| Eq. 86–89 | self-modifying DGD with retention/weight decay and associative objective | canonical code uses the fully derived linear specialization; Eq. 89 residual MLP is deferred | `self_modifying_titans.py` |
| Eq. 90–93 | chunk-wise parallel update | one memory chunk size and one shared auxiliary size for all other memories; generate a chunk from the previous state, then update at its boundary using the previous chunk state and gradient; the final chunk may be shorter | `self_modifying_titans.py` |
| HOPE q/k normalization and local convolution | reported HOPE sequence details | L2-normalize q and k; window 4 local convolution | `hope_block.py` |
| CMS Eq. 70 | sequential chain of frequency-specific MLPs | generic `K >= 1` ordered levels; `y=MLP_fk(...MLP_f1(x))` | `continuum_memory.py` |
| CMS Eq. 71 | scheduled optimization with objective chosen for task | image-event `update_periods`; detached per-event gradient accumulation and due-boundary plain GD mapping | `continuum_memory.py` |
| CMS Eq. 72–74 | nested, sequential, independent transfer choices | v1 implements only the sequential chain; nested reinitialization, meta-initialization, and head-wise aggregation remain deferred | `continuum_memory.py` |
| Titans objective/surprise | associative memory learns key→value; surprise is gradient-based memory error | linear `M(k)-v` diagnostic plus Eq. 93 parameter-gradient surprise; no extra momentum recurrence | `self_modifying_titans.py` |
| CADIC control | bounded Euclidean patch reference and common image score | normal-only reference updates; no HOPE semantics | existing `CADICPatchCoresetV1`, wrapped by future adapter |

The primary paper has a narrower, equation-level `M_q` ambiguity than the earlier v1 wording implied. Eq. 79–82 describe a preceding fully adaptive variant and include `M_q`. In the self-modifying formulation, Eq. 83 writes `q_t=x_tW_q` and the accompanying prose calls it “the only non-adaptive projection,” but Eq. 85 still lists `q` in the memories being optimized. Therefore adaptive `M_q` is **UNRESOLVED in the primary source** for the self-modifying formulation. v1 must retain `adaptive_q=True|False` as an explicit project switch, must not label the adaptive path paper-defined, and must record which path is selected for any experiment.

The primary source also leaves two implementation-level questions open. Eq. 88 contains the matrix-shaped `alpha_t I-eta_t k_t k_t^T` term while stating that memory architectures are arbitrary and then giving the residual MLP in Eq. 89; Eq. 93 derives the displayed rank-one form specifically for a linear memory. The paper does not provide a parameter-tensor version of this term for a deep MLP. Likewise, Eq. 76/86 write `eta_t` and `alpha_t` as learned projections but Eq. 88 uses them as scalar learning-rate and retention controls; no vector-to-scalar reduction is specified. These are **UNRESOLVED**, not paper-defined behavior.

## Canonical linear SMT equation lock

The user-selected canonical reference is the **linear** specialization explicitly derived in Eq. 93, not the reported experimental residual MLP architecture of Eq. 89/91. The residual/deep implementation and its parameter-space DGD rule are **DEFERRED**. Source checked directly: local `docs/papers/Nested_Learning_The_Illusion_of_Deep_Learning_Architectures.pdf`, printed pages 31–33, Eq. 79–93.

Use column-vector notation below; PyTorch stores each map as `[output_dim,D]` and applies `F.linear`. Let `h` be the local-convolution output and let `W_box^0` be the state at the start of the corresponding chunk.

| Memory | Object / state shape | Read input / output | Write key / target | Clock |
|---|---|---|---|---|
| `M_k` | linear `[D,D]` | `h → k_raw`, then L2 normalize | `k / W_k^0 v` | auxiliary chunk |
| `M_v` | linear `[D,D]` | `h → v` | `k / W_v^0 v` | auxiliary chunk |
| `M_eta` | linear `[1,D]` | `h → eta_raw → eta` scalar | `k / W_eta^0 v` (raw linear target) | auxiliary chunk |
| `M_alpha` | linear `[1,D]` | `h → alpha_raw → alpha` scalar | `k / W_alpha^0 v` (raw linear target) | auxiliary chunk |
| `M_memory` | linear `[D,D]` | normalized `q → o` | `k / W_memory^0 v` | memory chunk |
| optional `M_q` | linear `[D,D]` | `h → q_raw`, then L2 normalize | `k / W_q^0 v` | auxiliary chunk |

Canonical query is the ordinary, fixed-during-online-learning `q=normalize(W_q h)` with **`adaptive_q=False` by default**. The optional adaptive query uses the same update graph as other memories and is **PAPER-INFERRED**, not the canonical experiment.

Eq. 84/87 self-targets apply to **all** listed memories, including `memory`: `vhat_box,i = W_box^0 v_i`. Targets and keys are held fixed for each inner gradient. With Eq. 93's explicitly printed gradient convention,

```text
ell_box,i = (1/2) ||W_box^0 k_i - vhat_box,i||²
G_box,i = (W_box^0 k_i - vhat_box,i) k_i^T
W_box,i = W_box,i-1 (alpha_i I - eta_i k_i k_i^T) - eta_i G_box,i
```

Both the rank-one data-dependent retention and the gradient term are required. Within a chunk, the recurrent left state advances token by token, but every gradient/target uses the fixed chunk-start state. There is **no independent momentum recurrence in Eq. 90–93**; do not import the original Titans optimizer/momentum into this canonical equation path. Parameter-gradient surprise is momentary `G_box,i`; its chunk sum is only a temporary diagnostic, not a momentum buffer. The paper states squared L2 but its Eq. 93 omits the factor two; the half-squared loss convention is an explicit **PROJECT-MAPPING** to reproduce that printed recurrence exactly.

`M_memory(k)-v` and its squared L2 loss remain raw associative inspection values. The actual Eq. 93 write residual is separately `M_memory^0(k)-M_memory^0(v)`. Calling the raw residual the self-modifying write objective would contradict Eq. 84/87/93.

**PROJECT-MAPPING:** scalar controls are produced directly by `[1,D]` maps, then `eta=sigmoid(eta_raw)`, `alpha=sigmoid(alpha_raw)`; no vector averaging, epsilon offset, or silent clamp. The source does not specify range activations. These scalar postprocessors are outside the linear memories, so self-targets remain linear outputs. The Conv1D uses dense channels, bias, and same-length zero padding `(1,2)` over row-major tokens; those details are project mappings beyond the paper's window 4. Xavier-uniform slow initial matrices serve as reset sources; no outer-loop meta-training is implemented.

Two independent pending streams include every token exactly once. Flush full chunks and the shorter final remainder at each image boundary. At coincident boundaries, prepare all candidates/targets/gradients from captured pre-chunk states before committing any memory. For unequal clocks, hold each memory's read/gradient state until its own next boundary; this asynchronous cross-clock convention is an explicit **PROJECT-MAPPING** because Eq. 90 uses one displayed chunk index. Follow the prose's previous-chunk state; the printed `C ceil(t/C)` subscript is **UNRESOLVED** as a time index and must not create a look-ahead read.

## State inventory and scope

| state | shape/role | initialized | update clock | task boundary | image/batch/task persistence | checkpoint | official evaluation | label |
|---|---|---|---|---|---|---|---|---|
| frozen ViT parameters | ViT-B/8; block 8 output `[784,768]` | checkpoint load | never | no reset | persists/persists/persists | yes | unchanged | CAD-ADAPTATION |
| backbone patch sequence `x` | 784 tokens, row-major | each image | none | fresh | no/no/no | no | read-only | CAD-ADAPTATION |
| `M_k,M_v,M_q,M_eta,M_alpha,M_memory` persistent | adaptive linear memories; `M_eta,M_alpha` output scalars | new run/seed/replicate | auxiliary or memory chunk boundaries | **NO RESET** | persistent/persistent/persistent | yes | cloned to working state; persistent copy unchanged | PAPER-DEFINED + PAPER-INFERRED |
| Titans gradient/surprise diagnostics | temporary parameter-gradient tensors from Eq. 93 | each chunk | corresponding chunk boundary | fresh | working-state only | no | cloned and discarded with working state | PAPER-DEFINED |
| Titans working state | deep clone of all above | clone for each official test image | patch/chunk within image | discarded | ephemeral/ephemeral/ephemeral | no | may mutate causally within image | CAD-ADAPTATION |
| CMS level parameters | generic residual MLP `D -> H -> D` (default `H=D`, GELU, bias, no normalization) | new run/seed/replicate | configured image-event periods on train stream | **NO RESET** | persistent/persistent/persistent | yes | **frozen during each test image** | PAPER-INFERRED + PROJECT-MAPPING |
| CMS scheduler counters | image-event counters | new run | increment after each train image | **NO RESET** | persistent/persistent/persistent | yes | no increment | CAD-ADAPTATION |
| CMS update state | detached per-parameter gradient accumulators, pending counts, and update counts | new run | accumulate each image; commit at period boundary | **NO RESET** | persistent/persistent/persistent | yes | frozen | PROJECT-MAPPING |
| HOPE `y` | `[784,768]` representation | each image | forward | fresh | no/no/no | no | from working clone | PAPER-DEFINED |
| Head-A reference bank (deferred) | ≤2,500 HOPE patch vectors if later enabled | first normal stream | unspecified pending stable-reference design | **NO RESET** | persistent/persistent/persistent | yes | no mutation | CAD-ADAPTATION / BLOCKED |
| Head-A parent contexts | potentially many parent images because arbitrary patch coreset entries need context | unspecified | unspecified | **NO RESET** | persistent/persistent/persistent | TBD | no mutation | CAD-ADAPTATION / BLOCKED |
| Head-B residual/surprise map | `[784]` | each image | forward | fresh | no/no/no | no | working clone only | EXPERIMENTAL-HYPOTHESIS |
| RNG state affecting model | framework RNG | run | only deterministic declared operations | **NO RESET** within run | persistent/persistent/persistent | yes if needed | test scoring must not advance relevant RNG | CAD-ADAPTATION |

**Reset rule:** bottle→cable→…→zipper has no reset for any persistent HOPE, CMS, scheduler, optimizer, or reference state. A new independent run, seed, or replicate starts from initial checkpoint/empty memory. “Batch” is a transport unit, not a semantic reset.

## Clocks and schedules

Titans is the fast patch-local clock. `memory_chunk_size` updates `M_memory`; `auxiliary_memory_chunk_size` is shared by `M_k,M_v,M_eta,M_alpha` and optional `M_q`. They may both be 16 in the first run. Each stream includes every token once and flushes its shorter final remainder. For unequal clocks, each stream retains its own chunk-start state until its boundary; this asynchronous pending-stream convention is a **PROJECT-MAPPING** because Eq. 90 displays one chunk index.

CMS is the longer continual compression clock and is **not** scheduled every 16/128 patches. The generic core names its integer configuration `update_periods`: period `p` commits after events `p, 2p, ...`. A completed normal training image is one project-defined event. The first event is 1, there is no event-0 update, and an incomplete period is retained rather than flushed at a task boundary or run end. The baseline schedule remains `[1, 8]`, giving approximately 1 and 0.125 updates per image. CMS objectives are supplied per level through `objective(level_index, level_input, level_output, metadata)`; the primitive does not hard-code anomaly scoring or the Titans L2 loss.

## Canonical CMS core implementation

`models/hope_cad/continuum_memory.py` implements `ContinuumMemorySystem` with generic `K >= 1` sequential levels. Each level is a project-mapped residual two-layer MLP `x + W2(GELU(W1(x)+b1))+b2`, defaulting to `hidden_dim=D`, bias enabled, and no normalization. Static initialization parameters are ordinary `nn.Parameter` tensors; current weights and biases are registered buffers. Each current tensor has a same-shaped detached gradient accumulator. `commit_image` computes the complete pre-event chain, evaluates every local objective, accumulates detached gradients transactionally, and commits due levels with `current -= learning_rate[level] * accumulated_gradient`. `forward` is always read-only. `commit_batch` processes images sequentially as separate events. Native state serialization includes current state, static initialization, accumulators, counters, and schema/configuration metadata. `clone_for_evaluation` returns an isolated frozen read-only copy. Plain accumulated gradient descent, residual geometry, GELU, initialization, event clock, local detached optimization, no partial flush, and test-time CMS freezing are **PROJECT-MAPPING** decisions; CMS structure, sequential composition, scheduled updates, and task-selected objective boundary are **PAPER-DEFINED**.

## Canonical HOPE composition

`models/hope_cad/hope_block.py` implements `HopeBlock` as an orchestration layer only:

```text
x [B,N,D]
  -> one SMT read/update pass
  -> exact causal SMT memory_prediction o [B,N,D]
  -> CMS sequential chain
  -> y [B,N,D]
```

`forward(x)` is strictly read-only. `commit_image(x, objectives, metadata)` validates one normal image, snapshots only mutable SMT/CMS continuation state, runs exactly one `smt.forward(update=True)`, passes that exact causal `memory_prediction` to one `cms.commit_image`, and restores both online systems if any post-mutation operation fails. `commit_batch` is ordered singleton transport. `evaluate_image(x)` deep-clones the full wrapper, permits causal SMT evolution inside the clone, keeps CMS frozen through `cms.forward`, and discards the clone. The wrapper owns no scientific online tensors and introduces no second event clock. These transaction, rollback, and full-clone evaluation rules are **PROJECT-MAPPING** choices; the SMT-to-CMS order and final CMS output follow HOPE Eq. 94–97.

The measured FP32 tensor payload for the default D=768, CMS K=2 configuration is 25,967,640 B for SMT, 28,348,456 B for CMS, and 54,316,096 B for the composed core. Mutable continuation state is 25,983,040 B; wrapper accounting adds no tensor state.

## Official-evaluation working state

Let `S_persistent` be the state after train/good updates. For each official test image `i`, create `S_work_i = deep_clone(S_persistent)`. Run its 784 patches causally; Self-Modifying Titans may update `S_work_i` within the image. CMS parameters are frozen for the entire test image because their v1 clock is train-image based; this avoids a hidden image-level CMS update and preserves the longer-timescale interpretation. Produce `y_i`/scores, then discard `S_work_i`. `S_persistent` must be semantically and, where tensor order permits, bitwise unchanged. No reference bank, CMS counter, optimizer state, or relevant RNG state changes. Thus test image A cannot affect B and test order cannot affect results.

## Head-A stale-representation policy

A mutable HOPE encoder makes a bank entry `y_old=HOPE(S_old,x_old)` stale after state changes. A 2,500-patch bank may contain patches from many parent images; retaining only four full source sequences cannot guarantee context coverage, while retaining arbitrary parent contexts creates a hidden replay/storage budget. Therefore v1 does **not** claim to solve stale Head-A representations. The HOPE-transformed patch-NN head is **DEFERRED / BLOCKED** pending a bounded, context-correct stable-reference design. Candidate future designs include a frozen scoring/reference HOPE copy, a stable pre-HOPE reference space with HOPE kept on a separate prediction path, or a formally bounded parent-context store; each requires a new decision and storage/compute accounting.

## Spatial operator decision

The HOPE paper's local convolution window 4 is **PAPER-DEFINED** for its sequence experiments. v1 uses that paper-faithful 1D operator over row-major patch order. It therefore has a known raster-boundary artifact: `(row 0,col 27)` is adjacent to `(row 1,col 0)` in 1D despite not being a natural 2D neighbor. A 2D spatially aware equivalent would be a **CAD-ADAPTATION**, not a silent implementation substitution; it is deferred to an ablation.

## Three separate objectives

1. **Titans objective (paper-grounded):** associative L2 `L_T(M;k,v)=||M(k)-v||²`, with self-generated `vhat` and the linear Eq. 93 DGD/retention machinery. The printed canonical equation has no additional momentum recurrence.
2. **CMS optimization objective (paper-permitted, not fixed by HOPE):** `ContinuumMemorySystem.commit_image` accepts one pure `objective(level_index, level_input, level_output, metadata)` callable per level; the actual CAD self-supervised objective is a later design decision.
3. **Anomaly scoring objective:** primary Head B residual/surprise; deferred Head A Euclidean NN distance only after a stable-reference design. Neither trains CMS implicitly.

## Benchmark mapping

The future versioned adapter implements `fit_task`, `score_batch`, `state_dict`, `load_state_dict`, `memory_stats`, and `method_metadata`. `fit_task` updates only train/good data. `score_batch` clones persistent state per image, freezes CMS, scores read-only, and returns `image_scores` plus `[B,224,224]` maps. Existing MVTec order, official-test isolation, all-pixel P-AUPR, final macro metrics, and forgetting matrix remain unchanged.
