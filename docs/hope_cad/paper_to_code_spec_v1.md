# HOPE-CAD v1 paper-to-code specification

**Task 1 revision status: PASS.** Labels are **PAPER-DEFINED**, **PAPER-INFERRED**, **CAD-ADAPTATION**, or **EXPERIMENTAL-HYPOTHESIS**. This document is a specification only; it adds no model code.

Primary sources read directly: Nested Learning/HOPE arXiv:2512.24695v1 (especially Eq. 70–74, 79, 83–97), Titans arXiv:2501.00663v1 (neural-memory objective, surprise, momentum and decay), and CADIC arXiv:2511.08634v1 (control protocol and patch coreset).

## Equation-to-module map

| Paper equation/mechanism | Semantic meaning | State/input/output/update | Candidate module |
|---|---|---|---|
| NL Definition 2 | frequency is updates per unit time; one data update is the time unit | v1 makes the event unit explicit for each schedule | `models/hope_cad/state_v1.py` |
| HOPE Eq. 94–97 | Self-Modifying Titans precedes sequential CMS | `o_t=M_memory,t-1(q_t)`; `y_t=MLP^(f_k)(...MLP^(f_1)(o_t))` | `models/hope_cad/hope_block_v1.py` |
| Eq. 76 | projections produce key/value/query, learning rate and retention | `k=xW_k,v=xW_v,q=xW_q,eta=xW_eta,alpha=xW_alpha` | `self_modifying_titans_v1.py` |
| Eq. 79–82 | adaptive memories generate projections and retrieve | `k=M_k(x),v=M_v(x),q=M_q(x),eta=M_eta(x),alpha=M_alpha(x); o=M_memory(q)` | `self_modifying_titans_v1.py` |
| Eq. 83–85 | self-modification generates each memory's target | `vhat_box=M_box(v)` and optimize each memory on `(k,vhat_box)` | `self_modifying_titans_v1.py` |
| Eq. 86–89 | DGD with retention/weight decay and associative objective | `L_T(M;k,v)=||M(k)-v||²`; `M_box,t=M_box,t-1(alpha_t I-eta_t k_t k_t^T)-eta_t∇L_T`; reported memory is residual 2-layer MLP | `self_modifying_titans_v1.py` |
| Eq. 90–93 | chunk-wise parallel update | generate a chunk from prior state, then update at boundary using the prior chunk state and gradient | `self_modifying_titans_v1.py` |
| HOPE q/k normalization and local convolution | reported HOPE sequence details | L2-normalize q and k; window 4 local convolution | `hope_block_v1.py` |
| CMS Eq. 70 | sequential chain of frequency-specific MLPs | `y=MLP_fk(...MLP_f1(x))` | `continuum_memory_v1.py` |
| CMS Eq. 71 | scheduled optimization with objective chosen for task | update `theta_l` only at its schedule; generic `objective/update` interface required | `continuum_memory_v1.py` |
| CMS Eq. 72–74 | nested, sequential, independent transfer choices | v1 chooses sequential chain; no nested task reset | `continuum_memory_v1.py` |
| Titans objective/surprise | associative memory learns key→value; surprise is gradient-based memory error | residual and gradient surprise are exposed for Head B; momentum/decay are state | `self_modifying_titans_v1.py` |
| CADIC control | bounded Euclidean patch reference and common image score | normal-only reference updates; no HOPE semantics | existing `CADICPatchCoresetV1`, wrapped by future adapter |

The paper contains a real ambiguity: Eq. 79/83–88 place an adaptive `M_q` alongside `M_k,M_v,M_eta,M_alpha`, while nearby prose says `q_t=x_tW_q` is “the only non-adaptive projection.” This is inconsistent in the primary source. v1 interprets the explicit Eq. 79/83 computational graph as **PAPER-INFERRED** adaptive `M_q`, while retaining a learned fixed `W_q` as its base/initial projection; the implementation must expose a flag and metadata for `adaptive_q=True|False`, never silently call it unambiguously paper-defined.

## State inventory and scope

| state | shape/role | initialized | update clock | task boundary | image/batch/task persistence | checkpoint | official evaluation | label |
|---|---|---|---|---|---|---|---|---|
| frozen ViT parameters | ViT-B/8; block 8 output `[784,768]` | checkpoint load | never | no reset | persists/persists/persists | yes | unchanged | CAD-ADAPTATION |
| backbone patch sequence `x` | 784 tokens, row-major | each image | none | fresh | no/no/no | no | read-only | CAD-ADAPTATION |
| `M_k,M_v,M_q,M_eta,M_alpha,M_memory` persistent | adaptive residual MLP memories, width 768 | new run/seed/replicate | Titans chunk boundaries | **NO RESET** | persistent/persistent/persistent | yes | cloned to working state; persistent copy unchanged | PAPER-DEFINED + PAPER-INFERRED |
| Titans optimizer/momentum state | matching auxiliary tensors | new run | Titans chunk boundaries | **NO RESET** | persistent/persistent/persistent | yes | cloned; discarded after image | PAPER-DEFINED |
| Titans working state | deep clone of all above | clone for each official test image | patch/chunk within image | discarded | ephemeral/ephemeral/ephemeral | no | may mutate causally within image | CAD-ADAPTATION |
| CMS level-1/level-2 parameters | residual MLP width 768 | new run/seed/replicate | image events (see below) on train stream | **NO RESET** | persistent/persistent/persistent | yes | **frozen during each test image** | PAPER-DEFINED + EXPERIMENTAL-HYPOTHESIS |
| CMS scheduler counters | image-event counters | new run | increment after each train image | **NO RESET** | persistent/persistent/persistent | yes | no increment | CAD-ADAPTATION |
| CMS optimizer state | per-level objective/optimizer tensors | new run | level schedule | **NO RESET** | persistent/persistent/persistent | yes | frozen | PAPER-DEFINED |
| HOPE `y` | `[784,768]` representation | each image | forward | fresh | no/no/no | no | from working clone | PAPER-DEFINED |
| Head-A reference bank (deferred) | ≤2,500 HOPE patch vectors if later enabled | first normal stream | unspecified pending stable-reference design | **NO RESET** | persistent/persistent/persistent | yes | no mutation | CAD-ADAPTATION / BLOCKED |
| Head-A parent contexts | potentially many parent images because arbitrary patch coreset entries need context | unspecified | unspecified | **NO RESET** | persistent/persistent/persistent | TBD | no mutation | CAD-ADAPTATION / BLOCKED |
| Head-B residual/surprise map | `[784]` | each image | forward | fresh | no/no/no | no | working clone only | EXPERIMENTAL-HYPOTHESIS |
| RNG state affecting model | framework RNG | run | only deterministic declared operations | **NO RESET** within run | persistent/persistent/persistent | yes if needed | test scoring must not advance relevant RNG | CAD-ADAPTATION |

**Reset rule:** bottle→cable→…→zipper has no reset for any persistent HOPE, CMS, scheduler, optimizer, or reference state. A new independent run, seed, or replicate starts from initial checkpoint/empty memory. “Batch” is a transport unit, not a semantic reset.

## Clocks and schedules

Titans is the fast patch-local clock. Its v1 chunk contract has two explicit fields: `memory_chunk_size` for `M_memory`, and `auxiliary_memory_chunk_size` for each applicable `M_k,M_v,M_q,M_eta,M_alpha`. They may both be 16 in the first run (equal values are an explicit hypothesis, not a collapsed API). A 784-patch image therefore has approximately 49 Titans memory and auxiliary chunk boundaries when 16 is used.

CMS is the longer continual compression clock and is **not** scheduled every 16/128 patches. v1 uses image events: CMS level 1 updates once after every 1 completed normal training image; CMS level 2 updates once after every 8 completed normal training images. Approximate updates are therefore 1 and 0.125 per image, or for a task with `N` good images, `N` and `floor(N/8)` updates. The exact numbers are **EXPERIMENTAL-HYPOTHESIS**. They create a timescale distinct from the 784-token Titans dynamics while keeping one minimal two-level design. CMS update objectives are generic interfaces; the CMS primitive must not hard-code anomaly scoring or the Titans L2 loss.

## Official-evaluation working state

Let `S_persistent` be the state after train/good updates. For each official test image `i`, create `S_work_i = deep_clone(S_persistent)`. Run its 784 patches causally; Self-Modifying Titans may update `S_work_i` within the image. CMS parameters are frozen for the entire test image because their v1 clock is train-image based; this avoids a hidden image-level CMS update and preserves the longer-timescale interpretation. Produce `y_i`/scores, then discard `S_work_i`. `S_persistent` must be semantically and, where tensor order permits, bitwise unchanged. No reference bank, CMS counter, optimizer state, or relevant RNG state changes. Thus test image A cannot affect B and test order cannot affect results.

## Head-A stale-representation policy

A mutable HOPE encoder makes a bank entry `y_old=HOPE(S_old,x_old)` stale after state changes. A 2,500-patch bank may contain patches from many parent images; retaining only four full source sequences cannot guarantee context coverage, while retaining arbitrary parent contexts creates a hidden replay/storage budget. Therefore v1 does **not** claim to solve stale Head-A representations. The HOPE-transformed patch-NN head is **DEFERRED / BLOCKED** pending a bounded, context-correct stable-reference design. Candidate future designs include a frozen scoring/reference HOPE copy, a stable pre-HOPE reference space with HOPE kept on a separate prediction path, or a formally bounded parent-context store; each requires a new decision and storage/compute accounting.

## Spatial operator decision

The HOPE paper's local convolution window 4 is **PAPER-DEFINED** for its sequence experiments. v1 uses that paper-faithful 1D operator over row-major patch order. It therefore has a known raster-boundary artifact: `(row 0,col 27)` is adjacent to `(row 1,col 0)` in 1D despite not being a natural 2D neighbor. A 2D spatially aware equivalent would be a **CAD-ADAPTATION**, not a silent implementation substitution; it is deferred to an ablation.

## Three separate objectives

1. **Titans objective (paper-grounded):** associative L2 `L_T(M;k,v)=||M(k)-v||²`, with self-generated `vhat` and the paper's DGD/retention/momentum machinery.
2. **CMS optimization objective (paper-permitted, not fixed by HOPE):** `continuum_memory_v1.py` exposes `objective(state, chunk, context)` and `update(...)`; the actual CAD self-supervised objective is a later design decision (Task 3 if not defensible earlier).
3. **Anomaly scoring objective:** primary Head B residual/surprise; deferred Head A Euclidean NN distance only after a stable-reference design. Neither trains CMS implicitly.

## Benchmark mapping

The future versioned adapter implements `fit_task`, `score_batch`, `state_dict`, `load_state_dict`, `memory_stats`, and `method_metadata`. `fit_task` updates only train/good data. `score_batch` clones persistent state per image, freezes CMS, scores read-only, and returns `image_scores` plus `[B,224,224]` maps. Existing MVTec order, official-test isolation, all-pixel P-AUPR, final macro metrics, and forgetting matrix remain unchanged.
