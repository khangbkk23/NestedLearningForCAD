# Pre-Execution-Lock resource implementation amendment

The59-cell engineering dry-run exposes substantial duplicated prefix work.
All regimes have identical arriving distributions/images, masks, replay/anchor
draws and learner state through completed cycle64. At that boundary all16-cycle
graphs are released. The serialized tuple of fast parameters, Adam moments/clock,
memory values, direction RNG, stream/replay/anchor RNGs and replay contents is a
sufficient Markov state for subsequent updates. Evaluator witnesses/RNG remain
outside the learner and are restored separately.

Reuse that completed prefix within the same seed and arm across regimes only.
Different arms and seeds never share learned states. Prefixes contain no B
parameters, future image or phase-B witnesses. Set the harness regime only after
restoration; learner receives no phase metadata. Every subsequent16-cycle graph
and memory write remains unchanged. This neither removes a control nor shortens
a credit horizon. Prefix tensor artifacts are immutable shared links; complete
logical trajectories and per-arm12-gradient accounting remain available.

Before Execution Lock, test all15 arm restorations against independently trained
D16 full-shape results and test generator/replay continuation for every regime.
Actual physical GPU cost charges the shared prefix once; logical per-arm costs
include it. Paired regime statistics were already averaged within seed, so reuse
does not change statistical units or confidence intervals. Projection uses measured
engineering prefix/suffix timings plus20% evaluation headroom and includes all
extension engineering costs in the6hour ceiling. No evaluation seeds are used
for these checks or resource calibration.

This is deterministic implementation completion before Execution Lock, not a
change selected for a favorable retention result. Original P2h and extension
Design Locks remain unchanged; Execution Lock additionally pins this amendment.
