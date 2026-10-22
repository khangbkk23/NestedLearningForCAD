# Proposal: historical-query slow consolidation

Trạng thái **chưa approve/freeze/implement**. Tách khỏi workspace v7/port5090. Seed0 matrices có pixel-retention benefit, chưa image-retention benefit; saved metadata388/401slowepisodes và3199/3211delayedlabels cùngcategory. Đây là observations hậu nghiệm MVTec development, không chứng minh nguyên nhân/gain.

## Can thiệp tối thiểu

Câu hỏi: sau fast adaptation trên recent contexts, slow objective có giữ utility mapping của past contexts tốt hơn không? V5.1 query4later records trong cùng block8; block task stream khiến nó ít chịu explicit historical pressure. Nhiều clocks/width không tự sửa query distribution.

Giữ raw ViT/CADIC proposal/score/data rights, actualphi học cả8labels, fast rate/energy denominator và θ/φ ownership. Chỉ thay query sampler slow update:4chronological support, query recent+completed historical theo tỷ lệ khóa trước run, bounded memory và fixed forward budget. Không thêm full counterfactual rollouts/shadowbanks.

Historical record chứa decision tokens/stats, storedc/gamma và detached target của **original fixed bank pair tại thời điểm gốc**. Không recompute target bằng current bank hoặc gọi nó là reward hiện tại. Chỉ completedlabels vào buffer; không pending/future labels. Historical queries disjoint với current support/query; không task/categoryID sampling.

Sau slow commit giữ actualphi8, không clonephi4/reset task. Buffer/insertion/sampling/privateRNG/eligiblepool/collision/emptyfallback/partialepisode phải vào atomic checkpoint/rollback. Scoring không consume replayRNG.

## Căn cứ và giới hạn

MSE y=Delta/c có conditional optimum của original contrast; historical query thay distribution mà slow representation được chấm sau fast adaptation. Nó có thể tạo credit cho retained utility mapping. Nhưng old labels chịu bank/distribution drift, nén context/hard whole-image admission còn hạn chế. Giảm utility error không bảo đảm raw support retention/anomaly ranking/VisA gain; không convergence/regret theorem ở đây.

## Phải chốt trước implement

1. MethodID riêng, capacity/bytes/query split/privateRNG và fallback. Không tự chọn64/128 để lấp spec khi chưa review tradeoff.
2. Equal-cache recent-only vs recent+historical; GEOMETRIC/FAST_ONLY/FIRST_ORDER attribution; seeds/trials/primary decision rule.
3. Tests exact gradient quotient, immutable causal labels, disjoint/no-future access, buffer/clock/RNG resume/rollback, read-only score/equal-forward accounting.
4. Normal pilot utility/sign regret, functional meta response, actual bank effects, old/new normal coverage/compute/bytes/fail-stop rules khóa trước run; không tăng budget để cứu gate.
5. Giữ mọi development trials/negative results; freeze backbone/preprocessing/policy/scoring/seeds/protocol trước VisA.

Preservation/refactor → real5090preflight/timing → review nền/attribution và chốt full revision spec → implement/local tests → normal pilot → bounded development. Không tự đưa draft vào code khi cleanup; không gán preprocessing/GPU gain cho Nested Learning.
