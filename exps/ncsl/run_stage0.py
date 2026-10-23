"""Execute the corrected NCSL-01 normal-only Stage 0 experiment."""
from __future__ import annotations

import argparse, hashlib, json, os, platform, time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import core
from . import protocol as P


def bins_of(x: torch.Tensor, positions: list[np.ndarray]) -> list[torch.Tensor]:
    return [x[:, torch.as_tensor(pos), :].reshape(-1, x.shape[-1]).contiguous() for pos in positions]


def image_bins(x: torch.Tensor, positions: list[np.ndarray]) -> list[torch.Tensor]:
    return [x[:, torch.as_tensor(pos), :].reshape(-1, x.shape[-1]).contiguous() for pos in positions]


def score_grid(metric: core.LowRankMetric, queries: torch.Tensor,
               supports: list[torch.Tensor], positions: list[np.ndarray],
               device: str = "cpu") -> np.ndarray:
    # Exact per-bin NN on the shared metric; chunks avoid a giant 784 x N matrix.
    qbins = bins_of(queries, positions)
    out = torch.empty((queries.shape[0], 784), dtype=torch.float64)
    with torch.no_grad():
        for b, pos in enumerate(positions):
            q = metric.apply(qbins[b].to(device))
            s = metric.apply(supports[b].to(device))
            d = core.nearest(q, s).detach().cpu().to(torch.float64)
            out[:, torch.as_tensor(pos)] = d.reshape(queries.shape[0], len(pos))
    return out.numpy()


def full_teacher_grid(metric: core.LowRankMetric, queries: torch.Tensor,
                      full_bins: list[torch.Tensor], positions: list[np.ndarray],
                      device: str = "cpu") -> np.ndarray:
    return score_grid(metric, queries, full_bins, positions, device)


def train_metric(metric: core.LowRankMetric, full_bins: list[torch.Tensor],
                 hold_bins: list[torch.Tensor], quotas: np.ndarray, mode: str,
                 seed: int, device: str) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Alternating metric optimization with stop-gradient through FPS identities."""
    metric.to(device)
    full_dev = [x.to(device) for x in full_bins]
    hold_dev = [x.to(device) for x in hold_bins]
    support, ids = core.select_support(full_bins, metric, quotas)
    opt = torch.optim.Adam(metric.parameters(), lr=P.LR)
    g = torch.Generator(device="cpu").manual_seed(seed)
    losses, norms, overlaps = [], [], []
    for epoch in range(P.EPOCHS):
        opt.zero_grad(set_to_none=True)
        total, count = None, 0
        for b in range(core.BINS):
            q = hold_dev[b]
            n = min(q.shape[0], max(1, P.PROBE_BATCH // core.BINS))
            idx = torch.randperm(q.shape[0], generator=g)[:n]
            qb = q[idx]
            tq = metric.apply(qb)
            td = metric.apply(full_dev[b])
            ts = metric.apply(support[b].to(device))
            dd = core.nearest(tq, td)
            ds = core.nearest(tq, ts)
            term = ((ds - dd) / (dd + P.TAU)).sum()
            total = term if total is None else total + term
            count += int(n)
        loss = total / float(count)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite NCSL loss")
        loss.backward()
        opt.step()
        observed = metric.project(core.GAMMA)
        losses.append(float(loss.detach().cpu()))
        norms.append(observed)
        if mode == "nested" and (epoch + 1) % P.RESELECT_EVERY == 0:
            old_ids = [x.copy() for x in ids]
            support, ids = core.select_support(full_bins, metric, quotas)
            changed = sum(int(not np.array_equal(a, z)) for a, z in zip(old_ids, ids))
            overlaps.append(int(changed))
    if not all(torch.isfinite(p).all() for p in metric.parameters()):
        raise RuntimeError("non-finite metric state")
    return support, {"losses": losses, "spectral_norms": norms,
                     "reselection_changed_bins": overlaps,
                     "metric_digest": metric.state_digest(),
                     "final_support_ids": [x.tolist() for x in ids]}


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64); ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def correlations(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ra, rb = rankdata(a), rankdata(b)
    def corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0: return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    return {"spearman": corr(ra, rb), "kendall": float(np.mean(np.sign((a[:, None]-a[None, :]) * (b[:, None]-b[None, :])) >= 0) * 2 - 1) if len(a) < 2000 else float("nan")}


def bootstrap_ci(values_by_cat: dict[str, np.ndarray], seed: int = 20261022,
                 reps: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        means = []
        for cat in sorted(values_by_cat):
            x = values_by_cat[cat]
            means.append(float(x[rng.integers(0, len(x), len(x))].mean()))
        vals.append(float(np.mean(means)))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def metrics_for_arm(scores: np.ndarray, teacher: np.ndarray, categories: list[str]) -> dict[str, Any]:
    excess = np.maximum(scores - teacher, 0.0)
    rel = excess / (teacher + P.TAU)
    out: dict[str, Any] = {}
    cat_arr = np.asarray(categories)
    by = {}
    for cat in sorted(set(categories)):
        mask = cat_arr == cat
        e = excess[mask].reshape(-1); r = rel[mask].reshape(-1)
        tail = e[np.argsort(teacher[mask].reshape(-1))[-max(1, int(.05*e.size)):]]
        by[cat] = {"mean": float(e.mean()), "median": float(np.median(e)),
                   "p95": float(np.quantile(e,.95)), "max": float(e.max()),
                   "relative_mean": float(r.mean()), "tail5_mean": float(tail.mean()),
                   "image_mean": float(scores[mask].max(1).mean())}
    out["per_category"] = by
    out["equal_category_macro_mean"] = float(np.mean([v["mean"] for v in by.values()]))
    out["equal_category_macro_tail5_mean"] = float(np.mean([v["tail5_mean"] for v in by.values()]))
    out["pooled"] = {"mean": float(excess.mean()), "median": float(np.median(excess)),
                     "p95": float(np.quantile(excess,.95)), "max": float(excess.max()),
                     "relative_mean": float(rel.mean())}
    return out


def patch_overlap(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    k = max(1, int(round(a.shape[1] * frac)))
    vals = []
    for xa, xb in zip(a, b):
        ia = set(np.argpartition(xa, -k)[-k:].tolist())
        ib = set(np.argpartition(xb, -k)[-k:].tolist())
        vals.append(len(ia & ib) / float(k))
    return float(np.mean(vals))


def arm_auxiliary(scores: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    im, refim = scores.max(1), reference.max(1)
    return {"image_rank": correlations(im, refim),
            "top1_patch_overlap": patch_overlap(scores, reference, .01),
            "top5_patch_overlap": patch_overlap(scores, reference, .05)}


def run_boundary(data: dict, cache, t: int, device: str, outdir: Path) -> dict[str, Any]:
    positions = P.bin_positions(); updates = data["updates"]
    fit_idx, hold_idx = P.split_events(updates, t)
    fit = P.stack_events(cache, updates, fit_idx); hold = P.stack_events(cache, updates, hold_idx)
    eligible_witnesses = {200: 39, 300: 59, 350: 59}[t]
    witness_rows = data["probes"][:eligible_witnesses]
    witnesses = torch.stack([cache.row(row["category"], int(row["cache_index"]))
                             for row in witness_rows]).float()
    fit_bins, hold_bins = bins_of(fit, positions), bins_of(hold, positions)
    # Witnesses are diagnostic only and never enter fitting or selection.
    fit_cats = [updates[i]["category"] for i in fit_idx]
    hold_cats = [updates[i]["category"] for i in hold_idx]
    wit_cats = [x["category"] for x in witness_rows]
    uq = core.deterministic_uniform_quotas(); aq, weights = core.proportional_quotas(fit_bins)
    identity = core.LowRankMetric(seed=P.SEED); identity.project();
    sup_u, ids_u = core.select_support(fit_bins, identity, uq)
    sup_a, ids_a = core.select_support(fit_bins, identity, aq)
    ident_metric = core.LowRankMetric(seed=P.SEED)
    arms = {"A": (fit_bins, ident_metric), "B-uniform": (sup_u, ident_metric),
            "B-adaptive": (sup_a, ident_metric)}
    # Metrics use matched parameter count, optimizer and schedule.
    mf = core.LowRankMetric(seed=P.SEED); sf, tr_f = train_metric(mf, fit_bins, hold_bins, aq, "fixed", P.SEED+3, device)
    mn = core.LowRankMetric(seed=P.SEED); sn, tr_n = train_metric(mn, fit_bins, hold_bins, aq, "nested", P.SEED+3, device)
    arms.update({"C-fixedS": (sf, mf), "C-NCSL": (sn, mn)})
    # A uses full identity teacher and the other arms are scores under their metric.
    all_scores, wit_scores = {}, {}
    for name, (support, metric) in arms.items():
        all_scores[name] = score_grid(metric, hold, support, positions, device)
        wit_scores[name] = score_grid(metric, witnesses, support, positions, device)
    teacher_i = score_grid(ident_metric, hold, fit_bins, positions, device)
    teacher_w = score_grid(ident_metric, witnesses, fit_bins, positions, device)
    report = {"boundary": t, "fit_images": len(fit_idx), "heldout_images": len(hold_idx),
              "fit_events": [int(updates[i]["event"]) for i in fit_idx],
              "heldout_events": [int(updates[i]["event"]) for i in hold_idx],
              "eligible_witnesses": [x["relative_path"] for x in witness_rows],
              "quotas_uniform": uq.tolist(), "quotas_adaptive": aq.tolist(),
              "complexity_weights": weights.tolist(), "arms": {},
              "training": {"C-fixedS": tr_f, "C-NCSL": tr_n},
              "byte_ledger": core.byte_ledger(), "device": device}
    for name in arms:
        report["arms"][name] = {
            "heldout": metrics_for_arm(all_scores[name], teacher_i, hold_cats),
            "witness": metrics_for_arm(wit_scores[name], teacher_w, wit_cats),
            "heldout_auxiliary_vs_identity_teacher": arm_auxiliary(all_scores[name], teacher_i),
            "score_hash": hashlib.sha256(all_scores[name].tobytes()).hexdigest(),
        }
    # Learned full-teacher drift relative to immutable identity teacher.
    for name in ("C-fixedS", "C-NCSL"):
        metric = arms[name][1]
        learned_teacher = full_teacher_grid(metric, hold, fit_bins, positions, device)
        delta = learned_teacher - teacher_i
        report["arms"][name]["full_teacher_identity_drift"] = {
            "mean_abs": float(np.abs(delta).mean()), "p95_abs": float(np.quantile(np.abs(delta),.95)),
            "max_abs": float(np.abs(delta).max()),
            "spectral_norm": float(metric.spectral_norm().item()),
            "singular_range_W": [float(x) for x in torch.linalg.svdvals(torch.eye(768)+metric.update_matrix()).detach().cpu()[[0,-1]]],
        }
    # Paired whole-image uncertainty for C-NCSL against B-adaptive and C-fixedS.
    def image_diff(a, b):
        # Paired whole-image uncertainty is on the preregistered objective:
        # mean patch selection excess per image, stratified by category.
        cat_mask = np.asarray(hold_cats)
        ea = np.maximum(all_scores[a] - teacher_i, 0.0).mean(axis=1)
        eb = np.maximum(all_scores[b] - teacher_i, 0.0).mean(axis=1)
        return {cat: (ea[cat_mask == cat] - eb[cat_mask == cat])
                for cat in sorted(set(hold_cats))}
    report["paired_image_bootstrap"] = {
        "C-NCSL_minus_B-adaptive": bootstrap_ci(image_diff("C-NCSL", "B-adaptive")),
        "C-NCSL_minus_C-fixedS": bootstrap_ci(image_diff("C-NCSL", "C-fixedS")),
    }
    report["support_identity_overlap"] = {}
    support_ids = {"B-uniform": ids_u, "B-adaptive": ids_a,
                   "C-fixedS": [np.asarray(x) for x in tr_f["final_support_ids"]],
                   "C-NCSL": [np.asarray(x) for x in tr_n["final_support_ids"]]}
    for name, val in support_ids.items():
        report["support_identity_overlap"][name] = float(np.mean([
            len(set(np.asarray(ids_a[b]).tolist()) & set(np.asarray(val[b]).tolist())) / len(ids_a[b])
            for b in range(core.BINS)]))
    # The serialized score file is reopened before sealing; this is a real
    # persistence parity check rather than a hash of an in-memory object.
    # Provenance is local fit-image event/patch identities; no data payload copied.
    report["support_provenance"] = {"B-uniform": [x.tolist() for x in ids_u],
                                    "B-adaptive": [x.tolist() for x in ids_a],
                                    "C-fixedS": tr_f["final_support_ids"],
                                    "C-NCSL": tr_n["final_support_ids"]}
    np.savez_compressed(outdir / f"scores_t{t}.npz", **{f"{k}_held":v for k,v in all_scores.items()},
                        teacher_identity=teacher_i, **{f"{k}_witness":v for k,v in wit_scores.items()},
                        teacher_witness=teacher_w)
    reopened = np.load(outdir / f"scores_t{t}.npz")
    report["save_reload_scoring_parity"] = bool(np.array_equal(reopened["C-NCSL_held"], all_scores["C-NCSL"]))
    report["immutable_artifact_hashes"] = {
        f"scores_t{t}.npz": hashlib.sha256((outdir / f"scores_t{t}.npz").read_bytes()).hexdigest()
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--boundaries", default="350")
    args = ap.parse_args(); device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no fallback is implicit")
    root = P.ROOT / "results/hope_cad/ncsl" / args.run_id; root.mkdir(parents=True, exist_ok=False)
    data, cache, cache_checks = P.load_inputs(True)
    lock = {"schema":"ncsl_stage0_g0_v1", "config_hash":P.config_hash(), "protocol_sha256":P.digest(P.P0C),
            "checkpoint_sha256":P.CHECKPOINT_SHA256, "cache_sha256":data["cache_sha256"],
            "source_canonical_digest":data["source_canonical_digest"], "seed":P.SEED,
            "metric":"W=I+UV^T, dim=768, rank=32", "init":"U seeded QR nonzero (0.1), V=0",
            "spectral_bound":core.GAMMA, "preservation":"hard bounded distortion only",
            "tau":P.TAU, "lr":P.LR, "epochs":P.EPOCHS, "reselect_every":P.RESELECT_EVERY,
            "fit_split":"image-level deterministic 80/20 within causal prefix",
            "arms":["A","B-uniform","B-adaptive","C-fixedS","C-NCSL"],
            "anomaly_access":False, "cache_checks":cache_checks,
            "source_hashes": {str(p.relative_to(P.ROOT)): P.digest(p) for p in
                              [P.ROOT/"exps/ncsl/core.py", P.ROOT/"exps/ncsl/protocol.py",
                               P.ROOT/"exps/ncsl/run_stage0.py", P.ROOT/"AGENTS.md"]},
            "environment":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.cuda.is_available()}}
    (root/"G0_LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True)+"\n")
    reports = {}; started=time.perf_counter()
    for t in [int(x) for x in args.boundaries.split(",") if x]:
        reports[str(t)] = run_boundary(data, cache, t, device, root)
        (root/f"boundary_{t}.json").write_text(json.dumps(reports[str(t)], indent=2, sort_keys=True)+"\n")
    # Causal old-score drift uses only witness rows already eligible at both
    # boundaries.  No future-category witness is opened at an earlier prefix.
    ordered = sorted(int(x) for x in reports)
    for prev, cur in zip(ordered, ordered[1:]):
        a = np.load(root/f"scores_t{prev}.npz"); b = np.load(root/f"scores_t{cur}.npz")
        n = min(a["C-NCSL_witness"].shape[0], b["C-NCSL_witness"].shape[0])
        drift = {}
        for arm in ["B-adaptive", "C-fixedS", "C-NCSL"]:
            x = a[f"{arm}_witness"][:n].max(1); y = b[f"{arm}_witness"][:n].max(1)
            drift[arm] = {"n_common_images": int(n), "mean_abs": float(np.abs(y-x).mean()),
                          "max_abs": float(np.abs(y-x).max())}
        reports[str(cur)]["causal_old_score_drift"] = {f"{prev}_to_{cur}": drift}
        (root/f"boundary_{cur}.json").write_text(json.dumps(reports[str(cur)], indent=2, sort_keys=True)+"\n")
    seal = {"schema":"ncsl_stage0_g2_seal_v1", "run_id":args.run_id, "elapsed_seconds":time.perf_counter()-started,
            "boundaries":list(reports), "reports":reports,
            "source_hashes":{"G0_LOCK.json":P.digest(root/"G0_LOCK.json"),
                             "boundary_reports":{str(t):P.digest(root/f"boundary_{t}.json") for t in reports}},
            "anomaly_access":False, "sealed_exclusive":True}
    (root/"G2_SEAL.json").write_text(json.dumps(seal, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"run_id":args.run_id,"boundaries":list(reports),"elapsed_seconds":seal["elapsed_seconds"]}, indent=2))


if __name__ == "__main__": main()

