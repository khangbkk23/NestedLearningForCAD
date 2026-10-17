"""P1 preregistration and once-only causal development experiment (no tuning)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from exps.task_p.p0_normal_support import BUDGET, fingerprint, require, serialized_bytes
from exps.task_p.p1_data import (ROOT, CP, CP_SHA, CATEGORIES, BOUNDARIES, FeatureCache,
                                ArrivalReader, discover, sha, write_json)
from exps.task_p.p1_readout import ARMS, GRID16, new_models, readout, restore

OUT = ROOT / "results/hope_cad/task_p/p1"
P0C = ROOT / "results/hope_cad/task_p/p0c"
CODE = ("exps/task_p/p1_data.py", "exps/task_p/p1_readout.py", "exps/task_p/p1_protocol.py",
        "exps/task_p/analyze_p1.py", "exps/task_p/test_p1.py",
        "exps/task_p/p0_normal_support.py", "models/feature_extractors/cadic_vit_v1.py",
        "dataset/benchmark_protocol_v1.py", "exps/anomaly/hope_anomaly_signal.py")


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def verify_inherited():
    """Opaque file hashes only. Do not map numerical feature caches here."""
    old = json.loads((P0C / "protocol_resolved.json").read_text())
    require(sha(P0C / "protocol_resolved.json") == (P0C / "protocol.sha256").read_text().strip(),
            "P0c protocol digest mismatch")
    expected = {str(CP.relative_to(ROOT)): CP_SHA}
    feature_root = "results/hope_cad/frozen_features_v1/"
    expected.update({feature_root + k: v for k, v in old["cache_sha256"].items()})
    expected.update({feature_root + k: v for k, v in old["input_document_sha256"].items()})
    p0 = json.loads((ROOT / "results/hope_cad/task_p/p0/protocol_resolved.json").read_text())
    expected["exps/task_p/p0_normal_support.py"] = p0["source_code_sha256"]["exps/task_p/p0_normal_support.py"]
    for line in (P0C / "analysis_code.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    ledger = json.loads((P0C / "artifact_ledger.json").read_text())
    # P0c ledger maps artifact-relative filenames to byte identities.
    for name, value in ledger.items():
        if isinstance(value, dict) and "sha256" in value:
            expected[str((P0C / name).relative_to(ROOT))] = value["sha256"]
    checks = {name: {"expected": digest, "actual": sha(ROOT / name)}
              for name, digest in expected.items()}
    metadata_name = "results/hope_cad/task_p/p0c/analysis_code.txt"
    metadata = checks[metadata_name]
    # Existing P0c ledger predates its disclosed analyzer finalization. The text
    # ledger alone has a stale byte hash; all three source hashes inside it are
    # independently checked above, along with every numerical artifact.
    require(metadata["expected"] == "eb42e43422cb8d81034eed5e60c5c8ade164b4461fdc77aac7410053daf307e3"
            and metadata["actual"] == "4665446e4d81765e09ea7369555285f64985a82e4fa97b362864b0be4c576fe6",
            "unexpected P0c source-ledger drift")
    mismatches = {n: v for n, v in checks.items() if v["expected"] != v["actual"]}
    require(set(mismatches) == {metadata_name}, "inherited artifact/source mismatch")
    manifest = json.loads((ROOT / (feature_root + "source_manifest.json")).read_text())
    canonical = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "manifest_digest"},
                                          sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(canonical == old["source_canonical_digest"], "source manifest canonical digest mismatch")
    return {"passed": True, "checks": checks, "canonical_source_digest": canonical,
            "disclosed_preexisting_metadata_mismatch": mismatches,
            "interpretation": "numerical artifacts and executable sources match; analysis_code.txt byte ledger is stale; original untouched"}


def design():
    return {"schema": "task_p_p1_design_v1", "dataset": "MVTec AD v1",
            "exposure": "DEVELOPMENT EVIDENCE; historical official-test and AD-01/02 exposure",
            "authorization_required": "explicit user decision permitting consumed development evaluation",
            "categories": list(CATEGORIES), "boundaries": list(BOUNDARIES),
            "updates": "exact 350-event P0c normal stream; no regenerated normal features",
            "evaluation_rule": "historical first 10 normal + first 10 round-robin anomalies/category; next 10 sealed",
            "sealed_rule": "identity names only; never open, hash, decode, extract or evaluate next-10 payloads",
            "representation": {"checkpoint_sha256": CP_SHA, "shape": [784, 768], "dtype": "float32",
                               "block_index": 8, "normalization": "none", "RGB_resize": "canonical PIL bilinear 224 square"},
            "arms": list(ARMS), "capacities": {ARMS[0]: 2702, ARMS[1]: 2702, ARMS[2]: 2716},
            "budget": "8 MiB including RNG, serials, nearest-other graph, metadata; shared backbone separate",
            "selectors": "unmodified P0 Support STREAM_GREEDY / Algorithm R; reservoir seed 7202",
            "grid": {ARMS[0]: list(GRID16), ARMS[1]: "row-major all 784", ARMS[2]: "row-major all 784"},
            "score": "raw FP64 Euclidean 1-NN with P0 cancellation repair; all 784 queries; max patch image score",
            "map": "FP32 [28,28] -> native H,W; bilinear align_corners=False antialias=False; no smoothing",
            "mask": "native binary >0; permitted values {0,1,255}; no resize; require nonempty defect masks",
            "calibration": "none; no task ID/category scale, normalization, learned transform or threshold",
            "metrics": {"primary": "native pixel AP (non-interpolated AUPR), category then equal-category macro",
                        "secondary": "image AUROC, pixel AUROC, globally pooled pixel AP, prevalence, forgetting",
                        "false_positives": "normal image-score median/p95/max and raw pixel means",
                        "separation": "normal/defective image and normal/defect pixel raw means; no scale fitting"},
            "statistics": {"repetitions": 1000, "seed": 102701,
                           "unit": "whole source images, stratified by category and normal/anomaly label",
                           "pairing": "same image multiplicities across all arms and temporal boundaries",
                           "pixel_bootstrap": "exact point AP; 65536 bins for uncertainty only; harmonic-sum AP bounds over all within-bin rankings; no pixel sampling",
                           "interval": "percentile 95%; fixed categories/order/seed; no familywise discovery claim",
                           "contrasts": "C-B and B-A macro pixel AP; point >=.02 and lower CI>0 improves; CI within +/-.02 matches; otherwise inconclusive; reverse for underperformance"},
            "criteria": {"signal": "one arm: final image AUROC>=.70 and AP>=max(3*prevalence, prevalence+.05) in >=2/3 categories, macro image CI lower>.5 and AP-minus-prevalence CI lower>0",
                         "retention_failure": "an initially useful category (same .70 / AP criterion) loses >.10 absolute image AUROC or pixel AP at any later distinct-task boundary; require paired loss CI lower>0",
                         "failure": "all arms at acquisition: every category image AUROC<=.55 and pixel AP<=prevalence+.02; identifies only joint representation/scorer failure",
                         "otherwise": "P1_INCONCLUSIVE; report mixed findings and each arm separately",
                         "justification": "project feasibility margins, not paper-derived industrial requirements; chance AUROC .5, constant-score AP prevalence; sparse native defects require lift; .02 practical comparison and .10 retention margin fixed without anomaly outcomes"},
            "diagnostics": "no oracle/neural/extra arm; scorer-vs-representation cause remains unresolved absent separate authorized diagnostic",
            "heatmaps": "final highest-score normal and lowest-score defective image/category/arm; raw shared color limits for each image; annotations visualization only",
            "stop": "one run, exclusive start marker, no rescue/tuning; fail closed on provenance, gating or degenerate metrics",
            "source_sha256": {name: sha(ROOT / name) for name in CODE},
            "references": ["https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
                           "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html",
                           "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html",
                           "https://docs.pytorch.org/docs/main/generated/torch.nn.functional.interpolate.html"]}


def prepare(dataset_root, authorization, device):
    require(not (OUT / "protocol_resolved.json").exists(), "protocol exists; refuse overwrite")
    auth = json.loads(Path(authorization).read_text())
    require(auth.get("authorize_exposed_development_only") is True and bool(auth.get("user_decision")),
            "BLOCKED_DATA: consumed development authorization unresolved")
    frozen = json.loads((OUT / "protocol_design.json").read_text())
    require(sha(OUT / "protocol_design.json") == (OUT / "protocol_design.sha256").read_text().strip()
            and frozen == design(), "frozen design/source changed")
    verification = verify_inherited()
    updates, evaluations, sealed = discover(dataset_root)
    protocol = dict(frozen, updates=updates, evaluations=evaluations, sealed_identity_paths=sealed,
                    dataset_root=str(Path(dataset_root).resolve()),
                    locked_utc=datetime.now(timezone.utc).isoformat(), authorization=auth,
                    design_sha256=sha(OUT / "protocol_design.json"), device=device,
                    environment={"python": platform.python_version(), "torch": torch.__version__,
                                 "numpy": np.__version__, "threads": 4},
                    p0c_protocol_sha256=sha(P0C / "protocol_resolved.json"))
    write_json(OUT / "input_verification.json", verification)
    write_json(OUT / "protocol_resolved.json", protocol)
    (OUT / "protocol.sha256").write_text(sha(OUT / "protocol_resolved.json") + "\n")


def load_protocol():
    require((OUT / "protocol_resolved.json").is_file(), "BLOCKED_DATA: no resolved identity protocol")
    digest = sha(OUT / "protocol_resolved.json")
    require(digest == (OUT / "protocol.sha256").read_text().strip(), "protocol mismatch")
    p = json.loads((OUT / "protocol_resolved.json").read_text())
    require(all(sha(ROOT / name) == value for name, value in p["source_sha256"].items()), "code changed after lock")
    return p, digest


def run():
    p, digest = load_protocol()
    verify_inherited()
    write_json(OUT / "run_started.json", {"protocol_sha256": digest,
               "started_utc": datetime.now(timezone.utc).isoformat()})
    reader = ArrivalReader(p, FeatureCache(p["dataset_root"], OUT / "evaluation_features", digest, p["device"]))
    models = new_models()
    trace, records, ledgers = [], [], []
    previous = {a: fingerprint(m.state()) for a, m in models.items()}
    started = time.perf_counter()
    try:
        for row in p["updates"]:
            features = reader.update(row)
            before_features = fingerprint({"features": features})
            for a, m in models.items():
                require(fingerprint(m.state()) == previous[a], "state chain reset")
                positions = list(GRID16) if a == ARMS[0] else list(range(784))
                x = features[positions]
                serials = torch.arange((row["event"] - 1) * len(positions), row["event"] * len(positions))
                if row["event"] == 1:
                    unchanged = fingerprint(m.state())
                    m.update(x, serials, write=False)
                    require(unchanged == fingerprint(m.state()), "no-write mutated state")
                t = time.perf_counter()
                stats = m.update(x, serials)
                elapsed = time.perf_counter() - t
                after = fingerprint(m.state())
                ledger = m.ledger()
                trace.append({"event": row["event"], "arm": a, "source": row["relative_path"],
                              "before": previous[a], "after": after, "update_seconds": elapsed,
                              "input_sha256": before_features, "positions": len(positions),
                              "inserted": stats["inserted"], "replaced": stats["replaced"],
                              "rejected": stats["rejected"], "ledger": ledger})
                previous[a] = after
            require(before_features == fingerprint({"features": features}), "common features mutated")
            if row["event"] not in BOUNDARIES:
                continue
            b = row["event"]
            states = {a: m.state() for a, m in models.items()}
            target = OUT / f"states_t{b}.pt"
            require(not target.exists(), "boundary state exists")
            torch.save(states, target)
            reopened = torch.load(target, weights_only=False)
            historical = torch.load(P0C / f"states_t{b}.pt", weights_only=False)["models"]
            old_names = dict(zip(ARMS, ("GRID16_GREEDY_REFERENCE", "FULL784_GREEDY", "FULL784_RESERVOIR")))
            for a, m in models.items():
                require(fingerprint(reopened[a]) == fingerprint(m.state()), "serialization state mismatch")
                require(fingerprint(historical[old_names[a]]) == fingerprint(m.state()), "P0c state not reproduced bitwise")
                require(serialized_bytes(m.state()) <= BUDGET, "serialized state exceeds reference budget")
                ledgers.append({"boundary": b, "arm": a, **m.ledger(),
                                "serialized_bytes": serialized_bytes(m.state())})
            for item in p["evaluations"]:
                if item["category"] not in reader.admitted:
                    continue
                q, mask = reader.evaluate(item, b)
                for a, m in models.items():
                    t = time.perf_counter()
                    result = readout(m, q, item["hw"])
                    latency = time.perf_counter() - t
                    replay = readout(restore(reopened[a]), q, item["hw"])
                    require(np.array_equal(result["patch_distances"], replay["patch_distances"]),
                            "serialized readout differs")
                    records.append({"boundary": b, "arm": a, "source": item,
                                    "patch_distances": result["patch_distances"].tolist(),
                                    "image_score": result["image_score"], "inference_seconds": latency,
                                    "positive_pixels": int(mask.sum()), "pixels": int(mask.size),
                                    "state_sha256": result["state_sha256"]})
            write_json(OUT / f"predictions_t{b}.json", [r for r in records if r["boundary"] == b])
            print(f"boundary {b}: causal state and {sum(r['boundary'] == b for r in records)} arm/image scores saved", flush=True)
        write_json(OUT / "run_completed.json", {"records": len(records), "protocol_sha256": digest})
    except Exception as exc:
        write_json(OUT / "execution_error.json", {"error": repr(exc), "completed": reader.completed})
        raise
    finally:
        write_json(OUT / "update_trace.json", trace)
        write_json(OUT / "access_log.json", reader.access)
        write_json(OUT / "memory_ledger.json", ledgers)
        write_json(OUT / "resources.json", {"wall_seconds": time.perf_counter() - started,
                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   "extractions": reader.cache.timings, "device": p["device"],
                   "transient_image_feature_bytes": 784 * 768 * 4,
                   "shared_checkpoint_bytes": CP.stat().st_size,
                   "timing_scope": "support readout incl integrity hashes and native upsample; extraction separate"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("design", "verify", "prepare", "run"))
    parser.add_argument("--dataset-root", default=str(ROOT / "data/mvtec"))
    parser.add_argument("--authorization", default=str(OUT / "authorization.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    configure()
    if args.mode == "design":
        write_json(OUT / "protocol_design.json", design())
        (OUT / "protocol_design.sha256").write_text(sha(OUT / "protocol_design.json") + "\n")
    elif args.mode == "verify":
        write_json(OUT / "inherited_verification.json", verify_inherited())
    elif args.mode == "prepare":
        prepare(args.dataset_root, args.authorization, args.device)
    else:
        run()


if __name__ == "__main__":
    main()
