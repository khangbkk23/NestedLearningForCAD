# exps/hope_cad_ad01_phase0.py
"""AD-01 Phase 0 diagnostics: CADIC reference integrity and `screw` failure mode.

Read-only with respect to every archived artifact. This module loads an existing
frozen CADIC state, re-derives its scores through the unmodified benchmark
protocol, and decomposes the image score into its two published factors so the
failure mode can be attributed rather than guessed.

No training, no memory write, no hyperparameter tuning, and no use of test
labels for any choice. Labels are read only to compute the reported metrics.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

# The paper-faithful adapter is the one that produced the authoritative
# 15-task run. It is installed under the canonical module name exactly as
# `scripts/benchmarks/run_cadic_paperfaithful.sh` does.
def install_paperfaithful_adapter():
    sys.modules["models.cadic_benchmark_adapter_v1"] = importlib.import_module(
        "models.cadic_paperfaithful.cadic_benchmark_adapter"
    )
    from models.cadic_benchmark_adapter_v1 import CADICBenchmarkAdapterV1

    return CADICBenchmarkAdapterV1


@dataclass(frozen=True)
class ReferenceRun:
    """An archived benchmark run plus the identities needed to re-derive it."""

    run_dir: Path
    protocol: dict
    method: dict
    manifest: dict
    runmeta: dict

    @classmethod
    def load(cls, run_dir: str | Path) -> "ReferenceRun":
        run = Path(run_dir)
        if not run.is_absolute():
            run = (ROOT / run).resolve()
        for name in ("protocol_resolved.yaml", "method_resolved.yaml", "manifest_train.json", "run.json"):
            if not (run / name).is_file():
                raise FileNotFoundError(f"archived run is missing {name}: {run}")
        return cls(
            run_dir=run,
            protocol=yaml.safe_load((run / "protocol_resolved.yaml").read_text()),
            method=yaml.safe_load((run / "method_resolved.yaml").read_text()),
            manifest=json.loads((run / "manifest_train.json").read_text()),
            runmeta=json.loads((run / "run.json").read_text()),
        )

    def state_path(self, name: str) -> Path:
        path = self.run_dir / "states" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def load_adapter(self, device: str = "cpu"):
        adapter_cls = install_paperfaithful_adapter()
        adapter = adapter_cls(self.method, device)
        state = torch.load(self.state_path("final.pt"), map_location=device, weights_only=False)
        adapter.load_state_dict(state)
        return adapter


def sha256_file(path: Path, *, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def score_task(adapter, protocol_obj, task_id: int, *, b: int) -> dict[str, Any]:
    """Score one official test task and return raw score factors.

    Returns per-image records carrying the published image score together with
    the two factors it multiplies: the worst-patch distance `star_score` and the
    support-density weight `1 - exp(star - logsumexp(support))`. Separating them
    is what allows the failure mode to be attributed to a factor rather than to
    the composite.
    """
    coreset = adapter.coreset
    loader = protocol_obj.build_test_loader(task_id)

    rows: list[dict[str, Any]] = []
    for batch in loader:
        features = adapter.extractor.extract_patch_features(batch["images"].to(adapter.device))
        for index in range(features.shape[0]):
            patches = features[index]
            pixel, indices = coreset._nearest(patches, coreset.features)
            star_row = int(torch.argmax(pixel).item())
            star_score = float(pixel[star_row].item())
            c_star_index = int(indices[star_row].item())
            c_star = coreset.features[c_star_index]
            support_indices = coreset._topk_indices(c_star, min(b, coreset.count))
            support = coreset.features[support_indices]
            support_dist = torch.linalg.vector_norm(support - patches[star_row], dim=1)
            log_den = float(torch.logsumexp(support_dist, dim=0).item())
            weight = 1.0 - float(np.exp(star_score - log_den))

            raw_max = star_score
            raw_mean = float(pixel.mean().item())
            raw_top1pct = float(torch.topk(pixel, max(1, int(0.01 * pixel.numel()))).values.mean().item())

            label = int(batch["labels"][index].item())
            mask = batch["masks"][index]
            pixel_map = torch.nn.functional.interpolate(
                pixel.reshape(1, 1, 28, 28),
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            star_on_defect = None
            if label:
                mask_28 = torch.nn.functional.interpolate(
                    mask[None, None].float(), size=(28, 28), mode="nearest"
                )[0, 0].bool().reshape(-1)
                star_on_defect = bool(mask_28[star_row].item())
            else:
                mask_28 = torch.zeros(784, dtype=torch.bool)

            rows.append(
                dict(
                    relative_path=batch["relative_path"][index],
                    label=label,
                    image_score=weight * star_score,
                    star_score=star_score,
                    weight=weight,
                    log_den=log_den,
                    raw_max=raw_max,
                    raw_mean=raw_mean,
                    raw_top1pct=raw_top1pct,
                    star_row=star_row,
                    support_mean_dist=float(support_dist.mean().item()),
                    star_on_defect=star_on_defect,
                    pixel_map=pixel_map.numpy().astype(np.float32),
                    pixel_map_28=pixel.numpy().astype(np.float32),
                    mask=mask.numpy().astype(bool),
                    mask_28=mask_28.numpy().astype(bool),
                )
            )
    return {"task_id": task_id, "rows": rows}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def avg_precision_pixels(records: Iterable[dict[str, Any]], *, grid: str = "native") -> float:
    """Pixel average precision under a declared score/mask grid.

    ``native`` reproduces the benchmark path: the 28x28 patch score is bilinearly
    upsampled to the image grid and compared against the nearest-resized native
    mask. ``p28`` compares the raw 28x28 patch score against the 28x28
    nearest-downsampled mask, which introduces no interpolation on either side.
    """
    from sklearn.metrics import average_precision_score

    if grid == "native":
        scores = np.concatenate([r["pixel_map"].reshape(-1) for r in records])
        masks = np.concatenate([r["mask"].reshape(-1) for r in records])
    elif grid == "p28":
        scores = np.concatenate([r["pixel_map_28"].reshape(-1) for r in records])
        masks = np.concatenate([r["mask_28"].reshape(-1) for r in records])
    else:
        raise ValueError(f"unknown grid {grid}")
    return float(average_precision_score(masks, scores))


def summarise_task(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["rows"]
    labels = np.array([r["label"] for r in rows])
    def column(name: str) -> np.ndarray:
        return np.array([r[name] for r in rows], dtype=np.float64)

    defect_star = [r["star_on_defect"] for r in rows if r["label"]]
    return {
        "task_id": result["task_id"],
        "n_images": len(rows),
        "n_normal": int((labels == 0).sum()),
        "n_defect": int((labels == 1).sum()),
        "auroc_image_score": auroc(column("image_score"), labels),
        "auroc_star_score": auroc(column("star_score"), labels),
        "auroc_raw_max": auroc(column("raw_max"), labels),
        "auroc_raw_mean": auroc(column("raw_mean"), labels),
        "auroc_raw_top1pct": auroc(column("raw_top1pct"), labels),
        "auroc_weight": auroc(column("weight"), labels),
        "aupr_pixels_current": avg_precision_pixels(rows),
        "aupr_pixels_p28": avg_precision_pixels(rows, grid="p28"),
        "mask28_prevalence": float(
            np.mean(np.concatenate([r["mask_28"].reshape(-1) for r in rows]))
        ),
        "weight_normal_mean": float(column("weight")[labels == 0].mean()),
        "weight_defect_mean": float(column("weight")[labels == 1].mean()),
        "star_normal_mean": float(column("star_score")[labels == 0].mean()),
        "star_defect_mean": float(column("star_score")[labels == 1].mean()),
        "star_on_defect_fraction": (
            float(np.mean([bool(v) for v in defect_star])) if defect_star else None
        ),
    }


def compare_two_reference_banks(run_a: Path, run_b: Path, task_id: int, *, b: int) -> dict[str, Any]:
    """Score one task under two archived banks drawn from the same code lineage.

    Used to test whether the development-study and 15-task CADIC numbers differ
    because of the bank or because of the evaluation protocol.
    """
    out = {}
    for tag, run in (("a", run_a), ("b", run_b)):
        ref = ReferenceRun.load(run)
        adapter = ref.load_adapter("cpu")
        protocol_obj = _protocol(ref)
        result = score_task(adapter, protocol_obj, task_id, b=b)
        out[tag] = {"run_dir": str(ref.run_dir), **summarise_task(result)}
    return out


def _protocol(ref: ReferenceRun):
    from dataset.benchmark_protocol_v1 import MVTecContinualProtocol

    return MVTecContinualProtocol(
        ref.protocol, ref.manifest, ref.method, int(ref.runmeta["seed"]), 0
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AD-01 Phase 0 CADIC diagnostics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", default="screw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ref = ReferenceRun.load(args.run_dir)
    task_id = ref.protocol["task_order"].index(args.task)
    adapter = ref.load_adapter("cpu")
    protocol_obj = _protocol(ref)
    result = score_task(adapter, protocol_obj, task_id, b=int(ref.method["scoring"]["image_neighbors_b"]))
    summary = summarise_task(result)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.task}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    array_fields = ("pixel_map", "pixel_map_28", "mask", "mask_28")
    per_image = [
        {k: v for k, v in row.items() if k not in array_fields}
        for row in result["rows"]
    ]
    (out_dir / f"{args.task}_per_image.json").write_text(
        json.dumps(per_image, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
