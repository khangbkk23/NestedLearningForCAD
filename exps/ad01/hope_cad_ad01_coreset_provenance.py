# exps/hope_cad_ad01_coreset_provenance.py
"""Recover CADIC coreset provenance for one task by exact float32 matching.

The archived coreset stores only feature vectors, not source coordinates. Because
CADIC inserts training patch vectors verbatim (no averaging or projection), every
bank vector is bit-identical to some training patch. Re-extracting one task's
`train/good` corpus therefore lets us recover, for each bank vector, the exact
source image and the 28x28 patch position it came from.

This is a diagnostic of what the memory *covers*, not a performance claim. It
performs no memory write and reads no test label.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from exps.ad01.hope_cad_ad01_phase0 import ReferenceRun, _protocol, install_paperfaithful_adapter


@torch.no_grad()
def extract_task_corpus(adapter, protocol_obj, task_id: int) -> dict[str, Any]:
    """Extract every train/good patch for one task, keeping (image, position)."""
    loader = protocol_obj.build_train_loader(task_id)
    vectors: list[torch.Tensor] = []
    image_index: list[int] = []
    position: list[int] = []
    relative_paths: list[str] = []
    offset = 0
    for batch in loader:
        features = adapter.extractor.extract_patch_features(batch["images"].to(adapter.device))
        count, patches, dim = features.shape
        vectors.append(features.reshape(-1, dim).contiguous())
        image_index.extend(range(offset, offset + count) for _ in range(patches))
        position.extend(range(patches))
        offset += count
        relative_paths.extend(batch["relative_path"])
    # `image_index`/`position` are built per image; flatten in the same order as
    # `vectors` by repeating each image index `patches` times.
    flat_images = [i for i in range(offset) for _ in range(patches)]
    flat_positions = list(range(patches)) * offset
    return {
        "vectors": torch.cat(vectors, dim=0),
        "image_index": torch.tensor(flat_images, dtype=torch.long),
        "position": torch.tensor(flat_positions, dtype=torch.long),
        "relative_paths": relative_paths,
        "n_images": offset,
        "n_patches": patches,
    }


def match_bank_to_corpus(bank: torch.Tensor, corpus: torch.Tensor, *, tol: float = 1e-3) -> dict[str, Any]:
    """Match bank rows to corpus rows by Euclidean tolerance.

    Bit-exact matching is unreliable here: the archived bank was built on GPU
    while this diagnostic re-extracts on CPU, so genuine members can differ in
    the last float32 bits. A tolerance therefore separates "belongs to this
    task" from "does not", which is the question that matters. Membership is
    reported at several tolerances so the separation can be inspected rather
    than assumed.
    """
    best = torch.full((bank.shape[0],), float("inf"))
    arg = torch.zeros(bank.shape[0], dtype=torch.long)
    for start in range(0, corpus.shape[0], 4096):
        part = corpus[start : start + 4096]
        distances = torch.cdist(bank, part)
        values, indices = distances.min(dim=1)
        update = values < best
        best[update] = values[update]
        arg[update] = indices[update] + start

    thresholds = (1e-6, 1e-4, tol, 1e-1, 1.0)
    return {
        "nearest_corpus_index": arg,
        "nearest_distance": best,
        "tol": tol,
        "members_at_tolerance": int((best <= tol).sum()),
        "members_by_tolerance": {f"{t:g}": int((best <= t).sum()) for t in thresholds},
        "nearest_distance_min": float(best.min()),
        "nearest_distance_median": float(best.median()),
        "nearest_distance_max": float(best.max()),
    }


def provenance_report(run_dir: str | Path, task: str) -> dict[str, Any]:
    ref = ReferenceRun.load(run_dir)
    task_id = ref.protocol["task_order"].index(task)
    adapter = ref.load_adapter("cpu")
    protocol_obj = _protocol(ref)

    bank = adapter.coreset.features.clone()
    corpus = extract_task_corpus(adapter, protocol_obj, task_id)
    match = match_bank_to_corpus(bank, corpus["vectors"])

    member_mask = match["nearest_distance"] <= match["tol"]
    matched = match["nearest_corpus_index"][member_mask]
    positions = corpus["position"][matched]
    images = corpus["image_index"][matched]

    position_counts = Counter(int(p) for p in positions.tolist())
    image_counts = Counter(int(i) for i in images.tolist())
    n_patches = int(corpus["n_patches"])
    rows = n_patches and (corpus["vectors"].shape[0] // n_patches)

    # 28x28 occupancy: how many bank vectors claim each patch position.
    occupancy = torch.zeros(28 * 28, dtype=torch.long)
    for position, count in position_counts.items():
        occupancy[position] = count

    # Number of distinct source images contributing at least one vector.
    distinct_images = len(image_counts)
    top_images = image_counts.most_common(10)

    return {
        "task": task,
        "task_id": task_id,
        "bank_size": int(bank.shape[0]),
        "corpus_images": int(corpus["n_images"]),
        "corpus_patches": int(corpus["vectors"].shape[0]),
        "matched_bank_vectors": int(member_mask.sum()),
        "unmatched_bank_vectors": int((~member_mask).sum()),
        "members_by_tolerance": match["members_by_tolerance"],
        "nearest_distance_min": match["nearest_distance_min"],
        "nearest_distance_median": match["nearest_distance_median"],
        "nearest_distance_max": match["nearest_distance_max"],
        "membership_ratio": float(member_mask.float().mean()),
        "distinct_source_images": distinct_images,
        "source_image_coverage": distinct_images / max(1, int(corpus["n_images"])),
        "position_occupancy_min": int(occupancy.min()),
        "position_occupancy_max": int(occupancy.max()),
        "position_occupancy_mean": float(occupancy.float().mean()),
        "position_occupancy_std": float(occupancy.float().std()),
        "position_occupancy_nonzero": int((occupancy > 0).sum()),
        "position_occupancy_grid": occupancy.reshape(28, 28).tolist(),
        "top_source_images": [
            {"image_index": i, "relative_path": corpus["relative_paths"][i], "count": c}
            for i, c in top_images
        ],
        "config": adapter.coreset.config.__dict__,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CADIC coreset provenance diagnostic")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", default="screw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = provenance_report(args.run_dir, args.task)
    out = Path(args.out)
    if not out.is_absolute():
        out = (Path(__file__).resolve().parents[2] / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.task}_provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    summary = {k: v for k, v in report.items() if k != "position_occupancy_grid"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
