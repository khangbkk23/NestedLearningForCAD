import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_matrix(payload: Dict[str, Any], metric: str) -> Dict[str, Dict[str, float]]:
    if "matrix" in payload:
        return {
            str(after_task): {
                str(eval_task): float(score)
                for eval_task, score in row.items()
            }
            for after_task, row in payload["matrix"].items()
        }

    if isinstance(payload, list):
        matrix: Dict[str, Dict[str, float]] = {}
        for record in payload:
            if "forgetting_eval" not in record:
                continue
            after_task = str(record["task_id"])
            matrix[after_task] = {
                str(eval_task): float(metrics.get(metric, 0.0))
                for eval_task, metrics in record["forgetting_eval"].items()
            }
        return matrix

    raise ValueError("Input must be forgetting_matrix.json or task_records.json with forgetting_eval.")


def compute_forgetting(matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    if not matrix:
        return {"forgetting_measure": 0.0, "per_task_forgetting": {}}

    final_task = max(int(k) for k in matrix)
    final_key = str(final_task)
    per_task: Dict[str, float] = {}

    for eval_task in range(final_task):
        eval_key = str(eval_task)
        history = []
        for after_task in range(eval_task, final_task + 1):
            score = matrix.get(str(after_task), {}).get(eval_key)
            if score is not None:
                history.append(float(score))

        final_score = matrix.get(final_key, {}).get(eval_key)
        if not history or final_score is None:
            continue
        per_task[eval_key] = max(0.0, max(history) - float(final_score))

    fm = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return {
        "forgetting_measure": fm,
        "per_task_forgetting": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute standard continual-learning forgetting from R[i][j].")
    parser.add_argument("input_json", type=Path, help="forgetting_matrix.json or task_records.json")
    parser.add_argument("--metric", default="image_auroc")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = _load_json(args.input_json)
    metric = payload.get("metric", args.metric) if isinstance(payload, dict) else args.metric
    matrix = _extract_matrix(payload, metric=metric)
    result = {"metric": metric, "matrix": matrix, **compute_forgetting(matrix)}

    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
