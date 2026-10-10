# scripts/exps/hope_outer_learning_h50_v1.py
"""Run the predeclared contextual-signal screen."""

from __future__ import annotations

import argparse
import json

from exps.ol04e.hope_outer_learning_h50_v1 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

