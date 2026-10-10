# scripts/exps/hope_outer_learning_horizon_audit_v1.py
"""Run the bounded OL-04D cached teacher/horizon diagnostic."""

from __future__ import annotations

import argparse

from exps.ol04d.hope_outer_learning_horizon_audit_v1 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.device)


if __name__ == "__main__":
    main()
