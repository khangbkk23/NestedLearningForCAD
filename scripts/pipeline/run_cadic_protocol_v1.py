# scripts/pipeline/run_cadic_protocol_v1.py
"""Deprecated compatibility entry point.

The generic benchmark harness is the only supported CADIC-compatible runner.
Keeping this name avoids breaking old notebooks while preventing the historical
runner from silently using a different evaluator or test protocol.
"""
import argparse
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="legacy config path, retained for a clear error")
    parser.parse_args()
    parser.error("deprecated: use scripts/benchmarks/0_setup_benchmark.py followed by 1_run_benchmark.py")
if __name__ == "__main__":
    main()