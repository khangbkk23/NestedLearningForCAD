"""Dependency-light verification for NL-CORE-01R."""

from __future__ import annotations

from .forensics import checkpoint_parity, factorization_check, noncollinear_run, stationary_run


def main() -> None:
    check = factorization_check()
    assert check["max_abs_error"] < 1e-12
    stable = stationary_run(0.8, events=1000)
    neutral = stationary_run(1.1, events=1000)
    unstable = stationary_run(1.2, events=1000)
    assert stable["finite"] and neutral["finite"] and unstable["finite"]
    assert stable["max_generator_prediction_error"] < 1e-10
    assert neutral["max_generator_prediction_error"] < 1e-10
    assert unstable["max_generator_prediction_error"] < 1e-8
    assert stable["theory"]["stable"] and unstable["theory"]["stable"] is False
    assert noncollinear_run(events=1000)["finite"]
    parity = checkpoint_parity(events=200)
    assert all(parity.values())
    print("NL-CORE-01R verification: PASS")
    print("factorization=PASS stationary=PASS noncollinear=PASS checkpoint=PASS")


if __name__ == "__main__":
    main()
