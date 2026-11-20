"""Synchronized warmed resource timing, with every sample retained; no ML imports."""
import math
import time


def warmed_timing(fn, synchronize, *, warmups=1, repeats=3, clock=time.monotonic):
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (warmups, repeats)):
        raise ValueError("Positive warmup and measurement counts required")
    cold, samples = [], []
    for target, count in ((cold, warmups), (samples, repeats)):
        for _ in range(count):
            synchronize()
            start = clock()
            fn()
            synchronize()
            seconds = clock() - start
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Invalid synchronized timing sample")
            target.append(seconds)
    return {"warmup_seconds": cold, "samples_seconds": samples,
            "maximum_seconds": max(samples), "rule": "maximum after declared warmup; no sample filtering"}
