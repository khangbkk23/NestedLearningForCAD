"""B322 profile: measured cost and passivity of the shadow path on synthetic features.

Runs the real decoder class, ``Shadows.update`` and ``Shadows.forward`` on one
Linux GPU with random frozen-feature tensors. No dataset, no encoder weights, no
anomaly-detection score. It answers cost and passivity questions only; it says
nothing about whether shadows help (card B322, section 8).
"""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REGISTERED_TAUS = (512, 128)
ARRIVALS = 3629


def median(values):
    return statistics.median(values) if values else None


def main():
    parser = argparse.ArgumentParser(description="B322 shadow cost/passivity profile (synthetic)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--scored", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.steps < args.warmup + 5 or args.scored < args.warmup + 5:
        parser.error("Too few steps/images to measure after warm-up")

    from nested_replay_cad import paths
    from nested_replay_cad.trainer import configure

    paths.process_environment()
    torch = configure()
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats(0)
    from safetensors.torch import save_file

    from nested_replay_cad.model import decoder as official
    from nested_replay_cad.model import scoring
    from nested_replay_cad.shadow import Shadows

    torch.manual_seed(0)
    scoring.gauss(torch.device("cuda:0"))  # scorer init consumes RNG once; do it before the observer
    model = official.Decoder(checkpoint=True)
    official.init_trainable(model)
    model.to("cuda:0")
    optimizer = official.make_optimizer(model)
    shadows = Shadows(torch, model, REGISTERED_TAUS)
    checks = {}
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(1234)

    def feature():
        return torch.randn((2, 789, 768), generator=generator, device="cuda:0")

    def sync():
        torch.cuda.synchronize()
        return time.monotonic()

    # Training steps with the shadow update directly after each optimizer step.
    step_seconds, update_seconds = [], []
    for step in range(1, args.steps + 1):
        item = feature()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = sync()
        target, prediction = model(item[None, 0], item[None, 1])
        loss = official.global_cosine_hm_percent(target, prediction, p=0.9 / 26, factor=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        stepped = sync()
        shadows.update(model)
        updated = sync()
        if step > args.warmup:
            step_seconds.append(stepped - started)
            update_seconds.append(updated - stepped)
        if shadows.step != step:
            raise RuntimeError("Shadow clock differs from the optimizer clock")
        del target, prediction, loss
    checks["finite_parameters"] = all(
        torch.isfinite(t).all().item() for tau in REGISTERED_TAUS for t in shadows.params[tau].values()
    )

    # Observer: passivity and the functional_call path with a checkpoint=True decoder.
    model.eval()
    identity = Shadows(torch, model, [1])
    probe = feature()
    with torch.no_grad():
        expected = model(probe[None, 0], probe[None, 1])
        through = identity.forward(model, 1, probe[None, 0], probe[None, 1])
    def flat(value):
        # The decoder returns nested lists of tensors, not tensors.
        if isinstance(value, torch.Tensor):
            yield value
        else:
            for part in value:
                yield from flat(part)

    left, right = list(flat(expected)), list(flat(through))
    checks["tau1_equals_decoder"] = bool(left) and len(left) == len(right) and all(
        a.shape == b.shape and torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(left, right)
    )
    state_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    shadow_digests = {tau: shadows.digest(tau) for tau in REGISTERED_TAUS}
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    fast_seconds, ema_seconds = [], []
    pixel_bytes = []
    with tempfile.TemporaryDirectory() as folder, torch.no_grad():
        for index in range(args.scored):
            item = feature()
            started = sync()
            target, prediction = model(item[None, 0], item[None, 1])
            pixel, image = scoring.dinomaly(target, prediction)
            pixel, image = pixel[0].cpu().contiguous(), image[0].cpu().contiguous()
            fast = sync() - started
            started = sync()
            for tau in REGISTERED_TAUS:
                target, prediction = shadows.forward(model, tau, item[None, 0], item[None, 1])
                pixel_s, image_s = scoring.dinomaly(target, prediction)
                pixel_s, image_s = pixel_s[0].cpu().contiguous(), image_s[0].cpu().contiguous()
            ema = sync() - started
            if index >= args.warmup:
                fast_seconds.append(fast)
                ema_seconds.append(ema)
            if index == 0:
                path = Path(folder) / "score.safetensors"
                gt = torch.zeros((256, 256), dtype=torch.uint8)
                save_file(dict(pixel=pixel_s, image=image_s, gt=gt), str(path))
                pixel_bytes.append(path.stat().st_size)
        checks["observer_fast_parameters_unchanged"] = all(
            torch.equal(state_before[n], p) for n, p in model.named_parameters()
        )
        checks["observer_shadow_digests_unchanged"] = all(
            shadow_digests[tau] == shadows.digest(tau) for tau in REGISTERED_TAUS
        )
        checks["observer_rng_unchanged"] = bool(
            torch.equal(cpu_rng, torch.get_rng_state())
            and torch.equal(cuda_rng, torch.cuda.get_rng_state(0))
        )
        checks["shadow_differs_from_fast"] = any(
            not torch.equal(shadows.params[512][n], p) for n, p in model.named_parameters()
        )

        # Bytes of the shadow state as the trainer stores it.
        tensors = {f"shadow.ema{tau}.{n}": v.detach().cpu().contiguous()
                   for tau in REGISTERED_TAUS for n, v in shadows.params[tau].items()}
        path = Path(folder) / "shadows.safetensors"
        started = time.monotonic()
        save_file(tensors, str(path))
        shadow_checkpoint = dict(bytes=path.stat().st_size, seconds=time.monotonic() - started)
        fast_tensors = {n: v.detach().cpu().contiguous() for n, v in model.state_dict().items()}
        path = Path(folder) / "decoder.safetensors"
        save_file(fast_tensors, str(path))
        decoder_checkpoint_bytes = path.stat().st_size

    update = median(update_seconds)
    extra = median(ema_seconds)
    result = dict(
        status="PROFILE_COMPLETE" if all(checks.values()) else "PROFILE_CHECK_FAILED",
        experiment="B322",
        scope="synthetic features, real decoder/scorer/Shadows; no dataset, no AD metrics",
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        checks=checks,
        steps_measured=len(step_seconds),
        images_measured=len(fast_seconds),
        seconds=dict(
            train_step_median=median(step_seconds),
            shadow_update_median=update,
            shadow_update_max=max(update_seconds),
            score_fast_median=median(fast_seconds),
            score_two_shadows_median=extra,
        ),
        projections=dict(
            note="Linear extrapolation from medians; synthetic features, one GPU. Not an ETA.",
            shadow_update_total_seconds=update * ARRIVALS,
            extra_seconds_per_1000_scored_images=extra * 1000,
        ),
        bytes=dict(
            decoder_state=decoder_checkpoint_bytes,
            two_shadows=shadow_checkpoint["bytes"],
            score_file_each=pixel_bytes[0],
            shadow_save_seconds=shadow_checkpoint["seconds"],
        ),
        memory=dict(
            peak_VRAM_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    print("B322_PROFILE", json.dumps(result), flush=True)
    return 0 if result["status"] == "PROFILE_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
