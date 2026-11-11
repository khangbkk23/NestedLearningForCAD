"""B319 training/evaluation loop: frozen foundation and CTX learned replay."""

import gc
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    staged = path.with_name(path.name + ".partial")
    staged.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(staged, path)


def configure():
    if sys.platform != "linux":
        raise ValueError("Linux GPU scope required")
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for registered port")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


def arm_trial(output, config, arm, deadline):
    global ROOT
    ROOT = Path(config["project_root"])
    torch = configure()
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file, save_file

    from nested_replay_cad.core.data.manifest import (
        checked_path,
        checked_root,
        read_verified_bytes,
    )
    from nested_replay_cad.core.memory.rng import seed_sampling
    from nested_replay_cad.core.memory.sampler import (
        admission_slot,
        sample_reservations,
    )
    from nested_replay_cad.model import decoder as official
    from nested_replay_cad.model import encoder
    from nested_replay_cad.model import scoring as readout

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="FAILED",
        arm=arm,
        seed=config["seed"],
        committed=0,
        snapshots=[],
        counters=dict(encoder=0, support_forwards=0, optimizer=0),
        AD_executed=False,
        AD_inputs_enumerated=False,
        new_normal_training=False,
        slow_training=False,
        GPU_executed=True,
        scientific_acceptance=False,
    )
    dataset = checked_root(config["dataset_path"])
    selection = read(ROOT / config["selection_path"])
    full_cohort = read(ROOT / config["cohort_path"])["records"]
    result["AD_inputs_enumerated"] = True
    write(output / "study.json", result)
    store, reservoir, fifo, metas = {}, (), (), {}
    rng = seed_sampling(config["seed"])
    torch.manual_seed(config["seed"])
    service = encoder.load(ROOT / config["encoder"]["path"], device="cuda:0")
    torch.manual_seed(config["seed"])
    decoder = official.Decoder(checkpoint=True)
    official.init_trainable(decoder)
    decoder.to("cuda:0")
    initial_path = output / "initial.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in decoder.state_dict().items()
        },
        str(initial_path),
    )
    result["initial_sha256"] = sha(initial_path)
    optimizer = official.make_optimizer(decoder)
    controller = None
    if arm in ("a0_tail", "a0_tail_ctx"):
        from nested_replay_cad.replay import NestedReplay

        controller = NestedReplay(config["seed"], "tail", normalize_context=(arm == "a0_tail_ctx"))
        result["counters"].update(
            controller_context_forwards=0, slow_query_forwards=0, slow_replay_forwards=0
        )
    # Conv2d initialization inside the scorer consumes RNG once. Complete it
    # before the training stream seed so the observer remains read-only.
    readout.gauss(torch.device("cuda:0"))
    torch.manual_seed(config["seed"] + 1000000)
    result["timing"] = dict(
        encoder_seconds=0.0, observer_seconds=0.0, checkpoint_seconds=0.0
    )
    trace_path = output / "events.jsonl"

    from nested_replay_cad.storage import DiskGuard

    disk_guard = DiskGuard(output.parent.parent)
    encoder_cap = 3629 + sum(
        sum(r["category"] == c for r in full_cohort) * (15 - i)
        for i, c in enumerate(config["categories"])
    )

    def guard():
        disk_guard.check()
        import resource

        if (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            > config["caps"]["group_RSS_bytes"]
        ):
            raise RuntimeError("Host RAM cap exceeded")
        if torch.cuda.max_memory_allocated(0) > 12 * 1024**3:
            raise RuntimeError("Allocated VRAM12GiB cap exceeded")
        if time.monotonic() > deadline - 600:
            raise TimeoutError("Arm compute reserve")
        if (
            result["counters"]["encoder"] > encoder_cap
            or result["counters"]["support_forwards"] > 18145
        ):
            raise RuntimeError("Registered per-arm compute cap")
        if (
            result["counters"].get("controller_context_forwards", 0) > 14516
            or result["counters"].get("slow_query_forwards", 0) > 904
            or result["counters"].get("slow_replay_forwards", 0) > 904
        ):
            raise RuntimeError("Registered controller compute cap")

    def store_result(force=False):
        result["seconds"] = time.monotonic() - started
        if (
            force
            or result["committed"] % 8 == 0
            or result["committed"] in config["snapshots"]
        ):
            write(output / "study.json", result)

    def release_encoder():
        nonlocal service
        from nested_replay_cad.replay import trim_workspace

        service = None
        trim_workspace()
        print("FOUNDATION_SLOW_WORKSPACE", arm, result["committed"] + 1, flush=True)

    def extract(row, *, test=False):
        nonlocal service
        guard()
        torch.cuda.synchronize()
        extraction_started = time.monotonic()
        relative = row["path"] if test else row["relative_path"]
        expected = row["image_sha256"] if test else row["file_sha256"]
        raw = read_verified_bytes(dataset, relative)
        if hashlib.sha256(raw).hexdigest() != expected or len(raw) != row["bytes"]:
            raise ValueError("Image byte identity")
        if service is None:
            random_before = torch.get_rng_state().clone()
            try:
                service = encoder.load(
                    ROOT / config["encoder"]["path"], device="cuda:0"
                )
            finally:
                torch.set_rng_state(random_before)
        value = (
            encoder.preprocess(checked_path(dataset, relative))
            .unsqueeze(0)
            .to("cuda:0")
        )
        with torch.no_grad():
            feature = service.groups(value).detach()[0].contiguous()
        torch.cuda.synchronize()
        result["timing"]["encoder_seconds"] += time.monotonic() - extraction_started
        result["counters"]["encoder"] += 1
        if (
            feature.shape != (2, 789, 768)
            or feature.dtype != torch.float32
            or not torch.isfinite(feature).all()
        ):
            raise ValueError("Grouped frozen features")
        return feature

    def model_hash():
        digest = hashlib.sha256()
        for name, tensor in sorted(decoder.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def observe(cursor):
        seen = [
            category
            for category, boundary in zip(config["categories"], config["snapshots"])
            if boundary <= cursor
        ]
        cohort = [row for row in full_cohort if row["category"] in seen]
        torch.cuda.synchronize()
        observer_started = time.monotonic()
        guard()
        before = model_hash()
        random_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(0).clone()
        decoder.eval()
        if selection["external_records"]:
            raise ValueError("Full-test route has no external-normal observer")
        normal = []
        scores = []
        destination = output / "scores" / f"g{cursor}"
        destination.mkdir(parents=True)
        with torch.no_grad():
            for row in cohort:
                guard()
                result["AD_executed"] = True
                if len(scores) % 32 == 0:
                    print(
                        "FULLTEST_EVAL_PROGRESS",
                        arm,
                        cursor,
                        len(scores),
                        len(cohort),
                        flush=True,
                    )
                feature = extract(row, test=True)
                target, prediction = decoder(feature[None, 0], feature[None, 1])
                pixel, image = readout.dinomaly(target, prediction)
                pixel, image = pixel[0].cpu().contiguous(), image[0].cpu().contiguous()
                if row["image_label"] == 0:
                    gt = torch.zeros((256, 256), dtype=torch.uint8)
                    original_positive = 0
                else:
                    raw = read_verified_bytes(dataset, row["mask_path"])
                    if (
                        hashlib.sha256(raw).hexdigest() != row["mask_sha256"]
                        or len(raw) != row["mask_bytes"]
                    ):
                        raise ValueError("GT byte identity")
                    from io import BytesIO

                    with Image.open(BytesIO(raw)) as mask:
                        original_positive = int(np.count_nonzero(np.asarray(mask)))
                        gt = torch.from_numpy(
                            (
                                np.asarray(
                                    mask.resize((256, 256), Image.Resampling.NEAREST)
                                )
                                != 0
                            )
                            .astype(np.uint8)
                            .copy()
                        )
                    if not original_positive:
                        raise ValueError("Original GT is empty")
                    if bool(gt.any()) != row["mask_survives_nearest256"]:
                        raise ValueError(
                            "GT resizing differs from frozen full-test manifest"
                        )
                if (
                    pixel.shape != (256, 256)
                    or image.ndim != 0
                    or not torch.isfinite(pixel).all()
                    or not torch.isfinite(image)
                ):
                    raise ValueError("Score contract")
                path = destination / (row["image_id"] + ".safetensors")
                save_file(dict(pixel=pixel, image=image, gt=gt), str(path))
                scores.append(
                    dict(
                        image_id=row["image_id"],
                        path=path.relative_to(output).as_posix(),
                        sha256=sha(path),
                        bytes=path.stat().st_size,
                        original_positive=original_positive,
                        resized_positive=int(gt.sum()),
                    )
                )
        from nested_replay_cad.metrics import category_metrics

        write(destination / "score_manifest.json", dict(cursor=cursor, scores=scores))
        metrics = {}
        for category in seen:
            items = [row for row in cohort if row["category"] == category]
            metrics[category] = category_metrics(items, destination)
            print(
                "FULLTEST_CATEGORY",
                arm,
                cursor,
                category,
                len(items),
                json.dumps(metrics[category]),
                flush=True,
            )
        gc.collect()
        if (
            before != model_hash()
            or not torch.equal(random_before, torch.get_rng_state())
            or not torch.equal(cuda_rng_before, torch.cuda.get_rng_state(0))
        ):
            raise ValueError("Observer mutated model/RNG")
        decoder.train()
        snapshot = dict(
            cursor=cursor,
            normal=normal,
            scores=scores,
            metrics=metrics,
            initial_metrics=None,
            model_sha256=before,
            state_unchanged=True,
        )
        torch.cuda.synchronize()
        result["timing"]["observer_seconds"] += time.monotonic() - observer_started
        print(
            "FULLTEST_OBSERVER_SECONDS",
            arm,
            cursor,
            time.monotonic() - observer_started,
            flush=True,
        )
        from nested_replay_cad.metrics import memory_report

        snapshot["memory"] = memory_report(
            decoder, optimizer, store, controller, torch, service
        )
        snapshot["cohort_images"] = len(cohort)
        result["snapshots"].append(snapshot)
        store_result(force=True)
        print("FOUNDATION_SNAPSHOT", arm, cursor, json.dumps(metrics), flush=True)
        # Reduce peak disk before resuming the unchanged training loop.
        from nested_replay_cad.storage import archive_tensors

        chunks = archive_tensors(
            destination,
            output.parent,
            output.parent.parent,
            output.parent.name + "_" + arm + "_g" + str(cursor) + "_scores",
            deadline,
        )
        write(output / ("score_archives_g" + str(cursor) + ".json"), chunks)
        snapshot["score_archives"] = chunks["archives"]
        store_result(force=True)

    try:
        for arrival, row in enumerate(selection["records"], 1):
            guard()
            identity = row["opaque_id"]
            if row["arrival"] != arrival or identity in metas:
                raise ValueError("Arrival identity")
            reserved, advanced = sample_reservations(reservoir, fifo, rng)
            support_ids = reserved.R_reserved
            blocked = set((*reserved.Q_recent, *reserved.Q_history))
            if blocked & set(support_ids) or any(
                metas[key]["arrival"] >= arrival for key in support_ids
            ):
                raise ValueError("Support/query causality")
            current = extract(row)
            p = config["hard_mining_final"] * min(
                arrival / config["hard_mining_ramp_arrivals"], 1.0
            )
            torch.cuda.synchronize()
            event_start = time.monotonic()
            controller_event = None
            if controller is not None:
                controller_event = controller.step(
                    decoder,
                    optimizer,
                    identity,
                    current,
                    support_ids,
                    reserved.Q_recent,
                    reserved.Q_history,
                    store,
                    metas,
                    arrival,
                    reserved.eligible,
                    p,
                    release_encoder,
                )
                losses = controller_event["losses"]
                norm = controller_event["gradient_norm"]
                for key, count in controller_event["counters"].items():
                    result["counters"][key] += count
                result["slow_training"] = controller.slow_steps > 0
            else:
                decoder.train()
                optimizer.zero_grad(set_to_none=True)
                records = [(identity, current)] + [
                    (key, store[key]) for key in support_ids
                ]
                losses = []
                for index, (key, feature) in enumerate(records):
                    guard()
                    target, prediction = decoder(feature[None, 0], feature[None, 1])
                    loss = official.global_cosine_hm_percent(
                        target, prediction, p=p, factor=0.1
                    )
                    coefficient = (
                        1.0
                        if len(records) == 1
                        else (0.5 if index == 0 else 0.5 / len(support_ids))
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    losses.append(float(loss.detach()))
                    (coefficient * loss).backward()
                    result["counters"]["support_forwards"] += 1
                    del target, prediction, loss
                norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.1)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite gradient norm")
                optimizer.step()
            torch.cuda.synchronize()
            result["counters"]["optimizer"] += 1
            result["new_normal_training"] = True
            slot, admission_rng = admission_slot(advanced.admission, arrival)
            staged_reservoir = list(reservoir)
            if slot is not None:
                if arrival <= 24:
                    staged_reservoir.append(identity)
                else:
                    staged_reservoir[slot] = identity
            reservoir = tuple(staged_reservoir)
            fifo = (*fifo, identity)[-8:]
            retained = set(reservoir) | set(fifo)
            store[identity] = current
            metas[identity] = dict(arrival=arrival, raw_sha256=row["file_sha256"])
            store = {key: store[key] for key in retained}
            metas = {key: metas[key] for key in retained}
            rng = replace(advanced, admission=admission_rng)
            if len(retained) > 32 or any(
                value.shape != (2, 789, 768) for value in store.values()
            ):
                raise ValueError("Grouped memory bound")
            result["committed"] = arrival
            event = dict(
                u=arrival,
                current_id=identity,
                active_replay=list(support_ids),
                reserved_replay=list(reserved.R_reserved),
                recent=list(reserved.Q_recent),
                history=list(reserved.Q_history),
                reservoir=list(reservoir),
                fifo=list(fifo),
                raw_draws={
                    name: getattr(rng, name).raw_draws
                    for name in ("admission", "recent", "historical", "warmup")
                },
                losses=losses,
                hm_p=p,
                gradient_norm=float(norm),
                seconds=time.monotonic() - event_start,
            )
            if controller_event is not None:
                event["weights"] = controller_event["weights"]
                event["slow"] = controller_event["slow"]
                event["controller_counters"] = controller_event["counters"]
            with trace_path.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            store_result()
            if arrival % 8 == 0:
                print(
                    "FOUNDATION_TRAIN",
                    arm,
                    arrival,
                    "/3629",
                    "support_forwards",
                    result["counters"]["support_forwards"],
                    flush=True,
                )
            if arrival >= config.get("integration_arrivals", 256):
                if arrival in config["snapshots"]:
                    observe(arrival)
                break
            if arrival in config["snapshots"]:
                observe(arrival)
        checkpoint_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        tensors = {
            f"model.{name}": value.detach().cpu().contiguous()
            for name, value in decoder.state_dict().items()
        }
        optimizer_meta = {}
        for name, parameter in decoder.named_parameters():
            state = optimizer.state[parameter]
            optimizer_meta[name] = dict(step=state["step"])
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                tensors[f"optimizer.{name}.{key}"] = state[key].detach().contiguous()
        for key, value in store.items():
            tensors["memory." + key] = value.contiguous()
        tensors["rng.torch"] = torch.get_rng_state()
        tensors["rng.cuda0"] = torch.cuda.get_rng_state(0)
        if controller is not None:
            result["controller"] = controller.preserve(tensors)
            if (
                controller.slow_steps != controller.eligible_commits // 16
                or controller.slow_steps > 226
            ):
                raise ValueError("Final slow clock does not match eligible commits")
        path = output / "final-state.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in tensors.items()
            },
            str(path),
        )
        result["checkpoint"] = dict(
            path=path.name,
            sha256=sha(path),
            bytes=path.stat().st_size,
            reservoir=list(reservoir),
            fifo=list(fifo),
            metas=metas,
            rng=asdict(rng),
            optimizer=optimizer_meta,
            scope="Audit preservation checkpoint; continuation qualification separate",
        )
        result["timing"]["checkpoint_seconds"] = time.monotonic() - checkpoint_started
        result.update(
            peak_VRAM_allocated=torch.cuda.max_memory_allocated(0),
            peak_VRAM_reserved=torch.cuda.max_memory_reserved(0),
        )
        result["status"] = "COMPLETE_REVIEW_PENDING"
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        store_result(force=True)
    return result["status"] == "COMPLETE_REVIEW_PENDING"
