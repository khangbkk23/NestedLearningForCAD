"""Evaluator-only exact tied-group ranks. No model or learner imports."""

import math


def exact_numpy(scores, labels):
    import numpy as np

    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    raw_labels = np.asarray(labels).reshape(-1)
    if raw_labels.dtype.kind not in "buif" or not np.isin(raw_labels, (0, 1)).all():
        raise ValueError("Rank labels must be binary before conversion")
    y = raw_labels.astype(np.uint8, copy=False)
    if (
        not len(s)
        or s.shape != y.shape
        or not np.isfinite(s).all()
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError("Rank input contract")
    order = np.argsort(s, kind="stable")[::-1]
    v = s[order]
    p = np.cumsum(y[order], dtype=np.float64)
    del order
    ends = np.r_[np.flatnonzero(v[:-1] != v[1:]), len(v) - 1]
    del v
    tp = p[ends]
    del p
    total = ends.astype(np.float64) + 1
    del ends
    positive = float(tp[-1])
    negative = float(total[-1] - positive)
    if positive <= 0 or negative <= 0:
        raise ValueError("Undefined binary ranking")
    increments = np.diff(np.r_[0.0, tp])
    precision = tp / total
    recall = tp / positive
    ap = float(np.sum((increments / positive) * precision, dtype=np.float64))
    auc = float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, (total - tp) / negative]))
    prauc = float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))
    return dict(
        AP=ap, AUROC=auc, PR_AUC_trapezoid=prauc, pixels=len(s), positives=int(positive)
    )


def category_metrics(items, destination):
    import numpy as np
    from safetensors.numpy import load_file

    pixel = np.empty(len(items) * 256 * 256, dtype=np.float32)
    labels = np.empty(len(pixel), dtype=np.uint8)
    images = np.empty(len(items), dtype=np.float32)
    for i, row in enumerate(items):
        payload = load_file(str(destination / (row["image_id"] + ".safetensors")))
        pixel[i * 65536 : (i + 1) * 65536] = payload["pixel"].reshape(-1)
        gt = payload["gt"].reshape(-1)
        if gt.dtype.kind not in "buif" or not np.isin(gt, (0, 1)).all():
            raise ValueError("Pixel ground truth must be binary before conversion")
        labels[i * 65536 : (i + 1) * 65536] = gt
        images[i] = payload["image"].item()
    p = exact_numpy(pixel, labels)
    del pixel, labels
    image = exact_numpy(images, [r["image_label"] for r in items])
    return dict(pixel=p, image=image)


def memory_report(decoder, optimizer, store, controller, torch, encoder=None):
    import resource

    parts = dict(
        decoder_parameter_elements=sum(p.numel() for p in decoder.parameters()),
        decoder_parameters_bytes=sum(
            p.numel() * p.element_size() for p in decoder.parameters()
        ),
        decoder_buffers=sum(p.numel() * p.element_size() for p in decoder.buffers()),
        replay_unique_records=len(store),
        replay_features=sum(v.numel() * v.element_size() for v in store.values()),
        fast_optimizer_tensors=sum(
            v.numel() * v.element_size()
            for state in optimizer.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        ),
        peak_host_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        GPU_allocated_bytes=torch.cuda.memory_allocated(0),
        GPU_reserved_bytes=torch.cuda.memory_reserved(0),
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
    )
    parts["controller_parameters_bytes"] = (
        0
        if controller is None
        else sum(
            v.numel() * v.element_size() for v in controller.phi.parameters.values()
        )
    )
    parts["controller_slow_optimizer_bytes"] = (
        0
        if controller is None
        else sum(
            v.numel() * v.element_size()
            for group in (controller.adam.m, controller.adam.v)
            for v in group.values()
        )
    )
    parts["encoder_weights_bytes"] = (
        None
        if encoder is None
        else sum(v.numel() * v.element_size() for v in encoder.state_dict().values())
    )
    return parts
