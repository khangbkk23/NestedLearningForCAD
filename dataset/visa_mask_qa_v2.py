"""Evaluator-only mask QA. Loss of tiny resized defects is recorded, never relabeled."""
from pathlib import Path
import numpy as np
from PIL import Image
from .benchmark_protocol_v2 import safe_path, file_hash


def inspect_mask(root, row, size=224):
    image_path = safe_path(root, row['relative_path'])
    with Image.open(image_path) as image:
        geometry = image.size
    report = dict(identity=row['relative_path'], image_sha256=file_hash(image_path),
                  image_label=row['label'], geometry=list(geometry), original_positive_pixels=0,
                  resized_positive_pixels=0, emptied_after_resize=False)
    if not row['label']:
        return report
    path = safe_path(root, row['mask_path'])
    with Image.open(path) as source:
        if source.size != geometry:
            raise ValueError('mask/image geometry mismatch')
        raw = np.asarray(source)
        if raw.ndim == 3:
            if raw.shape[2] != 3 or not np.all(raw == raw[..., :1]):
                raise ValueError('mask channels disagree or unsupported channel encoding')
            raw = raw[..., 0]
        if raw.ndim != 2 or raw.dtype.kind not in 'bui':
            raise ValueError('unsupported mask encoding')
        binary = raw != 0
        if not binary.any():
            raise ValueError('original anomalous mask is empty')
        resized = np.asarray(Image.fromarray(binary.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)) != 0
    return dict(**{k: v for k, v in report.items() if k not in ('original_positive_pixels','resized_positive_pixels','emptied_after_resize')},
                mask_sha256=file_hash(path), original_positive_pixels=int(binary.sum()),
                resized_positive_pixels=int(resized.sum()), emptied_after_resize=not bool(resized.any()),
                encoding_values=np.unique(raw).tolist())


def mask_qa(protocol):
    tasks = {}
    for i, task in enumerate(protocol.task_names):
        rows = [inspect_mask(protocol.root, row, protocol.image_size) for row in protocol.test_paths(i)]
        positives = sum(r['resized_positive_pixels'] for r in rows)
        tasks[task] = dict(images=rows, resized_positive_pixels=positives,
                           emptied_identities=[r['identity'] for r in rows if r['emptied_after_resize']],
                           pixel_metric_defined=positives > 0)
    return dict(schema='visa_mask_qa_v2', policy='direct_square224_nearest_nonzero_keep_image_label',
                tasks=tasks, co_primary_complete=all(r['pixel_metric_defined'] for r in tasks.values()))
