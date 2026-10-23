"""Fresh canonical train/good stream; no historical prefix partition or test reads."""
from pathlib import Path
from hope_cad.data.prefix import safe_path,file_hash
from hope_cad.contract import digest
from hope_cad.data.protocol import CANONICAL_MVTEC_TASKS


def build_manifest(root,source_identity):
    root=Path(root).resolve();tasks=list(CANONICAL_MVTEC_TASKS);rows=[]
    for tid,task in enumerate(tasks):
        paths=sorted((root/task/'train/good').glob('*.png'))
        if not paths:raise ValueError('Empty full-train category: '+task)
        for p in paths:
            rel=p.relative_to(root).as_posix();checked=safe_path(root,rel)
            rows.append(dict(task_id=tid,task_name=task,split='train',relative_path=rel,sha256=file_hash(checked)))
    if len(rows)!=3629 or sum(r['task_id']==0 for r in rows)!=209 or len({r['relative_path'] for r in rows})!=3629:
        raise ValueError('Full-train requires3629unique normals including209Bottle')
    value=dict(schema='hope_fulltrain_manifest_v1',protocol_id='mvtec_1x15_fulltrain_v7_v1',dataset_name='MVTec AD',
        task_order=tasks,ordering_rule='task_order_then_lexicographic_relative_path',
        source_identity=source_identity,online=rows,excluded=[],
        data_rights='one_image_causal_delivery; observed_completed_context_replay; no offline_calibration')
    value['digest']=digest(value)
    return value,digest(rows)
