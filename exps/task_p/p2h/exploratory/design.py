"""Write-once extension locks, source verification and budget bookkeeping."""
import json
import time
import subprocess
from pathlib import Path
from ..design import ROOT,OUT as ORIGINAL,REGIMES,ARMS,sha,save

OUT=ORIGINAL/'exploratory'
PACKAGE=ROOT/'exps/task_p/p2h/exploratory'
POSITIVE=[a for a in ARMS if a!='relaxed_cap']
DIAGNOSTIC=['E','zero_memory','candidate','reset_memory','norm_matched_random','immediate_protection','post_onset_only']
MATRIX={r:POSITIVE if r in ['D8','D16','R'] else DIAGNOSTIC for r in ['D8','D16','R','C','D32']}


def source_hashes():
    files=list(PACKAGE.glob('*.py'))+list(PACKAGE.glob('*.md'))
    files+=list(PACKAGE.parent.glob('*.py'))+[PACKAGE.parent/'PROTOCOL.md']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def design_lock():
    assert not OUT.exists()
    original=json.loads((ORIGINAL/'verification.json').read_text())
    assert original['passed'] and original['verdict']=='RESOURCE_INFEASIBLE'
    save(OUT/'design_lock.json',dict(schema='p2h_x_DL_v1',utc=time.time(),
        user_authorization='explicit separate exploratory phase; waive only10x runtime',
        original_verdict='RESOURCE_INFEASIBLE',original_protocol=sha(ORIGINAL/'protocol_design.json'),
        original_artifact_manifest=sha(ORIGINAL/'artifact_manifest.json'),
        extension_protocol=sha(PACKAGE/'PROTOCOL.md'),matrix=MATRIX,
        engineering_seeds=list(range(1901,1911)),evaluation_seeds=list(range(3101,3111)),
        GPU_reserved_max_bytes=int(4.5*1024**3),extension_total_seconds=21600,
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        G3_untouched=True))


def validate_execution():
    lock=json.loads((OUT/'execution_lock.json').read_text())
    assert lock['passed'] and lock['G3_untouched_at_lock']
    assert sha(PACKAGE/'PROTOCOL.md')==lock['protocol_sha256']
    assert source_hashes()==lock['source_sha256'],'source changed after Execution Lock'
    assert lock['budget_projection_seconds']<=21600
    return lock


def budget_elapsed():
    # Conservatively count complete CUDA process wall time, including I/O/CPU waits.
    total=0.
    for p in OUT.rglob('cost.json'): total+=json.loads(p.read_text())['seconds']
    return total


if __name__=='__main__': design_lock()
