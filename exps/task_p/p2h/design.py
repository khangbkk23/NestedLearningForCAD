"""Stable, write-once Design Lock; no torch import or sample generation."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/hope_cad/task_p/p2h'
PYTHON = '/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python'
REPORTS = ['task_p_p2g_research_architecture_decision.md',
           'task_p_p2g_independent_architecture_review.md',
           'task_p_p2g_revision_r1_r15.md', 'task_p_p2g_revision_final_review.md',
           'task_p_p2g_revision_final_v3.md', 'task_p_p2f_independent_audit.md',
           'task_p_p2f_stage_a_corrigendum.md']
REGIMES = {'C':(8,8,8,8,80), 'D8':(8,8,8,8,80),
           'D16':(14,8,8,14,86), 'D32':(32,8,8,0,104), 'R':(8,8,56,8,128)}
ARMS = ['E','zero_memory','candidate','immediate_protection','extra_replay',
        'short_credit','reset_memory','collapsed_clock','single_memory_16',
        'M4_only_16','M16_only_16','disabled_guard','norm_matched_random',
        'post_onset_only','deployed_ablation','relaxed_cap']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(value))


def lock():
    assert not OUT.exists(), 'P2h output exists; no overwrite or rerun'
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    assert branch=='dikhang_space'
    expected={'p2':'be3a0fa0a351f4acf0e087667090d25b3257e5b5cfccc5dac8b0daa3b1ea0fa2',
              'p2d':'182601095371de52627e8a8e2bae26299a3573f5bcf7a63a88f07c762e44d94b',
              'p2e':'18ea5eb104b6cdccce1e89073f3b509d84a583bf37f6ac83212bd5c5fbb6fb75',
              'p2f':'d2bc36f298ef32fc219ec3c2448dd84bce5b95b0073198648a659ae4dd0fe0d5'}
    pins={}
    for phase,digest in expected.items():
        p=ROOT/'results/hope_cad/task_p'/phase/'protocol_resolved.json'
        assert sha(p)==digest
        for n,h in json.loads(p.read_text())['code_sha256'].items():
            assert sha(ROOT/n)==h,n
            pins[n]=h
    preserved={}
    for directory in ['agents/tasks','agents/reports','exps/task_p','exps/r1']:
        for p in sorted((ROOT/directory).rglob('*')):
            rel=str(p.relative_to(ROOT))
            if p.is_file() and p.suffix in ['.md','.py','.txt'] and 'p2h' not in rel:
                preserved[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
    for n in ['AGENTS.md','AGENTS.local.md']:
        p=ROOT/n; preserved[n]={'sha256':sha(p),'bytes':p.stat().st_size}
    for phase in ['p0','p0c','p1','p2','p2d','p2e','p2f']:
        for n in ['protocol_resolved.json','protocol.sha256','artifact_integrity.json',
                  'inheritance.json','verification.json','summary.json']:
            p=ROOT/'results/hope_cad/task_p'/phase/n
            if p.is_file():
                preserved[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size}
    save(OUT/'preservation.json',{'scope':'historical source/reports/aggregate/ledger metadata only; no feature, prediction, mask or sealed payload read','files':preserved})
    config={'schema':'p2h_design_v1','synthetic_only':True,
            'engineering_seeds':list(range(1901,1911)),'G3_seeds':list(range(3101,3111)),
            'regimes_LPMW_endpoint':REGIMES,'arms':ARMS,
            'model':dict(channels=768,latent=64,hidden=128,side=28),
            'mathematics_protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md')),
            'report_sha256':{f'agents/reports/{n}':sha(ROOT/'agents/reports'/n) for n in REPORTS},
            'canonical_source_sha256':pins,'historical_protocol_sha256':expected,
            'G3_untouched':True,'G3_max_GPU_hours':6,'G1b_timeout_s':1800,
            'G2_timeout_s':600,'G2_windows':3,'G2_reserved_GiB':4.5,
            'G2_time_ratio':10.,'benefit_orientation':'reference_excess - candidate_excess',
            'D16_user_corrigendum':dict(L=14,W=14),
            'implementation_policy':'engineering sources created after DL; actual sources frozen at EL; no scientific rule changes'}
    save(OUT/'protocol_design.json',config)
    save(OUT/'design_lock.json',{'schema':'p2h_DL_v1','utc':datetime.now(timezone.utc).isoformat(),
         'repository':str(ROOT),'branch':branch,
         'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'protocol_sha256':sha(OUT/'protocol_design.json'),
         'preservation_sha256':sha(OUT/'preservation.json'),
         'design_source_sha256':sha(__file__),'G3_untouched':True})
    print('DESIGN_LOCK',sha(OUT/'protocol_design.json'),'preserved_files',len(preserved),flush=True)


if __name__=='__main__':
    lock()
