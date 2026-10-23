"""Successor research identities, isolated from the frozen v5.1 resource contract."""
import json
import re
from hope_cad.paths import RESOURCES
from hope_cad.contract import digest, design_digest, file_sha

RESOURCE_SHA256='f2078c85782e5bdc4116fbea5d29bc6cf2d9e944f591f8a1b1531550f9f056ab'

VARIANTS=('CADIC','R_EXACT','R_FO','H32_EXACT','H32_FO','H32_DIRECT','FAST_ONLY','BIAS_VETO','PILOT_RATE_RANDOM_VETO')


def contract():
    if file_sha(RESOURCES/'research.json')!=RESOURCE_SHA256:raise ValueError('Locked research resource changed; register a new revision')
    value=json.loads((RESOURCES/'research.json').read_text(encoding='utf8'))
    if value['base_design_digest']!=design_digest() or value['study']['variants']!=list(VARIANTS):raise ValueError('Research contract/base mismatch')
    return value


def study_digest(): return digest(contract())


def method_identity(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered research variant')
    c=contract()
    return dict(schema='hope_research_method_identity_v1',variant=variant,
        method_id=c['candidate_id'] if variant.startswith('H32') else ('hope_cad_v5_1_fulltrain_v1' if variant.startswith('R_') else 'hope_fulltrain_'+variant.lower()+'_v1'),
        base_design_digest=design_digest(),study_digest=study_digest(),
        learner_schema='hope_hq_learner_v1' if variant in VARIANTS[1:7] else ('scalar_utility_control_v1' if variant!='CADIC' else None),
        session_schema='hope_hq_session_v1')


def job(name, stage=None):
    match=re.fullmatch(r'([A-Z0-9_]+)_seed([012])',name)
    if not match or match[1] not in VARIANTS:raise ValueError('Unregistered research job')
    variant,seed=match[1],int(match[2]);cfg=contract()['study']
    if stage=='pilot' and seed!=0:raise ValueError('Pilot seed is fixed at0')
    if stage=='main' and name not in cfg['C_full_seed0_jobs']:raise ValueError('Only three main jobs registered')
    if stage=='extension' and name not in [v+'_seed0' for v in cfg['D_conditional_remaining_seed0_controls']]+cfg['D_conditional_replicates']:raise ValueError('Unregistered conditional job')
    return variant,seed


def session_for(name,device='cpu',**kwargs):
    from hope_cad.models.research_session import ResearchSession
    variant,seed=job(name)
    return ResearchSession(variant,device=device,seed=seed,**kwargs)
