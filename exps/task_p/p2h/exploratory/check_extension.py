"""Final pre-lock CPU dispatch and full statistical-decision fixtures."""
import json
import torch
from .design import OUT,MATRIX,POSITIVE,save
from .test_controls import run,semantic_tests
from .analysis import summarize,known_signs


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    results={'semantics':semantic_tests()}
    for arm in POSITIVE:
        results[arm]=run(arm); print('FINAL_CPU',arm,'PASS',flush=True)
    assert known_signs()
    # Clearly named fabricated scalar fixtures test the entire analysis code,
    # never represent generated streams or measured performance.
    for scenario in ['beneficial','harmful']:
        root=OUT/'statistical_fixtures'/scenario
        for regime,arms in MATRIX.items():
            for arm in arms:
                for seed in [1901,1902]:
                    excess=(.8 if scenario=='beneficial' else 1.2) if arm=='candidate' else 1.
                    row=dict(regime=regime,arm=arm,seed=seed,excess=excess,new_normal=1.,
                        traces=[dict(live_sha256='fabricated-equal-forward')]*160,
                        seconds=0.,peak_reserved=0,peak_allocated=0,host_maxrss=0,meter={},witness_loss_forwards=0)
                    save(root/f'{regime}_{seed}_{arm}'/'trace.json',row)
        result=summarize(root,[1901,1902])
        assert result['primary']['benefit']>0 if scenario=='beneficial' else result['primary']['benefit']<0
        expected='MECHANISM_SUPPORTED' if scenario=='beneficial' else 'MECHANISM_NOT_SUPPORTED'
        assert result['verdict']==expected,(scenario,result['verdict'])
        assert result['secondary_family_size']==12
        save(root/'analysis.json',result)
    save(OUT/'extension_tests.json',dict(passed=True,results=results,full_statistical_sign_fixtures=True,G3_untouched=True))


if __name__=='__main__': main()
