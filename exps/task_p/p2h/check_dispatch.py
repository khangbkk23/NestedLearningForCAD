"""Unimplemented controls must never silently execute the candidate."""
import json
from .learner import Learner
from .design import OUT,save


def main():
    pending=['immediate_protection','extra_replay','collapsed_clock','single_memory_16',
             'M4_only_16','M16_only_16','post_onset_only','deployed_ablation','unknown']
    rejected=[]
    for arm in pending:
        try: Learner(arm=arm)
        except NotImplementedError: rejected.append(arm)
    assert rejected==pending
    save(OUT/'arm_dispatch_check.json',dict(passed=True,rejected=rejected,
         scope='fail-closed engineering validation only; update equations unchanged',G3_untouched=True))
    print(json.dumps(dict(passed=True,rejected=rejected)),flush=True)


if __name__=='__main__': main()
