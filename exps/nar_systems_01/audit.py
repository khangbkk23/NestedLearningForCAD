"""Recheck the historical audit with its output redirected; never mutate its folder."""
from pathlib import Path
import hashlib
import json
from exps.nar_pilot_00 import full as R

OUT=R.ROOT/'results/hope_cad/nar_systems_01'

def main():
    original=json.loads((R.OUT/'FINAL_AUDIT.json').read_text())
    assert original['all_passed'] and original['count']==1268
    for name,h in original['artifact_sha256'].items():
        assert R.sha(R.OUT/name)==h, name
    script=R.OUT/'AUDIT_SCRIPT.py'
    assert R.sha(script)==original['audit_script_sha256']
    assert not (OUT/'ARTIFACT_RECHECK.json').exists()
    source=script.read_text()
    old="R.write_json(P/'FINAL_AUDIT.json',"
    assert source.count(old)==1
    redirected=source.replace(old,"R.write_json(Path_recheck/'ARTIFACT_RECHECK.json',")
    env={'__file__':str(script),'__name__':'__recheck__','Path_recheck':OUT}
    exec(compile(redirected,str(script),'exec'),env)
    new=json.loads((OUT/'ARTIFACT_RECHECK.json').read_text())
    assert new['checks']==original['checks'] and new['summary']==original['summary']
    R.write_json(OUT/'RECHECK_PROVENANCE.json',{
      'historical_audit_sha256':R.sha(R.OUT/'FINAL_AUDIT.json'),
      'historical_script_sha256':R.sha(script),
      'wrapper_sha256':R.sha(Path(__file__)),
      'only_edit':'redirect FINAL_AUDIT.json write to new ARTIFACT_RECHECK.json',
      'identical_named_checks':True,
      'limitation':'New session source/artifact recheck; not independent algorithm implementation or training replication.'})

if __name__=='__main__': main()
