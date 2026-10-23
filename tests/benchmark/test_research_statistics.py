import json
import tempfile
import unittest
from pathlib import Path
from hope_cad.contract import digest,file_sha
from hope_cad.research import method_identity,study_digest
from hope_cad.benchmark.statistics import aggregate


class ResearchStatisticsContracts(unittest.TestCase):
    def fixture(self,path,bad_method=False):
        root=Path(path);source=dict(research_study_digest=study_digest(),runtime='synthetic_CPU_fixture')
        files={'suite_lock.json':dict(identity=source,digest=digest(source)),
            'test_manifest.json':dict(digest='synthetic_test_only'),
            'runs/H32_EXACT_seed0/metrics/final_macro.json':dict(task_count=15,macro_final_i_auroc=.5,macro_final_p_ap=.2,i_ap=.3,p_auroc=.6)}
        ji=dict(job='H32_EXACT_seed0',mode='H32_EXACT',seed=0,scope='fresh_fulltrain_3629',
            method_digest='wrong' if bad_method else digest(method_identity('H32_EXACT')),suite_digest=digest(source))
        files['runs/H32_EXACT_seed0/job_lock.json']=dict(identity=ji,digest=digest(ji))
        files['runs/H32_EXACT_seed0/train_complete.json']=dict(job_digest=digest(ji),images=3629,finalized=True)
        files['runs/H32_EXACT_seed0/final_complete.json']=dict(job_digest=digest(ji),status='final_complete',cell_count=15)
        for ti in range(15):
            ci=dict(job_digest=digest(ji),test_manifest_digest='synthetic_test_only',checkpoint_id=14,task_id=ti,rows=[])
            base=f'runs/H32_EXACT_seed0/cells/14_{ti:02d}/'
            files[base+'cell_lock.json']=dict(identity=ci,digest=digest(ci))
            files[base+'completion.json']=dict(cell_digest=digest(ci))
        for n,v in files.items():
            p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
        (root/'artifact_manifest.json').write_text(json.dumps(dict(schema='hope_hq_fulltrain_evidence_v1',files={n:file_sha(root/n) for n in files})))

    def test_per_job_identity_and_fifteen_job_coverage_and_missing_seeds(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'input';root.mkdir();self.fixture(root)
            out=Path(t)/'summary.json';v=aggregate([root],out,research=True)
            self.assertEqual(v['required_jobs'],15);self.assertFalse(v['complete_study']);self.assertFalse(v['scientific_superiority_validated'])
            with self.assertRaises(ValueError):aggregate([root],out,research=True,statistics=True)
            with self.assertRaises(ValueError):aggregate([root],out) # cannot relabel new schema as old cohort
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root,bad_method=True)
            with self.assertRaises(ValueError):aggregate([root],root/'summary.json',research=True)

    def reseal(self,root):
        manifest=root/'artifact_manifest.json'
        files={p.relative_to(root).as_posix():file_sha(p) for p in root.rglob('*.json') if p!=manifest and p.name!='summary.json'}
        manifest.write_text(json.dumps(dict(schema='hope_hq_fulltrain_evidence_v1',files=files)))

    def test_renamed_seed_and_mode_scope_and_cell_bindings_rejected(self):
        for mutation in ('rename_seed','seed','job','mode','scope','train','final','cell','cell_digest','matrix'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as t:
                root=Path(t);self.fixture(root);folder=root/'runs/H32_EXACT_seed0'
                if mutation=='rename_seed':folder.rename(root/'runs/H32_EXACT_seed1')
                elif mutation in ('seed','job','mode','scope'):
                    p=folder/'job_lock.json';v=json.loads(p.read_text())
                    v['identity'][mutation]={'seed':1,'job':'H32_EXACT_seed1','mode':'R_EXACT','scope':'discarded_pilot_256'}[mutation]
                    v['digest']=digest(v['identity']);p.write_text(json.dumps(v))
                elif mutation in ('train','final'):
                    p=folder/(mutation+'_complete.json');v=json.loads(p.read_text());v['job_digest']='wrong';p.write_text(json.dumps(v))
                elif mutation in ('cell','cell_digest'):
                    p=folder/'cells/14_00/cell_lock.json';v=json.loads(p.read_text());v['identity']['job_digest']='wrong'
                    if mutation=='cell':v['digest']=digest(v['identity'])
                    p.write_text(json.dumps(v))
                    (p.parent/'completion.json').write_text(json.dumps(dict(cell_digest=v['digest'])))
                else:(folder/'matrix_complete.json').write_text(json.dumps(dict(cell_count=120,status='matrix_complete',job_digest='wrong')))
                self.reseal(root)
                with self.assertRaises(ValueError):aggregate([root],root/'summary.json',research=True)
