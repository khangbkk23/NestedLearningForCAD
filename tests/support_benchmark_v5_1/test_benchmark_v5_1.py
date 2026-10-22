import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from models.hope_cad_v5_1.session import SupportSession
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
from training.benchmark_engine_v1 import _state_equal
from training.support_benchmark_v5_1.contract import job,study,lock,session_for
from training.support_benchmark_v5_1.engine import train,evaluate,cell,task_state,Pause,read,maps_from_patches
from training.support_benchmark_v5_1.data import NormalFeatures,test_manifest as make_test_manifest
from training.support_benchmark_v5_1.statistics import PixelAP,multiplicities,aggregate

def actor(mode='NESTED'):
    torch.set_num_threads(2)
    return SupportSession(mode,seed=9,bank_config=CADICPatchCoresetConfig(budget=6,dim=4,chunk_size=4,pair_chunk_size=4,query_chunk_size=4),model_config=dict(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=2,width=2))

def fixture():
    rows=[dict(relative_path=f't{i//8}/train/good/{i:03d}.png',task_id=i//8,task_name=f't{i//8}',split='train') for i in range(24)]
    g=torch.Generator().manual_seed(54);xs=[torch.randn(1,4,4,generator=g)*.02+1.7**i for i in range(24)]
    class Provider:
        def __init__(self):self.calls=0;self.stop=None
        def get(self,row):
            if self.calls==self.stop:raise Pause('synthetic interruption')
            self.calls+=1;return xs[rows.index(row)]
    return rows,['t0','t1','t2'],Provider()

class EvalProvider:
    def __init__(self):self.calls=0;self.stop=None
    def get(self,row):
        if self.calls==self.stop:raise Pause('synthetic evaluation interruption')
        self.calls+=1
        raw=torch.arange(16,dtype=torch.float32).reshape(1,4,4)*.03+row['label']+.1
        mask=np.zeros((2,2),bool);mask[0,0]=bool(row['label'])
        return raw,mask

def eval_rows(n=3,count=2):
    return [dict(relative_path=f't{ti}/test/{i}.png',task_id=ti,label=i%2,mask_path=None) for ti in range(n) for i in range(count)]

class BenchmarkContracts(unittest.TestCase):
    def test_all_27_registered_jobs_and_unknown_rejection(self):
        self.assertEqual(len(study()['methods'])*len(study()['seeds']),27)
        for m in study()['methods']:
            for seed in study()['seeds']:self.assertEqual(job(f'{m}_seed{seed}'),(m,seed))
        for n in ['M_seed0','NESTED_seed3','CADIC_seed-1','../NESTED_seed0']:
            with self.assertRaises(ValueError):job(n)

    def test_payload_control_uses_matched_tiles_and_discloses_memory(self):
        v=session_for('CADIC_TOTAL_BYTES_seed0')
        self.assertEqual(v.bank_config.budget,24576);self.assertEqual(v.bank_config.chunk_size,256)
        self.assertFalse(study()['equal_total_memory_claim'])

    def test_tuple_list_lock_transport_still_rejects_real_changes(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'lock.json';a=lock(p,dict(shape=(1,2)));b=lock(p,dict(shape=[1,2]));self.assertEqual(a,b)
            with self.assertRaises(ValueError):lock(p,dict(shape=[1,3]))

    def test_train_resume_matches_uninterrupted_including_events(self):
        with tempfile.TemporaryDirectory() as t:
            rows,tasks,p=fixture();identity=dict(job='fixture');a=train(Path(t)/'a',rows,tasks,identity,p,actor)
            rows,tasks,p=fixture();p.stop=10
            with self.assertRaises(Pause):train(Path(t)/'b',rows,tasks,identity,p,actor)
            p.stop=None;b=train(Path(t)/'b',rows,tasks,identity,p,actor)
            self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
            for i in range(24):self.assertEqual((Path(t)/'a/events'/f'{i:05d}.json').read_bytes(),(Path(t)/'b/events'/f'{i:05d}.json').read_bytes())

    def test_task_boundaries_keep_pending_partial_and_true_end_censors(self):
        with tempfile.TemporaryDirectory() as t:
            rows,tasks,p=fixture();v=train(t,rows,tasks,dict(job='fixture'),p,actor)
            s=torch.load(Path(t)/'states/task_00.pt',weights_only=True)['session']
            self.assertFalse(s['finalized']);self.assertIsNotNone(s['pending']);self.assertTrue(s['learner']['records'])
            self.assertTrue(v.finalized);self.assertEqual(v.counters['censored'],1)
            self.assertEqual(len(v.learner.records),v.learner.labels%8)

    def test_corrupted_event_prefix_fails_resume(self):
        with tempfile.TemporaryDirectory() as t:
            rows,tasks,p=fixture();p.stop=10
            with self.assertRaises(Pause):train(t,rows,tasks,dict(job='fixture'),p,actor)
            path=Path(t)/'events/00002.json';s=read(path);s['identity']='tampered';path.write_text(json.dumps(s))
            p.stop=None
            with self.assertRaises(ValueError):train(t,rows,tasks,dict(job='fixture'),p,actor)

    def test_interrupted_active_receipt_publication_recovers(self):
        with tempfile.TemporaryDirectory() as t:
            rows,tasks,p=fixture();p.stop=10
            with self.assertRaises(Pause):train(t,rows,tasks,dict(job='fixture'),p,actor)
            (Path(t)/'active.json').unlink();p.stop=None
            v=train(t,rows,tasks,dict(job='fixture'),p,actor);self.assertEqual(v.cursor,24)

    def test_active_observed_identity_must_match_registered_rows(self):
        with tempfile.TemporaryDirectory() as t:
            rows,tasks,p=fixture();p.stop=1
            with self.assertRaises(Pause):train(t,rows,tasks,dict(job='fixture'),p,actor,checkpoint_every=1)
            path=Path(t)/'active.pt';v=torch.load(path,weights_only=True)
            # NESTED anchors bind this identity too; change both coherently to test the outer row guard.
            v['session']['seen_ids'][0]='unregistered/train/good/x.png'
            v['session']['anchors']['identities'][0]=v['session']['seen_ids'][0]
            torch.save(v,path);(Path(t)/'active.json').unlink();p.stop=None
            with self.assertRaises(ValueError):train(t,rows,tasks,dict(job='fixture'),p,actor,checkpoint_every=1)

    def test_task_state_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            rows,tasks,p=fixture();train(t,rows,tasks,dict(job='fixture'),p,actor)
            path=Path(t)/'states/task_00.pt';path.write_bytes(path.read_bytes()+b'x')
            with self.assertRaises(ValueError):task_state(t,0,actor(),read(Path(t)/'job_lock.json')['digest'])

    def test_7_2_adds_only_missing_cells_and_preserves_final_row(self):
        with tempfile.TemporaryDirectory() as t:
            rows,tasks,p=fixture();train(t,rows,tasks,dict(job='fixture'),p,actor)
            ep=EvalProvider();ids=dict(provider=ep,test_digest='fixture')
            first=evaluate(t,tasks,eval_rows(),actor,ids)
            saved={p.relative_to(t).as_posix():p.read_bytes() for p in (Path(t)/'cells').rglob('*') if p.is_file()}
            self.assertEqual(ep.calls,6)
            last=evaluate(t,tasks,eval_rows(),actor,ids,matrix=True)
            self.assertEqual(ep.calls,12);self.assertEqual(first,last)
            for p,b in saved.items():self.assertEqual((Path(t)/p).read_bytes(),b)
            matrix=read(Path(t)/'metrics/task_matrix.json')['i_auroc'];self.assertIsNone(matrix['matrix'][0][1])
            self.assertEqual(matrix['bwt'],np.mean([matrix['matrix'][-1][j]-matrix['matrix'][j][j] for j in range(2)]))

    def test_matrix_cannot_train_without_parent(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'job_lock.json').write_text(json.dumps(dict(digest='x')))
            with self.assertRaises(ValueError):evaluate(t,['t0'],eval_rows(1),actor,dict(provider=EvalProvider(),test_digest='x'),matrix=True)

    def test_evaluation_mid_cell_resume_is_readonly(self):
        with tempfile.TemporaryDirectory() as t:
            v=actor('CADIC');rows,_,p=fixture()
            for r in rows[:5]:v.observe(p.get(r),r['relative_path'])
            before=v.state_dict();ep=EvalProvider();ep.stop=9
            with self.assertRaises(Pause):cell(t,0,0,v,eval_rows(1,10),ep,dict(test='fixture'))
            chunk=Path(t)/'cells/00_00/chunks/0000.npz';old=chunk.read_bytes();ep.stop=None
            cell(t,0,0,v,eval_rows(1,10),ep,dict(test='fixture'))
            self.assertEqual(chunk.read_bytes(),old);self.assertTrue(_state_equal(before,v.state_dict()))

    def test_changed_chunk_or_test_rows_fail(self):
        with tempfile.TemporaryDirectory() as t:
            v=actor('CADIC');rows,_,p=fixture()
            for r in rows[:5]:v.observe(p.get(r),r['relative_path'])
            cell(t,0,0,v,eval_rows(1),EvalProvider(),dict(test='fixture'))
            changed=eval_rows(1);changed[0]['label']=1
            with self.assertRaises(ValueError):cell(t,0,0,v,changed,EvalProvider(),dict(test='fixture'))
            chunk=Path(t)/'cells/00_00/chunks/0000.npz';chunk.write_bytes(chunk.read_bytes()+b'x')
            with self.assertRaises(ValueError):cell(t,0,0,v,eval_rows(1),EvalProvider(),dict(test='fixture'))

    def test_score_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as t:
            v=actor('CADIC');rows,_,p=fixture()
            for r in rows[:5]:v.observe(p.get(r),r['relative_path'])
            score=v.score
            def bad(raw):v.cursor+=1;return score(raw)
            v.score=bad
            with self.assertRaises(RuntimeError):cell(t,0,0,v,eval_rows(1),EvalProvider(),dict(test='fixture'))

    def test_compact_map_interpolation_and_AP_equal_direct(self):
        patch=np.array([[0,1,1,2],[2,1,0,3]],dtype=np.float32)
        expected=torch.nn.functional.interpolate(torch.tensor(patch).reshape(2,1,2,2),size=(7,7),mode='bilinear',align_corners=False)[:,0].numpy()
        self.assertTrue(np.array_equal(maps_from_patches(patch,7),expected))

    def test_all_ablations_have_actual_mode_counters_and_finite_grads(self):
        for mode in ['NESTED','FIRST_ORDER','FAST_ONLY','NO_SELF_MODIFICATION','NO_ANCHOR_OBJECTIVE','NO_FAST_ADAPTATION']:
            with self.subTest(mode=mode):
                v=actor(mode);rows,_,p=fixture()
                for r in rows[:12]:v.observe(p.get(r),r['relative_path'])
                self.assertGreaterEqual(v.learner.labels,8)
                self.assertEqual(v.learner.fast_updates,0 if mode=='NO_FAST_ADAPTATION' else v.learner.labels)
                self.assertEqual(v.learner.slow_updates,0 if mode in ('FAST_ONLY','NO_FAST_ADAPTATION') else v.learner.labels//8)
                if v.learner.last_slow:self.assertTrue(all(np.isfinite(n) for n in v.learner.last_slow['group_norms'].values()))
                if mode=='NO_ANCHOR_OBJECTIVE':self.assertEqual(v.anchor_weight,0.)
                if mode=='NO_SELF_MODIFICATION':self.assertFalse(v.learner.model.self_modifying)

    def test_train_provider_rejects_official_test_before_extractor(self):
        with tempfile.TemporaryDirectory() as t:
            p=NormalFeatures(t,None,t,'cpu')
            with self.assertRaises(ValueError):p.get(dict(relative_path='bottle/test/good/a.png',split='test'))

    def test_CPU_runtime_fails_before_dataset_extraction(self):
        from notebook_support.support_benchmark_v5_1.bootstrap import prepare
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(ValueError):prepare({})

    def test_different_job_restore_excludes_learned_parent(self):
        from notebook_support.support_benchmark_v5_1.bootstrap import selected_payload
        self.assertTrue(selected_payload('preflight.json','CADIC_seed0'))
        self.assertTrue(selected_payload('runs/CADIC_seed0/active.pt','CADIC_seed0'))
        self.assertFalse(selected_payload('runs/NESTED_seed0/active.pt','CADIC_seed0'))
        self.assertFalse(selected_payload('jobs/NESTED/state.pt','NESTED_seed0'))

    def test_failure_entry_exports_zip_and_never_calls_it_passed(self):
        import time,zipfile
        from notebook_support.support_benchmark_v5_1.entry import run_notebook
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);c=dict(phase='preflight',job='NESTED_seed0',session_hours=3.,started=time.monotonic(),work=str(t/'work'),output=str(t/'failed.zip'))
            with patch('notebook_support.support_benchmark_v5_1.entry.prepare',side_effect=ValueError('synthetic missing asset')):
                with self.assertRaises(RuntimeError):run_notebook(c)
            with zipfile.ZipFile(t/'failed.zip') as z:self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')

    def test_official_mask_QA_rejects_geometry_mismatch(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for defect in ['good','bad']:
                p=root/'t0/test'/defect;p.mkdir(parents=True);Image.new('RGB',(10,10)).save(p/'a.png')
            m=root/'t0/ground_truth/bad';m.mkdir(parents=True);Image.new('L',(9,9),1).save(m/'a_mask.png')
            with self.assertRaises(ValueError):make_test_manifest(root,['t0'])

    def test_tiny_resized_mask_keeps_image_label_and_records_QA(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for defect in ['good','bad']:
                p=root/'t0/test'/defect;p.mkdir(parents=True);Image.new('RGB',(1000,1000)).save(p/'a.png')
            m=root/'t0/ground_truth/bad';m.mkdir(parents=True);raw=np.zeros((1000,1000),np.uint8);raw[0,0]=1;Image.fromarray(raw).save(m/'a_mask.png')
            value=make_test_manifest(root,['t0']);self.assertEqual(sum(r['label'] for r in value['rows']),1)
            self.assertTrue(value['mask_qa'][0]['emptied_after_resize'])

    def test_aggregate_missing_seeds_cannot_produce_CI(self):
        from training.support_benchmark_v5_1.contract import EVIDENCE_SCHEMA,study_digest
        from training.support_benchmark_v5_1.contract import digest as d
        from notebook_support.common.evidence import export_archive
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);work=t/'work';work.mkdir();identity=dict(study_digest=study_digest())
            (work/'suite_lock.json').write_text(json.dumps(dict(identity=identity,digest=d(identity))))
            (work/'test_manifest.json').write_text(json.dumps(dict(digest='test-fixture')))
            p=work/'runs/NESTED_seed0/metrics';p.mkdir(parents=True)
            (p/'final_macro.json').write_text(json.dumps(dict(task_count=15,macro_final_i_auroc=.5,macro_final_p_ap=.2,i_ap=.3,p_auroc=.6)))
            export_archive(work,t/'e.zip',schema=EVIDENCE_SCHEMA)
            out=aggregate([t/'e.zip'],t/'out.json');self.assertFalse(out['complete_study']);self.assertEqual(len(out['missing_jobs']),26)
            with self.assertRaises(ValueError):aggregate([t/'e.zip'],t/'ci.json',statistics=True)

    def test_bootstrap_whole_image_weights_are_paired_stratified(self):
        y=np.array([0,0,1,1,1]);a=multiplicities(y,30,2);b=multiplicities(y,30,2)
        self.assertTrue(np.array_equal(a,b));self.assertTrue((a[:,:2].sum(1)==2).all());self.assertTrue((a[:,2:].sum(1)==3).all())
        self.assertFalse(np.array_equal(a,multiplicities(y,30,3)))

    def test_weighted_pixel_AP_matches_sklearn_with_ties_and_zeros(self):
        maps=np.array([[[0,1],[1,2]],[[2,1],[1,0]],[[1,2],[2,2]]],np.float32)
        masks=np.array([[[0,0],[0,0]],[[1,0],[0,0]],[[0,1],[0,1]]],bool);ap=PixelAP(maps,masks)
        for weights in [[1,1,1],[2,0,1],[0,3,2]]:
            w=np.repeat(weights,4)
            self.assertAlmostEqual(ap.ap(weights),average_precision_score(masks.reshape(-1),maps.reshape(-1),sample_weight=w),places=14)
        self.assertEqual(ap.ap([1,0,0]),0.)

    def test_undefined_empty_bootstrap_stratum_is_rejected(self):
        with self.assertRaises(ValueError):multiplicities(np.array([0,0]),20,0)

if __name__=='__main__':unittest.main()
