import copy,json,tempfile,time,unittest,zipfile,tarfile,io
from pathlib import Path
from unittest.mock import patch
import torch
from models.hope_cad_v2.predictor import HOPEFeaturePredictorV4
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.state import tensor_digest
from dataset.hope_mask_views import patch_masks
from scripts.hope_cad_v2.synthetic import feature_fixture
from training.hope_outer_learning_v4 import PrefixFeatures,make_schedule,capture_rng
from training.hope_w7_io import export_archive
from notebook_support.normal_causal import experiments as ex
from notebook_support.normal_causal.contract import event_order,PROTOCOL,digest,legal_rows
from notebook_support.normal_causal.bootstrap import verified_import,normal_members,prepare,discover_resume


def fixture():
    torch.manual_seed(129);torch.set_num_threads(2)
    model=HOPEFeaturePredictorV4(VisualBlockConfig(4,3,4,2,4,4),beta=.01).eval()
    state=model.cms.initial_state()
    # Start with a nonempty slow accumulator: no task-boundary flush may creep in.
    x=torch.randn(1,16,3);target=torch.nn.functional.normalize(torch.randn(1,16,4),dim=-1);masks=patch_masks(4,4)
    for v in range(7):state=model.cms.event(x,target,masks[v%4],model.decoder,state).state
    events=[[(x+image*.01,target,masks[v]) for v in range(4)] for image in range(4)]
    groups={'fit':[e for im in events for e in im],'aux_query':[(x+.2,target,masks[v]) for v in range(4)]}
    return model.requires_grad_(False),state,events,groups

class DiagnosticContracts(unittest.TestCase):
    def test_event_schedule_covers_every_image_view(self):
        for cursor in (0,7,123):
            order=event_order(cursor)
            self.assertEqual(len(order),96)
            for start in range(0,96,16):self.assertEqual(len(set(order[start:start+16])),16)

    def test_ownership_and_parent_unchanged_all_arms(self):
        model,state,events,groups=fixture();before=tensor_digest((dict(model.named_parameters()),));s=ex.state_fingerprint(state)
        with tempfile.TemporaryDirectory() as tmp:
            for arm in PROTOCOL['arms']:
                out=ex.capacity_arm(model,state,groups,events,state.clock,arm,Path(tmp)/arm,identity={'test':True},steps=3,checkpoints=[0,3])
                self.assertEqual(out['cms_change_l2']>0,arm!='decoder_adam')
                self.assertEqual(out['decoder_change_l2']>0,arm in ('decoder_adam','joint_adam'))
                self.assertEqual(out['clock_after'],state.clock+(3 if arm=='deployed_cms' else 0))
                self.assertFalse((Path(tmp)/arm/'active.pt').exists())
        self.assertEqual(before,tensor_digest((dict(model.named_parameters()),)));self.assertEqual(s,ex.state_fingerprint(state))

    def test_deployed_control_matches_direct_events_with_pending(self):
        model,state,events,groups=fixture();expected=state
        for image,view in event_order(state.clock,3):expected=model.cms.event(*events[image][view][:2],events[image][view][2],model.decoder,expected).state
        with tempfile.TemporaryDirectory() as tmp:
            out=ex.capacity_arm(model,state,groups,events,state.clock,'deployed_cms',tmp,identity={},steps=3,checkpoints=[0,3])
            self.assertEqual(out['trace'][-1]['groups'],ex.evaluate(model,expected,groups))

    def test_resume_all_four_arms_matches_uninterrupted(self):
        model,state,events,groups=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            for arm in PROTOCOL['arms']:
                kw=dict(identity={'test':True},steps=4,checkpoints=[0,2,4])
                full=ex.capacity_arm(model,state,groups,events,7,arm,Path(tmp)/(arm+'full'),**kw)
                original=ex.atomic_torch_save
                def pause(value,path):
                    original(value,path)
                    if value['step']==2:raise ex.BudgetPause('fixture')
                with patch.object(ex,'atomic_torch_save',pause):
                    with self.assertRaises(ex.BudgetPause):ex.capacity_arm(model,state,groups,events,7,arm,Path(tmp)/arm,**kw)
                resumed=ex.capacity_arm(model,state,groups,events,7,arm,Path(tmp)/arm,**kw)
                self.assertEqual(full,resumed)

    def test_query_changes_do_not_change_fitted_parameters(self):
        model,state,events,groups=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            a=ex.capacity_arm(model,state,groups,events,7,'joint_adam',Path(tmp)/'a',identity={},steps=2,checkpoints=[0,2])
            changed=copy.deepcopy(groups);changed['aux_query']=[(x,t+3,m) for x,t,m in groups['aux_query']]
            b=ex.capacity_arm(model,state,changed,events,7,'joint_adam',Path(tmp)/'b',identity={},steps=2,checkpoints=[0,2])
            self.assertEqual(a['cms_change_l2'],b['cms_change_l2']);self.assertEqual(a['decoder_change_l2'],b['decoder_change_l2'])
            self.assertNotEqual(a['trace'],b['trace'])

    def test_resume_rejects_real_identity_change(self):
        model,state,events,groups=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            ex.capacity_arm(model,state,groups,events,7,'decoder_adam',tmp,identity={'source':'a'},steps=1,checkpoints=[0,1])
            with self.assertRaises(ValueError):ex.capacity_arm(model,state,groups,events,7,'decoder_adam',tmp,identity={'source':'b'},steps=1,checkpoints=[0,1])

    def test_nonfinite_failure_keeps_preceding_atomic_state_and_parent(self):
        model,state,events,groups=fixture();before=ex.state_fingerprint(state)
        events=copy.deepcopy(events);events[0][3]=(events[0][3][0]*float('nan'),events[0][3][1],events[0][3][2])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((FloatingPointError,RuntimeError)):ex.capacity_arm(model,state,groups,events,7,'joint_adam',tmp,identity={},steps=2,checkpoints=[0,2])
            self.assertEqual(ex.load(Path(tmp)/'active.pt')['step'],0)
        self.assertEqual(before,ex.state_fingerprint(state))

    def test_latent_oracle_resume_and_discard(self):
        model,state,events,groups=fixture();event=events[0][0]
        with tempfile.TemporaryDirectory() as tmp:
            a=ex.latent_oracle(model,state,event,Path(tmp)/'a',identity={},steps=4)
            original=ex.atomic_torch_save
            def pause(v,p):
                original(v,p)
                if v['step']==2:raise ex.BudgetPause()
            with patch.object(ex,'atomic_torch_save',pause):
                with self.assertRaises(ex.BudgetPause):ex.latent_oracle(model,state,event,Path(tmp)/'b',identity={},steps=4)
            b=ex.latent_oracle(model,state,event,Path(tmp)/'b',identity={},steps=4)
            self.assertEqual(a,b);self.assertLessEqual(a['best_found_loss'],a['initial_loss'])
            self.assertFalse(a['lower_bound_certificate']);self.assertFalse((Path(tmp)/'b/active.pt').exists())

    def test_parent_and_rng_unchanged_by_auxiliary_fit(self):
        model,state,events,groups=fixture();rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:ex.capacity_arm(model,state,groups,events,7,'joint_adam',tmp,identity={},steps=2,checkpoints=[0,2])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_continuation_preserves_schedule_optimizer_and_resume(self):
        c=VisualBlockConfig(4,3,4,2,4,4);torch.manual_seed(3);model=HOPEFeaturePredictorV4(c,beta=.01).eval()
        fixture_data=feature_fixture(c);data=PrefixFeatures(fixture_data.manifest,fixture_data.provider)
        optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
        # Nonempty Adam moments, not a restarted optimizer.
        sum(p.square().sum() for p in model.parameters()).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        parent=dict(step=400,max_steps=400,outer_lr=.0001,metadata=model.metadata(),schedule=make_schedule(data,0,steps=400),weights=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),rng=capture_rng(),feature_reads_total=dict(data.reads))
        with tempfile.TemporaryDirectory() as tmp:
            a=ex.continuation(copy.deepcopy(model),PrefixFeatures(data.manifest,data.provider),parent,Path(tmp)/'a',identity={},extra_steps=2,validation_offsets=[0,1,2])
            original=ex.atomic_torch_save
            def pause(v,p):
                original(v,p)
                if v['step']==401:raise ex.BudgetPause()
            with patch.object(ex,'atomic_torch_save',pause):
                with self.assertRaises(ex.BudgetPause):ex.continuation(copy.deepcopy(model),PrefixFeatures(data.manifest,data.provider),parent,Path(tmp)/'b',identity={},extra_steps=2,validation_offsets=[0,1,2])
            saved=ex.load(Path(tmp)/'b/active.pt');self.assertEqual(int(next(iter(saved['optimizer']['state'].values()))['step']),2)
            b=ex.continuation(copy.deepcopy(model),PrefixFeatures(data.manifest,data.provider),parent,Path(tmp)/'b',identity={},extra_steps=2,validation_offsets=[0,1,2])
            self.assertEqual(a['trace'],b['trace']);self.assertEqual(a['reads'],b['reads']);self.assertEqual(a['probe_reads'],0)
            self.assertFalse(a['selected']);self.assertFalse((Path(tmp)/'b/active.pt').exists())
            wrong=copy.deepcopy(parent);wrong['schedule']['episodes'][0]['query']='forbidden'
            with self.assertRaises(ValueError):ex.continuation(copy.deepcopy(model),data,wrong,Path(tmp)/'bad',identity={},extra_steps=1,validation_offsets=[0,1])

class FeatureContracts(unittest.TestCase):
    def test_raw_normal_cache_hashes_and_rejects_changed_image(self):
        from PIL import Image
        from notebook_support.normal_causal.features import NormalFeatures
        from notebook_support.normal_causal.contract import sha
        class Extractor:
            def __init__(self):self.reads=0
            def extract_patch_features(self,image):
                self.reads+=1
                return torch.arange(1,769).float().reshape(1,1,768).expand(1,784,768).clone()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'bottle/train/good/000.png';path.parent.mkdir(parents=True);Image.new('RGB',(224,224),'white').save(path)
            row=dict(relative_path='bottle/train/good/000.png',sha256=sha(path),split='train',task_id=0,task_name='bottle')
            extractor=Extractor();provider=NormalFeatures(root,extractor,root/'cache',root/'receipts','cpu')
            first=provider.row(row,{row['relative_path']:row});self.assertEqual(extractor.reads,5)
            second=provider.row(row,{row['relative_path']:row});self.assertEqual(extractor.reads,5)
            self.assertTrue(torch.equal(first[0][0],second[0][0]));self.assertEqual(len(first),4)
            Image.new('RGB',(224,224),'black').save(path)
            with self.assertRaises(ValueError):provider.row(row,{row['relative_path']:row})

    def test_interrupted_cache_publish_can_regenerate_and_forbidden_row_cannot_read(self):
        from PIL import Image
        from notebook_support.normal_causal.features import NormalFeatures
        from notebook_support.normal_causal.contract import sha
        class Extractor:
            def extract_patch_features(self,image):return torch.ones(1,784,768)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'bottle/train/good/000.png';path.parent.mkdir(parents=True);Image.new('RGB',(224,224),'white').save(path)
            row=dict(relative_path='bottle/train/good/000.png',sha256=sha(path),split='train',task_id=0,task_name='bottle')
            provider=NormalFeatures(root,Extractor(),root/'cache',root/'receipts','cpu');provider.row(row,{row['relative_path']:row})
            next((root/'cache').glob('*.pt')).unlink();provider.row(row,{row['relative_path']:row})
            with self.assertRaises(ValueError):provider.row(row,{})
            tensor=next((root/'cache').glob('*.pt'));torch.save({'teacher':torch.zeros(1)},tensor)
            with self.assertRaises(ValueError):provider.row(row,{row['relative_path']:row})

class LifecycleContracts(unittest.TestCase):
    def test_three_cases_all_arms_oracles_resume_and_completion(self):
        import hashlib
        from notebook_support.normal_causal import runner
        model,state,events,groups=fixture()
        names=[f't{i}' for i in range(15)]
        rows=[dict(task_id=t,task_name=names[t],split='train',relative_path=f't{t}/train/good/{i}.png',sha256=str(i)) for t in range(15) for i in range(8)]
        samples={name:sorted([r for r in rows if r['task_id']==t],key=lambda r:hashlib.sha256(('recovery-normal-v1-20261016|'+r['relative_path']).encode()).hexdigest()) for t,name in enumerate(names)}
        real_capacity,real_oracle=runner.capacity_arm,runner.latent_oracle
        def reduced_arm(*a,**kw):return real_capacity(*a,**kw,steps=2,checkpoints=[0,2])
        def reduced_oracle(*a,**kw):return real_oracle(*a,**kw,steps=2)
        def fake_encoded(model,provider,selected,allowed):
            for row in selected:self.assertIn(row['relative_path'],allowed)
            return [events[0] for _ in selected]
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);ref=work/'parents/recovery';(ref/'reference_w7/runs/M_seed0/states').mkdir(parents=True);(ref/'diagnostics/M_seed0').mkdir(parents=True)
            (ref/'reference_w7/runs/M_seed0/manifest_train.json').write_text(json.dumps(dict(online=rows,task_order=names)))
            (ref/'diagnostics/M_seed0/sample_manifest.json').write_text(json.dumps(dict(samples=samples)))
            for t in [1,7,14]:(ref/f'reference_w7/runs/M_seed0/states/task_{t:02d}.pt').write_bytes(b'fixture-parent')
            (work/'diagnostic_lock.json').write_text(json.dumps({'digest':'fixture'}));(work/'preflight.json').write_text(json.dumps({'status':'passed'}))
            (work/'continuation').mkdir();(work/'continuation/result.json').write_text(json.dumps({'status':'complete','fixture':True}))
            prepared=dict(dataset=tmp,checkpoint='fixture',prefix_cache=str(work/'prefix.pt'))
            with patch.object(runner,'CADICViTFeatureExtractor',return_value=object()),patch.object(runner,'NormalFeatures',return_value=object()),patch.object(runner,'encode_rows',fake_encoded),patch.object(runner,'parent_model',side_effect=lambda p,d:(model,state,7)),patch.object(runner,'capacity_arm',side_effect=reduced_arm),patch.object(runner,'latent_oracle',side_effect=reduced_oracle),patch('builtins.print'):
                self.assertEqual(runner.run(work,work,prepared,'run','cpu',None),'complete_pending_review')
                self.assertEqual(runner.run(work,work,prepared,'run','cpu',None),'complete_pending_review')
            summary=json.loads((work/'summary.json').read_text());self.assertEqual(summary['capacity_fits'],12);self.assertEqual(summary['latent_same_image_fits'],48)
            self.assertFalse(list(work.rglob('active.pt')));self.assertFalse(summary['official_test_accessed']);self.assertFalse(summary['visa_accessed'])

class TransportContracts(unittest.TestCase):
    def test_import_requires_pinned_manifest_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'in').mkdir();(root/'in/x.txt').write_text('good')
            out=export_archive(root/'in',root/'e.zip',schema='fixture');from notebook_support.normal_causal.contract import read_manifest
            pinned=read_manifest(root/'e.zip')[1]
            verified_import(root/'e.zip',root/'ok',pinned,{'x.txt'},'fixture')
            with self.assertRaises(ValueError):verified_import(root/'e.zip',root/'bad','wrong',{'x.txt'},'fixture')
            with self.assertRaises(ValueError):verified_import(root/'e.zip',root/'bad',pinned,{'missing'},'fixture')

    def test_only_train_good_extracted_and_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bio=io.BytesIO()
            with tarfile.open(fileobj=bio,mode='w') as z:
                for name in ['bottle/train/good/000.png','bottle/test/broken/000.png','bottle/ground_truth/broken/000_mask.png']:z.addfile(tarfile.TarInfo(name),io.BytesIO())
            bio.seek(0)
            with tarfile.open(fileobj=bio) as z:self.assertEqual([m.name for m in normal_members(z,tmp)],['bottle/train/good/000.png'])
            bio=io.BytesIO()
            with tarfile.open(fileobj=bio,mode='w') as z:z.addfile(tarfile.TarInfo('../bottle/train/good/0.png'),io.BytesIO())
            bio.seek(0)
            with tarfile.open(fileobj=bio) as z:
                with self.assertRaises(ValueError):normal_members(z,tmp)

    def test_gpu_failure_happens_before_any_parent_import(self):
        with patch('torch.cuda.is_available',return_value=False),patch('notebook_support.normal_causal.bootstrap.find_reviewed',side_effect=AssertionError('must not import')):
            with self.assertRaisesRegex(RuntimeError,'Enable GPU'):prepare({})

    def test_sample_selection_rejects_future_nononline_and_tampering(self):
        from hashlib import sha256
        tasks=['t0','t1'];rows=[dict(task_id=t,task_name=tasks[t],split='train',relative_path=f't{t}/train/good/{i}.png',sha256=str(i)) for t in range(2) for i in range(9)]
        samples={name:sorted([r for r in rows if r['task_id']==t],key=lambda r:sha256(('recovery-normal-v1-20261016|'+r['relative_path']).encode()).hexdigest())[:8] for t,name in enumerate(tasks)}
        train=dict(online=rows,task_order=tasks);sample=dict(samples=samples)
        got=legal_rows(train,sample,1);self.assertEqual(len(got['fit']),4);self.assertEqual(len(got['query']),4)
        broken=copy.deepcopy(sample);broken['samples']['t1'][0]['task_id']=2
        with self.assertRaises(ValueError):legal_rows(train,broken,1)
        broken=copy.deepcopy(train);broken['online']=rows[1:]
        with self.assertRaises(ValueError):legal_rows(broken,sample,1)
        with self.assertRaises(ValueError):legal_rows(train,sample,2)

    def test_failure_is_exported_before_notebook_raises(self):
        from notebook_support.normal_causal.entry import run_notebook
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(work=str(Path(tmp)/'work'),output=str(Path(tmp)/'out.zip'),started=time.monotonic(),phase='run',session_hours=1)
            with patch('notebook_support.normal_causal.entry.prepare',side_effect=ValueError('fixture failure')),patch('notebook_support.normal_causal.entry.traceback.print_exc'),patch('builtins.print'):
                with self.assertRaises(RuntimeError):run_notebook(cfg)
            with zipfile.ZipFile(cfg['output']) as z:self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')

    def test_resume_different_outputs_fail_without_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in (1,2):
                d=root/str(i);d.mkdir();(d/'a.txt').write_text(str(i));export_archive(d,root/f'{i}.zip',schema='hope_normal_causal_evidence_v1')
            with self.assertRaises(ValueError):discover_resume(root)

if __name__=='__main__':unittest.main()
