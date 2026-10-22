import copy,hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import torch

from hope_cad.cli import parser
from hope_cad.runtime.identity import ROOT,source_digest,source_files,configure_runtime
from hope_cad.runtime.execution import require_preflight,exclusive_work
from hope_cad.runtime.evidence import export_archive,export_run,restore_archive
from hope_cad.contract import design_digest,digest
from hope_cad.baselines.cadic.bank import CADICPatchCoresetV1,CADICPatchCoresetConfig
from hope_cad.benchmark.stream import _state_equal
from hope_cad.runtime.backend import activation,ORIGINAL_NEAREST

class PortableContracts(unittest.TestCase):
    def test_v5_1_scientific_digest_is_unchanged(self):
        self.assertEqual(design_digest(),'6f6b2c847a815619b6ad5ed899a0859aab1d7b532d62b27ae3217d926695db99')

    def test_consolidated_layout_has_one_active_model_and_pinned_method(self):
        p=json.loads((ROOT/'src/hope_cad/resources/lineage.json').read_text(encoding='utf-8'))
        self.assertFalse(p['scientific_algorithm_changed'])
        self.assertEqual(p['method_id'],'hope_cad_v5_1_causal_scaled_utility')
        self.assertEqual(p['historical_design_digest'],design_digest())
        self.assertTrue((ROOT/'src/hope_cad/models/session.py').is_file())
        for old in ('models','training','dataset','hope_cad','docs/main_spec'):
            self.assertFalse((ROOT/old).exists(),old)

    def test_source_inventory_is_complete_and_has_no_external_runtime_dependency(self):
        files=source_files();self.assertIn('src/hope_cad/cli.py',files)
        self.assertIn('scripts/remote/setup.sh',files)
        self.assertFalse(any(n.startswith(('notebooks/','notebook_support/','continuation_runtime_v1/','.agents/')) for n in files))
        for area in ('src/hope_cad','scripts','tests'):
            for path in (ROOT/area).rglob('*.py'):
                self.assertIn(path.relative_to(ROOT).as_posix(),files)

    def test_outputs_cannot_write_source_or_overlap_inputs(self):
        from hope_cad.runtime.identity import output_directory
        for folder,kwargs in [(ROOT,{}),(ROOT/'src',{}),(ROOT.parent,{}),
            (ROOT/'results/run',dict(data_root=ROOT/'results')),
            (ROOT/'runs/one',dict(checkpoint=ROOT/'runs/one/input.safetensors'))]:
            with self.subTest(path=folder),self.assertRaises(ValueError):output_directory(folder,**kwargs)
        self.assertEqual(output_directory(ROOT/'runs/one'),(ROOT/'runs/one').resolve())

    def test_cannot_use_local_smoke_as_remote_preflight(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'preflight.json').write_text(json.dumps(dict(status='local_synthetic_smoke_passed',remote_acceptance=False)))
            with self.assertRaises(ValueError):require_preflight(t,{},'rows','checkpoint')

    def test_passed_preflight_rejects_changed_source_runtime_checkpoint_and_rows(self):
        rt={'device':'fixture'};binding=dict(source_digest=source_digest(),design_digest=design_digest(),runtime=rt,
            normal_rows_digest='rows',backbone_sha256='checkpoint')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'preflight.json';value=dict(status='remote_preflight_passed_pending_review',remote_acceptance=True,binding=binding,binding_digest=digest(binding))
            p.write_text(json.dumps(value));self.assertEqual(require_preflight(t,rt,'rows','checkpoint'),value)
            for key,new in [('source_digest','changed'),('runtime',{}),('normal_rows_digest','other'),('backbone_sha256','other')]:
                v=copy.deepcopy(value);v['binding'][key]=new;v['binding_digest']=digest(v['binding']);p.write_text(json.dumps(v))
                with self.subTest(field=key),self.assertRaises(ValueError):require_preflight(t,rt,'rows','checkpoint')

    def test_remote_gate_fails_without_gpu_before_asset_access(self):
        from hope_cad.runtime.identity import check_remote_runtime
        with patch('sys.platform','linux'),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(ValueError):check_remote_runtime('cuda:0','NVIDIA GeForce RTX 5090')

    def test_preflight_runtime_failure_exports_evidence_before_raise(self):
        from hope_cad.runtime.preflight import run
        parent=ROOT/'results/portable-fixtures';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as t:
            folder=Path(t)/'failed';config=dict(run_dir=str(folder),device='cuda:0',expected_gpu='fixture',data_root='missing',checkpoint='missing',local_smoke=False)
            with patch('hope_cad.runtime.identity.check_remote_runtime',side_effect=ValueError('fixture runtime mismatch')):
                with self.assertRaises(RuntimeError):run(config)
            with zipfile.ZipFile(folder.with_suffix('.zip')) as z:
                report=json.loads(z.read('preflight.json'))
                self.assertEqual(report['status'],'failed_preserve_evidence')
                self.assertFalse(report['remote_acceptance'])
                self.assertIn('fixture runtime mismatch',report['error'])

    def test_single_writer_guard_never_overwrites_owner(self):
        with tempfile.TemporaryDirectory() as t:
            with exclusive_work(t):
                before=(Path(t)/'.writer.lock').read_bytes()
                with self.assertRaises(FileExistsError):
                    with exclusive_work(t):pass
                self.assertEqual((Path(t)/'.writer.lock').read_bytes(),before)
            self.assertFalse((Path(t)/'.writer.lock').exists())

    def test_export_verifies_bytes_and_does_not_duplicate_manifest_on_reexport(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);root=t/'run';root.mkdir()
            (root/'handoff.json').write_text('{"status":"paused_at_atomic_boundary"}')
            (root/'active.pt').write_bytes(b'opaque-fixture')
            export_run(root,t/'first.zip')
            restore_archive(t/'first.zip',t/'restored',schema='hope_cad_portable_preflight_v1')
            # Existing manifest must not become its own payload when packaging a directory.
            (root/'artifact_manifest.json').write_text('{"old":true}')
            export_run(root,t/'second.zip')
            with zipfile.ZipFile(t/'second.zip') as z:
                self.assertEqual(len(z.namelist()),len(set(z.namelist())))
                self.assertNotIn('artifact_manifest.json',json.loads(z.read('artifact_manifest.json'))['files'])
            self.assertEqual((t/'restored/active.pt').read_bytes(),b'opaque-fixture')

    def test_export_refuses_output_inside_tree_and_active_writer(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'handoff.json').write_text('{}')
            with self.assertRaises(ValueError):export_run(root,root/'self.zip')
            with exclusive_work(root):
                with self.assertRaises(ValueError):export_run(root,root.parent/'outside.zip')

    def test_fixed_shape_operator_scores_state_and_counts_match_reference(self):
        cfg=CADICPatchCoresetConfig(budget=11,dim=4,chunk_size=3,query_chunk_size=4,pair_chunk_size=4)
        gen=torch.Generator().manual_seed(71);xs=[torch.randn(5,4,generator=gen) for _ in range(6)]
        a=CADICPatchCoresetV1(cfg);b=CADICPatchCoresetV1(cfg)
        for x in xs:
            reference=a.update(x)
            with activation(True):optimized=b.update(x)
            self.assertEqual(reference,optimized);self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
        reference=a.pixel_scores(xs[0][None])
        with activation(True):optimized=b.pixel_scores(xs[0][None])
        self.assertTrue(_state_equal(reference,optimized))
        self.assertIs(CADICPatchCoresetV1._nearest,ORIGINAL_NEAREST)

    def test_unknown_nn_operator_is_not_silently_overridden(self):
        with patch.object(CADICPatchCoresetV1,'_nearest',lambda *args:None):
            with self.assertRaises(RuntimeError):
                with activation(True):pass

    def test_source_shipping_roundtrip_is_self_contained(self):
        from hope_cad.runtime.shipping import package
        with tempfile.TemporaryDirectory() as t:
            target=Path(t)/'source.zip';receipt=package(target)
            with zipfile.ZipFile(target) as z:
                manifest=json.loads(z.read('bundle_manifest.json'))
                self.assertEqual(receipt['files'],len(manifest['files']))
                self.assertEqual(set(z.namelist()),set(manifest['files'])|{'bundle_manifest.json'})
                for name,expected in manifest['files'].items():self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),expected)
            with self.assertRaises(ValueError):package(target)

    def test_portable_commands_require_explicit_inputs(self):
        for name in ('train','evaluate','matrix'):
            with self.subTest(command=name):
                v=parser().parse_args([name,'--job','NESTED_seed0','--run-dir','run','--data-root','data','--checkpoint','weights','--preflight','gate'])
                self.assertEqual(v.command,name)

if __name__=='__main__':unittest.main()
