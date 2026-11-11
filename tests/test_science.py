"""Scoped CPU component checks; not B319/GPU qualification or AD evidence."""

import copy,hashlib,json,os,subprocess,sys

from pathlib import Path

import pytest

import torch

from nested_replay_cad.core.contracts import PhiState

from nested_replay_cad.core.policy.layout import phi_shapes

from nested_replay_cad.core.policy.policy import ReplayPolicy

from nested_replay_cad.core.memory.rng import seed_sampling

from nested_replay_cad.core.memory.sampler import sample_reservations

from nested_replay_cad import fast_update as vjp

from nested_replay_cad.context import normalize_contexts

from nested_replay_cad.data import selection

ROOT=Path(__file__).resolve().parents[1]

def test_policy_has_97_parameters_and_carries_gradient():

    torch.manual_seed(7)

    p={n:torch.zeros(s,requires_grad=True) for n,s in phi_shapes().items()}

    with torch.no_grad():torch.nn.init.xavier_uniform_(p['hidden.weight'])

    assert sum(t.numel() for t in p.values())==97

    c=torch.tensor([[.1,.2,.3,.4],[.8,.7,.6,.5]],dtype=torch.float32)

    w=ReplayPolicy().weights(PhiState(p),c)

    assert torch.all(w>0) and torch.isclose(w.sum(),torch.tensor(1.))

    w[0].backward()

    assert torch.count_nonzero(p['out.weight'].grad)>0

def test_context_normalization_is_detached_and_does_not_mutate():

    c=torch.tensor([[.1,.2,.3,.4],[.8,.7,.6,.5]],dtype=torch.float32);before=c.clone()

    out=normalize_contexts(c)

    assert torch.equal(c,before) and torch.equal(out[:,2:],c[:,2:])

    assert out.grad_fn is None and not out.requires_grad and torch.isfinite(out).all()

def test_replay_reservations_are_disjoint_reproducible_and_private():

    ids=tuple(hashlib.sha256(str(i).encode()).hexdigest() for i in range(24))

    rng=seed_sampling(0);a,an=sample_reservations(ids,ids[-8:],rng);b,bn=sample_reservations(ids,ids[-8:],rng)

    assert a==b and an==bn and rng==seed_sampling(0)

    assert a.eligible

    assert len(set(a.R_reserved+a.Q_recent+a.Q_history))==8

@pytest.mark.parametrize('maximum',[.1,.01])

def test_first_conditional_vjp_matches_autograd_small_smooth_case(maximum):

    p={'w':torch.tensor([.7,-.8],dtype=torch.float64)}

    g={'w':torch.tensor([.02,-.03],dtype=torch.float64,requires_grad=True)}

    m={'w':{'exp_avg':torch.tensor([.01,.02],dtype=torch.float64),'exp_avg_sq':torch.tensor([.02,.03],dtype=torch.float64),'max_exp_avg_sq':torch.tensor([.04,.05],dtype=torch.float64)}}

    q={'w':torch.tensor([.3,-.2],dtype=torch.float64)}

    result,_=vjp.functional_step(p,g,m,7,maximum=maximum)

    expected=torch.autograd.grad((result['w']*q['w']).sum(),g['w'])[0]

    actual,_=vjp.conditional_vjp({k:v.detach() for k,v in g.items()},m,7,q,maximum=maximum)

    assert torch.allclose(actual['w'],expected,atol=1e-10,rtol=1e-8)

def test_full_train_metadata_and_order_guard():

    full=json.loads((ROOT/'configs/train_manifest.json').read_text())

    out=selection(full);assert len(out['records'])==3629 and out['boundaries'][-1]==3629

    broken=copy.deepcopy(full);broken['records'][0]['arrival']=2

    with pytest.raises(ValueError):selection(broken)

def test_cli_help_and_missing_inputs_do_not_create_output(tmp_path):

    help_run=subprocess.run([sys.executable,'-B','scripts/train.py','--help'],cwd=ROOT,capture_output=True,text=True)

    assert help_run.returncode==0 and 'a0_tail' in help_run.stdout and 'all' in help_run.stdout

    output=tmp_path/'run'

    checked=subprocess.run([sys.executable,'-B','scripts/train.py','--check','--data',str(tmp_path/'absent'),'--output',str(output)],cwd=ROOT,capture_output=True,text=True)

    assert checked.returncode==2 and 'Missing inputs' in checked.stderr

    assert not output.exists()

def test_metrics_valid_and_invalid_labels():

    import numpy as np

    from nested_replay_cad.metrics import exact_numpy

    assert exact_numpy([.1,.3,.3,.9],[0,1,0,1])['AP']==pytest.approx(5/6)

    assert exact_numpy([0.,0.,0.,0.],[0,1,0,1])['AUROC']==.5

    for labels in ([.5,1.5],np.array([256,1],dtype=np.int64),[float('nan'),1]):

        with pytest.raises(ValueError):exact_numpy([0.,1.],labels)

def test_small_numerical_qualification():

    from numerical_oracle import run

    result=run('cpu')

    assert result['status']=='PASS' and result['checks']==128

def test_context_derivative_qualification():

    from context_oracle import run

    result=run()

    assert result['status']=='PASS' and result['checks']==74

def test_all_runtime_modules_import_without_gpu():

    import importlib

    for path in sorted((ROOT/'src').rglob('*.py')):

        module='.'.join(path.relative_to(ROOT/'src').with_suffix('').parts).removesuffix('.__init__')

        importlib.import_module(module)

    assert not torch.cuda.is_initialized()

def test_manifest_builder_contract_on_synthetic_images(tmp_path):

    from PIL import Image

    from nested_replay_cad.core.data.manifest import ManifestProtocol,build_manifest

    folder=tmp_path/'widget/train/good';folder.mkdir(parents=True)

    Image.new('RGB',(8,8),(0,0,0)).save(folder/'000.png')

    Image.new('RGB',(8,8),(1,1,1)).save(folder/'001.png')

    value=build_manifest(tmp_path,protocol=ManifestProtocol('synthetic:loader:v1',(('widget',2),)))

    assert isinstance(value,str)

    result=json.loads(value)

    assert len(result['records'])==2

    assert [r['arrival'] for r in result['records']]==[1,2]

def test_cli_input_loader_uses_exact_manifest_and_resolves_paths(tmp_path,monkeypatch):

    import importlib.util,argparse

    from nested_replay_cad.core.data import manifest

    spec=importlib.util.spec_from_file_location('run_cli',ROOT/'scripts/train.py')

    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

    recipe=json.loads((ROOT/'configs/b319.json').read_text())

    weights=tmp_path/'weights';weights.write_bytes(b'fixture')

    recipe['encoder']['sha256']=hashlib.sha256(b'fixture').hexdigest()

    config=tmp_path/'config.json';config.write_text(json.dumps(recipe))

    dataset=tmp_path/'data';dataset.mkdir()

    expected=json.loads((ROOT/'configs/train_manifest.json').read_text())

    monkeypatch.setattr(manifest,'build_manifest',lambda p:json.dumps(expected))

    loaded,full,versions,_=cli.load_inputs(argparse.Namespace(config=config,data=dataset,weights=weights))

    assert loaded['dataset_path']==str(dataset.resolve())

    assert loaded['encoder']['path']==str(weights.resolve())

    assert loaded['integration_arrivals']==3629 and len(loaded['snapshots'])==15

    assert full==expected and versions['torch']=='2.12.0+cu126'

def test_configure_has_platform_guard_instead_of_nameerror():

    from nested_replay_cad.trainer import configure

    if sys.platform!='linux':

        with pytest.raises(ValueError,match='Linux GPU'):configure()

def test_three_arms_and_correct_ctx_switch():

    from nested_replay_cad.replay import NestedReplay

    cfg=json.loads((ROOT/'configs/b319.json').read_text())

    assert cfg['arms']==['er_uniform','a0_tail','a0_tail_ctx']

    raw=NestedReplay(0,'tail',device='cpu',normalize_context=False)

    ctx=NestedReplay(0,'tail',device='cpu',normalize_context=True)

    assert not raw.normalize_context and ctx.normalize_context

    assert all(torch.equal(raw.phi.parameters[k],ctx.phi.parameters[k]) for k in raw.phi.parameters)

@pytest.mark.parametrize('arm',['er_uniform','a0_tail','a0_tail_ctx'])

def test_resolved_arm_metadata_and_handoff(arm,tmp_path,monkeypatch):

    import importlib.util

    from nested_replay_cad import trainer,data

    spec=importlib.util.spec_from_file_location('cli_arm_'+arm,ROOT/'scripts/train.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

    cfg=json.loads((ROOT/'configs/b319.json').read_text());cfg.update(dataset_path=str(tmp_path),encoder={'path':str(tmp_path/'w')})

    monkeypatch.setattr(cli,'load_inputs',lambda args:(cfg,{}, {},ROOT/'configs/b319.json'))
    monkeypatch.setattr(cli.provenance,'source_identity',lambda *args,**kwargs:{'git':{'commit':'fixture','dirty':True},'design_commit':'fixture','design_hashes':{}})

    monkeypatch.setattr(cli.platform,'system',lambda:'Linux')

    monkeypatch.setattr(data,'prepare',lambda *args:None)

    captured={}

    def trial(output,config,actual,deadline):captured.update(config=config,arm=actual);return True

    monkeypatch.setattr(trainer,'arm_trial',trial)

    monkeypatch.setattr(sys,'argv',['train.py','--arm',arm,'--output',str(tmp_path/'run')])

    assert cli.main()==0

    written=json.loads((tmp_path/'run/config.json').read_text())

    assert written['arm']==arm and written['arms']==[arm] and captured['arm']==arm

