"""One GPU fast update/scoring smoke and tiny conditional VJP check; no retained training state."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from nested_replay_cad import paths
    paths.process_environment()
    from nested_replay_cad.trainer import configure
    t=configure();t.cuda.set_device(0);t.empty(1,device='cuda');t.cuda.reset_peak_memory_stats(0)
    from nested_replay_cad.model import encoder,decoder,scoring
    from nested_replay_cad.fast_update import functional_step,conditional_vjp
    from nested_replay_cad.context import normalize_contexts
    from nested_replay_cad.core.policy.policy import ReplayPolicy
    from nested_replay_cad.core.policy.layout import phi_shapes
    from nested_replay_cad.core.contracts import PhiState
    config=json.loads((ROOT/'configs/b319.json').read_text());full=json.loads((ROOT/config['full_manifest_path']).read_text())
    t.manual_seed(0);service=encoder.load(a.weights,device='cuda:0')
    image=encoder.preprocess(a.data/full['records'][0]['relative_path']).unsqueeze(0).to('cuda')
    with t.no_grad():feature=service.groups(image)
    assert feature.shape==(1,2,789,768) and t.isfinite(feature).all()
    del service,image;t.cuda.empty_cache()
    model=decoder.Decoder(checkpoint=True);decoder.init_trainable(model);model.to('cuda');model.train();opt=decoder.make_optimizer(model)
    en,de=model(feature[:,0],feature[:,1]);loss=decoder.global_cosine_hm_percent(en,de,p=.9/26,factor=.1)
    loss.backward();t.nn.utils.clip_grad_norm_(model.parameters(),.1);opt.step();opt.zero_grad(set_to_none=True)
    del en,de,loss;model.eval()
    with t.no_grad():
        en,de=model(feature[:,0],feature[:,1]);pixel,image=scoring.dinomaly(en,de)
    assert pixel.shape==(1,256,256) and t.isfinite(pixel).all() and t.isfinite(image).all()
    contexts=t.tensor([[.1,.2,.3,.4],[.2,.5,.7,.9]],dtype=t.float32)
    phi=PhiState({n:t.zeros(shape,requires_grad=True) for n,shape in phi_shapes().items()})
    weights=ReplayPolicy().weights(phi,normalize_contexts(contexts));assert t.isfinite(weights).all() and abs(float(weights.sum())-1)<1e-6
    g={'w':t.tensor([.02,-.03],device='cuda',dtype=t.float64,requires_grad=True)};params={'w':t.tensor([.7,-.8],device='cuda',dtype=t.float64)}
    moments={'w':{'exp_avg':t.tensor([.01,.02],device='cuda',dtype=t.float64),'exp_avg_sq':t.tensor([.02,.03],device='cuda',dtype=t.float64),'max_exp_avg_sq':t.tensor([.04,.05],device='cuda',dtype=t.float64)}}
    q={'w':t.tensor([.3,-.2],device='cuda',dtype=t.float64)}
    stepped,_=functional_step(params,g,moments,7);expected=t.autograd.grad((stepped['w']*q['w']).sum(),g['w'])[0]
    actual,_=conditional_vjp({n:v.detach() for n,v in g.items()},moments,7,q)
    t.testing.assert_close(actual['w'],expected,rtol=1e-8,atol=1e-10)
    result=dict(status='PASS',scope='one GPU fast update/scorer plus tiny VJP and CTX-policy; full-model slow event not qualified by this smoke',device=t.cuda.get_device_name(0),peak_VRAM_bytes=t.cuda.max_memory_allocated(0),fresh_full_run_required=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print('B319_PREFLIGHT_PASS',json.dumps(result),flush=True)
