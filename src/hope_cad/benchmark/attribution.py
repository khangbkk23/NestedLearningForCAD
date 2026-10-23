"""Bounded, normal-event-only prequential analysis. Never imports a model/torch."""
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

IDS = ('OBS-NESTED', 'OBS-GAMMA', 'OBS-MEAN', 'OBS-GAMMA-BIAS',
       'OBS-AFFINE-GAMMA', 'OBS-AFFINE-CLOCK', 'OBS-NLMS-BIAS')
ARCHIVES = {
    'NESTED_seed0': ('hope_cad_matrix_NESTED_seed0_evidence.zip', '668f9e348561d5ec55c4cf51762c5890b19328c2a950c7ba2fb09f037b1bea0d'),
    'CADIC_seed0': ('hope_cad_matrix_CADIC_seed0_evidence.zip', '035f88639f8faab8bf02fdfbcd5f502677620c3756af53dcf4815dd40e8cc889')}


def strict_json(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out: raise ValueError('Duplicate JSON key: ' + k)
            out[k] = v
        return out
    def bad(v): raise ValueError('Nonfinite JSON: ' + v)
    value=json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    def finite(v):
        if type(v) is float and not math.isfinite(v):raise ValueError('Nonfinite numeric JSON')
        if isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,list):
            for x in v:finite(x)
    finite(value)
    return value


def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


class NormalArchive:
    def __init__(self, path, arm, deadline, expected_sha):
        self.path, self.arm, self.deadline = Path(path), arm, deadline
        h = hashlib.sha256()
        with self.path.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                self.boundary(); h.update(block)
        if h.hexdigest() != expected_sha: raise ValueError('Input ZIP SHA mismatch')
        self.archive = zipfile.ZipFile(self.path)
        names = self.archive.namelist()
        if len(names) != len(set(names)): raise ValueError('Duplicate ZIP member')
        for n in names:
            p = PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n or ':' in n or p.as_posix() != n:
                raise ValueError('Unsafe ZIP name')
        self.used, self.event_bytes = {}, 0
        self.manifest = self.read('artifact_manifest.json', 32*1024**2, manifest=True)
        if set(names) != set(self.manifest['files']) | {'artifact_manifest.json'}: raise ValueError('ZIP inventory differs')
        self.legacy = sum(n.startswith(f'benchmark/legacy/runs/{arm}/events/') and n.endswith('.json') for n in names)
        event_names={n for n in names if n.startswith(f'benchmark/runs/{arm}/events/')}
        if event_names!={f'benchmark/runs/{arm}/events/{i:05d}.json' for i in range(3599)}:raise ValueError('Missing/extra canonical events')
        self.states = sum(n.endswith('.pt') for n in names)  # Names only, never decode tensor states.

    def boundary(self):
        if time.monotonic() >= self.deadline: raise TimeoutError('D0 soft budget reached')

    def read(self, name, limit=None, manifest=False):
        self.boundary()
        event = name.startswith(f'benchmark/runs/{self.arm}/events/') and name.endswith('.json')
        allowed = {'artifact_manifest.json', 'benchmark/manifest_train.json', f'benchmark/runs/{self.arm}/train_complete.json'}
        if name not in allowed and not event: raise ValueError('D0 member outside allowlist')
        limit = limit or (1024**2 if event else 16*1024**2)
        info = self.archive.getinfo(name)
        if info.file_size > limit: raise ValueError('D0 member byte cap')
        if event:
            self.event_bytes += info.file_size
            if self.event_bytes > 256*1024**2: raise ValueError('D0 total event cap')
        raw = self.archive.read(name); checksum = hashlib.sha256(raw).hexdigest()
        if not manifest and checksum != self.manifest['files'][name]: raise ValueError('Used member SHA differs')
        self.used[name] = checksum
        return strict_json(raw)

    def close(self): self.archive.close()


class Predictors:
    def __init__(self):
        self.n = 0; self.sy = self.sr = self.bias = 0.
        self.S = {d: np.zeros((d,d), dtype=np.float64) for d in (2,3)}
        self.v = {d: np.zeros(d, dtype=np.float64) for d in (2,3)}

    def predict(self, e):
        gamma = e['gamma']; q = math.log1p(e['cursor']) / math.log1p(1000000)
        xs = {2: np.array([1.,gamma]), 3: np.array([1.,gamma,q])}
        out = dict(zip(IDS, [e['prediction'], gamma, gamma, gamma, gamma, gamma, gamma+self.bias]))
        if self.n >= 32:
            out['OBS-MEAN'] = self.sy / self.n
            out['OBS-GAMMA-BIAS'] = gamma + self.sr / self.n
            for d, name in ((2,'OBS-AFFINE-GAMMA'), (3,'OBS-AFFINE-CLOCK')):
                penalty = np.eye(d)*1e-6; penalty[0,0] = 0
                out[name] = float(xs[d] @ np.linalg.solve(self.S[d]+penalty, self.v[d]))
        if not all(math.isfinite(v) for v in out.values()): raise ValueError('Nonfinite prediction')
        return out, xs

    def update(self, e, credit, pred, xs):
        y = credit['target']; self.n += 1; self.sy += y; self.sr += y-e['gamma']
        self.bias -= .05*(pred['OBS-NLMS-BIAS']-y)
        for d, x in xs.items(): self.S[d] += np.outer(x,x); self.v[d] += x*y


def close(a, b): return abs(a-b) <= 1e-12 + 1e-10*max(abs(a), abs(b))


def joined_rows(archive):
    manifest = archive.read('benchmark/manifest_train.json'); online = manifest['online']
    if len(online) != 3599 or len({r['relative_path'] for r in online}) != 3599: raise ValueError('Wrong historical manifest')
    predictor = Predictors(); pending = None; decisions = warm = rejects = 0
    for cursor, row in enumerate(online):
        name = f'benchmark/runs/{archive.arm}/events/{cursor:05d}.json'; e = archive.read(name)
        if e['cursor'] != cursor or e['identity'] != row['relative_path']: raise ValueError('Event identity/chronology changed')
        credit = e.get('credit')
        if credit:
            if pending is None: raise ValueError('Credit without decision')
            old, pred, xs, member, checksum = pending
            for ck, ek in (('decision_identity','identity'), ('decision_cursor','cursor'), ('c','c'), ('gamma','gamma'), ('accepted','accepted'), ('prediction','prediction'), ('decision_d','decision_d')):
                if credit[ck] != old[ek]: raise ValueError('Credit fixed-pair join mismatch: '+ck)
            if credit['reward_observed_at'] != cursor or old['cursor']+1 != cursor: raise ValueError('Noncausal label')
            delta = credit['Jminus']-credit['Jplus']
            if credit['c'] < 1 or not close(credit['target'],delta/credit['c']): raise ValueError('Target QA')
            for suffix in ('minus','plus'):
                if credit['anchor_'+suffix] != old['anchor_'+suffix] or not close(credit['J'+suffix],credit['next_'+suffix]+credit['anchor_'+suffix]): raise ValueError('Fixed anchor QA')
            out = dict(cursor=old['cursor'], identity=old['identity'], category=row['task_name'] if row['task_id']==online[old['cursor']]['task_id'] else online[old['cursor']]['task_name'],
                evaluate=not old['warmup'], decision=old, credit=credit, predictions=pred,
                source_member=member, source_sha256=checksum, credit_member=name, credit_sha256=archive.used[name])
            predictor.update(old,credit,pred,xs); pending = None
            yield out
        elif pending is not None: raise ValueError('Missing next-event credit')
        if e['action'] in ('accept','reject'):
            for k in ('prediction','gamma','c','decision_d','current_minus','current_plus','anchor_minus','anchor_plus'):
                if type(e[k]) not in (int,float) or not math.isfinite(e[k]): raise ValueError('Decision scalar QA')
            if e['c'] < 1 or e['warmup'] != (predictor.n < 32): raise ValueError('Warmup/scale QA')
            scale=1+e['current_minus']+e['current_plus']+e['anchor_minus']+e['anchor_plus']
            prior=(e['current_minus']+e['anchor_minus']-e['current_plus']-e['anchor_plus'])/scale
            if not close(e['c'],scale) or not close(e['gamma'],prior):raise ValueError('Immediate scale/prior QA')
            decisions += 1; warm += int(e['warmup']); rejects += int(not e['accepted'])
            pred,xs = predictor.predict(e); pending = e,pred,xs,name,archive.used[name]
    completion = archive.read(f'benchmark/runs/{archive.arm}/train_complete.json')
    if pending is not None or (decisions,predictor.n,warm,rejects)!=(3211,3211,32,1935) or completion['images']!=3599 or completion['counters']['censored']!=0 or archive.legacy!=376:
        raise ValueError('Historical counts/censor changed')


def summary(values):
    if not values: return dict(mean=None,quantiles=None,reason='empty_group')
    a=np.asarray(values,dtype=np.float64)
    return dict(mean=float(a.mean()),quantiles=np.quantile(a,[0,.25,.5,.75,1],method='linear').tolist())


def corr(x,y):
    return None if len(x)<2 or np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1])


def metrics(rows, name):
    n=len(rows)
    if not n:return dict(count=0,reason='empty_group',MSE=None,MAE=None)
    p=np.array([r['predictions'][name] for r in rows]);y=np.array([r['credit']['target'] for r in rows])
    delta=np.array([r['credit']['Jminus']-r['credit']['Jplus'] for r in rows]);c=np.array([r['credit']['c'] for r in rows]);g=np.array([r['decision']['gamma'] for r in rows])
    a=p>=0; nonzero=delta!=0; regret=np.maximum(delta,0)-a*delta
    return dict(count=n,MSE=float(np.mean((p-y)**2)),MAE=float(np.mean(abs(p-y))),raw_advantage_MSE=float(np.mean((c*p-delta)**2)),
        regret_mean=float(regret.mean()),regret_sum=float(regret.sum()),accept_fraction=float(a.mean()),
        sign_agreement=float(np.mean(a[nonzero]==(delta[nonzero]>0))) if nonzero.any() else None,
        sign_agreement_reason=None if nonzero.any() else 'no_nonzero_advantages',zero_target_fraction=float((~nonzero).mean()),
        target=summary(y.tolist()),prediction=summary(p.tolist()),gamma=summary(g.tolist()),prediction_error=summary((p-y).tolist()),
        learned_residual=summary((p-g).tolist()),cancellation=summary((abs(p[g!=0])/abs(g[g!=0])).tolist()),gamma_zero_count=int((g==0).sum()),
        correlations=dict(p_y=corr(p,y),p_gamma=corr(p,g),p_clock=corr(p,[math.log1p(r['cursor'])/math.log1p(1000000) for r in rows])),
        undefined_correlation_reason='null iff fewer_than_two_or_zero_variance')


def run(input_root, output, *, nested_archive=None, cadic_archive=None):
    started=time.monotonic();deadline=started+1770;out=Path(output).resolve()
    if out.exists(): raise ValueError('D0 requires a new session directory; preserve partial attempts')
    if shutil.disk_usage(out.parent if out.parent.exists() else Path.cwd()).free<64*1024**2:raise ValueError('Need64MiB output reserve')
    out.mkdir(parents=True);archives={};status='failed_preserve_evidence';failure=None
    try:
        explicit={'NESTED_seed0':nested_archive,'CADIC_seed0':cadic_archive}
        for arm,(name,h) in ARCHIVES.items():archives[arm]=NormalArchive(explicit[arm] or Path(input_root)/name,arm,deadline,h)
        cad=archives['CADIC_seed0'];cm=cad.read('benchmark/manifest_train.json');cd=cad.read('benchmark/runs/CADIC_seed0/train_complete.json')
        for i,row in enumerate(cm['online']):
            e=cad.read(f'benchmark/runs/CADIC_seed0/events/{i:05d}.json')
            if e['cursor']!=i or e['identity']!=row['relative_path']:raise ValueError('CADIC event scope changed')
        if cd['images']!=3599:raise ValueError('CADIC scope changed')
        rows=[];nested=archives['NESTED_seed0']
        with (out/'prequential_predictions.jsonl.tmp').open('w',encoding='utf8') as f:
            for row in joined_rows(nested):
                rows.append(row);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        (out/'prequential_predictions.jsonl.tmp').replace(out/'prequential_predictions.jsonl')
        if cm['online']!=nested.read('benchmark/manifest_train.json')['online']:raise ValueError('Paired normal manifests differ')
        main=[r for r in rows if r['evaluate']]
        categories=sorted({r['category'] for r in main})
        report={n:metrics(main,n) for n in IDS}
        percat={cat:{n:metrics([r for r in main if r['category']==cat],n) for n in IDS} for cat in categories}
        timebins={str(b):{n:metrics([r for r in main if r['cursor']//256==b],n) for n in IDS} for b in range(15)}
        action={a:float(np.mean([max(r['credit']['Jminus']-r['credit']['Jplus'],0)-accept*(r['credit']['Jminus']-r['credit']['Jplus']) for r in main])) for a,accept in [('ALWAYS_ACCEPT',True),('ALWAYS_REJECT',False)]}
        publish(out/'metrics.json',dict(micro=report,per_category=percat,macro_category_MSE={n:float(np.mean([percat[k][n]['MSE'] for k in categories])) for n in IDS},time_bins=timebins,action_only_regret=action,scope='logged fixed pairs, not new policy or AD counterfactual'))
        bins={};edges=[0,1e-4,1e-3,1e-2,.05,.1,float('inf')]
        for n in IDS:
            bins[n]=[]
            for lo,hi in zip(edges,edges[1:]):
                group=[r for r in main if lo<=abs(r['predictions'][n])<hi];m=metrics(group,n)
                m.update(lower=lo,upper=hi if math.isfinite(hi) else None,mean_y=summary([r['credit']['target'] for r in group])['mean'],sum_abs_advantage=sum(abs(r['credit']['Jminus']-r['credit']['Jplus']) for r in group),nonzero_count=sum(r['credit']['Jminus']!=r['credit']['Jplus'] for r in group),sign_errors=sum((r['predictions'][n]>=0)!=(r['credit']['Jminus']>r['credit']['Jplus']) for r in group if r['credit']['Jminus']!=r['credit']['Jplus']))
                bins[n].append(m)
        publish(out/'margin_bins.json',bins)
        components={}
        for label,group in [('all',rows),('accepted',[r for r in rows if r['decision']['accepted']]),('rejected',[r for r in rows if not r['decision']['accepted']]),('warmup',[r for r in rows if not r['evaluate']]),('postwarm',main)]:
            vals={k:[] for k in ('changed_rows','replacement_ops','committed_changed_rows','next_advantage','anchor_advantage','current_advantage','y','Delta')}
            for r in group:
                d,cr=r['decision'],r['credit'];v=[d['changed_rows'],d['proposal']['replaced'],int(d['accepted'])*d['changed_rows'],cr['next_minus']-cr['next_plus'],cr['anchor_minus']-cr['anchor_plus'],d['current_minus']-d['current_plus'],cr['target'],cr['Jminus']-cr['Jplus']]
                for k,x in zip(vals,v):vals[k].append(x)
            components[label]=dict(count=len(group),values={k:summary(v) for k,v in vals.items()},correlations_to_Delta={k:corr(v,vals['Delta']) for k,v in vals.items()})
        publish(out/'proposal_components.json',components)
        publish(out/'record_qa.json',dict(events=3599,completed=3211,postwarm=len(main),legacy_excluded=376,censor=0,passed=True))
        publish(out/'state_availability.json',dict(tensor_state_member_names={k:v.states for k,v in archives.items()},tensors_loaded=False,neural_probe='not_performed',exact_WNR='unavailable_without_per_row_provenance'))
        status='complete_pending_review'
    except (BaseException,) as exc:
        failure=exc;status='paused_preserve_partial' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'failed_preserve_evidence'
    finally:
        publish(out/'input_receipt.json',dict(analysis_id='support_utility_attribution_v2',
            analyzer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            archives={k:dict(zip_sha256=ARCHIVES[k][1],used_members=a.used) for k,a in archives.items()}))
        for a in archives.values():a.close()
        publish(out/'resources.json',dict(elapsed_seconds=time.monotonic()-started,wall_cap_seconds=1800,CPU_thread_cap=2,target_RAM_GiB=2,model_forward=False))
        (out/'limitations.md').write_text('Observed normal fixed-pair analysis only; no independent AD or current-bank counterfactual. Neural/GT/test inputs not read. Negative predictors remain valid results.\n',encoding='utf8')
        publish(out/'handoff.json',dict(status=status,error=repr(failure) if failure else None,reportable=False))
        publish(out/'artifact_manifest.json',dict(schema='support_utility_attribution_v2',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='artifact_manifest.json' and not p.name.endswith('.tmp')}))
    if failure:raise RuntimeError('D0 stopped; preserve output '+str(out)) from failure
    return dict(status=status,output=str(out))
