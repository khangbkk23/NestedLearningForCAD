"""Verify/lock/run one P0. Only this module can open the three allowed clean caches."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from exps.task_p.p0_normal_support import (
    ARMS, BUDGET, GREEDY_CAPACITY, RESERVOIR_CAPACITY, Support, distances,
    fingerprint, require, serialized_bytes,
)

ROOT = Path('/mnt/d/code_for_fun/project/nestedlearningforcad/nestedlearningforcad')
FEATURES = ROOT / 'results/hope_cad/frozen_features_v1'
OUT = ROOT / 'results/hope_cad/task_p/p0'
CATEGORIES = ('bottle', 'carpet', 'hazelnut')
TRAIN_GRID = tuple(r * 28 + c for r in (3, 10, 17, 24) for c in (3, 10, 17, 24))
PROBE_GRID = tuple(r * 28 + c for r in (1, 4, 8, 11, 15, 18, 22, 26) for c in (1, 4, 8, 11, 15, 18, 22, 26))
CP_SHA = '755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415'
SOURCE_DIGEST = '060579457d344992eef159b1d4f1a86c34dd9283b97b70d3c3a386006b0f4dcc'
CACHE_SHA = {
    'clean/bottle_train.pt': 'ef4f3180d2381b64b4cdafea447b9df254b7522b9338f24992f4a857114ad6b3',
    'clean/carpet_train.pt': 'f3b1957d7b9d12add45c4ff9c33a5f9f5e237ff7ce6dbdb7693c81527c89b179',
    'clean/hazelnut_train.pt': '7bd97accb4a927285881d8c8b61cbc5db8106e57e2ceee1fe623f5903f62d180',
}
CODE = ('exps/task_p/p0_normal_support.py', 'exps/task_p/p0_protocol.py',
        'exps/task_p/test_p0_normal_support.py', 'exps/task_p/analyze_p0.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def save(name, value):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def configure():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)


def environment():
    return {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
            'device': 'cpu', 'threads': torch.get_num_threads(), 'deterministic': torch.are_deterministic_algorithms_enabled()}


def validate_roles(updates, probes, r1_probes):
    for rows in (updates, probes, r1_probes):
        for row in rows:
            path = row['relative_path']
            require(path == f"{row['category']}/train/good/{row['cache_index']:03}.png" and row['category'] in CATEGORIES,
                    'forbidden source path')
        require(len({r['relative_path'] for r in rows}) == len(rows), 'duplicate source identity')
        require(len({r['raw_png_sha256'] for r in rows}) == len(rows), 'duplicate source content')
    for i, left in enumerate((updates, probes, r1_probes)):
        for right in (updates, probes, r1_probes)[i+1:]:
            require(not {r['relative_path'] for r in left} & {r['relative_path'] for r in right}, 'split identity overlap')
            require(not {r['raw_png_sha256'] for r in left} & {r['raw_png_sha256'] for r in right}, 'split content overlap')


def roles(source):
    entries = {r['relative_path']: r for r in source['entries']}
    def row(cat, index, event=0):
        path = f'{cat}/train/good/{index:03}.png'
        require(path in entries, 'missing approved identity')
        return {'category': cat, 'cache_index': index, 'relative_path': path,
                'raw_png_sha256': entries[path]['raw_png_sha256'], 'event': event}
    generator = torch.Generator().manual_seed(0); updates = []
    for cat, start, stop in [('bottle', 0, 100), ('carpet', 0, 100), ('hazelnut', 0, 100), ('bottle', 100, 150)]:
        for i in torch.randperm(stop-start, generator=generator).tolist(): updates.append(row(cat, start+i, len(updates)+1))
    probes = [row(cat, i) for cat in CATEGORIES for i in range(170 if cat == 'bottle' else 120, 190 if cat == 'bottle' else 140)]
    r1 = [row(cat, i) for cat in CATEGORIES for i in range(150 if cat == 'bottle' else 100, 170 if cat == 'bottle' else 120)]
    validate_roles(updates, probes, r1)
    prior = json.loads((ROOT / 'results/hope_cad/r1_conditional_pairing/protocol_resolved.json').read_text())
    require([r['relative_path'] for r in updates] == [r['relative_path'] for r in prior['updates']], 'historical order mismatch')
    return updates, probes, r1


class Reader:
    """Sequential update reader and separately gated evaluator; no past-update replay."""
    def __init__(self, protocol, preflight=False):
        self.p = protocol; self.completed = 0; self.admitted = set(); self.current = None
        self.obj = None; self.access = []; self.preflight = preflight
        self.entries = json.loads((FEATURES / 'source_manifest.json').read_text())['entries']

    def _open(self, category):
        require(category in CATEGORIES, 'forbidden category')
        obj = torch.load(FEATURES / f'clean/{category}_train.pt', mmap=True, weights_only=True)
        entries = [r for r in self.entries if r['category'] == category]
        require(obj['view'] == 'clean' and obj['class_name'] == category, 'wrong cache view')
        require(obj['relative_paths'] == [r['relative_path'] for r in entries], 'cache paths mismatch')
        require(obj['per_image_source_sha256'] == [r['raw_png_sha256'] for r in entries], 'cache source hashes mismatch')
        require(obj['patches'].shape == (len(entries), 784, 768) and obj['patches'].dtype == torch.float32, 'cache geometry mismatch')
        require(obj['source_manifest_digest'] == SOURCE_DIGEST and obj['metadata']['checkpoint_sha256'] == CP_SHA, 'cache provenance mismatch')
        require(obj['metadata']['block_index'] == 8 and obj['metadata']['feature_normalization'] == 'none', 'cache transform mismatch')
        return obj

    def _select(self, obj, row, grid, role):
        require(obj['relative_paths'][row['cache_index']] == row['relative_path'], 'cache row mismatch')
        full = obj['patches'][row['cache_index']]
        # Opaque content hash of an authorized row; only selected vectors reach the learner/evaluator.
        h = hashlib.sha256(str((tuple(full.shape), full.dtype)).encode() + full.contiguous().numpy().tobytes()).hexdigest()
        require(h == obj['per_image_feature_sha256'][row['cache_index']], 'feature row digest mismatch')
        result = full[list(grid)].clone()
        require(bool(torch.isfinite(result).all()), 'nonfinite authorized features')
        self.access.append({'role': role, 'path': row['relative_path'], 'event': self.completed,
                            'positions': len(grid), 'selected_sha256': fingerprint({'x': result})})
        return result

    def update(self, row):
        require(self.completed < len(self.p['updates']) and row == self.p['updates'][self.completed], 'future or unauthorized update')
        require(not self.preflight or self.completed < 10, 'preflight beyond first admitted bottle block')
        category = row['category']
        if self.current != category:
            self.obj = None; self.current = category; self.admitted.add(category); self.obj = self._open(category)
        self.completed += 1
        return self._select(self.obj, row, TRAIN_GRID, 'feasibility_training' if self.preflight else 'update')

    def probe(self, row, boundary):
        require(not self.preflight, 'preflight probes forbidden')
        require(boundary == self.completed and boundary in (100, 200, 300, 350), 'probe at forbidden boundary')
        require(row in self.p['probes'] and row['category'] in self.admitted, 'future or unauthorized probe')
        obj = self._open(row['category'])
        return self._select(obj, row, PROBE_GRID, 'probe')


def feasibility(protocol):
    # Only the first admitted bottle identity is used numerically. No future/probe cache is mapped.
    reader = Reader(protocol, preflight=True); x = reader.update(protocol['updates'][0])
    g = torch.Generator().manual_seed(7204); timed = {}
    for method in ARMS:
        m = Support(method)
        # Full-size synthetic state, not an extra scientific run or feature transform.
        m.bank[:] = torch.randn(m.bank.shape, generator=g) * 2
        m.ids[:] = torch.arange(m.capacity); m.n = m.capacity; m.seen = m.capacity
        if method != 'RESERVOIR':
            for start in range(0, m.n, 128):
                stop = min(m.n, start+128); d = distances(m.bank[start:stop], m.bank[:m.n])
                d[torch.arange(stop-start), torch.arange(start, stop)] = torch.inf
                v, ix = d.min(1); m.nn_distance[start:stop] = v; m.nn_index[start:stop] = ix
        if method == 'WITNESS_PROTECTED':
            m.witnesses[:] = m.bank[:128]; m.witness_ids[:] = m.ids[:128]; m.wn = 128; m.wseen = m.n
        t = time.perf_counter(); stats = m.update(x, torch.arange(5600, 5616)); elapsed = time.perf_counter()-t
        timed[method] = {'full_bank_image_update_seconds': elapsed, 'ledger': m.ledger(), 'accepted': stats['inserted'] + stats['replaced']}
        del m; gc.collect()
    projected = 350 * sum(v['full_bank_image_update_seconds'] for v in timed.values())
    allowance = 3 * projected + 120
    result = {'no_probe_reads': True, 'first_admitted_training_path': protocol['updates'][0]['relative_path'],
              'full_size_synthetic_state_with_one_training_image': True, 'measurements': timed,
              'projected_update_seconds': projected, 'threefold_plus_120s_seconds': allowance,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              'access_log': reader.access}
    save('feasibility.json', result)
    require(allowance <= 3600, 'BLOCKED_COMPUTE: conservative throughput projection exceeds 60 minutes')
    require(result['peak_rss_bytes'] <= 6*1024**3, 'BLOCKED_COMPUTE: feasibility peak RSS exceeds 6 GiB')
    return result


def prepare():
    require(not (OUT / 'protocol_resolved.json').exists() and not (OUT / 'run_started.json').exists(), 'refusing overwrite or second attempt')
    require('passed' in (OUT / 'unit_tests.txt').read_text(), 'focused tests missing')
    source = json.loads((FEATURES / 'source_manifest.json').read_text())
    canonical = hashlib.sha256(json.dumps({k:v for k,v in source.items() if k!='manifest_digest'}, sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(canonical == SOURCE_DIGEST == source['manifest_digest'], 'source canonical digest mismatch')
    for name, expected in CACHE_SHA.items():
        require(sha(FEATURES/name) == expected, 'cache byte hash mismatch: '+name); print('verified', name, flush=True)
    require(sha(ROOT/'checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth') == CP_SHA, 'checkpoint digest mismatch')
    docs = {n:json.loads((FEATURES/n).read_text()) for n in ('extractor_contract.json','checkpoint_provenance.json','cache_integrity.json','independent_verification.json')}
    require(docs['independent_verification.json']['overall_passed'], 'rebuild independent verification failed')
    require(docs['extractor_contract.json']['checkpoint_sha256'] == CP_SHA, 'extractor checkpoint mismatch')
    updates, probes, r1 = roles(source)
    p = {'schema':'task_p_p0_v1','root':str(ROOT),'locked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'checkpoint_sha256':CP_SHA,'cache_sha256':CACHE_SHA,'source_canonical_digest':SOURCE_DIGEST,
         'input_document_sha256':{n:sha(FEATURES/n) for n in ('source_manifest.json',*docs)},
         'source_code_sha256':{n:sha(ROOT/n) for n in CODE},'environment':environment(),
         'updates':updates,'probes':probes,'excluded_r1_probes':r1,'train_grid':TRAIN_GRID,'probe_grid':PROBE_GRID,
         'arrival_boundaries':[100,200,300,350],'order_seed':0,'reservoir_seed':7202,'witness_seed':7203,
         'budget_bytes':BUDGET,'metadata_reserve_bytes':16384,
         'capacities':{'FROZEN_FIRST':GREEDY_CAPACITY,'RESERVOIR':RESERVOIR_CAPACITY,'STREAM_GREEDY':GREEDY_CAPACITY,'WITNESS_PROTECTED':2350},
         'witness_capacity':128,
         'methods':{
           'FROZEN_FIRST':'same exact closest-pair coverage selector as C; freeze score bank after 100 image events; counters persist',
           'RESERVOIR':'Algorithm R in row-major arrival order; full budget after metadata/RNG; no labels/quotas',
           'STREAM_GREEDY':'not exact ContCore; fill by farthest remaining image descriptor; thereafter repeatedly select farthest remaining descriptor; replace global most redundant row iff novelty exceeds its exact nearest-distinct distance; stable row-index ties',
           'WITNESS_PROTECTED':'same farthest remaining image order; fill greedy; full bank: 16 most locally redundant rows by exact incremental nearest-distinct graph; evaluate all 16 replacements; accept max(G-H) among G>H and tail90(relu witness increments)<=0.05*s_old; else reject proposal; continue remaining image descriptors'},
         'witness_semantics':{'update':'Algorithm R in original row-major order after all image bank decisions; W fixed and historical during the image',
             'scale':'median strictly positive nearest-other-index W/W distance before each image; filter zero minima; fewer than 2 witnesses or empty positive distances rejects all replacements',
             'tail':'linear-interpolated 90th percentile of positive-part increments across ALL witnesses, including zeros',
             'gain':'mean decrease of distances of all 16 current-image vectors, including already processed rows',
             'cost':'mean positive-part witness distance increase across ALL historical witnesses',
             'counterfactual':'at each proposal compare protected choice with same-state argmax positive G without witness constraints; synthetic guard-off only, no fifth benchmark arm'},
         'distance':'FP64 accumulated raw Euclidean norm-dot with direct-difference repair at <=64*float64_eps*(sum squared norms); stable first-index min ties',
         'risk':'mean largest 7 of 64 probe patch distances; per image then mean images; equal-category macro normalized by training scale',
         'scale_formula':'median of 150 L2 differences between consecutive row-major candidate positions j,j+1 from first ten admitted images of each category; evaluator-only, computed after tenth arrival',
         'primary':'final D versus C; no outcome-selected baseline; B and A descriptive',
         'bootstrap':{'replicates':10000,'seed':7201,'unit':'whole source image; paired stratified within category','interval':'percentile 2.5/97.5; fixed training/order seed'},
         'gates':{'final_gain':0.05,'ci_lower':0.,'pre_return_deterioration':0.20,'acquisition_gain':0.10,'per_category_final_deterioration':0.10,
                  'retention_contrasts':[['bottle',100,200],['bottle',100,300],['carpet',200,300]],
                  'near_zero_normalized_risk':1e-10,'actual_guard_effect_required':True,'all_integrity_required':True},
         'diagnostics':'raw and scaled risks; mean/median/p90/max patch distances; D-C patch distance rank correlation and matched descriptor-ID rate; no unbounded oracle',
         'stop_rules':{'computation_seconds':3600,'peak_rss_bytes':6*1024**3,'output_bytes':512*1024**2,'no_recovery_training_or_tuning':True},
         'limitations':['P0 images disjoint from R1 probes but not proven fresh to historical AD-01','unknown object/acquisition grouping','sparse training/probe grids','one seed/order','candidate and C differ in capacity and replacement objective; synthetic/actual guard counterfactual identifies decision effect, not isolated performance attribution'],
         'preserved_sha256':json.loads((OUT/'preflight_preservation.json').read_text())['preserved_sha256']}
    feasibility(p)
    save('input_verification.json',{'passed':True,'cache_sha256':CACHE_SHA,'checkpoint_sha256':CP_SHA,'canonical_source_digest':canonical,'source_file_sha256':sha(FEATURES/'source_manifest.json'),
         'role_counts':[len(updates),len(probes),len(r1)],'filename_content_disjoint':True,'numeric_future_cache_access_before_arrival':False})
    save('protocol_resolved.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol_resolved.json')+'\n')
    print('PROTOCOL_LOCKED',sha(OUT/'protocol_resolved.json'),flush=True)


def run():
    p=json.loads((OUT/'protocol_resolved.json').read_text()); protocol_sha=sha(OUT/'protocol_resolved.json')
    require(protocol_sha==(OUT/'protocol.sha256').read_text().strip(),'protocol changed')
    require(all(sha(ROOT/n)==h for n,h in p['source_code_sha256'].items()),'frozen source changed')
    require(p['environment']==environment(),'environment changed')
    with (OUT/'run_started.json').open('x') as stream:
        json.dump({'protocol_sha256':protocol_sha,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},stream)
    started=time.perf_counter(); reader=Reader(p); models={a:Support(a) for a in ARMS}
    records=[]; boundaries=[]; trace=[]; chain=[]; scales={}; scale_values={c:[] for c in CATEGORIES}; checks=[]
    timing={a:{'update_seconds':0.,'read_seconds':0.,'read_images':0,'max_persistent_bytes':0} for a in ARMS}
    previous={a:fingerprint(m.state()) for a,m in models.items()}
    for row in p['updates']:
        require(time.perf_counter()-started<3600,'BLOCKED_COMPUTE: run time exceeded')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=6*1024**3,'BLOCKED_COMPUTE: RSS exceeded')
        x=reader.update(row);event=row['event'];category=row['category']
        if category not in scales:
            scale_values[category].extend((x[1:].double()-x[:-1].double()).norm(dim=-1).tolist())
            if len(scale_values[category])==150:
                value=float(np.median(scale_values[category]));require(np.isfinite(value) and value>0,'invalid training-only scale')
                scales[category]={'value':value,'available_at_event':event,'source_paths':[q['relative_path'] for q in p['updates'][:event] if q['category']==category][:10]}
                scale_values[category].clear()
        ids=torch.arange((event-1)*16,event*16,dtype=torch.int64)
        input_hash=fingerprint({'x':x,'ids':ids}); before={a:fingerprint(m.state()) for a,m in models.items()}
        require(before==previous,'state reset between events')
        for a,m in models.items():
            if event==1:
                h=fingerprint(m.state());m.update(x,ids,write=False);require(h==fingerprint(m.state()),'no-write mutated state')
                checks.append({'arm':a,'no_write_state_unchanged':True})
            t=time.perf_counter(); stats=m.update(x,ids); update_seconds=time.perf_counter()-t; timing[a]['update_seconds']+=update_seconds
            require(input_hash==fingerprint({'x':x,'ids':ids}),'learner mutated common proposals')
            ledger=m.ledger();timing[a]['max_persistent_bytes']=max(timing[a]['max_persistent_bytes'],ledger['charged_persistent_bytes'])
            after=fingerprint(m.state());bank_after=fingerprint({'bank':m.bank,'ids':m.ids})
            trace.append({'event':event,'arm':a,'input_sha256':input_hash,'after_sha256':after,
                          'score_bank_sha256':bank_after,'ledger':ledger,'update_seconds':update_seconds,**stats})
        previous={a:fingerprint(m.state()) for a,m in models.items()}
        chain.append({'event':event,'before':before,'after':previous})
        if event in p['arrival_boundaries']:
            boundary={'event':event,'scales':dict(scales),'methods':{},'categories':{}}
            for a,m in models.items():
                by_category={c:0 for c in CATEGORIES}
                for serial in m.ids[:m.n].tolist():by_category[p['updates'][serial//16]['category']]+=1
                size=serialized_bytes(m.state());require(size<=BUDGET,'serialized state exceeds 8 MiB')
                boundary['methods'][a]={'ledger':m.ledger(),'serialized_state_bytes':size,'support_counts_audit_only':by_category,'seen_descriptors':m.seen,'events':m.events}
            for cat in CATEGORIES:
                if cat not in reader.admitted:continue
                group=[]
                for probe in [q for q in p['probes'] if q['category']==cat]:
                    q=reader.probe(probe,event); entry={'event':event,'category':cat,'path':probe['relative_path'],
                        'raw_png_sha256':probe['raw_png_sha256'],'scale':scales[cat]['value'],'arms':{}}
                    for a,m in models.items():
                        h=fingerprint(m.state());t=time.perf_counter();d,idx=m.score(q)
                        timing[a]['read_seconds']+=time.perf_counter()-t;timing[a]['read_images']+=1
                        repeat,_=m.score(q);require(torch.equal(d,repeat),'read replay differs')
                        require(h==fingerprint(m.state()),'read mutated learner state')
                        tail=float(d.topk(7).values.mean())
                        entry['arms'][a]={'tail_raw':tail,'tail_normalized':tail/scales[cat]['value'],
                            'mean':float(d.mean()),'median':float(torch.quantile(d,.5)),'p90':float(torch.quantile(d,.9)),
                            'max':float(d.max()),'distances':d.tolist(),'nearest_serials':idx.tolist()}
                    records.append(entry);group.append(entry)
                boundary['categories'][cat]={a:{'raw_risk':float(np.mean([e['arms'][a]['tail_raw'] for e in group])),
                    'normalized_risk':float(np.mean([e['arms'][a]['tail_normalized'] for e in group]))} for a in ARMS}
            boundaries.append(boundary)
            torch.save({'protocol_sha256':protocol_sha,'models':{a:m.state() for a,m in models.items()}},OUT/f'states_t{event}.pt')
            save('per_image_results.json',records);save('boundaries.json',boundaries)
            save('update_trace.json',trace);save('state_chain.json',chain);save('access_log.json',reader.access)
            print('boundary_complete',event,'occupancy',{a:m.n for a,m in models.items()},flush=True)
        del x
    require(all(sha(ROOT/n)==h for n,h in p['preserved_sha256'].items()),'protected source changed')
    resources={'methods':{a:{**v,'update_ms_per_arrival':1000*v['update_seconds']/350,
        'inference_ms_per_image':1000*v['read_seconds']/v['read_images']} for a,v in timing.items()},
        'shared':{'runtime_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'checkpoint_file_bytes':(ROOT/'checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth').stat().st_size,
                  'clean_cache_file_bytes':sum((FEATURES/n).stat().st_size for n in CACHE_SHA),'per_image_input_bytes':16*768*4,
                  'scale_evaluator_max_float_count':150,'timing_scope':'CPU support heads only; update includes selector guard tracing, excludes SHA ledger checks; read includes one distance call, excludes repeated-read checks'}}
    save('resources.json',resources);save('causal_checks.json',checks)
    print('MEASUREMENTS_COMPLETE',resources['shared']['runtime_seconds'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args();configure()
    try:prepare() if args.mode=='prepare' else run()
    except Exception as error:
        save('execution_error.json',{'mode':args.mode,'type':type(error).__name__,'error':str(error)});raise


if __name__=='__main__':main()
