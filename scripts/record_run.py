"""Publish all terminal execution receipts, preserving incomplete/failed runs."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nested_replay_cad.provenance import digest
from summarize import summarize,render_results

def write_once(path,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise FileExistsError('Closed receipt differs; preserve it and publish a separate analysis: '+str(path))
    else:path.write_bytes(raw)

def export_run(run,destination,version=None,analysis_id=None):
    run=Path(run).resolve();destination=Path(destination)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run.name):raise ValueError('Unsafe run ID')
    if analysis_id and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',analysis_id):raise ValueError('Unsafe analysis ID')
    if version and not re.fullmatch(r'v[0-9]+',version):raise ValueError('Unsafe version')
    record_path=run/'receipts/receipt.json' if not version else run/'receipts'/version/'provider-status.json'
    if not record_path.exists():raise ValueError('No closed local receipt or provider terminal status')
    record=json.loads(record_path.read_text());status=str(record.get('state',record.get('status','')))
    if not (status in ('failed','cancelled','complete_review_pending') or status.endswith(('.COMPLETE','.ERROR','.CANCEL_ACKNOWLEDGED'))):raise ValueError('Run is not terminal')
    base=run/'outputs'/version if version else run/'outputs'
    if version and (base/'B319').exists():base=base/'B319'
    elif not base.exists() and not version:base=run
    execution_receipt=run/'receipts'/version/'receipt.json' if version else record_path
    execution=json.loads(execution_receipt.read_text()) if execution_receipt.exists() else record
    result=None;analysis_error=None
    if all((base/arm/'arms'/arm/'study.json').is_file() for arm in ('er_uniform','a0_tail','a0_tail_ctx')):
        try:result=summarize(base)
        except (ValueError,KeyError,TypeError) as error:analysis_error=repr(error)
    files={p.relative_to(run).as_posix():digest(p) for p in (record_path,*sorted(base.glob('*/arms/*/study.json'))) if p.is_file()}
    bundles={}
    for p in sorted((run/'package').rglob('*.zip')):
        bundles[p.relative_to(run).as_posix()]={'sha256':digest(p),'bytes':p.stat().st_size}
    sources=[]
    for p in sorted(base.glob('*/config.json')):
        config=json.loads(p.read_text());sources.append({'arm':config.get('arm'),'source_sha256':config.get('source_sha256'),'recipe_sha256':config.get('recipe_sha256'),'runtime':config.get('runtime')})
    identifier=run.name+('-'+version if version else '')+('-analysis-'+analysis_id if analysis_id else '')
    analysis_source=Path(__file__).with_name('summarize.py');analysis_sha=digest(analysis_source)
    retained=run/'audit/analysis'/analysis_sha/'summarize.py'
    retained.parent.mkdir(parents=True,exist_ok=True)
    if not retained.exists():retained.write_bytes(analysis_source.read_bytes())
    elif digest(retained)!=analysis_sha:raise ValueError('Retained analysis source mismatch')
    receipt={'schema':'experiment_receipt_v1','experiment':'B319','run_id':identifier,'execution':execution,'provider_status':record if version else None,'local_artifact_root':str(run),
        'provider':record.get('kernel'),'files':files,'retained_source_and_archives':bundles,'arm_sources':sources,
        'matrix_complete':result is not None,'analysis_error':analysis_error,'scientific_acceptance':False,'analysis_source_sha256':analysis_sha}
    target=destination/identifier
    if result is not None:write_once(target/'metrics.json',result)
    write_once(target/'receipt.json',receipt)
    if result is not None:
        text=render_results(result);path=target/'results.md'
        if path.exists() and path.read_text()!=text:raise FileExistsError(path)
        if not path.exists():path.write_text(text)
    print('RECEIPT_RECORDED',identifier,'matrix_complete='+str(result is not None),flush=True)
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--version');p.add_argument('--analysis-id');a=p.parse_args()
    export_run(a.run,ROOT/'experiments/B319/runs',a.version,a.analysis_id)
