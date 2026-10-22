"""Prepare known normals and run isolated PREP/R1/O1/O2/R2 with failure export."""
import json
import sys
import time
import traceback
from pathlib import Path

from training.runtime import atomic_json
from training.support_throughput_v5_1.contract import ROOT, Evidence, contract, verify_parent, sha
from training.cadic_cost_validation import verify_delivery, summarize, PASSES, SCHEMA
from notebook_support.support_throughput_v5_1.bootstrap import prepare, extract_window, verify_window, one
from notebook_support.common.process import stream_process
from notebook_support.common.evidence import export_archive


def run_notebook(c):
    work = Path(c['work']); work.mkdir(parents=True, exist_ok=True)
    status = 'failed_preserve_evidence'
    try:
        source = verify_delivery()
        if c.get('RESUME_OVERRIDE'):
            raise ValueError('No automatic retry/resume in this bounded experiment; preserve partial output for review')
        if (work/'attempt_ledger.json').exists():
            raise ValueError('An attempt already exists; preserve output and request review, do not replay')
        # Durable reservation before expensive bootstrap. A failed PREP is not auto-retried.
        deadline = c['started']+180*60-180
        ledger = dict(source_digest=source, max_arrivals=243, reserved=[], automatic_retries=False)
        atomic_json(ledger, work/'attempt_ledger.json')
        prepared = prepare(dict(c, RESUME_OVERRIDE=None))
        historical = json.loads((ROOT/'notebook_support/cadic_cost_nested_reference.json').read_text())
        if prepared['runtime'] != historical['runtime']:
            raise ValueError('Runtime differs from reviewed historical NESTED comparison')
        with Evidence(prepared['parent']) as e:
            rows = verify_parent(e, contract())['rows'][:195]
        inputs = Path(c['inputs'])
        # Original prepare extracts only16. Build CADIC's own prefix from registered train normals.
        dataset = Path(prepared['dataset'])
        try:
            verify_window(dataset, rows)
        except FileNotFoundError:
            archive = one([*inputs.rglob('*mvtec*.tar.xz'), *inputs.rglob('*mvtec*.tar.gz')], 'MVTec tar')
            print('[CADIC COST] extracting195 permitted normals; no test/GT', flush=True)
            dataset = extract_window(archive, c['assets'], rows)
            verify_window(dataset, rows)
        atomic_json(dict(rows=rows, prefix=[0,179], measured_window=[179,195], future_pixels=False), work/'permitted_rows.json')
        values = {}
        for name in ('PREP', *PASSES):
            if time.monotonic() >= deadline:
                status = 'incomplete_budget'; break
            count = 179 if name == 'PREP' else 16
            ledger['reserved'].append(dict(pass_id=name, arrivals=count, state='reserved'))
            atomic_json(ledger, work/'attempt_ledger.json')
            cfg = dict(prepared, dataset=str(dataset), work=str(work), deadline=deadline)
            cfg['pass'] = name
            path = work/'configs'/(name+'.json'); atomic_json(cfg, path)
            print('[CADIC COST START]', name, 'remaining seconds', round(deadline-time.monotonic()), flush=True)
            worker_started = time.monotonic()
            code = stream_process([sys.executable, '-B', '-m', 'training.cadic_cost_validation', '--config', str(path)],
                work/'logs'/(name+'.log'), cwd=ROOT)
            ledger['reserved'][-1]['worker_wall_seconds'] = time.monotonic()-worker_started
            atomic_json(ledger, work/'attempt_ledger.json')
            folder = work/'passes'/name
            if code:
                raise RuntimeError('Worker failed: '+name)
            if not (folder/'complete.json').exists():
                status = 'incomplete_budget'; break
            m = json.loads((folder/'measurement.json').read_text())
            seal = json.loads((folder/'complete.json').read_text())
            if seal['status'] != 'complete' or not seal['features_exact_to_reviewed_parent'] or seal['measurement_sha256'] != sha(folder/'measurement.json') or seal['source_digest'] != source or seal['predecessor_sha256'] != sha(work/'predecessor179.pt'):
                raise ValueError('Worker seal mismatch')
            if name != 'PREP' and seal['full_state_exact_to_R1'] is not True:
                raise ValueError('Missing full state parity seal')
            expected = list(range(179)) if name == 'PREP' else list(range(179,195))
            if m['attempted_event_ids'] != expected:
                raise ValueError('Wrong attempt chronology')
            ledger['reserved'][-1]['state'] = 'complete'; atomic_json(ledger, work/'attempt_ledger.json')
            if name != 'PREP': values[name] = m
        else:
            nested = json.loads((ROOT/'notebook_support/cadic_cost_nested_reference.json').read_text())
            result = summarize(values, nested)
            result['prefix_preparation'] = json.loads((work/'passes/PREP/measurement.json').read_text())
            result['source_digest'] = source
            result['worker_wall_seconds'] = {v['pass_id']:v['worker_wall_seconds'] for v in ledger['reserved']}
            atomic_json(result, work/'summary.json')
            status = result['status']
    except BaseException as exc:
        traceback.print_exc(); atomic_json(dict(error=repr(exc)), work/'failure.json')
    finally:
        atomic_json(dict(status=status, reportable=False, adopted=False, resume_requires_review=True), work/'handoff.json')
        atomic_json(dict(elapsed_before_export_seconds=time.monotonic()-c['started'],
            budget_seconds=10800, export_reserve_seconds=180), work/'execution.json')
        receipt = export_archive(work, c['output'], schema=SCHEMA)
        print('[CADIC COST EXPORTED]', json.dumps(receipt), flush=True)
        try:
            from IPython.display import display, FileLink
            display(FileLink(receipt['path']))
        except ImportError:
            pass
    if status != 'complete_pending_review':
        raise RuntimeError('Not complete: '+status+'; preserve ZIP/log, no automatic retry')
    print('[CADIC COST STATUS]', status, flush=True)
    return receipt
