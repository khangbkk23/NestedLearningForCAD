"""Cumulative GPU ledger. Interrupted reservations stay charged until reviewed."""
import math
import time
import uuid
from pathlib import Path
from hope_cad.runtime.system import atomic_json
from hope_cad.contract import digest,canonical


class Budget:
    def __init__(self,root,stage,hours,external_seconds=0.,allocation=None):
        self.path=Path(root)/('budget_'+stage+'.json');self.stage=stage
        if not math.isfinite(hours) or not 0<hours<=4:raise ValueError('Research sessions must be in(0,4]hours')
        if not math.isfinite(external_seconds) or external_seconds<0:raise ValueError('External billed time must be nonnegative')
        cap={'B':12*3600,'C':24*3600}.get(stage)
        if stage=='D':
            if not allocation or allocation.get('stage')!='D' or allocation.get('authorized') is not True:raise ValueError('Extension needs explicit allocation record')
            cap=allocation.get('gpu_wall_seconds_cap')
        if allocation:
            if allocation.get('authorized') is not True or allocation.get('stage')!=stage:raise ValueError('Allocation not approved for this stage')
            if 'USD_cap' in allocation:
                usd=allocation['USD_cap'];rate=allocation.get('USD_per_hour')
                if not all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in (usd,rate)):raise ValueError('Invalid cost cap/hourly rate')
                cap=min(cap,usd/rate*3600)
        if not isinstance(cap,(int,float)) or not math.isfinite(cap) or cap<=0:raise ValueError('Invalid cumulative allocation')
        self.binding=canonical(dict(stage=stage,cap_seconds=cap,allocation=allocation))
        self.value=__import__('json').loads(self.path.read_text()) if self.path.exists() else dict(binding=self.binding,attempts=[])
        if self.value['binding']!=self.binding:raise ValueError('Budget allocation changed')
        charged=sum(x['charged_seconds'] for x in self.value['attempts'])
        remaining=cap-charged-external_seconds
        if remaining<=180:raise ValueError('budget_unresolved: cumulative allocation exhausted')
        seconds=min(hours*3600,remaining)
        self.started=time.monotonic();self.deadline=self.started+seconds-180
        self.entry=dict(id=uuid.uuid4().hex,status='reserved_incomplete',external_billed_seconds=external_seconds,
            charged_seconds=seconds+external_seconds,reserved_invocation_seconds=seconds)
        self.value['attempts'].append(self.entry);atomic_json(self.value,self.path)

    def begin_export(self):
        # Remain fully reserved through ZIP compression, verification and hashing.
        self.entry.update(status='export_incomplete',before_export_seconds=time.monotonic()-self.started)
        atomic_json(self.value,self.path)

    def finish(self,evidence):
        if self.entry['status']!='export_incomplete':raise ValueError('Cannot close a reservation before verified export')
        elapsed=time.monotonic()-self.started
        # ZIP contains the open reservation. This hash-bound sidecar closes it
        # without changing the already verified archive (no self-referential hash).
        self.entry.update(status='closed_after_verified_export',elapsed_seconds=elapsed,
            export_seconds=elapsed-self.entry['before_export_seconds'],
            publication_reserve_seconds=180,charged_seconds=elapsed+180+self.entry['external_billed_seconds'])
        reconciliation=dict(schema='hope_research_accounting_v1',archive=evidence,
            binding=self.binding,attempt=dict(self.entry),scope='Includes verified export; final publication reserve180s, not an invoice')
        atomic_json(reconciliation,Path(evidence['path']).with_suffix('.accounting.json'))
        atomic_json(self.value,self.path)
        return reconciliation

    def remaining(self):
        return self.binding['cap_seconds']-sum(x['charged_seconds'] for x in self.value['attempts'])
