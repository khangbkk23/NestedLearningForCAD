"""Opt-in host timing and synchronized phase memory; never part of learned state."""
from contextlib import contextmanager,nullcontext
from contextvars import ContextVar
from functools import wraps
import time
import torch
from hope_cad.runtime.system import atomic_json,resource_usage

_current=ContextVar('research_measurement',default=None)


def span(name):
    meter=_current.get()
    return meter.component(name) if meter is not None else nullcontext()


def timed(name):
    """Host inclusive/exclusive scopes, with no extra forward or per-call CUDA sync."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args,**kwargs):
            meter=_current.get()
            if meter is None:return fn(*args,**kwargs)
            with meter.component(name):return fn(*args,**kwargs)
        return wrapped
    return decorate


class Measurement:
    def __init__(self,work,attempt,device):
        self.path=work/'telemetry'/(attempt+'.json');self.device=device
        self.value=dict(schema='hope_research_measurement_v1',attempt=attempt,phases=[],
            scope='Host component times include dispatch/synchronization already performed by algorithm; not CUDA kernel times. Phase wall includes explicit boundary synchronization. Host peak is process lifetime; GPU peaks reset per phase and include backbone/allocator, not equal-memory matching. Nested inclusive times must not be summed.')
        self.active=None;self.stack=[]

    def sync(self):
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self,name,job=None):
        if self.active is not None:raise ValueError('Measurement phases cannot overlap')
        self.sync()
        if str(self.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(self.device)
        row=dict(phase=name,job=job,components={},status='running');self.active=row
        if str(self.device).startswith('cuda'):
            row['gpu_baseline_bytes']=dict(allocated=torch.cuda.memory_allocated(self.device),reserved=torch.cuda.memory_reserved(self.device))
        token=_current.set(self);started=time.monotonic()
        try:
            yield
            row['status']='complete'
        except BaseException as exc:
            row['status']='paused' if type(exc).__name__=='Pause' else 'failed'
            raise
        finally:
            try:
                self.sync();row['wall_seconds']=time.monotonic()-started
                row['resources']=resource_usage(self.device)
            except BaseException as exc:
                row['status']='measurement_failed';row['measurement_error']=repr(exc)
                raise
            finally:
                _current.reset(token);self.active=None
                self.value['phases'].append(row);atomic_json(self.value,self.path)

    @contextmanager
    def component(self,name):
        started=time.monotonic();frame=[0.];self.stack.append(frame)
        try:yield
        finally:
            elapsed=time.monotonic()-started;self.stack.pop()
            if self.stack:self.stack[-1][0]+=elapsed
            row=self.active['components'].setdefault(name,dict(calls=0,inclusive_host_seconds=0.,exclusive_host_seconds=0.))
            row['calls']+=1;row['inclusive_host_seconds']+=elapsed
            row['exclusive_host_seconds']+=max(0.,elapsed-frame[0])
