"""Revision-specific slow objective; actual fast trajectory remains the v5.1 rule."""
from copy import deepcopy
import torch
from hope_cad.contract import design_digest,digest
from hope_cad.models.learner import NestedUtilityLearner, prediction, fast_step, fast_step_from_prediction, norm
from hope_cad.models.history import CompletedHistory
from hope_cad.models.validation import keys, integer, tree_finite
from hope_cad.models.primitives.state import finite_tensors
from hope_cad.runtime.measurement import timed,span

NEURAL_VARIANTS=('R_EXACT','R_FO','H32_EXACT','H32_FO','H32_DIRECT','FAST_ONLY')


@timed('functional_unroll_and_queries')
def query_objective(model,initial,support,queries,variant):
    if len(support)!=4 or len(queries)!=4:raise ValueError('Four support/four queries required')
    phi={k:v.detach().clone().requires_grad_(True) for k,v in initial.items()}
    if variant!='H32_DIRECT':
        for r in support:phi,*_=fast_step(model,phi,r,exact=variant.endswith('EXACT'))
    losses=torch.stack([.5*(prediction(model,phi,r)[0]-r['target']).square() for r in queries])
    return losses.mean(),losses


class HistoricalUtilityLearner(NestedUtilityLearner):
    def __init__(self,model,variant,seed,method_digest):
        if variant not in NEURAL_VARIANTS:raise ValueError('Unknown consolidation variant')
        mode='FAST_ONLY' if variant=='FAST_ONLY' else ('FIRST_ORDER' if variant.endswith('_FO') else 'NESTED')
        super().__init__(model,mode)
        self.variant,self.seed,self.method_digest=variant,seed,method_digest
        self.history=CompletedHistory(seed);self.last_episode=None;self.debug_queries=None
        self.last_decision_cursor=-1

    def query_records(self):
        historical=self.history.selected()
        if self.variant.startswith('H32') and historical:
            return self.records[6:8]+[h['record'] for h in historical],['recent','recent','history','history']
        return self.records[4:8],['fallback']*4 if not historical else ['recent']*4

    @timed('learner_fast_and_slow')
    def observe(self,record):
        self.validate_record(record)
        if record['decision_cursor']<=self.last_decision_cursor:raise ValueError('Nonchronological completed record')
        record={k:v.detach().clone() if torch.is_tensor(v) else v for k,v in record.items()}
        if not self.records:
            self.history.begin(self.labels+1,record['decision_cursor']+1)
            self.block_initial={k:v.detach().clone() for k,v in self.phi.items()}
        before,denom=prediction(self.model,self.phi,record)
        if float(denom.detach())!=record['decision_d']:raise ValueError('Factual denominator changed')
        updated,loss,_,grads=fast_step_from_prediction(self.phi,record['target'],before,denom,exact=False)
        label=self.labels+1
        details=dict(labels=label,denominator=float(denom.detach()),prediction_before=float(before.detach()),
            fast_loss=float(loss.detach()),fast_gradient_norm=norm(grads),
            fast_step_norm=norm(updated[k]-self.phi[k] for k in self.phi),
            fast_group_step_norms={p:norm(updated[k]-self.phi[k] for k in self.phi if k.startswith(p)) for p in ('cms.','readout.')})
        self.phi={k:v.detach().requires_grad_(True) for k,v in updated.items()}
        self.labels=label;self.fast_updates+=1;self.last_decision_cursor=record['decision_cursor'];self.records.append(record)
        if label in (1,33,65,97,129,161,193,225):
            with torch.no_grad():after=float(prediction(self.model,self.phi,record)[0])
            error=abs(float(before.detach())-record['target'])
            details['response']=dict(prediction_after=after,small_error=error<=1e-12,response_gain=None if error<=1e-12 else (error-abs(after-record['target']))/error)
        if len(self.records)==8:
            queries,sources=self.query_records();snapshot=deepcopy(self.history.selection)
            episode_model=self._cpu(self.model.state_dict()) if label==40 else None
            probe=bool(self.optimizer) and (label==40 or (self.slow_updates+1)%32==0)
            with torch.no_grad():pre=[float(prediction(self.model,self.phi,q)[0]) for q in queries] if probe else None
            historical_gradient=None
            if self.optimizer:
                self.optimizer.zero_grad(set_to_none=True)
                outer,losses=query_objective(self.model,self.block_initial,self.records[:4],queries,self.variant)
                params=self.model.slow_parameters()
                if sources[-1]=='history' and (self.slow_updates==1 or probe):
                    hg=torch.autograd.grad(losses[2:].mean(),[p for _,p in params],retain_graph=True)
                    finite_tensors(hg,'historical query gradient');historical_gradient=norm(hg)
                with span('slow_backward_and_optimizer'):
                    outer.backward()
                    if any(p.grad is None for _,p in params):raise RuntimeError('Disconnected active slow parameter')
                    finite_tensors((p.grad for _,p in params),'slow gradient')
                    norms={n:norm([p.grad]) for n,p in params}
                    total=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True)
                    previous={n:p.detach().clone() for n,p in params}
                    self.optimizer.step();tree_finite(self.optimizer.state_dict());finite_tensors((p for _,p in params),'slow parameters')
                self.slow_updates+=1
                self.last_slow=dict(update=self.slow_updates,loss=float(outer.detach()),gradient_norm=float(total),group_norms=norms,
                    step_norms={n:norm([p-previous[n]]) for n,p in params},support_ids=[r['decision_identity'] for r in self.records[:4]],query_ids=[r['decision_identity'] for r in queries])
                details['slow']=deepcopy(self.last_slow)
                with torch.no_grad():post=[float(prediction(self.model,self.phi,q)[0]) for q in queries] if probe else None
                change=[abs(a-b) for a,b in zip(pre,post)] if probe else None
                functional_probe=dict(mean_abs_delta=sum(change)/4,max_abs_delta=max(change),phi='actual_phi8_fixed',before=pre,after=post) if probe else None
            else:losses=None;functional_probe=dict(status='not_applicable_no_slow_optimizer')
            self.last_episode=dict(variant=self.variant,completed_labels=label,snapshot=snapshot,
                support_ids=[r['decision_identity'] for r in self.records[:4]],query_ids=[r['decision_identity'] for r in queries],query_sources=sources,
                query_ages=[label-(label-7+next(i for i,r in enumerate(self.records) if r is q) if s!='history' else h['completed_label_index']) for q,s,h in
                    [(q,s,next((x for x in self.history.selected() if x['record'] is q),{})) for q,s in zip(queries,sources)]],
                recent_loss=float(losses[:2].detach().mean()) if losses is not None and sources[-1]=='history' else (float(losses.detach().mean()) if losses is not None else None),
                historical_loss=float(losses[2:].detach().mean()) if losses is not None and sources[-1]=='history' else None,
                historical_gradient_norm=historical_gradient,gradient_policy='direct' if self.variant=='H32_DIRECT' else ('FO' if self.variant.endswith('_FO') else 'exact'),
                functional_support_steps=0 if self.variant=='H32_DIRECT' else 4,
                zero_gradient=bool(self.last_slow['gradient_norm']==0) if self.optimizer else None,
                clipped=bool(self.last_slow['gradient_norm']>1) if self.optimizer else None,functional_probe=functional_probe,
                history_count=len(self.history.buffer),history_slot_ids=[x['record']['decision_identity'] for x in self.history.buffer],
                reservoir_rng_digest=digest(self.history.reservoir_rng.getstate()),query_rng_digest=digest(self.history.query_rng.getstate()))
            details['episode']=deepcopy(self.last_episode)
            if label==40:
                self.saved_meta_episode=dict(labels_inclusive=[33,40],phi_start=self._cpu(self.block_initial),records=self._cpu(self.records),model=episode_model,mode=self.mode)
                self.debug_queries=dict(queries=self._cpu(queries),sources=sources,snapshot=snapshot,variant=self.variant)
            self.history.commit(self.records,label)
            self.records,self.block_initial=[],None
        details.update(fast_updates=self.fast_updates,slow_updates=self.slow_updates,completed_label_blocks=label//8)
        return details

    def state_dict(self):
        return self._cpu(dict(schema='hope_hq_learner_v1',variant=self.variant,method_digest=self.method_digest,
            base=super().state_dict(),history=self.history.state_dict(),last_episode=self.last_episode,
            debug_queries=self.debug_queries,last_decision_cursor=self.last_decision_cursor))

    def load_state_dict(self,state):
        keys(state,self.state_dict(),'HQ learner')
        if state['schema']!='hope_hq_learner_v1' or state['variant']!=self.variant or state['method_digest']!=self.method_digest:raise ValueError('HQ learner identity changed')
        tree_finite(state)
        base=NestedUtilityLearner(deepcopy(self.model),self.mode);base.load_state_dict(state['base'])
        hist=CompletedHistory(self.seed);hist.load(state['history'],base.labels,base.records,base.validate_record)
        if type(state['last_decision_cursor']) is not int or (base.labels==0)!=(state['last_decision_cursor']==-1):raise ValueError('Completed cursor mismatch')
        if any(r['decision_cursor']>state['last_decision_cursor'] for r in base.records+[x['record'] for x in hist.buffer]):raise ValueError('Future record cursor')
        if bool(base.labels//8)!=(state['last_episode'] is not None) or (state['debug_queries'] is not None)!=(base.labels>=40):raise ValueError('Episode trace clock')
        if state['last_episode'] is not None and state['last_episode']['completed_labels']!=base.labels//8*8:raise ValueError('Episode label clock')
        if state['last_episode'] is not None:
            ep=state['last_episode']
            if ep['variant']!=self.variant or len(ep['query_ids'])!=4 or len(set(ep['support_ids']+ep['query_ids']))!=8:
                raise ValueError('Episode query provenance')
            if base.last_slow is not None and (ep['query_ids']!=base.last_slow['query_ids'] or ep['support_ids']!=base.last_slow['support_ids']):raise ValueError('Episode/optimizer trace mismatch')
        dbg=state['debug_queries']
        if dbg is not None:
            keys(dbg,('queries','sources','snapshot','variant'),'debug queries')
            if dbg['variant']!=self.variant or len(dbg['queries'])!=4 or len(dbg['sources'])!=4:raise ValueError('Debug variant/query shape')
            for q in dbg['queries']:base.validate_record(q)
            ids=[r['decision_identity'] for r in dbg['queries']]
            if len(set(ids))!=4 or set(ids)&{r['decision_identity'] for r in base.saved_meta_episode['records'][:4]}:raise ValueError('Debug support/query overlap')
            expected_sources=['recent','recent','history','history'] if self.variant.startswith('H32') else ['recent']*4
            if dbg['sources']!=expected_sources:raise ValueError('Debug query policy changed')
            for q,source in zip(dbg['queries'],dbg['sources']):
                if source=='history' and (q['decision_identity'] not in dbg['snapshot']['ids'] or q['decision_cursor']+1>=dbg['snapshot']['first_credit_observed_at']):raise ValueError('Debug future query')
        self.__dict__.update(base.__dict__);self.history=hist
        device=next(self.model.parameters()).device
        for item in hist.buffer:
            item['record']['tokens']=item['record']['tokens'].to(device);item['record']['stats']=item['record']['stats'].to(device)
        self.last_episode=deepcopy(state['last_episode']);self.debug_queries=deepcopy(state['debug_queries']);self.last_decision_cursor=state['last_decision_cursor']

    def validation_records(self):
        return self.records+[i['record'] for i in self.history.buffer]+(self.saved_meta_episode['records'] if self.saved_meta_episode else [])+(self.debug_queries['queries'] if self.debug_queries else [])
