"""Causal completed-record reservoir; two private RNGs and exact query ordering."""
from copy import deepcopy
import random
import torch
from hope_cad.contract import digest
from hope_cad.models.primitives.state import tensor_digest
from hope_cad.models.validation import keys, integer


def envelope(record, label):
    # record already owns detached clones; the envelope and selected IDs do not clone tensors.
    value = dict(completed_label_index=label, reward_observed_at=record['decision_cursor']+1, record=record)
    value['payload_sha256'] = payload_digest(value)
    return value


def payload_digest(value):
    r=value['record']
    return digest(dict(tensors=tensor_digest([{'tokens':r['tokens'],'stats':r['stats']}]),
        fields={k:v for k,v in r.items() if not torch.is_tensor(v)},
        completed_label_index=value['completed_label_index'],reward_observed_at=value['reward_observed_at']))


class CompletedHistory:
    schema='completed_utility_history_v1'

    def __init__(self, seed):
        self.buffer=[];self.seen=0;self.reservoir_rng=random.Random(seed+96136)
        self.query_rng=random.Random(seed+96137);self.selection=None

    def begin(self, label, observed):
        if self.selection is not None or self.seen!=label-1 or self.seen%8:raise ValueError('Invalid history block start')
        if bool(self.buffer)!=(self.seen>0) or len(self.buffer)!=min(32,self.seen):raise ValueError('Invalid history occupancy')
        slots=self.query_rng.sample(range(len(self.buffer)),2) if self.seen else []
        self.selection=dict(cutoff=label-1,first_credit_observed_at=observed,slots=slots,
            ids=[self.buffer[i]['record']['decision_identity'] for i in slots],
            payload_sha256=[self.buffer[i]['payload_sha256'] for i in slots],
            query_rng_after=digest(self.query_rng.getstate()))

    def selected(self):
        if self.selection is None:raise ValueError('No active query snapshot')
        return [self.buffer[i] for i in self.selection['slots']]

    def commit(self, records, labels):
        if self.selection is None or len(records)!=8 or self.seen+8!=labels:raise ValueError('History commit clock')
        for r in records:
            n=self.seen+1; item=envelope(r,n)
            if n<=32:self.buffer.append(item)
            else:
                j=self.reservoir_rng.randrange(n)
                if j<32:self.buffer[j]=item
            self.seen=n
        self.selection=None

    def state_dict(self):
        return dict(schema=self.schema,seen=self.seen,buffer=self.buffer,selection=self.selection,
            reservoir_rng=self.reservoir_rng.getstate(),query_rng=self.query_rng.getstate())

    def load(self, state, labels, records, validate):
        keys(state,self.state_dict(),'completed history')
        if state['schema']!=self.schema:raise ValueError('History schema changed')
        B,r=divmod(labels,8);integer(state['seen'],'history seen')
        if state['seen']!=8*B or len(state['buffer'])!=min(32,8*B):raise ValueError('History clock/occupancy')
        ids=[];indices=[];cursors=[]
        for item in state['buffer']:
            keys(item,('completed_label_index','reward_observed_at','record','payload_sha256'),'history envelope')
            validate(item['record']);i=integer(item['completed_label_index'],'history label',1)
            observed=integer(item['reward_observed_at'],'reward time',1)
            if i>8*B or observed!=item['record']['decision_cursor']+1 or payload_digest(item)!=item['payload_sha256']:
                raise ValueError('History payload/eligibility changed')
            ids.append(item['record']['decision_identity']);indices.append(i);cursors.append(item['record']['decision_cursor'])
        if any(len(set(v))!=len(v) for v in (ids,indices,cursors)):raise ValueError('Duplicate history records')
        ordered=sorted(zip(indices,cursors))
        if [v for _,v in ordered]!=sorted(cursors):raise ValueError('History label/cursor order')
        current_ids=[v['decision_identity'] for v in records]
        if set(ids)&set(current_ids):raise ValueError('History/current overlap')
        # Current records are chronological, and all committed observations precede this block.
        if records and cursors and max(cursors)>=records[0]['decision_cursor']:raise ValueError('Future history')
        sel=state['selection']
        if (sel is not None)!=bool(r):raise ValueError('Partial query selection mismatch')
        for name in ('reservoir_rng','query_rng'):
            temporary=random.Random();temporary.setstate(state[name])
        if sel is not None:
            keys(sel,('cutoff','first_credit_observed_at','slots','ids','payload_sha256','query_rng_after'),'query snapshot')
            if sel['cutoff']!=8*B or sel['first_credit_observed_at']!=records[0]['decision_cursor']+1:raise ValueError('Query cutoff')
            slots=sel['slots']
            if not isinstance(slots,list) or len(slots)!=(2 if B else 0) or len(set(slots))!=len(slots):raise ValueError('Query cardinality')
            for i in slots:
                integer(i,'query slot')
                if i>=len(ids) or state['buffer'][i]['reward_observed_at']>=sel['first_credit_observed_at']:raise ValueError('Ineligible historical query')
            if sel['ids']!=[ids[i] for i in slots] or sel['payload_sha256']!=[state['buffer'][i]['payload_sha256'] for i in slots] or sel['query_rng_after']!=digest(state['query_rng']):raise ValueError('Saved query selection changed')
        self.seen=state['seen'];self.buffer=deepcopy(state['buffer']);self.selection=deepcopy(sel)
        self.reservoir_rng.setstate(state['reservoir_rng']);self.query_rng.setstate(state['query_rng'])
