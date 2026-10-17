"""Synthetic only: no feature cache, normal probe or image file is opened."""
import copy
import numpy as np
import pytest
import torch

from exps.task_p.p0_normal_support import (
    ARMS, BUDGET, Support, distances, fingerprint, serialized_bytes,
)
from exps.task_p.p0_protocol import Reader, configure, validate_roles


@pytest.fixture(autouse=True)
def deterministic():
    configure();torch.manual_seed(7205)


def test_raw_euclidean_against_independent_difference_overflow_cancellation_ties():
    for scale in (1.,1e12,1e20):
        x=torch.randn(9,7)*scale;y=torch.cat((x[:3],torch.randn(8,7)*scale))
        ref=(x.double()[:,None]-y.double()[None]).square().sum(-1).sqrt()
        assert torch.allclose(distances(x,y),ref,atol=1e-10,rtol=1e-12)
        assert torch.equal(distances(x[:3],x[:3]).diagonal(),torch.zeros(3,dtype=torch.float64))
    m=Support('RESERVOIR',capacity=3,dim=1);m.update(torch.tensor([[0.],[0.],[3.]]),torch.arange(3))
    assert m.score(torch.tensor([[0.]]))[1].item()==0


def test_incremental_nearest_graph_matches_full_oracle_after_every_write():
    m=Support('STREAM_GREEDY',capacity=17,dim=7)
    for i in range(45):
        slot=m.n if m.n<m.capacity else (i*7)%m.capacity
        m._put(slot,torch.randn(7),i)
        if m.n<2:continue
        d=distances(m.bank[:m.n],m.bank[:m.n]);d.fill_diagonal_(torch.inf)
        v,idx=d.min(1)
        assert torch.allclose(m.nn_distance[:m.n],v,atol=1e-12,rtol=1e-12)
        assert torch.equal(m.nn_index[:m.n],idx)


def test_greedy_coverage_rule_rejects_redundancy_and_acquires_far_mode():
    m=Support('STREAM_GREEDY',capacity=3,dim=1)
    m.update(torch.tensor([[0.],[1.],[10.]]),torch.arange(3))
    d=m.update(torch.tensor([[.01]]),torch.tensor([3]));assert d['replaced']==0
    d=m.update(torch.tensor([[20.]]),torch.tensor([4]));assert d['replaced']==1
    assert m.score(torch.tensor([[20.]]))[0].item()==0
    assert m.score(torch.tensor([[10.]]))[0].item()==0


def witness_fixture():
    m=Support('WITNESS_PROTECTED',capacity=3,dim=1,witness_capacity=2)
    for i,v in enumerate((0.,.1,10.)):m._put(i,torch.tensor([v]),i)
    m.seen=3;m.witnesses[:,0]=torch.tensor([0.,.001]);m.witness_ids[:]=torch.tensor([0,1]);m.wn=2;m.wseen=3
    return m


def test_witness_guard_off_changes_actual_synthetic_decision_vs_greedy():
    protected=witness_fixture();unguarded=copy.deepcopy(protected)
    greedy=Support('STREAM_GREEDY',capacity=3,dim=1)
    for i,v in enumerate((0.,.1,10.)):greedy._put(i,torch.tensor([v]),i)
    x=torch.tensor([[20.]]);ids=torch.tensor([3])
    dp=protected.update(x,ids);du=unguarded.update(x,ids,guard_enabled=False);greedy.update(x,ids)
    assert dp['decisions'][0]['selected_slot']==1
    assert du['decisions'][0]['selected_slot']==0
    assert int(greedy.ids[0])==3 and dp['guard_effect_attempts']==1
    assert protected.score(torch.tensor([[0.]]))[0].item()==0
    assert unguarded.score(torch.tensor([[0.]]))[0].item()>0


def test_witness_costs_equal_literal_tentative_replacements():
    m=witness_fixture();x=torch.tensor([[20.],[21.]])
    before=m.bank[:m.n].clone();w=m.witnesses[:m.wn].clone()
    stats=m.update(x,torch.tensor([3,4]))
    first=stats['decisions'][0];slot=first['selected_slot'];chosen=first['candidate_row']
    tentative=before.clone();tentative[slot]=x[chosen]
    g=(distances(x,before).min(1).values-distances(x,tentative).min(1).values).mean()
    inc=(distances(w,tentative).min(1).values-distances(w,before).min(1).values).clamp_min(0)
    assert abs(first['gain']-float(g))<1e-12
    assert abs(first['witness_cost']-float(inc.mean()))<1e-12
    assert abs(first['tail_cost']-float(torch.quantile(inc,.9)))<1e-12


def test_degenerate_witness_scale_fails_closed_without_changing_score_bank():
    m=witness_fixture();m.witnesses.zero_()
    before=fingerprint({'bank':m.bank,'ids':m.ids})
    d=m.update(torch.tensor([[20.]]),torch.tensor([4]))
    assert d['replaced']==0 and d['witness_scale'] is None
    assert before==fingerprint({'bank':m.bank,'ids':m.ids})


def test_witnesses_are_not_deployment_support():
    m=witness_fixture();m.witnesses[:]=torch.tensor([[50.],[60.]])
    assert m.score(torch.tensor([[50.]]))[0].item()==40


def test_algorithm_r_uniformity_and_seed_replay():
    counts=np.zeros(12)
    for seed in range(400):
        m=Support('RESERVOIR',capacity=4,dim=1);m.rng.manual_seed(seed)
        m.update(torch.arange(12,dtype=torch.float32)[:,None],torch.arange(12))
        counts[m.ids.numpy()]+=1
    assert np.max(np.abs(counts-400/3))<5*np.sqrt(400*(1/3)*(2/3))
    a=Support('RESERVOIR',capacity=4,dim=1);b=copy.deepcopy(a)
    x=torch.arange(12,dtype=torch.float32)[:,None];ids=torch.arange(12)
    a.update(x,ids);b.update(x,ids)
    assert fingerprint(a.state())==fingerprint(b.state())


def test_frozen_retention_is_distinct_from_new_acquisition():
    a=Support('FROZEN_FIRST',capacity=5,dim=1);c=Support('STREAM_GREEDY',capacity=5,dim=1)
    for event in range(101):
        x=torch.tensor([[float(event%5)]]) if event<100 else torch.tensor([[100.]])
        for m in (a,c):m.update(x,torch.tensor([event]))
    assert a.score(torch.tensor([[100.]]))[0].item()==96
    assert c.score(torch.tensor([[100.]]))[0].item()==0
    assert a.score(torch.tensor([[0.]]))[0].item()==0


def test_memory_ledger_counts_rng_graph_witnesses_and_serialization():
    for arm in ARMS:
        m=Support(arm);assert m.ledger()['charged_persistent_bytes']<=BUDGET
        assert serialized_bytes(m.state())<=BUDGET
    m=Support('WITNESS_PROTECTED');assert m.ledger()['witness_slots']==128
    assert m.ledger()['tensor_bytes']>2478*768*4


def test_no_write_read_only_deterministic_replay_and_no_resets():
    for arm in ARMS:
        a=Support(arm,capacity=8,dim=4,witness_capacity=3);b=copy.deepcopy(a)
        for phase in range(3):
            x=torch.randn(4,4)+phase*3;ids=torch.arange(phase*4,(phase+1)*4)
            for m in (a,b):m.update(x,ids)
            assert fingerprint(a.state())==fingerprint(b.state())
            before=fingerprint(a.state());a.update(x,ids,write=False)
            assert fingerprint(a.state())==before
            assert torch.equal(a.score(x)[0],b.score(x)[0])
            assert fingerprint(a.state())==before and a.events==phase+1 and a.seen==(phase+1)*4


def test_roles_reject_forbidden_identity_and_r1_content_overlap():
    def row(i,h):return {'category':'bottle','cache_index':i,'relative_path':f'bottle/train/good/{i:03}.png','raw_png_sha256':h}
    u,p,r=[row(0,'u')],[row(170,'p')],[row(150,'r')]
    validate_roles(u,p,r)
    with pytest.raises(RuntimeError,match='content'):validate_roles(u,[row(170,'u')],r)
    bad=row(0,'u');bad['relative_path']='bottle/test/good/000.png'
    with pytest.raises(RuntimeError,match='forbidden'):validate_roles([bad],p,r)


def test_causal_reader_rejects_future_unused_rows_and_early_probes(monkeypatch):
    monkeypatch.setattr(Reader,'__init__',lambda self:None)
    reader=Reader();u={'category':'bottle','relative_path':'bottle/train/good/000.png'}
    p={'category':'hazelnut','relative_path':'hazelnut/train/good/120.png'}
    reader.p={'updates':[u],'probes':[p]};reader.completed=0;reader.preflight=False;reader.admitted=set()
    with pytest.raises(RuntimeError,match='future or unauthorized'):reader.update(p)
    with pytest.raises(RuntimeError,match='boundary'):reader.probe(p,100)
    reader.completed=100
    with pytest.raises(RuntimeError,match='future or unauthorized'):reader.probe(p,100)
    reader.preflight=True
    with pytest.raises(RuntimeError,match='forbidden'):reader.probe(p,100)
