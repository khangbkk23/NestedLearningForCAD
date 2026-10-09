# exps/test_hope_outer_learning_pilot_v1.py
"""Synthetic device-neutral algebra, protocol, attribution and restart checks."""

from dataclasses import replace
import gc
import json
import weakref

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from exps.hope_outer_learning_p1_v1 import (
    synthetic_parameters, initial_functional_state, propose_functional_event,
    functional_support_sequence, read_query_without_update,
)
from exps.hope_outer_learning_pilot_v1 import (
    CENTER_IDS, CENTERS, Parameters, ROLES, atomic_json, commit, copy_parameters,
    detached_state, event, initial_state, inventory, make_schedule, mask_rgb,
    nontriviality, paired_bootstrap, parameters_sha, predict_with_product,
    propose, read, restore_parameters, ridge_fit, ridge_read, schedule_sha,
    serialize_parameters, state_fingerprint, support_sequence, teacher_error, parameter_storage_bytes,
)
from scripts.exps.hope_outer_learning_pilot_v1 import save_fit_checkpoint, write_table


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(dim=4,trainable=True):
    old=synthetic_parameters(dim,requires_grad=trainable)
    p=Parameters(old.a0,old.wq,dict(old.auxiliary),old.conv_weight,old.conv_bias)
    return p,old


def images(p,sizes=(7,3,17)):
    g=torch.Generator().manual_seed(14)
    return [torch.randn(1,n,p.dim,generator=g,dtype=p.a0.dtype) for n in sizes]


@pytest.mark.parametrize("n",[1,3,17,784])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_forward_matches_locked_functional_oracle(n,dtype):
    p,old=tiny()
    # Preserve parameter ownership while selecting the same precision in both paths.
    if dtype==torch.float32:
        old=synthetic_parameters(4,dtype=dtype)
        p=Parameters(old.a0,old.wq,dict(old.auxiliary),old.conv_weight,old.conv_bias)
    x=images(p,(n,))[0]
    expected=propose_functional_event(old,initial_functional_state(old),x)
    actual=propose(p,initial_state(p),x)
    for name in ("keys","values","gates","queries"):
        assert torch.equal(getattr(actual,name),getattr(expected.projection,name))
    for name in ("C","D","transition"):
        assert torch.equal(getattr(actual,name),getattr(expected,name))
    for name in actual.weights:
        assert torch.equal(actual.weights[name],expected.weights[name])
    assert torch.equal(actual.pre_read,expected.pre_image_read)
    committed=commit(p,actual.source,actual)
    assert committed.counters.completed==1


def test_post_support_gradient_routing_matches_cpu_reference():
    p,old=tiny()
    xs=images(p)
    state,product=support_sequence(p,xs)
    original=functional_support_sequence(old,xs)
    query=images(p,(5,))[0]
    actual=read(p,state,query)
    expected=read_query_without_update(old,original.state,query)
    assert torch.equal(actual,expected)
    target=torch.ones_like(actual)*.3
    loss=teacher_error(actual,target,centers=tuple(range(5)))
    ga,gw=torch.autograd.grad(loss,(p.a0,p.wq))
    ga0,gw0=torch.autograd.grad(teacher_error(expected,target,centers=tuple(range(5))),(old.a0,old.wq))
    assert torch.equal(ga,ga0) and torch.equal(gw,gw0)
    assert float(ga.norm())>0 and float(gw.norm())>0
    assert not product.requires_grad
    for v in (*p.auxiliary.values(),p.conv_weight,p.conv_bias):
        assert not v.requires_grad and v.grad is None


def test_atomic_rejects_invalid_or_stale_and_source_is_unchanged():
    p,_=tiny();state=initial_state(p);identity=state_fingerprint(state)
    proposal=propose(p,state,images(p,(7,))[0])
    bad={**proposal.weights,"memory":torch.full_like(proposal.weights["memory"],float("nan"))}
    with pytest.raises(ValueError):
        commit(p,state,replace(proposal,weights=bad))
    with pytest.raises(ValueError,match="stale"):
        commit(p,initial_state(p),proposal)
    assert state_fingerprint(state)==identity


def test_candidate_isolation_read_only_and_event_clock():
    p,_=tiny();p2=copy_parameters(p,trainable=True)
    assert parameters_sha(p)==parameters_sha(p2)
    state,product=support_sequence(p,images(p))
    expected=parameters_sha(p2)
    fingerprint=state_fingerprint(state)
    read(p,state,images(p,(2,))[0])
    assert fingerprint==state_fingerprint(state)
    assert expected==parameters_sha(p2)
    assert state.counters.completed==3 and state.counters.online==6
    plain=detached_state(state)
    assert not inventory(p,plain)["graph_bearing"]
    assert all(torch.equal(state.weights[k],plain.weights[k]) for k in state.weights)
    assert torch.equal(product,product.detach())


def test_schedule_disjoint_roles_query_matching_and_locked_lengths():
    a,b=make_schedule(),make_schedule()
    assert a==b and schedule_sha(a)==schedule_sha(b)
    assert len(a["episodes"])==200 and len(a["pooled50"])==8
    for episode in a["episodes"]:
        assert len(set(episode["support"]))==4
        assert all(0<=i<60 for i in episode["support"])
        assert 60<=episode["query"]<80
    assert a["online"]==list(range(100,150))
    assert all(len(ids)==50 and max(ids)<60 for ids in a["pooled50"])
    assert len(set(a["wrong_history50"]))==50
    ids=[i for _,lo,hi in ROLES for i in range(lo,hi)]
    assert ids==list(range(170))


def test_one_shared_view_blocks_clean_center_copy_and_matches_coordinates():
    clean=torch.arange(3*224*224,dtype=torch.float32).reshape(1,3,224,224)
    masked,mask=mask_rgb(clean)
    assert int(mask.sum())==9216
    assert len(CENTER_IDS)==16
    for r,c in CENTERS:
        assert mask[(r-1)*8:(r+2)*8,(c-1)*8:(c+2)*8].all()
        assert CENTER_IDS[CENTERS.index((r,c))]==r*28+c
    changed=clean.clone();changed[:,:,mask]=9e6
    second,_=mask_rgb(changed)
    assert torch.equal(masked,second)
    assert torch.equal(clean[:,:,~mask],masked[:,:,~mask])


def test_teacher_loss_is_channel_sum_center_mean_and_target_frozen():
    prediction=torch.ones(784,4,requires_grad=True)
    target=torch.zeros(784,4)
    assert float(teacher_error(prediction,target).detach())==2
    with pytest.raises(ValueError,match="teacher"):
        teacher_error(prediction,target.requires_grad_())


def test_nontriviality_stops_identical_or_constant_targets():
    clean=F.normalize(torch.randn(8,784,4),dim=-1)
    trained=F.normalize(torch.randn(20,784,4),dim=-1)
    result=nontriviality(clean,clean.clone(),trained)
    assert not result["passed"] and result["identical_center_fraction"]==1
    constant=torch.ones_like(clean)
    result=nontriviality(constant,torch.randn_like(clean),trained)
    assert not result["passed"] and result["target_mean_squared_distance"]==0


def test_fixed_transition_can_be_absorbed_by_static_content():
    p,_=tiny();x=images(p,(5,))[0];product=torch.eye(p.dim,dtype=p.a0.dtype)*.8
    static=replace(p,a0=p.a0@product)
    assert torch.equal(predict_with_product(p,x,product),read(static,initial_state(static),x))


def test_long_sequence_matches_equal_order_oracle_without_reassociation():
    old=synthetic_parameters(12,dtype=torch.float32,requires_grad=False)
    p=Parameters(old.a0,old.wq,dict(old.auxiliary),old.conv_weight,old.conv_bias)
    sequence=images(p,(31,)*50)
    actual,product=support_sequence(p,sequence)
    expected=functional_support_sequence(old,sequence)
    for name in actual.weights:
        assert torch.equal(actual.weights[name],expected.state.weights[name])
    assert actual.counters==expected.state.counters
    # Reassociation is algebraically valid but is not the FP32 forward oracle.
    reassociated=p.a0@product
    assert torch.isfinite(reassociated).all()
    assert not torch.equal(actual.weights['memory'],reassociated)


def test_ridge_fit_and_image_paired_bootstrap_are_deterministic():
    g=torch.Generator().manual_seed(2)
    x=torch.randn(20,4,generator=g,dtype=torch.float64)
    y=x@torch.randn(4,4,generator=g,dtype=torch.float64)
    fitted=ridge_fit(x,y)
    assert (ridge_read(x,fitted)-y).norm()/(y.norm()+1e-12)<.005
    expected=.001*float(((x-x.mean(0)).T@(x-x.mean(0))).trace())/4
    assert fitted[-1]==expected
    result=paired_bootstrap([1,2,3],[2,3,4])
    assert result["estimate"]==result["ci_low"]==result["ci_high"]==1
    assert result==paired_bootstrap([1,2,3],[2,3,4])
    assert result["n_images"]==3


def test_checkpoint_restart_restores_independent_optimizer_and_parameters(tmp_path):
    p,_=tiny();optimizer=torch.optim.Adam([p.a0,p.wq],lr=1e-4)
    x=images(p,(5,))[0];target=torch.zeros(5,p.dim,dtype=p.a0.dtype)
    def step(parameters,opt):
        opt.zero_grad(set_to_none=True)
        teacher_error(read(parameters,initial_state(parameters),x),target,tuple(range(5))).backward()
        opt.step()
    step(p,optimizer)
    path=tmp_path/"fit.pt"
    save_fit_checkpoint(path,p,optimizer,1,"STATIC_META","schedule","manifest")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    q=restore_parameters(payload["parameters"],trainable=True)
    assert parameters_sha(q)==parameters_sha(p)
    other=torch.optim.Adam([q.a0,q.wq],lr=1e-4);other.load_state_dict(payload["optimizer"])
    step(p,optimizer);step(q,other)
    assert torch.equal(p.a0,q.a0) and torch.equal(p.wq,q.wq)
    assert payload["step"]==1


def test_artifact_round_trip_and_detached_state_schema(tmp_path):
    p,_=tiny();state=initial_state(p);plain=detached_state(state)
    atomic_json(tmp_path/"summary.json",{"steps":0,"values":[1,2]})
    write_table(tmp_path,"errors.parquet",[{"arm":"META_P1","identity":92,"error":.4}])
    frame=pd.read_parquet(tmp_path/"errors.parquet")
    assert len(frame)==1 and frame.iloc[0].identity==92
    assert json.loads((tmp_path/"summary.json").read_text())["steps"]==0
    assert inventory(p,plain)["key_count"]==9
    assert parameters_sha(restore_parameters(serialize_parameters(p)))==parameters_sha(p)


def test_actual_storage_includes_retained_unused_auxiliary_tensors():
    p,_=tiny()
    actual=sum(v.numel()*v.element_size() for v in
               (p.a0,p.wq,p.conv_weight,p.conv_bias,*p.auxiliary.values()))
    minimal=sum(v.numel()*v.element_size() for v in (p.a0,p.wq,p.conv_weight,p.conv_bias))
    assert parameter_storage_bytes(p)==actual and actual>minimal
    inv=inventory(p,initial_state(p))
    assert inv['total_deployed_bytes']==actual+inv['fast_bytes']


def test_graph_disposal_has_no_persistent_episode_references():
    refs=[]
    for _ in range(8):
        p,_=tiny();xs=images(p);state,_=support_sequence(p,xs)
        prediction=read(p,state,xs[0]);loss=prediction.square().mean();loss.backward()
        refs.extend([weakref.ref(p.a0),weakref.ref(state.weights["memory"]),weakref.ref(loss)])
        del p,state,prediction,loss,xs
    # The ignored old fixture holds the same A0 leaf until explicitly discarded.
    del _
    gc.collect()
    assert all(ref() is None for ref in refs)
