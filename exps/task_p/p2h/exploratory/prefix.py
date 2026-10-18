"""Exact state forking at a completed16-cycle boundary, never future data."""
import os
import torch
from ..preflight import tensor_file
from .data import SyntheticStream,Replay,WitnessEvaluator
from .design import OUT


def path(stage,seed,arm): return OUT/stage/'prefixes'/f'{seed}_{arm}.pt'


def capture(stage,seed,arm,learner,stream,replay,rng,evaluator,acquisition,seconds):
    assert learner.live.clock==64 and learner.window is None
    assert stream.next_event==32 and stream.muB is None,'prefix contains future phase'
    assert set(evaluator.samples)=={'A'}
    target=path(stage,seed,arm); target.parent.mkdir(exist_ok=True)
    tensor_file(target,dict(learner=learner.payload(),stream=stream.state(),replay=replay.state(),
        anchors=rng.get_state(),witness_rng=evaluator.rng.get_state(),witnesses=evaluator.samples,
        acquisition=acquisition,prefix_seconds=seconds,available_after_cycle=64))


def restore(stage,seed,regime,arm,learner,directory):
    saved=torch.load(path(stage,seed,arm),weights_only=False,map_location='cpu')
    assert saved['available_after_cycle']==64 and saved['stream']['muB'] is None
    learner.restore(saved['learner'])
    stream=SyntheticStream.restore(saved['stream']); stream.regime=regime
    replay=Replay.restore(saved['replay'])
    rng=torch.Generator().set_state(saved['anchors'])
    evaluator=WitnessEvaluator(seed,stream)
    evaluator.rng.set_state(saved['witness_rng']); evaluator.samples=saved['witnesses']
    source=OUT/stage/f'D8_{seed}_{arm}'
    for name in ['endpoint_64.pt']+[f'window_{i:02d}.pt' for i in range(4)]:
        os.link(source/name,directory/name)
    return stream,replay,rng,evaluator,saved['acquisition'],saved['prefix_seconds']
