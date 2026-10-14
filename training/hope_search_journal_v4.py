"""Atomic, identity-bound progress for each fit and stress gate in a finite search."""
from pathlib import Path
import torch
from training.hope_runtime_v4 import atomic_torch_save, atomic_json
from scripts.hope_cad_v2.diagnostics_v4 import visual_diagnostics


class SearchJournal:
    def __init__(self, root, data, *, device='cpu'):
        self.root, self.data, self.device = Path(root), data, device

    def path(self, method, beta, lr, kind):
        return self.root / f'{method}_b{beta:g}_lr{lr:g}_{kind}.pt'

    def load(self, path):
        return torch.load(path, map_location=self.device, weights_only=False) if path.is_file() else None

    def load_candidate(self, method, beta, lr):
        return self.load(self.path(method, beta, lr, 'complete'))

    def save_candidate(self, method, beta, lr, candidate):
        atomic_torch_save(candidate, self.path(method, beta, lr, 'complete'))

    def fit_options(self, method, beta, lr):
        path = self.path(method, beta, lr, 'fit_resume')
        def checkpoint(payload):
            atomic_torch_save(payload,path)
            if payload['step'] in (0,1,5,25,50,100,200,300,400):
                print(f"[{method}] beta={beta:g} outer_lr={lr:g} step={payload['step']} checkpoint saved",flush=True)
        def diagnose(step, model):
            from training.hope_outer_learning_v4 import capture_rng, restore_rng
            state = capture_rng()
            try:
                x, _, _ = self.data.get(self.data.ids('meta_support')[0], 0, 'meta_support')
                result = visual_diagnostics(model, x)
                atomic_json(dict(step=step, method=method, beta=beta, outer_lr=lr,
                                 identity=self.data.ids('meta_support')[0], view=0,
                                 data_scope='meta_support', diagnostics=result),
                            self.root / f'{method}_b{beta:g}_lr{lr:g}_diagnostics_{step}.json')
            finally:
                restore_rng(state)
        return dict(resume=self.load(path), checkpoint_callback=checkpoint,
                    diagnostic_callback=diagnose)

    def gate_options(self, method, beta, lr):
        path = self.path(method, beta, lr, 'gate_resume')
        return dict(resume=self.load(path), checkpoint_callback=lambda p: atomic_torch_save(p, path))
