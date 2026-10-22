"""Explicit effective CADIC routes. No import-time monkeypatch or legacy fallback."""
import importlib
from pathlib import Path
from hope_cad.data.prefix import file_hash

ROUTES = {
    'cadic_loader_batch_v1': ('hope_cad.baselines.cadic.compatible_adapter', 'loader_batch',
                            'hope_cad.baselines.cadic.compatible_bank', 'torch_cdist_euclidean'),
    'cadic_paperfaithful_image_v1': ('hope_cad.baselines.cadic.adapter', 'image',
                                    'hope_cad.baselines.cadic.bank', 'clamp_mm_then_sqrt'),
    'cadic_paperfaithful_loader_batch_v1': ('hope_cad.baselines.cadic.adapter', 'loader_batch',
                                          'hope_cad.baselines.cadic.bank', 'clamp_mm_then_sqrt'),
}


def route_descriptor(route):
    if route not in ROUTES:
        raise ValueError(f'unknown/unavailable adapter: {route}; full HOPE adapter belongs to W5')
    module, update, kernel_module, kernel = ROUTES[route]
    sources = (module, kernel_module, 'hope_cad.models.features',
               'hope_cad.benchmark.metrics', 'hope_cad.benchmark.stream',
               'hope_cad.benchmark.protocol_config', 'hope_cad.data.prefix',
               'hope_cad.data.masks', 'hope_cad.benchmark.matched',
               'hope_cad.benchmark.registry')
    hashes = {}
    # Resolve paths without importing timm or allocating a backbone.
    from hope_cad.paths import ROOT as root
    for source in sources:
        path = root / 'src' / (source.replace('.', '/') + '.py')
        hashes[path.relative_to(root).as_posix()] = file_hash(path)
    return dict(adapter_id=route, class_name=module + '.CADICBenchmarkAdapterV1',
                effective_update_unit=update, distance_kernel=kernel,
                score='s_star * (1 - exp(s_star - logsumexp(query_to_neighbors_of_c_star_distances)))',
                source_sha256=hashes, exact_parity_claim=False)


class RegisteredAdapter:
    def __init__(self, adapter, descriptor):
        self.adapter, self.descriptor = adapter, descriptor

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def method_metadata(self):
        return {**self.adapter.method_metadata(), 'effective_route': self.descriptor}


def build_adapter(config, device='cpu'):
    descriptor = route_descriptor(config['adapter'])
    if config['memory']['update_unit'] != descriptor['effective_update_unit']:
        raise ValueError('requested update unit differs from effective adapter implementation')
    path = Path(config['extractor']['checkpoint_path'])
    if file_hash(path) != config['extractor']['checkpoint_sha256']:
        raise ValueError('extractor checkpoint SHA256 mismatch')
    module, _, _, _ = ROUTES[config['adapter']]
    cls = getattr(importlib.import_module(module), 'CADICBenchmarkAdapterV1')
    descriptor['checkpoint_sha256'] = file_hash(path)
    return RegisteredAdapter(cls(config, device), descriptor)
