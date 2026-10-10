"""Explicit effective CADIC routes. No import-time monkeypatch or legacy fallback."""
import importlib
from pathlib import Path
from dataset.benchmark_protocol_v2 import file_hash

ROUTES = {
    'cadic_loader_batch_v1': ('models.cadic_benchmark_adapter_v1', 'loader_batch',
                            'models.cadic_patch_coreset_v1', 'torch_cdist_euclidean'),
    'cadic_paperfaithful_image_v1': ('models.cadic_paperfaithful.cadic_benchmark_adapter', 'image',
                                    'models.cadic_paperfaithful.cadic_patch_coreset', 'clamp_mm_then_sqrt'),
    'cadic_paperfaithful_loader_batch_v1': ('models.cadic_paperfaithful.cadic_benchmark_adapter', 'loader_batch',
                                          'models.cadic_paperfaithful.cadic_patch_coreset', 'clamp_mm_then_sqrt'),
}


def route_descriptor(route):
    if route not in ROUTES:
        raise ValueError(f'unknown/unavailable adapter: {route}; full HOPE adapter belongs to W5')
    module, update, kernel_module, kernel = ROUTES[route]
    sources = (module, kernel_module, 'models.feature_extractors.cadic_vit_v1',
               'training.benchmark_metrics_v1', 'training.benchmark_engine_v1',
               'training.benchmark_config_v2', 'dataset.benchmark_protocol_v2',
               'dataset.visa_mask_qa_v2', 'scripts.hope_cad_v2.run_matched_cadic',
               'models.benchmark_registry_v2')
    hashes = {}
    # Resolve paths without importing timm or allocating a backbone.
    root = Path(__file__).resolve().parents[1]
    for source in sources:
        path = root / (source.replace('.', '/') + '.py')
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
