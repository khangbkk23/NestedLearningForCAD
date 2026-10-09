"""W3 contract tests. Run with unittest; no dataset, weights, timm or pytest needed."""

import copy
import io
import unittest

import torch
from torch.nn import functional as F

from models.hope_cad_v2 import HOPECADVisualBlock, VisualBlockConfig, FourScanVisualMemory
from models.hope_cad_v2.reference import scan_reference
from models.hope_cad_v2.scan_geometry import chunk_spans, scan_routes
from models.hope_cad_v2.visual_memory import VisualMemoryState, scan_chunked, stability_matched_step


MEASUREMENTS = {}


def measure_error(name, actual, expected):
    difference = (actual.detach().double() - expected.detach().double()).abs()
    MEASUREMENTS[name] = {
        'max_absolute_error': difference.max().item(),
        'relative_l2_error': (difference.norm() / expected.detach().double().norm().clamp_min(1e-30)).item(),
    }


def fixture(leading=(2, 2), n=7, d=3, dtype=torch.float64):
    z = torch.randn(*leading, n, d, dtype=dtype)
    q = F.normalize(torch.randn_like(z), dim=-1)
    matrices = [0.04 * torch.randn(*leading, d, d, dtype=dtype) for _ in range(3)]
    controls = [0.05 * torch.randn(*leading, 1, d, dtype=dtype) for _ in range(2)]
    return z, q, VisualMemoryState(*matrices, *controls)


def close_state(left, right, *, atol=1e-9, rtol=1e-7):
    for a, b in zip(left.tensors(), right.tensors()):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


class W3TestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(103)


class GeometryTests(W3TestCase):
    def test_rectangular_coordinate_order_and_inverse(self):
        routes = scan_routes(2, 3)
        expected = [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0],
                    [0, 3, 1, 4, 2, 5], [5, 2, 4, 1, 3, 0]]
        tokens = torch.arange(6).reshape(1, 6, 1)
        for route, order in zip(routes, expected):
            self.assertEqual(route.permutation.tolist(), order)
            self.assertTrue(torch.equal(route.restore(route.scan(tokens)), tokens))
        self.assertEqual([r.chunk_size for r in routes], [3, 3, 2, 2])

    def test_canonical_grid_and_invalid_geometry(self):
        routes = scan_routes(28, 28)
        self.assertTrue(all(r.chunk_size == 28 for r in routes))
        self.assertEqual(len(chunk_spans(784, 28)), 28)
        self.assertEqual(chunk_spans(5, 2), ((0, 2), (2, 4), (4, 5)))
        for dims in ((0, 2), (2, -1), (True, 2), (2.5, 3)):
            with self.assertRaises(ValueError):
                scan_routes(*dims)
        with self.assertRaises(ValueError):
            chunk_spans(5, 0)
        with self.assertRaises(ValueError):
            routes[0].scan(torch.randn(1, 2, 4))


class RecurrenceTests(W3TestCase):
    def test_fp64_output_and_all_five_maps_match_independent_oracle(self):
        z, q, initial = fixture()
        for chunk in (1, 3, 7, 20):
            with self.subTest(chunk=chunk):
                actual = scan_chunked(z, q, initial, chunk)
                oracle = scan_reference(z, q, initial, chunk)
                torch.testing.assert_close(actual.output, oracle.output, atol=1e-9, rtol=1e-7)
                close_state(actual.final_state, oracle.final_state)
                measure_error(f'fp64_oracle_chunk_{chunk}', actual.output, oracle.output)

    def test_float32_matches_fp64_oracle_at_spec_tolerance(self):
        z, q, initial = fixture(n=28, d=16)
        expected = scan_reference(z, q, initial, 7)
        actual = scan_chunked(z.float(), q.float(), VisualMemoryState(*(t.float() for t in initial.tensors())), 7)
        torch.testing.assert_close(actual.output.double(), expected.output, atol=1e-5, rtol=1e-4)
        close_state(VisualMemoryState(*(t.double() for t in actual.final_state.tensors())),
                    expected.final_state, atol=1e-5, rtol=1e-4)
        measure_error('fp32_oracle_output', actual.output, expected.output)
        for name, a, b in zip(('content', 'key', 'value', 'eta', 'alpha'), actual.final_state.tensors(), expected.final_state.tensors()):
            measure_error(f'fp32_oracle_final_{name}', a, b)

    def test_readout_is_from_chunk_boundary_not_current_token(self):
        z, q, initial = fixture(leading=(1,), n=6, d=2)
        changed = z.clone()
        changed[:, 0] *= 30
        before, after = scan_chunked(z, q, initial, 3), scan_chunked(changed, q, initial, 3)
        expected = q[:, :3] @ initial.content.transpose(-1, -2)
        torch.testing.assert_close(before.output[:, :3], expected, atol=0, rtol=0)
        torch.testing.assert_close(after.output[:, :3], expected, atol=0, rtol=0)
        self.assertGreater((before.output[:, 3:] - after.output[:, 3:]).abs().max().item(), 1e-8)

    def test_later_chunks_do_not_modify_earlier_readout(self):
        z, q, initial = fixture(leading=(1,), n=6, d=2)
        modified = z.clone()
        modified[:, 3:] += 20
        a, b = scan_chunked(z, q, initial, 3), scan_chunked(modified, q, initial, 3)
        torch.testing.assert_close(a.output, b.output, atol=0, rtol=0)
        # The last chunk still updates all five final maps for diagnostics.
        self.assertGreater((a.final_state.key - b.final_state.key).abs().max().item(), 1e-8)

    def test_batch_transport_and_inputs_are_immutable(self):
        z, q, initial = fixture(leading=(3, 2), n=6)
        saved = [t.clone() for t in (z, q, *initial.tensors())]
        batched = scan_chunked(z, q, initial, 2)
        for i in range(3):
            one = scan_chunked(z[i:i+1], q[i:i+1], VisualMemoryState(*(t[i:i+1] for t in initial.tensors())), 2)
            torch.testing.assert_close(one.output[0], batched.output[i], atol=1e-9, rtol=1e-7)
            close_state(one.final_state, VisualMemoryState(*(t[i:i+1] for t in batched.final_state.tensors())))
        for old, current in zip(saved, (z, q, *initial.tensors())):
            self.assertTrue(torch.equal(old, current))

    def test_gain_and_memory_bounds_are_relative_to_each_chunk_boundary(self):
        z, q, initial = fixture(leading=(20,), n=56, d=4)
        result = scan_chunked(z, q, initial, 28, capture=True)
        MEASUREMENTS['max_chunk_gain_spectral_norm'] = max(
            torch.linalg.matrix_norm(c.gains, ord=2).max().item() for c in result.chunks
        )
        for chunk in result.chunks:
            self.assertFalse(chunk.gains.requires_grad)
            self.assertLessEqual(torch.linalg.matrix_norm(chunk.gains, ord=2).max().item(), 1 + 1e-12)
            self.assertTrue(bool((chunk.key_norm_squared <= 1 + 1e-12).all()))
            self.assertTrue(bool((chunk.final_norms <= chunk.boundary_norms + 1e-12).all()))

    def test_zero_key_and_saturated_alpha_are_finite_in_forward_and_backward(self):
        for dtype in (torch.float32, torch.float64):
            key = torch.zeros(3, 2, dtype=dtype, requires_grad=True)
            delta = torch.zeros_like(key, requires_grad=True)
            eta = torch.ones(3, dtype=dtype, requires_grad=True)
            alpha = torch.tensor([0., .9, 1.], dtype=dtype, requires_grad=True)
            step = stability_matched_step(key, delta, eta, alpha)
            self.assertEqual(step.executed[0].item(), 0)
            self.assertEqual(step.executed[2].item(), 0)
            gradients = torch.autograd.grad(step.executed.sum(), (key, delta, eta, alpha), create_graph=True)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))

    def test_spectral_and_injection_limits_and_ulp_rounding(self):
        key = torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64)
        delta = torch.tensor([[0., 0.], [10., 0.]], dtype=torch.float64)
        alpha = torch.tensor([.01, .9], dtype=torch.float64, requires_grad=True)
        eta = torch.full((2,), 100., dtype=torch.float64)
        guarded = stability_matched_step(key, delta, eta, alpha)
        continuous_cap = 2 * alpha * (1 - 1e-6)
        expected = torch.nextafter(continuous_cap.detach(), torch.zeros_like(alpha))
        self.assertTrue(torch.equal(guarded.spectral_limit, expected))
        self.assertTrue(torch.equal(guarded.executed[:1], expected[:1]))
        self.assertTrue(bool((guarded.executed <= guarded.injection_limit).all()))
        derivative = torch.autograd.grad(guarded.executed[0], alpha)[0]
        self.assertAlmostEqual(derivative[0].item(), 2 * (1 - 1e-6), places=12)
        self.assertGreater(derivative[0].item(), 0)  # no detached rate/retention path

    def test_invalid_inputs_fail_explicitly(self):
        z, q, initial = fixture(leading=(1,), n=3, d=2)
        with self.assertRaises(ValueError):
            scan_chunked(z.half(), q.half(), initial, 1)
        with self.assertRaises(ValueError):
            scan_chunked(z, q, initial, 0)
        with self.assertRaises(ValueError):
            scan_chunked(z, q[..., :1], initial, 1)
        with self.assertRaises(ValueError):
            scan_reference(z.float(), q.float(), VisualMemoryState(*(t.float() for t in initial.tensors())), 1)
        z[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            scan_chunked(z, q, initial, 1)


class GradientTests(W3TestCase):
    def test_optimized_and_independent_oracle_gradients_agree(self):
        z, q, state = fixture(leading=(1,), n=5, d=2)
        args = tuple(t.detach().requires_grad_() for t in (z, q, *state.tensors()))
        def loss(fn):
            result = fn(args[0], args[1], VisualMemoryState(*args[2:]), 2)
            return result.output.square().sum() + sum(t.square().sum() for t in result.final_state.tensors())
        actual = torch.autograd.grad(loss(scan_chunked), args)
        reference = torch.autograd.grad(loss(scan_reference), args)
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)

    def test_finite_difference_first_and_second_derivatives(self):
        z, q, state = fixture(leading=(1,), n=3, d=2)
        args = tuple(t.detach().requires_grad_() for t in (z, q, *state.tensors()))
        def function(*values):
            result = scan_chunked(values[0], values[1], VisualMemoryState(*values[2:]), 1)
            return (result.output, *result.final_state.tensors())
        self.assertTrue(torch.autograd.gradcheck(function, args, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True))
        self.assertTrue(torch.autograd.gradgradcheck(function, args, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True))

    def test_all_declared_visual_parameters_receive_finite_nonzero_gradients(self):
        model = HOPECADVisualBlock(VisualBlockConfig(10, 8, 4, 2, 3, 4)).double()
        features = torch.randn(2, 12, 10, dtype=torch.float64, requires_grad=True)
        output = model(features)
        loss = (output * torch.randn_like(output)).sum() + output.square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(parameter.grad.abs().max().item(), 0)
        self.assertGreater(features.grad.abs().max().item(), 0)


class SpatialBlockTests(W3TestCase):
    def test_four_route_fusion_matches_explicit_oracle_on_rectangle(self):
        model = FourScanVisualMemory(4, 2, (2, 3)).double()
        z = torch.randn(2, 6, 4, dtype=torch.float64)
        q = F.normalize(model.query(z).reshape(2, 6, 2, 2), dim=-1, eps=1e-6).flatten(-2)
        expected, finals = [], []
        for i, route in enumerate(scan_routes(2, 3)):
            state = VisualMemoryState(*(t[i].unsqueeze(0).expand(2, -1, -1, -1) for t in model.initial_state().tensors()))
            result = scan_reference(route.scan(z).reshape(2, 6, 2, 2).transpose(1, 2),
                                    route.scan(q).reshape(2, 6, 2, 2).transpose(1, 2), state, route.chunk_size)
            expected.append(route.restore(result.output.transpose(1, 2).reshape(2, 6, 4)))
            finals.append(result.final_state)
        inspection = model.inspect(z)
        outputs = torch.stack(expected, dim=1)
        torch.testing.assert_close(inspection.directional_outputs, outputs, atol=1e-9, rtol=1e-7)
        torch.testing.assert_close(inspection.output, (outputs * model.fusion[None, :, None]).sum(1), atol=1e-9, rtol=1e-7)
        close_state(inspection.final_state, VisualMemoryState(*(torch.stack([s.tensors()[m] for s in finals], 1) for m in range(5))))
        self.assertEqual(len(inspection.scan_groups), 2)

    def test_directions_have_independent_working_and_initial_maps(self):
        model = FourScanVisualMemory(4, 2, (3, 3)).double()
        z = torch.randn(1, 9, 4, dtype=torch.float64)
        before = model.inspect(z).directional_outputs.detach()
        with torch.no_grad():
            model.initial_content[0].add_(.03)
        after = model.inspect(z).directional_outputs.detach()
        self.assertGreater((before[:, 0] - after[:, 0]).abs().max().item(), 0)
        torch.testing.assert_close(before[:, 1:], after[:, 1:], atol=0, rtol=0)

    def test_every_call_resets_per_image_view_and_preserves_parameters_rng(self):
        model = HOPECADVisualBlock(VisualBlockConfig(10, 8, 4, 2, 3, 3)).double()
        batch = torch.randn(3, 9, 10, dtype=torch.float64)
        before = copy.deepcopy(model.state_dict())
        rng = torch.random.get_rng_state().clone()
        with torch.no_grad():
            together = model(batch)
            individual = torch.cat([model(image[None]) for image in batch])
            _ = model(batch + 50)
            repeated = model(batch)
            reversed_batch = model(batch.flip(0)).flip(0)
        for actual in (individual, repeated, reversed_batch):
            torch.testing.assert_close(actual, together, atol=1e-9, rtol=1e-7)
        for name, old in before.items():
            current = model.state_dict()[name]
            self.assertTrue(torch.equal(old, current) if isinstance(old, torch.Tensor) else old == current)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(list(model.named_buffers()), [])

    def test_state_dict_roundtrip_and_geometry_mismatch_rejected(self):
        config = VisualBlockConfig(10, 8, 4, 2, 2, 3)
        model = HOPECADVisualBlock(config).double()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        restored = HOPECADVisualBlock(config).double()
        restored.load_state_dict(state)
        x = torch.randn(1, 6, 10, dtype=torch.float64)
        torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)
        incompatible = HOPECADVisualBlock(VisualBlockConfig(10, 8, 4, 2, 3, 2)).double()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            incompatible.load_state_dict(state)

    def test_canonical_shapes_initialization_and_full_grid_forward_backward(self):
        model = HOPECADVisualBlock()
        self.assertTrue(model.config.canonical)
        self.assertEqual(model.memory.heads, 8)
        self.assertEqual(model.memory.initial_content.shape, (4, 8, 16, 16))
        self.assertEqual(model.memory.initial_eta.shape, (4, 8, 1, 16))
        self.assertEqual(model.depthwise.groups, 128)
        self.assertEqual(model.depthwise.kernel_size, (3, 3))
        self.assertEqual(model.ffn[0].out_features, 1024)
        for initial in model.memory.initial_state().tensors()[:3]:
            self.assertLessEqual(initial.abs().max().item(), .02)
        self.assertEqual(model.memory.initial_eta.count_nonzero().item(), 0)
        self.assertEqual(model.memory.initial_alpha.count_nonzero().item(), 0)
        self.assertTrue(bool((model.memory.fusion == .25).all()))
        x = torch.randn(1, 784, 768)
        output = model(x)
        self.assertEqual(output.shape, (1, 784, 256))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        self.assertFalse(model.method_metadata()['full_method_implemented'])

    def test_canonical_full_grid_float32_matches_float64_with_same_weights(self):
        model = HOPECADVisualBlock()
        reference = copy.deepcopy(model).double()
        x = torch.randn(1, 784, 768)
        with torch.no_grad():
            expected = reference(x.double())
            actual = model(x)
        torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=1e-4)
        measure_error('canonical_full_grid_fp32_vs_fp64', actual, expected)

    def test_wrapper_order_matches_equation_and_no_extra_branches(self):
        model = HOPECADVisualBlock(VisualBlockConfig(10, 8, 4, 2, 2, 3)).double()
        x = torch.randn(1, 6, 10, dtype=torch.float64)
        p = model.projection(x)
        z = model.in_projection(model.input_norm(p)).transpose(1, 2).reshape(1, 4, 2, 3)
        visual = model.memory(model.depthwise(z).flatten(2).transpose(1, 2))
        residual = p + model.out_projection(visual)
        expected = residual + model.ffn(model.ffn_norm(residual))
        torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
        self.assertFalse(any(isinstance(m, torch.nn.Dropout) for m in model.modules()))

    def test_invalid_config_inputs_and_failed_call_do_not_change_state(self):
        for arguments in ({'dim':0}, {'memory_dim':3, 'head_dim':2}, {'height':True}):
            with self.assertRaises(ValueError):
                VisualBlockConfig(**arguments)
        model = HOPECADVisualBlock(VisualBlockConfig(10, 8, 4, 2, 2, 3))
        for x in (torch.randn(1, 6, 11), torch.randn(0, 6, 10), torch.randn(1, 6, 10).half(), torch.full((1,6,10), float('inf'))):
            with self.assertRaises(ValueError):
                model(x)
        x = torch.randn(1, 6, 10)
        before = model(x)
        invalid = x.clone(); invalid[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            model(invalid)
        torch.testing.assert_close(model(x), before, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable; CPU tests still cover the contract')
    def test_cuda_float32_cpu_parity_autocast_and_backward(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            cpu = HOPECADVisualBlock(VisualBlockConfig(10, 8, 4, 2, 3, 4))
            gpu = copy.deepcopy(cpu).cuda()
            x = torch.randn(2, 12, 10)
            expected = cpu(x)
            with torch.autocast('cuda', dtype=torch.float16):
                actual = gpu(x.cuda())
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-4)
            measure_error('cuda_fp32_vs_cpu', actual.cpu(), expected)
            actual.square().sum().backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in gpu.parameters()))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
