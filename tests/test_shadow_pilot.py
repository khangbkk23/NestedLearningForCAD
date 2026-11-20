"""B322 shadow checks. Standard library by default; torch cases skip without torch.

Run directly (no project conftest, no data, no GPU):
    python -I -B tests/test_shadow_pilot.py
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shadow = load("shadow", "src/nested_replay_cad/shadow.py")
report = load("shadow_report", "scripts/shadow_report.py")
pilot = load("shadow_pilot", "scripts/shadow_pilot.py")
CONFIG = json.loads((ROOT / "configs/b322.json").read_text(encoding="utf8"))
TASKS = [f"t{i:02d}" for i in range(15)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stages(final, acquisition=None, peak=None):
    """Pixel AP is ``acquisition`` when learned, ``peak`` in between, ``final`` at the end."""
    acquisition = acquisition or final
    peak = peak or acquisition
    out = []
    for boundary in range(15):
        row = {}
        for index in range(boundary + 1):
            if boundary == 14:
                value = final[index]
            elif index == boundary:
                value = acquisition[index]
            else:
                value = peak[index]
            row[TASKS[index]] = {"pixel": {"AP": value, "AUROC": 0.9}, "image": {"AP": 0.9, "AUROC": 0.95}}
        out.append(row)
    return out


class CoefficientTests(unittest.TestCase):
    def test_tau_one_tracks_parameters(self):
        self.assertTrue(all(shadow.coefficient(t, 1) == 0.0 for t in (1, 2, 500)))

    def test_warmup_is_plain_mean_then_exponential(self):
        tau, thetas, s = 8, [float(i * i) for i in range(40)], 0.0
        for t in range(1, 40):
            a = shadow.coefficient(t, tau)
            s = a * s + (1 - a) * thetas[t]
            if t + 1 <= tau:
                self.assertAlmostEqual(s, sum(thetas[: t + 1]) / (t + 1), places=9)
            else:
                self.assertEqual(a, 1 - 1 / tau)

    def test_invalid_inputs_denied(self):
        for step, tau in ((0, 4), (1, 0), (True, 4), (1.0, 4)):
            with self.assertRaises(ValueError):
                shadow.coefficient(step, tau)
        for taus in ([], [4, 4], [0], "512"):
            with self.assertRaises(ValueError):
                shadow.validate_taus(taus)


class ReportTests(unittest.TestCase):
    def test_matches_recorded_b319_summary(self):
        recorded = json.loads(
            (ROOT / "experiments/B319/runs/b319-s0-20261110-kaggle-v5/metrics.json").read_text(encoding="utf8")
        )["arms"]["er_uniform"]
        tasks, values = recorded["matrix"]["tasks"], recorded["matrix"]["values"]
        rebuilt = [
            {
                c: {"pixel": {"AP": values["pixel_AP"][b][j], "AUROC": values["pixel_AUROC"][b][j]},
                    "image": {"AP": values["image_AP"][b][j], "AUROC": values["image_AUROC"][b][j]}}
                for j, c in enumerate(tasks[: b + 1])
            }
            for b in range(15)
        ]
        mine = report.summarize(rebuilt)
        for key in ("final_macro_pixel_AP", "final_macro_image_AUROC", "macro_Fmax", "macro_Facq"):
            self.assertAlmostEqual(mine[key], recorded[key], places=12)

    def test_forgetting_and_acquisition_definitions(self):
        out = report.summarize(stages([0.5] * 15, acquisition=[0.6] * 15, peak=[0.7] * 15))
        self.assertAlmostEqual(out["macro_acquisition_pixel_AP"], (0.6 * 14 + 0.5) / 15)
        self.assertAlmostEqual(out["macro_Facq"], 0.1)
        # Task 13 is seen once before final, so its prior maximum is its acquisition value.
        self.assertAlmostEqual(out["macro_Fmax"], (0.2 * 13 + 0.1) / 14)
        self.assertIsNone(out["per_category"][TASKS[-1]]["Fmax"])

    def labels(self, fast, slow):
        return report.compare(report.summarize(fast), report.summarize(slow), CONFIG["gate"], CONFIG["historical_screen"])

    def test_labels_need_all_conditions(self):
        fast = stages([0.50] * 15, acquisition=[0.60] * 15)
        good = self.labels(fast, stages([0.56] * 15, acquisition=[0.60] * 15))
        self.assertEqual(good["label"], "SIGNAL")
        self.assertEqual(good["historical_screen"]["status"], "SCREEN_MET")
        self.assertEqual(self.labels(fast, stages([0.502] * 15, acquisition=[0.60] * 15))["label"], "NO_SIGNAL")
        lagging = self.labels(fast, stages([0.56] * 15, acquisition=[0.58] * 15))
        self.assertEqual(lagging["label"], "MIXED")
        self.assertFalse(lagging["conditions"]["c_plasticity"])

    def test_historical_screen_floor_is_unchanged(self):
        final = [0.56] * 15
        final[3] = 0.48
        out = self.labels(stages([0.50] * 15, acquisition=[0.60] * 15), stages(final, acquisition=[0.60] * 15))
        self.assertEqual(out["historical_screen"], {"status": "SCREEN_NOT_MET", "floor_violations": [TASKS[3]]})

    def test_secondary_cannot_change_primary_verdict(self):
        fast = stages([0.50] * 15, acquisition=[0.60] * 15)
        snapshots = []
        for f, p, s in zip(fast, stages([0.50] * 15, acquisition=[0.60] * 15), stages([0.56] * 15, acquisition=[0.60] * 15)):
            snapshots.append({"metrics": f, "model_sha256": "x", "shadow": {"512": {"metrics": p}, "128": {"metrics": s}}})
        out = report.report({"status": "COMPLETE_REVIEW_PENDING", "committed": 3629, "snapshots": snapshots}, CONFIG)
        self.assertEqual(out["verdict"], "NO_SIGNAL")
        self.assertEqual(out["secondary_exploratory"]["label"], "SIGNAL")
        self.assertIn("primary verdict is unchanged", out["secondary_note"])


class ContractTests(unittest.TestCase):
    def test_registered_config_loads_and_binds_b319_bytes(self):
        loaded = pilot.load_pilot(ROOT / "configs/b322.json", digest)
        self.assertEqual(loaded["shadow"]["primary"], 512)
        self.assertEqual(loaded["caps_status"], "APPROVED")
        self.assertEqual(loaded["caps"], {"per_arm_seconds": 14400})

    def test_changed_contract_denied(self):
        import tempfile

        for key, value in (("shadow", {"taus": [128, 512], "primary": 128, "secondary": 512}),
                           ("seed", 1), ("base_recipe_sha256", "0" * 64)):
            changed = copy.deepcopy(CONFIG)
            changed[key] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "b322.json"
                path.write_text(json.dumps(changed), encoding="utf8")
                with self.assertRaises(ValueError):
                    pilot.load_pilot(path, digest)

    def test_b319_recipe_has_no_shadow(self):
        self.assertNotIn("shadow", json.loads((ROOT / "configs/b319.json").read_text(encoding="utf8")))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch not installed")
class TorchShadowTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(3)
        return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))

    def test_update_is_passive_and_tau_one_equals_model(self):
        model = self.model()
        shadows = shadow.Shadows(torch, model, [1, 8])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        history = [{n: p.detach().clone() for n, p in model.named_parameters()}]
        for _ in range(5):
            optimizer.zero_grad()
            model(torch.randn(3, 4)).sum().backward()
            optimizer.step()
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            rng = torch.get_rng_state().clone()
            shadows.update(model)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for n, p in model.named_parameters():
                self.assertTrue(torch.equal(before[n], p))
                self.assertTrue(torch.equal(shadows.params[1][n], p))
            history.append(before)
        for n in history[0]:
            mean = torch.stack([h[n] for h in history]).mean(0)
            self.assertTrue(torch.allclose(shadows.params[8][n], mean, atol=1e-6))

    def test_forward_uses_shadow_parameters_without_touching_module(self):
        model = self.model()
        shadows = shadow.Shadows(torch, model, [4])
        x = torch.randn(2, 4)
        expected = model(x)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        moved = {n: p.detach().clone() for n, p in model.named_parameters()}
        out = torch.func.functional_call(model, shadows.params[4], (x,))
        self.assertTrue(torch.allclose(out, expected))
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(moved[n], p))

    def test_buffers_denied(self):
        with self.assertRaises(ValueError):
            shadow.Shadows(torch, torch.nn.BatchNorm1d(4), [4])


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()
