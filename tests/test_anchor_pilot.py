"""B323 checks. Standard library by default; torch and numpy cases skip when absent.

    python -I -B tests/test_anchor_pilot.py
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load("anchor_pilot", "scripts/anchor_pilot.py")
report = load("anchor_report", "scripts/anchor_report.py")
anchor = load("anchor", "src/nested_replay_cad/anchor.py")
CONFIG = json.loads((ROOT / "configs/b323.json").read_text(encoding="utf8"))
TASKS = [f"t{i:02d}" for i in range(15)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def study(arm, final, acquisition=None, first_hash=None):
    acquisition = acquisition or final
    snapshots = []
    for boundary in range(15):
        row = {}
        for index in range(boundary + 1):
            value = final[index] if boundary == 14 else acquisition[index]
            row[TASKS[index]] = {"pixel": {"AP": value, "AUROC": 0.9}, "image": {"AP": 0.9, "AUROC": 0.95}}
        snapshots.append({"cursor": 209 if boundary == 0 else 1000 + boundary, "metrics": row,
                          "model_sha256": first_hash if boundary == 0 and first_hash else "x"})
    return {"arm": arm, "status": "COMPLETE_REVIEW_PENDING", "committed": 3629, "snapshots": snapshots,
            "counters": {"anchor_forwards": 7}, "reservoir_capacity": 74 if arm == "er_bytes" else 24}


class ContractTests(unittest.TestCase):
    def test_registered_config_loads(self):
        loaded = pilot.load_pilot(ROOT / "configs/b323.json", digest)
        self.assertEqual(loaded["arms"], ["er_bytes", "anchor"])
        self.assertEqual(loaded["caps_status"], "APPROVED")
        self.assertEqual(loaded["caps"], {"per_arm_seconds": 14400})

    def test_changed_contract_denied(self):
        for key, value in (("anchor", {"weight": 0.5}), ("er_bytes", {"reservoir_capacity": 24}), ("seed", 1)):
            changed = copy.deepcopy(CONFIG)
            changed[key] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "b323.json"
                path.write_text(json.dumps(changed), encoding="utf8")
                with self.assertRaises(ValueError):
                    pilot.load_pilot(path, digest)

    def test_arm_overrides_are_disjoint_and_byte_matched(self):
        control = pilot.arm_overrides(CONFIG, "er_bytes", False)
        candidate = pilot.arm_overrides(CONFIG, "anchor", False)
        self.assertEqual(control["reservoir_capacity"], 74)
        self.assertNotIn("anchor", control)
        self.assertEqual(candidate["anchor"], {"weight": 1.0})
        self.assertNotIn("reservoir_capacity", candidate)
        self.assertNotIn("integration_arrivals", candidate)
        # 50 extra replay features versus one decoder copy, in bytes.
        extra_replay = (74 - 24) * 2 * 789 * 768 * 4
        decoder = 61425408 * 4
        self.assertLess(abs(extra_replay - decoder) / decoder, 0.02)

    def test_qualification_stops_at_second_boundary(self):
        self.assertEqual(pilot.arm_overrides(CONFIG, "anchor", True)["integration_arrivals"], 433)
        with self.assertRaises(ValueError):
            pilot.arm_overrides(CONFIG, "er_uniform", False)

    def test_b319_recipe_has_no_b323_keys(self):
        base = json.loads((ROOT / "configs/b319.json").read_text(encoding="utf8"))
        self.assertNotIn("anchor", base)
        self.assertNotIn("reservoir_capacity", base)

    def test_one_gpu_slot_per_arm(self):
        self.assertEqual(sorted(pilot.SLOTS.values()), ["0", "1"])
        self.assertEqual(set(pilot.SLOTS), set(CONFIG["arms"]))


class ReportTests(unittest.TestCase):
    HASH = CONFIG["prefix_reference"]["model_sha256"]

    def run_report(self, control_final, candidate_final, candidate_acq=None, first_hash=None):
        return report.report(study("er_bytes", control_final, [0.60] * 15),
                             study("anchor", candidate_final, candidate_acq or [0.60] * 15, first_hash or self.HASH),
                             CONFIG)

    def test_labels(self):
        self.assertEqual(self.run_report([0.50] * 15, [0.51] * 15)["verdict"], "SIGNAL")
        self.assertEqual(self.run_report([0.50] * 15, [0.502] * 15)["verdict"], "NO_SIGNAL")
        floor = [0.53] * 15
        floor[4] = 0.48
        out = self.run_report([0.50] * 15, floor)
        self.assertEqual(out["verdict"], "MIXED")
        self.assertEqual(out["primary"]["floor_violations"], [TASKS[4]])
        lagging = self.run_report([0.50] * 15, [0.51] * 15, [0.58] * 15)
        self.assertEqual(lagging["verdict"], "MIXED")
        self.assertFalse(lagging["primary"]["conditions"]["c_plasticity"])

    def test_age_profile_and_compute_flag(self):
        control = study("er_bytes", [0.50] * 15, [0.60] * 15)
        candidate = study("anchor", [0.52] * 15, [0.60] * 15, self.HASH)
        out = report.report(control, candidate, CONFIG)
        profile = out["age_profile"]
        self.assertEqual(set(profile), {str(a) for a in range(15)})
        # Age 0: 14 acquisition cells equal at 0.60, plus the last task whose final values differ by 0.02.
        self.assertAlmostEqual(profile["0"]["mean"], 0.02 / 15)
        self.assertEqual(profile["0"]["cells"], 15)
        self.assertAlmostEqual(profile["14"]["mean"], 0.02)  # task 0 at the last boundary: final values
        self.assertEqual(profile["14"]["cells"], 1)
        self.assertFalse(out["compute"]["matched"])

    def test_prefix_check(self):
        self.assertEqual(self.run_report([0.5] * 15, [0.5] * 15)["prefix_check"]["status"], "PREFIX_REPRODUCED")
        self.assertEqual(self.run_report([0.5] * 15, [0.5] * 15, first_hash="other")["prefix_check"]["status"],
                         "PREFIX_MISMATCH")

    def test_incomplete_or_swapped_arms_denied(self):
        with self.assertRaises(ValueError):
            report.report(study("anchor", [0.5] * 15), study("er_bytes", [0.5] * 15), CONFIG)
        broken = study("anchor", [0.5] * 15)
        broken["committed"] = 433
        with self.assertRaises(ValueError):
            report.report(study("er_bytes", [0.5] * 15), broken, CONFIG)

    def test_reference_er32_is_reported_not_gated(self):
        recorded = json.loads((ROOT / "experiments/B319/runs/b319-s0-20261110-kaggle-v5/metrics.json")
                              .read_text(encoding="utf8"))["arms"]["er_uniform"]
        control, candidate = study("er_bytes", [0.5] * 15), study("anchor", [0.5] * 15, first_hash=self.HASH)
        tasks = recorded["matrix"]["tasks"]
        for item in (control, candidate):
            for snapshot in item["snapshots"]:
                snapshot["metrics"] = {tasks[i]: v for i, v in enumerate(snapshot["metrics"].values())}
        out = report.report(control, candidate, CONFIG, recorded["matrix"])
        self.assertAlmostEqual(out["reference_er32"]["summary"]["final_macro_pixel_AP"],
                               recorded["final_macro_pixel_AP"], places=12)
        self.assertEqual(out["verdict"], "NO_SIGNAL")


class TransportTests(unittest.TestCase):
    NOTEBOOKS = ("notebooks/b323.ipynb", "notebooks/b323_qualify.ipynb")

    def code(self, name):
        return "".join(json.loads((ROOT / name).read_text(encoding="utf8"))["cells"][0]["source"])

    def test_notebooks_compile_and_run_both_arms_in_parallel(self):
        for name in self.NOTEBOOKS:
            code = self.code(name)
            compile(code, name, "exec")
            self.assertIn("B323_TERMINAL", code)
            self.assertIn("'--parallel'", code)
            self.assertIn("16*1024**3", code)
            self.assertLess(code.index("anchor-tests"), code.index("gpu-preflight"))

    def test_full_notebook_gates_on_caps_and_qualification_is_capped(self):
        full, qualify = (self.code(name) for name in self.NOTEBOOKS)
        self.assertLess(full.index("caps_status"), full.index("'install'"))
        self.assertNotIn("'--qualify'", full)
        self.assertIn("'--qualify'", qualify)
        self.assertIn("DEADLINE=START+3600", qualify)
        self.assertNotIn("caps_status", qualify)

    def test_watcher_recognises_b323_marker(self):
        watch = load("watch_kaggle_b323", "scripts/watch_kaggle.py")
        payload = watch.terminal_payload('5s B323_TERMINAL {"status":"COMPLETE_REVIEW_PENDING"}')
        self.assertEqual(payload, {"status": "COMPLETE_REVIEW_PENDING"})
        self.assertIsNone(watch.terminal_payload('B323_PARALLEL_RESULT {"status":"FAILED"}'))

    def test_package_contains_b323_design_and_sources(self):
        import contextlib
        import io
        import shutil
        import zipfile
        from unittest.mock import patch

        pack = load("development_packager_b323", "scripts/package_development.py")
        run_id = "b323-synthetic-package-check"
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            names = [q.relative_to(ROOT).as_posix() for scope in ("src", "scripts", "tests")
                     for q in (ROOT / scope).rglob("*.py") if "__pycache__" not in q.parts]
            names += ["pixi.toml", "pixi.lock", "pyproject.toml", "LICENSE", ".gitattributes",
                      "scripts/activate_source.sh", "scripts/activate_source.bat", "configs/b319.json",
                      "configs/b323.json", "configs/train_manifest.json", "experiments/B323/card.md", *self.NOTEBOOKS]
            for name in set(names):
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            plan = folder / "experiments/B323/attempts" / (run_id + ".md")
            plan.parent.mkdir(parents=True)
            plan.write_text("Synthetic packaging fixture; never upload or execute.\n", encoding="utf8")

            def git_output(args, **kwargs):
                return "" if "status" in args else "1" * 40

            with patch.object(pack, "ROOT", folder), patch.object(pack.subprocess, "check_output", side_effect=git_output), \
                    patch.object(pack.subprocess, "run"), contextlib.redirect_stdout(io.StringIO()):
                run = pack.package("B323", "owner-b", run_id, "notebooks/b323_qualify.ipynb")
            metadata = json.loads((run / "package/kernel/kernel-metadata.json").read_text(encoding="utf8"))
            self.assertTrue(metadata["is_private"])
            self.assertEqual(metadata["code_file"], "b323.ipynb")
            packaged = "".join(json.loads((run / "package/kernel/b323.ipynb").read_text(encoding="utf8"))["cells"][0]["source"])
            self.assertIn("'--qualify'", packaged)
            self.assertNotIn("__RUN_ID__", packaged)
            with zipfile.ZipFile(run / "package/dataset/source.zip") as archive:
                for required in ("scripts/anchor_pilot.py", "scripts/anchor_report.py", "src/nested_replay_cad/anchor.py",
                                 "tests/test_anchor_pilot.py", "configs/b323.json", "experiments/B323/card.md"):
                    self.assertIn(required, archive.namelist())
                isolated = folder / "isolated"
                archive.extractall(isolated)
            # Every test file the notebook runs must import inside the bundle alone
            # (attempt b323-qual-20261115-01 failed on a test that read configs/b322.json).
            import re
            import subprocess

            calls = re.findall(r"'(tests/[A-Za-z0-9_]+\.py)'((?:,'[A-Za-z0-9_.]+')*)\]", packaged)
            self.assertTrue(calls)
            for test, names in calls:
                # Exactly the classes the notebook runs (here they skip without torch;
                # what matters is that the module imports from the bundle alone).
                classes = re.findall(r"'([A-Za-z0-9_.]+)'", names)
                result = subprocess.run([sys.executable, "-I", "-B", str(isolated / test), *classes], cwd=isolated,
                                        capture_output=True, text=True, encoding="utf8", timeout=120)
                self.assertEqual(result.returncode, 0, test + "\n" + result.stdout + result.stderr)
            self.assertNotIn("test_shadow_pilot", packaged)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch not installed")
class TorchAnchorTests(unittest.TestCase):
    class Toy(torch.nn.Module if torch is not None else object):
        """Two-input module returning (encoder groups, decoder groups) like the real decoder."""

        def __init__(self):
            super().__init__()
            self.a = torch.nn.Conv2d(3, 3, 1)
            self.b = torch.nn.Conv2d(3, 3, 1)
            self.noise = torch.nn.Dropout(0.5)

        def forward(self, first, second):
            return [first, second], [self.a(self.noise(first)), self.b(self.noise(second))]

    def setUp(self):
        torch.manual_seed(5)
        self.model = self.Toy()
        self.feature = torch.randn(2, 3, 4, 4)

    def forward(self):
        return self.model(self.feature[None, 0], self.feature[None, 1])

    def test_inactive_until_first_boundary(self):
        slow = anchor.SlowLevel(torch, self.model)
        self.assertFalse(slow.active)
        self.assertFalse(slow.covers(1))
        with self.assertRaises(ValueError):
            slow.distance(self.model, self.feature, self.forward()[1])

    def test_distance_is_zero_at_consolidation_and_positive_after_drift(self):
        slow = anchor.SlowLevel(torch, self.model)
        slow.consolidate(self.model, 10)
        self.model.eval()
        self.assertLess(float(slow.distance(self.model, self.feature, self.forward()[1])), 1e-6)
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p))
        self.assertGreater(float(slow.distance(self.model, self.feature, self.forward()[1])), 1e-3)
        self.assertTrue(slow.covers(10))
        self.assertFalse(slow.covers(11))
        with self.assertRaises(ValueError):
            slow.consolidate(self.model, 10)

    def test_slow_forward_keeps_mode_rng_and_fast_parameters(self):
        slow = anchor.SlowLevel(torch, self.model)
        slow.consolidate(self.model, 3)
        self.model.train()
        prediction = self.forward()[1]
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        rng = torch.get_rng_state().clone()
        digest_before = slow.digest()
        slow.distance(self.model, self.feature, prediction)
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(digest_before, slow.digest())
        for n, p in self.model.named_parameters():
            self.assertTrue(torch.equal(before[n], p))

    def test_anchor_gradient_is_not_scaled_by_later_hooks(self):
        """Trainer order: anchor backward first, then a loss that hooks ``prediction``."""
        slow = anchor.SlowLevel(torch, self.model)
        slow.consolidate(self.model, 3)
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(0.3)
        self.model.eval()

        def anchor_only():
            self.model.zero_grad()
            slow.distance(self.model, self.feature, self.forward()[1]).backward()
            return {n: p.grad.clone() for n, p in self.model.named_parameters()}

        expected = anchor_only()
        self.model.zero_grad()
        prediction = self.forward()[1]
        slow.distance(self.model, self.feature, prediction).backward(retain_graph=True)
        for tensor in prediction:
            tensor.register_hook(lambda g: g * 0.0)
        sum(t.sum() for t in prediction).backward()
        for n, p in self.model.named_parameters():
            self.assertTrue(torch.allclose(p.grad, expected[n], atol=1e-6))

    def test_buffers_denied(self):
        with self.assertRaises(ValueError):
            anchor.SlowLevel(torch, torch.nn.BatchNorm1d(4))


try:
    import numpy
    from nested_replay_cad.core.memory.rng import seed_sampling
    from nested_replay_cad.core.memory.sampler import admission_slot, sample_reservations
    SAMPLER = numpy.__version__ == "2.4.5"
except Exception:  # numpy absent or another version: the registered RNG cannot run
    SAMPLER = False


@unittest.skipUnless(SAMPLER, "reference numpy 2.4.5 not installed")
class SamplerCapacityTests(unittest.TestCase):
    def simulate(self, arrivals, **keywords):
        from dataclasses import replace

        rng = seed_sampling(0)
        reservoir, fifo, trace = (), (), []
        for arrival in range(1, arrivals + 1):
            identity = hashlib.sha256(str(arrival).encode()).hexdigest()
            reserved, advanced = sample_reservations(reservoir, fifo, rng, **keywords)
            slot, admission = admission_slot(advanced.admission, arrival, **keywords)
            staged = list(reservoir)
            capacity = keywords.get("capacity", 24)
            if slot is not None:
                if arrival <= capacity:
                    staged.append(identity)
                else:
                    staged[slot] = identity
            reservoir, fifo = tuple(staged), (*fifo, identity)[-8:]
            rng = replace(advanced, admission=admission)
            trace.append((reserved.R_reserved, reservoir))
        return trace

    def test_default_equals_explicit_24(self):
        self.assertEqual(self.simulate(300), self.simulate(300, capacity=24))

    def test_capacity_74_fills_and_never_overflows(self):
        trace = self.simulate(600, capacity=74)
        sizes = [len(reservoir) for _, reservoir in trace]
        self.assertEqual(max(sizes), 74)
        self.assertEqual(sizes[73], 74)
        self.assertTrue(all(len(support) <= 4 for support, _ in trace))
        self.assertNotEqual(trace[-1][1], self.simulate(600)[-1][1])

    def test_invalid_capacity_denied(self):
        from nested_replay_cad.core.errors import SchemaError

        rng = seed_sampling(0)
        for capacity in (0, 23, True, 24.0, 5000):
            with self.assertRaises(SchemaError):
                sample_reservations((), (), rng, capacity=capacity)
            with self.assertRaises(SchemaError):
                admission_slot(rng.admission, 1, capacity=capacity)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()
