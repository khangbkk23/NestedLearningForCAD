"""B324 checks. Standard library by default; torch and reference-numpy cases skip when absent.

    python -I -B tests/test_coverage_pilot.py
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load("coverage_pilot", "scripts/coverage_pilot.py")
report = load("coverage_report", "scripts/coverage_report.py")
from nested_replay_cad import coverage  # noqa: E402

CONFIG = json.loads((ROOT / "configs/b324.json").read_text(encoding="utf8"))
TASKS = [f"t{i:02d}" for i in range(15)]
NOTEBOOK = "notebooks/b324.ipynb"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def notebook_code():
    return "".join(json.loads((ROOT / NOTEBOOK).read_text(encoding="utf8"))["cells"][0]["source"])


def study(arm, final, seed=0, acquisition=None, boundaries=15, committed=3629):
    acquisition = acquisition or final
    snapshots = []
    for boundary in range(boundaries):
        count = boundary + 1 if boundaries == 15 else 15
        row = {}
        for index in range(count):
            value = final[index] if boundary == boundaries - 1 else acquisition[index]
            row[TASKS[index]] = {"pixel": {"AP": value, "AUROC": 0.9}, "image": {"AP": 0.9, "AUROC": 0.95}}
        snapshots.append({"cursor": 1000 + boundary, "metrics": row, "model_sha256": "x"})
    return {"arm": arm, "seed": seed, "status": "COMPLETE_REVIEW_PENDING", "committed": committed,
            "snapshots": snapshots, "counters": {"optimizer": 0, "extra_updates": 0}, "seconds": 1.0}


class ContractTests(unittest.TestCase):
    def test_registered_config_loads(self):
        loaded = pilot.load_pilot(ROOT / "configs/b324.json", digest)
        self.assertEqual(loaded["seeds"], [0, 1])
        self.assertEqual(loaded["caps_status"], "APPROVED")
        self.assertEqual(loaded["caps"], {"per_arm_seconds": 18000})

    def test_changed_contract_denied(self):
        for path, value in ((("arms", "nested", "consolidation", "updates"), 64),
                            (("arms", "raw1", "reservoir_capacity"), 400),
                            (("replay", "format"), "jpeg"), (("seeds",), [0, 1, 2])):
            changed = copy.deepcopy(CONFIG)
            node = changed
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / "b324.json"
                target.write_text(json.dumps(changed), encoding="utf8")
                with self.assertRaises(ValueError):
                    pilot.load_pilot(target, digest)

    def test_byte_budget(self):
        replay = CONFIG["replay"]
        self.assertEqual(replay["bytes_per_sample"], coverage.BYTES_PER_SAMPLE)
        self.assertEqual(replay["budget_bytes"], 32 * 2 * 789 * 768 * 4)
        self.assertLessEqual(336 * coverage.BYTES_PER_SAMPLE, replay["budget_bytes"])
        self.assertGreater(337 * coverage.BYTES_PER_SAMPLE, replay["budget_bytes"])

    def test_flat_and_nested_make_the_same_number_of_updates(self):
        nested = coverage.validate_schedule(dict(pilot.NESTED))
        self.assertEqual(coverage.expected_updates(pilot.FLAT, 3629), 10887)
        self.assertEqual(coverage.expected_updates(nested, 3629), 10887)
        self.assertEqual(coverage.expected_updates({"mode": "single"}, 3629), 3629)
        # Q0 prefix: six consolidation batches by arrival 433, no tail.
        self.assertEqual(coverage.expected_updates(nested, 433), 433 + 6 * 128)
        self.assertEqual(coverage.extra_updates(nested, 3584), 128)
        self.assertEqual(coverage.extra_updates(nested, 3629), 90)
        self.assertEqual(coverage.extra_updates(nested, 100), 0)

    def test_invalid_schedules_denied(self):
        for bad in ({"mode": "flat", "extra_per_arrival": 0, "samples": 4},
                    {"mode": "nested", "every": 64, "updates": 128, "samples": 5, "tail": 90, "final_arrival": 3584},
                    {"mode": "other"}):
            with self.assertRaises(ValueError):
                coverage.validate_schedule(bad)

    def test_arm_overrides(self):
        q0 = pilot.arm_overrides(CONFIG, "q0", 0, "q0")
        self.assertEqual(q0["reservoir_capacity"], 24)
        self.assertEqual(q0["integration_arrivals"], 433)
        iid = pilot.arm_overrides(CONFIG, "iid", 0, "s1")
        self.assertTrue(iid["shuffled_stream"])
        self.assertEqual(iid["observe_at"], [3629])
        nested = pilot.arm_overrides(CONFIG, "nested", 1, "s2b")
        self.assertEqual(nested["seed"], 1)
        self.assertEqual(nested["support_forwards_cap"], 54435)
        self.assertEqual(nested["replay_encoder_cap"], 4 * 3629 + 5 * (56 * 128 + 90))
        self.assertNotIn("integration_arrivals", nested)
        for args in (("raw1", 0, "s2a"), ("flat", 1, "s2a"), ("flat", 2, None), ("er_uniform", 0, None)):
            with self.assertRaises((ValueError, KeyError)):
                pilot.arm_overrides(CONFIG, *args)

    def test_b319_recipe_has_no_b324_keys(self):
        base = json.loads((ROOT / "configs/b319.json").read_text(encoding="utf8"))
        for key in ("replay_format", "schedule", "shuffled_stream", "observe_at"):
            self.assertNotIn(key, base)


class ReplicationTests(unittest.TestCase):
    def test_r1_runs_q0_to_the_end_on_seed_1(self):
        q0 = pilot.arm_overrides(CONFIG, "q0", 1, "r1")
        self.assertNotIn("integration_arrivals", q0)
        self.assertEqual((q0["seed"], q0["reservoir_capacity"]), (1, 24))
        self.assertEqual(pilot.arm_overrides(CONFIG, "raw1", 1, "r1")["reservoir_capacity"], 328)
        with self.assertRaises(ValueError):
            pilot.arm_overrides(CONFIG, "raw1", 0, "r1")

    def test_r1_rule(self):
        def pair(base, raw, raw_acq=None):
            q0 = study("q0", [base] * 15, seed=1)
            raw1 = study("raw1", [raw] * 15, seed=1, acquisition=raw_acq)
            return q0, raw1

        self.assertEqual(report.r1_report(*pair(0.60, 0.63), CONFIG)["label"], "REPLICATED")
        self.assertEqual(report.r1_report(*pair(0.60, 0.603), CONFIG)["label"], "NOT_REPLICATED")
        self.assertEqual(report.r1_report(*pair(0.60, 0.63, [0.66] * 15), CONFIG)["label"], "NOT_REPLICATED")
        with self.assertRaises(ValueError):
            report.r1_report(study("q0", [0.6] * 15, seed=0), study("raw1", [0.6] * 15, seed=0), CONFIG)


class ReportTests(unittest.TestCase):
    def test_s2_labels(self):
        def pairs(gain_seed0, gain_seed1):
            out = []
            for seed, gain in ((0, gain_seed0), (1, gain_seed1)):
                out.append((study("flat", [0.60] * 15, seed), study("nested", [0.60 + gain] * 15, seed), True))
            return out

        self.assertEqual(report.s2_report(pairs(0.01, 0.01), CONFIG)["label"], "STRUCTURE_SIGNAL")
        self.assertEqual(report.s2_report(pairs(0.002, -0.001), CONFIG)["label"], "STRUCTURE_ADDS_NOTHING")
        self.assertEqual(report.s2_report(pairs(0.01, 0.0), CONFIG)["label"], "MIXED")
        self.assertEqual(report.s2_report(pairs(0.01, 0.01)[:1], CONFIG)["label"], "INCOMPLETE_ONE_SEED")

    def test_s2_pair_needs_one_seed(self):
        with self.assertRaises(ValueError):
            report.s2_report([(study("flat", [0.6] * 15, 0), study("nested", [0.6] * 15, 1), True)], CONFIG)

    def test_s1_prediction(self):
        recorded = json.loads(report.REFERENCE.read_text(encoding="utf8"))["arms"]["er_uniform"]
        tasks = recorded["matrix"]["tasks"]

        def renamed(item):
            for snapshot in item["snapshots"]:
                snapshot["metrics"] = {tasks[i]: v for i, v in enumerate(snapshot["metrics"].values())}
            return item

        good = renamed(study("raw1", [0.63] * 15))
        out = report.s1_report(good, None, CONFIG, recorded["matrix"])
        self.assertEqual(out["label"], "COVERAGE_SUPPORTED")
        self.assertAlmostEqual(out["er32_reference"]["final_macro_pixel_AP"], recorded["final_macro_pixel_AP"], places=12)
        forgetting = renamed(study("raw1", [0.63] * 15, acquisition=[0.66] * 15))
        self.assertEqual(report.s1_report(forgetting, None, CONFIG, recorded["matrix"])["label"], "COVERAGE_NOT_SUPPORTED")
        iid = study("iid", [0.7] * 15, boundaries=1)
        self.assertAlmostEqual(report.s1_report(good, iid, CONFIG, recorded["matrix"])["iid_reference"]["final_macro_pixel_AP"], 0.7)

    def test_q0_identity(self):
        ref = CONFIG["identity_reference"]["model_sha256"]

        def prefix(arm, hashes, optimizer, extra):
            item = study(arm, [0.5] * 15, boundaries=2, committed=433)
            item["snapshots"] = [{"cursor": 209, "model_sha256": hashes[0], "metrics": {}},
                                 {"cursor": 433, "model_sha256": hashes[1], "metrics": {}}]
            item["counters"] = {"optimizer": optimizer, "extra_updates": extra}
            item["replay_memory_bytes"] = 1000
            return item

        good = report.q0_report(prefix("q0", [ref["209"], ref["433"]], 433, 0),
                                prefix("nested", ["a", "b"], 433 + 768, 768), CONFIG)
        self.assertEqual(good["status"], "IDENTITY_REPRODUCED")
        bad = report.q0_report(prefix("q0", [ref["209"], "z"], 433, 0), prefix("nested", ["a", "b"], 1201, 768), CONFIG)
        self.assertEqual(bad["status"], "IDENTITY_MISMATCH")


class TransportTests(unittest.TestCase):
    def test_notebook_compiles_and_gates(self):
        code = notebook_code()
        compile(code, NOTEBOOK, "exec")
        self.assertIn("B324_TERMINAL", code)
        self.assertIn("JOB=RUN_ID.split('-')[1]", code)
        self.assertIn("if JOB!='q0' and cfg.get('caps_status')!='APPROVED'", code)
        self.assertLess(code.index("caps_status"), code.index("'install'"))
        self.assertLess(code.index("coverage-tests"), code.index("gpu-preflight"))
        # The contract-test gate runs before data and GPU, fatal, with the basetemp parent created first.
        self.assertLess(code.index("'.local'/'cache'/'tests'"), code.index("pytest-"))
        self.assertLess(code.index("pytest_gate"), code.index("'assets'"))
        self.assertLess(code.index("pytest_gate"), code.index("gpu-preflight"))
        self.assertNotIn("except Exception as error:\n                status.setdefault('pytest_gate'", code)
        self.assertNotIn("'-e','cad-test','test'", code)
        self.assertIn("16*1024**3", code)
        self.assertNotIn("test_shadow_pilot", code)
        self.assertNotIn("test_anchor_pilot", code)

    def test_pytest_gate_files_read_only_bundled_experiment_files(self):
        gates = re.findall(r"for gate in \(([^)]*)\)", notebook_code())
        self.assertEqual(len(gates), 1)
        files = re.findall(r"'(tests/[A-Za-z0-9_]+\.py)'", gates[0])
        self.assertTrue(files)
        for name in files:
            text = (ROOT / name).read_text(encoding="utf8")
            self.assertIsNone(re.search(r"configs/b32[0-3]|notebooks/|experiments/B3", text), name)

    def test_watcher_recognises_b324_marker(self):
        watch = load("watch_kaggle_b324", "scripts/watch_kaggle.py")
        self.assertEqual(watch.terminal_payload('B324_TERMINAL {"status":"FAILED"}'), {"status": "FAILED"})

    def test_package_and_isolated_bundle_tests(self):
        pack = load("development_packager_b324", "scripts/package_development.py")
        run_id = "b324-q0-synthetic-check"
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            names = [q.relative_to(ROOT).as_posix() for scope in ("src", "scripts", "tests")
                     for q in (ROOT / scope).rglob("*.py") if "__pycache__" not in q.parts]
            names += ["pixi.toml", "pixi.lock", "pyproject.toml", "LICENSE", ".gitattributes",
                      "scripts/activate_source.sh", "scripts/activate_source.bat", "configs/b319.json",
                      "configs/b324.json", "configs/train_manifest.json", "experiments/B324/card.md", NOTEBOOK,
                      "experiments/B319/runs/b319-s0-20261110-kaggle-v5/metrics.json"]
            for name in set(names):
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            plan = folder / "experiments/B324/attempts" / (run_id + ".md")
            plan.parent.mkdir(parents=True)
            plan.write_text("Synthetic packaging fixture; never upload or execute.\n", encoding="utf8")

            def git_output(args, **kwargs):
                return "" if "status" in args else "1" * 40

            with patch.object(pack, "ROOT", folder), \
                    patch.object(pack.subprocess, "check_output", side_effect=git_output), \
                    patch.object(pack.subprocess, "run"), contextlib.redirect_stdout(io.StringIO()):
                run = pack.package("B324", "owner-b", run_id)
            metadata = json.loads((run / "package/kernel/kernel-metadata.json").read_text(encoding="utf8"))
            self.assertTrue(metadata["is_private"])
            packaged = "".join(json.loads((run / "package/kernel/b324.ipynb").read_text(encoding="utf8"))["cells"][0]["source"])
            self.assertNotIn("__RUN_ID__", packaged)
            self.assertIn("RUN_ID='b324-q0-synthetic-check'", packaged)
            isolated = folder / "isolated"
            with zipfile.ZipFile(run / "package/dataset/source.zip") as archive:
                for required in ("scripts/coverage_pilot.py", "scripts/coverage_report.py",
                                 "src/nested_replay_cad/coverage.py", "tests/test_coverage_pilot.py",
                                 "configs/b324.json", "experiments/B324/card.md"):
                    self.assertIn(required, archive.namelist())
                archive.extractall(isolated)
            calls = re.findall(r"'(tests/[A-Za-z0-9_]+\.py)'((?:,'[A-Za-z0-9_.]+')*)\]", packaged)
            self.assertTrue(calls)
            for test, names in calls:
                classes = re.findall(r"'([A-Za-z0-9_.]+)'", names)
                result = subprocess.run([sys.executable, "-I", "-B", str(isolated / test), *classes], cwd=isolated,
                                        capture_output=True, text=True, encoding="utf8", timeout=120)
                self.assertEqual(result.returncode, 0, test + "\n" + result.stdout + result.stderr)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch not installed")
class TorchCoverageTests(unittest.TestCase):
    def png(self, folder, size=(700, 513)):
        import numpy as np
        from PIL import Image

        generator = np.random.default_rng(7)
        path = Path(folder) / "sample.png"
        Image.fromarray(generator.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        return path

    def test_constants_match_encoder(self):
        from nested_replay_cad.model import encoder

        self.assertEqual((coverage.MEAN, coverage.STD, coverage.SIZE), (encoder.MEAN, encoder.STD, encoder.SIZE))

    def test_uint8_round_trip_is_bitwise_preprocess(self):
        from nested_replay_cad.model import encoder

        with tempfile.TemporaryDirectory() as folder:
            for size in ((700, 513), (392, 392), (1024, 1024)):
                path = self.png(folder, size)
                stored = coverage.load_uint8(path)
                self.assertEqual((stored.dtype, tuple(stored.shape)), (torch.uint8, (3, 392, 392)))
                self.assertEqual(stored.numel(), coverage.BYTES_PER_SAMPLE)
                self.assertTrue(torch.equal(coverage.normalize_uint8(stored), encoder.preprocess(path)))

    def test_wrong_stored_tensor_denied(self):
        with self.assertRaises(ValueError):
            coverage.normalize_uint8(torch.zeros((3, 392, 392), dtype=torch.float32))
        with self.assertRaises(ValueError):
            coverage.normalize_uint8(torch.zeros((3, 224, 224), dtype=torch.uint8))

    def test_store_digest_tracks_content(self):
        store = {"a" * 64: torch.zeros((3, 392, 392), dtype=torch.uint8)}
        before = coverage.store_digest(store)
        store["a" * 64][0, 0, 0] = 1
        self.assertNotEqual(before, coverage.store_digest(store))


try:
    import numpy
    from nested_replay_cad.core.memory.rng import seed_sampling
    SAMPLER = numpy.__version__ == "2.4.5"
except Exception:  # numpy absent or another version: the registered RNG cannot run
    SAMPLER = False


@unittest.skipUnless(SAMPLER, "reference numpy 2.4.5 not installed")
class SamplerCoverageTests(unittest.TestCase):
    IDS = tuple(hashlib.sha256(str(i).encode()).hexdigest() for i in range(400))

    def test_aux_streams_are_deterministic_and_separate(self):
        self.assertEqual(coverage.aux_stream(0, "consolidation"), coverage.aux_stream(0, "consolidation"))
        self.assertNotEqual(coverage.aux_stream(0, "consolidation"), coverage.aux_stream(1, "consolidation"))
        self.assertNotEqual(coverage.aux_stream(0, "consolidation"), coverage.aux_stream(0, "stream_order"))
        registered = seed_sampling(0)
        for state in (registered.admission, registered.recent, registered.historical, registered.warmup):
            self.assertNotEqual(state, coverage.aux_stream(0, "consolidation"))
        with self.assertRaises(ValueError):
            coverage.aux_stream(0, "admission")

    def test_uniform_sample_without_replacement(self):
        state = coverage.aux_stream(0, "consolidation")
        picked, state = coverage.uniform_sample(state, self.IDS[:328], 5)
        self.assertEqual(len(set(picked)), 5)
        self.assertTrue(set(picked) <= set(self.IDS[:328]))
        few, _ = coverage.uniform_sample(state, self.IDS[:3], 5)
        self.assertEqual(sorted(few), sorted(self.IDS[:3]))

    def test_shuffled_stream_is_a_seeded_permutation(self):
        records = [{"opaque_id": identity, "arrival": index, "task_audit_only": "x"}
                   for index, identity in enumerate(self.IDS, 1)]
        first = coverage.shuffled_records(records, 0)
        self.assertEqual(first, coverage.shuffled_records(records, 0))
        self.assertEqual([row["arrival"] for row in first], list(range(1, len(records) + 1)))
        self.assertEqual(sorted(row["opaque_id"] for row in first), sorted(self.IDS))
        self.assertNotEqual([row["opaque_id"] for row in first], list(self.IDS))
        self.assertNotEqual(first, coverage.shuffled_records(records, 1))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()
