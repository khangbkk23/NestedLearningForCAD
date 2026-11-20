"""B322 packaging and bundled-design checks. Standard library only; never uploads or launches.

    python -I -B tests/test_b322_transport.py
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pack = load("development_packager_b322", "scripts/package_development.py")
pilot = load("shadow_pilot_b322", "scripts/shadow_pilot.py")
sys.path.insert(0, str(ROOT / "src"))
from nested_replay_cad import provenance  # noqa: E402

NOTEBOOKS = ("notebooks/b322.ipynb", "notebooks/b322_profile.ipynb")


def source_text(relative):
    return "".join(json.loads((ROOT / relative).read_text(encoding="utf8"))["cells"][0]["source"])


def make_fixture(folder, run_id):
    """Copy the files a B322 package reads, plus a synthetic attempt plan."""
    names = [p.relative_to(ROOT).as_posix() for scope in ("src", "scripts", "tests") for p in (ROOT / scope).rglob("*.py")
             if "__pycache__" not in p.parts]
    names += ["pixi.toml", "pixi.lock", "pyproject.toml", "LICENSE", ".gitattributes",
              "scripts/activate_source.sh", "scripts/activate_source.bat", "configs/b319.json",
              "configs/b322.json", "configs/train_manifest.json", "experiments/B322/card.md", *NOTEBOOKS]
    for name in set(names):
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    plan = folder / "experiments/B322/attempts" / (run_id + ".md")
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("Synthetic packaging fixture; never upload or execute.\n", encoding="utf8")


def package(folder, owner, run_id, template=None):
    def git_output(args, **kwargs):
        return "" if "status" in args else "1" * 40

    with patch.object(pack, "ROOT", folder), patch.object(pack.subprocess, "check_output", side_effect=git_output), \
            patch.object(pack.subprocess, "run"), contextlib.redirect_stdout(io.StringIO()):
        return pack.package("B322", owner, run_id, template)


class NotebookTests(unittest.TestCase):
    def test_notebooks_compile_and_print_the_watcher_marker(self):
        for name in NOTEBOOKS:
            code = source_text(name)
            compile(code, name, "exec")
            self.assertIn("B322_TERMINAL", code)
            self.assertIn("__RUN_ID__", code)

    def test_full_notebook_refuses_provisional_caps_before_any_install(self):
        code = source_text("notebooks/b322.ipynb")
        self.assertIn("scripts/shadow_pilot.py", code)
        self.assertLess(code.index("caps_status"), code.index("'install'"))
        self.assertLess(code.index("torch-tests"), code.index("gpu-preflight"))
        self.assertIn("43200", code)

    def test_profile_notebook_uses_no_dataset_or_weights(self):
        code = source_text("notebooks/b322_profile.ipynb")
        self.assertIn("scripts/shadow_profile.py", code)
        for forbidden in ("prepare_assets", "scripts/preflight.py", "scripts/shadow_pilot.py", "scripts/train.py"):
            self.assertNotIn(forbidden, code)


class PackagingTests(unittest.TestCase):
    def test_two_owners_and_both_templates_package_privately(self):
        for owner, template in (("owner-a", None), ("owner-b", "notebooks/b322_profile.ipynb")):
            with tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                run_id = "b322-synthetic-package-check"
                make_fixture(folder, run_id)
                run = package(folder, owner, run_id, template)
                with self.assertRaises(FileExistsError):
                    package(folder, owner, run_id, template)
                record = json.loads((run / "run.json").read_text(encoding="utf8"))
                self.assertEqual(record["experiment"], "B322")
                self.assertTrue(record["is_private"])
                metadata = json.loads((run / "package/kernel/kernel-metadata.json").read_text(encoding="utf8"))
                self.assertTrue(metadata["is_private"])
                self.assertEqual(metadata["id"], owner + "/" + run_id)
                self.assertEqual(metadata["code_file"], "b322.ipynb")
                packaged = "".join(json.loads((run / "package/kernel/b322.ipynb").read_text(encoding="utf8"))["cells"][0]["source"])
                self.assertNotIn("__RUN_ID__", packaged)
                self.assertIn(record["source_zip_sha256"], packaged)
                with zipfile.ZipFile(run / "package/dataset/source.zip") as archive:
                    names = archive.namelist()
                    for required in ("scripts/shadow_pilot.py", "scripts/shadow_profile.py", "src/nested_replay_cad/shadow.py",
                                     "tests/test_shadow_pilot.py", "configs/b322.json", "configs/b319.json",
                                     "experiments/B322/card.md", "experiments/B322/attempts/" + run_id + ".md", *NOTEBOOKS):
                        self.assertIn(required, names)
                    for name, digest in record["files"].items():
                        self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)

    def test_invalid_owner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid owner"):
            pack.package("B320", "Not An Owner", "b320-synthetic-check")

    def test_unknown_experiment_and_wrong_template_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unregistered experiment"):
            pack.package("B399", "owner-a", "b399-synthetic-check")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            make_fixture(folder, "b322-synthetic-check")
            with self.assertRaisesRegex(ValueError, "Unregistered notebook"):
                package(folder, "owner-a", "b322-synthetic-check", "notebooks/b321.ipynb")


class BundledDesignTests(unittest.TestCase):
    def extracted(self, folder):
        run_id = "b322-synthetic-bundle-check"
        make_fixture(folder, run_id)
        run = package(folder, "owner-b", run_id)
        root = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(run / "package/dataset/source.zip") as archive:
            archive.extractall(root)
        return root

    def test_card_and_recipe_checked_as_subset_in_a_git_free_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.extracted(Path(folder))
            try:
                with patch.object(pilot, "ROOT", root):
                    self.assertEqual(pilot.check_bundled_design(provenance), "1" * 40)
                    (root / "configs/b322.json").write_text("{}\n", encoding="utf8")
                    with self.assertRaisesRegex(ValueError, "differs from SOURCE.json"):
                        pilot.check_bundled_design(provenance)
            finally:
                shutil.rmtree(root, ignore_errors=True)

    def test_card_tamper_is_denied(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.extracted(Path(folder))
            try:
                (root / "experiments/B322/card.md").write_text("changed\n", encoding="utf8")
                with patch.object(pilot, "ROOT", root):
                    with self.assertRaisesRegex(ValueError, "differs from SOURCE.json"):
                        pilot.check_bundled_design(provenance)
            finally:
                shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()
