"""One preregistered, cached-feature R1 comparison. No image/extractor access."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Preserve an explicitly supplied workspace spelling on case-insensitive WSL mounts.
ROOT = Path(os.environ.get("HOPE_CAD_WORKSPACE", str(Path(__file__).resolve().parents[2])))
FEATURES = ROOT / "results/hope_cad/frozen_features_v1"
OUT = ROOT / "results/hope_cad/r1_conditional_pairing"
CATEGORIES = ("bottle", "carpet", "hazelnut")
COUNTS = dict(zip(CATEGORIES, (209, 280, 391)))
CENTERS = tuple(r * 28 + c for r in (3, 10, 17, 24) for c in (3, 10, 17, 24))
CHECKPOINT_SHA = "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415"
MANIFEST_DIGEST = "060579457d344992eef159b1d4f1a86c34dd9283b97b70d3c3a386006b0f4dcc"
CACHE_SHA = {
    "clean/bottle_train.pt": "ef4f3180d2381b64b4cdafea447b9df254b7522b9338f24992f4a857114ad6b3",
    "clean/carpet_train.pt": "f3b1957d7b9d12add45c4ff9c33a5f9f5e237ff7ce6dbdb7693c81527c89b179",
    "clean/hazelnut_train.pt": "7bd97accb4a927285881d8c8b61cbc5db8106e57e2ceee1fe623f5903f62d180",
    "masked_rgb16/bottle_train.pt": "5ed6df107889499a0d2d552bc72ab7d007c1a531e17eda6b6ff48fd6bc6712ae",
    "masked_rgb16/carpet_train.pt": "69036fa883b6faf212b79d5eef5e58de83a01e38e12e33b7f0a034fbe302da6d",
    "masked_rgb16/hazelnut_train.pt": "a99680d755a581f36e5b20c0ff60f857dbb39cc54313887494b350d9f03120f4",
}
ARMS = ("mean", "ridge", "shuffled", "neural")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(value):
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, torch.Tensor):
            q = v.detach().cpu().contiguous()
            h.update(str((tuple(q.shape), q.dtype)).encode())
            h.update(q.numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); add(v[k])
        elif isinstance(v, (list, tuple)):
            for a in v: add(a)
        else:
            h.update(repr(v).encode())
    add(value)
    return h.hexdigest()


def tensor_bytes(value):
    if isinstance(value, torch.Tensor): return value.numel() * value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)): return sum(tensor_bytes(v) for v in value)
    return 0


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def environment():
    return {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "threads": torch.get_num_threads(), "device": "cpu",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def derangement(n, seed):
    require(n > 1, "singleton block cannot be deranged")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    result = torch.empty(n, dtype=torch.int64)
    result[order] = order.roll(1)
    require(bool((result != torch.arange(n)).all()), "derangement contains fixed point")
    return result


def validate_roles(updates, probes):
    u = {r["relative_path"] for r in updates}; p = {r["relative_path"] for r in probes}
    require(len(u) == len(updates) and len(p) == len(probes) and not u & p, "identity overlap")
    uh = {r["raw_png_sha256"] for r in updates}; ph = {r["raw_png_sha256"] for r in probes}
    require(len(uh) == len(updates) and len(ph) == len(probes) and not uh & ph, "content overlap")
    require(all("/train/good/" in r["relative_path"] for r in updates + probes), "non-normal path")


def validate_alignment(clean, masked, entries, category):
    paths = [r["relative_path"] for r in entries]
    hashes = [r["raw_png_sha256"] for r in entries]
    for view, obj in (("clean", clean), ("masked_rgb16", masked)):
        require(obj["view"] == view and obj["class_name"] == category, "wrong cache view/category")
        require(obj["relative_paths"] == paths and obj["per_image_source_sha256"] == hashes,
                "cache identity alignment failed")
        require(obj["source_manifest_digest"] == MANIFEST_DIGEST, "cache manifest mismatch")
        require(obj["patches"].shape == (len(paths), 784, 768) and obj["patches"].dtype == torch.float32,
                "cache shape/dtype mismatch")
        require(obj["metadata"]["checkpoint_sha256"] == CHECKPOINT_SHA, "cache checkpoint mismatch")
        require(obj["metadata"]["block_index"] == 8 and obj["metadata"]["feature_normalization"] == "none",
                "cache transform mismatch")


def make_input(context):
    # Only masked context and explicit position enter the predictor, never clean targets.
    n, p, _ = context.shape
    return torch.cat((context, torch.eye(p, dtype=context.dtype).expand(n, p, p)), dim=-1)


class Mean:
    def __init__(self, dim=768, positions=16):
        self.sum = torch.zeros(positions, dim, dtype=torch.float64)
        self.count = 0
    def update(self, z):
        self.sum += z.double().sum(0); self.count += len(z)
    def predict(self, c):
        require(self.count > 0, "empty mean")
        return (self.sum / self.count).expand(len(c), -1, -1)
    def state(self): return {"sum": self.sum, "count": self.count}


class Ridge:
    """Shared linear slope, unpenalized position intercepts; exact cumulative normal equations."""
    def __init__(self, dim=768, positions=16):
        self.sx = torch.zeros(positions, dim, dtype=torch.float64)
        self.sz = torch.zeros_like(self.sx)
        self.xx = torch.zeros(dim, dim, dtype=torch.float64)
        self.xz = torch.zeros_like(self.xx)
        self.count = 0
        self.w = torch.zeros_like(self.xx)
        self.b = torch.zeros_like(self.sz)
        self.lam = None
    def update(self, c, z):
        x, y = c.double(), z.double()
        self.sx += x.sum(0); self.sz += y.sum(0); self.count += len(x)
        a, b = x.flatten(0, 1), y.flatten(0, 1)
        self.xx += a.T @ a; self.xz += a.T @ b
    def fit(self):
        g = self.xx - self.sx.T @ self.sx / self.count
        g = (g + g.T) * 0.5
        cross = self.xz - self.sx.T @ self.sz / self.count
        self.lam = max(1e-12, 0.01 * float(torch.trace(g)) / len(g))
        system = g + self.lam * torch.eye(len(g), dtype=torch.float64)
        self.w = torch.cholesky_solve(cross, torch.linalg.cholesky(system))
        self.b = self.sz / self.count - (self.sx / self.count) @ self.w
        residual = float((system @ self.w - cross).norm() / max(float(cross.norm()), 1e-12))
        require(residual < 1e-8 and bool(torch.isfinite(self.w).all()), "ridge solve failed")
        return {"lambda": self.lam, "relative_normal_equation_residual": residual}
    def predict(self, c): return c.double() @ self.w + self.b
    def state(self): return vars(self)


class Neural(nn.Module):
    def __init__(self, dim=768, positions=16, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + positions, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, context): return self.net(make_input(context))


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0, foreach=False)


def image_loss(pred, z):
    return (pred.double() - z.double()).square().sum(-1).mean(-1)


def step(model, opt, c, z, write=True):
    before = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad(): old = model(c).clone()
    if write:
        opt.zero_grad(set_to_none=True)
        loss = (model(c) - z).square().sum(-1).mean()
        require(bool(torch.isfinite(loss)), "nonfinite neural loss")
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        after = model(c)
        change = sum(float((p - q).double().square().sum()) for p, q in zip(model.parameters(), before)) ** 0.5
        output_change = float((after - old).double().square().sum(-1).mean().sqrt())
        post_loss = float(image_loss(after, z).mean())
    require(np.isfinite(post_loss) and np.isfinite(change), "nonfinite neural update")
    return {"pre_loss": float(image_loss(old, z).mean()), "post_loss": post_loss,
            "parameter_delta_l2": change, "output_delta_rms_l2": output_change}


def inference(model, c):
    with torch.no_grad():
        return model(c) if isinstance(model, Neural) else model.predict(c)


def model_state(model):
    return model.state_dict() if isinstance(model, Neural) else model.state()


def feasibility():
    # Synthetic only: no cached observations, labels, or held-out losses.
    torch.manual_seed(7111)
    c = F.normalize(torch.randn(10, 16, 768), dim=-1); z = F.normalize(torch.randn_like(c), dim=-1)
    n = Neural(); o = optimizer(n)
    t = time.perf_counter()
    for _ in range(20): step(n, o, c, z)
    neural_step_seconds = (time.perf_counter() - t) / 20
    r = Ridge(); t = time.perf_counter(); r.update(c, z); r.fit()
    ridge_seconds = time.perf_counter() - t
    estimate = 2 * 35 * 50 * neural_step_seconds + 35 * ridge_seconds
    require(estimate * 3 < 3300, "threefold feasibility allowance exceeds experiment budget")
    return {"synthetic_only": True, "neural_step_seconds": neural_step_seconds,
            "ridge_update_and_solve_seconds": ridge_seconds, "projected_compute_seconds": estimate,
            "threefold_allowance_seconds": estimate * 3, "environment": environment()}


def prepare():
    require(not (OUT / "protocol_resolved.json").exists(), "protocol already frozen; refusing overwrite")
    require(not (OUT / "run_started.json").exists(), "experiment already attempted")
    t = time.perf_counter()
    docs = {name: json.loads((FEATURES / name).read_text()) for name in
            ("source_manifest.json", "extractor_contract.json", "checkpoint_provenance.json",
             "cache_integrity.json", "independent_verification.json")}
    manifest = docs["source_manifest.json"]
    canonical = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "manifest_digest"},
                                         sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(canonical == manifest["manifest_digest"] == MANIFEST_DIGEST, "source manifest digest failed")
    actual = {}
    for name, expected in CACHE_SHA.items():
        actual[name] = sha(FEATURES / name)
        require(actual[name] == expected, f"cache hash mismatch: {name}")
        print("verified", name, flush=True)
    cp = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
    require(sha(cp) == CHECKPOINT_SHA, "checkpoint hash mismatch")
    require(docs["independent_verification.json"]["overall_passed"], "rebuild verification failed")
    require(docs["extractor_contract.json"]["checkpoint_sha256"] == CHECKPOINT_SHA, "contract checkpoint")
    require(docs["checkpoint_provenance.json"]["canonical_sha256"] == CHECKPOINT_SHA, "provenance checkpoint")
    require(docs["cache_integrity.json"]["source_manifest_digest"] == MANIFEST_DIGEST, "integrity manifest")
    # Import only the inspected pure manifest function; never call prepare/extraction/evaluation utilities.
    from scripts.exps.hope_image_synchronous_memory import manifest as oracle_manifest
    updates, probes, available = oracle_manifest(0)
    entries = {r["relative_path"]: r for r in manifest["entries"]}
    for r in updates + probes:
        r["raw_png_sha256"] = entries[r["relative_path"]]["raw_png_sha256"]
    validate_roles(updates, probes)
    require(len(updates) == 350 and len(probes) == 60 and available == COUNTS, "role coverage mismatch")
    for category in CATEGORIES:
        group = [e for e in manifest["entries"] if e["category"] == category]
        objs = [torch.load(FEATURES / f"{v}/{category}_train.pt", mmap=True, weights_only=True) for v in ("clean", "masked_rgb16")]
        validate_alignment(*objs, group, category)
        del objs
    blocks = []
    for start in range(0, 350, 10):
        rows = updates[start:start + 10]
        require(len({r["phase"] for r in rows}) == 1, "block crosses task boundary")
        perm = derangement(10, 7102 + start // 10).tolist()
        blocks.append({"block": start // 10, "arrival_end": start + 10, "phase": rows[0]["phase"],
                       "category": rows[0]["class_name"], "rows": rows, "donor_permutation": perm,
                       "donor_paths": [rows[j]["relative_path"] for j in perm]})
    measured = feasibility()
    preserved_paths = [".gitignore", "pytest.ini", "logs_hope_gate3.pid",
                       "agents/reports/hope_cad_r0_architecture_decision.md", "agents/reports/hope_cad_r0_research_plan.md",
                       "scripts/hope_cad/provision_vit_b8_checkpoint.py", "scripts/hope_cad/rebuild_frozen_features_v1.py",
                       "models/hope_cad/self_modifying_titans.py", "models/hope_cad/continuum_memory.py",
                       "models/hope_cad/hope_block.py", "exps/ol04/hope_outer_learning_pilot_v1.py",
                       "exps/ol04e/hope_outer_learning_h50_v1.py", "scripts/exps/hope_image_synchronous_memory.py"]
    protocol = {
        "schema": "hope_cad_r1_conditional_pairing_v1", "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "representation": docs["extractor_contract.json"], "checkpoint_sha256": CHECKPOINT_SHA,
        "cache_sha256": actual, "manifest_canonical_digest": canonical,
        "input_document_file_sha256": {n: sha(FEATURES / n) for n in docs},
        "updates": updates, "probes": probes, "blocks": blocks, "boundaries": [100, 200, 300, 350],
        "centers": CENTERS, "order_seed": 0, "initialization_seed": 7111, "bootstrap_seed": 7123,
        "neural": {"architecture": "784-128-768 GELU MLP; zero final layer; no output normalization",
                   "optimizer": "Adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
                   "weight_decay": 0, "steps_per_block": 50, "total_steps_per_arm": 1750,
                   "replay": "none; current 10-image block reused for 50 full-batch steps then discarded",
                   "reset_at_boundary": False},
        "ridge": {"intercepts": "unpenalized position; shared full 768x768 slope",
                  "lambda": "max(1e-12,0.01*trace(within-position cumulative scatter)/768)",
                  "precision": "float64 sufficient statistics and Cholesky solve", "fit_boundaries_only": True},
        "metric": "sum squared channel errors, mean 16 positions, mean images, equal-category macro; no output L2 normalization",
        "bootstrap": {"repeats": 10000, "unit": "whole image, stratified within category, paired across arms",
                      "interval": "percentile 2.5,97.5; conditional on fixed training seed",
                      "statistic": "1-macro_neural/min(macro_mean,macro_ridge); reselect minimum each replicate"},
        "gate": {"relative_gain": 0.05, "bootstrap_lower_above": 0.0, "positive_categories": 2,
                 "pairing_relative_gain": 0.05, "pairing_bootstrap_lower_above": 0.0,
                 "near_zero_baseline": 1e-10, "integrity_required": True,
                 "inconclusive_is": "R1_FAIL after executed experiment; no rescue runs"},
        "justification": {"ridge": "1% of average within-position eigenvalue, scale-aware weak shrinkage; no probe tuning",
                          "neural": "one hidden layer, 128 units; 50 steps/block for bounded optimization exposure; no search",
                          "pairing_gate": "same project-defined 5% meaningful-effect threshold as primary comparison",
                          "near_zero": "unit-target RMS error <=1e-5 makes relative gain numerically uninformative",
                          "groups": "exact-content-disjoint images; object/acquisition independence unknown",
                          "delay": "all 10 images arrive before every arm receives this block; donors never cross blocks/tasks"},
        "environment": environment(), "runner_sha256": sha(__file__),
        "tests_sha256": sha(ROOT / "exps/r1/test_r1_conditional_pairing.py"),
        "preserved_sha256": {n: sha(ROOT / n) for n in preserved_paths},
        "budget_seconds": 3600, "output_budget_bytes": 1024**3,
    }
    atomic_json(OUT / "feasibility.json", measured)
    atomic_json(OUT / "input_verification.json", {"passed": True, "hashes": actual,
                "checkpoint_sha256": CHECKPOINT_SHA, "manifest_digest": canonical,
                "source_file_sha256": sha(FEATURES / "source_manifest.json"), "role_count": [350, 60],
                "exact_content_disjoint": True, "cache_metadata_alignment": True,
                "verification_seconds": time.perf_counter() - t})
    atomic_json(OUT / "protocol_resolved.json", protocol)
    (OUT / "protocol.sha256").write_text(sha(OUT / "protocol_resolved.json") + "\n")
    print("PROTOCOL_LOCKED", sha(OUT / "protocol_resolved.json"), measured, flush=True)


class Reader:
    def __init__(self, protocol):
        self.protocol = protocol; self.arrived = set(); self.objects = {}; self.access = []
    def arrive(self, category):
        if category in self.arrived: return
        entries = json.loads((FEATURES / "source_manifest.json").read_text())["entries"]
        objs = [torch.load(FEATURES / f"{v}/{category}_train.pt", mmap=True, weights_only=True)
                for v in ("clean", "masked_rgb16")]
        validate_alignment(*objs, [e for e in entries if e["category"] == category], category)
        self.objects[category] = objs; self.arrived.add(category)
    def pair(self, rows, role):
        permitted = {r["relative_path"]: r for r in self.protocol["updates" if role == "update" else "probes"]}
        clean, context = [], []
        for r in rows:
            category = r["class_name"]
            require(category in self.arrived, "future category read")
            require(r["relative_path"] in permitted and permitted[r["relative_path"]] == r, "unauthorized row")
            for obj, dest in zip(self.objects[category], (clean, context)):
                idx = r["cache_index"]
                require(obj["relative_paths"][idx] == r["relative_path"], "row index mismatch")
                full = obj["patches"][idx]
                require(digest(full) == obj["per_image_feature_sha256"][idx], "per-image feature hash mismatch")
                q = full[list(CENTERS)].clone()
                require(bool(torch.isfinite(q).all()) and bool((q.norm(dim=-1) > 1e-8).all()), "invalid feature")
                dest.append(F.normalize(q, dim=-1, eps=1e-8))
            self.access.append({"path": r["relative_path"], "role": role})
        return torch.stack(context), torch.stack(clean)


def summarize(per_image, protocol):
    final = [r for r in per_image if r["boundary"] == 350]
    a = np.array([[[r["losses"][arm] for r in final if r["category"] == cat]
                   for cat in CATEGORIES] for arm in ARMS])
    require(a.shape == (4, 3, 20), "final endpoint coverage differs")
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    indices = rng.integers(0, 20, size=(10000, 3, 20))
    boot = np.stack([np.stack([a[j, k][indices[:, k]].mean(-1) for k in range(3)], axis=1) for j in range(4)])
    means = a.mean(-1); macro = means.mean(-1); bm = boot.mean(-1)
    baseline = float(min(macro[0], macro[1])); near = protocol["gate"]["near_zero_baseline"]
    ci = lambda v: [float(x) for x in np.quantile(v, [0.025, 0.975])]
    valid = bool(baseline > near and np.all(np.minimum(means[0], means[1]) > near)
                 and np.all(np.minimum(bm[0], bm[1]) > near) and np.all(bm[2] > near))
    gain = 1 - macro[3] / baseline if valid else None
    gain_ci = ci(1 - bm[3] / np.minimum(bm[0], bm[1])) if valid else None
    pairing = 1 - macro[3] / macro[2] if valid else None
    pairing_ci = ci(1 - bm[3] / bm[2]) if valid else None
    positive = int(np.sum(means[3] < np.minimum(means[0], means[1])))
    gates = {"baseline_well_defined": valid, "gain_at_least_5pct": valid and gain >= .05,
             "gain_ci_above_zero": valid and gain_ci[0] > 0, "two_categories": positive >= 2,
             "pairing_at_least_5pct": valid and pairing >= .05,
             "pairing_ci_above_zero": valid and pairing_ci[0] > 0}
    return {"macro": {arm: {"mse": float(macro[j]), "ci95": ci(bm[j])} for j, arm in enumerate(ARMS)},
            "per_category": {cat: {arm: {"mse": float(means[j, k]), "ci95": ci(boot[j, :, k])}
                                   for j, arm in enumerate(ARMS)} for k, cat in enumerate(CATEGORIES)},
            "per_category_neural_gain": {cat: {"relative": float(1 - means[3, k] / min(means[0, k], means[1, k])),
                   "ci95": ci(1 - boot[3, :, k] / np.minimum(boot[0, :, k], boot[1, :, k]))}
                                         for k, cat in enumerate(CATEGORIES)} if valid else None,
            "strongest_macro_baseline": ARMS[int(np.argmin(macro[:2]))], "relative_gain": gain,
            "relative_gain_ci95": gain_ci, "pairing_relative_gain": pairing,
            "pairing_relative_gain_ci95": pairing_ci, "positive_categories": positive, "gates": gates}


def run():
    path = OUT / "protocol_resolved.json"
    p = json.loads(path.read_text())
    require(sha(path) == (OUT / "protocol.sha256").read_text().strip(), "protocol changed")
    require(p["runner_sha256"] == sha(__file__), "runner changed since lock")
    require(p["tests_sha256"] == sha(ROOT / "exps/r1/test_r1_conditional_pairing.py"), "tests changed since lock")
    require(p["environment"] == environment(), "environment changed since lock")
    require((OUT / "unit_tests.txt").is_file() and "passed" in (OUT / "unit_tests.txt").read_text(), "focused tests missing")
    with (OUT / "run_started.json").open("x") as f:
        json.dump({"protocol_sha256": sha(path), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    start = time.perf_counter()
    torch.manual_seed(p["initialization_seed"])
    models = {"mean": Mean(), "ridge": Ridge(), "shuffled": Neural()}
    models["neural"] = copy.deepcopy(models["shuffled"])
    opts = {a: optimizer(models[a]) for a in ("shuffled", "neural")}
    require(digest(model_state(models["shuffled"])) == digest(model_state(models["neural"])), "initialization mismatch")
    reader = Reader(p); trace = []; per_image = []; boundaries = []; timings = {a: {"update_seconds": 0., "inference_seconds": 0., "inference_images": 0} for a in ARMS}
    steps = {"shuffled": 0, "neural": 0}; no_write = []; state_chain = []
    initial = {a: digest(model_state(m)) for a, m in models.items()}
    for block in p["blocks"]:
        require(time.perf_counter() - start < p["budget_seconds"], "experiment time budget exceeded")
        reader.arrive(block["category"])
        c, z = reader.pair(block["rows"], "update")
        donor = torch.tensor(block["donor_permutation"])
        require(sorted(donor.tolist()) == list(range(10)) and bool((donor != torch.arange(10)).all()), "bad donor map")
        state_chain.append({"block": block["block"], "before": {a: digest(model_state(m)) for a, m in models.items()}})
        for a in ("mean", "ridge"):
            t = time.perf_counter()
            models[a].update(z) if a == "mean" else models[a].update(c, z)
            timings[a]["update_seconds"] += time.perf_counter() - t
        for a in ("shuffled", "neural"):
            target = z[donor] if a == "shuffled" else z
            if block["block"] == 0:
                h = digest((model_state(models[a]), opts[a].state_dict()))
                diagnostic = step(models[a], opts[a], c, target, write=False)
                require(h == digest((model_state(models[a]), opts[a].state_dict())), "no-write changed state")
                no_write.append({"arm": a, **diagnostic, "state_unchanged": True})
            t = time.perf_counter()
            for k in range(p["neural"]["steps_per_block"]):
                d = step(models[a], opts[a], c, target)
                require(d["parameter_delta_l2"] > 0 and d["output_delta_rms_l2"] > 0, "ineffective write")
                steps[a] += 1
                trace.append({"arm": a, "block": block["block"], "arrival_end": block["arrival_end"],
                              "step": steps[a], **d})
            timings[a]["update_seconds"] += time.perf_counter() - t
        state_chain[-1]["after"] = {a: digest(model_state(m)) for a, m in models.items()}
        if len(state_chain) > 1:
            require(state_chain[-2]["after"] == state_chain[-1]["before"], "state reset at block boundary")
        end = block["arrival_end"]
        if end in p["boundaries"]:
            t = time.perf_counter(); solve = models["ridge"].fit()
            timings["ridge"]["update_seconds"] += time.perf_counter() - t
            # Fitting the read weights is part of the ridge state transition at the boundary.
            state_chain[-1]["after"]["ridge"] = digest(models["ridge"].state())
            boundary = {"event": end, "phase": block["phase"], "ridge_solve": solve,
                        "optimizer_steps": dict(steps), "categories": {}}
            for a, opt in opts.items():
                require({int(v["step"]) for v in opt.state.values()} == {steps[a]}, "optimizer reset or unequal steps")
            for cat in CATEGORIES:
                if cat not in reader.arrived: continue
                rows = [r for r in p["probes"] if r["class_name"] == cat]
                pc, pz = reader.pair(rows, "probe")
                losses = {}; diag = {}
                for a, m in models.items():
                    before = digest((model_state(m), opts[a].state_dict() if a in opts else {}))
                    predictions = []
                    for x in pc.split(1):
                        t = time.perf_counter(); pred = inference(m, x)
                        timings[a]["inference_seconds"] += time.perf_counter() - t
                        timings[a]["inference_images"] += 1; predictions.append(pred)
                    prediction = torch.cat(predictions)
                    losses[a] = image_loss(prediction, pz).tolist()
                    repeat = torch.cat([inference(m, x) for x in pc.split(1)])
                    require(torch.equal(prediction, repeat), "evaluation not repeatable")
                    require(before == digest((model_state(m), opts[a].state_dict() if a in opts else {})), "evaluation mutated state")
                    variance = float(prediction.double().var(dim=0, unbiased=False).sum(-1).mean())
                    diag[a] = {"mse": float(np.mean(losses[a])), "prediction_variance": variance,
                               "prediction_mean_norm": float(prediction.double().norm(dim=-1).mean())}
                diag["target_variance"] = float(pz.double().var(dim=0, unbiased=False).sum(-1).mean())
                boundary["categories"][cat] = diag
                for i, row in enumerate(rows):
                    per_image.append({"boundary": end, "category": cat, "path": row["relative_path"],
                                      "raw_png_sha256": row["raw_png_sha256"], "losses": {a: losses[a][i] for a in ARMS}})
            boundaries.append(boundary)
            print("boundary_complete", end, "steps", steps, flush=True)
            atomic_json(OUT / "boundaries.json", boundaries)
            atomic_json(OUT / "per_image_results.json", per_image)
        del c, z
    require(steps == {"shuffled": 1750, "neural": 1750}, "unequal optimization exposure")
    require(all(sha(ROOT / name) == h for name, h in p["preserved_sha256"].items()), "protected source changed")
    results = summarize(per_image, p)
    results["gates"]["integrity"] = True
    results["status"] = "R1_PASS_PREDICTIVE_INFO" if all(results["gates"].values()) else "R1_FAIL"
    states = {a: model_state(m) for a, m in models.items()}
    optimizer_states = {a: o.state_dict() for a, o in opts.items()}
    resources = {a: {"model_state_tensor_bytes": tensor_bytes(states[a]),
                     "optimizer_tensor_bytes": tensor_bytes(optimizer_states.get(a, {})), **timings[a],
                     "inference_seconds_per_image": timings[a]["inference_seconds"] / timings[a]["inference_images"],
                     "update_seconds_per_arrival_image": timings[a]["update_seconds"] / 350} for a in ARMS}
    resources["shared"] = {"feature_cache_file_bytes": sum((FEATURES / n).stat().st_size for n in CACHE_SHA),
                           "checkpoint_file_bytes": (ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth").stat().st_size,
                           "block_context_target_tensor_bytes": 10 * 16 * 768 * 4 * 2,
                           "shuffled_target_copy_bytes": 10 * 16 * 768 * 4,
                           "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                           "experiment_runtime_seconds": time.perf_counter() - start,
                           "timing_scope": "neural update includes per-step causal diagnostics; CPU head only; inference batch one"}
    torch.save({"models": states, "optimizers": optimizer_states, "protocol_sha256": sha(path)}, OUT / "final_states.pt")
    atomic_json(OUT / "training_trace.json", trace)
    atomic_json(OUT / "state_chain.json", {"initial": initial, "blocks": state_chain, "no_write": no_write})
    atomic_json(OUT / "access_log.json", reader.access)
    atomic_json(OUT / "resources.json", resources)
    atomic_json(OUT / "summary.json", results)
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args(); configure()
    try:
        prepare() if args.mode == "prepare" else run()
    except Exception as exc:
        atomic_json(OUT / "execution_error.json", {"mode": args.mode, "type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__": main()
