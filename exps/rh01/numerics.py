# exps/rh01/numerics.py
"""RH-01 numerics-repair preflight (user option b). No development data, no labels.

Hypothesis: under BF16 autocast, phi(q) = elu(q) + 1 rounds to exactly 0 once q < ~-5.5 (BF16 spacing near 1 is
2^-7; FP32 needs q < ~-16.6). If a token's phi(q) vanishes in every channel of a head, the normalizer
phi(q).sum_s phi(k) is 0, z = 1/0 = inf and the output is 0 * inf = NaN.

Fix candidate (applied identically to every arm): the LinearAttention2 core (elu features, kv, normalizer, output
einsum) runs in FP32 with autocast disabled, exactly the official FP32 arithmetic; qkv/proj stay under autocast.
In FP32 evaluation the patched forward is the original forward, op for op.

  python -m exps.rh01.numerics probe    # zero-feature / zero-denominator rates on existing states (BF16 vs FP32)
  python -m exps.rh01.numerics parity   # FP32 identity, BF16 error vs FP32 reference, step time
  python -m exps.rh01.numerics stress   # 1,000-step continuation from J15 2711 at the failure-region LR
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.rh01 import common as C
from exps.rh01 import run as RR

OUT = C.OUT / "numerics"
ORIGINAL = O.LinearAttention2.forward


def forward_fp32core(self, x, attn_mask=None):
    B, N, Cc = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, Cc // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
    with torch.autocast("cuda", enabled=False):
        q = F.elu(q) + 1.
        k = F.elu(k) + 1.
        kv = torch.einsum('...sd,...se->...de', k, v)
        z = 1.0 / torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))
        x = torch.einsum('...de,...sd,...s->...se', kv, q, z)
    x = x.transpose(1, 2).reshape(B, N, Cc)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, kv


def use(fp32core):
    O.LinearAttention2.forward = forward_fp32core if fp32core else ORIGINAL


STATS = []


def forward_probe(self, x, attn_mask=None):
    """Original arithmetic, plus counts of vanished phi(q) rows and zero / non-finite normalizers."""
    B, N, Cc = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, Cc // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = F.elu(q) + 1.
    k = F.elu(k) + 1.
    kv = torch.einsum('...sd,...se->...de', k, v)
    den = torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))
    STATS.append({"module": getattr(self, "_name", None), "rows": den.numel(), "phi_q_zero_rows": int((q == 0).all(-1).sum()),
                  "den_zero": int((den == 0).sum()), "den_min": float(den.float().min()),
                  "den_nonfinite": int((~torch.isfinite(den)).sum()), "kv_absmax": float(kv.abs().amax()),
                  "kv_nonfinite": int((~torch.isfinite(kv)).sum()), "q_raw_min": float(qkv[0].amin()),
                  "q_raw_max": float(qkv[0].amax()), "k_raw_max": float(qkv[1].amax()),
                  "v_absmax": float(v.abs().amax()), "dtype": str(den.dtype), "nonfinite_out": 0})
    z = 1.0 / den
    x = torch.einsum('...de,...sd,...s->...se', kv, q, z)
    STATS[-1]["nonfinite_out"] = int((~torch.isfinite(x)).sum())
    x = x.transpose(1, 2).reshape(B, N, Cc)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, kv


def state(arm, seed, stage=None):
    rec = D.read(C.OUT / f"runs/{arm}_seed{seed}/TRAIN.json")
    st = rec["stages"][-1] if stage is None else next(s for s in rec["stages"] if s["index"] == stage)
    path = D.ROOT / st["checkpoint"]
    require(sha(path) == st["checkpoint_sha256"], f"checkpoint changed: {path}")
    return torch.load(path, weights_only=True)["state"]


def decoder(sd, train=True, ckpt=False):
    """ckpt=True for training steps (the recipe's exact gradient checkpointing, D4)."""
    dec = O.Decoder(768, 12, checkpoint=ckpt).cuda()
    dec.load_state_dict(sd)
    return dec.train() if train else dec.eval()


@torch.no_grad()
def probe():
    DR.configure()
    RR.verify_lock()
    target = OUT / "PROBE_MECHANISM.json"
    require(not target.exists(), "one-shot")
    pool = C.Pool([C.load_features(c) for c in C.STREAM])
    idx = torch.randperm(len(pool), generator=torch.Generator().manual_seed(7))[:800].tolist()
    O.LinearAttention2.forward = forward_probe
    res = {}
    for name, sd in (("J15_2711_final", state("J15", 2711)), ("R15_2711_final", state("R15", 2711)),
                     ("R15_2712_final", state("R15", 2712)), ("R15_2711_stage8", state("R15", 2711, 8))):
        dec = decoder(sd)
        for mode in ("bf16", "fp32"):
            STATS.clear()
            torch.manual_seed(0)
            for i in range(0, len(idx), 8):
                x = pool[idx[i:i + 8]].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(mode == "bf16")):
                    dec(x[:, 0], x[:, 1])
            res[f"{name}/{mode}"] = {"rows": sum(s["rows"] for s in STATS),
                                     "phi_q_zero_rows": sum(s["phi_q_zero_rows"] for s in STATS),
                                     "den_zero": sum(s["den_zero"] for s in STATS),
                                     "nonfinite_out": sum(s["nonfinite_out"] for s in STATS),
                                     "den_min": min(s["den_min"] for s in STATS),
                                     "q_raw_min": min(s["q_raw_min"] for s in STATS),
                                     "per_block_phi_q_zero": [sum(s["phi_q_zero_rows"] for s in STATS[b::8])
                                                              for b in range(8)]}
            print(name, mode, {k: v for k, v in res[f"{name}/{mode}"].items()}, flush=True)
        del dec
    use(False)
    D.write_json(target, {"images": 800, "train_mode_dropout": True, "results": res})


def parity():
    DR.configure()
    RR.verify_lock()
    target = OUT / "PARITY.json"
    require(not target.exists(), "one-shot")
    pool = C.Pool([C.load_features(c) for c in C.STREAM])
    idx = torch.randperm(len(pool), generator=torch.Generator().manual_seed(8))[:64].tolist()
    res = {}
    for name, sd in (("J15_2711_final", state("J15", 2711)), ("R15_2712_final", state("R15", 2712))):
        dec = decoder(sd, train=False)
        outs = {}
        with torch.no_grad():
            for tag, fp32core, bf16 in (("fp32_original", False, False), ("fp32_patched", True, False),
                                        ("bf16_original", False, True), ("bf16_patched", True, True)):
                use(fp32core)
                o = []
                for i in range(0, len(idx), 8):
                    x = pool[idx[i:i + 8]].cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                        en, de = dec(x[:, 0], x[:, 1])
                    o.append(torch.cat([d.float().flatten(1) for d in de], 1).cpu())
                outs[tag] = torch.cat(o)
        ref = outs["fp32_original"]
        rel = lambda a: float(((a - ref).norm(dim=1) / ref.norm(dim=1)).max())
        res[name] = {"fp32_patched_equals_original": bool(torch.equal(outs["fp32_patched"], ref)),
                     "fp32_patched_max_abs_diff": float((outs["fp32_patched"] - ref).abs().max()),
                     "bf16_original_max_rel_err": rel(outs["bf16_original"]),
                     "bf16_patched_max_rel_err": rel(outs["bf16_patched"]),
                     "nonfinite": {k: int((~torch.isfinite(v)).sum()) for k, v in outs.items()}}
        print(name, res[name], flush=True)
        del dec, outs
        torch.cuda.empty_cache()
    use(False)
    # step time: 40 training steps each (throwaway decoder from J15 2711 final)
    times = {}
    for tag, fp32core in (("original", False), ("patched", True)):
        use(fp32core)
        dec = decoder(state("J15", 2711), ckpt=True)
        gen = torch.Generator().manual_seed(1)
        RR.train_stage_guarded(dec, pool, 10, gen)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        RR.train_stage_guarded(dec, pool, 40, gen)
        torch.cuda.synchronize()
        times[tag] = (time.perf_counter() - t0) / 40
        del dec
    use(False)
    res["s_per_step"] = times
    res["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    print("s/step", times, flush=True)
    D.write_json(target, res)


def stress(steps=1000):
    """Continue from the saved J15 seed-2712 state at iteration 4,758 (the first non-finite step; fragile regime)
    for `steps` iterations, original vs FP32-core, identical seeds and batches. The optimizer state was not saved,
    so a fresh StableAdamW is warmed up for 100 iterations to the scheduled LR at 4,758 and then held constant.
    Non-finite steps are counted (skipped by the guard), never stopped on."""
    DR.configure()
    RR.verify_lock()
    target = OUT / "STRESS.json"
    require(not target.exists(), "one-shot")
    pool = C.Pool([C.load_features(c) for c in C.STREAM])
    snap = torch.load(SCRATCH / "j15_2712_iter4758.pt", weights_only=True)
    total, warm, i = RR.JOINT_ITERS, 100, 4758
    lr = 2e-4 + 0.5 * (2e-3 - 2e-4) * (1 + math.cos(math.pi * (i - warm) / (total - warm)))
    res = {"start": "J15 seed 2712 replayed state at iteration 4758", "lr": lr, "steps": steps}
    for tag, fp32core in (("original", False), ("patched", True)):
        use(fp32core)
        torch.manual_seed(11)
        dec = decoder(snap["state"], ckpt=True)
        gen = torch.Generator().manual_seed(12)
        orig_sched = O.make_scheduler
        O.make_scheduler = lambda opt, iters: torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: (lr / 2e-3) * min(1.0, (k + 1) / 100))
        RR.MAX_SKIPS, RR.MAX_CONSECUTIVE = 10 ** 9, 10 ** 9
        t0 = time.perf_counter()
        try:
            losses, skips = RR.train_stage_guarded(dec, pool, steps, gen)
        finally:
            O.make_scheduler = orig_sched
            RR.MAX_SKIPS, RR.MAX_CONSECUTIVE = 10, 3
        res[tag] = {"nonfinite_steps": len(skips), "first": skips[:10], "seconds": time.perf_counter() - t0,
                    "loss_last100": float(np.nanmean(losses[-100:])),
                    "params_finite": all(bool(torch.isfinite(q).all()) for q in dec.parameters())}
        print(tag, {k: v for k, v in res[tag].items() if k != "first"}, flush=True)
        del dec
        torch.cuda.empty_cache()
    use(False)
    D.write_json(target, res)

# --- localization: deterministic replay of a J15 run up to its first non-finite step ----------------------

SCRATCH = None


def instrument(dec, x, p, rng, mode):
    """One training step's forward/backward on (state, batch, dropout RNG) with module-level finiteness hooks."""
    first, stats = [], {}

    def hook(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                fin = bool(torch.isfinite(t).all())
                stats.setdefault(name, {"max_abs": 0.0, "finite": True})
                amax = float(t.detach().abs().amax())
                stats[name]["max_abs"] = max(stats[name]["max_abs"], amax if math.isfinite(amax) else 3e38)
                stats[name]["finite"] &= fin
                if not fin and not first:
                    first.append(name)
        return fn

    use(mode == "fp32core")
    if mode == "probe":
        O.LinearAttention2.forward = forward_probe
    STATS.clear()
    for n, m in dec.named_modules():
        if isinstance(m, O.LinearAttention2):
            m._name = n
    handles = [m.register_forward_hook(hook(n)) for n, m in dec.named_modules() if n]
    dec.zero_grad(set_to_none=True)
    torch.cuda.set_rng_state(rng)
    total = 0.
    for xb in x.split(DR.MICRO):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(mode != "fp32full")):
            en, de = dec(xb[:, 0], xb[:, 1])
        loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de], p=p,
                                          factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
        loss.backward()
        total += float(loss)
    for h in handles:
        h.remove()
    bad = [n for n, q in dec.named_parameters() if q.grad is not None and not bool(torch.isfinite(q.grad).all())]
    norm = float(torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=1e30))
    use(False)
    att = [s for s in STATS] if mode == "probe" else []
    return {"loss": total, "grad_norm": norm, "first_nonfinite_module": first[0] if first else None,
            "params_with_nonfinite_grad": bad[:20], "n_params_nonfinite_grad": len(bad),
            "attention_probe": [{k: v for k, v in s.items()} for s in att],
            "largest_activations": sorted(((v["max_abs"], k) for k, v in stats.items()), reverse=True)[:8]}


def localize(seed=2712, target=4758):
    DR.configure()
    RR.verify_lock()
    out = OUT / f"LOCALIZE_J15_{seed}_{target}.json"
    require(not out.exists(), "one-shot")
    pool = C.Pool([C.load_features(c) for c in C.STREAM])
    torch.manual_seed(RR.stage_seeds("J15", seed, 1)[0])                   # exactly as run.train (J15)
    dec = DR.new_decoder(seed)
    gen = torch.Generator().manual_seed(RR.stage_seeds("J15", seed, 1)[1])
    iters = RR.JOINT_ITERS
    opt, ramp = O.make_optimizer(dec), round(0.1 * iters)
    sch = O.make_scheduler(opt, iters)
    cur = DR.Batches(len(pool), O.RECIPE["batch_size"], gen)
    dec.train()
    t0 = time.perf_counter()
    for i in range(target + 1):                                            # mirrors train_stage_guarded exactly
        p = min(O.RECIPE["hm_p_final"] * i / ramp, O.RECIPE["hm_p_final"])
        x = pool[cur.next()].cuda(non_blocking=True)
        if i == target:
            break
        opt.zero_grad(set_to_none=True)
        total = 0.
        for xb in x.split(DR.MICRO):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                en, de = dec(xb[:, 0], xb[:, 1])
            loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de],
                                              p=p, factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
            loss.backward()
            total += float(loss)
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        require(np.isfinite(total) and bool(torch.isfinite(norm)), f"non-finite before target at {i}")
        opt.step()
        sch.step()
    rng = torch.cuda.get_rng_state()
    res = {"seed": seed, "target": target, "replay_seconds": time.perf_counter() - t0, "p": p,
           "lr": opt.param_groups[0]["lr"]}
    snap = {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
    res["weights_finite"] = all(bool(torch.isfinite(v).all()) for v in snap.values())
    torch.save({"state": snap, "batch": x.cpu(), "cuda_rng": rng, "p": p}, SCRATCH / f"j15_{seed}_iter{target}.pt")
    del opt, sch
    dec.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    print("replayed", {k: v for k, v in res.items()}, flush=True)
    analyse(dec, snap, x, p, rng, res, out)


def analyse(dec, snap, x, p, rng, res, out):
    for mode in ("probe", "fp32core", "fp32full", "bf16_original"):
        dec.load_state_dict(snap)
        try:
            res[mode] = instrument(dec, x, p, rng, mode)
        except torch.OutOfMemoryError as e:
            res[mode] = {"error": f"OOM: {e}"[:300]}
        torch.cuda.empty_cache()
        show = {k: v for k, v in res[mode].items() if k not in ("attention_probe", "params_with_nonfinite_grad")}
        print(mode, show, flush=True)
        if mode == "probe" and "attention_probe" in res[mode]:
            for b, s in enumerate(res[mode]["attention_probe"]):
                print("  probe call", b, s, flush=True)
    D.write_json(out, res)


def reanalyse(path, seed=2712, target=4758):
    """Re-run the step analysis from a saved snapshot (no replay)."""
    DR.configure()
    RR.verify_lock()
    snap = torch.load(path, weights_only=True)
    dec = O.Decoder(768, 12, checkpoint=True).cuda().train()
    analyse(dec, snap["state"], snap["batch"].cuda(), snap["p"], snap["cuda_rng"],
            {"seed": seed, "target": target, "from_snapshot": str(path)}, OUT / f"LOCALIZE_J15_{seed}_{target}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("probe", "parity", "stress", "localize"))
    ap.add_argument("--scratch")
    a = ap.parse_args()
    global SCRATCH
    from pathlib import Path
    SCRATCH = Path(a.scratch) if a.scratch else OUT
    {"probe": probe, "parity": parity, "stress": stress, "localize": localize}[a.stage]()


if __name__ == "__main__":
    main()
