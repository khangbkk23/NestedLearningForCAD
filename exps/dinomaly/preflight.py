# exps/dinomaly/preflight.py
"""Bounded normal-only resource preflight: encoder latency, decoder step time / peak VRAM per config.

Uses only the first 32 bottle P1 training identities (normal). No evaluation image or mask is read.
"""
from __future__ import annotations

import platform
import subprocess
import time

import torch

from exps.dinomaly import common as C
from exps.dinomaly import encoder as E
from exps.dinomaly import official as O


CAP_GIB = 2.6          # hard VRAM cap: the Windows driver otherwise spills to shared system memory


def configure():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)


def gpu_info():
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"nvidia_smi": q, "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "bf16": torch.cuda.is_bf16_supported(),
            "free_bytes": torch.cuda.mem_get_info()[0], "total_bytes": torch.cuda.mem_get_info()[1]}


def step_profile(dim, heads, x, ckpt, bf16, steps=20, warm=5, micro=16, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    dec = O.Decoder(dim, heads, checkpoint=ckpt).cuda().train()
    O.init_trainable(dec)
    opt = O.make_optimizer(dec)
    sch = O.make_scheduler(opt, 1000)
    times, losses = [], []
    try:
        for it in range(warm + steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad()
            loss = 0.
            for xb in x.split(micro):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    en, de = dec(xb[:, 0], xb[:, 1])
                en, de = [e.float() for e in en], [d.float() for d in de]
                part = O.global_cosine_hm_percent(en, de, p=0.9, factor=0.1) * len(xb) / len(x)
                part.backward()
                loss = loss + part.detach()
                del en, de, part
            torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=0.1)
            opt.step()
            sch.step()
            torch.cuda.synchronize()
            if it >= warm:
                times.append(time.perf_counter() - t0)
            elif time.perf_counter() - t0 > 30:
                return {"oom": True, "message": "step > 30 s: treated as memory spill"}
            losses.append(float(loss))
    except torch.OutOfMemoryError as e:
        return {"oom": True, "message": str(e).split("\n")[0][:200]}
    finally:
        del dec, opt
    return {"oom": False, "seconds_per_step": sum(times) / len(times),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "loss_first_last": [losses[0], losses[-1]], "finite": all(l == l for l in losses)}


def second():
    """PREFLIGHT_2: micro-batch / TF32 configurations under the same cap (after base OOM at micro=16)."""
    configure()
    ident = C.identities()
    path, digest = C.CKPT["base"]
    C.require(C.sha(path) == digest, "encoder checkpoint changed")
    rows = ident["train"]["bottle"][:16]
    enc = E.load(path, "base", "cuda")
    x16 = torch.cat([enc.groups(torch.stack([E.preprocess(C.source(r)) for r in rows[i:i + 8]]).cuda())
                     for i in (0, 8)]).contiguous()
    del enc
    torch.cuda.empty_cache()
    grid = {"base_bf16_ckpt_m8": (768, 12, True, True, 8, False), "base_bf16_ckpt_m4": (768, 12, True, True, 4, False),
            "base_tf32_ckpt_m4": (768, 12, True, False, 4, True), "base_fp32_ckpt_m4": (768, 12, True, False, 4, False),
            "base_bf16_m4": (768, 12, False, True, 4, False)}
    configs = {}
    for name, (dim, heads, ck, bf, mb, tf) in grid.items():
        configs[name] = step_profile(dim, heads, x16, ck, bf, micro=mb, tf32=tf)
        print(name, configs[name], flush=True)
    xs = torch.randn(16, 2, 789, 384, device="cuda")
    for name, (ck, bf, mb, tf) in {"small_bf16": (False, True, 16, False), "small_bf16_ckpt": (True, True, 16, False),
                                   "small_tf32_ckpt": (True, False, 16, True), "small_fp32_m8": (False, False, 8, False)}.items():
        configs[name + "_shape_only"] = step_profile(384, 6, xs, ck, bf, micro=mb, tf32=tf)
        print(name, configs[name + "_shape_only"], flush=True)
    C.write_json(C.OUT / "PREFLIGHT_2.json", {"cap_gib": CAP_GIB, "gpu": gpu_info(), "decoder_steps": configs,
                 "inputs": [r["relative_path"] for r in rows], "note": "normal training identities only"})


def main():
    configure()
    ident = C.identities()
    info = gpu_info()
    path, digest = C.CKPT["base"]
    C.require(C.sha(path) == digest, "encoder checkpoint changed")
    rows = ident["train"]["bottle"][:32]
    imgs = torch.stack([E.preprocess(C.source(r)) for r in rows])
    torch.cuda.reset_peak_memory_stats()
    enc = E.load(path, "base", "cuda")
    feats, lat = [], []
    for i in range(0, len(imgs), 8):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        feats.append(enc.groups(imgs[i:i + 8].cuda()))
        torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) / len(imgs[i:i + 8]))
    g = torch.cat(feats)
    cpu = E.load(path, "base", "cpu").groups(imgs[:2])
    encoder = {"seconds_per_image_batch8": lat[1:], "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "gpu_vs_cpu_max_abs": float((g[:2].cpu() - cpu).abs().max()), "feature_shape": list(g.shape[1:]),
               "parameters": sum(p.numel() for p in enc.parameters())}
    del enc
    torch.cuda.empty_cache()
    x16 = g[:16].contiguous()
    configs = {}
    for name, (ck, bf) in {"fp32": (False, False), "fp32_ckpt": (True, False),
                           "bf16": (False, True), "bf16_ckpt": (True, True)}.items():
        configs[f"base_{name}"] = step_profile(768, 12, x16, ck, bf)
        print(name, configs[f"base_{name}"], flush=True)
    xs = torch.randn(16, 2, 789, 384, device="cuda")                    # shape-only proxy for ViT-S
    for name, (ck, bf) in {"fp32": (False, False), "fp32_ckpt": (True, False)}.items():
        configs[f"small_{name}_shape_only"] = step_profile(384, 6, xs, ck, bf)
    dec = O.Decoder(768, 12)
    trainable = sum(p.numel() for p in dec.parameters())
    out = {"cap_gib": CAP_GIB, "gpu": info, "encoder": encoder, "decoder_steps": configs, "decoder_trainable_parameters": {
        "base": trainable, "small": sum(p.numel() for p in O.Decoder(384, 6).parameters())},
        "inputs": [r["relative_path"] for r in rows], "note": "normal training identities only; no evaluation access"}
    C.write_json(C.OUT / "PREFLIGHT.json", out)
    print(out["encoder"], out["decoder_trainable_parameters"], flush=True)


if __name__ == "__main__":
    import sys
    second() if sys.argv[1:] == ["second"] else main()
