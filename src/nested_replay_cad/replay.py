"""Learned replay with causal reservoir/FIFO queries and optional CTX normalization."""

import gc
import math


def trim_workspace():
    """Return unused Linux allocator pages between large conditional phases."""
    import ctypes

    gc.collect()
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)


class NestedReplay:
    def __init__(self, seed, objective, device="cuda:0", normalize_context=False):
        import torch

        from nested_replay_cad.core.contracts import PhiState
        from nested_replay_cad.core.learning.adam import adam_initial
        from nested_replay_cad.core.policy.layout import phi_shapes
        from nested_replay_cad.core.policy.policy import ReplayPolicy

        if objective not in ("tail", "mean"):
            raise ValueError("Foundation replay objective")
        generator = torch.Generator().manual_seed(seed + 2000000)
        parameters = {
            name: torch.zeros(shape, requires_grad=True)
            for name, shape in phi_shapes().items()
        }
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(
                parameters["hidden.weight"], generator=generator
            )
        self.normalize_context = bool(normalize_context)
        self.device = torch.device(device)
        self.phi = PhiState(
            parameters
        )  # Keep qualified 97-parameter policy/Adam on CPU.
        self.adam = adam_initial(self.phi)
        self.policy = ReplayPolicy()
        self.objective = objective
        self.eligible_commits = 0
        self.slow_steps = 0

    def step(
        self,
        decoder,
        optimizer,
        current_id,
        current,
        replay_ids,
        recent_ids,
        history_ids,
        store,
        metas,
        arrival,
        eligible,
        hard_mining_p,
        release_encoder,
    ):
        import torch
        import torch.nn.functional as functional

        from nested_replay_cad.core.contracts import PhiState
        from nested_replay_cad.core.learning.adam import adam_propose
        from nested_replay_cad.fast_update import conditional_vjp, functional_step
        from nested_replay_cad.model import decoder as official

        roles = [current_id, *replay_ids, *recent_ids, *history_ids]
        if len(set(roles)) != len(roles) or any(
            metas[identity]["arrival"] >= arrival for identity in roles[1:]
        ):
            raise ValueError("Foundation replay role disjointness/causality")
        parameters = dict(decoder.named_parameters())
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        counters = dict(
            support_forwards=0,
            controller_context_forwards=0,
            slow_query_forwards=0,
            slow_replay_forwards=0,
        )
        feature = current
        target, prediction = decoder(feature[None, 0], feature[None, 1])
        loss = official.global_cosine_hm_percent(
            target, prediction, p=hard_mining_p, factor=0.1
        )
        losses = [float(loss.detach())]
        (loss * (0.5 if replay_ids else 1.0)).backward()
        counters["support_forwards"] += 1
        del target, prediction, loss
        contexts, draws = [], {}
        with torch.no_grad():
            current_mean = current[:, 5:].mean((0, 1))
            for identity in replay_ids:
                draws[identity] = torch.cuda.get_rng_state(self.device).clone()
                feature = store[identity]
                target, prediction = decoder(feature[None, 0], feature[None, 1])
                patch = (
                    torch.stack(
                        [
                            1 - functional.cosine_similarity(a, b, dim=1)
                            for a, b in zip(target, prediction)
                        ]
                    )
                    .mean(0)
                    .flatten()
                )
                age = arrival - metas[identity]["arrival"]
                old_mean = feature[:, 5:].mean((0, 1))
                contexts.append(
                    torch.stack(
                        (
                            patch.mean() / 2,
                            torch.topk(patch, 79).values.mean() / 2,
                            patch.new_tensor(age / (age + 32)),
                            (
                                1
                                + functional.cosine_similarity(
                                    old_mean[None], current_mean[None], dim=1
                                )[0]
                            )
                            / 2,
                        )
                    )
                )
                counters["controller_context_forwards"] += 1
                del target, prediction, patch
        if replay_ids:
            contexts = torch.stack(contexts).detach().cpu().contiguous()
            if self.normalize_context:
                from nested_replay_cad.context import normalize_contexts

                contexts = normalize_contexts(contexts)
            weights = (
                self.policy.weights(self.phi, contexts)
                if eligible
                else torch.full((len(replay_ids),), 1 / len(replay_ids))
            )
        else:
            contexts = torch.empty((0, 4))
            weights = torch.empty(0)
        for index, identity in enumerate(replay_ids):
            torch.cuda.set_rng_state(draws[identity], self.device)
            feature = store[identity]
            target, prediction = decoder(feature[None, 0], feature[None, 1])
            loss = official.global_cosine_hm_percent(
                target, prediction, p=hard_mining_p, factor=0.1
            )
            losses.append(float(loss.detach()))
            (0.5 * weights[index].detach().to(self.device) * loss).backward()
            counters["support_forwards"] += 1
            del target, prediction, loss
        if not all(
            torch.isfinite(parameter.grad).all() for parameter in parameters.values()
        ):
            raise ValueError("Nonfinite foundation support gradient")
        due = eligible and (self.eligible_commits + 1) % 16 == 0
        slow = None
        next_phi, next_adam = self.phi, self.adam
        probes = None
        if due:
            release_encoder()
            gradient = {name: value.grad.detach() for name, value in parameters.items()}
            optimizer.zero_grad(set_to_none=True)
            parent = {
                name: value.detach().clone() for name, value in parameters.items()
            }
            moments = {
                name: {
                    key: optimizer.state[value][key]
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
                }
                for name, value in parameters.items()
            }
            step = arrival - 1
            if any(
                optimizer.state[value]["step"] != step for value in parameters.values()
            ):
                raise ValueError("Fast optimizer pre-event clock")
            with torch.no_grad():
                functional_step(parent, gradient, moments, step, destination=parameters)
            probes = {
                name: (
                    torch.linspace(
                        0,
                        value.numel() - 1,
                        min(32, value.numel()),
                        device=value.device,
                    ).long(),
                    None,
                )
                for name, value in parameters.items()
            }
            probes = {
                name: (indices, parameters[name].detach().flatten()[indices].clone())
                for name, (indices, _) in probes.items()
            }
            decoder.eval()
            decoder.training = True  # checkpoint blocks, child dropout remains eval
            query_rng = torch.cuda.get_rng_state(self.device).clone()
            query_values = []
            for index, identity in enumerate((*recent_ids, *history_ids)):
                feature = store[identity]
                target, prediction = decoder(feature[None, 0], feature[None, 1])
                patch = (
                    torch.stack(
                        [
                            1 - functional.cosine_similarity(a, b, dim=1)
                            for a, b in zip(target, prediction)
                        ]
                    )
                    .mean(0)
                    .flatten()
                )
                value = (
                    torch.sort(patch, descending=True, stable=True).values[:79].mean()
                    if self.objective == "tail" and index >= len(recent_ids)
                    else patch.mean()
                )
                query_values.append(float(value.detach()))
                (0.25 * value).backward()
                counters["slow_query_forwards"] += 1
                del target, prediction, patch, value
            if not torch.equal(query_rng, torch.cuda.get_rng_state(self.device)):
                raise ValueError("Slow query RNG mutation")
            query = {name: value.grad.detach() for name, value in parameters.items()}
            optimizer.zero_grad(set_to_none=True)
            adjoint, branches = conditional_vjp(gradient, moments, step, query)
            del query
            with torch.no_grad():
                for name, value in parameters.items():
                    value.copy_(parent[name])
            trim_workspace()
            decoder.train()
            alignments = []
            for identity in replay_ids:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.set_rng_state(draws[identity], self.device)
                feature = store[identity]
                target, prediction = decoder(feature[None, 0], feature[None, 1])
                loss = official.global_cosine_hm_percent(
                    target, prediction, p=hard_mining_p, factor=0.1
                )
                loss.backward()
                alignments.append(
                    0.5
                    * math.fsum(
                        float((adjoint[name] * value.grad.detach().double()).sum())
                        for name, value in parameters.items()
                    )
                )
                counters["slow_replay_forwards"] += 1
                del target, prediction, loss
            optimizer.zero_grad(set_to_none=True)
            del adjoint, parent, moments
            trim_workspace()
            kl = (weights * torch.log(len(replay_ids) * weights)).sum()
            upstream = torch.tensor(alignments, dtype=torch.float64)
            objective = (weights.double() * upstream).sum() + 0.001 * kl
            phi_values = tuple(self.phi.parameters.values())
            derivatives = torch.autograd.grad(objective, phi_values)
            policy_gradient = {
                name: value.detach().clone()
                for name, value in zip(self.phi.parameters, derivatives)
            }
            next_phi, next_adam = adam_propose(self.phi, self.adam, policy_gradient)
            slow = dict(
                objective=self.objective,
                query_values=query_values,
                query_objective=math.fsum(query_values) / 4,
                alignments=alignments,
                phi_gradient_norm=math.sqrt(
                    math.fsum(
                        float(value.double().square().sum())
                        for value in policy_gradient.values()
                    )
                ),
                gradient={
                    name: value.flatten().tolist()
                    for name, value in policy_gradient.items()
                },
                global_clip_active=branches["global_clip_active"],
                RMS_scaled_tensors=sum(
                    row["rms_scaled"] for row in branches["tensors"].values()
                ),
            )
            for name, value in parameters.items():
                value.grad = gradient[name]
        norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.1)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite foundation gradient norm")
        optimizer.step()
        if not all(torch.isfinite(value).all() for value in parameters.values()):
            raise ValueError("Nonfinite foundation fast update")
        if probes is not None:
            maximum = 0.0
            for name, (indices, expected) in probes.items():
                actual = parameters[name].detach().flatten()[indices]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                maximum = max(maximum, float((actual - expected).abs().max()))
            slow["sampled_fast_update_maxabs"] = maximum
            self.phi = PhiState(
                {
                    name: value.detach().clone().requires_grad_()
                    for name, value in next_phi.parameters.items()
                }
            )
            self.adam = next_adam
            self.slow_steps += 1
            slow["step"] = self.slow_steps
            slow["next_weights_same_context"] = (
                self.policy.weights(self.phi, contexts).detach().tolist()
            )
            optimizer.zero_grad(set_to_none=True)
            del gradient
            trim_workspace()
        if eligible:
            self.eligible_commits += 1
        return dict(
            losses=losses,
            gradient_norm=float(norm),
            weights=weights.detach().tolist(),
            slow=slow,
            counters=counters,
        )

    def preserve(self, tensors):
        for name, value in self.phi.parameters.items():
            tensors["policy." + name] = value.detach().contiguous()
            tensors["slow.m." + name] = self.adam.m[name].contiguous()
            tensors["slow.v." + name] = self.adam.v[name].contiguous()
        return dict(
            objective=self.objective,
            context_mode="CTX-RN-01" if self.normalize_context else "raw",
            numerical_mode="FND-VJP-FWD32-01",
            slow_steps=self.slow_steps,
            eligible_commits=self.eligible_commits,
            slow_optimizer_step=self.adam.step,
        )
