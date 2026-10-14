"""Read-only contraction diagnostics. Norms are FP64; recurrence remains FP32."""
from dataclasses import replace
from copy import deepcopy
import math
import torch
from torch.nn import functional as F
from models.hope_cad_v2.scan_geometry import scan_routes
from models.hope_cad_v2.visual_memory import stability_matched_step


def stats(tensor):
    value = tensor.detach().double()
    return dict(min=value.min().item(), mean=value.mean().item(), max=value.max().item(),
                variance=value.var(unbiased=False).item(),
                rms=value.square().mean().sqrt().item(),
                zero_fraction=(value == 0).double().mean().item(),
                subnormal_fraction=((value.abs() > 0) & (value.abs() < torch.finfo(tensor.dtype).tiny)).double().mean().item())


@torch.no_grad()
def visual_diagnostics(model, features):
    visual, c = model.visual, model.visual.config
    x = visual.projection(features)
    spatial = visual.in_projection(visual.input_norm(x)).transpose(1, 2).reshape(features.shape[0], c.memory_dim, c.height, c.width)
    z = visual.depthwise(spatial).flatten(2).transpose(1, 2)
    memory = visual.memory
    inspected = memory.inspect(z)
    routes = scan_routes(c.height, c.width, device=features.device)
    route_by_name = {r.name: (i, r) for i, r in enumerate(routes)}
    groups = []
    for names, chunks in inspected.scan_groups:
        indices = [route_by_name[n][0] for n in names]
        states = [m[indices].unsqueeze(0).expand(features.shape[0], -1, -1, -1, -1) for m in memory.initial_state().tensors()]
        ordered = torch.stack([route_by_name[n][1].scan(z) for n in names], 1)
        ordered = ordered.reshape(features.shape[0], len(names), c.height*c.width, memory.heads, memory.head_dim).transpose(2, 3)
        records = []
        for chunk in chunks:
            tokens = ordered[..., chunk.start:chunk.end, :]
            kraw = tokens @ states[1].transpose(-1, -2)
            key = kraw / (kraw.square().sum(-1, keepdim=True)+1e-6).sqrt()
            value = tokens @ states[2].transpose(-1, -2)
            eta = .025*F.softplus((tokens @ states[3].transpose(-1, -2)).squeeze(-1))
            alpha0 = 0.9 if memory.row_retention is None else memory.row_retention ** (1.0/routes[indices[0]].chunk_size)
            alpha_bias = math.log(9.) if memory.row_retention is None else math.log(alpha0/(1-alpha0))
            alpha = torch.sigmoid((tokens @ states[4].transpose(-1, -2)).squeeze(-1)+alpha_bias)
            guard = stability_matched_step(key, key-value, eta, alpha,higher_order_safe=True)
            pre = [s.double().norm(dim=(-1,-2)).cpu().tolist() for s in states]
            states = [s @ chunk.gains[..., -1, :, :] for s in states]
            # Per image/direction/head diagnostics, not just a global average.
            readouts = []
            for local, name in enumerate(names):
                index, route = route_by_name[name]
                output = route.scan(inspected.directional_outputs[:, index])[:, chunk.start:chunk.end]
                heads = output.reshape(features.shape[0], chunk.end-chunk.start, memory.heads, memory.head_dim)
                readouts.append(heads.double().square().mean((1,3)).sqrt().cpu().tolist())
            records.append(dict(start=chunk.start, end=chunk.end,
                boundary_map_norms=pre, final_map_norms=[s.double().norm(dim=(-1,-2)).cpu().tolist() for s in states],
                gain_spectral_norms=torch.linalg.matrix_norm(chunk.gains.double(), ord=2).amax(-1).cpu().tolist(),
                output_rms_by_direction_head=readouts, alpha=stats(alpha), raw_eta=stats(eta),
                executed_eta=stats(guard.executed),
                spectral_cap_fraction=(guard.executed == guard.spectral_limit).double().mean().item()))
        groups.append(dict(directions=names, chunks=records))
    branch = visual.out_projection(inspected.output)
    static = deepcopy(model)
    static.dynamics = replace(static.dynamics, visual_updates=False)
    active_output, static_output = model.encode(features), static.encode(features)
    per_position = inspected.directional_outputs.double().square().mean(-1).sqrt().cpu().tolist()
    spatial_rms=inspected.directional_outputs.double().square().mean(-1).sqrt().reshape(features.shape[0],4,c.height,c.width)
    border_mask=torch.zeros(c.height,c.width,dtype=torch.bool,device=features.device)
    band=min(2,max(1,min(c.height,c.width)//4))
    border_mask[:band]=True; border_mask[-band:]=True
    border_mask[:,:band]=True; border_mask[:,-band:]=True
    cy,cx=c.height//2,c.width//2
    center=spatial_rms[:,:,max(0,cy-2):min(c.height,cy+2),max(0,cx-2):min(c.width,cx+2)].square().mean((-1,-2)).sqrt()
    border=spatial_rms[:,:,border_mask].square().mean(-1).sqrt()
    return dict(scope='visual component, synthetic or permitted normal-prefix features; no CAD utility claim',
                model_visual_updates=model.dynamics.visual_updates,
                operator_trace_scope='dynamic operator trace; static controls execute their fixed-map readout',
                groups=groups, initial_maps=[stats(v) for v in memory.initial_state().tensors()],
                final_maps=[stats(v) for v in inspected.final_state.tensors()],
                restored_spatial_rms=per_position, output=stats(inspected.output),
                row_retention=memory.row_retention,
                center_rms_by_direction=center.cpu().tolist(),border_rms_by_direction=border.cpu().tolist(),
                center_to_border_ratio_by_direction=(center/border.clamp_min(1e-30)).cpu().tolist(),
                branch_to_input_rms=stats(branch)['rms']/max(stats(x)['rms'], 1e-12),
                visual_on_off_delta=stats(active_output-static_output),
                interpretation='tiny initialization is a warning, not an automatic architecture failure')
