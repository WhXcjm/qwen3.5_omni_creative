# Copyright (c) 2024 Keller Jordan
# Copyright (c) 2025 Moonshot AI
import math
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.optim.adamw import adamw as adamw_functional


def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    if len(G.shape) != 2:
        raise ValueError(
            f"zeropower_via_newtonschulz5 expects a 2-D tensor, got shape {G.shape}"
        )
    
    # Coefficients a,b,c are used to ensure convergence 
    # values are from the source code of Keller Jordan
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def _zeropower_via_newtonschulz5_batched(G, steps):
    """Apply Newton-Schulz independently to a batch of equally shaped matrices."""
    if len(G.shape) != 3:
        raise ValueError(
            f"batched Newton-Schulz expects a 3-D tensor, got shape {G.shape}"
        )

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.transpose(-2, -1)
    return X


def _valid_muon_parts(parts, arity):
    return bool(parts) and all(
        isinstance(part, (tuple, list))
        and len(part) == arity
        and isinstance(part[0], str)
        and all(isinstance(value, int) and value > 0 for value in part[1:])
        for part in parts
    )


def _validate_muon_split_spec(param, split_spec):
    if not isinstance(split_spec, dict):
        raise TypeError("Muon split metadata must be a dict")

    expected_shape = tuple(split_spec.get("expected_shape", ()))
    if tuple(param.shape) != expected_shape:
        raise ValueError(
            f"Muon split metadata expected shape {expected_shape}, got {tuple(param.shape)}"
        )

    layout = split_spec.get("layout")
    parts = tuple(split_spec.get("parts", ()))
    if layout == "contiguous_head_parts":
        # Contiguous by semantic type, e.g. GatedDeltaNet in_proj_qkv:
        #   [Q_head0 | ... | Q_headN | K_head0 | ... | K_headN | V_head0 | ... | V_headM]
        # parts: ((name, head_count, head_dim), ...).  Each [head_dim, input_dim] NS independently.
        if len(expected_shape) != 2 or not _valid_muon_parts(parts, 3):
            raise ValueError("Invalid contiguous_head_parts metadata")
        split_size = sum(head_count * head_dim for _, head_count, head_dim in parts)
        expected_size = expected_shape[0]
    elif layout == "periodic_head_parts":
        # Head-interleaved layout, e.g. MLA q_b_proj / kv_b_proj:
        #   [head0·nope, head0·rope, head1·nope, head1·rope, ..., headN·nope, headN·rope]
        # Requires explicit head_count.  parts: ((name, part_dim), ...).  Per-head per-part NS.
        head_count = split_spec.get("head_count")
        if (
            len(expected_shape) != 2
            or not isinstance(head_count, int)
            or head_count <= 0
            or not _valid_muon_parts(parts, 2)
        ):
            raise ValueError("Invalid periodic_head_parts metadata")
        split_size = head_count * sum(part_dim for _, part_dim in parts)
        expected_size = expected_shape[0]
    elif layout == "contiguous_parts":
        # Contiguous semantic blocks, e.g. MLA kv_a_proj_with_mqa:
        #   [ compressed_kv (d_comp rows) | k_rope (d_rope rows) ]
        # parts: ((name, part_dim), ...).  Whole block NS (not per-head).
        if len(expected_shape) != 2 or not _valid_muon_parts(parts, 2):
            raise ValueError("Invalid contiguous_parts metadata")
        split_size = sum(part_dim for _, part_dim in parts)
        expected_size = expected_shape[0]
    else:
        raise ValueError(f"Unsupported Muon split layout: {layout}")

    if split_size != expected_size:
        raise ValueError(
            f"Muon split parts cover {split_size} values, expected {expected_size}"
        )


def _scaled_zeropower(matrix, steps, matched_adamw_rms):
    update = zeropower_via_newtonschulz5(matrix, steps=steps)
    scale = matched_adamw_rms * math.sqrt(max(matrix.shape[-2:]))
    return update * scale


def _scaled_batched_zeropower(matrices, steps, matched_adamw_rms):
    update = _zeropower_via_newtonschulz5_batched(matrices, steps=steps)
    scale = matched_adamw_rms * math.sqrt(max(matrices.shape[-2:]))
    return update * scale


def _zeropower_via_newtonschulz5_split(G, split_spec, steps, matched_adamw_rms):
    """Split a fused update into semantic matrices, orthogonalize, and regroup."""
    layout = split_spec["layout"]
    expected_shape = tuple(split_spec["expected_shape"])
    parts = tuple(split_spec["parts"])

    if tuple(G.shape) != expected_shape:
        raise ValueError(
            f"Gathered Muon update shape {tuple(G.shape)} does not match {expected_shape}"
        )

    # TODO: Support configurable heads-per-NS-group (for example, 2, 4, or all heads).
    # The current policy treats each attention head as an independent NS unit.
    if layout == "contiguous_head_parts":
        # Contiguous by semantic type, e.g. GatedDeltaNet in_proj_qkv:
        #   G = [Q_head0 | ... | Q_headN | K_head0 | ... | K_headN | V_head0 | ... | V_headM]
        #   Step 1: narrow each part's row range → reshape to [head_count, head_dim, input_dim]
        #   Step 2: batched NS per head_part [head_dim, input_dim] → reshape back to 2D
        #   Step 3: concat along dim=0, restoring the original row order
        input_dim = G.shape[1]
        offset = 0
        updates = []
        for _, head_count, head_dim in parts:
            row_count = head_count * head_dim
            head_batch = G.narrow(0, offset, row_count).reshape(
                head_count, head_dim, input_dim
            )
            head_update = _scaled_batched_zeropower(
                head_batch, steps, matched_adamw_rms
            )
            updates.append(head_update.reshape(row_count, input_dim))
            offset += row_count
        return torch.cat(updates, dim=0)

    if layout == "periodic_head_parts":
        # Head-interleaved layout, e.g. MLA q_b_proj / kv_b_proj:
        #   G = [h0_nope, h0_rope, h1_nope, h1_rope, ..., hN_nope, hN_rope]
        #   Step 1: reshape to [head_count, head_width, input_dim] (head-major view)
        #   Step 2: split along dim=1 by part_dim → each part is [head_count, part_dim, input_dim]
        #   Step 3: batched NS per (head, part) → concat along dim=1 → reshape_as(G)
        head_count = split_spec["head_count"]
        part_dims = [part_dim for _, part_dim in parts]
        head_width = sum(part_dims)
        input_dim = G.shape[1]
        head_major = G.reshape(head_count, head_width, input_dim)
        part_updates = []
        for part_batch in torch.split(head_major, part_dims, dim=1):
            part_updates.append(
                _scaled_batched_zeropower(
                    part_batch, steps, matched_adamw_rms
                )
            )
        return torch.cat(part_updates, dim=1).reshape_as(G)

    if layout == "contiguous_parts":
        # Contiguous semantic blocks, e.g. MLA kv_a_proj_with_mqa:
        #   G = [ compressed_kv (d_c rows) | k_rope (d_r rows) ]
        #   Step 1: torch.split along dim=0 by part_dim
        #   Step 2: NS on each whole block individually (not per-head)
        #   Step 3: torch.cat along dim=0, preserving original row order
        part_dims = [part_dim for _, part_dim in parts]
        return torch.cat(
            [
                _scaled_zeropower(part, steps, matched_adamw_rms)
                for part in torch.split(G, part_dims, dim=0)
            ],
            dim=0,
        )

    raise ValueError(f"Unsupported Muon split layout: {layout}")


def normalize_range(value_range: Tuple[int, int], start: int):
    return (value_range[0] - start, value_range[1] - start)


def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape[:2]
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough).
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, 
                 param_groups,
                 lr: float = 2e-2,
                 weight_decay: float = 0.1,
                 matched_adamw_rms: float = 0.2,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 ns_steps: int = 5,
                 adamw_betas: Tuple[float, float] = (0.95, 0.95),
                 adamw_eps: float = 1e-8):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        
        super().__init__(param_groups, defaults)
        self._muon_split_spec_by_param: Dict[torch.nn.Parameter, Dict[str, Any]] = {}

    def _set_muon_split_specs(
        self,
        split_specs: Dict[torch.nn.Parameter, Dict[str, Any]],
    ) -> None:
        # NOTE: This dict is keyed by Parameter *object identity*, not by value.
        # Parameter does not override __eq__/__hash__, so membership checks
        # resolve via id().  Consequences:
        #   - Only use_muon=True params may carry a split spec; anything else
        #     (fallback AdamW params, non-trainable, or foreign objects) raises.
        #   - Specs are intentionally NOT serialized in state_dict().  After a
        #     checkpoint restore the caller must rebuild the model and re-inject
        #     via this method; stale Parameter references would otherwise fail
        #     the identity check or be silently dropped.
        if not isinstance(split_specs, dict):
            raise TypeError("Muon split specs must be a parameter-keyed dict")

        muon_params = {
            param
            for group in self.param_groups
            if group.get("use_muon", False)
            for param in group["params"]
        }
        for param, split_spec in split_specs.items():
            if param not in muon_params:
                raise ValueError("Muon split metadata can only target use_muon=True parameters")
            _validate_muon_split_spec(param, split_spec)

        self._muon_split_spec_by_param = split_specs.copy()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # First pass: update muon momentum buffers
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue
            
            momentum = group['momentum']
            nesterov = group['nesterov']
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Initialize momentum buffer
                if "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(p.grad)
                
                buf = state["muon_buffer"]
                buf.mul_(momentum).add_(p.grad)
                
                # Prepare for Newton-Schulz iteration
                if nesterov:
                    g = p.grad.add(buf, alpha=momentum)
                else:
                    g = buf
                
                state["ns_input"] = g.to(torch.bfloat16)

        # Second pass: apply updates
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue
            
            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                ns_input = state.pop("ns_input", None)
                if ns_input is None:
                    continue

                split_spec = self._muon_split_spec_by_param.get(p)
                
                # Handle FSDP sharding
                if hasattr(p, 'device_mesh'):
                    device_mesh = ns_input.device_mesh

                    # Rank-2 matrices must be complete before semantic splitting and NS.
                    new_placements = [
                        Replicate() if isinstance(placement, Shard) else placement
                        for placement in ns_input.placements
                    ]
                    ns_input_full = ns_input.redistribute(device_mesh, new_placements)
                    ns_input_local = ns_input_full.to_local()
                else:
                    ns_input_local = ns_input
                
                if split_spec is None:
                    # Preserve the original whole-matrix path for ordinary Muon parameters.
                    update = zeropower_via_newtonschulz5(
                        ns_input_local, steps=ns_steps
                    )
                    adjusted_lr = adjust_lr_wd_for_muon(
                        lr, matched_adamw_rms, ns_input.shape
                    )
                else:
                    update = _zeropower_via_newtonschulz5_split(
                        ns_input_local,
                        split_spec,
                        steps=ns_steps,
                        matched_adamw_rms=matched_adamw_rms,
                    )
                    adjusted_lr = lr
                
                # Handle FSDP sharding for output
                if hasattr(p, 'device_mesh'):
                    update_dtensor = DTensor.from_local(update, p.device_mesh, new_placements)
                    update_sharded = update_dtensor.redistribute(p.device_mesh, p.placements)
                    update = update_sharded
                
                # Apply weight decay
                p.mul_(1 - lr * weight_decay)
                
                # Split updates already carry their own shape-dependent Muon scaling.
                p.data.add_(update, alpha=-adjusted_lr)

        # Use AdamW for non-Muon parameters
        for group in self.param_groups:

            if group.get('use_muon', False):
                continue

            beta1, beta2 = group['adamw_betas']
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            has_complex = False

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients, please consider SparseAdam instead"
                    )

                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]

                if "step" not in state:
                    state["step"] = torch.tensor(
                        0.0, dtype=torch.float32, device="cpu"
                    )
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

            if params_with_grad:
                adamw_functional(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    [],
                    state_steps,
                    foreach=False,
                    capturable=False,
                    differentiable=False,
                    fused=False,
                    grad_scale=None,
                    found_inf=None,
                    has_complex=has_complex,
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['adamw_eps'],
                    maximize=False,
                )

        return loss
