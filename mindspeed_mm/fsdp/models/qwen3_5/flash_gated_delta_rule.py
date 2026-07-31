# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, List

import torch
import torch_npu
import fla_npu

from mindspeed.lite.ops.triton.l2norm import l2norm_bwd, l2norm_fwd
from mindspeed.lite.ops.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from mindspeed.lite.ops.triton.wy_fast import recompute_w_u_fwd
from mindspeed.lite.ops.triton.solve_tril import solve_tril
from mindspeed.lite.ops.triton.cumsum import chunk_local_cumsum
from mindspeed.lite.ops.triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

def prepare_chunk_indices_list(
    cu_seqlens: list,
    chunk_size:int,
) -> list:
    indices = []
    for i in range(len(cu_seqlens) - 1):
        length = cu_seqlens[i + 1] - cu_seqlens[i]
        if length <= 0:
            continue
        num_chunks = (length + chunk_size - 1) // chunk_size
        for chunk_id in range(num_chunks):
            indices.append(i)
            indices.append(chunk_id)
    return indices


def _prep_indices(cu_seqlens: Optional[torch.Tensor], chunk_size: int, device):
    if cu_seqlens is None:
        return None, None, None, None
    cu_list = cu_seqlens.tolist()
    ch_list = prepare_chunk_indices_list(cu_list, chunk_size)
    cu_tensor = torch.tensor(cu_list, dtype=torch.int64, device=device)
    ch_tensor = torch.tensor(ch_list, dtype=torch.int64, device=device)
    return cu_list, ch_list, cu_tensor, ch_tensor


def flash_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
):
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens)

    _, _, cu_t, ch_t = _prep_indices(cu_seqlens, chunk_size, q.device)

    # Transpose to head-first [B, H, T, K/V] for NPU ops
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    w = w.transpose(1, 2).contiguous()
    u = u.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()

    h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k, w, u, g,
        initial_state=initial_state,
        cu_seqlens=cu_t,
        chunk_indices=ch_t,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )

    o = torch.ops.npu.npu_chunk_fwd_o(
        q, k, v_new, h, scale,
        g=g,
        cu_seqlens=cu_t,
        chunk_indices=ch_t,
        chunk_size=chunk_size,
    )

    g = g.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()

    return g, o, A, final_state


def flash_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
):
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens)

    cu_list, ch_list, cu_t, ch_t = _prep_indices(cu_seqlens, chunk_size, q.device)

    w = w.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    do = do.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    u = u.transpose(1, 2).contiguous()
    A = A.transpose(1, 2).contiguous()

    # Recompute h for backward (fwd_h again, Tensor? indices for  fwd ops)
    h, v_new, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k, w, u, g,
        initial_state=initial_state,
        cu_seqlens=cu_t,
        chunk_indices=ch_t,
        output_final_state=False,
        chunk_size=chunk_size,
    )

    dv = torch.ops.npu.npu_chunk_bwd_dv_local(
        q, k, do, g, scale, chunk_size,
        g_gamma = None,
        A = A,
        cu_seqlens=cu_list,
        chunk_indices=ch_list,
    )

    dh, dh0, dv = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q, k, w, do, dv, scale, chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=dht,
        cu_seqlens=cu_list,
        chunk_indices=ch_list,
    )

    dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
        q, k, v_new, g, h, do, dh, dv, chunk_size,
        cu_seqlens=cu_list,
        chunk_indices=ch_list,
        scale=scale,
    )

    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k, v, beta, A, dw, dv, g,
        cu_seqlens=cu_list,
        chunk_indices=ch_list,
        chunk_size=chunk_size,
    )

    dk2, dv, db, dg2 = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
        k, v, beta, A, dA, dw, dv, g, chunk_size,
        cu_seqlens=cu_list,
        chunk_indices=ch_list,
    )

    dk2 = dk2.transpose(1, 2).contiguous()
    dv = dv.transpose(1, 2).contiguous()
    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()

    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg dtype {dg.dtype}, should be torch.float32")

    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, head_first=False)

    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, final_state = flash_chunk_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = flash_chunk_gated_delta_rule_bwd(
            q=q, k=k, v=v, g=g, beta=beta, A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do, dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        if initial_state is None:
            dh0 = None
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None

@torch.compiler.disable
def flash_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        head_first: bool = False,
):
    if output_final_state and initial_state is None:
        output_final_state = False

    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype} , k current type is {k.dtype} ,v current type is {v.dtype} , they should are equal"
        )
    if q.dtype == torch.float32:
        raise ValueError(
            "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
        )
    if len(beta.shape) != 3:
        raise ValueError(
            f"beta current shape len is {len(beta.shape)}, beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
        )

    if head_first:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    #     """This function is intended to align with the l2norm implementation in the FLA library."""
    #     original_dtype = x.dtype
    #     inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    #     # Counteract verl's autocast promotion (bf16 -> fp32) by restoring original dtype
    #     return (x * inv_norm).to(original_dtype)
    #
    # if use_qk_l2norm_in_kernel:
    #     q = l2norm(q, dim=-1, eps=1e-6)
    #     k = l2norm(k, dim=-1, eps=1e-6)
    #
    # use_qk_l2norm_in_kernel = False

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale,
        initial_state, output_final_state,
        cu_seqlens, use_qk_l2norm_in_kernel, chunk_size,
    )
    return o, final_state