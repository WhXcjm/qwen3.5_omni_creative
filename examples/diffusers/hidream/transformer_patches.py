#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_hidream_image import (
    HiDreamAttention,
    HiDreamAttnProcessor,
    HiDreamImageFeedForwardSwiGLU,
    MOEFeedForwardSwiGLU,
    MoEGate,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    cos = freqs_cis[..., 0, 0]
    sin = freqs_cis[..., 1, 0]

    # cos + i*sin
    freqs_complex = torch.complex(cos, sin)

    # Rotation with complex multiplication
    xq_rotated = xq_complex * freqs_complex
    xk_rotated = xk_complex * freqs_complex

    xq_out = torch.view_as_real(xq_rotated).reshape(*xq.shape)
    xk_out = torch.view_as_real(xk_rotated).reshape(*xk.shape)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm_npu(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


@maybe_allow_in_graph
class PatchedHiDreamAttention(HiDreamAttention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor=None,
        out_dim: int = None,
        single: bool = False,
    ):
        super(Attention, self).__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.single = single

        self.to_q = nn.Linear(query_dim, self.inner_dim)
        self.to_k = nn.Linear(self.inner_dim, self.inner_dim)
        self.to_v = nn.Linear(self.inner_dim, self.inner_dim)
        self.to_out = nn.Linear(self.inner_dim, self.out_dim)
        self.q_rms_norm = RMSNorm_npu(self.inner_dim, eps)
        self.k_rms_norm = RMSNorm_npu(self.inner_dim, eps)

        if not single:
            self.to_q_t = nn.Linear(query_dim, self.inner_dim)
            self.to_k_t = nn.Linear(self.inner_dim, self.inner_dim)
            self.to_v_t = nn.Linear(self.inner_dim, self.inner_dim)
            self.to_out_t = nn.Linear(self.inner_dim, self.out_dim)
            self.q_rms_norm_t = RMSNorm_npu(self.inner_dim, eps)
            self.k_rms_norm_t = RMSNorm_npu(self.inner_dim, eps)

        self.set_processor(processor)

    def forward(
        self,
        norm_hidden_states: torch.Tensor,
        hidden_states_masks: torch.Tensor = None,
        norm_encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states=norm_hidden_states,
            hidden_states_masks=hidden_states_masks,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )


class PatchedHiDreamAttnProcessor(HiDreamAttnProcessor):
    def __call__(
        self,
        attn: HiDreamAttention,
        hidden_states: torch.Tensor,
        hidden_states_masks: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        query_i = attn.q_rms_norm(attn.to_q(hidden_states)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(hidden_states)).to(dtype=dtype)
        value_i = attn.to_v(hidden_states)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if hidden_states_masks is not None:
            key_i = key_i * hidden_states_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            query_t = attn.q_rms_norm_t(attn.to_q_t(encoder_hidden_states)).to(
                dtype=dtype
            )
            key_t = attn.k_rms_norm_t(attn.to_k_t(encoder_hidden_states)).to(
                dtype=dtype
            )
            value_t = attn.to_v_t(encoder_hidden_states)

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
        else:
            query = query_i
            key = key_i
            value = value_i

        if query.shape[-1] == image_rotary_emb.shape[-3] * 2:
            query, key = apply_rope(query, key, image_rotary_emb)

        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, image_rotary_emb)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if query.dtype in (torch.float16, torch.bfloat16):
            hidden_states = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                attn.heads,
                input_layout="BNSD",
                pse=None,
                scale=1.0 / math.sqrt(query.shape[-1]),
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if not attn.single:
            hidden_states_i, hidden_states_t = torch.split(
                hidden_states, [num_image_tokens, num_text_tokens], dim=1
            )
            hidden_states_i = attn.to_out(hidden_states_i)
            hidden_states_t = attn.to_out_t(hidden_states_t)
            return hidden_states_i, hidden_states_t
        else:
            hidden_states = attn.to_out(hidden_states)
            return hidden_states


class PatchedMOEFeedForwardSwiGLU(MOEFeedForwardSwiGLU):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
        _force_inference_output: bool = False,
    ):
        nn.Module.__init__(self)
        self.shared_experts = HiDreamImageFeedForwardSwiGLU(dim, hidden_dim // 2)
        self.experts = nn.ModuleList(
            [
                HiDreamImageFeedForwardSwiGLU(dim, hidden_dim)
                for i in range(num_routed_experts)
            ]
        )
        self._force_inference_output = _force_inference_output
        self.gate = MoEGate(
            embed_dim=dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            _force_inference_output=_force_inference_output,
        )
        self.num_activated_experts = num_activated_experts
        self.num_routed_experts = num_routed_experts

    def forward(self, x):
        dtype = x.dtype
        identity = x
        orig_shape = x.shape
        N = x.shape[0]  # B * T

        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])  # Flatten to (N, dim)
        flat_topk_idx = topk_idx.view(-1)  # (N*k, )
        flat_topk_weight = topk_weight.view(-1)  # (N*k, )

        if self.training and not self._force_inference_output:
            x_rep = x.repeat_interleave(self.num_activated_experts, dim=0)  # (N*k, dim)

            # Sort tokens by expert index
            sorted_experts, indices = flat_topk_idx.sort(stable=True)
            tokens_sorted = x_rep[indices]
            weights_sorted = flat_topk_weight[indices].unsqueeze(-1)  # (N*k, 1)

            # Count tokens per expert
            counts = torch.histc(
                sorted_experts,
                bins=self.num_routed_experts,
                min=0,
                max=self.num_routed_experts,
            )
            offsets = counts.cumsum(0) - counts  # [0, c0, c0+c1, ...]

            # Output tensor
            y_rep = torch.zeros_like(tokens_sorted, dtype=dtype)

            for i, expert in enumerate(self.experts):
                start = offsets[i]
                end = start + counts[i]
                if counts[i] == 0:
                    continue
                tokens_i = tokens_sorted[start:end]
                weight_i = weights_sorted[start:end]
                out_i = expert(tokens_i).to(dtype)
                out_i.mul_(weight_i)
                y_rep[indices[start:end]] = out_i

            # Reshape & sum over experts
            y = (
                y_rep.view(N, self.num_activated_experts, -1)
                .sum(dim=1)
                .view(orig_shape)
            )
        else:
            y = self.moe_infer(x, flat_topk_idx, flat_topk_weight).view(orig_shape)

        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        sorted_experts, indices = flat_expert_indices.sort(stable=True)
        token_indices = indices // self.num_activated_experts  # Map to orig token idx

        counts = torch.histc(
            sorted_experts,
            bins=self.num_routed_experts,
            min=0,
            max=self.num_routed_experts,
        )
        offsets = counts.cumsum(0) - counts

        # Prepare tensor
        expert_cache = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            start = offsets[i]
            end = start + counts[i]
            if counts[i] == 0:
                continue
            curr_token_indices = token_indices[start:end]
            curr_weights = flat_expert_weights[indices[start:end]].unsqueeze(-1)
            tokens_i = x[curr_token_indices]
            out_i = expert(tokens_i) * curr_weights
            expert_cache.index_add_(0, curr_token_indices, out_i)

        return expert_cache


def apply_patches():
    module = sys.modules["diffusers.models.transformers.transformer_hidream_image"]

    module.HiDreamAttention = PatchedHiDreamAttention
    module.HiDreamAttnProcessor = PatchedHiDreamAttnProcessor
    module.MOEFeedForwardSwiGLU = PatchedMOEFeedForwardSwiGLU
