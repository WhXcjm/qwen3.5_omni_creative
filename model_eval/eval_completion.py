#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-process, read-only text completion from a MindSpeed-MM FSDP2 DCP checkpoint.

Loads ONLY the language model + lm_head weights (skips the frozen vision tower,
the MTP block, and all optimizer state), keeps the checkpoint strictly read-only,
and runs greedy (argmax) decoding.

Why a dedicated loader (instead of dcp.load with FSDP2): the checkpoint was saved
by 8-rank FSDP2. We reconstruct the *full* (unsharded) fp32 state dict in a single
process via `_EmptyStateDictLoadPlanner` + `no_dist`, then load the text branch into
a plain `Qwen3_5MoeForConditionalGeneration` built on CPU with the torch (non-fused)
attention / GDN / expert-matmul fallbacks.

Usage (from repo root, conda env `muon`):
    python model_eval/eval_completion.py \
        --ckpt /data/w50062357/weights/.../ckpt_02_dyn_lr4e-4_hg1/iter_0005000 \
        --prompt "Machine learning is a branch of artificial intelligence that"
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict

# The trimmed wxj0804 repo has no `megatron/`; mindspeed_mm's top-level __init__
# imports megatron-dependent patches unless this is set (the training launch
# scripts set the same flag). Must be set before any mindspeed_mm import.
os.environ.setdefault("NON_MEGATRON", "true")

# Cap CPU threads *before* torch is imported: batch=1 autoregressive decode is
# memory-bound and 192 cores spawn hundreds of OpenMP/oneDNN threads that livelock
# on futexes (observed: 538 threads, 0 progress). Keep it modest and leave headroom
# for the NPU training that is running concurrently on this box.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import torch


# ---------------------------------------------------------------------------
# Single-process distributed shim
# ---------------------------------------------------------------------------
def _setup_single_process_dist(device: str = "cpu"):
    """Bootstrap a 1-rank gloo process group so the model's `get_parallel_state()`
    (used unconditionally in the forward pass) can build its device meshes without
    a torchrun environment. For `device="cpu"` we also force every device mesh onto
    CPU (never touching the NPUs); for `device="npu"` we leave native NPU detection
    intact and only neutralise the (would-be HCCL) device meshes."""
    import torch.distributed as dist

    if dist.is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    import mindspeed_mm.fsdp.utils.device as _dev_mod
    import mindspeed_mm.fsdp.distributed.parallel_state as _ps_mod

    if device == "cpu":
        # Force device meshes onto CPU (all parallel sizes stay at 1 -> every
        # `is_*_enable()` check is False, so the parallel branches are no-ops).
        _dev_mod.get_device_type = lambda: "cpu"
        _dev_mod.IS_NPU_AVAILABLE = False
        _ps_mod.get_device_type = lambda: "cpu"
    else:
        # NPU: keep native detection so the model uses npu fused ops (rms_norm,
        # rotary_mul, …). torch_npu registers the `npu` device + HCCL new_group.
        import torch_npu  # noqa: F401

    # torch_npu monkeypatches `torch.distributed.new_group`, so a real
    # `init_device_mesh` routes through its HCCL helper and hangs a 1-rank eval.
    # Replace `init_device_mesh` with a single-rank dummy mesh that satisfies every
    # `get_*_group`/`is_*_enable` call in the model forward without ever creating a
    # real device mesh.
    class _DummyMesh:
        def _flatten(self, mesh_dim_name=None):
            return self

        def __getitem__(self, key):
            return self

        def get_group(self, name=None):
            return torch.distributed.group.WORLD

        def get_local_rank(self, name=None):
            return 0

    _ps_mod.init_device_mesh = lambda *a, **k: _DummyMesh()

    # The grad-offload module instantiates a singleton at import time that calls
    # `create_stream()`. CPU has no device stream, so stub the stream/event
    # factories (they are only ever consumed during backward, which eval never runs).
    # Must be patched here, *before* the model class is imported.
    _dev_mod.create_stream = lambda device=None, priority=0: None
    _dev_mod.create_event = lambda enable_timing=False, blocking=False: None
    _dev_mod.get_current_stream = lambda device=None: None
    _dev_mod.switch_to_specified_stream = lambda *a, **k: None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DCP checkpoint -> greedy text completion")
    p.add_argument("--ckpt", required=True,
                   help="Path to the DCP iteration dir (contains .metadata + __*.distcp).")
    p.add_argument("--prompt", default="Machine learning is a branch of artificial intelligence that",
                   help="Input prompt text.")
    p.add_argument("--config", default=None,
                   help="HF model asset dir (config.json + tokenizer). "
                        "Default: $REPO/hf_models/qwen3_5-10b-a2b-exp4")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                   help="Attention backend (CPU-safe; flash_attention_2 is NPU-only).")
    p.add_argument("--recompute", action="store_true",
                   help="Full-sequence forward each step (no cache). O(n^2) reference "
                        "path for cross-checking the cached decode; not for production.")
    p.add_argument("--device", default="npu", choices=["npu", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None,
                   help="Optional JSON path to dump a full result record.")
    p.add_argument("--repo-root", default=None,
                   help="Repo root for config default resolution + import path. "
                        "Defaults to the repo containing this file.")
    return p.parse_args()


def repo_root_of_this_file() -> str:
    # model_eval/ is at the repo root; this file is <root>/model_eval/eval_completion.py
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------
def build_model(config_dir: str, attn: str, dtype_str: str, device: str = "cpu",
                use_triton_gdn: bool = False, use_grouped_expert_matmul: bool = False):
    from transformers import AutoConfig
    from accelerate import init_empty_weights
    from mindspeed_mm.fsdp.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)

    tc = config.text_config
    tc._attn_implementation = attn
    # Default torch fallbacks (correct for decode/completion eval): the fused
    # triton-GDN chunk rule diverges on short (seq_len==1) prefill and is skipped.
    # PPL eval does a full-length (4096) prefill forward, the same as training, so it
    # may enable the Triton GDN + grouped-expert kernels (via the --fast flag) to match
    # training speed. causal_conv1d/fla are not installed here; the conv step falls back
    # to torch conv1d, which is minor relative to the chunked-GDN core.
    tc.use_triton_gdn = use_triton_gdn
    tc.use_ascend_gdn = False
    tc.use_grouped_expert_matmul = use_grouped_expert_matmul
    tc.use_cache = True
    # Skip MTP + aux-loss machinery for eval (weights are not loaded).
    tc.mtp_num_layers = 0
    tc.router_aux_loss_coef = 0.0
    tc.router_aux_loss_offload = False
    config._attn_implementation = attn

    # Build on meta device (instant); real weights are materialised by the
    # loader. Random init of 10B params would otherwise take minutes on CPU.
    with init_empty_weights():
        model = Qwen3_5MoeForConditionalGeneration(config)

    # MLA has no GQA (num_kv_heads == num_heads) but the standard eager attention
    # does repeat_kv(key, module.num_key_value_groups). Patch at runtime (idempotent)
    # so the repo's model file does not need to be edited on every checkout.
    for _m in model.modules():
        if type(_m).__name__ == "Qwen3_5MoeMLA" and not hasattr(_m, "num_key_value_groups"):
            _m.num_key_value_groups = 1

    _patch_gdn_conv1d_update(model)

    model.eval()
    return model, config


def _patch_gdn_conv1d_update(model):
    """Fix the torch-fallback `causal_conv1d_update` for cache-based decoding.

    `torch_causal_conv1d_update` is channels-first (`[B, D, S]`), but the GDN
    forward's `use_precomputed_states` (decode, seq_len==1) branch passes the
    seq-first `[B, S, D]` `mixed_qkv` without transposing — the no-cache branch
    transposes, and the fused causal_conv1d lib (which would mask the difference)
    isn't installed in the eval env. Wrap each GDN module's update fn so it
    transposes on entry; output stays `[B, D, S]`, which the caller transposes
    back (line `mixed_qkv.transpose(1, 2)` right after the call).
    """
    import mindspeed_mm.fsdp.models.qwen3_5_moe.modeling_qwen3_5_moe as _moe

    _orig = _moe.torch_causal_conv1d_update

    def _fixed(hidden_states, conv_state, weight, bias=None, activation=None):
        return _orig(hidden_states.transpose(1, 2), conv_state, weight, bias, activation)

    patched = 0
    for _m in model.modules():
        if getattr(_m, "causal_conv1d_update", None) is _orig:
            _m.causal_conv1d_update = _fixed
            patched += 1
    print(f"[patch] gdn causal_conv1d_update fixed on {patched} layers", flush=True)


# ---------------------------------------------------------------------------
# DCP loading (model-only, read-only)
# ---------------------------------------------------------------------------
def _flatten_dict(d, prefix="", out=None):
    """Flatten a nested state dict back into dotted FQNs {name: tensor}."""
    if out is None:
        out = OrderedDict()
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, name, out)
        else:
            out[name] = v
    return out


def load_text_weights(ckpt_dir: str):
    """
    Return a dict {param_name: tensor} for the text branch only.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from mindspeed_mm.fsdp.checkpoint.dcp_utils import (
        load_metadata,
        extract_metadata,
        partial_load_dcp_state_dict,
    )

    reader = FileSystemReader(ckpt_dir)
    metadata = load_metadata(reader)

    # Select only the text branch: language model + output head.
    # Metadata FQNs carry the DCP save-state "model." prefix, so a text tensor
    # looks like "model.model.language_model.layers.0...." and lm_head is
    # "model.lm_head.weight".
    selected = [
        k for k in metadata.state_dict_metadata
        if k.startswith("model.model.language_model.")
        or k.startswith("model.lm_head.")
    ]
    if not selected:
        raise RuntimeError(f"No language_model/lm_head keys found in {ckpt_dir}")

    partial_md = extract_metadata(selected, metadata)
    loaded = partial_load_dcp_state_dict(partial_md, reader)

    # `partial_load_dcp_state_dict` reconstructs the nested state-dict hierarchy
    # from the DCP FQNs (e.g. "model.model.language_model.embed_tokens.weight" ->
    # {"model": {"model": {"language_model": ...}}}). Drop the outer save-state
    # "model" wrapper, then flatten back to dotted FQNs that line up exactly with
    # Qwen3_5MoeForConditionalGeneration.state_dict() keys.
    inner = loaded["model"] if "model" in loaded else loaded
    return _flatten_dict(inner)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str,
                    use_cache: bool = True):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(model.config, "eos_token_id", None)

    t0 = time.time()
    new_ids = []

    if use_cache:
        # Prefill the prompt once (builds the Qwen3_5MoeDynamicCache), then decode
        # one token at a time reusing the cache (O(n) instead of O(n^2)).
        out = model(input_ids=input_ids, use_cache=True, output_router_logits=False)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        for step in range(max_new_tokens):
            tid = next_token.item()
            new_ids.append(tid)
            print(f"[token {step+1}/{max_new_tokens}] id={tid} "
                  f"piece={tokenizer.decode([tid], skip_special_tokens=True)!r}", flush=True)
            if eos_id is not None and tid == eos_id:
                break
            out = model(input_ids=next_token[:, None], past_key_values=past,
                        use_cache=True, output_router_logits=False)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1)
    else:
        # Reference path: full-sequence forward every step (no cache). O(n^2) and
        # slow, but the chunk-parallel GDN is the training formulation, so this is
        # the ground truth for cross-checking the cached (recurrent) decode.
        seq = input_ids
        for step in range(max_new_tokens):
            out = model(input_ids=seq, use_cache=False, output_router_logits=False)
            next_token = out.logits[:, -1, :].argmax(dim=-1)
            tid = next_token.item()
            new_ids.append(tid)
            print(f"[token {step+1}/{max_new_tokens}] id={tid} "
                  f"piece={tokenizer.decode([tid], skip_special_tokens=True)!r}", flush=True)
            if eos_id is not None and tid == eos_id:
                break
            seq = torch.cat([seq, next_token[:, None]], dim=-1)

    decode_seconds = time.time() - t0

    new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(input_ids[0].tolist() + new_ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "prompt_tokens": prompt_len,
        "new_tokens": len(new_ids),
        "new_token_ids": new_ids,
        "completion": new_text,
        "full_text": full_text,
        "decode_seconds": round(decode_seconds, 3),
        "tokens_per_sec": round(len(new_ids) / decode_seconds, 2) if decode_seconds > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    repo = args.repo_root or repo_root_of_this_file()
    sys.path.insert(0, repo)
    os.chdir(repo)

    config_dir = args.config or os.path.join(repo, "hf_models", "qwen3_5-10b-a2b-exp4")
    ckpt_dir = os.path.abspath(args.ckpt)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))

    _setup_single_process_dist(args.device)

    t0 = time.time()

    # 1. tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config_dir, trust_remote_code=True)

    # 2. model (meta); weights are materialised by the loader on CPU.
    model, config = build_model(config_dir, args.attn, args.dtype, args.device)

    # 3. weights
    weights = load_text_weights(ckpt_dir)
    load_report = load_into_model(model, weights, args.dtype, args.device)
    _move_to_device(model, args.device)

    # 4. generate
    result = greedy_generate(model, tokenizer, args.prompt, args.max_new_tokens, args.device,
                             use_cache=not args.recompute)

    result["ckpt"] = ckpt_dir
    result["config_dir"] = config_dir
    result["attn"] = args.attn
    result["dtype"] = args.dtype
    result["load_report"] = load_report
    result["wall_seconds"] = round(time.time() - t0, 1)

    print("=" * 78)
    print(f"CKPT      : {ckpt_dir}")
    print(f"PROMPT    : {args.prompt}")
    print(f"LOADED    : {load_report['loaded']}/{load_report['present']} text params "
          f"({load_report['bytes']/1e9:.2f} GB), missing={len(load_report['missing'])}, "
          f"unexpected={len(load_report['unexpected'])}")
    print(f"COMPLETION: {result['completion']}")
    print("=" * 78)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.out}")

    return result


def _move_to_device(model, device):
    """Move real (non-meta) params/buffers onto `device`. The text params are
    already materialised there by `load_into_model`; this catches non-persistent
    buffers (e.g. RoPE `inv_freq`) that were created on CPU and the meta vision
    tower (skipped)."""
    device = torch.device(device)
    for t in list(model.parameters()) + list(model.buffers()):
        if t.device.type == "meta" or t.device == device:
            continue
        t.data = t.to(device)


def load_into_model(model, weights, dtype_str, device="cpu"):
    """Materialise text-branch weights into the (meta) model; report diagnostics."""
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    state = model.state_dict()
    present = OrderedDict()
    for k, v in weights.items():
        if k in state:
            present[k] = v.to(dtype=dtype, device=device)
    missing = [k for k in state if k not in weights]
    unexpected = [k for k in weights if k not in state]
    nbytes = sum(v.numel() * v.element_size() for v in present.values())

    # assign=True materialises meta params in place; strict=False skips the
    # frozen vision tower (left on meta, never used by text-only generation).
    model.load_state_dict(present, strict=False, assign=True)

    # Keep a couple of stats to prove real (non-random) weights were loaded.
    stats = {}
    for probe in ("model.language_model.embed_tokens.weight",
                  "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"):
        if probe in present:
            t = present[probe].float()
            stats[probe] = {"mean": round(t.mean().item(), 6), "std": round(t.std().item(), 6)}
    return {
        "loaded": len(present),
        "present": len(weights),
        "bytes": nbytes,
        "missing": missing,
        "unexpected": unexpected,
        "stats": stats,
    }


if __name__ == "__main__":
    main()
