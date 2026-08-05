#!/usr/bin/env python3
"""F4 static/NumPy probes for the Qwen3.5 10B-A2B FSDP2 Muon audit.

This script intentionally imports no project modules and mutates no training state.
It reconstructs the trainable language/MTP parameter inventory from the pinned
model config, applies the routing rules from the three candidate commits, traces
the scheduler, and checks Muon split/regroup and Shard(0) equivalence with NumPy.

It is not a substitute for the server-side torch.distributed._tensor/NPU probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FALLBACK_F1 = (
    ("suffix", ".bias"),
    ("contains", "embedding"),
    ("contains", "output_layer"),
    ("suffix", "embed_tokens.weight"),
    ("suffix", "lm_head.weight"),
)


@dataclass(frozen=True)
class Param:
    fqn: str
    shape: tuple[int, ...]
    category: str
    split_layout: str | None = None

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def parse_scalar(value: str):
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_yaml_scalars(path: Path) -> dict[tuple[str, ...], object]:
    """Parse only indentation-based scalar mappings needed by this probe."""
    result: dict[tuple[str, ...], object] = {}
    stack: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("-"):
            continue
        match = re.match(r"^(\s*)([^:]+):(?:\s*(.*))?$", line)
        if not match:
            continue
        indent = len(match.group(1).replace("\t", "    "))
        key = match.group(2).strip()
        value = (match.group(3) or "").strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path_key = tuple(name for _, name in stack) + (key,)
        if value:
            result[path_key] = parse_scalar(value)
        else:
            stack.append((indent, key))
    return result


def add_mla(params: list[Param], prefix: str, cfg: dict, scope: str) -> None:
    h = cfg["hidden_size"]
    nheads = cfg["num_attention_heads"]
    mla = cfg["mla_args"]
    q_rank = mla["q_lora_rank"]
    kv_rank = mla["kv_lora_rank"]
    q_nope = mla["qk_nope_head_dim"]
    q_rope = mla["qk_rope_head_dim"]
    v_dim = mla["v_head_dim"]
    q_width = q_nope + q_rope
    cat = f"{scope}_mla_attention"
    params.extend(
        [
            Param(f"{prefix}.q_a_proj.weight", (q_rank, h), cat),
            Param(f"{prefix}.q_a_layernorm.weight", (q_rank,), f"{scope}_norm_misc"),
            Param(
                f"{prefix}.q_b_proj.weight",
                (nheads * q_width, q_rank),
                cat,
                "periodic_head_parts",
            ),
            Param(
                f"{prefix}.kv_a_proj_with_mqa.weight",
                (kv_rank + q_rope, h),
                cat,
                "contiguous_parts",
            ),
            Param(f"{prefix}.kv_a_layernorm.weight", (kv_rank,), f"{scope}_norm_misc"),
            Param(
                f"{prefix}.kv_b_proj.weight",
                (nheads * (q_nope + v_dim), kv_rank),
                cat,
                "periodic_head_parts",
            ),
            Param(f"{prefix}.o_proj.weight", (h, nheads * v_dim), cat),
        ]
    )


def add_gdn(params: list[Param], prefix: str, cfg: dict) -> None:
    h = cfg["hidden_size"]
    nk = cfg["linear_num_key_heads"]
    nv = cfg["linear_num_value_heads"]
    dk = cfg["linear_key_head_dim"]
    dv = cfg["linear_value_head_dim"]
    kernel = cfg["linear_conv_kernel_dim"]
    key_dim = nk * dk
    value_dim = nv * dv
    conv_dim = 2 * key_dim + value_dim
    params.extend(
        [
            Param(f"{prefix}.conv1d.weight", (conv_dim, 1, kernel), "base_norm_misc"),
            Param(f"{prefix}.dt_bias", (nv,), "base_norm_misc"),
            Param(f"{prefix}.A_log", (nv,), "base_norm_misc"),
            Param(f"{prefix}.norm.weight", (dv,), "base_norm_misc"),
            Param(f"{prefix}.out_proj.weight", (h, value_dim), "base_gdn_attention"),
            Param(
                f"{prefix}.in_proj_qkv.weight",
                (conv_dim, h),
                "base_gdn_attention",
                "contiguous_head_parts",
            ),
            Param(f"{prefix}.in_proj_z.weight", (value_dim, h), "base_gdn_attention"),
            Param(f"{prefix}.in_proj_b.weight", (nv, h), "base_gdn_attention"),
            Param(f"{prefix}.in_proj_a.weight", (nv, h), "base_gdn_attention"),
        ]
    )


def add_latent_moe(params: list[Param], prefix: str, cfg: dict, scope: str) -> None:
    h = cfg["hidden_size"]
    factor = cfg["latent_moe_factor"]
    latent_h = h // factor
    experts = cfg["num_experts"] * factor
    expert_i = cfg["moe_intermediate_size"]
    shared_i = cfg["shared_expert_intermediate_size"]
    params.extend(
        [
            Param(f"{prefix}.gate.weight", (experts, h), f"{scope}_router"),
            Param(
                f"{prefix}.experts.gate_up_proj",
                (experts, latent_h, 2 * expert_i),
                f"{scope}_latent_experts",
                "expert_batched_parts",
            ),
            Param(
                f"{prefix}.experts.down_proj",
                (experts, expert_i, latent_h),
                f"{scope}_latent_experts",
                "expert_batched_parts",
            ),
            Param(f"{prefix}.shared_expert.gate_proj.weight", (shared_i, h), f"{scope}_shared_expert"),
            Param(f"{prefix}.shared_expert.up_proj.weight", (shared_i, h), f"{scope}_shared_expert"),
            Param(f"{prefix}.shared_expert.down_proj.weight", (h, shared_i), f"{scope}_shared_expert"),
            Param(f"{prefix}.shared_expert_gate.weight", (1, h), f"{scope}_shared_expert"),
            Param(f"{prefix}.down_project.weight", (latent_h, h), f"{scope}_latent_projection"),
            Param(f"{prefix}.up_project.weight", (h, latent_h), f"{scope}_latent_projection"),
        ]
    )


def build_param_inventory(cfg: dict, mtp_layers: int) -> list[Param]:
    params: list[Param] = []
    h = cfg["hidden_size"]
    params.append(
        Param(
            "model.language_model.embed_tokens.weight",
            (cfg["vocab_size"], h),
            "embedding_tied_lm_head",
        )
    )
    for layer_idx, layer_type in enumerate(cfg["layer_types"]):
        layer = f"model.language_model.layers.{layer_idx}"
        if layer_type == "linear_attention":
            add_gdn(params, f"{layer}.linear_attn", cfg)
        elif layer_type == "full_attention":
            add_mla(params, f"{layer}.self_attn", cfg, "base")
        else:
            raise ValueError(f"Unexpected layer type: {layer_type}")
        add_latent_moe(params, f"{layer}.mlp", cfg, "base")
        params.extend(
            [
                Param(f"{layer}.input_layernorm.weight", (h,), "base_norm_misc"),
                Param(f"{layer}.post_attention_layernorm.weight", (h,), "base_norm_misc"),
            ]
        )
    params.append(Param("model.language_model.norm.weight", (h,), "base_norm_misc"))

    for layer_idx in range(mtp_layers):
        layer = f"mtp.layers.{layer_idx}"
        add_mla(params, f"{layer}.self_attn", cfg, "mtp")
        add_latent_moe(params, f"{layer}.mlp", cfg, "mtp")
        params.extend(
            [
                Param(f"{layer}.input_layernorm.weight", (h,), "mtp_norm_misc"),
                Param(f"{layer}.post_attention_layernorm.weight", (h,), "mtp_norm_misc"),
            ]
        )
    if mtp_layers:
        params.extend(
            [
                Param("mtp.pre_fc_norm_embedding.weight", (h,), "mtp_bridge"),
                Param("mtp.pre_fc_norm_hidden.weight", (h,), "mtp_bridge"),
                Param("mtp.fc.weight", (h, 2 * h), "mtp_bridge"),
                Param("mtp.norm.weight", (h,), "mtp_bridge"),
            ]
        )
    return params


def matches_rules(name: str, rules: Iterable[tuple[str, str]]) -> bool:
    for kind, pattern in rules:
        if kind == "suffix" and name.endswith(pattern):
            return True
        if kind == "contains" and pattern in name:
            return True
    return False


def route(version: str, param: Param) -> str:
    if version == "adamw":
        return "AdamW"
    if version == "d2da1c0":
        eligible = len(param.shape) == 2 and not (
            param.fqn.endswith(".bias")
            or "embedding" in param.fqn
            or "output_layer" in param.fqn
        )
        return "Muon" if eligible else "AdamW"
    eligible_f1 = len(param.shape) == 2 and not matches_rules(param.fqn, FALLBACK_F1)
    if version == "647a5f7":
        return "Muon" if eligible_f1 else "AdamW"
    if version == "d5a5decc":
        expert_opt_in = param.split_layout == "expert_batched_parts"
        return "Muon" if eligible_f1 or expert_opt_in else "AdamW"
    raise ValueError(version)


def cosine_lr(
    scheduler_step: int,
    train_steps: int,
    warmup_steps: int,
    init_lr: float,
    min_lr: float,
) -> float:
    if scheduler_step < warmup_steps:
        return init_lr * scheduler_step / max(1, warmup_steps)
    progress = (scheduler_step - warmup_steps) / max(1, train_steps - warmup_steps)
    min_ratio = min_lr / init_lr
    factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return init_lr * (factor * (1.0 - min_ratio) + min_ratio)


def zeropower(matrix: np.ndarray, steps: int = 5) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (np.linalg.norm(x) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


def scaled_zeropower(matrix: np.ndarray, matched: float = 0.2) -> np.ndarray:
    return zeropower(matrix) * (matched * math.sqrt(max(matrix.shape[-2:])))


def scaled_batched(matrices: np.ndarray, matched: float = 0.2) -> np.ndarray:
    return np.stack([scaled_zeropower(m, matched) for m in matrices], axis=0)


def split_update(matrix: np.ndarray, layout: str, spec: dict, matched: float = 0.2) -> np.ndarray:
    parts = spec["parts"]
    if layout == "contiguous_head_parts":
        out = []
        offset = 0
        for _, heads, dim in parts:
            rows = heads * dim
            batch = matrix[offset : offset + rows].reshape(heads, dim, matrix.shape[1])
            out.append(scaled_batched(batch, matched).reshape(rows, matrix.shape[1]))
            offset += rows
        return np.concatenate(out, axis=0)
    if layout == "periodic_head_parts":
        heads = spec["head_count"]
        dims = [dim for _, dim in parts]
        head_major = matrix.reshape(heads, sum(dims), matrix.shape[1])
        batches = np.split(head_major, np.cumsum(dims)[:-1], axis=1)
        return np.concatenate([scaled_batched(batch, matched) for batch in batches], axis=1).reshape(matrix.shape)
    if layout == "contiguous_parts":
        dims = [dim for _, dim in parts]
        blocks = np.split(matrix, np.cumsum(dims)[:-1], axis=0)
        return np.concatenate([scaled_zeropower(block, matched) for block in blocks], axis=0)
    if layout == "expert_batched_parts":
        dims = [dim for _, dim in parts]
        blocks = np.split(matrix, np.cumsum(dims)[:-1], axis=2)
        return np.concatenate([scaled_batched(block, matched) for block in blocks], axis=2)
    raise ValueError(layout)


def rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def adamw_updates(grads: list[np.ndarray], beta1=0.9, beta2=0.999, eps=1e-8) -> list[np.ndarray]:
    m = np.zeros_like(grads[0], dtype=np.float64)
    v = np.zeros_like(grads[0], dtype=np.float64)
    updates = []
    for step, grad in enumerate(grads, start=1):
        grad = grad.astype(np.float64)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        updates.append(m_hat / (np.sqrt(v_hat) + eps))
    return updates


def muon_updates(grads: list[np.ndarray], split: bool) -> list[np.ndarray]:
    momentum = 0.95
    buffer = np.zeros_like(grads[0], dtype=np.float32)
    updates = []
    spec = {"parts": (("left", grads[0].shape[0] // 2), ("right", grads[0].shape[0] // 2))}
    for grad in grads:
        buffer = momentum * buffer + grad
        ns_input = grad + momentum * buffer
        if split:
            updates.append(split_update(ns_input, "contiguous_parts", spec))
        else:
            updates.append(scaled_zeropower(ns_input))
    return updates


def print_inventory(params: list[Param], full: bool) -> None:
    versions = ("d2da1c0", "647a5f7", "d5a5decc", "adamw")
    total = sum(p.numel for p in params)
    print("\nPARAMETER ROUTING (trainable language + MTP; visual is frozen)")
    print(f"unique trainable numel={total:,}")
    print("lm_head.weight is tied to model.language_model.embed_tokens.weight and is not double-counted")
    print(f"{'version':12} {'Muon':>16} {'fallback AdamW':>16} {'Muon %':>9}")
    for version in versions:
        muon = sum(p.numel for p in params if route(version, p) == "Muon")
        print(f"{version:12} {muon:16,} {total - muon:16,} {100.0 * muon / total:8.3f}%")

    categories = sorted({p.category for p in params})
    print("\ncategory coverage by candidate version")
    print(f"{'category':30} {'numel':>16} {'d2 route':>10} {'F1 route':>10} {'F2 route':>10}")
    for category in categories:
        selected = [p for p in params if p.category == category]
        numel = sum(p.numel for p in selected)
        cells = []
        for version in versions[:3]:
            muon = sum(p.numel for p in selected if route(version, p) == "Muon")
            cells.append("Muon" if muon == numel else "AdamW" if muon == 0 else f"mix {muon/numel:.1%}")
        print(f"{category:30} {numel:16,} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}")

    split_attention = sum(
        p.numel
        for p in params
        if p.split_layout and p.split_layout != "expert_batched_parts"
    )
    expert_numel = sum(p.numel for p in params if p.split_layout == "expert_batched_parts")
    print("\nF2 semantic opt-in sizes")
    print(f"attention split params={split_attention:,} ({split_attention/1e9:.6f}B)")
    print(f"latent expert params={expert_numel:,} ({expert_numel/1e9:.6f}B)")
    print(f"expert/attention ratio={expert_numel/split_attention:.6f}x")

    if full:
        print("\nFULL FQN INVENTORY")
        print(f"{'FQN':105} {'shape':25} {'numel':>14} {'d2':>6} {'F1':>6} {'F2':>6} {'split'}")
        for p in params:
            print(
                f"{p.fqn:105} {str(p.shape):25} {p.numel:14,} "
                f"{route('d2da1c0', p):>6} {route('647a5f7', p):>6} {route('d5a5decc', p):>6} "
                f"{p.split_layout or '-'}"
            )


def print_lr_trace(yaml_values: dict[tuple[str, ...], object]) -> None:
    get = yaml_values.__getitem__
    train_steps = int(get(("training", "train_iters")))
    init_lr = float(get(("training", "lr")))
    min_lr = float(get(("training", "lr_min")))
    warmup_ratio = float(get(("training", "lr_warmup_ratio")))
    warmup_steps = int(train_steps * warmup_ratio)
    print("\nSCHEDULER TRACE")
    print(f"train_steps={train_steps}, warmup_steps={warmup_steps}, init_lr={init_lr:.9g}, min_lr={min_lr:.9g}")
    print("iteration i uses LambdaLR scheduler_step=i-1; scheduler.step() then prepares step i")
    print(f"{'iteration':>9} {'used/logged LR':>18} {'post-step/next LR':>20}")
    for iteration in (1, 18, 50, 100, 120):
        used = cosine_lr(iteration - 1, train_steps, warmup_steps, init_lr, min_lr)
        next_lr = cosine_lr(iteration, train_steps, warmup_steps, init_lr, min_lr)
        print(f"{iteration:9d} {used:18.10e} {next_lr:20.10e}")


def print_update_probe(base_lr: float, weight_decay: float) -> None:
    rng = np.random.default_rng(20260804)
    grads = [rng.normal(size=(32, 16)).astype(np.float32) for _ in range(5)]
    adam = adamw_updates(grads)
    whole = muon_updates(grads, split=False)
    split = muon_updates(grads, split=True)
    print("\nSYNTHETIC UPDATE RMS (constant base LR; no weight decay in main update column)")
    print(f"{'step':>4} {'AdamW':>14} {'Muon whole':>14} {'Muon split':>14}")
    for idx in range(5):
        print(
            f"{idx+1:4d} {base_lr*rms(adam[idx]):14.8e} "
            f"{base_lr*rms(whole[idx]):14.8e} {base_lr*rms(split[idx]):14.8e}"
        )
    weight_rms = 0.02
    wd_rms = base_lr * weight_decay * weight_rms
    print(f"nominal AdamW first-step RMS (epsilon not dominant)={base_lr:.8e}")
    print(f"nominal Muon target RMS=matched(0.2)*lr={0.2*base_lr:.8e}")
    print(f"weight-decay delta RMS at weight RMS 0.02={wd_rms:.8e}")
    print(f"AdamW update/weight={base_lr/weight_rms:.6e}; Muon target update/weight={0.2*base_lr/weight_rms:.6e}")


def print_split_and_shard_checks() -> None:
    rng = np.random.default_rng(7)
    checks = []

    gdn = rng.normal(size=(24, 12)).astype(np.float32)
    gdn_spec = {"parts": (("q", 2, 4), ("k", 2, 4), ("v", 2, 4))}
    gdn_actual = split_update(gdn, "contiguous_head_parts", gdn_spec)
    gdn_ref = np.concatenate(
        [
            scaled_batched(gdn[0:8].reshape(2, 4, 12)).reshape(8, 12),
            scaled_batched(gdn[8:16].reshape(2, 4, 12)).reshape(8, 12),
            scaled_batched(gdn[16:24].reshape(2, 4, 12)).reshape(8, 12),
        ],
        axis=0,
    )
    checks.append(("contiguous_head_parts", np.max(np.abs(gdn_actual - gdn_ref))))

    periodic = rng.normal(size=(20, 12)).astype(np.float32)
    periodic_spec = {"head_count": 4, "parts": (("nope", 3), ("rope", 2))}
    periodic_actual = split_update(periodic, "periodic_head_parts", periodic_spec)
    head_major = periodic.reshape(4, 5, 12)
    periodic_ref = np.concatenate(
        [scaled_batched(head_major[:, :3]), scaled_batched(head_major[:, 3:])], axis=1
    ).reshape(periodic.shape)
    checks.append(("periodic_head_parts", np.max(np.abs(periodic_actual - periodic_ref))))

    contiguous = rng.normal(size=(12, 9)).astype(np.float32)
    contiguous_spec = {"parts": (("compressed", 7), ("rope", 5))}
    contiguous_actual = split_update(contiguous, "contiguous_parts", contiguous_spec)
    contiguous_ref = np.concatenate(
        [scaled_zeropower(contiguous[:7]), scaled_zeropower(contiguous[7:])], axis=0
    )
    checks.append(("contiguous_parts", np.max(np.abs(contiguous_actual - contiguous_ref))))

    expert = rng.normal(size=(8, 10, 14)).astype(np.float32)
    expert_spec = {"parts": (("gate", 7), ("up", 7))}
    expert_actual = split_update(expert, "expert_batched_parts", expert_spec)
    expert_ref = np.concatenate(
        [scaled_batched(expert[:, :, :7]), scaled_batched(expert[:, :, 7:])], axis=2
    )
    checks.append(("expert_batched_parts", np.max(np.abs(expert_actual - expert_ref))))

    rank2 = rng.normal(size=(17, 8)).astype(np.float32)
    rank2_shards = np.array_split(rank2, 4, axis=0)
    rank2_gathered = np.concatenate(rank2_shards, axis=0)
    checks.append(("rank2 Shard(0)->Replicate gather", np.max(np.abs(rank2 - rank2_gathered))))

    expert_shards = np.array_split(expert, 4, axis=0)
    local_updates = [split_update(shard, "expert_batched_parts", expert_spec) for shard in expert_shards]
    expert_resharded = np.concatenate(local_updates, axis=0)
    checks.append(("expert Shard(0) local-vs-global", np.max(np.abs(expert_actual - expert_resharded))))

    print("\nSPLIT/REGROUP AND SHARD(0) EQUIVALENCE")
    for name, error in checks:
        print(f"{name:42} max_abs_error={float(error):.3e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("hf_models/qwen3_5-10b-a2b-exp4/config.json"),
    )
    parser.add_argument(
        "--data-parallel-size",
        "--world-size",
        dest="data_parallel_size",
        type=int,
        default=8,
        help=(
            "effective data-parallel size used only for global-batch arithmetic; "
            "--world-size is retained as a compatibility alias"
        ),
    )
    parser.add_argument("--full-params", action="store_true")
    args = parser.parse_args()

    yaml_bytes = args.yaml.read_bytes()
    yaml_hash = hashlib.sha256(yaml_bytes).hexdigest().upper()
    yaml_values = parse_yaml_scalars(args.yaml)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))["text_config"]
    mtp_layers = int(yaml_values[("model", "mtp_num_layers")])
    params = build_param_inventory(cfg, mtp_layers)

    required = {
        ("training", "optimizer"),
        ("training", "lr"),
        ("training", "lr_min"),
        ("training", "train_iters"),
        ("training", "lr_warmup_ratio"),
        ("training", "gradient_accumulation_steps"),
        ("training", "micro_batch_size"),
        ("training", "weight_decay"),
        ("parallel", "fully_shard_parallel_size"),
        ("parallel", "expert_parallel_size"),
    }
    missing = sorted(required - yaml_values.keys())
    if missing:
        raise KeyError(f"Missing YAML scalars: {missing}")

    micro = int(yaml_values[("training", "micro_batch_size")])
    gas = int(yaml_values[("training", "gradient_accumulation_steps")])
    train_iters = int(yaml_values[("training", "train_iters")])
    global_batch = micro * gas * args.data_parallel_size
    print("F4 PINNED INPUT")
    print(f"yaml={args.yaml.resolve()}")
    print(f"sha256={yaml_hash}")
    print(
        f"data_parallel_size={args.data_parallel_size}, micro_batch={micro}, "
        f"GAS={gas}, global_batch={global_batch}"
    )
    print(f"optimizer steps={train_iters}; microsteps={train_iters * gas}")

    print_lr_trace(yaml_values)
    print_inventory(params, args.full_params)
    print_update_probe(
        float(yaml_values[("training", "lr")]),
        float(yaml_values[("training", "weight_decay")]),
    )
    print_split_and_shard_checks()


if __name__ == "__main__":
    main()
