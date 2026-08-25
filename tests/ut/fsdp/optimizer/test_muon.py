# pylint: skip-file
import copy
import importlib.util
import logging
import math
import os
from pathlib import Path
import sys
import types

import pytest
import yaml

os.environ.setdefault("NON_MEGATRON", "true")
os.environ.setdefault("MINDSPEED_MM_DISABLE_FSDP_OPS_PATCH", "true")

torch = pytest.importorskip("torch")

import mindspeed_mm.optimizer.muon as muon_module
from mindspeed_mm.optimizer.muon import (
    Muon,
    _dynamic_matched_adamw_rms,
    _zeropower_via_newtonschulz5_split,
    zeropower_via_newtonschulz5,
)


def _fallback_optimizer(params, **kwargs):
    return Muon([{"params": params, "use_muon": False}], **kwargs)


def _load_optimizer_module():
    """Load optimizer.py without executing the optional FSDP ops patch."""
    package_prefix = "_muon_routing_test"
    module_names = [
        package_prefix,
        f"{package_prefix}.fsdp",
        f"{package_prefix}.fsdp.optimizer",
        f"{package_prefix}.fsdp.distributed",
        f"{package_prefix}.optimizer",
    ]
    inserted_names = []

    try:
        for module_name in module_names:
            package = types.ModuleType(module_name)
            package.__path__ = []
            sys.modules[module_name] = package
            inserted_names.append(module_name)

        parallel_state_module_name = (
            f"{package_prefix}.fsdp.distributed.parallel_state"
        )
        parallel_state_module = types.ModuleType(parallel_state_module_name)
        parallel_state_module.get_parallel_state = lambda: None
        sys.modules[parallel_state_module_name] = parallel_state_module
        inserted_names.append(parallel_state_module_name)

        muon_module_name = f"{package_prefix}.optimizer.muon"
        sys.modules[muon_module_name] = sys.modules[Muon.__module__]
        inserted_names.append(muon_module_name)

        optimizer_module_name = f"{package_prefix}.fsdp.optimizer.optimizer"
        optimizer_path = (
            Path(__file__).resolve().parents[4]
            / "mindspeed_mm"
            / "fsdp"
            / "optimizer"
            / "optimizer.py"
        )
        spec = importlib.util.spec_from_file_location(
            optimizer_module_name, optimizer_path
        )
        assert spec is not None and spec.loader is not None
        optimizer_module = importlib.util.module_from_spec(spec)
        sys.modules[optimizer_module_name] = optimizer_module
        inserted_names.append(optimizer_module_name)
        spec.loader.exec_module(optimizer_module)
        return optimizer_module
    finally:
        for module_name in reversed(inserted_names):
            sys.modules.pop(module_name, None)


def test_mark_muon_param_groups_routes_fallbacks_and_covers_all_params_once():
    optimizer_module = _load_optimizer_module()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.embedding = torch.nn.Linear(8, 8, bias=False)
            self.output_layer = torch.nn.Linear(8, 8, bias=False)
            self.embed_tokens = torch.nn.Embedding(4, 8)
            self.lm_head = torch.nn.Linear(8, 4, bias=False)
            self.router = torch.nn.Linear(8, 4, bias=False)
            self.gate = torch.nn.Linear(8, 4, bias=False)
            self.in_proj_a = torch.nn.Linear(8, 2, bias=False)
            self.in_proj_b = torch.nn.Linear(8, 2, bias=False)
            self.shared_expert_gate = torch.nn.Linear(8, 1, bias=False)
            self.norm = torch.nn.LayerNorm(8)

    model = TinyModel()
    original_params = list(model.parameters())
    groups, split_specs = optimizer_module._mark_muon_param_groups(
        model,
        [
            {
                "params": original_params,
                "lr": 0.25,
                "weight_decay": 0.01,
                "group_name": "test",
            }
        ],
    )
    assert split_specs == {}

    name_by_param = {param: name for name, param in model.named_parameters()}
    routed_names = {True: set(), False: set()}
    for group in groups:
        routed_names[group["use_muon"]].update(
            name_by_param[param] for param in group["params"]
        )

    assert {
        "proj.weight",
        "router.weight",
        "gate.weight",
        "in_proj_a.weight",
        "in_proj_b.weight",
        "shared_expert_gate.weight",
    } <= routed_names[True]
    assert {
        "proj.bias",
        "embedding.weight",
        "output_layer.weight",
        "embed_tokens.weight",
        "lm_head.weight",
        "norm.weight",
        "norm.bias",
    } <= routed_names[False]

    routed_params = [param for group in groups for param in group["params"]]
    assert len(routed_params) == len(original_params)
    assert len({id(param) for param in routed_params}) == len(original_params)
    assert {id(param) for param in routed_params} == {
        id(param) for param in original_params
    }
    for group in groups:
        assert group["lr"] == 0.25
        assert group["weight_decay"] == 0.01
        assert group["group_name"] == "test"


def _tiny_pattern_model():
    class TinyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = torch.nn.Module()
            self.linear_attn.in_proj_a = torch.nn.Linear(8, 2, bias=False)
            self.linear_attn.in_proj_b = torch.nn.Linear(8, 2, bias=False)
            self.linear_attn.in_proj_z = torch.nn.Linear(8, 8, bias=False)

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList(
        [TinyLayer(), TinyLayer()]
    )
    return model


def _load_repo_yaml(relative_path):
    repo_root = Path(__file__).resolve().parents[4]
    with (repo_root / relative_path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_task6_yaml_keeps_in_proj_a_b_on_muon():
    optimizer_module = _load_optimizer_module()
    config = _load_repo_yaml(
        "scripts_qwen3_5/longTest/longTest_06-new_dyn_lr4e-4_hg1.yaml"
    )
    training = config["training"]

    assert training["muon_head_group_size"] == 1
    assert "muon_adamw_param_patterns" not in training

    model = _tiny_pattern_model()
    forced_adamw_params = optimizer_module._resolve_muon_adamw_params(
        model, training.get("muon_adamw_param_patterns")
    )
    groups, _ = optimizer_module._mark_muon_param_groups(
        model,
        [{"params": list(model.parameters())}],
        forced_adamw_params,
    )

    assert forced_adamw_params == set()
    name_by_param = {param: name for name, param in model.named_parameters()}
    routed_names = {True: set(), False: set()}
    for group in groups:
        routed_names[group["use_muon"]].update(
            name_by_param[param] for param in group["params"]
        )

    expected_muon_names = {
        f"model.language_model.layers.{layer}.linear_attn.{projection}.weight"
        for layer in range(2)
        for projection in ("in_proj_a", "in_proj_b")
    }
    assert expected_muon_names <= routed_names[True]
    assert expected_muon_names.isdisjoint(routed_names[False])


def test_task6_yaml_matches_t2_dynamic_training_semantics():
    task6 = _load_repo_yaml(
        "scripts_qwen3_5/longTest/longTest_06-new_dyn_lr4e-4_hg1.yaml"
    )
    t2_dynamic = _load_repo_yaml(
        "scripts_qwen3_5/configs/qwen3_5_10B-A2B_exp4_config_dynamic_rms.yaml"
    )

    # Long tests intentionally disable checkpoint writes.  Apart from that
    # operational setting and the output path, every parsed field must match.
    assert t2_dynamic["training"]["save_interval"] == 1000
    assert task6["training"]["save_interval"] == 99999
    for config in (task6, t2_dynamic):
        config["training"].pop("save")
        config["training"].pop("save_interval")

    assert task6 == t2_dynamic


def test_muon_adamw_patterns_use_full_names_and_preserve_group_metadata():
    optimizer_module = _load_optimizer_module()
    model = _tiny_pattern_model()
    patterns = [
        "model.language_model.layers.{*}.linear_attn.in_proj_a.weight",
        "model.language_model.layers.{*}.linear_attn.in_proj_b.weight",
    ]

    forced_adamw_params = optimizer_module._resolve_muon_adamw_params(
        model, patterns
    )
    groups, split_specs = optimizer_module._mark_muon_param_groups(
        model,
        [{"params": list(model.parameters()), "lr": 0.25, "group_name": "test"}],
        forced_adamw_params,
    )

    assert len(forced_adamw_params) == 4
    assert split_specs == {}
    name_by_param = {param: name for name, param in model.named_parameters()}
    routed_names = {True: set(), False: set()}
    for group in groups:
        routed_names[group["use_muon"]].update(
            name_by_param[param] for param in group["params"]
        )
        assert group["lr"] == 0.25
        assert group["group_name"] == "test"
        assert "muon_adamw_param_patterns" not in group

    assert routed_names[False] == {
        f"model.language_model.layers.{layer}.linear_attn.{projection}.weight"
        for layer in range(2)
        for projection in ("in_proj_a", "in_proj_b")
    }
    assert routed_names[True] == {
        f"model.language_model.layers.{layer}.linear_attn.in_proj_z.weight"
        for layer in range(2)
    }


def test_muon_adamw_patterns_fail_fast_on_invalid_or_missing_matches():
    optimizer_module = _load_optimizer_module()
    model = _tiny_pattern_model()

    with pytest.raises(ValueError, match="matched no trainable parameters"):
        optimizer_module._resolve_muon_adamw_params(
            model,
            ["model.language_model.layers.{*}.linear_attn.missing.weight"],
        )
    with pytest.raises(ValueError, match="only a complete '\\{\\*\\}' segment"):
        optimizer_module._resolve_muon_adamw_params(
            model,
            ["model.language_model.layers.*.linear_attn.in_proj_a.weight"],
        )
    with pytest.raises(ValueError, match="requires optimizer='muon'"):
        optimizer_module.build_optimizer(
            model,
            optimizer_type="adamw",
            muon_adamw_param_patterns=[
                "model.language_model.layers.{*}.linear_attn.in_proj_a.weight"
            ],
        )


def test_muon_adamw_patterns_reject_overlapping_rules():
    optimizer_module = _load_optimizer_module()
    model = _tiny_pattern_model()

    with pytest.raises(ValueError, match="matched more than once"):
        optimizer_module._resolve_muon_adamw_params(
            model,
            [
                "model.language_model.layers.{*}.linear_attn.in_proj_a.weight",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            ],
        )


def test_forced_adamw_param_warns_when_split_spec_is_ignored(caplog):
    optimizer_module = _load_optimizer_module()

    class LayoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fused = torch.nn.Linear(4, 6, bias=False)
            self._muon_layout_specs = {
                "fused.weight": {
                    "layout": "contiguous_parts",
                    "expected_shape": (6, 4),
                    "parts": (("left", 2), ("right", 4)),
                }
            }

    model = LayoutModel()
    with caplog.at_level(logging.WARNING, logger=optimizer_module.__name__):
        marked_groups, split_specs = optimizer_module._mark_muon_param_groups(
            model,
            [{"params": list(model.parameters())}],
            {model.fused.weight},
        )

    assert split_specs == {}
    assert len(marked_groups) == 1
    assert marked_groups[0]["params"] == [model.fused.weight]
    assert marked_groups[0]["use_muon"] is False
    assert (
        "Ignoring Muon split metadata for fused.weight because it is forced to AdamW"
        in caplog.text
    )


def test_mark_muon_param_groups_collects_parameter_layouts_and_opts_in_experts():
    optimizer_module = _load_optimizer_module()

    class LayoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fused = torch.nn.Linear(4, 6, bias=False)
            self.expert_gate_up = torch.nn.Parameter(torch.empty(2, 4, 6))
            self.unannotated_rank3 = torch.nn.Parameter(torch.empty(2, 4, 6))
            self._muon_layout_specs = {
                "fused.weight": {
                    "layout": "contiguous_parts",
                    "expected_shape": (6, 4),
                    "parts": (("first", 4), ("second", 2)),
                },
                "expert_gate_up": {
                    "layout": "expert_batched_parts",
                    "expected_shape": (2, 4, 6),
                    "parts": (("gate", 3), ("up", 3)),
                },
            }

    model = LayoutModel()
    groups, split_specs = optimizer_module._mark_muon_param_groups(
        model, [{"params": list(model.parameters()), "weight_decay": 0.1}]
    )

    name_by_param = {param: name for name, param in model.named_parameters()}
    routed_names = {True: set(), False: set()}
    for group in groups:
        routed_names[group["use_muon"]].update(
            name_by_param[param] for param in group["params"]
        )
        assert not any(key.startswith("_muon") for key in group)

    assert routed_names[True] == {"expert_gate_up", "fused.weight"}
    assert routed_names[False] == {"unannotated_rank3"}
    assert {name_by_param[param] for param in split_specs} == {
        "expert_gate_up",
        "fused.weight",
    }


def test_muon_split_specs_are_copied_and_not_serialized():
    param = torch.nn.Parameter(torch.zeros(6, 4))
    optimizer = Muon([{"params": [param], "use_muon": True}])
    split_spec = {
        "layout": "contiguous_parts",
        "expected_shape": (6, 4),
        "parts": (("first", 4), ("second", 2)),
    }
    split_specs = {param: split_spec}

    optimizer._set_muon_split_specs(split_specs)
    split_specs.clear()

    assert optimizer._muon_head_group_size == 1
    assert optimizer._muon_split_spec_by_param == {param: split_spec}
    state_dict = optimizer.state_dict()
    assert all(
        not key.startswith("_muon")
        for group in state_dict["param_groups"]
        for key in group
    )


def _scaled_reference(matrix, steps, matched_adamw_rms):
    return zeropower_via_newtonschulz5(matrix, steps) * (
        matched_adamw_rms * math.sqrt(max(matrix.shape[-2:]))
    )


def test_contiguous_head_parts_match_independent_per_head_reference():
    torch.manual_seed(1)
    matrix = torch.randn(12, 5, dtype=torch.float64)
    split_spec = {
        "layout": "contiguous_head_parts",
        "expected_shape": (12, 5),
        "parts": (("q", 2, 2), ("k", 1, 3), ("value", 1, 5)),
    }
    steps = 2
    matched_adamw_rms = 0.2

    actual = _zeropower_via_newtonschulz5_split(
        matrix, split_spec, steps, matched_adamw_rms
    )
    expected_parts = []
    offset = 0
    for _, head_count, head_dim in split_spec["parts"]:
        for _ in range(head_count):
            head = matrix[offset: offset + head_dim]
            expected_parts.append(
                _scaled_reference(head, steps, matched_adamw_rms)
            )
            offset += head_dim
    expected = torch.cat(expected_parts, dim=0)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_contiguous_head_parts_support_column_head_axis():
    torch.manual_seed(2)
    matrix = torch.randn(5, 12, dtype=torch.float64)
    split_spec = {
        "layout": "contiguous_head_parts",
        "expected_shape": (5, 12),
        "head_axis": 1,
        "parts": (("value", 3, 4),),
    }

    actual = _zeropower_via_newtonschulz5_split(matrix, split_spec, 2, 0.2)
    expected = torch.cat(
        [
            _scaled_reference(head, 2, 0.2)
            for head in torch.split(matrix, 4, dim=1)
        ],
        dim=1,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("matrix_shape", "head_axis", "head_count", "head_dim"),
    (
        pytest.param((32, 5), 0, 16, 2, id="gdn-in-proj-z-row-heads"),
        pytest.param((5, 32), 1, 16, 2, id="gdn-out-proj-column-heads"),
        pytest.param((5, 24), 1, 8, 3, id="mla-o-proj-column-heads"),
    ),
)
def test_task6_strict_per_head_layouts_match_independent_reference(
    matrix_shape, head_axis, head_count, head_dim
):
    torch.manual_seed(26)
    matrix = torch.randn(*matrix_shape, dtype=torch.float64)
    split_spec = {
        "layout": "contiguous_head_parts",
        "expected_shape": matrix_shape,
        "head_axis": head_axis,
        "parts": (("value", head_count, head_dim),),
    }

    actual = _zeropower_via_newtonschulz5_split(
        matrix, split_spec, 2, 0.2, head_group_size=1
    )
    expected = torch.cat(
        [
            _scaled_reference(head, 2, 0.2)
            for head in torch.split(matrix, head_dim, dim=head_axis)
        ],
        dim=head_axis,
    )

    assert actual.shape == matrix.shape
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("matrix_shape", "head_axis", "head_count", "head_dim", "matrix_dim"),
    (
        pytest.param((32, 5), 0, 16, 2, 5, id="gdn-in-proj-z-row-heads"),
        pytest.param((5, 32), 1, 16, 2, 5, id="gdn-out-proj-column-heads"),
        pytest.param((5, 24), 1, 8, 3, 5, id="mla-o-proj-column-heads"),
    ),
)
def test_task6_hg1_uses_one_independent_ns_batch_per_matrix(
    monkeypatch, matrix_shape, head_axis, head_count, head_dim, matrix_dim
):
    matrix = torch.arange(math.prod(matrix_shape), dtype=torch.float64).reshape(
        matrix_shape
    )
    split_spec = {
        "layout": "contiguous_head_parts",
        "expected_shape": matrix_shape,
        "head_axis": head_axis,
        "parts": (("value", head_count, head_dim),),
    }
    calls = []

    def identity_batched_ns(matrices, steps, matched_adamw_rms):
        calls.append((tuple(matrices.shape), steps, matched_adamw_rms))
        return matrices

    monkeypatch.setattr(
        muon_module, "_scaled_batched_zeropower", identity_batched_ns
    )
    actual = muon_module._zeropower_via_newtonschulz5_split(
        matrix, split_spec, 2, 0.2, head_group_size=1
    )

    assert calls == [((head_count, head_dim, matrix_dim), 2, 0.2)]
    assert actual.shape == matrix.shape
    torch.testing.assert_close(actual, matrix, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(("shape", "head_axis"), [((12, 5), 0), ((5, 12), 1)])
def test_single_contiguous_head_part_hg0_matches_whole_matrix(shape, head_axis):
    torch.manual_seed(3)
    matrix = torch.randn(*shape, dtype=torch.float64)
    split_spec = {
        "layout": "contiguous_head_parts",
        "expected_shape": shape,
        "head_axis": head_axis,
        "parts": (("value", 3, 4),),
    }

    actual = _zeropower_via_newtonschulz5_split(
        matrix, split_spec, 2, 0.2, head_group_size=0
    )
    expected = _scaled_reference(matrix, 2, 0.2)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_periodic_head_parts_match_independent_per_head_part_reference():
    torch.manual_seed(2)
    matrix = torch.randn(8, 5, dtype=torch.float64)
    split_spec = {
        "layout": "periodic_head_parts",
        "expected_shape": (8, 5),
        "head_count": 2,
        "parts": (("nope", 3), ("rope", 1)),
    }
    steps = 2
    matched_adamw_rms = 0.2

    actual = _zeropower_via_newtonschulz5_split(
        matrix, split_spec, steps, matched_adamw_rms
    )
    head_major = matrix.reshape(2, 4, 5)
    expected = torch.empty_like(head_major)
    for head_idx in range(2):
        offset = 0
        for _, part_dim in split_spec["parts"]:
            part = head_major[head_idx, offset: offset + part_dim]
            expected[head_idx, offset: offset + part_dim] = _scaled_reference(
                part, steps, matched_adamw_rms
            )
            offset += part_dim

    torch.testing.assert_close(
        actual, expected.reshape_as(matrix), rtol=1e-12, atol=1e-12
    )


def _grouped_head_reference(matrix, split_spec, head_group_size):
    layout = split_spec["layout"]
    if layout == "contiguous_head_parts":
        head_axis = split_spec.get("head_axis", 0)
        axis_normalized = matrix.movedim(head_axis, 0)
        updates, offset = [], 0
        for _, head_count, head_dim in split_spec["parts"]:
            group_size = head_count if head_group_size == 0 else head_group_size
            row_count = head_count * head_dim
            part = axis_normalized[offset: offset + row_count]
            updates.extend(
                _scaled_reference(group, 2, 0.2)
                for group in torch.split(part, group_size * head_dim, dim=0)
            )
            offset += row_count
        return torch.cat(updates, dim=0).movedim(0, head_axis)

    head_count = split_spec["head_count"]
    group_size = head_count if head_group_size == 0 else head_group_size
    head_major = matrix.reshape(head_count, -1, matrix.shape[1])
    updates, offset = [], 0
    for _, part_dim in split_spec["parts"]:
        part = head_major[:, offset: offset + part_dim]
        groups = part.reshape(head_count // group_size, group_size * part_dim, -1)
        update = torch.stack([_scaled_reference(group, 2, 0.2) for group in groups])
        updates.append(update.reshape(head_count, part_dim, -1))
        offset += part_dim
    return torch.cat(updates, dim=1).reshape_as(matrix)


@pytest.mark.parametrize("head_group_size", [1, 2, 4, 0])
def test_head_layouts_apply_ns_per_consecutive_semantic_group(head_group_size):
    torch.manual_seed(11)
    cases = (
        (
            torch.randn(24, 5, dtype=torch.float64),
            {
                "layout": "contiguous_head_parts",
                "expected_shape": (24, 5),
                "parts": (("q", 4, 2), ("k", 4, 1), ("value", 4, 3)),
            },
        ),
        (
            torch.randn(16, 5, dtype=torch.float64),
            {
                "layout": "periodic_head_parts",
                "expected_shape": (16, 5),
                "head_count": 4,
                "parts": (("nope", 3), ("rope", 1)),
            },
        ),
        (
            torch.randn(5, 16, dtype=torch.float64),
            {
                "layout": "contiguous_head_parts",
                "expected_shape": (5, 16),
                "head_axis": 1,
                "parts": (("value", 4, 4),),
            },
        ),
    )
    for matrix, split_spec in cases:
        actual = _zeropower_via_newtonschulz5_split(
            matrix, split_spec, 2, 0.2, head_group_size=head_group_size
        )
        expected = _grouped_head_reference(matrix, split_spec, head_group_size)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        if head_group_size == 0:
            explicit_all = _zeropower_via_newtonschulz5_split(
                matrix, split_spec, 2, 0.2, head_group_size=4
            )
            torch.testing.assert_close(actual, explicit_all, rtol=0.0, atol=0.0)


def test_contiguous_head_parts_validates_head_axis_and_coverage():
    param = torch.nn.Parameter(torch.zeros(5, 12))
    optimizer = Muon([{"params": [param], "use_muon": True}])
    split_spec = {
        "layout": "contiguous_head_parts",
        "expected_shape": (5, 12),
        "head_axis": 2,
        "parts": (("value", 3, 4),),
    }

    with pytest.raises(ValueError, match="Invalid contiguous_head_parts metadata"):
        optimizer._set_muon_split_specs({param: split_spec})

    split_spec["head_axis"] = 1
    split_spec["parts"] = (("value", 2, 4),)
    with pytest.raises(ValueError, match="cover 8 values, expected 12"):
        optimizer._set_muon_split_specs({param: split_spec})


def test_head_group_size_must_divide_head_count():
    matrix = torch.randn(16, 5)
    split_spec = {
        "layout": "periodic_head_parts",
        "expected_shape": (16, 5),
        "head_count": 4,
        "parts": (("nope", 3), ("rope", 1)),
    }
    with pytest.raises(ValueError, match="muon_head_group_size=3.*head_count=4"):
        _zeropower_via_newtonschulz5_split(
            matrix, split_spec, 2, 0.2, head_group_size=3
        )


def test_contiguous_parts_match_independent_semantic_reference():
    torch.manual_seed(3)
    matrix = torch.randn(5, 4, dtype=torch.float64)
    split_spec = {
        "layout": "contiguous_parts",
        "expected_shape": (5, 4),
        "parts": (("compressed", 3), ("rope", 2)),
    }
    steps = 2
    matched_adamw_rms = 0.2

    actual = _zeropower_via_newtonschulz5_split(
        matrix, split_spec, steps, matched_adamw_rms
    )
    expected = torch.cat(
        [
            _scaled_reference(part, steps, matched_adamw_rms)
            for part in torch.split(matrix, [3, 2], dim=0)
        ],
        dim=0,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for head_group_size in (2, 4, 0):
        unchanged = _zeropower_via_newtonschulz5_split(
            matrix, split_spec, steps, matched_adamw_rms, head_group_size
        )
        torch.testing.assert_close(unchanged, actual, rtol=0.0, atol=0.0)


def test_expert_batched_parts_match_independent_per_expert_reference():
    torch.manual_seed(4)
    matrix = torch.randn(3, 4, 6, dtype=torch.float64)
    split_spec = {
        "layout": "expert_batched_parts",
        "expected_shape": (3, 4, 6),
        "parts": (("gate", 3), ("up", 3)),
    }
    steps = 2
    matched_adamw_rms = 0.2

    actual = _zeropower_via_newtonschulz5_split(
        matrix, split_spec, steps, matched_adamw_rms
    )
    expected = torch.empty_like(matrix)
    for expert_idx in range(matrix.shape[0]):
        for part_idx, part in enumerate(torch.split(matrix[expert_idx], 3, dim=1)):
            start = part_idx * 3
            expected[expert_idx, :, start: start + 3] = _scaled_reference(
                part, steps, matched_adamw_rms
            )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for head_group_size in (2, 4, 0):
        unchanged = _zeropower_via_newtonschulz5_split(
            matrix, split_spec, steps, matched_adamw_rms, head_group_size
        )
        torch.testing.assert_close(unchanged, actual, rtol=0.0, atol=0.0)


def test_build_optimizer_uses_runtime_policy_without_state_dict_pollution():
    optimizer_module = _load_optimizer_module()
    optimizer_module.get_parallel_state = lambda: types.SimpleNamespace(
        get_ep_group_size=lambda: 1
    )

    class LayoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fused = torch.nn.Linear(5, 16, bias=False)
            self._muon_layout_specs = {
                "fused.weight": {
                    "layout": "periodic_head_parts",
                    "expected_shape": (16, 5),
                    "head_count": 4,
                    "parts": (("nope", 3), ("rope", 1)),
                }
            }

    torch.manual_seed(14)
    model = LayoutModel()
    param = model.fused.weight
    initial = param.detach().clone()
    grad = torch.randn_like(param)
    lr = 0.05
    optimizer = optimizer_module.build_optimizer(
        model,
        lr=lr,
        weight_decay=0.0,
        optimizer_type="muon",
        fused=False,
        matched_adamw_rms=0.2,
        muon_momentum=0.0,
        ns_steps=2,
        muon_head_group_size=2,
    )
    expected_update = _zeropower_via_newtonschulz5_split(
        grad.to(torch.bfloat16),
        model._muon_layout_specs["fused.weight"],
        2,
        0.2,
        head_group_size=2,
    )
    param.grad = grad
    optimizer.step()

    assert optimizer._muon_head_group_size == 2
    torch.testing.assert_close(param, initial.add(expected_update, alpha=-lr))
    assert all("muon_head_group_size" not in group for group in optimizer.param_groups)
    assert all(
        "muon_head_group_size" not in group
        for group in optimizer.state_dict()["param_groups"]
    )


def test_muon_without_split_spec_preserves_whole_matrix_update():
    initial = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    grad = torch.tensor([[0.5, -0.25], [0.125, -0.75]])
    param = torch.nn.Parameter(initial.clone())
    lr = 0.05
    matched_adamw_rms = 0.2
    optimizer = Muon(
        [{"params": [param], "use_muon": True}],
        lr=lr,
        weight_decay=0.0,
        matched_adamw_rms=matched_adamw_rms,
        momentum=0.0,
        nesterov=False,
        ns_steps=2,
    )

    ns_input = grad.to(torch.bfloat16)
    expected_update = zeropower_via_newtonschulz5(ns_input, steps=2)
    expected = initial.clone()
    expected.add_(
        expected_update,
        alpha=-lr * matched_adamw_rms * math.sqrt(max(ns_input.shape)),
    )

    param.grad = grad.clone()
    optimizer.step()

    torch.testing.assert_close(param, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (1, 1.0),
        (2, 0.708085476162),
        (5, 0.452122806989),
        (10, 0.330125962050),
        (18, 0.266873040174),
        (20, 0.259230289798),
        (50, 0.230601153557),
        (177, 0.229415735697),
        (1000, 0.229415733871),
    ],
)
def test_dynamic_matched_adamw_rms_key_steps(step, expected):
    assert _dynamic_matched_adamw_rms(step, 0.9) == pytest.approx(
        expected, rel=0.0, abs=5e-13
    )


def test_dynamic_matched_adamw_rms_uses_per_param_step_and_roundtrips():
    initial = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    grad = torch.tensor([[0.5, -0.25], [0.125, -0.75]])
    param = torch.nn.Parameter(initial.clone())
    lr = 0.05
    optimizer = Muon(
        [{"params": [param], "use_muon": True}],
        lr=lr,
        weight_decay=0.0,
        matched_adamw_rms=9.0,
        dynamic_matched_adamw_rms=True,
        momentum=0.0,
        nesterov=False,
        ns_steps=2,
        adamw_betas=(0.9, 0.999),
    )

    # Like AdamW, a parameter's clock advances only when that parameter has a gradient.
    optimizer.step()
    assert param not in optimizer.state

    param.grad = grad.clone()
    optimizer.step()
    assert optimizer.state[param]["step"] == 1
    expected = initial.clone()
    expected.add_(
        zeropower_via_newtonschulz5(grad.to(torch.bfloat16), steps=2),
        alpha=(
            -lr
            * _dynamic_matched_adamw_rms(1, 0.9)
            * math.sqrt(max(grad.shape))
        ),
    )
    torch.testing.assert_close(param, expected, rtol=0.0, atol=0.0)

    state_dict = copy.deepcopy(optimizer.state_dict())
    restored = torch.nn.Parameter(param.detach().clone())
    restored_optimizer = Muon(
        [{"params": [restored], "use_muon": True}],
        lr=lr,
        weight_decay=0.0,
        matched_adamw_rms=9.0,
        dynamic_matched_adamw_rms=True,
        momentum=0.0,
        nesterov=False,
        ns_steps=2,
        adamw_betas=(0.8, 0.999),
    )
    restored_optimizer.load_state_dict(state_dict)
    assert restored_optimizer.param_groups[0]["adamw_betas"] == (0.9, 0.999)
    assert restored_optimizer.state[restored]["step"] == 1

    next_grad = grad.flip(0)
    param.grad = next_grad.clone()
    restored.grad = next_grad.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(param, restored, rtol=0.0, atol=0.0)
    assert optimizer.state[param]["step"] == 2
    assert restored_optimizer.state[restored]["step"] == 2


def test_fixed_matched_adamw_rms_keeps_optimizer_state_unchanged():
    param = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = Muon([{"params": [param], "use_muon": True}])

    param.grad = torch.ones_like(param)
    optimizer.step()
    assert set(optimizer.state[param]) == {"muon_buffer"}


def test_dynamic_matched_adamw_rms_multi_optimizer_dcp_roundtrip():
    optimizer_module = _load_optimizer_module()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ep = torch.nn.Parameter(torch.randn(2, 2))
            self.non_ep = torch.nn.Parameter(torch.randn(2, 2))

    def make_multi_optimizer(model):
        optimizers = {
            name: Muon(
                [{"params": [getattr(model, name)], "use_muon": True}],
                dynamic_matched_adamw_rms=True,
                adamw_betas=(0.9, 0.999),
                momentum=0.0,
                nesterov=False,
                weight_decay=0.0,
            )
            for name in ("ep", "non_ep")
        }
        return optimizer_module.MultiOptimizer(
            model, optimizers, key_names=["ep", "non_ep"]
        )

    torch.manual_seed(15)
    original_model = TinyModel()
    original_optimizer = make_multi_optimizer(original_model)
    for param in original_model.parameters():
        param.grad = torch.ones_like(param)
    original_optimizer.step()

    dcp_state = copy.deepcopy(original_optimizer.state_dict())
    restored_model = TinyModel()
    restored_model.load_state_dict(original_model.state_dict())
    restored_optimizer = make_multi_optimizer(restored_model)
    restored_optimizer.load_state_dict(dcp_state)

    for optimizer in restored_optimizer.optimizers_dict.values():
        param = optimizer.param_groups[0]["params"][0]
        assert optimizer.state[param]["step"] == 1

    for original, restored in zip(
        original_model.parameters(), restored_model.parameters()
    ):
        next_grad = torch.arange(4, dtype=original.dtype).reshape_as(original)
        original.grad = next_grad.clone()
        restored.grad = next_grad.clone()
    original_optimizer.step()
    restored_optimizer.step()
    for original, restored in zip(
        original_model.parameters(), restored_model.parameters()
    ):
        torch.testing.assert_close(original, restored, rtol=0.0, atol=0.0)


def test_split_muon_state_roundtrip_keeps_runtime_metadata():
    split_spec = {
        "layout": "contiguous_parts",
        "expected_shape": (4, 4),
        "parts": (("first", 2), ("second", 2)),
    }
    original = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    original_optimizer = Muon(
        [{"params": [original], "use_muon": True}],
        lr=0.01,
        weight_decay=0.0,
        ns_steps=2,
    )
    original_optimizer._set_muon_split_specs({original: split_spec})
    original.grad = torch.linspace(-0.5, 0.5, 16).reshape(4, 4)
    original_optimizer.step()

    state_dict = copy.deepcopy(original_optimizer.state_dict())
    restored = torch.nn.Parameter(original.detach().clone())
    restored_optimizer = Muon(
        [{"params": [restored], "use_muon": True}],
        lr=0.01,
        weight_decay=0.0,
        ns_steps=2,
    )
    restored_optimizer._set_muon_split_specs({restored: split_spec})
    restored_optimizer.load_state_dict(state_dict)

    assert restored_optimizer._muon_split_spec_by_param == {restored: split_spec}
    assert set(restored_optimizer.state[restored]) == {"muon_buffer"}

    next_grad = torch.linspace(0.25, -0.25, 16).reshape(4, 4)
    original.grad = next_grad.clone()
    restored.grad = next_grad.clone()
    original_optimizer.step()
    restored_optimizer.step()

    torch.testing.assert_close(original, restored, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        original_optimizer.state[original]["muon_buffer"],
        restored_optimizer.state[restored]["muon_buffer"],
        rtol=0.0,
        atol=0.0,
    )


def test_adamw_fallback_handles_tiny_gradient():
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    lr = 0.1
    grad = 1e-8
    eps = 1e-8
    optimizer = _fallback_optimizer(
        [param],
        lr=lr,
        weight_decay=0.0,
        adamw_betas=(0.9, 0.999),
        adamw_eps=eps,
    )

    param.grad = torch.full_like(param, grad)
    optimizer.step()

    expected_update = lr * grad / (abs(grad) + eps)
    torch.testing.assert_close(
        param.detach(),
        torch.tensor([1.0 - expected_update], dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_adamw_fallback_tracks_step_per_parameter():
    first = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = _fallback_optimizer([first, second], lr=0.01, weight_decay=0.0)

    first.grad = torch.ones_like(first)
    second.grad = None
    optimizer.step()
    assert optimizer.state[first]["step"].item() == 1
    assert second not in optimizer.state

    first.grad = None
    second.grad = torch.ones_like(second)
    optimizer.step()
    assert optimizer.state[first]["step"].item() == 1
    assert optimizer.state[second]["step"].item() == 1

    first.grad = torch.zeros_like(first)
    second.grad = None
    optimizer.step()
    assert optimizer.state[first]["step"].item() == 2
    assert optimizer.state[second]["step"].item() == 1
    assert "step" not in optimizer.param_groups[0]


def test_step_accepts_closure_none_and_returns_closure_loss():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = _fallback_optimizer([param])

    assert optimizer.step(closure=None) is None

    closure_saw_grad_enabled = False

    def closure():
        nonlocal closure_saw_grad_enabled
        closure_saw_grad_enabled = torch.is_grad_enabled()
        return param.square().sum()

    loss = optimizer.step(closure=closure)
    assert closure_saw_grad_enabled
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_state_dict_roundtrip_continues_identically():
    original = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    original_optimizer = _fallback_optimizer(
        [original],
        lr=0.01,
        weight_decay=0.1,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
    )
    original.grad = torch.tensor([0.25, -0.5], dtype=torch.float64)
    original_optimizer.step()

    state_dict = copy.deepcopy(original_optimizer.state_dict())
    restored = torch.nn.Parameter(original.detach().clone())
    restored_optimizer = _fallback_optimizer([restored])
    restored_optimizer.load_state_dict(state_dict)

    original_state = original_optimizer.state[original]
    restored_state = restored_optimizer.state[restored]
    assert set(original_state) == {"step", "exp_avg", "exp_avg_sq"}
    assert set(restored_state) == {"step", "exp_avg", "exp_avg_sq"}
    assert original_state["step"].device.type == "cpu"
    assert original_state["step"].ndim == 0
    assert "step" not in original_optimizer.param_groups[0]

    next_grad = torch.tensor([-0.1, 0.2], dtype=torch.float64)
    original.grad = next_grad.clone()
    restored.grad = next_grad.clone()
    original_optimizer.step()
    restored_optimizer.step()

    torch.testing.assert_close(original, restored, rtol=0.0, atol=0.0)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            original_optimizer.state[original][key],
            restored_optimizer.state[restored][key],
            rtol=0.0,
            atol=0.0,
        )
