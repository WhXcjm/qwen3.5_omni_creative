# pylint: skip-file
import ast
import copy
import importlib.util
import math
import os
from pathlib import Path
import sys
import types

import pytest

os.environ.setdefault("NON_MEGATRON", "true")
os.environ.setdefault("MINDSPEED_MM_DISABLE_FSDP_OPS_PATCH", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.optimizer.muon import (
    Muon,
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


_EXPECTED_SPLIT_NAMES = {
    "gdn.in_proj_qkv.weight",
    "mla.q_b_proj.weight",
    "mla.kv_a_proj_with_mqa.weight",
    "mla.kv_b_proj.weight",
}


def _split_only_layout_model():
    model = torch.nn.Module()
    model.gdn = torch.nn.Module()
    model.gdn.in_proj_qkv = torch.nn.Linear(5, 12, bias=False)
    model.gdn._muon_layout_specs = {
        "in_proj_qkv.weight": {
            "layout": "contiguous_head_parts",
            "expected_shape": (12, 5),
            "parts": (("q", 2, 2), ("k", 1, 3), ("value", 1, 5)),
        },
    }

    model.mla = torch.nn.Module()
    model.mla.q_b_proj = torch.nn.Linear(5, 8, bias=False)
    model.mla.kv_a_proj_with_mqa = torch.nn.Linear(5, 5, bias=False)
    model.mla.kv_b_proj = torch.nn.Linear(5, 8, bias=False)
    model.mla._muon_layout_specs = {
        "q_b_proj.weight": {
            "layout": "periodic_head_parts",
            "expected_shape": (8, 5),
            "head_count": 2,
            "parts": (("q_nope", 3), ("q_rope", 1)),
        },
        "kv_a_proj_with_mqa.weight": {
            "layout": "contiguous_parts",
            "expected_shape": (5, 5),
            "parts": (("compressed_kv", 3), ("k_rope", 2)),
        },
        "kv_b_proj.weight": {
            "layout": "periodic_head_parts",
            "expected_shape": (8, 5),
            "head_count": 2,
            "parts": (("k_nope", 3), ("value", 1)),
        },
    }

    model.experts = torch.nn.Module()
    model.experts.gate_up_proj = torch.nn.Parameter(torch.empty(3, 4, 6))
    model.experts.down_proj = torch.nn.Parameter(torch.empty(3, 3, 4))
    return model


def _class_muon_layouts(model_path, class_name):
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assignments = []
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "_muon_layout_specs"
            for target in node.targets
        ):
            assignments.append(node.value)

    if not assignments:
        return {}
    assert len(assignments) == 1
    layout_dict = assignments[0]
    assert isinstance(layout_dict, ast.Dict)

    layouts = {}
    for param_name_node, spec_node in zip(layout_dict.keys, layout_dict.values):
        assert isinstance(param_name_node, ast.Constant)
        assert isinstance(param_name_node.value, str)
        assert isinstance(spec_node, ast.Dict)
        spec_fields = {
            key.value: value
            for key, value in zip(spec_node.keys, spec_node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        layout_node = spec_fields["layout"]
        assert isinstance(layout_node, ast.Constant)
        layouts[param_name_node.value] = layout_node.value
    return layouts


def test_qwen35_production_metadata_is_split_only():
    model_path = (
        Path(__file__).resolve().parents[4]
        / "mindspeed_mm"
        / "fsdp"
        / "models"
        / "qwen3_5_moe"
        / "modeling_qwen3_5_moe.py"
    )
    assert _class_muon_layouts(model_path, "Qwen3_5MoeGatedDeltaNet") == {
        "in_proj_qkv.weight": "contiguous_head_parts",
    }
    assert _class_muon_layouts(model_path, "Qwen3_5MoeMLA") == {
        "q_b_proj.weight": "periodic_head_parts",
        "kv_a_proj_with_mqa.weight": "contiguous_parts",
        "kv_b_proj.weight": "periodic_head_parts",
    }
    assert _class_muon_layouts(model_path, "Qwen3_5MoeExperts") == {}
    assert _class_muon_layouts(model_path, "Qwen3_5LatentMoeExperts") == {}


def test_mark_muon_param_groups_selects_only_2d_fused_layouts():
    optimizer_module = _load_optimizer_module()
    model = _split_only_layout_model()
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

    assert routed_names[True] == _EXPECTED_SPLIT_NAMES
    assert routed_names[False] == {"experts.gate_up_proj", "experts.down_proj"}
    assert {name_by_param[param] for param in split_specs} == _EXPECTED_SPLIT_NAMES
    assert all(param.ndim == 2 for param in split_specs)


@pytest.mark.parametrize("builder_name", ["build_optimizer", "build_ep_fsdp2_optimizer"])
def test_muon_builders_inject_only_selected_split_specs(builder_name):
    optimizer_module = _load_optimizer_module()
    model = _split_only_layout_model()
    params = list(model.parameters())

    if builder_name == "build_optimizer":
        parallel_state = types.SimpleNamespace(get_ep_group_size=lambda: 1)
        optimizer_module.get_parallel_state = lambda: parallel_state
        optimizer = optimizer_module.build_optimizer(
            model,
            optimizer_type="muon",
            param_groups=[{"params": params, "weight_decay": 0.1}],
        )
    else:
        multi_optimizer = optimizer_module.build_ep_fsdp2_optimizer(
            model,
            optimizer_type="muon",
            param_groups=[{"params": params, "weight_decay": 0.1}],
        )
        assert set(multi_optimizer.optimizers_dict) == {"non_ep"}
        optimizer = multi_optimizer.optimizers_dict["non_ep"]

    name_by_param = {param: name for name, param in model.named_parameters()}
    assert {
        name_by_param[param] for param in optimizer._muon_split_spec_by_param
    } == _EXPECTED_SPLIT_NAMES

    use_muon_by_name = {}
    for group in optimizer.param_groups:
        assert not any(key.startswith("_muon") for key in group)
        for param in group["params"]:
            use_muon_by_name[name_by_param[param]] = group["use_muon"]

    assert all(use_muon_by_name[name] for name in _EXPECTED_SPLIT_NAMES)
    assert use_muon_by_name["experts.gate_up_proj"] is False
    assert use_muon_by_name["experts.down_proj"] is False


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

    assert optimizer._muon_split_spec_by_param == {param: split_spec}
    state_dict = optimizer.state_dict()
    assert set(state_dict) == {"state", "param_groups"}
    assert state_dict["state"] == {}
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
