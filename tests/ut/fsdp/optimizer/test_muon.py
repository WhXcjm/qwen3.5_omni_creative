# pylint: skip-file
import copy
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest

os.environ.setdefault("NON_MEGATRON", "true")
os.environ.setdefault("MINDSPEED_MM_DISABLE_FSDP_OPS_PATCH", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.optimizer.muon import Muon


def _fallback_optimizer(params, **kwargs):
    return Muon([{"params": params, "use_muon": False}], **kwargs)


def _load_param_group_marker():
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
        return optimizer_module._mark_muon_param_groups
    finally:
        for module_name in reversed(inserted_names):
            sys.modules.pop(module_name, None)


def test_mark_muon_param_groups_routes_fallbacks_and_covers_all_params_once():
    mark_muon_param_groups = _load_param_group_marker()

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
    groups = mark_muon_param_groups(
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
