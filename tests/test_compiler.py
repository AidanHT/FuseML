"""Tests for compiler.py — _node_to_descriptor, _descriptor_from_metadata, and FuseMLCompiler.

Run with:
    pytest tests/test_compiler.py -v
    pytest tests/ -m compiler -v
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from fuseml.codegen.kernel_generator import TensorDescriptor, TritonKernelGenerator
from fuseml.codegen.kernel_launcher import KernelLauncher
from fuseml.compiler import FuseMLCompiler, _descriptor_from_metadata, _node_to_descriptor
from fuseml.registry import SupportedOpsRegistry, build_default_registry

from conftest import trace_no_grad, find_groups_with_shapes, run_surgery


# ---------------------------------------------------------------------------
# Helpers — lightweight FX node stand-ins for _node_to_descriptor tests
# ---------------------------------------------------------------------------

def _make_fx_node(*, name="node", meta=None):
    """Minimal object that quacks like ``torch.fx.Node`` for metadata tests."""
    node = SimpleNamespace(name=name, meta=meta or {})
    return node


def _make_tensor_meta(*, shape, stride, dtype):
    """Mimic a ``TensorMetadata`` namedtuple (has .shape, .stride, .dtype)."""
    return SimpleNamespace(shape=shape, stride=stride, dtype=dtype)


def _make_fake_tensor(*, shape, stride, dtype):
    """Mimic a ``FakeTensor`` (has .shape, callable .stride(), .dtype)."""
    return SimpleNamespace(shape=shape, stride=lambda: stride, dtype=dtype)


# ---------------------------------------------------------------------------
# _node_to_descriptor
# ---------------------------------------------------------------------------

@pytest.mark.compiler
class TestNodeToDescriptor:
    """Verify _node_to_descriptor extracts metadata from FX node objects."""

    def test_tensor_meta_returns_descriptor(self):
        tm = _make_tensor_meta(shape=(128, 64), stride=(64, 1), dtype=torch.float32)
        node = _make_fx_node(name="addmm", meta={"tensor_meta": tm})
        desc = _node_to_descriptor(node)
        assert desc is not None
        assert desc.name == "addmm"
        assert desc.shape == (128, 64)
        assert desc.stride == (64, 1)
        assert desc.dtype == torch.float32

    def test_val_returns_descriptor(self):
        fake = _make_fake_tensor(shape=(4, 256), stride=(256, 1), dtype=torch.float16)
        node = _make_fx_node(name="relu", meta={"val": fake})
        desc = _node_to_descriptor(node)
        assert desc is not None
        assert desc.name == "relu"
        assert desc.shape == (4, 256)
        assert desc.stride == (256, 1)
        assert desc.dtype == torch.float16

    def test_tensor_meta_takes_priority_over_val(self):
        tm = _make_tensor_meta(shape=(10, 20), stride=(20, 1), dtype=torch.float32)
        fake = _make_fake_tensor(shape=(99, 99), stride=(99, 1), dtype=torch.float16)
        node = _make_fx_node(name="n", meta={"tensor_meta": tm, "val": fake})
        desc = _node_to_descriptor(node)
        assert desc.shape == (10, 20)
        assert desc.dtype == torch.float32

    def test_neither_returns_none(self):
        node = _make_fx_node(name="scalar", meta={})
        desc = _node_to_descriptor(node)
        assert desc is None

    def test_sequence_of_tensor_meta_takes_first(self):
        tm0 = _make_tensor_meta(shape=(8, 16), stride=(16, 1), dtype=torch.float32)
        tm1 = _make_tensor_meta(shape=(99, 99), stride=(99, 1), dtype=torch.float16)
        node = _make_fx_node(name="multi", meta={"tensor_meta": [tm0, tm1]})
        desc = _node_to_descriptor(node)
        assert desc is not None
        assert desc.shape == (8, 16)

    def test_empty_sequence_returns_none(self):
        node = _make_fx_node(name="empty", meta={"tensor_meta": []})
        desc = _node_to_descriptor(node)
        assert desc is None

    def test_uses_node_name(self):
        tm = _make_tensor_meta(shape=(2, 3), stride=(3, 1), dtype=torch.float32)
        node = _make_fx_node(name="my_custom_name", meta={"tensor_meta": tm})
        desc = _node_to_descriptor(node)
        assert desc.name == "my_custom_name"


# ---------------------------------------------------------------------------
# _descriptor_from_metadata
# ---------------------------------------------------------------------------

@pytest.mark.compiler
class TestDescriptorFromMetadata:
    """Verify _descriptor_from_metadata builds descriptors from plain dicts."""

    def test_valid_dict(self):
        meta = {"shape": (128, 256), "stride": (256, 1), "dtype": torch.float32}
        desc = _descriptor_from_metadata("out", meta)
        assert desc is not None
        assert desc.name == "out"
        assert desc.shape == (128, 256)
        assert desc.stride == (256, 1)
        assert desc.dtype == torch.float32

    def test_empty_dict_returns_none(self):
        desc = _descriptor_from_metadata("out", {})
        assert desc is None

    def test_missing_shape_returns_none(self):
        desc = _descriptor_from_metadata("out", {"stride": (1,), "dtype": torch.float32})
        assert desc is None

    def test_uses_provided_name(self):
        meta = {"shape": (4,), "stride": (1,), "dtype": torch.float16}
        desc = _descriptor_from_metadata("custom_name", meta)
        assert desc.name == "custom_name"


# ---------------------------------------------------------------------------
# FuseMLCompiler.__init__
# ---------------------------------------------------------------------------

@pytest.mark.compiler
class TestFuseMLCompilerInit:
    """Verify compiler construction."""

    def test_default_registry(self):
        compiler = FuseMLCompiler()
        assert compiler.registry is not None
        assert compiler.registry.is_supported(torch.ops.aten.addmm.default)

    def test_custom_registry(self):
        custom = SupportedOpsRegistry()
        custom.register(torch.ops.aten.relu.default, "elementwise")
        compiler = FuseMLCompiler(registry=custom)
        assert compiler.registry is custom

    def test_fusion_candidates_starts_empty(self):
        compiler = FuseMLCompiler()
        assert compiler.fusion_candidates == []

    def test_has_generator(self):
        compiler = FuseMLCompiler()
        assert isinstance(compiler._generator, TritonKernelGenerator)


# ---------------------------------------------------------------------------
# FuseMLCompiler.capture_and_print_graph
# ---------------------------------------------------------------------------

@pytest.mark.compiler
class TestFuseMLCompilerCaptureAndPrint:
    """Verify graph tagging for observability."""

    def test_tags_addmm_as_candidate(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        compiler = FuseMLCompiler()
        compiler.capture_and_print_graph(gm)
        addmm_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.addmm.default
        ]
        assert len(addmm_nodes) >= 1
        assert addmm_nodes[0].meta.get("fusion_candidate") is True

    def test_tags_relu_as_candidate(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        compiler = FuseMLCompiler()
        compiler.capture_and_print_graph(gm)
        relu_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.relu.default
        ]
        assert len(relu_nodes) >= 1
        assert relu_nodes[0].meta.get("fusion_candidate") is True

    def test_sets_fusion_category(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        compiler = FuseMLCompiler()
        compiler.capture_and_print_graph(gm)
        addmm_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.addmm.default
        )
        assert addmm_node.meta.get("fusion_category") == "linear"

    def test_non_matched_nodes_not_tagged(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        compiler = FuseMLCompiler()
        compiler.capture_and_print_graph(gm)
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        for ph in placeholder_nodes:
            assert ph.meta.get("fusion_candidate", False) is False


# ---------------------------------------------------------------------------
# FuseMLCompiler.__call__ — no-fusion path
# ---------------------------------------------------------------------------

@pytest.mark.compiler
class TestFuseMLCompilerCallNoFusion:
    """Verify __call__ returns a callable when no fusion groups are found."""

    def test_no_fusible_ops_returns_callable(self):
        """Graph of just a ReLU (no addmm trigger) returns the forward callable."""
        gm = trace_no_grad(nn.ReLU(), torch.randn(2, 64))
        compiler = FuseMLCompiler()
        # Use _compile_aten_graph directly — make_fx already produces aten ops.
        result = compiler._compile_aten_graph(gm, [torch.randn(2, 64)])
        assert callable(result)

    def test_empty_registry_returns_forward(self):
        """With an empty registry, no ops match — forward returned unchanged."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        empty_reg = SupportedOpsRegistry()
        compiler = FuseMLCompiler(registry=empty_reg)
        # Use _compile_aten_graph directly — make_fx already produces aten ops.
        result = compiler._compile_aten_graph(gm, [torch.randn(2, 64)])
        assert callable(result)


# ---------------------------------------------------------------------------
# FuseMLCompiler._build_launcher — integration with mock Triton
# ---------------------------------------------------------------------------

@pytest.mark.compiler
class TestFuseMLCompilerBuildLauncher:
    """Verify _build_launcher produces a KernelLauncher from a valid FusionGroup."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        """Inject mock triton so compile_and_bind can exec() the kernel string."""
        mock_tl = types.ModuleType("triton.language")
        mock_tl.constexpr = int
        mock_tl_math = types.ModuleType("triton.language.math")
        mock_tl.math = mock_tl_math

        mock_triton = types.ModuleType("triton")
        mock_triton.jit = lambda fn: fn
        mock_triton.language = mock_tl

        monkeypatch.setitem(sys.modules, "triton", mock_triton)
        monkeypatch.setitem(sys.modules, "triton.language", mock_tl)
        monkeypatch.setitem(sys.modules, "triton.language.math", mock_tl_math)

    def test_returns_launcher_for_valid_group(self):
        """Linear+ReLU group with shape metadata yields a KernelLauncher."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        groups = find_groups_with_shapes(gm, (x,))
        assert len(groups) >= 1

        compiler = FuseMLCompiler()
        launcher = compiler._build_launcher(groups[0])
        assert isinstance(launcher, KernelLauncher)

    def test_returns_none_when_input_lacks_metadata(self):
        """If an input node has no tensor_meta or val, _build_launcher returns None."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        groups = find_groups_with_shapes(gm, (x,))
        assert len(groups) >= 1

        group = groups[0]
        # Strip metadata from one of the input nodes AND clear
        # param_bindings so the get_attr fallback cannot rescue it.
        group.inputs[0].meta.pop("tensor_meta", None)
        group.inputs[0].meta.pop("val", None)
        group.param_bindings.clear()

        compiler = FuseMLCompiler()
        launcher = compiler._build_launcher(group)
        assert launcher is None

    def test_returns_none_when_output_lacks_metadata(self):
        """If output_metadata is empty and output_node has no meta, returns None."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        groups = find_groups_with_shapes(gm, (x,))
        assert len(groups) >= 1

        group = groups[0]
        group.output_metadata = {}
        group.output_node.meta.pop("tensor_meta", None)
        group.output_node.meta.pop("val", None)

        compiler = FuseMLCompiler()
        launcher = compiler._build_launcher(group)
        assert launcher is None
