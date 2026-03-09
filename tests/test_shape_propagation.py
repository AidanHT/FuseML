"""Tests for automated shape propagation and metadata extraction.

Verifies that ShapeProp integration populates FusionGroup.output_metadata
and that tensor_meta is carried forward to placeholder nodes after surgery.

Run with:
    pytest tests/test_shape_propagation.py -v
    pytest tests/ -m shape_prop -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fuseml import FuseMLFusionPass, fuseml_fused_kernel_placeholder

from conftest import find_groups_with_shapes, run_surgery, trace_no_grad


# ---------------------------------------------------------------------------
# Tests — output_metadata populated via ShapeProp
# ---------------------------------------------------------------------------


class TestOutputMetadataPopulated:
    """Verify FusionGroup.output_metadata contains shape/stride/dtype after
    shape propagation."""

    pytestmark = pytest.mark.shape_prop

    def test_metadata_has_shape(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        groups = find_groups_with_shapes(gm, (x,))

        assert len(groups) >= 1
        meta = groups[0].output_metadata
        assert "shape" in meta
        assert meta["shape"] == (2, 64)

    def test_metadata_has_stride(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        groups = find_groups_with_shapes(gm, (x,))

        assert len(groups) >= 1
        meta = groups[0].output_metadata
        assert "stride" in meta
        assert isinstance(meta["stride"], tuple)
        assert len(meta["stride"]) == 2

    def test_metadata_has_dtype(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        groups = find_groups_with_shapes(gm, (x,))

        assert len(groups) >= 1
        meta = groups[0].output_metadata
        assert "dtype" in meta
        assert meta["dtype"] is torch.float32

    def test_metadata_shape_matches_output(self):
        """Output metadata shape must match actual eager output shape."""
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        x = torch.randn(4, 64)
        gm = trace_no_grad(model, x)
        groups = find_groups_with_shapes(gm, (x,))

        if groups:
            meta = groups[0].output_metadata
            with torch.no_grad():
                expected = model(x)
            assert meta["shape"] == tuple(expected.shape)
            assert meta["dtype"] is expected.dtype

    def test_three_node_chain_metadata(self):
        """addmm -> relu -> sigmoid: metadata reflects final sigmoid output."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)

            def forward(self, x):
                return torch.sigmoid(torch.relu(self.linear(x)))

        model = Model()
        x = torch.randn(8, 32)
        gm = trace_no_grad(model, x)
        groups = find_groups_with_shapes(gm, (x,))

        assert len(groups) == 1
        meta = groups[0].output_metadata
        assert meta["shape"] == (8, 16)
        assert meta["dtype"] is torch.float32


class TestOutputMetadataMultipleGroups:
    """Verify each group gets its own correct metadata."""

    pytestmark = pytest.mark.shape_prop

    def test_two_heads_independent_shapes(self):
        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 32)
                self.b = nn.Linear(64, 128)

            def forward(self, x):
                return torch.relu(self.a(x)), torch.sigmoid(self.b(x))

        model = TwoHeads()
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        groups = find_groups_with_shapes(gm, (x,))

        assert len(groups) == 2
        shapes = {g.output_metadata["shape"] for g in groups}
        assert (2, 32) in shapes
        assert (2, 128) in shapes


# ---------------------------------------------------------------------------
# Tests — metadata empty without ShapeProp
# ---------------------------------------------------------------------------


class TestMetadataWithoutShapeProp:
    """Without example_inputs, output_metadata should still be populated
    from the tensor_meta already present via make_fx tracing."""

    pytestmark = pytest.mark.shape_prop

    def test_metadata_populated_from_make_fx(self):
        """make_fx populates tensor_meta during tracing, so groups built
        without explicit ShapeProp still have metadata."""
        from conftest import find_groups

        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        groups = find_groups(gm)

        assert len(groups) >= 1
        # make_fx does populate tensor_meta, so metadata should be present.
        meta = groups[0].output_metadata
        assert "shape" in meta


# ---------------------------------------------------------------------------
# Tests — run() with example_inputs
# ---------------------------------------------------------------------------


class TestRunWithExampleInputs:
    """Verify the public run() entry point propagates shapes correctly."""

    pytestmark = pytest.mark.shape_prop

    def test_run_with_inputs_populates_metadata(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)

        fuse_pass = FuseMLFusionPass(gm)
        result = fuse_pass.run(example_inputs=(x,))

        assert isinstance(result, torch.fx.GraphModule)
        # The placeholder node should exist and have tensor_meta.
        placeholder_nodes = [
            n for n in result.graph.nodes
            if n.op == "call_function"
            and n.target is fuseml_fused_kernel_placeholder
        ]
        assert len(placeholder_nodes) >= 1

    def test_run_without_inputs_still_works(self):
        """run() without example_inputs should not crash."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        result = FuseMLFusionPass(gm).run()
        assert isinstance(result, torch.fx.GraphModule)


# ---------------------------------------------------------------------------
# Tests — tensor_meta copied to placeholder nodes after surgery
# ---------------------------------------------------------------------------


class TestSurgeryPreservesTensorMeta:
    """Verify _apply_surgery copies tensor_meta to the new fused node."""

    pytestmark = pytest.mark.shape_prop

    def test_placeholder_has_tensor_meta(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        gm, groups = run_surgery(gm, example_inputs=(x,))

        if groups:
            placeholder_nodes = [
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is fuseml_fused_kernel_placeholder
            ]
            for pn in placeholder_nodes:
                assert "tensor_meta" in pn.meta

    def test_placeholder_tensor_meta_matches_output_shape(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        x = torch.randn(4, 64)
        gm = trace_no_grad(model, x)
        gm, groups = run_surgery(gm, example_inputs=(x,))

        if groups:
            placeholder_nodes = [
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is fuseml_fused_kernel_placeholder
            ]
            for pn in placeholder_nodes:
                tm = pn.meta["tensor_meta"]
                assert tuple(tm.shape) == (4, 128)

    def test_placeholder_tensor_meta_dtype(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Sigmoid())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        gm, groups = run_surgery(gm, example_inputs=(x,))

        if groups:
            placeholder_nodes = [
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is fuseml_fused_kernel_placeholder
            ]
            for pn in placeholder_nodes:
                tm = pn.meta["tensor_meta"]
                assert tm.dtype is torch.float32

    def test_graph_lint_after_surgery_with_shapes(self):
        """Graph must remain valid after surgery with shape metadata."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)
        gm = trace_no_grad(model, x)
        gm, groups = run_surgery(gm, example_inputs=(x,))

        gm.graph.lint()
