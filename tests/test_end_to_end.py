"""End-to-end tests for FuseMLFusionPass.run() and op set completeness.

Run with:
    pytest tests/test_end_to_end.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fuseml import FuseMLFusionPass

from conftest import trace_fn_no_grad, trace_no_grad


# ---------------------------------------------------------------------------
# Tests — run() end-to-end
# ---------------------------------------------------------------------------

class TestRunEndToEnd:
    """Verify FuseMLFusionPass.run() executes without error and returns
    a GraphModule with correct graph surgery applied."""

    def test_run_returns_graph_module(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        result = FuseMLFusionPass(gm).run()
        assert isinstance(result, torch.fx.GraphModule)

    def test_run_no_fusible_ops(self):
        """run() on a graph with no fusible ops should not crash."""

        def identity(x):
            return x + 1.0

        gm = trace_fn_no_grad(identity, torch.randn(2, 64))
        result = FuseMLFusionPass(gm).run()
        assert isinstance(result, torch.fx.GraphModule)

    def test_run_no_fusible_preserves_numerics(self):
        """A graph with no fusible ops still produces correct output."""

        def add_one(x):
            return x + 1.0

        x = torch.randn(2, 64)
        gm = trace_fn_no_grad(add_one, torch.randn(2, 64))
        result_gm = FuseMLFusionPass(gm).run()

        with torch.no_grad():
            assert torch.allclose(result_gm(x), x + 1.0)


# ---------------------------------------------------------------------------
# Tests — barrier/absorbable set completeness
# ---------------------------------------------------------------------------

class TestOpSets:
    """Verify the class-level op sets contain everything we expect."""

    @pytest.mark.parametrize("op", [
        torch.ops.aten._softmax.default,
        torch.ops.aten._log_softmax.default,
        torch.ops.aten.native_layer_norm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.convolution.default,
    ])
    def test_barrier_ops(self, op):
        assert op in FuseMLFusionPass._BARRIER_OPS

    @pytest.mark.parametrize("op", [
        torch.ops.aten.relu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    ])
    def test_absorbable_ops(self, op):
        assert op in FuseMLFusionPass._ABSORBABLE_OPS

    def test_barrier_and_absorbable_disjoint(self):
        """No op should be in both sets."""
        overlap = FuseMLFusionPass._BARRIER_OPS & FuseMLFusionPass._ABSORBABLE_OPS
        assert len(overlap) == 0, f"Overlap: {overlap}"
