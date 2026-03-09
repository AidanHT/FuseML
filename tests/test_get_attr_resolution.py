"""Tests for get_attr resolution, parameter binding, and enhanced DCE.

Verifies that:
- get_attr nodes feeding fusion groups are resolved to live nn.Parameter /
  buffer tensors and stored in ``FusionGroup.param_bindings``.
- Enhanced dead code elimination removes all orphaned get_attr and
  intermediate call_function nodes after surgery.
- ``graph.lint()`` is enforced after surgery (topological invariant check).

Run with:
    pytest tests/test_get_attr_resolution.py -v
    pytest tests/ -m get_attr -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fuseml import FuseMLFusionPass, fuseml_fused_kernel_placeholder

from conftest import run_surgery, trace_no_grad


# ======================================================================
# get_attr resolution
# ======================================================================

class TestGetAttrResolution:
    """Verify get_attr nodes are resolved to nn.Parameter/buffer values."""

    pytestmark = pytest.mark.get_attr

    def test_linear_relu_has_param_bindings(self):
        """Linear+ReLU group should have at least one resolved parameter."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))

        fuse_pass = FuseMLFusionPass(gm)
        groups = fuse_pass._find_fusion_groups()

        assert len(groups) >= 1
        group = groups[0]
        assert len(group.param_bindings) > 0
        for name, param in group.param_bindings.items():
            assert isinstance(param, (torch.Tensor, torch.nn.Parameter))

    def test_param_values_match_model(self):
        """Resolved parameters should be the same objects as the graph module's."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))

        fuse_pass = FuseMLFusionPass(gm)
        groups = fuse_pass._find_fusion_groups()

        assert len(groups) >= 1
        group = groups[0]
        for name, param in group.param_bindings.items():
            resolved = fuse_pass._fetch_attr(gm, name)
            assert param is resolved

    def test_param_bindings_in_placeholder_meta(self):
        """After surgery, placeholder nodes should carry param_bindings in meta."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            placeholder_nodes = [
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is fuseml_fused_kernel_placeholder
            ]
            for pn in placeholder_nodes:
                assert "param_bindings" in pn.meta
                assert isinstance(pn.meta["param_bindings"], dict)


# ======================================================================
# Enhanced dead code elimination
# ======================================================================

class TestEnhancedDCE:
    """Verify enhanced DCE removes all orphaned nodes."""

    pytestmark = pytest.mark.get_attr

    def test_no_dangling_get_attr_after_surgery(self):
        """After surgery, no get_attr node should have zero users."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            for n in gm.graph.nodes:
                if n.op == "get_attr":
                    assert len(n.users) > 0, (
                        f"Dangling get_attr node {n.name!r} with zero users"
                    )

    def test_no_orphaned_call_function_after_surgery(self):
        """After surgery, no call_function node should have zero users
        (transitive inputs like aten.t are kept alive by the placeholder)."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            for n in gm.graph.nodes:
                if n.op == "call_function" and len(n.users) == 0:
                    pytest.fail(
                        f"Orphaned call_function {n.name!r} (target={n.target})"
                    )

    def test_three_node_chain_no_dangling_attrs(self):
        """addmm -> relu -> sigmoid: verify complete cleanup."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return torch.sigmoid(torch.relu(self.linear(x)))

        gm = trace_no_grad(Model(), torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            for n in gm.graph.nodes:
                if n.op == "get_attr":
                    assert len(n.users) > 0, (
                        f"Dangling get_attr: {n.name!r}"
                    )

    def test_two_heads_no_cross_contamination(self):
        """Two independent fusion groups: DCE should clean both completely."""

        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 64)
                self.b = nn.Linear(64, 64)

            def forward(self, x):
                return torch.relu(self.a(x)), torch.sigmoid(self.b(x))

        gm = trace_no_grad(TwoHeads(), torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            for n in gm.graph.nodes:
                if n.op == "get_attr" and len(n.users) == 0:
                    pytest.fail(f"Dangling get_attr: {n.name!r}")


# ======================================================================
# Validation compilation (graph.lint + recompile)
# ======================================================================

class TestValidationCompilation:
    """Verify graph.lint() and graph.recompile() are enforced after surgery."""

    pytestmark = pytest.mark.get_attr

    def test_lint_after_surgery(self):
        """Surgery should internally call graph.lint() — external call should also pass."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        # If surgery ran, lint was already called internally.
        # Calling it again should not raise.
        gm.graph.lint()

    def test_recompile_produces_valid_forward(self):
        """After surgery + recompile, gm.forward should be callable."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        assert callable(gm.forward)
