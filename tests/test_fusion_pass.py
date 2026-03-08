"""Tests for FuseMLFusionPass — greedy pattern matcher.

Each test traces a small nn.Module at aten level using ``make_fx`` and then
runs ``FuseMLFusionPass._find_fusion_groups`` to verify the greedy matcher
produces the expected fusion groups.

Run with:
    pytest tests/test_fusion_pass.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx

from fuseml_backend import FuseMLFusionPass, FusionGroup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace_no_grad(model: nn.Module, x: torch.Tensor) -> torch.fx.GraphModule:
    """Trace *model* at aten level with gradients disabled."""
    with torch.no_grad():
        gm = make_fx(lambda inp: model(inp))(x)
    return gm


def _find_groups(gm: torch.fx.GraphModule) -> list[FusionGroup]:
    return FuseMLFusionPass(gm)._find_fusion_groups()


# ---------------------------------------------------------------------------
# Tests — basic fusion
# ---------------------------------------------------------------------------

class TestLinearReLU:
    """Linear -> ReLU should fuse into addmm -> relu."""

    @pytest.fixture()
    def groups(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = _trace_no_grad(model, torch.randn(2, 64))
        return _find_groups(gm)

    def test_one_group_found(self, groups):
        assert len(groups) == 1

    def test_group_length(self, groups):
        assert len(groups[0]) == 2

    def test_base_node_is_addmm(self, groups):
        assert groups[0].base_node.target is torch.ops.aten.addmm.default

    def test_fused_node_is_relu(self, groups):
        assert groups[0].fused_nodes[0].target is torch.ops.aten.relu.default

    def test_output_node_is_relu(self, groups):
        assert groups[0].output_node.target is torch.ops.aten.relu.default

    def test_has_external_inputs(self, groups):
        assert len(groups[0].inputs) >= 2  # at least weight/bias + activation


class TestLinearSigmoid:
    """Linear -> Sigmoid should fuse into addmm -> sigmoid."""

    def test_fuses(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Sigmoid())
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert groups[0].fused_nodes[0].target is torch.ops.aten.sigmoid.default


class TestLongerChain:
    """Linear -> ReLU -> Sigmoid should fuse the full chain."""

    def test_three_node_chain(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return torch.sigmoid(torch.relu(self.linear(x)))

        gm = _trace_no_grad(Model(), torch.randn(2, 64))
        groups = _find_groups(gm)

        assert len(groups) == 1
        assert len(groups[0]) == 3
        assert groups[0].base_node.target is torch.ops.aten.addmm.default
        assert groups[0].output_node.target is torch.ops.aten.sigmoid.default


# ---------------------------------------------------------------------------
# Tests — halting conditions
# ---------------------------------------------------------------------------

class TestBranchingHalts:
    """When addmm has multiple users, no group should be formed."""

    def test_no_fusion_on_branch(self):
        class Branching(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                h = self.linear(x)
                return torch.relu(h) + torch.sigmoid(h)

        gm = _trace_no_grad(Branching(), torch.randn(2, 64))
        groups = _find_groups(gm)
        assert len(groups) == 0


class TestBarrierOpHalts:
    """A barrier op (another addmm) after the first should stop absorption."""

    def test_two_linears_separate(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

        # Neither addmm should form a group because the first feeds into a
        # transpose (for the second linear) and the second has no absorbable
        # successor. Both stay at length 1, so they are filtered out.
        # If by chance one does fuse, it still must not cross the barrier.
        for g in groups:
            targets = [n.target for n in g.all_nodes]
            assert targets.count(torch.ops.aten.addmm.default) == 1


class TestLinearOnly:
    """A standalone Linear (addmm with no pointwise tail) should not fuse."""

    def test_no_group(self):
        model = nn.Sequential(nn.Linear(64, 64))
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# Tests — input tracking
# ---------------------------------------------------------------------------

class TestInputTracking:
    """External inputs must only contain nodes produced outside the group."""

    def test_no_internal_nodes_in_inputs(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

        assert len(groups) == 1
        group = groups[0]
        internal = set(group.all_nodes)

        for inp in group.inputs:
            assert inp not in internal, (
                f"Internal node {inp.name} should not appear in group.inputs"
            )

    def test_inputs_are_fx_nodes(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

        for inp in groups[0].inputs:
            assert isinstance(inp, torch.fx.Node)


# ---------------------------------------------------------------------------
# Tests — consumed set prevents overlapping groups
# ---------------------------------------------------------------------------

class TestNoOverlap:
    """Nodes claimed by one group must not appear in another."""

    def test_disjoint_groups(self):
        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 64)
                self.b = nn.Linear(64, 64)

            def forward(self, x):
                return torch.relu(self.a(x)), torch.sigmoid(self.b(x))

        gm = _trace_no_grad(TwoHeads(), torch.randn(2, 64))
        groups = _find_groups(gm)

        all_nodes = []
        for g in groups:
            all_nodes.extend(g.all_nodes)

        # No node should appear in more than one group.
        assert len(all_nodes) == len(set(all_nodes))


# ---------------------------------------------------------------------------
# Tests — FusionGroup dataclass basics
# ---------------------------------------------------------------------------

class TestFusionGroupDataclass:

    def test_len_single(self):
        """A FusionGroup with only a base_node has length 1."""
        # Use a simple placeholder node to avoid needing a real graph.
        model = nn.Linear(4, 4)
        gm = _trace_no_grad(model, torch.randn(1, 4))
        node = next(
            n for n in gm.graph.nodes if n.op == "call_function"
        )
        group = FusionGroup(base_node=node)
        assert len(group) == 1
        assert group.output_node is node

    def test_all_nodes(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

        if groups:
            g = groups[0]
            assert g.all_nodes == [g.base_node] + g.fused_nodes
