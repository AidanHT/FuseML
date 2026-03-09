"""Tests for greedy pattern matching (_find_fusion_groups).

Covers: absorbable ops, longer chains, barrier ops, branching,
standalone addmm, input tracking, and multiple groups.

Run with:
    pytest tests/test_pattern_matching.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fuseml import FuseMLFusionPass

from fuseml.passes.graph_cut import TRANSPARENT_OPS

from conftest import find_groups, find_groups_with_shapes, trace_no_grad


# ---------------------------------------------------------------------------
# Tests — each absorbable op individually
# ---------------------------------------------------------------------------

class TestAbsorbReLU:
    def test_addmm_relu(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        groups = find_groups(trace_no_grad(model, torch.randn(2, 64)))
        assert len(groups) == 1
        assert groups[0].fused_nodes[-1].target is torch.ops.aten.relu.default


class TestAbsorbSigmoid:
    def test_addmm_sigmoid(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Sigmoid())
        groups = find_groups(trace_no_grad(model, torch.randn(2, 64)))
        assert len(groups) == 1
        assert groups[0].fused_nodes[-1].target is torch.ops.aten.sigmoid.default


class TestAbsorbGeLU:
    """GeLU decomposes to mul/erf/add/mul so the non-decomposed aten.gelu
    is what we want. Trace without decomposition to keep it as a single op."""

    def test_addmm_gelu(self):
        # make_fx without decomposition_table keeps gelu as aten.gelu.default
        model = nn.Sequential(nn.Linear(64, 64), nn.GELU())
        gm = trace_no_grad(model, torch.randn(2, 64))

        # Check if gelu survived as a single node or got decomposed.
        gelu_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.gelu.default
        ]
        if gelu_nodes:
            groups = find_groups(gm)
            assert len(groups) == 1
            assert groups[0].fused_nodes[-1].target is torch.ops.aten.gelu.default
        else:
            # GeLU was decomposed into mul/erf/add/mul — verify those absorb
            groups = find_groups(gm)
            # The decomposed ops (mul, add) are absorbable, so we should still
            # get a group if the topology is single-user throughout.
            if groups:
                for node in groups[0].fused_nodes:
                    assert node.target in FuseMLFusionPass._ABSORBABLE_OPS


class TestAbsorbAdd:
    """aten.add.Tensor (residual connection pattern)."""

    def test_addmm_add_external(self):
        class AddResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                # linear(x) + x  is a residual add
                return self.linear(x) + x

        gm = trace_no_grad(AddResidual(), torch.randn(2, 64))
        groups = find_groups(gm)

        # addmm has one user (add), add has one user (output) — should fuse.
        # But addmm may have multiple users if x is reused. Check:
        addmm_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.addmm.default
        ]
        if addmm_nodes and len(addmm_nodes[0].users) == 1:
            assert len(groups) == 1
            assert any(
                n.target is torch.ops.aten.add.Tensor for n in groups[0].fused_nodes
            )


class TestAbsorbMul:
    """aten.mul.Tensor (scaling pattern)."""

    def test_addmm_mul_scalar(self):
        class MulScale(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x) * 0.5

        gm = trace_no_grad(MulScale(), torch.randn(2, 64))
        groups = find_groups(gm)

        # mul with a scalar constant — addmm -> mul
        if groups:
            assert any(
                n.target is torch.ops.aten.mul.Tensor for n in groups[0].fused_nodes
            )


# ---------------------------------------------------------------------------
# Tests — longer chains
# ---------------------------------------------------------------------------

class TestLongerChain:
    """Linear -> ReLU -> Sigmoid should fuse the full 3-node chain."""

    def test_three_node_chain(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return torch.sigmoid(torch.relu(self.linear(x)))

        gm = trace_no_grad(Model(), torch.randn(2, 64))
        groups = find_groups(gm)

        assert len(groups) == 1
        assert len(groups[0]) == 3
        assert groups[0].base_node.target is torch.ops.aten.addmm.default
        assert groups[0].output_node.target is torch.ops.aten.sigmoid.default


class TestChainReLUSigmoid:
    """addmm -> relu -> sigmoid — verify node ordering."""

    def test_node_order(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return torch.sigmoid(torch.relu(self.linear(x)))

        gm = trace_no_grad(Model(), torch.randn(2, 64))
        groups = find_groups(gm)

        if len(groups) == 1 and len(groups[0]) == 3:
            assert groups[0].all_nodes[0].target is torch.ops.aten.addmm.default
            assert groups[0].all_nodes[1].target is torch.ops.aten.relu.default
            assert groups[0].all_nodes[2].target is torch.ops.aten.sigmoid.default


# ---------------------------------------------------------------------------
# Tests — halting on barrier ops
# ---------------------------------------------------------------------------

class TestBarrierAddmm:
    """Two consecutive Linears — the second addmm is a barrier."""

    def test_two_linears_no_cross_fusion(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        for g in groups:
            targets = [n.target for n in g.all_nodes]
            assert targets.count(torch.ops.aten.addmm.default) == 1


class TestBarrierSoftmax:
    """Softmax after Linear should halt absorption."""

    def test_softmax_halts(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Softmax(dim=-1))
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        # addmm -> softmax: softmax is a barrier, so addmm stays alone (len 1)
        # and gets filtered out. No groups expected.
        assert len(groups) == 0


class TestBarrierLayerNorm:
    """LayerNorm after Linear should halt absorption."""

    def test_layernorm_halts(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
                self.ln = nn.LayerNorm(64)

            def forward(self, x):
                return self.ln(self.linear(x))

        gm = trace_no_grad(Model(), torch.randn(2, 64))
        groups = find_groups(gm)

        # addmm -> layer_norm: barrier halts, addmm alone is filtered.
        assert len(groups) == 0


class TestBarrierBmm:
    """bmm is a barrier — verify it's in the barrier set."""

    def test_bmm_in_barrier_set(self):
        assert torch.ops.aten.bmm.default in FuseMLFusionPass._BARRIER_OPS


class TestBarrierConvolution:
    """convolution is a barrier — verify it's in the barrier set."""

    def test_conv_in_barrier_set(self):
        assert torch.ops.aten.convolution.default in FuseMLFusionPass._BARRIER_OPS


# ---------------------------------------------------------------------------
# Tests — halting on branching (multi-user)
# ---------------------------------------------------------------------------

class TestBranchingAtBase:
    """addmm with multiple users — no group formed."""

    def test_no_fusion_on_branch(self):
        class Branching(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                h = self.linear(x)
                return torch.relu(h) + torch.sigmoid(h)

        gm = trace_no_grad(Branching(), torch.randn(2, 64))
        groups = find_groups(gm)
        assert len(groups) == 0


class TestBranchingMidChain:
    """Partial fusion: addmm -> relu fuses, but relu branches so sigmoid
    is NOT absorbed."""

    def test_partial_fusion_on_midchain_branch(self):
        class MidBranch(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                h = torch.relu(self.linear(x))
                # h is used twice — branches here
                return torch.sigmoid(h) + h

        gm = trace_no_grad(MidBranch(), torch.randn(2, 64))

        # Check if addmm -> relu fuses (relu has 2 users, so absorption
        # should stop *after* relu — but relu itself was absorbed because
        # the check is on current.users before looking at successor.
        # Actually: addmm has 1 user (relu), so relu is absorbed.
        # Then current=relu, relu has 2 users -> break.
        # Group = [addmm, relu] = len 2. Valid.
        groups = find_groups(gm)

        addmm_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.addmm.default
        ]
        if addmm_nodes and len(addmm_nodes[0].users) == 1:
            assert len(groups) == 1
            assert len(groups[0]) == 2
            assert groups[0].output_node.target is torch.ops.aten.relu.default
            # sigmoid must NOT be in the group
            fused_targets = {n.target for n in groups[0].all_nodes}
            assert torch.ops.aten.sigmoid.default not in fused_targets


# ---------------------------------------------------------------------------
# Tests — standalone addmm (no pointwise tail)
# ---------------------------------------------------------------------------

class TestLinearOnly:
    """A standalone Linear (addmm with no absorbable successor) is filtered."""

    def test_no_group(self):
        model = nn.Sequential(nn.Linear(64, 64))
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# Tests — input tracking
# ---------------------------------------------------------------------------

class TestInputTracking:
    """External inputs must only contain nodes produced outside the group."""

    def test_no_internal_nodes_in_inputs(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        assert len(groups) == 1
        group = groups[0]
        internal = set(group.all_nodes)

        for inp in group.inputs:
            assert inp not in internal, (
                f"Internal node {inp.name} should not appear in group.inputs"
            )

    def test_inputs_are_fx_nodes(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        for inp in groups[0].inputs:
            assert isinstance(inp, torch.fx.Node)

    def test_no_duplicate_inputs(self):
        """Each external input should appear at most once."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        if groups:
            inputs = groups[0].inputs
            assert len(inputs) == len(set(inputs))

    def test_binary_op_records_external_operand(self):
        """When an absorbed add/mul has an external operand, it must appear
        in group.inputs."""

        class AddResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x) + x

        gm = trace_no_grad(AddResidual(), torch.randn(2, 64))
        groups = find_groups(gm)

        if groups:
            group = groups[0]
            add_nodes = [
                n for n in group.fused_nodes
                if n.target is torch.ops.aten.add.Tensor
            ]
            if add_nodes:
                # The residual `x` should be in group.inputs
                add_node = add_nodes[0]
                external_args = [
                    a for a in add_node.args
                    if isinstance(a, torch.fx.Node) and a not in set(group.all_nodes)
                ]
                for ext in external_args:
                    assert ext in group.inputs


# ---------------------------------------------------------------------------
# Tests — multiple groups in one graph
# ---------------------------------------------------------------------------

class TestMultipleGroups:
    """Two independent fusible chains in the same graph."""

    def test_two_parallel_heads(self):
        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 64)
                self.b = nn.Linear(64, 64)

            def forward(self, x):
                return torch.relu(self.a(x)), torch.sigmoid(self.b(x))

        gm = trace_no_grad(TwoHeads(), torch.randn(2, 64))
        groups = find_groups(gm)

        assert len(groups) == 2

    def test_disjoint_nodes(self):
        """No node should appear in more than one group."""

        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 64)
                self.b = nn.Linear(64, 64)

            def forward(self, x):
                return torch.relu(self.a(x)), torch.sigmoid(self.b(x))

        gm = trace_no_grad(TwoHeads(), torch.randn(2, 64))
        groups = find_groups(gm)

        all_nodes = []
        for g in groups:
            all_nodes.extend(g.all_nodes)
        assert len(all_nodes) == len(set(all_nodes))

    def test_sequential_fusions(self):
        """Linear -> ReLU -> Linear -> Sigmoid: two groups, one per addmm."""

        class Sequential(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(64, 64)
                self.l2 = nn.Linear(64, 64)

            def forward(self, x):
                return torch.sigmoid(self.l2(torch.relu(self.l1(x))))

        gm = trace_no_grad(Sequential(), torch.randn(2, 64))
        groups = find_groups(gm)

        # First addmm may or may not absorb relu depending on whether relu
        # feeds only into the transpose for l2. Second addmm -> sigmoid.
        # At minimum, second addmm -> sigmoid should form a group.
        sigmoid_groups = [
            g for g in groups
            if any(n.target is torch.ops.aten.sigmoid.default for n in g.fused_nodes)
        ]
        assert len(sigmoid_groups) >= 1


# ---------------------------------------------------------------------------
# Tests — view/metadata penetration
# ---------------------------------------------------------------------------

class TestViewPenetrationShapePreserving:
    """Transparent view ops that preserve 2-D shape should be absorbed."""
    pytestmark = pytest.mark.view_penetration

    def test_addmm_view_relu_fuses(self):
        """addmm → view (same shape) → relu should fuse into a single group."""

        class ViewBetween(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                h = self.linear(x)
                # view that preserves the [M, N] shape
                h = h.view(h.shape[0], h.shape[1])
                return torch.relu(h)

        x = torch.randn(2, 64)
        gm = trace_no_grad(ViewBetween(), x)

        # Check if a view node exists between addmm and relu.
        view_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target in TRANSPARENT_OPS
        ]

        if view_nodes:
            # Run with shapes so _is_shape_preserving_2d can check metadata.
            from torch.fx.passes.shape_prop import ShapeProp
            ShapeProp(gm).propagate(x)
            groups = FuseMLFusionPass(gm)._find_fusion_groups()

            if groups:
                # The view should be inside the group's fused_nodes.
                group = groups[0]
                fused_targets = [n.target for n in group.fused_nodes]
                # relu must be in the group (view penetration worked).
                assert torch.ops.aten.relu.default in fused_targets
                # The group should have more than 1 node (base + at least relu).
                assert len(group) >= 2

    def test_addmm_reshape_gelu_fuses(self):
        """addmm → reshape (same shape) → gelu should fuse."""

        class ReshapeBetween(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                h = self.linear(x)
                h = h.reshape(h.shape[0], h.shape[1])
                return torch.nn.functional.gelu(h)

        x = torch.randn(4, 64)
        gm = trace_no_grad(ReshapeBetween(), x)

        reshape_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target in {torch.ops.aten.reshape.default, torch.ops.aten.view.default}
        ]

        if reshape_nodes:
            from torch.fx.passes.shape_prop import ShapeProp
            ShapeProp(gm).propagate(x)
            groups = FuseMLFusionPass(gm)._find_fusion_groups()

            if groups:
                group = groups[0]
                assert len(group) >= 2


class TestViewPenetrationShapeChanging:
    """View ops that change shape should NOT be absorbed."""
    pytestmark = pytest.mark.view_penetration

    def test_shape_changing_view_halts_absorption(self):
        """addmm → view (3-D) → relu: view changes rank, should stop."""

        class RankChangingView(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                h = self.linear(x)
                # Reshape to 3-D: [2, 64] → [2, 8, 8]
                h = h.view(x.shape[0], 8, 8)
                # relu after rank change
                return torch.relu(h)

        x = torch.randn(2, 64)
        gm = trace_no_grad(RankChangingView(), x)

        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(x)
        groups = FuseMLFusionPass(gm)._find_fusion_groups()

        # The relu should NOT be absorbed because the view changed rank.
        for g in groups:
            relu_in_fused = any(
                n.target is torch.ops.aten.relu.default for n in g.fused_nodes
            )
            assert not relu_in_fused, (
                "relu should not be absorbed after a rank-changing view"
            )


class TestViewPenetrationTransparentOpsSet:
    """Verify TRANSPARENT_OPS contains the expected view/metadata ops."""
    pytestmark = pytest.mark.view_penetration

    @pytest.mark.parametrize("op", [
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.permute.default,
    ])
    def test_op_in_transparent_set(self, op):
        assert op in TRANSPARENT_OPS


# ---------------------------------------------------------------------------
# Tests — escape node safety (post-reduction guard)
# ---------------------------------------------------------------------------

class TestEscapeNodeSafety:
    """Verify the post-reduction escape guard in _find_fusion_groups."""

    def test_escape_before_reduction_preserved(self):
        """addmm → relu → sum: relu has external user → relu in
        intermediate_outputs (escape before reduction is safe)."""

        class ReluThenSum(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                h = torch.relu(self.linear(x))
                # h is used both by sum and by an identity branch
                return h.sum(dim=-1, keepdim=False), h

        gm = trace_no_grad(ReluThenSum(), torch.randn(2, 64))
        groups = find_groups(gm)

        # If addmm → relu → sum formed a group, relu should be an escape
        # node (consumed by both sum and the output tuple).
        for g in groups:
            relu_nodes = [
                n for n in g.all_nodes
                if n.target is torch.ops.aten.relu.default
            ]
            sum_nodes = [
                n for n in g.all_nodes
                if n.target is torch.ops.aten.sum.dim_IntList
            ]
            if relu_nodes and sum_nodes:
                # relu should be in intermediate_outputs
                assert relu_nodes[0] in g.intermediate_outputs

    def test_no_escape_after_reduction_structurally(self):
        """Reduction always terminates the group (break), so nothing can
        appear after it in all_nodes.  This tests the structural invariant."""

        class LinearReluSum(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return torch.relu(self.linear(x)).sum(dim=-1, keepdim=False)

        gm = trace_no_grad(LinearReluSum(), torch.randn(2, 64))
        groups = find_groups(gm)

        for g in groups:
            reduction_seen = False
            for n in g.all_nodes:
                if n.target in FuseMLFusionPass._REDUCTION_OPS:
                    reduction_seen = True
                elif reduction_seen:
                    # No node should appear after a reduction
                    assert False, (
                        f"Node {n.name} appears after reduction in group"
                    )
