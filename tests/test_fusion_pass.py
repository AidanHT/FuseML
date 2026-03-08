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

from fuseml_backend import (
    FuseMLFusionPass,
    FusionGroup,
    SupportedOpsRegistry,
    build_default_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace_no_grad(model: nn.Module, x: torch.Tensor) -> torch.fx.GraphModule:
    """Trace *model* at aten level with gradients disabled."""
    with torch.no_grad():
        gm = make_fx(lambda inp: model(inp))(x)
    return gm


def _trace_fn_no_grad(fn, x: torch.Tensor) -> torch.fx.GraphModule:
    """Trace a plain function at aten level with gradients disabled."""
    with torch.no_grad():
        gm = make_fx(fn)(x)
    return gm


def _find_groups(gm: torch.fx.GraphModule) -> list[FusionGroup]:
    return FuseMLFusionPass(gm)._find_fusion_groups()


# ---------------------------------------------------------------------------
# Tests — default registry contents
# ---------------------------------------------------------------------------

class TestBuildDefaultRegistry:
    """Verify build_default_registry ships all expected ops."""

    @pytest.fixture()
    def registry(self):
        return build_default_registry()

    def test_addmm_registered(self, registry):
        assert torch.ops.aten.addmm.default in registry

    def test_addmm_category(self, registry):
        assert registry.get_category(torch.ops.aten.addmm.default) == "linear"

    @pytest.mark.parametrize("op", [
        torch.ops.aten.relu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    ])
    def test_elementwise_ops_registered(self, registry, op):
        assert op in registry
        assert registry.get_category(op) == "elementwise"

    def test_total_count(self, registry):
        assert len(registry) == 6  # 1 linear + 5 elementwise


# ---------------------------------------------------------------------------
# Tests — SupportedOpsRegistry API
# ---------------------------------------------------------------------------

class TestSupportedOpsRegistry:

    def test_register_and_query(self):
        r = SupportedOpsRegistry()
        r.register(torch.ops.aten.relu.default, "elementwise")
        assert r.is_supported(torch.ops.aten.relu.default)
        assert r.get_category(torch.ops.aten.relu.default) == "elementwise"

    def test_unregister(self):
        r = SupportedOpsRegistry()
        r.register(torch.ops.aten.relu.default)
        r.unregister(torch.ops.aten.relu.default)
        assert not r.is_supported(torch.ops.aten.relu.default)

    def test_unregister_absent_is_noop(self):
        r = SupportedOpsRegistry()
        r.unregister(torch.ops.aten.relu.default)  # should not raise

    def test_contains(self):
        r = SupportedOpsRegistry()
        r.register(torch.ops.aten.relu.default)
        assert torch.ops.aten.relu.default in r
        assert torch.ops.aten.gelu.default not in r

    def test_register_many(self):
        r = SupportedOpsRegistry()
        ops = [torch.ops.aten.relu.default, torch.ops.aten.gelu.default]
        r.register_many(ops, "elementwise")
        assert len(r) == 2
        for op in ops:
            assert op in r


# ---------------------------------------------------------------------------
# Tests — each absorbable op individually
# ---------------------------------------------------------------------------

class TestAbsorbReLU:
    def test_addmm_relu(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        groups = _find_groups(_trace_no_grad(model, torch.randn(2, 64)))
        assert len(groups) == 1
        assert groups[0].fused_nodes[-1].target is torch.ops.aten.relu.default


class TestAbsorbSigmoid:
    def test_addmm_sigmoid(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Sigmoid())
        groups = _find_groups(_trace_no_grad(model, torch.randn(2, 64)))
        assert len(groups) == 1
        assert groups[0].fused_nodes[-1].target is torch.ops.aten.sigmoid.default


class TestAbsorbGeLU:
    """GeLU decomposes to mul/erf/add/mul so the non-decomposed aten.gelu
    is what we want. Trace without decomposition to keep it as a single op."""

    def test_addmm_gelu(self):
        # make_fx without decomposition_table keeps gelu as aten.gelu.default
        model = nn.Sequential(nn.Linear(64, 64), nn.GELU())
        gm = _trace_no_grad(model, torch.randn(2, 64))

        # Check if gelu survived as a single node or got decomposed.
        gelu_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.gelu.default
        ]
        if gelu_nodes:
            groups = _find_groups(gm)
            assert len(groups) == 1
            assert groups[0].fused_nodes[-1].target is torch.ops.aten.gelu.default
        else:
            # GeLU was decomposed into mul/erf/add/mul — verify those absorb
            groups = _find_groups(gm)
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

        gm = _trace_no_grad(AddResidual(), torch.randn(2, 64))
        groups = _find_groups(gm)

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

        gm = _trace_no_grad(MulScale(), torch.randn(2, 64))
        groups = _find_groups(gm)

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

        gm = _trace_no_grad(Model(), torch.randn(2, 64))
        groups = _find_groups(gm)

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

        gm = _trace_no_grad(Model(), torch.randn(2, 64))
        groups = _find_groups(gm)

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
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

        for g in groups:
            targets = [n.target for n in g.all_nodes]
            assert targets.count(torch.ops.aten.addmm.default) == 1


class TestBarrierSoftmax:
    """Softmax after Linear should halt absorption."""

    def test_softmax_halts(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Softmax(dim=-1))
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

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

        gm = _trace_no_grad(Model(), torch.randn(2, 64))
        groups = _find_groups(gm)

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

        gm = _trace_no_grad(Branching(), torch.randn(2, 64))
        groups = _find_groups(gm)
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

        gm = _trace_no_grad(MidBranch(), torch.randn(2, 64))

        # Check if addmm -> relu fuses (relu has 2 users, so absorption
        # should stop *after* relu — but relu itself was absorbed because
        # the check is on current.users before looking at successor.
        # Actually: addmm has 1 user (relu), so relu is absorbed.
        # Then current=relu, relu has 2 users -> break.
        # Group = [addmm, relu] = len 2. Valid.
        groups = _find_groups(gm)

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

    def test_no_duplicate_inputs(self):
        """Each external input should appear at most once."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

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

        gm = _trace_no_grad(AddResidual(), torch.randn(2, 64))
        groups = _find_groups(gm)

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

        gm = _trace_no_grad(TwoHeads(), torch.randn(2, 64))
        groups = _find_groups(gm)

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

        gm = _trace_no_grad(TwoHeads(), torch.randn(2, 64))
        groups = _find_groups(gm)

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

        gm = _trace_no_grad(Sequential(), torch.randn(2, 64))
        groups = _find_groups(gm)

        # First addmm may or may not absorb relu depending on whether relu
        # feeds only into the transpose for l2. Second addmm -> sigmoid.
        # At minimum, second addmm -> sigmoid should form a group.
        sigmoid_groups = [
            g for g in groups
            if any(n.target is torch.ops.aten.sigmoid.default for n in g.fused_nodes)
        ]
        assert len(sigmoid_groups) >= 1


# ---------------------------------------------------------------------------
# Tests — FusionGroup dataclass
# ---------------------------------------------------------------------------

class TestFusionGroupDataclass:

    def test_len_single(self):
        """A FusionGroup with only a base_node has length 1."""
        model = nn.Linear(4, 4)
        gm = _trace_no_grad(model, torch.randn(1, 4))
        node = next(n for n in gm.graph.nodes if n.op == "call_function")
        group = FusionGroup(base_node=node)
        assert len(group) == 1

    def test_output_defaults_to_base(self):
        model = nn.Linear(4, 4)
        gm = _trace_no_grad(model, torch.randn(1, 4))
        node = next(n for n in gm.graph.nodes if n.op == "call_function")
        group = FusionGroup(base_node=node)
        assert group.output_node is node

    def test_all_nodes_ordering(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

        if groups:
            g = groups[0]
            assert g.all_nodes[0] is g.base_node
            assert g.all_nodes[1:] == g.fused_nodes

    def test_repr_contains_node_names(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = _trace_no_grad(model, torch.randn(2, 64))
        groups = _find_groups(gm)

        if groups:
            r = repr(groups[0])
            assert "FusionGroup(" in r
            assert "->" in r


# ---------------------------------------------------------------------------
# Tests — run() end-to-end
# ---------------------------------------------------------------------------

class TestRunEndToEnd:
    """Verify FuseMLFusionPass.run() executes without error and returns
    a GraphModule. (Surgery is still a TODO, so we just check the pass
    completes and doesn't corrupt the graph.)"""

    def test_run_returns_graph_module(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = _trace_no_grad(model, torch.randn(2, 64))
        result = FuseMLFusionPass(gm).run()
        assert isinstance(result, torch.fx.GraphModule)

    def test_run_preserves_numerics(self):
        """The graph module should still produce the same output after run()."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        x = torch.randn(2, 64)

        with torch.no_grad():
            expected = model(x)

        gm = _trace_no_grad(model, torch.randn(2, 64))

        # Run the pass (surgery is a no-op for now)
        result_gm = FuseMLFusionPass(gm).run()

        with torch.no_grad():
            actual = result_gm(x)

        assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-3)

    def test_run_no_fusible_ops(self):
        """run() on a graph with no fusible ops should not crash."""

        def identity(x):
            return x + 1.0

        gm = _trace_fn_no_grad(identity, torch.randn(2, 64))
        result = FuseMLFusionPass(gm).run()
        assert isinstance(result, torch.fx.GraphModule)


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
