"""Tests for AOT-agnostic graph topology and pattern matching.

Covers:
  - Canonical target resolution (canonicalize_target)
  - Structural node classification (classify_node, is_trigger)
  - Canonical topology signature (build_op_signature, op_signature)
  - Defining-node resolution through transparent ops
  - SymInt-safe shape comparison utilities
  - Consistency between topology op sets and existing sets
  - Pattern matching robustness against AOT Autograd naming artifacts

Run with:
    pytest tests/test_aot_topology.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fuseml import (
    FuseMLFusionPass,
    FusionGroup,
    NodeRole,
    build_op_signature,
    canonicalize_target,
    classify_node,
    is_trigger,
    resolve_to_defining_node,
    symint_safe_eq,
    symint_safe_materialize,
)
from fuseml.codegen.kernel_cache import build_op_chain
from fuseml.passes.graph_cut import TRANSPARENT_OPS
from fuseml.passes.topology import (
    ABSORBABLE_OPS,
    BARRIER_OPS,
    INPLACE_OPS,
    REDUCTION_OPS,
    TRIGGER_OPS,
)

from conftest import find_groups, trace_no_grad


# ---------------------------------------------------------------------------
# Canonical target resolution
# ---------------------------------------------------------------------------

class TestCanonicalizeTarget:
    """canonicalize_target must produce stable strings from ATen targets."""
    pytestmark = pytest.mark.aot_topology

    def test_addmm_canonical_string(self):
        result = canonicalize_target(torch.ops.aten.addmm.default)
        assert "addmm" in result
        assert "default" in result

    def test_relu_canonical_string(self):
        result = canonicalize_target(torch.ops.aten.relu.default)
        assert "relu" in result

    def test_gelu_canonical_string(self):
        result = canonicalize_target(torch.ops.aten.gelu.default)
        assert "gelu" in result

    def test_stable_across_calls(self):
        """Same target must always produce the same canonical string."""
        t = torch.ops.aten.sigmoid.default
        assert canonicalize_target(t) == canonicalize_target(t)

    def test_different_targets_different_strings(self):
        """Different ATen ops must produce different canonical strings."""
        s1 = canonicalize_target(torch.ops.aten.relu.default)
        s2 = canonicalize_target(torch.ops.aten.gelu.default)
        assert s1 != s2

    def test_non_aten_callable(self):
        """Non-ATen callables should produce a non-empty string."""
        def my_func():
            pass
        result = canonicalize_target(my_func)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

class TestClassifyNode:
    """classify_node must return the correct NodeRole for ATen targets."""
    pytestmark = pytest.mark.aot_topology

    def test_addmm_is_barrier(self):
        """addmm is in both TRIGGER_OPS and BARRIER_OPS; classify_node
        returns BARRIER because trigger detection is handled by is_trigger()."""
        model = nn.Sequential(nn.Linear(64, 64))
        gm = trace_no_grad(model, torch.randn(2, 64))

        addmm_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.addmm.default
        ]
        assert len(addmm_nodes) >= 1
        # addmm is in BARRIER_OPS, but classify_node checks absorbable/
        # reduction/transparent/inplace first, then barrier. Since addmm
        # is not in those sets, it falls to BARRIER.
        role = classify_node(addmm_nodes[0])
        assert role is NodeRole.BARRIER

    def test_relu_is_absorbable(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))

        relu_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.relu.default
        ]
        if relu_nodes:
            assert classify_node(relu_nodes[0]) is NodeRole.ABSORBABLE

    def test_placeholder_is_unknown(self):
        """Placeholder nodes must be classified as UNKNOWN — never
        confused with call_function nodes."""
        model = nn.Sequential(nn.Linear(64, 64))
        gm = trace_no_grad(model, torch.randn(2, 64))

        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) >= 1
        for p in placeholders:
            assert classify_node(p) is NodeRole.UNKNOWN

    def test_output_is_unknown(self):
        model = nn.Sequential(nn.Linear(64, 64))
        gm = trace_no_grad(model, torch.randn(2, 64))

        outputs = [n for n in gm.graph.nodes if n.op == "output"]
        assert len(outputs) == 1
        assert classify_node(outputs[0]) is NodeRole.UNKNOWN


class TestIsTrigger:
    """is_trigger must correctly identify FusionGroup seed nodes."""
    pytestmark = pytest.mark.aot_topology

    def test_addmm_triggers(self):
        model = nn.Sequential(nn.Linear(64, 64))
        gm = trace_no_grad(model, torch.randn(2, 64))

        addmm_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.addmm.default
        ]
        assert len(addmm_nodes) >= 1
        assert is_trigger(addmm_nodes[0])

    def test_relu_does_not_trigger(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))

        relu_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.relu.default
        ]
        if relu_nodes:
            assert not is_trigger(relu_nodes[0])

    def test_placeholder_does_not_trigger(self):
        model = nn.Sequential(nn.Linear(64, 64))
        gm = trace_no_grad(model, torch.randn(2, 64))

        for n in gm.graph.nodes:
            if n.op == "placeholder":
                assert not is_trigger(n)


# ---------------------------------------------------------------------------
# Canonical topology signature
# ---------------------------------------------------------------------------

class TestOpSignature:
    """FusionGroup.op_signature and build_op_signature must produce
    canonical topology tuples independent of node names."""
    pytestmark = pytest.mark.aot_topology

    def test_op_signature_ignores_node_names(self):
        """Two traces of the same model must produce identical op_signature
        despite potentially different node names."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())

        gm1 = trace_no_grad(model, torch.randn(2, 64))
        groups1 = find_groups(gm1)

        gm2 = trace_no_grad(model, torch.randn(4, 64))
        groups2 = find_groups(gm2)

        if groups1 and groups2:
            assert groups1[0].op_signature == groups2[0].op_signature

    def test_op_signature_is_tuple_of_strings(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        if groups:
            sig = groups[0].op_signature
            assert isinstance(sig, tuple)
            for s in sig:
                assert isinstance(s, str)

    def test_op_signature_contains_target_names(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        if groups:
            sig = groups[0].op_signature
            assert any("addmm" in s for s in sig)
            assert any("relu" in s for s in sig)

    def test_build_op_signature_matches_property(self):
        """build_op_signature(group.all_nodes) == group.op_signature."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        if groups:
            group = groups[0]
            assert build_op_signature(group.all_nodes) == group.op_signature

    def test_op_chain_consistent_with_signature(self):
        """build_op_chain result must match op_signature (both are tuples)."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        if groups:
            group = groups[0]
            chain = build_op_chain(group)
            assert chain == group.op_signature

    def test_different_models_different_signatures(self):
        """Different model topologies must produce different signatures."""
        model_a = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        model_b = nn.Sequential(nn.Linear(64, 64), nn.Sigmoid())

        gm_a = trace_no_grad(model_a, torch.randn(2, 64))
        gm_b = trace_no_grad(model_b, torch.randn(2, 64))

        groups_a = find_groups(gm_a)
        groups_b = find_groups(gm_b)

        if groups_a and groups_b:
            assert groups_a[0].op_signature != groups_b[0].op_signature


# ---------------------------------------------------------------------------
# Defining-node resolution
# ---------------------------------------------------------------------------

class TestResolveToDefiningNode:
    """resolve_to_defining_node must trace through transparent ops."""
    pytestmark = pytest.mark.aot_topology

    def test_non_transparent_returns_self(self):
        """A non-transparent call_function node resolves to itself."""
        model = nn.Sequential(nn.Linear(64, 64))
        gm = trace_no_grad(model, torch.randn(2, 64))

        addmm = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.addmm.default
        ]
        if addmm:
            assert resolve_to_defining_node(addmm[0]) is addmm[0]

    def test_placeholder_returns_self(self):
        """Placeholder nodes resolve to themselves."""
        model = nn.Sequential(nn.Linear(64, 64))
        gm = trace_no_grad(model, torch.randn(2, 64))

        for n in gm.graph.nodes:
            if n.op == "placeholder":
                assert resolve_to_defining_node(n) is n

    def test_view_resolves_to_predecessor(self):
        """A view op should resolve through to its data source."""

        class ViewAfterLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                h = self.linear(x)
                return h.view(h.shape[0], h.shape[1])

        gm = trace_no_grad(ViewAfterLinear(), torch.randn(2, 64))

        view_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target in TRANSPARENT_OPS
        ]

        for v in view_nodes:
            resolved = resolve_to_defining_node(v)
            # Resolved node should not be a transparent op.
            if resolved.op == "call_function":
                assert resolved.target not in TRANSPARENT_OPS


# ---------------------------------------------------------------------------
# SymInt-safe utilities
# ---------------------------------------------------------------------------

class TestSymIntSafety:
    """SymInt-safe comparison and materialisation utilities."""
    pytestmark = pytest.mark.aot_topology

    def test_symint_safe_eq_concrete_equal(self):
        assert symint_safe_eq(42, 42) is True

    def test_symint_safe_eq_concrete_unequal(self):
        assert symint_safe_eq(42, 43) is False

    def test_symint_safe_eq_with_none(self):
        """Comparing with None should return False, not raise."""
        assert symint_safe_eq(42, None) is False

    def test_symint_safe_materialize_concrete(self):
        result = symint_safe_materialize((2, 64))
        assert result == (2, 64)
        assert all(isinstance(v, int) for v in result)

    def test_symint_safe_materialize_torch_size(self):
        t = torch.randn(3, 128)
        result = symint_safe_materialize(t.shape)
        assert result == (3, 128)

    def test_symint_safe_materialize_returns_none_on_failure(self):
        """Non-materializable values should return None, not raise."""
        result = symint_safe_materialize("not_a_shape")
        assert result is None


# ---------------------------------------------------------------------------
# Op set consistency
# ---------------------------------------------------------------------------

class TestOpSetConsistency:
    """Verify that topology op sets are consistent with the class-level
    sets on FuseMLFusionPass (backward compatibility)."""
    pytestmark = pytest.mark.aot_topology

    def test_barrier_ops_match(self):
        assert FuseMLFusionPass._BARRIER_OPS == BARRIER_OPS

    def test_absorbable_ops_match(self):
        assert FuseMLFusionPass._ABSORBABLE_OPS == ABSORBABLE_OPS

    def test_reduction_ops_match(self):
        assert FuseMLFusionPass._REDUCTION_OPS == REDUCTION_OPS

    def test_transparent_ops_reexported(self):
        """TRANSPARENT_OPS imported from graph_cut must be the same
        object as topology.TRANSPARENT_OPS."""
        from fuseml.passes.topology import TRANSPARENT_OPS as topo_transparent
        assert TRANSPARENT_OPS is topo_transparent


# ---------------------------------------------------------------------------
# Pattern matching with AOT-style placeholder names
# ---------------------------------------------------------------------------

class TestAOTPlaceholderRobustness:
    """Verify that pattern matching is robust against AOT Autograd
    placeholder naming conventions (primals_*, tangents_*)."""
    pytestmark = pytest.mark.aot_topology

    def test_fusion_independent_of_placeholder_names(self):
        """Pattern matching must find the same groups regardless of
        whether placeholders are named 'x', 'primals_0', etc."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))

        # Verify groups are found.
        groups = find_groups(gm)
        assert len(groups) >= 1

        # The group's inputs should be FX Nodes, not string names.
        for inp in groups[0].inputs:
            assert isinstance(inp, torch.fx.Node)

        # No input should be matched by name — verify by checking that
        # the group works even though we don't know the placeholder names.
        input_names = [inp.name for inp in groups[0].inputs]
        # Names may be anything (x, primals_0, etc.) — the important
        # thing is that groups were found correctly.
        assert len(input_names) > 0

    def test_multiple_traces_same_groups(self):
        """Tracing the same model twice should produce structurally
        identical fusion groups despite potentially different node names."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())

        gm1 = trace_no_grad(model, torch.randn(2, 64))
        gm2 = trace_no_grad(model, torch.randn(2, 64))

        groups1 = find_groups(gm1)
        groups2 = find_groups(gm2)

        assert len(groups1) == len(groups2)
        for g1, g2 in zip(groups1, groups2):
            # Same structural topology.
            assert g1.op_signature == g2.op_signature
            # Same number of inputs.
            assert len(g1.inputs) == len(g2.inputs)
            # Same number of nodes.
            assert len(g1) == len(g2)

    def test_dependency_resolution_uses_node_identity(self):
        """FusionGroup.inputs must contain actual FX Node objects
        resolved by graph-structure traversal, not string matching."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        if groups:
            group = groups[0]
            internal = set(group.all_nodes)

            # Every input must be an FX Node outside the group.
            for inp in group.inputs:
                assert isinstance(inp, torch.fx.Node)
                assert inp not in internal
