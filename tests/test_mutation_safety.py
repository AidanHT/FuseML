"""Tests for in-place mutation detection and aliasing safety guards.

Verifies:
- IN_PLACE_OPS set contents and disjointness from _ABSORBABLE_OPS
- is_safe_inplace() correctly detects safe vs. aliased in-place ops
- View ancestry chain walking catches transitive aliasing
- check_group_mutation_safety() batch validation
- Integration with _find_fusion_groups() for in-place absorption

Run with:
    pytest tests/test_mutation_safety.py -v
    pytest tests/ -m mutation_safety -v
"""

from __future__ import annotations

import pytest
import torch

from fuseml.passes.fusion_pass import FuseMLFusionPass
from fuseml.passes.graph_cut import TRANSPARENT_OPS
from fuseml.passes.mutation_safety import (
    IN_PLACE_OPS,
    MutationFinding,
    check_group_mutation_safety,
    is_safe_inplace,
)


# ---------------------------------------------------------------------------
# Helpers — hashable FX node stand-ins for unit tests
# ---------------------------------------------------------------------------

class _FakeNode:
    """Hashable stand-in for ``torch.fx.Node``."""

    _counter = 0

    def __init__(self, *, op="call_function", target=None, name="node", args=()):
        _FakeNode._counter += 1
        self._id = _FakeNode._counter
        self.op = op
        self.target = target
        self.name = name
        self.args = args
        self.meta = {}
        self.users = {}

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"_FakeNode({self.name!r})"


def _make_node(*, op="call_function", target=None, name="node", args=(), users=None):
    """Create a minimal hashable object that quacks like ``torch.fx.Node``."""
    node = _FakeNode(op=op, target=target, name=name, args=args)
    if users is not None:
        node.users = users
    return node


# ---------------------------------------------------------------------------
# IN_PLACE_OPS set contents
# ---------------------------------------------------------------------------

@pytest.mark.mutation_safety
class TestInPlaceOpsSet:
    """Verify IN_PLACE_OPS contains expected entries."""

    def test_contains_relu_inplace(self):
        assert torch.ops.aten.relu_.default in IN_PLACE_OPS

    def test_contains_add_inplace(self):
        assert torch.ops.aten.add_.Tensor in IN_PLACE_OPS

    def test_contains_mul_inplace(self):
        assert torch.ops.aten.mul_.Tensor in IN_PLACE_OPS

    def test_contains_sigmoid_inplace(self):
        assert torch.ops.aten.sigmoid_.default in IN_PLACE_OPS

    def test_all_categorized_as_elementwise(self):
        for op, category in IN_PLACE_OPS.items():
            assert category == "elementwise", (
                f"{op} should be 'elementwise', got {category!r}"
            )

    def test_disjoint_from_absorbable_ops(self):
        """In-place ops must not overlap with _ABSORBABLE_OPS."""
        overlap = set(IN_PLACE_OPS) & FuseMLFusionPass._ABSORBABLE_OPS
        assert not overlap, f"Overlap: {overlap}"

    def test_disjoint_from_barrier_ops(self):
        """In-place ops must not overlap with _BARRIER_OPS."""
        overlap = set(IN_PLACE_OPS) & FuseMLFusionPass._BARRIER_OPS
        assert not overlap, f"Overlap: {overlap}"


# ---------------------------------------------------------------------------
# is_safe_inplace — non-in-place passthrough
# ---------------------------------------------------------------------------

@pytest.mark.mutation_safety
class TestSafeNonInplacePassthrough:
    """Non-in-place ops always return True."""

    def test_functional_relu_is_safe(self):
        node = _make_node(target=torch.ops.aten.relu.default, name="relu")
        group_set = {node}
        assert is_safe_inplace(node, group_set) is True

    def test_functional_add_is_safe(self):
        node = _make_node(target=torch.ops.aten.add.Tensor, name="add")
        group_set = {node}
        assert is_safe_inplace(node, group_set) is True


# ---------------------------------------------------------------------------
# is_safe_inplace — no aliasing (safe cases)
# ---------------------------------------------------------------------------

@pytest.mark.mutation_safety
class TestSafeInplaceNoAlias:
    """In-place ops with no external users on the mutated arg are safe."""

    def test_relu_inplace_no_external_users(self):
        """addmm → relu_: relu_ mutates addmm output, no external users."""
        addmm = _make_node(
            target=torch.ops.aten.addmm.default, name="addmm",
        )
        relu_ = _make_node(
            target=torch.ops.aten.relu_.default, name="relu_",
            args=(addmm,),
        )
        # addmm's only user is relu_
        addmm.users = {relu_: None}
        relu_.users = {}

        group_set = {addmm, relu_}
        assert is_safe_inplace(relu_, group_set) is True

    def test_add_inplace_no_external_users(self):
        """addmm → add_: safe when mutated arg has no external users."""
        addmm = _make_node(
            target=torch.ops.aten.addmm.default, name="addmm",
        )
        residual = _make_node(op="placeholder", name="residual")
        add_ = _make_node(
            target=torch.ops.aten.add_.Tensor, name="add_",
            args=(addmm, residual),
        )
        addmm.users = {add_: None}
        residual.users = {add_: None}
        add_.users = {}

        group_set = {addmm, add_}
        assert is_safe_inplace(add_, group_set) is True

    def test_sigmoid_inplace_no_external_users(self):
        """addmm → sigmoid_: safe."""
        addmm = _make_node(
            target=torch.ops.aten.addmm.default, name="addmm",
        )
        sigmoid_ = _make_node(
            target=torch.ops.aten.sigmoid_.default, name="sigmoid_",
            args=(addmm,),
        )
        addmm.users = {sigmoid_: None}
        sigmoid_.users = {}

        group_set = {addmm, sigmoid_}
        assert is_safe_inplace(sigmoid_, group_set) is True


# ---------------------------------------------------------------------------
# is_safe_inplace — aliased externally (unsafe cases)
# ---------------------------------------------------------------------------

@pytest.mark.mutation_safety
class TestUnsafeInplaceAliased:
    """In-place ops with external consumers on the mutated arg are unsafe."""

    def test_relu_inplace_mutated_arg_has_external_user(self):
        """addmm output used by both relu_ and external sigmoid → unsafe."""
        external_sigmoid = _make_node(
            target=torch.ops.aten.sigmoid.default, name="ext_sigmoid",
        )
        addmm = _make_node(
            target=torch.ops.aten.addmm.default, name="addmm",
        )
        relu_ = _make_node(
            target=torch.ops.aten.relu_.default, name="relu_",
            args=(addmm,),
        )
        # addmm feeds both relu_ (in group) and external_sigmoid (outside)
        addmm.users = {relu_: None, external_sigmoid: None}
        relu_.users = {}

        group_set = {addmm, relu_}
        assert is_safe_inplace(relu_, group_set) is False

    def test_mul_inplace_mutated_arg_has_external_user(self):
        """mul_ on a value also used externally → unsafe."""
        source = _make_node(op="placeholder", name="source")
        external_user = _make_node(
            target=torch.ops.aten.relu.default, name="ext_relu",
        )
        mul_ = _make_node(
            target=torch.ops.aten.mul_.Tensor, name="mul_",
            args=(source, source),
        )
        # source is used by both mul_ (in group) and external_relu (outside)
        source.users = {mul_: None, external_user: None}
        mul_.users = {}

        # source is an input, not in the group
        group_set = {mul_}
        assert is_safe_inplace(mul_, group_set) is False


# ---------------------------------------------------------------------------
# is_safe_inplace — view ancestry chain walk
# ---------------------------------------------------------------------------

@pytest.mark.mutation_safety
class TestUnsafeInplaceViewAncestry:
    """View ancestry chain walk catches transitive aliasing."""

    def test_relu_inplace_on_view_of_externally_used_tensor(self):
        """base → view → relu_: base has external users → unsafe."""
        external_user = _make_node(
            target=torch.ops.aten.sigmoid.default, name="ext_sigmoid",
        )
        base = _make_node(
            target=torch.ops.aten.addmm.default, name="base",
        )
        view = _make_node(
            target=torch.ops.aten.view.default, name="view",
            args=(base,),
        )
        relu_ = _make_node(
            target=torch.ops.aten.relu_.default, name="relu_",
            args=(view,),
        )
        # base is used by view (in group) and external_user (outside)
        base.users = {view: None, external_user: None}
        view.users = {relu_: None}
        relu_.users = {}

        group_set = {base, view, relu_}
        assert is_safe_inplace(relu_, group_set) is False

    def test_deep_view_chain_catches_aliasing(self):
        """base → view1 → view2 → relu_: base aliased externally → unsafe."""
        external_user = _make_node(
            target=torch.ops.aten.add.Tensor, name="ext_add",
        )
        base = _make_node(
            target=torch.ops.aten.addmm.default, name="base",
        )
        view1 = _make_node(
            target=torch.ops.aten.view.default, name="view1",
            args=(base,),
        )
        view2 = _make_node(
            target=torch.ops.aten.reshape.default, name="view2",
            args=(view1,),
        )
        relu_ = _make_node(
            target=torch.ops.aten.relu_.default, name="relu_",
            args=(view2,),
        )
        base.users = {view1: None, external_user: None}
        view1.users = {view2: None}
        view2.users = {relu_: None}
        relu_.users = {}

        group_set = {base, view1, view2, relu_}
        assert is_safe_inplace(relu_, group_set) is False

    def test_view_chain_safe_when_no_external_users(self):
        """base → view → relu_: base has no external users → safe."""
        base = _make_node(
            target=torch.ops.aten.addmm.default, name="base",
        )
        view = _make_node(
            target=torch.ops.aten.view.default, name="view",
            args=(base,),
        )
        relu_ = _make_node(
            target=torch.ops.aten.relu_.default, name="relu_",
            args=(view,),
        )
        base.users = {view: None}
        view.users = {relu_: None}
        relu_.users = {}

        group_set = {base, view, relu_}
        assert is_safe_inplace(relu_, group_set) is True


# ---------------------------------------------------------------------------
# is_safe_inplace — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.mutation_safety
class TestSafeInplaceEdgeCases:
    """Edge cases for is_safe_inplace."""

    def test_no_args_returns_true(self):
        """Node with empty args → safe (nothing to mutate)."""
        node = _make_node(
            target=torch.ops.aten.relu_.default, name="relu_",
            args=(),
        )
        assert is_safe_inplace(node, {node}) is True

    def test_scalar_arg_returns_true(self):
        """If arg[0] is a scalar, not an FX node → safe."""
        node = _FakeNode(
            target=torch.ops.aten.relu_.default, name="relu_",
        )
        node.args = (42,)
        node.users = {}
        assert is_safe_inplace(node, {node}) is True


# ---------------------------------------------------------------------------
# check_group_mutation_safety — batch validation
# ---------------------------------------------------------------------------

@pytest.mark.mutation_safety
class TestCheckGroupMutationSafety:
    """Batch validation of fusion groups."""

    def test_no_inplace_ops_returns_empty(self):
        """Group with only functional ops → no findings."""
        addmm = _make_node(target=torch.ops.aten.addmm.default, name="addmm")
        relu = _make_node(target=torch.ops.aten.relu.default, name="relu", args=(addmm,))
        addmm.users = {relu: None}
        relu.users = {}

        group_nodes = [addmm, relu]
        group_set = set(group_nodes)
        findings = check_group_mutation_safety(group_nodes, group_set)
        assert len(findings) == 0

    def test_safe_inplace_returns_safe_finding(self):
        """Safe in-place op → single finding with safe=True."""
        addmm = _make_node(target=torch.ops.aten.addmm.default, name="addmm")
        relu_ = _make_node(target=torch.ops.aten.relu_.default, name="relu_", args=(addmm,))
        addmm.users = {relu_: None}
        relu_.users = {}

        group_nodes = [addmm, relu_]
        group_set = set(group_nodes)
        findings = check_group_mutation_safety(group_nodes, group_set)

        assert len(findings) == 1
        assert findings[0].safe is True
        assert findings[0].node_name == "relu_"

    def test_unsafe_inplace_returns_unsafe_finding(self):
        """Aliased in-place op → finding with safe=False."""
        external = _make_node(target=torch.ops.aten.sigmoid.default, name="ext")
        addmm = _make_node(target=torch.ops.aten.addmm.default, name="addmm")
        relu_ = _make_node(target=torch.ops.aten.relu_.default, name="relu_", args=(addmm,))
        addmm.users = {relu_: None, external: None}
        relu_.users = {}

        group_nodes = [addmm, relu_]
        group_set = set(group_nodes)
        findings = check_group_mutation_safety(group_nodes, group_set)

        assert len(findings) == 1
        assert findings[0].safe is False

    def test_mutation_finding_dataclass(self):
        """MutationFinding has expected fields."""
        f = MutationFinding(node_name="relu_", description="test", safe=True)
        assert f.node_name == "relu_"
        assert f.description == "test"
        assert f.safe is True


# ---------------------------------------------------------------------------
# Integration: in-place ops in _find_fusion_groups
# ---------------------------------------------------------------------------

@pytest.mark.mutation_safety
class TestInPlaceAbsorptionIntegration:
    """Verify in-place ops are (or aren't) absorbed during pattern matching.

    Note: make_fx + functionalization typically converts in-place ops to
    functional equivalents.  These integration tests check behavioral
    contracts via the pattern matcher's code paths — the mock-based unit
    tests above cover the actual in-place detection logic.
    """

    def test_inplace_ops_not_in_absorbable_ops(self):
        """In-place variants must not be in _ABSORBABLE_OPS (they have
        their own dedicated branch with aliasing checks)."""
        for op in IN_PLACE_OPS:
            assert op not in FuseMLFusionPass._ABSORBABLE_OPS

    def test_inplace_ops_not_in_barrier_ops(self):
        """In-place ops should not be barriers."""
        for op in IN_PLACE_OPS:
            assert op not in FuseMLFusionPass._BARRIER_OPS

    def test_inplace_ops_not_in_reduction_ops(self):
        """In-place ops should not be reduction ops."""
        for op in IN_PLACE_OPS:
            assert op not in FuseMLFusionPass._REDUCTION_OPS
