"""Tests for the graph-cutting safeguard — SUPPORTED_TRITON_OPS, validation, and splitting."""

import pytest
import torch

from fuseml.fusion_group import FusionGroup
from fuseml.passes.graph_cut import (
    SUPPORTED_TRITON_OPS,
    GraphSegment,
    split_fusion_group,
    validate_fusion_group,
)


# ---------------------------------------------------------------------------
# Helpers — hashable FX node stand-ins (SimpleNamespace is not hashable)
# ---------------------------------------------------------------------------

class _FakeNode:
    """Hashable stand-in for ``torch.fx.Node`` used in graph-cut tests."""

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


def _make_group(targets, *, base_name="addmm", node_prefix="n"):
    """Build a FusionGroup from a list of aten op targets.

    The first target is always the base node; subsequent targets become
    fused_nodes.  Minimal wiring is set up so that ``validate_fusion_group``
    and ``split_fusion_group`` can inspect the group.
    """
    nodes = []
    for i, tgt in enumerate(targets):
        name = base_name if i == 0 else f"{node_prefix}{i}"
        n = _make_node(target=tgt, name=name)
        nodes.append(n)

    # Wire args so each fused node takes the previous node as input.
    for i in range(1, len(nodes)):
        nodes[i].args = (nodes[i - 1],)

    # Wire users so each node points to the next.
    for i in range(len(nodes) - 1):
        nodes[i].users = {nodes[i + 1]: None}
    nodes[-1].users = {}

    group = FusionGroup(base_node=nodes[0])
    group.fused_nodes = nodes[1:]
    group.output_node = nodes[-1]

    # Compute inputs: external dependencies (base_node's args are external).
    group_set = set(group.all_nodes)
    seen = set()
    for node in group.all_nodes:
        for arg in node.args:
            if hasattr(arg, "name") and arg not in group_set and arg not in seen:
                group.inputs.append(arg)
                seen.add(arg)

    return group


# A fake unsupported aten op for testing.
# A hashable fake target representing an unsupported aten op.
_FAKE_UNSUPPORTED = "aten.fake_custom_op.default"


# ---------------------------------------------------------------------------
# SUPPORTED_TRITON_OPS dictionary
# ---------------------------------------------------------------------------

@pytest.mark.graph_cut
class TestSupportedTritonOps:
    """Verify the SUPPORTED_TRITON_OPS dictionary is populated correctly."""

    def test_contains_gemm_base(self):
        assert torch.ops.aten.addmm.default in SUPPORTED_TRITON_OPS
        assert SUPPORTED_TRITON_OPS[torch.ops.aten.addmm.default] == "gemm"

    def test_contains_elementwise_ops(self):
        elementwise = [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
            torch.ops.aten.gelu.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]
        for op in elementwise:
            assert op in SUPPORTED_TRITON_OPS, f"{op} missing from SUPPORTED_TRITON_OPS"
            assert SUPPORTED_TRITON_OPS[op] == "elementwise"

    def test_contains_reduction_ops(self):
        reductions = [
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.amax.default,
            torch.ops.aten.mean.dim,
        ]
        for op in reductions:
            assert op in SUPPORTED_TRITON_OPS, f"{op} missing from SUPPORTED_TRITON_OPS"
            assert SUPPORTED_TRITON_OPS[op] == "reduction"

    def test_does_not_contain_barrier_ops(self):
        """Barrier ops like softmax / layernorm must NOT be in the supported set."""
        barrier_ops = [
            torch.ops.aten._softmax.default,
            torch.ops.aten._log_softmax.default,
            torch.ops.aten.native_layer_norm.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.convolution.default,
        ]
        for op in barrier_ops:
            assert op not in SUPPORTED_TRITON_OPS, (
                f"{op} should not be in SUPPORTED_TRITON_OPS"
            )


# ---------------------------------------------------------------------------
# validate_fusion_group
# ---------------------------------------------------------------------------

@pytest.mark.graph_cut
class TestValidateFusionGroup:
    """Test that validate_fusion_group correctly identifies unsupported nodes."""

    def test_all_supported_returns_empty(self):
        group = _make_group([
            torch.ops.aten.addmm.default,
            torch.ops.aten.relu.default,
            torch.ops.aten.add.Tensor,
        ])
        assert validate_fusion_group(group) == []

    def test_single_unsupported_node(self):
        group = _make_group([
            torch.ops.aten.addmm.default,
            torch.ops.aten.relu.default,
            _FAKE_UNSUPPORTED,
            torch.ops.aten.gelu.default,
        ])
        bad = validate_fusion_group(group)
        assert len(bad) == 1
        assert bad[0].target is _FAKE_UNSUPPORTED

    def test_multiple_unsupported_nodes(self):
        fake2 = "aten.another_fake.default"
        group = _make_group([
            torch.ops.aten.addmm.default,
            _FAKE_UNSUPPORTED,
            fake2,
        ])
        bad = validate_fusion_group(group)
        assert len(bad) == 2

    def test_non_call_function_nodes_ignored(self):
        """Nodes with op != 'call_function' are inherently safe."""
        group = _make_group([torch.ops.aten.addmm.default, torch.ops.aten.relu.default])
        # Manually inject a placeholder-type node.
        placeholder = _make_node(op="placeholder", target="x", name="x_input")
        group.fused_nodes.insert(0, placeholder)
        bad = validate_fusion_group(group)
        assert bad == []

    def test_unsupported_base_node(self):
        group = _make_group([
            _FAKE_UNSUPPORTED,
            torch.ops.aten.relu.default,
        ])
        bad = validate_fusion_group(group)
        assert len(bad) == 1
        assert bad[0] is group.base_node


# ---------------------------------------------------------------------------
# split_fusion_group
# ---------------------------------------------------------------------------

@pytest.mark.graph_cut
class TestSplitFusionGroup:
    """Test the graph-splitting logic."""

    def test_fully_valid_group_returns_single_fused_segment(self):
        group = _make_group([
            torch.ops.aten.addmm.default,
            torch.ops.aten.relu.default,
            torch.ops.aten.gelu.default,
        ])
        segments = split_fusion_group(group)
        assert len(segments) == 1
        assert segments[0].kind == "fused"
        assert segments[0].group is group

    def test_unsupported_base_returns_all_native(self):
        group = _make_group([
            _FAKE_UNSUPPORTED,
            torch.ops.aten.relu.default,
        ])
        segments = split_fusion_group(group)
        assert len(segments) == 1
        assert segments[0].kind == "native"
        assert len(segments[0].nodes) == 2

    def test_split_at_middle_produces_kernel_a_and_native(self):
        """addmm -> relu -> UNSUPPORTED -> gelu -> Kernel A + native."""
        group = _make_group([
            torch.ops.aten.addmm.default,
            torch.ops.aten.relu.default,
            _FAKE_UNSUPPORTED,
            torch.ops.aten.gelu.default,
        ])
        segments = split_fusion_group(group)

        # Kernel A: addmm -> relu
        assert segments[0].kind == "fused"
        kernel_a = segments[0].group
        assert kernel_a is not None
        assert len(kernel_a) == 2  # base + 1 fused
        assert kernel_a.base_node.target is torch.ops.aten.addmm.default
        assert kernel_a.fused_nodes[0].target is torch.ops.aten.relu.default
        assert kernel_a.output_node is kernel_a.fused_nodes[-1]

        # Native: unsupported + gelu
        assert segments[1].kind == "native"
        assert len(segments[1].nodes) == 2
        assert segments[1].nodes[0].target is _FAKE_UNSUPPORTED

    def test_split_at_first_fused_node_no_kernel_a(self):
        """addmm -> UNSUPPORTED -> relu -> base alone has no fusion benefit."""
        group = _make_group([
            torch.ops.aten.addmm.default,
            _FAKE_UNSUPPORTED,
            torch.ops.aten.relu.default,
        ])
        segments = split_fusion_group(group)

        # Base node alone -> native (no fusion benefit).
        assert segments[0].kind == "native"
        assert len(segments[0].nodes) == 1
        assert segments[0].nodes[0].target is torch.ops.aten.addmm.default

        # Unsupported + remaining -> native.
        assert segments[1].kind == "native"
        assert len(segments[1].nodes) == 2

    def test_split_preserves_kernel_a_inputs(self):
        """Kernel A's inputs should be recomputed for the truncated group."""
        # Create external input nodes that the base and fused nodes reference.
        ext_bias = _make_node(target="bias", name="bias_input", users={})
        ext_weight = _make_node(target="weight", name="weight_input", users={})

        base = _make_node(
            target=torch.ops.aten.addmm.default,
            name="addmm",
            args=(ext_bias, ext_weight),
        )
        relu = _make_node(
            target=torch.ops.aten.relu.default,
            name="relu",
            args=(base,),
        )
        bad = _make_node(target=_FAKE_UNSUPPORTED, name="bad_op", args=(relu,))
        gelu = _make_node(target=torch.ops.aten.gelu.default, name="gelu", args=(bad,))

        base.users = {relu: None}
        relu.users = {bad: None}
        bad.users = {gelu: None}
        gelu.users = {}

        group = FusionGroup(base_node=base)
        group.fused_nodes = [relu, bad, gelu]
        group.output_node = gelu
        group.inputs = [ext_bias, ext_weight]

        segments = split_fusion_group(group)
        kernel_a = segments[0].group
        assert kernel_a is not None
        # Kernel A should include both external inputs from the base node.
        assert ext_bias in kernel_a.inputs
        assert ext_weight in kernel_a.inputs

    def test_split_with_reduction_before_unsupported(self):
        """addmm -> relu -> sum -> UNSUPPORTED -> Kernel A covers the reduction."""
        group = _make_group([
            torch.ops.aten.addmm.default,
            torch.ops.aten.relu.default,
            torch.ops.aten.sum.dim_IntList,
            _FAKE_UNSUPPORTED,
        ])
        segments = split_fusion_group(group)

        # Kernel A: addmm -> relu -> sum
        assert segments[0].kind == "fused"
        assert len(segments[0].group) == 3

        # Native: unsupported node
        assert segments[1].kind == "native"
        assert len(segments[1].nodes) == 1

    def test_all_elementwise_chain_valid(self):
        """A long supported chain should produce a single fused segment."""
        group = _make_group([
            torch.ops.aten.addmm.default,
            torch.ops.aten.relu.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.gelu.default,
            torch.ops.aten.mul.Tensor,
        ])
        segments = split_fusion_group(group)
        assert len(segments) == 1
        assert segments[0].kind == "fused"


# ---------------------------------------------------------------------------
# GraphSegment dataclass
# ---------------------------------------------------------------------------

@pytest.mark.graph_cut
class TestGraphSegment:
    """Basic tests for the GraphSegment data structure."""

    def test_repr_fused(self):
        node = _make_node(name="addmm")
        seg = GraphSegment(kind="fused", nodes=[node])
        assert "fused" in repr(seg)
        assert "addmm" in repr(seg)

    def test_repr_native(self):
        node = _make_node(name="bad_op")
        seg = GraphSegment(kind="native", nodes=[node])
        assert "native" in repr(seg)
        assert "bad_op" in repr(seg)

    def test_default_fields(self):
        seg = GraphSegment(kind="native")
        assert seg.group is None
        assert seg.nodes == []
