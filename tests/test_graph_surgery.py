"""Tests for graph surgery (_apply_surgery).

Verifies the graph rewriting phase: placeholder insertion, input wiring,
output rewiring, dead code elimination, and metadata attachment.

Run with:
    pytest tests/test_graph_surgery.py -v
    pytest tests/ -m surgery -v
"""

from __future__ import annotations

import operator as _operator

import pytest
import torch
import torch.nn as nn

from fuseml import FuseMLFusionPass, FusionGroup, fuseml_fused_kernel_placeholder

from conftest import run_surgery, trace_no_grad


class TestSurgeryPlaceholderInserted:
    """Verify placeholder nodes are injected into the graph."""
    pytestmark = pytest.mark.surgery

    def test_placeholder_present_after_surgery(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        assert len(groups) >= 1
        placeholder_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is fuseml_fused_kernel_placeholder
        ]
        assert len(placeholder_nodes) == len(groups)

    def test_placeholder_count_matches_groups(self):
        """Two independent fusible chains yield two placeholder nodes."""

        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 64)
                self.b = nn.Linear(64, 64)

            def forward(self, x):
                return torch.relu(self.a(x)), torch.sigmoid(self.b(x))

        gm = trace_no_grad(TwoHeads(), torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        placeholder_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is fuseml_fused_kernel_placeholder
        ]
        assert len(placeholder_nodes) == len(groups)


class TestSurgeryOriginalNodesRemoved:
    """Dead code elimination should remove the original fused nodes."""
    pytestmark = pytest.mark.surgery

    def test_addmm_removed_after_surgery(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            remaining_targets = {
                n.target for n in gm.graph.nodes if n.op == "call_function"
            }
            # The original addmm and relu should be dead-code-eliminated.
            assert torch.ops.aten.addmm.default not in remaining_targets
            assert torch.ops.aten.relu.default not in remaining_targets

    def test_three_node_chain_cleaned(self):
        """addmm -> relu -> sigmoid: all three originals removed."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return torch.sigmoid(torch.relu(self.linear(x)))

        gm = trace_no_grad(Model(), torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            remaining_targets = {
                n.target for n in gm.graph.nodes if n.op == "call_function"
            }
            assert torch.ops.aten.addmm.default not in remaining_targets
            assert torch.ops.aten.relu.default not in remaining_targets
            assert torch.ops.aten.sigmoid.default not in remaining_targets


class TestSurgeryInputWiring:
    """Verify the placeholder node receives the correct external inputs."""
    pytestmark = pytest.mark.surgery

    def test_placeholder_args_match_group_inputs(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))

        fuse_pass = FuseMLFusionPass(gm)
        groups = fuse_pass._find_fusion_groups()
        assert len(groups) >= 1

        expected_inputs = groups[0].inputs
        fuse_pass._apply_surgery(groups)

        placeholder_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is fuseml_fused_kernel_placeholder
        ]
        assert len(placeholder_nodes) == 1
        assert list(placeholder_nodes[0].args) == expected_inputs

    def test_binary_op_external_input_wired(self):
        """Residual add's external operand must appear in placeholder args."""

        class AddResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x) + x

        gm = trace_no_grad(AddResidual(), torch.randn(2, 64))

        fuse_pass = FuseMLFusionPass(gm)
        groups = fuse_pass._find_fusion_groups()

        if groups:
            expected_input_count = len(groups[0].inputs)
            fuse_pass._apply_surgery(groups)

            placeholder_nodes = [
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is fuseml_fused_kernel_placeholder
            ]
            assert len(placeholder_nodes[0].args) == expected_input_count


class TestSurgeryOutputRewiring:
    """Verify downstream consumers are rewired to the placeholder node."""
    pytestmark = pytest.mark.surgery

    def test_output_node_uses_placeholder(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            # Find the output node
            output_nodes = [n for n in gm.graph.nodes if n.op == "output"]
            assert len(output_nodes) == 1

            # The output should reference a placeholder, not the original relu
            placeholder_nodes = [
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is fuseml_fused_kernel_placeholder
            ]
            assert len(placeholder_nodes) >= 1

            # Check that the output node's args reference the placeholder
            def _collect_node_refs(args):
                refs = []
                for a in args:
                    if isinstance(a, torch.fx.Node):
                        refs.append(a)
                    elif isinstance(a, (tuple, list)):
                        refs.extend(_collect_node_refs(a))
                return refs

            output_refs = _collect_node_refs(output_nodes[0].args)
            assert any(ref in placeholder_nodes for ref in output_refs)


class TestSurgeryMetadata:
    """Verify metadata is attached to placeholder nodes."""
    pytestmark = pytest.mark.surgery

    def test_fusion_group_in_meta(self):
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
                assert "fusion_group" in pn.meta
                assert "fused_op_names" in pn.meta
                assert isinstance(pn.meta["fused_op_names"], list)
                assert len(pn.meta["fused_op_names"]) > 0


class TestSurgeryGraphValid:
    """After surgery the graph must remain structurally valid."""
    pytestmark = pytest.mark.surgery

    def test_graph_lint_passes(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        # graph.lint() raises if the graph is structurally invalid.
        gm.graph.lint()

    def test_recompile_succeeds(self):
        """The graph module should have a valid forward after surgery."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        # recompile() would raise on an invalid graph
        gm.recompile()
        assert hasattr(gm, "forward")

    def test_no_fusible_ops_graph_unchanged(self):
        """Surgery on a graph with no groups should leave it intact."""

        def identity(x):
            return x + 1.0

        from conftest import trace_fn_no_grad

        gm = trace_fn_no_grad(identity, torch.randn(2, 64))
        node_count_before = len(list(gm.graph.nodes))

        gm, groups = run_surgery(gm)

        assert len(groups) == 0
        assert len(list(gm.graph.nodes)) == node_count_before


class TestSurgeryDanglingOutputGuard:
    """Verify fusion groups at the end of the graph are correctly wired."""
    pytestmark = pytest.mark.surgery

    def test_final_fusion_wired_to_output(self):
        """When the fused group is the last op before output, the output
        node's args must reference the placeholder, not the dead original."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            output_nodes = [n for n in gm.graph.nodes if n.op == "output"]
            assert len(output_nodes) == 1

            # Collect all node references from the output node's args.
            def _collect_refs(args):
                refs = []
                for a in args:
                    if isinstance(a, torch.fx.Node):
                        refs.append(a)
                    elif isinstance(a, (tuple, list)):
                        refs.extend(_collect_refs(a))
                return refs

            output_refs = _collect_refs(output_nodes[0].args)

            # None of the output refs should be dead fused nodes.
            for ref in output_refs:
                assert ref.op != "call_function" or (
                    ref.target is fuseml_fused_kernel_placeholder
                    or ref.target not in {
                        torch.ops.aten.addmm.default,
                        torch.ops.aten.relu.default,
                        torch.ops.aten.sigmoid.default,
                    }
                ), f"Output references dead node: {ref.name} ({ref.target})"

    def test_two_heads_both_wired_to_output(self):
        """Two parallel fusions that both feed into the output tuple."""

        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 64)
                self.b = nn.Linear(64, 64)

            def forward(self, x):
                return torch.relu(self.a(x)), torch.sigmoid(self.b(x))

        gm = trace_no_grad(TwoHeads(), torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if len(groups) == 2:
            placeholder_nodes = {
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is fuseml_fused_kernel_placeholder
            }

            output_nodes = [n for n in gm.graph.nodes if n.op == "output"]
            assert len(output_nodes) == 1

            def _collect_refs(args):
                refs = set()
                for a in args:
                    if isinstance(a, torch.fx.Node):
                        refs.add(a)
                    elif isinstance(a, (tuple, list)):
                        refs.update(_collect_refs(a))
                return refs

            output_refs = _collect_refs(output_nodes[0].args)
            # Both placeholders should be referenced by the output.
            assert placeholder_nodes.issubset(output_refs)

    def test_graph_lint_after_dangling_guard(self):
        """Graph must pass lint after surgery with dangling output guard."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)
        gm.graph.lint()


class TestSurgeryReverseTopologicalErasure:
    """Verify dead code sweeps use reverse topological order."""
    pytestmark = pytest.mark.surgery

    def test_cascading_get_attr_cleanup(self):
        """get_attr → t() → addmm → relu: after surgery, both t() and
        get_attr should be erased without error."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            # No orphaned get_attr nodes should remain.
            orphaned_get_attr = [
                n for n in gm.graph.nodes
                if n.op == "get_attr" and len(n.users) == 0
            ]
            assert len(orphaned_get_attr) == 0

    def test_orphaned_call_function_erased(self):
        """Orphaned call_function intermediates (e.g. aten.t) should be
        erased by the reverse-order sweep."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            orphaned_cf = [
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and len(n.users) == 0
                and n.target is not fuseml_fused_kernel_placeholder
            ]
            assert len(orphaned_cf) == 0

    def test_graph_lint_after_reverse_cleanup(self):
        """Graph must pass lint() after reverse-order DCE."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)
        gm.graph.lint()

    def test_multi_phase_cleanup_idempotent(self):
        """Running surgery twice should be safe — second run is a no-op."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups1 = run_surgery(gm)
        # Second run on already-fused graph — should find no new groups.
        gm, groups2 = run_surgery(gm)
        assert len(groups2) == 0
        gm.graph.lint()


class TestSurgeryPlaceholderRaises:
    """The placeholder should raise if actually called."""
    pytestmark = pytest.mark.surgery

    def test_placeholder_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="should never be called directly"):
            fuseml_fused_kernel_placeholder(torch.randn(2, 2))


# ---------------------------------------------------------------------------
# Multi-consumer safety (intermediate output rewiring)
# ---------------------------------------------------------------------------

def _build_multi_consumer_graph():
    """Build an FX graph with an intermediate node that has an external consumer.

    Graph topology::

        x, bias, weight (placeholders)
          └─→ addmm(bias, x, weight)          [base_node]
                └─→ relu(addmm)                [intermediate — has 2 consumers]
                      ├─→ sigmoid(relu)         [output_node]
                      └─→ add(relu, x)          [external consumer]
        output: (sigmoid, add)

    The fusion group is addmm → relu → sigmoid, with relu as an
    intermediate output because the external ``add`` node consumes it.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    bias = graph.placeholder("bias")
    weight = graph.placeholder("weight")

    addmm = graph.call_function(
        torch.ops.aten.addmm.default, args=(bias, x, weight),
    )
    relu = graph.call_function(
        torch.ops.aten.relu.default, args=(addmm,),
    )
    sigmoid = graph.call_function(
        torch.ops.aten.sigmoid.default, args=(relu,),
    )
    # External consumer of the intermediate relu — outside the fusion group.
    external_add = graph.call_function(
        torch.ops.aten.add.Tensor, args=(relu, x),
    )
    graph.output((sigmoid, external_add))

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    # Manually construct the FusionGroup: addmm → relu → sigmoid.
    group = FusionGroup(base_node=addmm)
    group.fused_nodes = [relu, sigmoid]
    group.output_node = sigmoid
    group.inputs = [bias, x, weight]

    # Escape-node analysis: relu has an external consumer (external_add).
    group_set = set(group.all_nodes)
    for n in group.all_nodes:
        if n is group.output_node:
            continue
        for user in n.users:
            if user not in group_set:
                group.intermediate_outputs.append(n)
                break

    return gm, group


class TestSurgeryMultiConsumerSafety:
    """Verify intermediate-output (escape-node) rewiring with getitem decomposition."""
    pytestmark = pytest.mark.surgery

    def test_intermediate_outputs_detected(self):
        """The manually constructed group should have relu as an escape node."""
        _gm, group = _build_multi_consumer_graph()
        assert len(group.intermediate_outputs) == 1
        assert group.intermediate_outputs[0].target is torch.ops.aten.relu.default

    def test_getitem_nodes_inserted(self):
        """Surgery should insert operator.getitem nodes: one for the primary
        output and one per intermediate output."""
        gm, group = _build_multi_consumer_graph()
        fuse_pass = FuseMLFusionPass(gm)
        fuse_pass._apply_surgery([group])

        getitem_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _operator.getitem
        ]
        # 1 primary + 1 intermediate = 2 getitem nodes.
        assert len(getitem_nodes) == 2

    def test_getitem_indices_correct(self):
        """getitem(placeholder, 0) is primary; getitem(placeholder, 1+) are
        intermediate outputs."""
        gm, group = _build_multi_consumer_graph()
        fuse_pass = FuseMLFusionPass(gm)
        fuse_pass._apply_surgery([group])

        getitem_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _operator.getitem
        ]
        indices = sorted(n.args[1] for n in getitem_nodes)
        assert indices == [0, 1]

    def test_external_consumer_rewired_to_getitem(self):
        """The external add node must now consume the intermediate getitem,
        not the dead relu node."""
        gm, group = _build_multi_consumer_graph()
        fuse_pass = FuseMLFusionPass(gm)
        fuse_pass._apply_surgery([group])

        # Find the surviving add node (external consumer).
        add_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.add.Tensor
        ]
        assert len(add_nodes) == 1

        # Its first arg should be a getitem node (index 1 — the intermediate).
        add_arg0 = add_nodes[0].args[0]
        assert add_arg0.target is _operator.getitem
        assert add_arg0.args[1] == 1  # intermediate index

    def test_primary_output_wired_through_getitem(self):
        """The graph's output should reference getitem(placeholder, 0)."""
        gm, group = _build_multi_consumer_graph()
        fuse_pass = FuseMLFusionPass(gm)
        fuse_pass._apply_surgery([group])

        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        # output args is a tuple: ((primary_getitem, add_node),)
        # Flatten to find the primary getitem.
        def _collect_refs(args):
            refs = []
            for a in args:
                if isinstance(a, torch.fx.Node):
                    refs.append(a)
                elif isinstance(a, (tuple, list)):
                    refs.extend(_collect_refs(a))
            return refs

        output_refs = _collect_refs(output_node.args)
        primary_getitems = [
            r for r in output_refs
            if r.op == "call_function"
            and r.target is _operator.getitem
            and r.args[1] == 0
        ]
        assert len(primary_getitems) == 1

    def test_num_outputs_metadata(self):
        """The placeholder node should carry num_outputs metadata."""
        gm, group = _build_multi_consumer_graph()
        fuse_pass = FuseMLFusionPass(gm)
        fuse_pass._apply_surgery([group])

        ph_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is fuseml_fused_kernel_placeholder
        ]
        assert len(ph_nodes) == 1
        assert ph_nodes[0].meta["num_outputs"] == 2

    def test_graph_lint_after_multi_output_surgery(self):
        """Graph must pass lint() after multi-output rewiring."""
        gm, group = _build_multi_consumer_graph()
        fuse_pass = FuseMLFusionPass(gm)
        fuse_pass._apply_surgery([group])
        gm.graph.lint()

    def test_original_fused_nodes_removed(self):
        """The original addmm, relu, sigmoid should be DCE'd after surgery."""
        gm, group = _build_multi_consumer_graph()
        fuse_pass = FuseMLFusionPass(gm)
        fuse_pass._apply_surgery([group])

        remaining_targets = {
            n.target for n in gm.graph.nodes if n.op == "call_function"
        }
        assert torch.ops.aten.addmm.default not in remaining_targets
        assert torch.ops.aten.relu.default not in remaining_targets
        assert torch.ops.aten.sigmoid.default not in remaining_targets


# ---------------------------------------------------------------------------
# Topology validation (SSA insertion-point guard)
# ---------------------------------------------------------------------------

class TestSurgeryTopologyValidation:
    """Verify _validate_insertion_topology catches SSA violations."""
    pytestmark = pytest.mark.surgery

    def test_valid_topology_passes(self):
        """A well-formed fusion group should pass topology validation."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))

        fuse_pass = FuseMLFusionPass(gm)
        groups = fuse_pass._find_fusion_groups()

        if groups:
            # Should not raise.
            FuseMLFusionPass._validate_insertion_topology(groups[0])

    def test_residual_input_topology_valid(self):
        """Residual connection (skip) should pass topology validation —
        the skip input is topologically before the output node."""

        class AddResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x) + x

        gm = trace_no_grad(AddResidual(), torch.randn(2, 64))
        fuse_pass = FuseMLFusionPass(gm)
        groups = fuse_pass._find_fusion_groups()

        if groups:
            # The residual x feeds into add.Tensor as a secondary input.
            # It must be topologically before the output_node.
            FuseMLFusionPass._validate_insertion_topology(groups[0])

    def test_malformed_group_raises(self):
        """A FusionGroup with an input after the output_node should raise."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        bias = graph.placeholder("bias")
        weight = graph.placeholder("weight")

        addmm = graph.call_function(
            torch.ops.aten.addmm.default, args=(bias, x, weight),
        )
        relu = graph.call_function(
            torch.ops.aten.relu.default, args=(addmm,),
        )
        # A node that appears AFTER relu in topological order.
        late_node = graph.call_function(
            torch.ops.aten.sigmoid.default, args=(x,),
        )
        graph.output(relu)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # Construct a malformed group whose inputs include late_node.
        group = FusionGroup(base_node=addmm)
        group.fused_nodes = [relu]
        group.output_node = relu
        group.inputs = [bias, x, weight, late_node]  # late_node is AFTER relu

        with pytest.raises(RuntimeError, match="SSA violation"):
            FuseMLFusionPass._validate_insertion_topology(group)

    def test_residual_inputs_all_before_output(self):
        """All external inputs (including residuals) must appear before
        the output_node in the graph's topological order."""

        class DoubleResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                h = self.linear(x)
                return h + x  # residual

        gm = trace_no_grad(DoubleResidual(), torch.randn(2, 64))
        fuse_pass = FuseMLFusionPass(gm)
        groups = fuse_pass._find_fusion_groups()

        if groups:
            group = groups[0]
            graph = group.base_node.graph
            topo_index = {n: i for i, n in enumerate(graph.nodes)}
            output_pos = topo_index[group.output_node]

            for inp in group.inputs:
                assert topo_index[inp] < output_pos, (
                    f"Input {inp.name} appears after output_node"
                )


# ---------------------------------------------------------------------------
# Acyclicity validation (post-surgery cycle detection)
# ---------------------------------------------------------------------------

class TestSurgeryCyclePrevention:
    """Verify _validate_acyclicity catches cycles in the graph."""
    pytestmark = pytest.mark.surgery

    def test_acyclic_graph_passes(self):
        """A standard post-surgery graph should pass cycle detection."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        # Should not raise — already validated inside _apply_surgery.
        if groups:
            FuseMLFusionPass._validate_acyclicity(gm.graph)

    def test_two_heads_acyclic(self):
        """Two parallel fusion groups should produce an acyclic graph."""

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
            FuseMLFusionPass._validate_acyclicity(gm.graph)

    def test_multi_consumer_surgery_acyclic(self):
        """Multi-output surgery should produce an acyclic graph."""
        gm, group = _build_multi_consumer_graph()
        fuse_pass = FuseMLFusionPass(gm)
        fuse_pass._apply_surgery([group])

        # Should not raise — validated inside _apply_surgery, but verify
        # it also passes when called independently.
        FuseMLFusionPass._validate_acyclicity(gm.graph)

    def test_synthetic_cycle_detected(self):
        """Manually introduce a cycle and verify detection."""
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        b = graph.call_function(torch.ops.aten.relu.default, args=(a,))
        c = graph.call_function(torch.ops.aten.sigmoid.default, args=(b,))
        graph.output(c)

        # Introduce a cycle: make b also use c (c → b → c).
        # We do this by directly mutating the user dict — this is NOT
        # how FX is meant to be used, but it simulates a bug in rewiring.
        b.args = (c,)  # b now uses c, but c uses b → cycle
        # Note: this also makes a orphaned but that's fine for the test.

        with pytest.raises(RuntimeError, match="Cycle detected"):
            FuseMLFusionPass._validate_acyclicity(graph)


# ---------------------------------------------------------------------------
# Residual topology mapping (SSA-preserving residual wiring)
# ---------------------------------------------------------------------------

class TestSurgeryResidualTopology:
    """Verify residual connections are correctly mapped during surgery."""
    pytestmark = pytest.mark.surgery

    def test_residual_add_input_in_placeholder_args(self):
        """The residual input (x) must appear in the placeholder's args
        after surgery when the group contains addmm + add(result, x)."""

        class AddResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x) + x

        gm = trace_no_grad(AddResidual(), torch.randn(2, 64))
        fuse_pass = FuseMLFusionPass(gm)
        groups = fuse_pass._find_fusion_groups()

        if groups:
            group = groups[0]
            # The residual x should be in group.inputs.
            placeholder_names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
            input_names = [n.name for n in group.inputs]

            # At least one placeholder (the data input x) should appear
            # as both a regular input and a residual input.
            x_placeholder = next(
                (n for n in gm.graph.nodes if n.op == "placeholder"),
                None,
            )
            assert x_placeholder is not None
            assert x_placeholder in group.inputs or any(
                inp.name in placeholder_names for inp in group.inputs
            )

            fuse_pass._apply_surgery(groups)

            # After surgery, the placeholder node should reference x.
            ph_nodes = [
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is fuseml_fused_kernel_placeholder
            ]
            assert len(ph_nodes) == 1
            assert x_placeholder in ph_nodes[0].args

    def test_mul_residual_wired_correctly(self):
        """A multiplicative residual (linear(x) * x) should wire x
        into the placeholder args."""

        class MulResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x) * x

        gm = trace_no_grad(MulResidual(), torch.randn(2, 64))
        gm, groups = run_surgery(gm)

        if groups:
            ph_nodes = [
                n for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is fuseml_fused_kernel_placeholder
            ]
            assert len(ph_nodes) == 1
            # The placeholder must have the data input x in its args.
            x_placeholder = next(
                n for n in gm.graph.nodes if n.op == "placeholder"
            )
            assert x_placeholder in ph_nodes[0].args

    def test_surgery_with_residual_graph_valid(self):
        """Full surgery on a residual model must produce a lint-clean graph."""

        class AddResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x) + x

        gm = trace_no_grad(AddResidual(), torch.randn(2, 64))
        gm, groups = run_surgery(gm)
        gm.graph.lint()
