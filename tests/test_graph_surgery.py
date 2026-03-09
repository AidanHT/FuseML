"""Tests for graph surgery (_apply_surgery).

Verifies the graph rewriting phase: placeholder insertion, input wiring,
output rewiring, dead code elimination, and metadata attachment.

Run with:
    pytest tests/test_graph_surgery.py -v
    pytest tests/ -m surgery -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fuseml import FuseMLFusionPass, fuseml_fused_kernel_placeholder

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


class TestSurgeryPlaceholderRaises:
    """The placeholder should raise if actually called."""
    pytestmark = pytest.mark.surgery

    def test_placeholder_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="should never be called directly"):
            fuseml_fused_kernel_placeholder(torch.randn(2, 2))
