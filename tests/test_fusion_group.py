"""Tests for FusionGroup dataclass.

Run with:
    pytest tests/test_fusion_group.py -v
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fuseml import FusionGroup

from conftest import find_groups, trace_no_grad


class TestFusionGroupDataclass:

    def test_len_single(self):
        """A FusionGroup with only a base_node has length 1."""
        model = nn.Linear(4, 4)
        gm = trace_no_grad(model, torch.randn(1, 4))
        node = next(n for n in gm.graph.nodes if n.op == "call_function")
        group = FusionGroup(base_node=node)
        assert len(group) == 1

    def test_output_defaults_to_base(self):
        model = nn.Linear(4, 4)
        gm = trace_no_grad(model, torch.randn(1, 4))
        node = next(n for n in gm.graph.nodes if n.op == "call_function")
        group = FusionGroup(base_node=node)
        assert group.output_node is node

    def test_all_nodes_ordering(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        if groups:
            g = groups[0]
            assert g.all_nodes[0] is g.base_node
            assert g.all_nodes[1:] == g.fused_nodes

    def test_repr_contains_node_names(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        gm = trace_no_grad(model, torch.randn(2, 64))
        groups = find_groups(gm)

        if groups:
            r = repr(groups[0])
            assert "FusionGroup(" in r
            assert "->" in r

    def test_param_bindings_default_empty(self):
        """param_bindings should default to an empty dict."""
        model = nn.Linear(4, 4)
        gm = trace_no_grad(model, torch.randn(1, 4))
        node = next(n for n in gm.graph.nodes if n.op == "call_function")
        group = FusionGroup(base_node=node)
        assert group.param_bindings == {}
