"""FusionGroup — data structure for a single fused operator sequence.

A FusionGroup represents a contiguous subgraph of memory-bound FX nodes
that will be replaced by a single Triton kernel during graph surgery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class FusionGroup:
    """A contiguous sequence of memory-bound FX nodes identified for fusion.

    Represents a subgraph that will be replaced by a single Triton kernel.
    All nodes between *base_node* and *output_node* are absorbed into the
    group, and the generated kernel only reads from *inputs* and writes
    the result of *output_node* — eliminating intermediate HBM round-trips.

    Attributes
    ----------
    base_node : torch.fx.Node
        The first compute node in the fusible sequence (e.g., ``aten.addmm``).
    fused_nodes : list[torch.fx.Node]
        All subsequently absorbed nodes, **excluding** *base_node* itself.
    inputs : list[torch.fx.Node]
        External dependencies — nodes consumed by the group but produced
        outside of it.  These become the Triton kernel's input pointers.
    output_node : torch.fx.Node
        The final node in the sequence whose result is visible to the rest
        of the graph.  The fused kernel's output replaces this node.
    """

    base_node: torch.fx.Node
    fused_nodes: List[torch.fx.Node] = field(default_factory=list)
    inputs: List[torch.fx.Node] = field(default_factory=list)
    output_node: torch.fx.Node = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Default output_node to base_node when the group is a single op.
        if self.output_node is None:
            self.output_node = self.base_node

    @property
    def all_nodes(self) -> List[torch.fx.Node]:
        """Return *base_node* followed by all *fused_nodes*."""
        return [self.base_node] + self.fused_nodes

    def __len__(self) -> int:
        return 1 + len(self.fused_nodes)

    def __repr__(self) -> str:
        names = [n.name for n in self.all_nodes]
        return f"FusionGroup({' -> '.join(names)}, inputs={len(self.inputs)})"
