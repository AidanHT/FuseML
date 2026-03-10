"""FusionGroup — data structure for a single fused operator sequence.

A FusionGroup represents a contiguous subgraph of FX nodes that will be
replaced by a single fused kernel during graph surgery.  The kernel may
be a Triton-generated GEMM+epilogue (for memory-bound GEMMs) or a
cuBLAS GEMM with cublasLt epilogue fusion (for compute-bound GEMMs
with fusible activations like GeLU/ReLU).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import torch


@dataclass
class FusionGroup:
    """A contiguous sequence of FX nodes identified for fusion.

    Represents a subgraph that will be replaced by a single kernel.
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
        outside of it.  These become the kernel's input pointers.
    output_node : torch.fx.Node
        The final node in the sequence whose result is visible to the rest
        of the graph.  The fused kernel's output replaces this node.
    output_metadata : dict[str, Any]
        Tensor metadata (shape, stride, dtype) of the group's output,
        extracted from ``output_node.meta['tensor_meta']`` by
        :class:`~fuseml.passes.fusion_pass.FuseMLFusionPass` after shape
        propagation.  Used downstream for Triton block grid calculation
        and memory pointer setup.
    intermediate_outputs : list[torch.fx.Node]
        Internal nodes whose results are consumed by at least one node
        **outside** the fused block (escape nodes).  Each entry requires an
        additional output pointer in the ``@triton.jit`` signature and an
        intermediate ``tl.store`` in the epilogue to write the activation
        back to HBM, preserving it for PyTorch Autograd's backward pass.
    param_bindings : dict[str, torch.nn.Parameter | torch.Tensor]
        Mapping from ``get_attr`` node targets (dot-separated attribute
        paths such as ``"0.weight"`` or ``"linear.bias"``) to the resolved
        ``nn.Parameter`` or buffer tensor extracted from the root
        ``fx.GraphModule``.  Populated by
        :meth:`~fuseml.passes.fusion_pass.FuseMLFusionPass._resolve_get_attr_bindings`
        so that downstream codegen and the compiler can access live parameter
        data without relying on potentially incomplete FX node metadata.
    fusion_strategy : str
        Execution backend for this group:

        - ``"triton"`` (default) — custom Triton GEMM+epilogue kernel.
        - ``"cublas_epilogue"`` — cuBLAS GEMM with cublasLt epilogue
          fusion (GELU/ReLU fused into the matmul write-back).

        The compiler dispatches to different launcher constructors based
        on this field.
    cublas_pattern : CublasEpiloguePattern | None
        When ``fusion_strategy == "cublas_epilogue"``, stores the matched
        pattern (epilogue type, absorbed nodes, etc.).  ``None`` for
        Triton-fused groups.
    """

    base_node: torch.fx.Node
    fused_nodes: List[torch.fx.Node] = field(default_factory=list)
    inputs: List[torch.fx.Node] = field(default_factory=list)
    output_node: torch.fx.Node = None  # type: ignore[assignment]
    output_metadata: Dict[str, Any] = field(default_factory=dict)
    intermediate_outputs: List[torch.fx.Node] = field(default_factory=list)
    param_bindings: Dict[str, Union[torch.nn.Parameter, torch.Tensor]] = field(
        default_factory=dict,
    )
    node_args_snapshot: Dict[str, tuple] = field(default_factory=dict)
    fusion_strategy: str = "triton"
    cublas_pattern: Any = None

    def __post_init__(self) -> None:
        # Default output_node to base_node when the group is a single op.
        if self.output_node is None:
            self.output_node = self.base_node

    @property
    def all_nodes(self) -> List[torch.fx.Node]:
        """Return *base_node* followed by all *fused_nodes*."""
        return [self.base_node] + self.fused_nodes

    @property
    def op_signature(self) -> Tuple[str, ...]:
        """Canonical topology tuple built exclusively from ``node.target``.

        Returns a tuple of canonical ATen operator strings for every
        ``call_function`` node in execution order.  This is completely
        independent of ``node.name``, placeholder naming conventions
        (``primals_*``, ``tangents_*``), and any other tracing artifact.

        Two :class:`FusionGroup` instances with identical ``op_signature``
        represent structurally isomorphic fused operator sequences.
        """
        from fuseml.passes.topology import canonicalize_target

        return tuple(
            canonicalize_target(n.target)
            for n in self.all_nodes
            if n.op == "call_function"
        )

    def __len__(self) -> int:
        return 1 + len(self.fused_nodes)

    def __repr__(self) -> str:
        names = [n.name for n in self.all_nodes]
        return f"FusionGroup({' -> '.join(names)}, inputs={len(self.inputs)})"
