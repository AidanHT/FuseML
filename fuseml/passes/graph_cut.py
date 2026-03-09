"""Graph-cutting safeguard — splits FusionGroups at unsupported operators.

Before passing FusionGroups to Triton code generation (Phase 3), this module
validates that every node in the group has a known Triton translation in the
kernel generator's epilogue.  If an unsupported operator is found, the group
is split precisely at that node:

* **Kernel A** — all nodes *before* the unsupported op form a shorter (but
  valid) FusionGroup that can still be compiled to a Triton kernel.
* **Native** — the unsupported op itself executes via standard PyTorch eager
  mode.
* **Kernel B** — remaining nodes after the unsupported op form a new
  FusionGroup *only if* they contain a valid GEMM base node (``addmm``).
  Otherwise they also fall back to native execution.

This prevents a single obscure ``aten.*`` op from crashing the entire JIT
compiler while preserving as much fusion benefit as possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Set

import torch

from fuseml._logging import logger
from fuseml.fusion_group import FusionGroup


# ---------------------------------------------------------------------------
# Authoritative set of ops the Triton kernel generator can translate
# ---------------------------------------------------------------------------

SUPPORTED_TRITON_OPS: Dict[Any, str] = {
    # GEMM base — the kernel's matmul core
    torch.ops.aten.addmm.default: "gemm",
    # Elementwise activations / arithmetic (register ops on `acc`)
    torch.ops.aten.relu.default: "elementwise",
    torch.ops.aten.relu_.default: "elementwise",
    torch.ops.aten.gelu.default: "elementwise",
    torch.ops.aten.add.Tensor: "elementwise",
    torch.ops.aten.mul.Tensor: "elementwise",
    # Reductions — cross-thread synchronization via tl.atomic_*
    torch.ops.aten.sum.dim_IntList: "reduction",
    torch.ops.aten.amax.default: "reduction",
    torch.ops.aten.mean.dim: "reduction",
}


# ---------------------------------------------------------------------------
# GraphSegment — represents one piece of a split FusionGroup
# ---------------------------------------------------------------------------

@dataclass
class GraphSegment:
    """A contiguous segment produced by splitting a FusionGroup.

    Attributes
    ----------
    kind :
        ``"fused"`` — this segment can be compiled to a Triton kernel.
        ``"native"`` — these nodes must run via PyTorch eager execution.
    group :
        The (possibly truncated) :class:`FusionGroup` when ``kind == "fused"``.
        ``None`` when ``kind == "native"``.
    nodes :
        The raw FX nodes belonging to this segment.  For ``"fused"``
        segments this mirrors ``group.all_nodes``; for ``"native"``
        segments it lists the nodes that PyTorch will execute eagerly.
    """

    kind: Literal["fused", "native"]
    group: FusionGroup | None = None
    nodes: List[torch.fx.Node] = field(default_factory=list)

    def __repr__(self) -> str:
        names = [n.name for n in self.nodes]
        return f"GraphSegment(kind={self.kind!r}, nodes={names})"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_fusion_group(group: FusionGroup) -> List[torch.fx.Node]:
    """Return every node in *group* whose target is **not** in
    :data:`SUPPORTED_TRITON_OPS`.

    Only ``call_function`` nodes are checked — other node types
    (``placeholder``, ``output``, ``get_attr``) are inherently safe.

    Parameters
    ----------
    group :
        The FusionGroup to validate.

    Returns
    -------
    list[torch.fx.Node]
        Unsupported nodes (empty when the entire group is valid).
    """
    unsupported: List[torch.fx.Node] = []
    for node in group.all_nodes:
        if node.op != "call_function":
            continue
        if node.target not in SUPPORTED_TRITON_OPS:
            unsupported.append(node)
    return unsupported


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _extract_tensor_metadata(node: torch.fx.Node) -> Dict[str, Any]:
    """Pull shape/stride/dtype from *node*'s ``tensor_meta``.

    Mirrors :meth:`FuseMLFusionPass._extract_tensor_metadata` so that the
    graph-cut module can rebuild output metadata for truncated sub-groups
    without importing the fusion pass (avoids circular dependencies).
    """
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return {}

    if not hasattr(meta, "shape"):
        if isinstance(meta, (tuple, list)) and len(meta) > 0:
            meta = meta[0]
        else:
            return {}

    if not hasattr(meta, "shape"):
        return {}

    return {
        "shape": tuple(meta.shape),
        "stride": tuple(meta.stride),
        "dtype": meta.dtype,
    }


def _build_sub_group(
    original: FusionGroup,
    fused_subset: List[torch.fx.Node],
) -> FusionGroup:
    """Build a new FusionGroup from *original*'s base node and a prefix of
    its fused nodes.

    Recomputes ``inputs``, ``output_node``, ``output_metadata``, and
    ``intermediate_outputs`` for the smaller group so downstream codegen
    sees consistent metadata.
    """
    sub = FusionGroup(base_node=original.base_node)
    sub.fused_nodes = list(fused_subset)
    sub.output_node = fused_subset[-1] if fused_subset else original.base_node

    # Recompute external inputs for the sub-group.  Use duck typing
    # (hasattr "op") rather than isinstance(torch.fx.Node) so the logic
    # also works with lightweight node stand-ins in tests.
    group_set: Set = set(sub.all_nodes)
    seen_inputs: Set = set()
    for node in sub.all_nodes:
        for arg in node.args:
            if hasattr(arg, "op") and arg not in group_set:
                if arg not in seen_inputs:
                    sub.inputs.append(arg)
                    seen_inputs.add(arg)

    # Recompute escape nodes (intermediate outputs consumed outside the
    # sub-group).
    for node in sub.all_nodes:
        if node is sub.output_node:
            continue
        for user in node.users:
            if user not in group_set:
                sub.intermediate_outputs.append(node)
                break

    # Rebuild output metadata from the new output node.
    sub.output_metadata = _extract_tensor_metadata(sub.output_node)

    return sub


def split_fusion_group(group: FusionGroup) -> List[GraphSegment]:
    """Split *group* at the first unsupported operator.

    Returns an ordered list of :class:`GraphSegment` objects describing how
    the original group should be executed:

    1. **Kernel A** (``kind="fused"``) — the longest valid prefix of the
       group that can be compiled.  Omitted when the base node itself is
       unsupported or the prefix contains only the base node (no fusion
       benefit).
    2. **Native** (``kind="native"``) — the unsupported node plus all
       remaining nodes that cannot form a valid kernel.
    3. **Kernel B** (``kind="fused"``) — if the remaining nodes after the
       native segment contain a GEMM base node (``addmm``), they form a
       new FusionGroup.  In practice, fusion groups are linear chains
       starting from a single ``addmm``, so Kernel B is rare; the
       architecture supports it for forward compatibility.

    When the entire group is valid, a single ``"fused"`` segment wrapping
    the original group is returned unchanged.

    Parameters
    ----------
    group :
        The FusionGroup to validate and potentially split.

    Returns
    -------
    list[GraphSegment]
        Ordered execution segments.
    """
    # --- Fast path: entire group is valid ---------------------------------
    unsupported = validate_fusion_group(group)
    if not unsupported:
        return [GraphSegment(kind="fused", group=group, nodes=list(group.all_nodes))]

    first_bad = unsupported[0]

    logger.warning(
        "Unsupported op %s (node %r) found in %s — splitting.",
        first_bad.target,
        first_bad.name,
        group,
    )

    # --- Base node itself is unsupported → entire group is native ---------
    if first_bad is group.base_node:
        logger.warning(
            "Base node %r is unsupported — entire group falls back to native.",
            group.base_node.name,
        )
        return [GraphSegment(kind="native", nodes=list(group.all_nodes))]

    # --- Find the split index in fused_nodes ------------------------------
    split_idx: int | None = None
    for i, node in enumerate(group.fused_nodes):
        if node is first_bad:
            split_idx = i
            break

    assert split_idx is not None, "Unsupported node not found in fused_nodes"

    segments: List[GraphSegment] = []

    # --- Kernel A: base_node + fused_nodes[:split_idx] --------------------
    pre_nodes = group.fused_nodes[:split_idx]
    if pre_nodes:
        # At least one absorbed node → fusion is worthwhile.
        group_a = _build_sub_group(group, pre_nodes)
        segments.append(
            GraphSegment(kind="fused", group=group_a, nodes=list(group_a.all_nodes))
        )
        logger.info(
            "Kernel A: %s (%d nodes)", group_a, len(group_a),
        )
    else:
        # Only the base node before the unsupported op — no fusion benefit,
        # but it still runs natively without issue (PyTorch handles addmm).
        segments.append(
            GraphSegment(kind="native", nodes=[group.base_node])
        )
        logger.info(
            "Base node %r alone offers no fusion benefit — marked native.",
            group.base_node.name,
        )

    # --- Native segment: unsupported node + trailing nodes ----------------
    trailing = group.fused_nodes[split_idx:]  # includes the unsupported node

    # Check whether any trailing node is a GEMM base that could seed
    # Kernel B.
    kernel_b_base_idx: int | None = None
    for j, node in enumerate(trailing):
        if j == 0:
            continue  # skip the unsupported node itself
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten.addmm.default
            and node.target in SUPPORTED_TRITON_OPS
        ):
            kernel_b_base_idx = j
            break

    if kernel_b_base_idx is not None:
        # Nodes between unsupported op and the new base → native.
        native_nodes = trailing[:kernel_b_base_idx]
        segments.append(GraphSegment(kind="native", nodes=native_nodes))
        logger.info(
            "Native segment: %d node(s) running via PyTorch.",
            len(native_nodes),
        )

        # Kernel B: new base node + everything after it.
        kb_base = trailing[kernel_b_base_idx]
        kb_fused = [
            n for n in trailing[kernel_b_base_idx + 1:]
            if n.op == "call_function" and n.target in SUPPORTED_TRITON_OPS
        ]
        if kb_fused:
            group_b = FusionGroup(base_node=kb_base)
            group_b.fused_nodes = kb_fused
            group_b.output_node = kb_fused[-1]

            # Recompute inputs / escapes for Kernel B.
            kb_set: Set[torch.fx.Node] = set(group_b.all_nodes)
            seen: Set[torch.fx.Node] = set()
            for node in group_b.all_nodes:
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg not in kb_set:
                        if arg not in seen:
                            group_b.inputs.append(arg)
                            seen.add(arg)
            for node in group_b.all_nodes:
                if node is group_b.output_node:
                    continue
                for user in node.users:
                    if user not in kb_set:
                        group_b.intermediate_outputs.append(node)
                        break

            group_b.output_metadata = _extract_tensor_metadata(group_b.output_node)
            segments.append(
                GraphSegment(kind="fused", group=group_b, nodes=list(group_b.all_nodes))
            )
            logger.info("Kernel B: %s (%d nodes)", group_b, len(group_b))
        else:
            # Base node alone — no fusion benefit, add to native.
            segments.append(
                GraphSegment(kind="native", nodes=trailing[kernel_b_base_idx:])
            )
    else:
        # No GEMM base in trailing nodes → all native.
        segments.append(GraphSegment(kind="native", nodes=trailing))
        logger.info(
            "Native segment: %d node(s) running via PyTorch.",
            len(trailing),
        )

    return segments
