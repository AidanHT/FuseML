"""In-place mutation detection and aliasing safety guards.

Before fusing an in-place operator (e.g. ``relu_``, ``add_``) into a Triton
kernel, we must verify that the mutated tensor is not aliased by nodes
outside the fusion group.  If it is, the fusion would silently erase the
mutation side-effect that downstream PyTorch code relies on.

This module provides:

* :data:`IN_PLACE_OPS` — canonical set of in-place ``aten`` ops the compiler
  can translate (mirrors :data:`SUPPORTED_TRITON_OPS` in ``graph_cut``).
* :func:`is_safe_inplace` — per-node aliasing check used during greedy
  absorption in :meth:`FuseMLFusionPass._find_fusion_groups`.
* :func:`check_group_mutation_safety` — batch validation over an entire
  fusion group, returning :class:`MutationFinding` diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set

import torch

from fuseml._logging import logger
from fuseml.passes.topology import TRANSPARENT_OPS


# ---------------------------------------------------------------------------
# Canonical set of in-place aten ops the Triton epilogue can translate
# ---------------------------------------------------------------------------

IN_PLACE_OPS: Dict[Any, str] = {
    torch.ops.aten.relu_.default: "elementwise",
    torch.ops.aten.add_.Tensor: "elementwise",
    torch.ops.aten.mul_.Tensor: "elementwise",
    torch.ops.aten.sigmoid_.default: "elementwise",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MutationFinding:
    """Result of an in-place mutation safety check for a single node.

    Attributes
    ----------
    node_name : str
        The FX node name that was inspected.
    description : str
        Human-readable explanation of the finding.
    safe : bool
        ``True`` when the in-place op can be safely absorbed into a fusion
        group without violating aliasing semantics.  ``False`` when
        external consumers alias the mutated tensor and fusion must be
        aborted at this node.
    """

    node_name: str
    description: str
    safe: bool


# ---------------------------------------------------------------------------
# Per-node aliasing check
# ---------------------------------------------------------------------------

def is_safe_inplace(
    node: torch.fx.Node,
    group_node_set: Set[torch.fx.Node],
) -> bool:
    """Return ``True`` if fusing *node* (an in-place op) is alias-safe.

    The check ensures that the tensor mutated by the in-place operation is
    not consumed by any node **outside** the fusion group.  If external
    consumers exist, fusing the op would silently remove the mutation
    side-effect they depend on.

    Additionally, the function walks the **view ancestry chain** of the
    mutated argument.  If the mutated tensor is itself a view (or chain of
    views) of a base tensor that has external consumers, the fusion is
    also unsafe — because the base tensor's memory is aliased through
    the view, and the in-place mutation would affect it.

    Parameters
    ----------
    node :
        An FX node whose ``target`` is in :data:`IN_PLACE_OPS`.
    group_node_set :
        The set of all nodes currently in the fusion group (including the
        base node and all absorbed nodes so far).

    Returns
    -------
    bool
        ``True`` when the in-place op can be safely fused.
    """
    if node.target not in IN_PLACE_OPS:
        return True  # Not an in-place op — always safe.

    # In-place aten ops always mutate args[0].
    if not node.args:
        return True  # Defensive: no args → nothing to mutate.

    mutated_arg = node.args[0]

    # The mutated arg must be an FX node (or node-like object) to inspect
    # its users.  Use duck-typing (hasattr "users") rather than isinstance
    # so the logic also works with lightweight node stand-ins in tests.
    if not hasattr(mutated_arg, "users"):
        return True  # Scalar or non-node arg — no aliasing concern.

    # --- Direct aliasing check -------------------------------------------
    # If the mutated argument has users outside the group (besides the
    # in-place node itself), the mutation would be invisible to those
    # external consumers after fusion replaces the subgraph.
    if _has_external_users(mutated_arg, group_node_set, exclude={node}):
        logger.debug(
            "Unsafe in-place: %s mutates %s which has external users.",
            node.name,
            mutated_arg.name,
        )
        return False

    # --- View ancestry chain walk ----------------------------------------
    # If the mutated arg is a view of another tensor, the underlying
    # storage is aliased.  Walk backwards through the view chain and check
    # each ancestor for external users.
    ancestor = mutated_arg
    while (
        hasattr(ancestor, "target")
        and ancestor.target in TRANSPARENT_OPS
        and ancestor.args
        and hasattr(ancestor.args[0], "users")
    ):
        ancestor = ancestor.args[0]
        if _has_external_users(ancestor, group_node_set, exclude={node}):
            logger.debug(
                "Unsafe in-place: %s mutates a view of %s which has "
                "external users.",
                node.name,
                ancestor.name,
            )
            return False

    return True


def _has_external_users(
    node: torch.fx.Node,
    group_node_set: Set[torch.fx.Node],
    exclude: Set[torch.fx.Node] | None = None,
) -> bool:
    """Return ``True`` if *node* has at least one user outside *group_node_set*.

    Parameters
    ----------
    node :
        The FX node to inspect.
    group_node_set :
        Nodes belonging to the current fusion group.
    exclude :
        Optional set of nodes to ignore when checking users (e.g. the
        in-place op itself, which is about to be absorbed).
    """
    exclude = exclude or set()
    for user in node.users:
        if user not in group_node_set and user not in exclude:
            return True
    return False


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def check_group_mutation_safety(
    group_nodes: List[torch.fx.Node],
    group_node_set: Set[torch.fx.Node],
) -> List[MutationFinding]:
    """Validate every node in a fusion group for in-place mutation safety.

    Iterates over *group_nodes* and calls :func:`is_safe_inplace` for each
    node whose target is in :data:`IN_PLACE_OPS`.  Returns a list of
    :class:`MutationFinding` diagnostics — one per in-place node found.

    Parameters
    ----------
    group_nodes :
        Ordered list of FX nodes in the fusion group.
    group_node_set :
        The same nodes as a set for O(1) membership checks.

    Returns
    -------
    list[MutationFinding]
        One entry per in-place node, indicating whether fusion is safe.
    """
    findings: List[MutationFinding] = []

    for node in group_nodes:
        if not hasattr(node, "target"):
            continue
        if node.target not in IN_PLACE_OPS:
            continue

        safe = is_safe_inplace(node, group_node_set)
        findings.append(
            MutationFinding(
                node_name=node.name if hasattr(node, "name") else str(node),
                description=(
                    f"In-place op {node.target} is "
                    f"{'safe' if safe else 'UNSAFE'} to fuse — "
                    f"{'no external aliasing detected' if safe else 'mutated tensor has external consumers'}."
                ),
                safe=safe,
            )
        )

    return findings
