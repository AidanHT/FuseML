"""
FuseML Backend — Custom JIT backend for torch.compile.

This module implements the first stage of the FuseML pipeline: graph capture
and foundational pattern matching. The backend intercepts torch.fx graphs
produced by torch.compile, traverses them to identify fusible memory-bound
operator sequences (e.g., Linear -> GeLU), and flags candidates for future
Triton kernel fusion.

Currently operates in *interception-only* mode — patterns are detected and
logged, but execution falls through to standard eager mode so correctness
is preserved while the fusion infrastructure is built out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[FuseML] %(levelname)s — %(message)s",
)
logger = logging.getLogger("fuseml")


# ---------------------------------------------------------------------------
# SupportedOpsRegistry — extensible registry of low arithmetic-intensity ops
# ---------------------------------------------------------------------------
class SupportedOpsRegistry:
    """Registry of ops eligible for fusion based on arithmetic intensity.

    Low arithmetic-intensity (memory-bound) ops spend most of their wall-clock
    time moving data between HBM and SRAM rather than doing FLOPs.  Fusing
    consecutive memory-bound ops into a single Triton kernel eliminates the
    intermediate HBM round-trips.

    Usage
    -----
    >>> registry = SupportedOpsRegistry()
    >>> registry.register(torch.ops.aten.relu.default, "elementwise")
    >>> registry.is_supported(torch.ops.aten.relu.default)
    True

    The *category* string is free-form metadata that downstream passes can use
    to select fusion strategies (e.g. "elementwise" ops can always tile 1-D,
    while "reduction" ops need an accumulator dimension).
    """

    def __init__(self) -> None:
        # Maps op target -> category string for downstream strategy selection.
        self._ops: Dict[Callable, str] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def register(self, op_target: Callable, category: str = "memory_bound") -> None:
        """Add *op_target* to the registry under *category*."""
        self._ops[op_target] = category
        logger.debug("Registered op: %s [%s]", op_target, category)

    def register_many(
        self, targets: List[Callable], category: str = "memory_bound"
    ) -> None:
        """Convenience batch registration."""
        for t in targets:
            self.register(t, category)

    def unregister(self, op_target: Callable) -> None:
        """Remove *op_target* from the registry (no-op if absent)."""
        self._ops.pop(op_target, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def is_supported(self, op_target: Callable) -> bool:
        return op_target in self._ops

    def get_category(self, op_target: Callable) -> str | None:
        return self._ops.get(op_target)

    @property
    def targets(self) -> Set[Callable]:
        """Snapshot of all registered op targets."""
        return set(self._ops)

    def __contains__(self, op_target: Callable) -> bool:
        return self.is_supported(op_target)

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{getattr(t, 'name', str(t))}({c})" for t, c in self._ops.items()
        )
        return f"SupportedOpsRegistry([{entries}])"


# ---------------------------------------------------------------------------
# FusionGroup — data structure for a single fused operator sequence
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# FuseMLFusionPass — compiler pass that discovers and applies fusions
# ---------------------------------------------------------------------------
class FuseMLFusionPass:
    """Graph-rewriting pass that fuses memory-bound operator sequences.

    The pass operates in two phases:

    1. **Discovery** (``_find_fusion_groups``): walk the FX graph and
       identify contiguous runs of registry-matched, memory-bound nodes
       that can be collapsed into single Triton kernels.
    2. **Surgery** (``_apply_surgery``): rewrite the graph by replacing
       each :class:`FusionGroup` with a call to the corresponding
       generated Triton kernel.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The traced graph to optimize.  Modified **in-place** during surgery.
    registry : SupportedOpsRegistry | None
        Op registry used to decide which nodes are fusion-eligible.
        Defaults to :func:`build_default_registry`.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        registry: SupportedOpsRegistry | None = None,
    ) -> None:
        self.graph_module = graph_module
        self.registry = registry or build_default_registry()

    # ------------------------------------------------------------------
    # Phase 1 — Discover fusible groups
    # ------------------------------------------------------------------
    def _find_fusion_groups(self) -> List[FusionGroup]:
        """Identify contiguous sequences of fusible memory-bound nodes.

        Walks ``self.graph_module.graph`` in topological order and groups
        consecutive nodes whose targets are present in ``self.registry``
        into :class:`FusionGroup` instances.

        Returns
        -------
        list[FusionGroup]
            Ordered list of fusion candidates.  Empty when no fusible
            sequences are found.
        """
        # TODO: implement pattern-matching traversal
        return []

    # ------------------------------------------------------------------
    # Phase 2 — Rewrite the graph
    # ------------------------------------------------------------------
    def _apply_surgery(self, groups: List[FusionGroup]) -> None:
        """Replace each *FusionGroup* in the graph with a fused kernel call.

        For every group, this method will:
        1. Generate the corresponding Triton kernel source.
        2. Insert a new ``call_function`` node targeting the compiled kernel.
        3. Rewire downstream consumers to read from the new node.
        4. Erase the original fused nodes from the graph.

        Parameters
        ----------
        groups : list[FusionGroup]
            Fusion groups produced by :meth:`_find_fusion_groups`.
        """
        # TODO: implement graph surgery + Triton codegen integration
        pass

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> torch.fx.GraphModule:
        """Execute the full fusion pass and return the optimized graph module.

        Calls :meth:`_find_fusion_groups` to discover candidates, then
        :meth:`_apply_surgery` to rewrite the graph.  The modified
        ``GraphModule`` is recompiled before being returned so that its
        ``forward`` method reflects the new graph topology.

        Returns
        -------
        torch.fx.GraphModule
            The same ``graph_module`` passed at construction, modified
            in-place with fused subgraphs replaced by Triton kernel calls.
        """
        groups = self._find_fusion_groups()

        if groups:
            logger.info("Found %d fusion group(s) — applying graph surgery.", len(groups))
            self._apply_surgery(groups)
            self.graph_module.recompile()
        else:
            logger.info("No fusible sequences detected — graph unchanged.")

        return self.graph_module


def build_default_registry() -> SupportedOpsRegistry:
    """Create a registry pre-loaded with the baseline set of fusible ops.

    Currently registers only ``aten.addmm.default`` (the aten decomposition of
    ``nn.Linear``).  This is the canonical memory-bound producer: it writes a
    full intermediate activation tensor to HBM that downstream elementwise ops
    immediately re-read.

    To extend, call ``registry.register(op, category)`` after construction::

        registry = build_default_registry()
        registry.register(torch.ops.aten.relu.default,  "elementwise")
        registry.register(torch.ops.aten.gelu.default,  "elementwise")
        registry.register(torch.ops.aten.add.Tensor,    "elementwise")
    """
    registry = SupportedOpsRegistry()

    # --- Baseline: linear (matmul + bias) is the primary HBM producer ------
    registry.register(torch.ops.aten.addmm.default, "linear")

    return registry


# ---------------------------------------------------------------------------
# FuseMLCompiler — torch.compile backend
# ---------------------------------------------------------------------------
class FuseMLCompiler:
    """Custom ``torch.compile`` backend that inspects FX graphs for fusible
    memory-bound operator patterns using a :class:`SupportedOpsRegistry`.

    The compiler is invoked by TorchDynamo once per unique graph structure.
    It walks the graph nodes, tags those whose targets appear in the registry
    as ``fusion_candidates``, prints the annotated graph, and returns the
    *unmodified* forward callable so eager execution remains intact during
    this interception phase.

    Parameters
    ----------
    registry : SupportedOpsRegistry | None
        Op registry to match against.  Defaults to :func:`build_default_registry`.

    Attributes
    ----------
    fusion_candidates : list[torch.fx.Node]
        Accumulated nodes tagged as fusion candidates across all graphs
        processed by this compiler instance.
    """

    def __init__(self, registry: SupportedOpsRegistry | None = None) -> None:
        self.registry: SupportedOpsRegistry = registry or build_default_registry()
        self.fusion_candidates: List[torch.fx.Node] = []

    # ------------------------------------------------------------------
    # Entry point called by torch.compile for each captured sub-graph
    # ------------------------------------------------------------------
    def __call__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        """Analyse *gm*'s computational graph and return its forward method.

        Parameters
        ----------
        gm:
            The ``GraphModule`` produced by TorchDynamo containing an FX
            graph of aten-level operations.
        example_inputs:
            Representative tensors used during tracing.  Not consumed here
            but required by the backend protocol.

        Returns
        -------
        Callable
            ``gm.forward`` — the original eager implementation, unchanged.
            Future iterations will swap fused subgraphs with compiled
            Triton kernels here.
        """
        logger.info(
            "Captured FX graph with %d nodes — scanning for fusion candidates …",
            len(list(gm.graph.nodes)),
        )

        self.capture_and_print_graph(gm)

        # ── Passthrough: return unmodified forward for eager execution ──
        return gm.forward

    # ------------------------------------------------------------------
    # Graph capture, tagging, and printing
    # ------------------------------------------------------------------
    def capture_and_print_graph(self, gm: torch.fx.GraphModule) -> None:
        """Walk the FX graph, tag registry-matched nodes, and print the result.

        For every ``call_function`` node whose target is in ``self.registry``,
        we attach ``node.meta['fusion_candidate'] = True`` and the op's
        registry category.  This metadata propagates to downstream passes
        (pattern grouping, Triton codegen) without mutating the graph
        structure itself.

        A formatted summary of the graph is logged at INFO level so the
        developer can see the full node list with fusion annotations.
        """
        found: int = 0

        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue

            if self.registry.is_supported(node.target):
                node.meta["fusion_candidate"] = True
                node.meta["fusion_category"] = self.registry.get_category(node.target)
                self.fusion_candidates.append(node)
                found += 1

        # ── Print annotated graph ──────────────────────────────────────
        logger.info("--- FX Graph Dump (fusion candidates marked with *) ---")
        for node in gm.graph.nodes:
            is_candidate = node.meta.get("fusion_candidate", False)
            marker = " * [FUSION CANDIDATE]" if is_candidate else ""
            category = ""
            if is_candidate:
                category = f" (category={node.meta.get('fusion_category', '?')})"
            logger.info(
                "  %-12s %-30s target=%-40s%s%s",
                node.op,
                node.name,
                str(getattr(node.target, "name", node.target))[:40],
                marker,
                category,
            )
        logger.info("--- End Graph Dump ---")

        if found:
            logger.info(
                "Tagged %d node(s) as fusion candidates via registry.", found
            )
        else:
            logger.info("No registry-matched ops detected in this graph.")


# ======================================================================
# Smoke test — run directly to verify graph interception
# ======================================================================
def main() -> None:
    """Compile a minimal Linear+GeLU block through the FuseML backend and
    validate that eager-mode numerics are preserved."""

    # ── Build a simple test module ────────────────────────────────────
    model: nn.Module = nn.Sequential(
        nn.Linear(128, 128),
        nn.GELU(),
    )

    compiler = FuseMLCompiler()  # uses default registry (addmm only)
    compiled_model = torch.compile(model, backend=compiler)

    # ── Run a forward pass to trigger graph capture ───────────────────
    x: torch.Tensor = torch.randn(4, 128)

    with torch.no_grad():
        eager_out: torch.Tensor = model(x)
        fused_out: torch.Tensor = compiled_model(x)

    # ── Correctness gate: fused path must match eager path ────────────
    assert torch.allclose(eager_out, fused_out, atol=1e-3, rtol=1e-3), (
        "Mismatch between eager and compiled outputs — "
        "backend must not alter numerics during interception phase."
    )

    logger.info(
        "Validation passed — compiled output matches eager output "
        "(atol=1e-3, rtol=1e-3).  %d fusion candidate(s) logged.",
        len(compiler.fusion_candidates),
    )


if __name__ == "__main__":
    main()
