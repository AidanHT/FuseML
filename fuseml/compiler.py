"""FuseMLCompiler — torch.compile backend.

This module provides the custom ``torch.compile`` backend that intercepts
FX graphs produced by TorchDynamo, tags fusion candidates, and (in the
future) swaps fused subgraphs with compiled Triton kernels.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn

from fuseml._logging import logger
from fuseml.registry import SupportedOpsRegistry, build_default_registry


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
