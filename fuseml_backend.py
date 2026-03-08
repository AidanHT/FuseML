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
from typing import Callable, List

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
# Fusion-candidate op targets
# ---------------------------------------------------------------------------
# TorchDynamo may emit graphs at different decomposition levels:
#   - High-level: torch.nn.functional.linear / gelu  (default backend)
#   - Aten-level: aten.addmm.default / aten.gelu.default  (with aot_autograd)
# We match both so the pattern detector works regardless of decomposition.
#
# Why these matter:
#   addmm / linear  — produces an intermediate tensor, writes it to HBM.
#   gelu             — immediately reads that tensor back from HBM into SRAM.
# Fusing them keeps the intermediate in SRAM, eliminating the round-trip.
_LINEAR_TARGETS = {
    torch.ops.aten.addmm.default,       # aten decomposition of nn.Linear
    torch.nn.functional.linear,          # high-level functional op
}
_GELU_TARGETS = {
    torch.ops.aten.gelu.default,         # aten decomposition of nn.GELU
    torch.nn.functional.gelu,            # high-level functional op
}


class FuseMLBackend:
    """Custom ``torch.compile`` backend that inspects FX graphs for fusible
    memory-bound operator patterns.

    The backend is invoked by TorchDynamo once per unique graph structure.
    It walks the graph nodes, applies lightweight pattern-matching passes,
    and returns the *unmodified* forward callable so that eager execution
    remains intact during this interception phase.

    Attributes
    ----------
    fusion_candidates : list[tuple[torch.fx.Node, torch.fx.Node]]
        Accumulated (linear_node, activation_node) pairs identified across
        all graphs processed by this backend instance.
    """

    def __init__(self) -> None:
        self.fusion_candidates: list[tuple[torch.fx.Node, torch.fx.Node]] = []

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

        self._match_linear_activation(gm)

        # ── Passthrough: return unmodified forward for eager execution ──
        return gm.forward

    # ------------------------------------------------------------------
    # Pattern-matching passes
    # ------------------------------------------------------------------
    def _match_linear_activation(self, gm: torch.fx.GraphModule) -> None:
        """Identify contiguous ``Linear -> GeLU`` sequences in *gm*.

        Matches both high-level (functional.linear/gelu) and aten-level
        (addmm/gelu) decompositions so detection works at any graph level.

        Why this matters from a hardware perspective:
        * ``addmm`` / ``linear`` produces an intermediate tensor and writes
          it to HBM (~1.5 TB/s on A100).
        * ``gelu`` immediately reads that tensor back from HBM into SRAM
          (~19 TB/s on A100) just to apply an element-wise activation.
        * Fusing these two ops into a single Triton kernel keeps the
          intermediate entirely in SRAM, eliminating one full round-trip
          to global memory per token per layer.
        """
        found: int = 0

        for node in gm.graph.nodes:
            # We only care about call_function nodes — skip placeholders,
            # getattr, output, etc.
            if node.op != "call_function":
                continue

            if node.target not in _LINEAR_TARGETS:
                continue

            # Check downstream consumers: does any user of this linear op
            # immediately apply gelu?  We inspect node.users which maps
            # consumer nodes -> number of uses.
            for user in node.users:
                if (
                    user.op == "call_function"
                    and user.target in _GELU_TARGETS
                ):
                    self.fusion_candidates.append((node, user))
                    found += 1
                    logger.info(
                        "Fusion candidate found: [%s (Linear) -> %s (GeLU)]  "
                        "— intermediate tensor can be kept in SRAM, "
                        "avoiding HBM round-trip.",
                        node.name,
                        user.name,
                    )

        if found == 0:
            logger.info("No Linear -> GeLU patterns detected in this graph.")


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

    backend = FuseMLBackend()
    compiled_model = torch.compile(model, backend=backend)

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
        len(backend.fusion_candidates),
    )


if __name__ == "__main__":
    main()
