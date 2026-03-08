"""SupportedOpsRegistry — extensible registry of low arithmetic-intensity ops.

This module owns the registry that tracks which aten ops are eligible for
fusion, along with the factory function that builds the default set.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Set

import torch

from fuseml._logging import logger


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


def build_default_registry() -> SupportedOpsRegistry:
    """Create a registry pre-loaded with the baseline set of fusible ops.

    Currently registers ``aten.addmm.default`` (the aten decomposition of
    ``nn.Linear``) as the canonical memory-bound producer, plus the standard
    set of low arithmetic-intensity pointwise ops that can be absorbed into
    a fused kernel.
    """
    registry = SupportedOpsRegistry()

    # --- Baseline: linear (matmul + bias) is the primary HBM producer ------
    registry.register(torch.ops.aten.addmm.default, "linear")

    # --- Low arithmetic-intensity pointwise ops (fusion-absorbable) ---------
    registry.register_many(
        [
            torch.ops.aten.relu.default,
            torch.ops.aten.gelu.default,
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ],
        category="elementwise",
    )

    return registry
