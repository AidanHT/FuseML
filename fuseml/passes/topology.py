"""AOT-agnostic node classification and canonical topology utilities.

Isolates pattern matching from PyTorch AOT Autograd tracing artifacts by
providing structural classification and topology hashing that rely
exclusively on ``node.target`` (ATen operator identity) and graph
connectivity — never on ``node.name``, ``node.op`` string contents
beyond the ``"call_function"`` discriminant, or placeholder naming
conventions (``primals_*``, ``tangents_*``).

Key utilities:

* :class:`NodeRole` — enum classifying each node's structural role.
* :func:`classify_node` — single-dispatch classifier using target identity.
* :func:`canonicalize_target` — stable string for any FX node target.
* :func:`build_op_signature` — canonical topology tuple for a node list.
* :func:`resolve_to_defining_node` — traces through transparent ops to
  the closest data-producing computation.
* :func:`symint_safe_eq` — SymInt-safe equality for shape comparisons.
* :func:`symint_safe_materialize` — deferred SymInt → int conversion.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable, Dict, List, Set, Tuple

import torch

from fuseml._logging import logger


# ---------------------------------------------------------------------------
# NodeRole — structural role of an FX node in the fusion pipeline
# ---------------------------------------------------------------------------

class NodeRole(Enum):
    """Structural role assigned to an FX node during fusion discovery."""

    TRIGGER = auto()       # Starts a new FusionGroup (e.g., addmm)
    ABSORBABLE = auto()    # Pointwise — absorbed into an existing group
    REDUCTION = auto()     # Absorbed as the *final* op (terminates group)
    BARRIER = auto()       # Halts forward absorption
    TRANSPARENT = auto()   # View/metadata — absorbed silently if shape-safe
    INPLACE = auto()       # In-place mutation — conditionally absorbed
    UNKNOWN = auto()       # Not recognised by the fusion pipeline


# ---------------------------------------------------------------------------
# Canonical op sets — single source of truth
# ---------------------------------------------------------------------------
# These are identity sets keyed on the actual ATen function objects.
# Pattern matching and classification both consult these sets, ensuring
# that adding a new op only requires a single edit.

TRIGGER_OPS: Set[Callable] = {
    torch.ops.aten.addmm.default,
}

BARRIER_OPS: Set[Callable] = {
    torch.ops.aten._softmax.default,
    torch.ops.aten._log_softmax.default,
    torch.ops.aten.native_layer_norm.default,
    torch.ops.aten.addmm.default,  # second addmm is a barrier
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.convolution.default,
}

ABSORBABLE_OPS: Set[Callable] = {
    torch.ops.aten.relu.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.mul.Tensor,
}

REDUCTION_OPS: Set[Callable] = {
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.amax.default,
    torch.ops.aten.mean.dim,
}

TRANSPARENT_OPS: Set[Callable] = {
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.expand.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
}

INPLACE_OPS: Set[Callable] = {
    torch.ops.aten.relu_.default,
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.mul_.Tensor,
    torch.ops.aten.sigmoid_.default,
}


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def classify_node(node: torch.fx.Node) -> NodeRole:
    """Classify *node* purely by its ``target`` attribute.

    Only ``call_function`` nodes are classified; all other op types
    (``placeholder``, ``output``, ``get_attr``) return :attr:`NodeRole.UNKNOWN`.

    The classification uses identity comparison against ATen function
    objects — it never inspects ``node.name`` or string-parses
    ``node.op``.
    """
    if node.op != "call_function":
        return NodeRole.UNKNOWN

    target = node.target

    # Order matters: TRIGGER before BARRIER because addmm is in both
    # sets (the *first* addmm triggers, subsequent ones are barriers).
    # The caller decides context (trigger vs barrier) — we report the
    # most specific role.
    if target in INPLACE_OPS:
        return NodeRole.INPLACE
    if target in REDUCTION_OPS:
        return NodeRole.REDUCTION
    if target in TRANSPARENT_OPS:
        return NodeRole.TRANSPARENT
    if target in ABSORBABLE_OPS:
        return NodeRole.ABSORBABLE
    if target in BARRIER_OPS:
        return NodeRole.BARRIER
    return NodeRole.UNKNOWN


def is_trigger(node: torch.fx.Node) -> bool:
    """Return ``True`` if *node* can seed a new FusionGroup."""
    return (
        node.op == "call_function"
        and node.target in TRIGGER_OPS
    )


# ---------------------------------------------------------------------------
# Canonical target string
# ---------------------------------------------------------------------------

def canonicalize_target(target: Any) -> str:
    """Convert an FX node target to a stable canonical string.

    Uses ``str(target)`` which for ATen ops produces deterministic
    names like ``"aten.addmm.default"``.  Falls back to
    ``repr(target)`` for non-ATen callables.

    This is the **only** place target → string conversion should happen
    so that all consumers (op_chain, op_signature, logging) share a
    single canonical form.
    """
    # ATen ops have a reliable str() representation.
    s = str(target)
    if s and s != "<unknown>":
        return s
    # Fallback for non-ATen callables.
    if hasattr(target, "__qualname__"):
        return target.__qualname__
    return repr(target)


# ---------------------------------------------------------------------------
# Canonical topology signature
# ---------------------------------------------------------------------------

def build_op_signature(nodes: List[torch.fx.Node]) -> Tuple[str, ...]:
    """Build a canonical topology tuple from a sequence of FX nodes.

    Returns a tuple of canonical target strings for every
    ``call_function`` node in *nodes*, preserving execution order.
    The result is hashable, comparable, and completely independent
    of node names or tracing-specific metadata.
    """
    return tuple(
        canonicalize_target(n.target)
        for n in nodes
        if n.op == "call_function"
    )


# ---------------------------------------------------------------------------
# Defining-node resolution
# ---------------------------------------------------------------------------

def resolve_to_defining_node(node: torch.fx.Node) -> torch.fx.Node:
    """Trace through transparent ops to the closest data-producing node.

    AOT Autograd may insert intermediate view/reshape/unsqueeze operations
    between the actual computation and its consumer.  This function walks
    backward through such transparent ops (following ``args[0]``) to find
    the node that actually defines the tensor data.

    Stops at:
    * Non-transparent ``call_function`` nodes (the defining computation).
    * ``placeholder`` / ``get_attr`` nodes (graph-level inputs).
    * Nodes with missing or non-Node ``args[0]``.

    Returns the original *node* unchanged if it is not a transparent op.
    """
    current = node
    visited: Set[torch.fx.Node] = set()

    while (
        current.op == "call_function"
        and current.target in TRANSPARENT_OPS
        and current not in visited
    ):
        visited.add(current)
        if not current.args or not isinstance(current.args[0], torch.fx.Node):
            break
        current = current.args[0]

    return current


# ---------------------------------------------------------------------------
# SymInt-safe shape utilities
# ---------------------------------------------------------------------------

def symint_safe_eq(a: Any, b: Any) -> bool:
    """Compare two values that may be ``torch.SymInt`` instances.

    During pattern matching discovery, shape dimensions may be symbolic
    integers introduced by Dynamo.  Direct ``==`` on SymInts can raise
    ``GuardOnDataDependentSymNode`` or return a symbolic boolean.

    This function attempts a concrete comparison and falls back to
    ``False`` (conservative) when the values cannot be resolved.

    Parameters
    ----------
    a, b :
        Values to compare (typically shape dimensions).

    Returns
    -------
    bool
        ``True`` only when both values are concretely equal.
    """
    try:
        return bool(a == b)
    except Exception:
        # SymInt comparison failed — cannot prove equality at trace time.
        return False


def symint_safe_len(shape: Any) -> int | None:
    """Return the length of *shape*, or ``None`` if it cannot be determined.

    Handles both concrete tuples and symbolic shape objects.
    """
    try:
        return len(shape)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Compute-bound GEMM detection
# ---------------------------------------------------------------------------

# Bytes-per-element for arithmetic intensity calculation.
_BYTES_PER_ELEM: Dict[Any, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
}

# Arithmetic intensity threshold above which a GEMM is compute-bound.
# For BF16 on Ada Lovelace: peak compute ~110 TFLOPS, peak BW ~256 GB/s,
# ridge point = 110e12 / 256e9 ≈ 430 FLOP/byte.  In practice, cuBLAS
# outperforms custom Triton GEMM at intensities above ~50 FLOP/byte.
_COMPUTE_BOUND_INTENSITY_THRESHOLD: float = 50.0


def is_compute_bound_gemm(
    M: int, N: int, K: int, dtype: Any = torch.bfloat16,
) -> bool:
    """Return ``True`` if a GEMM with dimensions (M, N, K) is compute-bound.

    Computes the arithmetic intensity (FLOP / byte of HBM traffic) and
    compares against a threshold.  When the intensity exceeds the
    threshold, cuBLAS will outperform a custom Triton GEMM kernel because
    the bottleneck is compute throughput (where cuBLAS excels with
    CUTLASS-optimized tiles, persistent kernels, and split-K), not
    memory bandwidth (where fusion eliminates redundant HBM traffic).

    The arithmetic intensity formula::

        intensity = 2·M·N·K / ((M·K + K·N + M·N) · bytes_per_element)

    This measures FLOPs per byte of data moved for a single GEMM
    (reading A, B, writing C).

    Parameters
    ----------
    M, N, K :
        Matrix dimensions.
    dtype :
        Element precision (determines bytes per element).

    Returns
    -------
    bool
        ``True`` when the GEMM is compute-bound and cuBLAS should be
        preferred over a custom Triton fused kernel.
    """
    bpe = _BYTES_PER_ELEM.get(dtype, 4)
    flops = 2.0 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * bpe
    if bytes_moved == 0:
        return False
    intensity = flops / bytes_moved
    return intensity > _COMPUTE_BOUND_INTENSITY_THRESHOLD


def symint_safe_materialize(vals: Any) -> Tuple[int, ...] | None:
    """Attempt to materialize a shape/stride tuple to concrete ints.

    Returns ``None`` if any element cannot be resolved to a concrete
    ``int`` (e.g., a truly dynamic SymInt guard).  This should **only**
    be called at kernel launch time, not during pattern matching.
    """
    try:
        return tuple(int(v) for v in vals)
    except Exception:
        return None
