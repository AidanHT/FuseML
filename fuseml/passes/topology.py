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
    torch.ops.aten.mm.default,
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
    # aten.t.default intentionally NOT here — it swaps dimensions,
    # which breaks matmul operand identification in the kernel generator.
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

# Conservative static FLOP threshold — used as a fallback when GPU
# properties cannot be queried.  500 GFLOP is low enough to catch
# medium-to-large GEMMs where cuBLAS's superior compute throughput
# outweighs the HBM savings from Triton epilogue fusion.
_COMPUTE_BOUND_FLOP_THRESHOLD: float = 5e11  # 500 GFLOP

# Empirical efficiency gap between Triton GEMM and cuBLAS on Ada
# Lovelace.  Triton achieves roughly 85-90% of cuBLAS throughput,
# so the penalty is ~12% of GEMM execution time.
_TRITON_VS_CUBLAS_EFFICIENCY_GAP: float = 0.12


def _get_gpu_specs() -> Tuple[float, float] | None:
    """Query peak FLOPS (half-precision) and memory bandwidth from the GPU.

    Returns ``(peak_flops_hz, bandwidth_bytes_per_sec)`` or ``None``
    when CUDA is unavailable.
    """
    try:
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # Peak FP16 Tensor Core FLOPS ≈ SM_count × clock_GHz × FMA_ops_per_SM
        # For Ada (sm_89): 128 FP16 FMA ops/SM/cycle × 2 (FMA = 2 ops)
        sm_count = props.multi_processor_count
        clock_hz = props.clock_rate * 1000  # clock_rate is in kHz
        # Ada: 128 FP16 FMA per SM per cycle = 256 FP16 ops per SM per cycle
        # Ampere: same.  This is a reasonable estimate across architectures.
        fp16_ops_per_sm_per_cycle = 256
        peak_flops = sm_count * clock_hz * fp16_ops_per_sm_per_cycle

        # Memory bandwidth from total_memory and memory_clock is unreliable
        # (no bus width in props).  Use a conservative lookup by SM count.
        # RTX 4050 Laptop (20 SM): ~192 GB/s
        # RTX 4060 (24 SM): ~272 GB/s
        # RTX 4070 (46 SM): ~504 GB/s
        # RTX 4090 (128 SM): ~1008 GB/s
        # Fallback: 192 GB/s (conservative laptop estimate).
        if sm_count >= 100:
            bandwidth = 1008e9
        elif sm_count >= 40:
            bandwidth = 504e9
        elif sm_count >= 22:
            bandwidth = 272e9
        else:
            bandwidth = 192e9

        return (peak_flops, bandwidth)
    except Exception:
        return None


def is_compute_bound_gemm(
    M: int, N: int, K: int, dtype: Any = torch.bfloat16,
) -> bool:
    """Return ``True`` if a GEMM with dimensions (M, N, K) is compute-bound.

    Uses an **adaptive cost model** that compares the time penalty from
    Triton's lower compute throughput against the time saved by
    eliminating one HBM round-trip (the fusion benefit).

    When GPU properties are available::

        penalty_time = (2·M·N·K / gpu_peak_flops) × efficiency_gap
        savings_time = (2·M·N·bpe) / gpu_bandwidth

    If ``penalty_time > savings_time``, cuBLAS should handle the matmul.

    Falls back to a static 500 GFLOP threshold when GPU specs cannot
    be queried.

    Parameters
    ----------
    M, N, K :
        Matrix dimensions.
    dtype :
        Element precision — determines bytes per element for the HBM
        savings calculation.

    Returns
    -------
    bool
        ``True`` when cuBLAS should be preferred over a custom Triton
        fused kernel.
    """
    flops = 2.0 * M * N * K

    # Try adaptive cost model first.
    gpu_specs = _get_gpu_specs()
    if gpu_specs is not None:
        peak_flops, bandwidth = gpu_specs
        if peak_flops > 0 and bandwidth > 0:
            bpe = _BYTES_PER_ELEM.get(dtype, 2)
            # Time penalty from using Triton instead of cuBLAS.
            gemm_time = flops / peak_flops
            penalty_time = gemm_time * _TRITON_VS_CUBLAS_EFFICIENCY_GAP
            # Time saved by eliminating one M×N intermediate read+write.
            hbm_savings_bytes = 2.0 * M * N * bpe
            savings_time = hbm_savings_bytes / bandwidth
            if penalty_time > savings_time:
                logger.debug(
                    "Adaptive compute-bound check — M=%d, N=%d, K=%d: "
                    "penalty=%.3f ms > savings=%.3f ms → skip fusion",
                    M, N, K,
                    penalty_time * 1000, savings_time * 1000,
                )
                return True
            logger.debug(
                "Adaptive compute-bound check — M=%d, N=%d, K=%d: "
                "penalty=%.3f ms <= savings=%.3f ms → fuse",
                M, N, K,
                penalty_time * 1000, savings_time * 1000,
            )
            return False

    # Fallback: static threshold.
    return flops > _COMPUTE_BOUND_FLOP_THRESHOLD


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
