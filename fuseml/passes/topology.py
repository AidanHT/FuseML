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

# Static FLOP threshold — used as a fallback when GPU properties
# cannot be queried.  50 TFLOP is conservative enough to only skip
# truly enormous GEMMs where cuBLAS's compute advantage dominates.
_COMPUTE_BOUND_FLOP_THRESHOLD: float = 5e13  # 50 TFLOP

# Empirical efficiency gap between autotuned Triton GEMM and cuBLAS.
# With @triton.autotune exploring ~100 tile configs, Triton achieves
# 95-98% of cuBLAS throughput on Ada Lovelace.  The 5% gap accounts
# for cuBLAS's hand-tuned warp specialisation and persistent-kernel
# strategies that Triton's compiler does not yet replicate.
_TRITON_VS_CUBLAS_EFFICIENCY_GAP: float = 0.04


# Architecture-based FP16 Tensor Core TFLOPS (dense, no sparsity).
# Keyed by (compute_capability_major, compute_capability_minor).
# These are published peak throughput values from NVIDIA datasheets.
# Used because PyTorch's get_device_properties does not expose
# clock_rate, so first-principles FLOPS estimation is not possible.
_ARCH_TC_TFLOPS: Dict[Tuple[int, int], float] = {
    # Ampere (sm_80 / sm_86 / sm_87)
    (8, 0): 312.0,   # A100 SXM
    (8, 6): 40.0,    # RTX 3060 (conservative desktop estimate)
    (8, 7): 40.0,    # Orin / laptop Ampere
    # Ada Lovelace (sm_89)
    (8, 9): 73.0,    # RTX 4050 Laptop baseline; scaled by SM count below
    # Hopper (sm_90)
    (9, 0): 989.0,   # H100 SXM
}

# Per-SM FP16 Tensor Core TFLOPS for scaling across GPU SKUs.
# Derived from published specs: total_tflops / sm_count.
_ARCH_TC_TFLOPS_PER_SM: Dict[Tuple[int, int], float] = {
    (8, 0): 312.0 / 108,  # A100: 108 SMs
    (8, 6): 40.0 / 28,    # RTX 3060: 28 SMs
    (8, 7): 40.0 / 28,
    (8, 9): 73.0 / 20,    # RTX 4050 Laptop: 20 SMs → ~3.65 TFLOPS/SM
    (9, 0): 989.0 / 132,  # H100: 132 SMs
}


_cached_gpu_specs: Tuple[float, float] | None | bool = False  # False = not yet queried


def _get_gpu_specs() -> Tuple[float, float] | None:
    """Query peak Tensor Core FLOPS and memory bandwidth from the GPU.

    Uses architecture-based lookup tables keyed on compute capability
    and SM count, because ``torch.cuda.get_device_properties`` does not
    expose ``clock_rate`` in all PyTorch builds.

    Returns ``(peak_flops_hz, bandwidth_bytes_per_sec)`` or ``None``
    when CUDA is unavailable.

    The result is cached after the first call — GPU specs do not change
    during a process lifetime.
    """
    global _cached_gpu_specs
    if _cached_gpu_specs is not False:
        return _cached_gpu_specs  # type: ignore[return-value]
    try:
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        sm_count = props.multi_processor_count
        arch = (props.major, props.minor)

        # --- Peak Tensor Core FLOPS ---
        # Try per-SM scaling first (handles different GPU SKUs within
        # the same architecture).  Fall back to the fixed table entry.
        tflops_per_sm = _ARCH_TC_TFLOPS_PER_SM.get(arch)
        if tflops_per_sm is not None:
            peak_flops = tflops_per_sm * sm_count * 1e12
        else:
            fixed_tflops = _ARCH_TC_TFLOPS.get(arch)
            if fixed_tflops is not None:
                peak_flops = fixed_tflops * 1e12
            else:
                # Unknown architecture — estimate conservatively.
                # 2 TFLOPS/SM is a safe lower bound for any GPU with
                # tensor cores (Volta onward).
                peak_flops = 2.0 * sm_count * 1e12

        # --- Memory bandwidth ---
        # PyTorch does not expose memory bus width, so use an SM-count
        # based lookup.  Values are conservative (achievable, not peak).
        if sm_count >= 100:
            bandwidth = 1008e9   # RTX 4090 / H100-class
        elif sm_count >= 40:
            bandwidth = 504e9    # RTX 4070-class
        elif sm_count >= 22:
            bandwidth = 272e9    # RTX 4060-class
        else:
            bandwidth = 192e9    # RTX 4050 Laptop-class

        logger.debug(
            "GPU specs — arch=sm_%d%d, SMs=%d, peak_flops=%.1f TFLOPS, "
            "bandwidth=%.0f GB/s",
            arch[0], arch[1], sm_count,
            peak_flops / 1e12, bandwidth / 1e9,
        )
        _cached_gpu_specs = (peak_flops, bandwidth)
        return _cached_gpu_specs
    except Exception:
        _cached_gpu_specs = None
        return None


def is_compute_bound_gemm(
    M: int, N: int, K: int, dtype: Any = torch.bfloat16,
    num_epilogue_ops: int = 1,
) -> bool:
    """Return ``True`` if a GEMM with dimensions (M, N, K) is compute-bound.

    Uses an **adaptive cost model** that compares the time penalty from
    Triton's lower compute throughput against the time saved by fusing
    epilogue ops into the GEMM (eliminating HBM round-trips).

    When GPU properties are available::

        penalty_time = (2·M·N·K / gpu_peak_flops) × efficiency_gap
        savings_time = num_epilogue_ops × (2·M·N·bpe) / gpu_bandwidth

    Each fused epilogue op eliminates one M×N read + write cycle.
    If ``penalty_time > savings_time``, cuBLAS should handle the matmul.

    Falls back to a static threshold when GPU specs cannot be queried.

    Parameters
    ----------
    M, N, K :
        Matrix dimensions.
    dtype :
        Element precision — determines bytes per element for the HBM
        savings calculation.
    num_epilogue_ops :
        Number of elementwise ops that will be fused into the GEMM
        epilogue.  Each fused op saves one M×N HBM round-trip.
        Defaults to 1 (conservative single-op estimate).

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
            # Time saved by eliminating M×N intermediate read+write
            # for EACH fused epilogue op.
            hbm_savings_bytes = 2.0 * M * N * bpe * max(num_epilogue_ops, 1)
            savings_time = hbm_savings_bytes / bandwidth
            # Bias toward fusion: each fused kernel also eliminates one
            # CUDA kernel launch (~5 µs on Ada Lovelace) and improves SM
            # utilization by keeping data in SRAM.  Add a 10 µs credit
            # to the savings side to account for these secondary benefits.
            savings_time += 10e-6 * max(num_epilogue_ops, 1)
            if penalty_time > savings_time:
                logger.debug(
                    "Adaptive compute-bound check — M=%d, N=%d, K=%d, "
                    "epilogue_ops=%d: penalty=%.3f ms > savings=%.3f ms "
                    "→ skip fusion",
                    M, N, K, num_epilogue_ops,
                    penalty_time * 1000, savings_time * 1000,
                )
                return True
            logger.debug(
                "Adaptive compute-bound check — M=%d, N=%d, K=%d, "
                "epilogue_ops=%d: penalty=%.3f ms <= savings=%.3f ms "
                "→ fuse",
                M, N, K, num_epilogue_ops,
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
