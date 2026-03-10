"""KernelCacheKey and caching engine for compiled Triton kernels.

Prevents redundant kernel compilation by uniquely identifying each fused
kernel configuration through its operator topology, tensor memory layout,
storage offsets, pointer alignment, data types, and GPU compute capability.

The cache key accounts for PyTorch's underlying C++ ATen tensor
representations to prevent out-of-bounds memory accesses and silent
miscalculations when tensors are views, slices, or non-contiguously strided.

**Design invariants:**

* The cache never forces ``.contiguous()`` on input tensors.  Strided
  layouts are cached and dispatched as separate kernel variants whenever
  the operator chain is mathematically valid for the given layout, avoiding
  unnecessary HBM copy overhead.
* ``storage_offset`` is part of the key because Triton kernels must offset
  their base pointers by this value; a kernel compiled for offset-zero
  would read/write the wrong memory region on a view with a non-zero
  offset.
* ``aligned`` (``data_ptr() % 16 == 0``) is part of the key so that
  aligned configurations map to kernels using wide vectorised loads, while
  misaligned configurations map to a safe variant that avoids GPU memory
  faults from over-wide vector instructions.
* ``device_capability`` captures ``torch.cuda.get_device_capability()``
  (e.g., ``(8, 0)`` for A100 vs ``(8, 9)`` for Ada Lovelace) to ensure
  optimal ``BLOCK_SIZE`` and ``num_warps`` selection per GPU architecture.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from fuseml._logging import logger


# ---------------------------------------------------------------------------
# SymInt materialisation helpers
# ---------------------------------------------------------------------------

def _materialize_ints(vals) -> tuple[int, ...]:
    """Force-evaluate every element to a concrete Python ``int``.

    PyTorch 2.0+ may represent dimensions as ``torch.SymInt`` in FX graphs
    with dynamic shapes.  ``SymInt`` objects are not reliably hashable with
    ``hash()`` and cannot be stored in frozen-dataclass fields that expect
    ``int``.  This helper resolves them eagerly.
    """
    return tuple(int(v) for v in vals)


def _materialize_int(val) -> int:
    """Force-evaluate a single value (possibly ``torch.SymInt``) to ``int``."""
    return int(val)


# ---------------------------------------------------------------------------
# TensorFingerprint — per-tensor memory layout descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorFingerprint:
    """Physical memory-layout fingerprint for a single tensor.

    Captures every property of a tensor's ATen storage representation that
    affects Triton kernel correctness:

    - **shape** and **stride**: determine pointer arithmetic and tile bounds
      (vital for boundary-masked offset calculations ``offs_m``, ``offs_n``,
      ``offs_k``).
    - **storage_offset**: the offset (in elements) from the storage base
      pointer.  Triton kernels must add this to their base pointers to
      avoid reading/writing the wrong memory region (handles view ancestry).
    - **aligned**: whether ``data_ptr() % 16 == 0``.  Misaligned pointers
      prevent the use of vectorised ``tl.load`` instructions, which would
      cause GPU memory faults.
    - **dtype**: the scalar type, stored as a string for deterministic
      hashing across sessions.
    - **broadcast_dims**: a boolean tuple recording which dimensions have
      stride 0 (the standard convention for broadcast/expanded dimensions).
      A ``torch.Tensor`` produced by ``.expand()`` has stride 0 on the
      expanded axis; this must be part of the cache key so that kernels
      compiled for one broadcast pattern are never dispatched on another.

    No FX node references, parameter names, or symbolic variables are
    stored — only physical memory layout properties.
    """

    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    storage_offset: int
    aligned: bool
    dtype: str  # str(torch.dtype) for deterministic hashing
    broadcast_dims: Tuple[bool, ...] = ()

    # ── Deterministic hash and equality ────────────────────────────────

    def __hash__(self) -> int:
        """Deterministic hash from all physical-layout fields.

        Reduces every field to a Python primitive (int, str, bool,
        tuple-of-int) and hashes the resulting flat tuple.  This is
        deterministic within a Python session — sufficient for an
        in-memory cache.
        """
        return hash((
            self.shape,
            self.stride,
            self.storage_offset,
            self.aligned,
            self.dtype,
            self.broadcast_dims,
        ))

    def __eq__(self, other: object) -> bool:
        """Field-by-field equality — two fingerprints match iff every
        physical-layout property is identical."""
        if not isinstance(other, TensorFingerprint):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.stride == other.stride
            and self.storage_offset == other.storage_offset
            and self.aligned == other.aligned
            and self.dtype == other.dtype
            and self.broadcast_dims == other.broadcast_dims
        )

    def __lt__(self, other: object) -> bool:
        """Total ordering for deterministic sorting of fingerprint tuples.

        Required so that :class:`KernelCacheKey` can sort its input
        fingerprints before hashing, making the cache key immune to
        graph node reorderings introduced by AOT Autograd.
        """
        if not isinstance(other, TensorFingerprint):
            return NotImplemented
        return (
            self.shape, self.stride, self.storage_offset,
            self.aligned, self.dtype, self.broadcast_dims,
        ) < (
            other.shape, other.stride, other.storage_offset,
            other.aligned, other.dtype, other.broadcast_dims,
        )

    # ── Factory methods ────────────────────────────────────────────────

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> TensorFingerprint:
        """Build a fingerprint from a live ``torch.Tensor``.

        Extracts all physical layout properties directly from the tensor's
        ATen storage — shape, stride, storage_offset, pointer alignment,
        and dtype.

        When ``t`` is a ``FakeTensor`` (as produced by ``aot_autograd``),
        ``.data_ptr()`` is unavailable, so we default to ``aligned=True``
        (PyTorch allocations are 256-byte aligned in practice).
        """
        stride = _materialize_ints(t.stride())
        try:
            aligned = (t.data_ptr() % 16 == 0)
        except RuntimeError:
            # FakeTensor / FunctionalTensor — no real storage.
            aligned = True
        return cls(
            shape=_materialize_ints(t.shape),
            stride=stride,
            storage_offset=_materialize_int(t.storage_offset()),
            aligned=aligned,
            dtype=str(t.dtype),
            broadcast_dims=tuple(s == 0 for s in stride),
        )

    @classmethod
    def from_node(cls, node: torch.fx.Node) -> Optional[TensorFingerprint]:
        """Build a fingerprint from an FX node's attached metadata.

        Consults ``tensor_meta`` (from ``ShapeProp``) first, then ``val``
        (``FakeTensor`` from ``torch.compile``).

        Uses ``storage_offset=0`` and ``aligned=True`` as defaults because
        intermediate computation results produced by PyTorch ops are freshly
        allocated with naturally aligned, zero-offset storage.

        Returns ``None`` when no usable metadata is present.
        """
        # Try tensor_meta first (populated by ShapeProp).
        meta = node.meta.get("tensor_meta")
        if meta is not None:
            if not hasattr(meta, "shape"):
                if isinstance(meta, (tuple, list)) and len(meta) > 0:
                    meta = meta[0]
                else:
                    meta = None

        if meta is not None and hasattr(meta, "shape"):
            stride = _materialize_ints(meta.stride)
            return cls(
                shape=_materialize_ints(meta.shape),
                stride=stride,
                storage_offset=0,
                aligned=True,
                dtype=str(meta.dtype),
                broadcast_dims=tuple(s == 0 for s in stride),
            )

        # Fall back to FakeTensor stored by torch.compile.
        val = node.meta.get("val")
        if val is not None and hasattr(val, "shape"):
            stride = (
                _materialize_ints(val.stride())
                if callable(getattr(val, "stride", None))
                else (1,) * len(val.shape)
            )
            offset = (
                _materialize_int(val.storage_offset())
                if callable(getattr(val, "storage_offset", None))
                else 0
            )
            return cls(
                shape=_materialize_ints(val.shape),
                stride=stride,
                storage_offset=offset,
                aligned=True,  # FakeTensors don't have a real data_ptr
                dtype=str(val.dtype),
                broadcast_dims=tuple(s == 0 for s in stride),
            )

        return None


# ---------------------------------------------------------------------------
# KernelCacheKey
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelCacheKey:
    """Unique cache key for a compiled fused Triton kernel.

    Deterministically hashes the full tensor configuration — operator
    topology, physical memory layout, storage offsets, pointer alignment,
    data types, and GPU compute capability — so that kernels compiled for
    one configuration are never dispatched on a different one, preventing
    out-of-bounds memory accesses and silent miscalculations.

    The ``__hash__`` method uses SHA-256 over a canonical byte
    representation of all fields, ensuring deterministic, collision-
    resistant hashing that is immune to FX graph node renamings.  Input
    fingerprints are sorted before hashing to guarantee order-invariance
    against tracing artifacts.

    Attributes
    ----------
    op_chain :
        Canonicalized operator sequence as a tuple of strings, e.g.
        ``("aten.addmm.default", "aten.gelu.default")``.
    input_fingerprints :
        Tuple of :class:`TensorFingerprint` objects, one per kernel input
        tensor.  Sorted during hashing and equality comparison for
        determinism against graph node reorderings.
    output_shapes :
        Shapes of the kernel's output tensor(s).
    output_dtypes :
        String representations of the output ``torch.dtype``\\(s).
    device_capability :
        GPU compute capability from ``torch.cuda.get_device_capability()``,
        e.g. ``(8, 0)`` for A100 or ``(8, 9)`` for Ada Lovelace.
        ``(0, 0)`` for CPU-only configurations.
    """

    op_chain: Tuple[str, ...]
    input_fingerprints: Tuple[TensorFingerprint, ...]
    output_shapes: Tuple[Tuple[int, ...], ...]
    output_dtypes: Tuple[str, ...]
    device_capability: Tuple[int, int]

    def __hash__(self) -> int:
        """SHA-256 based deterministic hash over all fields.

        Constructs a canonical byte representation from:

        1. The canonicalized ``op_chain`` tuple.
        2. A sorted tuple of :class:`TensorFingerprint` dataclasses
           (order-invariant against graph node renamings).
        3. Output shapes and dtypes.
        4. GPU device capability.

        Returns a 64-bit signed integer derived from the first 8 bytes of
        the SHA-256 digest.  Deterministic across Python sessions.
        """
        sorted_fps = tuple(sorted(self.input_fingerprints))
        canonical = repr((
            self.op_chain,
            sorted_fps,
            self.output_shapes,
            self.output_dtypes,
            self.device_capability,
        ))
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=True)

    def __eq__(self, other: object) -> bool:
        """Field-by-field equality with sorted input fingerprints — ensures
        cache lookups never conflate two distinct kernel configurations
        and are immune to graph node reorderings."""
        if not isinstance(other, KernelCacheKey):
            return NotImplemented
        return (
            self.op_chain == other.op_chain
            and sorted(self.input_fingerprints) == sorted(other.input_fingerprints)
            and self.output_shapes == other.output_shapes
            and self.output_dtypes == other.output_dtypes
            and self.device_capability == other.device_capability
        )


# ---------------------------------------------------------------------------
# KernelCache
# ---------------------------------------------------------------------------

class KernelCache:
    """In-memory cache mapping :class:`KernelCacheKey` → compiled launcher.

    On a cache miss the caller triggers kernel generation and stores the
    result via :meth:`store`.  On a hit the previously compiled executable
    is returned directly, avoiding redundant Triton code generation and JIT
    compilation.

    The cache does **not** force ``.contiguous()`` on input tensors — it
    caches and dispatches strided kernel variants whenever the operator
    chain is mathematically valid for the given layout, avoiding
    unnecessary HBM copy overhead.
    """

    def __init__(self) -> None:
        self._store: Dict[KernelCacheKey, Any] = {}
        self._hits: int = 0
        self._misses: int = 0

    def lookup(self, key: KernelCacheKey) -> Any | None:
        """Return the cached launcher for *key*, or ``None`` on a miss."""
        launcher = self._store.get(key)
        if launcher is not None:
            self._hits += 1
            logger.debug(
                "Kernel cache HIT for %s (total hits=%d, size=%d)",
                "->".join(key.op_chain), self._hits, len(self._store),
            )
            return launcher
        self._misses += 1
        logger.debug(
            "Kernel cache MISS for %s (total misses=%d, size=%d)",
            "->".join(key.op_chain), self._misses, len(self._store),
        )
        return None

    def store(self, key: KernelCacheKey, launcher: Any) -> None:
        """Store a compiled launcher under *key*."""
        self._store[key] = launcher
        logger.debug(
            "Cached kernel for %s (cache size=%d)",
            "->".join(key.op_chain), len(self._store),
        )

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._store)

    @property
    def hits(self) -> int:
        """Total number of cache hits since creation or last :meth:`clear`."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total number of cache misses since creation or last :meth:`clear`."""
        return self._misses

    def clear(self) -> None:
        """Drop all cached entries and reset hit/miss counters."""
        self._store.clear()
        self._hits = 0
        self._misses = 0


# ---------------------------------------------------------------------------
# Device capability helper
# ---------------------------------------------------------------------------

def _get_device_capability(
    tensor_map: Dict[str, torch.Tensor],
) -> Tuple[int, int]:
    """Extract CUDA device capability from the first CUDA tensor found.

    Returns ``(0, 0)`` for CPU-only configurations where no CUDA tensor
    is available.
    """
    for t in tensor_map.values():
        if isinstance(t, torch.Tensor) and t.is_cuda:
            return torch.cuda.get_device_capability(t.device)
    return (0, 0)


# ---------------------------------------------------------------------------
# Key construction helpers
# ---------------------------------------------------------------------------

def build_op_chain(group: Any) -> Tuple[str, ...]:
    """Build a stable operator-chain tuple from a FusionGroup.

    Returns a tuple of **canonical** string representations of every
    ``call_function`` target in execution order.
    Uses :func:`~fuseml.passes.topology.canonicalize_target` as the
    single source of truth for target → string conversion, ensuring
    consistency with :attr:`FusionGroup.op_signature`.

    The resulting tuple is completely independent of ``node.name``,
    placeholder naming conventions (``primals_*``, ``tangents_*``),
    and any other AOT Autograd tracing artifact.

    Parameters
    ----------
    group :
        A :class:`~fuseml.fusion_group.FusionGroup` (typed as ``Any`` to
        avoid circular imports).

    Returns
    -------
    Tuple[str, ...]
        e.g. ``("aten.addmm.default", "aten.gelu.default")``.
    """
    from fuseml.passes.topology import canonicalize_target

    return tuple(
        canonicalize_target(node.target)
        for node in group.all_nodes
        if node.op == "call_function"
    )


def build_cache_key(
    group: Any,
    tensor_map: Dict[str, torch.Tensor],
    output_shape: Tuple[int, ...] | None = None,
    output_dtype: str | None = None,
) -> KernelCacheKey | None:
    """Construct a :class:`KernelCacheKey` from a FusionGroup and live tensors.

    For each fusion-group input, the fingerprint is built from the live
    tensor in *tensor_map* when available (capturing the true storage
    offset and pointer alignment), falling back to FX node metadata for
    intermediate nodes (which are freshly allocated by PyTorch and
    therefore use offset=0 / aligned=True defaults).

    Parameters
    ----------
    group :
        The :class:`~fuseml.fusion_group.FusionGroup` being compiled.
    tensor_map :
        Mapping from FX node name → live ``torch.Tensor`` for all
        graph-level placeholder inputs.
    output_shape :
        Override for the output shape.  When ``None``, extracted from
        ``group.output_metadata`` or ``group.output_node``.
    output_dtype :
        Override for the output dtype string.  When ``None``, extracted
        from ``group.output_metadata`` or ``group.output_node``.

    Returns
    -------
    KernelCacheKey | None
        The fully constructed key, or ``None`` if required metadata is
        missing (the caller should skip caching and fall through to
        unconditional generation).
    """
    op_chain = build_op_chain(group)

    # ── Input fingerprints ────────────────────────────────────────────
    fingerprints: List[TensorFingerprint] = []
    for input_node in group.inputs:
        tensor = tensor_map.get(input_node.name)
        if tensor is not None:
            fp = TensorFingerprint.from_tensor(tensor)
        else:
            fp = TensorFingerprint.from_node(input_node)
            if fp is None:
                logger.debug(
                    "Cannot fingerprint input node %s — skipping cache key.",
                    input_node.name,
                )
                return None
        fingerprints.append(fp)

    # ── Output metadata ───────────────────────────────────────────────
    resolved_shape = output_shape
    resolved_dtype = output_dtype

    if resolved_shape is None or resolved_dtype is None:
        out_meta = getattr(group, "output_metadata", {})
        if out_meta and "shape" in out_meta:
            resolved_shape = resolved_shape or tuple(out_meta["shape"])
            resolved_dtype = resolved_dtype or str(out_meta["dtype"])
        else:
            # Fall back to the output node's FX metadata.
            fp = TensorFingerprint.from_node(group.output_node)
            if fp is None:
                logger.debug(
                    "Cannot determine output metadata for cache key."
                )
                return None
            resolved_shape = resolved_shape or fp.shape
            resolved_dtype = resolved_dtype or fp.dtype

    # ── Device capability ─────────────────────────────────────────────
    device_capability = _get_device_capability(tensor_map)

    return KernelCacheKey(
        op_chain=op_chain,
        input_fingerprints=tuple(fingerprints),
        output_shapes=(resolved_shape,),
        output_dtypes=(resolved_dtype,),
        device_capability=device_capability,
    )
