"""KernelCacheKey and caching engine for compiled Triton kernels.

Prevents redundant kernel compilation by uniquely identifying each fused
kernel configuration through its operator topology, tensor memory layout,
storage offsets, pointer alignment, data types, and device.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from fuseml._logging import logger


# ---------------------------------------------------------------------------
# TensorFingerprint — per-tensor memory layout descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorFingerprint:
    """Physical memory-layout fingerprint for a single tensor.

    Captures every property of a tensor's ATen storage representation that
    affects Triton kernel correctness:

    - **shape** and **stride**: determine pointer arithmetic and tile bounds.
    - **storage_offset**: the offset (in elements) from the storage base
      pointer.  Triton kernels must add this to their base pointers to
      avoid reading/writing the wrong memory region.
    - **aligned**: whether ``data_ptr() % 16 == 0``.  Misaligned pointers
      prevent the use of vectorised loads (e.g., ``tl.load`` with wide
      vector widths), which would cause GPU memory faults.
    - **dtype**: the scalar type, stored as a string for deterministic
      hashing across sessions.
    """

    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    storage_offset: int
    aligned: bool
    dtype: str  # str(torch.dtype) for deterministic hashing

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
        )

    # ── Factory methods ────────────────────────────────────────────────

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> TensorFingerprint:
        """Build a fingerprint from a live ``torch.Tensor``.

        Extracts all physical layout properties directly from the tensor's
        ATen storage — shape, stride, storage_offset, pointer alignment,
        and dtype.
        """
        return cls(
            shape=tuple(t.shape),
            stride=tuple(t.stride()),
            storage_offset=t.storage_offset(),
            aligned=(t.data_ptr() % 16 == 0),
            dtype=str(t.dtype),
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
            return cls(
                shape=tuple(meta.shape),
                stride=tuple(meta.stride),
                storage_offset=0,
                aligned=True,
                dtype=str(meta.dtype),
            )

        # Fall back to FakeTensor stored by torch.compile.
        val = node.meta.get("val")
        if val is not None and hasattr(val, "shape"):
            stride = (
                tuple(val.stride())
                if callable(getattr(val, "stride", None))
                else (1,) * len(val.shape)
            )
            offset = (
                val.storage_offset()
                if callable(getattr(val, "storage_offset", None))
                else 0
            )
            return cls(
                shape=tuple(val.shape),
                stride=stride,
                storage_offset=offset,
                aligned=True,  # FakeTensors don't have a real data_ptr
                dtype=str(val.dtype),
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
    data types, and device — so that kernels compiled for one configuration
    are never dispatched on a different one, preventing out-of-bounds
    memory accesses and silent miscalculations.

    Attributes
    ----------
    op_chain :
        Stable string encoding the fused operator sequence, e.g.
        ``"aten.addmm.default->aten.gelu.default->aten.add.Tensor"``.
    input_fingerprints :
        Ordered tuple of :class:`TensorFingerprint` objects, one per
        kernel input tensor.
    output_shape :
        Shape of the kernel's primary output tensor.
    output_dtype :
        String representation of the output ``torch.dtype``.
    device :
        String representation of the target ``torch.device``.
    """

    op_chain: str
    input_fingerprints: Tuple[TensorFingerprint, ...]
    output_shape: Tuple[int, ...]
    output_dtype: str
    device: str

    def __hash__(self) -> int:
        """Deterministic hash over all fields.

        Flattens every constituent value into a single tuple of Python
        primitives (int, str, bool, tuple-of-int) and hashes the result.
        This avoids relying on auto-generated hashing of nested frozen
        dataclasses and makes the hash strategy explicit and auditable.

        Deterministic within a Python session — sufficient for an in-memory
        cache.  Cross-session determinism is unnecessary because the cache
        is not persisted to disk.
        """
        parts: List[Any] = [self.op_chain]
        for fp in self.input_fingerprints:
            parts.append(fp.shape)
            parts.append(fp.stride)
            parts.append(fp.storage_offset)
            parts.append(fp.aligned)
            parts.append(fp.dtype)
        parts.append(self.output_shape)
        parts.append(self.output_dtype)
        parts.append(self.device)
        return hash(tuple(parts))

    def __eq__(self, other: object) -> bool:
        """Field-by-field equality — ensures cache lookups never conflate
        two distinct kernel configurations."""
        if not isinstance(other, KernelCacheKey):
            return NotImplemented
        return (
            self.op_chain == other.op_chain
            and self.input_fingerprints == other.input_fingerprints
            and self.output_shape == other.output_shape
            and self.output_dtype == other.output_dtype
            and self.device == other.device
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
                key.op_chain, self._hits, len(self._store),
            )
            return launcher
        self._misses += 1
        logger.debug(
            "Kernel cache MISS for %s (total misses=%d, size=%d)",
            key.op_chain, self._misses, len(self._store),
        )
        return None

    def store(self, key: KernelCacheKey, launcher: Any) -> None:
        """Store a compiled launcher under *key*."""
        self._store[key] = launcher
        logger.debug(
            "Cached kernel for %s (cache size=%d)",
            key.op_chain, len(self._store),
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
# Key construction helpers
# ---------------------------------------------------------------------------

def build_op_chain(group: Any) -> str:
    """Build a stable operator-chain string from a FusionGroup.

    Concatenates the string representations of every ``call_function``
    target in execution order, joined by ``"->"``.

    Parameters
    ----------
    group :
        A :class:`~fuseml.fusion_group.FusionGroup` (typed as ``Any`` to
        avoid circular imports).

    Returns
    -------
    str
        e.g. ``"aten.addmm.default->aten.gelu.default"``.
    """
    targets: List[str] = []
    for node in group.all_nodes:
        if node.op == "call_function":
            targets.append(str(node.target))
    return "->".join(targets)


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

    # ── Device ────────────────────────────────────────────────────────
    device = "cpu"
    for t in tensor_map.values():
        if isinstance(t, torch.Tensor):
            device = str(t.device)
            break

    return KernelCacheKey(
        op_chain=op_chain,
        input_fingerprints=tuple(fingerprints),
        output_shape=resolved_shape,
        output_dtype=resolved_dtype,
        device=device,
    )
