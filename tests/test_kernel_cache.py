"""Tests for KernelCacheKey, TensorFingerprint, and KernelCache.

Validates deterministic SHA-256 hashing, field-by-field equality, cache
hit/miss behaviour, sorted-fingerprint order-invariance, device capability
keying, and correct differentiation of tensor memory layouts (strides,
storage offsets, pointer alignment, dtypes, broadcast patterns).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fuseml.codegen.kernel_cache import (
    KernelCache,
    KernelCacheKey,
    TensorFingerprint,
    build_cache_key,
    build_op_chain,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_node(
    name: str,
    op: str = "call_function",
    target: str = "aten.add.Tensor",
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
) -> SimpleNamespace:
    """Create a minimal mock FX node with tensor_meta in .meta."""
    meta: dict = {}
    if shape is not None:
        stride = stride or tuple(
            _compute_contiguous_strides(shape)
        )
        meta["tensor_meta"] = SimpleNamespace(
            shape=shape,
            stride=stride,
            dtype=dtype,
        )
    return SimpleNamespace(name=name, op=op, target=target, meta=meta, args=(), users={})


def _compute_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute row-major contiguous strides for *shape*."""
    if not shape:
        return ()
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def _make_fingerprint(**overrides) -> TensorFingerprint:
    """Build a TensorFingerprint with sensible defaults, applying overrides."""
    defaults = dict(
        shape=(4, 128),
        stride=(128, 1),
        storage_offset=0,
        aligned=True,
        dtype="torch.float32",
        broadcast_dims=(False, False),
    )
    defaults.update(overrides)
    return TensorFingerprint(**defaults)


def _make_key(**overrides) -> KernelCacheKey:
    """Build a KernelCacheKey with sensible defaults, applying overrides."""
    defaults = dict(
        op_chain=("aten.addmm.default", "aten.gelu.default"),
        input_fingerprints=(
            _make_fingerprint(shape=(4, 128), stride=(128, 1)),
            _make_fingerprint(shape=(128, 128), stride=(128, 1)),
        ),
        output_shapes=((4, 128),),
        output_dtypes=("torch.float32",),
        device_capability=(0, 0),
    )
    defaults.update(overrides)
    return KernelCacheKey(**defaults)


# ======================================================================
# TensorFingerprint — creation
# ======================================================================

@pytest.mark.cache
class TestTensorFingerprintCreation:
    """TensorFingerprint.from_tensor and from_node factory methods."""

    def test_from_tensor_contiguous(self):
        t = torch.randn(4, 128)
        fp = TensorFingerprint.from_tensor(t)
        assert fp.shape == (4, 128)
        assert fp.stride == (128, 1)
        assert fp.storage_offset == 0
        assert fp.dtype == "torch.float32"

    def test_from_tensor_preserves_strides(self):
        t = torch.randn(128, 4).t()  # transposed → non-contiguous
        fp = TensorFingerprint.from_tensor(t)
        assert fp.shape == (4, 128)
        assert fp.stride == (1, 4)  # transposed strides

    def test_from_tensor_storage_offset(self):
        base = torch.randn(8, 128)
        view = base[2:]  # storage_offset = 2 * 128 = 256
        fp = TensorFingerprint.from_tensor(view)
        assert fp.shape == (6, 128)
        assert fp.storage_offset == 256

    def test_from_tensor_alignment(self):
        t = torch.randn(4, 128)
        fp = TensorFingerprint.from_tensor(t)
        # Freshly allocated tensors are 16-byte aligned.
        assert fp.aligned is True

    def test_from_tensor_misaligned(self):
        base = torch.randn(65)
        # Slice off one float32 (4 bytes) to potentially misalign.
        view = base[1:]
        fp = TensorFingerprint.from_tensor(view)
        # data_ptr shifted by 4 bytes; may or may not be 16-aligned
        # depending on allocator — just verify the field is a bool.
        assert isinstance(fp.aligned, bool)
        assert fp.storage_offset == 1

    def test_from_tensor_dtype(self):
        t = torch.randn(4, 4, dtype=torch.float16)
        fp = TensorFingerprint.from_tensor(t)
        assert fp.dtype == "torch.float16"

    def test_from_tensor_no_subclass_field(self):
        """TensorFingerprint must not store tensor_subclass — pure physical."""
        t = torch.randn(4, 4)
        fp = TensorFingerprint.from_tensor(t)
        assert not hasattr(fp, "tensor_subclass")

    def test_from_node_with_tensor_meta(self):
        node = _make_node("x", shape=(8, 64), dtype=torch.float16)
        fp = TensorFingerprint.from_node(node)
        assert fp is not None
        assert fp.shape == (8, 64)
        assert fp.stride == (64, 1)
        assert fp.storage_offset == 0
        assert fp.aligned is True
        assert fp.dtype == "torch.float16"

    def test_from_node_no_metadata_returns_none(self):
        node = SimpleNamespace(name="x", meta={})
        assert TensorFingerprint.from_node(node) is None

    def test_from_node_with_val_metadata(self):
        """Fall back to node.meta['val'] when tensor_meta is absent."""
        fake = SimpleNamespace(
            shape=(2, 32),
            stride=lambda: (32, 1),
            storage_offset=lambda: 0,
            dtype=torch.float32,
        )
        node = SimpleNamespace(name="y", meta={"val": fake})
        fp = TensorFingerprint.from_node(node)
        assert fp is not None
        assert fp.shape == (2, 32)
        assert fp.stride == (32, 1)

    def test_from_node_tuple_tensor_meta(self):
        """tensor_meta may be a tuple of TensorMetadata — unwrap first."""
        inner = SimpleNamespace(
            shape=(3, 16), stride=(16, 1), dtype=torch.float32,
        )
        node = SimpleNamespace(
            name="z", meta={"tensor_meta": (inner,)}
        )
        fp = TensorFingerprint.from_node(node)
        assert fp is not None
        assert fp.shape == (3, 16)


# ======================================================================
# TensorFingerprint — hash & equality
# ======================================================================

@pytest.mark.cache
class TestTensorFingerprintHashEq:
    """Deterministic __hash__ and __eq__ for TensorFingerprint."""

    def test_same_config_equal(self):
        a = _make_fingerprint()
        b = _make_fingerprint()
        assert a == b
        assert hash(a) == hash(b)

    def test_same_hash_across_calls(self):
        fp = _make_fingerprint()
        assert hash(fp) == hash(fp)

    def test_different_shape(self):
        a = _make_fingerprint(shape=(4, 128), stride=(128, 1))
        b = _make_fingerprint(shape=(8, 128), stride=(128, 1))
        assert a != b

    def test_different_stride(self):
        a = _make_fingerprint(stride=(128, 1))
        b = _make_fingerprint(stride=(1, 4))  # transposed
        assert a != b

    def test_different_storage_offset(self):
        a = _make_fingerprint(storage_offset=0)
        b = _make_fingerprint(storage_offset=256)
        assert a != b

    def test_different_alignment(self):
        a = _make_fingerprint(aligned=True)
        b = _make_fingerprint(aligned=False)
        assert a != b

    def test_different_dtype(self):
        a = _make_fingerprint(dtype="torch.float32")
        b = _make_fingerprint(dtype="torch.float16")
        assert a != b

    def test_not_equal_to_non_fingerprint(self):
        fp = _make_fingerprint()
        assert fp != "not a fingerprint"
        assert fp.__eq__("not a fingerprint") is NotImplemented

    def test_usable_as_dict_key(self):
        fp = _make_fingerprint()
        d = {fp: "value"}
        assert d[_make_fingerprint()] == "value"

    def test_usable_in_set(self):
        s = {_make_fingerprint(), _make_fingerprint()}
        assert len(s) == 1

    def test_lt_ordering(self):
        """TensorFingerprint supports __lt__ for deterministic sorting."""
        fp_small = _make_fingerprint(shape=(2, 64), stride=(64, 1))
        fp_large = _make_fingerprint(shape=(4, 128), stride=(128, 1))
        assert fp_small < fp_large
        assert not fp_large < fp_small

    def test_lt_returns_not_implemented_for_non_fingerprint(self):
        fp = _make_fingerprint()
        assert fp.__lt__("not a fingerprint") is NotImplemented

    def test_sorted_fingerprints_deterministic(self):
        """sorted() on fingerprints produces a consistent ordering."""
        fps = [
            _make_fingerprint(shape=(128, 128), stride=(128, 1)),
            _make_fingerprint(shape=(4, 128), stride=(128, 1)),
            _make_fingerprint(shape=(64, 64), stride=(64, 1)),
        ]
        assert sorted(fps) == sorted(fps)
        assert sorted(fps)[0].shape == (4, 128)


# ======================================================================
# KernelCacheKey — hash & equality
# ======================================================================

@pytest.mark.cache
class TestKernelCacheKeyHashEq:
    """Deterministic SHA-256 __hash__ and __eq__ for KernelCacheKey."""

    def test_identical_keys_equal(self):
        a = _make_key()
        b = _make_key()
        assert a == b
        assert hash(a) == hash(b)

    def test_hash_deterministic(self):
        k = _make_key()
        assert hash(k) == hash(k)

    def test_hash_is_int(self):
        """SHA-256 based hash must return a proper int."""
        k = _make_key()
        assert isinstance(hash(k), int)

    def test_different_op_chain(self):
        a = _make_key(op_chain=("aten.addmm.default", "aten.gelu.default"))
        b = _make_key(op_chain=("aten.addmm.default", "aten.relu.default"))
        assert a != b

    def test_different_input_fingerprints(self):
        fp_a = _make_fingerprint(shape=(4, 128), stride=(128, 1))
        fp_b = _make_fingerprint(shape=(4, 64), stride=(64, 1))
        a = _make_key(input_fingerprints=(fp_a,))
        b = _make_key(input_fingerprints=(fp_b,))
        assert a != b

    def test_different_output_shapes(self):
        a = _make_key(output_shapes=((4, 128),))
        b = _make_key(output_shapes=((8, 128),))
        assert a != b

    def test_different_output_dtypes(self):
        a = _make_key(output_dtypes=("torch.float32",))
        b = _make_key(output_dtypes=("torch.float16",))
        assert a != b

    def test_different_device_capability(self):
        a = _make_key(device_capability=(0, 0))
        b = _make_key(device_capability=(8, 0))
        assert a != b

    def test_not_equal_to_non_key(self):
        k = _make_key()
        assert k != 42
        assert k.__eq__(42) is NotImplemented

    def test_usable_as_dict_key(self):
        k = _make_key()
        d = {k: "launcher"}
        assert d[_make_key()] == "launcher"

    def test_fingerprint_order_invariant(self):
        """Sorted fingerprints make keys immune to input reordering."""
        fp_a = _make_fingerprint(shape=(4, 128), stride=(128, 1))
        fp_b = _make_fingerprint(shape=(128, 128), stride=(128, 1))
        k1 = _make_key(input_fingerprints=(fp_a, fp_b))
        k2 = _make_key(input_fingerprints=(fp_b, fp_a))
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_storage_offset_differentiates_keys(self):
        fp_zero = _make_fingerprint(storage_offset=0)
        fp_offset = _make_fingerprint(storage_offset=512)
        a = _make_key(input_fingerprints=(fp_zero,))
        b = _make_key(input_fingerprints=(fp_offset,))
        assert a != b

    def test_alignment_differentiates_keys(self):
        fp_aligned = _make_fingerprint(aligned=True)
        fp_misaligned = _make_fingerprint(aligned=False)
        a = _make_key(input_fingerprints=(fp_aligned,))
        b = _make_key(input_fingerprints=(fp_misaligned,))
        assert a != b

    def test_device_capability_a100_vs_ada(self):
        """Different GPU architectures produce different cache keys."""
        a = _make_key(device_capability=(8, 0))   # A100
        b = _make_key(device_capability=(8, 9))   # Ada Lovelace
        assert a != b
        assert hash(a) != hash(b)


# ======================================================================
# KernelCache — lookup, store, counters
# ======================================================================

@pytest.mark.cache
class TestKernelCache:
    """KernelCache lookup/store and hit/miss tracking."""

    def test_empty_cache_miss(self):
        cache = KernelCache()
        assert cache.lookup(_make_key()) is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_store_then_hit(self):
        cache = KernelCache()
        key = _make_key()
        sentinel = object()
        cache.store(key, sentinel)
        assert cache.lookup(key) is sentinel
        assert cache.hits == 1
        assert cache.size == 1

    def test_different_key_misses(self):
        cache = KernelCache()
        key_a = _make_key(op_chain=("a",))
        key_b = _make_key(op_chain=("b",))
        cache.store(key_a, "launcher_a")
        assert cache.lookup(key_b) is None
        assert cache.misses == 1

    def test_multiple_entries(self):
        cache = KernelCache()
        k1 = _make_key(op_chain=("chain1",))
        k2 = _make_key(op_chain=("chain2",))
        cache.store(k1, "L1")
        cache.store(k2, "L2")
        assert cache.size == 2
        assert cache.lookup(k1) == "L1"
        assert cache.lookup(k2) == "L2"
        assert cache.hits == 2

    def test_overwrite_entry(self):
        cache = KernelCache()
        key = _make_key()
        cache.store(key, "old")
        cache.store(key, "new")
        assert cache.lookup(key) == "new"
        assert cache.size == 1

    def test_clear_resets_everything(self):
        cache = KernelCache()
        key = _make_key()
        cache.store(key, "val")
        cache.lookup(key)  # hit
        cache.lookup(_make_key(op_chain=("miss",)))  # miss
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.lookup(key) is None

    def test_hit_miss_counters(self):
        cache = KernelCache()
        key = _make_key()
        cache.store(key, "val")

        for _ in range(5):
            cache.lookup(key)
        for _ in range(3):
            cache.lookup(_make_key(op_chain=("nonexistent",)))

        assert cache.hits == 5
        assert cache.misses == 3


# ======================================================================
# KernelCache — strided layout differentiation
# ======================================================================

@pytest.mark.cache
class TestCacheStridedLayouts:
    """Cache must differentiate contiguous vs strided without .contiguous()."""

    def test_contiguous_vs_transposed(self):
        cache = KernelCache()

        fp_contig = _make_fingerprint(shape=(4, 128), stride=(128, 1))
        fp_trans = _make_fingerprint(shape=(4, 128), stride=(1, 4))

        key_contig = _make_key(input_fingerprints=(fp_contig,))
        key_trans = _make_key(input_fingerprints=(fp_trans,))

        cache.store(key_contig, "contig_launcher")
        cache.store(key_trans, "transposed_launcher")

        assert cache.lookup(key_contig) == "contig_launcher"
        assert cache.lookup(key_trans) == "transposed_launcher"
        assert cache.size == 2

    def test_view_with_offset_vs_base(self):
        base = torch.randn(8, 128)
        view = base[2:]  # offset = 256

        fp_base = TensorFingerprint.from_tensor(base)
        fp_view = TensorFingerprint.from_tensor(view)

        assert fp_base.storage_offset == 0
        assert fp_view.storage_offset == 256
        assert fp_base != fp_view

    def test_same_shape_different_strides_cached_separately(self):
        cache = KernelCache()

        # Row-major (4, 128) with stride (128, 1)
        fp_row = _make_fingerprint(shape=(4, 128), stride=(128, 1))
        # Column-major (4, 128) with stride (1, 4)
        fp_col = _make_fingerprint(shape=(4, 128), stride=(1, 4))

        k_row = _make_key(input_fingerprints=(fp_row,))
        k_col = _make_key(input_fingerprints=(fp_col,))

        cache.store(k_row, "row_kernel")
        cache.store(k_col, "col_kernel")

        assert cache.lookup(k_row) == "row_kernel"
        assert cache.lookup(k_col) == "col_kernel"


# ======================================================================
# build_op_chain
# ======================================================================

@pytest.mark.cache
class TestBuildOpChain:

    def test_single_op(self):
        base = _make_node("addmm", target="aten.addmm.default")
        group = SimpleNamespace(
            all_nodes=[base],
            inputs=[],
            output_node=base,
            output_metadata={},
        )
        assert build_op_chain(group) == ("aten.addmm.default",)

    def test_multi_op_chain(self):
        n1 = _make_node("addmm", target="aten.addmm.default")
        n2 = _make_node("gelu", target="aten.gelu.default")
        n3 = _make_node("add", target="aten.add.Tensor")
        group = SimpleNamespace(
            all_nodes=[n1, n2, n3],
            inputs=[],
            output_node=n3,
            output_metadata={},
        )
        assert build_op_chain(group) == (
            "aten.addmm.default", "aten.gelu.default", "aten.add.Tensor"
        )

    def test_skips_non_call_function(self):
        n1 = _make_node("addmm", op="call_function", target="aten.addmm.default")
        n2 = _make_node("placeholder", op="placeholder", target="x")
        group = SimpleNamespace(all_nodes=[n1, n2])
        assert build_op_chain(group) == ("aten.addmm.default",)


# ======================================================================
# build_cache_key
# ======================================================================

@pytest.mark.cache
class TestBuildCacheKey:

    def test_builds_key_from_tensor_map(self):
        inp1 = _make_node("x", shape=(4, 128))
        inp2 = _make_node("w", shape=(128, 128))
        base = _make_node("addmm", target="aten.addmm.default", shape=(4, 128))
        out = _make_node("gelu", target="aten.gelu.default", shape=(4, 128))

        group = SimpleNamespace(
            all_nodes=[base, out],
            inputs=[inp1, inp2],
            output_node=out,
            output_metadata={"shape": (4, 128), "stride": (128, 1), "dtype": torch.float32},
        )

        t1 = torch.randn(4, 128)
        t2 = torch.randn(128, 128)
        tensor_map = {"x": t1, "w": t2}

        key = build_cache_key(group, tensor_map)
        assert key is not None
        assert key.op_chain == ("aten.addmm.default", "aten.gelu.default")
        assert len(key.input_fingerprints) == 2
        assert key.output_shapes == ((4, 128),)
        assert key.output_dtypes == ("torch.float32",)
        assert key.device_capability == (0, 0)  # CPU tensors

    def test_falls_back_to_node_metadata(self):
        """When a tensor is not in tensor_map, from_node is used."""
        inp = _make_node("intermediate", shape=(4, 64))
        base = _make_node("addmm", target="aten.addmm.default", shape=(4, 64))

        group = SimpleNamespace(
            all_nodes=[base],
            inputs=[inp],
            output_node=base,
            output_metadata={"shape": (4, 64), "stride": (64, 1), "dtype": torch.float32},
        )

        key = build_cache_key(group, {})  # empty tensor_map
        assert key is not None
        # from_node defaults: offset=0, aligned=True
        assert key.input_fingerprints[0].storage_offset == 0
        assert key.input_fingerprints[0].aligned is True

    def test_returns_none_when_metadata_missing(self):
        inp = SimpleNamespace(name="unknown", meta={})
        base = _make_node("addmm", target="aten.addmm.default")

        group = SimpleNamespace(
            all_nodes=[base],
            inputs=[inp],
            output_node=base,
            output_metadata={},
        )

        assert build_cache_key(group, {}) is None

    def test_captures_view_storage_offset(self):
        """Live tensor view with non-zero offset is fingerprinted correctly."""
        base_t = torch.randn(8, 128)
        view_t = base_t[4:]  # offset = 4 * 128 = 512

        inp = _make_node("x", shape=(4, 128))
        node = _make_node("addmm", target="aten.addmm.default", shape=(4, 128))

        group = SimpleNamespace(
            all_nodes=[node],
            inputs=[inp],
            output_node=node,
            output_metadata={"shape": (4, 128), "stride": (128, 1), "dtype": torch.float32},
        )

        key = build_cache_key(group, {"x": view_t})
        assert key is not None
        assert key.input_fingerprints[0].storage_offset == 512

    def test_captures_alignment(self):
        t = torch.randn(4, 128)
        inp = _make_node("x", shape=(4, 128))
        node = _make_node("addmm", target="aten.addmm.default", shape=(4, 128))

        group = SimpleNamespace(
            all_nodes=[node],
            inputs=[inp],
            output_node=node,
            output_metadata={"shape": (4, 128), "stride": (128, 1), "dtype": torch.float32},
        )

        key = build_cache_key(group, {"x": t})
        assert key is not None
        assert key.input_fingerprints[0].aligned == (t.data_ptr() % 16 == 0)

    def test_output_metadata_from_node_fallback(self):
        """When output_metadata is empty, from_node on output_node is used."""
        inp = _make_node("x", shape=(4, 128))
        node = _make_node("addmm", target="aten.addmm.default", shape=(4, 128))

        group = SimpleNamespace(
            all_nodes=[node],
            inputs=[inp],
            output_node=node,
            output_metadata={},  # empty
        )

        t = torch.randn(4, 128)
        key = build_cache_key(group, {"x": t})
        assert key is not None
        assert key.output_shapes == ((4, 128),)

    def test_device_capability_from_cpu_tensor_map(self):
        t = torch.randn(4, 128)  # CPU tensor
        inp = _make_node("x", shape=(4, 128))
        node = _make_node("addmm", target="aten.addmm.default", shape=(4, 128))

        group = SimpleNamespace(
            all_nodes=[node],
            inputs=[inp],
            output_node=node,
            output_metadata={"shape": (4, 128), "stride": (128, 1), "dtype": torch.float32},
        )

        key = build_cache_key(group, {"x": t})
        assert key is not None
        assert key.device_capability == (0, 0)  # CPU → (0, 0)


# ======================================================================
# Integration: FuseMLCompiler._cache attribute
# ======================================================================

@pytest.mark.cache
class TestCompilerCacheAttribute:
    """Verify that FuseMLCompiler owns a KernelCache instance."""

    def test_compiler_has_cache(self):
        from fuseml.compiler import FuseMLCompiler
        compiler = FuseMLCompiler()
        assert isinstance(compiler._cache, KernelCache)
        assert compiler._cache.size == 0


# ======================================================================
# SymInt unpacking
# ======================================================================

class _FakeSymInt:
    """Mock SymInt-like wrapper that stores an int but is not ``int``."""

    def __init__(self, val: int) -> None:
        self._val = val

    def __int__(self) -> int:
        return self._val

    def __index__(self) -> int:
        return self._val

    def __repr__(self) -> str:
        return f"SymInt({self._val})"


@pytest.mark.cache
class TestSymIntUnpacking:
    """SymInt values must be force-evaluated to concrete int in fingerprints."""

    def test_from_tensor_resolves_symint_shape(self):
        t = torch.randn(4, 128)
        fp = TensorFingerprint.from_tensor(t)
        # All shape/stride elements must be plain int.
        assert all(type(s) is int for s in fp.shape)
        assert all(type(s) is int for s in fp.stride)
        assert type(fp.storage_offset) is int

    def test_from_node_resolves_symint_shape(self):
        """SymInt-like values in tensor_meta.shape are resolved to int."""
        sym_shape = (_FakeSymInt(8), _FakeSymInt(64))
        sym_stride = (_FakeSymInt(64), _FakeSymInt(1))
        meta = SimpleNamespace(shape=sym_shape, stride=sym_stride, dtype=torch.float32)
        node = SimpleNamespace(name="x", meta={"tensor_meta": meta})

        fp = TensorFingerprint.from_node(node)
        assert fp is not None
        assert fp.shape == (8, 64)
        assert fp.stride == (64, 1)
        assert all(type(s) is int for s in fp.shape)
        assert all(type(s) is int for s in fp.stride)

    def test_symint_fingerprints_equal_and_hashable(self):
        """Fingerprints from SymInt-wrapped and plain-int values must match."""
        fp_plain = _make_fingerprint(shape=(8, 64), stride=(64, 1))

        sym_shape = (_FakeSymInt(8), _FakeSymInt(64))
        sym_stride = (_FakeSymInt(64), _FakeSymInt(1))
        meta = SimpleNamespace(shape=sym_shape, stride=sym_stride, dtype=torch.float32)
        node = SimpleNamespace(name="x", meta={"tensor_meta": meta})
        fp_sym = TensorFingerprint.from_node(node)

        assert fp_sym == fp_plain
        assert hash(fp_sym) == hash(fp_plain)

    def test_symint_cache_key_round_trip(self):
        """Store with SymInt-derived key, lookup with plain-int key — must hit."""
        cache = KernelCache()

        # Build fingerprint from SymInt-like node metadata.
        sym_shape = (_FakeSymInt(4), _FakeSymInt(128))
        sym_stride = (_FakeSymInt(128), _FakeSymInt(1))
        meta = SimpleNamespace(shape=sym_shape, stride=sym_stride, dtype=torch.float32)
        node = SimpleNamespace(name="x", meta={"tensor_meta": meta})
        fp_sym = TensorFingerprint.from_node(node)

        key_sym = _make_key(input_fingerprints=(fp_sym,))
        cache.store(key_sym, "sym_launcher")

        # Lookup with a plain-int key that has the same concrete values.
        fp_plain = _make_fingerprint(shape=(4, 128), stride=(128, 1))
        key_plain = _make_key(input_fingerprints=(fp_plain,))
        assert cache.lookup(key_plain) == "sym_launcher"


# ======================================================================
# Broadcast stride differentiation
# ======================================================================

@pytest.mark.cache
class TestBroadcastStrideDifferentiation:
    """Cache must track broadcast (stride-0) dimensions explicitly."""

    def test_broadcast_dims_from_expanded_tensor(self):
        base = torch.randn(1, 64)
        expanded = base.expand(128, 64)
        fp = TensorFingerprint.from_tensor(expanded)
        assert fp.broadcast_dims == (True, False)
        assert fp.shape == (128, 64)
        assert fp.stride[0] == 0

    def test_broadcast_dims_contiguous(self):
        t = torch.randn(4, 128)
        fp = TensorFingerprint.from_tensor(t)
        assert fp.broadcast_dims == (False, False)

    def test_broadcast_vs_nonbroadcast_not_equal(self):
        fp_normal = _make_fingerprint(
            shape=(128, 64), stride=(64, 1),
            broadcast_dims=(False, False),
        )
        fp_broadcast = _make_fingerprint(
            shape=(128, 64), stride=(0, 1),
            broadcast_dims=(True, False),
        )
        assert fp_normal != fp_broadcast

    def test_broadcast_pattern_differentiates_cache_keys(self):
        cache = KernelCache()

        fp_normal = _make_fingerprint(
            shape=(128, 64), stride=(64, 1),
            broadcast_dims=(False, False),
        )
        fp_broadcast = _make_fingerprint(
            shape=(128, 64), stride=(0, 1),
            broadcast_dims=(True, False),
        )

        k_normal = _make_key(input_fingerprints=(fp_normal,))
        k_broadcast = _make_key(input_fingerprints=(fp_broadcast,))

        cache.store(k_normal, "normal_launcher")
        assert cache.lookup(k_broadcast) is None  # must miss
        assert cache.misses == 1

    def test_1d_bias_no_broadcast(self):
        t = torch.randn(64)
        fp = TensorFingerprint.from_tensor(t)
        assert fp.broadcast_dims == (False,)
        assert fp.shape == (64,)

    def test_broadcast_dims_from_node_metadata(self):
        """Stride-0 in node metadata produces broadcast_dims=(True, False)."""
        meta = SimpleNamespace(
            shape=(128, 64), stride=(0, 1), dtype=torch.float32,
        )
        node = SimpleNamespace(name="x", meta={"tensor_meta": meta})
        fp = TensorFingerprint.from_node(node)
        assert fp is not None
        assert fp.broadcast_dims == (True, False)


# ======================================================================
# SHA-256 hash determinism
# ======================================================================

@pytest.mark.cache
class TestSHA256HashDeterminism:
    """KernelCacheKey hash must be SHA-256 based and deterministic."""

    def test_hash_reproducible_across_constructions(self):
        """Two independently constructed identical keys produce the same hash."""
        h1 = hash(_make_key())
        h2 = hash(_make_key())
        assert h1 == h2

    def test_hash_stable_across_repeated_calls(self):
        """hash() on the same key object returns the same value every time."""
        k = _make_key()
        hashes = [hash(k) for _ in range(100)]
        assert len(set(hashes)) == 1

    def test_hash_changes_with_any_field(self):
        """Changing any single field must produce a different hash."""
        base_hash = hash(_make_key())

        assert hash(_make_key(op_chain=("different",))) != base_hash
        assert hash(_make_key(output_shapes=((999,),))) != base_hash
        assert hash(_make_key(output_dtypes=("torch.float16",))) != base_hash
        assert hash(_make_key(device_capability=(9, 0))) != base_hash

        fp_different = _make_fingerprint(shape=(1, 1), stride=(1, 1))
        assert hash(_make_key(input_fingerprints=(fp_different,))) != base_hash

    def test_no_fx_node_references_in_key(self):
        """Cache key fields must contain only physical layout primitives."""
        key = _make_key()
        # op_chain: tuple of strings
        assert all(isinstance(s, str) for s in key.op_chain)
        # fingerprints: physical layout only
        for fp in key.input_fingerprints:
            assert all(isinstance(d, int) for d in fp.shape)
            assert all(isinstance(d, int) for d in fp.stride)
            assert isinstance(fp.storage_offset, int)
            assert isinstance(fp.aligned, bool)
            assert isinstance(fp.dtype, str)
            assert all(isinstance(b, bool) for b in fp.broadcast_dims)
            assert not hasattr(fp, "tensor_subclass")
        # output shapes/dtypes: tuples of primitives
        for shape in key.output_shapes:
            assert all(isinstance(d, int) for d in shape)
        for dtype in key.output_dtypes:
            assert isinstance(dtype, str)
        # device_capability: tuple of ints
        assert all(isinstance(d, int) for d in key.device_capability)