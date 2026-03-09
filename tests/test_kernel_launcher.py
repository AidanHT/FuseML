"""Tests for KernelLauncher — runtime Triton kernel dispatch wrapper.

Run with:
    pytest tests/test_kernel_launcher.py -v
    pytest tests/ -m launcher -v
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from fuseml.codegen.kernel_generator import TensorDescriptor
from fuseml.codegen.kernel_launcher import (
    KernelLauncher,
    _DEFAULT_SRAM_BUDGET_BYTES,
    _LARGE_WORKING_SET_BYTES,
    _MIN_BLOCK_DIM,
)


# ---------------------------------------------------------------------------
# Mock kernel that records all call args and supports kernel_fn[grid](...)
# ---------------------------------------------------------------------------

class _MockKernelFn:
    """Mimics the ``kernel_fn[grid](*args, **kwargs)`` Triton launch syntax.

    Triton kernels are invoked as ``fn[grid](...)``.  ``__getitem__``
    returns a bound launcher that, when called, records the grid and all
    positional/keyword arguments.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __getitem__(self, grid):
        def _launcher(*args, **kwargs):
            resolved_grid = grid
            if callable(grid):
                resolved_grid = grid(kwargs)
            self.calls.append({
                "grid": resolved_grid,
                "args": args,
                "kwargs": kwargs,
            })
        return _launcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def input_descs():
    """Standard A(128x64) @ B(64x256) inputs."""
    a = TensorDescriptor(name="a", shape=(128, 64), stride=(64, 1), dtype=torch.float32)
    b = TensorDescriptor(name="b", shape=(64, 256), stride=(256, 1), dtype=torch.float32)
    return [a, b]


@pytest.fixture
def output_desc():
    return TensorDescriptor(name="out", shape=(128, 256), stride=(256, 1), dtype=torch.float32)


@pytest.fixture
def mock_kernel_fn():
    return _MockKernelFn()


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestKernelLauncherInit:
    """Verify construction stores fields and validates operand names."""

    def test_stores_kernel_fn(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        assert launcher._kernel_fn is mock_kernel_fn

    def test_stores_left_right_indices(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        assert launcher._left_idx == 0
        assert launcher._right_idx == 1

    def test_raises_on_invalid_left_name(self, mock_kernel_fn, input_descs, output_desc):
        with pytest.raises(ValueError, match="left_name='nonexistent'"):
            KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "nonexistent", "b")

    def test_raises_on_invalid_right_name(self, mock_kernel_fn, input_descs, output_desc):
        with pytest.raises(ValueError, match="right_name='nonexistent'"):
            KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "nonexistent")

    def test_default_block_sizes(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        assert launcher._block_size_m == 64
        assert launcher._block_size_n == 64
        assert launcher._block_size_k == 32

    def test_custom_block_sizes(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=128, block_size_n=128, block_size_k=64,
        )
        assert launcher._block_size_m == 128
        assert launcher._block_size_n == 128
        assert launcher._block_size_k == 64


# ---------------------------------------------------------------------------
# __call__ — runtime dispatch
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestKernelLauncherCall:
    """Verify __call__ extracts dims, allocates output, assembles args, and launches."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        """Inject a mock triton module so the deferred ``import triton`` succeeds."""
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_returns_tensor_with_correct_shape(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (128, 256)

    def test_returns_tensor_with_correct_dtype(self, mock_kernel_fn, input_descs):
        out_fp16 = TensorDescriptor("out", (128, 256), (256, 1), torch.float16)
        launcher = KernelLauncher(mock_kernel_fn, input_descs, out_fp16, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.dtype == torch.float16

    def test_raises_on_wrong_input_count(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        with pytest.raises(ValueError, match="Expected 2 input tensor"):
            launcher(torch.randn(128, 64))

    def test_extracts_mnk_correctly(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # M=128, N=256, K=64 appear after pointers in the positional args.
        # Pointers: 2 inputs + 1 output = 3, then M, N, K.
        assert call["args"][3] == 128  # M
        assert call["args"][4] == 256  # N
        assert call["args"][5] == 64   # K

    def test_output_device_matches_input(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.device == a.device

    def test_correct_number_of_pointer_args(self, mock_kernel_fn, input_descs, output_desc):
        """2 inputs + 1 output = 3 tensor args before M, N, K."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # args[0], args[1] = input tensors; args[2] = output tensor; args[3..5] = M,N,K
        # Triton 3.x expects actual torch.Tensor objects as pointer parameters.
        for i in range(3):
            assert isinstance(call["args"][i], torch.Tensor)

    def test_passes_strides_for_each_input(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # After 3 ptrs + M,N,K = 6 args, strides follow:
        # a has 2 strides, b has 2 strides, out has 2 strides = 6 stride args total
        stride_start = 6  # 3 ptrs + 3 dims
        a_stride_m = call["args"][stride_start]
        a_stride_k = call["args"][stride_start + 1]
        assert a_stride_m == a.stride(0)
        assert a_stride_k == a.stride(1)
        b_stride_k = call["args"][stride_start + 2]
        b_stride_n = call["args"][stride_start + 3]
        assert b_stride_k == b.stride(0)
        assert b_stride_n == b.stride(1)

    def test_noncontiguous_strides_propagated(self, mock_kernel_fn, input_descs, output_desc):
        """Non-contiguous tensors pass their actual strides, not assumed row-major."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a_contig = torch.randn(64, 128).T  # shape (128,64) with stride (1,128) — column-major
        b = torch.randn(64, 256)
        assert not a_contig.is_contiguous()
        launcher(a_contig, b)
        call = mock_kernel_fn.calls[0]
        stride_start = 6
        assert call["args"][stride_start] == a_contig.stride(0)       # 1 (non-contiguous)
        assert call["args"][stride_start + 1] == a_contig.stride(1)   # 128

    def test_block_sizes_passed_as_kwargs(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=32, block_size_n=64, block_size_k=16,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        assert call["kwargs"]["BLOCK_SIZE_M"] == 32
        assert call["kwargs"]["BLOCK_SIZE_N"] == 64
        assert call["kwargs"]["BLOCK_SIZE_K"] == 16


# ---------------------------------------------------------------------------
# __call__ with intermediate (escape) tensors
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestKernelLauncherCallWithIntermediates:
    """Verify intermediate escape buffers are allocated and their args passed."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_one_intermediate_adds_ptr_and_strides(self, mock_kernel_fn, input_descs, output_desc):
        intm = TensorDescriptor("esc0", (128, 256), (256, 1), torch.float32)
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [intm], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # Pointers: 2 inputs + 1 output + 1 intermediate = 4
        # Then M, N, K = 3
        # Strides: a(2) + b(2) + out(2) + intm(2) = 8
        # Total positional args = 4 + 3 + 8 = 15
        assert len(call["args"]) == 15

    def test_two_intermediates_adds_two_ptrs_and_four_strides(self, mock_kernel_fn, input_descs, output_desc):
        intm0 = TensorDescriptor("esc0", (128, 256), (256, 1), torch.float32)
        intm1 = TensorDescriptor("esc1", (128, 256), (256, 1), torch.float16)
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [intm0, intm1], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # Pointers: 2 + 1 + 2 = 5
        # Then M, N, K = 3
        # Strides: a(2) + b(2) + out(2) + intm0(2) + intm1(2) = 10
        # Total positional args = 5 + 3 + 10 = 18
        assert len(call["args"]) == 18


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestKernelLauncherRepr:
    """Verify repr includes essential diagnostic info."""

    def test_repr_contains_input_names(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        r = repr(launcher)
        assert "'a'" in r
        assert "'b'" in r

    def test_repr_contains_output_name(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        r = repr(launcher)
        assert "'out'" in r

    def test_repr_contains_block_sizes(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=32, block_size_n=64, block_size_k=16,
        )
        r = repr(launcher)
        assert "(32,64,16)" in r


# ---------------------------------------------------------------------------
# Power-of-two block size enforcement
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestBlockSizePowerOfTwo:
    """Verify KernelLauncher rounds block sizes up to the next power of two."""

    def test_already_power_of_two_unchanged(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=64, block_size_n=128, block_size_k=32,
        )
        assert launcher._block_size_m == 64
        assert launcher._block_size_n == 128
        assert launcher._block_size_k == 32

    def test_non_power_of_two_rounded_up(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=50, block_size_n=100, block_size_k=24,
        )
        assert launcher._block_size_m == 64
        assert launcher._block_size_n == 128
        assert launcher._block_size_k == 32

    def test_default_block_sizes_are_power_of_two(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
        )
        for bs in (launcher._block_size_m, launcher._block_size_n, launcher._block_size_k):
            assert bs & (bs - 1) == 0, f"{bs} is not a power of 2"


# ---------------------------------------------------------------------------
# _has_negative_strides — only negative strides need materialisation
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestHasNegativeStrides:
    """Verify _has_negative_strides only flags tensors with negative strides."""

    def test_contiguous_row_major(self):
        t = torch.randn(128, 64)
        assert not KernelLauncher._has_negative_strides(t)

    def test_simple_transpose(self):
        """A transposed 2-D tensor has non-negative strides."""
        t = torch.randn(64, 128).T  # shape (128, 64), stride (1, 128)
        assert not KernelLauncher._has_negative_strides(t)

    def test_1d_tensor(self):
        t = torch.randn(256)
        assert not KernelLauncher._has_negative_strides(t)

    def test_1d_sliced(self):
        """1-D tensors with stride != 1 are fine — no negative stride."""
        t = torch.randn(512)[::2]  # stride (2,)
        assert not KernelLauncher._has_negative_strides(t)

    def test_scalar(self):
        t = torch.tensor(3.14)
        assert not KernelLauncher._has_negative_strides(t)

    def test_negative_stride_detected(self):
        """Negative strides are the only layout Triton cannot handle.

        PyTorch doesn't currently allow creating negative-stride tensors
        via as_strided, so we use a lightweight mock.
        """
        class _FakeNegStrideTensor:
            ndim = 2
            def stride(self): return (-64, 1)
            @property
            def shape(self): return (128, 64)

        assert KernelLauncher._has_negative_strides(_FakeNegStrideTensor())

    def test_zero_stride_broadcast_not_negative(self):
        """expand() creates zero strides — handled by stride params, not negative."""
        t = torch.randn(1, 64).expand(128, 64)  # stride (0, 1)
        assert t.stride(0) == 0
        assert not KernelLauncher._has_negative_strides(t)

    def test_zero_stride_on_size_1_dim(self):
        """Size-1 dims with stride 0 are fine."""
        t_bcast = torch.as_strided(torch.randn(64), (1, 64), (0, 1))
        assert not KernelLauncher._has_negative_strides(t_bcast)

    def test_no_unit_stride_not_negative(self):
        """Non-unit strides are handled by stride params, not flagged."""
        base = torch.randn(256, 128)
        t = base[::2, ::2]  # stride (256, 2)
        assert 1 not in t.stride()
        assert not KernelLauncher._has_negative_strides(t)


# ---------------------------------------------------------------------------
# _materialize_if_needed — only negative strides trigger copy
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestMaterializeIfNeeded:
    """Verify _materialize_if_needed copies ONLY negative-stride tensors."""

    def test_contiguous_tensor_returned_as_is(self):
        t = torch.randn(128, 64)
        result = KernelLauncher._materialize_if_needed(t)
        assert result is t  # same object, no copy

    def test_transposed_tensor_returned_as_is(self):
        """Transposed 2-D passes through — stride params handle it."""
        t = torch.randn(64, 128).T
        assert not t.is_contiguous()
        result = KernelLauncher._materialize_if_needed(t)
        assert result is t

    def test_zero_stride_broadcast_returned_as_is(self):
        """Zero-stride expanded tensor passes through — stride params handle it."""
        t = torch.randn(1, 64).expand(128, 64)  # stride (0, 1)
        assert t.stride(0) == 0
        result = KernelLauncher._materialize_if_needed(t)
        assert result is t  # NO copy — zero stride is fine for Triton

    def test_no_unit_stride_returned_as_is(self):
        """Non-unit-stride view passes through — stride params handle it."""
        base = torch.randn(256, 128)
        t = base[::2, ::2]  # stride (256, 2)
        result = KernelLauncher._materialize_if_needed(t)
        assert result is t  # NO copy

    def test_negative_stride_returns_contiguous_copy(self):
        """Negative strides are the only case that triggers materialisation."""
        class _FakeNegStrideTensor:
            ndim = 2
            def stride(self): return (-64, 1)
            @property
            def shape(self): return (128, 64)
            def contiguous(self):
                return torch.randn(128, 64)  # simulate copy

        fake = _FakeNegStrideTensor()
        result = KernelLauncher._materialize_if_needed(fake)
        assert isinstance(result, torch.Tensor)
        assert result.is_contiguous()


# ---------------------------------------------------------------------------
# _select_num_warps — heuristic warp selection
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestSelectNumWarps:
    """Verify num_warps heuristic returns correct values for various configs."""

    def test_tiny_problem_returns_2(self):
        """M*N < 1024 → 2 warps regardless of dtype."""
        assert KernelLauncher._select_num_warps(16, 16, 64, torch.float32) == 2
        assert KernelLauncher._select_num_warps(16, 16, 64, torch.float16) == 2

    def test_fp32_medium_returns_4(self):
        """FP32 with moderate tile size → 4 warps."""
        assert KernelLauncher._select_num_warps(128, 256, 64, torch.float32) == 4

    def test_fp16_large_returns_8(self):
        """FP16 with large tile (>= 4096 elements) → 8 warps for tensor cores."""
        assert KernelLauncher._select_num_warps(128, 256, 64, torch.float16) == 8

    def test_bf16_large_returns_8(self):
        """BF16 behaves like FP16."""
        assert KernelLauncher._select_num_warps(128, 256, 64, torch.bfloat16) == 8

    def test_fp16_small_tile_returns_4(self):
        """FP16 with tile area < 4096 but >= 1024 → 4 warps (not enough to justify 8)."""
        assert KernelLauncher._select_num_warps(32, 64, 64, torch.float16) == 4

    def test_return_value_is_power_of_2(self):
        """All returned num_warps should be powers of 2 (Triton requirement)."""
        for M, N, K in [(16, 16, 8), (64, 64, 32), (256, 512, 128)]:
            for dt in (torch.float32, torch.float16, torch.bfloat16):
                nw = KernelLauncher._select_num_warps(M, N, K, dt)
                assert nw & (nw - 1) == 0, f"num_warps={nw} is not a power of 2"


# ---------------------------------------------------------------------------
# _select_num_stages — heuristic pipeline depth
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestSelectNumStages:
    """Verify num_stages heuristic returns correct values."""

    def test_small_working_set_returns_2(self):
        """When total bytes < threshold → 2 stages."""
        # (M+N)*K*bpe = (32+64)*32*4 = 12288 bytes << 512KB
        assert KernelLauncher._select_num_stages(32, 64, 32, torch.float32) == 2

    def test_large_fp32_returns_3(self):
        """Large FP32 working set → 3 stages."""
        # Need (M+N)*K*4 >= 512*1024 = 524288
        # e.g. M=1024, N=1024, K=128 → (2048)*128*4 = 1,048,576 > 524,288
        assert KernelLauncher._select_num_stages(1024, 1024, 128, torch.float32) == 3

    def test_large_fp16_returns_4(self):
        """Large FP16 working set → 4 stages (deeper pipelining)."""
        # (M+N)*K*2 >= 524288 → e.g. M=1024, N=1024, K=256
        # (2048)*256*2 = 1,048,576 > 524,288
        assert KernelLauncher._select_num_stages(1024, 1024, 256, torch.float16) == 4

    def test_large_bf16_returns_4(self):
        """BF16 behaves like FP16 for stage selection."""
        assert KernelLauncher._select_num_stages(1024, 1024, 256, torch.bfloat16) == 4

    def test_returned_values_are_positive(self):
        for M, N, K in [(8, 8, 4), (128, 256, 64), (2048, 2048, 512)]:
            for dt in (torch.float32, torch.float16, torch.bfloat16):
                ns = KernelLauncher._select_num_stages(M, N, K, dt)
                assert ns >= 2, f"num_stages={ns} should be >= 2"


# ---------------------------------------------------------------------------
# __call__ integration — num_warps and num_stages passed to kernel
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestKernelLaunchHeuristics:
    """Verify __call__ passes num_warps and num_stages as kwargs."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_num_warps_passed_as_kwarg(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        assert "num_warps" in call["kwargs"]
        assert isinstance(call["kwargs"]["num_warps"], int)

    def test_num_stages_passed_as_kwarg(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        assert "num_stages" in call["kwargs"]
        assert isinstance(call["kwargs"]["num_stages"], int)

    def test_fp32_medium_gets_4_warps(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # M=128, N=256 → tile_area=32768 >= 4096 but FP32 → 4 warps
        assert call["kwargs"]["num_warps"] == 4

    def test_fp16_output_gets_8_warps(self, mock_kernel_fn, input_descs):
        out_fp16 = TensorDescriptor("out", (128, 256), (256, 1), torch.float16)
        launcher = KernelLauncher(mock_kernel_fn, input_descs, out_fp16, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # M=128, N=256 → tile_area=32768 >= 4096 and FP16 → 8 warps
        assert call["kwargs"]["num_warps"] == 8


# ---------------------------------------------------------------------------
# __call__ with native stride handling — no unnecessary copies
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestKernelLaunchNativeStrides:
    """Verify __call__ passes non-contiguous layouts through without copying."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_zero_stride_broadcast_passes_through(self, mock_kernel_fn, input_descs, output_desc):
        """Zero-stride expanded tensor passes through with original strides (0, 1)."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a_bcast = torch.randn(1, 64).expand(128, 64)  # stride (0, 1)
        assert a_bcast.stride(0) == 0
        b = torch.randn(64, 256)
        launcher(a_bcast, b)
        call = mock_kernel_fn.calls[0]
        stride_start = 6  # 3 ptrs + 3 dims
        # Zero stride passed through — kernel handles via stride params
        assert call["args"][stride_start] == 0       # stride_m (broadcast)
        assert call["args"][stride_start + 1] == 1   # stride_k

    def test_transposed_input_preserves_original_strides(self, mock_kernel_fn, input_descs, output_desc):
        """Transposed input keeps its original strides."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a_t = torch.randn(64, 128).T  # stride (1, 128) — column-major
        b = torch.randn(64, 256)
        launcher(a_t, b)
        call = mock_kernel_fn.calls[0]
        stride_start = 6
        assert call["args"][stride_start] == 1       # stride_m (column-major)
        assert call["args"][stride_start + 1] == 128  # stride_k

    def test_output_shape_correct_with_broadcast_input(self, mock_kernel_fn, input_descs, output_desc):
        """Output dimensions correct even with zero-stride broadcast input."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a_bcast = torch.randn(1, 64).expand(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a_bcast, b)
        assert result.shape == (128, 256)

    def test_non_unit_stride_passes_through(self, mock_kernel_fn, input_descs, output_desc):
        """Non-unit-stride views pass through with their actual strides."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        # Create a (128, 64) tensor with non-unit strides via slicing
        base = torch.randn(256, 128)
        a_strided = base[::2, ::2]  # shape (128, 64), stride (256, 2)
        b = torch.randn(64, 256)
        launcher(a_strided, b)
        call = mock_kernel_fn.calls[0]
        stride_start = 6
        assert call["args"][stride_start] == 256     # stride_m
        assert call["args"][stride_start + 1] == 2   # stride_k


# ---------------------------------------------------------------------------
# Grid computation — boundary masking support (non-divisible shapes)
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestGridBoundaryMasking:
    """Verify grid handles non-divisible M, N dimensions correctly.

    The kernel generator emits boundary masks like
    ``mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)``
    which rely on M, N being passed as positional args.  These tests
    verify the launcher computes the correct grid and passes M, N.
    """

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_non_divisible_m(self, mock_kernel_fn, output_desc):
        """M=100 with BLOCK_SIZE_M=64 → grid_x = ceil(100/64) = 2."""
        a_desc = TensorDescriptor("a", (100, 64), (64, 1), torch.float32)
        b_desc = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        out_desc = TensorDescriptor("out", (100, 256), (256, 1), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, [a_desc, b_desc], out_desc, [], "a", "b",
        )
        a = torch.randn(100, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        grid_x, grid_y = call["grid"]
        assert grid_x == 2   # ceil(100/64)
        assert grid_y == 4   # ceil(256/64)
        # M=100, N=256 passed for boundary mask
        assert call["args"][3] == 100
        assert call["args"][4] == 256

    def test_non_divisible_n(self, mock_kernel_fn):
        """N=200 with BLOCK_SIZE_N=64 → grid_y = ceil(200/64) = 4."""
        a_desc = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b_desc = TensorDescriptor("b", (64, 200), (200, 1), torch.float32)
        out_desc = TensorDescriptor("out", (128, 200), (200, 1), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, [a_desc, b_desc], out_desc, [], "a", "b",
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 200)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        grid_x, grid_y = call["grid"]
        assert grid_x == 2   # ceil(128/64)
        assert grid_y == 4   # ceil(200/64) = 3.125 → 4
        assert call["args"][4] == 200  # N passed for mask

    def test_both_non_divisible(self, mock_kernel_fn):
        """Vocab-like shape: M=32001, N=127 with block 64/64."""
        a_desc = TensorDescriptor("a", (32001, 64), (64, 1), torch.float32)
        b_desc = TensorDescriptor("b", (64, 127), (127, 1), torch.float32)
        out_desc = TensorDescriptor("out", (32001, 127), (127, 1), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, [a_desc, b_desc], out_desc, [], "a", "b",
        )
        a = torch.randn(32001, 64)
        b = torch.randn(64, 127)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        grid_x, grid_y = call["grid"]
        assert grid_x == 501  # ceil(32001/64)
        assert grid_y == 2    # ceil(127/64)
        assert call["args"][3] == 32001
        assert call["args"][4] == 127


# ---------------------------------------------------------------------------
# _enforce_sram_capacity — SRAM budget enforcement
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestSRAMCapacityEnforcement:
    """Verify _enforce_sram_capacity downscales blocks to fit SRAM budget."""

    def test_small_tile_unchanged(self):
        """64x64 FP32 = 16KB < 48KB → no change."""
        m, n = KernelLauncher._enforce_sram_capacity(64, 64, torch.float32)
        assert (m, n) == (64, 64)

    def test_large_tile_downscaled(self):
        """256x256 FP32 = 256KB > 48KB → must shrink."""
        m, n = KernelLauncher._enforce_sram_capacity(256, 256, torch.float32)
        assert m * n * 4 <= _DEFAULT_SRAM_BUDGET_BYTES
        assert m & (m - 1) == 0  # power of 2
        assert n & (n - 1) == 0

    def test_halves_larger_dim_first(self):
        """128x128 FP32 = 64KB > 48KB → halve M first since M >= N."""
        m, n = KernelLauncher._enforce_sram_capacity(128, 128, torch.float32)
        assert m * n * 4 <= _DEFAULT_SRAM_BUDGET_BYTES
        # M was halved first (128 >= 128, so M goes first)
        assert m == 64
        assert n == 128

    def test_fp16_allows_larger_tiles(self):
        """128x128 FP16 = 32KB < 48KB → unchanged."""
        m, n = KernelLauncher._enforce_sram_capacity(128, 128, torch.float16)
        assert (m, n) == (128, 128)

    def test_respects_minimum_floor(self):
        """Even with tiny budget, dimensions don't go below _MIN_BLOCK_DIM."""
        m, n = KernelLauncher._enforce_sram_capacity(
            64, 64, torch.float32, sram_budget_bytes=64,
        )
        assert m >= _MIN_BLOCK_DIM
        assert n >= _MIN_BLOCK_DIM

    def test_custom_budget(self):
        """Custom budget is respected."""
        m, n = KernelLauncher._enforce_sram_capacity(
            64, 64, torch.float32, sram_budget_bytes=8 * 1024,
        )
        # 64x64x4 = 16KB > 8KB → need downscaling
        assert m * n * 4 <= 8 * 1024

    def test_result_always_power_of_2(self):
        """Downscaled dimensions must remain powers of 2."""
        for bm, bn in [(256, 256), (128, 256), (512, 64)]:
            for dt in (torch.float32, torch.float16, torch.bfloat16):
                m, n = KernelLauncher._enforce_sram_capacity(bm, bn, dt)
                assert m & (m - 1) == 0, f"m={m} not power of 2"
                assert n & (n - 1) == 0, f"n={n} not power of 2"


# ---------------------------------------------------------------------------
# SRAM enforcement integration — verify __call__ passes downscaled blocks
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestSRAMEnforcementInCall:
    """Verify __call__ passes SRAM-safe block sizes to the kernel launch."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_oversized_blocks_downscaled_in_launch(self, mock_kernel_fn, input_descs, output_desc):
        """Block sizes that exceed SRAM budget are downscaled before launch."""
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=256, block_size_n=256,  # 256*256*4 = 256KB >> 48KB
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        bsm = call["kwargs"]["BLOCK_SIZE_M"]
        bsn = call["kwargs"]["BLOCK_SIZE_N"]
        assert bsm * bsn * 4 <= _DEFAULT_SRAM_BUDGET_BYTES

    def test_safe_blocks_unchanged_in_launch(self, mock_kernel_fn, input_descs, output_desc):
        """Block sizes within SRAM budget pass through unchanged."""
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=32, block_size_n=64,  # 32*64*4 = 8KB << 48KB
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        assert call["kwargs"]["BLOCK_SIZE_M"] == 32
        assert call["kwargs"]["BLOCK_SIZE_N"] == 64

    def test_sram_downscale_bumps_num_stages(self, mock_kernel_fn, input_descs, output_desc):
        """When SRAM enforcement reduces blocks, num_stages is bumped by 1."""
        # First: get baseline num_stages with default blocks (64x64 fits)
        launcher_baseline = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher_baseline(a, b)
        baseline_stages = mock_kernel_fn.calls[0]["kwargs"]["num_stages"]

        # Now: use oversized blocks that trigger SRAM downscaling
        mock_kernel_fn_2 = _MockKernelFn()
        launcher_downscaled = KernelLauncher(
            mock_kernel_fn_2, input_descs, output_desc, [], "a", "b",
            block_size_m=256, block_size_n=256,
        )
        launcher_downscaled(a, b)
        downscaled_stages = mock_kernel_fn_2.calls[0]["kwargs"]["num_stages"]

        # Downscaled version should have +1 stage (capped at 4)
        assert downscaled_stages == min(baseline_stages + 1, 4)


# ---------------------------------------------------------------------------
# _get_launch_stream — CUDA stream acquisition
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestCUDAStreamSync:
    """Verify the launcher handles CUDA stream synchronization."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_no_stream_kwarg_on_cpu(self, mock_kernel_fn, input_descs, output_desc):
        """CPU tensors should not pass stream= to Triton."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)  # CPU
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        assert "stream" not in call["kwargs"]

    def test_get_launch_stream_returns_none_on_cpu(self):
        """_get_launch_stream returns None for CPU devices."""
        cpu_device = torch.device("cpu")
        assert KernelLauncher._get_launch_stream(cpu_device) is None

    def test_stream_kwarg_not_passed_triton3(self, mock_kernel_fn, input_descs, output_desc, monkeypatch):
        """Triton 3.x deprecated stream= kwarg — it should NOT be in kwargs."""
        monkeypatch.setattr(
            KernelLauncher, "_get_launch_stream", staticmethod(lambda device: 42)
        )
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # Triton 3.x automatically uses the current CUDA stream;
        # the launcher no longer passes stream= explicitly.
        assert "stream" not in call["kwargs"]
