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
from fuseml.codegen.kernel_launcher import KernelLauncher


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
        """2 inputs + 1 output = 3 pointer args before M, N, K."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # args[0], args[1] = input ptrs; args[2] = output ptr; args[3..5] = M,N,K
        # All pointers are ints (from data_ptr())
        for i in range(3):
            assert isinstance(call["args"][i], int)

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
