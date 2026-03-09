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
    LaunchParams,
    _DEFAULT_GROUP_SIZE_M,
    _DEFAULT_SRAM_BUDGET_BYTES,
    _LARGE_WORKING_SET_BYTES,
    _MIN_BLOCK_DIM,
    _enforce_sram_capacity,
    _select_num_stages,
    _select_num_warps,
    compute_launch_params,
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
        assert launcher._launch_params.block_m == 64
        assert launcher._launch_params.block_n == 64
        assert launcher._launch_params.block_k == 32

    def test_custom_block_sizes(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=128, block_size_n=128, block_size_k=64,
        )
        assert launcher._launch_params.block_m == 128
        assert launcher._launch_params.block_n == 128
        assert launcher._launch_params.block_k == 64

    def test_explicit_launch_params(self, mock_kernel_fn, input_descs, output_desc):
        lp = LaunchParams(block_m=32, block_n=64, block_k=16, group_size_m=4, num_warps=2, num_stages=3)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            launch_params=lp,
        )
        assert launcher._launch_params is lp
        assert launcher._launch_params.block_m == 32
        assert launcher._launch_params.num_warps == 2


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
        lp = LaunchParams(block_m=32, block_n=64, block_k=16, group_size_m=8, num_warps=4, num_stages=2)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            launch_params=lp,
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
        assert launcher._launch_params.block_m == 64
        assert launcher._launch_params.block_n == 128
        assert launcher._launch_params.block_k == 32

    def test_non_power_of_two_rounded_up(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=50, block_size_n=100, block_size_k=24,
        )
        assert launcher._launch_params.block_m == 64
        assert launcher._launch_params.block_n == 128
        assert launcher._launch_params.block_k == 32

    def test_default_block_sizes_are_power_of_two(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
        )
        lp = launcher._launch_params
        for bs in (lp.block_m, lp.block_n, lp.block_k):
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
        assert _select_num_warps(16, 16, 64, torch.float32) == 2
        assert _select_num_warps(16, 16, 64, torch.float16) == 2

    def test_fp32_medium_returns_4(self):
        """FP32 with moderate tile size → 4 warps."""
        assert _select_num_warps(128, 256, 64, torch.float32) == 4

    def test_fp16_large_returns_8(self):
        """FP16 with large tile (>= 4096 elements) → 8 warps for tensor cores."""
        assert _select_num_warps(128, 256, 64, torch.float16) == 8

    def test_bf16_large_returns_8(self):
        """BF16 behaves like FP16."""
        assert _select_num_warps(128, 256, 64, torch.bfloat16) == 8

    def test_fp16_small_tile_returns_4(self):
        """FP16 with tile area < 4096 but >= 1024 → 4 warps (not enough to justify 8)."""
        assert _select_num_warps(32, 64, 64, torch.float16) == 4

    def test_return_value_is_power_of_2(self):
        """All returned num_warps should be powers of 2 (Triton requirement)."""
        for M, N, K in [(16, 16, 8), (64, 64, 32), (256, 512, 128)]:
            for dt in (torch.float32, torch.float16, torch.bfloat16):
                nw = _select_num_warps(M, N, K, dt)
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
        assert _select_num_stages(32, 64, 32, torch.float32) == 2

    def test_large_fp32_returns_3(self):
        """Large FP32 working set → 3 stages."""
        # Need (M+N)*K*4 >= 512*1024 = 524288
        # e.g. M=1024, N=1024, K=128 → (2048)*128*4 = 1,048,576 > 524,288
        assert _select_num_stages(1024, 1024, 128, torch.float32) == 3

    def test_large_fp16_returns_5(self):
        """Large FP16 working set → 5 stages (deep cp.async pipelining on Ada)."""
        # (M+N)*K*2 >= 524288 → e.g. M=1024, N=1024, K=256
        # (2048)*256*2 = 1,048,576 > 524,288
        assert _select_num_stages(1024, 1024, 256, torch.float16) == 5

    def test_large_bf16_returns_5(self):
        """BF16 behaves like FP16 for stage selection (Ada cp.async)."""
        assert _select_num_stages(1024, 1024, 256, torch.bfloat16) == 5

    def test_returned_values_are_positive(self):
        for M, N, K in [(8, 8, 4), (128, 256, 64), (2048, 2048, 512)]:
            for dt in (torch.float32, torch.float16, torch.bfloat16):
                ns = _select_num_stages(M, N, K, dt)
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
        # Pre-compute with FP16 output dtype to get 8 warps
        lp = compute_launch_params(128, 256, 64, torch.float16)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out_fp16, [], "a", "b",
            launch_params=lp,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # M=128, N=256 → tile_area=32768 >= 4096 and FP16 → 8 warps
        assert call["kwargs"]["num_warps"] == 8


# ---------------------------------------------------------------------------
# __call__ — GROUP_SIZE_M (L2 swizzle width) passed to kernel
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestGroupSizeMKwarg:
    """Verify GROUP_SIZE_M is passed as a constexpr kwarg."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_default_group_size_m(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        assert call["kwargs"]["GROUP_SIZE_M"] == _DEFAULT_GROUP_SIZE_M

    def test_custom_group_size_m(self, mock_kernel_fn, input_descs, output_desc):
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            group_size_m=16,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        assert call["kwargs"]["GROUP_SIZE_M"] == 16

    def test_grid_is_1d(self, mock_kernel_fn, input_descs, output_desc):
        """Grid is 1-D (single integer tuple) for L2 swizzled block indexing."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        grid = call["grid"]
        assert isinstance(grid, tuple)
        assert len(grid) == 1
        # total_programs = ceil(128/64) * ceil(256/64) = 2 * 4 = 8
        assert grid[0] == 8


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
    verify the launcher computes the correct 1-D grid (with L2 swizzling)
    and passes M, N.
    """

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_non_divisible_m(self, mock_kernel_fn, output_desc):
        """M=100 with BLOCK_SIZE_M=64 → grid = ceil(100/64) * ceil(256/64) = 2 * 4 = 8."""
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
        (total_programs,) = call["grid"]
        grid_m = (100 + 64 - 1) // 64   # 2
        grid_n = (256 + 64 - 1) // 64   # 4
        assert total_programs == grid_m * grid_n
        # M=100, N=256 passed for boundary mask
        assert call["args"][3] == 100
        assert call["args"][4] == 256

    def test_non_divisible_n(self, mock_kernel_fn):
        """N=200 with BLOCK_SIZE_N=64 → grid = 2 * 4 = 8."""
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
        (total_programs,) = call["grid"]
        grid_m = (128 + 64 - 1) // 64   # 2
        grid_n = (200 + 64 - 1) // 64   # 4
        assert total_programs == grid_m * grid_n
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
        (total_programs,) = call["grid"]
        grid_m = (32001 + 64 - 1) // 64  # 501
        grid_n = (127 + 64 - 1) // 64    # 2
        assert total_programs == grid_m * grid_n
        assert call["args"][3] == 32001
        assert call["args"][4] == 127


# ---------------------------------------------------------------------------
# _enforce_sram_capacity — SRAM budget enforcement
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestSRAMCapacityEnforcement:
    """Verify _enforce_sram_capacity downscales blocks to fit SRAM budget."""

    def test_small_tile_unchanged(self):
        """64x64 FP32 = 16KB < 100KB → no change."""
        m, n = _enforce_sram_capacity(64, 64, torch.float32)
        assert (m, n) == (64, 64)

    def test_large_tile_downscaled(self):
        """256x256 FP32 = 256KB > 100KB → must shrink."""
        m, n = _enforce_sram_capacity(256, 256, torch.float32)
        assert m * n * 4 <= _DEFAULT_SRAM_BUDGET_BYTES
        assert m & (m - 1) == 0  # power of 2
        assert n & (n - 1) == 0

    def test_halves_larger_dim_first(self):
        """256x256 FP32 = 256KB > 100KB → halve M first since M >= N."""
        m, n = _enforce_sram_capacity(256, 256, torch.float32)
        assert m * n * 4 <= _DEFAULT_SRAM_BUDGET_BYTES
        # With 100KB budget, 256x256 FP32 = 256KB needs downscaling
        assert m <= 256
        assert n <= 256
        assert m & (m - 1) == 0  # power of 2
        assert n & (n - 1) == 0

    def test_128x128_fp32_fits_ada(self):
        """128x128 FP32 = 64KB < 100KB → unchanged on Ada."""
        m, n = _enforce_sram_capacity(128, 128, torch.float32)
        assert (m, n) == (128, 128)

    def test_fp16_allows_larger_tiles(self):
        """128x128 FP16 = 32KB < 100KB → unchanged."""
        m, n = _enforce_sram_capacity(128, 128, torch.float16)
        assert (m, n) == (128, 128)

    def test_respects_minimum_floor(self):
        """Even with tiny budget, dimensions don't go below _MIN_BLOCK_DIM."""
        m, n = _enforce_sram_capacity(
            64, 64, torch.float32, sram_budget_bytes=64,
        )
        assert m >= _MIN_BLOCK_DIM
        assert n >= _MIN_BLOCK_DIM

    def test_custom_budget(self):
        """Custom budget is respected."""
        m, n = _enforce_sram_capacity(
            64, 64, torch.float32, sram_budget_bytes=8 * 1024,
        )
        # 64x64x4 = 16KB > 8KB → need downscaling
        assert m * n * 4 <= 8 * 1024

    def test_result_always_power_of_2(self):
        """Downscaled dimensions must remain powers of 2."""
        for bm, bn in [(256, 256), (128, 256), (512, 64)]:
            for dt in (torch.float32, torch.float16, torch.bfloat16):
                m, n = _enforce_sram_capacity(bm, bn, dt)
                assert m & (m - 1) == 0, f"m={m} not power of 2"
                assert n & (n - 1) == 0, f"n={n} not power of 2"


# ---------------------------------------------------------------------------
# SRAM enforcement integration — verify __call__ passes downscaled blocks
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestSRAMEnforcementInCall:
    """Verify pre-computed LaunchParams passes SRAM-safe block sizes to kernel."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_oversized_blocks_downscaled_at_init(self, mock_kernel_fn, input_descs, output_desc):
        """Block sizes that exceed SRAM budget are downscaled at construction."""
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=256, block_size_n=256,  # 256*256*4 = 256KB >> 100KB
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        bsm = call["kwargs"]["BLOCK_SIZE_M"]
        bsn = call["kwargs"]["BLOCK_SIZE_N"]
        assert bsm * bsn * 4 <= _DEFAULT_SRAM_BUDGET_BYTES

    def test_safe_blocks_unchanged_at_init(self, mock_kernel_fn, input_descs, output_desc):
        """Block sizes within SRAM budget pass through unchanged."""
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [], "a", "b",
            block_size_m=32, block_size_n=64,  # 32*64*4 = 8KB << 100KB
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


# ---------------------------------------------------------------------------
# Mean reduction — FP32 accumulation and axis-aware division
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestMeanReductionFP32:
    """Mean reduction: FP32 accumulation buffer, correct division, dtype cast."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_mean_output_allocated_fp32(self, mock_kernel_fn, input_descs):
        """Mean accumulation buffer must be FP32 regardless of output dtype."""
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        # Output must be in the target dtype (bf16) after division + cast
        assert result.dtype == torch.bfloat16

    def test_mean_fp32_output_stays_fp32(self, mock_kernel_fn, input_descs):
        """When output is already FP32, no extra cast needed."""
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.dtype == torch.float32

    def test_mean_divides_by_n_for_axis_1(self, mock_kernel_fn, input_descs):
        """axis=1 (reduce N): division should be by N (=256)."""
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        # Result shape should be (M=128,)
        assert result.shape == (128,)

    def test_mean_divides_by_m_for_axis_0(self, mock_kernel_fn, input_descs):
        """axis=0 (reduce M): division should be by M (=128), output shape (N=256,)."""
        out = TensorDescriptor("out", (256,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=0,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        # Surviving dim is N=256
        assert result.shape == (256,)

    def test_sum_output_not_fp32_promoted(self, mock_kernel_fn, input_descs):
        """Sum reduction should NOT use FP32 accumulation (only mean does)."""
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="sum", reduction_axis=1,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        # Sum output stays in the requested dtype
        assert result.dtype == torch.bfloat16

    def test_max_output_uses_neg_inf(self, mock_kernel_fn, input_descs):
        """Max reduction should pre-fill with -inf in the output dtype."""
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="max", reduction_axis=1,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.dtype == torch.float32

    def test_reduction_axis_default_is_1(self, mock_kernel_fn, input_descs):
        """When reduction_axis is not specified, it defaults to 1."""
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="mean",
        )
        assert launcher._reduction_axis == 1

    def test_repr_includes_axis(self, mock_kernel_fn, input_descs):
        """Repr should show reduction op and axis."""
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
        )
        r = repr(launcher)
        assert "mean" in r
        assert "axis=1" in r


# ---------------------------------------------------------------------------
# Zero-overhead dispatcher — grid, allocation, and stride fast-paths
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestZeroOverheadGrid:
    """Verify native integer division grid replaces triton.cdiv."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_no_triton_cdiv_dependency(self, mock_kernel_fn, input_descs, output_desc):
        """__call__ must not reference triton.cdiv — uses native // division."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        # If triton.cdiv were still used, this would raise AttributeError
        # because the mock triton module has no cdiv attribute.
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        (total_programs,) = call["grid"]
        assert total_programs == 8  # ceil(128/64) * ceil(256/64)

    def test_native_division_matches_cdiv(self, mock_kernel_fn, input_descs, output_desc):
        """Native (a + b - 1) // b matches triton.cdiv for non-divisible dims."""
        a_desc = TensorDescriptor("a", (100, 64), (64, 1), torch.float32)
        b_desc = TensorDescriptor("b", (64, 200), (200, 1), torch.float32)
        out_desc = TensorDescriptor("out", (100, 200), (200, 1), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, [a_desc, b_desc], out_desc, [], "a", "b",
        )
        a = torch.randn(100, 64)
        b = torch.randn(64, 200)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        (total_programs,) = call["grid"]
        expected = ((100 + 64 - 1) // 64) * ((200 + 64 - 1) // 64)
        assert total_programs == expected

    def test_grid_is_lambda_closure(self, mock_kernel_fn, input_descs, output_desc):
        """Grid is a lambda that captures M, N from the enclosing scope."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        grid = call["grid"]
        assert isinstance(grid, tuple)
        assert len(grid) == 1


@pytest.mark.launcher
class TestSmartOutputAllocation:
    """Verify torch.empty_strided allocation with async reduction init."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_2d_output_is_row_major_strided(self, mock_kernel_fn, input_descs, output_desc):
        """Non-reduced output must be row-major with stride (N, 1)."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.stride() == (256, 1)

    def test_2d_output_is_contiguous(self, mock_kernel_fn, input_descs, output_desc):
        """Row-major empty_strided output must be contiguous."""
        launcher = KernelLauncher(mock_kernel_fn, input_descs, output_desc, [], "a", "b")
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.is_contiguous()

    def test_sum_output_stride_is_1(self, mock_kernel_fn, input_descs):
        """Sum reduction output must have stride (1,)."""
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="sum", reduction_axis=1,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.stride() == (1,)

    def test_max_output_stride_is_1(self, mock_kernel_fn, input_descs):
        """Max reduction output must have stride (1,)."""
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, out, [], "a", "b",
            reduction_op="max", reduction_axis=1,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.stride() == (1,)

    def test_intermediate_buffers_row_major(self, mock_kernel_fn, input_descs, output_desc):
        """Intermediate escape buffers must be row-major strided."""
        intm = TensorDescriptor("esc0", (128, 256), (256, 1), torch.float32)
        launcher = KernelLauncher(
            mock_kernel_fn, input_descs, output_desc, [intm], "a", "b",
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_kernel_fn.calls[0]
        # The intermediate tensor is args[3] (after 2 inputs + 1 output)
        intm_tensor = call["args"][3]
        assert intm_tensor.stride() == (256, 1)
        assert intm_tensor.is_contiguous()

    def test_dtype_extracted_from_descriptor(self, mock_kernel_fn, input_descs):
        """Output dtype must come from the descriptor, not hardcoded."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            out = TensorDescriptor("out", (128, 256), (256, 1), dtype)
            launcher = KernelLauncher(mock_kernel_fn, input_descs, out, [], "a", "b")
            a = torch.randn(128, 64)
            b = torch.randn(64, 256)
            result = launcher(a, b)
            assert result.dtype == dtype, f"Expected {dtype}, got {result.dtype}"


@pytest.mark.launcher
class TestBitwiseNegativeStrideCheck:
    """Verify the bitwise OR fold in _has_negative_strides."""

    def test_positive_strides_bitwise_or_non_negative(self):
        """OR of positive strides is non-negative."""
        t = torch.randn(128, 64)
        # All strides positive → bits non-negative → False
        assert not KernelLauncher._has_negative_strides(t)

    def test_mixed_positive_strides(self):
        """Various positive strides OR'd together stay non-negative."""
        base = torch.randn(256, 128)
        t = base[::2, ::2]  # stride (256, 2)
        assert not KernelLauncher._has_negative_strides(t)

    def test_zero_stride_or_keeps_non_negative(self):
        """Zero stride OR'd with positive stays non-negative."""
        t = torch.randn(1, 64).expand(128, 64)  # stride (0, 1)
        assert not KernelLauncher._has_negative_strides(t)

    def test_negative_stride_sets_sign_bit(self):
        """Any negative stride sets the sign bit in the OR result."""
        class _FakeNeg:
            ndim = 2
            def stride(self): return (-64, 1)
        assert KernelLauncher._has_negative_strides(_FakeNeg())

    def test_single_negative_among_positives(self):
        """One negative stride among positives triggers detection."""
        class _FakeMixed:
            ndim = 3
            def stride(self): return (256, -1, 1)
        assert KernelLauncher._has_negative_strides(_FakeMixed())

    def test_all_negative_strides(self):
        """All negative strides detected."""
        class _FakeAllNeg:
            ndim = 2
            def stride(self): return (-128, -1)
        assert KernelLauncher._has_negative_strides(_FakeAllNeg())

    def test_scalar_fast_path(self):
        """0-D tensors skip the bitwise check entirely."""
        t = torch.tensor(3.14)
        assert not KernelLauncher._has_negative_strides(t)


# ---------------------------------------------------------------------------
# LaunchParams — frozen dataclass
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestLaunchParams:
    """Verify LaunchParams is immutable and stores all fields."""

    def test_frozen(self):
        lp = LaunchParams(block_m=64, block_n=64, block_k=32, group_size_m=8, num_warps=4, num_stages=2)
        with pytest.raises(AttributeError):
            lp.block_m = 128  # type: ignore[misc]

    def test_fields(self):
        lp = LaunchParams(block_m=128, block_n=64, block_k=32, group_size_m=16, num_warps=8, num_stages=5)
        assert lp.block_m == 128
        assert lp.block_n == 64
        assert lp.block_k == 32
        assert lp.group_size_m == 16
        assert lp.num_warps == 8
        assert lp.num_stages == 5

    def test_equality(self):
        a = LaunchParams(64, 64, 32, 8, 4, 2)
        b = LaunchParams(64, 64, 32, 8, 4, 2)
        assert a == b

    def test_inequality(self):
        a = LaunchParams(64, 64, 32, 8, 4, 2)
        b = LaunchParams(128, 64, 32, 8, 4, 2)
        assert a != b


# ---------------------------------------------------------------------------
# compute_launch_params — static pre-computation
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestComputeLaunchParams:
    """Verify compute_launch_params pre-computes SRAM-safe configurations."""

    def test_returns_launch_params(self):
        lp = compute_launch_params(128, 256, 64, torch.float32)
        assert isinstance(lp, LaunchParams)

    def test_default_block_sizes(self):
        lp = compute_launch_params(128, 256, 64, torch.float32)
        assert lp.block_m == 64
        assert lp.block_n == 64
        assert lp.block_k == 32
        assert lp.group_size_m == 8

    def test_custom_block_sizes(self):
        lp = compute_launch_params(
            128, 256, 64, torch.float32,
            block_size_m=32, block_size_n=128, block_size_k=64,
        )
        # 32*128*4 = 16KB < 100KB → no downscaling
        assert lp.block_m == 32
        assert lp.block_n == 128
        assert lp.block_k == 64

    def test_oversized_blocks_downscaled(self):
        lp = compute_launch_params(
            128, 256, 64, torch.float32,
            block_size_m=256, block_size_n=256,
        )
        assert lp.block_m * lp.block_n * 4 <= _DEFAULT_SRAM_BUDGET_BYTES

    def test_non_power_of_two_rounded(self):
        lp = compute_launch_params(
            128, 256, 64, torch.float32,
            block_size_m=50, block_size_n=100, block_size_k=24,
        )
        assert lp.block_m & (lp.block_m - 1) == 0
        assert lp.block_n & (lp.block_n - 1) == 0
        assert lp.block_k & (lp.block_k - 1) == 0

    def test_fp16_gets_more_warps(self):
        lp_fp32 = compute_launch_params(128, 256, 64, torch.float32)
        lp_fp16 = compute_launch_params(128, 256, 64, torch.float16)
        # FP16 large tile → 8 warps; FP32 → 4 warps
        assert lp_fp16.num_warps == 8
        assert lp_fp32.num_warps == 4

    def test_sram_downscale_bumps_stages(self):
        lp_safe = compute_launch_params(128, 256, 64, torch.float32)
        lp_big = compute_launch_params(
            128, 256, 64, torch.float32,
            block_size_m=256, block_size_n=256,
        )
        # Downscaled blocks get +1 stage (capped at 4)
        assert lp_big.num_stages == min(lp_safe.num_stages + 1, 4)

    def test_sram_autotuner_path(self):
        from fuseml.codegen.sram_autotuner import SRAMAutotuner
        autotuner = SRAMAutotuner()
        lp = compute_launch_params(128, 256, 64, torch.float32, sram_autotuner=autotuner)
        assert isinstance(lp, LaunchParams)
        # The autotuner should produce valid power-of-2 blocks
        assert lp.block_m & (lp.block_m - 1) == 0
        assert lp.block_n & (lp.block_n - 1) == 0

    def test_launch_params_injected_into_kwargs(self):
        """Pre-computed LaunchParams are injected into kernel kwargs at dispatch."""
        mock_fn = _MockKernelFn()
        input_descs = [
            TensorDescriptor("a", (128, 64), (64, 1), torch.float32),
            TensorDescriptor("b", (64, 256), (256, 1), torch.float32),
        ]
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float32)
        lp = LaunchParams(block_m=32, block_n=64, block_k=16, group_size_m=4, num_warps=2, num_stages=3)
        launcher = KernelLauncher(mock_fn, input_descs, out, [], "a", "b", launch_params=lp)
        launcher(torch.randn(128, 64), torch.randn(64, 256))
        call = mock_fn.calls[0]
        assert call["kwargs"]["BLOCK_SIZE_M"] == 32
        assert call["kwargs"]["BLOCK_SIZE_N"] == 64
        assert call["kwargs"]["BLOCK_SIZE_K"] == 16
        assert call["kwargs"]["GROUP_SIZE_M"] == 4
        assert call["kwargs"]["num_warps"] == 2
        assert call["kwargs"]["num_stages"] == 3


# ---------------------------------------------------------------------------
# CUDA Graph safety — pre-computed dispatch state, no host-side branching
# ---------------------------------------------------------------------------

@pytest.mark.launcher
class TestCUDAGraphSafety:
    """Verify KernelLauncher dispatch is CUDA-graph-safe.

    CUDA Graph capture requires:
    1. No data-dependent host-side control flow during dispatch
    2. All allocations with explicit device/dtype (no implicit sync)
    3. No per-dispatch dict literal construction (frozen kwargs)
    4. No secondary PyTorch kernels for post-kernel fixups
    """

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    @pytest.fixture
    def input_descs(self):
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        return [a, b]

    @pytest.fixture
    def output_desc(self):
        return TensorDescriptor("out", (128, 256), (256, 1), torch.float32)

    # ── Frozen launch kwargs ──────────────────────────────────────

    def test_frozen_kwargs_built_at_init(self, input_descs, output_desc):
        """Launch kwargs must be built once at init, not per-dispatch."""
        mock_fn = _MockKernelFn()
        lp = LaunchParams(32, 64, 16, 4, 2, 3)
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
            launch_params=lp,
        )
        assert hasattr(launcher, "_frozen_launch_kwargs")
        assert launcher._frozen_launch_kwargs["BLOCK_SIZE_M"] == 32
        assert launcher._frozen_launch_kwargs["num_warps"] == 2

    def test_frozen_kwargs_same_object_across_calls(self, input_descs, output_desc):
        """The same dict object is reused across dispatches — no per-call construction."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
        )
        frozen_id = id(launcher._frozen_launch_kwargs)
        launcher(torch.randn(128, 64), torch.randn(64, 256))
        launcher(torch.randn(128, 64), torch.randn(64, 256))
        # The frozen dict is the same object — not rebuilt per call
        assert id(launcher._frozen_launch_kwargs) == frozen_id

    def test_autotuned_has_empty_frozen_kwargs(self, input_descs, output_desc):
        """Autotuned kernels have empty frozen kwargs (Triton manages them)."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
            is_autotuned=True,
        )
        assert launcher._frozen_launch_kwargs == {}

    # ── Pre-computed allocation state ─────────────────────────────

    def test_is_reduced_precomputed(self, input_descs, output_desc):
        """_is_reduced is pre-computed at init, not checked per-dispatch."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
        )
        assert launcher._is_reduced is False

    def test_is_reduced_true_for_1d_output(self, input_descs):
        """1-D output descriptor → _is_reduced is True."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="sum", reduction_axis=1,
        )
        assert launcher._is_reduced is True

    def test_alloc_fill_value_precomputed_for_max(self, input_descs):
        """Max reduction pre-computes fill value as -inf."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="max", reduction_axis=1,
        )
        assert launcher._alloc_fill_value == float("-inf")

    def test_alloc_fill_value_precomputed_for_sum(self, input_descs):
        """Sum reduction pre-computes fill value as 0.0."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="sum", reduction_axis=1,
        )
        assert launcher._alloc_fill_value == 0.0

    def test_alloc_dtype_fp32_for_mean(self, input_descs):
        """Mean reduction forces FP32 accumulation regardless of output dtype."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
        )
        assert launcher._alloc_dtype == torch.float32

    def test_alloc_dtype_matches_output_for_non_mean(self, input_descs):
        """Non-mean reductions use the output descriptor dtype."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="sum", reduction_axis=1,
        )
        assert launcher._alloc_dtype == torch.bfloat16

    # ── Post-kernel strategy ──────────────────────────────────────

    def test_post_kernel_noop_for_non_reduction(self, input_descs, output_desc):
        """Non-reduction kernels use NOOP post-kernel strategy."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
        )
        assert launcher._post_kernel_strategy == KernelLauncher._POST_KERNEL_NOOP

    def test_post_kernel_cast_for_fused_mean(self, input_descs):
        """Fused mean uses CAST_ONLY — epilogue already divided."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
            mean_epilogue_fused=True,
        )
        assert launcher._post_kernel_strategy == KernelLauncher._POST_KERNEL_CAST_ONLY

    def test_post_kernel_div_for_legacy_mean(self, input_descs):
        """Legacy mean (unfused epilogue) uses MEAN_DIV strategy."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
            mean_epilogue_fused=False,
        )
        assert launcher._post_kernel_strategy == KernelLauncher._POST_KERNEL_MEAN_DIV

    def test_fused_mean_returns_correct_dtype(self, input_descs):
        """Fused mean: output is cast from FP32 to target dtype."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
            mean_epilogue_fused=True,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        # Fused path: cast FP32 → bf16
        assert result.dtype == torch.bfloat16

    def test_fused_mean_fp32_output_stays_fp32(self, input_descs):
        """When output is already FP32, cast is a no-op."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
            mean_epilogue_fused=True,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.dtype == torch.float32

    def test_legacy_mean_uses_inplace_mul(self, input_descs):
        """Legacy mean: mul_(1/N) instead of tensor / N (avoids scalar tensor)."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
            mean_epilogue_fused=False,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        # Result should have correct shape and dtype
        assert result.shape == (128,)
        assert result.dtype == torch.float32

    def test_mean_epilogue_fused_default_is_true(self, input_descs, output_desc):
        """mean_epilogue_fused defaults to True."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
        )
        assert launcher._mean_epilogue_fused is True

    # ── Allocation safeness ───────────────────────────────────────

    def test_output_uses_torch_full_for_reduction(self, input_descs):
        """Reduction output uses torch.full with explicit device/dtype."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.float32)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="sum", reduction_axis=1,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.shape == (128,)
        assert result.device == a.device

    def test_explicit_dtype_on_all_allocations(self, input_descs):
        """All tensor factory calls must specify dtype explicitly."""
        mock_fn = _MockKernelFn()
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        launcher = KernelLauncher(
            mock_fn, input_descs, out, [], "a", "b",
            reduction_op="mean", reduction_axis=1,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        # Mean: accumulator is FP32, final result is bf16
        assert result.dtype == torch.bfloat16

    # ── No per-dispatch dict construction ─────────────────────────

    def test_no_per_dispatch_dict_construction(self, input_descs, output_desc):
        """Kernel kwargs must reference the frozen dict, not build a new one."""
        mock_fn = _MockKernelFn()
        lp = LaunchParams(64, 64, 32, 8, 4, 2)
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
            launch_params=lp,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        call = mock_fn.calls[0]
        # The kwargs passed to the kernel must match the frozen dict
        assert call["kwargs"]["BLOCK_SIZE_M"] == lp.block_m
        assert call["kwargs"]["BLOCK_SIZE_N"] == lp.block_n
        assert call["kwargs"]["num_warps"] == lp.num_warps

    # ── Deterministic dispatch across repeated calls ──────────────

    def test_repeated_calls_produce_identical_kwargs(self, input_descs, output_desc):
        """Multiple dispatches produce bit-identical kwargs (CUDA Graph replay)."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        launcher(a, b)
        assert mock_fn.calls[0]["kwargs"] == mock_fn.calls[1]["kwargs"]

    def test_repeated_calls_produce_identical_grid(self, input_descs, output_desc):
        """Same inputs produce identical grid across dispatches."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        launcher(a, b)
        assert mock_fn.calls[0]["grid"] == mock_fn.calls[1]["grid"]

    def test_repeated_calls_produce_identical_arg_count(self, input_descs, output_desc):
        """Positional arg count is stable across dispatches."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        launcher(a, b)
        assert len(mock_fn.calls[0]["args"]) == len(mock_fn.calls[1]["args"])
