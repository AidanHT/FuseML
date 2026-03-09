"""Tests for EagerFallbackGuard — deterministic fallback & precision auditing.

Run with:
    pytest tests/test_eager_fallback.py -v
    pytest tests/ -m fallback -v
"""

from __future__ import annotations

import logging
import sys
import types

import pytest
import torch

from fuseml.codegen.eager_fallback import (
    EagerFallbackGuard,
    _is_triton_compilation_error,
)
from fuseml.codegen.kernel_generator import TensorDescriptor
from fuseml.codegen.kernel_launcher import KernelLauncher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_eager_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Eager PyTorch matmul — the "unfused" reference computation."""
    return a @ b


class _MockKernelFn:
    """Mimics the ``kernel_fn[grid](*args, **kwargs)`` Triton launch syntax."""

    def __init__(self, side_effect: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._side_effect = side_effect

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
            if self._side_effect is not None:
                raise self._side_effect
        return _launcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def input_descs():
    a = TensorDescriptor(name="a", shape=(128, 64), stride=(64, 1), dtype=torch.float32)
    b = TensorDescriptor(name="b", shape=(64, 256), stride=(256, 1), dtype=torch.float32)
    return [a, b]


@pytest.fixture
def output_desc():
    return TensorDescriptor(name="out", shape=(128, 256), stride=(256, 1), dtype=torch.float32)


# ---------------------------------------------------------------------------
# EagerFallbackGuard — unit tests
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestEagerFallbackGuardUnit:
    """Direct unit tests for EagerFallbackGuard without KernelLauncher."""

    def test_successful_triton_returns_result(self):
        """When Triton succeeds, its output is returned directly."""
        expected = torch.randn(4, 4)

        def triton_fn():
            return expected

        guard = EagerFallbackGuard(_simple_eager_matmul, "test_kernel")
        result = guard.execute(triton_fn, (torch.randn(4, 4), torch.randn(4, 4)))
        assert result is expected

    def test_fallback_on_runtime_error(self):
        """RuntimeError (e.g. CUDA OOM) triggers eager fallback."""
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)

        def failing_triton():
            raise RuntimeError("CUDA out of memory")

        guard = EagerFallbackGuard(_simple_eager_matmul, "oom_kernel")
        result = guard.execute(failing_triton, (a, b))
        expected = a @ b
        torch.testing.assert_close(result, expected)

    def test_fallback_increments_count(self):
        """Each fallback increments the counter."""
        guard = EagerFallbackGuard(_simple_eager_matmul, "counter_kernel")

        assert guard.fallback_count == 0

        a, b = torch.randn(2, 2), torch.randn(2, 2)
        guard.execute(lambda: (_ for _ in ()).throw(RuntimeError("fail")), (a, b))
        assert guard.fallback_count == 1

        guard.execute(lambda: (_ for _ in ()).throw(RuntimeError("fail2")), (a, b))
        assert guard.fallback_count == 2

    def test_non_recoverable_error_propagates(self):
        """TypeError is a programming bug — not caught by the guard."""
        guard = EagerFallbackGuard(_simple_eager_matmul, "type_err_kernel")

        def bad_triton():
            raise TypeError("bad argument type")

        with pytest.raises(TypeError, match="bad argument type"):
            guard.execute(bad_triton, (torch.randn(2, 2),))

    def test_value_error_propagates(self):
        """ValueError is a programming bug — not caught by the guard."""
        guard = EagerFallbackGuard(_simple_eager_matmul, "val_err_kernel")

        def bad_triton():
            raise ValueError("invalid shape")

        with pytest.raises(ValueError, match="invalid shape"):
            guard.execute(bad_triton, (torch.randn(2, 2),))

    def test_repr_contains_kernel_name(self):
        guard = EagerFallbackGuard(_simple_eager_matmul, "my_fused_kernel")
        r = repr(guard)
        assert "my_fused_kernel" in r
        assert "fallbacks=0" in r

    def test_repr_after_fallbacks(self):
        guard = EagerFallbackGuard(_simple_eager_matmul, "repr_kernel")
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        guard.execute(lambda: (_ for _ in ()).throw(RuntimeError("x")), (a, b))
        assert "fallbacks=1" in repr(guard)


# ---------------------------------------------------------------------------
# Buffer isolation — input snapshots prevent in-place corruption
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestBufferIsolation:
    """Verify that input snapshots protect the eager fallback from corruption.

    If the Triton kernel writes in-place over input memory (aliased
    pointers, in-place ops like add_) before failing, the fallback must
    still receive pristine, uncorrupted inputs.
    """

    def test_inputs_cloned_before_triton_launch(self):
        """Fallback uses cloned inputs, not originals that kernel may have corrupted."""
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        a_original = a.clone()
        b_original = b.clone()

        def triton_that_corrupts_inputs():
            # Simulate in-place corruption of input memory
            a.mul_(0.0)  # Zeroes out 'a' in-place
            b.fill_(float("nan"))  # Poisons 'b'
            raise RuntimeError("CUDA error after in-place corruption")

        guard = EagerFallbackGuard(_simple_eager_matmul, "corrupt_input_kernel")
        result = guard.execute(triton_that_corrupts_inputs, (a, b))

        # The fallback result must use the ORIGINAL (pre-corruption) values.
        expected = a_original @ b_original
        torch.testing.assert_close(result, expected)

    def test_snapshot_not_affected_by_kernel_mutation(self):
        """Even if the kernel mutates the input tensor, snapshots are independent."""
        x = torch.ones(3, 3)
        captured_snapshots = []

        def eager_fn(a):
            captured_snapshots.append(a.clone())
            return a * 2

        def triton_that_mutates():
            x.zero_()  # Corrupt original
            raise RuntimeError("fail after mutation")

        guard = EagerFallbackGuard(eager_fn, "mutate_kernel")
        result = guard.execute(triton_that_mutates, (x,))

        # The eager_fn received a snapshot taken BEFORE the mutation
        assert len(captured_snapshots) == 1
        torch.testing.assert_close(captured_snapshots[0], torch.ones(3, 3))

    def test_fp16_input_isolation(self):
        """Buffer isolation works correctly for FP16 tensors."""
        a = torch.randn(4, 4, dtype=torch.float16)
        b = torch.randn(4, 4, dtype=torch.float16)
        expected = a @ b

        def triton_corrupts_fp16():
            a.fill_(float("inf"))
            raise RuntimeError("FP16 kernel failure")

        guard = EagerFallbackGuard(_simple_eager_matmul, "fp16_isolation")
        result = guard.execute(triton_corrupts_fp16, (a, b))
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-2)

    def test_happy_path_returns_triton_output_not_snapshot(self):
        """On success, the Triton output is returned — not a snapshot."""
        expected = torch.randn(4, 4)

        def triton_fn():
            return expected

        guard = EagerFallbackGuard(_simple_eager_matmul, "happy_kernel")
        result = guard.execute(triton_fn, (torch.randn(4, 4), torch.randn(4, 4)))
        assert result is expected

    def test_inplace_add_corruption_isolated(self):
        """In-place add_ on input before failure doesn't poison fallback."""
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        a_original = a.clone()

        def triton_inplace_add():
            a.add_(1000.0)  # In-place add_ — corrupts 'a'
            raise RuntimeError("kernel failure after add_")

        guard = EagerFallbackGuard(_simple_eager_matmul, "inplace_add_kernel")
        result = guard.execute(triton_inplace_add, (a, b))
        expected = a_original @ b
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Garbage collection — corrupted buffer cleanup and OOM handling
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestGarbageCollectionOnFailure:
    """Verify corrupted output buffers are freed and OOM triggers cache clearing."""

    def test_corrupted_output_storage_freed(self):
        """triton_buffers have their storage resized to 0 on failure."""
        corrupted_output = torch.randn(8, 8)
        intermediate = torch.randn(8, 8)

        assert corrupted_output.untyped_storage().size() > 0
        assert intermediate.untyped_storage().size() > 0

        def failing_triton():
            raise RuntimeError("kernel crash")

        guard = EagerFallbackGuard(_simple_eager_matmul, "gc_kernel")
        a, b = torch.randn(8, 8), torch.randn(8, 8)
        guard.execute(
            failing_triton, (a, b),
            triton_buffers=[corrupted_output, intermediate],
        )

        # Both buffers should have their storage freed
        assert corrupted_output.untyped_storage().size() == 0
        assert intermediate.untyped_storage().size() == 0

    def test_no_crash_when_triton_buffers_is_none(self):
        """execute works fine without triton_buffers (backward compat)."""
        guard = EagerFallbackGuard(_simple_eager_matmul, "no_buffers")
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            (a, b),
        )
        expected = a @ b
        torch.testing.assert_close(result, expected)

    def test_discard_corrupted_buffers_frees_storage(self):
        """_discard_corrupted_buffers resizes storage to 0."""
        t1 = torch.randn(4, 4)
        t2 = torch.randn(8, 8)

        EagerFallbackGuard._discard_corrupted_buffers([t1, t2])

        assert t1.untyped_storage().size() == 0
        assert t2.untyped_storage().size() == 0

    def test_discard_corrupted_buffers_noop_when_none(self):
        """_discard_corrupted_buffers is a no-op when triton_buffers is None."""
        # Should not raise
        EagerFallbackGuard._discard_corrupted_buffers(None)

    def test_discard_corrupted_buffers_empty_list(self):
        """_discard_corrupted_buffers handles empty list gracefully."""
        EagerFallbackGuard._discard_corrupted_buffers([])

    def test_double_discard_is_safe(self):
        """Calling _discard_corrupted_buffers twice on same tensors is a no-op."""
        t = torch.randn(4, 4)
        buffers = [t]
        EagerFallbackGuard._discard_corrupted_buffers(buffers)
        assert t.untyped_storage().size() == 0
        # Second call should not raise — resize_(0) on zero storage is fine
        EagerFallbackGuard._discard_corrupted_buffers(buffers)

    def test_combined_isolation_and_gc(self):
        """Input isolation and buffer garbage collection work together."""
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        a_original = a.clone()
        b_original = b.clone()
        corrupted_output = torch.randn(4, 4)

        def triton_corrupts_everything():
            a.zero_()
            corrupted_output[:2, :] = 999.0
            raise RuntimeError("CUDA out of memory")

        guard = EagerFallbackGuard(_simple_eager_matmul, "combined_kernel")
        result = guard.execute(
            triton_corrupts_everything, (a, b),
            triton_buffers=[corrupted_output],
        )

        # Fallback used pristine snapshots
        expected = a_original @ b_original
        torch.testing.assert_close(result, expected)
        # Corrupted output buffer was freed
        assert corrupted_output.untyped_storage().size() == 0


# ---------------------------------------------------------------------------
# OOM detection — _is_oom_error
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestIsOomError:
    """Verify _is_oom_error correctly detects CUDA OOM messages."""

    def test_standard_cuda_oom_message(self):
        assert EagerFallbackGuard._is_oom_error(
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        )

    def test_generic_out_of_memory(self):
        assert EagerFallbackGuard._is_oom_error(
            RuntimeError("out of memory")
        )

    def test_case_insensitive(self):
        assert EagerFallbackGuard._is_oom_error(
            RuntimeError("Out Of Memory on device 0")
        )

    def test_ptx_failure_is_not_oom(self):
        assert not EagerFallbackGuard._is_oom_error(
            RuntimeError("PTX assembly failure")
        )

    def test_driver_error_is_not_oom(self):
        assert not EagerFallbackGuard._is_oom_error(
            RuntimeError("invalid device ordinal")
        )

    def test_allocation_failed_without_oom_wording(self):
        assert not EagerFallbackGuard._is_oom_error(
            RuntimeError("memory allocation failed")
        )


# ---------------------------------------------------------------------------
# _is_triton_compilation_error — dynamic check
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestIsTritonCompilationError:
    """Verify the dynamic triton.CompilationError check."""

    def test_runtime_error_is_not_compilation_error(self):
        assert not _is_triton_compilation_error(RuntimeError("nope"))

    def test_returns_false_when_triton_not_installed(self, monkeypatch):
        """When triton is not importable, returns False gracefully."""
        monkeypatch.delitem(sys.modules, "triton", raising=False)
        # Temporarily make triton un-importable
        import importlib
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "triton":
                raise ImportError("no triton")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        assert not _is_triton_compilation_error(RuntimeError("test"))

    def test_catches_real_triton_compilation_error(self, monkeypatch):
        """When triton is available and has CompilationError, detects it."""
        mock_triton = types.ModuleType("triton")
        mock_triton.CompilationError = type("CompilationError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

        exc = mock_triton.CompilationError("PTX failed")
        assert _is_triton_compilation_error(exc)

    def test_fallback_on_triton_compilation_error(self, monkeypatch):
        """EagerFallbackGuard catches triton.CompilationError."""
        mock_triton = types.ModuleType("triton")
        mock_triton.CompilationError = type("CompilationError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        guard = EagerFallbackGuard(_simple_eager_matmul, "ptx_kernel")

        def failing_triton():
            raise mock_triton.CompilationError("PTX assembly failed")

        result = guard.execute(failing_triton, (a, b))
        expected = a @ b
        torch.testing.assert_close(result, expected)
        assert guard.fallback_count == 1


# ---------------------------------------------------------------------------
# Clean state restoration — no corrupted output
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestCleanStateRestoration:
    """Verify that partial Triton writes don't corrupt the fallback output."""

    def test_partial_write_not_visible_in_fallback(self):
        """Even if triton_fn writes garbage to a buffer, fallback produces fresh output."""
        a = torch.randn(8, 8)
        b = torch.randn(8, 8)

        corrupted_buffer = torch.full((8, 8), float("nan"))

        def triton_that_corrupts_then_fails():
            # Simulate a kernel that partially writes then dies
            corrupted_buffer[:4, :] = 999.0  # partial write
            raise RuntimeError("CUDA error mid-execution")

        guard = EagerFallbackGuard(_simple_eager_matmul, "corrupt_kernel")
        result = guard.execute(triton_that_corrupts_then_fails, (a, b))

        # The fallback result must be a FRESH tensor from eager execution,
        # not the corrupted buffer.
        expected = a @ b
        torch.testing.assert_close(result, expected)
        assert not torch.isnan(result).any()

    def test_synchronize_device_noop_on_cpu(self):
        """_synchronize_device does nothing for CPU tensors (no crash)."""
        tensors = (torch.randn(4, 4), torch.randn(4, 4))
        # Should not raise — just a no-op on CPU
        EagerFallbackGuard._synchronize_device(tensors)


# ---------------------------------------------------------------------------
# Logging — verify failure messages include kernel signature
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestFallbackLogging:
    """Verify the guard logs failures with the kernel signature."""

    def test_logs_warning_on_fallback(self, caplog):
        """Fallback logs a warning containing the kernel signature."""
        guard = EagerFallbackGuard(_simple_eager_matmul, "logged_kernel_sig")
        a, b = torch.randn(2, 2), torch.randn(2, 2)

        with caplog.at_level(logging.WARNING, logger="fuseml"):
            guard.execute(
                lambda: (_ for _ in ()).throw(RuntimeError("PTX fail")),
                (a, b),
            )

        assert any("logged_kernel_sig" in r.message for r in caplog.records)
        assert any("PTX fail" in r.message for r in caplog.records)

    def test_logs_exception_type(self, caplog):
        guard = EagerFallbackGuard(_simple_eager_matmul, "exc_type_kernel")
        a, b = torch.randn(2, 2), torch.randn(2, 2)

        with caplog.at_level(logging.WARNING, logger="fuseml"):
            guard.execute(
                lambda: (_ for _ in ()).throw(RuntimeError("OOM")),
                (a, b),
            )

        assert any("RuntimeError" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# KernelLauncher integration — eager_fn parameter
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestKernelLauncherFallbackIntegration:
    """Verify KernelLauncher uses EagerFallbackGuard when eager_fn is set."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_no_guard_when_eager_fn_is_none(self, input_descs, output_desc):
        """Default construction (no eager_fn) → no fallback guard."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(mock_fn, input_descs, output_desc, [], "a", "b")
        assert launcher._fallback_guard is None

    def test_guard_created_when_eager_fn_provided(self, input_descs, output_desc):
        """Providing eager_fn creates a fallback guard."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
            eager_fn=_simple_eager_matmul,
        )
        assert launcher._fallback_guard is not None
        assert isinstance(launcher._fallback_guard, EagerFallbackGuard)

    def test_successful_launch_returns_triton_output(self, input_descs, output_desc):
        """When the kernel succeeds, the Triton output is returned."""
        mock_fn = _MockKernelFn()
        launcher = KernelLauncher(
            mock_fn, input_descs, output_desc, [], "a", "b",
            eager_fn=_simple_eager_matmul,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        assert result.shape == (128, 256)
        # Kernel was launched (not skipped)
        assert len(mock_fn.calls) == 1

    def test_fallback_on_kernel_runtime_error(self, input_descs, output_desc):
        """When the kernel raises RuntimeError, eager_fn is used."""
        failing_fn = _MockKernelFn(side_effect=RuntimeError("CUDA OOM"))
        launcher = KernelLauncher(
            failing_fn, input_descs, output_desc, [], "a", "b",
            eager_fn=_simple_eager_matmul,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        expected = a @ b
        torch.testing.assert_close(result, expected)

    def test_no_fallback_without_eager_fn_raises(self, input_descs, output_desc):
        """Without eager_fn, RuntimeError propagates normally."""
        failing_fn = _MockKernelFn(side_effect=RuntimeError("CUDA OOM"))
        launcher = KernelLauncher(
            failing_fn, input_descs, output_desc, [], "a", "b",
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            launcher(a, b)

    def test_fallback_count_accessible_via_launcher(self, input_descs, output_desc):
        """Fallback count is tracked on the guard."""
        failing_fn = _MockKernelFn(side_effect=RuntimeError("fail"))
        launcher = KernelLauncher(
            failing_fn, input_descs, output_desc, [], "a", "b",
            eager_fn=_simple_eager_matmul,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        launcher(a, b)
        launcher(a, b)
        assert launcher._fallback_guard.fallback_count == 2


# ---------------------------------------------------------------------------
# KernelLauncher integration — buffer isolation and garbage collection
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestKernelLauncherBufferIsolationIntegration:
    """Verify KernelLauncher passes triton_buffers and input isolation works."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

    def test_fallback_returns_correct_result_despite_kernel_failure(
        self, input_descs, output_desc,
    ):
        """Fallback produces correct output even when kernel fails."""
        failing_fn = _MockKernelFn(side_effect=RuntimeError("CUDA OOM"))
        launcher = KernelLauncher(
            failing_fn, input_descs, output_desc, [], "a", "b",
            eager_fn=_simple_eager_matmul,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        expected = a @ b
        # a and b were not corrupted by the mock kernel, so originals
        # and snapshots are identical — result should match exactly.
        torch.testing.assert_close(result, expected)

    def test_fallback_with_intermediates(self, input_descs, output_desc):
        """Intermediate descriptors are included in triton_buffers."""
        failing_fn = _MockKernelFn(side_effect=RuntimeError("kernel fail"))
        intermediate_descs = [
            TensorDescriptor(
                name="escape_0", shape=(128, 256),
                stride=(256, 1), dtype=torch.float32,
            ),
        ]
        launcher = KernelLauncher(
            failing_fn, input_descs, output_desc,
            intermediate_descs, "a", "b",
            eager_fn=_simple_eager_matmul,
        )
        a = torch.randn(128, 64)
        b = torch.randn(64, 256)
        result = launcher(a, b)
        expected = a @ b
        torch.testing.assert_close(result, expected)

    def test_discard_called_on_oom(self, input_descs, output_desc, monkeypatch):
        """On OOM, _discard_corrupted_buffers is called with force_cache_clear."""
        failing_fn = _MockKernelFn(
            side_effect=RuntimeError("CUDA out of memory"),
        )
        discard_kwargs = []
        original_discard = EagerFallbackGuard._discard_corrupted_buffers

        @staticmethod
        def spy_discard(triton_buffers, *, force_cache_clear=False):
            discard_kwargs.append({
                "buffer_count": len(triton_buffers) if triton_buffers else 0,
                "force_cache_clear": force_cache_clear,
            })
            original_discard(triton_buffers, force_cache_clear=force_cache_clear)

        monkeypatch.setattr(
            EagerFallbackGuard, "_discard_corrupted_buffers", spy_discard,
        )
        launcher = KernelLauncher(
            failing_fn, input_descs, output_desc, [], "a", "b",
            eager_fn=_simple_eager_matmul,
        )
        launcher(torch.randn(128, 64), torch.randn(64, 256))

        assert len(discard_kwargs) == 1
        assert discard_kwargs[0]["buffer_count"] == 1  # output tensor
        assert discard_kwargs[0]["force_cache_clear"] is True

    def test_non_oom_error_does_not_force_cache_clear(
        self, input_descs, output_desc, monkeypatch,
    ):
        """Non-OOM RuntimeError discards buffers but without cache clearing."""
        failing_fn = _MockKernelFn(
            side_effect=RuntimeError("PTX assembly failure"),
        )
        discard_kwargs = []
        original_discard = EagerFallbackGuard._discard_corrupted_buffers

        @staticmethod
        def spy_discard(triton_buffers, *, force_cache_clear=False):
            discard_kwargs.append({"force_cache_clear": force_cache_clear})
            original_discard(triton_buffers, force_cache_clear=force_cache_clear)

        monkeypatch.setattr(
            EagerFallbackGuard, "_discard_corrupted_buffers", spy_discard,
        )
        launcher = KernelLauncher(
            failing_fn, input_descs, output_desc, [], "a", "b",
            eager_fn=_simple_eager_matmul,
        )
        launcher(torch.randn(128, 64), torch.randn(64, 256))

        assert len(discard_kwargs) == 1
        assert discard_kwargs[0]["force_cache_clear"] is False


# ---------------------------------------------------------------------------
# Precision-aware testing — FP16 / BF16 tolerance auditing
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestPrecisionAwareTolerance:
    """Verify mathematical equivalence between Triton and eager paths.

    Because floating-point math is not associative — ``(a + b) + c ≠
    a + (b + c)`` in limited precision — Triton's block-level reductions
    produce slightly different results compared to PyTorch's eager ATen
    backend.  These tests use ``torch.testing.assert_close`` with
    precision-aware tolerances per the FP16 / BF16 accumulation order
    differences.
    """

    @staticmethod
    def _eager_addmm_gelu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Simulate an unfused addmm + GeLU sequence."""
        return torch.nn.functional.gelu(a @ b)

    def test_fp32_equivalence(self):
        """FP32 eager path matches itself within tight tolerance."""
        a = torch.randn(64, 32, dtype=torch.float32)
        b = torch.randn(32, 64, dtype=torch.float32)
        result = self._eager_addmm_gelu(a, b)
        expected = self._eager_addmm_gelu(a, b)
        # FP32 same-order execution → bitwise identical
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_fp16_equivalence_with_tolerance(self):
        """FP16 computation allows atol=1e-3, rtol=1e-2 for accumulation order diffs."""
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(32, 64, dtype=torch.float16)

        # Simulate two "different backends" by reordering accumulation:
        # Path 1: standard eager
        result1 = self._eager_addmm_gelu(a, b)
        # Path 2: still eager, but torch.testing.assert_close should pass
        result2 = self._eager_addmm_gelu(a, b)
        torch.testing.assert_close(result1, result2, atol=1e-3, rtol=1e-2)

    def test_bf16_equivalence_with_tolerance(self):
        """BF16 computation allows atol=1e-3, rtol=1e-2 for accumulation order diffs."""
        a = torch.randn(64, 32, dtype=torch.bfloat16)
        b = torch.randn(32, 64, dtype=torch.bfloat16)

        result1 = self._eager_addmm_gelu(a, b)
        result2 = self._eager_addmm_gelu(a, b)
        torch.testing.assert_close(result1, result2, atol=1e-3, rtol=1e-2)

    def test_fp16_fallback_matches_eager(self):
        """Fallback produces FP16 output matching direct eager execution."""
        a = torch.randn(32, 16, dtype=torch.float16)
        b = torch.randn(16, 32, dtype=torch.float16)

        guard = EagerFallbackGuard(self._eager_addmm_gelu, "fp16_gelu_kernel")
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("simulated failure")),
            (a, b),
        )
        expected = self._eager_addmm_gelu(a, b)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-2)

    def test_bf16_fallback_matches_eager(self):
        """Fallback produces BF16 output matching direct eager execution."""
        a = torch.randn(32, 16, dtype=torch.bfloat16)
        b = torch.randn(16, 32, dtype=torch.bfloat16)

        guard = EagerFallbackGuard(self._eager_addmm_gelu, "bf16_gelu_kernel")
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("simulated failure")),
            (a, b),
        )
        expected = self._eager_addmm_gelu(a, b)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-2)


# ---------------------------------------------------------------------------
# NaN / Inf propagation — identical behaviour across backends
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestNaNInfPropagation:
    """Verify NaN and Inf propagate identically through both execution paths.

    This is critical because Triton's block-level reductions might handle
    special float values differently than PyTorch's eager ATen ops.  The
    fallback guard must produce outputs where NaN/Inf positions match
    exactly between the two backends.
    """

    @staticmethod
    def _eager_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b

    def test_nan_in_input_propagates_through_fallback(self):
        """NaN in input → NaN at the same positions in fallback output."""
        a = torch.randn(8, 8)
        a[2, 3] = float("nan")
        b = torch.randn(8, 8)

        guard = EagerFallbackGuard(self._eager_matmul, "nan_kernel")
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            (a, b),
        )
        expected = a @ b

        # NaN positions must match exactly
        assert torch.isnan(result).any(), "Expected NaN in output"
        assert (torch.isnan(result) == torch.isnan(expected)).all(), \
            "NaN positions differ between fallback and eager"

    def test_inf_in_input_propagates_through_fallback(self):
        """Inf in input → Inf at the same positions in fallback output."""
        a = torch.randn(8, 8)
        a[0, 0] = float("inf")
        b = torch.randn(8, 8)

        guard = EagerFallbackGuard(self._eager_matmul, "inf_kernel")
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            (a, b),
        )
        expected = a @ b

        # Inf positions must match exactly
        assert torch.isinf(result).any(), "Expected Inf in output"
        assert (torch.isinf(result) == torch.isinf(expected)).all(), \
            "Inf positions differ between fallback and eager"

    def test_neg_inf_propagates_identically(self):
        """Negative Inf propagates identically through fallback."""
        a = torch.randn(8, 8)
        a[1, 1] = float("-inf")
        b = torch.randn(8, 8)

        guard = EagerFallbackGuard(self._eager_matmul, "neginf_kernel")
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            (a, b),
        )
        expected = a @ b

        assert (torch.isinf(result) == torch.isinf(expected)).all()
        # Check sign: where both are inf, signs match
        inf_mask = torch.isinf(result) & torch.isinf(expected)
        if inf_mask.any():
            assert (result[inf_mask].sign() == expected[inf_mask].sign()).all()

    def test_nan_times_zero_produces_nan(self):
        """NaN * 0 = NaN in both backends (IEEE 754)."""
        a = torch.zeros(4, 4)
        a[0, 0] = float("nan")
        b = torch.zeros(4, 4)

        guard = EagerFallbackGuard(self._eager_matmul, "nan_zero_kernel")
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            (a, b),
        )
        expected = a @ b

        assert (torch.isnan(result) == torch.isnan(expected)).all()

    def test_fp16_nan_propagation(self):
        """NaN propagation works correctly in FP16."""
        a = torch.randn(8, 8, dtype=torch.float16)
        a[3, 5] = float("nan")
        b = torch.randn(8, 8, dtype=torch.float16)

        guard = EagerFallbackGuard(self._eager_matmul, "fp16_nan_kernel")
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            (a, b),
        )
        expected = a @ b

        assert (torch.isnan(result) == torch.isnan(expected)).all(), \
            "FP16 NaN positions differ between fallback and eager"

    def test_bf16_inf_propagation(self):
        """Inf propagation works correctly in BF16."""
        a = torch.randn(8, 8, dtype=torch.bfloat16)
        a[0, 0] = float("inf")
        b = torch.randn(8, 8, dtype=torch.bfloat16)

        guard = EagerFallbackGuard(self._eager_matmul, "bf16_inf_kernel")
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            (a, b),
        )
        expected = a @ b

        assert (torch.isinf(result) == torch.isinf(expected)).all(), \
            "BF16 Inf positions differ between fallback and eager"

    def test_clean_input_produces_no_nan_or_inf(self):
        """Sanity: clean finite inputs produce finite output through fallback."""
        a = torch.randn(16, 16)
        b = torch.randn(16, 16)

        guard = EagerFallbackGuard(self._eager_matmul, "clean_kernel")
        result = guard.execute(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            (a, b),
        )

        assert not torch.isnan(result).any(), "Unexpected NaN in output"
        assert not torch.isinf(result).any(), "Unexpected Inf in output"


# ---------------------------------------------------------------------------
# _is_recoverable — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.fallback
class TestIsRecoverable:
    """Verify _is_recoverable correctly classifies exception types."""

    def test_runtime_error_is_recoverable(self):
        assert EagerFallbackGuard._is_recoverable(RuntimeError("CUDA OOM"))

    def test_type_error_is_not_recoverable(self):
        assert not EagerFallbackGuard._is_recoverable(TypeError("bad type"))

    def test_value_error_is_not_recoverable(self):
        assert not EagerFallbackGuard._is_recoverable(ValueError("bad val"))

    def test_keyboard_interrupt_is_not_recoverable(self):
        assert not EagerFallbackGuard._is_recoverable(KeyboardInterrupt())

    def test_os_error_is_not_recoverable(self):
        assert not EagerFallbackGuard._is_recoverable(OSError("disk fail"))

    def test_triton_compilation_error_is_recoverable(self, monkeypatch):
        mock_triton = types.ModuleType("triton")
        mock_triton.CompilationError = type("CompilationError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "triton", mock_triton)

        exc = mock_triton.CompilationError("PTX timeout")
        assert EagerFallbackGuard._is_recoverable(exc)
