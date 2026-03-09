"""EagerFallbackGuard — deterministic fallback from Triton to eager PyTorch.

Wraps Triton JIT compilation and kernel execution in a guarded
try/except block.  When the Triton path fails — due to PTX assembly
errors, ``triton.CompilationError``, CUDA out-of-memory during
intermediate allocations, or hardware-specific compilation timeouts —
the guard:

1. **Restores clean state** by synchronising the CUDA device so that no
   partially written data remains in the pre-allocated output tensor.
   A fresh output is then produced by the eager fallback, guaranteeing
   that downstream consumers never observe corrupted memory.

2. **Logs the failure** with the kernel signature so operators can
   diagnose which fused group triggered the fallback without sifting
   through stack traces.

3. **Routes the exact inputs** through the original, unfused PyTorch
   ``fx.Graph`` node sequence (provided as *eager_fn*), producing a
   mathematically equivalent result via ATen eager execution.

Because floating-point math is *not* associative in limited precision
(``(a + b) + c ≠ a + (b + c)``), the Triton kernel and the eager
fallback will generally differ by a small epsilon.  Callers should use
precision-aware tolerances (e.g. ``atol=1e-3, rtol=1e-2`` for FP16 /
BF16) when comparing the two paths.
"""

from __future__ import annotations

from typing import Callable

import torch

from fuseml._logging import logger


def _is_triton_compilation_error(exc: BaseException) -> bool:
    """Check whether *exc* is a ``triton.CompilationError``.

    Triton is an optional dependency — it may not be installed on
    CPU-only builds.  This helper avoids a top-level import and
    gracefully returns ``False`` when Triton is absent.
    """
    try:
        import triton  # noqa: F811
        if hasattr(triton, "CompilationError"):
            return isinstance(exc, triton.CompilationError)
    except ImportError:
        pass
    return False


class EagerFallbackGuard:
    """Deterministic fallback wrapper for Triton kernel execution.

    Parameters
    ----------
    eager_fn :
        A callable ``(*input_tensors) -> torch.Tensor`` that reproduces
        the fused kernel's result via standard PyTorch eager execution
        (i.e. running the original, unfused ``fx.Graph`` node sequence).
        Must accept the same positional tensor arguments that the fused
        kernel consumes and return a single output tensor.
    kernel_signature :
        Human-readable identifier for the kernel (e.g.
        ``"fused_addmm_gelu_layernorm_128x256x64_fp16"``).  Used in
        log messages so operators can quickly identify which fusion
        group triggered a fallback.
    """

    def __init__(
        self,
        eager_fn: Callable[..., torch.Tensor],
        kernel_signature: str,
    ) -> None:
        self._eager_fn = eager_fn
        self._kernel_signature = kernel_signature
        self._fallback_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        triton_launch_fn: Callable[[], torch.Tensor],
        input_tensors: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Try the Triton path; fall back to eager on recoverable failure.

        Parameters
        ----------
        triton_launch_fn :
            Zero-argument callable that launches the Triton kernel and
            returns the output tensor.  This closure captures all
            launch parameters (grid, block sizes, stream handle, etc.).
        input_tensors :
            The original input tensors, passed through to *eager_fn*
            if the Triton path fails.

        Returns
        -------
        torch.Tensor
            The kernel output — either from Triton (fast path) or from
            eager PyTorch execution (fallback path).
        """
        try:
            output = triton_launch_fn()
            return output
        except Exception as exc:
            if not self._is_recoverable(exc):
                raise
            return self._handle_failure(exc, input_tensors)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _is_recoverable(exc: BaseException) -> bool:
        """Decide whether *exc* warrants an eager fallback.

        Recoverable errors:
        - ``RuntimeError`` — covers CUDA OOM, PTX assembly failures,
          driver errors, and Triton launch failures.
        - ``triton.CompilationError`` — Triton-specific compilation
          failure (bad IR, unsupported feature, timeout).

        Non-recoverable errors (re-raised):
        - ``TypeError``, ``ValueError`` — programming bugs that should
          not be silently masked.
        - ``KeyboardInterrupt``, ``SystemExit`` — user/system signals.
        """
        if isinstance(exc, RuntimeError):
            return True
        if _is_triton_compilation_error(exc):
            return True
        return False

    def _handle_failure(
        self,
        exc: BaseException,
        input_tensors: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Log the failure, restore clean CUDA state, run eager fallback."""
        self._fallback_count += 1

        logger.warning(
            "Triton kernel '%s' failed (attempt #%d): %s: %s — "
            "falling back to eager PyTorch execution",
            self._kernel_signature,
            self._fallback_count,
            type(exc).__name__,
            exc,
        )

        # ── Clean state restoration ──────────────────────────────────
        # If the Triton kernel failed mid-execution (e.g. CUDA OOM
        # during an intermediate allocation), the pre-allocated output
        # tensor may contain partially written garbage.  Synchronising
        # the device ensures all pending CUDA work completes (or errors
        # out) so the stream is in a known-good state before we produce
        # a fresh output through the eager path.
        self._synchronize_device(input_tensors)

        return self._execute_eager_fallback(input_tensors)

    @staticmethod
    def _synchronize_device(
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Synchronise the CUDA device to flush partial kernel writes.

        After a failed kernel launch, the CUDA stream may still have
        pending asynchronous work.  ``torch.cuda.synchronize()``
        blocks until all kernels on the device complete, ensuring no
        stale data lingers in the output buffer.

        No-op when all tensors reside on CPU.
        """
        for t in tensors:
            if t.device.type == "cuda":
                torch.cuda.synchronize(t.device)
                return  # all tensors should be on the same device

    def _execute_eager_fallback(
        self,
        input_tensors: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Route inputs through the original unfused PyTorch execution.

        Produces a fresh output tensor via ``eager_fn`` — the pre-
        allocated Triton output buffer is discarded, preventing any
        corrupted partial writes from reaching downstream consumers.
        """
        logger.debug(
            "Executing eager fallback for '%s' with %d input tensor(s)",
            self._kernel_signature,
            len(input_tensors),
        )
        return self._eager_fn(*input_tensors)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fallback_count(self) -> int:
        """Number of times the eager fallback has been triggered."""
        return self._fallback_count

    def __repr__(self) -> str:
        return (
            f"EagerFallbackGuard("
            f"kernel={self._kernel_signature!r}, "
            f"fallbacks={self._fallback_count})"
        )
