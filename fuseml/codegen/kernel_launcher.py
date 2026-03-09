"""KernelLauncher — runtime wrapper for compiled Triton kernels.

Bridges the gap between the @triton.jit callable produced by
``TritonKernelGenerator.compile_and_bind`` and the PyTorch FX graph:
at each forward pass it intercepts the live ``torch.Tensor`` arguments,
extracts their shapes and strides, computes the CUDA launch grid, and
dispatches the compiled kernel.

Non-contiguous tensors are handled correctly: every dimension stride is
read from ``tensor.stride()`` at call time, so arbitrary memory layouts
propagate to Triton's stride-based pointer arithmetic without copies.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from fuseml._logging import logger
from fuseml.codegen.kernel_generator import TensorDescriptor, _classify, _identify_matmul_operands


# ---------------------------------------------------------------------------
# Default block sizes — tune per-GPU architecture; 64/64/32 is a safe start
# for Ampere/Ada.  These are passed as tl.constexpr at launch time.
# ---------------------------------------------------------------------------
_DEFAULT_BLOCK_SIZE_M: int = 64
_DEFAULT_BLOCK_SIZE_N: int = 64
_DEFAULT_BLOCK_SIZE_K: int = 32


class KernelLauncher:
    """Wraps a compiled ``@triton.jit`` fused kernel with runtime dispatch.

    Responsibilities:
    - Extract ``M``, ``N``, ``K`` from the actual runtime tensors.
    - Allocate the output tensor (and any intermediate escape tensors).
    - Compute the 2-D CUDA launch grid via ``triton.cdiv``.
    - Assemble the flat argument list required by the generated kernel
      signature (pointers → dimensions → strides → constexpr block sizes).
    - Call the compiled kernel and return the output tensor.

    Parameters
    ----------
    kernel_fn :
        The ``@triton.jit``-decorated callable returned by
        ``TritonKernelGenerator.compile_and_bind``.
    input_descriptors :
        Ordered list of :class:`~fuseml.codegen.kernel_generator.TensorDescriptor`
        objects for all kernel inputs — **same order** as the generated
        kernel's pointer parameters.  The matmul operands must be among
        these descriptors.
    output_descriptor :
        Descriptor for the 2-D output tensor.
    intermediate_descriptors :
        Descriptors for every escape node that requires an intermediate
        ``tl.store`` (i.e. ``group.intermediate_outputs``).  Pass an empty
        list when no intermediate stores are needed.
    left_name :
        ``name`` field of the left (M×K) matmul operand descriptor.
    right_name :
        ``name`` field of the right (K×N) matmul operand descriptor.
    block_size_m, block_size_n, block_size_k :
        ``tl.constexpr`` tile sizes forwarded to the kernel at launch.
        Defaults to 64/64/32.
    """

    def __init__(
        self,
        kernel_fn: Callable,
        input_descriptors: list[TensorDescriptor],
        output_descriptor: TensorDescriptor,
        intermediate_descriptors: list[TensorDescriptor],
        left_name: str,
        right_name: str,
        block_size_m: int = _DEFAULT_BLOCK_SIZE_M,
        block_size_n: int = _DEFAULT_BLOCK_SIZE_N,
        block_size_k: int = _DEFAULT_BLOCK_SIZE_K,
    ) -> None:
        self._kernel_fn = kernel_fn
        self._input_descriptors = input_descriptors
        self._output_descriptor = output_descriptor
        self._intermediate_descriptors = intermediate_descriptors
        self._block_size_m = block_size_m
        self._block_size_n = block_size_n
        self._block_size_k = block_size_k

        # Pre-compute indices so __call__ does not re-scan the list every time.
        input_names = [d.name for d in input_descriptors]
        try:
            self._left_idx: int = input_names.index(left_name)
        except ValueError:
            raise ValueError(
                f"left_name={left_name!r} not found in input_descriptors: {input_names}"
            )
        try:
            self._right_idx: int = input_names.index(right_name)
        except ValueError:
            raise ValueError(
                f"right_name={right_name!r} not found in input_descriptors: {input_names}"
            )

        logger.debug(
            "KernelLauncher created — inputs=%s, output=%s, intermediates=%s, "
            "left=%s[%d], right=%s[%d], block=(%d,%d,%d)",
            input_names,
            output_descriptor.name,
            [d.name for d in intermediate_descriptors],
            left_name, self._left_idx,
            right_name, self._right_idx,
            block_size_m, block_size_n, block_size_k,
        )

    # ------------------------------------------------------------------
    # Runtime dispatch
    # ------------------------------------------------------------------

    def __call__(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        """Launch the fused Triton kernel and return the output tensor.

        Parameters
        ----------
        *input_tensors :
            Runtime ``torch.Tensor`` objects in the same order as
            *input_descriptors* was supplied at construction.  Non-contiguous
            tensors are supported — strides are read via ``tensor.stride()``
            so the Triton kernel's stride-parameterised pointer arithmetic
            handles any layout without copies.

        Returns
        -------
        torch.Tensor
            The kernel's primary output (shape M×N, dtype from
            *output_descriptor*).  When intermediate escape tensors are
            present they are written to allocated buffers but not returned
            here; their downstream consumers must be wired separately by the
            compiler's graph-substitution phase.
        """
        import triton  # deferred — Triton is not required on CPU-only builds

        if len(input_tensors) != len(self._input_descriptors):
            raise ValueError(
                f"Expected {len(self._input_descriptors)} input tensor(s), "
                f"got {len(input_tensors)}."
            )

        left_t = input_tensors[self._left_idx]
        right_t = input_tensors[self._right_idx]

        # ── Dimension extraction ──────────────────────────────────────
        # Support both 2-D (M, K) and batched (*, M, K) shapes; always
        # use the last two dims.
        M: int = int(left_t.shape[-2])
        K: int = int(left_t.shape[-1])
        N: int = int(right_t.shape[-1])

        device = left_t.device
        out_dtype = self._output_descriptor.dtype

        logger.debug(
            "KernelLauncher dispatch — M=%d, N=%d, K=%d, device=%s, dtype=%s",
            M, N, K, device, out_dtype,
        )

        # ── Output allocation ─────────────────────────────────────────
        output = torch.empty(M, N, dtype=out_dtype, device=device)

        # Intermediate escape buffers (one per escape node).
        intermediate_outputs: list[torch.Tensor] = [
            torch.empty(M, N, dtype=d.dtype, device=device)
            for d in self._intermediate_descriptors
        ]

        # ── Launch grid ───────────────────────────────────────────────
        # Captures M and N by value through the closure.  META is Triton's
        # auto-tuner dict; reading BLOCK_SIZE_* from META ensures the grid
        # is always consistent with the constexpr values passed below.
        bsm = self._block_size_m
        bsn = self._block_size_n

        def grid(META: dict) -> tuple[int, int]:
            return (
                triton.cdiv(M, META["BLOCK_SIZE_M"]),
                triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )

        # ── Argument assembly ─────────────────────────────────────────
        # Order must match the generated kernel signature exactly:
        #   pointers (inputs, output, intermediates)
        #   M, N, K
        #   strides per input (each in dim order from the descriptor)
        #   output strides (m, n)
        #   intermediate strides (m, n) per escape buffer

        # Pointers
        ptr_args: list[int] = [int(t.data_ptr()) for t in input_tensors]
        ptr_args.append(int(output.data_ptr()))
        ptr_args.extend(int(t.data_ptr()) for t in intermediate_outputs)

        # Strides — each input tensor contributes one stride per dimension,
        # in the same axis order as dim_labels used during codegen.  For a
        # 2-D left (m,k) tensor: stride_m, stride_k.  For a 1-D bias (n):
        # stride_n.  Non-contiguous tensors are handled transparently because
        # Triton's pointer arithmetic uses the stride argument directly.
        stride_args: list[int] = []
        for t in input_tensors:
            stride_args.extend(int(s) for s in t.stride())

        # Output strides (always M×N — row-major from torch.empty)
        stride_args.extend([int(output.stride(0)), int(output.stride(1))])

        # Intermediate escape tensor strides (each M×N — row-major)
        for t in intermediate_outputs:
            stride_args.extend([int(t.stride(0)), int(t.stride(1))])

        # ── Kernel launch ─────────────────────────────────────────────
        self._kernel_fn[grid](
            *ptr_args,
            M, N, K,
            *stride_args,
            BLOCK_SIZE_M=bsm,
            BLOCK_SIZE_N=bsn,
            BLOCK_SIZE_K=self._block_size_k,
        )

        return output

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        in_names = [d.name for d in self._input_descriptors]
        intm_names = [d.name for d in self._intermediate_descriptors]
        return (
            f"KernelLauncher("
            f"inputs={in_names}, "
            f"output={self._output_descriptor.name!r}, "
            f"intermediates={intm_names}, "
            f"block=({self._block_size_m},{self._block_size_n},{self._block_size_k}))"
        )
