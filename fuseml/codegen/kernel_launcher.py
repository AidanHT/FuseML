"""KernelLauncher — runtime wrapper for compiled Triton kernels.

Bridges the gap between the @triton.jit callable produced by
``TritonKernelGenerator.compile_and_bind`` and the PyTorch FX graph:
at each forward pass it intercepts the live ``torch.Tensor`` arguments,
extracts their shapes and strides, computes the CUDA launch grid, and
dispatches the compiled kernel.

**Boundary masking**: The generated Triton kernel always wraps every
``tl.load`` and ``tl.store`` with a 2-D boundary mask of the form
``mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)`` (plus a
K-dimension mask inside the GEMM loop).  The launcher passes the
dynamic tensor dimensions ``M``, ``N``, ``K`` so these masks correctly
guard against out-of-bounds access on non-divisible matrix shapes.

**Native stride arithmetic**: The generated kernel uses stride-
parameterised pointer arithmetic (``ptr + row * stride_row + col *
stride_col``), so non-contiguous layouts — transposed matrices, zero-
stride broadcasts, non-unit-stride views — are handled without any
memory copy.  Only tensors with **negative strides** (e.g. from
``torch.flip()``) are materialised via ``.contiguous()``; Triton's
pointer arithmetic cannot produce valid VRAM offsets for negative
strides.

**SRAM capacity enforcement**: Before launch, the output accumulator
tile size (``BLOCK_SIZE_M × BLOCK_SIZE_N × sizeof(dtype)``) is checked
against the SM shared-memory budget (default 48 KB).  Oversized tiles
are dynamically downscaled by halving the larger dimension until the
tile fits, preventing Triton "Out of Shared Memory" PTX failures.

**CUDA stream synchronization**: The kernel is launched on
``torch.cuda.current_stream()`` so it executes on the same stream as
upstream/downstream PyTorch ops, preventing cross-stream data races
under ``torch.compile``, CUDA graphs, or custom stream contexts.

**Heuristic tuning**: ``num_warps`` and ``num_stages`` are dynamically
selected based on the output precision (FP16/BF16 vs FP32) and the
overall bytes accessed, balancing compute throughput against register
pressure and memory latency hiding.
"""

from __future__ import annotations

from typing import Callable

import torch

from fuseml._logging import logger
from fuseml.codegen.eager_fallback import EagerFallbackGuard
from fuseml.codegen.kernel_generator import (
    TensorDescriptor,
    next_power_of_2,
)


# ---------------------------------------------------------------------------
# Default block sizes — tune per-GPU architecture; 64/64/32 is a safe start
# for Ampere/Ada.  These are passed as tl.constexpr at launch time.
# ---------------------------------------------------------------------------
_DEFAULT_BLOCK_SIZE_M: int = 64
_DEFAULT_BLOCK_SIZE_N: int = 64
_DEFAULT_BLOCK_SIZE_K: int = 32

# Default L2 swizzle group width — controls how many M-blocks are grouped
# together in the 1-D → 2-D block-index mapping.  Adjacent programs in a
# group share the same A-tile rows, maximising L2 cache reuse.  8 is the
# standard value from the Triton GEMM tutorial and works well on Ampere/Ada.
_DEFAULT_GROUP_SIZE_M: int = 8

# Bytes-per-element lookup for heuristic tuning.
_BYTES_PER_ELEMENT: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
}

# Threshold (in bytes) above which we increase num_stages to exploit
# deeper software pipelining.  ~512 KB is roughly where L2 pressure
# starts dominating on Ada Lovelace (sm_89) GPUs.
_LARGE_WORKING_SET_BYTES: int = 512 * 1024

# Ada Lovelace (sm_89) SRAM capacity budget (bytes).  Ada supports up to
# 100 KB configurable shared memory per SM.  This is used by SRAM capacity
# enforcement to prevent "Out of Shared Memory" PTX failures at runtime.
_DEFAULT_SRAM_BUDGET_BYTES: int = 100 * 1024

# Minimum block dimension after SRAM downscaling.  Below 16 the
# Triton tile is so small that launch overhead dominates compute.
_MIN_BLOCK_DIM: int = 16


class KernelLauncher:
    """Wraps a compiled ``@triton.jit`` fused kernel with runtime dispatch.

    Responsibilities:
    - Extract ``M``, ``N``, ``K`` from the actual runtime tensors.
    - Materialise tensors with negative strides (the only layout Triton
      cannot handle via stride-parameterised pointer arithmetic).
    - Enforce SRAM capacity limits by downscaling block dimensions.
    - Launch the kernel on ``torch.cuda.current_stream()``.
    - Allocate the output tensor (zero-initialised when atomic ops are used).
    - Compute the 1-D CUDA launch grid with L2-swizzled block indexing:
      ``total_programs = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)``.
    - Dynamically select ``num_warps`` and ``num_stages`` based on precision
      and working-set size.
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
    group_size_m :
        L2 swizzle group width — controls how many M-blocks are grouped
        together in the 1-D → 2-D block-index mapping.  Adjacent programs
        in a group share the same A-tile rows, maximising L2 cache reuse.
        Defaults to 8.
    reduction_op :
        Optional reduction kind (``"sum"``, ``"max"``, ``"mean"``).
    eager_fn :
        Optional callable ``(*input_tensors) -> torch.Tensor`` that
        reproduces the kernel's result via standard PyTorch eager
        execution.  When provided, the launcher wraps every kernel
        dispatch in an :class:`~fuseml.codegen.eager_fallback.EagerFallbackGuard`
        so that ``triton.CompilationError``, CUDA OOM, and PTX assembly
        failures fall back deterministically to the original unfused
        computation.  When ``None``, errors propagate as before.
    is_autotuned :
        When ``True``, the kernel has a ``@triton.autotune`` decorator
        that manages tile sizes, warp counts, and pipeline stages
        automatically.  The launcher skips its own heuristic tuning and
        SRAM enforcement, and does not pass block-size / num_warps /
        num_stages kwargs — the autotuner selects these at runtime.
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
        group_size_m: int = _DEFAULT_GROUP_SIZE_M,
        reduction_op: str | None = None,
        eager_fn: Callable[..., torch.Tensor] | None = None,
        is_autotuned: bool = False,
    ) -> None:
        # FX graph.call_function() uses target.__name__ to generate node names.
        self.__name__ = "fuseml_fused_kernel"
        self._kernel_fn = kernel_fn
        self._input_descriptors = input_descriptors
        self._output_descriptor = output_descriptor
        self._intermediate_descriptors = intermediate_descriptors
        # Triton requires all BLOCK_SIZE constexprs to be strict powers of
        # two — invalid values produce PTX compilation failures.  Round up
        # any non-power-of-two value so callers don't need to worry about it.
        self._block_size_m = next_power_of_2(block_size_m)
        self._block_size_n = next_power_of_2(block_size_n)
        self._block_size_k = next_power_of_2(block_size_k)
        self._group_size_m = group_size_m
        self._reduction_op = reduction_op
        self._is_autotuned = is_autotuned

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

        # ── Eager fallback guard ──────────────────────────────────────
        # When an eager_fn is provided, wrap every kernel dispatch in a
        # guard that catches triton.CompilationError / RuntimeError and
        # falls back to the original unfused PyTorch execution path.
        kernel_sig = (
            f"fused_{'_'.join(input_names)}"
            f"_{output_descriptor.name}"
            f"_{'x'.join(str(s) for s in output_descriptor.shape)}"
            f"_{output_descriptor.dtype}"
        )
        self._fallback_guard: EagerFallbackGuard | None = None
        if eager_fn is not None:
            self._fallback_guard = EagerFallbackGuard(
                eager_fn=eager_fn,
                kernel_signature=kernel_sig,
            )

        logger.debug(
            "KernelLauncher created — inputs=%s, output=%s, intermediates=%s, "
            "left=%s[%d], right=%s[%d], block=(%d,%d,%d), fallback=%s",
            input_names,
            output_descriptor.name,
            [d.name for d in intermediate_descriptors],
            left_name, self._left_idx,
            right_name, self._right_idx,
            block_size_m, block_size_n, block_size_k,
            "enabled" if self._fallback_guard else "disabled",
        )

    # ------------------------------------------------------------------
    # Negative-stride guard — the only layout Triton cannot handle
    # ------------------------------------------------------------------

    @staticmethod
    def _has_negative_strides(tensor: torch.Tensor) -> bool:
        """Return ``True`` if any stride is negative.

        Triton's pointer arithmetic assumes non-negative strides.
        Negative strides (e.g. from ``torch.flip()``) produce invalid
        VRAM offsets and must be materialised via ``.contiguous()``
        before dispatch.  All other non-contiguous layouts — zero
        strides (broadcast/expand), non-unit strides (sliced views),
        transposed tensors — are handled natively by the kernel's
        stride-parameterised pointer arithmetic.
        """
        if tensor.ndim == 0:
            return False
        return any(s < 0 for s in tensor.stride())

    @staticmethod
    def _materialize_if_needed(tensor: torch.Tensor) -> torch.Tensor:
        """Return a contiguous copy ONLY if the tensor has negative strides.

        This is the minimal safety net for tensors produced by
        ``torch.flip()`` or exotic ``as_strided()`` views with negative
        strides.  Zero-stride (expand/broadcast), non-unit-stride
        (sliced views), and transposed tensors all work correctly with
        Triton's stride-parameterised pointer arithmetic and are passed
        through without any HBM allocation or copy.
        """
        if KernelLauncher._has_negative_strides(tensor):
            logger.debug(
                "Negative-stride materialisation — tensor shape=%s stride=%s, "
                "copying to row-major layout (Triton cannot handle negative strides)",
                tuple(tensor.shape), tuple(tensor.stride()),
            )
            return tensor.contiguous()
        return tensor

    # ------------------------------------------------------------------
    # Heuristic launch-parameter selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_num_warps(
        M: int, N: int, K: int, dtype: torch.dtype,
    ) -> int:
        """Choose ``num_warps`` based on precision and tile size.

        Heuristic rationale:

        * **Half-precision (FP16 / BF16)** kernels benefit from tensor-core
          throughput which is 2× that of FP32.  More warps (8) keep the
          tensor-core pipeline saturated and hide global-memory latency.
        * **FP32** has lower ALU throughput per SM; fewer warps (4) reduce
          register pressure and avoid occupancy cliffs.
        * **Very small problems** (tile area < 1024 elements) get only 2
          warps to avoid scheduling overhead exceeding useful compute.
        """
        tile_area = M * N
        if tile_area < 1024:
            # Tiny problem — minimal warp count to reduce launch overhead.
            return 2
        is_half = dtype in (torch.float16, torch.bfloat16)
        if is_half and tile_area >= 4096:
            # Large half-precision tile — 8 warps saturate tensor cores.
            return 8
        return 4

    @staticmethod
    def _select_num_stages(
        M: int, N: int, K: int, dtype: torch.dtype,
    ) -> int:
        """Choose ``num_stages`` (software pipelining depth) for the K-loop.

        Ada Lovelace (sm_89) uses ``cp.async`` for asynchronous global→shared
        memory copies.  Higher stage counts overlap more ``cp.async`` prefetch
        operations with Tensor Core compute, but each extra stage consumes
        additional registers for the prefetch buffers.

        Heuristic rationale for sm_89:

        * **Deep K dimension** (large working set) benefits from 4–5 stages
          so that the next tile's ``cp.async`` load overlaps with the current
          tile's ``tl.dot``.  Ada's 64K register file per SM can sustain up
          to 5 stages without spilling for typical bf16 tile sizes.
        * **Shallow K** or **small problems** should use 2–3 stages to
          avoid wasting registers on prefetch buffers that never overlap
          enough compute.
        * **Half-precision** (bf16/fp16) data is half the bytes, so for the
          same K the shared-memory footprint per stage is smaller — we can
          afford deeper pipelining without exceeding the 100 KB SRAM budget.
        """
        bpe = _BYTES_PER_ELEMENT.get(dtype, 4)
        # Approximate bytes touched per K-tile iteration:
        #   left tile (M × BLOCK_K) + right tile (BLOCK_K × N)
        # We use K as a proxy for loop depth, not BLOCK_K, since more
        # iterations = more overlap opportunity.
        working_bytes = (M + N) * K * bpe
        is_half = dtype in (torch.float16, torch.bfloat16)
        if working_bytes < _LARGE_WORKING_SET_BYTES:
            # Small working set — 2 stages for shallow K.
            return 2
        if is_half:
            # Half-precision, large K — 5 stages for deep cp.async
            # pipelining on Ada.  bf16 tiles are half the size of fp32,
            # so 5 stages fit comfortably within the 100 KB SRAM budget
            # without overflowing the register file.
            return 5
        # FP32, large K — 3 stages balances register use vs. latency hiding.
        return 3

    # ------------------------------------------------------------------
    # SRAM capacity enforcement
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_sram_capacity(
        block_m: int,
        block_n: int,
        dtype: torch.dtype,
        sram_budget_bytes: int = _DEFAULT_SRAM_BUDGET_BYTES,
    ) -> tuple[int, int]:
        """Downscale block dimensions if the output tile exceeds SRAM budget.

        The SRAM requirement for the output accumulator tile is::

            BLOCK_SIZE_M × BLOCK_SIZE_N × sizeof(dtype)

        If this exceeds *sram_budget_bytes*, the method halves the larger
        dimension iteratively until the tile fits, maintaining power-of-2
        alignment.  A floor of ``_MIN_BLOCK_DIM`` prevents degenerate tiles.

        Returns
        -------
        (new_block_m, new_block_n) — downscaled to fit within the budget.
        """
        bpe = _BYTES_PER_ELEMENT.get(dtype, 4)
        while block_m * block_n * bpe > sram_budget_bytes:
            if block_m <= _MIN_BLOCK_DIM and block_n <= _MIN_BLOCK_DIM:
                logger.warning(
                    "SRAM budget %d bytes exceeded even at minimum tile "
                    "(%d × %d × %d bpe = %d bytes); proceeding with "
                    "minimum tile",
                    sram_budget_bytes, block_m, block_n, bpe,
                    block_m * block_n * bpe,
                )
                break
            # Halve the larger dimension first.
            if block_m >= block_n and block_m > _MIN_BLOCK_DIM:
                block_m //= 2
            elif block_n > _MIN_BLOCK_DIM:
                block_n //= 2
            else:
                block_m //= 2
        logger.debug(
            "SRAM capacity check — block=(%d, %d), dtype=%s, "
            "tile_bytes=%d, budget=%d",
            block_m, block_n, dtype, block_m * block_n * bpe,
            sram_budget_bytes,
        )
        return block_m, block_n

    # ------------------------------------------------------------------
    # CUDA stream acquisition
    # ------------------------------------------------------------------

    @staticmethod
    def _get_launch_stream(device: torch.device) -> int | None:
        """Return the raw CUDA stream handle for Triton, or ``None`` on CPU.

        Triton's ``@triton.jit`` kernel accepts a ``stream=`` keyword
        argument with the raw ``cudaStream_t`` handle (an integer).
        Launching on ``torch.cuda.current_stream()`` ensures the kernel
        executes on the same stream as upstream/downstream PyTorch ops.
        """
        if device.type == "cuda":
            return torch.cuda.current_stream(device).cuda_stream
        return None

    # ------------------------------------------------------------------
    # Runtime dispatch
    # ------------------------------------------------------------------

    def __call__(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        """Launch the fused Triton kernel and return the output tensor.

        Parameters
        ----------
        *input_tensors :
            Runtime ``torch.Tensor`` objects in the same order as
            *input_descriptors* was supplied at construction.  Only
            tensors with **negative strides** (e.g. ``torch.flip()``)
            are materialised to contiguous memory; all other non-
            contiguous layouts (zero-stride broadcasts, non-unit-stride
            views, transposed matrices) are passed through as-is — the
            kernel's stride-parameterised pointer arithmetic handles
            them natively without any HBM allocation.

        Returns
        -------
        torch.Tensor
            The kernel's primary output (shape M×N, dtype from
            *output_descriptor*).  When atomic reductions are used the
            output is pre-zeroed (or ``-inf`` for max).  When intermediate
            escape tensors are present they are written to allocated
            buffers but not returned here; their downstream consumers
            must be wired separately by the compiler's graph-substitution
            phase.
        """
        import triton  # deferred — Triton is not required on CPU-only builds

        if len(input_tensors) != len(self._input_descriptors):
            raise ValueError(
                f"Expected {len(self._input_descriptors)} input tensor(s), "
                f"got {len(input_tensors)}."
            )

        # ── Negative-stride guard ─────────────────────────────────────
        # Only tensors with negative strides need materialisation.
        # All other layouts (zero-stride, non-unit-stride, transposed)
        # are handled natively by the kernel's stride-parameterised
        # pointer arithmetic — no HBM allocation or copy needed.
        input_tensors = tuple(self._materialize_if_needed(t) for t in input_tensors)

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

        # ── Heuristic launch parameters ──────────────────────────────
        # When the kernel is autotuned, block sizes, warp counts, and
        # pipeline stages are embedded in the @triton.autotune configs
        # and selected automatically at runtime — skip static heuristics.
        if not self._is_autotuned:
            num_warps = self._select_num_warps(M, N, K, out_dtype)
            num_stages = self._select_num_stages(M, N, K, out_dtype)

            # ── SRAM capacity enforcement ─────────────────────────────
            # Downscale block dims if the output tile exceeds shared memory.
            block_m, block_n = self._enforce_sram_capacity(
                self._block_size_m, self._block_size_n, out_dtype,
            )
            block_k = self._block_size_k

            # If SRAM enforcement reduced the tile, compensate with +1
            # stage to improve latency hiding (more loop iterations).
            if block_m < self._block_size_m or block_n < self._block_size_n:
                num_stages = min(num_stages + 1, 4)

        # ── CUDA stream synchronization ───────────────────────────────
        # Launch on PyTorch's current stream to avoid cross-stream races.
        stream_handle = self._get_launch_stream(device)

        if self._is_autotuned:
            logger.debug(
                "KernelLauncher dispatch (autotuned) — M=%d, N=%d, K=%d, "
                "device=%s, dtype=%s",
                M, N, K, device, out_dtype,
            )
        else:
            logger.debug(
                "KernelLauncher dispatch — M=%d, N=%d, K=%d, device=%s, "
                "dtype=%s, num_warps=%d, num_stages=%d, block=(%d,%d,%d)",
                M, N, K, device, out_dtype, num_warps, num_stages,
                block_m, block_n, block_k,
            )

        # ── Output allocation ─────────────────────────────────────────
        # Zero-initialise when atomic additions are used in the kernel
        # (reductions); use -inf for atomic max so the first real value
        # always wins.  Non-reduction outputs use torch.empty for speed.
        is_reduced = len(self._output_descriptor.shape) == 1
        if is_reduced:
            if self._reduction_op == "max":
                output = torch.full((M,), float("-inf"), dtype=out_dtype, device=device)
            else:
                # sum / mean — zero-init for tl.atomic_add correctness.
                output = torch.zeros(M, dtype=out_dtype, device=device)
        else:
            output = torch.empty(M, N, dtype=out_dtype, device=device)

        # Intermediate escape buffers (one per escape node).
        intermediate_outputs: list[torch.Tensor] = [
            torch.empty(M, N, dtype=d.dtype, device=device)
            for d in self._intermediate_descriptors
        ]

        # ── Launch grid ───────────────────────────────────────────────
        # 1-D grid: total_programs = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)
        # The kernel's L2 swizzling logic maps this 1-D program index to
        # 2-D (pid_m, pid_n) using GROUP_SIZE_M-wide column-major stripes,
        # maximising L2 data reuse across adjacent thread blocks.
        def grid(META: dict) -> tuple[int]:
            return (
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )

        # ── Argument assembly ─────────────────────────────────────────
        # Order must match the generated kernel signature exactly:
        #   pointers (inputs, output, intermediates)
        #   M, N, K              ← dynamic dims for boundary masks
        #   strides per input (each in dim order from the descriptor)
        #   output strides (m, n)
        #   intermediate strides (m, n) per escape buffer

        # Tensor args — Triton 3.x expects actual torch.Tensor objects as
        # pointer parameters (not raw data_ptr() ints).  Triton extracts
        # the device pointer internally from the tensor's storage.
        tensor_args: list[torch.Tensor] = list(input_tensors)
        tensor_args.append(output)
        tensor_args.extend(intermediate_outputs)

        # Strides — each input tensor contributes one stride per dimension,
        # in the same axis order as dim_labels used during codegen.  For a
        # 2-D left (m,k) tensor: stride_m, stride_k.  For a 1-D bias (n):
        # stride_n.  Non-contiguous tensors are handled transparently because
        # Triton's pointer arithmetic uses the stride argument directly.
        stride_args: list[int] = []
        for t in input_tensors:
            stride_args.extend(int(s) for s in t.stride())

        # Output strides — 1-D for reduced output, 2-D otherwise.
        if is_reduced:
            stride_args.append(int(output.stride(0)))
        else:
            stride_args.extend([int(output.stride(0)), int(output.stride(1))])

        # Intermediate escape tensor strides (each M×N — row-major)
        for t in intermediate_outputs:
            stride_args.extend([int(t.stride(0)), int(t.stride(1))])

        # ── Kernel launch ─────────────────────────────────────────────
        # M, N, K are passed as positional args so the generated kernel
        # can use them in boundary masks:
        #   mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        #
        # When the kernel is autotuned, block sizes and tuning params
        # are managed by @triton.autotune — we pass an empty kwargs dict
        # so the autotuner selects the optimal config at runtime.
        if self._is_autotuned:
            launch_kwargs: dict = {}
        else:
            launch_kwargs = {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": self._group_size_m,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        # Note: Triton 3.x deprecated the ``stream=`` kwarg — the kernel
        # automatically uses the current CUDA stream.

        def _triton_launch() -> torch.Tensor:
            """Closure capturing all launch state for the fallback guard."""
            self._kernel_fn[grid](
                *tensor_args,
                M, N, K,
                *stride_args,
                **launch_kwargs,
            )

            # ── Post-kernel fixup for mean reduction ─────────────────
            # The kernel accumulates partial sums via tl.atomic_add;
            # the division by the full dimension size is applied here
            # for better numerical precision than dividing inside each
            # program.
            result = output
            if self._reduction_op == "mean":
                result = result / N
            return result

        # ── Guarded execution ────────────────────────────────────────
        # When a fallback guard is configured, wrap the Triton launch
        # so that CompilationError / RuntimeError (CUDA OOM, PTX
        # failures) are caught and the inputs are re-routed through
        # the original unfused PyTorch execution path.  The guard:
        #   1. Snapshots inputs (clone) before the launch so the
        #      fallback always operates on pristine, uncorrupted data.
        #   2. Frees the pre-allocated output + intermediate buffers
        #      on failure (resizes storage to 0) to reclaim GPU memory.
        #   3. Calls torch.cuda.empty_cache() on OOM to defragment
        #      the allocator before the eager fallback allocates.
        if self._fallback_guard is not None:
            return self._fallback_guard.execute(
                _triton_launch,
                input_tensors,
                triton_buffers=[output] + intermediate_outputs,
            )

        return _triton_launch()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        in_names = [d.name for d in self._input_descriptors]
        intm_names = [d.name for d in self._intermediate_descriptors]
        reduction = f", reduction={self._reduction_op!r}" if self._reduction_op else ""
        tuning = ", autotuned=True" if self._is_autotuned else ""
        return (
            f"KernelLauncher("
            f"inputs={in_names}, "
            f"output={self._output_descriptor.name!r}, "
            f"intermediates={intm_names}, "
            f"block=({self._block_size_m},{self._block_size_n},{self._block_size_k})"
            f"{reduction}{tuning})"
        )
