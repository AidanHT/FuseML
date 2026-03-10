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

from dataclasses import dataclass
from typing import Callable

import torch

from fuseml._logging import logger
from fuseml.codegen.eager_fallback import EagerFallbackGuard
from fuseml.codegen.kernel_generator import (
    TensorDescriptor,
    next_power_of_2,
)
from fuseml.codegen.sram_autotuner import SRAMAutotuner


# ---------------------------------------------------------------------------
# Default block sizes — tune per-GPU architecture; 64/64/32 is a safe start
# for Ampere/Ada.  These are passed as tl.constexpr at launch time.
# ---------------------------------------------------------------------------
_DEFAULT_BLOCK_SIZE_M: int = 128
_DEFAULT_BLOCK_SIZE_N: int = 128
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


# ---------------------------------------------------------------------------
# LaunchParams — pre-computed, immutable kernel launch configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaunchParams:
    """Pre-computed, immutable kernel launch configuration.

    All SRAM budgeting, heuristic warp/stage selection, and block-size
    enforcement is resolved **once** at compilation time (inside
    :func:`compute_launch_params`) and frozen into this dataclass.  The
    :class:`KernelLauncher` reads these constants in *O(1)* with zero
    conditional branching on the dispatch hot-path.

    Attributes
    ----------
    block_m, block_n, block_k :
        Tile dimensions (all powers of two) passed as ``tl.constexpr``
        to the Triton kernel.  Already SRAM-safe.
    group_size_m :
        L2 swizzle group width for the 1-D → 2-D block-index mapping.
    num_warps :
        Number of warps per thread block.
    num_stages :
        Software-pipelining depth for the K-loop.
    """

    block_m: int
    block_n: int
    block_k: int
    group_size_m: int
    num_warps: int
    num_stages: int


# ---------------------------------------------------------------------------
# Static heuristic pre-computation
# ---------------------------------------------------------------------------

def _select_num_warps(
    block_m: int, block_n: int, dtype: torch.dtype,
) -> int:
    """Choose ``num_warps`` based on precision and **tile** (block) size.

    Heuristic rationale:

    * **Half-precision (FP16 / BF16)** kernels benefit from tensor-core
      throughput which is 2x that of FP32.  More warps (8) keep the
      tensor-core pipeline saturated and hide global-memory latency.
    * **FP32** has lower ALU throughput per SM; fewer warps (4) reduce
      register pressure and avoid occupancy cliffs.
    * **Very small tiles** (area < 1024 elements) get only 2 warps to
      avoid scheduling overhead exceeding useful compute.

    Note: ``block_m`` and ``block_n`` are the tile dimensions (not the
    full matrix dimensions) so the heuristic correctly adapts to the
    actual per-SM workload.
    """
    tile_area = block_m * block_n
    if tile_area < 1024:
        return 2
    is_half = dtype in (torch.float16, torch.bfloat16)
    if is_half and tile_area >= 4096:
        return 8
    return 4


def _select_num_stages(
    M: int, N: int, K: int, dtype: torch.dtype,
) -> int:
    """Choose ``num_stages`` (software pipelining depth) for the K-loop.

    Ada Lovelace (sm_89) uses ``cp.async`` for asynchronous global→shared
    memory copies.  Higher stage counts overlap more ``cp.async`` prefetch
    operations with Tensor Core compute, but each extra stage consumes
    additional registers for the prefetch buffers.
    """
    bpe = _BYTES_PER_ELEMENT.get(dtype, 4)
    working_bytes = (M + N) * K * bpe
    is_half = dtype in (torch.float16, torch.bfloat16)
    if working_bytes < _LARGE_WORKING_SET_BYTES:
        return 2
    if is_half:
        return 5
    return 3


def _enforce_sram_capacity(
    block_m: int,
    block_n: int,
    dtype: torch.dtype,
    sram_budget_bytes: int = _DEFAULT_SRAM_BUDGET_BYTES,
) -> tuple[int, int]:
    """Downscale block dimensions if the output tile exceeds SRAM budget.

    Returns ``(new_block_m, new_block_n)`` — downscaled to fit within
    the budget while maintaining power-of-2 alignment and a floor of
    ``_MIN_BLOCK_DIM``.
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


def compute_launch_params(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    block_size_m: int = _DEFAULT_BLOCK_SIZE_M,
    block_size_n: int = _DEFAULT_BLOCK_SIZE_N,
    block_size_k: int = _DEFAULT_BLOCK_SIZE_K,
    group_size_m: int = _DEFAULT_GROUP_SIZE_M,
    sram_autotuner: SRAMAutotuner | None = None,
    sram_budget_bytes: int = _DEFAULT_SRAM_BUDGET_BYTES,
) -> LaunchParams:
    """Pre-compute all kernel launch parameters at compilation time.

    Encapsulates SRAM budgeting, heuristic warp/stage selection, and
    block-size enforcement into a single call that returns a frozen
    :class:`LaunchParams`.  This removes all conditional branching from
    the :class:`KernelLauncher` dispatch hot-path.

    Three resolution paths, in priority order:

    1. **SRAMAutotuner** — AOT dynamic search over all SRAM-safe configs,
       scored against ``(M, N, K, dtype)`` and cached.
    2. **Static heuristics** — ``_select_num_warps`` /
       ``_select_num_stages`` / ``_enforce_sram_capacity``.

    Parameters
    ----------
    M, N, K :
        Matrix dimensions from the traced tensor descriptors.
    dtype :
        Output precision — drives SRAM footprint and warp selection.
    block_size_m, block_size_n, block_size_k :
        Initial tile sizes (rounded to power-of-2 internally).
    group_size_m :
        L2 swizzle group width.
    sram_autotuner :
        When provided, replaces static heuristics with a dynamic
        SRAM-aware configuration search.
    sram_budget_bytes :
        Maximum SRAM budget for static heuristic path.

    Returns
    -------
    LaunchParams
        Frozen, SRAM-safe launch configuration.
    """
    block_m = next_power_of_2(block_size_m)
    block_n = next_power_of_2(block_size_n)
    block_k = next_power_of_2(block_size_k)

    if sram_autotuner is not None:
        cfg = sram_autotuner.select_config(M, N, K, dtype)
        return LaunchParams(
            block_m=cfg.block_m,
            block_n=cfg.block_n,
            block_k=cfg.block_k,
            group_size_m=group_size_m,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    # Static heuristic path
    num_stages = _select_num_stages(M, N, K, dtype)

    orig_block_m, orig_block_n = block_m, block_n
    block_m, block_n = _enforce_sram_capacity(
        block_m, block_n, dtype, sram_budget_bytes,
    )

    # Select num_warps based on final (post-SRAM-enforcement) tile dims.
    num_warps = _select_num_warps(block_m, block_n, dtype)

    # Compensate for SRAM-induced tile shrinkage with deeper pipelining.
    if block_m < orig_block_m or block_n < orig_block_n:
        num_stages = min(num_stages + 1, 4)

    return LaunchParams(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_size_m=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


class KernelLauncher:
    """Wraps a compiled ``@triton.jit`` fused kernel with runtime dispatch.

    The launcher is a **dumb executor** — all SRAM budgeting, heuristic
    warp/stage selection, and block-size enforcement are resolved at
    compilation time and stored in a frozen :class:`LaunchParams`.  At
    dispatch, the launcher reads these pre-computed constants in *O(1)*
    with zero conditional branching for hardware limits.

    Responsibilities:
    - Extract ``M``, ``N``, ``K`` from the actual runtime tensors.
    - Materialise tensors with negative strides (the only layout Triton
      cannot handle via stride-parameterised pointer arithmetic).
    - Launch the kernel on ``torch.cuda.current_stream()``.
    - Allocate the output tensor (zero-initialised when atomic ops are used).
    - Compute the 1-D CUDA launch grid with L2-swizzled block indexing:
      ``total_programs = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)``.
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
    launch_params :
        Pre-computed :class:`LaunchParams` containing SRAM-safe block
        sizes, warp count, and pipeline stages.  When provided, the
        launcher uses these directly with zero runtime computation.
        When ``None`` and *is_autotuned* is ``False``, the launcher
        auto-computes parameters from the descriptor shapes and the
        *block_size_m/n/k* defaults via :func:`compute_launch_params`.
    block_size_m, block_size_n, block_size_k :
        Initial ``tl.constexpr`` tile sizes used **only** when
        *launch_params* is ``None`` (backward compatibility).
        Defaults to 64/64/32.
    group_size_m :
        L2 swizzle group width used **only** when *launch_params*
        is ``None``.  Defaults to 8.
    reduction_op :
        Optional reduction kind (``"sum"``, ``"max"``, ``"mean"``).
    reduction_axis :
        Triton tile axis that was collapsed (``0`` = M, ``1`` = N).
        Required when *reduction_op* is set.  Determines the surviving
        dimension for output allocation and the reduced dimension for
        mean division.
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
        automatically.  The launcher passes an empty kwargs dict and
        *launch_params* is ignored.
    sram_autotuner :
        Optional :class:`~fuseml.codegen.sram_autotuner.SRAMAutotuner`
        used **only** when *launch_params* is ``None`` (backward
        compatibility fallback for auto-computation at init time).
    mean_epilogue_fused :
        When ``True`` (the default), the Triton epilogue already contains
        the reciprocal multiply (``partial_sum * (1.0 / dim)``), so no
        post-kernel division is needed — only an FP32→output dtype cast.
        Set to ``False`` for backward compatibility with kernels whose
        epilogues accumulate raw sums and rely on the launcher for the
        ``result / dim`` fixup.
    """

    # ── Post-kernel strategy constants ─────────────────────────────
    # Pre-computed at __init__ and stored in ``_post_kernel_strategy``
    # so that ``__call__`` dispatches via integer comparison — zero
    # data-dependent branching on the hot-path.
    _POST_KERNEL_NOOP: int = 0        # Return output directly
    _POST_KERNEL_CAST_ONLY: int = 1   # Fused mean: just cast FP32 → out_dtype
    _POST_KERNEL_MEAN_DIV: int = 2    # Legacy mean: mul_(reciprocal) then cast

    def __init__(
        self,
        kernel_fn: Callable,
        input_descriptors: list[TensorDescriptor],
        output_descriptor: TensorDescriptor,
        intermediate_descriptors: list[TensorDescriptor],
        left_name: str,
        right_name: str,
        launch_params: LaunchParams | None = None,
        block_size_m: int = _DEFAULT_BLOCK_SIZE_M,
        block_size_n: int = _DEFAULT_BLOCK_SIZE_N,
        block_size_k: int = _DEFAULT_BLOCK_SIZE_K,
        group_size_m: int = _DEFAULT_GROUP_SIZE_M,
        reduction_op: str | None = None,
        reduction_axis: int | None = None,
        eager_fn: Callable[..., torch.Tensor] | None = None,
        is_autotuned: bool = False,
        sram_autotuner: SRAMAutotuner | None = None,
        mean_epilogue_fused: bool = True,
    ) -> None:
        # FX graph.call_function() uses target.__name__ to generate node names.
        self.__name__ = "fuseml_fused_kernel"
        self._kernel_fn = kernel_fn
        self._input_descriptors = input_descriptors
        self._output_descriptor = output_descriptor
        self._intermediate_descriptors = intermediate_descriptors
        self._reduction_op = reduction_op
        self._reduction_axis = reduction_axis if reduction_axis is not None else 1
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

        # ── Static launch-parameter pre-computation ───────────────────
        # Resolve all SRAM budgeting, block sizing, warp/stage selection
        # once at construction time.  The __call__ hot-path reads these
        # frozen constants in O(1) with zero conditional branching.
        if launch_params is not None:
            self._launch_params: LaunchParams | None = launch_params
        elif not is_autotuned:
            # Backward compat: auto-compute from descriptor shapes.
            left_desc = input_descriptors[self._left_idx]
            right_desc = input_descriptors[self._right_idx]
            M = int(left_desc.shape[-2])
            K = int(left_desc.shape[-1])
            N = int(right_desc.shape[-1])
            self._launch_params = compute_launch_params(
                M, N, K, output_descriptor.dtype,
                block_size_m=block_size_m,
                block_size_n=block_size_n,
                block_size_k=block_size_k,
                group_size_m=group_size_m,
                sram_autotuner=sram_autotuner,
            )
        else:
            # Autotuned — Triton's @triton.autotune manages everything.
            self._launch_params = None

        # ── Pre-frozen launch kwargs ──────────────────────────────────
        # Built once at init; reused every dispatch — avoids dict literal
        # construction on the hot-path.  ``__call__`` passes these via
        # ``**self._frozen_launch_kwargs`` which is O(6) pointer copy.
        if not is_autotuned and self._launch_params is not None:
            lp = self._launch_params
            self._frozen_launch_kwargs: dict[str, int] = {
                "BLOCK_SIZE_M": lp.block_m,
                "BLOCK_SIZE_N": lp.block_n,
                "BLOCK_SIZE_K": lp.block_k,
                "GROUP_SIZE_M": lp.group_size_m,
                "num_warps": lp.num_warps,
                "num_stages": lp.num_stages,
            }
        else:
            self._frozen_launch_kwargs = {}

        # ── Pre-computed output allocation parameters ─────────────────
        # Eliminates per-dispatch branching on reduction_op for output
        # tensor initialization.  All factory calls use explicit device=
        # and dtype= to avoid implicit host synchronization.
        self._is_reduced: bool = len(output_descriptor.shape) == 1
        if self._is_reduced:
            if reduction_op == "max":
                self._alloc_fill_value: float = float("-inf")
            else:
                # sum and mean both need zero-initialized accumulators
                self._alloc_fill_value = 0.0
            # Mean accumulates in FP32 for numerical stability across atomics
            self._alloc_dtype: torch.dtype = (
                torch.float32 if reduction_op == "mean" else output_descriptor.dtype
            )
        else:
            self._alloc_fill_value = 0.0  # unused for 2-D path
            self._alloc_dtype = output_descriptor.dtype

        # ── Pre-computed post-kernel strategy ─────────────────────────
        # Determines what (if anything) happens after the Triton kernel
        # returns, encoded as an integer for branchless dispatch.
        #
        # _POST_KERNEL_NOOP     : non-mean ops → return output directly
        # _POST_KERNEL_CAST_ONLY: fused mean   → cast FP32 → out_dtype
        # _POST_KERNEL_MEAN_DIV : legacy mean  → mul_(1/dim), then cast
        self._mean_epilogue_fused: bool = mean_epilogue_fused
        if reduction_op == "mean":
            if mean_epilogue_fused:
                self._post_kernel_strategy: int = self._POST_KERNEL_CAST_ONLY
            else:
                self._post_kernel_strategy = self._POST_KERNEL_MEAN_DIV
        else:
            self._post_kernel_strategy = self._POST_KERNEL_NOOP

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

        lp_info = (
            f"block=({self._launch_params.block_m},"
            f"{self._launch_params.block_n},"
            f"{self._launch_params.block_k})"
            if self._launch_params else "autotuned"
        )
        logger.debug(
            "KernelLauncher created — inputs=%s, output=%s, intermediates=%s, "
            "left=%s[%d], right=%s[%d], %s, fallback=%s",
            input_names,
            output_descriptor.name,
            [d.name for d in intermediate_descriptors],
            left_name, self._left_idx,
            right_name, self._right_idx,
            lp_info,
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
        # Fast-path bitwise check: OR all strides together.  If any
        # stride is negative, the sign bit propagates through OR,
        # making the result negative — avoids Python-level generator.
        bits = 0
        for s in tensor.stride():
            bits |= s
        return bits < 0

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
    # Runtime dispatch — CUDA-graph-safe, zero host-side data-dependent
    # branching.  All per-instance decisions were resolved at __init__.
    # ------------------------------------------------------------------

    def __call__(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        """Launch the fused Triton kernel and return the output tensor.

        **CUDA Graph safety guarantees**:

        1. **No host-side data-dependent control flow.**  Every branch in
           this method is resolved by instance-level constants that were
           pre-computed at ``__init__`` time (``_is_reduced``,
           ``_post_kernel_strategy``, ``_frozen_launch_kwargs``).  No
           branch inspects runtime tensor *values* or dynamically-
           computed tensor *shapes*.

        2. **Allocation safeness.**  All ``torch.empty_strided`` /
           ``torch.full`` calls specify ``device=`` and ``dtype=``
           explicitly to avoid implicit host synchronisation or
           uncaptured memory-pool expansion.

        3. **Flat argument passing.**  Dynamic shapes (``M``, ``N``,
           ``K``) and strides are passed as explicit Python ``int``
           positional arguments.  Launch kwargs (block sizes, warps,
           stages) are unpacked from ``self._frozen_launch_kwargs``
           which was built once at ``__init__`` — no per-dispatch dict
           literal construction.

        4. **No post-kernel fixup kernels.**  Mean division is fused
           into the Triton epilogue (reciprocal multiply before
           ``tl.atomic_add``).  The only post-kernel op is an optional
           FP32 → output-dtype cast via ``Tensor.to()`` which is a
           single capturable CUDA kernel (and a no-op when dtypes
           already match).

        Parameters
        ----------
        *input_tensors :
            Runtime ``torch.Tensor`` objects in the same order as
            *input_descriptors* was supplied at construction.  Only
            tensors with **negative strides** (e.g. ``torch.flip()``)
            are materialised to contiguous memory; all other non-
            contiguous layouts are passed through as-is.

        Returns
        -------
        torch.Tensor
            The kernel's primary output.
        """
        if len(input_tensors) != len(self._input_descriptors):
            raise ValueError(
                f"Expected {len(self._input_descriptors)} input tensor(s), "
                f"got {len(input_tensors)}."
            )

        # ── Negative-stride fast-path ─────────────────────────────────
        # CPU-side metadata check only — no GPU synchronisation.
        input_tensors = tuple(self._materialize_if_needed(t) for t in input_tensors)

        left_t = input_tensors[self._left_idx]
        right_t = input_tensors[self._right_idx]

        # ── Dimension extraction (CPU metadata, no host-GPU sync) ────
        M: int = int(left_t.shape[-2])
        K: int = int(left_t.shape[-1])
        N: int = int(right_t.shape[-1])

        device: torch.device = left_t.device
        out_dtype: torch.dtype = self._output_descriptor.dtype

        logger.debug(
            "KernelLauncher dispatch — M=%d, N=%d, K=%d, device=%s, "
            "dtype=%s, kwargs=%s",
            M, N, K, device, out_dtype, self._frozen_launch_kwargs,
        )

        # ── Output allocation — explicit device/dtype, pre-computed ───
        # The allocation kind (_is_reduced, _alloc_fill_value,
        # _alloc_dtype) was resolved at __init__; only M, N are dynamic.
        if self._is_reduced:
            surviving_size: int = M if self._reduction_axis == 1 else N
            output = torch.full(
                (surviving_size,),
                self._alloc_fill_value,
                dtype=self._alloc_dtype,
                device=device,
            )
        else:
            output = torch.empty_strided(
                (M, N), (N, 1), dtype=out_dtype, device=device,
            )

        # Intermediate escape buffers — explicit device/dtype.
        intermediate_outputs: list[torch.Tensor] = [
            torch.empty_strided(
                (M, N), (N, 1), dtype=d.dtype, device=device,
            )
            for d in self._intermediate_descriptors
        ]

        # ── Launch grid — deferred lambda for Triton constexpr ────────
        grid = lambda META: (  # noqa: E731
            ((M + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M'])
            * ((N + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N']),
        )

        # ── Flat argument assembly ────────────────────────────────────
        # All dynamic shapes and strides are passed as explicit Python
        # ints — no dict construction, no runtime type coercion.
        tensor_args: list[torch.Tensor] = list(input_tensors)
        tensor_args.append(output)
        tensor_args.extend(intermediate_outputs)

        stride_args: list[int] = []
        for t in input_tensors:
            stride_args.extend(int(s) for s in t.stride())

        if self._is_reduced:
            stride_args.append(int(output.stride(0)))
        else:
            stride_args.extend([int(output.stride(0)), int(output.stride(1))])

        for t in intermediate_outputs:
            stride_args.extend([int(t.stride(0)), int(t.stride(1))])

        # ── Kernel launch + post-kernel fixup ─────────────────────────
        # The closure is kept for EagerFallbackGuard compatibility but
        # contains zero data-dependent branching — the post-kernel
        # strategy integer was resolved at __init__.
        post_strategy = self._post_kernel_strategy

        def _triton_launch() -> torch.Tensor:
            """Closure capturing all launch state for the fallback guard."""
            self._kernel_fn[grid](
                *tensor_args,
                M, N, K,
                *stride_args,
                **self._frozen_launch_kwargs,
            )

            # Post-kernel: integer dispatch, no data-dependent branching.
            if post_strategy == KernelLauncher._POST_KERNEL_CAST_ONLY:
                # Fused mean: epilogue already divided — just cast.
                # Tensor.to() is a no-op when dtypes already match.
                return output.to(out_dtype)
            if post_strategy == KernelLauncher._POST_KERNEL_MEAN_DIV:
                # Legacy mean: in-place scalar multiply (single CUDA op)
                # instead of tensor / scalar (which constructs a scalar
                # tensor on the fly).
                reduced_dim_size = N if self._reduction_axis == 1 else M
                return output.mul_(1.0 / reduced_dim_size).to(out_dtype)
            return output

        # ── Guarded execution ────────────────────────────────────────
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
        reduction = (
            f", reduction={self._reduction_op!r}(axis={self._reduction_axis})"
            if self._reduction_op else ""
        )
        tuning = ", autotuned=True" if self._is_autotuned else ""
        if self._launch_params is not None:
            lp = self._launch_params
            block_str = f"block=({lp.block_m},{lp.block_n},{lp.block_k})"
        else:
            block_str = "block=(autotuned)"
        return (
            f"KernelLauncher("
            f"inputs={in_names}, "
            f"output={self._output_descriptor.name!r}, "
            f"intermediates={intm_names}, "
            f"{block_str}"
            f"{reduction}{tuning})"
        )
