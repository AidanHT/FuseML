"""Triton kernel code generation — signature, pointers, K-loop, epilogue, store, and compilation.

Generates the ``@triton.jit`` function header, pointer parameters, stride
parameters, ``tl.constexpr`` block sizes, ``tl.program_id`` block offsets,
initial pointer arithmetic, the blocked GEMM loop over the K dimension,
the fused epilogue (elementwise post-ops), and the final ``tl.store`` that
writes the accumulated result from SRAM back to HBM exactly once.

The ``compile_and_bind`` method appends the store section and compiles the
complete kernel string via ``exec()`` into a live ``@triton.jit`` callable.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from fuseml._logging import logger
from fuseml.passes.graph_cut import TRANSPARENT_OPS


# ---------------------------------------------------------------------------
# Dtype mapping — PyTorch → Triton type strings for accumulator casts
# ---------------------------------------------------------------------------

_TRITON_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float32: "tl.float32",
    torch.float16: "tl.float16",
    torch.bfloat16: "tl.bfloat16",
}

# Maximum finite value representable in IEEE 754 FP16.  Values beyond this
# overflow to ±inf during downcast from FP32.  Used for saturating clamp
# before the final tl.store to prevent silent precision catastrophes.
_FP16_MAX: float = 65504.0

# Half-precision dtypes that require explicit FP32 upcast before tl.dot()
# accumulation to prevent numerical underflow/overflow in the K-loop.
_HALF_PRECISION_DTYPES: frozenset[torch.dtype] = frozenset({
    torch.float16, torch.bfloat16,
})

# In-place op targets that reuse the accumulator registers directly without
# allocating temporary SRAM buffers.  The codegen emits a register-reuse
# annotation for these ops so the generated kernel is self-documenting.
_IN_PLACE_EPILOGUE_OPS: frozenset = frozenset({
    torch.ops.aten.relu_.default,
    torch.ops.aten.sigmoid_.default,
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.mul_.Tensor,
})


# ---------------------------------------------------------------------------
# Autotuning — candidate tile sizes, warp counts, and SRAM budget
# ---------------------------------------------------------------------------

# Bytes-per-element for SRAM footprint estimation.
_DTYPE_BYTES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}

# Default SRAM budget fallback — used when no CUDA device is available.
# Ada Lovelace (sm_89) supports up to 100 KB configurable shared memory
# per SM.  Configs whose shared-memory footprint exceeds the budget are
# pruned before writing the @triton.autotune decorator, preventing
# "Out of Shared Memory" PTX failures.
_DEFAULT_SRAM_BUDGET_BYTES: int = 100 * 1024


def _get_sram_budget() -> int:
    """Query the GPU's actual shared memory capacity, with a static fallback.

    Uses ``torch.cuda.get_device_properties`` to read the hardware's
    maximum configurable shared memory per block (``max_shared_memory_per_block``).
    Falls back to 100 KB when no CUDA device is available.
    """
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            budget = getattr(props, "max_shared_memory_per_block", 0)
            if budget > 0:
                return budget
    except Exception:
        pass
    return _DEFAULT_SRAM_BUDGET_BYTES


_AUTOTUNE_SRAM_BUDGET_BYTES: int = _DEFAULT_SRAM_BUDGET_BYTES

# Candidate tile dimensions for autotune config generation.
# Covers small tiles (32) for occupancy on low-SM-count laptop GPUs
# through large tiles (256) for data reuse on high-end desktop GPUs.
# SRAM pruning + heuristic filtering keeps the effective config count
# manageable (target < 100 surviving configs).
_AUTOTUNE_BLOCK_M_CHOICES: tuple[int, ...] = (32, 64, 128, 256)
_AUTOTUNE_BLOCK_N_CHOICES: tuple[int, ...] = (32, 64, 128, 256)
_AUTOTUNE_BLOCK_K_CHOICES: tuple[int, ...] = (32, 64, 128)

# Standard warp counts and software-pipelining depths.
# Ada uses cp.async (not TMA) for async global→shared copies.  Higher
# stage counts (up to 5) improve latency hiding by overlapping more
# cp.async copies with Tensor Core compute, but increase register
# pressure.  The autotuner selects the optimal balance at runtime.
_AUTOTUNE_NUM_WARPS_CHOICES: tuple[int, ...] = (4, 8, 16)
_AUTOTUNE_NUM_STAGES_CHOICES: tuple[int, ...] = (2, 3, 4, 5)

# Reduction-specialised warp counts — higher counts saturate the SMs
# during tl.atomic_add / tl.atomic_max cross-thread synchronisation.
_AUTOTUNE_REDUCTION_NUM_WARPS_CHOICES: tuple[int, ...] = (2, 4, 8, 16)

# L2 swizzle group width candidates — controls how many M-blocks are
# grouped in the 1-D → 2-D block-index mapping.  Different values suit
# different matrix aspect ratios.
_AUTOTUNE_GROUP_SIZE_M_CHOICES: tuple[int, ...] = (4, 8, 16)


# ---------------------------------------------------------------------------
# Reduction info — tracks what happened to acc during the epilogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReductionInfo:
    """Metadata describing a reduction emitted during the epilogue.

    Attributes
    ----------
    axis : Triton tile axis that was collapsed (0 = M, 1 = N).
    op   : Reduction kind — ``"sum"``, ``"max"``, or ``"mean"``.
    keepdim : Whether the reduced dimension is kept as size-1.
    """

    axis: int
    op: str
    keepdim: bool = False


# ---------------------------------------------------------------------------
# Tensor descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorDescriptor:
    """Lightweight metadata describing a tensor for kernel codegen.

    Attributes
    ----------
    name  : short identifier used in generated variable names (e.g. ``"a"``,
            ``"bias"``).
    shape : concrete shape tuple, e.g. ``(128, 64)``.
    stride: concrete stride tuple matching *shape*, e.g. ``(64, 1)``.
    dtype : PyTorch scalar type, e.g. ``torch.float32``.
    aligned : whether ``data_ptr() % 16 == 0``.  Aligned pointers enable
              maximally coalesced ``tl.load`` operations via wider vector
              transactions.  Mirrors the ``aligned`` field tracked by
              :class:`~fuseml.codegen.kernel_cache.TensorFingerprint`.
    """

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    aligned: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def next_power_of_2(val: int) -> int:
    """Round *val* up to the next power of two (unchanged if already a power of two).

    Triton requires all ``BLOCK_SIZE_*`` constexpr parameters to be strict
    powers of two, otherwise the PTX backend will reject the kernel.  Wrap
    every block-size heuristic through this function before passing values to
    ``@triton.jit``.

    >>> next_power_of_2(1)
    1
    >>> next_power_of_2(31)
    32
    >>> next_power_of_2(64)
    64
    >>> next_power_of_2(100)
    128
    """
    if val <= 0:
        raise ValueError(f"Block size must be positive, got {val}")
    # Bit trick: subtract 1, propagate highest bit, add 1.
    val -= 1
    val |= val >> 1
    val |= val >> 2
    val |= val >> 4
    val |= val >> 8
    val |= val >> 16
    return val + 1


def _stride_param(tensor_name: str, dim_label: str) -> str:
    """Build a stride parameter name.

    Single-char names keep the Triton-tutorial style (``stride_am``),
    multi-char names insert an underscore for clarity (``stride_bias_n``).
    """
    sep = "" if len(tensor_name) == 1 else "_"
    return f"stride_{tensor_name}{sep}{dim_label}"


def _block_const(dim: str) -> str:
    """Return the ``BLOCK_SIZE_*`` constant for *dim* (e.g. ``"m"`` -> ``"BLOCK_SIZE_M"``)."""
    return f"BLOCK_SIZE_{dim.upper()}"


def _deduplicate(tensors: list[TensorDescriptor]) -> list[TensorDescriptor]:
    """Remove duplicate descriptors (by name), preserving order."""
    seen: set[str] = set()
    out: list[TensorDescriptor] = []
    for t in tensors:
        if t.name not in seen:
            seen.add(t.name)
            out.append(t)
    return out


def _determine_block_order(tensor: TensorDescriptor) -> tuple[int, int]:
    """Determine the optimal memory access order for ``tl.make_block_ptr``.

    Returns ``(1, 0)`` for row-major layouts (last dimension contiguous)
    and ``(0, 1)`` for column-major layouts (first dimension contiguous).
    The ``order`` parameter tells the Triton compiler which dimension
    varies fastest in memory, enabling maximally coalesced loads.
    """
    if len(tensor.stride) >= 2 and tensor.stride[1] <= tensor.stride[0]:
        return (1, 0)  # row-major: last dim is fastest
    return (0, 1)  # column-major: first dim is fastest


def _classify(
    tensors: list[TensorDescriptor],
) -> tuple[list[TensorDescriptor], list[TensorDescriptor]]:
    """Split tensors into 2-D matrices and 1-D vectors."""
    matrices = [t for t in tensors if len(t.shape) == 2]
    vectors = [t for t in tensors if len(t.shape) == 1]
    return matrices, vectors


def _vector_dim(tensor: TensorDescriptor, M: int, N: int) -> str:
    """Decide whether a 1-D tensor broadcasts along ``"n"`` or ``"m"``."""
    if tensor.shape[0] == N:
        return "n"
    if tensor.shape[0] == M:
        return "m"
    # Ambiguous — default to n (standard bias convention for addmm).
    return "n"


def _surviving_dim(output: TensorDescriptor, M: int, N: int) -> str:
    """For a 1-D reduced output, determine which dimension label survived.

    Compares the output's single dimension against the matmul M and N to
    decide whether the result aligns with ``"m"`` (rows) or ``"n"`` (columns).
    Defaults to ``"m"`` (the common case for ``dim=-1`` reductions on an
    (M, N) matmul result).
    """
    if output.shape[0] == N and N != M:
        return "n"
    return "m"


def _identify_matmul_operands(
    matrices: list[TensorDescriptor],
) -> tuple[TensorDescriptor, TensorDescriptor]:
    """Find the (left, right) matmul pair from a list of 2-D tensors.

    Tries both orderings of the first two matrices to find the one where
    the inner (contracting) dimensions align: ``left.shape[1] == right.shape[0]``.

    Returns
    -------
    (left, right) : tuple[TensorDescriptor, TensorDescriptor]
        ``left`` is the (M, K) operand, ``right`` is the (K, N) operand.

    Raises
    ------
    ValueError
        If neither ordering produces matching inner dimensions.
    """
    m0, m1 = matrices[0], matrices[1]
    if m0.shape[1] == m1.shape[0]:
        return m0, m1
    if m1.shape[1] == m0.shape[0]:
        return m1, m0
    raise ValueError(
        f"Contracting dimension mismatch: "
        f"{m0.name}{m0.shape} vs {m1.name}{m1.shape}"
    )


# ---------------------------------------------------------------------------
# TritonKernelGenerator
# ---------------------------------------------------------------------------

class TritonKernelGenerator:
    """Generates Triton kernel source code from tensor descriptors.

    Current scope: matmul-based fusions (``addmm`` / ``mm`` as the base op)
    with optional bias and elementwise post-ops.

    Compiled kernels are cached by a SHA-256 hash of their full source so
    that repeated ``compile_and_bind`` calls with the same kernel string
    skip ``exec()`` entirely and return the previously compiled callable.
    """

    def __init__(self) -> None:
        self._kernel_cache: dict[str, object] = {}
        self._last_reduction: ReductionInfo | None = None

    # ------------------------------------------------------------------
    # Autotuning — SRAM-aware config generation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sram_footprint(
        block_m: int,
        block_n: int,
        block_k: int,
        dtype_bytes: int,
        num_stages: int,
    ) -> int:
        """Estimate shared-memory bytes for one GEMM config.

        Each software-pipeline stage holds one A-tile and one B-tile in
        shared memory simultaneously::

            per_stage = (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × dtype_bytes
            total     = num_stages × per_stage

        Returns the total shared-memory footprint in bytes.
        """
        a_tile = block_m * block_k * dtype_bytes
        b_tile = block_k * block_n * dtype_bytes
        return num_stages * (a_tile + b_tile)

    @staticmethod
    def _build_autotune_configs(
        dtype: torch.dtype,
        has_reduction: bool = False,
    ) -> list[dict]:
        """Generate the full Cartesian product of candidate autotune configs.

        Each config is a dict with keys ``BLOCK_SIZE_M``, ``BLOCK_SIZE_N``,
        ``BLOCK_SIZE_K``, ``GROUP_SIZE_M``, ``num_warps``, ``num_stages``.

        When *has_reduction* is ``True``, the warp-count candidates are
        expanded to include higher values (up to 16) that better saturate
        SMs during ``tl.atomic_add`` / ``tl.atomic_max`` synchronisation.
        """
        warp_choices = (
            _AUTOTUNE_REDUCTION_NUM_WARPS_CHOICES
            if has_reduction
            else _AUTOTUNE_NUM_WARPS_CHOICES
        )
        configs: list[dict] = []
        for bm in _AUTOTUNE_BLOCK_M_CHOICES:
            for bn in _AUTOTUNE_BLOCK_N_CHOICES:
                for bk in _AUTOTUNE_BLOCK_K_CHOICES:
                    for gsm in _AUTOTUNE_GROUP_SIZE_M_CHOICES:
                        for nw in warp_choices:
                            for ns in _AUTOTUNE_NUM_STAGES_CHOICES:
                                configs.append({
                                    "BLOCK_SIZE_M": bm,
                                    "BLOCK_SIZE_N": bn,
                                    "BLOCK_SIZE_K": bk,
                                    "GROUP_SIZE_M": gsm,
                                    "num_warps": nw,
                                    "num_stages": ns,
                                })
        return configs

    @staticmethod
    def _prune_configs_by_sram(
        configs: list[dict],
        dtype: torch.dtype,
        sram_budget: int = _AUTOTUNE_SRAM_BUDGET_BYTES,
    ) -> list[dict]:
        """Remove configs whose shared-memory footprint exceeds the SRAM budget.

        Uses :meth:`_compute_sram_footprint` to estimate the shared-memory
        requirement for each candidate config.  Configs that exceed
        *sram_budget* are silently dropped — the surviving list is
        guaranteed to fit within the SM's shared-memory capacity.
        """
        dtype_bytes = _DTYPE_BYTES.get(dtype, 4)
        survivors: list[dict] = []
        for cfg in configs:
            footprint = TritonKernelGenerator._compute_sram_footprint(
                cfg["BLOCK_SIZE_M"],
                cfg["BLOCK_SIZE_N"],
                cfg["BLOCK_SIZE_K"],
                dtype_bytes,
                cfg["num_stages"],
            )
            if footprint <= sram_budget:
                survivors.append(cfg)
        logger.debug(
            "SRAM pruning — %d / %d configs survive (budget=%d bytes, dtype=%s)",
            len(survivors), len(configs), sram_budget, dtype,
        )
        return survivors

    @staticmethod
    def _prune_configs_by_heuristic(configs: list[dict]) -> list[dict]:
        """Remove configs that are obviously sub-optimal via cheap heuristics.

        Applied after SRAM pruning to keep the surviving config count
        under ~100, limiting first-run Triton JIT compilation time.

        Rules:
        * 16 warps only make sense with large tiles (area >= 16384).
        * GROUP_SIZE_M=16 is wasteful with very small spatial tiles.
        * BLOCK_K=128 with num_stages > 2 almost always exceeds SRAM
          for useful spatial tile sizes — any that survived SRAM pruning
          are likely tiny spatial tiles where BLOCK_K=128 is overkill.
        """
        survivors: list[dict] = []
        for cfg in configs:
            tile_area = cfg["BLOCK_SIZE_M"] * cfg["BLOCK_SIZE_N"]
            # 16 warps are wasted on small tiles.
            if cfg["num_warps"] == 16 and tile_area < 16384:
                continue
            # BLOCK_K=128 + deep pipelining is almost never SRAM-safe
            # for useful spatial tiles; skip to avoid compiling duds.
            if cfg["BLOCK_SIZE_K"] == 128 and cfg["num_stages"] > 2:
                continue
            # GROUP_SIZE_M=16 is excessive for tiny output tiles.
            if cfg["GROUP_SIZE_M"] == 16 and tile_area < 4096:
                continue
            survivors.append(cfg)
        logger.debug(
            "Heuristic pruning — %d / %d configs survive",
            len(survivors), len(configs),
        )
        return survivors

    @staticmethod
    def _section_autotune_decorator(
        configs: list[dict],
    ) -> str:
        """Format the ``@triton.autotune(...)`` decorator from pruned configs.

        The decorator is placed *above* ``@triton.jit`` so that Triton's
        autotune framework selects the optimal tile size for each unique
        (M, N, K) combination at runtime.  ``key=['M', 'N', 'K']``
        ensures re-tuning when the dynamic batch dimensions change.
        """
        lines: list[str] = ["@triton.autotune("]
        lines.append("    configs=[")
        for cfg in configs:
            meta = {
                "BLOCK_SIZE_M": cfg["BLOCK_SIZE_M"],
                "BLOCK_SIZE_N": cfg["BLOCK_SIZE_N"],
                "BLOCK_SIZE_K": cfg["BLOCK_SIZE_K"],
                "GROUP_SIZE_M": cfg["GROUP_SIZE_M"],
            }
            nw = cfg["num_warps"]
            ns = cfg["num_stages"]
            lines.append(
                f"        triton.Config({meta}, num_warps={nw}, num_stages={ns}),"
            )
        lines.append("    ],")
        lines.append("    key=['M', 'N', 'K'],")
        lines.append(")")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_autotune_configs(
        self,
        input_tensors: list[TensorDescriptor],
        output_tensor: TensorDescriptor,
        has_reduction: bool = False,
    ) -> str:
        """Return a ``@triton.autotune(...)`` decorator string with SRAM-pruned configs.

        Generates a Cartesian product of candidate tile sizes, warp counts,
        and pipeline stages, then prunes any config whose shared-memory
        footprint exceeds FuseML's rigid 48 KB SRAM budget.  The surviving
        configs are formatted as ``triton.Config(...)`` objects inside the
        decorator.

        When *has_reduction* is ``True``, higher ``num_warps`` values (up
        to 16) are included to saturate SMs during the
        ``tl.atomic_add`` / ``tl.atomic_max`` synchronisation phases that
        follow reduction epilogues.

        ``M``, ``N``, ``K`` are listed as ``key`` meta-parameters so Triton
        re-tunes when dynamic batch sizes change.

        Parameters
        ----------
        input_tensors :
            Same descriptor list as :meth:`generate_signature_and_pointers`.
            Used to determine the operand dtype for SRAM footprint estimation.
        output_tensor :
            Descriptor for the output tensor.
        has_reduction :
            Whether the fusion group ends in a reduction op
            (``sum.dim_IntList``, ``amax``, ``mean.dim``).

        Returns
        -------
        str
            The ``@triton.autotune(...)`` decorator, ready to be prepended
            before ``@triton.jit`` in the generated kernel source.
        """
        unique = _deduplicate(input_tensors)
        matrices, _ = _classify(unique)

        # Use the left operand's dtype for SRAM estimation — the K-loop
        # loads A and B tiles into shared memory at their native precision.
        if len(matrices) >= 2:
            left, _ = _identify_matmul_operands(matrices)
            operand_dtype = left.dtype
        else:
            operand_dtype = output_tensor.dtype

        all_configs = self._build_autotune_configs(operand_dtype, has_reduction)
        pruned = self._prune_configs_by_sram(
            all_configs, operand_dtype, sram_budget=_get_sram_budget(),
        )
        pruned = self._prune_configs_by_heuristic(pruned)

        logger.info(
            "Autotune config generation — %d configs after SRAM + heuristic "
            "pruning (dtype=%s, reduction=%s).  First-run JIT compilation "
            "may take 1-3 minutes on laptop GPUs.",
            len(pruned), operand_dtype, has_reduction,
        )

        return self._section_autotune_decorator(pruned)

    def generate_signature_and_pointers(
        self,
        input_tensors: list[TensorDescriptor],
        output_tensor: TensorDescriptor,
        intermediate_tensors: list[TensorDescriptor] | None = None,
        autotune: bool = False,
        has_reduction: bool = False,
    ) -> str:
        """Return a Triton kernel source string (no compute loop).

        The returned string contains, in order:

        1. When *autotune* is ``True``, a ``@triton.autotune(...)``
           decorator with SRAM-pruned configs and ``key=['M', 'N', 'K']``.
        2. ``@triton.jit`` decorator.
        3. ``def fused_kernel(...)`` with dynamically generated pointer,
           dimension, stride, and ``tl.constexpr`` block-size parameters.
           Pointers and strides follow the caller-supplied input order.
           When *intermediate_tensors* are provided, their output pointers
           and M×N strides are appended after the primary output parameters.
        4. ``tl.program_id`` block-offset computation for the M and N
           dimensions.
        5. Initial pointer arithmetic for every input tensor, the output
           tensor, and each intermediate output tensor, in order.

        Parameters
        ----------
        input_tensors :
            Descriptors for each unique input.  Must include at least two
            2-D tensors (the matmul operands A and B); additional 2-D
            tensors are treated as auxiliary (M x N) operands (e.g.
            residual connections).  1-D tensors are treated as bias
            vectors broadcast along the matching dimension.
        output_tensor :
            Descriptor for the 2-D output tensor.
        intermediate_tensors :
            Descriptors for internal nodes whose results escape the fused
            block (users outside the group).  Each entry receives its own
            output pointer + M×N strides in the kernel signature and its
            own ``tl.store`` in the epilogue.  Pass ``None`` or an empty
            list when no intermediate stores are required.
        autotune :
            When ``True``, prepend a ``@triton.autotune(...)`` decorator
            with SRAM-pruned configs before ``@triton.jit``.
        has_reduction :
            When ``True`` (and *autotune* is also ``True``), include
            higher ``num_warps`` candidates (up to 16) to saturate SMs
            during reduction synchronisation phases.

        Raises
        ------
        ValueError
            If fewer than two 2-D inputs are provided or the inner
            (contracting) dimensions of the two matrices do not match.
        """
        intermediate_tensors = intermediate_tensors or []
        unique = _deduplicate(input_tensors)
        matrices, _ = _classify(unique)

        if len(matrices) < 2:
            raise ValueError(
                f"Need >= 2 two-dimensional inputs for matmul, got {len(matrices)}"
            )

        # Identify the matmul pair — tries both orderings.
        left, right = _identify_matmul_operands(matrices)
        M, K = left.shape
        N = right.shape[1]

        logger.debug(
            "Generating kernel signature — M=%d, N=%d, K=%d, inputs=%s, intermediates=%s",
            M, N, K, [t.name for t in unique], [t.name for t in intermediate_tensors],
        )

        # Map every input tensor to its dimension labels.
        dim_labels: dict[str, tuple[str, ...]] = {
            left.name: ("m", "k"),
            right.name: ("k", "n"),
        }
        for t in unique:
            if t.name in dim_labels:
                continue
            if len(t.shape) == 2:
                # Auxiliary 2-D tensor (e.g. residual) — same tile as output.
                dim_labels[t.name] = ("m", "n")
            elif len(t.shape) == 1:
                dim_labels[t.name] = (_vector_dim(t, M, N),)

        sections: list[str] = []
        if autotune:
            sections.append(
                self.generate_autotune_configs(input_tensors, output_tensor, has_reduction)
            )
        sections.extend([
            self._section_decorator(),
            self._section_function_def(unique, output_tensor, dim_labels, intermediate_tensors),
            self._section_docstring(),
            self._section_program_ids(),
            self._section_block_offsets(),
        ])
        # Pointer arithmetic — iterate inputs in caller-supplied order.
        # Matmul operands (left, right) use block pointers for hardware-
        # accelerated cp.async loads on Ada; other 2-D tensors (residuals)
        # and 1-D tensors (biases) use scalar offset arithmetic.
        matmul_names = {left.name, right.name}
        for t in unique:
            labels = dim_labels[t.name]
            if t.name in matmul_names and len(labels) == 2:
                sections.append(self._section_matmul_block_ptr(t, labels))
            elif len(labels) == 2:
                sections.append(self._section_matrix_ptrs(t, labels))
            else:
                sections.append(self._section_vector_ptrs(t, labels[0]))
        # Output pointer arithmetic — 1-D for reduced output, 2-D otherwise.
        if len(output_tensor.shape) == 1:
            surviving = _surviving_dim(output_tensor, M, N)
            sections.append(self._section_output_ptrs_reduced(output_tensor, surviving))
        else:
            sections.append(self._section_output_ptrs(output_tensor))
        # Pointer arithmetic for each intermediate output (all M x N tiles).
        for t in intermediate_tensors:
            sections.append(self._section_output_ptrs(t))
        sections.append(self._section_footer())

        return "\n".join(sections)

    def generate_k_loop(
        self,
        input_tensors: list[TensorDescriptor],
        output_tensor: TensorDescriptor,
    ) -> str:
        """Return a Triton source string for the blocked GEMM loop over K.

        The returned code is meant to follow the pointer arithmetic
        generated by :meth:`generate_signature_and_pointers`.  It contains:

        1. Accumulator initialisation (``tl.zeros`` in fp32).
        2. A ``for k in range(…)`` loop that, on each iteration:

           - loads tiles of the left and right operands from HBM into SRAM
             via ``tl.load`` on block pointers with ``boundary_check``,
           - accumulates ``tl.dot(left, right, acc=acc)`` with fp32
             accumulation using native bf16/fp16 Tensor Core throughput,
           - advances both block pointers via ``tl.advance``.

        3. Bias addition (if any 1-D inputs are present).
        4. **Epilogue downcast** — when the output dtype is narrower than
           fp32 (e.g. ``bfloat16``), the accumulator is cast to the output
           dtype *before* the epilogue post-ops.  This means the epilogue
           operates at the target precision, trading marginal numerical
           accuracy for higher throughput on Ada Lovelace Tensor Cores.

        Parameters
        ----------
        input_tensors :
            Same descriptor list as passed to
            :meth:`generate_signature_and_pointers`.
        output_tensor :
            Descriptor for the output tensor.  Its ``dtype`` determines
            the epilogue downcast precision.

        Raises
        ------
        ValueError
            If the matmul operands cannot be identified (same rules as
            :meth:`generate_signature_and_pointers`).
        """
        unique = _deduplicate(input_tensors)
        matrices, _ = _classify(unique)

        if len(matrices) < 2:
            raise ValueError(
                f"Need >= 2 two-dimensional inputs for matmul, got {len(matrices)}"
            )

        left, right = _identify_matmul_operands(matrices)

        logger.debug(
            "Generating K-loop — left=%s, right=%s", left.name, right.name,
        )

        sections = [
            self._section_accumulator(),
            self._section_k_loop(left, right),
        ]

        # After the GEMM loop, load and add any 1-D bias vectors.
        # addmm computes: bias + left @ right — the bias is broadcast
        # along the appropriate axis and added to the accumulator.
        _, vectors = _classify(unique)
        for v in vectors:
            dim = _vector_dim(v, left.shape[0], right.shape[-1])
            # Upcast half-precision bias to FP32 to match the accumulator
            # dtype and prevent silent precision loss during addition.
            bias_cast = ".to(tl.float32)" if v.dtype in _HALF_PRECISION_DTYPES else ""
            if dim == "n":
                sections.append(
                    f"\n    # Bias addition — load {v.name} and broadcast along N axis"
                    f"\n    {v.name} = tl.load({v.name}_ptrs, mask=offs_n < N, other=0.0, eviction_policy='evict_first'){bias_cast}"
                    f"\n    acc = acc + {v.name}[None, :]"
                )
            else:
                sections.append(
                    f"\n    # Bias addition — load {v.name} and broadcast along M axis"
                    f"\n    {v.name} = tl.load({v.name}_ptrs, mask=offs_m < M, other=0.0, eviction_policy='evict_first'){bias_cast}"
                    f"\n    acc = acc + {v.name}[:, None]"
                )

        # ── Epilogue precision note ─────────────────────────────────
        # The accumulator stays in FP32 through the entire epilogue so
        # that fused post-ops (GeLU, ReLU, add, …) execute at full
        # precision.  The narrowing cast to the output dtype happens
        # once in _section_store() right before tl.store, avoiding
        # double-cast and preserving numerical accuracy.

        return "\n".join(sections)

    def generate_epilogue(
        self,
        fusion_group_nodes: list[torch.fx.Node],
        escape_stores: dict[int, TensorDescriptor] | None = None,
        output_descriptor: TensorDescriptor | None = None,
        all_group_node_ids: set[int] | None = None,
        node_args_snapshot: dict[str, tuple] | None = None,
    ) -> str:
        """Return Triton source for fused post-GEMM operations on ``acc``.

        This is the dynamic AST translation phase: each PyTorch FX node in
        *fusion_group_nodes* is mapped to the equivalent Triton register
        operation, keeping all intermediate values in SRAM (no round-trip
        through HBM between fused ops).

        The method iterates through a **topologically sorted** list of
        ``torch.fx.Node`` objects and emits Triton code that modifies the
        ``acc`` register in-place.  Only ``call_function`` nodes produce
        code; other node types (``placeholder``, ``output``, …) are
        skipped.

        After emitting the register operation for each node, if that node
        is present in *escape_stores*, an intermediate ``tl.store`` is
        immediately appended to write the current value of ``acc`` back to
        HBM.  This ensures that PyTorch Autograd can retrieve the
        activation during the backward pass even though the value
        continues to be used (and may be overwritten) by subsequent fused
        ops in the same kernel.

        **Reduction support** — when a reduction operator (``aten.sum``,
        ``aten.amax``, ``aten.mean``) is encountered, the generator emits
        cross-thread synchronization commands:

        1. ``tl.sum(acc, axis=…)`` / ``tl.max(acc, axis=…)`` collapses the
           tile within a single program instance.
        2. ``tl.atomic_add`` / ``tl.atomic_max`` accumulates partial results
           across program instances that share the surviving dimension, so
           the complete reduction is correct even when the reduced dimension
           spans multiple tile blocks.

        The output tensor dimensions are updated accordingly (the reduced
        axis is collapsed), and the subsequent ``tl.store`` is replaced by
        the atomic store to prevent memory segmentation faults.

        Supported targets
        -----------------
        * ``torch.ops.aten.relu.default`` — ``tl.where(acc > 0, acc, 0.0)``
        * ``torch.ops.aten.gelu.default`` — fast tanh approximation
        * ``torch.ops.aten.add.Tensor`` — load a residual tensor tile from
          HBM into SRAM, then ``acc = acc + residual``
        * ``torch.ops.aten.mul.Tensor`` — element-wise multiply
        * ``torch.ops.aten.sum.dim_IntList`` — cross-thread sum reduction
        * ``torch.ops.aten.amax.default`` — cross-thread max reduction
        * ``torch.ops.aten.mean.dim`` — cross-thread mean reduction

        Parameters
        ----------
        fusion_group_nodes :
            Topologically sorted FX nodes representing the fused
            operations to apply after the GEMM K-loop.
        escape_stores :
            Mapping of ``id(node)`` → :class:`TensorDescriptor` for every
            node inside the group whose output is consumed by a user
            **outside** the fused block.  After the register op for each
            such node is emitted, a ``tl.store`` is appended to write
            ``acc`` to the corresponding HBM pointer.  Pass ``None`` or
            an empty dict when no intermediate stores are needed.
        output_descriptor :
            Descriptor for the final output tensor.  Required when the
            epilogue contains a reduction so that the atomic store can
            reference the correct output pointer name and dtype.  When
            ``None``, the output pointer name defaults to ``"out"``.
        all_group_node_ids :
            ``id()`` set of **all** nodes belonging to the fusion group
            (base node + fused nodes).  Used by binary-op classification
            to distinguish the GEMM result (already in ``acc``) from
            truly external tensors.  When ``None``, falls back to
            ``{id(n) for n in fusion_group_nodes}``.

        Returns
        -------
        str
            Triton source lines (indented at the kernel body level).
        """
        escape_stores = escape_stores or {}
        node_ids = all_group_node_ids or {id(n) for n in fusion_group_nodes}
        node_args_snapshot = node_args_snapshot or {}
        self._last_reduction = None  # reset for this epilogue

        out_name = output_descriptor.name if output_descriptor else "out"
        out_dtype = output_descriptor.dtype if output_descriptor else torch.float32
        triton_dtype = _TRITON_DTYPE_MAP.get(out_dtype, "tl.float32")

        lines: list[str] = [
            "",
            "    # -----------------------------------------------------------",
            "    # Epilogue — fused post-GEMM operations (all in SRAM registers)",
            "    # -----------------------------------------------------------",
        ]

        for node in fusion_group_nodes:
            if node.op != "call_function":
                continue

            # After graph surgery + DCE, erased nodes have args set to None.
            # Use the pre-surgery snapshot when available.
            saved_args = node_args_snapshot.get(node.name)

            # In-place mutation annotation — emit register-reuse guarantee.
            # When the FusionGroup contains an in-place op (relu_, add_, etc.)
            # the generator reuses the accumulator registers directly instead
            # of allocating a temporary SRAM buffer.  This is safe because:
            # 1. Triton's acc is a register tile, not a memory-backed tensor.
            # 2. The aliasing safety check in mutation_safety.py guarantees
            #    the in-place variant is only fused when alias-safe.
            if node.target in _IN_PLACE_EPILOGUE_OPS:
                lines.append(
                    "    # In-place mutation — accumulator registers reused directly,\n"
                    "    # no temporary SRAM buffer allocated (zero extra memory traffic)"
                )

            if node.target in (
                torch.ops.aten.relu.default,
                torch.ops.aten.relu_.default,
            ):
                lines.append(self._emit_relu())
            elif node.target == torch.ops.aten.gelu.default:
                lines.append(self._emit_gelu())
            elif node.target in (
                torch.ops.aten.sigmoid.default,
                torch.ops.aten.sigmoid_.default,
            ):
                lines.append(self._emit_sigmoid())
            elif node.target in (
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add_.Tensor,
            ):
                lines.append(self._emit_add(node, node_ids, saved_args))
            elif node.target in (
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.mul_.Tensor,
            ):
                lines.append(self._emit_mul(node, node_ids, saved_args))
            # ----- Reduction operators — cross-thread synchronization -----
            elif node.target == torch.ops.aten.sum.dim_IntList:
                axis, keepdim = self._determine_reduction_axis(node)
                self._last_reduction = ReductionInfo(axis=axis, op="sum", keepdim=keepdim)
                lines.append(self._emit_sum(axis, out_name, triton_dtype))
            elif node.target == torch.ops.aten.amax.default:
                axis, keepdim = self._determine_reduction_axis(node)
                self._last_reduction = ReductionInfo(axis=axis, op="max", keepdim=keepdim)
                lines.append(self._emit_max(axis, out_name, triton_dtype))
            elif node.target == torch.ops.aten.mean.dim:
                axis, keepdim = self._determine_reduction_axis(node)
                self._last_reduction = ReductionInfo(axis=axis, op="mean", keepdim=keepdim)
                lines.append(self._emit_mean(axis, out_name, triton_dtype))
            # ----- Transparent view/metadata ops (no Triton code needed) --
            # Shape-preserving view/reshape/unsqueeze ops absorbed during
            # pattern matching.  acc remains in SRAM registers unchanged;
            # stride transformation is handled at the store boundary via
            # runtime stride parameters passed to tl.store.
            elif node.target in TRANSPARENT_OPS:
                lines.append(
                    f"    # (no-op) Transparent view/metadata: "
                    f"{node.target} — acc unchanged in SRAM"
                )
            else:
                logger.warning(
                    "Unsupported epilogue target: %s — skipped", node.target,
                )

            # If this node's result is consumed outside the fused block,
            # flush acc to HBM immediately so Autograd can access it.
            # (Only for pre-reduction nodes; reductions emit their own
            # atomic store.)
            if id(node) in escape_stores and self._last_reduction is None:
                lines.append(self._section_intermediate_store(escape_stores[id(node)]))

        return "\n".join(lines)

    def compile_and_bind(
        self,
        kernel_string: str,
        output_tensor: TensorDescriptor,
    ) -> object:
        """Append store logic and compile the kernel into a callable.

        This is the final stage of the codegen pipeline:

        1. Generate the ``tl.store`` section that writes the accumulated
           result from SRAM back to HBM exactly once, with a 2-D boundary
           mask to prevent out-of-bounds writes.  If the output dtype
           differs from the fp32 accumulator, an explicit cast is emitted.
        2. Hash the full kernel source (SHA-256) and check the
           instance-level cache.  On a hit the previously compiled
           callable is returned immediately — no ``exec()`` overhead.
        3. On a miss, compile the kernel string via ``exec()`` in an
           isolated namespace pre-populated with ``triton`` and
           ``triton.language as tl``, cache the result, and return the
           ``@triton.jit``-decorated ``fused_kernel`` function so the
           PyTorch frontend can invoke it immediately.

        Parameters
        ----------
        kernel_string :
            Concatenated output of :meth:`generate_signature_and_pointers`,
            :meth:`generate_k_loop`, and optionally
            :meth:`generate_epilogue`.
        output_tensor :
            Descriptor for the output tensor (needed for the store
            pointer name and potential dtype cast).

        Returns
        -------
        callable
            The compiled ``@triton.jit`` kernel function.
        """
        import triton                   # noqa: F811 — runtime import
        import triton.language as tl    # noqa: F811

        # 1. Append the store section — or skip when a reduction already
        #    emitted an atomic store in the epilogue.
        if self._last_reduction is not None:
            # The epilogue's atomic store (tl.atomic_add / tl.atomic_max)
            # replaces the normal tl.store.  No additional store needed.
            full_kernel = kernel_string
            logger.debug(
                "Reduction detected (op=%s, axis=%d) — skipping normal tl.store.",
                self._last_reduction.op,
                self._last_reduction.axis,
            )
        else:
            store_section = self._section_store(output_tensor)
            full_kernel = kernel_string + "\n" + store_section

        # 2. Check the compilation cache (keyed by SHA-256 of full source)
        cache_key = hashlib.sha256(full_kernel.encode()).hexdigest()

        cached = self._kernel_cache.get(cache_key)
        if cached is not None:
            logger.debug("Kernel cache hit (%s…)", cache_key[:12])
            return cached

        logger.debug(
            "Kernel cache miss — compiling (%d chars, %s…)",
            len(full_kernel),
            cache_key[:12],
        )

        # 3. Compile by writing to a temporary .py file and importing.
        #    Triton's @jit decorator requires functions defined in real
        #    Python files (not exec'd code) for source inspection.
        fn = self._compile_from_source(full_kernel, cache_key)
        self._kernel_cache[cache_key] = fn
        return fn

    @staticmethod
    def _compile_from_source(source: str, cache_key: str) -> object:
        """Write kernel source to a temp file and import the function.

        This avoids ``exec()`` which is incompatible with some Triton
        builds (notably ``triton-windows``) that inspect the source file
        of ``@triton.jit``-decorated functions.
        """
        # Prepend the required imports so the file is self-contained.
        full_source = (
            "import triton\n"
            "import triton.language as tl\n"
            "from triton.language.extra.cuda import libdevice\n\n"
            + source
        )
        module_name = f"_fuseml_kernel_{cache_key[:16]}"

        # Write to a temp file that persists for the process lifetime.
        tmp_dir = Path(tempfile.gettempdir()) / "fuseml_kernels"
        tmp_dir.mkdir(exist_ok=True)
        kernel_path = tmp_dir / f"{module_name}.py"
        kernel_path.write_text(full_source, encoding="utf-8")

        # Import the module.
        spec = importlib.util.spec_from_file_location(module_name, kernel_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module.fused_kernel

    # ------------------------------------------------------------------
    # Code-section builders (each returns a ready-to-join string)
    # ------------------------------------------------------------------

    @staticmethod
    def _section_decorator() -> str:
        return "@triton.jit"

    @staticmethod
    def _section_function_def(
        inputs: list[TensorDescriptor],
        output: TensorDescriptor,
        dim_labels: dict[str, tuple[str, ...]],
        intermediates: list[TensorDescriptor] | None = None,
    ) -> str:
        """Build ``def fused_kernel(...):``.

        Parameter groups, in order:
        pointers (inputs, output, intermediates) -> dimensions -> strides
        (each input, then output, then each intermediate) -> constexpr
        block sizes.

        When *intermediates* is non-empty, each escape node contributes an
        additional ``{name}_ptr`` pointer and ``stride_{name}_m /
        stride_{name}_n`` parameters placed after the primary output
        strides and before the ``tl.constexpr`` block sizes.
        """
        intermediates = intermediates or []
        params: list[str] = []

        # -- Pointers: inputs in caller-supplied order, then output, then
        #    intermediate outputs (escape nodes) --
        primary_ptrs = ", ".join(f"{t.name}_ptr" for t in [*inputs, output])
        if intermediates:
            intm_ptrs = ", ".join(f"{t.name}_ptr" for t in intermediates)
            all_ptrs = f"{primary_ptrs}, {intm_ptrs}"
            params.append("    # Pointers to input / output / intermediate tensors")
        else:
            all_ptrs = primary_ptrs
            params.append("    # Pointers to input / output tensors")
        params.append(f"    {all_ptrs},")

        # -- Matrix dimensions --
        params.append("    # Matrix dimensions")
        params.append("    M, N, K,")

        # -- Strides: one block per input, in caller-supplied order --
        for t in inputs:
            labels = dim_labels[t.name]
            dim_desc = " x ".join(l.upper() for l in labels)
            stride_names = ", ".join(_stride_param(t.name, l) for l in labels)
            comment = "Strides" if len(labels) > 1 else "Stride"
            params.append(f"    # {comment} for {t.name} ({dim_desc})")
            params.append(f"    {stride_names},")

        # -- Strides: output --
        if len(output.shape) == 1:
            # Reduced output — single stride for the surviving dimension.
            matrices_tmp, _ = _classify(inputs)
            try:
                left_d, right_d = _identify_matmul_operands(matrices_tmp)
                dim_label = _surviving_dim(output, left_d.shape[0], right_d.shape[1])
            except (ValueError, IndexError):
                dim_label = "m"
            s_o = _stride_param(output.name, dim_label)
            params.append(f"    # Stride for reduced output {output.name} ({dim_label.upper()} dimension)")
            params.append(f"    {s_o},")
        else:
            s_om = _stride_param(output.name, "m")
            s_on = _stride_param(output.name, "n")
            params.append(f"    # Strides for {output.name} (M x N)")
            params.append(f"    {s_om}, {s_on},")

        # -- Strides: intermediate outputs (each M x N), one block each --
        for t in intermediates:
            s_im = _stride_param(t.name, "m")
            s_in = _stride_param(t.name, "n")
            params.append(f"    # Strides for intermediate output {t.name} (M x N)")
            params.append(f"    {s_im}, {s_in},")

        # -- Constexpr block sizes and L2 swizzle group width --
        params.append("    # Block sizes and L2 swizzle group width (compile-time constants)")
        params.append("    BLOCK_SIZE_M: tl.constexpr,")
        params.append("    BLOCK_SIZE_N: tl.constexpr,")
        params.append("    BLOCK_SIZE_K: tl.constexpr,")
        params.append("    GROUP_SIZE_M: tl.constexpr,")

        body = "\n".join(params)
        return f"def fused_kernel(\n{body}\n):"

    @staticmethod
    def _section_docstring() -> str:
        return '    """Auto-generated fused kernel — signature and pointer skeleton."""'

    @staticmethod
    def _section_program_ids() -> str:
        return "\n".join([
            "    # -----------------------------------------------------------",
            "    # L2 cache swizzling — reorder thread blocks so that adjacent",
            "    # programs share L2-resident rows/columns.  A 1-D program id",
            "    # is mapped to 2-D (pid_m, pid_n) in a GROUP_SIZE_M-wide",
            "    # column-major stripe, maximising L2 data reuse between",
            "    # neighbouring blocks that read the same A-tile rows.",
            "    # -----------------------------------------------------------",
            "    pid = tl.program_id(0)",
            "    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)",
            "    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)",
            "    num_pid_in_group = GROUP_SIZE_M * num_pid_n",
            "    group_id = pid // num_pid_in_group",
            "    first_pid_m = group_id * GROUP_SIZE_M",
            "    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)",
            "    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)",
            "    pid_n = (pid % num_pid_in_group) // group_size_m",
        ])

    @staticmethod
    def _section_block_offsets() -> str:
        """Row and column offsets for output stores and epilogue operations.

        The K-dimension offset (``offs_k``) is no longer emitted here because
        the matmul operands use ``tl.make_block_ptr`` which manages the
        reduction-axis indexing internally via ``tl.advance``.
        """
        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            "    # Block offsets — row and column index ranges for output /",
            "    # epilogue operations.  K-axis handled by block pointers.",
            "    # -----------------------------------------------------------",
            "    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)",
            "    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)",
        ])

    @staticmethod
    def _section_matrix_ptrs(
        tensor: TensorDescriptor,
        dim_labels: tuple[str, str],
    ) -> str:
        """Pointer arithmetic for a 2-D matrix tile (scalar offset path).

        Used for auxiliary 2-D tensors (residuals, intermediates) that are
        loaded in the epilogue — NOT for matmul operands, which use
        :meth:`_section_matmul_block_ptr` for hardware-accelerated loads.
        """
        n = tensor.name
        d0, d1 = dim_labels
        s0 = _stride_param(n, d0)
        s1 = _stride_param(n, d1)
        blk0 = _block_const(d0)
        blk1 = _block_const(d1)
        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            f"    # Pointer arithmetic for {n} — loads a ({blk0}, {blk1}) tile",
            f"    # {n}_ptrs[i, j] = {n}_ptr + offs_{d0}[i]*{s0} + offs_{d1}[j]*{s1}",
            "    # -----------------------------------------------------------",
            (
                f"    {n}_ptrs = {n}_ptr + "
                f"(offs_{d0}[:, None] * {s0} + offs_{d1}[None, :] * {s1})"
            ),
        ])

    @staticmethod
    def _section_matmul_block_ptr(
        tensor: TensorDescriptor,
        dim_labels: tuple[str, str],
    ) -> str:
        """Block pointer setup for a matmul operand (A or B).

        Uses ``tl.make_block_ptr`` instead of scalar offset calculations,
        enabling hardware-accelerated ``cp.async`` loads on Ada Lovelace
        (sm_89) and automatic boundary handling via ``boundary_check``.

        The ``order`` parameter is derived from the tensor's stride layout
        to ensure maximally coalesced global memory transactions.  Aligned
        pointers (``data_ptr() % 16 == 0``) benefit from wider vector loads
        that the hardware can issue when the base address is properly aligned.
        """
        n = tensor.name
        d0, d1 = dim_labels
        s0 = _stride_param(n, d0)
        s1 = _stride_param(n, d1)
        blk0 = _block_const(d0)
        blk1 = _block_const(d1)

        # Map dim labels to the appropriate dynamic dimensions and pid offsets.
        dim_map = {"m": "M", "n": "N", "k": "K"}
        shape_d0 = dim_map[d0]
        shape_d1 = dim_map[d1]

        # Determine which dimension is the "outer" (pid-indexed) dimension
        # vs the "inner" (reduction) dimension for the initial offset.
        if d0 in ("m", "n"):
            # First dim is a spatial dim (pid-indexed), second is reduction or spatial
            offset_d0 = f"pid_{d0} * {blk0}"
        else:
            offset_d0 = "0"

        if d1 in ("m", "n"):
            offset_d1 = f"pid_{d1} * {blk1}"
        else:
            offset_d1 = "0"

        order = _determine_block_order(tensor)
        aligned_note = "aligned" if tensor.aligned else "unaligned"

        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            f"    # Block pointer for {n} — ({blk0}, {blk1}) tile via",
            f"    # tl.make_block_ptr (cp.async on sm_89, {aligned_note} base ptr)",
            "    # -----------------------------------------------------------",
            (
                f"    {n}_block_ptr = tl.make_block_ptr("
            ),
            f"        base={n}_ptr,",
            f"        shape=({shape_d0}, {shape_d1}),",
            f"        strides=({s0}, {s1}),",
            f"        offsets=({offset_d0}, {offset_d1}),",
            f"        block_shape=({blk0}, {blk1}),",
            f"        order={order},",
            "    )",
        ])

    @staticmethod
    def _section_vector_ptrs(
        tensor: TensorDescriptor,
        dim: str,
    ) -> str:
        """Pointer arithmetic for a 1-D vector (bias), broadcast along one dim."""
        n = tensor.name
        sv = _stride_param(n, dim)
        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            f"    # Pointer arithmetic for {n} — 1-D, broadcast along {dim} axis",
            "    # -----------------------------------------------------------",
            f"    {n}_ptrs = {n}_ptr + offs_{dim} * {sv}",
        ])

    @staticmethod
    def _section_output_ptrs(tensor: TensorDescriptor) -> str:
        """Pointer arithmetic for the 2-D output tile (M x N)."""
        n = tensor.name
        sm = _stride_param(n, "m")
        sn = _stride_param(n, "n")
        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            f"    # Pointer arithmetic for output {n} (M x N)",
            f"    # {n}_ptrs[i, j] = {n}_ptr + offs_m[i]*{sm} + offs_n[j]*{sn}",
            "    # -----------------------------------------------------------",
            (
                f"    {n}_ptrs = {n}_ptr + "
                f"(offs_m[:, None] * {sm} + offs_n[None, :] * {sn})"
            ),
        ])

    @staticmethod
    def _section_output_ptrs_reduced(
        tensor: TensorDescriptor,
        surviving_dim: str,
    ) -> str:
        """Pointer arithmetic for a 1-D reduced output.

        After a reduction collapses one tile dimension, the output is a
        vector along the surviving dimension.  Only ``offs_{surviving_dim}``
        and the corresponding single stride are used.
        """
        n = tensor.name
        s = _stride_param(n, surviving_dim)
        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            f"    # Pointer arithmetic for reduced output {n} ({surviving_dim.upper()} — after reduction)",
            f"    # {n}_ptrs[i] = {n}_ptr + offs_{surviving_dim}[i]*{s}",
            "    # -----------------------------------------------------------",
            f"    {n}_ptrs = {n}_ptr + offs_{surviving_dim} * {s}",
        ])

    @staticmethod
    def _section_accumulator() -> str:
        """Accumulator initialisation — fp32 to preserve numerical precision.

        The dtype is intentionally hard-coded to tl.float32 regardless of the
        input or output dtype.  Accumulating in a lower-precision type risks
        catastrophic cancellation over long K dimensions.  The output cast is
        applied once, unconditionally, in _section_store before tl.store.
        """
        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            "    # Accumulator — strictly tl.float32 to prevent precision loss",
            "    # across the K-loop.  Cast to output dtype happens in the store",
            "    # section below; do NOT change this dtype to a narrower type.",
            "    # -----------------------------------------------------------",
            "    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)",
        ])

    @staticmethod
    def _section_k_loop(
        left: TensorDescriptor,
        right: TensorDescriptor,
    ) -> str:
        """Blocked GEMM loop using block pointers — Ada-optimised (sm_89).

        Uses ``tl.make_block_ptr`` / ``tl.advance`` instead of scalar offset
        arithmetic.  This enables ``cp.async``-based software pipelining on
        Ada Lovelace, where the hardware asynchronously prefetches the next
        K-tile from L2 into shared memory while the current tile's
        ``tl.dot`` executes on the Tensor Cores.

        **Tensor Core utilisation** — input tiles are loaded in their native
        dtype (e.g. ``bfloat16``) and passed directly to ``tl.dot``.  The
        accumulator is ``tl.float32`` (set via ``acc=acc``), so the hardware
        performs ``bf16 × bf16 → fp32`` fused multiply-accumulate in the
        Tensor Core pipeline.  Pre-casting inputs to ``tl.float32`` would
        bypass the Tensor Cores and fall back to the slower FP32 CUDA cores.

        **Boundary handling** — ``boundary_check=(0, 1)`` replaces explicit
        compound masks (``offs_m < M & offs_k < K``).  The Triton compiler
        inserts the minimal guard code at the PTX level, which is both
        shorter and avoids materialising intermediate mask tensors in
        registers.
        """
        ln = left.name
        rn = right.name

        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            "    # Blocked GEMM loop over the K dimension (block-pointer path)",
            f"    # Each iteration loads one (BLOCK_SIZE_M, BLOCK_SIZE_K) tile of {ln}",
            f"    # and one (BLOCK_SIZE_K, BLOCK_SIZE_N) tile of {rn} via block ptrs,",
            "    # accumulates the partial dot-product in fp32, then advances.",
            "    # Total HBM reads: 2 * ceil(K / BLOCK_SIZE_K) tile loads per program.",
            "    #",
            "    # boundary_check=(0, 1) — the Triton compiler inserts minimal PTX",
            "    # guard code for both tile dimensions, replacing the explicit",
            "    # compound masks used in the scalar-offset path.",
            "    #",
            "    # eviction_policy='evict_last' — K-loop tiles are streamed once;",
            "    # deprioritise in L2 to preserve SRAM residency for epilogue",
            "    # buffers and L2-swizzled A-tile rows.",
            "    # -----------------------------------------------------------",
            "    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):",
            "",
            f"        # Load (BLOCK_SIZE_M x BLOCK_SIZE_K) tile of {ln} from HBM → SRAM",
            (
                f"        {ln} = tl.load({ln}_block_ptr, "
                f"boundary_check=(0, 1), "
                f"eviction_policy='evict_last')"
            ),
            f"        # Load (BLOCK_SIZE_K x BLOCK_SIZE_N) tile of {rn} from HBM → SRAM",
            (
                f"        {rn} = tl.load({rn}_block_ptr, "
                f"boundary_check=(0, 1), "
                f"eviction_policy='evict_last')"
            ),
            "",
            "        # Tile-level matrix multiply — bf16/fp16 Tensor Core path with",
            "        # fp32 accumulation; inputs stay in native dtype for peak throughput",
            f"        acc = tl.dot({ln}, {rn}, acc=acc)",
            "",
            "        # Advance block pointers to the next K-tile — tl.advance handles",
            "        # stride arithmetic internally, supporting non-contiguous layouts",
            f"        {ln}_block_ptr = tl.advance({ln}_block_ptr, (0, BLOCK_SIZE_K))",
            f"        {rn}_block_ptr = tl.advance({rn}_block_ptr, (BLOCK_SIZE_K, 0))",
        ])

    @staticmethod
    def _section_store(output: TensorDescriptor) -> str:
        """Store the accumulated result from SRAM back to HBM exactly once.

        Emits a ``tl.store`` guarded by a 2-D boundary mask so that
        out-of-bounds threads do not corrupt memory.  When the output
        dtype differs from the fp32 accumulator, an explicit cast is
        emitted first to match the target precision.

        **Safe downcasting** — for FP16 targets the accumulator is clamped
        to the finite representable range (±65504) before the narrowing
        cast.  Without this saturation, large FP32 values silently overflow
        to ±inf during ``.to(tl.float16)``, corrupting downstream
        computation.  BF16 shares FP32's exponent range so no saturation
        is needed; FP32 → FP32 is an identity cast (zero runtime cost).
        """
        n = output.name
        triton_dtype = _TRITON_DTYPE_MAP.get(output.dtype, "tl.float32")

        lines = [
            "",
            "    # -----------------------------------------------------------",
            "    # Store — write the (BLOCK_SIZE_M, BLOCK_SIZE_N) result tile",
            "    # from SRAM back to HBM exactly once per program instance",
            "    # -----------------------------------------------------------",
        ]

        # FP16 saturating clamp — prevent overflow to ±inf during downcast.
        # FP16 max finite = 65504; anything larger becomes inf after cast.
        # BF16 shares FP32's 8-bit exponent so its range is identical to
        # FP32 (~3.4e38) — no saturation needed.
        if output.dtype == torch.float16:
            lines.extend([
                f"    # FP16 safe downcast — saturate to ±{_FP16_MAX} before",
                f"    # narrowing cast to prevent overflow to ±inf",
                f"    acc = tl.where(acc > {_FP16_MAX}, {_FP16_MAX}, acc)",
                f"    acc = tl.where(acc < -{_FP16_MAX}, -{_FP16_MAX}, acc)",
            ])

        # Unconditional cast: the accumulator is always tl.float32.
        # For fp32 outputs this is an identity cast (zero runtime cost);
        # for fp16/bf16 outputs it narrows the result to the target dtype.
        # Keeping the cast unconditional makes the dtype contract explicit
        # and ensures correctness survives future dtype changes.
        lines.append(f"    acc = acc.to({triton_dtype})")

        lines.append(
            f"    tl.store({n}_ptrs, acc, "
            f"mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Epilogue emitters (each returns a ready-to-join code snippet)
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_relu() -> str:
        """ReLU: zero out negative values in the accumulator."""
        return "\n".join([
            "    # ReLU activation — zero out negatives (register-only, no HBM traffic)",
            "    acc = tl.where(acc > 0, acc, 0.0)",
        ])

    @staticmethod
    def _emit_gelu() -> str:
        """GeLU activation via ``libdevice.tanh`` (hardware intrinsic).

        Matches PyTorch's ``gelu(approximate='tanh')`` formula::

            x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

        Uses ``libdevice.tanh`` which maps to a single-instruction
        CUDA libdevice call — faster and more accurate than a polynomial
        approximation.  The accumulator stays in FP32 through the entire
        computation for maximum precision.

        All arithmetic stays in SRAM registers — no HBM round-trip.
        """
        return "\n".join([
            "    # GeLU activation — libdevice.tanh intrinsic (all in SRAM registers)",
            "    # Formula: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
            "    _gelu_inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)",
            "    acc = 0.5 * acc * (1.0 + libdevice.tanh(_gelu_inner))",
        ])

    @staticmethod
    def _emit_sigmoid() -> str:
        """Sigmoid: 1/(1+exp(-x)) in SRAM registers.

        Uses Triton's built-in ``tl.sigmoid`` which computes the logistic
        function element-wise on the accumulator tile.  The explicit cast
        to ``tl.float32`` prevents precision loss when the accumulator is
        in a lower-precision format.
        """
        return "\n".join([
            "    # Sigmoid activation — 1/(1+exp(-x)) (register-only, no HBM traffic)",
            "    acc = tl.sigmoid(acc.to(tl.float32))",
        ])

    @staticmethod
    def _classify_binary_args(
        node: torch.fx.Node,
        node_ids: set[int],
        saved_args: tuple | None = None,
    ) -> tuple[int | float | None, str | None, torch.fx.Node | None]:
        """Classify arguments of a binary FX node as scalar or external tensor.

        Returns ``(scalar_value, ext_name, ext_node)`` where exactly one of
        *scalar_value* or *ext_name* is non-``None`` when an external operand
        is found.  If both args are internal to the fusion group, all three
        values are ``None``.

        When *saved_args* is provided (pre-surgery snapshot), it is used
        instead of ``node.args`` — this is necessary because FX nullifies
        ``node.args`` when a node is erased during dead code elimination.
        """
        args = saved_args if saved_args is not None else node.args

        scalar_value: int | float | None = None
        ext_name: str | None = None
        ext_node: torch.fx.Node | None = None

        for arg in args:
            if isinstance(arg, (int, float)):
                scalar_value = arg
                break
            if hasattr(arg, "name") and id(arg) not in node_ids:
                ext_name = arg.name
                ext_node = arg  # type: ignore[assignment]
                break

        return scalar_value, ext_name, ext_node

    @staticmethod
    def _is_1d_tensor(ext_node: torch.fx.Node | None) -> bool:
        """Return ``True`` if *ext_node* represents a 1-D tensor (e.g. bias).

        Probes the FX node's ``meta["val"]`` (a ``FakeTensor`` from
        ``torch.compile``'s abstract interpretation) or ``meta["tensor_meta"]``
        as a fallback.
        """
        if ext_node is None:
            return False
        fake = ext_node.meta.get("val")
        if fake is None:
            fake = ext_node.meta.get("tensor_meta")
        return fake is not None and len(fake.shape) == 1

    @staticmethod
    def _detect_epilogue_broadcast(ext_node: torch.fx.Node | None) -> str | None:
        """Detect stride-0 broadcast dimensions on a 2-D external tensor.

        Inspects the FX node's metadata (``TensorFingerprint``-equivalent
        stride information) to determine if one dimension has stride == 0,
        indicating that PyTorch's ``.expand()`` or broadcasting produced a
        zero-stride axis.  When detected, the epilogue can generate 1-D
        pointer arithmetic and let Triton broadcast to ``[BLOCK_M, BLOCK_N]``
        instead of loading a full 2-D tile of identical rows/columns.

        Returns
        -------
        ``"m"`` if dim 0 is broadcast (stride[0] == 0) — load along N.
        ``"n"`` if dim 1 is broadcast (stride[1] == 0) — load along M.
        ``None`` if no broadcast detected, tensor is not 2-D, or no metadata.
        """
        if ext_node is None:
            return None
        fake = ext_node.meta.get("val")
        if fake is None:
            fake = ext_node.meta.get("tensor_meta")
        if fake is None or not hasattr(fake, "shape") or len(fake.shape) != 2:
            return None
        # Extract stride — FakeTensor uses .stride() callable,
        # TensorMeta uses .stride attribute.
        raw_stride = getattr(fake, "stride", None)
        if raw_stride is None:
            return None
        stride = raw_stride() if callable(raw_stride) else raw_stride
        if len(stride) != 2:
            return None
        if stride[0] == 0:
            return "m"  # broadcast along M — load 1-D along N
        if stride[1] == 0:
            return "n"  # broadcast along N — load 1-D along M
        return None

    @staticmethod
    def _emit_add(
        node: torch.fx.Node,
        node_ids: set[int],
        saved_args: tuple | None = None,
    ) -> str:
        """Residual / bias add: load an external tensor tile from HBM, fuse into acc.

        Identifies the *external* argument (the one not produced by a prior
        node inside the fusion group) as the residual tensor.  Also handles
        scalar arguments (e.g. ``add(acc, 1.0)``) without emitting a load.

        Bias broadcasting
        -----------------
        When the external argument is a 1-D tensor (e.g. a per-column bias
        from ``aten.addmm``), the load must use a 1-D mask over ``offs_n``
        and the loaded tile must be broadcast across the M dimension using
        ``[None, :]`` before being added to ``acc``.  Using the 2-D residual
        load path for a 1-D pointer would produce incorrect pointer arithmetic
        and an out-of-bounds access.

        Shape detection uses the FX node's ``meta["val"]`` (a ``FakeTensor``
        produced by ``torch.compile``'s abstract interpretation pass) or
        ``meta["tensor_meta"]`` as a fallback.  If neither is present the
        method conservatively falls back to the 2-D residual path.
        """
        scalar_value, residual_name, residual_arg_node = (
            TritonKernelGenerator._classify_binary_args(node, node_ids, saved_args)
        )

        if scalar_value is not None:
            return "\n".join([
                "    # Scalar add — register-only (no HBM traffic)",
                f"    acc = acc + {scalar_value}",
            ])

        if residual_name is not None:
            is_1d_bias = TritonKernelGenerator._is_1d_tensor(residual_arg_node)

            if is_1d_bias:
                return "\n".join([
                    f"    # Bias broadcast add — load 1-D bias tile along N with offs_n",
                    f"    # mask, then broadcast across M via [None, :] before fusing",
                    f"    # into acc.  Using a 2-D mask here would be incorrect because",
                    f"    # {residual_name}_ptrs is a 1-D pointer block (no M stride).",
                    f"    {residual_name} = tl.load({residual_name}_ptrs, mask=offs_n < N, other=0.0, eviction_policy='evict_first')",
                    f"    acc = acc + {residual_name}[None, :]",
                ])

            broadcast_dim = TritonKernelGenerator._detect_epilogue_broadcast(
                residual_arg_node,
            )
            if broadcast_dim == "m":
                s_n = _stride_param(residual_name, "n")
                return "\n".join([
                    f"    # Broadcast add — {residual_name} has stride=0 along M",
                    f"    # (broadcast_dims[0]=True); load 1-D along N, broadcast to",
                    f"    # [BLOCK_M, BLOCK_N] via [None, :] (no redundant row loads).",
                    f"    {residual_name} = tl.load({residual_name}_ptr + offs_n * {s_n}, mask=offs_n < N, other=0.0, eviction_policy='evict_first')",
                    f"    acc = acc + {residual_name}[None, :]",
                ])
            if broadcast_dim == "n":
                s_m = _stride_param(residual_name, "m")
                return "\n".join([
                    f"    # Broadcast add — {residual_name} has stride=0 along N",
                    f"    # (broadcast_dims[1]=True); load 1-D along M, broadcast to",
                    f"    # [BLOCK_M, BLOCK_N] via [:, None] (no redundant column loads).",
                    f"    {residual_name} = tl.load({residual_name}_ptr + offs_m * {s_m}, mask=offs_m < M, other=0.0, eviction_policy='evict_first')",
                    f"    acc = acc + {residual_name}[:, None]",
                ])

            return "\n".join([
                f"    # Residual add — load 2-D {residual_name} tile from HBM into SRAM",
                (
                    f"    {residual_name} = tl.load({residual_name}_ptrs, "
                    f"mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0, eviction_policy='evict_first')"
                ),
                f"    acc = acc + {residual_name}",
            ])

        return "\n".join([
            "    # Element-wise add (both operands already in SRAM)",
            "    acc = acc + acc",
        ])

    @staticmethod
    def _emit_mul(
        node: torch.fx.Node,
        node_ids: set[int],
        saved_args: tuple | None = None,
    ) -> str:
        """Element-wise multiply: scalar, external tensor, or internal operands.

        Mirrors :meth:`_emit_add` but emits ``*`` instead of ``+``.  The same
        three cases apply:

        1. **Scalar** (``int`` / ``float``) — pure register multiply, no HBM
           traffic (e.g. ``x * 0.5``).
        2. **External 1-D tensor** — ``tl.load`` with a 1-D ``offs_n`` mask,
           broadcast via ``[None, :]``, then multiply into ``acc``.
        3. **External 2-D tensor** — ``tl.load`` with a full 2-D boundary mask,
           then element-wise multiply.
        4. **Both internal** — ``acc = acc * acc`` (square).
        """
        scalar_value, ext_name, ext_node = (
            TritonKernelGenerator._classify_binary_args(node, node_ids, saved_args)
        )

        if scalar_value is not None:
            return "\n".join([
                "    # Scalar mul — register-only (no HBM traffic)",
                f"    acc = acc * {scalar_value}",
            ])

        if ext_name is not None:
            is_1d = TritonKernelGenerator._is_1d_tensor(ext_node)

            if is_1d:
                return "\n".join([
                    f"    # Broadcast mul — load 1-D tile along N with offs_n mask,",
                    f"    # then broadcast across M via [None, :] before multiplying",
                    f"    # into acc.  {ext_name}_ptrs is a 1-D pointer block (no M stride).",
                    f"    {ext_name} = tl.load({ext_name}_ptrs, mask=offs_n < N, other=0.0, eviction_policy='evict_first')",
                    f"    acc = acc * {ext_name}[None, :]",
                ])

            broadcast_dim = TritonKernelGenerator._detect_epilogue_broadcast(ext_node)
            if broadcast_dim == "m":
                s_n = _stride_param(ext_name, "n")
                return "\n".join([
                    f"    # Broadcast mul — {ext_name} has stride=0 along M",
                    f"    # (broadcast_dims[0]=True); load 1-D along N, broadcast to",
                    f"    # [BLOCK_M, BLOCK_N] via [None, :] (no redundant row loads).",
                    f"    {ext_name} = tl.load({ext_name}_ptr + offs_n * {s_n}, mask=offs_n < N, other=0.0, eviction_policy='evict_first')",
                    f"    acc = acc * {ext_name}[None, :]",
                ])
            if broadcast_dim == "n":
                s_m = _stride_param(ext_name, "m")
                return "\n".join([
                    f"    # Broadcast mul — {ext_name} has stride=0 along N",
                    f"    # (broadcast_dims[1]=True); load 1-D along M, broadcast to",
                    f"    # [BLOCK_M, BLOCK_N] via [:, None] (no redundant column loads).",
                    f"    {ext_name} = tl.load({ext_name}_ptr + offs_m * {s_m}, mask=offs_m < M, other=0.0, eviction_policy='evict_first')",
                    f"    acc = acc * {ext_name}[:, None]",
                ])

            return "\n".join([
                f"    # Tensor mul — load 2-D {ext_name} tile from HBM into SRAM",
                (
                    f"    {ext_name} = tl.load({ext_name}_ptrs, "
                    f"mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0, eviction_policy='evict_first')"
                ),
                f"    acc = acc * {ext_name}",
            ])

        return "\n".join([
            "    # Element-wise mul (both operands already in SRAM)",
            "    acc = acc * acc",
        ])

    # ------------------------------------------------------------------
    # Reduction emitters — cross-thread synchronization
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_reduction_axis(
        node: torch.fx.Node,
    ) -> tuple[int, bool]:
        """Extract the Triton tile axis and ``keepdim`` flag from a reduction FX node.

        The FX node's ``args[1]`` is the list of PyTorch dimensions to
        reduce.  For a 2-D (M, N) matmul output:

        * dim 0 / -2  →  Triton axis 0 (M rows)
        * dim 1 / -1  →  Triton axis 1 (N columns)

        Returns ``(axis, keepdim)``.
        """
        dims = node.args[1] if len(node.args) > 1 else [-1]
        keepdim = node.args[2] if len(node.args) > 2 else False
        # Normalise negative dims for a 2-D tensor (ndim = 2).
        normalised = [d % 2 for d in dims]
        if 1 in normalised:
            return 1, keepdim
        if 0 in normalised:
            return 0, keepdim
        return 1, keepdim  # default: reduce columns

    @staticmethod
    def _emit_sum(axis: int, output_name: str, triton_dtype: str) -> str:
        """Sum reduction with two-stage block-local then atomic accumulation.

        **Stage 1 (register-local)**: ``tl.sum`` collapses the tile-local
        accumulator along *axis*, performing ``BLOCK_{dim}`` independent
        additions entirely in SRAM registers — zero HBM traffic.

        **Stage 2 (atomic to HBM)**: ``tl.atomic_add`` writes the partial
        sums to global memory.  Because the block-local reduction already
        collapsed one full tile dimension, the number of atomic operations
        per program is reduced from ``BLOCK_M × BLOCK_N`` to just
        ``BLOCK_{surviving}`` — minimizing atomic serialization penalties
        on the memory bus.
        """
        dim_name = "N" if axis == 1 else "M"
        surviving = "M" if axis == 1 else "N"
        offs = "offs_m" if axis == 1 else "offs_n"
        bound = "M" if axis == 1 else "N"
        return "\n".join([
            f"    # ── Reduction: sum along {dim_name} (two-stage) ──────────────────",
            f"    # Stage 1: block-local tl.sum collapses axis={axis}, producing one",
            f"    # partial sum per surviving-dimension lane — entirely in registers.",
            f"    # Stage 2: tl.atomic_add coalesces partials across programs that",
            f"    # share the same {bound}-tile.  Block-local reduction cuts atomics",
            f"    # from BLOCK_{dim_name}×BLOCK_{surviving} down to BLOCK_{surviving}.",
            f"    partial_sum = tl.sum(acc, axis={axis}).to({triton_dtype})",
            f"    tl.atomic_add({output_name}_ptrs, partial_sum, mask={offs} < {bound})",
        ])

    @staticmethod
    def _emit_max(axis: int, output_name: str, triton_dtype: str) -> str:
        """Max reduction with two-stage block-local then atomic accumulation.

        Same two-stage strategy as :meth:`_emit_sum`:

        **Stage 1**: ``tl.max`` collapses the tile in registers (zero HBM).
        **Stage 2**: ``tl.atomic_max`` writes block-local maxima to global
        memory, cutting atomic operations from ``BLOCK_M × BLOCK_N`` to
        ``BLOCK_{surviving}``.
        """
        dim_name = "N" if axis == 1 else "M"
        surviving = "M" if axis == 1 else "N"
        offs = "offs_m" if axis == 1 else "offs_n"
        bound = "M" if axis == 1 else "N"
        return "\n".join([
            f"    # ── Reduction: max along {dim_name} (two-stage) ──────────────────",
            f"    # Stage 1: block-local tl.max collapses axis={axis} in registers.",
            f"    # Stage 2: tl.atomic_max coalesces block-local maxima to HBM,",
            f"    # cutting atomics from BLOCK_{dim_name}×BLOCK_{surviving} to BLOCK_{surviving}.",
            f"    partial_max = tl.max(acc, axis={axis}).to({triton_dtype})",
            f"    tl.atomic_max({output_name}_ptrs, partial_max, mask={offs} < {bound})",
        ])

    @staticmethod
    def _emit_mean(axis: int, output_name: str, triton_dtype: str) -> str:
        """Mean reduction — division fused into epilogue for CUDA Graph safety.

        Each program contributes its block-local ``tl.sum`` **multiplied
        by the reciprocal** ``1.0 / dim_size`` via ``tl.atomic_add``.
        Fusing the reciprocal multiply into the epilogue eliminates the
        secondary PyTorch scalar-tensor kernel that would otherwise be
        launched during dispatch, making the entire dispatch sequence
        capturable by ``torch.cuda.CUDAGraph`` with zero host-side fixups.

        **Numerical precision**: partial sums and the reciprocal multiply
        are all in FP32 regardless of output dtype.  The
        :class:`~fuseml.codegen.kernel_launcher.KernelLauncher` only
        needs to cast the final FP32 result to the target dtype — no
        division is performed on the host.
        """
        dim_name = "N" if axis == 1 else "M"
        surviving = "M" if axis == 1 else "N"
        offs = "offs_m" if axis == 1 else "offs_n"
        bound = "M" if axis == 1 else "N"
        return "\n".join([
            f"    # ── Reduction: mean along {dim_name} (two-stage, division fused) ────",
            f"    # Stage 1: block-local tl.sum collapses axis={axis} in registers.",
            f"    # Stage 2: multiply by 1/{dim_name} then tl.atomic_add in FP32.",
            f"    # Division is fused here (not deferred to KernelLauncher) so that",
            f"    # no secondary PyTorch kernel is launched during dispatch — this is",
            f"    # required for CUDA Graph capture/replay safety.",
            f"    partial_sum = tl.sum(acc, axis={axis})  # keep FP32 — acc is always FP32",
            f"    partial_mean = partial_sum * (1.0 / {dim_name})  # fused reciprocal",
            f"    tl.atomic_add({output_name}_ptrs, partial_mean, mask={offs} < {bound})",
        ])

    @staticmethod
    def _section_intermediate_store(tensor: TensorDescriptor) -> str:
        """Emit a ``tl.store`` for an intermediate (escape) activation.

        Called inside :meth:`generate_epilogue` immediately after the
        register op for an escape node.  At that point ``acc`` holds the
        node's result, so we cast and store it to the intermediate HBM
        pointer before ``acc`` is overwritten by the next fused op.

        The store uses the same 2-D boundary mask as the final store so
        that out-of-bounds threads never corrupt memory.
        """
        n = tensor.name
        triton_dtype = _TRITON_DTYPE_MAP.get(tensor.dtype, "tl.float32")
        return "\n".join([
            f"    # Intermediate store — flush {n} from SRAM to HBM before",
            f"    # acc is overwritten; required by PyTorch Autograd backward.",
            f"    tl.store({n}_ptrs, acc.to({triton_dtype}),",
            f"             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))",
        ])

    @staticmethod
    def _section_footer() -> str:
        return "\n    # --- Compute loop and store logic to be generated separately ---"
