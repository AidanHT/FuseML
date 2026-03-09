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
from dataclasses import dataclass

import torch

from fuseml._logging import logger


# ---------------------------------------------------------------------------
# Dtype mapping — PyTorch → Triton type strings for accumulator casts
# ---------------------------------------------------------------------------

_TRITON_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float32: "tl.float32",
    torch.float16: "tl.float16",
    torch.bfloat16: "tl.bfloat16",
}


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
    """

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signature_and_pointers(
        self,
        input_tensors: list[TensorDescriptor],
        output_tensor: TensorDescriptor,
    ) -> str:
        """Return a Triton kernel source string (no compute loop).

        The returned string contains, in order:

        1. ``@triton.jit`` decorator.
        2. ``def fused_kernel(...)`` with dynamically generated pointer,
           dimension, stride, and ``tl.constexpr`` block-size parameters.
           Pointers and strides follow the caller-supplied input order.
        3. ``tl.program_id`` block-offset computation for the M and N
           dimensions.
        4. Initial pointer arithmetic for every input tensor and the
           output tensor, in the caller-supplied order.

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

        Raises
        ------
        ValueError
            If fewer than two 2-D inputs are provided or the inner
            (contracting) dimensions of the two matrices do not match.
        """
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
            "Generating kernel signature — M=%d, N=%d, K=%d, inputs=%s",
            M, N, K, [t.name for t in unique],
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

        sections = [
            self._section_decorator(),
            self._section_function_def(unique, output_tensor, dim_labels),
            self._section_docstring(),
            self._section_program_ids(),
            self._section_block_offsets(),
        ]
        # Pointer arithmetic — iterate inputs in caller-supplied order.
        for t in unique:
            labels = dim_labels[t.name]
            if len(labels) == 2:
                sections.append(self._section_matrix_ptrs(t, labels))
            else:
                sections.append(self._section_vector_ptrs(t, labels[0]))
        sections.append(self._section_output_ptrs(output_tensor))
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

           - computes a K-dimension mask to prevent OOB reads,
           - loads a ``(BLOCK_SIZE_M, BLOCK_SIZE_K)`` tile of the left
             operand and a ``(BLOCK_SIZE_K, BLOCK_SIZE_N)`` tile of the
             right operand from HBM into SRAM,
           - accumulates ``tl.dot(left, right)`` into the accumulator,
           - advances both pointer blocks to the next K tile.

        Parameters
        ----------
        input_tensors :
            Same descriptor list as passed to
            :meth:`generate_signature_and_pointers`.
        output_tensor :
            Descriptor for the 2-D output tensor (used only for
            validation).

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
        return "\n".join(sections)

    def generate_epilogue(
        self,
        fusion_group_nodes: list[torch.fx.Node],
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

        Supported targets
        -----------------
        * ``torch.ops.aten.relu.default`` — ``tl.where(acc > 0, acc, 0.0)``
        * ``torch.ops.aten.gelu.default`` — fast tanh approximation
        * ``torch.ops.aten.add.Tensor`` — load a residual tensor tile from
          HBM into SRAM, then ``acc = acc + residual``

        Parameters
        ----------
        fusion_group_nodes :
            Topologically sorted FX nodes representing the fused
            elementwise operations to apply after the GEMM K-loop.

        Returns
        -------
        str
            Triton source lines (indented at the kernel body level).
        """
        node_ids = {id(n) for n in fusion_group_nodes}

        lines: list[str] = [
            "",
            "    # -----------------------------------------------------------",
            "    # Epilogue — fused post-GEMM operations (all in SRAM registers)",
            "    # -----------------------------------------------------------",
        ]

        for node in fusion_group_nodes:
            if node.op != "call_function":
                continue

            if node.target == torch.ops.aten.relu.default:
                lines.append(self._emit_relu())
            elif node.target == torch.ops.aten.gelu.default:
                lines.append(self._emit_gelu())
            elif node.target == torch.ops.aten.add.Tensor:
                lines.append(self._emit_add(node, node_ids))
            else:
                logger.warning(
                    "Unsupported epilogue target: %s — skipped", node.target,
                )

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

        # 1. Append the store section to write results back to HBM
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

        # 3. Compile in an isolated namespace with Triton available
        namespace: dict[str, object] = {"triton": triton, "tl": tl}
        exec(full_kernel, namespace)  # noqa: S102

        fn = namespace["fused_kernel"]
        self._kernel_cache[cache_key] = fn
        return fn

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
    ) -> str:
        """Build ``def fused_kernel(...):``.

        Parameter groups, in order:
        pointers -> dimensions -> strides (each input in caller order,
        then output) -> constexpr block sizes.
        """
        params: list[str] = []

        # -- Pointers: inputs in caller-supplied order, then output --
        all_tensors = [*inputs, output]
        ptr_names = ", ".join(f"{t.name}_ptr" for t in all_tensors)
        params.append("    # Pointers to input / output tensors")
        params.append(f"    {ptr_names},")

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

        # -- Strides: output (M x N) --
        s_om = _stride_param(output.name, "m")
        s_on = _stride_param(output.name, "n")
        params.append(f"    # Strides for {output.name} (M x N)")
        params.append(f"    {s_om}, {s_on},")

        # -- Constexpr block sizes --
        params.append("    # Block sizes (compile-time constants)")
        params.append("    BLOCK_SIZE_M: tl.constexpr,")
        params.append("    BLOCK_SIZE_N: tl.constexpr,")
        params.append("    BLOCK_SIZE_K: tl.constexpr,")

        body = "\n".join(params)
        return f"def fused_kernel(\n{body}\n):"

    @staticmethod
    def _section_docstring() -> str:
        return '    """Auto-generated fused kernel — signature and pointer skeleton."""'

    @staticmethod
    def _section_program_ids() -> str:
        return "\n".join([
            "    # -----------------------------------------------------------",
            "    # Block index — each program instance owns one (M, N) tile",
            "    # -----------------------------------------------------------",
            "    pid_m = tl.program_id(0)",
            "    pid_n = tl.program_id(1)",
        ])

    @staticmethod
    def _section_block_offsets() -> str:
        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            "    # Block offsets — row, col, and reduction-axis index ranges",
            "    # -----------------------------------------------------------",
            "    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)",
            "    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)",
            "    offs_k = tl.arange(0, BLOCK_SIZE_K)",
        ])

    @staticmethod
    def _section_matrix_ptrs(
        tensor: TensorDescriptor,
        dim_labels: tuple[str, str],
    ) -> str:
        """Pointer arithmetic for a 2-D matrix tile.

        Loads a ``(BLOCK_SIZE_{d0}, BLOCK_SIZE_{d1})`` tile from global memory
        into SRAM.  The stride-based addressing allows non-contiguous layouts.
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
    def _section_accumulator() -> str:
        """Accumulator initialisation — fp32 to preserve numerical precision."""
        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            "    # Accumulator — stays in SRAM across the entire K-loop",
            "    # -----------------------------------------------------------",
            "    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)",
        ])

    @staticmethod
    def _section_k_loop(
        left: TensorDescriptor,
        right: TensorDescriptor,
    ) -> str:
        """Blocked GEMM loop — tiles left and right from HBM, accumulates in SRAM."""
        ln = left.name
        rn = right.name
        s_lk = _stride_param(ln, "k")
        s_rk = _stride_param(rn, "k")
        return "\n".join([
            "",
            "    # -----------------------------------------------------------",
            "    # Blocked GEMM loop over the K dimension",
            f"    # Each iteration loads one (BLOCK_SIZE_M, BLOCK_SIZE_K) tile of {ln}",
            f"    # and one (BLOCK_SIZE_K, BLOCK_SIZE_N) tile of {rn} from HBM into SRAM,",
            "    # accumulates the partial dot-product, then advances pointers.",
            "    # Total HBM reads: 2 * ceil(K / BLOCK_SIZE_K) tile loads per program.",
            "    # -----------------------------------------------------------",
            "    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):",
            "        # K-boundary mask — changes each iteration as k advances",
            "        k_mask = offs_k < K - k * BLOCK_SIZE_K",
            "",
            f"        # Load (BLOCK_SIZE_M x BLOCK_SIZE_K) tile of {ln} from HBM → SRAM",
            f"        # Full 2-D mask guards both the M boundary and the K boundary",
            f"        {ln} = tl.load({ln}_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)",
            f"        # Load (BLOCK_SIZE_K x BLOCK_SIZE_N) tile of {rn} from HBM → SRAM",
            f"        # Full 2-D mask guards both the K boundary and the N boundary",
            f"        {rn} = tl.load({rn}_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)",
            "",
            "        # Tile-level matrix multiply — accumulated entirely in SRAM",
            f"        acc += tl.dot({ln}, {rn})",
            "",
            "        # Advance to the next K-block (stride along the reduction axis)",
            f"        {ln}_ptrs += BLOCK_SIZE_K * {s_lk}",
            f"        {rn}_ptrs += BLOCK_SIZE_K * {s_rk}",
        ])

    @staticmethod
    def _section_store(output: TensorDescriptor) -> str:
        """Store the accumulated result from SRAM back to HBM exactly once.

        Emits a ``tl.store`` guarded by a 2-D boundary mask so that
        out-of-bounds threads do not corrupt memory.  When the output
        dtype differs from the fp32 accumulator, an explicit cast is
        emitted first to match the target precision.
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

        # Cast accumulator from fp32 to output dtype if needed
        if output.dtype != torch.float32:
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
        """GeLU: fast tanh approximation matching PyTorch's default.

        Formula: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        All arithmetic stays in SRAM registers — no HBM round-trip.
        """
        return "\n".join([
            "    # GeLU activation — fast tanh approximation (all in SRAM registers)",
            "    # Formula: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
            "    acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))",
        ])

    @staticmethod
    def _emit_add(
        node: torch.fx.Node,
        node_ids: set[int],
    ) -> str:
        """Residual add: load an external tensor tile from HBM, fuse into acc.

        Identifies the *external* argument (the one not produced by a prior
        node inside the fusion group) as the residual tensor.  Also handles
        scalar arguments (e.g. ``add(acc, 1.0)``) without emitting a load.
        """
        # Scan args for a scalar or an external tensor node.
        residual_name: str | None = None
        scalar_value: int | float | None = None

        for arg in node.args:
            if isinstance(arg, (int, float)):
                scalar_value = arg
                break
            if hasattr(arg, "name") and id(arg) not in node_ids:
                residual_name = arg.name
                break

        # Case 1: scalar add — no HBM traffic, register-only.
        if scalar_value is not None:
            return "\n".join([
                f"    # Scalar add — register-only (no HBM traffic)",
                f"    acc = acc + {scalar_value}",
            ])

        # Case 2: external tensor (residual connection) — one HBM load.
        if residual_name is not None:
            return "\n".join([
                f"    # Residual add — load {residual_name} tile from HBM into SRAM",
                (
                    f"    {residual_name} = tl.load({residual_name}_ptrs, "
                    f"mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)"
                ),
                f"    acc = acc + {residual_name}",
            ])

        # Case 3: both args are internal — just emit a plain add.
        return "\n".join([
            "    # Element-wise add (both operands already in SRAM)",
            "    acc = acc + acc",
        ])

    @staticmethod
    def _section_footer() -> str:
        return "\n    # --- Compute loop and store logic to be generated separately ---"
