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
    """

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


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
    # Public API
    # ------------------------------------------------------------------

    def generate_signature_and_pointers(
        self,
        input_tensors: list[TensorDescriptor],
        output_tensor: TensorDescriptor,
        intermediate_tensors: list[TensorDescriptor] | None = None,
    ) -> str:
        """Return a Triton kernel source string (no compute loop).

        The returned string contains, in order:

        1. ``@triton.jit`` decorator.
        2. ``def fused_kernel(...)`` with dynamically generated pointer,
           dimension, stride, and ``tl.constexpr`` block-size parameters.
           Pointers and strides follow the caller-supplied input order.
           When *intermediate_tensors* are provided, their output pointers
           and M×N strides are appended after the primary output parameters.
        3. ``tl.program_id`` block-offset computation for the M and N
           dimensions.
        4. Initial pointer arithmetic for every input tensor, the output
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

        sections = [
            self._section_decorator(),
            self._section_function_def(unique, output_tensor, dim_labels, intermediate_tensors),
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

        # After the GEMM loop, load and add any 1-D bias vectors.
        # addmm computes: bias + left @ right — the bias is broadcast
        # along the appropriate axis and added to the accumulator.
        _, vectors = _classify(unique)
        for v in vectors:
            dim = _vector_dim(v, left.shape[0], right.shape[-1])
            if dim == "n":
                sections.append(
                    f"\n    # Bias addition — load {v.name} and broadcast along N axis"
                    f"\n    {v.name} = tl.load({v.name}_ptrs, mask=offs_n < N, other=0.0)"
                    f"\n    acc = acc + {v.name}[None, :]"
                )
            else:
                sections.append(
                    f"\n    # Bias addition — load {v.name} and broadcast along M axis"
                    f"\n    {v.name} = tl.load({v.name}_ptrs, mask=offs_m < M, other=0.0)"
                    f"\n    acc = acc + {v.name}[:, None]"
                )

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

            # Both the functional (relu) and in-place (relu_) variants produce
            # identical register semantics — tl.where(acc > 0, acc, 0.0) —
            # because the Triton accumulator is a register tile, not a
            # memory-backed tensor.  No separate in-place path is needed.
            # The aliasing safety check in mutation_safety.py guarantees
            # the in-place variant is only fused when alias-safe.
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
            "import triton.language as tl\n\n"
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
        """Blocked GEMM loop — tiles left and right from HBM, accumulates in SRAM.

        No contiguity assumption is made anywhere in this loop.  Both the
        initial pointer tiles and the per-iteration pointer advances rely
        exclusively on the dynamically passed stride arguments (stride_*m,
        stride_*k, stride_*n).  A layout of stride=1 along any axis is
        handled identically to any other stride value — the caller is
        responsible for passing the correct runtime stride.
        """
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
            "        # Advance to the next K-block using the dynamically passed stride",
            "        # argument — no assumption is made that the tensor is contiguous",
            "        # along the K axis (stride=1 is handled identically to any other).",
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
            # Unconditional cast: the accumulator is always tl.float32.
            # For fp32 outputs this is an identity cast (zero runtime cost);
            # for fp16/bf16 outputs it narrows the result to the target dtype.
            # Keeping the cast unconditional makes the dtype contract explicit
            # and ensures correctness survives future dtype changes.
            f"    acc = acc.to({triton_dtype})",
        ]

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
            "    _gelu_inner = 0.7978845608 * (acc + 0.044715 * acc * acc * acc)",
            "    acc = 0.5 * acc * (1.0 + tl.extra.cuda.libdevice.tanh(_gelu_inner))",
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
                    f"    {residual_name} = tl.load({residual_name}_ptrs, mask=offs_n < N, other=0.0)",
                    f"    acc = acc + {residual_name}[None, :]",
                ])

            return "\n".join([
                f"    # Residual add — load 2-D {residual_name} tile from HBM into SRAM",
                (
                    f"    {residual_name} = tl.load({residual_name}_ptrs, "
                    f"mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)"
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
                    f"    {ext_name} = tl.load({ext_name}_ptrs, mask=offs_n < N, other=0.0)",
                    f"    acc = acc * {ext_name}[None, :]",
                ])

            return "\n".join([
                f"    # Tensor mul — load 2-D {ext_name} tile from HBM into SRAM",
                (
                    f"    {ext_name} = tl.load({ext_name}_ptrs, "
                    f"mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)"
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
        """Sum reduction with cross-thread synchronization via ``tl.atomic_add``.

        ``tl.sum`` collapses the tile-local accumulator along *axis*,
        producing a partial sum per program instance.  ``tl.atomic_add``
        accumulates the partials across all programs that share the
        surviving dimension so that the final result is correct even when
        the reduced dimension spans multiple tile blocks.
        """
        dim_name = "N" if axis == 1 else "M"
        offs = "offs_m" if axis == 1 else "offs_n"
        bound = "M" if axis == 1 else "N"
        return "\n".join([
            f"    # Reduction: sum along {dim_name} — cross-thread synchronization",
            f"    # tl.sum collapses axis={axis} within this program's tile.",
            f"    # tl.atomic_add accumulates partial sums across programs that",
            f"    # share the same {bound}-tile, producing the complete result.",
            f"    partial_sum = tl.sum(acc, axis={axis}).to({triton_dtype})",
            f"    tl.atomic_add({output_name}_ptrs, partial_sum, mask={offs} < {bound})",
        ])

    @staticmethod
    def _emit_max(axis: int, output_name: str, triton_dtype: str) -> str:
        """Max reduction with cross-thread synchronization via ``tl.atomic_max``.

        Same partial-result strategy as :meth:`_emit_sum` but uses
        ``tl.max`` / ``tl.atomic_max`` instead.
        """
        dim_name = "N" if axis == 1 else "M"
        offs = "offs_m" if axis == 1 else "offs_n"
        bound = "M" if axis == 1 else "N"
        return "\n".join([
            f"    # Reduction: max along {dim_name} — cross-thread synchronization",
            f"    # tl.max collapses axis={axis}, tl.atomic_max accumulates partials.",
            f"    partial_max = tl.max(acc, axis={axis}).to({triton_dtype})",
            f"    tl.atomic_max({output_name}_ptrs, partial_max, mask={offs} < {bound})",
        ])

    @staticmethod
    def _emit_mean(axis: int, output_name: str, triton_dtype: str) -> str:
        """Mean reduction — partial sums via ``tl.atomic_add``, division deferred.

        Each program contributes its partial ``tl.sum`` via
        ``tl.atomic_add``.  The division by the full dimension size is
        deferred to :class:`~fuseml.codegen.kernel_launcher.KernelLauncher`
        for better numerical precision (avoids many small floating-point
        divisions inside the kernel).
        """
        dim_name = "N" if axis == 1 else "M"
        offs = "offs_m" if axis == 1 else "offs_n"
        bound = "M" if axis == 1 else "N"
        return "\n".join([
            f"    # Reduction: mean along {dim_name} — cross-thread synchronization",
            f"    # Partial sum accumulated via tl.atomic_add; division by {dim_name}",
            f"    # is deferred to the kernel launcher for numerical precision.",
            f"    partial_sum = tl.sum(acc, axis={axis}).to({triton_dtype})",
            f"    tl.atomic_add({output_name}_ptrs, partial_sum, mask={offs} < {bound})",
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
