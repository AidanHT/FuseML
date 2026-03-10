"""FuseMLCompiler — torch.compile backend.

Intercepts FX graphs produced by TorchDynamo, discovers fusible operator
sequences, generates and compiles Triton kernels for each group, and
substitutes the compiled kernels back into the graph so that subsequent
forward passes dispatch directly to hardware.

Pipeline (per captured graph):
1. Tag all registry-matched nodes as fusion candidates (observability).
2. Run ``FuseMLFusionPass`` with shape propagation to discover groups and
   rewrite the graph with ``fuseml_fused_kernel_placeholder`` nodes.
3. For every placeholder node (one per fusion group):
   a. Build ``TensorDescriptor`` objects from FX node metadata.
   b. Generate the full Triton kernel source string.
   c. Compile the source via ``exec()`` into a ``@triton.jit`` callable.
   d. Wrap the callable in a ``KernelLauncher`` for runtime dispatch.
   e. Replace the placeholder node with a ``call_function`` to the launcher.
4. Recompile and return the optimised ``gm.forward``.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

from fuseml._logging import logger
from fuseml.codegen.cublas_epilogue import (
    CublasEpilogueLauncher,
    CublasResidualLauncher,
    match_cublas_epilogue,
)
from fuseml.codegen.kernel_cache import (
    KernelCache,
    KernelCacheKey,
    _materialize_ints,
    build_cache_key,
)
from fuseml.codegen.kernel_generator import (
    TensorDescriptor,
    TritonKernelGenerator,
    _classify,
    _identify_matmul_operands,
)
from fuseml.codegen.kernel_launcher import KernelLauncher
from fuseml.fusion_group import FusionGroup
from fuseml.passes.control_flow_validation import (
    ControlFlowError,
    validate_graph_control_flow,
)
from fuseml.passes.fusion_pass import FuseMLFusionPass, fuseml_fused_kernel_placeholder
from fuseml.passes.topology import is_trigger
from fuseml.registry import SupportedOpsRegistry, build_default_registry


# ---------------------------------------------------------------------------
# Sentinel exception — bypasses aot_module_simplified overhead
# ---------------------------------------------------------------------------

class _NoFusionPossible(Exception):
    """Raised when the ATen graph has no profitable fusion opportunities.

    Caught by :meth:`FuseMLCompiler.__call__` to bypass the
    ``aot_module_simplified`` wrapper and return the original (non-decomposed)
    forward function, eliminating per-invocation AOT overhead.
    """


# ---------------------------------------------------------------------------
# Node-to-descriptor helper
# ---------------------------------------------------------------------------

def _node_to_descriptor(node: torch.fx.Node) -> TensorDescriptor | None:
    """Build a :class:`TensorDescriptor` from an FX node's attached metadata.

    Two metadata sources are consulted in priority order:

    1. ``node.meta["tensor_meta"]`` — a ``TensorMetadata`` namedtuple
       populated by :class:`torch.fx.passes.shape_prop.ShapeProp`.
       Its ``.stride`` attribute is a plain tuple.
    2. ``node.meta["val"]`` — a ``FakeTensor`` stored by
       ``torch.compile``'s abstract-interpretation pass.
       Its ``.stride()`` attribute is a method.

    Returns ``None`` when neither source is present (e.g. scalar nodes or
    nodes for which shape propagation was not run).
    """
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is not None:
        # tensor_meta may be a single TensorMetadata or a sequence of them
        # (when the op returns multiple tensors).  Normalise to a single entry.
        if not hasattr(tensor_meta, "shape"):
            if isinstance(tensor_meta, (tuple, list)) and len(tensor_meta) > 0:
                tensor_meta = tensor_meta[0]
            else:
                tensor_meta = None

    if tensor_meta is not None and hasattr(tensor_meta, "shape"):
        shape = _materialize_ints(tensor_meta.shape)
        # TensorMetadata stores stride as a plain tuple.
        stride = _materialize_ints(tensor_meta.stride)
        dtype = tensor_meta.dtype
        return TensorDescriptor(name=node.name, shape=shape, stride=stride, dtype=dtype)

    # Fall back to FakeTensor stored by torch.compile.
    val = node.meta.get("val")
    if val is not None and hasattr(val, "shape"):
        shape = _materialize_ints(val.shape)
        # FakeTensor exposes stride() as a method.
        stride = _materialize_ints(val.stride()) if callable(getattr(val, "stride", None)) else (1,) * len(shape)
        dtype = val.dtype
        return TensorDescriptor(name=node.name, shape=shape, stride=stride, dtype=dtype)

    return None


def _descriptor_from_metadata(name: str, meta: dict) -> TensorDescriptor | None:
    """Build a :class:`TensorDescriptor` from a pre-extracted metadata dict.

    Used for ``FusionGroup.output_metadata`` which is already a plain
    ``{"shape": ..., "stride": ..., "dtype": ...}`` dict.
    """
    if not meta or "shape" not in meta:
        return None
    return TensorDescriptor(
        name=name,
        shape=_materialize_ints(meta["shape"]),
        stride=_materialize_ints(meta["stride"]),
        dtype=meta["dtype"],
    )


# ---------------------------------------------------------------------------
# FuseMLCompiler
# ---------------------------------------------------------------------------

class FuseMLCompiler:
    """Custom ``torch.compile`` backend that fuses memory-bound operator
    sequences into compiled Triton kernels.

    The compiler is invoked by TorchDynamo once per unique graph structure.
    It runs the full fusion pipeline — discovery, codegen, compilation, and
    graph surgery — and returns an optimised callable that dispatches every
    fusible subgraph to a single Triton kernel.

    Parameters
    ----------
    registry : SupportedOpsRegistry | None
        Op registry used to decide which nodes are fusion-eligible.
        Defaults to :func:`build_default_registry`.

    Attributes
    ----------
    fusion_candidates : list[torch.fx.Node]
        Accumulated nodes tagged as fusion candidates across all graphs
        processed by this compiler instance.
    """

    def __init__(self, registry: SupportedOpsRegistry | None = None) -> None:
        self.registry: SupportedOpsRegistry = registry or build_default_registry()
        self.fusion_candidates: List[torch.fx.Node] = []
        self.fusion_applied: bool = False
        self.fusion_strategy: str = "none"
        self._generator = TritonKernelGenerator()
        self._cache = KernelCache()

    # ------------------------------------------------------------------
    # Entry point called by torch.compile for each captured sub-graph
    # ------------------------------------------------------------------
    def __call__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        """Run the full fusion pipeline on *gm* and return its forward method.

        TorchDynamo may deliver a graph with high-level ops (e.g.
        ``torch.nn.functional.linear``) rather than decomposed ATen ops.
        We use ``aot_module_simplified`` to lower the graph to ATen level
        (producing ``aten.addmm.default``, ``aten.gelu.default``, etc.)
        before running the fusion pipeline.

        When the ATen-level analysis determines no profitable fusion is
        possible (e.g. all GEMMs are compute-bound), the compiler raises
        :class:`_NoFusionPossible` to bypass the ``aot_module_simplified``
        wrapper entirely — returning the original (non-decomposed) forward
        function so that the no-fusion path has zero per-invocation overhead.

        Parameters
        ----------
        gm:
            The ``GraphModule`` produced by TorchDynamo.
        example_inputs:
            Representative tensors used during tracing.

        Returns
        -------
        Callable
            The (possibly optimised) forward callable.
        """

        def _aten_compiler(
            aten_gm: torch.fx.GraphModule,
            aten_inputs: List[torch.Tensor],
        ) -> Callable:
            """Inner compiler that receives the ATen-decomposed graph."""
            optimised = self._compile_aten_graph(aten_gm, aten_inputs)
            return make_boxed_func(optimised)

        try:
            return aot_module_simplified(
                gm,
                example_inputs,
                fw_compiler=_aten_compiler,
            )
        except _NoFusionPossible as exc:
            # No profitable fusion found — return the original forward
            # WITHOUT the aot_module_simplified wrapper.  This eliminates
            # the per-invocation overhead of AOT Autograd's functionalization
            # layer, ensuring the no-fusion path matches eager speed.
            self.fusion_applied = False
            logger.info(
                "Bypassing AOT decomposition — %s", exc,
            )
            return gm.forward

    # ------------------------------------------------------------------
    # ATen-level compilation (called after aot_autograd decomposition)
    # ------------------------------------------------------------------

    def _compile_aten_graph(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        """Run the fusion pipeline on an ATen-decomposed graph.

        This is the core compilation logic, now guaranteed to receive
        ATen-level ops (``aten.addmm.default``, ``aten.gelu.default``,
        etc.) regardless of the PyTorch version.
        """
        # Clear per-compilation state to avoid stale references when
        # TorchDynamo triggers recompilation (e.g. different grad context).
        self.fusion_candidates = []

        logger.info(
            "Captured FX graph with %d nodes — scanning for fusion candidates …",
            sum(1 for _ in gm.graph.nodes),
        )

        # ── Step 0: validate for data-dependent control flow ──────────
        try:
            validate_graph_control_flow(gm)
        except ControlFlowError as exc:
            raise _NoFusionPossible(
                f"data-dependent control flow: {exc}"
            ) from exc

        # ── Step 1: single-pass candidate tagging + trigger pre-scan ──
        # Combines the observability tagging, trigger detection, and
        # early compute-bound rejection into one graph traversal.
        has_any_trigger = False
        has_fusible_trigger = False
        found: int = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if self.registry.is_supported(node.target):
                    node.meta["fusion_candidate"] = True
                    node.meta["fusion_category"] = self.registry.get_category(node.target)
                    self.fusion_candidates.append(node)
                    found += 1
                if is_trigger(node):
                    has_any_trigger = True
                    if FuseMLFusionPass._is_tiny_output(node):
                        logger.debug(
                            "Tiny GEMM trigger %s — skipping (output "
                            "too small for profitable fusion).",
                            node.name,
                        )
                    elif not FuseMLFusionPass._is_compute_bound_trigger(
                        node, min_penalty=100e-6,
                    ):
                        has_fusible_trigger = True  # Triton-fusible
                    elif (
                        node.meta.get("fusion_candidate", False)
                        and match_cublas_epilogue(node) is not None
                    ):
                        has_fusible_trigger = True  # cuBLAS-epilogue-fusible

        if not has_any_trigger:
            raise _NoFusionPossible("no trigger ops in graph")

        if not has_fusible_trigger:
            raise _NoFusionPossible(
                "all trigger ops are compute-bound — cuBLAS preferred"
            )

        self._log_graph_dump(gm, found)

        # ── Step 2: run fusion pass (discovery + surgery) ─────────────
        fusion_pass = FuseMLFusionPass(gm, self.registry)
        gm = fusion_pass.run(example_inputs=example_inputs)

        # ── Step 3: replace placeholder nodes with compiled launchers ──
        nodes_to_process = [
            node for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is fuseml_fused_kernel_placeholder
        ]

        if not nodes_to_process:
            raise _NoFusionPossible(
                "fusion pass found no fusible sequences"
            )

        # ── Build tensor map for cache key construction ────────────────
        # Map graph-level placeholder node names to the corresponding
        # example_inputs tensors so that build_cache_key() can extract
        # true storage offsets and pointer alignment from live tensors
        # (which may be views or slices with non-trivial ATen layout).
        graph_placeholders = [
            n for n in gm.graph.nodes if n.op == "placeholder"
        ]
        tensor_map: dict[str, torch.Tensor] = {}
        for node, inp in zip(graph_placeholders, example_inputs):
            if isinstance(inp, torch.Tensor):
                tensor_map[node.name] = inp

        strategies_used: set[str] = set()

        for placeholder_node in nodes_to_process:
            group: FusionGroup | None = placeholder_node.meta.get("fusion_group")
            if group is None:
                logger.warning(
                    "Placeholder node %s has no fusion_group metadata — skipping.",
                    placeholder_node.name,
                )
                continue

            # ── cuBLAS epilogue path — no Triton codegen needed ───────
            if group.fusion_strategy == "cublas_epilogue":
                launcher_fn = self._build_cublas_launcher(group)
                if launcher_fn is None:
                    logger.warning(
                        "Could not build CublasEpilogueLauncher for %s "
                        "— keeping placeholder.",
                        placeholder_node.name,
                    )
                    continue

                with gm.graph.inserting_after(placeholder_node):
                    new_node = gm.graph.call_function(
                        launcher_fn,
                        args=placeholder_node.args,
                    )
                if "tensor_meta" in placeholder_node.meta:
                    new_node.meta["tensor_meta"] = placeholder_node.meta["tensor_meta"]

                placeholder_node.replace_all_uses_with(new_node)
                gm.graph.erase_node(placeholder_node)
                strategies_used.add("cublas_epilogue")

                logger.debug(
                    "Replaced placeholder %s with CublasEpilogueLauncher "
                    "node %s.",
                    placeholder_node.name,
                    new_node.name,
                )
                continue

            # ── Triton path (existing) ────────────────────────────────

            # Enrich tensor_map with resolved parameters.
            if group.param_bindings:
                for attr_name, param_tensor in group.param_bindings.items():
                    if isinstance(param_tensor, torch.Tensor):
                        tensor_map[attr_name] = param_tensor

            # Cache lookup.
            cache_key = build_cache_key(group, tensor_map)
            launcher: KernelLauncher | None = None
            if cache_key is not None:
                launcher = self._cache.lookup(cache_key)

            if launcher is None:
                launcher = self._build_launcher(group)
                if launcher is not None and cache_key is not None:
                    self._cache.store(cache_key, launcher)

            if launcher is None:
                logger.warning(
                    "Could not build KernelLauncher for %s — keeping placeholder.",
                    placeholder_node.name,
                )
                continue

            with gm.graph.inserting_after(placeholder_node):
                new_node = gm.graph.call_function(
                    launcher,
                    args=placeholder_node.args,
                )
            if "tensor_meta" in placeholder_node.meta:
                new_node.meta["tensor_meta"] = placeholder_node.meta["tensor_meta"]

            placeholder_node.replace_all_uses_with(new_node)
            gm.graph.erase_node(placeholder_node)
            strategies_used.add("triton")

            logger.debug(
                "Replaced placeholder %s with KernelLauncher node %s.",
                placeholder_node.name,
                new_node.name,
            )

        gm.graph.eliminate_dead_code()
        gm.recompile()

        self.fusion_applied = True
        if strategies_used == {"cublas_epilogue"}:
            self.fusion_strategy = "cublas_epilogue"
        elif "cublas_epilogue" in strategies_used and "triton" in strategies_used:
            self.fusion_strategy = "mixed"
        elif "triton" in strategies_used:
            self.fusion_strategy = "triton"
        else:
            self.fusion_strategy = "none"
        logger.info(
            "Kernel substitution complete — %d launcher(s) inserted "
            "(strategy: %s).",
            len(nodes_to_process),
            self.fusion_strategy,
        )

        return gm.forward

    # ------------------------------------------------------------------
    # Launcher construction
    # ------------------------------------------------------------------

    def _build_launcher(self, group: FusionGroup) -> KernelLauncher | None:
        """Generate, compile, and wrap a Triton kernel for *group*.

        Returns ``None`` if any required metadata is missing (e.g. shape
        propagation was not run and descriptors cannot be built).
        """
        # ── Input descriptors ─────────────────────────────────────────
        input_descs: list[TensorDescriptor] = []
        for n in group.inputs:
            desc = _node_to_descriptor(n)
            # Fallback: if the node is a get_attr with no FX metadata,
            # build the descriptor directly from the resolved parameter
            # tensor in param_bindings.
            if (
                desc is None
                and hasattr(n, "op")
                and n.op == "get_attr"
                and n.target in group.param_bindings
            ):
                param = group.param_bindings[n.target]
                desc = TensorDescriptor(
                    name=n.name,
                    shape=_materialize_ints(param.shape),
                    stride=_materialize_ints(param.stride()),
                    dtype=param.dtype,
                )
            if desc is None:
                logger.warning(
                    "Cannot build descriptor for input node %s — "
                    "missing tensor_meta / val metadata.  Skipping group.",
                    n.name,
                )
                return None
            input_descs.append(desc)

        # ── Output descriptor ─────────────────────────────────────────
        out_desc = _descriptor_from_metadata(
            name=group.output_node.name,
            meta=group.output_metadata,
        )
        if out_desc is None:
            # Fall back to building from the output node's own metadata.
            out_desc = _node_to_descriptor(group.output_node)
        if out_desc is None:
            logger.warning(
                "Cannot build output descriptor for group %s — skipping.", group
            )
            return None

        # ── Intermediate (escape) descriptors ────────────────────────
        intermediate_descs: list[TensorDescriptor] = []
        for n in group.intermediate_outputs:
            desc = _node_to_descriptor(n)
            if desc is None:
                logger.warning(
                    "Cannot build descriptor for intermediate node %s — "
                    "skipping escape store for this node.",
                    n.name,
                )
                # Skip the escape store rather than aborting the whole group.
                continue
            intermediate_descs.append(desc)

        # Rebuild intermediate_outputs list to match only those descriptors
        # that were successfully built (guard against metadata gaps).
        valid_escape_nodes = [
            n for n in group.intermediate_outputs
            if _node_to_descriptor(n) is not None
        ]
        escape_stores = {
            id(n): desc
            for n, desc in zip(valid_escape_nodes, intermediate_descs)
        }

        # ── Identify matmul operands for the launcher ─────────────────
        matrices, _ = _classify(input_descs)
        if len(matrices) < 2:
            logger.warning(
                "Group %s has fewer than 2 matrix inputs — cannot identify "
                "matmul operands.  Skipping.",
                group,
            )
            return None
        try:
            left, right = _identify_matmul_operands(matrices)
        except ValueError as exc:
            logger.warning("Matmul operand identification failed for %s: %s", group, exc)
            return None

        # ── Detect reduction early (needed for autotune config) ───────
        has_reduction = any(
            hasattr(n, "target") and "sum" in str(getattr(n.target, "__name__", ""))
            or hasattr(n, "target") and "mean" in str(getattr(n.target, "__name__", ""))
            or hasattr(n, "target") and "amax" in str(getattr(n.target, "__name__", ""))
            for n in group.fused_nodes
        )

        # ── Code generation (autotuned) ──────────────────────────────
        sig = self._generator.generate_signature_and_pointers(
            input_descs, out_desc, intermediate_descs,
            autotune=True, has_reduction=has_reduction,
        )
        kloop = self._generator.generate_k_loop(input_descs, out_desc)
        all_ids = {id(n) for n in group.all_nodes}
        epilogue = self._generator.generate_epilogue(
            group.fused_nodes, escape_stores, output_descriptor=out_desc,
            all_group_node_ids=all_ids,
            node_args_snapshot=group.node_args_snapshot,
        )

        full_kernel_str = sig + "\n" + kloop + "\n" + epilogue

        # ── Compilation ───────────────────────────────────────────────
        try:
            kernel_fn = self._generator.compile_and_bind(full_kernel_str, out_desc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Triton compilation failed for %s: %s", group, exc)
            return None

        # ── Detect reduction for launcher initialisation ─────────────
        # The generator records ReductionInfo (op + axis) during the
        # epilogue.  Use it directly instead of re-inspecting the FX nodes.
        reduction_op: str | None = None
        reduction_axis: int | None = None
        if self._generator._last_reduction is not None:
            reduction_op = self._generator._last_reduction.op
            reduction_axis = self._generator._last_reduction.axis

        # ── Assemble launcher (autotuned — Triton selects block sizes) ─
        launcher = KernelLauncher(
            kernel_fn=kernel_fn,
            input_descriptors=input_descs,
            output_descriptor=out_desc,
            intermediate_descriptors=intermediate_descs,
            left_name=left.name,
            right_name=right.name,
            is_autotuned=True,
            reduction_op=reduction_op,
            reduction_axis=reduction_axis,
        )

        logger.info("Built %r for group %s.", launcher, group)
        return launcher

    # ------------------------------------------------------------------
    # cuBLAS epilogue launcher construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cublas_launcher(
        group: FusionGroup,
    ) -> CublasEpilogueLauncher | CublasResidualLauncher | None:
        """Build a cuBLAS epilogue launcher for a cuBLAS epilogue group.

        Dispatches based on the pattern's ``epilogue_type``:
        - ``GELU_BIAS`` / ``RELU_BIAS`` → :class:`CublasEpilogueLauncher`
        - ``BIAS_RESIDUAL`` → :class:`CublasResidualLauncher`

        Returns ``None`` if the group lacks a valid ``cublas_pattern``.
        """
        pattern = group.cublas_pattern
        if pattern is None:
            return None

        if pattern.epilogue_type == "BIAS_RESIDUAL":
            if len(group.inputs) < 4:
                logger.warning(
                    "BIAS_RESIDUAL group %s has %d inputs (expected 4) "
                    "— skipping.",
                    group, len(group.inputs),
                )
                return None
            launcher = CublasResidualLauncher()
        else:
            if len(group.inputs) < 3:
                logger.warning(
                    "cuBLAS epilogue group %s has %d inputs (expected 3) "
                    "— skipping.",
                    group, len(group.inputs),
                )
                return None
            launcher = CublasEpilogueLauncher(
                use_gelu=pattern.use_gelu,
                epilogue_type=pattern.epilogue_type,
            )

        logger.info("Built %r for group %s.", launcher, group)
        return launcher

    # ------------------------------------------------------------------
    # Graph logging (observability)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_graph_dump(gm: torch.fx.GraphModule, found: int) -> None:
        """Log the annotated FX graph (nodes must already be tagged)."""
        logger.info("--- FX Graph Dump (fusion candidates marked with *) ---")
        for node in gm.graph.nodes:
            is_candidate = node.meta.get("fusion_candidate", False)
            marker = " * [FUSION CANDIDATE]" if is_candidate else ""
            category = ""
            if is_candidate:
                category = f" (category={node.meta.get('fusion_category', '?')})"
            logger.info(
                "  %-12s %-30s target=%-40s%s%s",
                node.op,
                node.name,
                str(getattr(node.target, "name", node.target))[:40],
                marker,
                category,
            )
        logger.info("--- End Graph Dump ---")

        if found:
            logger.info(
                "Tagged %d node(s) as fusion candidates via registry.", found
            )
        else:
            logger.info("No registry-matched ops detected in this graph.")

    def capture_and_print_graph(self, gm: torch.fx.GraphModule) -> None:
        """Walk the FX graph, tag registry-matched nodes, and log the result.

        Backward-compatible entry point.  The main compilation pipeline
        uses the combined single-pass tagging in ``_compile_aten_graph``
        and calls ``_log_graph_dump`` directly.
        """
        found: int = 0
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if self.registry.is_supported(node.target):
                node.meta["fusion_candidate"] = True
                node.meta["fusion_category"] = self.registry.get_category(node.target)
                self.fusion_candidates.append(node)
                found += 1
        self._log_graph_dump(gm, found)


# ======================================================================
# Smoke test — run directly to verify graph interception
# ======================================================================
def main() -> None:
    """Compile a minimal Linear+GeLU block through the FuseML backend and
    validate that eager-mode numerics are preserved."""

    # ── Build a simple test module ────────────────────────────────────
    model: nn.Module = nn.Sequential(
        nn.Linear(128, 128),
        nn.GELU(),
    )

    compiler = FuseMLCompiler()  # uses default registry (addmm only)
    compiled_model = torch.compile(model, backend=compiler)

    # ── Run a forward pass to trigger graph capture ───────────────────
    x: torch.Tensor = torch.randn(4, 128)

    with torch.no_grad():
        eager_out: torch.Tensor = model(x)
        fused_out: torch.Tensor = compiled_model(x)

    # ── Correctness gate: fused path must match eager path ────────────
    assert torch.allclose(eager_out, fused_out, atol=1e-3, rtol=1e-3), (
        "Mismatch between eager and compiled outputs — "
        "backend must not alter numerics during interception phase."
    )

    logger.info(
        "Validation passed — compiled output matches eager output "
        "(atol=1e-3, rtol=1e-3).  %d fusion candidate(s) logged.",
        len(compiler.fusion_candidates),
    )


if __name__ == "__main__":
    main()
