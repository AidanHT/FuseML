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

from fuseml._logging import logger
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
from fuseml.registry import SupportedOpsRegistry, build_default_registry


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
        shape = tuple(tensor_meta.shape)
        # TensorMetadata stores stride as a plain tuple.
        stride = tuple(tensor_meta.stride)
        dtype = tensor_meta.dtype
        return TensorDescriptor(name=node.name, shape=shape, stride=stride, dtype=dtype)

    # Fall back to FakeTensor stored by torch.compile.
    val = node.meta.get("val")
    if val is not None and hasattr(val, "shape"):
        shape = tuple(val.shape)
        # FakeTensor exposes stride() as a method.
        stride = tuple(val.stride()) if callable(getattr(val, "stride", None)) else (1,) * len(shape)
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
        shape=tuple(meta["shape"]),
        stride=tuple(meta["stride"]),
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
        self._generator = TritonKernelGenerator()

    # ------------------------------------------------------------------
    # Entry point called by torch.compile for each captured sub-graph
    # ------------------------------------------------------------------
    def __call__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        """Run the full fusion pipeline on *gm* and return its forward method.

        Parameters
        ----------
        gm:
            The ``GraphModule`` produced by TorchDynamo containing an FX
            graph of aten-level operations.
        example_inputs:
            Representative tensors used during tracing.  Forwarded to
            ``FuseMLFusionPass`` for shape propagation so that every node's
            ``tensor_meta`` is populated before codegen.

        Returns
        -------
        Callable
            The (possibly optimised) ``gm.forward``.  When no fusible
            sequences are detected the original forward is returned
            unchanged.
        """
        logger.info(
            "Captured FX graph with %d nodes — scanning for fusion candidates …",
            len(list(gm.graph.nodes)),
        )

        # ── Step 0: validate for data-dependent control flow ──────────
        try:
            validate_graph_control_flow(gm)
        except ControlFlowError as exc:
            logger.warning(
                "Data-dependent control flow detected — skipping fusion "
                "and falling back to eager execution for this subgraph: %s",
                exc,
            )
            return gm.forward

        # ── Step 1: tag candidates for observability ──────────────────
        self.capture_and_print_graph(gm)

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
            logger.info("No placeholder nodes found — returning unmodified forward.")
            return gm.forward

        for placeholder_node in nodes_to_process:
            group: FusionGroup | None = placeholder_node.meta.get("fusion_group")
            if group is None:
                logger.warning(
                    "Placeholder node %s has no fusion_group metadata — skipping.",
                    placeholder_node.name,
                )
                continue

            launcher = self._build_launcher(group)
            if launcher is None:
                logger.warning(
                    "Could not build KernelLauncher for %s — keeping placeholder.",
                    placeholder_node.name,
                )
                continue

            # Replace the placeholder with a call_function to the launcher.
            with gm.graph.inserting_after(placeholder_node):
                new_node = gm.graph.call_function(
                    launcher,
                    args=placeholder_node.args,
                )
            # Copy tensor_meta so downstream passes see shape info.
            if "tensor_meta" in placeholder_node.meta:
                new_node.meta["tensor_meta"] = placeholder_node.meta["tensor_meta"]

            placeholder_node.replace_all_uses_with(new_node)
            gm.graph.erase_node(placeholder_node)

            logger.debug(
                "Replaced placeholder %s with KernelLauncher node %s.",
                placeholder_node.name,
                new_node.name,
            )

        gm.graph.eliminate_dead_code()
        gm.recompile()

        logger.info(
            "Kernel substitution complete — %d launcher(s) inserted.",
            len(nodes_to_process),
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

        # ── Code generation ───────────────────────────────────────────
        sig = self._generator.generate_signature_and_pointers(
            input_descs, out_desc, intermediate_descs
        )
        kloop = self._generator.generate_k_loop(input_descs, out_desc)
        epilogue = self._generator.generate_epilogue(
            group.all_nodes, escape_stores, output_descriptor=out_desc,
        )

        full_kernel_str = sig + "\n" + kloop + "\n" + epilogue

        # ── Compilation ───────────────────────────────────────────────
        try:
            kernel_fn = self._generator.compile_and_bind(full_kernel_str, out_desc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Triton compilation failed for %s: %s", group, exc)
            return None

        # ── Detect reduction for launcher initialisation ─────────────
        reduction_op: str | None = None
        if group.fused_nodes:
            last_target = group.fused_nodes[-1].target
            if last_target == torch.ops.aten.sum.dim_IntList:
                reduction_op = "sum"
            elif last_target == torch.ops.aten.amax.default:
                reduction_op = "max"
            elif last_target == torch.ops.aten.mean.dim:
                reduction_op = "mean"

        # ── Assemble launcher ─────────────────────────────────────────
        launcher = KernelLauncher(
            kernel_fn=kernel_fn,
            input_descriptors=input_descs,
            output_descriptor=out_desc,
            intermediate_descriptors=intermediate_descs,
            left_name=left.name,
            right_name=right.name,
            reduction_op=reduction_op,
        )

        logger.info("Built %r for group %s.", launcher, group)
        return launcher

    # ------------------------------------------------------------------
    # Graph capture, tagging, and printing (observability)
    # ------------------------------------------------------------------
    def capture_and_print_graph(self, gm: torch.fx.GraphModule) -> None:
        """Walk the FX graph, tag registry-matched nodes, and log the result.

        For every ``call_function`` node whose target is in ``self.registry``,
        attaches ``node.meta['fusion_candidate'] = True`` and the op's
        registry category.  A formatted summary is logged at INFO level.
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

        # ── Print annotated graph ──────────────────────────────────────
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
