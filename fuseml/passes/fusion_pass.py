"""FuseMLFusionPass — compiler pass that discovers and applies fusions.

This module contains the two-phase fusion pass (discovery + surgery) and the
placeholder function used as a symbolic target for fused kernel nodes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Set, Tuple

import torch
from torch.fx.passes.shape_prop import ShapeProp

from fuseml._logging import logger
from fuseml.codegen.kernel_cache import _materialize_ints
from fuseml.fusion_group import FusionGroup
from fuseml.passes.graph_cut import (
    TRANSPARENT_OPS,
    split_fusion_group,
    validate_fusion_group,
)
from fuseml.passes.mutation_safety import IN_PLACE_OPS, is_safe_inplace
from fuseml.registry import SupportedOpsRegistry, build_default_registry


# ---------------------------------------------------------------------------
# Placeholder for fused kernel calls (replaced by Triton codegen later)
# ---------------------------------------------------------------------------
def fuseml_fused_kernel_placeholder(*args):
    """Placeholder target for fused kernel nodes inserted during graph surgery.

    This function is never executed directly — it serves as a symbolic target
    in the FX graph that downstream Triton codegen will replace with a
    compiled kernel call.
    """
    raise RuntimeError(
        "fuseml_fused_kernel_placeholder should never be called directly. "
        "It must be replaced by a compiled Triton kernel before execution."
    )


class FuseMLFusionPass:
    """Graph-rewriting pass that fuses memory-bound operator sequences.

    The pass operates in two phases:

    1. **Discovery** (``_find_fusion_groups``): walk the FX graph and
       identify contiguous runs of registry-matched, memory-bound nodes
       that can be collapsed into single Triton kernels.
    2. **Surgery** (``_apply_surgery``): rewrite the graph by replacing
       each :class:`FusionGroup` with a call to the corresponding
       generated Triton kernel.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The traced graph to optimize.  Modified **in-place** during surgery.
    registry : SupportedOpsRegistry | None
        Op registry used to decide which nodes are fusion-eligible.
        Defaults to :func:`build_default_registry`.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        registry: SupportedOpsRegistry | None = None,
    ) -> None:
        self.graph_module = graph_module
        self.registry = registry or build_default_registry()

    # ------------------------------------------------------------------
    # Phase 1 — Discover fusible groups
    # ------------------------------------------------------------------
    # Ops that act as fusion barriers — require cross-thread synchronization,
    # reductions, or are heavy compute nodes that should start their own group.
    _BARRIER_OPS: Set[Callable] = {
        torch.ops.aten._softmax.default,
        torch.ops.aten._log_softmax.default,
        torch.ops.aten.native_layer_norm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.convolution.default,
    }

    # Pointwise ops eligible for absorption into an existing FusionGroup.
    _ABSORBABLE_OPS: Set[Callable] = {
        torch.ops.aten.relu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    }

    # Reduction ops that can be absorbed as the *final* operation in a
    # fusion group.  They require cross-thread synchronization (tl.atomic_*)
    # and collapse one tile dimension, so nothing further is absorbed after
    # a reduction.  Only keepdim=False reductions are absorbed; keepdim=True
    # is left as a barrier because the output remains 2-D while ``acc`` is
    # collapsed to 1-D, which the current store path does not handle.
    _REDUCTION_OPS: Set[Callable] = {
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.amax.default,
        torch.ops.aten.mean.dim,
    }

    def _find_fusion_groups(self) -> List[FusionGroup]:
        """Identify contiguous sequences of fusible memory-bound nodes.

        Walks ``self.graph_module.graph`` in topological order.  When a
        trigger node (``aten.addmm.default``) is encountered, a new
        :class:`FusionGroup` is initialized and the algorithm greedily
        absorbs downstream pointwise nodes that satisfy:

        * **Op type** — the node's target is in ``_ABSORBABLE_OPS``.
        * **Topology** — the node has exactly one user (no branching).

        Absorption halts on barrier ops, multi-user nodes, or any op not
        in the absorbable set.  Only groups with at least one absorbed
        node (i.e. len > 1) are returned, since a lone ``addmm`` offers
        no fusion benefit.

        Returns
        -------
        list[FusionGroup]
            Ordered list of fusion candidates.  Empty when no fusible
            sequences are found.
        """
        groups: List[FusionGroup] = []
        # Track nodes already claimed by a group to avoid overlapping fusions.
        consumed: Set[torch.fx.Node] = set()

        for node in self.graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            # Only trigger on the core compute op.
            if node.target is not torch.ops.aten.addmm.default:
                continue
            if node in consumed:
                continue

            group = FusionGroup(base_node=node)
            consumed.add(node)

            # Collect external inputs for the base node.
            group_node_set: Set[torch.fx.Node] = {node}
            self._collect_external_inputs(node, group_node_set, group)

            # --- Greedy forward absorption --------------------------------
            current = node
            while True:
                # The current node must feed exactly one downstream consumer.
                if len(current.users) != 1:
                    break

                successor = next(iter(current.users))

                # Must be a call_function to inspect its target.
                if successor.op != "call_function":
                    break

                # Halt on barrier / heavy-compute ops.
                if successor.target in self._BARRIER_OPS:
                    break

                # --- Reduction absorption (terminates the group) ----------
                if successor.target in self._REDUCTION_OPS:
                    # Only absorb keepdim=False; keepdim=True produces a
                    # 2-D output while the accumulator is collapsed to 1-D,
                    # which the store path cannot handle yet.
                    keepdim = (
                        successor.args[2]
                        if len(successor.args) > 2
                        else False
                    )
                    if keepdim:
                        break
                    group.fused_nodes.append(successor)
                    group.output_node = successor
                    consumed.add(successor)
                    group_node_set.add(successor)
                    self._collect_external_inputs(
                        successor, group_node_set, group,
                    )
                    break  # nothing further after a reduction

                # --- Transparent view/metadata penetration --------------------
                # If the successor is a view/reshape/unsqueeze that does not
                # change the 2-D tile shape, absorb it silently and keep
                # looking for the next compute op.  This lets chains like
                # addmm → view → relu fuse correctly.
                if successor.target in TRANSPARENT_OPS:
                    if self._is_shape_preserving_2d(successor):
                        group.fused_nodes.append(successor)
                        consumed.add(successor)
                        group_node_set.add(successor)
                        self._collect_external_inputs(
                            successor, group_node_set, group,
                        )
                        # Do NOT update output_node — a transparent op is
                        # never the "real" output; the next compute op is.
                        current = successor
                        continue
                    # Shape-changing view → stop absorption.
                    break

                # --- In-place op absorption with aliasing guard --------
                # In-place aten ops (relu_, add_, mul_, sigmoid_) can be
                # fused only when the mutated tensor is not aliased by
                # nodes outside the group.  The safety check walks the
                # view ancestry chain to catch transitive aliasing.
                if successor.target in IN_PLACE_OPS:
                    if is_safe_inplace(successor, group_node_set):
                        group.fused_nodes.append(successor)
                        group.output_node = successor
                        consumed.add(successor)
                        group_node_set.add(successor)
                        self._collect_external_inputs(
                            successor, group_node_set, group,
                        )
                        current = successor
                        continue
                    # Aliased externally — halt absorption to preserve
                    # PyTorch's expected mutation side-effects.
                    logger.debug(
                        "Aborting absorption at in-place op %s — "
                        "mutated tensor is aliased outside the group.",
                        successor.name,
                    )
                    break

                # Only absorb low-intensity pointwise ops.
                if successor.target not in self._ABSORBABLE_OPS:
                    break

                # Absorb the successor.
                group.fused_nodes.append(successor)
                group.output_node = successor
                consumed.add(successor)
                group_node_set.add(successor)

                # Record any external inputs the absorbed node requires.
                self._collect_external_inputs(successor, group_node_set, group)

                current = successor

            # --- Escape-node analysis -------------------------------------
            # For every node inside the group (except the final output_node,
            # which already has a dedicated tl.store), check whether any of
            # its users live *outside* the fused block.  Such nodes are
            # "escape nodes": their intermediate activation must be written
            # back to HBM so that PyTorch Autograd can retrieve it during
            # the backward pass.
            group_set = set(group.all_nodes)
            for n in group.all_nodes:
                if n is group.output_node:
                    continue  # handled by the final tl.store
                for user in n.users:
                    if user not in group_set:
                        group.intermediate_outputs.append(n)
                        break

            # --- Safety: no escape node after a reduction (defensive) ----
            # A reduction collapses acc from 2-D to 1-D.  An intermediate
            # tl.store emitted *after* the reduction would reference the
            # wrong shape.  The break-after-reduction rule (line 184)
            # structurally prevents this, but we guard against future
            # regressions by stripping any such escape nodes here.
            intermediate_set = set(group.intermediate_outputs)
            if intermediate_set:
                reduction_seen = False
                for n in group.all_nodes:
                    if n.target in self._REDUCTION_OPS:
                        reduction_seen = True
                    elif reduction_seen and n in intermediate_set:
                        logger.warning(
                            "Escape node %s after reduction in group %s "
                            "— removing from intermediate_outputs "
                            "(unsafe store).",
                            n.name,
                            group,
                        )
                        group.intermediate_outputs.remove(n)

            # --- Resolve get_attr parameter bindings ----------------------
            self._resolve_get_attr_bindings(group)

            # Only keep groups that actually fuse something (len > 1).
            if len(group) > 1:
                group.output_metadata = self._extract_tensor_metadata(
                    group.output_node
                )
                logger.debug("Fusion group found: %s", group)
                groups.append(group)

        return groups

    @staticmethod
    def _extract_tensor_metadata(node: torch.fx.Node) -> Dict[str, Any]:
        """Pull shape/stride/dtype from *node*'s ``tensor_meta``.

        Handles three cases:
        * ``tensor_meta`` is a single ``TensorMetadata`` namedtuple.
        * ``tensor_meta`` is a sequence of ``TensorMetadata`` (e.g. ops
          returning multiple tensors) — we take the first element.
        * ``tensor_meta`` is missing — returns an empty dict.
        """
        meta = node.meta.get("tensor_meta")
        if meta is None:
            return {}

        # A TensorMetadata namedtuple is itself a tuple subclass, so check
        # for the expected attributes first to distinguish a single metadata
        # entry from a sequence of them.
        if not hasattr(meta, "shape"):
            # Sequence of TensorMetadata — take the first element.
            if isinstance(meta, (tuple, list)) and len(meta) > 0:
                meta = meta[0]
            else:
                return {}

        if not hasattr(meta, "shape"):
            return {}

        return {
            "shape": _materialize_ints(meta.shape),
            "stride": _materialize_ints(meta.stride),
            "dtype": meta.dtype,
        }

    @staticmethod
    def _collect_external_inputs(
        node: torch.fx.Node,
        group_node_set: Set[torch.fx.Node],
        group: FusionGroup,
    ) -> None:
        """Add *node*'s args to ``group.inputs`` if produced outside the group."""
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg not in group_node_set:
                if arg not in group.inputs:
                    group.inputs.append(arg)

    @staticmethod
    def _is_shape_preserving_2d(node: torch.fx.Node) -> bool:
        """Return ``True`` if *node* is a view op that preserves 2-D shape.

        The Triton kernel stores its GEMM accumulator to a 2-D ``[M, N]``
        output.  A view op can only be absorbed transparently if its output
        shape matches the GEMM output dimensions.  Shape-changing views
        (different rank or different M/N) would cause a store mismatch and
        are conservatively rejected.

        Returns ``False`` when ``tensor_meta`` is missing (e.g. ShapeProp
        was not run), ensuring the conservative fallback: views are NOT
        absorbed and existing behavior is preserved.
        """
        node_meta = node.meta.get("tensor_meta")
        if node_meta is None or not hasattr(node_meta, "shape"):
            return False

        # The predecessor is the first arg (the tensor being viewed).
        if not node.args or not isinstance(node.args[0], torch.fx.Node):
            return False
        pred = node.args[0]
        pred_meta = pred.meta.get("tensor_meta")
        if pred_meta is None or not hasattr(pred_meta, "shape"):
            return False

        # Both must be 2-D with identical dimensions.
        if len(node_meta.shape) != 2 or len(pred_meta.shape) != 2:
            return False
        return (
            node_meta.shape[0] == pred_meta.shape[0]
            and node_meta.shape[1] == pred_meta.shape[1]
        )

    # ------------------------------------------------------------------
    # get_attr resolution — bind nn.Parameters to the FusionGroup
    # ------------------------------------------------------------------
    @staticmethod
    def _fetch_attr(
        gm: torch.fx.GraphModule,
        target: str,
    ) -> torch.nn.Parameter | torch.Tensor:
        """Resolve a ``get_attr`` *target* string to the live tensor on *gm*.

        The *target* is a dot-separated attribute path (e.g. ``"0.weight"``,
        ``"linear.bias"``) walked on the root :class:`GraphModule`.
        """
        attr: Any = gm
        for atom in target.split("."):
            attr = getattr(attr, atom)
        return attr

    def _resolve_get_attr_bindings(self, group: FusionGroup) -> None:
        """Populate ``group.param_bindings`` with resolved parameters.

        Scans ``group.inputs`` and their transitive args for ``get_attr``
        nodes, fetches the underlying ``nn.Parameter`` or buffer from
        ``self.graph_module``, and stores the mapping in
        ``group.param_bindings``.

        This makes parameter dependencies explicit so that downstream
        codegen and the compiler can access live tensor data without
        relying on potentially incomplete FX node metadata.
        """
        seen: Set[str] = set()

        def _resolve_node(node: torch.fx.Node) -> None:
            if not hasattr(node, "op"):
                return
            if node.op == "get_attr" and node.target not in seen:
                seen.add(node.target)
                try:
                    group.param_bindings[node.target] = self._fetch_attr(
                        self.graph_module, node.target,
                    )
                except AttributeError:
                    logger.warning(
                        "Could not resolve get_attr target %r on graph module.",
                        node.target,
                    )

        # Direct get_attr inputs.
        for inp in group.inputs:
            _resolve_node(inp)

        # Transitive: walk one level of args for each input to catch
        # patterns like  get_attr(weight) → t() → addmm  where t() is
        # the direct input but weight get_attr is one hop away.
        for inp in group.inputs:
            if hasattr(inp, "args"):
                for arg in inp.args:
                    if isinstance(arg, torch.fx.Node):
                        _resolve_node(arg)

    # ------------------------------------------------------------------
    # Phase 2 — Rewrite the graph
    # ------------------------------------------------------------------
    def _apply_surgery(self, groups: List[FusionGroup]) -> None:
        """Replace each *FusionGroup* in the graph with a fused kernel call.

        For every group, this method:
        1. Inserts a new ``call_function`` node targeting
           :func:`fuseml_fused_kernel_placeholder` immediately after the
           group's output node.
        2. Wires the new node's args to the group's external ``inputs``.
        3. Rewires all downstream consumers of the output node to read
           from the new placeholder node instead.
        4. After all groups are processed, eliminates dead code (the
           now-disconnected original sequences) and recompiles the
           ``GraphModule``.

        Parameters
        ----------
        groups : list[FusionGroup]
            Fusion groups produced by :meth:`_find_fusion_groups`.
        """
        graph = self.graph_module.graph

        for group in groups:
            # --- Insert placeholder node after the group's output ----------
            with graph.inserting_after(group.output_node):
                new_fused_node = graph.call_function(
                    fuseml_fused_kernel_placeholder,
                    args=tuple(group.inputs),
                )

            # Carry forward tensor_meta so downstream passes see shape info.
            if "tensor_meta" in group.output_node.meta:
                new_fused_node.meta["tensor_meta"] = group.output_node.meta[
                    "tensor_meta"
                ]

            # Copy metadata so downstream passes can inspect the group.
            new_fused_node.meta["fusion_group"] = group
            new_fused_node.meta["fused_op_names"] = [
                n.name for n in group.all_nodes
            ]
            new_fused_node.meta["param_bindings"] = group.param_bindings

            logger.debug(
                "Inserted fused placeholder %s for group %s",
                new_fused_node.name,
                group,
            )

            # --- Rewire downstream consumers to the new node ---------------
            group.output_node.replace_all_uses_with(
                new_fused_node,
                propagate_meta=False,
            )

            # --- Patch remaining groups' input lists ----------------------
            # replace_all_uses_with only updates FX graph args — it does
            # NOT touch the Python lists in other FusionGroup.inputs.
            # Without this patch, a later group whose inputs contained
            # the old output_node would create its placeholder with a
            # dangling reference, and DCE would remove the current
            # placeholder (zero graph-level users).
            for other_group in groups:
                if other_group is group:
                    continue
                for i, inp in enumerate(other_group.inputs):
                    if inp is group.output_node:
                        other_group.inputs[i] = new_fused_node

            # --- Dangling output guard ------------------------------------
            # If the fusion group is the final computation before the
            # graph's output node, verify the output node's args now
            # reference our new placeholder (not the dead original).
            for graph_node in graph.nodes:
                if graph_node.op == "output":
                    def _patch_arg(arg):
                        if arg is group.output_node:
                            logger.warning(
                                "Dangling output guard: output node still "
                                "references dead node %s — patching.",
                                group.output_node.name,
                            )
                            return new_fused_node
                        if isinstance(arg, (tuple, list)):
                            patched = [_patch_arg(a) for a in arg]
                            return type(arg)(patched)
                        return arg

                    new_args = tuple(_patch_arg(a) for a in graph_node.args)
                    if new_args != graph_node.args:
                        graph_node.args = new_args
                    break

        # --- Cleanup: rigorous dead code elimination -----------------------
        # Phase 1: Standard DCE removes nodes with zero users.
        graph.eliminate_dead_code()

        # Phase 2: Sweep orphaned get_attr nodes that may persist after
        # standard DCE (e.g. a weight get_attr whose sole consumer was a
        # now-dead transpose node).  Iterate in reverse topological order
        # so that dependents are erased before their dependencies — this
        # prevents erase_node from raising on nodes that still have users
        # within the orphaned set.
        for n in reversed(list(graph.nodes)):
            if n.op == "get_attr" and len(n.users) == 0:
                logger.debug("Removing orphaned get_attr node: %s", n.name)
                graph.erase_node(n)

        # Phase 3: Sweep orphaned call_function intermediates (e.g. aten.t
        # that only fed fused nodes).  Skip placeholder nodes — they are
        # the newly inserted targets.  Reverse order for the same
        # dependency-safety reason as Phase 2.
        for n in reversed(list(graph.nodes)):
            if (
                n.op == "call_function"
                and len(n.users) == 0
                and n.target is not fuseml_fused_kernel_placeholder
            ):
                logger.debug("Removing orphaned call_function: %s", n.name)
                graph.erase_node(n)

        # Phase 4: Final DCE to catch cascading orphans from phases 2-3.
        graph.eliminate_dead_code()

        # --- Validation: catch topological invariant violations early ------
        graph.lint()
        self.graph_module.recompile()

        logger.info(
            "Graph surgery complete — %d group(s) replaced with placeholder kernels.",
            len(groups),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        example_inputs: Tuple[torch.Tensor, ...] | None = None,
    ) -> torch.fx.GraphModule:
        """Execute the full fusion pass and return the optimized graph module.

        When *example_inputs* are provided, PyTorch's :class:`ShapeProp` is
        executed first so that every node's ``meta['tensor_meta']`` is
        populated with concrete shape / stride / dtype information.  This
        metadata is then extracted into each :class:`FusionGroup`'s
        ``output_metadata`` field for downstream Triton codegen.

        Parameters
        ----------
        example_inputs : tuple[torch.Tensor, ...] | None
            Representative input tensors matching the graph's signature.
            When ``None``, shape propagation is skipped and
            ``output_metadata`` will be empty.

        Returns
        -------
        torch.fx.GraphModule
            The same ``graph_module`` passed at construction, modified
            in-place with fused subgraphs replaced by Triton kernel calls.
        """
        # --- Shape propagation (requires concrete example tensors) ----------
        if example_inputs is not None:
            logger.debug("Running ShapeProp with %d example input(s).", len(example_inputs))
            ShapeProp(self.graph_module).propagate(*example_inputs)

        groups = self._find_fusion_groups()

        # --- Graph-cutting safeguard: validate before surgery -------------
        # Ensure every node in each group has a known Triton translation.
        # If an unsupported op slipped through pattern matching, the group
        # is split so that compilable prefixes are preserved and the rest
        # falls back to native PyTorch execution.
        validated: List[FusionGroup] = []
        for group in groups:
            segments = split_fusion_group(group)
            for seg in segments:
                if seg.kind == "fused" and seg.group is not None:
                    validated.append(seg.group)
                else:
                    logger.info(
                        "Native fallback for %d node(s): %s",
                        len(seg.nodes),
                        [n.name for n in seg.nodes],
                    )
        groups = validated

        if groups:
            logger.info("Found %d fusion group(s) — applying graph surgery.", len(groups))
            self._apply_surgery(groups)
        else:
            logger.info("No fusible sequences detected — graph unchanged.")

        return self.graph_module
