"""FuseMLFusionPass — compiler pass that discovers and applies fusions.

This module contains the two-phase fusion pass (discovery + surgery) and the
placeholder function used as a symbolic target for fused kernel nodes.
"""

from __future__ import annotations

import operator as _operator
from typing import Any, Callable, Dict, List, Set, Tuple

import torch
from torch.fx.passes.shape_prop import ShapeProp

from fuseml._logging import logger
from fuseml.codegen.kernel_cache import _materialize_ints
from fuseml.fusion_group import FusionGroup
from fuseml.passes.graph_cut import (
    split_fusion_group,
    validate_fusion_group,
)
from fuseml.passes.mutation_safety import is_safe_inplace
from fuseml.passes.topology import (
    ABSORBABLE_OPS,
    BARRIER_OPS,
    INPLACE_OPS,
    REDUCTION_OPS,
    TRANSPARENT_OPS,
    TRIGGER_OPS,
    NodeRole,
    classify_node,
    is_compute_bound_gemm,
    is_trigger,
    resolve_to_defining_node,
    symint_safe_eq,
    symint_safe_len,
)
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
    # Canonical op sets are defined in ``fuseml.passes.topology`` and
    # re-exported here as class attributes for backward compatibility.
    # Pattern matching uses ``classify_node()`` and ``is_trigger()``
    # from the topology module for structural dispatch.
    _BARRIER_OPS: Set[Callable] = BARRIER_OPS
    _ABSORBABLE_OPS: Set[Callable] = ABSORBABLE_OPS
    _REDUCTION_OPS: Set[Callable] = REDUCTION_OPS

    def _find_fusion_groups(self) -> List[FusionGroup]:
        """Identify contiguous sequences of fusible memory-bound nodes.

        Walks ``self.graph_module.graph`` in topological order.  When a
        trigger node is encountered (classified by
        :func:`~fuseml.passes.topology.is_trigger`), a new
        :class:`FusionGroup` is initialized and the algorithm greedily
        absorbs downstream nodes based on their
        :class:`~fuseml.passes.topology.NodeRole`:

        * **ABSORBABLE** — pointwise ops absorbed unconditionally.
        * **REDUCTION** — absorbed as the final op (terminates group).
        * **TRANSPARENT** — view/metadata ops absorbed if shape-preserving.
        * **INPLACE** — absorbed only when alias-safe.
        * **BARRIER / UNKNOWN** — halt absorption.

        Classification relies exclusively on ``node.target`` (ATen op
        identity) and structural connectivity — never on ``node.name``
        or placeholder naming conventions introduced by AOT Autograd.

        Only groups with at least one absorbed node (i.e. len > 1) are
        returned, since a lone trigger offers no fusion benefit.

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
            # --- Trigger detection via structural classification -----------
            if not is_trigger(node):
                continue
            if node in consumed:
                continue

            # --- Compute-bound GEMM check --------------------------------
            # For large matmuls where compute throughput (not memory
            # bandwidth) is the bottleneck, cuBLAS significantly
            # outperforms custom Triton GEMM kernels.  Skip fusion so
            # eager cuBLAS handles the matmul and the elementwise
            # post-ops run as separate (cheap) CUDA kernels.
            if self._is_compute_bound_trigger(node):
                logger.debug(
                    "Skipping compute-bound GEMM trigger %s — "
                    "cuBLAS will outperform custom Triton kernel.",
                    node.name,
                )
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

                # Classify the successor purely by its ATen target.
                role = classify_node(successor)

                # Non-call_function nodes (placeholder, output, get_attr)
                # are classified as UNKNOWN — halt absorption.
                if role is NodeRole.UNKNOWN:
                    break

                # --- BARRIER — halt absorption ----------------------------
                if role is NodeRole.BARRIER:
                    break

                # --- REDUCTION — absorb and terminate the group -----------
                if role is NodeRole.REDUCTION:
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

                # --- TRANSPARENT — view/metadata penetration --------------
                if role is NodeRole.TRANSPARENT:
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

                # --- INPLACE — conditionally absorbed with alias guard ----
                if role is NodeRole.INPLACE:
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
                    logger.debug(
                        "Aborting absorption at in-place op %s — "
                        "mutated tensor is aliased outside the group.",
                        successor.name,
                    )
                    break

                # --- ABSORBABLE — unconditional absorption ----------------
                if role is NodeRole.ABSORBABLE:
                    group.fused_nodes.append(successor)
                    group.output_node = successor
                    consumed.add(successor)
                    group_node_set.add(successor)
                    self._collect_external_inputs(
                        successor, group_node_set, group,
                    )
                    current = successor
                    continue

                # Unrecognised role — conservative halt.
                break

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
                # Snapshot node args so the epilogue can inspect them
                # after graph surgery / DCE has erased the original nodes
                # (FX nullifies node.args on erase_node).
                for n in group.all_nodes:
                    group.node_args_snapshot[n.name] = tuple(n.args)
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
    def _is_compute_bound_trigger(node: torch.fx.Node) -> bool:
        """Return ``True`` if the addmm trigger node has compute-bound dims.

        Extracts M, N, K from the node's tensor metadata (populated by
        ShapeProp) and delegates to
        :func:`~fuseml.passes.topology.is_compute_bound_gemm`.

        Before calling the cost model, counts the number of downstream
        absorbable ops (elementwise post-ops like gelu, add, relu) to
        estimate the total HBM savings from fusion — each fused op
        eliminates one M×N read+write round-trip.

        For ``aten.addmm(bias, input, weight)``:
        - ``input`` has shape (M, K)
        - ``weight`` has shape (K, N)

        Returns ``False`` (conservative — proceed with fusion) when
        metadata is unavailable.
        """
        # addmm args: (bias, input, weight)
        if len(node.args) < 3:
            return False

        input_node = node.args[1]
        weight_node = node.args[2]

        # Try tensor_meta first, then val (FakeTensor).
        def _get_shape(n: Any) -> Tuple | None:
            if not isinstance(n, torch.fx.Node):
                return None
            meta = n.meta.get("tensor_meta")
            if meta is not None and hasattr(meta, "shape"):
                return meta.shape
            val = n.meta.get("val")
            if val is not None and hasattr(val, "shape"):
                return val.shape
            return None

        input_shape = _get_shape(input_node)
        weight_shape = _get_shape(weight_node)

        if input_shape is None or weight_shape is None:
            return False
        if len(input_shape) < 2 or len(weight_shape) < 2:
            return False

        try:
            M = int(input_shape[-2])
            K = int(input_shape[-1])
            N = int(weight_shape[-1])
        except (TypeError, ValueError):
            return False

        # Determine dtype from the output node's metadata.
        out_meta = node.meta.get("tensor_meta")
        dtype = out_meta.dtype if out_meta is not None and hasattr(out_meta, "dtype") else torch.float32

        # Count downstream absorbable ops to estimate fusion savings.
        # Each fused epilogue op eliminates one M×N HBM round-trip.
        num_epilogue_ops = 0
        current = node
        while len(current.users) == 1:
            successor = next(iter(current.users))
            role = classify_node(successor)
            if role in (NodeRole.ABSORBABLE, NodeRole.INPLACE):
                num_epilogue_ops += 1
                current = successor
            elif role is NodeRole.REDUCTION:
                num_epilogue_ops += 1
                break
            else:
                break

        return is_compute_bound_gemm(M, N, K, dtype, num_epilogue_ops)

    @staticmethod
    def _collect_external_inputs(
        node: torch.fx.Node,
        group_node_set: Set[torch.fx.Node],
        group: FusionGroup,
    ) -> None:
        """Add *node*'s args to ``group.inputs`` if produced outside the group.

        Each argument is resolved to its closest data-producing defining
        node via :func:`~fuseml.passes.topology.resolve_to_defining_node`,
        tracing through any transparent view/metadata ops inserted by
        AOT Autograd.  This ensures the input list references the actual
        computation that produced the data, not an intermediate reshape
        artifact — making the FusionGroup's dependency graph robust
        against tracing-level variability.

        The resolved node is only used if it is *also* outside the group;
        if resolution lands on a node already inside the group, we fall
        back to the original arg to preserve correct wiring.
        """
        for arg in node.args:
            if not isinstance(arg, torch.fx.Node):
                continue
            if arg in group_node_set:
                continue

            # Resolve through transparent ops to the defining computation.
            resolved = resolve_to_defining_node(arg)

            # Use the resolved node if it's still external; otherwise
            # fall back to the original arg (the transparent op itself
            # may be needed for correct shape wiring).
            target_input = resolved if resolved not in group_node_set else arg

            if target_input not in group.inputs:
                group.inputs.append(target_input)

    @staticmethod
    def _is_shape_preserving_2d(node: torch.fx.Node) -> bool:
        """Return ``True`` if *node* is a view op that preserves 2-D shape.

        The Triton kernel stores its GEMM accumulator to a 2-D ``[M, N]``
        output.  A view op can only be absorbed transparently if its output
        shape matches the GEMM output dimensions.  Shape-changing views
        (different rank or different M/N) would cause a store mismatch and
        are conservatively rejected.

        Shape comparisons use :func:`~fuseml.passes.topology.symint_safe_eq`
        to handle ``torch.SymInt`` dimensions introduced by Dynamo without
        raising ``GuardOnDataDependentSymNode`` errors.  When a dimension
        cannot be concretely resolved, the comparison conservatively returns
        ``False``.

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
        # Use SymInt-safe comparison to handle symbolic shapes from Dynamo.
        node_rank = symint_safe_len(node_meta.shape)
        pred_rank = symint_safe_len(pred_meta.shape)
        if node_rank != 2 or pred_rank != 2:
            return False

        return (
            symint_safe_eq(node_meta.shape[0], pred_meta.shape[0])
            and symint_safe_eq(node_meta.shape[1], pred_meta.shape[1])
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
    # Phase 2 — Rewrite the graph (SSA-preserving)
    # ------------------------------------------------------------------
    def _apply_surgery(self, groups: List[FusionGroup]) -> None:
        """Replace each *FusionGroup* in the graph with a fused kernel call.

        For every group, this method:

        1. **Validates insertion topology** — ensures all external inputs
           precede the insertion point, preventing cyclic dependencies
           when residual connections are topologically parallel to the
           GEMM base node.
        2. **Inserts** a ``call_function`` node targeting
           :func:`fuseml_fused_kernel_placeholder` immediately after the
           group's output node, wired to the group's external ``inputs``.
        3. **Rewires consumers** — for single-output groups, a direct
           ``replace_all_uses_with``; for groups with
           ``intermediate_outputs`` (escape nodes), inserts
           ``operator.getitem`` nodes to decompose the multi-output
           tuple and selectively rewires each intermediate node's
           external consumers to the corresponding ``getitem``.
        4. **Patches cross-group references** so that later groups whose
           inputs referenced a now-replaced output node point to the
           correct replacement.
        5. **Eliminates dead code** and **validates acyclicity** via
           iterative DFS before calling ``graph.lint()`` and
           ``recompile()``.

        Parameters
        ----------
        groups : list[FusionGroup]
            Fusion groups produced by :meth:`_find_fusion_groups`.
        """
        graph = self.graph_module.graph

        # Build topo-index once before the surgery loop — avoids
        # O(groups * nodes) repeated dict construction.
        topo_index: Dict[torch.fx.Node, int] = {
            n: i for i, n in enumerate(graph.nodes)
        }

        for group in groups:
            # --- Phase 0: Validate insertion topology ---------------------
            self._validate_insertion_topology(group, topo_index)

            # --- Phase 1: Insert placeholder node after the output --------
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

            # --- Phase 2: Rewire downstream consumers ---------------------
            rewire_map = self._rewire_consumers(graph, group, new_fused_node)

            # --- Phase 3: Patch remaining groups' input lists -------------
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
                    replacement = rewire_map.get(inp)
                    if replacement is not None:
                        other_group.inputs[i] = replacement

            # --- Phase 4: Dangling output guard ---------------------------
            # If the fusion group is the final computation before the
            # graph's output node, verify the output node's args now
            # reference the replacement (not the dead original).
            for graph_node in graph.nodes:
                if graph_node.op == "output":
                    def _patch_arg(arg, _rmap=rewire_map):
                        replacement = _rmap.get(arg)
                        if replacement is not None:
                            logger.warning(
                                "Dangling output guard: output node still "
                                "references dead node %s — patching.",
                                arg.name,
                            )
                            return replacement
                        if isinstance(arg, (tuple, list)):
                            patched = [_patch_arg(a, _rmap) for a in arg]
                            return type(arg)(patched)
                        return arg

                    new_args = tuple(_patch_arg(a) for a in graph_node.args)
                    if new_args != graph_node.args:
                        graph_node.args = new_args
                    break

        # --- Cleanup: dead code elimination ----------------------------------
        # Single reverse sweep removes orphaned get_attr and call_function
        # nodes (e.g. weight get_attrs, aten.t transposes) that standard
        # DCE may miss.  Reverse topological order ensures dependents are
        # erased before their dependencies.
        for n in reversed(list(graph.nodes)):
            if len(n.users) == 0 and n.op in ("get_attr", "call_function"):
                if n.op == "call_function" and n.target is fuseml_fused_kernel_placeholder:
                    continue  # keep newly inserted placeholders
                logger.debug("Removing orphaned %s node: %s", n.op, n.name)
                graph.erase_node(n)

        # Final DCE catches any cascading orphans from the sweep above.
        graph.eliminate_dead_code()

        # --- Post-surgery validation --------------------------------------
        self._validate_acyclicity(graph)
        graph.lint()
        self.graph_module.recompile()

        logger.info(
            "Graph surgery complete — %d group(s) replaced with placeholder kernels.",
            len(groups),
        )

    # ------------------------------------------------------------------
    # Surgery helpers — consumer rewiring
    # ------------------------------------------------------------------
    def _rewire_consumers(
        self,
        graph: torch.fx.Graph,
        group: FusionGroup,
        new_fused_node: torch.fx.Node,
    ) -> Dict[torch.fx.Node, torch.fx.Node]:
        """Rewire downstream consumers of fused nodes to the new placeholder.

        **Single-output** (no ``intermediate_outputs``):
        Direct ``replace_all_uses_with`` from the group's ``output_node``
        to *new_fused_node*.

        **Multi-output** (``intermediate_outputs`` present):
        The Triton kernel must write intermediate activations back to HBM
        so that external consumers (e.g. Autograd's backward pass) can
        read them.  The FX graph models this as a tuple return:

        * ``new_fused_node`` → returns ``(primary, inter_0, inter_1, …)``
        * ``getitem(new_fused_node, 0)`` → primary output
        * ``getitem(new_fused_node, 1)`` → first intermediate, etc.

        External consumers of each intermediate node are selectively
        rewired to the corresponding ``getitem`` node using
        ``replace_input_with`` (not ``replace_all_uses_with``) to avoid
        disturbing internal group wiring that DCE will clean up.

        Returns
        -------
        dict[torch.fx.Node, torch.fx.Node]
            Mapping from each replaced original node to its replacement.
            Used by the caller to patch cross-group references and the
            dangling-output guard.
        """
        if not group.intermediate_outputs:
            # --- Single-output: simple replacement ------------------------
            group.output_node.replace_all_uses_with(
                new_fused_node,
                propagate_meta=False,
            )
            return {group.output_node: new_fused_node}

        # --- Multi-output: getitem decomposition --------------------------
        prev_node = new_fused_node

        # Primary output (index 0) — replaces group.output_node.
        with graph.inserting_after(prev_node):
            primary_getitem = graph.call_function(
                _operator.getitem, args=(new_fused_node, 0),
            )
        if "tensor_meta" in group.output_node.meta:
            primary_getitem.meta["tensor_meta"] = (
                group.output_node.meta["tensor_meta"]
            )
        prev_node = primary_getitem

        rewire_map: Dict[torch.fx.Node, torch.fx.Node] = {
            group.output_node: primary_getitem,
        }

        # Intermediate outputs (indices 1, 2, …).
        for idx, intermediate_node in enumerate(group.intermediate_outputs):
            with graph.inserting_after(prev_node):
                inter_getitem = graph.call_function(
                    _operator.getitem,
                    args=(new_fused_node, idx + 1),
                )
            if "tensor_meta" in intermediate_node.meta:
                inter_getitem.meta["tensor_meta"] = (
                    intermediate_node.meta["tensor_meta"]
                )
            rewire_map[intermediate_node] = inter_getitem
            prev_node = inter_getitem

        # Rewire output_node's ALL consumers → primary_getitem.
        # (getitem nodes reference new_fused_node, not output_node,
        # so replace_all_uses_with cannot create self-references.)
        group.output_node.replace_all_uses_with(
            primary_getitem, propagate_meta=False,
        )

        # Selectively rewire each intermediate node's EXTERNAL consumers
        # to the corresponding getitem.  Internal group wiring is left
        # intact — those nodes will be removed by DCE.
        group_set = set(group.all_nodes)
        for intermediate_node, getitem_node in rewire_map.items():
            if intermediate_node is group.output_node:
                continue  # already handled above
            for user in list(intermediate_node.users):
                if user not in group_set and user is not getitem_node:
                    user.replace_input_with(intermediate_node, getitem_node)

        # Attach multi-output metadata for downstream codegen.
        new_fused_node.meta["num_outputs"] = (
            1 + len(group.intermediate_outputs)
        )
        new_fused_node.meta["intermediate_output_names"] = [
            n.name for n in group.intermediate_outputs
        ]

        logger.debug(
            "Multi-output rewiring: %d getitem node(s) for group %s",
            1 + len(group.intermediate_outputs),
            group,
        )
        return rewire_map

    # ------------------------------------------------------------------
    # Surgery helpers — topology validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_insertion_topology(
        group: FusionGroup,
        topo_index: Dict[torch.fx.Node, int] | None = None,
    ) -> None:
        """Verify all group inputs precede the output_node topologically.

        The fused placeholder is inserted immediately after the group's
        ``output_node``.  If any external input appears *after* the
        output_node in the FX graph's node list (which maintains
        topological order), the insertion would create a use-before-def
        violation — breaking the SSA property.

        AOT Autograd flattening can separate secondary inputs (e.g. the
        residual connection feeding ``aten.add.Tensor``) from the main
        computational trunk.  In a valid FX graph these inputs are always
        topologically *before* the nodes that consume them, but this
        guard catches any graph corruption that would silently produce
        incorrect rewiring.

        Parameters
        ----------
        group : FusionGroup
            The fusion group to validate.
        topo_index : dict or None
            Pre-computed ``{node: position}`` mapping.  When ``None``,
            built on the fly (backward-compatible but slower).

        Raises
        ------
        RuntimeError
            If an input appears after the insertion point.
        """
        if topo_index is None:
            graph = group.base_node.graph
            topo_index = {n: i for i, n in enumerate(graph.nodes)}

        output_pos = topo_index.get(group.output_node)
        if output_pos is None:
            return  # output_node missing from graph — caller will fail later

        for inp in group.inputs:
            inp_pos = topo_index.get(inp)
            if inp_pos is not None and inp_pos > output_pos:
                raise RuntimeError(
                    f"SSA violation: input '{inp.name}' (topo position "
                    f"{inp_pos}) appears after output_node "
                    f"'{group.output_node.name}' (topo position "
                    f"{output_pos}).  The fused placeholder would create "
                    f"a cyclic dependency.  This indicates a malformed FX "
                    f"graph or incorrect FusionGroup construction."
                )

    @staticmethod
    def _validate_acyclicity(graph: torch.fx.Graph) -> None:
        """Verify the graph contains no cycles after surgery.

        Uses iterative DFS with three-colour marking
        (WHITE → GRAY → BLACK) to detect back edges.  A back edge
        exists when a GRAY node (currently on the DFS stack) is
        encountered as a successor, proving a cycle in the directed
        graph.

        Raises
        ------
        RuntimeError
            If a cycle is detected.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[torch.fx.Node, int] = {n: WHITE for n in graph.nodes}

        for start in graph.nodes:
            if color[start] != WHITE:
                continue

            # Iterative DFS avoids stack overflow on deep graphs.
            stack: List[Tuple[torch.fx.Node, bool]] = [(start, False)]
            while stack:
                node, post_visit = stack.pop()

                if post_visit:
                    color[node] = BLACK
                    continue

                if color[node] == BLACK:
                    continue
                if color[node] == GRAY:
                    # Revisited via another path — safe to finalize.
                    color[node] = BLACK
                    continue

                color[node] = GRAY
                stack.append((node, True))  # schedule post-visit

                for user in node.users:
                    if color[user] == GRAY:
                        raise RuntimeError(
                            f"Cycle detected after graph surgery: "
                            f"'{node.name}' → '{user.name}' forms a "
                            f"back edge.  This indicates incorrect "
                            f"node rewiring during fusion."
                        )
                    if color[user] == WHITE:
                        stack.append((user, False))

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
        # --- Shape propagation (skip when torch.compile already provided metadata) ---
        # torch.compile's abstract-interpretation pass populates node.meta["val"]
        # with FakeTensors.  When this metadata is present, ShapeProp is
        # redundant — all downstream code (compute-bound checks, output
        # metadata extraction) already falls back to meta["val"].
        if example_inputs is not None:
            has_faketensor_meta = any(
                n.meta.get("val") is not None
                for n in self.graph_module.graph.nodes
                if n.op == "call_function"
            )
            if has_faketensor_meta:
                logger.debug(
                    "FakeTensor metadata already present — skipping ShapeProp."
                )
            else:
                logger.debug(
                    "Running ShapeProp with %d example input(s).",
                    len(example_inputs),
                )
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
