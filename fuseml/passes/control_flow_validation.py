"""Pre-compilation validation pass for data-dependent control flow.

torch.fx symbolic tracing evaluates Python control flow at trace time,
permanently baking in whichever branch the dummy inputs happen to take.
Compiling that single-path graph into a Triton kernel produces silently
incorrect results for inputs that would have taken a different branch.

This module provides :func:`validate_graph_control_flow`, which inspects
a captured FX graph (and optionally the original callable's source) for
indicators of untraceable control flow.  When violations are found it
raises :class:`ControlFlowError` so the compiler can fall back to
standard eager-mode PyTorch execution.

Detection is split into two tiers:

* **Tier 1 — FX graph node inspection** (always runs): flags higher-order
  control-flow ops, scalar-extraction methods (``.item()``, ``.tolist()``,
  ``.bool()``), and ``torch.where`` combined with comparison ops.
* **Tier 2 — Source AST inspection** (when the original callable is
  available): parses the ``forward()`` source with :mod:`ast` and detects
  ``if`` / ``while`` / ``for`` statements whose conditions depend on
  tensor-derived values.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Sequence

import torch

from fuseml._logging import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ControlFlowViolation:
    """A single detected indicator of untraceable control flow.

    Attributes
    ----------
    node_name : str
        The FX node name or AST location that triggered the finding.
    description : str
        Human-readable explanation of why this is problematic.
    severity : ``"error"`` | ``"warning"``
        ``"error"`` triggers :class:`ControlFlowError` and forces eager
        fallback.  ``"warning"`` is logged but does not abort compilation.
    """

    node_name: str
    description: str
    severity: Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        tag = self.severity.upper()
        return f"[{tag}] {self.node_name}: {self.description}"


class ControlFlowError(Exception):
    """Raised when data-dependent control flow is detected in a traced graph.

    Attributes
    ----------
    violations : list[ControlFlowViolation]
        All findings that contributed to the error (severity ``"error"``).
    warnings : list[ControlFlowViolation]
        Advisory findings (severity ``"warning"``) collected alongside.
    """

    def __init__(
        self,
        violations: Sequence[ControlFlowViolation],
        warnings: Sequence[ControlFlowViolation] | None = None,
    ) -> None:
        self.violations: list[ControlFlowViolation] = list(violations)
        self.warnings: list[ControlFlowViolation] = list(warnings or [])
        summary = "; ".join(str(v) for v in self.violations)
        super().__init__(
            f"Data-dependent control flow detected in traced graph — "
            f"{len(self.violations)} violation(s): {summary}"
        )


# ---------------------------------------------------------------------------
# Constants — op / method sets for Tier 1
# ---------------------------------------------------------------------------

_SCALAR_EXTRACTION_METHODS: frozenset[str] = frozenset({
    "item",
    "tolist",
    "bool",
})

_COMPARISON_OPS: frozenset[Callable[..., Any]] = frozenset({
    torch.ops.aten.gt.Scalar,
    torch.ops.aten.gt.Tensor,
    torch.ops.aten.lt.Scalar,
    torch.ops.aten.lt.Tensor,
    torch.ops.aten.ge.Scalar,
    torch.ops.aten.ge.Tensor,
    torch.ops.aten.le.Scalar,
    torch.ops.aten.le.Tensor,
    torch.ops.aten.eq.Scalar,
    torch.ops.aten.eq.Tensor,
    torch.ops.aten.ne.Scalar,
    torch.ops.aten.ne.Tensor,
})

_BOOLEAN_REDUCTION_OPS: frozenset[Callable[..., Any]] = frozenset({
    torch.ops.aten.any.default,
    torch.ops.aten.any.dim,
    torch.ops.aten.all.default,
    torch.ops.aten.all.dim,
})

_WHERE_OPS: frozenset[Callable[..., Any]] = frozenset({
    torch.ops.aten.where.self,
    torch.ops.aten.where.ScalarSelf,
    torch.ops.aten.where.ScalarOther,
})

# Higher-order ops that represent explicit control flow in the graph.
_HIGHER_ORDER_CF_NAMES: frozenset[str] = frozenset({
    "cond",
    "while_loop",
    "map_impl",
})


# ---------------------------------------------------------------------------
# Constants — attribute names for Tier 2 AST inspection
# ---------------------------------------------------------------------------

_TENSOR_REDUCTION_ATTRS: frozenset[str] = frozenset({
    "item", "tolist", "bool",
    "sum", "mean", "max", "min", "prod",
    "any", "all",
    "argmax", "argmin",
    "norm", "count_nonzero",
})

_SAFE_CONDITION_ATTRS: frozenset[str] = frozenset({
    "training",
})


# ---------------------------------------------------------------------------
# Tier 1 — FX graph node inspection
# ---------------------------------------------------------------------------

def _check_graph_nodes(
    gm: torch.fx.GraphModule,
) -> tuple[list[ControlFlowViolation], list[ControlFlowViolation]]:
    """Walk *gm*'s FX graph and flag control-flow indicators.

    Returns (errors, warnings).
    """
    errors: list[ControlFlowViolation] = []
    warnings: list[ControlFlowViolation] = []

    comparison_nodes: set[int] = set()

    for node in gm.graph.nodes:
        # --- Higher-order control flow ops ---
        if node.op == "call_function":
            target_name = getattr(node.target, "__name__", str(node.target))
            if target_name in _HIGHER_ORDER_CF_NAMES:
                errors.append(ControlFlowViolation(
                    node_name=node.name,
                    description=(
                        f"Higher-order control flow op '{target_name}' "
                        f"detected — graph contains an explicit conditional "
                        f"branch that cannot be statically compiled."
                    ),
                ))
                continue

            # Track comparison ops for the torch.where heuristic.
            if node.target in _COMPARISON_OPS:
                comparison_nodes.add(id(node))

            # Boolean reductions feeding scalar extraction are suspicious.
            if node.target in _BOOLEAN_REDUCTION_OPS:
                for user in node.users:
                    if (
                        user.op == "call_method"
                        and user.target in _SCALAR_EXTRACTION_METHODS
                    ):
                        errors.append(ControlFlowViolation(
                            node_name=node.name,
                            description=(
                                f"Boolean reduction '{target_name}' feeds "
                                f"scalar extraction method '.{user.target}()' "
                                f"— likely used in data-dependent control flow."
                            ),
                        ))

            # torch.where with a comparison input → flattened if/else.
            if node.target in _WHERE_OPS:
                cond_arg = node.args[0] if node.args else None
                if (
                    isinstance(cond_arg, torch.fx.Node)
                    and cond_arg.op == "call_function"
                    and cond_arg.target in _COMPARISON_OPS
                ):
                    warnings.append(ControlFlowViolation(
                        node_name=node.name,
                        description=(
                            "torch.where with a comparison condition detected "
                            "— may indicate flattened data-dependent control "
                            "flow.  Verify the original source does not use "
                            "if/else with tensor conditions."
                        ),
                        severity="warning",
                    ))

        # --- Scalar extraction methods ---
        if node.op == "call_method" and node.target in _SCALAR_EXTRACTION_METHODS:
            already_flagged = any(
                v.node_name == node.name for v in errors
            )
            if not already_flagged:
                errors.append(ControlFlowViolation(
                    node_name=node.name,
                    description=(
                        f"Tensor scalar extraction method '.{node.target}()' "
                        f"found in the traced graph — the extracted value was "
                        f"likely used in data-dependent Python control flow "
                        f"that has been baked into a single path."
                    ),
                ))

    return errors, warnings


# ---------------------------------------------------------------------------
# Tier 2 — Source AST inspection
# ---------------------------------------------------------------------------

class _ControlFlowVisitor(ast.NodeVisitor):
    """AST visitor that detects data-dependent control flow statements."""

    def __init__(self) -> None:
        self.violations: list[ControlFlowViolation] = []

    # -- helpers ----------------------------------------------------------

    def _involves_tensor_op(self, node: ast.AST) -> bool:
        """Return True if *node* references a tensor reduction / extraction."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in _TENSOR_REDUCTION_ATTRS:
                    if not self._is_safe_receiver(func.value):
                        return True
            for arg in node.args:
                if self._involves_tensor_op(arg):
                    return True
            for kw in node.keywords:
                if self._involves_tensor_op(kw.value):
                    return True
        elif isinstance(node, ast.Compare):
            if self._involves_tensor_op(node.left):
                return True
            for comp in node.comparators:
                if self._involves_tensor_op(comp):
                    return True
        elif isinstance(node, ast.BoolOp):
            return any(self._involves_tensor_op(v) for v in node.values)
        elif isinstance(node, ast.UnaryOp):
            return self._involves_tensor_op(node.operand)
        elif isinstance(node, ast.BinOp):
            return (
                self._involves_tensor_op(node.left)
                or self._involves_tensor_op(node.right)
            )
        elif isinstance(node, ast.Attribute):
            if node.attr in _TENSOR_REDUCTION_ATTRS:
                if not self._is_safe_receiver(node.value):
                    return True
        return False

    @staticmethod
    def _is_safe_receiver(node: ast.AST) -> bool:
        """Return True for attribute access known not to involve tensors."""
        if isinstance(node, ast.Attribute):
            if node.attr in _SAFE_CONDITION_ATTRS:
                return True
        if isinstance(node, ast.Name):
            if node.id == "self":
                return False
        return False

    def _record(self, stmt: ast.stmt, kind: str) -> None:
        lineno = getattr(stmt, "lineno", "?")
        self.violations.append(ControlFlowViolation(
            node_name=f"source:L{lineno}",
            description=(
                f"Data-dependent '{kind}' statement at line {lineno} — "
                f"condition involves tensor-derived value(s) that would "
                f"be baked into a single path during tracing."
            ),
        ))

    # -- visitors ---------------------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        if self._involves_tensor_op(node.test):
            self._record(node, "if")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if self._involves_tensor_op(node.test):
            self._record(node, "while")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if self._involves_tensor_op(node.iter):
            self._record(node, "for")
        self.generic_visit(node)


def _check_source_ast(
    callable_obj: Any,
) -> list[ControlFlowViolation]:
    """Parse the source of *callable_obj* and detect control flow violations.

    Returns an empty list when source is unavailable (e.g. built-in or
    C-extension modules).
    """
    target = callable_obj
    if hasattr(target, "forward") and callable(target.forward):
        target = target.forward

    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return []

    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    visitor = _ControlFlowVisitor()
    visitor.visit(tree)
    return visitor.violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_graph_control_flow(
    gm: torch.fx.GraphModule,
    original_callable: Any | None = None,
) -> None:
    """Validate that *gm* is free of data-dependent control flow indicators.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The traced FX graph to inspect.
    original_callable : module, function, or None
        When provided, the original Python callable (typically an
        ``nn.Module``) whose source code will be inspected via
        :mod:`ast` for Tier 2 detection.

    Raises
    ------
    ControlFlowError
        If any *error*-level violations are found.  Warning-level
        findings are logged but do not raise.
    """
    # Tier 1: graph node inspection
    errors, warnings = _check_graph_nodes(gm)

    # Tier 2: source AST inspection
    if original_callable is not None:
        ast_violations = _check_source_ast(original_callable)
        errors.extend(ast_violations)

    # Log warnings
    for w in warnings:
        logger.warning("Control flow advisory: %s", w)

    if errors:
        raise ControlFlowError(violations=errors, warnings=warnings)

    logger.debug(
        "Control flow validation passed — no data-dependent branches detected."
    )
