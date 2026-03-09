"""Tests for the control flow validation pass.

Covers: clean graphs passing validation, Tier 1 FX graph detection
(.item(), higher-order ops, torch.where warnings), Tier 2 source AST
detection, whitelisting, ControlFlowError structure, and compiler
fallback integration.

Run with:
    pytest tests/test_control_flow_validation.py -v
    pytest tests/ -m control_flow -v
"""

from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from fuseml.passes.control_flow_validation import (
    ControlFlowError,
    ControlFlowViolation,
    _check_graph_nodes,
    _check_source_ast,
    validate_graph_control_flow,
)
from fuseml.compiler import FuseMLCompiler

from conftest import trace_no_grad


# ---------------------------------------------------------------------------
# Helpers — small modules that exercise specific patterns
# ---------------------------------------------------------------------------

class CleanLinearReLU(nn.Module):
    """No control flow — should always pass validation."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        return torch.relu(self.linear(x))


class ItemBranchModule(nn.Module):
    """Uses .item() to drive an if statement — classic untraceable pattern."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        out = self.linear(x)
        if out.sum().item() > 0:
            return torch.relu(out)
        return out


class SumBranchModule(nn.Module):
    """Uses tensor.sum() > 0 in an if condition — Tier 2 detectable."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        out = self.linear(x)
        if out.sum() > 0:
            return torch.relu(out)
        return out


class TrainingGuardModule(nn.Module):
    """Uses self.training — a plain bool, should be whitelisted."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        out = self.linear(x)
        if self.training:
            out = self.drop(out)
        return out


class WhereModule(nn.Module):
    """Uses torch.where with a comparison — should produce a warning."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        out = self.linear(x)
        return torch.where(out > 0, out, torch.zeros_like(out))


class AnyItemModule(nn.Module):
    """Calls .any().item() — boolean reduction + scalar extraction."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if (x > 0).any().item():
            return x * 2
        return x


# ---------------------------------------------------------------------------
# ControlFlowViolation dataclass
# ---------------------------------------------------------------------------

@pytest.mark.control_flow
class TestControlFlowViolation:
    def test_str_includes_severity_and_name(self):
        v = ControlFlowViolation(
            node_name="my_node", description="bad branch", severity="error"
        )
        s = str(v)
        assert "ERROR" in s
        assert "my_node" in s
        assert "bad branch" in s

    def test_warning_severity_tag(self):
        v = ControlFlowViolation(
            node_name="w", description="maybe bad", severity="warning"
        )
        assert "WARNING" in str(v)


# ---------------------------------------------------------------------------
# ControlFlowError
# ---------------------------------------------------------------------------

@pytest.mark.control_flow
class TestControlFlowError:
    def test_carries_violations(self):
        vs = [
            ControlFlowViolation("a", "desc a"),
            ControlFlowViolation("b", "desc b"),
        ]
        err = ControlFlowError(violations=vs)
        assert len(err.violations) == 2
        assert err.warnings == []

    def test_carries_warnings(self):
        vs = [ControlFlowViolation("a", "desc a")]
        ws = [ControlFlowViolation("w", "warn", severity="warning")]
        err = ControlFlowError(violations=vs, warnings=ws)
        assert len(err.warnings) == 1

    def test_message_includes_count(self):
        vs = [ControlFlowViolation("a", "desc")]
        err = ControlFlowError(violations=vs)
        assert "1 violation(s)" in str(err)

    def test_is_exception(self):
        err = ControlFlowError(violations=[ControlFlowViolation("a", "d")])
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# Tier 1 — FX graph node inspection
# ---------------------------------------------------------------------------

@pytest.mark.control_flow
class TestTier1CleanGraph:
    """Standard Linear+ReLU graph has no control flow indicators."""

    def test_clean_graph_passes(self):
        model = CleanLinearReLU()
        gm = trace_no_grad(model, torch.randn(2, 64))
        errors, warnings = _check_graph_nodes(gm)
        assert errors == []
        assert warnings == []

    def test_validate_does_not_raise_on_clean_graph(self):
        model = CleanLinearReLU()
        gm = trace_no_grad(model, torch.randn(2, 64))
        validate_graph_control_flow(gm)


@pytest.mark.control_flow
class TestTier1ItemDetection:
    """Graphs containing .item() must be flagged as errors.

    Note: make_fx itself refuses to trace modules that call .item() on
    proxy tensors (raising RuntimeError for data-dependent ops).  So we
    cannot produce a naturally-traced graph with .item() nodes — we
    verify that tracing correctly rejects such modules, then rely on the
    synthetic-injection tests in TestTier1SyntheticItem for Tier 1 coverage.
    """

    def test_make_fx_rejects_item_module(self):
        """make_fx raises RuntimeError when .item() is encountered."""
        model = ItemBranchModule()
        with pytest.raises(RuntimeError, match="data-dependent"):
            trace_no_grad(model, torch.randn(2, 64))


@pytest.mark.control_flow
class TestTier1WhereWarning:
    """torch.where with a comparison condition produces a warning, not error."""

    def test_where_produces_warning(self):
        model = WhereModule()
        gm = trace_no_grad(model, torch.randn(2, 64))

        where_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and "where" in str(n.target)
        ]
        if not where_nodes:
            pytest.skip("Tracer decomposed torch.where")

        errors, warnings = _check_graph_nodes(gm)
        assert errors == [], "torch.where alone should not cause an error"
        assert len(warnings) >= 1
        assert warnings[0].severity == "warning"

    def test_where_does_not_raise(self):
        model = WhereModule()
        gm = trace_no_grad(model, torch.randn(2, 64))
        validate_graph_control_flow(gm)


@pytest.mark.control_flow
class TestTier1BooleanReductionScalar:
    """Boolean reduction (.any()) feeding .item() is flagged.

    Like TestTier1ItemDetection, make_fx refuses to trace .any().item()
    patterns.  We verify the rejection, then test the node-level
    detection via synthetic graph injection.
    """

    def test_make_fx_rejects_any_item_module(self):
        model = AnyItemModule()
        with pytest.raises(RuntimeError, match="data-dependent"):
            trace_no_grad(model, torch.randn(2, 64))

    def test_synthetic_any_then_item_detected(self):
        """Inject aten.any + call_method('item') and verify both are flagged."""
        model = CleanLinearReLU()
        gm = trace_no_grad(model, torch.randn(2, 64))

        graph = gm.graph
        output_node = next(n for n in graph.nodes if n.op == "output")
        call_fn_node = next(
            n for n in graph.nodes if n.op == "call_function"
        )
        with graph.inserting_before(output_node):
            any_node = graph.call_function(
                torch.ops.aten.any.default, args=(call_fn_node,)
            )
            graph.call_method("item", args=(any_node,))
        gm.recompile()

        errors, _ = _check_graph_nodes(gm)
        assert len(errors) >= 1
        descs = " ".join(v.description for v in errors)
        assert "item" in descs.lower()


# ---------------------------------------------------------------------------
# Tier 1 — synthetic graph with injected .item() node
# ---------------------------------------------------------------------------

@pytest.mark.control_flow
class TestTier1SyntheticItem:
    """Inject a call_method 'item' node into a graph to guarantee detection."""

    def test_synthetic_item_node_triggers_error(self):
        model = CleanLinearReLU()
        gm = trace_no_grad(model, torch.randn(2, 64))

        graph = gm.graph
        output_node = next(n for n in graph.nodes if n.op == "output")
        with graph.inserting_before(output_node):
            some_node = next(
                n for n in graph.nodes if n.op == "call_function"
            )
            item_node = graph.call_method("item", args=(some_node,))
        gm.recompile()

        errors, _ = _check_graph_nodes(gm)
        assert any(v.node_name == item_node.name for v in errors)

    def test_synthetic_item_raises_via_validate(self):
        model = CleanLinearReLU()
        gm = trace_no_grad(model, torch.randn(2, 64))

        graph = gm.graph
        output_node = next(n for n in graph.nodes if n.op == "output")
        with graph.inserting_before(output_node):
            some_node = next(
                n for n in graph.nodes if n.op == "call_function"
            )
            graph.call_method("item", args=(some_node,))
        gm.recompile()

        with pytest.raises(ControlFlowError):
            validate_graph_control_flow(gm)


# ---------------------------------------------------------------------------
# Tier 2 — Source AST inspection
# ---------------------------------------------------------------------------

@pytest.mark.control_flow
class TestTier2SourceAST:
    """Inspect original callable source for data-dependent branches."""

    def test_item_branch_detected_in_source(self):
        violations = _check_source_ast(ItemBranchModule)
        assert len(violations) >= 1
        assert any("if" in v.description for v in violations)

    def test_sum_branch_detected_in_source(self):
        violations = _check_source_ast(SumBranchModule)
        assert len(violations) >= 1
        assert any("if" in v.description for v in violations)

    def test_clean_module_no_violations(self):
        violations = _check_source_ast(CleanLinearReLU)
        assert violations == []

    def test_training_guard_whitelisted(self):
        violations = _check_source_ast(TrainingGuardModule)
        training_violations = [
            v for v in violations
            if "training" in v.description.lower()
        ]
        assert training_violations == [], (
            "self.training should be whitelisted — it is a plain bool"
        )

    def test_source_unavailable_returns_empty(self):
        violations = _check_source_ast(torch.relu)
        assert violations == []

    def test_validate_with_original_callable_raises(self):
        model = CleanLinearReLU()
        gm = trace_no_grad(model, torch.randn(2, 64))
        with pytest.raises(ControlFlowError):
            validate_graph_control_flow(gm, original_callable=ItemBranchModule)


# ---------------------------------------------------------------------------
# Compiler integration — eager fallback
# ---------------------------------------------------------------------------

@pytest.mark.control_flow
@pytest.mark.compiler
class TestCompilerFallback:
    """When control flow is detected, the compiler returns gm.forward."""

    def test_clean_model_compiles_normally(self):
        model = CleanLinearReLU()
        gm = trace_no_grad(model, torch.randn(2, 64))
        compiler = FuseMLCompiler()
        result = compiler(gm, [torch.randn(2, 64)])
        assert callable(result)

    def test_item_graph_falls_back_to_forward(self):
        """Graph with an injected .item() node triggers fallback."""
        model = CleanLinearReLU()
        gm = trace_no_grad(model, torch.randn(2, 64))

        graph = gm.graph
        output_node = next(n for n in graph.nodes if n.op == "output")
        with graph.inserting_before(output_node):
            some_node = next(
                n for n in graph.nodes if n.op == "call_function"
            )
            graph.call_method("item", args=(some_node,))
        gm.recompile()

        compiler = FuseMLCompiler()
        result = compiler(gm, [torch.randn(2, 64)])
        assert callable(result), "Compiler should return a callable on fallback"
        # gm.forward is a bound method — each access creates a new object,
        # so we compare the underlying __func__ and __self__ instead.
        assert result.__func__ is gm.forward.__func__
        assert result.__self__ is gm
