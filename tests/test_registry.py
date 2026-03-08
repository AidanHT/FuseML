"""Tests for SupportedOpsRegistry and build_default_registry.

Run with:
    pytest tests/test_registry.py -v
"""

from __future__ import annotations

import pytest
import torch

from fuseml import SupportedOpsRegistry, build_default_registry


# ---------------------------------------------------------------------------
# Tests — default registry contents
# ---------------------------------------------------------------------------

class TestBuildDefaultRegistry:
    """Verify build_default_registry ships all expected ops."""

    @pytest.fixture()
    def registry(self):
        return build_default_registry()

    def test_addmm_registered(self, registry):
        assert torch.ops.aten.addmm.default in registry

    def test_addmm_category(self, registry):
        assert registry.get_category(torch.ops.aten.addmm.default) == "linear"

    @pytest.mark.parametrize("op", [
        torch.ops.aten.relu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    ])
    def test_elementwise_ops_registered(self, registry, op):
        assert op in registry
        assert registry.get_category(op) == "elementwise"

    def test_total_count(self, registry):
        assert len(registry) == 6  # 1 linear + 5 elementwise


# ---------------------------------------------------------------------------
# Tests — SupportedOpsRegistry API
# ---------------------------------------------------------------------------

class TestSupportedOpsRegistry:

    def test_register_and_query(self):
        r = SupportedOpsRegistry()
        r.register(torch.ops.aten.relu.default, "elementwise")
        assert r.is_supported(torch.ops.aten.relu.default)
        assert r.get_category(torch.ops.aten.relu.default) == "elementwise"

    def test_unregister(self):
        r = SupportedOpsRegistry()
        r.register(torch.ops.aten.relu.default)
        r.unregister(torch.ops.aten.relu.default)
        assert not r.is_supported(torch.ops.aten.relu.default)

    def test_unregister_absent_is_noop(self):
        r = SupportedOpsRegistry()
        r.unregister(torch.ops.aten.relu.default)  # should not raise

    def test_contains(self):
        r = SupportedOpsRegistry()
        r.register(torch.ops.aten.relu.default)
        assert torch.ops.aten.relu.default in r
        assert torch.ops.aten.gelu.default not in r

    def test_register_many(self):
        r = SupportedOpsRegistry()
        ops = [torch.ops.aten.relu.default, torch.ops.aten.gelu.default]
        r.register_many(ops, "elementwise")
        assert len(r) == 2
        for op in ops:
            assert op in r
