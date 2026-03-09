"""Tests for reduction codegen — cross-thread synchronization in the epilogue.

Verifies that reduction operators (sum, max, mean) emit correct Triton code:
- tl.sum / tl.max for tile-local partial reductions
- tl.atomic_add / tl.atomic_max for cross-program accumulation
- 1-D output pointer arithmetic and stride parameters
- No race conditions from blindly treating reductions as element-wise ops

Uses the same lightweight FX node stand-ins as test_kernel_generator.py.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from fuseml.codegen.kernel_generator import (
    ReductionInfo,
    TensorDescriptor,
    TritonKernelGenerator,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight FX node stand-ins
# ---------------------------------------------------------------------------

def _make_node(*, op="call_function", target=None, name="node", args=(), meta=None):
    """Create a minimal object that quacks like ``torch.fx.Node``."""
    ns = SimpleNamespace(op=op, target=target, name=name, args=args)
    ns.meta = meta or {}
    return ns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    return TritonKernelGenerator()


@pytest.fixture
def matmul_inputs():
    """Standard A(128x64) @ B(64x256) inputs."""
    a = TensorDescriptor(name="a", shape=(128, 64), stride=(64, 1), dtype=torch.float32)
    b = TensorDescriptor(name="b", shape=(64, 256), stride=(256, 1), dtype=torch.float32)
    return [a, b]


@pytest.fixture
def output_2d():
    """Standard 2-D output (M=128, N=256)."""
    return TensorDescriptor(name="out", shape=(128, 256), stride=(256, 1), dtype=torch.float32)


@pytest.fixture
def output_reduced_m():
    """1-D output after reducing along N — shape (M=128,)."""
    return TensorDescriptor(name="out", shape=(128,), stride=(1,), dtype=torch.float32)


@pytest.fixture
def output_reduced_n():
    """1-D output after reducing along M — shape (N=256,)."""
    return TensorDescriptor(name="out", shape=(256,), stride=(1,), dtype=torch.float32)


# ---------------------------------------------------------------------------
# ReductionInfo dataclass
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestReductionInfo:
    """Verify the ReductionInfo metadata container."""

    def test_frozen(self):
        ri = ReductionInfo(axis=1, op="sum")
        with pytest.raises(AttributeError):
            ri.axis = 0  # type: ignore[misc]

    def test_defaults(self):
        ri = ReductionInfo(axis=1, op="sum")
        assert ri.keepdim is False

    def test_fields(self):
        ri = ReductionInfo(axis=0, op="max", keepdim=True)
        assert ri.axis == 0
        assert ri.op == "max"
        assert ri.keepdim is True


# ---------------------------------------------------------------------------
# Epilogue: sum reduction
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestEpilogueSum:
    """Sum reduction: tl.sum + tl.atomic_add for cross-thread correctness."""

    def test_sum_emits_tl_sum(self, gen, output_reduced_m):
        relu_node = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(relu_node, [-1], False),
        )
        code = gen.generate_epilogue(
            [relu_node, sum_node], output_descriptor=output_reduced_m,
        )
        assert "tl.sum(acc, axis=1)" in code

    def test_sum_emits_atomic_add(self, gen, output_reduced_m):
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [sum_node], output_descriptor=output_reduced_m,
        )
        assert "tl.atomic_add(out_ptrs," in code

    def test_sum_uses_output_name(self, gen):
        """Atomic store must use the output descriptor's name, not a hardcoded 'out'."""
        out = TensorDescriptor("result", (128,), (1,), torch.float32)
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue([sum_node], output_descriptor=out)
        assert "tl.atomic_add(result_ptrs," in code

    def test_sum_mask_uses_offs_m(self, gen, output_reduced_m):
        """N-reduction: surviving dim is M, so mask should use offs_m < M."""
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [1], False),
        )
        code = gen.generate_epilogue(
            [sum_node], output_descriptor=output_reduced_m,
        )
        assert "mask=offs_m < M" in code

    def test_sum_no_regular_store(self, gen, output_reduced_m):
        """Reduction emits atomic store — regular tl.store must NOT appear."""
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [sum_node], output_descriptor=output_reduced_m,
        )
        # Should have atomic_add but NOT a bare tl.store
        assert "tl.atomic_add" in code
        assert "tl.store(" not in code

    def test_sum_sets_last_reduction(self, gen, output_reduced_m):
        """generate_epilogue must record reduction info on the instance."""
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        gen.generate_epilogue(
            [sum_node], output_descriptor=output_reduced_m,
        )
        assert gen._last_reduction is not None
        assert gen._last_reduction.op == "sum"
        assert gen._last_reduction.axis == 1

    def test_sum_axis_0(self, gen, output_reduced_n):
        """M-reduction: tl.sum(acc, axis=0), mask uses offs_n < N."""
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [0], False),
        )
        code = gen.generate_epilogue(
            [sum_node], output_descriptor=output_reduced_n,
        )
        assert "tl.sum(acc, axis=0)" in code
        assert "mask=offs_n < N" in code

    def test_sum_comment_mentions_two_stage(self, gen, output_reduced_m):
        """Comment must describe the two-stage reduction strategy."""
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [sum_node], output_descriptor=output_reduced_m,
        )
        assert "two-stage" in code.lower() or "block-local" in code.lower()


# ---------------------------------------------------------------------------
# Epilogue: max reduction
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestEpilogueMax:
    """Max reduction: tl.max + tl.atomic_max."""

    def test_max_emits_tl_max(self, gen, output_reduced_m):
        max_node = _make_node(
            target=torch.ops.aten.amax.default,
            name="max_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [max_node], output_descriptor=output_reduced_m,
        )
        assert "tl.max(acc, axis=1)" in code

    def test_max_emits_atomic_max(self, gen, output_reduced_m):
        max_node = _make_node(
            target=torch.ops.aten.amax.default,
            name="max_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [max_node], output_descriptor=output_reduced_m,
        )
        assert "tl.atomic_max(out_ptrs," in code

    def test_max_sets_last_reduction(self, gen, output_reduced_m):
        max_node = _make_node(
            target=torch.ops.aten.amax.default,
            name="max_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        gen.generate_epilogue(
            [max_node], output_descriptor=output_reduced_m,
        )
        assert gen._last_reduction is not None
        assert gen._last_reduction.op == "max"


# ---------------------------------------------------------------------------
# Epilogue: mean reduction
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestEpiloqueMean:
    """Mean reduction: tl.sum + fused reciprocal multiply + tl.atomic_add."""

    def test_mean_emits_tl_sum(self, gen, output_reduced_m):
        mean_node = _make_node(
            target=torch.ops.aten.mean.dim,
            name="mean_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [mean_node], output_descriptor=output_reduced_m,
        )
        # Mean uses tl.sum internally, NOT tl.mean (which doesn't exist in Triton)
        assert "tl.sum(acc, axis=1)" in code

    def test_mean_emits_fused_reciprocal(self, gen, output_reduced_m):
        """Mean epilogue must multiply by 1/dim before atomic_add."""
        mean_node = _make_node(
            target=torch.ops.aten.mean.dim,
            name="mean_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [mean_node], output_descriptor=output_reduced_m,
        )
        # Reciprocal multiply is fused into the epilogue
        assert "1.0 / N" in code or "1.0 / M" in code
        assert "partial_mean" in code

    def test_mean_emits_atomic_add(self, gen, output_reduced_m):
        mean_node = _make_node(
            target=torch.ops.aten.mean.dim,
            name="mean_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [mean_node], output_descriptor=output_reduced_m,
        )
        # atomic_add now receives partial_mean (not partial_sum)
        assert "tl.atomic_add(out_ptrs, partial_mean" in code

    def test_mean_comment_mentions_fused_division(self, gen, output_reduced_m):
        """Reciprocal multiply is fused into epilogue for CUDA Graph safety."""
        mean_node = _make_node(
            target=torch.ops.aten.mean.dim,
            name="mean_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [mean_node], output_descriptor=output_reduced_m,
        )
        assert "fused" in code.lower() or "reciprocal" in code.lower()

    def test_mean_sets_last_reduction(self, gen, output_reduced_m):
        mean_node = _make_node(
            target=torch.ops.aten.mean.dim,
            name="mean_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        gen.generate_epilogue(
            [mean_node], output_descriptor=output_reduced_m,
        )
        assert gen._last_reduction is not None
        assert gen._last_reduction.op == "mean"

    def test_mean_keeps_fp32_accumulation(self, gen, output_reduced_m):
        """Mean partial sums must stay in FP32 — no .to(triton_dtype) cast."""
        mean_node = _make_node(
            target=torch.ops.aten.mean.dim,
            name="mean_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [mean_node], output_descriptor=output_reduced_m,
        )
        # Must NOT cast before atomic — accumulate in FP32
        assert ".to(tl.float32)" not in code
        assert ".to(tl.float16)" not in code
        assert ".to(tl.bfloat16)" not in code
        # Must have "FP32" or "fp32" in comments explaining the strategy
        assert "fp32" in code.lower()

    def test_mean_bf16_output_still_fp32_accumulation(self, gen):
        """Even with bf16 output, mean partial sums must be FP32."""
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        mean_node = _make_node(
            target=torch.ops.aten.mean.dim,
            name="mean_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue([mean_node], output_descriptor=out)
        # No cast to bf16 before atomic — stays FP32
        assert ".to(tl.bfloat16)" not in code
        assert "tl.sum(acc, axis=1)" in code
        assert "tl.atomic_add(out_ptrs, partial_mean" in code

    def test_mean_comment_mentions_two_stage(self, gen, output_reduced_m):
        """Mean comment must describe the two-stage strategy."""
        mean_node = _make_node(
            target=torch.ops.aten.mean.dim,
            name="mean_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [mean_node], output_descriptor=output_reduced_m,
        )
        assert "two-stage" in code.lower() or "block-local" in code.lower()


# ---------------------------------------------------------------------------
# Chained element-wise then reduction
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestEpilogueChainedReduction:
    """Element-wise ops followed by a reduction in the same epilogue."""

    def test_relu_then_sum(self, gen, output_reduced_m):
        relu_node = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(relu_node, [-1], False),
        )
        code = gen.generate_epilogue(
            [relu_node, sum_node], output_descriptor=output_reduced_m,
        )
        # ReLU must come before the reduction
        assert code.index("tl.where") < code.index("tl.sum")
        # Atomic add must come after the reduction
        assert code.index("tl.sum") < code.index("tl.atomic_add")

    def test_gelu_then_max(self, gen, output_reduced_m):
        gelu_node = _make_node(target=torch.ops.aten.gelu.default, name="gelu_out")
        max_node = _make_node(
            target=torch.ops.aten.amax.default,
            name="max_out",
            args=(gelu_node, [-1], False),
        )
        code = gen.generate_epilogue(
            [gelu_node, max_node], output_descriptor=output_reduced_m,
        )
        assert code.index("_tanh_approx") < code.index("tl.max")

    def test_add_then_sum(self, gen, output_reduced_m):
        residual = _make_node(op="placeholder", name="res")
        prev_node = _make_node(name="prev", op="get_attr")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(prev_node, residual),
        )
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(add_node, [-1], False),
        )
        code = gen.generate_epilogue(
            [prev_node, add_node, sum_node], output_descriptor=output_reduced_m,
        )
        assert code.index("acc = acc + res") < code.index("tl.sum")


# ---------------------------------------------------------------------------
# Epilogue: no reduction resets state
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestEpilogueNoReduction:
    """Non-reduction epilogues must NOT set _last_reduction."""

    def test_relu_only_no_reduction(self, gen):
        relu_node = _make_node(target=torch.ops.aten.relu.default)
        gen.generate_epilogue([relu_node])
        assert gen._last_reduction is None

    def test_empty_no_reduction(self, gen):
        gen.generate_epilogue([])
        assert gen._last_reduction is None


# ---------------------------------------------------------------------------
# _determine_reduction_axis
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestDetermineReductionAxis:
    """Verify axis mapping from PyTorch dims to Triton tile axes."""

    def test_dim_minus_1(self):
        node = _make_node(args=(_make_node(name="x"), [-1], False))
        axis, keepdim = TritonKernelGenerator._determine_reduction_axis(node)
        assert axis == 1
        assert keepdim is False

    def test_dim_1(self):
        node = _make_node(args=(_make_node(name="x"), [1], False))
        axis, _ = TritonKernelGenerator._determine_reduction_axis(node)
        assert axis == 1

    def test_dim_0(self):
        node = _make_node(args=(_make_node(name="x"), [0], False))
        axis, _ = TritonKernelGenerator._determine_reduction_axis(node)
        assert axis == 0

    def test_dim_minus_2(self):
        node = _make_node(args=(_make_node(name="x"), [-2], False))
        axis, _ = TritonKernelGenerator._determine_reduction_axis(node)
        assert axis == 0

    def test_keepdim_true(self):
        node = _make_node(args=(_make_node(name="x"), [-1], True))
        _, keepdim = TritonKernelGenerator._determine_reduction_axis(node)
        assert keepdim is True


# ---------------------------------------------------------------------------
# Signature / pointer generation with 1-D output
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestSignatureReducedOutput:
    """Verify that 1-D reduced output generates correct signature and pointers."""

    def test_reduced_output_has_single_stride(self, gen, matmul_inputs, output_reduced_m):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_reduced_m)
        # Should have stride_out_m but NOT stride_out_n
        assert "stride_out_m" in code
        assert "stride_out_n" not in code

    def test_reduced_output_pointer_uses_offs_m(self, gen, matmul_inputs, output_reduced_m):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_reduced_m)
        assert "out_ptrs = out_ptr + offs_m * stride_out_m" in code

    def test_reduced_output_no_2d_pointer(self, gen, matmul_inputs, output_reduced_m):
        """Must NOT emit 2-D output pointer arithmetic for reduced output."""
        code = gen.generate_signature_and_pointers(matmul_inputs, output_reduced_m)
        assert "offs_m[:, None]" not in code.split("output out")[0] if "output out" in code else True
        # Check the output pointer line directly
        lines = [l for l in code.split("\n") if "out_ptrs = " in l]
        for line in lines:
            if "out_ptr" in line and "input" not in line:
                assert "[:, None]" not in line

    def test_2d_output_still_has_two_strides(self, gen, matmul_inputs, output_2d):
        """Normal 2-D output must still generate both stride params."""
        code = gen.generate_signature_and_pointers(matmul_inputs, output_2d)
        assert "stride_out_m" in code
        assert "stride_out_n" in code

    def test_reduced_output_still_has_all_inputs(self, gen, matmul_inputs, output_reduced_m):
        """Input pointers and strides are unchanged by output reduction."""
        code = gen.generate_signature_and_pointers(matmul_inputs, output_reduced_m)
        assert "a_ptr" in code
        assert "b_ptr" in code
        assert "stride_am" in code
        assert "stride_bn" in code
        assert "M, N, K" in code


# ---------------------------------------------------------------------------
# compile_and_bind with reduction (skips normal store)
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestCompileAndBindReduction:
    """Verify that compile_and_bind skips normal tl.store after a reduction."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        mock_tl = types.ModuleType("triton.language")
        mock_tl.constexpr = int

        mock_triton = types.ModuleType("triton")
        mock_triton.jit = lambda fn: fn
        mock_triton.language = mock_tl

        monkeypatch.setitem(sys.modules, "triton", mock_triton)
        monkeypatch.setitem(sys.modules, "triton.language", mock_tl)

    def test_reduction_skips_store(self, gen, matmul_inputs, output_reduced_m):
        """Full pipeline with sum: should compile and NOT contain bare tl.store."""
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_reduced_m)
        kloop = gen.generate_k_loop(matmul_inputs, output_reduced_m)
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        epilogue = gen.generate_epilogue(
            [sum_node], output_descriptor=output_reduced_m,
        )
        kernel_string = sig + "\n" + kloop + "\n" + epilogue
        # _last_reduction should be set, so compile_and_bind skips the store
        fn = gen.compile_and_bind(kernel_string, output_reduced_m)
        assert callable(fn)

    def test_non_reduction_still_has_store(self, gen, matmul_inputs, output_2d):
        """Normal epilogue must still append the tl.store."""
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_2d)
        kloop = gen.generate_k_loop(matmul_inputs, output_2d)
        relu = _make_node(target=torch.ops.aten.relu.default)
        epilogue = gen.generate_epilogue([relu])
        kernel_string = sig + "\n" + kloop + "\n" + epilogue
        fn = gen.compile_and_bind(kernel_string, output_2d)
        assert callable(fn)

    def test_relu_then_sum_compiles(self, gen, matmul_inputs, output_reduced_m):
        """Chained ReLU + sum must compile without errors."""
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_reduced_m)
        kloop = gen.generate_k_loop(matmul_inputs, output_reduced_m)
        relu = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(relu, [-1], False),
        )
        epilogue = gen.generate_epilogue(
            [relu, sum_node], output_descriptor=output_reduced_m,
        )
        kernel_string = sig + "\n" + kloop + "\n" + epilogue
        fn = gen.compile_and_bind(kernel_string, output_reduced_m)
        assert callable(fn)
        assert fn.__name__ == "fused_kernel"


# ---------------------------------------------------------------------------
# Dtype cast in reduction emitters
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestReductionDtypeCast:
    """Reduction partial results must be cast to the output dtype."""

    def test_sum_fp16_output_casts(self, gen):
        out = TensorDescriptor("out", (128,), (1,), torch.float16)
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue([sum_node], output_descriptor=out)
        assert ".to(tl.float16)" in code

    def test_sum_fp32_output_casts(self, gen, output_reduced_m):
        sum_node = _make_node(
            target=torch.ops.aten.sum.dim_IntList,
            name="sum_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue(
            [sum_node], output_descriptor=output_reduced_m,
        )
        assert ".to(tl.float32)" in code

    def test_max_bf16_output_casts(self, gen):
        out = TensorDescriptor("out", (128,), (1,), torch.bfloat16)
        max_node = _make_node(
            target=torch.ops.aten.amax.default,
            name="max_out",
            args=(_make_node(name="prev"), [-1], False),
        )
        code = gen.generate_epilogue([max_node], output_descriptor=out)
        assert ".to(tl.bfloat16)" in code


# ---------------------------------------------------------------------------
# Registry: reduction ops registered
# ---------------------------------------------------------------------------

@pytest.mark.reduction
class TestRegistryReductions:
    """Verify that reduction ops are registered in the default registry."""

    def test_sum_registered(self):
        from fuseml.registry import build_default_registry
        reg = build_default_registry()
        assert reg.is_supported(torch.ops.aten.sum.dim_IntList)

    def test_amax_registered(self):
        from fuseml.registry import build_default_registry
        reg = build_default_registry()
        assert reg.is_supported(torch.ops.aten.amax.default)

    def test_mean_registered(self):
        from fuseml.registry import build_default_registry
        reg = build_default_registry()
        assert reg.is_supported(torch.ops.aten.mean.dim)

    def test_reduction_category(self):
        from fuseml.registry import build_default_registry
        reg = build_default_registry()
        assert reg.get_category(torch.ops.aten.sum.dim_IntList) == "reduction"
        assert reg.get_category(torch.ops.aten.amax.default) == "reduction"
        assert reg.get_category(torch.ops.aten.mean.dim) == "reduction"
