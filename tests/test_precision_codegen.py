"""Tests for mixed-precision boundary handling, safe downcasting, and vectorized epilogues.

Covers four precision-management features in TritonKernelGenerator:

1. **Upcasting on Load** — FP16/BF16 operands are explicitly cast to FP32
   before ``tl.dot()`` accumulation to prevent numerical underflow/overflow.
2. **Broadcast-Aware Epilogue** — 2-D tensors with stride-0 broadcast dims
   use 1-D pointer arithmetic and Triton broadcasting instead of redundant
   2-D tile loads.
3. **Safe Downcasting** — FP16 outputs get saturating clamp (±65504) before
   the narrowing cast to prevent silent overflow to ±inf.
4. **In-Place Mutation** — In-place ops (relu_, add_, etc.) emit a register-
   reuse annotation confirming no temporary SRAM buffer is allocated.
"""

from types import SimpleNamespace

import pytest
import torch

from fuseml.codegen.kernel_generator import (
    TensorDescriptor,
    TritonKernelGenerator,
    _FP16_MAX,
    _HALF_PRECISION_DTYPES,
    _IN_PLACE_EPILOGUE_OPS,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight FX node stand-ins for epilogue tests
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
def output_fp32():
    return TensorDescriptor("out", (128, 256), (256, 1), torch.float32)


@pytest.fixture
def output_fp16():
    return TensorDescriptor("out", (128, 256), (256, 1), torch.float16)


@pytest.fixture
def output_bf16():
    return TensorDescriptor("out", (128, 256), (256, 1), torch.bfloat16)


# ===========================================================================
# 1. Upcasting on Load — FP16/BF16 → FP32 before tl.dot()
# ===========================================================================

@pytest.mark.precision
class TestUpcastingOnLoad:
    """Verify block-pointer loads use native dtype for Tensor Core throughput.

    With the Ada-optimised block-pointer K-loop, matmul operands are loaded
    in their native dtype (bf16/fp16) and passed directly to ``tl.dot``
    which accumulates in fp32 via ``acc=acc``.  Pre-casting to fp32 would
    bypass the Tensor Core pipeline.
    """

    def test_fp16_left_no_upcast(self, gen, output_fp32):
        """FP16 left operand must NOT chain .to(tl.float32) — Tensor Cores handle it."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float16)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        code = gen.generate_k_loop([a, b], output_fp32)
        # Block pointer loads should not have fp32 upcast
        for line in code.splitlines():
            if "a = tl.load(a_block_ptr," in line:
                assert ".to(tl.float32)" not in line, (
                    "FP16 matmul operand must stay in native dtype for Tensor Cores"
                )
                break
        else:
            pytest.fail("Left operand block pointer load line not found")

    def test_fp16_right_no_upcast(self, gen, output_fp32):
        """FP16 right operand must NOT chain .to(tl.float32)."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float16)
        code = gen.generate_k_loop([a, b], output_fp32)
        for line in code.splitlines():
            if "b = tl.load(b_block_ptr," in line:
                assert ".to(tl.float32)" not in line, (
                    "FP16 matmul operand must stay in native dtype for Tensor Cores"
                )
                break
        else:
            pytest.fail("Right operand block pointer load line not found")

    def test_bf16_both_no_upcast(self, gen, output_fp32):
        """BF16 matmul operands must NOT get .to(tl.float32) on loads."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.bfloat16)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.bfloat16)
        code = gen.generate_k_loop([a, b], output_fp32)
        # Block pointer load lines must not have upcast
        for line in code.splitlines():
            if "tl.load(" in line and "_block_ptr" in line:
                assert ".to(tl.float32)" not in line, (
                    "BF16 matmul operands must stay in native dtype for Tensor Cores"
                )

    def test_fp32_no_upcast(self, gen, output_fp32):
        """FP32 inputs must NOT emit .to(tl.float32) on K-loop loads."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        code = gen.generate_k_loop([a, b], output_fp32)
        for line in code.splitlines():
            if "tl.load(" in line and "_block_ptr" in line:
                assert ".to(tl.float32)" not in line, (
                    "FP32 operands should not chain .to(tl.float32)"
                )

    def test_mixed_precision_no_matmul_upcast(self, gen, output_fp32):
        """Neither operand gets upcast — Tensor Cores handle mixed precision."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float16)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        code = gen.generate_k_loop([a, b], output_fp32)
        for line in code.splitlines():
            if "tl.load(" in line and "_block_ptr" in line:
                assert ".to(tl.float32)" not in line

    def test_fp32_accumulation_via_acc_param(self, gen, output_fp32):
        """fp32 accumulation is achieved via acc=acc in tl.dot, not upcast."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float16)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float16)
        code = gen.generate_k_loop([a, b], output_fp32)
        assert "acc = tl.dot(a, b, acc=acc)" in code


@pytest.mark.precision
class TestUpcastingBiasLoad:
    """Verify FP32 upcast on half-precision bias vector loads."""

    def test_fp16_bias_upcast(self, gen, output_fp32):
        """FP16 bias loaded after K-loop must be upcast to FP32."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        bias = TensorDescriptor("bias", (256,), (1,), torch.float16)
        code = gen.generate_k_loop([a, b, bias], output_fp32)
        # Find the bias load line
        for line in code.splitlines():
            if "tl.load(bias" in line:
                assert ".to(tl.float32)" in line, (
                    "FP16 bias load must chain .to(tl.float32)"
                )
                break
        else:
            pytest.fail("Bias load line not found")

    def test_bf16_bias_upcast(self, gen, output_fp32):
        """BF16 bias also gets upcast."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        bias = TensorDescriptor("bias", (256,), (1,), torch.bfloat16)
        code = gen.generate_k_loop([a, b, bias], output_fp32)
        for line in code.splitlines():
            if "tl.load(bias" in line:
                assert ".to(tl.float32)" in line
                break
        else:
            pytest.fail("Bias load line not found")

    def test_fp32_bias_no_upcast(self, gen, output_fp32):
        """FP32 bias does NOT get upcast."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        bias = TensorDescriptor("bias", (256,), (1,), torch.float32)
        code = gen.generate_k_loop([a, b, bias], output_fp32)
        for line in code.splitlines():
            if "tl.load(bias" in line:
                assert ".to(tl.float32)" not in line
                break

    def test_m_axis_bias_upcast(self, gen, output_fp32):
        """FP16 bias along M axis also gets upcast."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        scale = TensorDescriptor("scale", (128,), (1,), torch.float16)
        code = gen.generate_k_loop([a, b, scale], output_fp32)
        for line in code.splitlines():
            if "tl.load(scale" in line:
                assert ".to(tl.float32)" in line
                break
        else:
            pytest.fail("Scale load line not found")


# ===========================================================================
# 2. Broadcast-Aware Epilogue — stride-0 detection and 1-D loads
# ===========================================================================

@pytest.mark.precision
class TestBroadcastAwareEpilogueAdd:
    """Verify 2-D tensors with stride-0 broadcast dims use 1-D loads in add."""

    def test_broadcast_dim0_add_uses_1d_load_along_n(self, gen):
        """stride[0]==0 (broadcast along M) → load 1-D along N, broadcast [None, :]."""
        # 2D tensor with stride=(0, 1) — row vector broadcast along M
        ext = _make_node(
            op="placeholder", name="res",
            meta={"val": SimpleNamespace(shape=(128, 256), stride=(0, 1))},
        )
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor, name="add_out",
            args=(acc_node, ext),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        # Should use 1-D pointer arithmetic with base pointer, not 2-D _ptrs
        assert "res_ptr + offs_n" in code
        assert "stride_res_n" in code
        assert "[None, :]" in code
        assert "broadcast_dims[0]=True" in code

    def test_broadcast_dim1_add_uses_1d_load_along_m(self, gen):
        """stride[1]==0 (broadcast along N) → load 1-D along M, broadcast [:, None]."""
        ext = _make_node(
            op="placeholder", name="col",
            meta={"val": SimpleNamespace(shape=(128, 256), stride=(256, 0))},
        )
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor, name="add_out",
            args=(acc_node, ext),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "col_ptr + offs_m" in code
        assert "stride_col_m" in code
        assert "[:, None]" in code
        assert "broadcast_dims[1]=True" in code

    def test_no_broadcast_uses_2d_load(self, gen):
        """No stride-0 dims → standard 2-D residual load path."""
        ext = _make_node(
            op="placeholder", name="res",
            meta={"val": SimpleNamespace(shape=(128, 256), stride=(256, 1))},
        )
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor, name="add_out",
            args=(acc_node, ext),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        # Should use the pre-computed 2-D _ptrs
        assert "res_ptrs" in code
        assert "offs_m[:, None] < M" in code

    def test_1d_tensor_still_uses_1d_bias_path(self, gen):
        """1-D tensors use the existing bias path (not broadcast detection)."""
        ext = _make_node(
            op="placeholder", name="bias",
            meta={"val": SimpleNamespace(shape=(256,), stride=(1,))},
        )
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor, name="add_out",
            args=(acc_node, ext),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "bias_ptrs" in code  # uses pre-computed 1-D pointers
        assert "Bias broadcast add" in code


@pytest.mark.precision
class TestBroadcastAwareEpilogueMul:
    """Verify 2-D tensors with stride-0 broadcast dims use 1-D loads in mul."""

    def test_broadcast_dim0_mul_uses_1d_load(self, gen):
        """stride[0]==0 → 1-D load along N for mul."""
        ext = _make_node(
            op="placeholder", name="scale",
            meta={"val": SimpleNamespace(shape=(128, 256), stride=(0, 1))},
        )
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        mul_node = _make_node(
            target=torch.ops.aten.mul.Tensor, name="mul_out",
            args=(acc_node, ext),
        )
        code = gen.generate_epilogue([acc_node, mul_node])
        assert "scale_ptr + offs_n" in code
        assert "[None, :]" in code
        assert "acc = acc * scale" in code

    def test_broadcast_dim1_mul_uses_1d_load(self, gen):
        """stride[1]==0 → 1-D load along M for mul."""
        ext = _make_node(
            op="placeholder", name="gate",
            meta={"val": SimpleNamespace(shape=(128, 256), stride=(1, 0))},
        )
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        mul_node = _make_node(
            target=torch.ops.aten.mul.Tensor, name="mul_out",
            args=(acc_node, ext),
        )
        code = gen.generate_epilogue([acc_node, mul_node])
        assert "gate_ptr + offs_m" in code
        assert "[:, None]" in code
        assert "acc = acc * gate" in code

    def test_no_broadcast_mul_uses_2d(self, gen):
        """No stride-0 → 2-D tile load for mul."""
        ext = _make_node(
            op="placeholder", name="mask_t",
            meta={"val": SimpleNamespace(shape=(128, 256), stride=(256, 1))},
        )
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        mul_node = _make_node(
            target=torch.ops.aten.mul.Tensor, name="mul_out",
            args=(acc_node, ext),
        )
        code = gen.generate_epilogue([acc_node, mul_node])
        assert "mask_t_ptrs" in code
        assert "mask_t = tl.load(mask_t_ptrs," in code


@pytest.mark.precision
class TestDetectEpilogueBroadcast:
    """Unit tests for the _detect_epilogue_broadcast static method."""

    def test_returns_m_for_stride0_dim0(self):
        node = _make_node(
            meta={"val": SimpleNamespace(shape=(128, 256), stride=(0, 1))},
        )
        assert TritonKernelGenerator._detect_epilogue_broadcast(node) == "m"

    def test_returns_n_for_stride0_dim1(self):
        node = _make_node(
            meta={"val": SimpleNamespace(shape=(128, 256), stride=(256, 0))},
        )
        assert TritonKernelGenerator._detect_epilogue_broadcast(node) == "n"

    def test_returns_none_for_no_broadcast(self):
        node = _make_node(
            meta={"val": SimpleNamespace(shape=(128, 256), stride=(256, 1))},
        )
        assert TritonKernelGenerator._detect_epilogue_broadcast(node) is None

    def test_returns_none_for_1d(self):
        node = _make_node(
            meta={"val": SimpleNamespace(shape=(256,), stride=(1,))},
        )
        assert TritonKernelGenerator._detect_epilogue_broadcast(node) is None

    def test_returns_none_for_no_metadata(self):
        node = _make_node(meta={})
        assert TritonKernelGenerator._detect_epilogue_broadcast(node) is None

    def test_returns_none_for_none_node(self):
        assert TritonKernelGenerator._detect_epilogue_broadcast(None) is None

    def test_tensor_meta_fallback(self):
        """Falls back to tensor_meta when val is not present."""
        node = _make_node(
            meta={"tensor_meta": SimpleNamespace(shape=(128, 256), stride=(0, 1))},
        )
        assert TritonKernelGenerator._detect_epilogue_broadcast(node) == "m"

    def test_callable_stride(self):
        """Handles FakeTensor-style .stride() callable."""
        class FakeTensor:
            shape = (128, 256)
            def stride(self):
                return (0, 1)
        node = _make_node(meta={"val": FakeTensor()})
        assert TritonKernelGenerator._detect_epilogue_broadcast(node) == "m"


# ===========================================================================
# 3. Safe Downcasting — FP16 saturation before narrowing cast
# ===========================================================================

@pytest.mark.precision
class TestSafeDowncasting:
    """Verify saturating clamp for FP16 targets in the store section."""

    def test_fp16_saturation_present(self, gen, output_fp16):
        """FP16 output must emit tl.where saturation before cast."""
        code = gen._section_store(output_fp16)
        assert "tl.where(acc > 65504.0" in code
        assert "tl.where(acc < -65504.0" in code

    def test_fp16_saturation_before_cast(self, gen, output_fp16):
        """Saturation clamp must appear before the .to(tl.float16) cast."""
        code = gen._section_store(output_fp16)
        sat_idx = code.index("tl.where(acc > 65504.0")
        cast_idx = code.index("acc.to(tl.float16)")
        assert sat_idx < cast_idx, (
            "FP16 saturation must precede the narrowing cast"
        )

    def test_fp16_saturation_before_store(self, gen, output_fp16):
        """Saturation must appear before tl.store."""
        code = gen._section_store(output_fp16)
        sat_idx = code.index("tl.where(acc > 65504.0")
        store_idx = code.index("tl.store(")
        assert sat_idx < store_idx

    def test_fp16_clamp_uses_correct_max(self, gen, output_fp16):
        """Saturation uses IEEE 754 FP16 max (65504.0)."""
        code = gen._section_store(output_fp16)
        assert "65504.0" in code
        assert "-65504.0" in code

    def test_fp16_cast_still_present(self, gen, output_fp16):
        """The .to(tl.float16) cast is still emitted after saturation."""
        code = gen._section_store(output_fp16)
        assert "acc = acc.to(tl.float16)" in code

    def test_bf16_no_saturation(self, gen, output_bf16):
        """BF16 shares FP32's exponent range — no saturation needed."""
        code = gen._section_store(output_bf16)
        assert "65504" not in code
        assert "tl.where" not in code
        assert "acc = acc.to(tl.bfloat16)" in code

    def test_fp32_no_saturation(self, gen, output_fp32):
        """FP32→FP32 is identity — no saturation needed."""
        code = gen._section_store(output_fp32)
        assert "65504" not in code
        assert "tl.where" not in code
        assert "acc = acc.to(tl.float32)" in code

    def test_store_mask_still_present(self, gen, output_fp16):
        """2-D boundary mask must still guard the tl.store after saturation."""
        code = gen._section_store(output_fp16)
        assert "mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)" in code

    def test_store_hbm_comment(self, gen, output_fp16):
        """Hardware sympathy: comment must mention HBM and SRAM."""
        code = gen._section_store(output_fp16)
        assert "HBM" in code
        assert "SRAM" in code

    def test_saturation_comment_mentions_fp16(self, gen, output_fp16):
        """Generated comment explains why saturation is needed."""
        code = gen._section_store(output_fp16)
        assert "FP16" in code
        assert "saturate" in code or "downcast" in code


@pytest.mark.precision
class TestFP16MaxConstant:
    """Verify the _FP16_MAX module constant."""

    def test_value(self):
        assert _FP16_MAX == 65504.0

    def test_is_float(self):
        assert isinstance(_FP16_MAX, float)


# ===========================================================================
# 4. In-Place Mutation — register-reuse annotations
# ===========================================================================

@pytest.mark.precision
class TestInPlaceMutationAnnotations:
    """Verify in-place ops emit register-reuse annotations in the epilogue."""

    def test_relu_inplace_annotation(self, gen):
        """relu_ must emit the in-place mutation annotation."""
        node = _make_node(target=torch.ops.aten.relu_.default)
        code = gen.generate_epilogue([node])
        assert "In-place mutation" in code
        assert "accumulator registers reused" in code
        assert "no temporary SRAM buffer" in code

    def test_relu_functional_no_annotation(self, gen):
        """Functional relu must NOT emit the in-place annotation."""
        node = _make_node(target=torch.ops.aten.relu.default)
        code = gen.generate_epilogue([node])
        assert "In-place mutation" not in code

    def test_sigmoid_inplace_annotation(self, gen):
        """sigmoid_ must emit the in-place mutation annotation."""
        node = _make_node(target=torch.ops.aten.sigmoid_.default)
        code = gen.generate_epilogue([node])
        assert "In-place mutation" in code

    def test_add_inplace_annotation(self, gen):
        """add_ must emit the in-place mutation annotation."""
        residual = _make_node(op="placeholder", name="res")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add_.Tensor, name="add_",
            args=(acc_node, residual),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "In-place mutation" in code

    def test_mul_inplace_annotation(self, gen):
        """mul_ must emit the in-place mutation annotation."""
        scale = _make_node(op="placeholder", name="scale")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        mul_node = _make_node(
            target=torch.ops.aten.mul_.Tensor, name="mul_",
            args=(acc_node, scale),
        )
        code = gen.generate_epilogue([acc_node, mul_node])
        assert "In-place mutation" in code

    def test_add_functional_no_annotation(self, gen):
        """Functional add.Tensor must NOT emit in-place annotation."""
        residual = _make_node(op="placeholder", name="res")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor, name="add_out",
            args=(acc_node, residual),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "In-place mutation" not in code

    def test_inplace_relu_still_emits_correct_code(self, gen):
        """In-place annotation does not break the actual relu emitter."""
        node = _make_node(target=torch.ops.aten.relu_.default)
        code = gen.generate_epilogue([node])
        assert "acc = tl.where(acc > 0, acc, 0.0)" in code

    def test_inplace_reuses_acc_no_temp_buffer(self, gen):
        """relu_ must NOT allocate a separate register tile — only modify acc."""
        node = _make_node(target=torch.ops.aten.relu_.default)
        code = gen.generate_epilogue([node])
        # No tl.zeros or tl.full allocations for temporary buffers
        assert "tl.zeros" not in code
        assert "tl.full" not in code


@pytest.mark.precision
class TestInPlaceOpsConstant:
    """Verify the _IN_PLACE_EPILOGUE_OPS module constant."""

    def test_contains_relu_inplace(self):
        assert torch.ops.aten.relu_.default in _IN_PLACE_EPILOGUE_OPS

    def test_contains_sigmoid_inplace(self):
        assert torch.ops.aten.sigmoid_.default in _IN_PLACE_EPILOGUE_OPS

    def test_contains_add_inplace(self):
        assert torch.ops.aten.add_.Tensor in _IN_PLACE_EPILOGUE_OPS

    def test_contains_mul_inplace(self):
        assert torch.ops.aten.mul_.Tensor in _IN_PLACE_EPILOGUE_OPS

    def test_does_not_contain_functional_relu(self):
        assert torch.ops.aten.relu.default not in _IN_PLACE_EPILOGUE_OPS

    def test_does_not_contain_functional_add(self):
        assert torch.ops.aten.add.Tensor not in _IN_PLACE_EPILOGUE_OPS

    def test_is_frozenset(self):
        assert isinstance(_IN_PLACE_EPILOGUE_OPS, frozenset)


@pytest.mark.precision
class TestHalfPrecisionDtypesConstant:
    """Verify the _HALF_PRECISION_DTYPES module constant."""

    def test_contains_fp16(self):
        assert torch.float16 in _HALF_PRECISION_DTYPES

    def test_contains_bf16(self):
        assert torch.bfloat16 in _HALF_PRECISION_DTYPES

    def test_does_not_contain_fp32(self):
        assert torch.float32 not in _HALF_PRECISION_DTYPES

    def test_is_frozenset(self):
        assert isinstance(_HALF_PRECISION_DTYPES, frozenset)


# ===========================================================================
# Integration: full pipeline with precision features
# ===========================================================================

@pytest.mark.precision
class TestPrecisionIntegration:
    """End-to-end pipeline tests combining multiple precision features."""

    def test_fp16_inputs_fp16_output_full_pipeline(self, gen):
        """FP16 matmul with fp32 accumulation, epilogue downcast, and saturation store."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float16)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float16)
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float16)

        kloop = gen.generate_k_loop([a, b], out)
        store = gen._section_store(out)

        # K-loop uses native dtype loads + fp32 accumulation via acc=acc
        assert "acc = tl.dot(a, b, acc=acc)" in kloop
        # K-loop epilogue downcast: FP16 saturation + narrowing cast
        assert "65504" in kloop
        assert "acc.to(tl.float16)" in kloop or ".to(tl.float16)" in kloop
        # Store has saturation + cast
        assert "65504.0" in store
        assert "acc.to(tl.float16)" in store

    def test_bf16_inputs_bf16_output_no_saturation(self, gen):
        """BF16 matmul with fp32 accumulation and epilogue downcast, no saturation."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.bfloat16)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.bfloat16)
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.bfloat16)

        kloop = gen.generate_k_loop([a, b], out)
        store = gen._section_store(out)

        # K-loop uses native dtype + epilogue downcast to bf16
        assert "acc = tl.dot(a, b, acc=acc)" in kloop
        assert ".to(tl.bfloat16)" in kloop
        assert "65504" not in store
        assert "acc.to(tl.bfloat16)" in store

    def test_fp16_bias_with_broadcast_residual(self, gen):
        """FP16 bias upcast + broadcast-aware residual add in epilogue."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        bias = TensorDescriptor("bias", (256,), (1,), torch.float16)

        kloop = gen.generate_k_loop([a, b, bias],
            TensorDescriptor("out", (128, 256), (256, 1), torch.float32))
        # FP16 bias should be upcast
        for line in kloop.splitlines():
            if "tl.load(bias" in line:
                assert ".to(tl.float32)" in line
                break

    def test_inplace_relu_with_fp16_output(self, gen):
        """relu_ annotation + FP16 saturation store."""
        node = _make_node(target=torch.ops.aten.relu_.default)
        epilogue = gen.generate_epilogue([node])
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float16)
        store = gen._section_store(out)

        assert "In-place mutation" in epilogue
        assert "65504.0" in store
