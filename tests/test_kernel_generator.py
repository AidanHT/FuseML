"""Tests for TritonKernelGenerator — signature, pointers, K-loop, epilogue, store, and compilation."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from fuseml.codegen.kernel_generator import TensorDescriptor, TritonKernelGenerator, next_power_of_2


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
def matmul_inputs():
    """Standard A(128x64) @ B(64x256) inputs."""
    a = TensorDescriptor(name="a", shape=(128, 64), stride=(64, 1), dtype=torch.float32)
    b = TensorDescriptor(name="b", shape=(64, 256), stride=(256, 1), dtype=torch.float32)
    return [a, b]


@pytest.fixture
def bias_vec():
    return TensorDescriptor(name="bias", shape=(256,), stride=(1,), dtype=torch.float32)


@pytest.fixture
def output_tensor():
    return TensorDescriptor(name="out", shape=(128, 256), stride=(256, 1), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Signature structure
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestSignatureGeneration:
    """Verify the generated kernel source structure."""

    def test_contains_decorator(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "@triton.jit" in code

    def test_contains_function_def(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "def fused_kernel(" in code

    def test_contains_pointer_params(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "a_ptr" in code
        assert "b_ptr" in code
        assert "out_ptr" in code

    def test_contains_dimension_params(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "M, N, K" in code

    def test_contains_stride_params_a(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "stride_am" in code
        assert "stride_ak" in code

    def test_contains_stride_params_b(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "stride_bk" in code
        assert "stride_bn" in code

    def test_contains_stride_params_output(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "stride_out_m" in code
        assert "stride_out_n" in code

    def test_contains_constexpr_block_sizes(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "BLOCK_SIZE_M: tl.constexpr" in code
        assert "BLOCK_SIZE_N: tl.constexpr" in code
        assert "BLOCK_SIZE_K: tl.constexpr" in code

    def test_contains_program_ids(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "tl.program_id(0)" in code
        assert "tl.program_id(1)" in code

    def test_contains_block_offsets(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "tl.arange(0, BLOCK_SIZE_M)" in code
        assert "tl.arange(0, BLOCK_SIZE_N)" in code
        assert "tl.arange(0, BLOCK_SIZE_K)" in code


# ---------------------------------------------------------------------------
# Pointer arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestPointerArithmetic:
    """Verify pointer arithmetic patterns in generated code."""

    def test_a_pointer_arithmetic(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)" in code

    def test_b_pointer_arithmetic(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)" in code

    def test_output_pointer_arithmetic(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)" in code

    def test_bias_pointer_arithmetic(self, gen, matmul_inputs, bias_vec, output_tensor):
        inputs = matmul_inputs + [bias_vec]
        code = gen.generate_signature_and_pointers(inputs, output_tensor)
        assert "bias_ptrs = bias_ptr + offs_n * stride_bias_n" in code

    def test_bias_pointer_in_signature(self, gen, matmul_inputs, bias_vec, output_tensor):
        inputs = matmul_inputs + [bias_vec]
        code = gen.generate_signature_and_pointers(inputs, output_tensor)
        assert "bias_ptr" in code
        assert "stride_bias_n" in code

    def test_bias_broadcast_along_m(self, gen, output_tensor):
        """Bias matching M dimension broadcasts along m."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        vec = TensorDescriptor("scale", (128,), (1,), torch.float32)
        code = gen.generate_signature_and_pointers([a, b, vec], output_tensor)
        assert "scale_ptrs = scale_ptr + offs_m * stride_scale_m" in code


# ---------------------------------------------------------------------------
# Input ordering (Bug A fix)
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestInputOrdering:
    """Pointer list and stride blocks must follow caller-supplied order."""

    def test_preserves_input_order_bias_first(self, gen, output_tensor):
        """addmm convention: [bias, A, B] — pointers must appear in that order."""
        bias = TensorDescriptor("bias", (256,), (1,), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        code = gen.generate_signature_and_pointers([bias, a, b], output_tensor)
        # The pointer line must list bias_ptr before a_ptr
        ptr_line = [l for l in code.splitlines() if "bias_ptr" in l and "a_ptr" in l][0]
        assert ptr_line.index("bias_ptr") < ptr_line.index("a_ptr")

    def test_stride_order_matches_input_order(self, gen, output_tensor):
        """Stride blocks must appear in the same order as inputs."""
        bias = TensorDescriptor("bias", (256,), (1,), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        code = gen.generate_signature_and_pointers([bias, a, b], output_tensor)
        # stride_bias_n must appear before stride_am in the source
        assert code.index("stride_bias_n") < code.index("stride_am")

    def test_pointer_arithmetic_order_matches_input_order(self, gen, output_tensor):
        """Pointer arithmetic sections follow caller order."""
        bias = TensorDescriptor("bias", (256,), (1,), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        code = gen.generate_signature_and_pointers([bias, a, b], output_tensor)
        # bias_ptrs section comes before a_ptrs section
        assert code.index("bias_ptrs") < code.index("a_ptrs")


# ---------------------------------------------------------------------------
# Reversed matrix order (Bug B fix)
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestReversedMatrixOrder:
    """Generator must auto-detect left/right operands regardless of input order."""

    def test_reversed_order_does_not_crash(self, gen, output_tensor):
        """Passing B first, then A — should still infer M, N, K correctly."""
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        code = gen.generate_signature_and_pointers([b, a], output_tensor)
        assert "def fused_kernel(" in code

    def test_reversed_order_correct_labels(self, gen, output_tensor):
        """A gets (m, k) labels and B gets (k, n) regardless of input order."""
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        code = gen.generate_signature_and_pointers([b, a], output_tensor)
        # A always gets m/k strides, B always gets k/n strides
        assert "stride_am" in code
        assert "stride_ak" in code
        assert "stride_bk" in code
        assert "stride_bn" in code

    def test_reversed_order_preserves_caller_pointer_order(self, gen, output_tensor):
        """Even with auto-detection, pointer list follows caller order [b, a]."""
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        code = gen.generate_signature_and_pointers([b, a], output_tensor)
        ptr_line = [l for l in code.splitlines() if "b_ptr" in l and "a_ptr" in l][0]
        assert ptr_line.index("b_ptr") < ptr_line.index("a_ptr")

    def test_reversed_order_correct_pointer_arithmetic(self, gen, output_tensor):
        """Pointer arithmetic uses the correct dimension labels after swap."""
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        code = gen.generate_signature_and_pointers([b, a], output_tensor)
        assert "a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)" in code
        assert "b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)" in code


# ---------------------------------------------------------------------------
# Auxiliary 2-D tensors (Bug C fix)
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestAuxiliary2DTensor:
    """Extra 2-D inputs (e.g. residual connections) get (M x N) treatment."""

    def test_residual_gets_pointer(self, gen, output_tensor):
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        res = TensorDescriptor("res", (128, 256), (256, 1), torch.float32)
        code = gen.generate_signature_and_pointers([a, b, res], output_tensor)
        assert "res_ptr" in code

    def test_residual_gets_mn_strides(self, gen, output_tensor):
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        res = TensorDescriptor("res", (128, 256), (256, 1), torch.float32)
        code = gen.generate_signature_and_pointers([a, b, res], output_tensor)
        assert "stride_res_m" in code
        assert "stride_res_n" in code

    def test_residual_gets_pointer_arithmetic(self, gen, output_tensor):
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        res = TensorDescriptor("res", (128, 256), (256, 1), torch.float32)
        code = gen.generate_signature_and_pointers([a, b, res], output_tensor)
        assert "res_ptrs = res_ptr + (offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n)" in code


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestEdgeCases:
    """Error handling and deduplication."""

    def test_duplicate_inputs_deduplicated(self, gen, matmul_inputs, output_tensor):
        duped = matmul_inputs + [matmul_inputs[0]]  # a appears twice
        code = gen.generate_signature_and_pointers(duped, output_tensor)
        # Pointer line should list a_ptr only once
        ptr_lines = [l for l in code.splitlines() if "a_ptr, b_ptr" in l]
        assert len(ptr_lines) == 1

    def test_raises_on_single_matrix(self, gen, output_tensor):
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        with pytest.raises(ValueError, match="2 two-dimensional"):
            gen.generate_signature_and_pointers([a], output_tensor)

    def test_raises_on_dimension_mismatch(self, gen, output_tensor):
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (32, 256), (256, 1), torch.float32)
        with pytest.raises(ValueError, match="Contracting dimension"):
            gen.generate_signature_and_pointers([a, b], output_tensor)

    def test_no_vectors_still_works(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert "def fused_kernel(" in code
        assert "bias" not in code

    def test_returns_nonempty_string(self, gen, matmul_inputs, output_tensor):
        result = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multiple_bias_vectors(self, gen, output_tensor):
        """Two 1-D vectors both appear in the generated code."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        bias = TensorDescriptor("bias", (256,), (1,), torch.float32)
        scale = TensorDescriptor("scale", (256,), (1,), torch.float32)
        code = gen.generate_signature_and_pointers([a, b, bias, scale], output_tensor)
        assert "bias_ptr" in code
        assert "scale_ptr" in code
        assert "bias_ptrs = bias_ptr + offs_n * stride_bias_n" in code
        assert "scale_ptrs = scale_ptr + offs_n * stride_scale_n" in code


# ---------------------------------------------------------------------------
# K-loop: accumulator
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestKLoopAccumulator:
    """Verify accumulator initialization in the K-loop."""

    def test_accumulator_init(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)" in code

    def test_accumulator_before_loop(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert code.index("tl.zeros") < code.index("for k in range")


# ---------------------------------------------------------------------------
# K-loop: loop structure
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestKLoopStructure:
    """Verify the blocked GEMM loop structure."""

    def test_loop_header(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):" in code

    def test_dot_product(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "acc += tl.dot(a, b)" in code

    def test_load_left(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "a = tl.load(a_ptrs," in code

    def test_load_right(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "b = tl.load(b_ptrs," in code


# ---------------------------------------------------------------------------
# K-loop: mask handling
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestKLoopMasking:
    """Verify 2-D compound masks prevent OOB reads on all axes.

    A K-only mask segfaults when M or N are not divisible by their
    block sizes.  Each load must guard *both* the spatial boundary
    (M for left, N for right) and the K boundary.
    """

    def test_k_mask_computed(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "k_mask = offs_k < K - k * BLOCK_SIZE_K" in code

    def test_left_m_boundary(self, gen, matmul_inputs, output_tensor):
        """Left operand load guards the M boundary."""
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "offs_m[:, None] < M" in code

    def test_left_compound_mask(self, gen, matmul_inputs, output_tensor):
        """Left load uses full 2-D mask: M boundary & K boundary."""
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "mask=(offs_m[:, None] < M) & (k_mask[None, :])" in code

    def test_right_n_boundary(self, gen, matmul_inputs, output_tensor):
        """Right operand load guards the N boundary."""
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "offs_n[None, :] < N" in code

    def test_right_compound_mask(self, gen, matmul_inputs, output_tensor):
        """Right load uses full 2-D mask: K boundary & N boundary."""
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "mask=(k_mask[:, None]) & (offs_n[None, :] < N)" in code

    def test_other_zero(self, gen, matmul_inputs, output_tensor):
        """Masked-out elements must be zero to avoid corrupting the dot product."""
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "other=0.0" in code


# ---------------------------------------------------------------------------
# K-loop: pointer advancement
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestKLoopPointerAdvance:
    """Verify pointers advance to the next K-block after each iteration."""

    def test_advance_left_ptrs(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "a_ptrs += BLOCK_SIZE_K * stride_ak" in code

    def test_advance_right_ptrs(self, gen, matmul_inputs, output_tensor):
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "b_ptrs += BLOCK_SIZE_K * stride_bk" in code

    def test_advance_after_dot(self, gen, matmul_inputs, output_tensor):
        """Pointer advancement must come after the dot product."""
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert code.index("tl.dot") < code.index("a_ptrs += BLOCK_SIZE_K")


# ---------------------------------------------------------------------------
# K-loop: reversed input order
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestKLoopReversedOrder:
    """K-loop auto-detects matmul operands regardless of input order."""

    def test_reversed_order_correct_loads(self, gen, output_tensor):
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        code = gen.generate_k_loop([b, a], output_tensor)
        assert "a = tl.load(a_ptrs," in code
        assert "b = tl.load(b_ptrs," in code

    def test_reversed_order_correct_advance(self, gen, output_tensor):
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        code = gen.generate_k_loop([b, a], output_tensor)
        assert "a_ptrs += BLOCK_SIZE_K * stride_ak" in code
        assert "b_ptrs += BLOCK_SIZE_K * stride_bk" in code

    def test_reversed_order_correct_dot(self, gen, output_tensor):
        """dot(left, right) must use the correct operand names after auto-swap."""
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        code = gen.generate_k_loop([b, a], output_tensor)
        assert "acc += tl.dot(a, b)" in code


# ---------------------------------------------------------------------------
# K-loop: edge cases
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestKLoopEdgeCases:
    """Error handling and hardware-sympathy comments."""

    def test_raises_on_single_matrix(self, gen, output_tensor):
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        with pytest.raises(ValueError, match="2 two-dimensional"):
            gen.generate_k_loop([a], output_tensor)

    def test_raises_on_dimension_mismatch(self, gen, output_tensor):
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (32, 256), (256, 1), torch.float32)
        with pytest.raises(ValueError, match="Contracting dimension"):
            gen.generate_k_loop([a, b], output_tensor)

    def test_returns_nonempty_string(self, gen, matmul_inputs, output_tensor):
        result = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sram_comment_present(self, gen, matmul_inputs, output_tensor):
        """Generated code must explain SRAM usage per coding guidelines."""
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "SRAM" in code

    def test_hbm_comment_present(self, gen, matmul_inputs, output_tensor):
        """Generated code must mention HBM for hardware sympathy."""
        code = gen.generate_k_loop(matmul_inputs, output_tensor)
        assert "HBM" in code

    def test_with_bias_ignored(self, gen, output_tensor):
        """K-loop only involves matmul operands — bias is not loaded."""
        bias = TensorDescriptor("bias", (256,), (1,), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        code = gen.generate_k_loop([bias, a, b], output_tensor)
        assert "a = tl.load(a_ptrs," in code
        assert "b = tl.load(b_ptrs," in code
        assert "tl.load(bias" not in code

    def test_multichar_tensor_names(self, gen, output_tensor):
        """Multi-char names use underscore-separated stride params."""
        left = TensorDescriptor("input", (128, 64), (64, 1), torch.float32)
        right = TensorDescriptor("weight", (64, 256), (256, 1), torch.float32)
        code = gen.generate_k_loop([left, right], output_tensor)
        assert "input = tl.load(input_ptrs," in code
        assert "weight = tl.load(weight_ptrs," in code
        assert "input_ptrs += BLOCK_SIZE_K * stride_input_k" in code
        assert "weight_ptrs += BLOCK_SIZE_K * stride_weight_k" in code
        assert "acc += tl.dot(input, weight)" in code


# ---------------------------------------------------------------------------
# Epilogue: ReLU
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestEpilogueRelu:
    """ReLU post-op: acc = tl.where(acc > 0, acc, 0.0)."""

    def test_relu_emits_tl_where(self, gen):
        node = _make_node(target=torch.ops.aten.relu.default)
        code = gen.generate_epilogue([node])
        assert "acc = tl.where(acc > 0, acc, 0.0)" in code

    def test_relu_comment(self, gen):
        node = _make_node(target=torch.ops.aten.relu.default)
        code = gen.generate_epilogue([node])
        assert "ReLU" in code

    def test_relu_no_hbm_load(self, gen):
        """ReLU is register-only — must not emit tl.load."""
        node = _make_node(target=torch.ops.aten.relu.default)
        code = gen.generate_epilogue([node])
        assert "tl.load" not in code


# ---------------------------------------------------------------------------
# Epilogue: GeLU
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestEpilogueGelu:
    """GeLU post-op: fast tanh approximation."""

    def test_gelu_emits_tanh(self, gen):
        node = _make_node(target=torch.ops.aten.gelu.default)
        code = gen.generate_epilogue([node])
        assert "tl.math.tanh" in code

    def test_gelu_sqrt_2_over_pi(self, gen):
        """Must use sqrt(2/pi) ≈ 0.7978845608 constant."""
        node = _make_node(target=torch.ops.aten.gelu.default)
        code = gen.generate_epilogue([node])
        assert "0.7978845608" in code

    def test_gelu_cubic_coefficient(self, gen):
        """Must use the 0.044715 coefficient for the x^3 term."""
        node = _make_node(target=torch.ops.aten.gelu.default)
        code = gen.generate_epilogue([node])
        assert "0.044715" in code

    def test_gelu_half_factor(self, gen):
        """GeLU formula multiplies by 0.5."""
        node = _make_node(target=torch.ops.aten.gelu.default)
        code = gen.generate_epilogue([node])
        assert "0.5 * acc" in code

    def test_gelu_modifies_acc(self, gen):
        """Result must be assigned back to acc."""
        node = _make_node(target=torch.ops.aten.gelu.default)
        code = gen.generate_epilogue([node])
        assert "acc = 0.5 * acc * (1.0 + tl.math.tanh(" in code

    def test_gelu_no_hbm_load(self, gen):
        """GeLU is register-only — must not emit tl.load."""
        node = _make_node(target=torch.ops.aten.gelu.default)
        code = gen.generate_epilogue([node])
        assert "tl.load" not in code

    def test_gelu_comment(self, gen):
        node = _make_node(target=torch.ops.aten.gelu.default)
        code = gen.generate_epilogue([node])
        assert "GeLU" in code


# ---------------------------------------------------------------------------
# Epilogue: add.Tensor (residual connection)
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestEpilogueAdd:
    """Residual add: load external tensor from HBM, fuse into acc."""

    def test_add_loads_residual(self, gen):
        residual = _make_node(op="placeholder", name="res")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(acc_node, residual),
        )
        # acc_node is internal (in the group), residual is external
        code = gen.generate_epilogue([acc_node, add_node])
        assert "tl.load(res_ptrs," in code

    def test_add_mask(self, gen):
        """Residual load must use a 2-D boundary mask."""
        residual = _make_node(op="placeholder", name="res")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(acc_node, residual),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)" in code

    def test_add_other_zero(self, gen):
        """Masked-out elements default to 0.0."""
        residual = _make_node(op="placeholder", name="res")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(acc_node, residual),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "other=0.0" in code

    def test_add_acc_plus_residual(self, gen):
        residual = _make_node(op="placeholder", name="res")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(acc_node, residual),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "acc = acc + res" in code

    def test_add_uses_residual_name(self, gen):
        """Variable names derive from the residual node's name attribute."""
        residual = _make_node(op="placeholder", name="skip_conn")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(acc_node, residual),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "skip_conn = tl.load(skip_conn_ptrs," in code
        assert "acc = acc + skip_conn" in code

    def test_add_residual_as_first_arg(self, gen):
        """Residual may appear as first arg — still correctly identified."""
        residual = _make_node(op="placeholder", name="res")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(residual, acc_node),
        )
        # acc_node IS in the fusion group, residual is NOT
        code = gen.generate_epilogue([acc_node, add_node])
        assert "tl.load(res_ptrs," in code
        assert "acc = acc + res" in code

    def test_add_comment_mentions_hbm(self, gen):
        """Hardware-sympathy: comment must mention HBM load."""
        residual = _make_node(op="placeholder", name="res")
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(acc_node, residual),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "HBM" in code

    def test_add_scalar_float(self, gen):
        """Scalar add emits register-only arithmetic, no tl.load."""
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(acc_node, 1.0),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "acc = acc + 1.0" in code
        assert "tl.load" not in code

    def test_add_scalar_int(self, gen):
        """Integer scalar add — same register-only path."""
        acc_node = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(acc_node, 3),
        )
        code = gen.generate_epilogue([acc_node, add_node])
        assert "acc = acc + 3" in code
        assert "tl.load" not in code


# ---------------------------------------------------------------------------
# Epilogue: chained operations
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestEpilogueChained:
    """Multiple fused post-ops in sequence."""

    def test_relu_then_add(self, gen):
        """ReLU followed by a residual add."""
        residual = _make_node(op="placeholder", name="res")
        relu_node = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(relu_node, residual),
        )
        code = gen.generate_epilogue([relu_node, add_node])
        assert code.index("tl.where") < code.index("tl.load")
        assert code.index("tl.load") < code.index("acc = acc + res")

    def test_gelu_then_add(self, gen):
        """GeLU followed by a residual add."""
        residual = _make_node(op="placeholder", name="res")
        gelu_node = _make_node(target=torch.ops.aten.gelu.default, name="gelu_out")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(gelu_node, residual),
        )
        code = gen.generate_epilogue([gelu_node, add_node])
        assert code.index("tl.math.tanh") < code.index("tl.load")

    def test_add_then_relu(self, gen):
        """Residual add followed by ReLU."""
        residual = _make_node(op="placeholder", name="res")
        # prev_node is an internal predecessor (e.g. the matmul result)
        # that doesn't emit epilogue code — use "get_attr" so it's skipped.
        prev_node = _make_node(name="prev", op="get_attr")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(prev_node, residual),
        )
        relu_node = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        code = gen.generate_epilogue([prev_node, add_node, relu_node])
        assert code.index("acc = acc + res") < code.index("tl.where")


# ---------------------------------------------------------------------------
# Epilogue: edge cases
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestEpilogueEdgeCases:
    """Edge cases and structural guarantees."""

    def test_empty_nodes_returns_header_only(self, gen):
        code = gen.generate_epilogue([])
        assert "Epilogue" in code
        assert "tl.where" not in code
        assert "tl.math.tanh" not in code
        assert "tl.load" not in code

    def test_placeholder_nodes_skipped(self, gen):
        """Placeholder nodes must not produce any code."""
        ph = _make_node(op="placeholder", name="x")
        code = gen.generate_epilogue([ph])
        assert "tl.where" not in code
        assert "tl.math.tanh" not in code
        assert "tl.load" not in code

    def test_output_nodes_skipped(self, gen):
        out = _make_node(op="output", name="output")
        code = gen.generate_epilogue([out])
        assert "tl.where" not in code

    def test_returns_nonempty_string(self, gen):
        node = _make_node(target=torch.ops.aten.relu.default)
        result = gen.generate_epilogue([node])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sram_comment(self, gen):
        """Epilogue header must mention SRAM — operations stay in registers."""
        node = _make_node(target=torch.ops.aten.relu.default)
        code = gen.generate_epilogue([node])
        assert "SRAM" in code

    def test_unsupported_target_skipped(self, gen):
        """Unsupported call_function targets produce no code (only a log warning)."""
        node = _make_node(
            target=torch.ops.aten.sigmoid.default,
            name="sig",
        )
        code = gen.generate_epilogue([node])
        # Only the epilogue header should be emitted, no op code
        assert "Epilogue" in code
        assert "tl.where" not in code
        assert "tl.math.tanh" not in code
        assert "tl.load" not in code


# ---------------------------------------------------------------------------
# Store section
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestStoreSection:
    """Verify the tl.store section that writes results back to HBM."""

    def test_store_contains_tl_store(self, gen, output_tensor):
        code = gen._section_store(output_tensor)
        assert "tl.store(" in code

    def test_store_uses_output_pointer_name(self, gen, output_tensor):
        code = gen._section_store(output_tensor)
        assert "tl.store(out_ptrs," in code

    def test_store_2d_boundary_mask(self, gen, output_tensor):
        """Store must guard both M and N boundaries to prevent OOB writes."""
        code = gen._section_store(output_tensor)
        assert "mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)" in code

    def test_store_unconditional_cast_fp32(self, gen, output_tensor):
        """fp32 output — unconditional cast to tl.float32 (identity, zero cost)."""
        code = gen._section_store(output_tensor)
        assert "acc = acc.to(tl.float32)" in code

    def test_store_cast_fp16(self, gen):
        """fp32 accumulator → fp16 output requires explicit cast."""
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float16)
        code = gen._section_store(out)
        assert "acc = acc.to(tl.float16)" in code

    def test_store_cast_bf16(self, gen):
        """fp32 accumulator → bf16 output requires explicit cast."""
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.bfloat16)
        code = gen._section_store(out)
        assert "acc = acc.to(tl.bfloat16)" in code

    def test_store_cast_before_store(self, gen):
        """Dtype cast must appear before tl.store — can't store wrong type."""
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float16)
        code = gen._section_store(out)
        assert code.index("acc.to(") < code.index("tl.store(")

    def test_store_hbm_comment(self, gen, output_tensor):
        """Hardware sympathy: comment must explain the HBM write."""
        code = gen._section_store(output_tensor)
        assert "HBM" in code

    def test_store_sram_comment(self, gen, output_tensor):
        """Hardware sympathy: comment must mention SRAM source."""
        code = gen._section_store(output_tensor)
        assert "SRAM" in code

    def test_store_custom_output_name(self, gen):
        """Store uses the output tensor's name, not a hardcoded 'out'."""
        out = TensorDescriptor("result", (128, 256), (256, 1), torch.float32)
        code = gen._section_store(out)
        assert "tl.store(result_ptrs," in code

    def test_store_exactly_once_comment(self, gen, output_tensor):
        """Comment must emphasise single write — the whole point of fusion."""
        code = gen._section_store(output_tensor)
        assert "exactly once" in code


# ---------------------------------------------------------------------------
# compile_and_bind — runtime compilation
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestCompileAndBind:
    """Verify runtime compilation via exec()."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        """Inject a minimal mock triton so compile_and_bind can import it."""
        mock_tl = types.ModuleType("triton.language")
        mock_tl.constexpr = int  # Valid annotation type for kernel params

        mock_triton = types.ModuleType("triton")
        mock_triton.jit = lambda fn: fn  # Identity decorator
        mock_triton.language = mock_tl

        monkeypatch.setitem(sys.modules, "triton", mock_triton)
        monkeypatch.setitem(sys.modules, "triton.language", mock_tl)

    def test_returns_callable(self, gen, matmul_inputs, output_tensor):
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        kloop = gen.generate_k_loop(matmul_inputs, output_tensor)
        kernel_string = sig + "\n" + kloop
        fn = gen.compile_and_bind(kernel_string, output_tensor)
        assert callable(fn)

    def test_function_name_is_fused_kernel(self, gen, matmul_inputs, output_tensor):
        """The returned callable must be the 'fused_kernel' function."""
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        kloop = gen.generate_k_loop(matmul_inputs, output_tensor)
        kernel_string = sig + "\n" + kloop
        fn = gen.compile_and_bind(kernel_string, output_tensor)
        assert fn.__name__ == "fused_kernel"

    def test_with_epilogue_relu(self, gen, matmul_inputs, output_tensor):
        """Full pipeline: signature + K-loop + epilogue + store compiles OK."""
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        kloop = gen.generate_k_loop(matmul_inputs, output_tensor)
        relu = _make_node(target=torch.ops.aten.relu.default)
        epilogue = gen.generate_epilogue([relu])
        kernel_string = sig + "\n" + kloop + "\n" + epilogue
        fn = gen.compile_and_bind(kernel_string, output_tensor)
        assert callable(fn)

    def test_with_epilogue_gelu(self, gen, matmul_inputs, output_tensor):
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        kloop = gen.generate_k_loop(matmul_inputs, output_tensor)
        gelu = _make_node(target=torch.ops.aten.gelu.default)
        epilogue = gen.generate_epilogue([gelu])
        kernel_string = sig + "\n" + kloop + "\n" + epilogue
        fn = gen.compile_and_bind(kernel_string, output_tensor)
        assert callable(fn)

    def test_with_residual_add(self, gen, output_tensor):
        """Pipeline with residual add compiles successfully."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        res = TensorDescriptor("res", (128, 256), (256, 1), torch.float32)
        sig = gen.generate_signature_and_pointers([a, b, res], output_tensor)
        kloop = gen.generate_k_loop([a, b, res], output_tensor)
        residual = _make_node(op="placeholder", name="res")
        prev = _make_node(target=torch.ops.aten.relu.default, name="prev")
        add_node = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(prev, residual),
        )
        epilogue = gen.generate_epilogue([prev, add_node])
        kernel_string = sig + "\n" + kloop + "\n" + epilogue
        fn = gen.compile_and_bind(kernel_string, output_tensor)
        assert callable(fn)

    def test_fp16_output_includes_cast(self, gen, matmul_inputs):
        """fp16 output → store section includes dtype cast."""
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float16)
        sig = gen.generate_signature_and_pointers(matmul_inputs, out)
        kloop = gen.generate_k_loop(matmul_inputs, out)
        kernel_string = sig + "\n" + kloop
        fn = gen.compile_and_bind(kernel_string, out)
        assert callable(fn)

    def test_with_bias_input(self, gen, output_tensor):
        """addmm-style kernel with bias compiles OK."""
        bias = TensorDescriptor("bias", (256,), (1,), torch.float32)
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        sig = gen.generate_signature_and_pointers([bias, a, b], output_tensor)
        kloop = gen.generate_k_loop([bias, a, b], output_tensor)
        kernel_string = sig + "\n" + kloop
        fn = gen.compile_and_bind(kernel_string, output_tensor)
        assert callable(fn)
        assert fn.__name__ == "fused_kernel"

    def test_chained_epilogue_compiles(self, gen, matmul_inputs, output_tensor):
        """Multiple chained post-ops compile without errors."""
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        kloop = gen.generate_k_loop(matmul_inputs, output_tensor)
        relu = _make_node(target=torch.ops.aten.relu.default, name="relu_out")
        scalar_add = _make_node(
            target=torch.ops.aten.add.Tensor,
            name="add_out",
            args=(relu, 1.0),
        )
        epilogue = gen.generate_epilogue([relu, scalar_add])
        kernel_string = sig + "\n" + kloop + "\n" + epilogue
        fn = gen.compile_and_bind(kernel_string, output_tensor)
        assert callable(fn)


# ---------------------------------------------------------------------------
# compile_and_bind — kernel compilation cache
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestCompileCache:
    """Verify that identical kernels are cached and not recompiled."""

    @pytest.fixture(autouse=True)
    def _inject_mock_triton(self, monkeypatch):
        """Inject a minimal mock triton so compile_and_bind can import it."""
        mock_tl = types.ModuleType("triton.language")
        mock_tl.constexpr = int

        mock_triton = types.ModuleType("triton")
        mock_triton.jit = lambda fn: fn
        mock_triton.language = mock_tl

        monkeypatch.setitem(sys.modules, "triton", mock_triton)
        monkeypatch.setitem(sys.modules, "triton.language", mock_tl)

    def test_cache_returns_same_object(self, gen, matmul_inputs, output_tensor):
        """Second call with identical kernel must return the exact same object."""
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        kloop = gen.generate_k_loop(matmul_inputs, output_tensor)
        kernel_string = sig + "\n" + kloop
        fn1 = gen.compile_and_bind(kernel_string, output_tensor)
        fn2 = gen.compile_and_bind(kernel_string, output_tensor)
        assert fn1 is fn2

    def test_cache_size_after_one_compile(self, gen, matmul_inputs, output_tensor):
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        kloop = gen.generate_k_loop(matmul_inputs, output_tensor)
        kernel_string = sig + "\n" + kloop
        gen.compile_and_bind(kernel_string, output_tensor)
        assert len(gen._kernel_cache) == 1

    def test_cache_no_growth_on_repeat(self, gen, matmul_inputs, output_tensor):
        """Repeated calls must not grow the cache."""
        sig = gen.generate_signature_and_pointers(matmul_inputs, output_tensor)
        kloop = gen.generate_k_loop(matmul_inputs, output_tensor)
        kernel_string = sig + "\n" + kloop
        gen.compile_and_bind(kernel_string, output_tensor)
        gen.compile_and_bind(kernel_string, output_tensor)
        gen.compile_and_bind(kernel_string, output_tensor)
        assert len(gen._kernel_cache) == 1

    def test_different_kernels_cached_separately(self, gen, output_tensor):
        """Two different kernel strings must produce two cache entries."""
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        sig1 = gen.generate_signature_and_pointers([a, b], output_tensor)
        kloop1 = gen.generate_k_loop([a, b], output_tensor)
        ks1 = sig1 + "\n" + kloop1

        # Different tensor names → different generated variable names
        x = TensorDescriptor("x", (128, 64), (64, 1), torch.float32)
        y = TensorDescriptor("y", (64, 256), (256, 1), torch.float32)
        sig2 = gen.generate_signature_and_pointers([x, y], output_tensor)
        kloop2 = gen.generate_k_loop([x, y], output_tensor)
        ks2 = sig2 + "\n" + kloop2

        fn1 = gen.compile_and_bind(ks1, output_tensor)
        fn2 = gen.compile_and_bind(ks2, output_tensor)

        assert fn1 is not fn2
        assert len(gen._kernel_cache) == 2

    def test_same_kernel_different_output_dtype_cached_separately(self, gen, matmul_inputs):
        """Same kernel_string but different output dtype → different store section → different cache entry."""
        sig = gen.generate_signature_and_pointers(matmul_inputs,
            TensorDescriptor("out", (128, 256), (256, 1), torch.float32))
        kloop = gen.generate_k_loop(matmul_inputs,
            TensorDescriptor("out", (128, 256), (256, 1), torch.float32))
        kernel_string = sig + "\n" + kloop

        out_fp32 = TensorDescriptor("out", (128, 256), (256, 1), torch.float32)
        out_fp16 = TensorDescriptor("out", (128, 256), (256, 1), torch.float16)

        gen.compile_and_bind(kernel_string, out_fp32)
        gen.compile_and_bind(kernel_string, out_fp16)

        # Store section differs (fp16 adds a cast) → two entries
        assert len(gen._kernel_cache) == 2

    def test_cache_is_per_instance(self, matmul_inputs, output_tensor):
        """Each generator instance has its own independent cache."""
        gen1 = TritonKernelGenerator()
        gen2 = TritonKernelGenerator()
        sig = gen1.generate_signature_and_pointers(matmul_inputs, output_tensor)
        kloop = gen1.generate_k_loop(matmul_inputs, output_tensor)
        kernel_string = sig + "\n" + kloop

        gen1.compile_and_bind(kernel_string, output_tensor)
        assert len(gen1._kernel_cache) == 1
        assert len(gen2._kernel_cache) == 0

    def test_cache_empty_on_init(self):
        """Fresh generator starts with an empty cache."""
        gen = TritonKernelGenerator()
        assert len(gen._kernel_cache) == 0


# ---------------------------------------------------------------------------
# next_power_of_2 helper
# ---------------------------------------------------------------------------

@pytest.mark.codegen
class TestNextPowerOf2:
    """Verify the power-of-two rounding helper used for BLOCK_SIZE validation."""

    @pytest.mark.parametrize("val, expected", [
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (7, 8),
        (8, 8),
        (31, 32),
        (32, 32),
        (33, 64),
        (64, 64),
        (100, 128),
        (127, 128),
        (128, 128),
        (129, 256),
        (255, 256),
        (256, 256),
        (1024, 1024),
        (1025, 2048),
    ])
    def test_rounds_up_correctly(self, val, expected):
        assert next_power_of_2(val) == expected

    def test_result_is_always_power_of_two(self):
        """Exhaustive check over a wide range."""
        for v in range(1, 2049):
            result = next_power_of_2(v)
            assert result >= v
            assert result & (result - 1) == 0, f"{v} -> {result} is not a power of 2"

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive"):
            next_power_of_2(0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="positive"):
            next_power_of_2(-1)
