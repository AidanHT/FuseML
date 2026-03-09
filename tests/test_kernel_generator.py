"""Tests for TritonKernelGenerator — signature and pointer arithmetic."""

import pytest
import torch

from fuseml.codegen.kernel_generator import TensorDescriptor, TritonKernelGenerator


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
