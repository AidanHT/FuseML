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
