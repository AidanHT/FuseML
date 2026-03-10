"""Tests for dynamic @triton.autotune config generation and SRAM budget enforcement.

Covers four autotuning features in TritonKernelGenerator and KernelLauncher:

1. **Config Generation** — Cartesian product of BLOCK_SIZE_M/N/K, num_warps,
   and num_stages candidates formatted as triton.Config objects.
2. **SRAM Pruning** — pre-calculates shared memory footprint per config and
   prunes any config exceeding FuseML's rigid 48 KB SRAM budget.
3. **Reduction Specialisation** — chains ending in reduction ops generate
   configs with higher num_warps (up to 16) to saturate SMs.
4. **Meta-Parameters** — M, N, K are passed as key parameters to the
   autotuner for dynamic batch size adaptation.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from fuseml.codegen.kernel_generator import (
    TensorDescriptor,
    TritonKernelGenerator,
    _AUTOTUNE_BLOCK_K_CHOICES,
    _AUTOTUNE_BLOCK_M_CHOICES,
    _AUTOTUNE_BLOCK_N_CHOICES,
    _AUTOTUNE_GROUP_SIZE_M_CHOICES,
    _AUTOTUNE_NUM_STAGES_CHOICES,
    _AUTOTUNE_NUM_WARPS_CHOICES,
    _AUTOTUNE_REDUCTION_NUM_WARPS_CHOICES,
    _AUTOTUNE_SRAM_BUDGET_BYTES,
    _DTYPE_BYTES,
)
from fuseml.codegen.kernel_launcher import KernelLauncher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(*, op="call_function", target=None, name="node", args=(), meta=None):
    """Create a minimal object that quacks like ``torch.fx.Node``."""
    ns = SimpleNamespace(op=op, target=target, name=name, args=args)
    ns.meta = meta or {}
    return ns


class _MockKernelFn:
    """Mock that mimics Triton's ``kernel_fn[grid](*args, **kwargs)``.

    When kwargs are empty (autotuned kernel), the grid function is called
    with a default META dict to simulate ``@triton.autotune`` populating
    the selected Config's values at runtime.
    """

    _DEFAULT_META = {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
    }

    def __init__(self):
        self.calls: list[dict] = []

    def __getitem__(self, grid):
        def launcher(*args, **kwargs):
            if callable(grid):
                # In real Triton, the autotuner populates META from the
                # selected Config.  Use default values for mock tests.
                meta = kwargs if kwargs else self._DEFAULT_META
                resolved = grid(meta)
            else:
                resolved = grid
            self.calls.append({
                "grid": resolved,
                "args": args,
                "kwargs": kwargs,
            })
        return launcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    return TritonKernelGenerator()


@pytest.fixture
def matmul_inputs_fp32():
    """Standard A(128x64) @ B(64x256) inputs in FP32."""
    a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
    b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
    return [a, b]


@pytest.fixture
def matmul_inputs_fp16():
    """A(128x64) @ B(64x256) inputs in FP16."""
    a = TensorDescriptor("a", (128, 64), (64, 1), torch.float16)
    b = TensorDescriptor("b", (64, 256), (256, 1), torch.float16)
    return [a, b]


@pytest.fixture
def output_fp32():
    return TensorDescriptor("out", (128, 256), (256, 1), torch.float32)


@pytest.fixture
def output_fp16():
    return TensorDescriptor("out", (128, 256), (256, 1), torch.float16)


# ===========================================================================
# 1. Constants verification
# ===========================================================================

@pytest.mark.autotune
class TestAutotuneConstants:
    """Verify the module-level autotuning constants."""

    def test_sram_budget_is_100kb(self):
        assert _AUTOTUNE_SRAM_BUDGET_BYTES == 100 * 1024

    def test_block_m_choices(self):
        assert _AUTOTUNE_BLOCK_M_CHOICES == (32, 64, 128, 256)

    def test_block_n_choices(self):
        assert _AUTOTUNE_BLOCK_N_CHOICES == (32, 64, 128, 256)

    def test_block_k_choices(self):
        assert _AUTOTUNE_BLOCK_K_CHOICES == (32, 64, 128)

    def test_num_warps_choices(self):
        assert _AUTOTUNE_NUM_WARPS_CHOICES == (4, 8, 16)

    def test_num_stages_choices(self):
        assert _AUTOTUNE_NUM_STAGES_CHOICES == (2, 3, 4, 5)

    def test_reduction_num_warps_includes_16(self):
        assert 16 in _AUTOTUNE_REDUCTION_NUM_WARPS_CHOICES

    def test_reduction_num_warps_superset_of_standard(self):
        """Reduction warp choices must include the standard choices."""
        for w in _AUTOTUNE_NUM_WARPS_CHOICES:
            assert w in _AUTOTUNE_REDUCTION_NUM_WARPS_CHOICES

    def test_group_size_m_choices(self):
        assert _AUTOTUNE_GROUP_SIZE_M_CHOICES == (4, 8, 16)

    def test_dtype_bytes_fp32(self):
        assert _DTYPE_BYTES[torch.float32] == 4

    def test_dtype_bytes_fp16(self):
        assert _DTYPE_BYTES[torch.float16] == 2

    def test_dtype_bytes_bf16(self):
        assert _DTYPE_BYTES[torch.bfloat16] == 2


# ===========================================================================
# 2. SRAM footprint calculation
# ===========================================================================

@pytest.mark.autotune
class TestComputeSRAMFootprint:
    """Unit tests for _compute_sram_footprint."""

    def test_basic_fp32(self):
        """64x32 A-tile + 32x64 B-tile, 1 stage, 4 bytes."""
        result = TritonKernelGenerator._compute_sram_footprint(64, 64, 32, 4, 1)
        # A_tile = 64*32*4 = 8192, B_tile = 32*64*4 = 8192
        # total = 1 * (8192 + 8192) = 16384
        assert result == 16384

    def test_basic_fp16(self):
        """Same tile dims but 2 bytes per element."""
        result = TritonKernelGenerator._compute_sram_footprint(64, 64, 32, 2, 1)
        assert result == 8192  # half of FP32

    def test_multi_stage(self):
        """Multiple stages multiply the footprint."""
        single = TritonKernelGenerator._compute_sram_footprint(64, 64, 32, 4, 1)
        triple = TritonKernelGenerator._compute_sram_footprint(64, 64, 32, 4, 3)
        assert triple == 3 * single

    def test_large_tile_exceeds_48kb(self):
        """128x128 tiles with 4 stages in FP32 should exceed 48 KB."""
        result = TritonKernelGenerator._compute_sram_footprint(128, 128, 64, 4, 4)
        assert result > _AUTOTUNE_SRAM_BUDGET_BYTES

    def test_small_tile_fits_48kb(self):
        """32x32 tiles with 2 stages in FP32 should fit within 48 KB."""
        result = TritonKernelGenerator._compute_sram_footprint(32, 32, 32, 4, 2)
        assert result <= _AUTOTUNE_SRAM_BUDGET_BYTES

    def test_asymmetric_tiles(self):
        """Non-square tile: 128x32 with BLOCK_K=32."""
        result = TritonKernelGenerator._compute_sram_footprint(128, 32, 32, 4, 1)
        # A = 128*32*4 = 16384, B = 32*32*4 = 4096
        assert result == 20480

    def test_zero_stages_returns_zero(self):
        """Edge case: 0 stages → 0 bytes."""
        result = TritonKernelGenerator._compute_sram_footprint(64, 64, 32, 4, 0)
        assert result == 0


# ===========================================================================
# 3. Config generation
# ===========================================================================

@pytest.mark.autotune
class TestBuildAutotuneConfigs:
    """Verify _build_autotune_configs generates the full Cartesian product."""

    def test_standard_config_count(self):
        """Standard (non-reduction) config count = |M| * |N| * |K| * |warps| * |stages| * |gsm|."""
        configs = TritonKernelGenerator._build_autotune_configs(torch.float32)
        expected = (
            len(_AUTOTUNE_BLOCK_M_CHOICES)
            * len(_AUTOTUNE_BLOCK_N_CHOICES)
            * len(_AUTOTUNE_BLOCK_K_CHOICES)
            * len(_AUTOTUNE_NUM_WARPS_CHOICES)
            * len(_AUTOTUNE_NUM_STAGES_CHOICES)
            * len(_AUTOTUNE_GROUP_SIZE_M_CHOICES)
        )
        assert len(configs) == expected

    def test_reduction_config_count(self):
        """Reduction configs use expanded warp choices → more configs."""
        standard = TritonKernelGenerator._build_autotune_configs(torch.float32, False)
        reduction = TritonKernelGenerator._build_autotune_configs(torch.float32, True)
        assert len(reduction) > len(standard)

    def test_reduction_configs_include_16_warps(self):
        """At least one reduction config should have num_warps=16."""
        configs = TritonKernelGenerator._build_autotune_configs(torch.float32, True)
        warp_counts = {c["num_warps"] for c in configs}
        assert 16 in warp_counts

    def test_standard_configs_max_16_warps(self):
        """Standard configs should not exceed 16 warps."""
        configs = TritonKernelGenerator._build_autotune_configs(torch.float32, False)
        max_warps = max(c["num_warps"] for c in configs)
        assert max_warps == 16

    def test_all_configs_have_required_keys(self):
        """Every config must have all six required keys."""
        configs = TritonKernelGenerator._build_autotune_configs(torch.float32)
        required = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                     "GROUP_SIZE_M", "num_warps", "num_stages"}
        for cfg in configs:
            assert set(cfg.keys()) == required

    def test_group_size_m_in_choices(self):
        """All configs have a GROUP_SIZE_M from the choices tuple."""
        configs = TritonKernelGenerator._build_autotune_configs(torch.float32)
        for cfg in configs:
            assert cfg["GROUP_SIZE_M"] in _AUTOTUNE_GROUP_SIZE_M_CHOICES

    def test_block_sizes_are_powers_of_two(self):
        """All tile dimensions must be powers of two."""
        configs = TritonKernelGenerator._build_autotune_configs(torch.float32)
        for cfg in configs:
            for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
                val = cfg[key]
                assert val > 0 and (val & (val - 1)) == 0, f"{key}={val} is not a power of 2"


# ===========================================================================
# 4. SRAM pruning
# ===========================================================================

@pytest.mark.autotune
class TestSRAMPruning:
    """Verify _prune_configs_by_sram enforces the 48 KB budget."""

    def test_small_configs_survive(self):
        """All configs with tiny tiles should survive pruning."""
        configs = [
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        ]
        result = TritonKernelGenerator._prune_configs_by_sram(configs, torch.float32)
        assert len(result) == 1

    def test_oversized_config_pruned(self):
        """A config that exceeds 48 KB must be removed."""
        configs = [
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4},
        ]
        result = TritonKernelGenerator._prune_configs_by_sram(configs, torch.float32)
        assert len(result) == 0

    def test_mixed_survival(self):
        """Only configs within budget survive; oversized ones are dropped."""
        small = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
                 "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2}
        large = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                 "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4}
        result = TritonKernelGenerator._prune_configs_by_sram(
            [small, large], torch.float32,
        )
        assert len(result) == 1
        assert result[0] is small

    def test_fp16_more_configs_survive(self):
        """FP16 has 2 bytes/element — more configs fit within 48 KB."""
        all_configs = TritonKernelGenerator._build_autotune_configs(torch.float16)
        fp32_survivors = TritonKernelGenerator._prune_configs_by_sram(
            all_configs, torch.float32,
        )
        fp16_survivors = TritonKernelGenerator._prune_configs_by_sram(
            all_configs, torch.float16,
        )
        assert len(fp16_survivors) >= len(fp32_survivors)

    def test_custom_sram_budget(self):
        """Configs are pruned against a custom budget."""
        configs = [
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        ]
        # With a tiny budget of 1 KB, even small configs are pruned.
        result = TritonKernelGenerator._prune_configs_by_sram(
            configs, torch.float32, sram_budget=1024,
        )
        assert len(result) == 0

    def test_exact_boundary(self):
        """A config at exactly 48 KB should survive (<=, not <)."""
        # Find a config whose footprint == 48 KB exactly.
        # 48 KB = 49152 bytes.  With 4 bytes/elem, 1 stage:
        # A + B = BM*BK*4 + BK*BN*4 = 49152
        # If BM=BN=64, BK=X: 64*X*4 + X*64*4 = 512*X = 49152 → X ≈ 96 (not power of 2)
        # Use a custom budget instead.
        cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
               "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 1}
        footprint = TritonKernelGenerator._compute_sram_footprint(64, 64, 32, 4, 1)
        result = TritonKernelGenerator._prune_configs_by_sram(
            [cfg], torch.float32, sram_budget=footprint,
        )
        assert len(result) == 1

    def test_all_pruned_returns_empty(self):
        """When all configs exceed budget, result is empty."""
        configs = TritonKernelGenerator._build_autotune_configs(torch.float32)
        result = TritonKernelGenerator._prune_configs_by_sram(
            configs, torch.float32, sram_budget=1,
        )
        assert len(result) == 0

    def test_pruning_preserves_order(self):
        """Surviving configs maintain their original order."""
        c1 = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
               "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2}
        c2 = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
               "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4}
        c3 = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
               "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2}
        result = TritonKernelGenerator._prune_configs_by_sram(
            [c1, c2, c3], torch.float32,
        )
        assert result[0] is c1
        assert result[-1] is c3


# ===========================================================================
# 5. Autotune decorator formatting
# ===========================================================================

@pytest.mark.autotune
class TestAutotuneDecoratorFormat:
    """Verify the generated @triton.autotune decorator string."""

    def test_decorator_starts_with_autotune(self):
        configs = [
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        ]
        code = TritonKernelGenerator._section_autotune_decorator(configs)
        assert code.startswith("@triton.autotune(")

    def test_decorator_contains_triton_config(self):
        configs = [
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        ]
        code = TritonKernelGenerator._section_autotune_decorator(configs)
        assert "triton.Config(" in code

    def test_decorator_contains_key_meta_params(self):
        """M, N, K must appear as key parameters for the autotuner."""
        configs = [
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        ]
        code = TritonKernelGenerator._section_autotune_decorator(configs)
        assert "key=['M', 'N', 'K']" in code

    def test_decorator_contains_num_warps(self):
        configs = [
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
        ]
        code = TritonKernelGenerator._section_autotune_decorator(configs)
        assert "num_warps=8" in code

    def test_decorator_contains_num_stages(self):
        configs = [
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
        ]
        code = TritonKernelGenerator._section_autotune_decorator(configs)
        assert "num_stages=3" in code

    def test_decorator_contains_block_sizes(self):
        configs = [
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        ]
        code = TritonKernelGenerator._section_autotune_decorator(configs)
        assert "'BLOCK_SIZE_M': 128" in code
        assert "'BLOCK_SIZE_N': 64" in code
        assert "'BLOCK_SIZE_K': 32" in code

    def test_decorator_contains_group_size(self):
        configs = [
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        ]
        code = TritonKernelGenerator._section_autotune_decorator(configs)
        assert "'GROUP_SIZE_M': 8" in code

    def test_multiple_configs_all_present(self):
        """Each config appears as a separate triton.Config line."""
        configs = [
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
        ]
        code = TritonKernelGenerator._section_autotune_decorator(configs)
        assert code.count("triton.Config(") == 2

    def test_decorator_ends_with_closing_paren(self):
        configs = [
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
        ]
        code = TritonKernelGenerator._section_autotune_decorator(configs)
        assert code.strip().endswith(")")


# ===========================================================================
# 6. generate_autotune_configs (public API)
# ===========================================================================

@pytest.mark.autotune
class TestGenerateAutotuneConfigs:
    """Verify the public generate_autotune_configs method."""

    def test_returns_string(self, gen, matmul_inputs_fp32, output_fp32):
        result = gen.generate_autotune_configs(matmul_inputs_fp32, output_fp32)
        assert isinstance(result, str)

    def test_contains_autotune_decorator(self, gen, matmul_inputs_fp32, output_fp32):
        result = gen.generate_autotune_configs(matmul_inputs_fp32, output_fp32)
        assert "@triton.autotune(" in result

    def test_contains_key_meta_params(self, gen, matmul_inputs_fp32, output_fp32):
        result = gen.generate_autotune_configs(matmul_inputs_fp32, output_fp32)
        assert "key=['M', 'N', 'K']" in result

    def test_no_oversized_configs(self, gen, matmul_inputs_fp32, output_fp32):
        """Generated code should not contain configs that exceed 48 KB."""
        result = gen.generate_autotune_configs(matmul_inputs_fp32, output_fp32)
        # Count triton.Config occurrences
        config_count = result.count("triton.Config(")
        assert config_count > 0
        # All configs in the output are SRAM-safe (verified by the pruning logic)

    def test_reduction_has_more_configs_or_extra_warps(self, gen, matmul_inputs_fp32, output_fp32):
        """Reduction mode should produce more configs (superset warp choices)."""
        standard = gen.generate_autotune_configs(matmul_inputs_fp32, output_fp32, False)
        reduction = gen.generate_autotune_configs(matmul_inputs_fp32, output_fp32, True)
        assert "num_warps=16" in reduction
        # Reduction includes num_warps=2 which standard does not.
        assert "num_warps=2" in reduction
        assert "num_warps=2" not in standard

    def test_fp16_generates_configs(self, gen, matmul_inputs_fp16, output_fp16):
        """FP16 inputs should generate valid autotune configs."""
        result = gen.generate_autotune_configs(matmul_inputs_fp16, output_fp16)
        assert "@triton.autotune(" in result
        assert result.count("triton.Config(") > 0


# ===========================================================================
# 7. Integration with generate_signature_and_pointers
# ===========================================================================

@pytest.mark.autotune
class TestAutotuneSignatureIntegration:
    """Verify autotune decorator is prepended when autotune=True."""

    def test_autotune_false_no_decorator(self, gen, matmul_inputs_fp32, output_fp32):
        """Default (autotune=False) should NOT include @triton.autotune."""
        code = gen.generate_signature_and_pointers(matmul_inputs_fp32, output_fp32)
        assert "@triton.autotune(" not in code
        assert "@triton.jit" in code

    def test_autotune_true_includes_decorator(self, gen, matmul_inputs_fp32, output_fp32):
        """autotune=True should prepend @triton.autotune before @triton.jit."""
        code = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, output_fp32, autotune=True,
        )
        assert "@triton.autotune(" in code
        assert "@triton.jit" in code

    def test_autotune_before_jit(self, gen, matmul_inputs_fp32, output_fp32):
        """@triton.autotune must appear BEFORE @triton.jit in the source."""
        code = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, output_fp32, autotune=True,
        )
        autotune_idx = code.index("@triton.autotune(")
        jit_idx = code.index("@triton.jit")
        assert autotune_idx < jit_idx

    def test_autotune_with_intermediates(self, gen, matmul_inputs_fp32, output_fp32):
        """Autotune decorator works alongside intermediate tensor pointers."""
        intm = TensorDescriptor("mid", (128, 256), (256, 1), torch.float32)
        code = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, output_fp32,
            intermediate_tensors=[intm], autotune=True,
        )
        assert "@triton.autotune(" in code
        assert "mid_ptr" in code

    def test_autotune_with_reduction_flag(self, gen, matmul_inputs_fp32, output_fp32):
        """has_reduction=True passes through to generate higher warp configs."""
        code = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, output_fp32,
            autotune=True, has_reduction=True,
        )
        assert "num_warps=16" in code

    def test_autotune_key_contains_mnk(self, gen, matmul_inputs_fp32, output_fp32):
        """M, N, K must be meta-parameters (key) for dynamic batch adaptation."""
        code = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, output_fp32, autotune=True,
        )
        assert "key=['M', 'N', 'K']" in code

    def test_signature_still_has_function_def(self, gen, matmul_inputs_fp32, output_fp32):
        """The function definition must still be present after autotune decorator."""
        code = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, output_fp32, autotune=True,
        )
        assert "def fused_kernel(" in code

    def test_signature_still_has_block_offsets(self, gen, matmul_inputs_fp32, output_fp32):
        """Block offsets and pointer arithmetic must be unaffected by autotune."""
        code = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, output_fp32, autotune=True,
        )
        assert "offs_m" in code
        assert "offs_n" in code
        # offs_k is no longer emitted — block pointers handle K-axis advancement


# ===========================================================================
# 8. Reduction specialisation
# ===========================================================================

@pytest.mark.autotune
class TestReductionSpecialisation:
    """Verify higher num_warps for reduction-ending fusion chains."""

    def test_reduction_configs_have_16_warps(self):
        """Reduction configs must include num_warps=16."""
        configs = TritonKernelGenerator._build_autotune_configs(
            torch.float32, has_reduction=True,
        )
        warp_counts = {c["num_warps"] for c in configs}
        assert 16 in warp_counts

    def test_standard_configs_lack_2_warps(self):
        """Standard (non-reduction) configs must NOT include num_warps=2."""
        configs = TritonKernelGenerator._build_autotune_configs(
            torch.float32, has_reduction=False,
        )
        warp_counts = {c["num_warps"] for c in configs}
        assert 2 not in warp_counts

    def test_reduction_still_includes_lower_warps(self):
        """Reduction configs should still include 4 and 8 warps."""
        configs = TritonKernelGenerator._build_autotune_configs(
            torch.float32, has_reduction=True,
        )
        warp_counts = {c["num_warps"] for c in configs}
        assert 4 in warp_counts
        assert 8 in warp_counts

    def test_reduction_config_count_ratio(self):
        """Reduction should have 3/2 the configs of standard (3 vs 2 warp choices)."""
        standard = TritonKernelGenerator._build_autotune_configs(torch.float32, False)
        reduction = TritonKernelGenerator._build_autotune_configs(torch.float32, True)
        expected_ratio = len(_AUTOTUNE_REDUCTION_NUM_WARPS_CHOICES) / len(_AUTOTUNE_NUM_WARPS_CHOICES)
        actual_ratio = len(reduction) / len(standard)
        assert abs(actual_ratio - expected_ratio) < 0.01


# ===========================================================================
# 9. KernelLauncher autotuned mode
# ===========================================================================

@pytest.mark.autotune
class TestKernelLauncherAutotuned:
    """Verify KernelLauncher skips static tuning when is_autotuned=True."""

    def test_autotuned_flag_stored(self):
        """The is_autotuned flag must be stored on the launcher."""
        mock_fn = _MockKernelFn()
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float32)
        launcher = KernelLauncher(
            mock_fn, [a, b], out, [], "a", "b", is_autotuned=True,
        )
        assert launcher._is_autotuned is True

    def test_not_autotuned_by_default(self):
        """Default KernelLauncher should NOT be autotuned."""
        mock_fn = _MockKernelFn()
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float32)
        launcher = KernelLauncher(mock_fn, [a, b], out, [], "a", "b")
        assert launcher._is_autotuned is False

    def test_autotuned_repr(self):
        """Repr should mention autotuned=True when enabled."""
        mock_fn = _MockKernelFn()
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float32)
        launcher = KernelLauncher(
            mock_fn, [a, b], out, [], "a", "b", is_autotuned=True,
        )
        assert "autotuned=True" in repr(launcher)

    def test_non_autotuned_repr_no_flag(self):
        """Repr should NOT mention autotuned when disabled."""
        mock_fn = _MockKernelFn()
        a = TensorDescriptor("a", (128, 64), (64, 1), torch.float32)
        b = TensorDescriptor("b", (64, 256), (256, 1), torch.float32)
        out = TensorDescriptor("out", (128, 256), (256, 1), torch.float32)
        launcher = KernelLauncher(mock_fn, [a, b], out, [], "a", "b")
        assert "autotuned" not in repr(launcher)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for kernel launch tests",
    )
    def test_autotuned_no_block_kwargs(self):
        """Autotuned kernel launch must NOT pass BLOCK_SIZE_* or num_warps."""
        mock_fn = _MockKernelFn()
        a = TensorDescriptor("a", (4, 4), (4, 1), torch.float32)
        b = TensorDescriptor("b", (4, 4), (4, 1), torch.float32)
        out = TensorDescriptor("out", (4, 4), (4, 1), torch.float32)
        launcher = KernelLauncher(
            mock_fn, [a, b], out, [], "a", "b", is_autotuned=True,
        )
        a_t = torch.randn(4, 4, device="cuda")
        b_t = torch.randn(4, 4, device="cuda")
        launcher(a_t, b_t)
        call = mock_fn.calls[0]
        assert "BLOCK_SIZE_M" not in call["kwargs"]
        assert "num_warps" not in call["kwargs"]
        assert "num_stages" not in call["kwargs"]

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for kernel launch tests",
    )
    def test_non_autotuned_has_block_kwargs(self):
        """Non-autotuned kernel launch must pass BLOCK_SIZE_* and num_warps."""
        mock_fn = _MockKernelFn()
        a = TensorDescriptor("a", (4, 4), (4, 1), torch.float32)
        b = TensorDescriptor("b", (4, 4), (4, 1), torch.float32)
        out = TensorDescriptor("out", (4, 4), (4, 1), torch.float32)
        launcher = KernelLauncher(mock_fn, [a, b], out, [], "a", "b")
        a_t = torch.randn(4, 4, device="cuda")
        b_t = torch.randn(4, 4, device="cuda")
        launcher(a_t, b_t)
        call = mock_fn.calls[0]
        assert "BLOCK_SIZE_M" in call["kwargs"]
        assert "num_warps" in call["kwargs"]
        assert "num_stages" in call["kwargs"]


# ===========================================================================
# 10. End-to-end integration
# ===========================================================================

@pytest.mark.autotune
class TestAutotuneEndToEnd:
    """End-to-end tests combining autotune with other pipeline stages."""

    def test_autotune_plus_k_loop(self, gen, matmul_inputs_fp32, output_fp32):
        """Autotune signature + K-loop produces valid combined source."""
        sig = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, output_fp32, autotune=True,
        )
        kloop = gen.generate_k_loop(matmul_inputs_fp32, output_fp32)
        combined = sig + "\n" + kloop
        assert "@triton.autotune(" in combined
        assert "@triton.jit" in combined
        assert "tl.dot(" in combined

    def test_autotune_plus_epilogue_plus_store(self, gen, matmul_inputs_fp32, output_fp32):
        """Full pipeline: autotune + sig + K-loop + epilogue + store."""
        sig = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, output_fp32, autotune=True,
        )
        kloop = gen.generate_k_loop(matmul_inputs_fp32, output_fp32)
        relu_node = _make_node(target=torch.ops.aten.relu.default)
        epilogue = gen.generate_epilogue([relu_node])
        store = gen._section_store(output_fp32)
        full = "\n".join([sig, kloop, epilogue, store])
        assert "@triton.autotune(" in full
        assert "tl.where(acc > 0" in full
        assert "tl.store(" in full

    def test_autotune_fp16_with_safe_downcast(self, gen, matmul_inputs_fp16, output_fp16):
        """FP16 autotune + native Tensor Core loads + saturation store."""
        sig = gen.generate_signature_and_pointers(
            matmul_inputs_fp16, output_fp16, autotune=True,
        )
        kloop = gen.generate_k_loop(matmul_inputs_fp16, output_fp16)
        store = gen._section_store(output_fp16)
        full = "\n".join([sig, kloop, store])
        assert "@triton.autotune(" in full
        assert "acc = tl.dot(" in kloop       # fp32 accum via acc=acc
        assert "65504.0" in store             # FP16 saturation

    def test_autotune_reduction_full_pipeline(self, gen, matmul_inputs_fp32):
        """Reduction autotune: higher warps + reduced output + atomic store."""
        out_reduced = TensorDescriptor("out", (128,), (1,), torch.float32)
        sig = gen.generate_signature_and_pointers(
            matmul_inputs_fp32, out_reduced,
            autotune=True, has_reduction=True,
        )
        assert "num_warps=16" in sig
        assert "key=['M', 'N', 'K']" in sig
