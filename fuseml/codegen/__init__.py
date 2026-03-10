"""Code generation package — Triton kernels and cuBLAS epilogue fusion."""

from fuseml.codegen.cublas_epilogue import (
    CublasEpilogueLauncher,
    CublasEpiloguePattern,
    CublasResidualLauncher,
    cublas_epilogue_available,
    match_cublas_epilogue,
)
from fuseml.codegen.eager_fallback import EagerFallbackGuard
from fuseml.codegen.kernel_cache import (
    KernelCache,
    KernelCacheKey,
    TensorFingerprint,
    build_cache_key,
    build_op_chain,
)
from fuseml.codegen.kernel_generator import (
    ReductionInfo,
    TensorDescriptor,
    TritonKernelGenerator,
    next_power_of_2,
)
from fuseml.codegen.kernel_launcher import (
    KernelLauncher,
    LaunchParams,
    compute_launch_params,
)
from fuseml.codegen.sram_autotuner import (
    SRAMAutotuner,
    TuneConfig,
    compute_sram_bytes,
    generate_sram_safe_configs,
)

__all__ = [
    "CublasEpilogueLauncher",
    "CublasEpiloguePattern",
    "EagerFallbackGuard",
    "KernelCache",
    "KernelCacheKey",
    "KernelLauncher",
    "LaunchParams",
    "ReductionInfo",
    "SRAMAutotuner",
    "TensorDescriptor",
    "TensorFingerprint",
    "TritonKernelGenerator",
    "TuneConfig",
    "build_cache_key",
    "build_op_chain",
    "compute_launch_params",
    "cublas_epilogue_available",
    "compute_sram_bytes",
    "generate_sram_safe_configs",
    "next_power_of_2",
]
