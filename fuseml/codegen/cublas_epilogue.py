"""cuBLAS epilogue fusion via cublasLt.

Provides pattern matching and runtime dispatch for compute-bound GEMMs
whose epilogue ops (GeLU, ReLU, residual add) can be fused into the
cublasLt matmul kernel — eliminating intermediate HBM round-trips
without paying the Triton GEMM throughput penalty.

Three-tier execution model (after integration)::

    Trigger GEMM detected
            │
            ▼
      is_compute_bound_gemm()?
            │
       ┌────┴────┐
       No        Yes
       │         │
       ▼         ▼
    Triton    Has fusible epilogue ops?
    fused     (GeLU, ReLU, residual add)
    kernel         │
               ┌───┴───┐
               No      Yes
               │       │
               ▼       ▼
            Eager    cuBLAS + cublasLt
            bypass   epilogue fusion

Supported patterns:

1. ``addmm → gelu`` — via ``_addmm_activation(use_gelu=True)``
2. ``addmm → relu`` — via ``_addmm_activation(use_gelu=False)``
3. ``addmm → add(residual)`` — via custom cublasLt CUDA extension
   that computes ``D = A @ B + bias + residual`` in a single kernel
   using ``CUBLASLT_EPILOGUE_BIAS`` with ``beta=1, C=residual``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import torch

from fuseml._logging import logger


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_cublas_epilogue_available_cache: bool | None = None


def cublas_epilogue_available() -> bool:
    """Check if cublasLt epilogue fusion is available on this system.

    Requirements:
    - CUDA device available
    - CUDA version >= 11.4
    - ``torch.ops.aten._addmm_activation`` exists (PyTorch 2.1+)
    """
    global _cublas_epilogue_available_cache
    if _cublas_epilogue_available_cache is not None:
        return _cublas_epilogue_available_cache

    available = False
    try:
        if not torch.cuda.is_available():
            logger.debug("cuBLAS epilogue: CUDA not available.")
        else:
            cuda_version = torch.version.cuda
            if cuda_version is None:
                logger.debug("cuBLAS epilogue: torch.version.cuda is None.")
            else:
                major, minor = (int(x) for x in cuda_version.split(".")[:2])
                if (major, minor) < (11, 4):
                    logger.debug(
                        "cuBLAS epilogue: CUDA %s < 11.4, not supported.",
                        cuda_version,
                    )
                elif not hasattr(torch.ops.aten, "_addmm_activation"):
                    logger.debug(
                        "cuBLAS epilogue: aten._addmm_activation not found "
                        "(requires PyTorch 2.1+).",
                    )
                else:
                    available = True
                    logger.debug(
                        "cuBLAS epilogue: available (CUDA %s, "
                        "aten._addmm_activation present).",
                        cuda_version,
                    )
    except Exception:
        pass

    _cublas_epilogue_available_cache = available
    return available


# ---------------------------------------------------------------------------
# CublasEpiloguePattern — describes a matched epilogue pattern
# ---------------------------------------------------------------------------


@dataclass
class CublasEpiloguePattern:
    """A matched cuBLAS epilogue fusion pattern.

    Attributes
    ----------
    epilogue_type : str
        ``"GELU_BIAS"``, ``"RELU_BIAS"``, or ``"BIAS_RESIDUAL"``.
    trigger_node : torch.fx.Node
        The matmul trigger node (``aten.addmm.default``).
    absorbed_nodes : list[torch.fx.Node]
        Epilogue nodes absorbed into the cublasLt call.
    output_node : torch.fx.Node
        The final node in the fused pattern.
    has_bias : bool
        Whether the trigger is ``addmm`` (has a bias vector).
    use_gelu : bool
        ``True`` for GELU epilogue, ``False`` otherwise.
    residual_arg_index : int | None
        For ``BIAS_RESIDUAL``: index of the residual tensor within
        the add node's args (0 or 1).  ``None`` for activation patterns.
    """

    epilogue_type: str
    trigger_node: Any  # torch.fx.Node
    absorbed_nodes: List[Any] = field(default_factory=list)
    output_node: Any = None  # torch.fx.Node
    has_bias: bool = True
    use_gelu: bool = True
    residual_arg_index: int | None = None


# ---------------------------------------------------------------------------
# Pattern matcher
# ---------------------------------------------------------------------------

_GELU_TARGETS = frozenset()
_RELU_TARGETS = frozenset()
_ADD_TARGETS = frozenset()

_targets_initialized = False


def _ensure_targets() -> None:
    """Lazily populate target sets."""
    global _GELU_TARGETS, _RELU_TARGETS, _ADD_TARGETS, _targets_initialized
    if _targets_initialized:
        return
    _GELU_TARGETS = frozenset({torch.ops.aten.gelu.default})
    _RELU_TARGETS = frozenset({
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
    })
    _ADD_TARGETS = frozenset({
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
    })
    _targets_initialized = True


def match_cublas_epilogue(
    trigger_node: torch.fx.Node,
) -> CublasEpiloguePattern | None:
    """Check if *trigger_node* + downstream ops form a cublasLt epilogue.

    Supported patterns (checked in priority order):

    1. ``addmm(bias, input, weight) → gelu(result)``
       → ``GELU_BIAS`` via ``_addmm_activation(..., use_gelu=True)``
    2. ``addmm(bias, input, weight) → relu(result)``
       → ``RELU_BIAS`` via ``_addmm_activation(..., use_gelu=False)``
    3. ``addmm(bias, input, weight) → add(result, residual)``
       → ``BIAS_RESIDUAL`` via custom cublasLt extension

    Returns ``None`` when no pattern matches or cublasLt is unavailable.
    """
    if not cublas_epilogue_available():
        return None

    _ensure_targets()

    if trigger_node.target is not torch.ops.aten.addmm.default:
        return None

    if len(trigger_node.args) < 3:
        return None

    if len(trigger_node.users) != 1:
        return None

    successor = next(iter(trigger_node.users))

    if successor.op != "call_function":
        return None

    # --- Pattern 1: addmm → gelu ---
    if successor.target in _GELU_TARGETS:
        approx = successor.kwargs.get("approximate", "none")
        if approx not in ("none", None):
            logger.debug(
                "cuBLAS epilogue: skipping gelu with approximate=%r "
                "(cublasLt uses exact erf-based GELU).",
                approx,
            )
            return None
        return CublasEpiloguePattern(
            epilogue_type="GELU_BIAS",
            trigger_node=trigger_node,
            absorbed_nodes=[successor],
            output_node=successor,
            has_bias=True,
            use_gelu=True,
        )

    # --- Pattern 2: addmm → relu ---
    if successor.target in _RELU_TARGETS:
        return CublasEpiloguePattern(
            epilogue_type="RELU_BIAS",
            trigger_node=trigger_node,
            absorbed_nodes=[successor],
            output_node=successor,
            has_bias=True,
            use_gelu=False,
        )

    # --- Pattern 3: addmm → add(residual) ---
    if successor.target in _ADD_TARGETS:
        if len(successor.args) < 2:
            return None
        arg0, arg1 = successor.args[0], successor.args[1]
        if not isinstance(arg0, torch.fx.Node) or not isinstance(arg1, torch.fx.Node):
            return None
        if arg0 is trigger_node:
            residual_idx = 1
        elif arg1 is trigger_node:
            residual_idx = 0
        else:
            return None

        if not _cublaslt_extension_available():
            return None

        return CublasEpiloguePattern(
            epilogue_type="BIAS_RESIDUAL",
            trigger_node=trigger_node,
            absorbed_nodes=[successor],
            output_node=successor,
            has_bias=True,
            use_gelu=False,
            residual_arg_index=residual_idx,
        )

    return None


# ---------------------------------------------------------------------------
# cublasLt CUDA extension — lazy JIT compilation
# ---------------------------------------------------------------------------

_cublaslt_ext = None
_cublaslt_ext_failed = False


def _cublaslt_extension_available() -> bool:
    """Check if the cublasLt CUDA extension can be compiled."""
    if _cublaslt_ext_failed:
        return False
    if _cublaslt_ext is not None:
        return True
    try:
        _get_cublaslt_extension()
        return True
    except Exception:
        return False


def _setup_msvc_env() -> None:
    """On Windows, ensure cl.exe is on PATH for JIT compilation.

    ``torch.utils.cpp_extension.load_inline`` needs the MSVC C++ compiler.
    When running outside a VS Developer Command Prompt, cl.exe may not be
    on PATH.  This function locates ``vcvarsall.bat`` via ``vswhere.exe``
    and imports the environment variables it sets.
    """
    import os
    import platform
    import shutil
    import subprocess

    if platform.system() != "Windows":
        return
    if shutil.which("cl") is not None:
        return

    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not os.path.isfile(vswhere):
        return

    try:
        result = subprocess.run(
            [vswhere, "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True, timeout=10,
        )
        vs_path = result.stdout.strip()
        if not vs_path:
            return

        vcvarsall = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
        if not os.path.isfile(vcvarsall):
            return

        env_cmd = f'"{vcvarsall}" x64 >nul 2>&1 && set'
        env_result = subprocess.run(
            env_cmd, shell=True, capture_output=True, text=True, timeout=30,
        )
        for line in env_result.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key] = value

        logger.debug("MSVC environment imported from %s", vcvarsall)
    except Exception as exc:
        logger.debug("Failed to set up MSVC environment: %s", exc)


def _get_cublaslt_extension():
    """Lazy-load the JIT-compiled cublasLt extension.

    Uses ``torch.utils.cpp_extension.load_inline`` to compile a small
    CUDA extension that wraps ``cublasLtMatmul`` with
    ``CUBLASLT_EPILOGUE_BIAS`` and ``beta=1`` (residual).  The compiled
    module is cached after first compilation.
    """
    global _cublaslt_ext, _cublaslt_ext_failed
    if _cublaslt_ext is not None:
        return _cublaslt_ext
    if _cublaslt_ext_failed:
        raise RuntimeError("cublasLt extension previously failed to compile.")

    try:
        from torch.utils.cpp_extension import load_inline
    except ImportError as exc:
        _cublaslt_ext_failed = True
        raise RuntimeError("torch.utils.cpp_extension not available.") from exc

    _setup_msvc_env()

    cpp_source = 'torch::Tensor addmm_bias_residual(torch::Tensor mat1, torch::Tensor mat2, torch::Tensor bias, torch::Tensor residual);\n'

    cuda_source = r"""
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace {

cudaDataType_t torch_to_cuda_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:    return CUDA_R_32F;
        case at::kHalf:     return CUDA_R_16F;
        case at::kBFloat16: return CUDA_R_16BF;
        default: TORCH_CHECK(false, "Unsupported dtype for cublasLt: ", dtype);
    }
}

cublasComputeType_t get_compute_type(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:    return CUBLAS_COMPUTE_32F;
        case at::kHalf:     return CUBLAS_COMPUTE_32F;
        case at::kBFloat16: return CUBLAS_COMPUTE_32F;
        default:            return CUBLAS_COMPUTE_32F;
    }
}

} // anonymous namespace

// D = mat1 @ mat2 + bias + residual   (single cublasLtMatmul call)
//
// mat1:     [M, K]  row-major
// mat2:     [K, N]  row-major
// bias:     [N]     1-D vector
// residual: [M, N]  row-major
// output:   [M, N]  row-major
//
// Row-major C = A @ B maps to col-major C^T = B^T @ A^T.
// cuBLAS sees:  m_cb = N,  n_cb = M,  k_cb = K
//   "A_cb" = B  (N x K col-major, i.e. B row-major transposed)
//   "B_cb" = A  (K x M col-major, i.e. A row-major transposed)
//   "C_cb" = residual  (N x M col-major)
//   "D_cb" = output    (N x M col-major)
// Bias is per-row in col-major = per-column in row-major = length N. OK.
torch::Tensor addmm_bias_residual(
    torch::Tensor mat1,
    torch::Tensor mat2,
    torch::Tensor bias,
    torch::Tensor residual
) {
    CHECK_CUDA(mat1); CHECK_CUDA(mat2); CHECK_CUDA(bias); CHECK_CUDA(residual);

    mat1 = mat1.contiguous();
    mat2 = mat2.contiguous();
    bias = bias.contiguous();
    residual = residual.contiguous();

    const int64_t M = mat1.size(0);
    const int64_t K = mat1.size(1);
    const int64_t N = mat2.size(1);

    TORCH_CHECK(mat2.size(0) == K, "mat2 rows must match mat1 cols");
    TORCH_CHECK(bias.numel() == N, "bias must have N elements");
    TORCH_CHECK(residual.size(0) == M && residual.size(1) == N,
                "residual must be [M, N]");

    auto output = torch::empty({M, N}, mat1.options());

    auto dtype = mat1.scalar_type();
    cudaDataType_t cuda_dtype = torch_to_cuda_dtype(dtype);
    cublasComputeType_t compute_type = get_compute_type(dtype);
    // scale type is always FP32 for mixed-precision GEMM
    cudaDataType_t scale_type = CUDA_R_32F;

    // Col-major dimensions after transpose trick
    int m_cb = static_cast<int>(N);
    int n_cb = static_cast<int>(M);
    int k_cb = static_cast<int>(K);

    // Leading dimensions (row-major stride-0 = col-major leading dim)
    int lda_cb = static_cast<int>(N);   // B row-major: stride(0) = N
    int ldb_cb = static_cast<int>(K);   // A row-major: stride(0) = K
    int ldc_cb = static_cast<int>(N);   // residual row-major: stride(0) = N
    int ldd_cb = static_cast<int>(N);   // output row-major: stride(0) = N

    float alpha = 1.0f, beta_val = 1.0f;

    // Get cublasLt handle from PyTorch's pool
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasLtHandle_t lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --- Descriptors ---
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr, d_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    auto cleanup = [&]() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (d_desc) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    };

    auto check = [&](cublasStatus_t status, const char* msg) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            cleanup();
            TORCH_CHECK(false, "cublasLt error in ", msg, ": ", static_cast<int>(status));
        }
    };

    // Operation descriptor
    check(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type),
          "MatmulDescCreate");

    // No transpose needed — the row-to-col trick means we pass
    // B^T and A^T as-is (their row-major layout IS col-major transposed).
    cublasOperation_t no_trans = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(op_desc,
          CUBLASLT_MATMUL_DESC_TRANSA, &no_trans, sizeof(no_trans)),
          "set TRANSA");
    check(cublasLtMatmulDescSetAttribute(op_desc,
          CUBLASLT_MATMUL_DESC_TRANSB, &no_trans, sizeof(no_trans)),
          "set TRANSB");

    // BIAS epilogue — adds bias vector (length m_cb = N) to each column
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    check(cublasLtMatmulDescSetAttribute(op_desc,
          CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
          "set EPILOGUE");

    const void* bias_ptr = bias.data_ptr();
    check(cublasLtMatmulDescSetAttribute(op_desc,
          CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)),
          "set BIAS_POINTER");

    cudaDataType_t bias_type = cuda_dtype;
    check(cublasLtMatmulDescSetAttribute(op_desc,
          CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)),
          "set BIAS_DATA_TYPE");

    // Matrix layouts (col-major, after transpose trick)
    check(cublasLtMatrixLayoutCreate(&a_desc, cuda_dtype, m_cb, k_cb, lda_cb),
          "A layout");
    check(cublasLtMatrixLayoutCreate(&b_desc, cuda_dtype, k_cb, n_cb, ldb_cb),
          "B layout");
    check(cublasLtMatrixLayoutCreate(&c_desc, cuda_dtype, m_cb, n_cb, ldc_cb),
          "C layout");
    check(cublasLtMatrixLayoutCreate(&d_desc, cuda_dtype, m_cb, n_cb, ldd_cb),
          "D layout");

    // Algorithm heuristic
    check(cublasLtMatmulPreferenceCreate(&preference), "PrefCreate");
    size_t workspace_size = 4 * 1024 * 1024;  // 4 MB
    check(cublasLtMatmulPreferenceSetAttribute(preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_size, sizeof(workspace_size)),
          "set workspace pref");

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned = 0;
    check(cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc,
          a_desc, b_desc, c_desc, d_desc,
          preference, 1, &heuristic, &returned),
          "AlgoGetHeuristic");

    if (returned == 0) {
        cleanup();
        TORCH_CHECK(false, "cublasLt: no suitable algorithm found");
    }

    // Workspace
    auto workspace = torch::empty(
        {static_cast<int64_t>(heuristic.workspaceSize)},
        torch::TensorOptions().dtype(torch::kByte).device(mat1.device()));

    // Execute: D = alpha * (B^T @ A^T) + beta * residual + bias
    //        = alpha * (mat1 @ mat2)^T + beta * residual^T + bias  (col-major)
    //        = mat1 @ mat2 + residual + bias                       (row-major)
    check(cublasLtMatmul(lt_handle, op_desc,
          &alpha,
          mat2.data_ptr(), a_desc,   // "A_cb" = mat2 (row-major as col-major B^T)
          mat1.data_ptr(), b_desc,   // "B_cb" = mat1 (row-major as col-major A^T)
          &beta_val,
          residual.data_ptr(), c_desc,  // C = residual
          output.data_ptr(), d_desc,    // D = output
          &heuristic.algo,
          workspace.data_ptr(), heuristic.workspaceSize,
          stream),
          "Matmul");

    cleanup();
    return output;
}
"""

    try:
        logger.info(
            "Compiling cublasLt CUDA extension (first-time JIT, may take "
            "10-30s)..."
        )
        ext = load_inline(
            name="fuseml_cublaslt",
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            functions=["addmm_bias_residual"],
            verbose=False,
            with_cuda=True,
        )
        _cublaslt_ext = ext
        logger.info("cublasLt CUDA extension compiled successfully.")
        return ext
    except Exception as exc:
        _cublaslt_ext_failed = True
        logger.warning("cublasLt CUDA extension failed to compile: %s", exc)
        raise


# ---------------------------------------------------------------------------
# CublasEpilogueLauncher — activation fusion (GELU/ReLU)
# ---------------------------------------------------------------------------


class CublasEpilogueLauncher:
    """Runtime launcher for cuBLAS GEMM with fused epilogue activation.

    Wraps ``torch.ops.aten._addmm_activation`` to dispatch a single
    cublasLt call that performs ``GEMM + bias + activation`` in one kernel.
    """

    def __init__(
        self,
        use_gelu: bool,
        epilogue_type: str = "GELU_BIAS",
    ) -> None:
        self.__name__ = "fuseml_cublas_epilogue"
        self._use_gelu = use_gelu
        self._epilogue_type = epilogue_type

    def __call__(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        if len(input_tensors) < 3:
            raise ValueError(
                f"CublasEpilogueLauncher expects at least 3 inputs "
                f"(bias, input, weight), got {len(input_tensors)}."
            )

        bias = input_tensors[0]
        mat1 = input_tensors[1]
        mat2 = input_tensors[2]

        try:
            return torch.ops.aten._addmm_activation(
                bias, mat1, mat2, use_gelu=self._use_gelu,
            )
        except Exception as exc:
            logger.warning(
                "cublasLt _addmm_activation failed (%s), falling back to "
                "separate addmm + activation.",
                exc,
            )
            result = torch.addmm(bias, mat1, mat2)
            if self._use_gelu:
                return torch.nn.functional.gelu(result)
            return torch.nn.functional.relu(result)

    def __repr__(self) -> str:
        return (
            f"CublasEpilogueLauncher("
            f"epilogue={self._epilogue_type!r}, "
            f"use_gelu={self._use_gelu})"
        )


# ---------------------------------------------------------------------------
# CublasResidualLauncher — bias + residual fusion
# ---------------------------------------------------------------------------


class CublasResidualLauncher:
    """Runtime launcher for cuBLAS GEMM with fused bias + residual add.

    Computes ``D = mat1 @ mat2 + bias + residual`` in a single
    ``cublasLtMatmul`` kernel call using ``CUBLASLT_EPILOGUE_BIAS``
    with ``beta=1, C=residual``.

    Falls back to ``addmm(bias, mat1, mat2) + residual`` (2 kernels)
    if the cublasLt extension is unavailable.
    """

    def __init__(self) -> None:
        self.__name__ = "fuseml_cublas_residual"

    def __call__(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        if len(input_tensors) < 4:
            raise ValueError(
                f"CublasResidualLauncher expects 4 inputs "
                f"(bias, input, weight, residual), got {len(input_tensors)}."
            )

        bias = input_tensors[0]
        mat1 = input_tensors[1]
        mat2 = input_tensors[2]
        residual = input_tensors[3]

        try:
            ext = _get_cublaslt_extension()
            return ext.addmm_bias_residual(mat1, mat2, bias, residual)
        except Exception as exc:
            logger.warning(
                "cublasLt residual fusion failed (%s), falling back to "
                "addmm + add.",
                exc,
            )
            return torch.addmm(bias, mat1, mat2) + residual

    def __repr__(self) -> str:
        return "CublasResidualLauncher(epilogue='BIAS_RESIDUAL')"
