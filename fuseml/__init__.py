"""FuseML — JIT deep learning compiler for fusing memory-bound operators.

Public API re-exports for convenient access::

    from fuseml import FuseMLFusionPass, FusionGroup, SupportedOpsRegistry
"""

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
from fuseml.compiler import FuseMLCompiler
from fuseml.fusion_group import FusionGroup
from fuseml.passes.control_flow_validation import (
    ControlFlowError,
    validate_graph_control_flow,
)
from fuseml.passes.fusion_pass import FuseMLFusionPass, fuseml_fused_kernel_placeholder
from fuseml.passes.graph_cut import (
    SUPPORTED_TRITON_OPS,
    GraphSegment,
    split_fusion_group,
    validate_fusion_group,
)
from fuseml.registry import SupportedOpsRegistry, build_default_registry

__all__ = [
    "ControlFlowError",
    "FuseMLCompiler",
    "FuseMLFusionPass",
    "FusionGroup",
    "GraphSegment",
    "KernelCache",
    "KernelCacheKey",
    "ReductionInfo",
    "SUPPORTED_TRITON_OPS",
    "SupportedOpsRegistry",
    "TensorDescriptor",
    "TensorFingerprint",
    "TritonKernelGenerator",
    "build_cache_key",
    "build_default_registry",
    "build_op_chain",
    "fuseml_fused_kernel_placeholder",
    "next_power_of_2",
    "split_fusion_group",
    "validate_fusion_group",
    "validate_graph_control_flow",
]
