"""FuseML — JIT deep learning compiler for fusing memory-bound operators.

Public API re-exports for convenient access::

    from fuseml import FuseMLFusionPass, FusionGroup, SupportedOpsRegistry
"""

from fuseml.codegen.kernel_generator import TensorDescriptor, TritonKernelGenerator
from fuseml.compiler import FuseMLCompiler
from fuseml.fusion_group import FusionGroup
from fuseml.passes.control_flow_validation import (
    ControlFlowError,
    validate_graph_control_flow,
)
from fuseml.passes.fusion_pass import FuseMLFusionPass, fuseml_fused_kernel_placeholder
from fuseml.registry import SupportedOpsRegistry, build_default_registry

__all__ = [
    "ControlFlowError",
    "FuseMLCompiler",
    "FuseMLFusionPass",
    "FusionGroup",
    "SupportedOpsRegistry",
    "TensorDescriptor",
    "TritonKernelGenerator",
    "build_default_registry",
    "fuseml_fused_kernel_placeholder",
    "validate_graph_control_flow",
]
