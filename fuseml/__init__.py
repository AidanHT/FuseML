"""FuseML — JIT deep learning compiler for fusing memory-bound operators.

Public API re-exports for convenient access::

    from fuseml import FuseMLFusionPass, FusionGroup, SupportedOpsRegistry
"""

from fuseml.compiler import FuseMLCompiler
from fuseml.fusion_group import FusionGroup
from fuseml.passes.fusion_pass import FuseMLFusionPass, fuseml_fused_kernel_placeholder
from fuseml.registry import SupportedOpsRegistry, build_default_registry

__all__ = [
    "FuseMLCompiler",
    "FuseMLFusionPass",
    "FusionGroup",
    "SupportedOpsRegistry",
    "build_default_registry",
    "fuseml_fused_kernel_placeholder",
]
