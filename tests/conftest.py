"""Shared test fixtures and tracing helpers for FuseML tests."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import ShapeProp

from fuseml import FuseMLFusionPass, FusionGroup


def trace_no_grad(model: nn.Module, x: torch.Tensor) -> torch.fx.GraphModule:
    """Trace *model* at aten level with gradients disabled."""
    with torch.no_grad():
        gm = make_fx(lambda inp: model(inp))(x)
    return gm


def trace_fn_no_grad(fn, x: torch.Tensor) -> torch.fx.GraphModule:
    """Trace a plain function at aten level with gradients disabled."""
    with torch.no_grad():
        gm = make_fx(fn)(x)
    return gm


def find_groups(gm: torch.fx.GraphModule) -> list[FusionGroup]:
    """Run pattern matching only (no surgery) and return discovered groups."""
    return FuseMLFusionPass(gm)._find_fusion_groups()


def find_groups_with_shapes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[torch.Tensor, ...],
) -> list[FusionGroup]:
    """Run shape propagation + pattern matching, return groups with metadata."""
    fuse_pass = FuseMLFusionPass(gm)
    ShapeProp(gm).propagate(*example_inputs)
    return fuse_pass._find_fusion_groups()


def run_surgery(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.fx.GraphModule, list[FusionGroup]]:
    """Run full fusion pass (discovery + surgery) and return (modified gm, groups).

    When *example_inputs* is provided, shape propagation runs first so
    ``FusionGroup.output_metadata`` and ``node.meta['tensor_meta']`` are
    populated.
    """
    fuse_pass = FuseMLFusionPass(gm)
    if example_inputs is not None:
        ShapeProp(gm).propagate(*example_inputs)
    groups = fuse_pass._find_fusion_groups()
    if groups:
        fuse_pass._apply_surgery(groups)
    return gm, groups
