# FuseML: A JIT Graph Compiler for Deep Learning Operator Fusion

FuseML is a lightweight, Just-In-Time (JIT) deep learning compiler that intercepts standard PyTorch workloads, analyzes the computational graph, and automatically fuses memory-bound sequential operators into highly optimized, bare-metal OpenAI Triton kernels.

## The Problem: The Memory Wall

In modern AI inference, compute is rarely the bottleneck — memory bandwidth is. When running standard eager-mode PyTorch, each individual operation (e.g., a `Linear` layer, followed by a `GeLU` activation, followed by a `LayerNorm`) requires a separate GPU kernel launch. More critically, the GPU must write intermediate tensors back to the slow High Bandwidth Memory (HBM) and read them back into fast SRAM for the next step. This constant VRAM thrashing destroys performance.

## The Solution: FuseML

FuseML acts as a highly efficient translation layer between the PyTorch frontend and the GPU hardware. By applying classic compiler theory directly to neural network execution, FuseML keeps data locked in ultra-fast SRAM for as long as possible — minimizing HBM round-trips.

---

## Architecture & Pipeline

FuseML implements a ten-stage compilation pipeline:

```
PyTorch Module
     │
     ▼
1. Graph Capture        torch.fx tracing + ATen decomposition (aot_module_simplified)
     │
     ▼
2. Control Flow         Data-dependent branching detection (rejects unsafe graphs)
   Validation
     │
     ▼
3. Fusion Discovery     Greedy forward absorption from GEMM base nodes
     │
     ▼
4. Safety Checks        In-place mutation aliasing + view ancestry analysis
     │
     ▼
5. Graph Surgery        Placeholder insertion, downstream rewiring, dead-code elimination
     │
     ▼
6. Graph Cutting        Splits groups at unsupported ops to preserve maximum fusion
     │
     ▼
7. Code Generation      Triton kernel source (signature + K-loop + epilogue + store)
     │
     ▼
8. Caching              Per-kernel SHA-256 source hash (avoids recompilation)
     │
     ▼
9. Kernel Launching     Runtime dispatch with heuristic tuning + SRAM budget enforcement
     │
     ▼
10. Fallback Recovery   EagerFallbackGuard routes to PyTorch eager on Triton failures
     │
     ▼
Standard PyTorch Tensor (transparent to caller)
```

---

## Project Structure

```
FuseML/
├── fuseml/                            # Main package
│   ├── __init__.py                    # Public API re-exports
│   ├── _logging.py                    # Centralized logging ([FuseML] prefix)
│   ├── registry.py                    # SupportedOpsRegistry + build_default_registry()
│   ├── fusion_group.py                # FusionGroup dataclass
│   ├── compiler.py                    # FuseMLCompiler (torch.compile backend)
│   ├── passes/                        # Graph optimization passes
│   │   ├── __init__.py
│   │   ├── fusion_pass.py             # FuseMLFusionPass (discovery + surgery)
│   │   ├── control_flow_validation.py # Data-dependent control flow detection
│   │   ├── graph_cut.py               # Unsupported-op group splitting
│   │   └── mutation_safety.py         # In-place aliasing safety checks
│   └── codegen/                       # Triton kernel generation & execution
│       ├── __init__.py
│       ├── kernel_generator.py        # TritonKernelGenerator
│       ├── kernel_cache.py            # KernelCache + KernelCacheKey + TensorFingerprint
│       ├── kernel_launcher.py         # KernelLauncher (runtime dispatch)
│       └── eager_fallback.py          # EagerFallbackGuard
├── tests/                             # Test suite (mirrors source structure)
│   ├── conftest.py                    # Shared fixtures
│   ├── test_registry.py
│   ├── test_fusion_group.py
│   ├── test_pattern_matching.py
│   ├── test_graph_surgery.py
│   ├── test_mutation_safety.py
│   ├── test_control_flow_validation.py
│   ├── test_graph_cut.py
│   ├── test_get_attr_resolution.py
│   ├── test_shape_propagation.py
│   ├── test_kernel_generator.py
│   ├── test_kernel_launcher.py
│   ├── test_kernel_cache.py
│   ├── test_eager_fallback.py
│   ├── test_reduction_codegen.py
│   ├── test_compiler.py
│   └── test_end_to_end.py
├── requirements.txt
├── pytest.ini
└── CLAUDE.md
```

---

## Implemented Modules

### `fuseml/registry.py` — Op Eligibility Registry

`SupportedOpsRegistry` is an extensible map from ATen ops to semantic categories. `build_default_registry()` pre-loads the baseline set of fusible operators:

| Category | ATen Ops |
|----------|----------|
| Linear (GEMM) | `aten.addmm.default` |
| Elementwise | `relu`, `gelu`, `sigmoid`, `add.Tensor`, `mul.Tensor` |
| Reduction | `sum.dim_IntList`, `amax.default`, `mean.dim` |

---

### `fuseml/fusion_group.py` — Fusion Target Descriptor

`FusionGroup` is a dataclass carrying all metadata for a single fused operator sequence:

- `base_node` — first compute node (GEMM anchor)
- `fused_nodes` — absorbed nodes after the base
- `inputs` — external dependencies (become kernel inputs)
- `output_node` — final node whose result the kernel replaces
- `output_metadata` — shape, stride, dtype
- `intermediate_outputs` — escape nodes consumed outside the group
- `param_bindings` — resolved `nn.Parameter` / buffer tensors
- `node_args_snapshot` — pre-surgery node arguments (for epilogue introspection)

---

### `fuseml/compiler.py` — Compiler Entry Point

`FuseMLCompiler` is the `torch.compile` backend. It:

1. Receives the FX graph from TorchDynamo via `aot_module_simplified`
2. Runs control flow validation
3. Executes the fusion pass (discovery + surgery)
4. Replaces placeholder nodes with compiled `KernelLauncher` callables
5. Returns a fully executable function backed by Triton kernels

Key method: `_build_launcher(group)` — generates, compiles, caches, and wraps the Triton kernel for a single `FusionGroup`.

---

### `fuseml/passes/fusion_pass.py` — Discovery & Surgery

`FuseMLFusionPass` runs two phases:

**Phase 1 — Discovery (`_find_fusion_groups`):**

Performs greedy forward absorption starting from `aten.addmm` base nodes:

- **Absorbable ops**: `relu`, `gelu`, `sigmoid`, `add`, `mul` (elementwise)
- **Barrier ops**: `softmax`, `layer_norm`, `mm`, `bmm`, `convolution` (halt absorption)
- **Reduction ops**: `sum`, `amax`, `mean` (absorbed only when `keepdim=False`)
- **Transparent ops**: `view`, `reshape`, `unsqueeze` (absorbed silently, no codegen required)

Also performs:
- Escape-node analysis (identifies results consumed outside the group)
- In-place aliasing safety checks
- Parameter binding resolution (`get_attr` → live `nn.Parameter`)

**Phase 2 — Surgery (`_apply_surgery`):**

- Inserts `fuseml_fused_kernel_placeholder` nodes after each group's output node
- Rewires all downstream consumers to read from the placeholder
- Dead-code elimination in 4 phases: standard DCE → orphaned `get_attr` → orphaned `call_function` → final DCE
- Graph lint validation + recompile

---

### `fuseml/passes/control_flow_validation.py` — Control Flow Guard

Two-tier validation before any fusion attempt:

**Tier 1 — FX Graph Inspection:** Walks FX nodes for higher-order ops (`cond`, `while_loop`, `map_impl`), scalar extractions (`.item()`, `.tolist()`), and boolean reductions (`any`, `all`) feeding conditional branches.

**Tier 2 — Source AST Inspection:** Parses the original module's source via Python's `ast` module. Detects data-dependent `if/while/for` conditions that involve tensor reductions (`sum`, `mean`, `max`, `min`, etc.). Raises `ControlFlowError` on violations.

---

### `fuseml/passes/graph_cut.py` — Group Splitter

Handles cases where a `FusionGroup` contains operators not yet supported in Triton codegen.

- `validate_fusion_group(group)` — returns unsupported nodes
- `split_fusion_group(group)` — splits at the first unsupported op, returning an ordered list of `GraphSegment` objects:
  - `"fused"` segments: valid prefix/suffix that can be compiled
  - `"native"` segments: unsupported ops routed to PyTorch eager

This maximizes fusion coverage even when a full group cannot be compiled.

---

### `fuseml/passes/mutation_safety.py` — Aliasing Safety

Detects unsafe in-place ops within a fusion group:

- `is_safe_inplace(node, group_node_set)` — checks if the tensor mutated by an in-place op has external users (or is a view of a tensor with external users)
- `check_group_mutation_safety(group_nodes, group_node_set)` — batch validation returning `MutationFinding` diagnostics

Tracked in-place ops: `relu_`, `add_`, `mul_`, `sigmoid_`

---

### `fuseml/codegen/kernel_generator.py` — Triton Code Generator

`TritonKernelGenerator` dynamically writes Triton kernel source as a Python string. The generated kernel is structured in four sections:

**1. Signature & Pointers**
- `@triton.jit` decorator with `BLOCK_SIZE_M/N/K` constexpr parameters
- Pointer arithmetic for all 2-D matrix inputs, 1-D vector inputs, and output tensors
- Boundary-masked offset calculation (`offs_m`, `offs_n`, `offs_k`)

**2. K-Loop (GEMM)**
- Blocked matrix multiplication over the K dimension
- Accumulates in `fp32` for numerical stability
- Adds bias vector after GEMM (from `aten.addmm` semantics)

**3. Epilogue**
- Applies fused post-GEMM operations in sequence:

| PyTorch Op | Triton Translation |
|------------|--------------------|
| `relu` / `relu_` | `tl.where(acc > 0, acc, 0.0)` |
| `gelu` | Fast tanh approximation |
| `sigmoid` / `sigmoid_` | `1 / (1 + tl.exp(-acc))` |
| `add.Tensor` / `add_.Tensor` | Load residual tile, add to acc |
| `mul.Tensor` / `mul_.Tensor` | Element-wise multiply |
| `sum.dim_IntList` | `tl.sum(acc, axis)` + `tl.atomic_add` |
| `amax.default` | `tl.max(acc, axis)` + `tl.atomic_max` |
| `mean.dim` | `tl.sum()` + `tl.atomic_add` + post-kernel division |
| `view`, `reshape`, `unsqueeze` | No-op (transparent) |

**4. Store & Compilation**
- Boundary-masked `tl.store` to output pointer
- Writes full kernel source to a temporary `.py` file (required by Triton for source inspection)
- Imports via `importlib` (avoids `exec()` incompatibilities)
- Caches compiled kernels by SHA-256 hash of source (avoids recompilation)

---

### `fuseml/codegen/kernel_cache.py` — Compilation Cache

**`TensorFingerprint`** captures the full physical layout of a tensor:
- `shape`, `stride` — tile dimensions
- `storage_offset` — offset from storage base pointer (critical for views)
- `aligned` — `data_ptr() % 16 == 0` (enables vectorized loads)
- `dtype` — string representation
- `broadcast_dims` — tuple of bools (stride == 0 indicates broadcast)
- `tensor_subclass` — type name (`"Tensor"`, `"Parameter"`, etc.)

**`KernelCacheKey`** uniquely identifies a compiled kernel variant by:
- `op_chain` — operator topology string (e.g., `"aten.addmm.default->aten.gelu.default"`)
- `input_fingerprints` — one `TensorFingerprint` per input
- `output_shape`, `output_dtype`
- `device`

Different tensor layouts, storage offsets, or pointer alignments produce different keys — preventing OOB access from cached kernels being reused on incompatible tensor layouts.

**`KernelCache`** is an in-memory dictionary with `lookup()`, `store()`, and `clear()` operations.

---

### `fuseml/codegen/kernel_launcher.py` — Runtime Dispatch

`KernelLauncher` wraps a compiled `@triton.jit` function and handles all runtime concerns:

**Heuristic Launch Parameter Selection:**
- `num_warps`: 2 (tiny tiles), 4 (FP32), 8 (large FP16/BF16 tiles)
- `num_stages`: 2 (small K), 3–4 (large K, adjusted for precision)

**SRAM Capacity Enforcement:**
- Configured with a 48 KB budget (safe minimum across SM70/SM80/SM89)
- Iteratively halves block dimensions if the output tile exceeds budget

**Runtime Dispatch Steps:**
1. Negative-stride guard: materializes tensors with negative strides (Triton limitation)
2. Dimension extraction: M, K from left operand; N from right (supports batched shapes)
3. Output allocation:
   - Reduced outputs: zero-init for `sum`/`mean`, `-inf` init for `max`
   - Full outputs: uninitialized (faster)
   - Intermediate buffers: empty M×N tiles
4. Argument assembly: tensors → dimensions (M, N, K) → strides → block size constants
5. Kernel launch on `torch.cuda.current_stream()`
6. Post-kernel fixup: divides by N for `mean` reduction

Default tile dimensions: `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`

---

### `fuseml/codegen/eager_fallback.py` — Error Recovery

`EagerFallbackGuard` provides deterministic fallback from Triton to eager PyTorch:

**Recoverable Errors (fallback triggered):**
- `RuntimeError`: CUDA OOM, PTX failures, driver errors
- `triton.CompilationError`: Triton-specific compilation issues

**Non-Recoverable (re-raised):**
- `TypeError`, `ValueError`: Programming bugs
- `KeyboardInterrupt`, `SystemExit`: User/system signals

**Failure Handling Protocol:**
1. Clones inputs before Triton launch (pristine fallback data)
2. Device synchronization to flush partial kernel writes
3. Pre-allocated buffer cleanup (resizes corrupted tensors to size 0)
4. `torch.cuda.empty_cache()` on OOM
5. Routes input snapshots through the eager fallback callable
6. Logs failure with kernel signature and attempt count

---

## Supported Fusion Patterns

FuseML currently fuses the following operator chains when the base is `aten.addmm` (Linear):

```
Linear → ReLU
Linear → GeLU
Linear → Sigmoid
Linear → Add (residual)
Linear → Mul
Linear → GeLU → Add (residual)
Linear → ReLU → Sum (reduction)
Linear → GeLU → AMax
Linear → GeLU → Mean
```

View-like ops (`reshape`, `unsqueeze`, `view`) are absorbed transparently within any of the above chains.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Graph Capture | `torch.fx`, `aot_module_simplified` |
| Kernel Backend | OpenAI Triton (Windows: `triton-windows`) |
| Testing | `pytest`, `pytest-cov` |
| Linting | `black`, `flake8`, `mypy` |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```python
import torch
import torch.nn as nn
from fuseml import FuseMLCompiler

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

model = MLP(1024).cuda().half()
x = torch.randn(32, 1024, device="cuda", dtype=torch.float16)

# Compile with FuseML backend
compiled = torch.compile(model, backend=FuseMLCompiler())
out = compiled(x)  # Fused Linear→GeLU kernel runs here
```

---

## Development Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run only surgery tests
python -m pytest tests/ -m surgery -v

# Run a specific test file
python -m pytest tests/test_kernel_generator.py -v

# Smoke test (torch.compile backend)
python -m fuseml.compiler
```

---

## Testing

The test suite mirrors the source structure with ~7,700 lines of tests across 17 files:

| Test File | Coverage Area |
|-----------|--------------|
| `test_registry.py` | Op registration and lookup |
| `test_fusion_group.py` | FusionGroup dataclass invariants |
| `test_pattern_matching.py` | Greedy absorption, barriers, branching, chains |
| `test_graph_surgery.py` | Placeholder insertion, rewiring, dead-code elimination |
| `test_mutation_safety.py` | In-place aliasing checks |
| `test_control_flow_validation.py` | Data-dependent control flow detection (Tier 1 + 2) |
| `test_graph_cut.py` | Unsupported-op splitting and segment validation |
| `test_get_attr_resolution.py` | Parameter binding resolution |
| `test_shape_propagation.py` | Tensor metadata extraction |
| `test_kernel_generator.py` | Triton codegen: signature, K-loop, epilogue, compilation |
| `test_kernel_launcher.py` | Runtime dispatch, grid sizing, SRAM enforcement |
| `test_kernel_cache.py` | Cache key construction and hit/miss behavior |
| `test_eager_fallback.py` | Error recovery and fallback routing |
| `test_reduction_codegen.py` | Reduction ops with atomic cross-thread sync |
| `test_compiler.py` | Full pipeline integration |
| `test_end_to_end.py` | MLP and Transformer block end-to-end correctness |

All fused kernels are validated against PyTorch eager execution using `torch.allclose(atol=1e-3, rtol=1e-3)`.

---

## Design Principles

- **Hardware Sympathy First**: Triton templates are built around GPU memory hierarchies. SRAM budgets are enforced at runtime. Stride-parameterized pointer arithmetic handles transposed, sliced, and broadcast tensors natively without forced `.contiguous()`.
- **Separation of Concerns**: Pattern matching is entirely separate from Triton code generation. Each pipeline stage lives in its own module.
- **Correctness Before Performance**: Every fused kernel is validated against eager PyTorch output. The `EagerFallbackGuard` ensures production reliability.
- **Zero-Copy by Default**: Storage offsets and pointer alignment are tracked in the cache key. Negative-stride tensors are materialized only when Triton requires it.
