"""bench_mlp_block.py -- Latency & memory-bandwidth benchmark for a Transformer MLP block.

Compares two execution modes:
  1. Eager     -- standard PyTorch model(x)
  2. FuseML    -- torch.compile(model, backend=FuseMLCompiler())

Metrics reported:
  - Latency (ms)           via torch.utils.benchmark.Timer or CUDA events
  - HBM Traffic (MB)       analytical byte-traffic model
  - Memory Throughput (GB/s)  traffic / latency

Usage:
    python benchmarks/bench_mlp_block.py
    python benchmarks/bench_mlp_block.py --preset small --no-plot
    python benchmarks/bench_mlp_block.py --precise --save-plot results.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure the project root is on sys.path so ``import fuseml`` works when
# running the script directly (e.g. ``python benchmarks/bench_mlp_block.py``).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.benchmark import Timer

# ---------------------------------------------------------------------------
# Constants (defaults -- all overridable via CLI)
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 8
SEQ_LEN: int = 2048
D_MODEL: int = 4096
D_INTERMEDIATE: int = 16384

DTYPE = torch.bfloat16

WARMUP_ITERS: int = 50
MEASURE_ITERS: int = 200
MIN_RUN_TIME: float = 5.0  # seconds, for Timer.blocked_autorange

# Number of GPU warmup iterations run BEFORE any benchmark.  These force
# the GPU clocks to ramp up and stabilise, preventing cold-start artifacts
# (thermal throttling, DVFS ramp, driver init) from contaminating the
# first measured mode.  20 dummy matmuls at full problem size is enough to
# trigger boost clocks on laptop GPUs (RTX 4050/4060/4070); 25 is conservative.
GPU_WARMUP_ITERS: int = 25

L2_FLUSH_SIZE_BYTES: int = 128 * 1024 * 1024  # 128 MB (exceeds all known GPU L2 caches)

COLORS: dict[str, str] = {
    "Eager": "#4C72B0",
    "FuseML": "#55A868",
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TransformerMLP(nn.Module):
    """Transformer MLP block: Linear -> GeLU -> Linear -> Add(residual).

    FuseML produces two fusion groups for this pattern:
      - Group 1: addmm + gelu  (Linear1 + GeLU fused in SRAM)
      - Group 2: addmm + add   (Linear2 + residual add fused in SRAM)
    """

    def __init__(self, d_model: int, d_intermediate: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_intermediate)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_intermediate, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# HBM traffic model
# ---------------------------------------------------------------------------


def _bytes_per_element(dtype: torch.dtype) -> int:
    """Return bytes per element for the given dtype."""
    return {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}[dtype]


@dataclass
class HBMTrafficEstimate:
    """Theoretical HBM byte-traffic estimate for one forward pass."""

    mode: str
    total_bytes: int
    breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


def _fuseml_breakdown(M: int, D: int, I: int, bpe: int) -> dict[str, int]:
    """HBM traffic breakdown for two fused kernels (addmm+gelu, addmm+add)."""
    return {
        "K1_read_x": M * D * bpe,
        "K1_read_W1": D * I * bpe,
        "K1_read_bias1": I * bpe,
        "K1_write_gelu_out": M * I * bpe,
        "K2_read_gelu_out": M * I * bpe,
        "K2_read_W2": I * D * bpe,
        "K2_read_bias2": D * bpe,
        "K2_read_residual": M * D * bpe,
        "K2_write_final": M * D * bpe,
    }


def compute_hbm_traffic(
    M: int,
    d_model: int,
    d_intermediate: int,
    dtype: torch.dtype,
    mode: str,
) -> HBMTrafficEstimate:
    """Compute theoretical HBM traffic for one forward pass.

    Parameters
    ----------
    M : int
        Effective batch dimension (batch_size * seq_len).
    d_model : int
        Model dimension.
    d_intermediate : int
        Intermediate (expanded) dimension.
    dtype : torch.dtype
        Element precision.
    mode : str
        One of ``"eager"``, ``"inductor"``, ``"fuseml"``.
    """
    bpe = _bytes_per_element(dtype)
    D, I = d_model, d_intermediate

    if mode == "eager":
        breakdown = {
            # Linear1 (addmm)
            "Linear1_read_x": M * D * bpe,
            "Linear1_read_W1": D * I * bpe,
            "Linear1_read_bias1": I * bpe,
            "Linear1_write": M * I * bpe,
            # GeLU (separate kernel -- reads intermediate from HBM, writes back)
            "GeLU_read": M * I * bpe,
            "GeLU_write": M * I * bpe,
            # Linear2 (addmm)
            "Linear2_read_gelu": M * I * bpe,
            "Linear2_read_W2": I * D * bpe,
            "Linear2_read_bias2": D * bpe,
            "Linear2_write": M * D * bpe,
            # Add (separate kernel -- reads both operands from HBM)
            "Add_read_linear2": M * D * bpe,
            "Add_read_residual": M * D * bpe,
            "Add_write": M * D * bpe,
        }
    elif mode == "fuseml":
        # FuseML fuses elementwise post-ops into GEMM epilogues.
        breakdown = _fuseml_breakdown(M, D, I, bpe)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    total = sum(breakdown.values())
    return HBMTrafficEstimate(mode=mode, total_bytes=total, breakdown=breakdown)


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Stores measured and computed results for one execution mode."""

    mode: str
    latency_ms: float
    iqr_ms: float
    hbm_traffic: HBMTrafficEstimate
    throughput_gb_s: float


# ---------------------------------------------------------------------------
# L2 cache flush
# ---------------------------------------------------------------------------


def flush_l2_cache(device: torch.device) -> None:
    """Write a 128 MB tensor to GPU to evict L2 cache contents.

    128 MB exceeds all known GPU L2 sizes (A100=40MB, H100=50MB, RTX4090=72MB).
    """
    n_elements = L2_FLUSH_SIZE_BYTES // 4  # float32 = 4 bytes
    buf = torch.empty(n_elements, dtype=torch.float32, device=device)
    buf.fill_(1.0)
    del buf
    torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------


def validate_correctness(
    eager_model: nn.Module,
    compiled_models: dict[str, nn.Module],
    x: torch.Tensor,
) -> None:
    """Verify all backends produce numerically identical output.

    Uses ``torch.allclose`` per CLAUDE.md validation protocol.
    Tolerances are dtype-aware: bfloat16 has only 7 mantissa bits
    (precision ~0.008 near 1.0), so chained matmul+activation+matmul+add
    naturally accumulates ~0.02 rounding error.
    """
    # Dtype-appropriate tolerances
    _tol = {
        torch.float32: (1e-3, 1e-3),
        torch.float16: (1e-2, 1e-2),
        torch.bfloat16: (2e-2, 1e-2),
    }
    atol, rtol = _tol.get(x.dtype, (1e-3, 1e-3))

    with torch.no_grad():
        ref = eager_model(x)
        for name, model in compiled_models.items():
            out = model(x)
            if not torch.allclose(ref, out, atol=atol, rtol=rtol):
                max_diff = (ref - out).abs().max().item()
                raise AssertionError(
                    f"Correctness check FAILED for {name}: "
                    f"max_diff={max_diff:.6f} exceeds tolerance "
                    f"(atol={atol}, rtol={rtol})."
                )
            max_diff = (ref - out).abs().max().item()
            print(f"  [PASS] {name} matches eager (max_diff={max_diff:.6f}, atol={atol}, rtol={rtol})")


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def _progress_bar(current: int, total: int, *, width: int = 28) -> str:
    """Return an inline ASCII progress bar: [████░░░░]  42/200  21%."""
    frac = current / total if total > 0 else 0.0
    filled = int(width * frac)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = int(100 * frac)
    pad = len(str(total))
    return f"[{bar}] {current:>{pad}}/{total}  {pct:3d}%"


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def benchmark_mode(
    label: str,
    model: nn.Module,
    x: torch.Tensor,
    hbm_traffic: HBMTrafficEstimate,
    *,
    use_timer: bool = True,
    warmup_iters: int = WARMUP_ITERS,
    measure_iters: int = MEASURE_ITERS,
) -> BenchmarkResult:
    """Measure latency and compute throughput for a single execution mode.

    Two measurement paths:
      - **Timer** (default): ``torch.utils.benchmark.Timer.blocked_autorange``
        for steady-state latency with built-in outlier rejection.
      - **Precise** (``use_timer=False``): manual CUDA-event loop with
        per-iteration L2 cache flush for cache-cold measurements.
    """
    device = x.device

    if use_timer:
        latency_ms, iqr_ms = _bench_timer(label, model, x)
    else:
        latency_ms, iqr_ms = _bench_precise(model, x, device, warmup_iters, measure_iters)

    throughput_gb_s = (hbm_traffic.total_bytes / 1e9) / (latency_ms / 1e3)

    return BenchmarkResult(
        mode=label,
        latency_ms=latency_ms,
        iqr_ms=iqr_ms,
        hbm_traffic=hbm_traffic,
        throughput_gb_s=throughput_gb_s,
    )


def _bench_timer(label: str, model: nn.Module, x: torch.Tensor) -> tuple[float, float]:
    """Steady-state measurement via ``torch.utils.benchmark.Timer``.

    Uses ``torch.no_grad()`` to match the compilation context and prevent
    TorchDynamo from triggering a guard-mismatch recompilation that would
    produce different fusion results.
    """
    timer = Timer(
        stmt="with torch.no_grad(): model(x)",
        globals={"model": model, "x": x, "torch": torch},
        num_threads=1,
        label="MLP Block",
        sub_label=label,
    )
    print(
        f"    [timer] blocked_autorange  (min {MIN_RUN_TIME:.0f}s, "
        f"outlier rejection enabled)...",
        end="",
        flush=True,
    )
    t0 = time.perf_counter()
    measurement = timer.blocked_autorange(min_run_time=MIN_RUN_TIME)
    elapsed = time.perf_counter() - t0
    total_iters = measurement.number_per_run * len(measurement.times)
    print(f"  done  ({elapsed:.1f}s elapsed, {total_iters} iters)")
    latency_ms = measurement.median * 1e3
    iqr_ms = measurement.iqr * 1e3
    return latency_ms, iqr_ms


def _bench_precise(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    measure_iters: int,
) -> tuple[float, float]:
    """Cache-cold measurement with per-iteration L2 flush and CUDA events."""
    # Warmup — GPU ops are async so we sync once at the end rather than per iter.
    print(f"    [warmup]  {warmup_iters} iters...", end="", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(warmup_iters):
            model(x)
        torch.cuda.synchronize(device)
    print(f"  done  ({time.perf_counter() - t0:.1f}s)")

    # Measurement — each iter syncs the device, so the progress bar reflects
    # true GPU progress.
    latencies_ms: list[float] = []
    sys.stdout.write(f"\r    [measure] {_progress_bar(0, measure_iters)}")
    sys.stdout.flush()
    t0 = time.perf_counter()

    for i in range(measure_iters):
        flush_l2_cache(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            model(x)
        end.record()

        torch.cuda.synchronize(device)
        latencies_ms.append(start.elapsed_time(end))  # milliseconds

        sys.stdout.write(f"\r    [measure] {_progress_bar(i + 1, measure_iters)}")
        sys.stdout.flush()

    elapsed = time.perf_counter() - t0
    print(f"  done  ({elapsed:.1f}s)")

    t = torch.tensor(latencies_ms)
    latency_ms = t.median().item()
    q1 = t.quantile(0.25).item()
    q3 = t.quantile(0.75).item()
    iqr_ms = q3 - q1
    return latency_ms, iqr_ms


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_header(args: argparse.Namespace, M: int, gpu_name: str) -> None:
    """Print the benchmark configuration header."""
    sep = "=" * 64
    print(sep)
    print("  FuseML Benchmark -- Transformer MLP Block")
    print(
        f"  Config: batch={args.batch_size}, seq={args.seq_len}, "
        f"d_model={args.d_model}, d_inter={args.d_intermediate}"
    )
    print(f"  dtype={args.dtype}  |  M={M}  |  GPU: {gpu_name}")
    mode_str = "precise (per-iter L2 flush)" if args.precise else "Timer (steady-state)"
    print(f"  Warmup: {args.warmup}  |  Measurement: {args.iters}  |  Mode: {mode_str}")
    print(sep)


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted ASCII table of benchmark results."""
    header = (
        f"+{'':->14}+{'':->14}+{'':->10}+{'':->12}+{'':->12}+\n"
        f"| {'Mode':<12} | {'Latency (ms)':>12} | {'IQR (ms)':>8} | {'HBM* (MB)':>10} | {'BW* (GB/s)':>10} |\n"
        f"+{'':->14}+{'':->14}+{'':->10}+{'':->12}+{'':->12}+"
    )
    print(f"\n{header}")
    for r in results:
        print(
            f"| {r.mode:<12} "
            f"| {r.latency_ms:>12.2f} "
            f"| {r.iqr_ms:>8.2f} "
            f"| {r.hbm_traffic.total_mb:>10.1f} "
            f"| {r.throughput_gb_s:>10.1f} |"
        )
    print(f"+{'':->14}+{'':->14}+{'':->10}+{'':->12}+{'':->12}+")
    print("  * HBM/BW are analytical estimates, not measured.  Use `ncu` for actual DRAM traffic.")


def print_speedup_summary(results: list[BenchmarkResult]) -> None:
    """Print speedup ratios and HBM savings relative to eager mode."""
    eager = next((r for r in results if r.mode == "Eager"), None)
    if eager is None:
        return

    # Speedup
    parts: list[str] = []
    for r in results:
        if r.mode == "Eager":
            continue
        speedup = eager.latency_ms / r.latency_ms
        parts.append(f"{r.mode} {speedup:.2f}x")
    if parts:
        print(f"\nSpeedup vs Eager:  {'  |  '.join(parts)}")

    # HBM savings (compare first non-eager mode that has different traffic)
    for r in results:
        if r.mode == "Eager":
            continue
        saved_mb = eager.hbm_traffic.total_mb - r.hbm_traffic.total_mb
        if saved_mb > 0:
            pct = saved_mb / eager.hbm_traffic.total_mb * 100
            print(
                f"HBM Savings vs Eager: {saved_mb:.1f} MB eliminated "
                f"({pct:.1f}% reduction)"
            )
            break


# ---------------------------------------------------------------------------
# Matplotlib chart
# ---------------------------------------------------------------------------


def plot_results(results: list[BenchmarkResult], save_path: str | None = None) -> None:
    """Generate a 3-subplot bar chart comparing execution modes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed -- skipping plot generation.")
        return

    modes = [r.mode for r in results]
    colors = [COLORS.get(m, "#888888") for m in modes]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "FuseML Benchmark: Transformer MLP Block",
        fontsize=14,
        fontweight="bold",
    )

    # -- Subplot 1: Latency --
    ax = axes[0]
    latencies = [r.latency_ms for r in results]
    iqrs = [r.iqr_ms for r in results]
    bars = ax.bar(modes, latencies, color=colors, yerr=iqrs, capsize=5)
    ax.set_title("Latency (ms)\nlower is better", fontsize=11)
    ax.set_ylabel("ms")
    for bar, val in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # -- Subplot 2: HBM Traffic --
    ax = axes[1]
    traffic = [r.hbm_traffic.total_mb for r in results]
    bars = ax.bar(modes, traffic, color=colors)
    ax.set_title("HBM Traffic (MB)\nlower is better", fontsize=11)
    ax.set_ylabel("MB")
    for bar, val in zip(bars, traffic):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # -- Subplot 3: Memory Throughput --
    ax = axes[2]
    bws = [r.throughput_gb_s for r in results]
    bars = ax.bar(modes, bws, color=colors)
    ax.set_title("Effective BW (GB/s)\nhigher is better", fontsize=11)
    ax.set_ylabel("GB/s")
    for bar, val in zip(bars, bws):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FuseML Transformer MLP block vs Eager",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--d-intermediate", type=int, default=D_INTERMEDIATE)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=MEASURE_ITERS)
    parser.add_argument(
        "--precise",
        action="store_true",
        help="Use manual CUDA-event loop with per-iteration L2 cache flush",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib chart")
    parser.add_argument("--save-plot", type=str, default=None, help="Save chart to file")
    parser.add_argument("--skip-fuseml", action="store_true", help="Skip FuseML mode")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["large", "medium", "small"],
        default=None,
        help=(
            "Dimension presets: 'large' (default, compute-bound GEMMs — "
            "cuBLAS wins), 'medium' (balanced), 'small' (memory-bound — "
            "fusion wins).  Overrides --batch-size/--seq-len/--d-model/"
            "--d-intermediate."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _gpu_warmup(model: nn.Module, x: torch.Tensor, *, iters: int = 20) -> None:
    """Run *iters* forward passes to force GPU clock ramp-up.

    Laptop GPUs (and desktop GPUs with power limits) start at low base
    clocks and only reach boost frequency after sustained compute load.
    Running warmup iterations before any Timer measurement ensures all
    benchmark modes see the same stable clock speed.
    """
    with torch.no_grad():
        for _ in range(iters):
            model(x)
    torch.cuda.synchronize(x.device)


def main() -> None:
    args = parse_args()

    # -- Apply dimension preset (overrides individual CLI args) --
    _PRESETS = {
        "small":  {"batch_size": 1, "seq_len": 128,  "d_model": 256,  "d_intermediate": 1024},
        "medium": {"batch_size": 4, "seq_len": 512,  "d_model": 1024, "d_intermediate": 4096},
        "large":  {"batch_size": 8, "seq_len": 2048, "d_model": 4096, "d_intermediate": 16384},
    }
    if args.preset is not None:
        for k, v in _PRESETS[args.preset].items():
            setattr(args, k, v)

    # -- Resolve dtype --
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # -- GPU check --
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)

    # -- bfloat16 fallback --
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("WARNING: bfloat16 not supported on this GPU. Falling back to float16.")
        dtype = torch.float16
        args.dtype = "float16"

    M = args.batch_size * args.seq_len

    # -- Header --
    print_header(args, M, gpu_name)

    # -- Allocate model & input --
    try:
        model = TransformerMLP(args.d_model, args.d_intermediate)
        model = model.to(device=device, dtype=dtype).eval()
        x = torch.randn(M, args.d_model, device=device, dtype=dtype)
    except torch.cuda.OutOfMemoryError:
        print("ERROR: CUDA out of memory. Reduce --batch-size or --d-intermediate.")
        sys.exit(1)

    # -- Build compiled models --
    compiled_models: dict[str, nn.Module] = {"Eager": model}

    # FuseML
    if not args.skip_fuseml:
        try:
            from fuseml import FuseMLCompiler

            fuseml_model = torch.compile(model, backend=FuseMLCompiler())
            print("\nTriggering FuseML compilation (first forward pass)...", end="", flush=True)
            t0 = time.perf_counter()
            with torch.no_grad():
                fuseml_model(x)  # trigger compilation
            print(f"  done  ({time.perf_counter() - t0:.1f}s)")
            compiled_models["FuseML"] = fuseml_model
        except Exception as exc:
            print(f"WARNING: FuseML compilation failed: {exc}")

    # -- GPU warmup (clock stabilisation) --
    # Laptop GPUs (RTX 4050/4060/4070) use aggressive DVFS — the GPU
    # starts at low clocks and only ramps to boost frequency after
    # sustained load.  Without warmup, the first benchmark mode (Eager)
    # runs at lower clocks and reports anomalously high latency, creating
    # a false "speedup" for later modes.
    #
    # We warm up ALL compiled models, not just Eager, so that every
    # backend's first Timer invocation sees stabilised clocks.
    n_modes = len(compiled_models)
    print(f"\nGPU Warmup (stabilising clocks)  [{n_modes} model(s), {GPU_WARMUP_ITERS} iters each]:")
    for idx, (name, warmup_model) in enumerate(compiled_models.items(), 1):
        print(f"  [{idx}/{n_modes}] {name}...", end="", flush=True)
        t0 = time.perf_counter()
        _gpu_warmup(warmup_model, x, iters=GPU_WARMUP_ITERS)
        print(f"  done  ({time.perf_counter() - t0:.1f}s)")

    # -- Correctness validation --
    print("\nCorrectness Validation:")
    others = {k: v for k, v in compiled_models.items() if k != "Eager"}
    if others:
        validate_correctness(model, others, x)
    else:
        print("  (no compiled backends available -- skipping)")

    # -- Compute HBM traffic estimates --
    traffic: dict[str, HBMTrafficEstimate] = {}
    for mode_name in compiled_models:
        mode_key = mode_name.lower()
        traffic[mode_name] = compute_hbm_traffic(M, args.d_model, args.d_intermediate, dtype, mode_key)

    # -- Print HBM traffic breakdown (always shown, even with only Eager) --
    eager_traffic = traffic.get("Eager")
    fused_traffic_est = compute_hbm_traffic(M, args.d_model, args.d_intermediate, dtype, "fuseml")
    if eager_traffic:
        saved_mb = eager_traffic.total_mb - fused_traffic_est.total_mb
        print(f"\nHBM Traffic Estimate (analytical model — NOT measured):")
        print(f"  These numbers are computed from matrix dimensions and dtype,")
        print(f"  not from hardware performance counters.  Use NVIDIA Nsight")
        print(f"  Compute (ncu) for actual DRAM traffic measurement.")
        print(f"  Eager (4 kernels):  {eager_traffic.total_mb:,.1f} MB")
        print(f"  Fused (2 kernels):  {fused_traffic_est.total_mb:,.1f} MB")
        print(f"  Savings:            {saved_mb:,.1f} MB ({saved_mb / eager_traffic.total_mb * 100:.1f}% reduction)")

    # -- Run benchmarks --
    n_modes = len(compiled_models)
    method = "precise (CUDA events)" if args.precise else f"timer (min {MIN_RUN_TIME:.0f}s)"
    print(f"\nRunning benchmarks  [{n_modes} mode(s), method: {method}]:")
    results: list[BenchmarkResult] = []
    for idx, (mode_name, mode_model) in enumerate(compiled_models.items(), 1):
        print(f"\n  [{idx}/{n_modes}] {mode_name}")
        t0 = time.perf_counter()
        result = benchmark_mode(
            label=mode_name,
            model=mode_model,
            x=x,
            hbm_traffic=traffic[mode_name],
            use_timer=not args.precise,
            warmup_iters=args.warmup,
            measure_iters=args.iters,
        )
        elapsed = time.perf_counter() - t0
        results.append(result)
        print(
            f"  -> {result.latency_ms:.3f} ms median  "
            f"(IQR: {result.iqr_ms:.3f} ms, total: {elapsed:.1f}s)"
        )

    # -- Results --
    print_results_table(results)
    print_speedup_summary(results)

    # -- Plot --
    if not args.no_plot:
        plot_results(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()
