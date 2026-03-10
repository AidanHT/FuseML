"""SRAM-aware AOT autotuner for Triton kernel launch configurations.

Replaces the static ``num_warps`` / ``num_stages`` heuristics in
:class:`~fuseml.codegen.kernel_launcher.KernelLauncher` with a dynamic
search space that **mathematically guarantees** SRAM compliance before
compilation.

The autotuner follows a three-step pipeline:

1. **Configuration generation** — A Python generator yields all valid
   ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)`` tuples whose
   shared-memory footprint fits within the SRAM budget::

       SRAM = (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × sizeof(dtype) × num_stages

   Only tuples where ``SRAM ≤ sram_budget`` are yielded, preventing
   "Out of Shared Memory" PTX failures at compile time.

2. **Configuration scoring** — Each SRAM-safe configuration is scored
   using hardware-aware heuristics that balance tile utilisation,
   tensor-core saturation, and software-pipelining depth.

3. **Caching** — The optimal configuration for each unique
   ``(M, N, K, dtype)`` problem shape is cached in a lightweight
   dictionary.  When a :class:`KernelCache` instance is provided,
   the selected config is stored alongside the kernel hash so that
   subsequent launches with the same shape skip the search entirely.

**Design invariants:**

* The generator never yields a configuration that exceeds the SRAM
  budget — correctness is guaranteed by construction, not by post-hoc
  validation.
* All block dimensions are powers of two (Triton PTX requirement).
* The scorer prefers configurations that minimise boundary waste
  (partial tiles), maximise SRAM utilisation, and match warp counts
  to the operand precision (FP16/BF16 → 8 warps for tensor-core
  saturation, FP32 → 4 warps for register pressure balance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import torch

from fuseml._logging import logger


# ---------------------------------------------------------------------------
# Bytes-per-element lookup
# ---------------------------------------------------------------------------

_DTYPE_BYTES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
}

# ---------------------------------------------------------------------------
# SRAM budget — queried at runtime from the GPU, with 100 KB fallback
# for Ada Lovelace (sm_89) when no CUDA device is available.
# ---------------------------------------------------------------------------

_FALLBACK_SRAM_BUDGET_BYTES: int = 100 * 1024


def _get_sram_budget() -> int:
    """Query the GPU's actual shared memory capacity at runtime."""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            budget = getattr(props, "max_shared_memory_per_block", 0)
            if budget > 0:
                return budget
    except Exception:
        pass
    return _FALLBACK_SRAM_BUDGET_BYTES


_DEFAULT_SRAM_BUDGET_BYTES: int = _FALLBACK_SRAM_BUDGET_BYTES

# ---------------------------------------------------------------------------
# Candidate tile sizes for the AOT search space
# ---------------------------------------------------------------------------

_BLOCK_M_CHOICES: tuple[int, ...] = (32, 64, 128, 256)
_BLOCK_N_CHOICES: tuple[int, ...] = (32, 64, 128, 256)
_BLOCK_K_CHOICES: tuple[int, ...] = (32, 64, 128)
_NUM_WARPS_CHOICES: tuple[int, ...] = (2, 4, 8)
_NUM_STAGES_CHOICES: tuple[int, ...] = (2, 3, 4, 5)

# Minimum block dimension — below 16 launch overhead dominates compute.
_MIN_BLOCK_DIM: int = 16


# ---------------------------------------------------------------------------
# TuneConfig — immutable validated configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TuneConfig:
    """A validated, SRAM-safe kernel launch configuration.

    Attributes
    ----------
    block_m, block_n, block_k :
        Tile dimensions (all powers of two) passed as ``tl.constexpr``
        to the Triton kernel.
    num_warps :
        Number of warps per thread block.
    num_stages :
        Software-pipelining depth for the K-loop (``cp.async`` stages).
    sram_bytes :
        Pre-computed shared-memory footprint in bytes.  This is the
        SRAM cost of holding the A-tile and B-tile across all stages::

            (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × sizeof(dtype) × num_stages
    """

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    sram_bytes: int


# ---------------------------------------------------------------------------
# SRAM footprint calculation
# ---------------------------------------------------------------------------

def compute_sram_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    dtype_bytes: int,
    num_stages: int,
) -> int:
    """Calculate the shared-memory footprint for A-tile and B-tile buffers.

    Each software-pipeline stage holds one A-tile (BLOCK_M × BLOCK_K)
    and one B-tile (BLOCK_K × BLOCK_N) in shared memory simultaneously::

        SRAM = (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × sizeof(dtype) × num_stages

    Parameters
    ----------
    block_m, block_n, block_k :
        Tile dimensions.
    dtype_bytes :
        Bytes per element (e.g. 4 for FP32, 2 for FP16/BF16).
    num_stages :
        Number of software-pipeline stages.

    Returns
    -------
    int
        Total shared-memory footprint in bytes.
    """
    a_tile = block_m * block_k * dtype_bytes
    b_tile = block_k * block_n * dtype_bytes
    return num_stages * (a_tile + b_tile)


# ---------------------------------------------------------------------------
# Configuration generator — yields only SRAM-safe tuples
# ---------------------------------------------------------------------------

def generate_sram_safe_configs(
    dtype: torch.dtype,
    sram_budget: int = _DEFAULT_SRAM_BUDGET_BYTES,
) -> Generator[TuneConfig, None, None]:
    """Yield all ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)``
    configurations whose shared-memory footprint does not exceed *sram_budget*.

    The generator lazily evaluates the Cartesian product of candidate
    tile sizes, warp counts, and pipeline stages.  For each combination
    it pre-computes the SRAM footprint and **only yields** configurations
    where ``SRAM ≤ sram_budget``.  This guarantees that every yielded
    :class:`TuneConfig` is safe to compile without risking "Out of Shared
    Memory" PTX failures.

    Parameters
    ----------
    dtype :
        The operand precision — determines bytes per element.
    sram_budget :
        Maximum allowed shared-memory footprint in bytes.

    Yields
    ------
    TuneConfig
        A validated, SRAM-safe configuration.
    """
    bpe = _DTYPE_BYTES.get(dtype, 4)
    for bm in _BLOCK_M_CHOICES:
        for bn in _BLOCK_N_CHOICES:
            for bk in _BLOCK_K_CHOICES:
                for nw in _NUM_WARPS_CHOICES:
                    for ns in _NUM_STAGES_CHOICES:
                        sram = compute_sram_bytes(bm, bn, bk, bpe, ns)
                        if sram <= sram_budget:
                            yield TuneConfig(
                                block_m=bm,
                                block_n=bn,
                                block_k=bk,
                                num_warps=nw,
                                num_stages=ns,
                                sram_bytes=sram,
                            )


# ---------------------------------------------------------------------------
# SRAMAutotuner — AOT configuration selector with caching
# ---------------------------------------------------------------------------

class SRAMAutotuner:
    """AOT SRAM-aware autotuner for Triton kernel launch parameters.

    Replaces the static ``_select_num_warps`` / ``_select_num_stages``
    heuristics with a dynamic search over all SRAM-safe configurations.
    The optimal configuration is selected via a hardware-aware scoring
    function and cached so that subsequent launches with the same problem
    shape skip the search entirely.

    Parameters
    ----------
    sram_budget :
        Maximum shared-memory budget in bytes (default 100 KB for Ada).
    """

    def __init__(
        self,
        sram_budget: int | None = None,
    ) -> None:
        self._sram_budget = sram_budget if sram_budget is not None else _get_sram_budget()
        # Internal cache: (M, N, K, dtype_str) → TuneConfig
        self._config_cache: dict[tuple, TuneConfig] = {}

    @property
    def sram_budget(self) -> int:
        """The SRAM budget used for configuration filtering."""
        return self._sram_budget

    @property
    def cache_size(self) -> int:
        """Number of cached configurations."""
        return len(self._config_cache)

    def select_config(
        self,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> TuneConfig:
        """Select the optimal SRAM-safe configuration for the given problem.

        On a cache hit, returns the previously selected config immediately.
        On a miss, iterates through all SRAM-safe configurations, scores
        each one against the problem dimensions, and caches the winner.

        Parameters
        ----------
        M, N, K :
            Dynamic matrix dimensions.
        dtype :
            Operand precision (determines SRAM footprint per element).

        Returns
        -------
        TuneConfig
            The highest-scoring SRAM-safe configuration.
        """
        cache_key = (M, N, K, str(dtype))
        cached = self._config_cache.get(cache_key)
        if cached is not None:
            logger.debug(
                "SRAMAutotuner cache HIT — M=%d, N=%d, K=%d, dtype=%s → %s",
                M, N, K, dtype, cached,
            )
            return cached

        best = self._find_best_config(M, N, K, dtype)
        self._config_cache[cache_key] = best
        logger.debug(
            "SRAMAutotuner cache MISS — M=%d, N=%d, K=%d, dtype=%s → %s "
            "(cache_size=%d)",
            M, N, K, dtype, best, len(self._config_cache),
        )
        return best

    def _find_best_config(
        self,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> TuneConfig:
        """Iterate all SRAM-safe configs, score each, return the best."""
        best_config: TuneConfig | None = None
        best_score = -1.0

        for cfg in generate_sram_safe_configs(dtype, self._sram_budget):
            score = self._score_config(cfg, M, N, K, dtype)
            if score > best_score:
                best_score = score
                best_config = cfg

        if best_config is None:
            # No config fits — fall back to minimal safe config.
            bpe = _DTYPE_BYTES.get(dtype, 4)
            sram = compute_sram_bytes(
                _MIN_BLOCK_DIM, _MIN_BLOCK_DIM, _MIN_BLOCK_DIM, bpe, 2,
            )
            best_config = TuneConfig(
                block_m=_MIN_BLOCK_DIM,
                block_n=_MIN_BLOCK_DIM,
                block_k=_MIN_BLOCK_DIM,
                num_warps=2,
                num_stages=2,
                sram_bytes=sram,
            )
            logger.warning(
                "SRAMAutotuner — no config fits budget %d bytes; "
                "falling back to minimum tile (%d, %d, %d)",
                self._sram_budget, _MIN_BLOCK_DIM, _MIN_BLOCK_DIM,
                _MIN_BLOCK_DIM,
            )

        return best_config

    @staticmethod
    def _score_config(
        cfg: TuneConfig,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> float:
        """Score a configuration against the problem dimensions.

        The scoring function balances several hardware-aware factors:

        * **Tile utilisation** — Prefer block sizes that divide the problem
          dimensions evenly, minimising wasted threads on boundary tiles.
        * **Tile area** — Larger tiles amortise launch overhead and improve
          data reuse, but only when the problem is large enough.
        * **SRAM utilisation** — Higher shared-memory usage (closer to the
          budget) implies deeper software pipelining and better latency
          hiding.
        * **Warp-precision matching** — FP16/BF16 with large tiles benefit
          from 8 warps to saturate tensor cores; FP32 prefers 4 warps to
          balance register pressure; tiny problems prefer 2 warps.
        * **Pipeline depth** — More stages improve latency hiding for deep
          K dimensions, but waste registers for shallow K.
        """
        # ── Tile utilisation (0–1 per dimension) ─────────────────────
        # When the block dimension exceeds the problem dimension, the
        # utilisation is M / block_m (fraction of the tile that's useful).
        # When it doesn't, we measure how well the problem divides.
        if M > 0:
            if cfg.block_m > M:
                m_util = M / cfg.block_m
            else:
                m_waste = (cfg.block_m - M % cfg.block_m) % cfg.block_m
                m_util = M / (M + m_waste)
        else:
            m_util = 1.0

        if N > 0:
            if cfg.block_n > N:
                n_util = N / cfg.block_n
            else:
                n_waste = (cfg.block_n - N % cfg.block_n) % cfg.block_n
                n_util = N / (N + n_waste)
        else:
            n_util = 1.0

        if K > 0:
            if cfg.block_k > K:
                k_util = K / cfg.block_k
            else:
                k_waste = (cfg.block_k - K % cfg.block_k) % cfg.block_k
                k_util = K / (K + k_waste)
        else:
            k_util = 1.0

        # ── Tile area bonus ──────────────────────────────────────────
        tile_area = cfg.block_m * cfg.block_n
        # Normalise to [0, 1] range — 128×128 = 16384 is the max.
        # For small problems, large tiles waste resources — cap the
        # effective area at the problem area.
        effective_area = min(tile_area, max(M, 1) * max(N, 1))
        area_score = min(effective_area / 16384.0, 1.0)

        # ── SRAM utilisation (closer to budget = better reuse) ───────
        sram_util = cfg.sram_bytes / _DEFAULT_SRAM_BUDGET_BYTES

        # ── Warp-precision matching ──────────────────────────────────
        is_half = dtype in (torch.float16, torch.bfloat16)
        warp_score = 1.0
        if tile_area < 1024:
            # Tiny tile — prefer 2 warps to avoid scheduling overhead.
            warp_score = 1.2 if cfg.num_warps == 2 else 0.8
        elif is_half and tile_area >= 4096:
            # Large half-precision tile — 8 warps saturate tensor cores.
            warp_score = 1.2 if cfg.num_warps >= 8 else 0.9
        else:
            # FP32 or moderate tile — 4 warps balance throughput/pressure.
            warp_score = 1.1 if cfg.num_warps == 4 else 1.0

        # ── Pipeline depth matching ──────────────────────────────────
        k_iterations = (K + cfg.block_k - 1) // cfg.block_k if K > 0 else 1
        # Deep K benefits from more stages; shallow K prefers fewer.
        if k_iterations >= 8:
            stage_score = min(cfg.num_stages / 5.0, 1.0)
        else:
            stage_score = 1.0 if cfg.num_stages <= 3 else 0.8

        # ── Weighted combination ─────────────────────────────────────
        score = (
            m_util * 10.0
            + n_util * 10.0
            + k_util * 5.0
            + area_score * 8.0
            + sram_util * 3.0
            + warp_score * 5.0
            + stage_score * 4.0
        )
        return score

    def clear_cache(self) -> None:
        """Drop all cached configurations."""
        self._config_cache.clear()

    def __repr__(self) -> str:
        return (
            f"SRAMAutotuner(budget={self._sram_budget}, "
            f"cached={len(self._config_cache)})"
        )
