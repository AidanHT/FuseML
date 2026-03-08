"""Centralized logging configuration for FuseML."""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[FuseML] %(levelname)s — %(message)s",
)

logger = logging.getLogger("fuseml")
