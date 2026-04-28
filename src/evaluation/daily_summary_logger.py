"""Compact daily summary logger for KPI source-of-truth.

Writes one JSONL row per finalized scheduling day. The payload is expected to be
aggregate-only (no per-task records) so it stays small and directly consumable by
`src/evaluation/kpis.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DailySummaryLogger:
    """Append-only JSONL logger for per-day KPI aggregates."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[Any] = None
        self._row_index = 0

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = open(self.output_path, "a", encoding="utf-8")
            logger.info("Daily summary logger writing to: %s", self.output_path)

    def log_day(self, summary: Dict[str, Any]) -> None:
        """Append one finalized day summary as a single JSON line."""
        self._ensure_open()
        line = json.dumps(summary, separators=(",", ":"), default=str)
        self._file.write(line + "\n")
        self._file.flush()
        self._row_index += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info(
                "Daily summary logger closed (%d rows written to %s)",
                self._row_index,
                self.output_path,
            )

    def __del__(self) -> None:
        self.close()
