"""Logging, timing, JSON export and grid-cell naming helpers."""
from __future__ import annotations

import json
import logging
import string
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROW_LETTERS = string.ascii_uppercase


def get_logger(name: str = "floodwing", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger (idempotent: no duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


@contextmanager
def timed(stage: str, timings: dict[str, float], logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager that records wall time of a stage into ``timings``."""
    start = time.perf_counter()
    if logger:
        logger.info("start: %s", stage)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[stage] = timings.get(stage, 0.0) + elapsed
        if logger:
            logger.info("done:  %s (%.1fs)", stage, elapsed)


def cell_name(index: int, cols: int) -> str:
    """Cell index -> name. Letter = row (A..), number = column (1..)."""
    r, c = divmod(int(index), cols)
    return f"{ROW_LETTERS[r]}{c + 1}"


def cell_index(name: str, rows: int, cols: int) -> int | None:
    """Cell name -> index, or None if out of range or malformed."""
    name = name.strip().upper().replace("-", "")
    if len(name) < 2 or name[0] not in ROW_LETTERS or not name[1:].isdigit():
        return None
    r = ROW_LETTERS.index(name[0])
    c = int(name[1:]) - 1
    if 0 <= r < rows and 0 <= c < cols:
        return r * cols + c
    return None


def cell_coords(rows: int, cols: int) -> np.ndarray:
    """(n, 2) array of (row, col) coordinates for each cell index."""
    return np.array([(i // cols, i % cols) for i in range(rows * cols)], dtype=float)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def save_json(obj: Any, path: Path | str) -> Path:
    """Write ``obj`` to ``path`` as indented JSON (handles NumPy types)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def finite_or_none(value: Any) -> float | None:
    """Convert to float, mapping NaN/inf/None to None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
