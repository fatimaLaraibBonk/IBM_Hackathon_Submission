"""Field-report agent: turns plain-text reports into structured heatmap updates.

Default: deterministic regex + keyword parser (no network, no API keys).
Optional: ``LocalLLMParser`` behind ``Config.use_local_llm`` (default OFF). It
emits the same schema, is validated by the same checker, and falls back to the
rule-based parser on any malformed output or load failure.

Cell naming: letter = row (A..), number = column (1..). On a 3x4 grid the
valid cells are A1..C4.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import Config
from .utils import cell_index, cell_name, get_logger

ACTIONS = ("boost", "block", "reduce")

# Longer phrases win over shorter overlapping ones (e.g. "no survivors" beats "survivors").
KEYWORDS: dict[str, tuple[str, ...]] = {
    "boost": ("survivor", "survivors", "spotted", "stranded", "rooftop", "roof", "boat", "trapped",
              "waving", "sos", "distress", "people", "family", "calling for help", "signal"),
    "block": ("collapsed", "collapse", "impassable", "bridge down", "no-fly", "no fly", "power lines",
              "fire", "unsafe", "gas leak"),
    "reduce": ("cleared", "evacuated", "empty", "searched", "no one", "nobody", "all clear",
               "rescued", "no survivors", "no people"),
}
ACTION_PRIORITY = {"block": 0, "boost": 1, "reduce": 2}

_CELL_RE = re.compile(r"\b([A-Z])-?(\d{1,2})\b")
_CELL_CTX_RE = re.compile(r"\b(?:cell|sector|grid|square)\s+([a-zA-Z])-?(\d{1,2})\b", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"[.;!?\n]+")

SAMPLE_REPORTS: tuple[str, ...] = (
    "Family of four waving from a rooftop near C3. Small boat seen drifting at B4.",
    "Bridge down at A2, area is impassable. Power lines across B1, declare no-fly.",
    "Sector C1 has been evacuated and cleared by ground teams.",
    "Survivors stranded on a roof at A4; the building at B3 has collapsed.",
    "SOS signal from cell b2. Also unconfirmed chatter about Z9 and sector D5.",
)


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    action: str
    phrase: str


def _keyword_matches(clause: str) -> list[_Match]:
    low = clause.lower()
    found: list[_Match] = []
    for action, words in KEYWORDS.items():
        for w in words:
            for m in re.finditer(r"\b" + re.escape(w) + r"\b", low):
                found.append(_Match(m.start(), m.end(), action, w))
    # Drop matches fully contained in a longer one.
    found.sort(key=lambda m: (m.start, -(m.end - m.start)))
    kept: list[_Match] = []
    for m in found:
        if any(k.start <= m.start and m.end <= k.end and (k.end - k.start) > (m.end - m.start) for k in kept):
            continue
        if any(k.start == m.start and k.end == m.end for k in kept):
            continue
        kept.append(m)
    return kept


def _cell_mentions(clause: str) -> list[tuple[int, str]]:
    """(position, raw cell token) for every cell-like token in a clause."""
    # Both regexes report the position of the row letter, so a token found by
    # both ("sector C1") collapses to one entry in the set.
    found = {(m.start(), f"{m.group(1).upper()}{m.group(2)}") for m in _CELL_RE.finditer(clause)}
    found |= {(m.start(1), f"{m.group(1).upper()}{m.group(2)}") for m in _CELL_CTX_RE.finditer(clause)}
    return sorted(found)


def parse_report(text: str, rows: int, cols: int, cfg: Config | None = None,
                 logger: logging.Logger | None = None) -> list[dict[str, Any]]:
    """Parse a field report into a list of update dicts.

    Each update: ``{"cell", "index", "action", "factor", "confidence", "evidence"}``.
    Out-of-range cells and cells with no action keyword in their clause are
    ignored and logged. Duplicate (cell, action) pairs are collapsed.
    """
    logger = logger or get_logger("floodwing.agent")
    boost = cfg.boost_factor if cfg else 1.5
    reduce = cfg.reduce_factor if cfg else 0.5
    updates: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for clause in _CLAUSE_RE.split(text):
        if not clause.strip():
            continue
        kws = _keyword_matches(clause)
        for pos, token in _cell_mentions(clause):
            idx = cell_index(token, rows, cols)
            if idx is None:
                logger.info("ignored out-of-range cell %r in report", token)
                continue
            if not kws:
                logger.info("ignored cell %s: no action keyword in clause %r", token, clause.strip())
                continue
            best = min(kws, key=lambda m: (min(abs(m.start - pos), abs(m.end - pos)), ACTION_PRIORITY[m.action]))
            if (idx, best.action) in seen:
                continue
            seen.add((idx, best.action))
            factor = {"boost": boost, "reduce": reduce, "block": 0.0}[best.action]
            updates.append({"cell": cell_name(idx, cols), "index": idx, "action": best.action,
                            "factor": factor, "confidence": 1.0, "evidence": best.phrase})
    return updates


def validate_updates(raw: Iterable[Any], rows: int, cols: int,
                     logger: logging.Logger | None = None) -> list[dict[str, Any]]:
    """Schema check shared by the rule-based and LLM parsers. Drops bad items."""
    logger = logger or get_logger("floodwing.agent")
    clean: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            logger.info("ignored non-dict update %r", item)
            continue
        cell = str(item.get("cell", ""))
        action = str(item.get("action", "")).lower()
        idx = cell_index(cell, rows, cols)
        if idx is None or action not in ACTIONS:
            logger.info("ignored invalid update %r", item)
            continue
        try:
            factor = float(item.get("factor", 0.0 if action == "block" else 1.0))
            conf = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            logger.info("ignored update with non-numeric fields %r", item)
            continue
        if action == "block":
            factor = 0.0
        if not (0.0 <= factor <= 10.0 and 0.0 <= conf <= 1.0):
            logger.info("ignored out-of-range factor/confidence %r", item)
            continue
        clean.append({"cell": cell_name(idx, cols), "index": idx, "action": action, "factor": factor,
                      "confidence": conf, "evidence": str(item.get("evidence", ""))})
    return clean


def apply_updates(w: np.ndarray, updates: Iterable[dict[str, Any]], w_cap: float = 2.0) -> np.ndarray:
    """Pure function: return a NEW weight vector with updates applied.

    boost/reduce multiply the cell weight; block sets it to 0 (the QUBO also
    receives the blocked set via :func:`blocked_cells`).
    """
    out = np.array(w, dtype=float, copy=True)
    for u in updates:
        i = int(u["index"])
        out[i] = 0.0 if u["action"] == "block" else out[i] * float(u["factor"])
    return np.clip(out, 0.0, w_cap)


def blocked_cells(updates: Iterable[dict[str, Any]]) -> set[int]:
    """Indices of cells marked as blocked."""
    return {int(u["index"]) for u in updates if u["action"] == "block"}


class LocalLLMParser:
    """Optional local-LLM parser (Config.use_local_llm). Same schema, validated,
    falls back to rules on any failure. Not used by default; requires
    ``transformers`` and a model download, which the default path never needs.
    """

    PROMPT = ("Extract drone-search updates from the report. Grid rows are letters A-{last_row}, "
              "columns are numbers 1-{cols}. Return ONLY a JSON list of objects with keys "
              "cell, action (boost|block|reduce), factor, confidence.\nReport: {text}\nJSON:")

    def __init__(self, cfg: Config, logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger or get_logger("floodwing.agent")
        self._pipe = None

    def _load(self) -> Any:
        if self._pipe is None:
            from transformers import pipeline  # lazy: optional dependency

            self._pipe = pipeline("text-generation", model=self.cfg.local_llm_model, device_map="auto")
        return self._pipe

    def parse(self, text: str, rows: int, cols: int) -> list[dict[str, Any]]:
        try:
            pipe = self._load()
            prompt = self.PROMPT.format(last_row=chr(ord("A") + rows - 1), cols=cols, text=text)
            out = pipe(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"][len(prompt):]
            match = re.search(r"\[.*\]", out, re.DOTALL)
            raw = json.loads(match.group(0)) if match else []
            updates = validate_updates(raw, rows, cols, self.logger)
            if updates:
                return updates
            self.logger.warning("LLM output empty or invalid; falling back to rules")
        except Exception as exc:  # noqa: BLE001 - any LLM failure falls back
            self.logger.warning("local LLM unavailable (%s); falling back to rules", exc)
        return parse_report(text, rows, cols, self.cfg, self.logger)


def make_parser(cfg: Config, logger: logging.Logger | None = None):
    """Return a callable ``(text, rows, cols) -> updates`` honouring Config.use_local_llm."""
    if cfg.use_local_llm:
        return LocalLLMParser(cfg, logger).parse
    return lambda text, rows, cols: validate_updates(parse_report(text, rows, cols, cfg, logger), rows, cols, logger)
