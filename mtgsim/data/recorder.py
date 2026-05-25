"""Recorder: persists matches as JSONL for analysis + NN training.

Modes (compact-only; debug uses ReplayCapture separately):
  - "summary":   ~200B/match. match_start + game_start/end + match_end.
  - "decisions": ~50KB/match gzipped. summary + (state_vec, legal, mcts_visits, chosen, value, outcome_z).
                 Directly trainable for AlphaZero-style NN.

Layout:
  data/matches/YYYY-MM-DD/<prefix>_<match_id>_<pid>_<ms>.jsonl[.gz]

Backfill: outcome_z is filled at game_end before flush.
"""
from __future__ import annotations
import gzip
import json
import os
import random
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 2


def safe_serialize(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [safe_serialize(x) for x in v]
    if isinstance(v, dict):
        return {str(k): safe_serialize(x) for k, x in v.items()}
    if hasattr(v, "name"):
        return v.name
    return str(v)


class Recorder:
    def __init__(self, root: str, mode: str = "summary", sample: float = 1.0,
                 rng: Optional[random.Random] = None,
                 file_prefix: str = "match",
                 compress: bool = True):
        """root: base dir; daily subfolder appended.
        mode: 'summary' | 'decisions'.
        sample: probability (0..1) that a match is recorded.
        compress: gzip files (default True). Loader reads both transparently."""
        assert mode in ("summary", "decisions"), f"unknown mode {mode!r}; use ReplayCapture for full traces"
        self.root = Path(root)
        self.mode = mode
        self.sample = sample
        self.rng = rng or random.Random()
        self.file_prefix = file_prefix
        self.compress = compress
        self._match_buf: List[Dict[str, Any]] = []
        self._dec_buf: List[Dict[str, Any]] = []
        self._all_dec_for_match: List[Dict[str, Any]] = []
        self._enabled_for_match: bool = False
        self._current_match_id: Optional[int] = None
        self._current_game_id: Optional[int] = None
        self._lock = threading.Lock()
        self.stats = {"matches_written": 0, "decisions": 0, "bytes": 0}
        self._date_dir = self._daily_dir()
        self._date_dir.mkdir(parents=True, exist_ok=True)

    def __deepcopy__(self, memo):
        return None

    def _daily_dir(self) -> Path:
        return self.root / "matches" / time.strftime("%Y-%m-%d")

    def start_match(self, match_id: int, deck0: str, deck1: str, meta: Optional[dict] = None):
        with self._lock:
            self._enabled_for_match = (self.rng.random() < self.sample)
            self._current_match_id = match_id
            self._match_buf = []
            self._all_dec_for_match = []
            if not self._enabled_for_match:
                return
            self._match_buf.append({
                "type": "match_start", "schema": SCHEMA_VERSION,
                "match_id": match_id, "deck0": deck0, "deck1": deck1,
                "ts": time.time(), "meta": meta or {},
            })

    def start_game(self, game_id: int, play_first: int):
        with self._lock:
            self._current_game_id = game_id
            self._dec_buf = []
            if not self._enabled_for_match:
                return
            self._match_buf.append({"type": "game_start", "game_id": game_id, "play_first": play_first})

    def end_game(self, game_id: int, winner_idx: Optional[int], turns: int, draw: bool):
        with self._lock:
            if not self._enabled_for_match:
                return
            for d in self._dec_buf:
                if draw or winner_idx is None:
                    d["outcome_z"] = 0.0
                else:
                    d["outcome_z"] = 1.0 if d["root_idx"] == winner_idx else -1.0
            self._all_dec_for_match.extend(self._dec_buf)
            self._dec_buf = []
            self._match_buf.append({
                "type": "game_end", "game_id": game_id,
                "winner_idx": winner_idx, "turns": turns, "draw": draw,
            })

    def end_match(self, summary: dict):
        with self._lock:
            if not self._enabled_for_match:
                self._match_buf = []
                self._all_dec_for_match = []
                return
            self._match_buf.append({"type": "match_end", **summary, "ts": time.time()})
            if self.mode == "decisions":
                self._match_buf.extend(self._all_dec_for_match)
            self._flush()
            self.stats["matches_written"] += 1
            self._enabled_for_match = False
            self._match_buf = []
            self._all_dec_for_match = []

    def record_event(self, ev):
        # no-op: compact recorder doesn't store events; use ReplayCapture for debugging
        return

    def record_metrics(self, metrics: dict):
        """Append a 'metrics' record. Called once per match (compact, ~500B)."""
        with self._lock:
            if not self._enabled_for_match:
                return
            self._match_buf.append({"type": "metrics", **metrics})

    def record_decision(self, root_idx: int, state_vec: List[float],
                        legal_actions_desc: List[Dict[str, Any]],
                        visits: List[int], chosen_idx: int, value: float):
        with self._lock:
            if not self._enabled_for_match or self.mode != "decisions":
                return
            self._dec_buf.append({
                "type": "decision",
                "game_id": self._current_game_id,
                "root_idx": root_idx,
                "state": state_vec,
                "legal": legal_actions_desc,
                "visits": visits,
                "chosen": chosen_idx,
                "value": value,
                "outcome_z": None,
            })
            self.stats["decisions"] += 1

    def _flush(self):
        if not self._match_buf:
            return
        cur = self._daily_dir()
        if cur != self._date_dir:
            self._date_dir = cur
            self._date_dir.mkdir(parents=True, exist_ok=True)
        ext = ".jsonl.gz" if self.compress else ".jsonl"
        fn = self._date_dir / (f"{self.file_prefix}_{self._current_match_id}"
                               f"_{os.getpid()}_{int(time.time()*1000)}{ext}")
        if self.compress:
            with gzip.open(fn, "wt", encoding="utf-8") as f:
                for line in self._match_buf:
                    f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        else:
            with fn.open("a", encoding="utf-8") as f:
                for line in self._match_buf:
                    f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        try:
            self.stats["bytes"] += fn.stat().st_size
        except Exception:
            pass
