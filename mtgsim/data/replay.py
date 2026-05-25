"""ReplayCapture: captures ONE match end-to-end for debugging.

Heavy: per-event state snapshot + decisions + seeds. Use only when flagged.
Triggered by UI button or programmatically. One file per match.

Layout:
  data/replays/replay_slot<N>_match<id>_<ts>.jsonl.gz
"""
from __future__ import annotations
import gzip
import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


def _serialize_card(c) -> dict:
    return {
        "cid": c.cid, "name": c.name,
        "tap": c.tapped, "sick": c.summoning_sick,
        "p": c.cdef.power, "t": c.cdef.toughness,
        "counters": dict(c.counters), "is_token": c.is_token,
        "is_land": c.cdef.is_land(), "is_creature": c.cdef.is_creature(),
        "attacking": c.attacking, "blocking": list(c.blocking),
        "blocked_by": list(c.blocked_by),
    }


def _serialize_player(p) -> dict:
    return {
        "idx": p.idx, "name": p.name, "life": p.life,
        "hand": [_serialize_card(c) for c in p.hand],
        "hand_count": len(p.hand),
        "library_count": len(p.library),
        "graveyard": [_serialize_card(c) for c in p.graveyard],
        "exile": [_serialize_card(c) for c in p.exile],
        "battlefield": [_serialize_card(c) for c in p.battlefield],
        "mana": {k: v for k, v in p.mana_pool.pool.items() if v},
        "city_blessing": p.city_blessing,
        "mulligans": p.mulligans_taken,
        "lost": p.lost,
    }


def serialize_full_state(game) -> dict:
    return {
        "turn": game.turn, "active_idx": game.active_idx,
        "phase": game.phase.name if game.phase else "",
        "step": game.step.name if game.step else "",
        "stack_size": len(game.stack),
        "players": [_serialize_player(p) for p in game.players],
        "winner_idx": game.winner_idx, "draw": game.draw_game,
    }


class ReplayCapture:
    """One-shot capture for a single match. Save when match ends."""
    def __init__(self, root: str, slot: int, match_id: int,
                 seed: int, compress: bool = True):
        self.root = Path(root) / "replays"
        self.root.mkdir(parents=True, exist_ok=True)
        self.slot = slot
        self.match_id = match_id
        self.seed = seed
        self.compress = compress
        self._buf: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._game = None  # set by record_event_with_state via thread-local trick
        self.done = False
        self.file_path: Optional[Path] = None

    def __deepcopy__(self, memo):
        return None

    def attach_game(self, game):
        """Engine call: set current game reference so record_event can snapshot."""
        self._game = game

    def record_event(self, ev):
        """Subscribe target on observer. Captures event + full state snapshot."""
        with self._lock:
            if self.done:
                return
            game = self._game
            state = serialize_full_state(game) if game is not None else None
            self._buf.append({
                "type": "event",
                "ts": ev.ts, "turn": ev.turn, "step": ev.step,
                "active": ev.active_idx, "kind": ev.kind,
                "data": {k: (v.name if hasattr(v, "name") else (str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v))
                         for k, v in (ev.data or {}).items()},
                "state": state,
            })

    def record_decision(self, root_idx: int, state_vec: List[float],
                        legal_actions_desc: List[Dict[str, Any]],
                        visits: List[int], chosen_idx: int, value: float):
        with self._lock:
            if self.done:
                return
            self._buf.append({
                "type": "decision",
                "root_idx": root_idx, "state_vec": state_vec,
                "legal": legal_actions_desc, "visits": visits,
                "chosen": chosen_idx, "value": value,
            })

    def header(self, deck0: str, deck1: str):
        with self._lock:
            self._buf.append({
                "type": "header",
                "slot": self.slot, "match_id": self.match_id,
                "seed": self.seed, "deck0": deck0, "deck1": deck1,
                "ts": time.time(),
            })

    def finalize(self, match_result_summary: dict):
        with self._lock:
            if self.done:
                return
            self.done = True
            self._buf.append({"type": "footer", **match_result_summary, "ts": time.time()})
            ext = ".jsonl.gz" if self.compress else ".jsonl"
            fn = self.root / f"replay_slot{self.slot}_match{self.match_id}_{int(time.time()*1000)}{ext}"
            self.file_path = fn
            if self.compress:
                with gzip.open(fn, "wt", encoding="utf-8") as f:
                    for line in self._buf:
                        f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            else:
                with fn.open("a", encoding="utf-8") as f:
                    for line in self._buf:
                        f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            self._buf = []
