"""Load JSONL recordings for analysis or ML training. Reads .jsonl and .jsonl.gz transparently."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Tuple


def _open_text(fp: Path):
    if fp.suffix == ".gz":
        return gzip.open(fp, "rt", encoding="utf-8")
    return fp.open("r", encoding="utf-8")


def iter_records(root: str, types: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
    p = Path(root)
    files = sorted(list(p.rglob("*.jsonl")) + list(p.rglob("*.jsonl.gz")))
    for fp in files:
        with _open_text(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if types and rec.get("type") not in types:
                    continue
                yield rec


def load_match_summaries(root: str) -> List[Dict[str, Any]]:
    return list(iter_records(root, types=["match_end"]))


def load_decisions(root: str) -> List[Dict[str, Any]]:
    """Returns decision tuples with outcome_z populated. Ready for NN training."""
    out = []
    for rec in iter_records(root, types=["decision"]):
        if rec.get("outcome_z") is None:
            continue  # skip undone games
        out.append(rec)
    return out


def to_training_arrays(decisions: List[Dict[str, Any]]):
    """Pack decisions into (states, policy_targets, values) numpy arrays.
    Requires numpy (optional dep; raises ImportError if missing).
    policy_targets: visit-distribution over legal actions (variable per-decision)."""
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy required for to_training_arrays — pip install numpy") from e

    states = np.asarray([d["state"] for d in decisions], dtype="float32")
    values = np.asarray([d["outcome_z"] for d in decisions], dtype="float32")
    # ragged policy targets — return as list of normalized visit distributions
    policy_targets = []
    for d in decisions:
        visits = d.get("visits") or [1] * len(d.get("legal", []))
        total = sum(visits) or 1
        policy_targets.append([v / total for v in visits])
    return states, policy_targets, values


def summarize(root: str) -> Dict[str, Any]:
    n_matches = 0
    n_games = 0
    n_decisions = 0
    n_events = 0
    wins = {}
    for rec in iter_records(root):
        t = rec.get("type")
        if t == "match_end":
            n_matches += 1
            w = rec.get("winner")
            if w:
                wins[w] = wins.get(w, 0) + 1
        elif t == "game_end":
            n_games += 1
        elif t == "decision":
            n_decisions += 1
        elif t == "event":
            n_events += 1
    return {
        "matches": n_matches,
        "games": n_games,
        "decisions": n_decisions,
        "events": n_events,
        "wins": wins,
    }
