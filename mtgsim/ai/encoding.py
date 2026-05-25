"""State + action encoding. Stub now; flesh out for NN later.

Design notes for NN later:
- state_to_tensor(game, root_idx) -> 1D float32 ndarray of fixed dim.
  Encode: life (both), library count, hand count, mana availability, BF counts by type,
  per-card-name presence/counts (top-k cards both decks).
- action_to_idx(action) -> int in [0, action_dim).
  Encode: kind + card-name slot (1-hot over deck card-name vocab) + target slot.
- legal_mask(actions, action_dim) -> bool ndarray.

For now we just give skeletons + a tiny encoder used by HeuristicPolicy (for diagnostics).
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ..engine.game import GameState
    from .action import Action


# Stable name vocab for both decks (mainboards + sideboards). Add new cards here.
CARD_NAMES = [
    # red
    "Fanatical Firebrand", "Ghitu Lavarunner", "Viashino Pyromancer",
    "Runaway Steam-Kin", "Goblin Chainwhirler",
    "Shock", "Lightning Strike", "Wizard's Lightning",
    "Light Up the Stage", "Skewer the Critics", "Experimental Frenzy",
    "Mountain",
    "Banefire", "Dire Fleet Daredevil", "Fiery Cannonade", "Fight with Fire",
    "Lava Coil", "Rekindling Phoenix", "Treasure Map",
    # white
    "Dauntless Bodyguard", "Skymarcher Aspirant", "Snubhorn Sentry",
    "Healer's Hawk", "Tithe Taker", "Benalish Marshal", "Venerated Loxodon",
    "Legion's Landing", "History of Benalia", "Conclave Tribunal",
    "Unbreakable Formation", "Plains", "Adanto, the First Fort",
    "Tocatli Honor Guard", "Baffling End", "Ajani, Adversary of Tyrants",
    "Adanto Vanguard", "Demystify",
    # tokens
    "Vampire token", "Knight token", "Spirit token", "Cat token",
    "Dinosaur token", "Elemental token (Rekindling)",
]
NAME_TO_IDX = {n: i for i, n in enumerate(CARD_NAMES)}
N_NAMES = len(CARD_NAMES)


# kind vocab
ACTION_KINDS = ["pass", "play_land", "cast", "activate", "attackers", "blockers"]
KIND_TO_IDX = {k: i for i, k in enumerate(ACTION_KINDS)}
N_KINDS = len(ACTION_KINDS)


def state_dim() -> int:
    """Dimension of state tensor."""
    # per-player: 6 scalars + N_NAMES (BF presence) + N_NAMES (grave presence) + 5 (mana)
    per_player = 6 + N_NAMES + N_NAMES + 5
    return 2 * per_player + 4  # +4 globals: turn, active_idx, phase_oh(none, main, combat, end approx)


def action_dim() -> int:
    """Coarse upper bound for action encoding (kind + card_slot)."""
    return N_KINDS * (1 + N_NAMES)


def state_to_vector(game: "GameState", root_idx: int) -> List[float]:
    """Naive encoding for now. NN will use this (or evolve it)."""
    vec: List[float] = []
    for offset in (0, 1):
        idx = (root_idx + offset) % 2
        p = game.players[idx]
        vec += [
            p.life / 20.0,
            len(p.library) / 60.0,
            len(p.hand) / 10.0,
            len(p.graveyard) / 30.0,
            len(p.battlefield) / 15.0,
            1.0 if p.city_blessing else 0.0,
        ]
        # BF presence by name
        bf_counts = [0.0] * N_NAMES
        for c in p.battlefield:
            i = NAME_TO_IDX.get(c.name)
            if i is not None:
                bf_counts[i] += 1.0
        vec += bf_counts
        # grave presence
        gr_counts = [0.0] * N_NAMES
        for c in p.graveyard:
            i = NAME_TO_IDX.get(c.name)
            if i is not None:
                gr_counts[i] += 1.0
        vec += gr_counts
        # mana pool (W,U,B,R,G)
        for sym in ("W", "U", "B", "R", "G"):
            vec.append(p.mana_pool.pool.get(sym, 0) / 5.0)
    vec += [
        game.turn / 30.0,
        float(game.active_idx),
        1.0 if game.step.name.startswith("PRECOMBAT") else 0.0,
        1.0 if "COMBAT" in game.step.name else 0.0,
    ]
    return vec


def action_to_idx(action: "Action") -> int:
    """Coarse: kind * (1 + N_NAMES) + (name_slot or 0)."""
    kind = KIND_TO_IDX.get(action.kind, 0)
    slot = 0
    # find name slot from card_cid? We need name lookup. For now, return 0 (NN later does richer encoding).
    return kind * (1 + N_NAMES) + slot
