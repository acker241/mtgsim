"""State + action encoding. Richer state for NN training.

Per-player block:
  - Scalars: life, lib_count, hand_count, gy_count, bf_count, city_blessing (6)
  - Mana pool by color: W,U,B,R,G,C (6)
  - Untapped lands by produced color: W,U,B,R,G,C (6)
  - Hand multi-hot by card name (N_NAMES)
  - BF multi-hot by card name (N_NAMES)
  - Graveyard multi-hot by card name (N_NAMES)
  - Aggregate creature stats: sum_power, sum_tough, max_power (3)
  - Hand burn potential: sum_damage_in_burn_cards (1)
  - Keyword counts on BF: flying, lifelink, vigilance, first_strike, menace, indestructible (6)
  - Saga lore counters total, planeswalker loyalty total (2)
  - Steam-Kin max counters in play, attacked_with last turn (2)
Total per player: 6+6+6 + 3*N_NAMES + 3+1+6+2+2 = 32 + 3*38 = 146

Globals:
  - turn / 30, active_idx (one-hot 2) (3)
  - Phase one-hot: BEGIN, MAIN1, COMBAT, MAIN2, END (5)
  - Step one-hot: 13 steps (13)
Total globals: 21

Grand total: 2*146 + 21 = 313 dims (was 198).

Action encoding unchanged (action_to_features in nn.py).
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ..engine.game import GameState
    from .action import Action


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


ACTION_KINDS = ["pass", "play_land", "cast", "activate", "attackers", "blockers"]
KIND_TO_IDX = {k: i for i, k in enumerate(ACTION_KINDS)}
N_KINDS = len(ACTION_KINDS)

PHASES = ["BEGIN", "MAIN1", "COMBAT", "MAIN2", "END"]
STEPS = ["UNTAP", "UPKEEP", "DRAW", "PRECOMBAT_MAIN",
         "BEGIN_COMBAT", "DECLARE_ATTACKERS", "DECLARE_BLOCKERS",
         "FIRST_STRIKE_DAMAGE", "COMBAT_DAMAGE", "END_COMBAT",
         "POSTCOMBAT_MAIN", "END_STEP", "CLEANUP"]

_BURN_DMG_BY_NAME = {
    "Shock": 2, "Lightning Strike": 3, "Wizard's Lightning": 3,
    "Skewer the Critics": 3, "Lava Coil": 4, "Fight with Fire": 5,
    "Banefire": 5,  # X — estimate
}


def state_dim() -> int:
    per_player = 6 + 6 + 6 + 3 * N_NAMES + 3 + 1 + 6 + 2 + 2
    globals_dim = 1 + 2 + len(PHASES) + len(STEPS)
    return 2 * per_player + globals_dim


def action_dim() -> int:
    return N_KINDS * (1 + N_NAMES)


def _encode_player(p, root_idx_relative: int) -> List[float]:
    """Encode one player block. root_idx_relative: 0 if this is root player, 1 if opp."""
    from ..engine.enums import Subtype, Keyword
    vec: List[float] = []
    # 6 scalars
    vec += [
        p.life / 20.0,
        len(p.library) / 60.0,
        len(p.hand) / 10.0,
        len(p.graveyard) / 30.0,
        len(p.battlefield) / 15.0,
        1.0 if p.city_blessing else 0.0,
    ]
    # mana pool (6)
    for sym in ("W", "U", "B", "R", "G", "C"):
        vec.append(p.mana_pool.pool.get(sym, 0) / 5.0)
    # untapped lands by produced color (6)
    untap_counts = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
    for c in p.battlefield:
        if not c.cdef.is_land() or c.tapped:
            continue
        if Subtype.MOUNTAIN in c.cdef.subtypes:
            untap_counts["R"] += 1
        elif Subtype.PLAINS in c.cdef.subtypes:
            untap_counts["W"] += 1
        else:
            untap_counts["C"] += 1
    for sym in ("W", "U", "B", "R", "G", "C"):
        vec.append(untap_counts[sym] / 6.0)
    # hand multi-hot (N_NAMES)
    hand_counts = [0.0] * N_NAMES
    for c in p.hand:
        i = NAME_TO_IDX.get(c.name)
        if i is not None:
            hand_counts[i] += 1.0
    vec += hand_counts
    # BF multi-hot
    bf_counts = [0.0] * N_NAMES
    for c in p.battlefield:
        i = NAME_TO_IDX.get(c.name)
        if i is not None:
            bf_counts[i] += 1.0
    vec += bf_counts
    # graveyard multi-hot
    gy_counts = [0.0] * N_NAMES
    for c in p.graveyard:
        i = NAME_TO_IDX.get(c.name)
        if i is not None:
            gy_counts[i] += 1.0
    vec += gy_counts
    # aggregate creature stats — use raw values without invoking game (no static effects context here)
    # caller should pass game when needed; for simplicity use base + counters
    sum_power = 0
    sum_tough = 0
    max_power = 0
    for c in p.battlefield:
        if not c.cdef.is_creature():
            continue
        pwr = (c.cdef.power or 0) + c.counters.get("+1/+1", 0)
        tgh = (c.cdef.toughness or 0) + c.counters.get("+1/+1", 0)
        sum_power += pwr
        sum_tough += tgh
        if pwr > max_power:
            max_power = pwr
    vec += [sum_power / 15.0, sum_tough / 15.0, max_power / 7.0]
    # hand burn potential (sum of damage in burn cards in hand)
    burn_total = 0
    for c in p.hand:
        burn_total += _BURN_DMG_BY_NAME.get(c.name, 0)
    vec += [burn_total / 15.0]
    # keyword counts on BF (6) — count instances, not just unique
    kw_counts = {Keyword.FLYING: 0, Keyword.LIFELINK: 0, Keyword.VIGILANCE: 0,
                 Keyword.FIRST_STRIKE: 0, Keyword.MENACE: 0, Keyword.INDESTRUCTIBLE: 0}
    for c in p.battlefield:
        if not c.cdef.is_creature():
            continue
        for kw in c.cdef.keywords:
            if kw in kw_counts:
                kw_counts[kw] += 1
    for kw in (Keyword.FLYING, Keyword.LIFELINK, Keyword.VIGILANCE,
               Keyword.FIRST_STRIKE, Keyword.MENACE, Keyword.INDESTRUCTIBLE):
        vec.append(kw_counts[kw] / 5.0)
    # saga lore counters total, planeswalker loyalty total (2)
    saga_total = 0
    pw_loyalty_total = 0
    for c in p.battlefield:
        if Subtype.SAGA in c.cdef.subtypes:
            saga_total += c.chapter
        if c.cdef.is_planeswalker():
            pw_loyalty_total += c.counters.get("loyalty", 0)
    vec += [saga_total / 6.0, pw_loyalty_total / 12.0]
    # Steam-Kin max counters in play, attacked_with last turn (2)
    steamkin_max = 0
    for c in p.battlefield:
        if c.name == "Runaway Steam-Kin":
            cnt = c.counters.get("+1/+1", 0)
            if cnt > steamkin_max:
                steamkin_max = cnt
    vec += [steamkin_max / 3.0, p.attacked_with / 5.0]
    return vec


def state_to_vector(game: "GameState", root_idx: int) -> List[float]:
    vec: List[float] = []
    # root player first, opponent second
    vec += _encode_player(game.players[root_idx], root_idx_relative=0)
    vec += _encode_player(game.players[1 - root_idx], root_idx_relative=1)
    # globals: turn, active one-hot, phase one-hot, step one-hot
    vec.append(game.turn / 30.0)
    # active idx (2) — relative: 1.0 if root is active
    vec.append(1.0 if game.active_idx == root_idx else 0.0)
    vec.append(1.0 if game.active_idx != root_idx else 0.0)
    # phase one-hot
    phase_name = game.phase.name if game.phase else ""
    for p in PHASES:
        vec.append(1.0 if p == phase_name else 0.0)
    # step one-hot
    step_name = game.step.name if game.step else ""
    for s in STEPS:
        vec.append(1.0 if s == step_name else 0.0)
    return vec


def action_to_idx(action: "Action") -> int:
    kind = KIND_TO_IDX.get(action.kind, 0)
    return kind * (1 + N_NAMES)
