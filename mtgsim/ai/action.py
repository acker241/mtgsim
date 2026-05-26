"""Canonical Action type and enumeration. Bridges AI decisions and engine.
Designed to be friendly for NN policy heads later (each Action maps to a class+params)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from ..engine.card import Card
from ..engine.player import Player
from ..engine.enums import Subtype, Keyword, CardType, Zone

if TYPE_CHECKING:
    from ..engine.game import GameState


@dataclass(frozen=False)
class Action:
    """Canonical action descriptor.
    For NN later: kind + (card_cid, target_cids, attackers, blocks) form a discrete head per kind."""
    kind: str                                       # "pass", "play_land", "cast", "activate", "attackers", "blockers"
    card_cid: Optional[int] = None                  # spell/land/source perm
    target_cids: List[int] = field(default_factory=list)  # cids; -1 = own player, -2 = opp player
    convoke_cids: List[int] = field(default_factory=list)
    x: int = 0
    use_alt_cost: bool = False                      # spectacle or wizard discount
    attacker_cids: List[int] = field(default_factory=list)
    block_map: Dict[int, List[int]] = field(default_factory=dict)
    activated_idx: int = 0

    def __repr__(self):
        if self.kind == "pass":
            return "pass"
        if self.kind == "play_land":
            return f"land#{self.card_cid}"
        if self.kind == "cast":
            return f"cast#{self.card_cid}->{self.target_cids}{'*' if self.use_alt_cost else ''}"
        if self.kind == "attackers":
            return f"atk{self.attacker_cids}"
        if self.kind == "blockers":
            return f"blk{self.block_map}"
        return f"<{self.kind}>"


def _by_cid(game, cid: int):
    """Resolve cid to Card or Player. -1/-2 = own/opp player at AP perspective (resolved by caller)."""
    for p in game.players:
        for lst in (p.battlefield, p.hand, p.library, p.graveyard, p.exile):
            for c in lst:
                if c.cid == cid:
                    return c
    return None


def resolve_target(game, idx: int, ref: int):
    """Resolve target reference: -1 self player, -2 opp player, else card cid."""
    if ref == -1:
        return game.players[idx]
    if ref == -2:
        return game.players[1 - idx]
    return _by_cid(game, ref)


def encode_target(game, idx: int, target: Any) -> int:
    if isinstance(target, Player):
        return -1 if target.idx == idx else -2
    return target.cid


# ---------------- Legal action enumeration ----------------
def legal_main_actions(game: "GameState", idx: int) -> List[Action]:
    """Enumerate main-phase actions for player idx. Always includes 'pass'."""
    from ..ai.heuristic import HeuristicAI  # for helper checks only
    pl = game.players[idx]
    actions: List[Action] = []
    helper = _Helper(game)

    # 1) play land (if not done)
    if pl.lands_played_this_turn < 1:
        for c in pl.hand:
            if c.cdef.is_land():
                actions.append(Action(kind="play_land", card_cid=c.cid))
                break  # only one land — enumerate one (all basics same)

    # 2) cast spells from hand
    pool = helper.available_mana(idx)
    for c in pl.hand:
        if c.cdef.is_land() or c.cdef.is_instant():
            continue
        cost = c.cdef.cost
        if not cost:
            continue
        use_alt = False
        cost_eff = cost
        if c.cdef.spectacle_cost is not None:
            if c.name == "Wizard's Lightning" and helper.has_wizard(idx):
                use_alt = True
                cost_eff = c.cdef.spectacle_cost
            elif c.cdef.is_sorcery() and pl.opp_lost_life_this_turn:
                use_alt = True
                cost_eff = c.cdef.spectacle_cost
        # convoke
        convoke = []
        if c.name in ("Conclave Tribunal", "Venerated Loxodon"):
            avail = helper.convoke_creatures(idx)
            deficit = max(0, cost_eff.cmc() - helper.total(pool))
            if deficit > len(avail):
                continue
            convoke = avail[:deficit]
        # check pay
        if not helper.can_pay_with_convoke(pool, cost_eff, convoke):
            continue
        # X
        x = 0
        if c.cdef.has_x:
            x = max(0, helper.total(pool) - cost.symbols.get("GENERIC", 0)
                    - sum(cost.colored_required().values()))
            if x < 1:
                continue
        # targets
        target_cids: List[int] = []
        if c.cdef.needs_targets > 0:
            t = helper.pick_target(idx, c, dmg=helper.dmg_estimate(c, x))
            if t is None:
                continue
            target_cids = [encode_target(game, idx, t)]
        actions.append(Action(kind="cast", card_cid=c.cid,
                              target_cids=target_cids,
                              convoke_cids=[k.cid for k in convoke],
                              x=x, use_alt_cost=use_alt))

    # 3) burn instants in main (only if useful: lethal or kill threat)
    for c in pl.hand:
        if not c.cdef.is_instant():
            continue
        dmg = helper.dmg_estimate(c, 0)
        if dmg <= 0:
            continue
        cost_eff = c.cdef.cost
        use_alt = False
        if c.name == "Wizard's Lightning" and helper.has_wizard(idx):
            cost_eff = c.cdef.spectacle_cost
            use_alt = True
        if not helper.can_pay_with_convoke(pool, cost_eff, []):
            continue
        # face target
        actions.append(Action(kind="cast", card_cid=c.cid, target_cids=[-2],
                              use_alt_cost=use_alt))
        # threat targets
        opp = game.players[1 - idx]
        for t in opp.battlefield:
            if not t.cdef.is_creature():
                continue
            if c.cdef.target_filter and not c.cdef.target_filter(game, pl, t):
                continue
            actions.append(Action(kind="cast", card_cid=c.cid, target_cids=[t.cid],
                                  use_alt_cost=use_alt))

    # 4) activate Steam-Kin (3 counters -> RRR) — only relevant if can chain spells; skip enumerating
    # 5) Adanto activated (1W,T: token) — useful at end of turn; enumerate when free
    for c in pl.battlefield:
        if c.name == "Adanto, the First Fort" and not c.tapped:
            if pool.get("W", 0) >= 1 and helper.total(pool) >= 2:
                actions.append(Action(kind="activate", card_cid=c.cid, activated_idx=0))

    # always pass
    actions.append(Action(kind="pass"))
    return actions


def legal_attacker_sets(game: "GameState", idx: int, k_options: int = 4) -> List[Action]:
    """Return a small set of attacker subsets to consider. Heuristic-driven enumeration.
    Returns 1-4 candidates: all, none, smart-only, heuristic."""
    pl = game.players[idx]
    candidates = [c for c in pl.battlefield if c.can_attack(game)]
    if not candidates:
        return [Action(kind="attackers", attacker_cids=[])]
    sets = []
    # 1) attack with all
    sets.append([c.cid for c in candidates])
    # 2) attack with none
    sets.append([])
    # 3) attack only with creatures with evasion (flying/menace) or that survive any block
    opp = game.players[1 - idx]
    blockers = [b for b in opp.battlefield if b.cdef.is_creature() and not b.tapped]

    def safe(c):
        if Keyword.FLYING in c.keywords(game) and not any(
                Keyword.FLYING in b.keywords(game) or Keyword.REACH in b.keywords(game) for b in blockers):
            return True
        # if biggest blocker power < c.toughness and c.power >= biggest blocker tough
        if not blockers:
            return True
        if c.power(game) > max((b.toughness(game) - b.damage_marked for b in blockers), default=0) \
                and max((b.power(game) for b in blockers), default=0) < c.toughness(game):
            return True
        return False
    smart = [c.cid for c in candidates if safe(c)]
    if smart and smart != sets[0]:
        sets.append(smart)
    # de-dup
    uniq = []
    seen = set()
    for s in sets:
        k = tuple(sorted(s))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    return [Action(kind="attackers", attacker_cids=s) for s in uniq[:k_options]]


def legal_blocker_options(game: "GameState", idx: int, k_options: int = 4) -> List[Action]:
    """Return small set of block assignments. Default: heuristic-best + chump-all + no-block."""
    from .heuristic import HeuristicAI
    nap = game.players[idx]
    if not game.attackers:
        return [Action(kind="blockers", block_map={})]
    # build candidate block maps:
    options: List[Dict[int, List[int]]] = []
    # 1) no block
    options.append({})
    # 2) heuristic block
    ai = HeuristicAI()
    h_block = ai._declare_blockers(game)
    if h_block:
        options.append(h_block)
    # 3) chump-all: chump every attacker with cheapest blocker
    avail = [b for b in nap.battlefield if b.cdef.is_creature() and not b.tapped]
    if avail:
        ca = dict()
        for atk in game.attackers:
            if avail:
                weakest = min(avail, key=lambda b: b.power(game) + b.toughness(game))
                if Keyword.MENACE in atk.keywords(game):
                    if len(avail) >= 2:
                        avail.sort(key=lambda b: b.power(game) + b.toughness(game))
                        ca[atk.cid] = [avail[0].cid, avail[1].cid]
                        avail = avail[2:]
                else:
                    ca[atk.cid] = [weakest.cid]
                    avail.remove(weakest)
        options.append(ca)
    # dedup
    uniq = []
    seen = set()
    for o in options:
        k = tuple(sorted((a, tuple(b)) for a, b in o.items()))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(o)
    return [Action(kind="blockers", block_map=o) for o in uniq[:k_options]]


# ---------------- Apply action ----------------
def apply_action(game: "GameState", idx: int, action: Action) -> bool:
    """Apply an action to the live game state. Returns True on success."""
    from ..engine import actions as eng_actions
    pl = game.players[idx]
    if action.kind == "pass":
        return True
    if action.kind == "play_land":
        card = _find_in_hand(pl, action.card_cid)
        if not card:
            return False
        return eng_actions.play_land(game, idx, card)
    if action.kind == "cast":
        card = _find_in_hand(pl, action.card_cid)
        if not card:
            return False
        targets = [resolve_target(game, idx, ref) for ref in action.target_cids]
        targets = [t for t in targets if t is not None]
        convoke = [_find_on_bf(pl, cid) for cid in action.convoke_cids]
        convoke = [c for c in convoke if c is not None]
        ok = eng_actions.cast_spell(game, idx, card, targets=targets, x=action.x,
                                    use_spectacle=action.use_alt_cost,
                                    convoke_creatures=convoke or None)
        if ok and convoke:
            card.ai_choice = convoke
        game.resolve_all()
        return ok
    if action.kind == "activate":
        perm = _find_on_bf(pl, action.card_cid)
        if not perm:
            return False
        if not perm.cdef.activated or action.activated_idx >= len(perm.cdef.activated):
            return False
        ab = perm.cdef.activated[action.activated_idx]
        if not ab.cost_fn(game, perm):
            return False
        ab.effect(game, pl, perm, [])
        game.resolve_all()
        return True
    if action.kind == "attackers":
        attackers = [_find_on_bf(pl, cid) for cid in action.attacker_cids]
        attackers = [a for a in attackers if a is not None]
        from ..engine import combat
        combat.declare_attackers(game, attackers)
        return True
    if action.kind == "blockers":
        from ..engine import combat
        combat.declare_blockers(game, action.block_map)
        return True
    return False


def _find_in_hand(pl: Player, cid: int) -> Optional[Card]:
    for c in pl.hand:
        if c.cid == cid:
            return c
    return None


def _find_on_bf(pl: Player, cid: int) -> Optional[Card]:
    for c in pl.battlefield:
        if c.cid == cid:
            return c
    return None


# ---------------- Helper: re-use parts of HeuristicAI ----------------
class _Helper:
    def __init__(self, game):
        self.game = game

    def available_mana(self, idx: int) -> Dict[str, int]:
        pl = self.game.players[idx]
        pool = dict(pl.mana_pool.pool)
        for p in pl.battlefield:
            if p.cdef.is_land() and not p.tapped:
                if Subtype.MOUNTAIN in p.cdef.subtypes:
                    pool["R"] = pool.get("R", 0) + 1
                elif Subtype.PLAINS in p.cdef.subtypes:
                    pool["W"] = pool.get("W", 0) + 1
        return pool

    def total(self, pool: Dict[str, int]) -> int:
        return sum(pool.values())

    def has_wizard(self, idx: int) -> bool:
        for c in self.game.players[idx].battlefield:
            if c.cdef.is_creature() and Subtype.WIZARD in c.cdef.subtypes:
                return True
        return False

    def convoke_creatures(self, idx: int) -> List[Card]:
        # convoke ignores summoning sickness (not an activated tap ability)
        return [c for c in self.game.players[idx].battlefield
                if c.cdef.is_creature() and not c.tapped]

    def can_pay_with_convoke(self, pool: Dict[str, int], cost, convoke: List[Card]) -> bool:
        p = dict(pool)
        from ..engine.enums import Color as Col
        for c in convoke:
            paid = False
            for sym, flag in [("W", Col.W), ("R", Col.R)]:
                if c.cdef.colors & flag and cost.colored_required().get(sym, 0) > p.get(sym, 0):
                    p[sym] = p.get(sym, 0) + 1
                    paid = True
                    break
            if not paid:
                p["C"] = p.get("C", 0) + 1
        for sym, n in cost.colored_required().items():
            if p.get(sym, 0) < n:
                return False
            p[sym] -= n
        gen = cost.generic_required()
        return sum(p.values()) >= gen

    def dmg_estimate(self, c: Card, x: int) -> int:
        n = c.name
        return {"Shock": 2, "Lightning Strike": 3, "Wizard's Lightning": 3,
                "Skewer the Critics": 3, "Lava Coil": 4, "Fight with Fire": 5,
                "Banefire": x}.get(n, 0)

    def pick_target(self, idx: int, c: Card, dmg: int = 0):
        opp = self.game.players[1 - idx]
        if c.cdef.target_filter is None:
            return opp
        # try opp first
        if c.cdef.target_filter(self.game, self.game.players[idx], opp):
            if dmg >= opp.life and dmg > 0:
                return opp
        # try opp's biggest creature
        cands = [k for k in opp.battlefield if c.cdef.target_filter(self.game, self.game.players[idx], k)]
        if cands:
            cands.sort(key=lambda k: -(k.power(self.game) if k.cdef.is_creature() else 0))
            return cands[0]
        if c.cdef.target_filter(self.game, self.game.players[idx], opp):
            return opp
        return None
