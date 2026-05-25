"""Combat steps: declare attackers, declare blockers, damage assignment."""
from __future__ import annotations
from typing import List, Dict
from .enums import Keyword, TriggerEvent
from .card import Card
from .game import GameState
from .actions import deal_damage


def declare_attackers(game: GameState, attackers: List[Card]):
    """Tap (if no vigilance), set attacking flag, fire ATTACKS trigger."""
    ap = game.active()
    for atk in attackers:
        if atk not in ap.battlefield or not atk.can_attack(game):
            continue
        atk.attacking = True
        if Keyword.VIGILANCE not in atk.keywords(game):
            atk.tapped = True
        game.attackers.append(atk)
        ap.attacked_with += 1
        game.log(f"P{ap.idx} attacks with {atk.name}")
        game.fire_event(TriggerEvent.ATTACKS, {"card": atk})
    game.resolve_all()


def declare_blockers(game: GameState, blocks: Dict[int, List[int]]):
    """blocks: {attacker_cid: [blocker_cid, ...]}"""
    nap = game.nonactive()
    # build lookup
    by_cid = {c.cid: c for c in game.battlefield()}
    for atk_cid, blocker_cids in blocks.items():
        atk = by_cid.get(atk_cid)
        if not atk or not atk.attacking:
            continue
        # menace: needs >=2 blockers or no blocks at all
        if Keyword.MENACE in atk.keywords(game) and 0 < len(blocker_cids) < 2:
            continue
        for bcid in blocker_cids:
            blk = by_cid.get(bcid)
            if not blk or blk not in nap.battlefield:
                continue
            if not blk.can_block(atk, game):
                continue
            blk.blocking.append(atk.cid)
            atk.blocked_by.append(blk.cid)
            game.fire_event(TriggerEvent.BLOCKS, {"attacker": atk, "blocker": blk})
    game.resolve_all()


def _strike_damage(game: GameState, first_strike_substep: bool):
    """Deal combat damage. first_strike_substep=True for first-strike step; otherwise normal step.
    A creature deals damage in first strike step if it has first_strike or double_strike.
    In normal step, double-strike + non-first-strike creatures deal damage."""
    by_cid = {c.cid: c for c in game.battlefield()}
    ap = game.active()
    nap = game.nonactive()

    def deals_now(c: Card) -> bool:
        kw = c.keywords(game)
        fs = Keyword.FIRST_STRIKE in kw
        ds = Keyword.DOUBLE_STRIKE in kw
        if first_strike_substep:
            return fs or ds
        else:
            return ds or (not fs)

    # build attacker -> blockers map (only attackers still alive)
    for atk in list(game.attackers):
        if atk not in ap.battlefield:
            continue
        blocker_cids = [bc for bc in atk.blocked_by if by_cid.get(bc) and by_cid[bc] in nap.battlefield]
        # damage from attacker
        if deals_now(atk):
            power = atk.power(game)
            if not blocker_cids:
                # unblocked: damage to defending player
                deal_damage(game, atk, nap, power, is_combat=True)
            else:
                # distribute among blockers in declared order (simplified: hit first blocker first)
                remaining = power
                for bcid in blocker_cids:
                    blk = by_cid[bcid]
                    if remaining <= 0:
                        break
                    needed = max(0, blk.toughness(game) - blk.damage_marked)
                    dmg = min(remaining, needed) if needed > 0 else remaining
                    # actually rule: must assign lethal in order before next. simplify: assign needed (or all if needed=0)
                    assign = needed if needed > 0 else remaining
                    assign = min(assign, remaining)
                    deal_damage(game, atk, blk, assign, is_combat=True)
                    remaining -= assign
        # damage from blockers to this attacker
        for bcid in blocker_cids:
            blk = by_cid[bcid]
            if not deals_now(blk):
                continue
            deal_damage(game, blk, atk, blk.power(game), is_combat=True)

    game.check_state_based()


def first_strike_damage(game: GameState):
    # only run if any combatant has FS or DS
    any_fs = False
    by_cid = {c.cid: c for c in game.battlefield()}
    for atk in game.attackers:
        kw = atk.keywords(game)
        if Keyword.FIRST_STRIKE in kw or Keyword.DOUBLE_STRIKE in kw:
            any_fs = True
            break
        for bc in atk.blocked_by:
            blk = by_cid.get(bc)
            if blk and (Keyword.FIRST_STRIKE in blk.keywords(game) or Keyword.DOUBLE_STRIKE in blk.keywords(game)):
                any_fs = True
                break
        if any_fs:
            break
    if any_fs:
        game.on_first_strike_substep = True
        _strike_damage(game, first_strike_substep=True)
        game.on_first_strike_substep = False
        game.resolve_all()


def combat_damage(game: GameState):
    _strike_damage(game, first_strike_substep=False)
    game.resolve_all()


def end_combat(game: GameState):
    for c in game.battlefield():
        c.attacking = False
        c.blocking = []
        c.blocked_by = []
    game.attackers.clear()
