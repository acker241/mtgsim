"""Engine actions: cast spell, activate ability, play land, deal damage, etc."""
from __future__ import annotations
from typing import List, Optional, Any
from .enums import Zone, TriggerEvent, Keyword, CardType
from .mana import ManaCost, ManaPool, GENERIC, COLOR_SYMS, W, U, B, R, G
from .card import Card
from .game import GameState, StackItem


def tap_for_mana(game: GameState, perm: Card) -> bool:
    """Standard basic land tap. Returns False if can't."""
    if perm.tapped:
        return False
    if not perm.cdef.is_land():
        return False
    perm.tapped = True
    # determine produced
    from .enums import Subtype
    pl = game.players[perm.controller_idx]
    if Subtype.MOUNTAIN in perm.cdef.subtypes:
        pl.mana_pool.add(R, 1)
    elif Subtype.PLAINS in perm.cdef.subtypes:
        pl.mana_pool.add(W, 1)
    else:
        # generic colorless fallback
        from .mana import C
        pl.mana_pool.add(C, 1)
    return True


def play_land(game: GameState, player_idx: int, land: Card) -> bool:
    pl = game.players[player_idx]
    if pl.lands_played_this_turn >= 1:
        return False
    if land not in pl.hand:
        return False
    # move to BF
    pl.hand.remove(land)
    game.move_to_battlefield(land, player_idx)
    pl.lands_played_this_turn += 1
    return True


def _can_pay_with_floating_and_lands(game: GameState, player_idx: int, cost: ManaCost,
                                     allow_convoke: bool = False,
                                     convoke_creatures: Optional[List[Card]] = None) -> bool:
    """Check if player can pay cost using current pool + untapped lands + (optionally) convoke creatures."""
    pl = game.players[player_idx]
    # snapshot pool
    pool = dict(pl.mana_pool.pool)
    # add land potential
    for perm in pl.battlefield:
        if perm.cdef.is_land() and not perm.tapped:
            from .enums import Subtype
            if Subtype.MOUNTAIN in perm.cdef.subtypes:
                pool[R] = pool.get(R, 0) + 1
            elif Subtype.PLAINS in perm.cdef.subtypes:
                pool[W] = pool.get(W, 0) + 1
    # add convoke contributions (each creature taps for 1 of its color or 1 generic)
    if allow_convoke and convoke_creatures:
        for c in convoke_creatures:
            # contribute by color if possible
            from .card import Card as _C
            cols = c.cdef.colors
            from .enums import Color as Col
            for sym, flag in [(W, Col.W), (U, Col.U), (B, Col.B), (R, Col.R), (G, Col.G)]:
                if cols & flag:
                    pool[sym] = pool.get(sym, 0) + 1
                    break
            else:
                # colorless creature: generic only
                from .mana import C as Csym
                pool[Csym] = pool.get(Csym, 0) + 1
    # check
    for sym, n in cost.colored_required().items():
        if pool.get(sym, 0) < n:
            return False
        pool[sym] -= n
    gen = cost.generic_required()
    return sum(pool.values()) >= gen


def pay_cost(game: GameState, player_idx: int, cost: ManaCost,
             tap_lands: Optional[List[Card]] = None,
             convoke_creatures: Optional[List[Card]] = None) -> bool:
    """Auto-tap untapped lands to fill pool, then pay. Returns True on success."""
    pl = game.players[player_idx]
    # tap requested lands first; if none, auto-tap untapped lands in greedy way
    from .enums import Subtype
    if tap_lands is None:
        # collect untapped lands
        tap_lands = [p for p in pl.battlefield if p.cdef.is_land() and not p.tapped]
    # determine how many of each color we need
    need = dict(cost.colored_required())
    gen_need = cost.generic_required()
    # start: pay from existing pool
    pool = pl.mana_pool
    # greedy: tap lands matching colored need first
    used: List[Card] = []
    for sym in (R, W, U, B, G):
        if need.get(sym, 0) > pool.pool.get(sym, 0):
            # tap colored lands
            for land in tap_lands:
                if land in used or land.tapped:
                    continue
                if sym == R and Subtype.MOUNTAIN in land.cdef.subtypes:
                    tap_for_mana(game, land)
                    used.append(land)
                elif sym == W and Subtype.PLAINS in land.cdef.subtypes:
                    tap_for_mana(game, land)
                    used.append(land)
                if pool.pool.get(sym, 0) >= need.get(sym, 0):
                    break
    # convoke contributions: tap chosen creatures
    if convoke_creatures:
        for c in convoke_creatures:
            if c.tapped:
                continue
            c.tapped = True
            # pick color contribution
            from .enums import Color as Col
            added = False
            for sym, flag in [(W, Col.W), (U, Col.U), (B, Col.B), (R, Col.R), (G, Col.G)]:
                if c.cdef.colors & flag and need.get(sym, 0) > pool.pool.get(sym, 0):
                    pool.add(sym, 1)
                    added = True
                    break
            if not added:
                from .mana import C as Csym
                pool.add(Csym, 1)
    # if still need generic, tap any untapped land
    have = sum(pool.pool.values())
    needed_total = sum(need.values()) + gen_need
    if have < needed_total:
        for land in tap_lands:
            if land in used or land.tapped:
                continue
            tap_for_mana(game, land)
            used.append(land)
            if sum(pool.pool.values()) >= needed_total:
                break
    if not pool.can_pay(cost):
        return False
    return pool.pay(cost)


def cast_spell(game: GameState, player_idx: int, card: Card,
               targets: Optional[List[Any]] = None,
               x: int = 0,
               use_spectacle: bool = False,
               convoke_creatures: Optional[List[Card]] = None) -> bool:
    """Cast spell from hand. Pushes on stack. Caller responsible to call game.resolve_all() if applicable."""
    pl = game.players[player_idx]
    if card not in pl.hand:
        return False
    cdef = card.cdef
    cost = cdef.cost
    if use_spectacle and cdef.spectacle_cost is not None:
        # Wizard's Lightning hack: uses spectacle_cost slot for wizard discount
        if cdef.name == "Wizard's Lightning":
            from .enums import Subtype
            has_wiz = any(c.cdef.is_creature() and Subtype.WIZARD in c.cdef.subtypes
                          for c in pl.battlefield)
            if not has_wiz:
                return False
            cost = cdef.spectacle_cost
        else:
            if not pl.opp_lost_life_this_turn:
                return False
            cost = cdef.spectacle_cost
    if cost is None:
        return False
    if cdef.has_x:
        cost = cost.copy()
        cost.x = x
    # validate targets count
    targets = list(targets or [])
    if len(targets) < cdef.needs_targets:
        return False
    # convoke check
    if convoke_creatures:
        # eligible: untapped creatures controller controls that didn't enter this turn (no summoning sickness need not apply to convoke; rules: any untapped creature)
        for c in convoke_creatures:
            if c.controller_idx != player_idx or c.tapped or not c.cdef.is_creature():
                return False
    # can pay?
    if not _can_pay_with_floating_and_lands(game, player_idx, cost,
                                            allow_convoke=bool(convoke_creatures),
                                            convoke_creatures=convoke_creatures or []):
        return False
    # pay
    if not pay_cost(game, player_idx, cost, convoke_creatures=convoke_creatures):
        return False
    # move to stack
    pl.hand.remove(card)
    card.zone = Zone.STACK
    card.targets = targets
    card.x_value = x
    card.cost_paid_alt = "spectacle" if use_spectacle else None
    item = StackItem(
        source=card,
        controller_idx=player_idx,
        is_spell=True,
        targets=targets,
        x=x,
        description=f"cast:{card.name}",
    )
    game.stack.append(item)
    pl.spells_cast_this_turn += 1
    if cdef.is_creature():
        pl.creatures_cast_this_turn += 1
    game.log(f"P{player_idx} casts {card.name} (cost {cost.symbols}{' spectacle' if use_spectacle else ''})")
    # fire CAST event
    game.fire_event(TriggerEvent.CAST, {"card": card, "controller_idx": player_idx})
    return True


def deal_damage(game: GameState, source: Optional[Card], target: Any, n: int,
                is_combat: bool = False) -> int:
    """Apply damage. target: Player or Card. Handles lifelink. Returns damage applied."""
    if n <= 0:
        return 0
    from .player import Player
    if isinstance(target, Player):
        target.lose_life(n, game)
        game.log(f"{source.name if source else '?'} deals {n} to P{target.idx} (life {target.life})")
    else:
        if Keyword.INDESTRUCTIBLE in target.keywords(game) and not is_combat:
            # indestructible damage still marked but won't destroy
            target.damage_marked += n
        else:
            target.damage_marked += n
        game.log(f"{source.name if source else '?'} deals {n} to {target.name} (marked {target.damage_marked}/{target.toughness(game)})")
    # lifelink: source controller gains life equal to damage
    if source and Keyword.LIFELINK in source.keywords(game):
        game.players[source.controller_idx].gain_life(n)
        game.log(f"  lifelink: P{source.controller_idx} gains {n} (life {game.players[source.controller_idx].life})")
    return n
