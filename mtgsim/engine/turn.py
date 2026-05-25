"""Turn structure driver. Yields control to AI at priority windows."""
from __future__ import annotations
from typing import Callable, List
from .enums import Phase, Step, TriggerEvent, Subtype, Keyword
from .game import GameState
from .card import Card
from . import combat as combat_mod


def _empty_mana_pools_all(game: GameState):
    """Per MTG rule 106.4: mana pool empties at end of each step/phase."""
    for p in game.players:
        p.mana_pool.empty()


def untap_step(game: GameState):
    game.step = Step.UNTAP
    ap = game.active()
    ap.reset_turn()
    for c in ap.battlefield:
        c.tapped = False
        c.damage_marked = 0
        c.summoning_sick = False
    _empty_mana_pools_all(game)


def upkeep_step(game: GameState, ai_step: Callable):
    game.step = Step.UPKEEP
    game.fire_event(TriggerEvent.UPKEEP, {})
    game.resolve_all()
    ai_priority(game, ai_step)
    _empty_mana_pools_all(game)


def draw_step(game: GameState, ai_step: Callable, skip: bool = False):
    game.step = Step.DRAW
    ap = game.active()
    if not skip:
        ap.draw(1, game)
        game.fire_event(TriggerEvent.DRAWS_CARD, {"player": ap})
    game.resolve_all()
    ai_priority(game, ai_step)
    _empty_mana_pools_all(game)


def precombat_main(game: GameState, ai_step: Callable):
    game.step = Step.PRECOMBAT_MAIN
    game.phase = Phase.MAIN1
    # saga: at beginning of precombat main, put a lore counter on each saga you control and execute that chapter
    for c in list(game.active().battlefield):
        if Subtype.SAGA in c.cdef.subtypes:
            c.chapter += 1
            game.log(f"{c.name} chapter {c.chapter}")
            if c.cdef.chapters and 1 <= c.chapter <= len(c.cdef.chapters):
                c.cdef.chapters[c.chapter - 1](game, game.players[c.controller_idx], c)
                # if final chapter, sacrifice after the ability resolves
                if c.chapter >= len(c.cdef.chapters):
                    # mark for sacrifice at next SBA
                    c.counters["__sacrifice__"] = 1
    # sacrifice sagas at end of chapter ability resolution
    for c in list(game.active().battlefield):
        if c.counters.pop("__sacrifice__", 0):
            game._move_to_graveyard(c, reason="saga complete")
    game.resolve_all()
    # AI loop: take main-phase actions
    ai_main_loop(game, ai_step)
    _empty_mana_pools_all(game)


def begin_combat(game: GameState, ai_step: Callable):
    game.step = Step.BEGIN_COMBAT
    game.phase = Phase.COMBAT
    game.fire_event(TriggerEvent.BEGIN_COMBAT, {})
    game.resolve_all()
    ai_priority(game, ai_step)
    _empty_mana_pools_all(game)


def declare_attackers_step(game: GameState, ai_step: Callable):
    game.step = Step.DECLARE_ATTACKERS
    attackers = ai_step(game, "declare_attackers")
    if attackers is None:
        attackers = []
    combat_mod.declare_attackers(game, attackers)
    ai_priority(game, ai_step)
    _empty_mana_pools_all(game)


def declare_blockers_step(game: GameState, ai_step: Callable):
    game.step = Step.DECLARE_BLOCKERS
    if not game.attackers:
        return
    blocks = ai_step(game, "declare_blockers")
    if blocks is None:
        blocks = {}
    combat_mod.declare_blockers(game, blocks)
    ai_priority(game, ai_step)
    _empty_mana_pools_all(game)


def first_strike_step(game: GameState, ai_step: Callable):
    game.step = Step.FIRST_STRIKE_DAMAGE
    combat_mod.first_strike_damage(game)
    ai_priority(game, ai_step)
    _empty_mana_pools_all(game)


def combat_damage_step(game: GameState, ai_step: Callable):
    game.step = Step.COMBAT_DAMAGE
    combat_mod.combat_damage(game)
    ai_priority(game, ai_step)
    _empty_mana_pools_all(game)


def end_combat_step(game: GameState, ai_step: Callable):
    game.step = Step.END_COMBAT
    combat_mod.end_combat(game)
    ai_priority(game, ai_step)
    _empty_mana_pools_all(game)


def postcombat_main(game: GameState, ai_step: Callable):
    game.step = Step.POSTCOMBAT_MAIN
    game.phase = Phase.MAIN2
    ai_main_loop(game, ai_step)
    _empty_mana_pools_all(game)


def end_step(game: GameState, ai_step: Callable):
    game.step = Step.END_STEP
    game.phase = Phase.END
    game.fire_event(TriggerEvent.END_STEP, {})
    game.resolve_all()
    ai_priority(game, ai_step)
    _empty_mana_pools_all(game)


def cleanup_step(game: GameState):
    game.step = Step.CLEANUP
    ap = game.active()
    # discard to 7
    while len(ap.hand) > 7:
        # AI: discard lowest-value (simplified: highest cmc — extra lands first)
        ap.hand.sort(key=lambda c: (0 if c.cdef.is_land() and sum(1 for x in ap.hand if x.cdef.is_land()) > 4 else 1,
                                    -(c.cdef.cost.cmc() if c.cdef.cost else 0)))
        d = ap.hand.pop()
        d.zone = ap.graveyard  # type: ignore
        from .enums import Zone as Z
        d.zone = Z.GRAVEYARD
        ap.graveyard.append(d)
        game.log(f"P{ap.idx} discards {d.name}")
    # damage wears off; came_in_this_turn cleared; EOT counters expire
    for c in game.battlefield():
        c.damage_marked = 0
        c.came_in_this_turn = False
        for k in list(c.counters.keys()):
            if k.endswith("_eot") or k.endswith("_eot_p") or k.endswith("_eot_t"):
                c.counters.pop(k, None)
    # decrement light_up_stage exile timer on each player's exile zone
    for p in game.players:
        for c in list(p.exile):
            if "light_up_stage_castable" in c.counters:
                c.counters["light_up_stage_castable"] -= 1
                if c.counters["light_up_stage_castable"] <= 0:
                    c.counters.pop("light_up_stage_castable", None)
    ap.mana_pool.empty()


def ai_priority(game: GameState, ai_step: Callable, only_active: bool = False):
    """Offer priority to active and then nonactive players to cast instants/abilities. Loop until both pass."""
    if game.is_over():
        return
    passes = 0
    order = [game.active_idx, 1 - game.active_idx] if not only_active else [game.active_idx]
    while passes < len(order):
        progressed = False
        for idx in order:
            action = ai_step(game, "priority", player_idx=idx)
            if action:
                progressed = True
                break  # state changed; restart loop
        if progressed:
            passes = 0
            game.resolve_all()
            if game.is_over():
                return
        else:
            passes = len(order)
    game.resolve_all()


def ai_main_loop(game: GameState, ai_step: Callable):
    """In main phase, AP keeps making main-phase actions (land drops, sorceries, abilities, etc) until done."""
    while True:
        if game.is_over():
            return
        action = ai_step(game, "main")
        if not action:
            break
        game.resolve_all()
    # offer priority for instants (mostly NAP)
    ai_priority(game, ai_step)


def take_turn(game: GameState, ai_step: Callable):
    """Run a full turn for the active player."""
    if game.is_over():
        return
    untap_step(game)
    if game.is_over():
        return
    upkeep_step(game, ai_step)
    if game.is_over():
        return
    skip_draw = (game.turn == 1 and game.active_idx == 0)
    draw_step(game, ai_step, skip=skip_draw)
    if game.is_over():
        return
    precombat_main(game, ai_step)
    if game.is_over():
        return
    begin_combat(game, ai_step)
    if game.is_over():
        return
    declare_attackers_step(game, ai_step)
    if game.is_over():
        return
    declare_blockers_step(game, ai_step)
    if game.is_over():
        return
    first_strike_step(game, ai_step)
    if game.is_over():
        return
    combat_damage_step(game, ai_step)
    if game.is_over():
        return
    end_combat_step(game, ai_step)
    if game.is_over():
        return
    postcombat_main(game, ai_step)
    if game.is_over():
        return
    end_step(game, ai_step)
    if game.is_over():
        return
    cleanup_step(game)


def advance_turn(game: GameState):
    game.active_idx = 1 - game.active_idx
    if game.active_idx == 0:
        game.turn += 1
