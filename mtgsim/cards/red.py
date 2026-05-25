"""Mono-Red Aggro card defs."""
from __future__ import annotations
from typing import List, Any
from ..engine.enums import CardType, Subtype, Keyword, Color, TriggerEvent, Zone
from ..engine.mana import ManaCost, R, W
from ..engine.card import CardDef, TriggeredAbility, ActivatedAbility, StaticEffect
from ..engine.actions import deal_damage


# ---------- Mountain ----------
def mountain():
    return CardDef(
        name="Mountain",
        types=CardType.LAND,
        subtypes={Subtype.MOUNTAIN},
        text="Tap: add R",
    )


# ---------- Fanatical Firebrand ----------
def _firebrand_tap_dmg(game, controller, perm, targets):
    if not targets:
        return
    perm.tapped = True
    # sac
    game._move_to_graveyard(perm, "sacrificed for ability")
    deal_damage(game, perm, targets[0], 1)


def fanatical_firebrand():
    cdef = CardDef(
        name="Fanatical Firebrand",
        types=CardType.CREATURE,
        subtypes={Subtype.GOBLIN, Subtype.PIRATE},
        cost=ManaCost.parse("R"),
        colors=Color.R,
        power=1, toughness=1,
        keywords={Keyword.HASTE},
        text="T, Sac: 1 dmg any target",
    )
    cdef.activated.append(ActivatedAbility(
        cost_fn=lambda g, p: not p.tapped,
        effect=_firebrand_tap_dmg,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: True,
        description="T,Sac: 1 dmg",
    ))
    return cdef


# ---------- Ghitu Lavarunner ----------
# 1/2; has haste as long as 2+ instants/sorceries in your graveyard.
def _ghitu_haste(game, src, target):
    if target.cid != src.cid:
        return (0, 0, set())
    ctrl = game.players[src.controller_idx]
    count = sum(1 for c in ctrl.graveyard if c.cdef.is_instant() or c.cdef.is_sorcery())
    if count >= 2:
        return (0, 0, {Keyword.HASTE})
    return (0, 0, set())


def ghitu_lavarunner():
    cdef = CardDef(
        name="Ghitu Lavarunner",
        types=CardType.CREATURE,
        subtypes={Subtype.HUMAN, Subtype.WIZARD},
        cost=ManaCost.parse("R"),
        colors=Color.R,
        power=2, toughness=2,
        text="As long as 2+ instants/sorceries in your graveyard, Ghitu Lavarunner has haste.",
    )
    # static self-haste
    cdef.static_mods.append(StaticEffect(apply=_ghitu_haste, description="self-haste conditional"))
    return cdef


# Actually Ghitu Lavarunner is 1/2 base, gets +1/+0 if condition? Let me re-check…
# No: Ghitu Lavarunner is 2/2. "Ghitu Lavarunner can attack as though it had haste" if 2+ instants/sorceries in graveyard.
# I'll keep haste impl.


# ---------- Viashino Pyromancer ----------
def _viashino_etb(game, controller, perm):
    opp = game.opponent_of(controller.idx)
    deal_damage(game, perm, opp, 2)


def viashino_pyromancer():
    cdef = CardDef(
        name="Viashino Pyromancer",
        types=CardType.CREATURE,
        subtypes={Subtype.VIASHINO, Subtype.WIZARD},
        cost=ManaCost.parse("1R"),
        colors=Color.R,
        power=2, toughness=1,
        text="ETB: 2 damage to an opponent",
        on_etb=_viashino_etb,
    )
    return cdef


# ---------- Runaway Steam-Kin ----------
def _steamkin_cast_red(game, src, data):
    spell = data.get("card")
    if not spell:
        return False
    if spell.cdef.colors & Color.R == 0:
        return False
    if spell.cid == src.cid:
        return False  # itself doesn't trigger
    if src.counters.get("+1/+1", 0) >= 3:
        return False
    return True


def _steamkin_cast_red_effect(game, src, data):
    src.counters["+1/+1"] = src.counters.get("+1/+1", 0) + 1
    game.log(f"Steam-Kin +1/+1 counter (now {src.counters['+1/+1']})")


def _steamkin_activated(game, controller, perm, targets):
    if perm.counters.get("+1/+1", 0) < 3:
        return
    perm.counters["+1/+1"] -= 3
    controller.mana_pool.add(R, 3)
    game.log(f"Steam-Kin: removed 3 counters, added RRR")


def runaway_steamkin():
    cdef = CardDef(
        name="Runaway Steam-Kin",
        types=CardType.CREATURE,
        subtypes={Subtype.ELEMENTAL},
        cost=ManaCost.parse("RR"),
        colors=Color.R,
        power=1, toughness=1,
        text="Whenever you cast red spell, +1/+1 counter (max 3). Remove 3: add RRR.",
    )
    cdef.triggers.append(TriggeredAbility(
        event=TriggerEvent.CAST,
        condition=_steamkin_cast_red,
        effect=_steamkin_cast_red_effect,
        description="cast-red->+1/+1",
    ))
    cdef.activated.append(ActivatedAbility(
        cost_fn=lambda g, p: p.counters.get("+1/+1", 0) >= 3,
        effect=_steamkin_activated,
        is_mana=True,
        description="remove 3: RRR",
    ))
    return cdef


# ---------- Goblin Chainwhirler ----------
def _chainwhirler_etb(game, controller, perm):
    opp = game.opponent_of(controller.idx)
    deal_damage(game, perm, opp, 1)
    # 1 dmg to each creature opponents control
    for c in list(opp.battlefield):
        if c.cdef.is_creature():
            deal_damage(game, perm, c, 1)
        if c.cdef.is_planeswalker():
            # damage to planeswalker = loss of loyalty counters
            c.counters["loyalty"] = c.counters.get("loyalty", 0) - 1


def goblin_chainwhirler():
    cdef = CardDef(
        name="Goblin Chainwhirler",
        types=CardType.CREATURE,
        subtypes={Subtype.GOBLIN, Subtype.WARRIOR},
        cost=ManaCost.parse("RRR"),
        colors=Color.R,
        power=3, toughness=3,
        keywords={Keyword.FIRST_STRIKE},
        text="ETB: 1 dmg to each opp, each creature opp controls, each PW opp controls",
        on_etb=_chainwhirler_etb,
    )
    return cdef


# ---------- Shock ----------
def _shock_resolve(game, controller, card, targets, x):
    if targets:
        deal_damage(game, card, targets[0], 2)


def shock():
    return CardDef(
        name="Shock",
        types=CardType.INSTANT,
        cost=ManaCost.parse("R"),
        colors=Color.R,
        text="2 damage any target",
        on_resolve=_shock_resolve,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: True,
    )


# ---------- Lightning Strike ----------
def _lstrike_resolve(game, controller, card, targets, x):
    if targets:
        deal_damage(game, card, targets[0], 3)


def lightning_strike():
    return CardDef(
        name="Lightning Strike",
        types=CardType.INSTANT,
        cost=ManaCost.parse("1R"),
        colors=Color.R,
        text="3 damage any target",
        on_resolve=_lstrike_resolve,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: True,
    )


# ---------- Wizard's Lightning ----------
def _wizards_lightning_resolve(game, controller, card, targets, x):
    if targets:
        deal_damage(game, card, targets[0], 3)


def wizards_lightning():
    # base 2R; reduced to R if controller controls a wizard.
    # We'll model reduction at cast-time in AI: try to cast with R if has wizard.
    # Engine-side: we keep cost at 2R; AI checks reduction and pays accordingly.
    # Simplest: we expose both costs via two paths.
    cdef = CardDef(
        name="Wizard's Lightning",
        types=CardType.INSTANT,
        cost=ManaCost.parse("2R"),
        colors=Color.R,
        text="3 damage any target; costs R if you control a Wizard",
        on_resolve=_wizards_lightning_resolve,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: True,
    )
    # alternative reduced cost stored as spectacle slot (reusing field for alt-cost gating)
    cdef.spectacle_cost = ManaCost.parse("R")  # only valid when controller has a wizard — gating in AI/cast
    # We'll use a custom flag — actually let's add a separate field via wizard-discount logic in cast.
    # For simplicity, repurpose: AI will check; cast_spell uses use_spectacle for the alt R cost.
    return cdef


# ---------- Light Up the Stage ----------
def _light_up_stage_resolve(game, controller, card, targets, x):
    # exile top 2 cards; mark as castable until end of next turn (we'll simplify: allow casting from exile until end of opponent's next turn)
    n = 2
    exiled = []
    for _ in range(n):
        if controller.library:
            c = controller.library.pop(0)
            c.zone = Zone.EXILE
            c.counters["light_up_stage_castable"] = 2  # decrement at each cleanup; cast when >0
            controller.exile.append(c)
            exiled.append(c)
    game.log(f"Light Up the Stage exiled {len(exiled)} cards")


def light_up_the_stage():
    cdef = CardDef(
        name="Light Up the Stage",
        types=CardType.SORCERY,
        cost=ManaCost.parse("2R"),
        colors=Color.R,
        text="Exile top 2, may play UEOT next turn. Spectacle R.",
        on_resolve=_light_up_stage_resolve,
        spectacle_cost=ManaCost.parse("R"),
    )
    return cdef


# ---------- Skewer the Critics ----------
def _skewer_resolve(game, controller, card, targets, x):
    if targets:
        deal_damage(game, card, targets[0], 3)


def skewer_the_critics():
    return CardDef(
        name="Skewer the Critics",
        types=CardType.SORCERY,
        cost=ManaCost.parse("2R"),
        colors=Color.R,
        text="3 damage any target. Spectacle R.",
        on_resolve=_skewer_resolve,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: True,
        spectacle_cost=ManaCost.parse("R"),
    )


# ---------- Experimental Frenzy ----------
# Static: you may play the top of your library. Plus restriction: you can't play cards from hand.
# Activated: 3R, sac: put top of library into graveyard? — Actually: 3R: Destroy Experimental Frenzy. Skip for AI complexity.
# We'll model as flag on controller: ai_data['experimental_frenzy_active']=True; ai handles play-from-top.
def _frenzy_etb(game, controller, perm):
    controller.ai_data["experimental_frenzy"] = perm.cid


def _frenzy_ltb(game, controller, perm):
    if controller.ai_data.get("experimental_frenzy") == perm.cid:
        controller.ai_data.pop("experimental_frenzy", None)


def experimental_frenzy():
    cdef = CardDef(
        name="Experimental Frenzy",
        types=CardType.ENCHANTMENT,
        cost=ManaCost.parse("3R"),
        colors=Color.R,
        text="Play top of library; can't play from hand. 3R: destroy this.",
        on_etb=_frenzy_etb,
        on_ltb=_frenzy_ltb,
    )
    return cdef


# ---------- Banefire (SB) ----------
def _banefire_resolve(game, controller, card, targets, x):
    if targets:
        deal_damage(game, card, targets[0], x)


def banefire():
    return CardDef(
        name="Banefire",
        types=CardType.SORCERY,
        cost=ManaCost.parse("XR"),
        colors=Color.R,
        text="X damage; if X>=5, uncounterable/unpreventable",
        on_resolve=_banefire_resolve,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: True,
        has_x=True,
    )


# ---------- Dire Fleet Daredevil (SB) ----------
# ETB: exile target instant/sorcery from opp graveyard; you may cast it this turn. (Complex.)
# Simplified: ETB no-op (skip casting from exile). Still useful as 2/1 FS body.
def dire_fleet_daredevil():
    return CardDef(
        name="Dire Fleet Daredevil",
        types=CardType.CREATURE,
        subtypes={Subtype.HUMAN, Subtype.PIRATE},
        cost=ManaCost.parse("1R"),
        colors=Color.R,
        power=2, toughness=1,
        keywords={Keyword.FIRST_STRIKE},
        text="ETB: (simplified) [exile inst/sor from opp grave skipped]",
    )


# ---------- Fiery Cannonade (SB) ----------
def _fiery_cannonade_resolve(game, controller, card, targets, x):
    opp = game.opponent_of(controller.idx)
    # 2 dmg to each non-Pirate creature
    for p in game.players:
        for c in list(p.battlefield):
            if c.cdef.is_creature() and Subtype.PIRATE not in c.cdef.subtypes:
                deal_damage(game, card, c, 2)


def fiery_cannonade():
    return CardDef(
        name="Fiery Cannonade",
        types=CardType.INSTANT,
        cost=ManaCost.parse("2R"),
        colors=Color.R,
        text="2 damage to each non-Pirate creature",
        on_resolve=_fiery_cannonade_resolve,
    )


# ---------- Fight with Fire (SB) ----------
def _fight_with_fire_resolve(game, controller, card, targets, x):
    # if x >= 5 (kicker paid), 5 damage divided as you choose; else 5 dmg to one target
    # simplified: deal 5 dmg to first target
    if targets:
        deal_damage(game, card, targets[0], 5)
    # ignore kicker mode for sim (rare in this deck)


def fight_with_fire():
    return CardDef(
        name="Fight with Fire",
        types=CardType.SORCERY,
        cost=ManaCost.parse("4R"),
        colors=Color.R,
        text="5 damage. Kicker 5R: 5 divided as you choose.",
        on_resolve=_fight_with_fire_resolve,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: True,
    )


# ---------- Lava Coil (SB) ----------
def _lava_coil_resolve(game, controller, card, targets, x):
    if not targets:
        return
    t = targets[0]
    from ..engine.player import Player
    if isinstance(t, Player):
        return
    if not t.cdef.is_creature():
        return
    deal_damage(game, card, t, 4)
    # if it would die, exile instead — simplified: if marked >= toughness, exile
    if t.damage_marked >= t.toughness(game):
        game.exile_card(t, "Lava Coil")


def lava_coil():
    return CardDef(
        name="Lava Coil",
        types=CardType.SORCERY,
        cost=ManaCost.parse("1R"),
        colors=Color.R,
        text="4 damage target creature; if dies, exile.",
        on_resolve=_lava_coil_resolve,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: hasattr(cand, "cdef") and cand.cdef.is_creature(),
    )


# ---------- Rekindling Phoenix (SB) ----------
def _rekindling_dies(game, controller, perm):
    # create a 0/1 red Elemental token; at controller's upkeep, may sac token to return Phoenix from graveyard
    from ..engine.card import Card as CardInst
    token_def = CardDef(
        name="Elemental token (Rekindling)",
        types=CardType.CREATURE,
        subtypes={Subtype.ELEMENTAL},
        colors=Color.R,
        power=0, toughness=1,
        text="0/1 elemental token; sac at upkeep returns Phoenix",
    )
    # the token, on its controller's upkeep, sacs itself to return Rekindling Phoenix from graveyard
    def _tok_upkeep(g, src, data):
        # only trigger on controller's upkeep
        if g.active_idx != src.controller_idx:
            return False
        return True

    def _tok_upkeep_eff(g, src, data):
        # sac token; if so, return Phoenix from grave to BF
        owner_grave = g.players[src.owner_idx].graveyard
        phoenix = None
        for cc in owner_grave:
            if cc.name == "Rekindling Phoenix":
                phoenix = cc
                break
        if phoenix:
            g._move_to_graveyard(src, "sac for Phoenix return")
            owner_grave.remove(phoenix)
            g.move_to_battlefield(phoenix, src.controller_idx)
            g.log("Rekindling Phoenix returns")

    token_def.triggers.append(TriggeredAbility(
        event=TriggerEvent.UPKEEP,
        condition=_tok_upkeep,
        effect=_tok_upkeep_eff,
        description="sac->return phoenix",
    ))
    tok = CardInst(cid=game.alloc_cid(), card_def=token_def,
                   owner_idx=controller.idx, controller_idx=controller.idx, is_token=True)
    game.move_to_battlefield(tok, controller.idx)


def rekindling_phoenix():
    cdef = CardDef(
        name="Rekindling Phoenix",
        types=CardType.CREATURE,
        subtypes={Subtype.PHOENIX},
        cost=ManaCost.parse("3R"),
        colors=Color.R,
        power=4, toughness=3,
        keywords={Keyword.FLYING, Keyword.HASTE},
        text="When dies: create 0/1 elemental token. At your upkeep, sac it: return Phoenix.",
        on_dies=_rekindling_dies,
    )
    return cdef


# ---------- Treasure Map (SB) ----------
# Simplified: 2 mana artifact; T,1: scry 1, add a counter; if 3+ counters, transform: tap, sac treasure for 1 any color, draw card.
# We'll implement minimal: T,1: add R (treat as mana rock after transform). For sim simplicity: T to add 1 generic (after a few turns).
def treasure_map():
    cdef = CardDef(
        name="Treasure Map",
        types=CardType.ARTIFACT,
        cost=ManaCost.parse("2"),
        text="(simplified) Tap: add 1 mana (as treasure ramp)",
    )
    # tap for 1 generic
    def _tap_for_one(g, ctrl, perm, targets):
        if perm.tapped:
            return
        perm.tapped = True
        from ..engine.mana import C as Csym
        ctrl.mana_pool.add(Csym, 1)
    cdef.activated.append(ActivatedAbility(
        cost_fn=lambda g, p: not p.tapped,
        effect=_tap_for_one,
        is_mana=True,
        description="T: +1",
    ))
    return cdef


def red_deck() -> list:
    """Return list of (CardDef, count) for the main deck. Mountains last."""
    return [
        (fanatical_firebrand(), 2),
        (ghitu_lavarunner(), 4),
        (viashino_pyromancer(), 4),
        (runaway_steamkin(), 4),
        (goblin_chainwhirler(), 4),
        (shock(), 4),
        (lightning_strike(), 4),
        (wizards_lightning(), 4),
        (light_up_the_stage(), 4),
        (skewer_the_critics(), 4),
        (experimental_frenzy(), 2),
        (mountain(), 20),
    ]


def red_sideboard() -> list:
    return [
        (banefire(), 2),
        (dire_fleet_daredevil(), 2),
        (fiery_cannonade(), 2),
        (fight_with_fire(), 1),
        (lava_coil(), 4),
        (rekindling_phoenix(), 2),
        (treasure_map(), 2),
    ]
