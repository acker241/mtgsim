"""Mono-White Aggro card defs."""
from __future__ import annotations
from typing import List, Any
from ..engine.enums import CardType, Subtype, Keyword, Color, TriggerEvent, Zone
from ..engine.mana import ManaCost, W
from ..engine.card import CardDef, TriggeredAbility, ActivatedAbility, StaticEffect, Card
from ..engine.actions import deal_damage


# ---------- Plains ----------
def plains():
    return CardDef(
        name="Plains",
        types=CardType.LAND,
        subtypes={Subtype.PLAINS},
        text="T: add W",
    )


# ---------- Dauntless Bodyguard ----------
# 1/1 Knight. ETB: choose another creature. Sac: chosen has indestructible UEOT.
def _bodyguard_etb(game, controller, perm):
    # choose another creature: pick most valuable (highest power) friendly creature
    candidates = [c for c in controller.battlefield if c.cdef.is_creature() and c.cid != perm.cid]
    if not candidates:
        return
    candidates.sort(key=lambda c: -(c.power(game) + c.toughness(game)))
    perm.ai_choice = candidates[0].cid


def _bodyguard_sac_eff(game, controller, perm, targets):
    chosen = getattr(perm, "ai_choice", None)
    if chosen is None:
        return
    by_cid = {c.cid: c for c in game.battlefield()}
    target = by_cid.get(chosen)
    game._move_to_graveyard(perm, "sac for indestructible")
    if target:
        target.counters["indestructible_eot"] = 1
        # apply via static? Simpler: just add a temp keyword via per-perm marker
        game.log(f"{target.name} indestructible UEOT")


def dauntless_bodyguard():
    cdef = CardDef(
        name="Dauntless Bodyguard",
        types=CardType.CREATURE,
        subtypes={Subtype.HUMAN, Subtype.KNIGHT},
        cost=ManaCost.parse("W"),
        colors=Color.W,
        power=2, toughness=1,
        text="ETB: choose creature. Sac: chosen indestructible UEOT.",
        on_etb=_bodyguard_etb,
    )
    cdef.activated.append(ActivatedAbility(
        cost_fn=lambda g, p: True,
        effect=_bodyguard_sac_eff,
        description="sac: indestructible UEOT",
    ))
    return cdef


# Patch: damage step checks 'indestructible_eot' counter via static keyword apply
def _indestructible_eot_static(game, src, target):
    if target.counters.get("indestructible_eot", 0) > 0 and target.cid == src.cid:
        return (0, 0, {Keyword.INDESTRUCTIBLE})
    return (0, 0, set())


# We won't bother making each card carry this; instead patch state-based check to honor the counter directly.
# (See game.check_state_based — we add the check there.)


# ---------- Skymarcher Aspirant ----------
def _skymarcher_flying(game, src, target):
    if target.cid != src.cid:
        return (0, 0, set())
    ctrl = game.players[src.controller_idx]
    if ctrl.city_blessing:
        return (0, 0, {Keyword.FLYING})
    return (0, 0, set())


def skymarcher_aspirant():
    cdef = CardDef(
        name="Skymarcher Aspirant",
        types=CardType.CREATURE,
        subtypes={Subtype.VAMPIRE, Subtype.SOLDIER},
        cost=ManaCost.parse("W"),
        colors=Color.W,
        power=1, toughness=1,
        text="W: Skymarcher has flying. (gain flying with city's blessing)",
    )
    cdef.static_mods.append(StaticEffect(apply=_skymarcher_flying, description="flying w/ blessing"))
    return cdef


# Actually Skymarcher Aspirant is 1/1, has flying with city's blessing, AND has activated ability W:scale... simplified: 1/1 with conditional flying. Good enough.


# ---------- Snubhorn Sentry ----------
# 1/4 defender; can attack if city's blessing.
def _snubhorn_can_attack(game, src, target):
    if target.cid != src.cid:
        return (0, 0, set())
    ctrl = game.players[src.controller_idx]
    if ctrl.city_blessing:
        # we don't have an anti-keyword. Instead, the card's can_attack must allow it.
        # Use a "lose-defender" hack: we'll remove DEFENDER via a sentinel keyword. Cleaner: handle in can_attack override.
        return (0, 0, {Keyword.HASTE})  # not the same; but workaround: with blessing, just give haste? no.
    return (0, 0, set())


def snubhorn_sentry():
    cdef = CardDef(
        name="Snubhorn Sentry",
        types=CardType.CREATURE,
        subtypes={Subtype.SOLDIER},  # actual card: Dinosaur Soldier (skipping DINOSAUR enum — irrelevant)
        cost=ManaCost.parse("W"),
        colors=Color.W,
        power=1, toughness=4,
        keywords={Keyword.DEFENDER},
        text="Can attack as though it didn't have defender, while city's blessing.",
    )
    # tag so can_attack helper allows
    cdef.text += " [ascend_attack]"
    return cdef


# Override Card.can_attack? Simpler: add a check in AI's attacker selection (treats ascend-attack tag).
# We'll handle ASCend-attack via a custom check by name in ai.


# ---------- Healer's Hawk ----------
def healers_hawk():
    return CardDef(
        name="Healer's Hawk",
        types=CardType.CREATURE,
        subtypes={Subtype.BIRD},
        cost=ManaCost.parse("W"),
        colors=Color.W,
        power=1, toughness=1,
        keywords={Keyword.FLYING, Keyword.LIFELINK},
        text="1/1 flying lifelink",
    )


# ---------- Tithe Taker ----------
# 2/1 Human Cleric. Spells opp casts during your turn cost 1 more. When dies: create 1/1 W Spirit flying token.
def _tithe_dies(game, controller, perm):
    from ..engine.card import Card as CardInst
    tok_def = CardDef(
        name="Spirit token",
        types=CardType.CREATURE,
        subtypes=set(),
        colors=Color.W,
        power=1, toughness=1,
        keywords={Keyword.FLYING},
    )
    tok = CardInst(cid=game.alloc_cid(), card_def=tok_def,
                   owner_idx=controller.idx, controller_idx=controller.idx, is_token=True)
    game.move_to_battlefield(tok, controller.idx)


def tithe_taker():
    cdef = CardDef(
        name="Tithe Taker",
        types=CardType.CREATURE,
        subtypes={Subtype.HUMAN},
        cost=ManaCost.parse("1W"),
        colors=Color.W,
        power=2, toughness=1,
        text="Opp spells on your turn cost 1 more. ETB: also makes Spirit token (simplified: token on death).",
        on_dies=_tithe_dies,
    )
    # NB: opp-spells-cost-more is not modeled in AI (rare trigger). Skipped for sim simplicity.
    return cdef


# ---------- Benalish Marshal ----------
def _marshal_static(game, src, target):
    if target.cid == src.cid:
        return (0, 0, set())
    if target.controller_idx != src.controller_idx:
        return (0, 0, set())
    if not target.cdef.is_creature():
        return (0, 0, set())
    return (1, 1, set())


def benalish_marshal():
    cdef = CardDef(
        name="Benalish Marshal",
        types=CardType.CREATURE,
        subtypes={Subtype.HUMAN, Subtype.KNIGHT, Subtype.SOLDIER},
        cost=ManaCost.parse("WWW"),
        colors=Color.W,
        power=3, toughness=3,
        text="Other creatures you control get +1/+1.",
    )
    cdef.static_mods.append(StaticEffect(apply=_marshal_static, description="+1/+1 to other creatures"))
    return cdef


# ---------- Venerated Loxodon ----------
# 4/4 4W convoke. ETB: each creature that convoked it gets +1/+1 counter.
def _loxodon_etb(game, controller, perm):
    # x = convoke creatures used (stored on perm.ai_data via cast helper)
    used = perm.ai_choice if hasattr(perm, "ai_choice") else 0
    if isinstance(used, list):
        # apply +1/+1 to each used creature
        for c in used:
            if c in controller.battlefield:
                c.counters["+1/+1"] = c.counters.get("+1/+1", 0) + 1


def venerated_loxodon():
    cdef = CardDef(
        name="Venerated Loxodon",
        types=CardType.CREATURE,
        subtypes={Subtype.ELEPHANT, Subtype.SOLDIER},
        cost=ManaCost.parse("4W"),
        colors=Color.W,
        power=4, toughness=4,
        text="Convoke. ETB: +1/+1 counter on each creature that convoked it.",
        on_etb=_loxodon_etb,
    )
    return cdef


# ---------- Legion's Landing ----------
# Legendary enchantment. ETB: create a 1/1 W Vampire lifelink. When you attack with 3+ creatures, transform.
def _legions_landing_etb(game, controller, perm):
    from ..engine.card import Card as CardInst
    tok_def = CardDef(
        name="Vampire token",
        types=CardType.CREATURE,
        subtypes={Subtype.VAMPIRE},
        colors=Color.W,
        power=1, toughness=1,
        keywords={Keyword.LIFELINK},
    )
    tok = CardInst(cid=game.alloc_cid(), card_def=tok_def,
                   owner_idx=controller.idx, controller_idx=controller.idx, is_token=True)
    game.move_to_battlefield(tok, controller.idx)


def _legions_landing_attacks_trigger(game, src, data):
    ctrl = game.players[src.controller_idx]
    if src.flipped:
        return False
    if ctrl.attacked_with < 3:
        return False
    return True


def _legions_landing_flip(game, src, data):
    # transform into Adanto, the First Fort (a land)
    src.flipped = True
    # change card_def to Adanto on the fly via wrapper land def
    # simpler: replace its cdef with a new land-type cdef
    src.card_def = _adanto_def()
    game.log("Legion's Landing flips → Adanto, the First Fort")


def _adanto_def():
    cdef = CardDef(
        name="Adanto, the First Fort",
        types=CardType.LAND,
        legendary=True,
        subtypes={Subtype.PLAINS},
        text="T: add W. 1W, T: create 1/1 Vampire token with lifelink.",
    )

    def _make_vamp(g, ctrl, perm, targets):
        from ..engine.mana import W as Wsym
        if perm.tapped:
            return
        # cost: 1W and tap
        cost = ManaCost.parse("1W")
        from ..engine.actions import pay_cost, _can_pay_with_floating_and_lands
        if not _can_pay_with_floating_and_lands(g, ctrl.idx, cost):
            return
        if not pay_cost(g, ctrl.idx, cost):
            return
        perm.tapped = True
        tok_def = CardDef(
            name="Vampire token",
            types=CardType.CREATURE,
            subtypes={Subtype.VAMPIRE},
            colors=Color.W,
            power=1, toughness=1,
            keywords={Keyword.LIFELINK},
        )
        tok = Card(cid=g.alloc_cid(), card_def=tok_def, owner_idx=ctrl.idx,
                   controller_idx=ctrl.idx, is_token=True)
        g.move_to_battlefield(tok, ctrl.idx)

    cdef.activated.append(ActivatedAbility(
        cost_fn=lambda g, p: not p.tapped,
        effect=_make_vamp,
        description="1W,T: vamp token",
    ))
    return cdef


def legions_landing():
    cdef = CardDef(
        name="Legion's Landing",
        types=CardType.ENCHANTMENT,
        legendary=True,
        cost=ManaCost.parse("W"),
        colors=Color.W,
        text="Legendary. ETB: 1/1 W Vampire lifelink. Attack with 3+: transform.",
        on_etb=_legions_landing_etb,
    )
    cdef.triggers.append(TriggeredAbility(
        event=TriggerEvent.ATTACKS,
        condition=lambda g, s, d: _legions_landing_attacks_trigger(g, s, d),
        effect=_legions_landing_flip,
        description="3+ attack -> flip",
    ))
    return cdef


# ---------- History of Benalia ----------
def _knight_token(game, controller):
    tok_def = CardDef(
        name="Knight token",
        types=CardType.CREATURE,
        subtypes={Subtype.KNIGHT},
        colors=Color.W,
        power=2, toughness=2,
        keywords={Keyword.VIGILANCE},
    )
    tok = Card(cid=game.alloc_cid(), card_def=tok_def, owner_idx=controller.idx,
               controller_idx=controller.idx, is_token=True)
    game.move_to_battlefield(tok, controller.idx)


def _hob_chap_i(game, controller, perm):
    _knight_token(game, controller)


def _hob_chap_ii(game, controller, perm):
    _knight_token(game, controller)


def _hob_chap_iii(game, controller, perm):
    # Knights +2/+1 UEOT
    for c in controller.battlefield:
        if c.cdef.is_creature() and Subtype.KNIGHT in c.cdef.subtypes:
            c.counters["+2/+1_eot_p"] = c.counters.get("+2/+1_eot_p", 0) + 2
            c.counters["+2/+1_eot_t"] = c.counters.get("+2/+1_eot_t", 0) + 1


def _hob_etb(game, controller, perm):
    """Dominaria-era saga rule: 'As this Saga enters', add lore counter and fire chapter I."""
    perm.chapter = 1
    _hob_chap_i(game, controller, perm)


def history_of_benalia():
    cdef = CardDef(
        name="History of Benalia",
        types=CardType.ENCHANTMENT,
        subtypes={Subtype.SAGA},
        cost=ManaCost.parse("1WW"),
        colors=Color.W,
        text="Saga I,II: 2/2 Knight vigilance token. III: Knights +2/+1 UEOT. (Dominaria timing: chapter I fires on ETB.)",
        chapters=[_hob_chap_i, _hob_chap_ii, _hob_chap_iii],
        on_etb=_hob_etb,
    )
    return cdef


# ---------- Conclave Tribunal ----------
# 3W convoke. ETB: exile target nonland permanent of opp until this leaves.
def _tribunal_etb(game, controller, perm):
    # pick target stored on perm.targets (set during cast)
    if not perm.targets:
        return
    target = perm.targets[0]
    if not hasattr(target, "cdef") or target.cdef.is_land():
        return
    # exile it; remember to return on LTB
    perm.ai_data = {"exiled_cid": target.cid, "exiled_controller": target.controller_idx,
                    "exiled_card_ref": target}
    game.exile_card(target, "Conclave Tribunal")


def _tribunal_ltb(game, controller, perm):
    data = getattr(perm, "ai_data", None) or {}
    target = data.get("exiled_card_ref")
    if target and target.zone == Zone.EXILE:
        # return to BF under original controller
        owner = game.players[target.owner_idx]
        if target in owner.exile:
            owner.exile.remove(target)
        game.move_to_battlefield(target, data["exiled_controller"])


def conclave_tribunal():
    cdef = CardDef(
        name="Conclave Tribunal",
        types=CardType.ENCHANTMENT,
        cost=ManaCost.parse("3W"),
        colors=Color.W,
        text="Convoke. ETB: exile target nonland perm until this leaves.",
        on_etb=_tribunal_etb,
        on_ltb=_tribunal_ltb,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: (hasattr(cand, "cdef") and not cand.cdef.is_land()
                                             and cand.controller_idx != ctrl.idx),
    )
    return cdef


# ---------- Unbreakable Formation ----------
def _unbreakable_resolve(game, controller, card, targets, x):
    in_main = game.phase.name in ("MAIN1", "MAIN2")
    for c in controller.battlefield:
        if c.cdef.is_creature():
            c.counters["indestructible_eot"] = 1
            if in_main:
                c.counters["+1/+1"] = c.counters.get("+1/+1", 0) + 1
                c.counters["vigilance_eot"] = 1


def unbreakable_formation():
    return CardDef(
        name="Unbreakable Formation",
        types=CardType.INSTANT,
        cost=ManaCost.parse("2W"),
        colors=Color.W,
        text="Creatures you control indestructible UEOT. Addendum: +1/+1 counters + vigilance.",
        on_resolve=_unbreakable_resolve,
    )


# ---------- Tocatli Honor Guard (SB) ----------
# Static: creatures entering BF don't have ETB triggers. (Affects opp & you.)
# We model with a flag stored on game (any Tocatli on BF -> on_etb suppressed via central check).
def tocatli_honor_guard():
    cdef = CardDef(
        name="Tocatli Honor Guard",
        types=CardType.CREATURE,
        subtypes={Subtype.HUMAN},
        cost=ManaCost.parse("1W"),
        colors=Color.W,
        power=1, toughness=3,
        text="Creatures ETB don't trigger abilities. (We suppress on_etb when this is on BF.)",
    )
    return cdef


# ---------- Baffling End (SB) ----------
def _baffling_etb(game, controller, perm):
    if not perm.targets:
        return
    target = perm.targets[0]
    if not hasattr(target, "cdef") or not target.cdef.is_creature():
        return
    cost = target.cdef.cost
    if cost and cost.cmc() > 3:
        return
    perm.ai_data = {"exiled_card_ref": target, "exiled_orig_controller": target.controller_idx}
    game.exile_card(target, "Baffling End")


def _baffling_ltb(game, controller, perm):
    data = getattr(perm, "ai_data", None) or {}
    orig_ctrl = data.get("exiled_orig_controller")
    if orig_ctrl is None:
        return
    # create 3/3 dinosaur token for that player
    tok_def = CardDef(
        name="Dinosaur token",
        types=CardType.CREATURE,
        subtypes=set(),
        colors=Color.NONE,
        power=3, toughness=3,
    )
    tok = Card(cid=game.alloc_cid(), card_def=tok_def, owner_idx=orig_ctrl,
               controller_idx=orig_ctrl, is_token=True)
    game.move_to_battlefield(tok, orig_ctrl)


def baffling_end():
    cdef = CardDef(
        name="Baffling End",
        types=CardType.ENCHANTMENT,
        cost=ManaCost.parse("1W"),
        colors=Color.W,
        text="ETB: exile target creature cmc≤3. LTB: 3/3 token for orig controller.",
        on_etb=_baffling_etb,
        on_ltb=_baffling_ltb,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: (hasattr(cand, "cdef") and cand.cdef.is_creature()
                                             and (cand.cdef.cost.cmc() if cand.cdef.cost else 0) <= 3
                                             and cand.controller_idx != ctrl.idx),
    )
    return cdef


# ---------- Ajani, Adversary of Tyrants (SB) ----------
# Planeswalker, loyalty 4. +1: +1/+1 counters on up to 2 target creatures. -2: 2 1/1 cat tokens w/ lifelink.
def _ajani_etb(game, controller, perm):
    perm.counters["loyalty"] = perm.cdef.starting_loyalty or 4


def _ajani_plus1(game, controller, perm, targets):
    perm.counters["loyalty"] = perm.counters.get("loyalty", 0) + 1
    # apply +1/+1 to up to 2 friendly creatures
    creatures = [c for c in controller.battlefield if c.cdef.is_creature()]
    creatures.sort(key=lambda c: -(c.power(game) + c.toughness(game)))
    for c in creatures[:2]:
        c.counters["+1/+1"] = c.counters.get("+1/+1", 0) + 1


def _ajani_minus2(game, controller, perm, targets):
    perm.counters["loyalty"] = perm.counters.get("loyalty", 0) - 2
    for _ in range(2):
        tok_def = CardDef(
            name="Cat token",
            types=CardType.CREATURE,
            subtypes={Subtype.CAT},
            colors=Color.W,
            power=1, toughness=1,
            keywords={Keyword.LIFELINK},
        )
        tok = Card(cid=game.alloc_cid(), card_def=tok_def,
                   owner_idx=controller.idx, controller_idx=controller.idx, is_token=True)
        game.move_to_battlefield(tok, controller.idx)


def ajani_adversary_of_tyrants():
    cdef = CardDef(
        name="Ajani, Adversary of Tyrants",
        types=CardType.PLANESWALKER,
        legendary=True,
        subtypes={Subtype.AJANI},
        cost=ManaCost.parse("2WW"),
        colors=Color.W,
        text="+1: +1/+1 to 2 creatures. -2: two 1/1 cat lifelink tokens.",
        starting_loyalty=4,
        on_etb=_ajani_etb,
    )
    cdef.activated.append(ActivatedAbility(cost_fn=lambda g, p: True, effect=_ajani_plus1, description="+1"))
    cdef.activated.append(ActivatedAbility(cost_fn=lambda g, p: p.counters.get("loyalty", 0) >= 2,
                                           effect=_ajani_minus2, description="-2"))
    return cdef


# ---------- Adanto Vanguard (SB) ----------
# 1/1 Vampire Soldier. Pay 4 life: indestructible UEOT.
def _adanto_van_indestructible(game, controller, perm, targets):
    controller.lose_life(4, game)
    perm.counters["indestructible_eot"] = 1


def adanto_vanguard():
    cdef = CardDef(
        name="Adanto Vanguard",
        types=CardType.CREATURE,
        subtypes={Subtype.VAMPIRE, Subtype.SOLDIER},
        cost=ManaCost.parse("1W"),
        colors=Color.W,
        power=3, toughness=1,  # FIX: card is 3/1, not 1/1
        text="Pay 4 life: indestructible UEOT. 3/1.",
    )
    cdef.activated.append(ActivatedAbility(cost_fn=lambda g, p: True, effect=_adanto_van_indestructible,
                                           description="4 life: indestructible UEOT"))
    return cdef


# ---------- Demystify (SB) ----------
def _demystify_resolve(game, controller, card, targets, x):
    if not targets:
        return
    t = targets[0]
    if hasattr(t, "cdef") and t.cdef.is_enchantment():
        game._move_to_graveyard(t, "destroyed by Demystify")


def demystify():
    return CardDef(
        name="Demystify",
        types=CardType.INSTANT,
        cost=ManaCost.parse("W"),
        colors=Color.W,
        text="Destroy target enchantment.",
        on_resolve=_demystify_resolve,
        needs_targets=1,
        target_filter=lambda g, ctrl, cand: (hasattr(cand, "cdef") and cand.cdef.is_enchantment()
                                             and cand.controller_idx != ctrl.idx),
    )


def white_deck() -> list:
    return [
        (dauntless_bodyguard(), 4),
        (skymarcher_aspirant(), 4),
        (snubhorn_sentry(), 4),
        (healers_hawk(), 1),
        (tithe_taker(), 4),
        (benalish_marshal(), 4),
        (venerated_loxodon(), 4),
        (legions_landing(), 4),
        (history_of_benalia(), 4),
        (conclave_tribunal(), 4),
        (unbreakable_formation(), 3),
        (plains(), 20),
    ]


def white_sideboard() -> list:
    return [
        (tocatli_honor_guard(), 4),
        (baffling_end(), 4),
        (ajani_adversary_of_tyrants(), 3),
        (adanto_vanguard(), 2),
        (demystify(), 2),
    ]
