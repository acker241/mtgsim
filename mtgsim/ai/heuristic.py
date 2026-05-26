"""Heuristic AI controller. Decides land drops, casts, attacks, blocks, priority responses."""
from __future__ import annotations
from typing import List, Dict, Optional, Any
import random
from ..engine.enums import CardType, Subtype, Keyword, Phase, Step, Zone, Color
from ..engine.mana import ManaCost, R, W, GENERIC
from ..engine.card import Card
from ..engine.player import Player
from ..engine.game import GameState
from ..engine import actions as eng_actions


class HeuristicAI:
    """Single AI instance per game. Both players in a match use independent copies."""

    def __init__(self, name: str = "AI", rng: Optional[random.Random] = None,
                 archetype: str = "midrange"):
        self.name = name
        self.rng = rng or random.Random()
        self.archetype = archetype  # 'aggro' | 'midrange' | 'control'
        self.is_aggro = (archetype == "aggro")
        self._priority_pass_for: set = set()

    # ---------------- Top-level dispatcher ----------------

    def __call__(self, game: GameState, kind: str, **kwargs):
        if kind == "main":
            return self._main_phase(game)
        if kind == "priority":
            idx = kwargs.get("player_idx", game.active_idx)
            return self._priority(game, idx)
        if kind == "declare_attackers":
            return self._declare_attackers(game)
        if kind == "declare_blockers":
            return self._declare_blockers(game)
        if kind == "mulligan":
            return self._mulligan(game, kwargs["player_idx"])
        return None

    # ---------------- Mulligan ----------------

    def _mulligan(self, game: GameState, idx: int) -> bool:
        """Return True to mulligan, False to keep. London mulligan. Stricter for aggro."""
        pl = game.players[idx]
        lands = sum(1 for c in pl.hand if c.cdef.is_land())
        mulls = pl.mulligans_taken
        if mulls >= 3:
            return False
        target_size = 7 - mulls
        if target_size <= 4:
            return False
        # aggro: needs 2-4 lands and at least 1 cheap (1-2 drop creature/spell)
        if self.is_aggro:
            if lands < 2 or lands > 4:
                return True
            cheap_creature = sum(1 for c in pl.hand
                                 if c.cdef.is_creature() and c.cdef.cost and c.cdef.cost.cmc() <= 2)
            if cheap_creature == 0 and target_size >= 6:
                return True
            return False
        # midrange/control: more flexible
        if lands < 2 or lands > 5:
            return True
        cheap = sum(1 for c in pl.hand if c.cdef.cost and c.cdef.cost.cmc() <= 2 and not c.cdef.is_land())
        if cheap == 0 and target_size == 7:
            return True
        return False

    # ---------------- Helpers ----------------

    def _has_wizard(self, game: GameState, idx: int) -> bool:
        for c in game.players[idx].battlefield:
            if c.cdef.is_creature() and (Subtype.WIZARD in c.cdef.subtypes):
                return True
        return False

    def _available_mana(self, game: GameState, idx: int, extra_creature_taps: int = 0) -> Dict[str, int]:
        """Estimated mana: pool + untapped lands by color."""
        pl = game.players[idx]
        pool = dict(pl.mana_pool.pool)
        for p in pl.battlefield:
            if p.cdef.is_land() and not p.tapped:
                if Subtype.MOUNTAIN in p.cdef.subtypes:
                    pool[R] = pool.get(R, 0) + 1
                elif Subtype.PLAINS in p.cdef.subtypes:
                    pool[W] = pool.get(W, 0) + 1
        return pool

    def _total_mana(self, pool: Dict[str, int]) -> int:
        return sum(pool.values())

    def _can_pay(self, pool: Dict[str, int], cost: ManaCost, x: int = 0) -> bool:
        p = dict(pool)
        for sym, n in cost.colored_required().items():
            if p.get(sym, 0) < n:
                return False
            p[sym] -= n
        gen = cost.symbols.get(GENERIC, 0) + x * cost.symbols.get("X", 0)
        return sum(p.values()) >= gen

    def _convoke_untapped_creatures(self, game: GameState, idx: int) -> List[Card]:
        return [c for c in game.players[idx].battlefield
                if c.cdef.is_creature() and not c.tapped and not c.summoning_sick]

    def _opp(self, game: GameState, idx: int) -> Player:
        return game.players[1 - idx]

    def _threats_on_opp(self, game: GameState, idx: int) -> List[Card]:
        opp = self._opp(game, idx)
        return [c for c in opp.battlefield if c.cdef.is_creature() or c.cdef.is_planeswalker()]

    def _best_burn_target(self, game: GameState, idx: int, dmg: int) -> Any:
        """Returns Player or Card to point burn at. Prefers lethal-on-creature; else face for clock; else face."""
        opp = self._opp(game, idx)
        # if can kill opponent, target face
        if dmg >= opp.life:
            return opp
        # if a creature is a major threat and lethal-able, kill it
        creatures = [c for c in opp.battlefield if c.cdef.is_creature()]
        creatures.sort(key=lambda c: -(c.power(game) + c.toughness(game)))
        for c in creatures:
            t = c.toughness(game) - c.damage_marked
            if t <= dmg and Keyword.INDESTRUCTIBLE not in c.keywords(game):
                # worth killing if power >=2 or has key abilities
                if c.power(game) >= 2 or Keyword.LIFELINK in c.keywords(game) or c.name in ("Benalish Marshal", "Tithe Taker"):
                    return c
        # else face for clock
        return opp

    # ---------------- Main phase ----------------

    def _main_phase(self, game: GameState) -> Optional[dict]:
        """Return one action dict to execute (engine continues main loop) or None to end main."""
        idx = game.active_idx
        pl = game.players[idx]

        # 1) play land if haven't yet
        if pl.lands_played_this_turn < 1:
            lands_in_hand = [c for c in pl.hand if c.cdef.is_land()]
            if lands_in_hand:
                land = lands_in_hand[0]
                eng_actions.play_land(game, idx, land)
                game.observer.emit("play_land", game, {"player_idx": idx, "card": land.name})
                return {"action": "play_land"}

        # 1.5) lethal-burn-in-main check: if we can kill opp by stacking all burn, do it now
        act = self._try_lethal_burn_main(game, idx)
        if act:
            return act

        # 1.6) clear-path burn: kill small blocker to let attackers through (aggro priority)
        if self.is_aggro:
            act = self._try_clear_blocker(game, idx)
            if act:
                return act

        # 1.7) main-phase removal burn: kill a key threat to clear board
        act = self._try_kill_threat_main(game, idx)
        if act:
            return act

        # 1.8) Steam-Kin activation: if 3 counters AND extra mana lets us cast more burn/spell
        act = self._try_steamkin_activation(game, idx)
        if act:
            return act

        # 2) consider casting sorceries / creatures / activated abilities
        action = self._try_cast_sorcery_or_creature(game, idx)
        if action:
            return action

        # 2.5) main-phase chip burn: Wizard's Lightning at face if discounted + spare mana
        act = self._try_chip_burn_main(game, idx)
        if act:
            return act

        # 3) experimental frenzy: play top of library if applicable
        if pl.ai_data.get("experimental_frenzy"):
            action = self._try_play_top(game, idx)
            if action:
                return action

        # 4) Light up the stage exiled cards may be cast in main too
        action = self._try_cast_from_exile(game, idx)
        if action:
            return action

        # 5) activated abilities (Adanto land, Treasure Map, etc)
        action = self._try_main_activated(game, idx)
        if action:
            return action

        return None

    def _try_cast_sorcery_or_creature(self, game: GameState, idx: int) -> Optional[dict]:
        pl = game.players[idx]
        pool = self._available_mana(game, idx)
        candidates = []
        in_main1 = (game.step.name == "PRECOMBAT_MAIN")
        # hold-mana check: if Chainwhirler in hand AND have ≥3 mountains untap AND Chainwhirler castable now, skip everything else this iteration
        chainwhirler_in_hand = any(c.name == "Goblin Chainwhirler" for c in pl.hand)
        mountains_untap = sum(1 for c in pl.battlefield
                              if c.cdef.is_land() and not c.tapped
                              and any(s.name == "MOUNTAIN" for s in c.cdef.subtypes))
        # if we can cast Chainwhirler RRR THIS turn, prioritize it: skip 1-drops that would tap mountains
        reserve_for_chainwhirler = (chainwhirler_in_hand and mountains_untap >= 3
                                    and pool.get("R", 0) >= 3)
        for c in pl.hand:
            if c.cdef.is_land():
                continue
            if c.cdef.is_instant():
                continue  # save instants for priority window
            cost = c.cdef.cost
            if not cost:
                continue
            # special handling: spectacle for sorceries (cheaper)
            use_spec = False
            cost_eff = cost
            if c.cdef.spectacle_cost is not None and c.cdef.is_sorcery() and pl.opp_lost_life_this_turn:
                use_spec = True
                cost_eff = c.cdef.spectacle_cost
            # skip spectacle sorceries at full cost in main1 — wait for combat damage to enable spectacle
            if (c.name in ("Skewer the Critics", "Light Up the Stage")
                    and not use_spec and in_main1):
                continue
            # hold mana: if we'd cast a non-Chainwhirler card that taps mountains needed for Chainwhirler this turn, skip
            if reserve_for_chainwhirler and c.name != "Goblin Chainwhirler":
                # if casting this would leave us unable to cast Chainwhirler (still need RRR)
                # — check post-cost mountains untap
                cost_r_needed = cost_eff.colored_required().get("R", 0) + cost_eff.symbols.get("GENERIC", 0)
                if mountains_untap - cost_r_needed < 3:
                    continue
            if c.cdef.has_x:
                # X spell: spend everything (Banefire)
                x = max(0, self._total_mana(pool) - cost.symbols.get(GENERIC, 0)
                        - sum(cost.colored_required().values()))
                # need at least 1 X for it to matter
                if x < 1:
                    continue
                if not self._can_pay(pool, cost, x=x):
                    continue
                candidates.append({"card": c, "use_spec": False, "x": x, "convoke": [], "cost_eff": cost})
                continue
            # convoke for Conclave Tribunal / Venerated Loxodon
            convoke = []
            if c.name in ("Conclave Tribunal", "Venerated Loxodon"):
                avail = self._convoke_untapped_creatures(game, idx)
                # use only enough to make cost payable
                need_cmc = cost_eff.cmc()
                have_mana = self._total_mana(pool)
                deficit = max(0, need_cmc - have_mana)
                convoke = avail[:deficit]
                # check colored req still met (each convoke creature contributes any color of itself or generic)
                if not self._can_pay_with_convoke(pool, cost_eff, convoke):
                    continue
            else:
                if not self._can_pay(pool, cost_eff):
                    continue
            candidates.append({"card": c, "use_spec": use_spec, "x": 0, "convoke": convoke, "cost_eff": cost_eff})
        if not candidates:
            return None
        # rank: prefer highest cmc that fits (curve out big), creatures over sorceries when both legal, except removal that hits big
        def rank(d):
            c = d["card"]
            cmc = d["cost_eff"].cmc()
            score = cmc * 10
            if c.cdef.is_creature():
                score += 5
            if c.name == "Goblin Chainwhirler":
                score += 50
            if c.name == "Benalish Marshal":
                score += 30
            if c.name == "History of Benalia":
                score += 20
            if c.name == "Conclave Tribunal":
                # only if there's a target
                if not self._tribunal_target(game, idx):
                    return -1000
                score += 25
            if c.name == "Venerated Loxodon":
                # only if convoke makes it free or cheap and BF has creatures to buff
                if not d["convoke"]:
                    score -= 20
                score += 25
            if c.name == "Skewer the Critics" and not d["use_spec"]:
                # heavily penalize full-cost spectacle sorcery in main1; cast in main2 after combat
                if game.step.name == "PRECOMBAT_MAIN":
                    score -= 80
                else:
                    score -= 5
            if c.name == "Light Up the Stage" and not d["use_spec"]:
                if game.step.name == "PRECOMBAT_MAIN":
                    score -= 80
                else:
                    score -= 5
            if c.name == "Experimental Frenzy":
                # only worth if our hand has dead cards (lands + many in hand)
                pl = game.players[idx]
                if len(pl.hand) > 3:
                    score += 30
            return score
        candidates.sort(key=lambda d: -rank(d))
        best = candidates[0]
        if rank(best) <= -500:
            return None
        # build targets if needed
        targets = []
        c = best["card"]
        if c.cdef.needs_targets > 0:
            tgt = self._pick_target_for(game, idx, c, dmg_estimate=self._spell_damage_estimate(c, best["x"]))
            if tgt is None:
                return None
            targets = [tgt]
        # execute
        ok = eng_actions.cast_spell(game, idx, c,
                                    targets=targets, x=best["x"],
                                    use_spectacle=best["use_spec"],
                                    convoke_creatures=best["convoke"] or None)
        if not ok:
            return None
        # record convoke list on card for Loxodon ETB
        if best["convoke"]:
            c.ai_choice = best["convoke"]
        game.resolve_all()
        return {"action": "cast", "card": c.name}

    def _can_pay_with_convoke(self, pool: Dict[str, int], cost: ManaCost, convoke: List[Card]) -> bool:
        p = dict(pool)
        # each convoke creature can pay any 1 of its color or 1 generic
        for c in convoke:
            paid = False
            from ..engine.enums import Color as Col
            for sym, flag in [(W, Col.W), (R, Col.R)]:
                if c.cdef.colors & flag and cost.colored_required().get(sym, 0) > p.get(sym, 0):
                    p[sym] = p.get(sym, 0) + 1
                    paid = True
                    break
            if not paid:
                # adds to generic pool
                p["C"] = p.get("C", 0) + 1
        return self._can_pay(p, cost)

    def _spell_damage_estimate(self, card: Card, x: int) -> int:
        n = card.name
        if n == "Shock":
            return 2
        if n in ("Lightning Strike", "Wizard's Lightning", "Skewer the Critics"):
            return 3
        if n == "Lava Coil":
            return 4
        if n == "Fight with Fire":
            return 5
        if n == "Banefire":
            return x
        return 0

    def _tribunal_target(self, game: GameState, idx: int) -> Optional[Card]:
        opp = self._opp(game, idx)
        cands = [c for c in opp.battlefield if not c.cdef.is_land()]
        if not cands:
            return None
        cands.sort(key=lambda c: -((c.power(game) if c.cdef.is_creature() else 0)
                                    + (c.cdef.cost.cmc() if c.cdef.cost else 0)))
        return cands[0]

    def _pick_target_for(self, game: GameState, idx: int, card: Card, dmg_estimate: int = 0) -> Optional[Any]:
        cdef = card.cdef
        opp = self._opp(game, idx)
        if cdef.target_filter is None:
            return opp
        # for burn: best_burn_target
        if dmg_estimate > 0:
            t = self._best_burn_target(game, idx, dmg_estimate)
            # validate filter
            if cdef.target_filter(game, game.players[idx], t):
                return t
            # fallback: face if allowed
            if cdef.target_filter(game, game.players[idx], opp):
                return opp
        # for Tribunal etc: pick first matching opp permanent
        if card.name == "Conclave Tribunal":
            return self._tribunal_target(game, idx)
        if card.name == "Baffling End":
            for c in opp.battlefield:
                if cdef.target_filter(game, game.players[idx], c):
                    return c
        if card.name == "Demystify":
            for c in opp.battlefield:
                if cdef.target_filter(game, game.players[idx], c):
                    return c
        # generic creature-target
        for c in opp.battlefield:
            if cdef.target_filter(game, game.players[idx], c):
                return c
        return None

    def _try_play_top(self, game: GameState, idx: int) -> Optional[dict]:
        pl = game.players[idx]
        if not pl.library:
            return None
        top = pl.library[0]
        # while Experimental Frenzy is out, AP may play top card; can't play from hand
        if top.cdef.is_land():
            if pl.lands_played_this_turn < 1:
                pl.library.pop(0)
                game.move_to_battlefield(top, idx)
                pl.lands_played_this_turn += 1
                game.observer.emit("frenzy_land", game, {"card": top.name})
                return {"action": "frenzy_land"}
            return None
        # try cast top card
        pool = self._available_mana(game, idx)
        cost = top.cdef.cost
        if not cost:
            return None
        if not self._can_pay(pool, cost):
            return None
        # cast it directly (no spectacle/convoke handling for simplicity)
        # move top to hand transiently? Rules: Experimental Frenzy lets you play it directly from top.
        # Simulate: temporarily move to hand, cast, restore if fails
        pl.library.pop(0)
        pl.hand.append(top)
        top.zone = Zone.HAND
        targets = []
        if top.cdef.needs_targets > 0:
            tgt = self._pick_target_for(game, idx, top, dmg_estimate=self._spell_damage_estimate(top, 0))
            if tgt is None:
                pl.hand.remove(top)
                pl.library.insert(0, top)
                top.zone = Zone.LIBRARY
                return None
            targets = [tgt]
        ok = eng_actions.cast_spell(game, idx, top, targets=targets)
        if not ok:
            pl.hand.remove(top)
            pl.library.insert(0, top)
            top.zone = Zone.LIBRARY
            return None
        game.observer.emit("frenzy_cast", game, {"card": top.name})
        game.resolve_all()
        return {"action": "frenzy_cast"}

    def _try_cast_from_exile(self, game: GameState, idx: int) -> Optional[dict]:
        pl = game.players[idx]
        pool = self._available_mana(game, idx)
        for c in list(pl.exile):
            if c.counters.get("light_up_stage_castable", 0) <= 0:
                continue
            if c.cdef.is_land():
                continue
            cost = c.cdef.cost
            if not cost:
                continue
            if not self._can_pay(pool, cost):
                continue
            # treat as casting from hand: move to hand temporarily
            pl.exile.remove(c)
            pl.hand.append(c)
            c.zone = Zone.HAND
            counters_save = dict(c.counters)
            c.counters.pop("light_up_stage_castable", None)
            targets = []
            if c.cdef.needs_targets > 0:
                tgt = self._pick_target_for(game, idx, c, dmg_estimate=self._spell_damage_estimate(c, 0))
                if tgt is None:
                    pl.hand.remove(c)
                    pl.exile.append(c)
                    c.zone = Zone.EXILE
                    c.counters = counters_save
                    continue
                targets = [tgt]
            ok = eng_actions.cast_spell(game, idx, c, targets=targets)
            if not ok:
                pl.hand.remove(c)
                pl.exile.append(c)
                c.zone = Zone.EXILE
                c.counters = counters_save
                continue
            game.observer.emit("exile_cast", game, {"card": c.name})
            game.resolve_all()
            return {"action": "exile_cast"}
        return None

    def _try_main_activated(self, game: GameState, idx: int) -> Optional[dict]:
        pl = game.players[idx]
        # Adanto, the First Fort — make vamp token at any main if have W mana to spare
        for c in pl.battlefield:
            if c.name == "Adanto, the First Fort" and not c.tapped:
                # check we have 1W in pool/lands (besides keeping mana for instants)
                pool = self._available_mana(game, idx)
                if pool.get(W, 0) >= 1 and self._total_mana(pool) >= 2:
                    # don't use if we still need to cast a spell this turn? Heuristic: only at end of main, run as last resort
                    # for simplicity always do it
                    if c.cdef.activated:
                        c.cdef.activated[0].effect(game, pl, c, [])
                        game.observer.emit("adanto_token", game, {})
                        return {"action": "adanto_token"}
        # Treasure Map: tap for mana (added as ramp; not very useful, skip if no other use)
        # Skip — not impactful in sim.
        # Runaway Steam-Kin: if has 3 counters and no spells left to cast this turn? rarely worth at end of turn — skip
        return None

    # ---------------- Priority (instant-speed responses) ----------------

    def _priority(self, game: GameState, idx: int) -> Optional[dict]:
        """Return action or None to pass."""
        pl = game.players[idx]
        if pl.lost or game.is_over():
            return None
        is_my_turn = (idx == game.active_idx)

        # combat tricks: between first-strike and normal damage, kill opposing creature to save our combatant
        if game.step == Step.FIRST_STRIKE_DAMAGE:
            act = self._try_save_combatant(game, idx)
            if act:
                return act

        burn_windows_nap = (
            Step.UPKEEP, Step.PRECOMBAT_MAIN, Step.BEGIN_COMBAT,
            Step.DECLARE_ATTACKERS, Step.DECLARE_BLOCKERS,
            Step.FIRST_STRIKE_DAMAGE,
            Step.POSTCOMBAT_MAIN, Step.END_STEP,
        )

        # NAP windows: cast burn / removal at opp's turn
        if not is_my_turn and game.step in burn_windows_nap:
            # 1) lethal burn if total available burn kills opp
            act = self._try_cast_burn(game, idx, target_face_priority=True)
            if act:
                return act
            # 2) kill key threat before it attacks/triggers (BEGIN_COMBAT + DECLARE_ATTACKERS most valuable)
            if game.step in (Step.UPKEEP, Step.PRECOMBAT_MAIN, Step.BEGIN_COMBAT, Step.DECLARE_ATTACKERS):
                act = self._try_kill_threat_main(game, idx)
                if act:
                    return act
            # 3) end-step dump: chip burn at face when end of opp turn (use mana before untap)
            if game.step == Step.END_STEP:
                act = self._try_chip_burn_face_priority(game, idx)
                if act:
                    return act

        # AP windows during combat: protect creatures
        if is_my_turn and game.step == Step.DECLARE_BLOCKERS:
            act = self._try_cast_unbreakable_formation(game, idx)
            if act:
                return act

        return None

    def _try_chip_burn_face_priority(self, game: GameState, idx: int) -> Optional[dict]:
        """End-of-opp-turn: dump any burn that we can pay at face. Don't waste mana before our untap."""
        pl = game.players[idx]
        opp = self._opp(game, idx)
        if opp.lost:
            return None
        pool = self._available_mana(game, idx)
        for c in pl.hand:
            if not c.cdef.is_instant():
                continue
            dmg = self._spell_damage_estimate(c, 0)
            if dmg <= 0:
                continue
            cost_eff = c.cdef.cost
            use_alt = False
            if c.name == "Wizard's Lightning" and self._has_wizard(game, idx):
                cost_eff = c.cdef.spectacle_cost
                use_alt = True
            if not self._can_pay(pool, cost_eff):
                continue
            ok = eng_actions.cast_spell(game, idx, c, targets=[opp], use_spectacle=use_alt)
            if ok:
                game.observer.emit("end_step_burn", game, {"card": c.name})
                game.resolve_all()
                return {"action": "end_step_burn"}
        return None

    def _try_steamkin_activation(self, game: GameState, idx: int) -> Optional[dict]:
        """Activate Steam-Kin (remove 3 counters -> RRR) ONLY when net positive.

        Trade-off:
          - Lose: 3 power AND 3 toughness on Steam-Kin (was X/X, becomes (X-3)/(X-3))
          - Gain: 3 R mana usable THIS PHASE ONLY (rule 106.4)

        Activate if any of:
          (a) The RRR unlocks a cast that wasn't payable before AND new cast deals >=3 damage
              (or kills key threat / closes lethal)
          (b) Steam-Kin is doomed this turn anyway (blocker too big, lethal incoming)
          (c) Combined with intent: still want to attack but mana is needed for combat trick
        """
        pl = game.players[idx]
        opp = self._opp(game, idx)
        # find Steam-Kin with >= 3 counters
        sk = None
        for c in pl.battlefield:
            if c.name == "Runaway Steam-Kin" and c.counters.get("+1/+1", 0) >= 3:
                sk = c
                break
        if sk is None:
            return None
        # available mana NOW (lands + pool)
        pool = self._available_mana(game, idx)
        # mana after activation: add 3 R
        pool_after = dict(pool)
        pool_after["R"] = pool_after.get("R", 0) + 3
        # find spells in hand castable AFTER but NOT castable now (unlocked by RRR)
        unlocked_value = 0
        unlocked_card = None
        for c in pl.hand:
            cost = c.cdef.cost
            if not cost:
                continue
            cost_eff = cost
            if c.cdef.spectacle_cost and c.cdef.is_sorcery() and pl.opp_lost_life_this_turn:
                cost_eff = c.cdef.spectacle_cost
            elif c.name == "Wizard's Lightning" and self._has_wizard(game, idx):
                cost_eff = c.cdef.spectacle_cost
            now_ok = self._can_pay(pool, cost_eff)
            after_ok = self._can_pay(pool_after, cost_eff)
            if not now_ok and after_ok:
                v = self._spell_damage_estimate(c, 0)
                if c.cdef.is_creature() and c.cdef.power is not None:
                    v = max(v, c.cdef.power + c.cdef.toughness)
                if v > unlocked_value:
                    unlocked_value = v
                    unlocked_card = c
        # heuristic: activate if unlocked value >= 3 (worth at least the 3 power loss)
        # OR lethal (unlocked_value + already_payable_burn >= opp.life)
        # OR Steam-Kin is about to die (already damaged near lethal)
        sk_dying = sk.damage_marked > 0 and (sk.toughness(game) - sk.damage_marked) <= 0
        if unlocked_value >= 3 or sk_dying:
            # execute activation
            for act in sk.cdef.activated:
                if act.cost_fn(game, sk):
                    act.effect(game, pl, sk, [])
                    game.observer.emit("steamkin_activate", game,
                                       {"unlocked": unlocked_card.name if unlocked_card else None,
                                        "value": unlocked_value, "sk_dying": sk_dying})
                    pl.ai_data["steamkin_activated"] = pl.ai_data.get("steamkin_activated", 0) + 1
                    return {"action": "steamkin_activate"}
        return None

    def _try_chip_burn_main(self, game: GameState, idx: int) -> Optional[dict]:
        """Chip burn at face when surplus mana would otherwise float away.

        Runs in either MAIN phase. Wizard's Lightning at discount preferred.
        Skips if hand has cards we WANT to cast (creatures) and mana would leave them stranded.
        """
        if game.step.name not in ("PRECOMBAT_MAIN", "POSTCOMBAT_MAIN"):
            return None
        pl = game.players[idx]
        opp = self._opp(game, idx)
        if opp.lost:
            return None
        pool = self._available_mana(game, idx)
        # try Wizard's Lightning with discount first (best mana efficiency)
        if self._has_wizard(game, idx):
            for c in pl.hand:
                if c.name != "Wizard's Lightning":
                    continue
                if not self._can_pay(pool, c.cdef.spectacle_cost):
                    continue
                ok = eng_actions.cast_spell(game, idx, c, targets=[opp], use_spectacle=True)
                if ok:
                    game.observer.emit("chip_burn", game, {"card": c.name})
                    game.resolve_all()
                    return {"action": "chip_burn", "card": c.name}
        # then plain Shock / Lightning Strike if floating mana would waste
        for c in pl.hand:
            if not c.cdef.is_instant():
                continue
            dmg = self._spell_damage_estimate(c, 0)
            if dmg < 2:
                continue
            cost_eff = c.cdef.cost
            if c.name == "Wizard's Lightning":
                continue  # handled above
            if not self._can_pay(pool, cost_eff):
                continue
            ok = eng_actions.cast_spell(game, idx, c, targets=[opp])
            if ok:
                game.observer.emit("chip_burn", game, {"card": c.name})
                game.resolve_all()
                return {"action": "chip_burn", "card": c.name}
        return None

    def _try_lethal_burn_main(self, game: GameState, idx: int) -> Optional[dict]:
        """If total available burn (instants+sorceries) >= opp life, cast one burn at face."""
        pl = game.players[idx]
        opp = self._opp(game, idx)
        pool = self._available_mana(game, idx)
        burns = []
        for c in pl.hand:
            if not (c.cdef.is_instant() or c.cdef.is_sorcery()):
                continue
            dmg = self._spell_damage_estimate(c, 0)
            if dmg <= 0:
                continue
            cost_eff = c.cdef.cost
            use_alt = False
            if c.name == "Wizard's Lightning" and self._has_wizard(game, idx):
                cost_eff = c.cdef.spectacle_cost
                use_alt = True
            elif c.cdef.spectacle_cost and c.cdef.is_sorcery() and pl.opp_lost_life_this_turn:
                cost_eff = c.cdef.spectacle_cost
                use_alt = True
            burns.append((c, cost_eff, dmg, use_alt))
        if not burns:
            return None
        # check we can sequence them within current pool
        burns.sort(key=lambda t: t[1].cmc())
        total_payable = 0
        remaining_pool = dict(pool)
        for c, cost, dmg, alt in burns:
            if self._can_pay(remaining_pool, cost):
                # subtract from pool
                for sym, n in cost.colored_required().items():
                    remaining_pool[sym] = remaining_pool.get(sym, 0) - n
                gen = cost.symbols.get("GENERIC", 0)
                while gen > 0:
                    for sym in ("R", "W", "U", "B", "G", "C"):
                        if remaining_pool.get(sym, 0) > 0:
                            remaining_pool[sym] -= 1
                            gen -= 1
                            break
                    else:
                        break
                total_payable += dmg
        if total_payable < opp.life:
            return None
        # cast the cheapest first
        c, cost, dmg, alt = burns[0]
        ok = eng_actions.cast_spell(game, idx, c, targets=[opp], use_spectacle=alt)
        if ok:
            game.observer.emit("lethal_burn", game, {"card": c.name})
            game.resolve_all()
            return {"action": "lethal_burn", "card": c.name}
        return None

    def _try_save_combatant(self, game: GameState, idx: int) -> Optional[dict]:
        """In FIRST_STRIKE_DAMAGE step: if own creature in combat would die next step,
        cast burn to finish opposing creature first (uses opposing.damage_marked from FS)."""
        pl = game.players[idx]
        # find own creatures in combat
        my_in_combat = [c for c in pl.battlefield
                        if c.cdef.is_creature() and (c.attacking or c.blocking)]
        if not my_in_combat:
            return None
        by_cid = {c.cid: c for c in game.battlefield()}
        pool = self._available_mana(game, idx)
        for mine in my_in_combat:
            # find opposing creature(s) this is in combat with
            opposing = []
            if mine.attacking:
                opposing = [by_cid[bc] for bc in mine.blocked_by if bc in by_cid]
            elif mine.blocking:
                opposing = [by_cid[ac] for ac in mine.blocking if ac in by_cid]
            if not opposing:
                continue
            # would my creature die in normal damage step?
            incoming = sum(o.power(game) for o in opposing
                           if not o.has_kw(Keyword.FIRST_STRIKE, game)
                           and not o.has_kw(Keyword.DOUBLE_STRIKE, game))
            my_tough_left = mine.toughness(game) - mine.damage_marked
            if incoming < my_tough_left:
                continue  # already survives
            # try to burn an opposing creature to remove its damage output
            for opp_c in opposing:
                if Keyword.INDESTRUCTIBLE in opp_c.keywords(game):
                    continue
                tough_left = opp_c.toughness(game) - opp_c.damage_marked
                for c in pl.hand:
                    if not c.cdef.is_instant():
                        continue
                    dmg = self._spell_damage_estimate(c, 0)
                    if dmg < tough_left:
                        continue
                    cost_eff = c.cdef.cost
                    use_alt = False
                    if c.name == "Wizard's Lightning" and self._has_wizard(game, idx):
                        cost_eff = c.cdef.spectacle_cost
                        use_alt = True
                    if not self._can_pay(pool, cost_eff):
                        continue
                    if c.cdef.target_filter and not c.cdef.target_filter(game, pl, opp_c):
                        continue
                    ok = eng_actions.cast_spell(game, idx, c, targets=[opp_c],
                                                 use_spectacle=use_alt)
                    if ok:
                        game.observer.emit("save_combatant", game,
                                           {"card": c.name, "saved": mine.name,
                                            "killed": opp_c.name})
                        game.resolve_all()
                        return {"action": "save_combatant"}
        return None

    def _try_clear_blocker(self, game: GameState, idx: int) -> Optional[dict]:
        """Aggro: burn small blocker to free attackers. Only fires when attackers <= blockers
        AND burning the blocker enables >= burn_dmg in unblocked face damage."""
        pl = game.players[idx]
        opp = self._opp(game, idx)
        if opp.lost:
            return None
        # only fires during PRECOMBAT_MAIN before combat
        if game.step.name != "PRECOMBAT_MAIN":
            return None
        # count attackers I'll have (already on BF, can attack)
        my_attackers = [c for c in pl.battlefield if c.can_attack(game)]
        if not my_attackers:
            return None
        # count opp blockers (untapped creatures)
        opp_blockers = [c for c in opp.battlefield if c.cdef.is_creature() and not c.tapped]
        if not opp_blockers:
            return None  # already free path
        if len(my_attackers) > len(opp_blockers):
            return None  # have surplus already, no need
        # candidate blockers: small toughness, killable by burn in hand
        pool = self._available_mana(game, idx)
        # find burn cards we can cast NOW that kill cheapest blocker
        best = None  # (priority_score, card, target, use_alt)
        for blocker in sorted(opp_blockers, key=lambda b: b.toughness(game) - b.damage_marked):
            tough_left = blocker.toughness(game) - blocker.damage_marked
            if Keyword.INDESTRUCTIBLE in blocker.keywords(game):
                continue
            for c in pl.hand:
                if not (c.cdef.is_instant() or c.cdef.is_sorcery()):
                    continue
                dmg = self._spell_damage_estimate(c, 0)
                if dmg < tough_left:
                    continue
                cost_eff = c.cdef.cost
                use_alt = False
                if c.name == "Wizard's Lightning" and self._has_wizard(game, idx):
                    cost_eff = c.cdef.spectacle_cost
                    use_alt = True
                elif (c.cdef.spectacle_cost and c.cdef.is_sorcery()
                      and pl.opp_lost_life_this_turn):
                    cost_eff = c.cdef.spectacle_cost
                    use_alt = True
                if not self._can_pay(pool, cost_eff):
                    continue
                if c.cdef.target_filter and not c.cdef.target_filter(game, pl, blocker):
                    continue
                # estimate value: face damage we'd gain = power of attacker that was being blocked
                # heuristic: pick attacker with highest power that would have been blocked
                attackers_sorted = sorted(my_attackers, key=lambda a: -a.power(game))
                # attacker absorbed = min(blocker.power, attacker.power); face gained = attacker.power
                # use top attacker's power as rough gain estimate
                gain = attackers_sorted[0].power(game) if attackers_sorted else 0
                if gain < dmg:
                    # not worth: burning costs more damage than we gain
                    continue
                key = (-gain, cost_eff.cmc())
                if best is None or key < best[0]:
                    best = (key, c, blocker, use_alt)
            if best:
                break  # take first viable blocker (smallest toughness)
        if best is None:
            return None
        _, card, target, use_alt = best
        ok = eng_actions.cast_spell(game, idx, card, targets=[target], use_spectacle=use_alt)
        if ok:
            game.observer.emit("clear_blocker", game,
                               {"card": card.name, "target": target.name})
            game.resolve_all()
            return {"action": "clear_blocker", "card": card.name}
        return None

    def _try_kill_threat_main(self, game: GameState, idx: int) -> Optional[dict]:
        """Cast burn in main to kill a key threat creature."""
        pl = game.players[idx]
        opp = self._opp(game, idx)
        # ranked threats
        threats = [c for c in opp.battlefield if c.cdef.is_creature()]
        if not threats:
            return None
        def threat_score(c):
            s = c.power(game) * 2 + c.toughness(game)
            if c.name == "Benalish Marshal":
                s += 20
            if Keyword.LIFELINK in c.keywords(game):
                s += 10
            if Keyword.FLYING in c.keywords(game):
                s += 3
            if c.name == "Venerated Loxodon":
                s += 8
            if c.name == "Tithe Taker":
                s += 5
            return s
        threats.sort(key=threat_score, reverse=True)
        # aggro mode: only spend burn on creatures if HIGH-value threat or lifelink/anthem
        threshold = 12 if self.is_aggro else 6
        if threat_score(threats[0]) < threshold:
            return None
        pool = self._available_mana(game, idx)
        # find cheapest burn that can kill highest threat
        for target in threats[:2]:
            tough_left = target.toughness(game) - target.damage_marked
            if Keyword.INDESTRUCTIBLE in target.keywords(game):
                continue
            best = None
            for c in pl.hand:
                if not (c.cdef.is_instant() or c.cdef.is_sorcery()):
                    continue
                dmg = self._spell_damage_estimate(c, 0)
                if dmg <= 0 or dmg < tough_left:
                    continue
                cost_eff = c.cdef.cost
                use_alt = False
                if c.name == "Wizard's Lightning" and self._has_wizard(game, idx):
                    cost_eff = c.cdef.spectacle_cost
                    use_alt = True
                elif c.cdef.spectacle_cost and c.cdef.is_sorcery() and pl.opp_lost_life_this_turn:
                    cost_eff = c.cdef.spectacle_cost
                    use_alt = True
                if not self._can_pay(pool, cost_eff):
                    continue
                # prefer cheaper, prefer matching damage
                key = (cost_eff.cmc(), dmg - tough_left)
                if best is None or key < best[0]:
                    best = (key, c, cost_eff, use_alt, target)
            if best:
                _, c, _, alt, target = best
                # check filter
                if c.cdef.target_filter and not c.cdef.target_filter(game, pl, target):
                    continue
                ok = eng_actions.cast_spell(game, idx, c, targets=[target], use_spectacle=alt)
                if ok:
                    game.observer.emit("kill_threat", game, {"card": c.name, "target": target.name})
                    game.resolve_all()
                    return {"action": "kill_threat", "card": c.name}
        return None

    def _try_cast_burn(self, game: GameState, idx: int, target_face_priority: bool = False) -> Optional[dict]:
        pl = game.players[idx]
        opp = self._opp(game, idx)
        pool = self._available_mana(game, idx)
        burns = []
        for c in pl.hand:
            if not c.cdef.is_instant():
                continue
            dmg = self._spell_damage_estimate(c, 0)
            if dmg <= 0:
                continue
            cost_eff = c.cdef.cost
            # Wizard's Lightning discount
            if c.name == "Wizard's Lightning" and self._has_wizard(game, idx):
                cost_eff = c.cdef.spectacle_cost  # which we set to R
            if not self._can_pay(pool, cost_eff):
                continue
            burns.append((c, cost_eff, dmg))
        if not burns:
            return None
        # if lethal possible (sum dmg to face >= opp life), cast all
        total_face = sum(d for _, _, d in burns)
        if total_face >= opp.life and target_face_priority:
            # cast the cheapest first
            burns.sort(key=lambda t: t[1].cmc())
            c, cost_eff, dmg = burns[0]
            use_alt = (c.name == "Wizard's Lightning" and self._has_wizard(game, idx))
            ok = eng_actions.cast_spell(game, idx, c, targets=[opp], use_spectacle=use_alt)
            if ok:
                game.observer.emit("burn", game, {"player_idx": idx, "card": c.name, "target": "face"})
                game.resolve_all()
                return {"action": "burn", "card": c.name}
        return None

    def _try_cast_unbreakable_formation(self, game: GameState, idx: int) -> Optional[dict]:
        pl = game.players[idx]
        # find Unbreakable Formation in hand
        for c in pl.hand:
            if c.name != "Unbreakable Formation":
                continue
            pool = self._available_mana(game, idx)
            if not self._can_pay(pool, c.cdef.cost):
                continue
            # check if my attackers blocked and would die
            risky = 0
            for atk in game.attackers:
                if atk.controller_idx != idx:
                    continue
                blockers_power = sum(game._opp_block_power_estimate(atk) if hasattr(game, "_opp_block_power_estimate") else 0
                                     for _ in [None])
                # simpler: if blocked, formation saves it
                if atk.blocked_by:
                    risky += 1
            if risky == 0:
                return None
            ok = eng_actions.cast_spell(game, idx, c)
            if ok:
                game.observer.emit("formation", game, {"player_idx": idx})
                game.resolve_all()
                return {"action": "formation"}
        return None

    # ---------------- Combat: attackers ----------------

    def _declare_attackers(self, game: GameState) -> List[Card]:
        idx = game.active_idx
        pl = game.players[idx]
        opp = self._opp(game, idx)

        # race pressure: total power on my side that could threaten lethal soon
        my_power_total = sum(c.power(game) for c in pl.battlefield
                             if c.cdef.is_creature() and c.can_attack(game))
        opp_power_total = sum(c.power(game) for c in opp.battlefield
                              if c.cdef.is_creature() and not c.tapped)
        # racing if we'd kill opp faster than they kill us
        my_turns_to_lethal = (opp.life // max(1, my_power_total)) if my_power_total else 99
        opp_turns_to_lethal = (pl.life // max(1, opp_power_total)) if opp_power_total else 99
        racing = my_turns_to_lethal <= opp_turns_to_lethal
        # forced attack: very low life, must close out
        forced = opp.life <= 6 or pl.life <= 6

        atks: List[Card] = []
        for c in pl.battlefield:
            if not c.can_attack(game):
                continue
            p = c.power(game)
            if p <= 0:
                continue
            # blockers opp could use
            blockers = [b for b in opp.battlefield
                        if b.cdef.is_creature() and b.can_block(c, game)]
            if not blockers:
                # unblockable — always attack
                atks.append(c)
                continue
            # simulate opp's best block from opp's perspective
            # opp will pick block set with highest _block_score (positive = good for opp)
            best_opp_score = None
            best_block_set = None
            # candidate sets: each single legal blocker + each pair
            cand_sets = [[b] for b in blockers]
            if len(blockers) >= 2:
                for i in range(len(blockers)):
                    for j in range(i+1, len(blockers)):
                        cand_sets.append([blockers[i], blockers[j]])
            for cs in cand_sets:
                s = self._block_score(c, cs, game, lethal_pressure=False)
                if s is None:
                    continue
                if best_opp_score is None or s > best_opp_score:
                    best_opp_score = s
                    best_block_set = cs
            # if opp's best block is profitable for opp, attacker likely dies/trades for opp gain
            if best_opp_score is not None and best_opp_score > 0:
                # check if attacker still deals face damage (e.g., blocker too small)
                atk_dies, dying = self._simulate_block(c, best_block_set, game)
                # if blocker absorbs all damage and attacker dies, this attack only loses our creature
                if atk_dies and not forced and not racing:
                    continue
                # if attacker survives but takes damage that prevents it being useful next turn,
                # OK to attack only if racing or forced
                if atk_dies and (racing or forced):
                    # only ok if attacker is "expendable" (small) or we're closing out
                    # use a simple rule: attack if our power total kills opp within next turn
                    if my_turns_to_lethal <= 1:
                        atks.append(c)
                    elif p >= 2 and not forced:
                        # don't suicide medium creatures
                        continue
                    else:
                        atks.append(c)
                    continue
                # if attacker survives the block, attack (creature absorbs damage but lives)
                if not atk_dies:
                    atks.append(c)
            else:
                # opp won't profitably block; attacker likely deals face damage
                atks.append(c)
        return atks

    # ---------------- Combat: blockers ----------------

    def _simulate_block(self, atk: Card, blockers: List[Card], game: GameState):
        """Simulate combat (FS step + normal step) honoring first strike / double strike.
        Returns (attacker_dies, list_of_dying_blocker_cids, damage_to_attacker_player_if_unblocked)."""
        atk_kw = atk.keywords(game)
        atk_fs = Keyword.FIRST_STRIKE in atk_kw or Keyword.DOUBLE_STRIKE in atk_kw
        atk_ds = Keyword.DOUBLE_STRIKE in atk_kw
        atk_indestr = Keyword.INDESTRUCTIBLE in atk_kw
        atk_tough = atk.toughness(game)
        atk_dmg = atk.damage_marked
        atk_power = atk.power(game)

        blk_state = []
        for b in blockers:
            b_kw = b.keywords(game)
            blk_state.append({
                "c": b,
                "tough": b.toughness(game),
                "dmg": b.damage_marked,
                "power": b.power(game),
                "fs": Keyword.FIRST_STRIKE in b_kw or Keyword.DOUBLE_STRIKE in b_kw,
                "ds": Keyword.DOUBLE_STRIKE in b_kw,
                "indestr": Keyword.INDESTRUCTIBLE in b_kw,
                "alive": True,
            })
        # AP chooses damage assignment order: smallest toughness first (kills with lethal-first rule)
        blk_state.sort(key=lambda b: b["tough"] - b["dmg"])

        def assign_atk_damage():
            nonlocal blk_state
            remaining = atk_power
            for b in blk_state:
                if not b["alive"]:
                    continue
                if remaining <= 0:
                    break
                needed = max(0, b["tough"] - b["dmg"])
                # must assign lethal to first before going to next
                if needed > 0 and remaining >= needed:
                    b["dmg"] += needed
                    remaining -= needed
                else:
                    # not enough for lethal: dump all remaining here
                    b["dmg"] += remaining
                    remaining = 0

        def update_alive():
            for b in blk_state:
                if b["alive"] and not b["indestr"] and b["dmg"] >= b["tough"]:
                    b["alive"] = False

        # FIRST STRIKE step
        if atk_fs:
            assign_atk_damage()
        for b in blk_state:
            if b["alive"] and b["fs"]:
                atk_dmg += b["power"]
        # SBA after FS
        update_alive()
        atk_alive = atk_indestr or atk_dmg < atk_tough

        # NORMAL damage step
        if atk_alive and (not atk_fs or atk_ds):
            assign_atk_damage()
        for b in blk_state:
            if b["alive"] and (not b["fs"] or b["ds"]):
                atk_dmg += b["power"]
        update_alive()
        atk_dies = not atk_indestr and atk_dmg >= atk_tough

        dying = [b["c"].cid for b in blk_state if not b["alive"]]
        return atk_dies, dying

    def _block_score(self, atk: Card, blockers: List[Card], game: GameState,
                     lethal_pressure: bool) -> Optional[float]:
        """Returns score for a candidate block. Higher = better. None if not viable."""
        if not blockers:
            return None
        atk_dies, dying = self._simulate_block(atk, blockers, game)
        # value of attacker (gone if dies)
        atk_val = atk.power(game) + atk.toughness(game)
        # priority kill bonus
        if atk.name in ("Benalish Marshal", "Venerated Loxodon",
                        "Goblin Chainwhirler", "Rekindling Phoenix",
                        "Ajani, Adversary of Tyrants"):
            atk_val += 4
        # value of blockers lost
        blockers_lost = 0
        for b in blockers:
            if b.cid in dying:
                blockers_lost += b.power(game) + b.toughness(game)
        # face damage absorbed (if attacker doesn't die, it deals damage anyway,
        # but at least blockers absorbed some)
        # we credit absorbed = attacker.power if attacker has no trample
        absorbed = atk.power(game) if blockers else 0
        net = (atk_val if atk_dies else 0) - blockers_lost + (absorbed if lethal_pressure else 0)
        return net

    def _declare_blockers(self, game: GameState) -> Dict[int, List[int]]:
        idx = 1 - game.active_idx
        pl = game.players[idx]
        blocks: Dict[int, List[int]] = {}
        available = [b for b in pl.battlefield if b.cdef.is_creature() and not b.tapped]
        total_incoming = sum(a.power(game) for a in game.attackers)
        lethal = total_incoming >= pl.life

        for atk in sorted(game.attackers, key=lambda a: -a.power(game)):
            akw = atk.keywords(game)
            legal = [b for b in available if b.can_block(atk, game)]
            need_min = 2 if Keyword.MENACE in akw else 1
            if len(legal) < need_min:
                continue

            best_block = None
            best_score = None

            # candidate block sets:
            # - each single legal blocker (unless menace requires 2+)
            # - each pair of legal blockers
            # - chump single (smallest blocker) if lethal threat
            candidates: List[List[Card]] = []
            if Keyword.MENACE not in akw:
                for b in legal:
                    candidates.append([b])
            if len(legal) >= 2:
                for i in range(len(legal)):
                    for j in range(i + 1, len(legal)):
                        candidates.append([legal[i], legal[j]])

            for cset in candidates:
                score = self._block_score(atk, cset, game, lethal)
                if score is None:
                    continue
                # only commit block if net positive OR lethal threat
                if score < 0 and not lethal:
                    continue
                key = -score  # lower = better in sort
                if best_score is None or key < best_score:
                    best_score = key
                    best_block = cset

            # forced chump if lethal threat and no positive block found
            if not best_block and lethal:
                weakest = min(legal, key=lambda b: b.power(game) + b.toughness(game))
                if Keyword.MENACE in akw and len(legal) >= 2:
                    legal_sorted = sorted(legal, key=lambda b: b.power(game) + b.toughness(game))
                    best_block = legal_sorted[:2]
                else:
                    best_block = [weakest]

            if best_block:
                blocks[atk.cid] = [b.cid for b in best_block]
                for b in best_block:
                    if b in available:
                        available.remove(b)
        return blocks
