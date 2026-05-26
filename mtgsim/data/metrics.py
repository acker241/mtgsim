"""Per-match metrics + issue detection. Light weight (~500B/match) added to recorder.

Tracks:
  - Card cast turns (Chainwhirler, Loxodon, Marshal, etc.) — curve quality
  - Spectacle: used vs wasted (cast at full cost when available cheap)
  - Wizard's Lightning: discounted vs full cost
  - Steam-Kin: max counters, RRR activations
  - Lethal-miss heuristic: at end of own turn, total burn in hand >= opp life and didn't kill
  - Mulligan distribution

Issue detector emits flags on suspected bad plays. Useful to target heuristic fixes.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from ..engine.enums import Subtype, Keyword
from ..engine.card import Card


# burn damage dict (same as ai)
_BURN_DMG = {
    "Shock": 2, "Lightning Strike": 3, "Wizard's Lightning": 3,
    "Skewer the Critics": 3, "Lava Coil": 4, "Fight with Fire": 5,
}


def _is_burn_card(name: str) -> bool:
    return name in _BURN_DMG


class MetricsCollector:
    """Subscribe to observer events; at game/match end, snapshot battlefield/hand info."""

    def __init__(self):
        # per-player metrics
        self.m: Dict[str, Any] = {
            # cast turns
            "chainwhirler_cast_turns_p0": [],
            "chainwhirler_cast_turns_p1": [],
            "loxodon_cast_turns_p0": [],
            "loxodon_cast_turns_p1": [],
            "marshal_cast_turns_p0": [],
            "marshal_cast_turns_p1": [],
            # spectacle
            "spectacle_uses_p0": 0, "spectacle_uses_p1": 0,
            "spectacle_full_cost_p0": 0, "spectacle_full_cost_p1": 0,
            # wizard's lightning
            "wiz_lightning_disc_p0": 0, "wiz_lightning_disc_p1": 0,
            "wiz_lightning_full_p0": 0, "wiz_lightning_full_p1": 0,
            # steam-kin
            "steamkin_max_counters_p0": 0, "steamkin_max_counters_p1": 0,
            "steamkin_activated_p0": 0, "steamkin_activated_p1": 0,
            # lethal miss
            "lethal_missed_p0": 0, "lethal_missed_p1": 0,
            "lethal_taken_p0": 0, "lethal_taken_p1": 0,
            # first creature on bf
            "first_creature_turn_p0": None, "first_creature_turn_p1": None,
            # mana flood
            "lands_in_play_t5_p0": None, "lands_in_play_t5_p1": None,
            # hand size at end
            "hand_at_end_p0": 0, "hand_at_end_p1": 0,
            # mulligans
            "mulls_p0": 0, "mulls_p1": 0,
            # turn count
            "final_turn": 0,
            # outcome
            "winner": None, "winner_idx": None, "draw": False,
            # deck names (for tagging)
            "p0_deck": "", "p1_deck": "",
            # detected issues
            "issues": [],
        }
        self._last_steamkin_counters = {0: 0, 1: 0}
        self._game = None

    def attach_game(self, game):
        self._game = game

    def on_event(self, ev):
        kind = ev.kind
        if kind == "cast":
            card = ev.data.get("card")
            active = ev.data.get("player_idx", ev.active_idx)
            spec = bool(ev.data.get("spec", False))
            turn = ev.turn
            if card == "Goblin Chainwhirler":
                self.m[f"chainwhirler_cast_turns_p{active}"].append(turn)
            elif card == "Venerated Loxodon":
                self.m[f"loxodon_cast_turns_p{active}"].append(turn)
            elif card == "Benalish Marshal":
                self.m[f"marshal_cast_turns_p{active}"].append(turn)
            if card in ("Skewer the Critics", "Light Up the Stage"):
                if spec:
                    self.m[f"spectacle_uses_p{active}"] += 1
                else:
                    self.m[f"spectacle_full_cost_p{active}"] += 1
            elif card == "Wizard's Lightning":
                if spec:
                    self.m[f"wiz_lightning_disc_p{active}"] += 1
                else:
                    self.m[f"wiz_lightning_full_p{active}"] += 1
        elif kind == "steamkin_activate":
            # use game.active_idx (only active player can activate during own main)
            active = ev.active_idx
            self.m[f"steamkin_activated_p{active}"] += 1
        elif kind == "game_end":
            self._snapshot_game_end(ev)

    def snapshot_end_of_turn(self, game):
        """Called by hook at end_step. Detect lethal-miss AND snapshot Steam-Kin counters."""
        ap_idx = game.active_idx
        opp = game.players[1 - ap_idx]
        # snapshot Steam-Kin counters at end of turn (most accurate timing)
        for pi in (0, 1):
            for c in game.players[pi].battlefield:
                if c.name == "Runaway Steam-Kin":
                    cnt = c.counters.get("+1/+1", 0)
                    if cnt > self.m[f"steamkin_max_counters_p{pi}"]:
                        self.m[f"steamkin_max_counters_p{pi}"] = cnt
        if opp.lost:
            return
        ap = game.players[ap_idx]
        total_burn = 0
        for c in ap.hand:
            if not (c.cdef.is_instant() or c.cdef.is_sorcery()):
                continue
            d = _BURN_DMG.get(c.name, 0)
            if d > 0:
                total_burn += d
        if total_burn >= opp.life and total_burn > 0:
            self.m[f"lethal_missed_p{ap_idx}"] += 1
            self.m["issues"].append({
                "turn": game.turn, "kind": "lethal_missed",
                "player": ap_idx, "burn_total": total_burn, "opp_life": opp.life,
            })

    def snapshot_pre_main(self, game):
        """At start of precombat main, capture Steam-Kin max counters."""
        for pi in (0, 1):
            for c in game.players[pi].battlefield:
                if c.name == "Runaway Steam-Kin":
                    cnt = c.counters.get("+1/+1", 0)
                    if cnt > self.m[f"steamkin_max_counters_p{pi}"]:
                        self.m[f"steamkin_max_counters_p{pi}"] = cnt
            # first creature on bf
            if self.m[f"first_creature_turn_p{pi}"] is None:
                for c in game.players[pi].battlefield:
                    if c.cdef.is_creature():
                        self.m[f"first_creature_turn_p{pi}"] = game.turn
                        break
            # lands at T5 snapshot
            if game.turn == 5 and self.m[f"lands_in_play_t5_p{pi}"] is None:
                self.m[f"lands_in_play_t5_p{pi}"] = sum(
                    1 for c in game.players[pi].battlefield if c.cdef.is_land()
                )

    def _snapshot_game_end(self, ev):
        if self._game is None:
            return
        g = self._game
        self.m["final_turn"] = g.turn
        self.m["winner_idx"] = g.winner_idx
        self.m["draw"] = g.draw_game
        if g.winner_idx is not None:
            self.m["winner"] = g.players[g.winner_idx].name
        for pi in (0, 1):
            self.m[f"hand_at_end_p{pi}"] = len(g.players[pi].hand)
            self.m[f"mulls_p{pi}"] = g.players[pi].mulligans_taken

    def finalize(self, deck0: str, deck1: str) -> dict:
        self.m["p0_deck"] = deck0
        self.m["p1_deck"] = deck1
        return dict(self.m)


def install_metrics_hooks(game, collector: MetricsCollector):
    """Wire collector to game: subscribe to observer; patch turn module to snapshot."""
    collector.attach_game(game)
    game.observer.subscribe(collector.on_event)
    # we also need turn-level snapshots; we use observer events with kind='turn_start' / 'end_step'
    # already emitted from log() but not as discrete events. So patch by attaching wrappers.
    # simpler: use the existing 'log' kind to detect end_step / precombat_main markers.
    # Engine emits step changes via game.step setter? It does game.step = X then log().
    # We'll piggyback on a tick by subscribing and inspecting game.step.

    last_step = {"v": ""}

    def _step_watcher(ev):
        if not game or game.is_over():
            return
        step = game.step.name if game.step else ""
        if step != last_step["v"]:
            last_step["v"] = step
            if step == "PRECOMBAT_MAIN":
                collector.snapshot_pre_main(game)
            elif step == "END_STEP":
                collector.snapshot_end_of_turn(game)

    game.observer.subscribe(_step_watcher)
