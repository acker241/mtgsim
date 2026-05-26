"""GameState: top-level state object. Owns players, stack, turn info, RNG, log."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Tuple
import random
from .enums import Zone, Phase, Step, TriggerEvent, Keyword, CardType
from .card import Card
from .player import Player
from .observer import Observer, NULL_OBSERVER


@dataclass
class StackItem:
    """Spell or ability on the stack."""
    source: Card                       # the spell card (for spells) or the source perm (for abilities)
    controller_idx: int
    is_spell: bool
    is_ability: bool = False
    targets: List[Any] = field(default_factory=list)
    x: int = 0
    resolve: Optional[Callable] = None  # (game) -> None
    description: str = ""


@dataclass
class GameState:
    players: List[Player]
    rng: random.Random
    turn: int = 1
    active_idx: int = 0
    phase: Phase = Phase.BEGIN
    step: Step = Step.UNTAP
    stack: List[StackItem] = field(default_factory=list)
    next_cid: int = 1
    log_enabled: bool = False
    log_lines: List[str] = field(default_factory=list)
    max_turns: int = 30
    winner_idx: Optional[int] = None
    draw_game: bool = False
    # combat state
    attackers: List[Card] = field(default_factory=list)  # creatures declared as attackers this combat
    # used for ai
    on_first_strike_substep: bool = False
    # observer for UI / logs
    observer: Observer = field(default_factory=lambda: NULL_OBSERVER)
    game_id: int = 0
    # data recorder (optional)
    recorder: Optional[Any] = None
    # ai_step callback (set by match runner) — enables priority windows in resolve_all
    ai_step: Optional[Callable] = None

    def log(self, msg: str):
        if self.log_enabled:
            line = f"T{self.turn} P{self.active_idx} {self.step.name}: {msg}"
            self.log_lines.append(line)
        # emit even when log not enabled — observer may still want it
        self.observer.emit("log", self, {"msg": msg})

    def alloc_cid(self) -> int:
        c = self.next_cid
        self.next_cid += 1
        return c

    def active(self) -> Player:
        return self.players[self.active_idx]

    def nonactive(self) -> Player:
        return self.players[1 - self.active_idx]

    def opponent_of(self, idx: int) -> Player:
        return self.players[1 - idx]

    def battlefield(self) -> List[Card]:
        out = []
        for p in self.players:
            out.extend(p.battlefield)
        return out

    def all_creatures(self) -> List[Card]:
        return [c for c in self.battlefield() if c.cdef.is_creature()]

    def creatures_of(self, idx: int) -> List[Card]:
        return [c for c in self.players[idx].battlefield if c.cdef.is_creature()]

    def is_over(self) -> bool:
        if self.winner_idx is not None or self.draw_game:
            return True
        alive = [p for p in self.players if not p.lost]
        if len(alive) == 0:
            self.draw_game = True
            return True
        if len(alive) == 1:
            self.winner_idx = alive[0].idx
            return True
        if self.turn > self.max_turns:
            self.draw_game = True
            return True
        return False

    def check_state_based(self):
        """SBAs: lethal damage, 0 toughness, players at 0 life, lib-out flag."""
        for p in self.players:
            if p.life <= 0:
                p.lost = True
        # creatures with lethal damage or 0 toughness die
        dying: List[Card] = []
        for c in list(self.battlefield()):
            if c.cdef.is_creature():
                t = c.toughness(self)
                indest = (Keyword.INDESTRUCTIBLE in c.keywords(self)
                          or c.counters.get("indestructible_eot", 0) > 0)
                if t <= 0 and not indest:
                    dying.append(c)
                elif c.damage_marked >= t and not indest:
                    dying.append(c)
            if c.cdef.is_planeswalker():
                if c.counters.get("loyalty", 0) <= 0:
                    dying.append(c)
        # legend rule (704.5k): if a player controls 2+ legendary permanents with same name,
        # they choose one and the rest go to their owners' graveyards.
        for p in self.players:
            by_name: dict = {}
            for c in p.battlefield:
                if c.cdef.legendary:
                    by_name.setdefault(c.name, []).append(c)
            for name, lst in by_name.items():
                if len(lst) > 1:
                    # keep first (arbitrary AI choice); rest die
                    for c in lst[1:]:
                        if c not in dying:
                            dying.append(c)
        for c in dying:
            self._move_to_graveyard(c, reason="dies")
        # ascend check
        for p in self.players:
            p.check_ascend()

    def _move_to_graveyard(self, c: Card, reason: str = ""):
        owner = self.players[c.controller_idx]
        if c in owner.battlefield:
            owner.battlefield.remove(c)
        # tokens cease to exist instead of going to graveyard
        if c.is_token:
            c.zone = Zone.EXILE
            self.log(f"{c.name} token ceases to exist ({reason})")
            return
        ow = self.players[c.owner_idx]
        c.zone = Zone.GRAVEYARD
        c.tapped = False
        c.damage_marked = 0
        c.counters.clear()
        c.attacking = False
        c.blocking = []
        c.blocked_by = []
        c.chapter = 0
        ow.graveyard.append(c)
        self.log(f"{c.name} -> graveyard ({reason})")
        # on_ltb / on_dies
        if c.cdef.on_ltb:
            c.cdef.on_ltb(self, self.players[c.controller_idx], c)
        if c.cdef.on_dies:
            c.cdef.on_dies(self, self.players[c.controller_idx], c)
        # fire LTB / DIES triggers from other permanents
        self.fire_event(TriggerEvent.DIES, {"card": c})

    def move_to_battlefield(self, c: Card, controller_idx: int, tapped: bool = False, with_counters: dict = None):
        # remove from any prior zone
        self._remove_from_current_zone(c)
        c.controller_idx = controller_idx
        c.zone = Zone.BATTLEFIELD
        c.tapped = tapped
        c.summoning_sick = c.cdef.is_creature()
        c.came_in_this_turn = True
        c.damage_marked = 0
        c.attacking = False
        c.blocking = []
        c.blocked_by = []
        if with_counters:
            for k, v in with_counters.items():
                c.counters[k] = c.counters.get(k, 0) + v
        # planeswalker starting loyalty
        if c.cdef.is_planeswalker() and c.cdef.starting_loyalty:
            c.counters["loyalty"] = c.cdef.starting_loyalty
        self.players[controller_idx].battlefield.append(c)
        self.log(f"{c.name} enters BF (P{controller_idx})")
        # Tocatli Honor Guard suppresses creature ETB triggers
        tocatli_active = any(p.name == "Tocatli Honor Guard"
                             for pl in self.players for p in pl.battlefield)
        suppress_etb = tocatli_active and c.cdef.is_creature()
        if c.cdef.on_etb and not suppress_etb:
            c.cdef.on_etb(self, self.players[controller_idx], c)
        if not suppress_etb:
            self.fire_event(TriggerEvent.ETB, {"card": c})

    def exile_card(self, c: Card, source_desc: str = ""):
        self._remove_from_current_zone(c)
        c.zone = Zone.EXILE
        self.players[c.owner_idx].exile.append(c)
        self.log(f"{c.name} exiled ({source_desc})")

    def _remove_from_current_zone(self, c: Card):
        p_ctrl = self.players[c.controller_idx]
        p_own = self.players[c.owner_idx]
        for lst in (p_ctrl.battlefield, p_own.hand, p_own.library, p_own.graveyard, p_own.exile):
            if c in lst:
                lst.remove(c)
                return

    def fire_event(self, ev: TriggerEvent, data: dict):
        """Walk battlefield and stack of cards to find triggered abilities matching event; queue them on stack."""
        triggers_to_fire: List[Tuple[Card, Any]] = []
        for src in self.battlefield():
            for trig in src.cdef.triggers:
                if trig.event != ev:
                    continue
                if trig.condition and not trig.condition(self, src, data):
                    continue
                triggers_to_fire.append((src, trig))
        # push each as stack item (LIFO; AP triggers ordered before NAP per APNAP rule, simplified)
        for src, trig in triggers_to_fire:
            self._push_triggered(src, trig, data)

    def _push_triggered(self, src: Card, trig, data: dict):
        item = StackItem(
            source=src,
            controller_idx=src.controller_idx,
            is_spell=False,
            is_ability=True,
            resolve=lambda g, s=src, t=trig, d=data: t.effect(g, s, d),
            description=f"trigger:{src.name}/{trig.event.name}",
        )
        self.stack.append(item)

    def resolve_top(self):
        if not self.stack:
            return
        item = self.stack.pop()
        self.log(f"resolve {item.description or item.source.name}")
        if item.is_spell:
            spell = item.source
            spell.targets = item.targets
            spell.x_value = item.x
            # remove from stack zone
            ow = self.players[spell.owner_idx]
            # spell is in zone STACK on owner... we'll just be sure to move accordingly
            ctrl = self.players[item.controller_idx]
            if spell.cdef.is_permanent_type():
                # permanent enters BF
                self._remove_from_current_zone(spell)
                self.move_to_battlefield(spell, item.controller_idx)
            else:
                # non-permanent: resolve effect, then go to graveyard
                if spell.cdef.on_resolve:
                    spell.cdef.on_resolve(self, ctrl, spell, item.targets, item.x)
                # cleanup: move to graveyard
                self._remove_from_current_zone(spell)
                spell.zone = Zone.GRAVEYARD
                self.players[spell.owner_idx].graveyard.append(spell)
        else:
            if item.resolve:
                item.resolve(self)
        self.check_state_based()

    def resolve_all(self):
        """Process stack with APNAP priority windows per rule 116.4 / 117 / 608.
        If ai_step is set, each spell/ability on stack triggers a priority round
        where both players can respond. Only when both pass is the top resolved."""
        ai_step = self.ai_step
        if ai_step is None:
            while self.stack and not self.is_over():
                self.resolve_top()
                self.check_state_based()
            return
        # Priority-aware resolution
        while self.stack and not self.is_over():
            # SBA loop until stable
            self.check_state_based()
            # Priority round: APNAP order, all pass → resolve top
            passes = 0
            order = [self.active_idx, 1 - self.active_idx]
            i = 0
            iter_limit = 16  # safety cap against infinite loops
            while passes < 2 and iter_limit > 0:
                if self.is_over():
                    return
                pidx = order[i]
                try:
                    act = ai_step(self, "stack_response", player_idx=pidx)
                except Exception:
                    act = None
                if act:
                    passes = 0
                    i = 0
                else:
                    passes += 1
                    i = (i + 1) % 2
                iter_limit -= 1
            if self.stack:
                self.resolve_top()
                self.check_state_based()
