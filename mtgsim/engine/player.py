from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
import random
from .enums import Zone
from .mana import ManaPool
from .card import Card

if TYPE_CHECKING:
    from .game import GameState


@dataclass
class Player:
    idx: int
    name: str
    life: int = 20
    library: List[Card] = field(default_factory=list)
    hand: List[Card] = field(default_factory=list)
    graveyard: List[Card] = field(default_factory=list)
    exile: List[Card] = field(default_factory=list)
    battlefield: List[Card] = field(default_factory=list)
    mana_pool: ManaPool = field(default_factory=ManaPool)
    lands_played_this_turn: int = 0
    poison: int = 0
    lost: bool = False
    # ascend / city's blessing
    city_blessing: bool = False
    # spectacle / opponent life-loss this turn
    opp_lost_life_this_turn: bool = False
    # tracking
    cards_drawn_this_turn: int = 0
    creatures_cast_this_turn: int = 0
    spells_cast_this_turn: int = 0
    attacked_with: int = 0  # creatures that attacked this turn (for Legion's Landing flip)
    mulligans_taken: int = 0
    # ai memo
    ai_data: dict = field(default_factory=dict)

    def shuffle(self, rng: random.Random):
        rng.shuffle(self.library)

    def draw(self, n: int = 1, game: Optional["GameState"] = None) -> List[Card]:
        drawn = []
        for _ in range(n):
            if not self.library:
                self.lost = True
                return drawn
            c = self.library.pop(0)
            c.zone = Zone.HAND
            self.hand.append(c)
            self.cards_drawn_this_turn += 1
            drawn.append(c)
        return drawn

    def lose_life(self, n: int, game: "GameState"):
        if n <= 0:
            return
        self.life -= n
        # mark opponents' tracker (mono-red spectacle gating uses their own — opp lost life)
        for p in game.players:
            if p.idx != self.idx:
                p.opp_lost_life_this_turn = True
        if self.life <= 0:
            self.lost = True

    def gain_life(self, n: int):
        if n <= 0:
            return
        self.life += n

    def permanents_count(self) -> int:
        return len(self.battlefield)

    def check_ascend(self):
        if not self.city_blessing and self.permanents_count() >= 10:
            self.city_blessing = True

    def reset_turn(self):
        self.lands_played_this_turn = 0
        self.cards_drawn_this_turn = 0
        self.creatures_cast_this_turn = 0
        self.spells_cast_this_turn = 0
        self.attacked_with = 0
        self.opp_lost_life_this_turn = False

    def __repr__(self):
        return f"<P{self.idx}:{self.name} L{self.life} H{len(self.hand)} BF{len(self.battlefield)} Lib{len(self.library)}>"
