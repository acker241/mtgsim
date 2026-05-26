"""Card definitions (static templates) and Card instances (per-game state)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Dict, Any, TYPE_CHECKING
from .enums import CardType, Subtype, Keyword, Zone, Color, TriggerEvent
from .mana import ManaCost

if TYPE_CHECKING:
    from .game import GameState
    from .player import Player


@dataclass
class CardDef:
    """Immutable card template. One instance per unique card name.
    Singleton for deepcopy: never duplicated when GameState is cloned (callables would break)."""
    name: str
    types: CardType
    subtypes: Set[Subtype] = field(default_factory=set)
    legendary: bool = False  # for rule 704.5k (legend rule)

    def __deepcopy__(self, memo):
        # treat as singleton — never duplicated
        return self
    cost: Optional[ManaCost] = None
    colors: Color = Color.NONE
    power: Optional[int] = None
    toughness: Optional[int] = None
    keywords: Set[Keyword] = field(default_factory=set)
    text: str = ""
    # behaviour hooks
    on_resolve: Optional[Callable] = None       # for spells, signature (game, controller, card, targets, x)
    on_etb: Optional[Callable] = None           # signature (game, controller, perm)
    on_ltb: Optional[Callable] = None
    on_dies: Optional[Callable] = None
    triggers: List["TriggeredAbility"] = field(default_factory=list)
    activated: List["ActivatedAbility"] = field(default_factory=list)
    static_mods: List["StaticEffect"] = field(default_factory=list)
    # spell metadata
    needs_targets: int = 0                      # number of targets required
    target_filter: Optional[Callable] = None    # (game, controller, candidate) -> bool. candidate is Player or Permanent
    # alt costs / kickers
    spectacle_cost: Optional[ManaCost] = None   # mono-red set spectacle
    has_x: bool = False
    # planeswalker
    starting_loyalty: Optional[int] = None
    # saga
    chapters: Optional[List[Callable]] = None   # list of fns (game, controller, perm) — chapter I..N

    def is_creature(self) -> bool:
        return bool(self.types & CardType.CREATURE)

    def is_land(self) -> bool:
        return bool(self.types & CardType.LAND)

    def is_instant(self) -> bool:
        return bool(self.types & CardType.INSTANT)

    def is_sorcery(self) -> bool:
        return bool(self.types & CardType.SORCERY)

    def is_artifact(self) -> bool:
        return bool(self.types & CardType.ARTIFACT)

    def is_enchantment(self) -> bool:
        return bool(self.types & CardType.ENCHANTMENT)

    def is_planeswalker(self) -> bool:
        return bool(self.types & CardType.PLANESWALKER)

    def is_historic(self) -> bool:
        # historic = artifact, legendary, or saga
        return self.is_artifact() or Subtype.SAGA in self.subtypes or "legendary" in self.text.lower()

    def is_permanent_type(self) -> bool:
        return bool(self.types & (CardType.CREATURE | CardType.LAND | CardType.ARTIFACT
                                  | CardType.ENCHANTMENT | CardType.PLANESWALKER))


@dataclass
class TriggeredAbility:
    event: TriggerEvent
    # condition(game, source_perm, event_data) -> bool
    condition: Optional[Callable] = None
    # effect(game, source_perm, event_data)
    effect: Callable = None
    description: str = ""


@dataclass
class ActivatedAbility:
    cost_fn: Callable  # (game, perm) -> bool tries to pay, returns True if paid
    effect: Callable   # (game, controller, perm, targets)
    is_mana: bool = False
    needs_targets: int = 0
    target_filter: Optional[Callable] = None
    description: str = ""


@dataclass
class StaticEffect:
    """Continuous effect, e.g. Benalish Marshal: other creatures you control +1/+1."""
    apply: Callable  # (game, source_perm, target_perm) -> (dp, dt, extra_keywords)
    description: str = ""


@dataclass
class Card:
    """Per-game instance of a card. Tracks zone, ownership, control, state."""
    cid: int                       # unique instance id
    card_def: CardDef
    owner_idx: int
    controller_idx: int
    zone: Zone = Zone.LIBRARY
    tapped: bool = False
    summoning_sick: bool = False
    damage_marked: int = 0
    counters: Dict[str, int] = field(default_factory=dict)
    attached_to: Optional[int] = None       # for auras/equipment
    is_token: bool = False
    flipped: bool = False                   # for transform/flip
    # saga
    chapter: int = 0
    # spell-on-stack data
    targets: List[Any] = field(default_factory=list)
    x_value: int = 0
    cost_paid_alt: Optional[str] = None     # 'spectacle' etc
    # tracking
    attacking: bool = False
    blocking: List[int] = field(default_factory=list)
    blocked_by: List[int] = field(default_factory=list)
    came_in_this_turn: bool = False
    dealt_first_strike: bool = False

    @property
    def name(self) -> str:
        return self.card_def.name

    @property
    def cdef(self) -> CardDef:
        return self.card_def

    def power(self, game: "GameState") -> int:
        base = self.cdef.power if self.cdef.power is not None else 0
        base += self.counters.get("+1/+1", 0)
        base -= self.counters.get("-1/-1", 0)
        base += self.counters.get("+2/+1_eot_p", 0)
        for src in game.battlefield():
            for st in src.cdef.static_mods:
                dp, _dt, _ = st.apply(game, src, self)
                base += dp
        return base

    def toughness(self, game: "GameState") -> int:
        base = self.cdef.toughness if self.cdef.toughness is not None else 0
        base += self.counters.get("+1/+1", 0)
        base -= self.counters.get("-1/-1", 0)
        base += self.counters.get("+2/+1_eot_t", 0)
        for src in game.battlefield():
            for st in src.cdef.static_mods:
                _dp, dt, _ = st.apply(game, src, self)
                base += dt
        return base

    def keywords(self, game: "GameState") -> Set[Keyword]:
        kw = set(self.cdef.keywords)
        if self.counters.get("vigilance_eot", 0) > 0:
            kw.add(Keyword.VIGILANCE)
        if self.counters.get("indestructible_eot", 0) > 0:
            kw.add(Keyword.INDESTRUCTIBLE)
        for src in game.battlefield():
            for st in src.cdef.static_mods:
                _dp, _dt, extra = st.apply(game, src, self)
                if extra:
                    kw |= extra
        return kw

    def has_kw(self, kw: Keyword, game: "GameState") -> bool:
        return kw in self.keywords(game)

    def can_attack(self, game: "GameState") -> bool:
        if not self.cdef.is_creature():
            return False
        if self.tapped:
            return False
        if self.summoning_sick and not self.has_kw(Keyword.HASTE, game):
            return False
        if self.has_kw(Keyword.DEFENDER, game):
            # Snubhorn Sentry-style ascend bypass
            if "[ascend_attack]" in self.cdef.text and game.players[self.controller_idx].city_blessing:
                return True
            return False
        return True

    def can_block(self, attacker: "Card", game: "GameState") -> bool:
        if not self.cdef.is_creature():
            return False
        if self.tapped:
            return False
        akw = attacker.keywords(game)
        if Keyword.FLYING in akw and Keyword.FLYING not in self.keywords(game) and Keyword.REACH not in self.keywords(game):
            return False
        return True

    def __repr__(self):
        z = self.zone.name[:3]
        t = "T" if self.tapped else ""
        return f"<{self.name}#{self.cid} {z}{t}>"
