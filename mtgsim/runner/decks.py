"""Concrete deck specs."""
from .match import DeckSpec
from ..cards.red import red_deck, red_sideboard
from ..cards.white import white_deck, white_sideboard


def mono_red() -> DeckSpec:
    return DeckSpec(name="Mono-Red", main=red_deck(), sideboard=red_sideboard())


def mono_white() -> DeckSpec:
    return DeckSpec(name="Mono-White", main=white_deck(), sideboard=white_sideboard())
