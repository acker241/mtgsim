"""Concrete deck specs."""
from .match import DeckSpec
from ..cards.red import red_deck, red_sideboard
from ..cards.white import white_deck, white_sideboard


def mono_red() -> DeckSpec:
    # vs aggro mirror: bring in removal (Lava Coil) + Phoenix; cut high-cmc Frenzy + slow Firebrand
    red_sb_plan = {
        "aggro": [
            ("Lava Coil", "Light Up the Stage", 2),
            ("Lava Coil", "Experimental Frenzy", 2),
            ("Rekindling Phoenix", "Fanatical Firebrand", 2),
        ],
    }
    return DeckSpec(name="Mono-Red", main=red_deck(), sideboard=red_sideboard(),
                    archetype="aggro", sb_plan=red_sb_plan)


def mono_white() -> DeckSpec:
    # vs aggro: bring in Tocatli (kills ETBs), Baffling End (removes 1-3 cmc threats); cut defenders + lifelink hawk
    white_sb_plan = {
        "aggro": [
            ("Tocatli Honor Guard", "Snubhorn Sentry", 4),
            ("Baffling End", "Tithe Taker", 3),
            ("Baffling End", "Healer's Hawk", 1),
        ],
    }
    return DeckSpec(name="Mono-White", main=white_deck(), sideboard=white_sideboard(),
                    archetype="aggro", sb_plan=white_sb_plan)
