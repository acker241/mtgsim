"""Deck-building helpers: instantiate Card objects from (CardDef, count) lists."""
from __future__ import annotations
from typing import List, Tuple
from ..engine.card import Card, CardDef
from ..engine.enums import Zone


def build_library(defs: List[Tuple[CardDef, int]], owner_idx: int, start_cid: int) -> Tuple[List[Card], int]:
    cards: List[Card] = []
    cid = start_cid
    for cdef, count in defs:
        for _ in range(count):
            c = Card(cid=cid, card_def=cdef, owner_idx=owner_idx, controller_idx=owner_idx, zone=Zone.LIBRARY)
            cards.append(c)
            cid += 1
    return cards, cid


def build_sideboard(defs: List[Tuple[CardDef, int]], owner_idx: int, start_cid: int) -> Tuple[List[Card], int]:
    """Sideboard cards live in a separate list; not in any game zone until swapped in."""
    cards: List[Card] = []
    cid = start_cid
    for cdef, count in defs:
        for _ in range(count):
            c = Card(cid=cid, card_def=cdef, owner_idx=owner_idx, controller_idx=owner_idx, zone=Zone.EXILE)
            cards.append(c)
            cid += 1
    return cards, cid
