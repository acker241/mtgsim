"""Mana cost + pool. Costs as dict {symbol: count}. Symbols: W,U,B,R,G,C,X,GENERIC."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import re

# symbols
W, U, B, R, G, C = "W", "U", "B", "R", "G", "C"
GENERIC = "GENERIC"
X = "X"
COLOR_SYMS = (W, U, B, R, G)


@dataclass
class ManaCost:
    """e.g. {GENERIC:2, R:1} for 2R. X cost separate."""
    symbols: Dict[str, int] = field(default_factory=dict)
    x: int = 0  # X value at cast time

    @staticmethod
    def parse(s: str) -> "ManaCost":
        """'2R' -> {GENERIC:2, R:1}. '1WW' -> {GENERIC:1, W:2}. 'XR' -> {X:1, R:1}."""
        cost = ManaCost()
        if not s:
            return cost
        # tokens: digits = generic, letters = colored or X
        i = 0
        while i < len(s):
            c = s[i]
            if c.isdigit():
                # collect digits
                j = i
                while j < len(s) and s[j].isdigit():
                    j += 1
                cost.symbols[GENERIC] = cost.symbols.get(GENERIC, 0) + int(s[i:j])
                i = j
            elif c.upper() in COLOR_SYMS:
                sym = c.upper()
                cost.symbols[sym] = cost.symbols.get(sym, 0) + 1
                i += 1
            elif c.upper() == X:
                cost.symbols[X] = cost.symbols.get(X, 0) + 1
                i += 1
            else:
                i += 1
        return cost

    def cmc(self) -> int:
        n = 0
        for k, v in self.symbols.items():
            if k == X:
                n += self.x * v
            else:
                n += v
        return n

    def colored_required(self) -> Dict[str, int]:
        return {k: v for k, v in self.symbols.items() if k in COLOR_SYMS}

    def generic_required(self) -> int:
        g = self.symbols.get(GENERIC, 0)
        g += self.x * self.symbols.get(X, 0)
        return g

    def copy(self) -> "ManaCost":
        return ManaCost(symbols=dict(self.symbols), x=self.x)

    def __repr__(self):
        return f"ManaCost({self.symbols}, x={self.x})"


@dataclass
class ManaPool:
    pool: Dict[str, int] = field(default_factory=lambda: {W: 0, U: 0, B: 0, R: 0, G: 0, C: 0})

    def add(self, sym: str, n: int = 1):
        self.pool[sym] = self.pool.get(sym, 0) + n

    def total(self) -> int:
        return sum(self.pool.values())

    def can_pay(self, cost: ManaCost) -> bool:
        """Check if pool can pay cost, ignoring conditional/alt costs."""
        pool = dict(self.pool)
        # pay colored
        for sym, n in cost.colored_required().items():
            if pool.get(sym, 0) < n:
                return False
            pool[sym] -= n
        # pay generic from any
        gen = cost.generic_required()
        avail = sum(pool.values())
        return avail >= gen

    def pay(self, cost: ManaCost) -> bool:
        """Pay cost from pool. Generic paid from most-abundant non-colorless first, then colorless."""
        if not self.can_pay(cost):
            return False
        for sym, n in cost.colored_required().items():
            self.pool[sym] -= n
        gen = cost.generic_required()
        # prefer colorless first, then non-required colors arbitrarily
        order = [C, W, U, B, R, G]
        for sym in order:
            while gen > 0 and self.pool.get(sym, 0) > 0:
                self.pool[sym] -= 1
                gen -= 1
        return True

    def empty(self):
        for k in self.pool:
            self.pool[k] = 0

    def copy(self) -> "ManaPool":
        return ManaPool(pool=dict(self.pool))

    def __repr__(self):
        nonzero = {k: v for k, v in self.pool.items() if v}
        return f"ManaPool({nonzero})"
