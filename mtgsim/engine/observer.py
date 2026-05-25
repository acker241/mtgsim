"""Observer hook: emits events for UI / logging. Engine calls Observer.emit() at key moments."""
from __future__ import annotations
from typing import Any, Callable, List, Optional
from dataclasses import dataclass, field
import time


@dataclass
class Event:
    kind: str
    data: dict
    turn: int
    active_idx: int
    step: str
    ts: float = field(default_factory=time.time)


class Observer:
    """Synchronous observer. Engine emits events; callbacks consume them.
    Used by UI to render state. Also used to pump pause/speed control."""
    def __init__(self):
        self.subs: List[Callable[[Event], Any]] = []
        self.tick: Optional[Callable[[], Any]] = None  # called per action — UI uses to pause/sleep

    def __deepcopy__(self, memo):
        # subs hold bound methods to UI/recorder objects (non-picklable threads/locks).
        # Clones (used by MCTS) never need to notify — return the null observer singleton.
        return NULL_OBSERVER

    def subscribe(self, cb: Callable[[Event], Any]):
        self.subs.append(cb)

    def emit(self, kind: str, game, data: Optional[dict] = None):
        ev = Event(
            kind=kind,
            data=data or {},
            turn=game.turn if game else 0,
            active_idx=game.active_idx if game else 0,
            step=(game.step.name if (game and game.step) else ""),
        )
        for s in list(self.subs):
            try:
                s(ev)
            except Exception:
                pass

    def wait_if_paused(self):
        if self.tick:
            self.tick()


# default global no-op observer
NULL_OBSERVER = Observer()
