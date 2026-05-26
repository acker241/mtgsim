"""HumanController: AI-compatible callable that blocks on a queue waiting for user input.

UI sends action descriptors via the queue. Game thread calls HumanController like any
other AI; controller blocks until action arrives.
"""
from __future__ import annotations
import threading
import queue
from typing import Optional, Dict, Any, List
from ..engine.game import GameState
from ..engine.card import Card
from ..engine import actions as eng_actions
from ..engine import combat as combat_mod


class HumanController:
    def __init__(self, name: str = "Human", side: str = "red"):
        self.name = name
        self.side = side
        # action queue: UI pushes, game thread pops
        self.q: "queue.Queue[dict]" = queue.Queue()
        # request flag: game thread sets when waiting for input, UI reads
        self.waiting_for = None  # str like 'main', 'declare_attackers', etc.
        self.waiting_context: dict = {}
        self.lock = threading.Lock()

    def __deepcopy__(self, memo):
        return None

    # AI-callable interface
    # do_mulligans calls this directly on AI (not via __call__)
    def _mulligan(self, game: GameState, idx: int) -> bool:
        return self._request("mulligan", game, idx, {})

    def __call__(self, game: GameState, kind: str, **kwargs):
        idx = kwargs.get("player_idx", game.active_idx)
        if kind == "mulligan":
            return self._request("mulligan", game, idx, {})
        if kind == "main":
            return self._request("main", game, idx, {})
        if kind == "priority":
            # auto-pass priority for MVP (instant-speed responses skipped)
            return None
        if kind == "stack_response":
            # auto-pass stack response for MVP
            return None
        if kind == "declare_attackers":
            return self._request("declare_attackers", game, idx, {})
        if kind == "declare_blockers":
            return self._request("declare_blockers", game, idx, {})
        return None

    # ---- request / response plumbing ----
    def _request(self, kind: str, game: GameState, idx: int, ctx: dict):
        with self.lock:
            self.waiting_for = kind
            self.waiting_context = {"idx": idx, **ctx}
        # block until UI sends action
        action = self.q.get()
        with self.lock:
            self.waiting_for = None
            self.waiting_context = {}
        return self._apply_or_translate(kind, game, idx, action)

    def submit(self, action: dict):
        """Called by UI thread: push action dict to queue."""
        self.q.put(action)

    # ---- action translation: convert UI dicts to engine effects ----
    def _apply_or_translate(self, kind: str, game: GameState, idx: int, action: dict):
        pl = game.players[idx]
        a = action.get("action") if action else None

        if kind == "mulligan":
            return bool(action.get("mulligan", False))

        if kind == "main":
            if a == "pass" or a is None:
                return None  # signals end of main
            if a == "play_land":
                cid = action.get("card_cid")
                land = next((c for c in pl.hand if c.cid == cid and c.cdef.is_land()), None)
                if not land:
                    return None
                eng_actions.play_land(game, idx, land)
                return {"action": "play_land"}
            if a == "cast":
                cid = action.get("card_cid")
                card = next((c for c in pl.hand if c.cid == cid), None)
                if not card:
                    return None
                targets = []
                for ref in action.get("target_cids", []):
                    if ref == -2:
                        targets.append(game.players[1 - idx])
                    elif ref == -1:
                        targets.append(pl)
                    else:
                        t = self._find_card_anywhere(game, ref)
                        if t:
                            targets.append(t)
                convoke = []
                for ccid in action.get("convoke_cids", []):
                    c = next((x for x in pl.battlefield if x.cid == ccid), None)
                    if c:
                        convoke.append(c)
                use_alt = bool(action.get("use_alt_cost", False))
                x = int(action.get("x", 0))
                ok = eng_actions.cast_spell(game, idx, card, targets=targets, x=x,
                                            use_spectacle=use_alt,
                                            convoke_creatures=convoke or None)
                if ok and convoke:
                    card.ai_choice = convoke
                game.resolve_all()
                return {"action": "cast"}
            if a == "activate":
                cid = action.get("card_cid")
                perm = next((c for c in pl.battlefield if c.cid == cid), None)
                if not perm:
                    return None
                ab_idx = int(action.get("ability_idx", 0))
                if ab_idx >= len(perm.cdef.activated):
                    return None
                ab = perm.cdef.activated[ab_idx]
                if not ab.cost_fn(game, perm):
                    return None
                ab.effect(game, pl, perm, [])
                game.resolve_all()
                return {"action": "activate"}
            return None

        if kind == "declare_attackers":
            cids = action.get("attacker_cids", [])
            by_cid = {c.cid: c for c in pl.battlefield}
            attackers = [by_cid[c] for c in cids if c in by_cid and by_cid[c].can_attack(game)]
            return attackers

        if kind == "declare_blockers":
            return action.get("block_map", {})

        return None

    def _find_card_anywhere(self, game, cid):
        for p in game.players:
            for lst in (p.battlefield, p.hand, p.library, p.graveyard, p.exile):
                for c in lst:
                    if c.cid == cid:
                        return c
        return None
