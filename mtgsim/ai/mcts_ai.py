"""Drop-in MCTS-driven AI controller. Uses MCTS for critical decisions, heuristic elsewhere."""
from __future__ import annotations
import random
from typing import List, Dict, Optional, Any
from ..engine.game import GameState
from .heuristic import HeuristicAI
from .action import (
    Action, legal_main_actions, legal_attacker_sets, legal_blocker_options,
    apply_action,
)
from .mcts import FlatMCTS
from .policy import Policy, UniformPolicy


class MctsAI:
    """Composes HeuristicAI (for fast decisions) with FlatMCTS (for critical ones)."""

    def __init__(self, name: str = "MctsAI",
                 rng: Optional[random.Random] = None,
                 n_sims: int = 32,
                 max_rollout_turns: int = 5,
                 mcts_for_main: bool = True,
                 mcts_for_attacks: bool = True,
                 mcts_for_blocks: bool = True,
                 policy: Optional[Policy] = None):
        self.name = name
        self.rng = rng or random.Random()
        self.heur = HeuristicAI(name=name, rng=self.rng)
        self.mcts = FlatMCTS(policy=policy or UniformPolicy(),
                             n_sims=n_sims,
                             max_rollout_turns=max_rollout_turns,
                             rng=self.rng)
        self.use_main = mcts_for_main
        self.use_attacks = mcts_for_attacks
        self.use_blocks = mcts_for_blocks

    def __call__(self, game: GameState, kind: str, **kwargs):
        idx = kwargs.get("player_idx", game.active_idx)
        if kind == "mulligan":
            return self.heur._mulligan(game, idx)
        if kind == "priority":
            return self.heur._priority(game, idx)
        if kind == "main":
            return self._main_with_mcts(game)
        if kind == "declare_attackers":
            return self._attackers_with_mcts(game)
        if kind == "declare_blockers":
            return self._blockers_with_mcts(game)
        return None

    def _main_with_mcts(self, game):
        if not self.use_main:
            return self.heur._main_phase(game)
        idx = game.active_idx
        legal = legal_main_actions(game, idx)
        # if only 'pass' is legal, end main
        non_pass = [a for a in legal if a.kind != "pass"]
        if not non_pass:
            return None
        # if obvious lethal: skip MCTS
        if self._obvious_lethal(game, idx):
            return self.heur._main_phase(game)
        chosen = self.mcts.search(game, idx, legal)
        if chosen.kind == "pass":
            return None
        ok = apply_action(game, idx, chosen)
        return {"action": chosen.kind} if ok else None

    def _attackers_with_mcts(self, game):
        if not self.use_attacks:
            return self.heur._declare_attackers(game)
        idx = game.active_idx
        legal = legal_attacker_sets(game, idx)
        if not legal:
            return []
        if len(legal) == 1:
            chosen = legal[0]
        else:
            chosen = self.mcts.search(game, idx, legal)
        # convert to list of Card objects
        by_cid = {c.cid: c for c in game.players[idx].battlefield}
        return [by_cid[cid] for cid in chosen.attacker_cids if cid in by_cid]

    def _blockers_with_mcts(self, game):
        if not self.use_blocks:
            return self.heur._declare_blockers(game)
        idx = 1 - game.active_idx
        legal = legal_blocker_options(game, idx)
        if len(legal) == 1:
            return legal[0].block_map
        chosen = self.mcts.search(game, idx, legal)
        return chosen.block_map

    def _obvious_lethal(self, game, idx) -> bool:
        opp = game.players[1 - idx]
        # if can dump enough burn to kill
        burns = 0
        for c in game.players[idx].hand:
            if c.cdef.is_instant() or c.cdef.is_sorcery():
                burns += {"Shock": 2, "Lightning Strike": 3, "Wizard's Lightning": 3,
                          "Skewer the Critics": 3, "Lava Coil": 4,
                          "Fight with Fire": 5}.get(c.name, 0)
        return burns >= opp.life

    # expose mulligan helper for the runner
    def _mulligan(self, game, idx):
        return self.heur._mulligan(game, idx)
