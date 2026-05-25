"""Flat Monte Carlo with UCB1 selection, structured to upgrade to PUCT+NN later.

Algorithm (root-only, no tree):
  For each legal action a from root state s:
    repeat n_sims times:
      clone s
      apply a
      rollout via HeuristicAI until terminal or max_rollout_turns
      compute value from root player's POV
    avg_value[a] = mean of rollout values
  return argmax(avg_value[a])

UCB1 variant: instead of equal sims, select action by Q + c*sqrt(ln N / n_a) each iter.
PUCT-ready: priors come from Policy.evaluate(); easy switch to NN later.
"""
from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .action import Action, apply_action
from .policy import Policy, UniformPolicy
from ..engine.game import GameState
from ..engine import turn as turn_mod
from ..engine.observer import Observer, NULL_OBSERVER


@dataclass
class ActionStats:
    action: Action
    visits: int = 0
    total_value: float = 0.0
    prior: float = 1.0

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0


class FlatMCTS:
    """Root-only MCTS: UCB1 over actions, rollouts via heuristic, value from rollout end."""
    def __init__(self,
                 policy: Optional[Policy] = None,
                 n_sims: int = 64,
                 max_rollout_turns: int = 6,
                 c_ucb: float = 1.4,
                 c_puct: float = 1.0,
                 rng: Optional[random.Random] = None):
        self.policy = policy or UniformPolicy()
        self.n_sims = n_sims
        self.max_rollout_turns = max_rollout_turns
        self.c_ucb = c_ucb
        self.c_puct = c_puct
        self.rng = rng or random.Random()

    def search(self, game: GameState, root_idx: int, legal: List[Action]) -> Action:
        if not legal:
            return Action(kind="pass")
        if len(legal) == 1:
            self._record_if_any(game, root_idx, legal, [1], 0, 0.0)
            return legal[0]
        priors, _ = self.policy.evaluate(game, root_idx, legal)
        stats = [ActionStats(action=a, prior=p) for a, p in zip(legal, priors)]
        total_n = 0
        for _ in range(self.n_sims):
            best = max(stats, key=lambda s: s.q + self.c_puct * s.prior
                       * math.sqrt(total_n + 1) / (1 + s.visits))
            value = self._rollout_with(game, root_idx, best.action)
            best.visits += 1
            best.total_value += value
            total_n += 1
        # pick most-visited action; record decision tuple to recorder
        ordered = sorted(range(len(stats)), key=lambda i: (-stats[i].visits, -stats[i].q))
        chosen_idx = ordered[0]
        visits = [s.visits for s in stats]
        # root value estimate (avg over visits)
        root_val = sum(s.total_value for s in stats) / max(1, total_n)
        self._record_if_any(game, root_idx, legal, visits, chosen_idx, root_val)
        return stats[chosen_idx].action

    def _record_if_any(self, game, root_idx, legal, visits, chosen_idx, value):
        rec = getattr(game, "recorder", None)
        if rec is None:
            return
        try:
            from .encoding import state_to_vector
            state_vec = state_to_vector(game, root_idx)
            legal_desc = [self._action_to_dict(a) for a in legal]
            rec.record_decision(root_idx=root_idx, state_vec=state_vec,
                                legal_actions_desc=legal_desc,
                                visits=visits, chosen_idx=chosen_idx, value=value)
        except Exception:
            pass

    @staticmethod
    def _action_to_dict(a: Action) -> dict:
        return {
            "kind": a.kind,
            "card_cid": a.card_cid,
            "target_cids": list(a.target_cids),
            "convoke_cids": list(a.convoke_cids),
            "x": a.x,
            "use_alt_cost": a.use_alt_cost,
            "attacker_cids": list(a.attacker_cids),
            "block_map": {str(k): list(v) for k, v in a.block_map.items()},
            "activated_idx": a.activated_idx,
        }

    def _rollout_with(self, game: GameState, root_idx: int, action: Action) -> float:
        """Clone game, apply root action, then continue with heuristic AI until end / turn cap."""
        snapshot = self._clone(game)
        # apply action on snapshot
        try:
            apply_action(snapshot, snapshot.active_idx, action)
        except Exception:
            return self._value(snapshot, root_idx)
        # run heuristic until end
        from .heuristic import HeuristicAI
        ai = HeuristicAI(rng=self.rng)

        def step_fn(g, kind, **kwargs):
            return ai(g, kind, **kwargs)
        # finish current turn first (we already applied an action mid-main)
        # we'll just keep running steps: end current step's loop then continue turns
        start_turn = snapshot.turn
        try:
            # if we were in main, continue main loop
            if snapshot.step.name in ("PRECOMBAT_MAIN", "POSTCOMBAT_MAIN"):
                turn_mod.ai_main_loop(snapshot, step_fn)
                if not snapshot.is_over():
                    turn_mod.begin_combat(snapshot, step_fn)
                    if not snapshot.is_over():
                        turn_mod.declare_attackers_step(snapshot, step_fn)
                        turn_mod.declare_blockers_step(snapshot, step_fn)
                        turn_mod.first_strike_step(snapshot, step_fn)
                        turn_mod.combat_damage_step(snapshot, step_fn)
                        turn_mod.end_combat_step(snapshot, step_fn)
                        turn_mod.postcombat_main(snapshot, step_fn)
                        turn_mod.end_step(snapshot, step_fn)
                        turn_mod.cleanup_step(snapshot)
                        turn_mod.advance_turn(snapshot)
            elif snapshot.step.name == "DECLARE_ATTACKERS":
                # action was attackers; resume from declare_blockers
                turn_mod.declare_blockers_step(snapshot, step_fn)
                turn_mod.first_strike_step(snapshot, step_fn)
                turn_mod.combat_damage_step(snapshot, step_fn)
                turn_mod.end_combat_step(snapshot, step_fn)
                turn_mod.postcombat_main(snapshot, step_fn)
                turn_mod.end_step(snapshot, step_fn)
                turn_mod.cleanup_step(snapshot)
                turn_mod.advance_turn(snapshot)
            elif snapshot.step.name == "DECLARE_BLOCKERS":
                turn_mod.first_strike_step(snapshot, step_fn)
                turn_mod.combat_damage_step(snapshot, step_fn)
                turn_mod.end_combat_step(snapshot, step_fn)
                turn_mod.postcombat_main(snapshot, step_fn)
                turn_mod.end_step(snapshot, step_fn)
                turn_mod.cleanup_step(snapshot)
                turn_mod.advance_turn(snapshot)
            # now play remaining turns until cap or end
            while not snapshot.is_over() and snapshot.turn - start_turn < self.max_rollout_turns:
                turn_mod.take_turn(snapshot, step_fn)
                if snapshot.is_over():
                    break
                turn_mod.advance_turn(snapshot)
        except Exception:
            pass
        return self._value(snapshot, root_idx)

    def _value(self, game: GameState, root_idx: int) -> float:
        if game.winner_idx is not None:
            return 1.0 if game.winner_idx == root_idx else -1.0
        if game.draw_game:
            return 0.0
        # static eval if rollout truncated
        me = game.players[root_idx]
        opp = game.players[1 - root_idx]
        if me.lost:
            return -1.0
        if opp.lost:
            return 1.0
        v = (me.life - opp.life) / 40.0
        my_power = sum(c.power(game) for c in me.battlefield if c.cdef.is_creature())
        opp_power = sum(c.power(game) for c in opp.battlefield if c.cdef.is_creature())
        v += (my_power - opp_power) / 30.0
        v += (len(me.hand) - len(opp.hand)) / 25.0
        return max(-1.0, min(1.0, v))

    def _clone(self, game: GameState) -> GameState:
        # detach observer + recorder (both hold thread-bound state / locks)
        obs = game.observer
        rec = game.recorder
        game.observer = NULL_OBSERVER
        game.recorder = None
        try:
            new_game = copy.deepcopy(game)
        finally:
            game.observer = obs
            game.recorder = rec
        new_game.observer = NULL_OBSERVER
        new_game.recorder = None
        new_game.log_enabled = False
        return new_game
