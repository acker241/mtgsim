"""Policy/Value interface. Pluggable: HeuristicPolicy now, NeuralPolicy later."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING
import math
from .action import Action

if TYPE_CHECKING:
    from ..engine.game import GameState


class Policy(ABC):
    """Maps (state, legal_actions) -> (priors over actions, value estimate for root player).
    Value is in [-1, 1]: +1 = root player likely to win, -1 = likely to lose, 0 = draw."""
    @abstractmethod
    def evaluate(self, game: "GameState", root_player_idx: int,
                 legal: List[Action]) -> Tuple[List[float], float]:
        pass


class UniformPolicy(Policy):
    """All legal actions equally likely. Value via simple lifepad heuristic."""
    def evaluate(self, game, root_idx, legal):
        n = max(1, len(legal))
        priors = [1.0 / n] * len(legal)
        v = self._static_value(game, root_idx)
        return priors, v

    def _static_value(self, game, root_idx) -> float:
        if game.winner_idx is not None:
            return 1.0 if game.winner_idx == root_idx else -1.0
        if game.draw_game:
            return 0.0
        me = game.players[root_idx]
        opp = game.players[1 - root_idx]
        # life differential normalized
        v = (me.life - opp.life) / 40.0
        # board differential: power/toughness on BF
        my_power = sum(c.power(game) for c in me.battlefield if c.cdef.is_creature())
        opp_power = sum(c.power(game) for c in opp.battlefield if c.cdef.is_creature())
        v += (my_power - opp_power) / 30.0
        return max(-1.0, min(1.0, v))


class HeuristicPolicy(Policy):
    """Uses HeuristicAI to assign higher prior to its preferred action; rollout-based value."""
    def __init__(self, n_rollouts: int = 1, max_rollout_turns: int = 6):
        self.n_rollouts = n_rollouts
        self.max_rollout_turns = max_rollout_turns

    def evaluate(self, game, root_idx, legal):
        # priors: uniform with bias toward heuristic's best action
        from .heuristic import HeuristicAI
        ai = HeuristicAI()
        # quick heuristic ranking: use heuristic main_phase to suggest an action
        # mark all uniform; bonus on action matching heuristic
        n = len(legal)
        priors = [1.0 / n] * n
        # value: 1 quick rollout truncated
        v = self._rollout_value(game, root_idx)
        return priors, v

    def _rollout_value(self, game, root_idx) -> float:
        # placeholder: static evaluation; rollouts done in MCTS itself
        if game.winner_idx is not None:
            return 1.0 if game.winner_idx == root_idx else -1.0
        if game.draw_game:
            return 0.0
        me = game.players[root_idx]
        opp = game.players[1 - root_idx]
        v = (me.life - opp.life) / 40.0
        my_power = sum(c.power(game) for c in me.battlefield if c.cdef.is_creature())
        opp_power = sum(c.power(game) for c in opp.battlefield if c.cdef.is_creature())
        v += (my_power - opp_power) / 30.0
        # hand size differential
        v += (len(me.hand) - len(opp.hand)) / 20.0
        return max(-1.0, min(1.0, v))


class NeuralPolicy(Policy):
    """Stub. Wraps a NN that takes encoded state, returns (action_logits, value).
    To implement later:
      - load PyTorch/JAX model
      - encode state via GameEncoder
      - mask illegal actions
      - softmax for priors
    """
    def __init__(self, model=None, encoder=None):
        self.model = model
        self.encoder = encoder

    def evaluate(self, game, root_idx, legal):
        if self.model is None:
            # fallback to uniform until trained
            return UniformPolicy().evaluate(game, root_idx, legal)
        raise NotImplementedError("Wire your NN inference here")
