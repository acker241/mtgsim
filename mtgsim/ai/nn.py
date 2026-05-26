"""PyTorch model + NeuralPolicy. Trained on recorded MCTS decisions.

PolicyValueNet:
  in: state vector (encoding.state_dim() dims)
  trunk: 2-3 hidden layers
  policy head: per-action logits (variable len; we operate on legal subset)
  value head: scalar in [-1, 1] (tanh)

Since legal action set is variable per state, we don't enumerate over a fixed action vocab.
Instead, the policy head outputs a HIDDEN action embedding, and we score each legal action by
a small action-encoder pipeline → dot product with state embedding. This keeps it tractable.

Simpler v1 used here: predict a SCALAR score per legal action using (state_emb + action_feat).
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    torch = None  # type: ignore

from .encoding import state_dim, action_dim, ACTION_KINDS, KIND_TO_IDX, NAME_TO_IDX, N_NAMES
from .policy import Policy
from .action import Action


def action_to_features(action: Action) -> List[float]:
    """Small fixed-len feature vector per action. Used by NN head."""
    vec = [0.0] * (len(ACTION_KINDS) + 3)
    kind_idx = KIND_TO_IDX.get(action.kind, 0)
    vec[kind_idx] = 1.0
    # extras
    vec[-3] = 1.0 if action.use_alt_cost else 0.0
    vec[-2] = float(len(action.target_cids)) / 4.0
    vec[-1] = float(len(action.convoke_cids)) / 4.0
    return vec


ACTION_FEAT_DIM = len(ACTION_KINDS) + 3


if TORCH_OK:

    class PolicyValueNet(nn.Module):
        """State encoder + per-action score head + scalar value head."""

        def __init__(self, state_in: int = None, hidden: int = 128):
            super().__init__()
            self.state_in = state_in or state_dim()
            self.hidden = hidden
            self.trunk = nn.Sequential(
                nn.Linear(self.state_in, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            # value head
            self.value_head = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Tanh(),
            )
            # action score head: takes (state_emb, action_feat) → scalar
            self.action_proj = nn.Linear(ACTION_FEAT_DIM, hidden)
            self.score_head = nn.Sequential(
                nn.Linear(hidden * 2, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )

        def encode_state(self, state_vec):
            return self.trunk(state_vec)

        def value(self, state_emb):
            return self.value_head(state_emb).squeeze(-1)

        def action_scores(self, state_emb, action_feats):
            """state_emb: [B, H]; action_feats: [B, K, F]; out: [B, K]"""
            B, K, F_ = action_feats.shape
            a_emb = self.action_proj(action_feats)         # [B, K, H]
            s_emb = state_emb.unsqueeze(1).expand(-1, K, -1)  # [B, K, H]
            combined = torch.cat([s_emb, a_emb], dim=-1)   # [B, K, 2H]
            scores = self.score_head(combined).squeeze(-1)  # [B, K]
            return scores

        def forward(self, state_vec, action_feats=None, action_mask=None):
            s_emb = self.encode_state(state_vec)
            v = self.value(s_emb)
            if action_feats is None:
                return None, v
            scores = self.action_scores(s_emb, action_feats)
            if action_mask is not None:
                scores = scores.masked_fill(~action_mask, -1e9)
            return scores, v


class NeuralPolicy(Policy):
    """Wraps a PolicyValueNet. Returns priors (softmax over action scores) + value."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        if not TORCH_OK:
            raise ImportError("PyTorch required; pip install torch")
        self.device = device
        self.model = PolicyValueNet().to(device)
        self.model.eval()
        if model_path:
            self.load(model_path)

    def load(self, path: str):
        sd = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(sd)
        self.model.eval()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def evaluate(self, game, root_idx, legal: List[Action]) -> Tuple[List[float], float]:
        from .encoding import state_to_vector
        state_vec = state_to_vector(game, root_idx)
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        feats = [action_to_features(a) for a in legal]
        if not feats:
            return [], 0.0
        af = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            scores, value = self.model(s, af)
        priors = F.softmax(scores[0], dim=-1).cpu().tolist()
        return priors, float(value.item())
