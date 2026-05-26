"""Train PolicyValueNet on recorded MCTS decisions.

Usage:
  py -m mtgsim.scripts.train data/ --epochs 20 --batch 128 --out model.pt

Reads (state, legal_actions, visits, outcome_z) tuples from JSONL.
Loss: KL(softmax(logits) || visit_distribution) + MSE(value, outcome_z)
"""
from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..data.loader import iter_records
from ..ai.nn import PolicyValueNet, action_to_features, ACTION_FEAT_DIM
from ..ai.encoding import state_dim


class DecisionDataset(Dataset):
    def __init__(self, root: str, max_actions: int = 32):
        self.samples: List[dict] = []
        self.max_actions = max_actions
        for rec in iter_records(root, types=["decision"]):
            if rec.get("outcome_z") is None:
                continue
            if not rec.get("legal") or not rec.get("visits"):
                continue
            if len(rec["legal"]) > max_actions:
                continue
            self.samples.append(rec)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rec = self.samples[i]
        state = torch.tensor(rec["state"], dtype=torch.float32)
        K = len(rec["legal"])
        # build action feature tensor [max_actions, ACTION_FEAT_DIM]
        feats = torch.zeros(self.max_actions, ACTION_FEAT_DIM)
        for j, a in enumerate(rec["legal"]):
            from ..ai.action import Action
            act = Action(
                kind=a.get("kind", "pass"),
                card_cid=a.get("card_cid"),
                target_cids=list(a.get("target_cids") or []),
                convoke_cids=list(a.get("convoke_cids") or []),
                x=a.get("x", 0),
                use_alt_cost=a.get("use_alt_cost", False),
            )
            feats[j] = torch.tensor(action_to_features(act))
        mask = torch.zeros(self.max_actions, dtype=torch.bool)
        mask[:K] = True
        # visit distribution → policy target
        visits = list(rec["visits"]) + [0] * (self.max_actions - K)
        v = torch.tensor(visits, dtype=torch.float32)
        s = v.sum()
        if s > 0:
            policy_tgt = v / s
        else:
            policy_tgt = mask.float() / max(1, K)
        value_tgt = torch.tensor(rec["outcome_z"], dtype=torch.float32)
        return state, feats, mask, policy_tgt, value_tgt


def train(data_dir: str, out_path: str, epochs: int = 20, batch_size: int = 128,
          lr: float = 1e-3, val_frac: float = 0.1, device: str = "cpu"):
    print(f"Loading data from {data_dir}...")
    ds = DecisionDataset(data_dir)
    print(f"Loaded {len(ds)} decision samples")
    if len(ds) < 10:
        print("Not enough samples. Generate more with --record-mode decisions.")
        return
    # split
    rng = random.Random(0)
    n_val = int(len(ds) * val_frac)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    val_idx = set(idxs[:n_val])
    train_set = [ds[i] for i in idxs if i not in val_idx]
    val_set = [ds[i] for i in idxs if i in val_idx]
    # collate
    def collate(batch):
        s = torch.stack([b[0] for b in batch])
        f = torch.stack([b[1] for b in batch])
        m = torch.stack([b[2] for b in batch])
        p = torch.stack([b[3] for b in batch])
        v = torch.stack([b[4] for b in batch])
        return s, f, m, p, v
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = PolicyValueNet(state_in=state_dim()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Model: state_in={state_dim()}, params={sum(p.numel() for p in model.parameters())}")

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        train_pol_loss = 0.0
        train_val_loss = 0.0
        n_batches = 0
        for s, f, m, p_tgt, v_tgt in train_loader:
            s, f, m, p_tgt, v_tgt = s.to(device), f.to(device), m.to(device), p_tgt.to(device), v_tgt.to(device)
            scores, value = model(s, f, m)
            # policy loss: KL divergence over masked actions
            log_p = F.log_softmax(scores, dim=-1)
            pol_loss = -(p_tgt * log_p).sum(dim=-1).mean()
            val_loss = F.mse_loss(value, v_tgt)
            loss = pol_loss + val_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_pol_loss += pol_loss.item()
            train_val_loss += val_loss.item()
            n_batches += 1
        # validation
        model.eval()
        val_pol = 0.0
        val_val = 0.0
        nb = 0
        with torch.no_grad():
            for s, f, m, p_tgt, v_tgt in val_loader:
                s, f, m, p_tgt, v_tgt = s.to(device), f.to(device), m.to(device), p_tgt.to(device), v_tgt.to(device)
                scores, value = model(s, f, m)
                log_p = F.log_softmax(scores, dim=-1)
                val_pol += -(p_tgt * log_p).sum(dim=-1).mean().item()
                val_val += F.mse_loss(value, v_tgt).item()
                nb += 1
        dt = time.time() - t0
        print(f"Epoch {ep+1:>2}/{epochs} | train pol={train_pol_loss/max(1,n_batches):.3f} val={train_val_loss/max(1,n_batches):.3f}"
              f" | val pol={val_pol/max(1,nb):.3f} val={val_val/max(1,nb):.3f} | {dt:.1f}s")
    # save
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="models/policyvaluenet.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    train(args.data_dir, args.out, epochs=args.epochs, batch_size=args.batch,
          lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()
