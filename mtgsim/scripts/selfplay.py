"""Self-play loop: generate MCTS decisions → train NN → evaluate → promote.

Usage:
  py -m mtgsim.scripts.selfplay --iters 3 --games-per-iter 500 --workers 8
"""
from __future__ import annotations
import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list, **kw):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--games-per-iter", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-sims", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workspace", default="selfplay_runs")
    args = ap.parse_args()

    ws = Path(args.workspace)
    ws.mkdir(exist_ok=True)
    py = sys.executable

    for it in range(args.iters):
        print(f"\n===== ITER {it+1}/{args.iters} =====")
        iter_dir = ws / f"iter{it:02d}"
        data_dir = iter_dir / "data"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        # 1) generate self-play with MCTS (heuristic policy or last NN if exists)
        gen = [py, "-m", "mtgsim.runner.cli",
               "--matches", str(args.games_per_iter),
               "--workers", str(args.workers),
               "--ai", "mcts", "--n-sims", str(args.n_sims),
               "--record-to", str(data_dir),
               "--record-mode", "decisions",
               "--seed", str(args.seed + it * 1000)]
        t0 = time.time()
        run(gen, check=False)
        print(f"  generation: {time.time()-t0:.1f}s")
        # 2) train
        model_out = iter_dir / "model.pt"
        tr = [py, "-m", "mtgsim.scripts.train",
              str(data_dir), "--out", str(model_out),
              "--epochs", str(args.epochs)]
        run(tr, check=False)
        # 3) (TODO) evaluate new model vs prev: play 100 matches, promote if >55% wins
        # for now: just keep
        if model_out.exists():
            print(f"  model: {model_out} ({model_out.stat().st_size//1024} KB)")
        # next iteration would point NeuralPolicy at iter_dir/model.pt
        # (integration: pass --use-model in CLI of mtgsim.runner.cli — left as a follow-up)
    print("\nDone. Models in", ws)


if __name__ == "__main__":
    main()
