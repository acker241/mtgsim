"""Evaluate AI vs AI: heuristic vs heuristic, mcts+nn vs heuristic, etc.

Usage:
  py -m mtgsim.scripts.eval --matches 200 --workers 8 \
      --ai0 mcts --model0 models/policy.pt --ai1 heuristic
"""
from __future__ import annotations
import argparse
import random
import time
from collections import Counter
from multiprocessing import Pool
from ..runner.match import play_match
from ..runner.decks import mono_red, mono_white


def _make_factory(ai_kind: str, n_sims: int, model_path: str = None):
    """Returns a factory(name, rng, archetype) -> AI."""
    if ai_kind == "heuristic":
        from ..ai.heuristic import HeuristicAI
        def f(name, rng, archetype="midrange"):
            return HeuristicAI(name=name, rng=rng, archetype=archetype)
        return f
    if ai_kind == "mcts":
        from ..ai.mcts_ai import MctsAI
        from ..ai.nn import NeuralPolicy
        policy = NeuralPolicy(model_path=model_path) if model_path else None
        def f(name, rng, archetype="midrange"):
            return MctsAI(name=name, rng=rng, n_sims=n_sims, max_rollout_turns=4,
                          mcts_for_main=True, policy=policy)
        return f
    raise ValueError(f"unknown ai_kind: {ai_kind}")


def _worker(args):
    seed, max_turns, ai0_kind, ai0_model, ai1_kind, ai1_model, n_sims = args
    rng = random.Random(seed)
    d0, d1 = mono_red(), mono_white()
    f0 = _make_factory(ai0_kind, n_sims, ai0_model)
    f1 = _make_factory(ai1_kind, n_sims, ai1_model)
    # custom factory: player 0 uses f0, player 1 uses f1
    def factory(name, rng_, archetype="midrange"):
        # decide by name: AI:Mono-Red is player 0 (red), AI:Mono-White is player 1 (white)
        if "Mono-Red" in name:
            return f0(name, rng_, archetype)
        return f1(name, rng_, archetype)
    res = play_match(d0, d1, rng, max_turns=max_turns, match_id=seed,
                     ai_factory=factory)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--ai0", choices=["heuristic", "mcts"], default="heuristic",
                    help="AI for player 0 (Mono-Red)")
    ap.add_argument("--model0", default=None, help="Model path for ai0 if mcts")
    ap.add_argument("--ai1", choices=["heuristic", "mcts"], default="heuristic",
                    help="AI for player 1 (Mono-White)")
    ap.add_argument("--model1", default=None, help="Model path for ai1 if mcts")
    ap.add_argument("--n-sims", type=int, default=16)
    args = ap.parse_args()

    base_rng = random.Random(args.seed)
    seeds = [base_rng.randint(0, 2**31 - 1) for _ in range(args.matches)]
    worker_args = [(s, args.max_turns, args.ai0, args.model0,
                    args.ai1, args.model1, args.n_sims) for s in seeds]
    print(f"Eval: red={args.ai0}{f'(NN={args.model0})' if args.model0 else ''} "
          f"vs white={args.ai1}{f'(NN={args.model1})' if args.model1 else ''}")
    print(f"Matches: {args.matches}, workers: {args.workers}, n_sims: {args.n_sims}")

    t0 = time.time()
    if args.workers == 1:
        results = [_worker(a) for a in worker_args]
    else:
        with Pool(args.workers) as pool:
            results = pool.map(_worker, worker_args)
    dt = time.time() - t0

    wins0 = sum(1 for r in results if r.winner_name == "Mono-Red")
    wins1 = sum(1 for r in results if r.winner_name == "Mono-White")
    draws = args.matches - wins0 - wins1
    games_total = sum(len(r.games) for r in results)
    avg_turns = sum(g.turns for r in results for g in r.games) / max(1, games_total)

    print(f"\n=== {args.matches} matches in {dt:.1f}s ({args.matches/dt:.2f} match/s) ===")
    print(f"Mono-Red:   {wins0:>4} ({wins0/args.matches*100:.1f}%)")
    print(f"Mono-White: {wins1:>4} ({wins1/args.matches*100:.1f}%)")
    print(f"Draws:      {draws:>4} ({draws/args.matches*100:.1f}%)")
    print(f"Avg turns/game: {avg_turns:.2f}")


if __name__ == "__main__":
    main()
