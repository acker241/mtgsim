"""CLI: run N matches headless, print stats."""
from __future__ import annotations
import argparse
import random
import sys
import time
from typing import Optional
from collections import Counter
from multiprocessing import Pool
from .match import play_match, MatchResult
from .decks import mono_red, mono_white


def _worker(args):
    (seed, max_turns, ai_mode, n_sims, rec_dir, rec_mode, rec_sample,
     compress, collect_metrics) = args
    rng = random.Random(seed)
    d0, d1 = mono_red(), mono_white()
    ai_factory = None
    if ai_mode == "mcts":
        from ..ai.mcts_ai import MctsAI
        sims = n_sims
        def ai_factory(name, rng_):
            return MctsAI(name=name, rng=rng_, n_sims=sims, max_rollout_turns=4,
                          mcts_for_main=True, mcts_for_attacks=False, mcts_for_blocks=False)
    recorder = None
    if rec_dir:
        from ..data.recorder import Recorder
        recorder = Recorder(root=rec_dir, mode=rec_mode, sample=rec_sample,
                            rng=random.Random(seed), compress=compress)
    res = play_match(d0, d1, rng, max_turns=max_turns, match_id=seed,
                     ai_factory=ai_factory, recorder=recorder,
                     collect_metrics=collect_metrics)
    return res


def run_headless(n_matches: int, workers: int, seed: int, max_turns: int,
                 ai_mode: str = "heuristic", n_sims: int = 16,
                 rec_dir: Optional[str] = None, rec_mode: str = "summary",
                 rec_sample: float = 1.0, compress: bool = True,
                 collect_metrics: bool = False):
    base_rng = random.Random(seed)
    seeds = [base_rng.randint(0, 2**31 - 1) for _ in range(n_matches)]
    args = [(s, max_turns, ai_mode, n_sims, rec_dir, rec_mode, rec_sample,
             compress, collect_metrics) for s in seeds]
    t0 = time.time()
    if workers == 1:
        results = [_worker(a) for a in args]
    else:
        with Pool(workers) as pool:
            results = pool.map(_worker, args)
    dt = time.time() - t0

    wins0 = sum(1 for r in results if r.winner_name == "Mono-Red")
    wins1 = sum(1 for r in results if r.winner_name == "Mono-White")
    draws = n_matches - wins0 - wins1
    total_games = sum(len(r.games) for r in results)
    avg_turns = sum(g.turns for r in results for g in r.games) / max(1, total_games)
    avg_mulls_p0 = sum(g.mulligans_p0 for r in results for g in r.games) / max(1, total_games)
    avg_mulls_p1 = sum(g.mulligans_p1 for r in results for g in r.games) / max(1, total_games)

    print(f"\n=== {n_matches} matches in {dt:.1f}s ({n_matches/dt:.1f} matches/s, {workers} workers) ===")
    print(f"Mono-Red   wins: {wins0:>5} ({wins0/n_matches*100:.1f}%)")
    print(f"Mono-White wins: {wins1:>5} ({wins1/n_matches*100:.1f}%)")
    print(f"Draws/incomplete: {draws} ({draws/n_matches*100:.1f}%)")
    print(f"Avg turns/game: {avg_turns:.2f}")
    print(f"Avg mulls Mono-Red:   {avg_mulls_p0:.2f}")
    print(f"Avg mulls Mono-White: {avg_mulls_p1:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--ui", action="store_true", help="Launch FastAPI UI instead of headless")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ui-matches", type=int, default=8, help="Parallel matches in UI grid")
    ap.add_argument("--ai", choices=["heuristic", "mcts"], default="heuristic")
    ap.add_argument("--n-sims", type=int, default=16, help="MCTS rollouts per decision")
    ap.add_argument("--record-to", type=str, default=None, help="Dir to save JSONL recordings")
    ap.add_argument("--record-mode", choices=["summary", "decisions"], default="summary",
                    help="summary=match results only; decisions=+ MCTS decision tuples (trainable)")
    ap.add_argument("--no-compress", action="store_true", help="Disable gzip (default: gzipped)")
    ap.add_argument("--metrics", action="store_true",
                    help="Collect per-match tactical metrics (Chainwhirler turn, spectacle usage, lethal-miss, etc)")
    ap.add_argument("--record-sample", type=float, default=1.0,
                    help="Fraction of matches to record (0..1)")
    args = ap.parse_args()
    compress = not args.no_compress
    if args.ui:
        from ..ui.app import run_ui
        run_ui(port=args.port, n_matches=args.ui_matches, seed=args.seed,
               max_turns=args.max_turns, ai_mode=args.ai, n_sims=args.n_sims,
               record_to=args.record_to, record_mode=args.record_mode,
               record_sample=args.record_sample, compress=compress)
    else:
        run_headless(args.matches, args.workers, args.seed, args.max_turns,
                     ai_mode=args.ai, n_sims=args.n_sims,
                     rec_dir=args.record_to, rec_mode=args.record_mode,
                     rec_sample=args.record_sample, compress=compress,
                     collect_metrics=args.metrics)


if __name__ == "__main__":
    main()
