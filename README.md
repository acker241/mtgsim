# mtgsim — MTG Standard Dominaria-era simulator

Python sim of Magic: The Gathering Standard from the Dominaria era. Pits **Mono-Red Aggro** vs **Mono-White Aggro** with pluggable AI (heuristic + MCTS), live web UI, replay capture, and ML-ready data recording.

## Why

Compare two real decklists head-to-head at scale with consistent AI. Designed as a foundation for training a neural-network policy via self-play (AlphaZero-style).

## Status

- Engine: 40 unique cards (mainboards + sideboards), 13 keywords, mechanics (spectacle, convoke, ascend, saga, transform, mentor, planeswalker)
- AI: heuristic (mulligan→main→burn→combat→blocks) + MCTS Flat MC with PUCT-ready policy interface
- UI: FastAPI + WebSocket grid showing N parallel matches with pause/resume/step/speed
- Replay: per-match capture with HTML viewer (`/replay`)
- Recorder: JSONL gzipped, modes `summary` / `decisions` (NN-trainable)
- Metrics: per-game tactical metrics + `analyze.py` aggregator

## Quick start

```powershell
# install (editable)
py -m pip install -e .

# run 1000 matches headless (parallel workers)
py -m mtgsim.runner.cli --matches 1000 --workers 8

# launch UI with 4 parallel matches in browser
py -m mtgsim.runner.cli --ui --ui-matches 4 --port 8765
# open http://127.0.0.1:8765

# collect training data (decisions + metrics, gzipped)
py -m mtgsim.runner.cli --matches 2000 --workers 8 \
    --record-to data --record-mode decisions --metrics --ai mcts --n-sims 16

# analyze recorded matches
py -m mtgsim.scripts.analyze data

# train NN on recorded decisions
py -m mtgsim.scripts.train data --epochs 20 --out models/policy.pt

# run MCTS guided by trained NN
py -m mtgsim.runner.cli --matches 100 --workers 4 --ai mcts --n-sims 16 \
    --use-model models/policy.pt

# full self-play loop (multi-iteration)
py -m mtgsim.scripts.selfplay --iters 3 --games-per-iter 500 --workers 8
```

## Layout

```
mtgsim/
  engine/        # game state, turns, combat, cards, mana — pure Python, deterministic
  cards/         # red.py / white.py — card definitions
  ai/            # heuristic.py, mcts.py, mcts_ai.py, policy.py, encoding.py, action.py
  data/          # recorder.py, replay.py, metrics.py, loader.py
  runner/        # match.py, deck.py, cli.py, decks.py
  ui/            # FastAPI app + static HTML
  scripts/       # analyze.py + ad-hoc tools
data/            # recorded matches (JSONL.gz) — committed
docs/
  HISTORY.md     # design decisions + chronological build log
  NEXT_STEPS.md  # what's broken, what to do, NN roadmap
```

## Current results (heuristic AI + sideboard, 2000 BO3)

Mono-Red **47.3%**  /  Mono-White **52.6%** (close to real meta).

Improvements that closed the gap:
- Mana pool empties between phases (rule 106.4 fix)
- Aggro mode flag: stricter mulligan, default burn → face, attack through bad trades
- Spectacle sequencing: skip Skewer/LightUp full-cost in main1 (wait for combat trigger)
- Steam-Kin RRR activation with utility check (only if net positive)
- Wizard's Lightning chip burn in own main when wizard out
- Per-archetype sideboard swap between BO3 games (+10% red winrate)

## NN-ready hooks

The MCTS uses a `Policy` interface; swap `UniformPolicy` for `NeuralPolicy(model, encoder)` to plug a trained network. State encoder + action encoder stubs in `mtgsim/ai/encoding.py`. Recorder mode `decisions` saves `(state_vec, legal_actions, mcts_visits, chosen, value, outcome_z)` tuples directly trainable for AlphaZero-style policy+value loss.

See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) for the NN training roadmap.

## License

Personal project. No license file (treat as all rights reserved unless we add one).
