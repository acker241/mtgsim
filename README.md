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

## Current results (heuristic AI, 2000 BO3)

Mono-Red ~30%  /  Mono-White ~70%

Red is underperforming vs. real meta (~50%). See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) for diagnosed bugs and fix plan.

## NN-ready hooks

The MCTS uses a `Policy` interface; swap `UniformPolicy` for `NeuralPolicy(model, encoder)` to plug a trained network. State encoder + action encoder stubs in `mtgsim/ai/encoding.py`. Recorder mode `decisions` saves `(state_vec, legal_actions, mcts_visits, chosen, value, outcome_z)` tuples directly trainable for AlphaZero-style policy+value loss.

See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) for the NN training roadmap.

## License

Personal project. No license file (treat as all rights reserved unless we add one).
