# History — design decisions log

Chronological log of what was built, why, and what was learned. Read top-to-bottom for full context.

---

## 0. Problem statement

User wants to pit two AIs against each other playing real MTG Standard decks from the Dominaria era (Mono-Red Aggro vs Mono-White Aggro) to measure:
- Which AI plays better
- Which deck is better
- Need to simulate MTG rules + matches at scale

## 1. Engine choice — Python from scratch

Considered: Forge (Java), MTGSDK + custom, Python from scratch, LLM-as-player.

**Picked Python from scratch** because:
- Forge JVM startup is heavy per process (~300MB), hurts parallelism
- LLM has rate-limits + cost + non-determinism
- For 2 specific decks (~40 unique cards), implementing the subset of rules is tractable (~3-4k LoC)
- Pure Python state = picklable, deepcopy-able, supports multiprocessing and MCTS rollouts

Trade-off accepted: not all MTG rules implemented (no counterspells, no flash interactions, simplified prioridade). Those don't matter for the two decks here.

## 2. Engine architecture

Decided on a "data + functions" layout (no classes-with-methods for game logic) so state can be pickled and copied cleanly:

- `engine/enums.py` — Zone, Phase, Step, CardType, Keyword
- `engine/mana.py` — ManaCost (parse "2R" → {GENERIC:2, R:1}) + ManaPool
- `engine/card.py` — `CardDef` (immutable template, singleton via `__deepcopy__`) + `Card` (per-game instance)
- `engine/player.py` — Player state (life, hand, library, BF, etc.)
- `engine/game.py` — GameState + stack + observer + state-based actions
- `engine/actions.py` — `cast_spell`, `play_land`, `pay_cost`, `deal_damage`
- `engine/combat.py` — declare/damage steps, first strike, blocking
- `engine/turn.py` — turn loop driver, calls AI at decision points

Pluggable callback `ai_step(game, kind, **kwargs)` invoked at: `main`, `priority`, `declare_attackers`, `declare_blockers`, `mulligan`. Same signature for heuristic and MCTS AIs.

## 3. Card framework

Each card = `CardDef` with optional hooks: `on_resolve`, `on_etb`, `on_ltb`, `on_dies`, plus lists of `TriggeredAbility` / `ActivatedAbility` / `StaticEffect`.

Mechanics implemented:
- Spectacle (alt cost when opp lost life this turn) — `cdef.spectacle_cost` slot
- Convoke (tap creatures for mana when casting) — `cast_spell` accepts `convoke_creatures`
- Ascend / city's blessing — `Player.city_blessing` flag set when permanents ≥ 10
- Saga (3 chapters, sacrifice after final) — `cdef.chapters: List[Callable]`
- Transform (Legion's Landing → Adanto) — `card.card_def = new_def()` swap
- Historic trigger (Wizard's Lightning discount with Wizard) — reuses `spectacle_cost` slot as discount cost (mild hack, gated by card name check in `cast_spell`)
- Planeswalker loyalty (Ajani)
- Tokens (lifelink, flying, etc.)
- ETB suppression by Tocatli Honor Guard (global static)

Static effects walk all permanents each time stats are computed. Slow but simple. Cards: Benalish Marshal (+1/+1 to other creatures), Ghitu Lavarunner (conditional self-haste), etc.

## 4. AI heuristic (v1)

`HeuristicAI` (mtgsim/ai/heuristic.py). Decisions made per callback kind:

- **Mulligan** — London. Keep if 2-5 lands, mull otherwise (up to 3).
- **Main phase** — sequence: play_land → try lethal burn → try kill threat with burn → cast spell ranked by curve+priority → activated abilities (Adanto token).
- **Priority** — instants on opponent's end step (lethal burn first); Unbreakable Formation in own declare_blockers if attackers blocked.
- **Attackers** — attack unless trade clearly bad AND not racing.
- **Blockers** — kill attacker free > trade > chump if lethal incoming.

Scoring per card (in `_try_cast_sorcery_or_creature`):
- Base: `cmc * 10`
- `+5` for creatures
- `+50` Goblin Chainwhirler (deck's best 3-drop)
- `+30` Benalish Marshal
- `+25` History of Benalia / Conclave Tribunal
- `+25` Venerated Loxodon (if convoke savings)
- `+30` Experimental Frenzy (if hand >3)
- `-5` Skewer/LightUp without spectacle (prefer them at R cost)

Mostly "midrange-ish" play. Result vs. real meta: Mono-Red ~30% (should be ~50%). See NEXT_STEPS.md.

## 5. MCTS

Flat Monte Carlo with UCB1/PUCT at the root (no deep tree). Structured to be AlphaZero-ready:

- `Policy` abstract → `UniformPolicy` / `HeuristicPolicy` / `NeuralPolicy(stub)`
- MCTS root expands all legal actions, runs rollouts via heuristic, picks most-visited action
- Rollouts truncated at `max_rollout_turns`, static value heuristic at cutoff (life diff + power diff + hand diff)
- `_clone(game)` uses `copy.deepcopy` with `CardDef.__deepcopy__ = self` (singleton) + Observer/Recorder detached
- Records `(state_vec, legal_actions, visits, chosen, value)` to game.recorder

Performance: ~46 BO3/s heuristic, ~0.22 BO3/s with `n_sims=16` MCTS (rollouts dominate). Acceptable for "thoughtful but slow" mode.

## 6. UI (FastAPI + WebSocket)

Grid layout, N matches running in parallel threads, each with its own observer + recorder. WS pushes state JSON every 250ms. Controls: pause/resume/step/speed (global). Per-match: `⚐ Flag` button captures the NEXT match end-to-end as a replay.

Decisions:
- Threading (not multiprocessing) because each thread shares the Hub state + observer subs
- `observer.tick()` hook called per AI action — implements pause + speed throttle
- `serialize_game(game)` produces compact JSON for client
- HTML/CSS in static/index.html — minimal vanilla JS, no framework

## 7. Recorder — modes evolution

Iteration 1: `summary` / `trace` / `full`. `full` was ~3MB/match (events alone are 95% of size).

Iteration 2 (current): dropped `trace`/`full`. Keep `summary` / `decisions`. Plus gzip default. Replay capture moved to separate on-demand `ReplayCapture` class (only fires for flagged matches in UI).

Sizes:
- `summary` mode: ~200B/match
- `decisions` mode: ~30-50KB/match gzipped
- ReplayCapture (per match, all events + state snapshots): ~25KB gzipped

Rationale: most matches are noise for ML; only decisions matter. Debug needs are on-demand, not bulk. Flag a match in UI when something looks off, get full trace.

## 8. Metrics + analyze script

`MetricsCollector` subscribes to observer for chosen events (`cast`, `game_end`) + snapshots state at `precombat_main` / `end_step`. Tracks per game:
- Card cast turns (Chainwhirler, Loxodon, Marshal)
- Spectacle: used vs full cost
- Wizard's Lightning: discounted vs full
- Steam-Kin: max counters, activations
- Lethal-miss heuristic (own turn end: total burn in hand >= opp life and didn't kill)
- First-creature turn, lands at T5, mulligans

`mtgsim/scripts/analyze.py` aggregates across all `metrics` records in a recording dir and prints a tactical report.

First analysis (2000 BO3 matches):
- Mono-Red 32%, Mono-White 68%
- Chainwhirler avg cast turn 5.58 (should be T3)
- Wizard's Lightning: **0 casts in 5127 games** (bug)
- Steam-Kin RRR ability: **0 activations** (bug — no code path for it)
- Spectacle efficiency 41.8% (often cast at full cost when discount was available)

## 9. Engine fix — mana empties between phases

MTG rule 106.4 says mana pool empties at end of each step/phase. My engine only emptied at untap. **Bug.** Fixed: added `_empty_mana_pools_all(game)` at end of every step function. Doesn't change "mana available" because AI computes `pool + untapped lands` each time.

## 10. Heuristic v2 fixes attempted

Added:
- `_try_steamkin_activation` — activates Steam-Kin RRR ability ONLY if net positive (RRR enables ≥3 damage worth of casts, OR Steam-Kin doomed anyway)
- `_try_chip_burn_main` — Wizard's Lightning at face when surplus mana otherwise floats (initially main2 only, then expanded to main1/main2)
- Wizard's Lightning discount path validated in `_try_lethal_burn_main` and `_try_kill_threat_main`

Re-ran 2000 matches. **Still 30% red.** Wizard's Lightning still 0 casts. Steam-Kin RRR still 0 activations.

Diagnosis: heuristic windows too restrictive. NAP red only tries burn in 3 of ~13 steps. Main-phase burn only fires for lethal or threat-kill, never for chip damage. Aggro/control mode not differentiated.

## 11. Decision point: heuristic refinement vs NN training

User asked: "if NN can learn the patterns by self-play, why am I hand-tuning heuristics?"

Honest answer: NN can learn via AlphaZero loop, but needs:
1. Correct simulator (or NN learns wrong game)
2. Rich state encoder (current 198-dim is shallow)
3. Lots of self-play data (1M+ games)
4. Heuristic baseline as bootstrap (random play converges too slow)

Project paused on heuristic-refinement vs NN-training fork. Pushed to repo to continue at home.

---

## Code stats (current commit)

```
mtgsim/
  engine/        ~900 LoC
  cards/         ~700 LoC (red+white)
  ai/            ~1100 LoC (heuristic+mcts+policy+encoding+action)
  data/          ~500 LoC (recorder+replay+metrics+loader)
  runner/        ~250 LoC
  ui/            ~600 LoC (app + 2 html pages)
  scripts/       ~150 LoC (analyze)
total            ~4200 LoC Python + ~600 LoC HTML/JS
```

## Performance benchmarks

- Heuristic headless 8 workers: ~70 BO3 matches/s
- Heuristic single-thread: ~10 BO3 matches/s
- MCTS n_sims=16 single-thread: ~0.22 BO3/s
- Recorder mode=decisions gzipped: ~30KB/match
- Recorder mode=summary: ~200B/match

## Key files for someone picking this up

- [README.md](../README.md) — overview + quick start
- [docs/NEXT_STEPS.md](NEXT_STEPS.md) — what to do next
- [mtgsim/data/SCHEMA.md](../mtgsim/data/SCHEMA.md) — recording format
- [mtgsim/ai/heuristic.py](../mtgsim/ai/heuristic.py) — the AI's decisions live here
- [mtgsim/ai/mcts.py](../mtgsim/ai/mcts.py) — MCTS engine
- [mtgsim/ai/policy.py](../mtgsim/ai/policy.py) — where to plug a NN
