# Next Steps

Two parallel tracks. Pick one, or do both. Engine fixes (#1) are prerequisite for either AI track.

---

## Track 0 — Engine correctness (DO FIRST)

Current heuristic + MCTS rest on top of an engine that has gaps. NN trained on a broken engine will learn a broken game. Fix engine first.

### 0.1 Priority windows
Currently `ai_priority` in `engine/turn.py` lets each player respond once per step, but heuristic `_priority` in `ai/heuristic.py` only does anything in 3 steps (END_STEP, DECLARE_ATTACKERS, DECLARE_BLOCKERS). Expand:
- NAP should be able to cast burn in opp's upkeep, draw, main1, main2 (any time they want to remove a threat)
- AP should be able to cast their own burn in own main phases freely (already partial)

### 0.2 Stack responses
Spells resolve immediately on `resolve_top`. There's no window to respond to a spell on the stack. For these decks:
- White Unbreakable Formation in response to red Lava Coil → not modeled
- Wizard's Lightning in response to white attack triggers → not modeled

Implement minimal: after `cast_spell` pushes to stack, call `ai_priority` BEFORE `resolve_top`. Then each player gets a chance to respond. Loop until both pass, then resolve top.

### 0.3 Combat damage assignment
Defender currently doesn't get to order multiple blockers optimally. Heuristic always picks the first listed. For double-block on a big attacker, defender should pick the ordering that maximizes blocker survival.

### 0.4 First-strike step skip
Currently always runs even if no creature has first-strike. Mild perf, not correctness.

### 0.5 Verify card implementations
Run a single game with `log_enabled=True` and walk through events for each card. Check:
- Steam-Kin counter cap at 3 (rule text)
- Wizard's Lightning discount fires correctly
- Spectacle resets correctly when turn ends
- Saga sacrifices on final chapter (currently uses `__sacrifice__` counter hack)
- Adanto (transformed Legion's Landing) taps for W properly

---

## Track A — Heuristic refinement

Hand-tune AI for measurable improvement. Goal: red 45-55% win rate (close to real meta).

### A.1 Aggro mode flag
Currently heuristic is generic. Add per-deck flag `is_aggro` and default burn → face when aggro:
- Aggro: burn face unless target is Marshal/Loxodon/Tithe-Taker/lifelinker
- Control: burn threats first

### A.2 Wizard's Lightning fix
Diagnose why 0 casts despite explicit code paths. Likely candidates:
- `_has_wizard` returning false when wizard is tapped/sick
- Cast cost computation returning `R` not enforced in `cast_spell` validation
- Mana not available because mountains were spent on creatures

Add log/print to confirm and fix.

### A.3 Steam-Kin reachability
Max counters mean 2.11 — never reaches 3. Causes:
- Steam-Kin dies before 3 red spells cast after it lands
- Cards used DON'T trigger (e.g., not actually colored R despite Color.R flag)
- The trigger condition `spell.cdef.colors & Color.R == 0` may be skipping things

Add explicit counter logging during run. Also consider: AI should protect Steam-Kin (don't attack with it if it survives = anthem).

### A.4 Spectacle sequencing
Spectacle efficiency 41.8% means Skewer/Light Up are cast at full cost often. Reason: cast BEFORE Viashino/Lavarunner triggered damage that turn.

Fix sequencing: within main_phase ranking, ANY card that deals damage to opp (Viashino ETB, Chainwhirler ETB, Wizard's Lightning) gets cast BEFORE any spectacle sorcery.

### A.5 Hold mana for Chainwhirler
T3+ when Chainwhirler is in hand AND we have ≥3 mountains untap, don't cast 1-drops that tap mountains. Currently scoring favors Chainwhirler (+50 bonus), but if a 1-drop is also castable AI may cast it first because of curve heuristics.

Add explicit "reserved mana" check.

### A.6 Experimental Frenzy heuristic
Need data on whether Frenzy is being cast at all and how productive it is. Add metrics:
- `frenzy_cast_turn`
- `frenzy_plays_after_cast` (count of cards played via Frenzy)
- `frenzy_lands_drawn_via` (lands stuck on top)

Heuristic should cast Frenzy when hand ≤ 3 cards AND BF has ≥ 2 creatures. Don't cast Frenzy when Chainwhirler is castable.

### A.7 Estimated impact
Implementing A.1 through A.5 should move red from 30% → 45-55%. ~600 LoC of changes.

---

## Track B — Neural network training (AlphaZero-style)

Replace heuristic with a learned policy+value network via self-play.

### B.1 Prerequisites
- Track 0 done (engine correct)
- Track A.1+A.2 at minimum (so baseline heuristic isn't broken — needed for bootstrap)

### B.2 Richer state encoder
Current `state_to_vector` in `mtgsim/ai/encoding.py` is 198 dims (life, library size, hand count, BF counts by card name). Too shallow for a NN.

Expand to ~1000-3000 dims:
- For each player: hand cards as multi-hot over `CARD_NAMES` (~38 dims × 2 = 76)
- BF cards as multi-hot per type (creatures separate from non-creatures)
- BF creature stats: power, toughness, counters, tapped, attacking
- Mana available by color (W, R) per player
- Graveyard counts per card (instant/sorcery for Lavarunner trigger)
- Phase/step one-hot (8 dims)
- Turn number, active player
- Saga chapter counters
- Steam-Kin counters
- City's blessing flag

### B.3 Richer action encoder
Current `action_to_idx` returns coarse kind*100 buckets. Pra NN com action mask, precisa de:
- Discrete action space: enumerate every possible (kind, source_card_slot, target_slot)
- Card slots: one slot per CARD_NAME (~38)
- Target slots: per player (2) + per creature (variable, use top-K) + "no target"
- Action dim: ~6 kinds × 38 cards × 12 targets ≈ 2700 (rough)

Provide `legal_mask(actions, action_dim)` for NN to mask illegal actions before softmax.

### B.4 Model
Small PyTorch MLP to start:
```python
class PolicyValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), torch.tanh(self.value_head(h))
```

Loss: `kl_div(softmax(policy), mcts_visits/sum) + mse(value, outcome_z)`.

### B.5 Training loop (self-play iteration)
```
for iteration in range(N):
    # generate self-play data
    run 1000+ matches with current policy via MCTS
    save (state, mcts_visits, outcome_z) tuples
    
    # train
    load decisions from this iter's recordings
    train PolicyValueNet for K epochs
    
    # evaluate
    play 100 matches: new_net vs prev_net
    if new_net wins > 55%: promote, else discard
```

Already implemented: data collection (`Recorder` mode=decisions), MCTS structure (PUCT-ready), policy interface.

Need: training script (`mtgsim/scripts/train.py`), evaluation harness, model save/load, NeuralPolicy implementation.

### B.6 Expected outcomes
- 1k self-play games + 50 epochs training = ~30 min CPU, modest improvement
- 10k games + 100 epochs = few hours, real improvement
- Multiple iterations = converge to balanced (maybe 50/50 on real matchups)

### B.7 Risks
- State encoding may not capture enough — NN learns surface patterns, loses to heuristic
- MCTS too shallow with n_sims=16 — NN's policy hint dominates badly
- Hidden info (opp hand, deck shuffling) limits ceiling — would need POMDP variant for true ceiling

### B.8 Estimated effort
- Richer encoders: ~300 LoC + 1 day testing
- Model + training script: ~500 LoC + 1 day
- Iteration loop: ~200 LoC + 1 day
- Tuning: open-ended

---

## Track C — Ergonomics / polish (optional)

### C.1 Sideboard plans
Currently BO3 plays game 2/3 with same 60-card maindeck. Implement per-deck sideboard plan:
- Red SB plan: vs white-aggro, board in Lava Coil + Rekindling Phoenix
- White SB plan: vs red-aggro, board in Tocatli Honor Guard + Baffling End

### C.2 UI improvements
- Show MCTS decision tree at flagged moments
- Filter aggregate stats by deck on play / on draw
- Live histogram: turn-of-win distribution

### C.3 Replay viewer
- Add "next decision" / "prev decision" buttons
- Show inline burn calculus when AI made a face vs creature choice

### C.4 More decks
Extend to other Dominaria-era archetypes (UB Pirates, GW Tokens). Each new deck = ~30-40 unique cards.

---

## Recommended next sprint (if starting fresh at home)

**Day 1**: Track 0 fixes (priority windows + stack responses + log a game manually)
**Day 2**: Track A.4 (spectacle sequencing) + A.2 (Wizard's Lightning) + re-run 2000 matches to measure
**Day 3**: Track A.1 (aggro mode) + A.5 (hold mana) + A.3 (Steam-Kin reach)
**Day 4**: Track B.2 (richer encoder) + B.3 (action encoder)
**Day 5**: Track B.4 (PyTorch model) + B.5 (training script, first iteration)

After day 5, decide: more heuristic tuning vs more NN iteration based on results.
