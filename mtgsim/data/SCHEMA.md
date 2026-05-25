# Recording Schema (JSONL)

One JSONL file per match. Each line is a JSON object with a `type` field.

## Record types

### `match_start`
```
{"type":"match_start","schema":1,"match_id":42,"deck0":"Mono-Red","deck1":"Mono-White","ts":1716580001.23,"meta":{}}
```

### `game_start`
```
{"type":"game_start","game_id":420,"play_first":0}
```

### `event` (mode=trace|full)
Emitted per engine observer event. Captures cast/etb/combat/etc.
```
{"type":"event","turn":5,"step":"PRECOMBAT_MAIN","active":0,"kind":"cast",
 "data":{"player_idx":0,"card":"Goblin Chainwhirler","spec":false,"x":0}}
```
Common `kind` values: `cast`, `play_land`, `burn`, `lethal_burn`, `kill_threat`,
`formation`, `frenzy_cast`, `frenzy_land`, `exile_cast`, `adanto_token`,
`mulligan_done`, `game_start`, `game_end`.

### `decision` (mode=full)
One per MCTS decision. Backfilled with `outcome_z` at game_end.
```
{"type":"decision","game_id":420,"root_idx":0,
 "state":[0.95,0.42,0.30,...],          # state vector (see mtgsim/ai/encoding.py)
 "legal":[{"kind":"cast","card_cid":12,...}, ...],
 "visits":[8,16,4, ...],                  # MCTS visit counts per legal action
 "chosen":1,
 "value":0.23,                            # MCTS value estimate at root
 "outcome_z":1.0}                         # +1 if root_idx won, -1 if lost, 0 draw
```

For NN training (AlphaZero-style):
- input  = `state`
- policy target = `visits` normalized (softmax over visits)
- value  target = `outcome_z`

### `game_end`
```
{"type":"game_end","game_id":420,"winner_idx":1,"turns":7,"draw":false}
```

### `match_end`
```
{"type":"match_end","match_id":42,"deck0_name":"Mono-Red","deck1_name":"Mono-White",
 "wins0":1,"wins1":2,"draws":0,"winner":"Mono-White","ts":1716580015.71}
```

## File layout
```
<root>/matches/YYYY-MM-DD/match_<match_id>_<pid>_<ms>.jsonl
```

## Sampling
`--record-sample 0.1` records 10% of matches. Each Recorder instance decides
per-match via its own RNG.

## Loading
```python
from mtgsim.data.loader import load_decisions, to_training_arrays
decisions = load_decisions("data/")
states, policies, values = to_training_arrays(decisions)
# train your NN on (states, policies, values)
```
