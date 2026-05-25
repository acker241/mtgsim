"""FastAPI + WebSocket. N partidas em threads. Cliente recebe estado serializado."""
from __future__ import annotations
import asyncio
import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..engine.observer import Observer
from ..engine.enums import Subtype
from ..runner.match import play_match, DeckSpec
from ..runner.decks import mono_red, mono_white


STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class MatchRunState:
    match_id: int
    pause_event: threading.Event
    speed_ms: int = 100
    step_remaining: int = 0   # if >0, allow N ticks then auto-pause
    thread: Optional[threading.Thread] = None
    log: deque = field(default_factory=lambda: deque(maxlen=400))
    last_state: dict = field(default_factory=dict)
    cumulative: dict = field(default_factory=dict)  # tally of game/match results
    running: bool = True
    seed: int = 0
    flag_replay: bool = False  # if True, capture the NEXT match in this slot as a replay
    last_replay_file: Optional[str] = None


class Hub:
    """Owns runners + clients. Broadcasts state."""
    def __init__(self, n_matches: int, base_seed: int, max_turns: int,
                 ai_mode: str = "heuristic", n_sims: int = 16,
                 record_to: Optional[str] = None, record_mode: str = "summary",
                 record_sample: float = 1.0, compress: bool = True):
        self.n_matches = n_matches
        self.base_seed = base_seed
        self.max_turns = max_turns
        self.ai_mode = ai_mode  # "heuristic" or "mcts"
        self.n_sims = n_sims
        self.record_to = record_to
        self.record_mode = record_mode
        self.record_sample = record_sample
        self.compress = compress
        self.record_stats = {"matches_written": 0, "events": 0, "decisions": 0, "bytes": 0}
        self.runs: Dict[int, MatchRunState] = {}
        self.clients: List[WebSocket] = []
        self.aggregate = {"wins": {"Mono-Red": 0, "Mono-White": 0}, "draws": 0, "matches_done": 0}
        self.lock = threading.Lock()
        self.global_pause = threading.Event()
        self.global_pause.set()  # initially not paused (event set = running)
        self.global_speed_ms = 100
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def _ai_factory(self):
        if self.ai_mode == "mcts":
            from ..ai.mcts_ai import MctsAI
            sims = self.n_sims
            def f(name, rng):
                return MctsAI(name=name, rng=rng, n_sims=sims, max_rollout_turns=4,
                              mcts_for_main=True, mcts_for_attacks=False, mcts_for_blocks=False)
            return f
        return None  # heuristic default

    def start_all(self):
        for i in range(self.n_matches):
            self._spawn_match(i, self.base_seed + i)

    def _spawn_match(self, slot: int, seed: int):
        state = MatchRunState(
            match_id=slot,
            pause_event=threading.Event(),
            speed_ms=self.global_speed_ms,
            seed=seed,
        )
        state.pause_event.set()
        self.runs[slot] = state
        t = threading.Thread(target=self._run_match_thread, args=(slot,), daemon=True)
        state.thread = t
        t.start()

    def _run_match_thread(self, slot: int):
        state = self.runs[slot]
        rng = random.Random(state.seed)
        hub = self
        observer = Observer()
        recorder = None
        if hub.record_to:
            from ..data.recorder import Recorder
            recorder = Recorder(root=hub.record_to, mode=hub.record_mode,
                                sample=hub.record_sample,
                                rng=random.Random(state.seed),
                                file_prefix=f"slot{slot}",
                                compress=hub.compress)

        def on_event(ev):
            line = self._fmt_event(ev)
            if line:
                state.log.append(line)
            # snapshot game state passed in observer? We don't have direct access here.
            # The tick callback below pulls state from a closure variable.

        def tick():
            # global pause
            while not hub.global_pause.is_set():
                time.sleep(0.05)
            # per-match pause
            while not state.pause_event.is_set():
                if state.step_remaining > 0:
                    state.step_remaining -= 1
                    state.pause_event.set()  # let one through
                    break
                time.sleep(0.05)
            if state.step_remaining > 0:
                state.step_remaining -= 1
                if state.step_remaining == 0:
                    state.pause_event.clear()
            # speed-based throttle
            ms = state.speed_ms if state.speed_ms is not None else hub.global_speed_ms
            if ms > 0:
                time.sleep(ms / 1000.0)
            # snapshot serialize is done from main loop via push_state

        observer.subscribe(on_event)
        observer.tick = tick

        # patch observer.emit to also push state after every event (for tight UI updates)
        orig_emit = observer.emit

        def emit_and_snapshot(kind, game, data=None):
            orig_emit(kind, game, data)
            try:
                state.last_state = serialize_game(game)
            except Exception:
                pass

        observer.emit = emit_and_snapshot

        # play match in loop forever (restart on completion to keep grid alive)
        match_counter = 0
        while True:
            match_counter += 1
            d0, d1 = mono_red(), mono_white()
            # check if this slot was flagged for replay
            replay = None
            if state.flag_replay:
                from ..data.replay import ReplayCapture
                replay = ReplayCapture(
                    root=str(Path(hub.record_to or "data")),
                    slot=slot, match_id=slot * 100000 + match_counter,
                    seed=state.seed + match_counter,
                )
                replay.header(d0.name, d1.name)
                state.flag_replay = False
                state.log.append(f"** capturing replay for match #{match_counter} **")
                # subscribe replay to observer
                observer.subscribe(replay.record_event)
            try:
                # if replay active, attach to all games via observer hook
                if replay is not None:
                    # patch observer.emit to ensure replay.attach_game(game) before each event
                    orig_emit2 = observer.emit
                    def emit_with_replay(kind, game, data=None, _o=orig_emit2, _r=replay):
                        _r.attach_game(game)
                        _o(kind, game, data)
                    observer.emit = emit_with_replay
                res = play_match(d0, d1, rng, observer=observer,
                                 max_turns=hub.max_turns, match_id=slot,
                                 ai_factory=hub._ai_factory(),
                                 recorder=recorder)
                if recorder is not None:
                    state.cumulative["recorder"] = dict(recorder.stats)
                if replay is not None:
                    replay.finalize({
                        "match_id": replay.match_id, "winner": res.winner_name,
                        "wins0": res.wins0, "wins1": res.wins1,
                        "games": len(res.games),
                    })
                    state.last_replay_file = replay.file_path.name if replay.file_path else None
                    state.log.append(f"** replay saved: {state.last_replay_file} **")
                    # unsubscribe replay
                    if replay.record_event in observer.subs:
                        observer.subs.remove(replay.record_event)
                    # restore emit
                    observer.emit = emit_and_snapshot
                with hub.lock:
                    if res.winner_name:
                        hub.aggregate["wins"][res.winner_name] = hub.aggregate["wins"].get(res.winner_name, 0) + 1
                    hub.aggregate["matches_done"] += 1
                state.cumulative["last_winner"] = res.winner_name
                state.cumulative["games"] = len(res.games)
            except Exception as e:
                state.log.append(f"!! crash: {e}")
                time.sleep(0.5)
            # new seed for next match to keep variation
            rng = random.Random(rng.randint(0, 2**31 - 1))

    def _fmt_event(self, ev) -> Optional[str]:
        if ev.kind == "log":
            return f"[T{ev.turn} {ev.step}] {ev.data.get('msg','')}"
        if ev.kind == "cast":
            return f"[T{ev.turn}] P{ev.data.get('player_idx')} casts {ev.data.get('card')}"
        if ev.kind == "game_start":
            return f"--- GAME START: {ev.data.get('p0')} vs {ev.data.get('p1')} (P{ev.data.get('play_first')} first) ---"
        if ev.kind == "game_end":
            return f"=== GAME END: winner={ev.data.get('winner')} turns={ev.data.get('turns')} ==="
        return None

    # ----- controls -----
    def set_global_speed(self, ms: int):
        self.global_speed_ms = ms
        for s in self.runs.values():
            s.speed_ms = ms

    def set_global_pause(self, paused: bool):
        if paused:
            self.global_pause.clear()
        else:
            self.global_pause.set()

    def step_global(self, n: int = 1):
        # set step on each match
        for s in self.runs.values():
            s.step_remaining = n
            s.pause_event.set()

    def snapshot(self) -> dict:
        rec_agg = {"matches_written": 0, "events": 0, "decisions": 0, "bytes": 0}
        for s in self.runs.values():
            r = s.cumulative.get("recorder") or {}
            for k in rec_agg:
                rec_agg[k] += r.get(k, 0)
        return {
            "n_matches": self.n_matches,
            "aggregate": self.aggregate,
            "paused": not self.global_pause.is_set(),
            "speed_ms": self.global_speed_ms,
            "record_to": self.record_to,
            "record_mode": self.record_mode,
            "record_stats": rec_agg,
            "matches": [
                {
                    "id": s.match_id,
                    "seed": s.seed,
                    "log": list(s.log)[-30:],
                    "state": s.last_state,
                    "cumulative": s.cumulative,
                    "flagged": s.flag_replay,
                    "last_replay": s.last_replay_file,
                }
                for s in self.runs.values()
            ],
        }


# -------- serialization --------
def serialize_card_short(c) -> dict:
    return {
        "cid": c.cid,
        "name": c.name,
        "tap": c.tapped,
        "sick": c.summoning_sick,
        "p": (c.cdef.power if c.cdef.power is not None else None),
        "t": (c.cdef.toughness if c.cdef.toughness is not None else None),
        "counters": dict(c.counters),
        "is_token": c.is_token,
        "is_land": c.cdef.is_land(),
        "is_creature": c.cdef.is_creature(),
        "attacking": c.attacking,
        "blocking": list(c.blocking),
    }


def serialize_player(p) -> dict:
    return {
        "idx": p.idx,
        "name": p.name,
        "life": p.life,
        "hand_count": len(p.hand),
        "library_count": len(p.library),
        "graveyard_count": len(p.graveyard),
        "mana": {k: v for k, v in p.mana_pool.pool.items() if v},
        "city_blessing": p.city_blessing,
        "battlefield": [serialize_card_short(c) for c in p.battlefield],
        "mulligans": p.mulligans_taken,
        "lost": p.lost,
    }


def serialize_game(game) -> dict:
    return {
        "turn": game.turn,
        "active_idx": game.active_idx,
        "phase": game.phase.name if game.phase else "",
        "step": game.step.name if game.step else "",
        "stack_size": len(game.stack),
        "players": [serialize_player(p) for p in game.players],
        "winner_idx": game.winner_idx,
        "draw": game.draw_game,
    }


# -------- FastAPI app --------
app = FastAPI()
HUB: Optional[Hub] = None


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/replay")
def replay_page():
    return FileResponse(STATIC_DIR / "replay.html")


@app.get("/api/replays")
def list_replays():
    base = Path(HUB.record_to or "data") / "replays"
    if not base.exists():
        return JSONResponse([])
    files = []
    for fp in sorted(base.glob("*.jsonl*"), key=lambda p: -p.stat().st_mtime):
        files.append({
            "name": fp.name,
            "size": fp.stat().st_size,
            "mtime": fp.stat().st_mtime,
        })
    return JSONResponse(files[:200])


@app.get("/api/replay/{name}")
def fetch_replay(name: str):
    import gzip
    base = Path(HUB.record_to or "data") / "replays"
    fp = base / name
    if not fp.exists() or not str(fp.resolve()).startswith(str(base.resolve())):
        raise HTTPException(404, "not found")
    if fp.suffix == ".gz":
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            text = f.read()
    else:
        with fp.open("r", encoding="utf-8") as f:
            text = f.read()
    return JSONResponse({"lines": text.splitlines()})


@app.post("/api/flag/{slot}")
def flag_slot(slot: int):
    if slot not in HUB.runs:
        raise HTTPException(404, "slot not found")
    HUB.runs[slot].flag_replay = True
    return {"ok": True, "slot": slot, "flagged": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    HUB.clients.append(ws)
    try:
        # state push loop
        async def pusher():
            while True:
                snap = HUB.snapshot()
                try:
                    await ws.send_text(json.dumps(snap))
                except Exception:
                    break
                await asyncio.sleep(0.25)

        push_task = asyncio.create_task(pusher())
        try:
            while True:
                msg = await ws.receive_text()
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                cmd = data.get("cmd")
                if cmd == "pause":
                    HUB.set_global_pause(True)
                elif cmd == "resume":
                    HUB.set_global_pause(False)
                elif cmd == "step":
                    HUB.step_global(int(data.get("n", 1)))
                elif cmd == "speed":
                    HUB.set_global_speed(int(data.get("ms", 100)))
        finally:
            push_task.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in HUB.clients:
            HUB.clients.remove(ws)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run_ui(port: int = 8765, n_matches: int = 8, seed: int = 42, max_turns: int = 25,
           ai_mode: str = "heuristic", n_sims: int = 16,
           record_to: Optional[str] = None, record_mode: str = "summary",
           record_sample: float = 1.0, compress: bool = True):
    import uvicorn
    global HUB
    HUB = Hub(n_matches=n_matches, base_seed=seed, max_turns=max_turns,
              ai_mode=ai_mode, n_sims=n_sims,
              record_to=record_to, record_mode=record_mode, record_sample=record_sample,
              compress=compress)
    HUB.start_all()
    msg = f"UI at http://127.0.0.1:{port} (ai={ai_mode}, n_sims={n_sims}, matches={n_matches})"
    if record_to:
        msg += f" recording->{record_to} mode={record_mode} sample={record_sample}"
    print(msg)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
