"""Match / game runner. BO3 with sideboard, mulligans, alternating play/draw."""
from __future__ import annotations
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import random
import time
from ..engine.game import GameState
from ..engine.player import Player
from ..engine.card import Card, CardDef
from ..engine.enums import Zone, Phase, Step
from ..engine.observer import Observer, NULL_OBSERVER
from ..engine import turn as turn_mod
from ..ai.heuristic import HeuristicAI
from .deck import build_library, build_sideboard


def make_heuristic(name, rng, archetype="midrange"):
    return HeuristicAI(name=name, rng=rng, archetype=archetype)


@dataclass
class DeckSpec:
    name: str
    main: list   # [(CardDef, count), ...]
    sideboard: list
    archetype: str = "midrange"   # 'aggro' | 'midrange' | 'control'
    # sideboard plan: {opp_archetype: [(in_card_name, out_card_name, count), ...]}
    sb_plan: dict = field(default_factory=dict)


@dataclass
class GameResult:
    winner_name: Optional[str]
    turns: int
    mulligans_p0: int
    mulligans_p1: int
    draw: bool = False
    duration_s: float = 0.0


@dataclass
class MatchResult:
    deck0_name: str
    deck1_name: str
    games: List[GameResult] = field(default_factory=list)
    wins0: int = 0
    wins1: int = 0
    draws: int = 0

    @property
    def winner_name(self) -> Optional[str]:
        if self.wins0 >= 2:
            return self.deck0_name
        if self.wins1 >= 2:
            return self.deck1_name
        return None


def setup_game(deck0: DeckSpec, deck1: DeckSpec, rng: random.Random,
               play_first_idx: int, observer: Observer = NULL_OBSERVER,
               game_id: int = 0,
               ai_factory: Optional[Callable] = None):
    p0 = Player(idx=0, name=deck0.name)
    p1 = Player(idx=1, name=deck1.name)
    lib0, next_cid = build_library(deck0.main, owner_idx=0, start_cid=1)
    lib1, next_cid = build_library(deck1.main, owner_idx=1, start_cid=next_cid)
    p0.library = lib0
    p1.library = lib1
    p0.shuffle(rng)
    p1.shuffle(rng)

    game = GameState(players=[p0, p1], rng=rng, observer=observer, game_id=game_id)
    game.next_cid = next_cid
    game.active_idx = play_first_idx
    factory = ai_factory or make_heuristic
    # pass archetype if factory accepts it
    import inspect
    sig = inspect.signature(factory)
    if "archetype" in sig.parameters:
        ai0 = factory(f"AI:{deck0.name}", rng, archetype=deck0.archetype)
        ai1 = factory(f"AI:{deck1.name}", rng, archetype=deck1.archetype)
    else:
        ai0 = factory(f"AI:{deck0.name}", rng)
        ai1 = factory(f"AI:{deck1.name}", rng)
    return game, ai0, ai1


def do_mulligans(game: GameState, ai0: HeuristicAI, ai1: HeuristicAI):
    """London mulligan. Each player draws 7, decides keep/mull; mull = shuffle & draw 7 again; on keep with k mulligans, put k cards bottom."""
    ais = [ai0, ai1]
    for idx, ai in enumerate(ais):
        pl = game.players[idx]
        while True:
            # draw 7
            for c in pl.hand:
                c.zone = Zone.LIBRARY
            pl.library = pl.hand + pl.library
            pl.hand = []
            pl.shuffle(game.rng)
            pl.draw(7, game)
            if not ai._mulligan(game, idx):
                break
            pl.mulligans_taken += 1
            if pl.mulligans_taken >= 4:
                break
        # london: put bottom mulligans_taken cards
        for _ in range(pl.mulligans_taken):
            if not pl.hand:
                break
            # bottom the worst (highest cmc if many lands, else lowest-impact land)
            lands = [c for c in pl.hand if c.cdef.is_land()]
            nonlands = [c for c in pl.hand if not c.cdef.is_land()]
            target = None
            if len(lands) > 4 and lands:
                target = lands[0]
            elif nonlands:
                # discard highest cmc
                nonlands.sort(key=lambda c: -(c.cdef.cost.cmc() if c.cdef.cost else 0))
                target = nonlands[0]
            else:
                target = pl.hand[0]
            pl.hand.remove(target)
            target.zone = Zone.LIBRARY
            pl.library.append(target)
        game.observer.emit("mulligan_done", game, {"player_idx": idx,
                                                   "mulligans": pl.mulligans_taken,
                                                   "hand_size": len(pl.hand)})


def play_game(deck0: DeckSpec, deck1: DeckSpec, rng: random.Random,
              play_first_idx: int = 0, observer: Observer = NULL_OBSERVER,
              max_turns: int = 25, game_id: int = 0,
              log_enabled: bool = False,
              ai_factory: Optional[Callable] = None,
              recorder: Optional[Any] = None,
              metrics_collector: Optional[Any] = None) -> GameResult:
    t0 = time.time()
    game, ai0, ai1 = setup_game(deck0, deck1, rng, play_first_idx, observer, game_id, ai_factory=ai_factory)
    game.max_turns = max_turns
    game.log_enabled = log_enabled
    game.recorder = recorder
    if recorder is not None:
        recorder.start_game(game_id, play_first_idx)
        observer.subscribe(recorder.record_event)
    if metrics_collector is not None:
        from ..data.metrics import install_metrics_hooks
        install_metrics_hooks(game, metrics_collector)
    do_mulligans(game, ai0, ai1)
    observer.emit("game_start", game, {"p0": deck0.name, "p1": deck1.name,
                                       "play_first": play_first_idx})

    # main loop
    def step_fn(g, kind, **kwargs):
        idx = kwargs.get("player_idx", g.active_idx)
        ai = ai0 if idx == 0 else ai1
        return ai(g, kind, **kwargs)

    while not game.is_over():
        observer.wait_if_paused()
        turn_mod.take_turn(game, step_fn)
        if game.is_over():
            break
        turn_mod.advance_turn(game)

    winner_name = None
    if game.winner_idx is not None:
        winner_name = game.players[game.winner_idx].name
    res = GameResult(
        winner_name=winner_name,
        turns=game.turn,
        mulligans_p0=game.players[0].mulligans_taken,
        mulligans_p1=game.players[1].mulligans_taken,
        draw=game.draw_game,
        duration_s=time.time() - t0,
    )
    observer.emit("game_end", game, {"winner": winner_name, "turns": game.turn, "draw": game.draw_game})
    if recorder is not None:
        recorder.end_game(game_id, game.winner_idx, game.turn, game.draw_game)
    return res


def _apply_sb(spec: DeckSpec, opp_archetype: str) -> DeckSpec:
    """Return new DeckSpec with sideboard plan applied for given opponent archetype."""
    plan = spec.sb_plan.get(opp_archetype) or []
    if not plan:
        return spec
    # build new main: copy then swap
    main_counts = {cdef.name: [cdef, count] for cdef, count in spec.main}
    sb_counts = {cdef.name: [cdef, count] for cdef, count in spec.sideboard}
    for in_name, out_name, n in plan:
        if in_name in sb_counts and out_name in main_counts:
            take = min(n, sb_counts[in_name][1], main_counts[out_name][1])
            sb_counts[in_name][1] -= take
            main_counts[out_name][1] -= take
            # add to main (or merge)
            if in_name in main_counts:
                main_counts[in_name][1] += take
            else:
                main_counts[in_name] = [sb_counts[in_name][0], take]
    new_main = [(cdef, cnt) for cdef, cnt in main_counts.values() if cnt > 0]
    return DeckSpec(name=spec.name, main=new_main, sideboard=spec.sideboard,
                    archetype=spec.archetype, sb_plan=spec.sb_plan)


def play_match(deck0: DeckSpec, deck1: DeckSpec, rng: random.Random,
               observer: Observer = NULL_OBSERVER, max_turns: int = 25,
               match_id: int = 0,
               ai_factory: Optional[Callable] = None,
               recorder: Optional[Any] = None,
               collect_metrics: bool = False) -> MatchResult:
    res = MatchResult(deck0_name=deck0.name, deck1_name=deck1.name)
    if recorder is not None:
        recorder.start_match(match_id, deck0.name, deck1.name)
    play_first = 0
    # game 1 uses main decks as-is
    deck0_g = deck0
    deck1_g = deck1
    for game_idx in range(3):
        if res.wins0 >= 2 or res.wins1 >= 2:
            break
        # game 2+: apply sideboard plan
        if game_idx >= 1:
            deck0_g = _apply_sb(deck0, deck1.archetype)
            deck1_g = _apply_sb(deck1, deck0.archetype)
        mc = None
        if collect_metrics:
            from ..data.metrics import MetricsCollector
            mc = MetricsCollector()
        gr = play_game(deck0_g, deck1_g, rng, play_first_idx=play_first,
                       observer=observer, max_turns=max_turns,
                       game_id=match_id * 10 + game_idx,
                       ai_factory=ai_factory,
                       recorder=recorder,
                       metrics_collector=mc)
        if mc is not None and recorder is not None:
            recorder.record_metrics(mc.finalize(deck0.name, deck1.name))
        res.games.append(gr)
        if gr.draw or gr.winner_name is None:
            res.draws += 1
        elif gr.winner_name == deck0.name:
            res.wins0 += 1
        else:
            res.wins1 += 1
        # next game: loser plays first
        if gr.winner_name == deck0.name:
            play_first = 1
        else:
            play_first = 0
        # sideboard swap (simplified: skip — both decks have small SBs)
        # TODO: implement sideboard plans
    if recorder is not None:
        recorder.end_match({
            "match_id": match_id,
            "deck0_name": deck0.name,
            "deck1_name": deck1.name,
            "wins0": res.wins0,
            "wins1": res.wins1,
            "draws": res.draws,
            "winner": res.winner_name,
            "games": len(res.games),
        })
    return res
