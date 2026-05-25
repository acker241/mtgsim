"""Aggregate metrics records across all games. Produce tactical report.

Usage:
  py -m mtgsim.scripts.analyze data/
"""
from __future__ import annotations
import argparse
import statistics
from collections import Counter
from ..data.loader import iter_records


def fmt_dist(values):
    if not values:
        return "(none)"
    return (f"n={len(values)} mean={statistics.mean(values):.2f} "
            f"median={statistics.median(values):.1f} "
            f"min={min(values)} max={max(values)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", default="data", nargs="?")
    ap.add_argument("--top-issues", type=int, default=10)
    args = ap.parse_args()

    metrics = list(iter_records(args.root, types=["metrics"]))
    summaries = list(iter_records(args.root, types=["match_end"]))
    print(f"Loaded {len(metrics)} game metrics, {len(summaries)} match summaries from {args.root}\n")

    if not metrics:
        print("No 'metrics' records found. Run with --metrics flag:")
        print("  py -m mtgsim.runner.cli --matches 1000 --workers 8 \\")
        print("      --record-to data --record-mode summary --metrics")
        return

    # determine which player is which deck (assumes deck0=Mono-Red consistently)
    # group: red metrics = data tagged for player whose deck name contains 'Red'
    red_chainwhirler_turns = []
    red_steamkin_max = []
    red_steamkin_activated = 0
    red_lethal_missed = 0
    red_first_creature_turn = []
    red_lands_t5 = []
    red_mulls = []
    spec_used_red = 0
    spec_wasted_red = 0
    wiz_disc_red = 0
    wiz_full_red = 0

    white_loxodon_turns = []
    white_marshal_turns = []
    white_first_creature_turn = []
    white_lands_t5 = []
    white_mulls = []

    wins = Counter()
    games_total = 0
    turns_all = []
    issues_counter = Counter()

    for m in metrics:
        games_total += 1
        p0_deck = m.get("p0_deck", "")
        p1_deck = m.get("p1_deck", "")
        # map players to red/white
        red_idx = 0 if "Red" in p0_deck else 1
        white_idx = 1 - red_idx

        # red metrics
        red_chainwhirler_turns += m.get(f"chainwhirler_cast_turns_p{red_idx}", [])
        if m.get(f"steamkin_max_counters_p{red_idx}", 0):
            red_steamkin_max.append(m[f"steamkin_max_counters_p{red_idx}"])
        red_steamkin_activated += m.get(f"steamkin_activated_p{red_idx}", 0)
        red_lethal_missed += m.get(f"lethal_missed_p{red_idx}", 0)
        if m.get(f"first_creature_turn_p{red_idx}") is not None:
            red_first_creature_turn.append(m[f"first_creature_turn_p{red_idx}"])
        if m.get(f"lands_in_play_t5_p{red_idx}") is not None:
            red_lands_t5.append(m[f"lands_in_play_t5_p{red_idx}"])
        red_mulls.append(m.get(f"mulls_p{red_idx}", 0))
        spec_used_red += m.get(f"spectacle_uses_p{red_idx}", 0)
        spec_wasted_red += m.get(f"spectacle_full_cost_p{red_idx}", 0)
        wiz_disc_red += m.get(f"wiz_lightning_disc_p{red_idx}", 0)
        wiz_full_red += m.get(f"wiz_lightning_full_p{red_idx}", 0)

        # white metrics
        white_loxodon_turns += m.get(f"loxodon_cast_turns_p{white_idx}", [])
        white_marshal_turns += m.get(f"marshal_cast_turns_p{white_idx}", [])
        if m.get(f"first_creature_turn_p{white_idx}") is not None:
            white_first_creature_turn.append(m[f"first_creature_turn_p{white_idx}"])
        if m.get(f"lands_in_play_t5_p{white_idx}") is not None:
            white_lands_t5.append(m[f"lands_in_play_t5_p{white_idx}"])
        white_mulls.append(m.get(f"mulls_p{white_idx}", 0))

        if m.get("winner"):
            wins[m["winner"]] += 1
        if m.get("draw"):
            wins["draw"] += 1
        turns_all.append(m.get("final_turn", 0))

        for iss in (m.get("issues") or []):
            issues_counter[iss.get("kind", "?")] += 1

    print("================== TACTICAL REPORT ==================")
    print(f"Games analyzed: {games_total}")
    print(f"Avg turns/game: {statistics.mean(turns_all):.1f}\n")

    print("--- WINRATE ---")
    total_decided = wins["Mono-Red"] + wins["Mono-White"]
    if total_decided > 0:
        print(f"  Mono-Red:   {wins['Mono-Red']:>5} ({wins['Mono-Red']/games_total*100:.1f}%)")
        print(f"  Mono-White: {wins['Mono-White']:>5} ({wins['Mono-White']/games_total*100:.1f}%)")
        print(f"  Draws:      {wins['draw']:>5} ({wins['draw']/games_total*100:.1f}%)")

    print("\n--- MONO-RED METRICS ---")
    print(f"  Chainwhirler cast turn:  {fmt_dist(red_chainwhirler_turns)}  (target T3)")
    print(f"  Chainwhirler cast rate:  {len(red_chainwhirler_turns)/games_total*100:.1f}% of games castable >=1")
    print(f"  First creature turn:     {fmt_dist(red_first_creature_turn)}  (target T1-2)")
    print(f"  Lands at T5:             {fmt_dist(red_lands_t5)}")
    print(f"  Mulls/game:              {fmt_dist(red_mulls)}")
    print()
    print(f"  Steam-Kin max counters:  {fmt_dist(red_steamkin_max)}  (3 = optimal)")
    print(f"  Steam-Kin RRR activated: {red_steamkin_activated} times across {games_total} games")
    print()
    spec_total = spec_used_red + spec_wasted_red
    if spec_total > 0:
        pct = spec_used_red/spec_total*100
        print(f"  Spectacle (Skewer/LightUp):  used {spec_used_red} | full-cost {spec_wasted_red}  ({pct:.1f}% efficient)")
    else:
        print(f"  Spectacle: 0 cast")
    wiz_total = wiz_disc_red + wiz_full_red
    if wiz_total > 0:
        pct = wiz_disc_red/wiz_total*100
        print(f"  Wizard's Lightning:          R-cost {wiz_disc_red} | 2R-cost {wiz_full_red}  ({pct:.1f}% discounted)")
    else:
        print(f"  Wizard's Lightning: 0 cast")
    print(f"\n  ! LETHAL MISSED (red had burn>=opp.life and didn't kill): {red_lethal_missed}")
    if games_total:
        print(f"    ({red_lethal_missed/games_total:.2f} per game)")

    print("\n--- MONO-WHITE METRICS ---")
    print(f"  Loxodon cast turn:       {fmt_dist(white_loxodon_turns)}  (target T3-4 with convoke)")
    print(f"  Loxodon cast rate:       {len(white_loxodon_turns)/games_total*100:.1f}% of games")
    print(f"  Marshal cast turn:       {fmt_dist(white_marshal_turns)}  (target T3 — WWW)")
    print(f"  First creature turn:     {fmt_dist(white_first_creature_turn)}")
    print(f"  Lands at T5:             {fmt_dist(white_lands_t5)}")
    print(f"  Mulls/game:              {fmt_dist(white_mulls)}")

    print("\n--- ISSUES DETECTED ---")
    if issues_counter:
        for k, v in issues_counter.most_common(args.top_issues):
            print(f"  {k}: {v}")
    else:
        print("  (none)")
    print("=====================================================")


if __name__ == "__main__":
    main()
