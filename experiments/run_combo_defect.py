"""
Combo-going-Black defect test.

Background: the 2026-04-17 Hamkins-echo sweep surfaced an anomaly — in
ComboAgent-vs-ComboAgent matches, White won roughly 3× more often than
Black (29 vs 10 at horizon=480). This contradicts the strategy-stealing
argument (there can be no second-player winning strategy in Connect-6-
style games), so it must be an ComboAgent flaw, not a game-truth
statement.

Diagnosis: the opening move has no potential gradient — an empty board
gives the same Erdős–Selfridge score to every candidate cell — so ComboAgent
fell through to the epsilon-noise term. `make_combo_v2_ca` adds
`feat_opening_center_bias` (hex-distance-decayed preference for cells
near the origin during the first few stones), inactive after move 4.

Test design:
  1. Head-to-head: ca_combo vs ca_combo_v2 in each colour seat. If v2
     dominates symmetrically, the opening bias is real improvement.
  2. Self-play colour symmetry: ca_combo_v2-vs-ca_combo_v2 should show
     Black winrate ≥ White winrate (strategy-stealing lower bound) or
     at worst a tie.

Outputs:
  - results/combo_defect.json
  - figures/fig_combo_defect.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# worktree shim — see CLAUDE.md > "Worktree gotcha"
_REAL_HEXGO = Path(r"C:\Users\Leon\Desktop\Psychograph\hexgo")
if _REAL_HEXGO.exists() and str(_REAL_HEXGO) not in sys.path:
    sys.path.insert(0, str(_REAL_HEXGO))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.harness import run_matchup, _wilson


def _run(n_games: int, parallelism: int, max_moves: int, seed: int) -> dict:
    print(f"\n── Combo-defect sweep ──  n={n_games}  parallel={parallelism}  "
          f"horizon={max_moves}\n")

    pairs = [
        ("ca_combo",    "ca_combo"),
        ("ca_combo_v2", "ca_combo_v2"),
        ("ca_combo",    "ca_combo_v2"),
        ("ca_combo_v2", "ca_combo"),
    ]

    out = {}
    t0 = time.perf_counter()
    # Per-matchup seed offsets break a subtle correlation: with the same base
    # seed across matchups, worker i of every matchup gets the same game RNG,
    # so tiny-eps-noise agents produce near-identical trajectories.
    for idx, (b, w) in enumerate(pairs):
        r = run_matchup(b, w, n_games=n_games, parallelism=parallelism,
                        seed=seed + idx * 100_000, max_moves=max_moves)
        print("  " + r.summary())
        out[f"{b}__vs__{w}"] = r.to_dict()
    out["_wall_time"] = time.perf_counter() - t0
    out["_params"] = dict(n_games=n_games, parallelism=parallelism,
                          max_moves=max_moves, seed=seed)
    return out


def _plot(results: dict, path: str) -> None:
    """Bar chart: Black win, White win, unfinished for each matchup."""
    pairs = [
        ("ca_combo__vs__ca_combo", "v1 B / v1 W"),
        ("ca_combo_v2__vs__ca_combo_v2", "v2 B / v2 W"),
        ("ca_combo__vs__ca_combo_v2", "v1 B / v2 W"),
        ("ca_combo_v2__vs__ca_combo", "v2 B / v1 W"),
    ]
    labels = [p[1] for p in pairs]
    n = [results[p[0]]["n_games"] for p in pairs]
    wb = [results[p[0]]["wins_black"] for p in pairs]
    ww = [results[p[0]]["wins_white"] for p in pairs]
    un = [results[p[0]]["unfinished"] for p in pairs]

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_b = ax.bar(x - width, wb, width, label="Black wins", color="#222")
    bars_w = ax.bar(x, ww, width, label="White wins", color="#bbb",
                    edgecolor="#333")
    bars_u = ax.bar(x + width, un, width, label="unfinished", color="#e58")

    for bars in (bars_b, bars_w, bars_u):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.4, f"{int(h)}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"games (of n={n[0]})")
    ax.set_title("Combo vs Combo-v2: does the opening-centre bias help?\n"
                 "(v2 = make_combo_v2_ca with feat_opening_center_bias)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def _verdict(results: dict) -> str:
    """Short human-readable verdict on whether v2 fixes the defect."""
    v1 = results["ca_combo__vs__ca_combo"]
    v2 = results["ca_combo_v2__vs__ca_combo_v2"]

    def _black_rate(r):
        n = r["wins_black"] + r["wins_white"]
        return r["wins_black"] / n if n > 0 else 0.5

    r1 = _black_rate(v1)
    r2 = _black_rate(v2)
    lo2, hi2 = _wilson(v2["wins_black"], v2["wins_black"] + v2["wins_white"])

    lines = [
        f"v1 self-play: Black share of decisive games = {r1:.2f}  "
        f"(B={v1['wins_black']} / W={v1['wins_white']})",
        f"v2 self-play: Black share of decisive games = {r2:.2f}  "
        f"(B={v2['wins_black']} / W={v2['wins_white']})  Wilson95=[{lo2:.2f},{hi2:.2f}]",
    ]
    if r2 >= 0.5:
        lines.append("→ v2 restores Black >= White (strategy-stealing lower bound respected).")
    elif hi2 >= 0.5:
        lines.append("→ v2 not statistically worse than parity.")
    else:
        lines.append("→ v2 still Black-disadvantaged; opening bias is insufficient.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--parallelism", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max-moves", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20260417)
    ap.add_argument("--quick", action="store_true",
                    help="small n for fast iteration")
    args = ap.parse_args()

    if args.quick:
        args.n = 20
        args.max_moves = 120

    results = _run(args.n, args.parallelism, args.max_moves, args.seed)

    results_path = Path("results") / "combo_defect.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {results_path}")

    fig_path = Path("figures") / "fig_combo_defect.png"
    _plot(results, str(fig_path))
    print(f"[saved] {fig_path}")

    print("\n── Verdict ──\n" + _verdict(results))


if __name__ == "__main__":
    main()
