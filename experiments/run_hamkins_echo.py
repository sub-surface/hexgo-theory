"""
Hamkins-echo experiment — does the draw fraction rise as we let games run longer?

Motivation: Hamkins & Leonessi (2022) showed that Infinite Hex (connection-style,
doubly-infinite winning path) is a draw under optimal play. Our game is alignment-
style 6-in-a-row, not connection-style, so the theorem does not transfer
directly — but the *intuition* does: as the playable horizon grows, two
competent players should force draws more often, because each side can keep
interdicting the other's threats indefinitely.

This experiment is a finite-horizon probe of that intuition.

Design:
  - Fix WIN_LENGTH = 6 (hardcoded in the engine).
  - Vary max_moves (temporal horizon) over a geometric sweep.
  - For each horizon, play N games per matchup and record
    (Black wins, White wins, unfinished).
  - Three matchups:
      combo_vs_combo     — symmetric strong play
      greedy_vs_combo    — moderate asymmetry
      random_vs_combo    — extreme asymmetry

Prediction (Hamkins echo):
  - combo_vs_combo:   unfinished fraction rises with horizon, then plateaus.
  - random_vs_combo:  strong player keeps winning regardless — draws require
                      *mutual* competence.
  - greedy_vs_combo:  intermediate.

Run:
    python -X utf8 experiments/run_hamkins_echo.py
    python -X utf8 experiments/run_hamkins_echo.py --quick
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Worktree fix: engine/__init__.py expects hexgo as sibling of the theory repo,
# but inside a .claude/worktrees/* checkout that resolution is wrong. Point it
# at the real hexgo repo explicitly before touching `engine`.
_REAL_HEXGO = Path(r"C:\Users\Leon\Desktop\Psychograph\hexgo")
if _REAL_HEXGO.exists() and str(_REAL_HEXGO) not in sys.path:
    sys.path.insert(0, str(_REAL_HEXGO))

import matplotlib.pyplot as plt
import numpy as np

from engine import (
    HexGame, RandomAgent, EisensteinGreedyAgent,
    ComboAgent,
)

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(exist_ok=True, parents=True)


AGENT_FACTORIES = {
    "random": lambda: RandomAgent(),
    "greedy": lambda: EisensteinGreedyAgent("greedy_def", defensive=True),
    "combo":  lambda: ComboAgent("combo"),
}


def play_one(fac_black, fac_white, max_moves: int, seed: int) -> dict:
    """Play a single game, return outcome dict."""
    import random as _r
    _r.seed(seed)
    a, b = fac_black(), fac_white()
    g = HexGame()
    m = 0
    while g.winner is None and m < max_moves:
        ag = a if g.current_player == 1 else b
        legal = g.legal_moves()
        if not legal:
            break
        mv = ag.choose_move(g)
        if mv not in set(legal):
            mv = _r.choice(legal)
        g.make(*mv)
        m += 1
    return {
        "winner": g.winner,            # 1=Black, 2=White, None=unfinished
        "length": len(g.move_history),
        "hit_cap": g.winner is None and m >= max_moves,
    }


def run_matchup(black: str, white: str, horizons: list[int], n_per_cell: int,
                seed0: int = 20260417) -> dict:
    fac_b = AGENT_FACTORIES[black]
    fac_w = AGENT_FACTORIES[white]
    rows = []
    for h in horizons:
        t0 = time.time()
        outcomes = [play_one(fac_b, fac_w, h, seed=seed0 + i) for i in range(n_per_cell)]
        b_wins = sum(1 for o in outcomes if o["winner"] == 1)
        w_wins = sum(1 for o in outcomes if o["winner"] == 2)
        unfin  = sum(1 for o in outcomes if o["winner"] is None)
        lengths = [o["length"] for o in outcomes]
        rows.append({
            "horizon": h,
            "n_games": n_per_cell,
            "black_wins": b_wins,
            "white_wins": w_wins,
            "unfinished": unfin,
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
        })
        print(f"  {black:>6s} vs {white:<6s}  horizon={h:>4d}  "
              f"B={b_wins:>3d}  W={w_wins:>3d}  unfin={unfin:>3d}  "
              f"mean_len={np.mean(lengths):6.1f}  ({time.time()-t0:.1f}s)")
    return {"black": black, "white": white, "rows": rows}


def plot_outcomes(all_results: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    matchup_names = list(all_results.keys())

    for ax, mkey in zip(axes, matchup_names):
        rows = all_results[mkey]["rows"]
        horizons = np.array([r["horizon"] for r in rows])
        n = np.array([r["n_games"] for r in rows])
        b = np.array([r["black_wins"] for r in rows]) / n
        w = np.array([r["white_wins"] for r in rows]) / n
        u = np.array([r["unfinished"] for r in rows]) / n

        ax.stackplot(horizons, b, w, u,
                     labels=["Black wins", "White wins", "Unfinished"],
                     colors=["#1f1f1f", "#bfbfbf", "#d62728"],
                     alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("horizon  (max_moves, log scale)")
        ax.set_title(mkey.replace("_", " "))
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("fraction of games")
    axes[-1].legend(loc="center right", framealpha=0.95)
    fig.suptitle("Hamkins echo — does 'unfinished' fraction rise with horizon?\n"
                 "Prediction: yes for symmetric strong play, no for asymmetric.",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_lengths(all_results: dict, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"combo_vs_combo": "#d62728",
              "greedy_vs_combo": "#1f77b4",
              "random_vs_combo": "#888"}
    for mkey, res in all_results.items():
        rows = res["rows"]
        horizons = np.array([r["horizon"] for r in rows])
        mean_len = np.array([r["mean_length"] for r in rows])
        ax.plot(horizons, mean_len, "o-", color=colors.get(mkey, None),
                label=mkey.replace("_", " "))
    # reference: y = x (every game uses the full horizon)
    ax.plot(horizons, horizons, "k--", alpha=0.4, label="y = horizon (always ran out)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("horizon  (max_moves)")
    ax.set_ylabel("mean game length")
    ax.set_title("Game length vs horizon\n"
                 "Games that hug the diagonal are running out the clock (draw-like).")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--horizons", type=int, nargs="*", default=None)
    parser.add_argument("--n", type=int, default=None, help="games per cell")
    args = parser.parse_args()

    if args.quick:
        horizons = [30, 60, 120]
        n = 20
    else:
        horizons = args.horizons or [30, 60, 120, 240, 480]
        n = args.n or 50

    print(f"\n── Hamkins echo scan ──  horizons={horizons}  n_per_cell={n}")

    matchups = [
        ("random", "combo"),
        ("greedy", "combo"),
        ("combo",  "combo"),
    ]
    all_results = {}
    for black, white in matchups:
        mkey = f"{black}_vs_{white}"
        print(f"\n[{mkey}]")
        all_results[mkey] = run_matchup(black, white, horizons, n)

    RESULTS_PATH = RESULTS_DIR / "hamkins_echo.json"
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\n[saved] {RESULTS_PATH}")

    plot_outcomes(all_results, FIGURES_DIR / "fig_hamkins_echo_outcomes.png")
    plot_lengths(all_results,  FIGURES_DIR / "fig_hamkins_echo_lengths.png")
    print(f"[saved] figures/fig_hamkins_echo_outcomes.png")
    print(f"[saved] figures/fig_hamkins_echo_lengths.png")


if __name__ == "__main__":
    main()
