"""
Neural CA agent benchmark — untrained baseline.

Shows the GPU pipeline is alive, measures the untrained random-weight
hex-conv NCA against Random (floor) and Combo-v2 (ceiling). Training
comes in a follow-up experiment.

Output:
  results/neural_ca_benchmark.json
  figures/fig_neural_ca_benchmark.png
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

from experiments.harness import run_matchup


PAIRS = [
    ("random",      "neural_ca"),
    ("neural_ca",   "random"),
    ("ca_combo_v2", "neural_ca"),
    ("neural_ca",   "ca_combo_v2"),
]


def _run(n_games: int, max_moves: int, seed: int) -> dict:
    print(f"\n── NeuralCA benchmark ──  n={n_games}  horizon={max_moves}  "
          f"parallelism=1 (CUDA)\n")
    out = {}
    t0 = time.perf_counter()
    for idx, (b, w) in enumerate(PAIRS):
        r = run_matchup(b, w, n_games=n_games, parallelism=1,
                        seed=seed + idx * 100_000, max_moves=max_moves)
        print("  " + r.summary())
        out[f"{b}__vs__{w}"] = r.to_dict()
    out["_wall_time"] = time.perf_counter() - t0
    out["_params"] = dict(n_games=n_games, max_moves=max_moves, seed=seed)
    return out


def _plot(results: dict, path: str) -> None:
    labels = ["NCA-W vs Rand-B", "NCA-B vs Rand-W",
              "NCA-W vs Combo-B", "NCA-B vs Combo-W"]
    # NCA wins (i.e. from the NCA seat).
    nca_wins = []
    other_wins = []
    unfin = []
    for (b, w) in PAIRS:
        r = results[f"{b}__vs__{w}"]
        if b == "neural_ca":
            nca_wins.append(r["wins_black"])
            other_wins.append(r["wins_white"])
        else:
            nca_wins.append(r["wins_white"])
            other_wins.append(r["wins_black"])
        unfin.append(r["unfinished"])
    n = results[f"{PAIRS[0][0]}__vs__{PAIRS[0][1]}"]["n_games"]

    x = np.arange(len(labels))
    width = 0.28
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, nca_wins,   width, label="NCA wins",  color="#36a")
    ax.bar(x,         other_wins, width, label="opp wins",  color="#bbb",
           edgecolor="#333")
    ax.bar(x + width, unfin,      width, label="unfinished", color="#e58")
    for i in range(len(labels)):
        ax.text(x[i] - width, nca_wins[i] + 0.4, f"{nca_wins[i]}",
                ha="center", fontsize=8)
        ax.text(x[i],         other_wins[i] + 0.4, f"{other_wins[i]}",
                ha="center", fontsize=8)
        ax.text(x[i] + width, unfin[i] + 0.4, f"{unfin[i]}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"games (of n={n})")
    ax.set_title("Untrained NeuralCAAgent (hex-conv stack, random weights) benchmark")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260417)
    args = ap.parse_args()

    results = _run(args.n, args.max_moves, args.seed)

    rpath = Path("results") / "neural_ca_benchmark.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {rpath}")

    fig = Path("figures") / "fig_neural_ca_benchmark.png"
    _plot(results, str(fig))
    print(f"[saved] {fig}")


if __name__ == "__main__":
    main()
