"""
MirrorAgent validation — tests Proposition P2 of the Hamkins synthesis.

P2 (from docs/theory/2026-04-17-hamkins-synthesis.md):
  A point-reflection MirrorAgent achieves non-loss >= 90% against RandomAgent
  but strictly less than 50% as second player against any stronger agent than
  Random (consistent with strategy-stealing).

Matchups:
  - Mirror-W vs Random-B  (Mirror as second player, weak opponent)
  - Mirror-B vs Random-W
  - Mirror-W vs ca_combo_v2-B  (Mirror as second player, strong opponent)
  - Mirror-B vs ca_combo_v2-W

Outputs results/mirror_agent.json + figures/fig_mirror_agent.png.
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


PAIRS = [
    ("random",      "mirror"),
    ("mirror",      "random"),
    ("ca_combo_v2", "mirror"),
    ("mirror",      "ca_combo_v2"),
]


def _run(n_games: int, parallelism: int, max_moves: int, seed: int) -> dict:
    print(f"\n── MirrorAgent P2 test ──  n={n_games}  parallel={parallelism}  "
          f"horizon={max_moves}\n")

    out = {}
    t0 = time.perf_counter()
    for idx, (b, w) in enumerate(PAIRS):
        r = run_matchup(b, w, n_games=n_games, parallelism=parallelism,
                        seed=seed + idx * 100_000, max_moves=max_moves)
        print("  " + r.summary())
        out[f"{b}__vs__{w}"] = r.to_dict()
    out["_wall_time"] = time.perf_counter() - t0
    out["_params"] = dict(n_games=n_games, parallelism=parallelism,
                          max_moves=max_moves, seed=seed)
    return out


def _plot(results: dict, path: str) -> None:
    """Grouped bar chart: each matchup's B-wins / W-wins / unfinished."""
    labels = ["Rand-B / Mirr-W", "Mirr-B / Rand-W",
              "Combo-B / Mirr-W", "Mirr-B / Combo-W"]
    wb = [results[f"{b}__vs__{w}"]["wins_black"] for b, w in PAIRS]
    ww = [results[f"{b}__vs__{w}"]["wins_white"] for b, w in PAIRS]
    un = [results[f"{b}__vs__{w}"]["unfinished"] for b, w in PAIRS]
    n = results[f"{PAIRS[0][0]}__vs__{PAIRS[0][1]}"]["n_games"]

    x = np.arange(len(labels))
    width = 0.28
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, wb, width, label="Black wins", color="#222")
    ax.bar(x,         ww, width, label="White wins", color="#bbb",
           edgecolor="#333")
    ax.bar(x + width, un, width, label="unfinished", color="#e58")

    for i in range(len(labels)):
        ax.text(x[i] - width, wb[i] + 0.4, f"{wb[i]}", ha="center", fontsize=8)
        ax.text(x[i],         ww[i] + 0.4, f"{ww[i]}", ha="center", fontsize=8)
        ax.text(x[i] + width, un[i] + 0.4, f"{un[i]}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"games (of n={n})")
    ax.set_title("MirrorAgent (point-reflection pairing) outcomes\n"
                 "P2 prediction: non-loss >= 90% vs Random-P1; < 50% as P2 vs stronger agent")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def _verdict(results: dict) -> str:
    lines = ["P2 (MirrorAgent non-loss) test:"]

    # Mirror-as-White vs Random — non-loss.
    r = results["random__vs__mirror"]
    nl = r["wins_white"] + r["unfinished"]
    n = r["n_games"]
    lo, hi = _wilson(nl, n)
    status = "OK" if nl / n >= 0.9 else "FAIL"
    lines.append(f"  Mirror-W vs Random-B non-loss = {nl}/{n} = {nl/n:.2f}  "
                 f"Wilson95=[{lo:.2f},{hi:.2f}]  [{status}; target >=0.90]")

    # Mirror-as-White vs Combo — must be < 50% wins (strategy-stealing).
    r = results["ca_combo_v2__vs__mirror"]
    wins = r["wins_white"]
    n = r["n_games"]
    lo, hi = _wilson(wins, n)
    status = "OK" if wins / n < 0.5 else "FAIL"
    lines.append(f"  Mirror-W vs Combo-B   wins     = {wins}/{n} = {wins/n:.2f}  "
                 f"Wilson95=[{lo:.2f},{hi:.2f}]  [{status}; target <0.50]")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--parallelism", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max-moves", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20260417)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n = 20
        args.max_moves = 120

    results = _run(args.n, args.parallelism, args.max_moves, args.seed)

    results_path = Path("results") / "mirror_agent.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {results_path}")

    fig_path = Path("figures") / "fig_mirror_agent.png"
    _plot(results, str(fig_path))
    print(f"[saved] {fig_path}")

    print("\n── Verdict ──\n" + _verdict(results))


if __name__ == "__main__":
    main()
