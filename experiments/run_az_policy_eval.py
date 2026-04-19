"""Phase 1 policy-head evaluation: how well does the pretrained trunk
play by itself, *without* MCTS?

If the pretrained trunk's policy head already beats ca_combo_v2 on a
n=50 sample, Phase 2 (MCTS self-play iteration) is pure upside. If it
loses badly, the pretrain is under-fitting and we should extend Phase 0
before investing search-time compute.

Runs three matchups with the `harness.play_one` helper:
    az_policy  vs  random
    az_policy  vs  ca_combo_v2
    ca_combo_v2 vs az_policy

Saves n_games, wins_black, wins_white, unfinished to results/az_policy_eval.json
and produces a small bar chart.

Usage:
    python experiments/run_az_policy_eval.py             # 50 games per matchup
    python experiments/run_az_policy_eval.py --quick     # 10 games per matchup
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_REAL_HEXGO = Path(r"C:\Users\Leon\Desktop\Psychograph\hexgo")
if _REAL_HEXGO.exists() and str(_REAL_HEXGO) not in sys.path:
    sys.path.insert(0, str(_REAL_HEXGO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import HexGame, RandomAgent
from engine.az_agent import make_az_agent
from engine.ca_policy import make_combo_v2_ca


def play_match(black_factory, white_factory, n: int, max_moves: int = 240) -> dict:
    wins_b = wins_w = unfinished = 0
    t0 = time.time()
    for g in range(n):
        blk = black_factory()
        wht = white_factory()
        game = HexGame()
        m = 0
        while game.winner is None and m < max_moves:
            mover = blk if game.current_player == 1 else wht
            mv = mover.choose_move(game)
            if mv is None or mv in game.board:
                break
            if not game.make(*mv):
                break
            m += 1
        if game.winner == 1:
            wins_b += 1
        elif game.winner == 2:
            wins_w += 1
        else:
            unfinished += 1
    return {
        "n_games": n, "wins_black": wins_b, "wins_white": wins_w,
        "unfinished": unfinished, "wall_time": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--ckpt", type=str,
                        default=str(ROOT / "artifacts" / "checkpoints" / "az_pretrain.pt"))
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if args.quick:
        args.n = 10

    def _az():
        return make_az_agent(args.ckpt, temperature=args.temperature, seed=0)

    def _rand():
        return RandomAgent()

    def _combo_v2():
        return make_combo_v2_ca()

    print(f"Phase 1 policy-head eval:  n={args.n}  temp={args.temperature}")

    out = {}
    for name, blk, wht in [
        ("az_vs_random",  _az,       _rand),
        ("az_vs_combov2", _az,       _combo_v2),
        ("combov2_vs_az", _combo_v2, _az),
    ]:
        print(f"  [{name}] ", end="", flush=True)
        r = play_match(blk, wht, args.n)
        out[name] = r
        n_dec = r["wins_black"] + r["wins_white"]
        pB = r["wins_black"] / max(1, n_dec)
        print(f"B={r['wins_black']:3d} W={r['wins_white']:3d} unf={r['unfinished']:3d}  "
              f"pB={pB:.2f}  ({r['wall_time']:.1f}s)")

    RESULTS_JSON = ROOT / "results" / "az_policy_eval.json"
    FIG = ROOT / "figures" / "fig_az_policy_eval.png"

    out["_args"] = vars(args)
    out["_wall_time"] = sum(v.get("wall_time", 0) for k, v in out.items() if isinstance(v, dict) and "wall_time" in v)
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(out, indent=2))

    # bar chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    keys = ["az_vs_random", "az_vs_combov2", "combov2_vs_az"]
    b_vals = [out[k]["wins_black"] / out[k]["n_games"] for k in keys]
    w_vals = [out[k]["wins_white"] / out[k]["n_games"] for k in keys]
    u_vals = [out[k]["unfinished"] / out[k]["n_games"] for k in keys]
    import numpy as np
    x = np.arange(len(keys))
    ax.bar(x - 0.25, b_vals, width=0.25, label="Black wins", color="#4c78a8")
    ax.bar(x,        w_vals, width=0.25, label="White wins", color="#e45756")
    ax.bar(x + 0.25, u_vals, width=0.25, label="Unfinished", color="gray", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=10)
    ax.set_ylabel("share of games")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Phase 1 policy-head eval: pretrained UnifiedNet vs baselines (n={args.n})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, k in enumerate(keys):
        ax.text(i, 1.01, f"n={out[k]['n_games']}", ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    plt.close(fig)
    print(f"\nwrote {RESULTS_JSON}")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
