r"""
Train each of the five NCA prior variants via REINFORCE self-play.

For each prior in {random, d6_tied, line_detector, erdos_selfridge, combo}:
  - instantiate a fresh NeuralCAAgent with that prior
  - train via self-play against a frozen copy of itself
    (engine.neural_ca.train_self_play)
  - save checkpoint to checkpoints/nca_<prior>.pt
  - save training curve to results/nca_train_<prior>.json

Runtime budget: --quick does 60 games/variant (~3 min each on 2060),
full default is 400 games/variant (~20 min each, ~1.7h total).

This intentionally trains each variant in isolation (no cross-prior
play) — we want to measure what each prior enables, not whether one
prior can be bullied into the same answer as another. Cross-prior
evaluation is the tournament in run_nca_zoo.py.

Outputs:
  checkpoints/nca_<prior>.pt           — trained model state_dicts
  results/nca_train_<prior>.json       — per-game reward / loss trajectory
  figures/fig_nca_train_curves.png     — overlay of learning curves
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
import torch

# worktree shim — see CLAUDE.md > "Worktree gotcha"
_REAL_HEXGO = Path(r"C:\Users\Leon\Desktop\Psychograph\hexgo")
if _REAL_HEXGO.exists() and str(_REAL_HEXGO) not in sys.path:
    sys.path.insert(0, str(_REAL_HEXGO))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine.neural_ca import make_nca_variant, train_self_play


PRIORS = ["random", "d6_tied", "line_detector", "erdos_selfridge", "combo"]


def _teacher_factory():
    """ca_combo_v2 — known to win vs untrained NCAs, provides dense signal."""
    from engine.ca_policy import make_combo_v2_ca
    return make_combo_v2_ca()


def _moving_avg(x: list[float], window: int) -> list[float]:
    if len(x) < window:
        return [float(np.mean(x[:i + 1])) for i in range(len(x))]
    out = []
    s = sum(x[:window])
    out.extend([s / window] * window)
    for i in range(window, len(x)):
        s += x[i] - x[i - window]
        out.append(s / window)
    return out


def _run(total_games: int, max_moves: int, step_every: int,
          refresh_every: int, temperature: float, lr: float,
          teacher_phase: int, seed: int) -> dict:
    out: dict = {}
    for prior in PRIORS:
        print(f"\n── training {prior} "
              f"({total_games} games, step_every={step_every}, "
              f"teacher_phase={teacher_phase}) ──")
        agent = make_nca_variant(prior, seed=seed)
        ckpt = Path("checkpoints") / f"nca_{prior}.pt"
        t0 = time.perf_counter()
        history = train_self_play(
            agent,
            total_games=total_games,
            step_every=step_every,
            refresh_opponent_every=refresh_every,
            temperature=temperature,
            learning_rate=lr,
            max_moves=max_moves,
            seed=seed + hash(prior) % 100_000,
            log_every=max(8, total_games // 20),
            checkpoint_path=str(ckpt),
            teacher_factory=_teacher_factory,
            teacher_phase_games=teacher_phase,
        )
        history["_wall_time"] = time.perf_counter() - t0
        history["_prior"] = prior
        out[prior] = history

        rpath = Path("results") / f"nca_train_{prior}.json"
        rpath.parent.mkdir(parents=True, exist_ok=True)
        with open(rpath, "w") as f:
            # History["reward"] and ["decisive"] can be long; dump directly.
            json.dump(history, f, indent=2)
        print(f"  [saved] {ckpt}  {rpath}")

    return out


def _plot(out: dict, path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"random": "#777", "d6_tied": "#36a", "line_detector": "#c83",
              "erdos_selfridge": "#2a7", "combo": "#a22"}

    for prior, hist in out.items():
        rewards = hist["reward"]
        smooth = _moving_avg(rewards, max(4, len(rewards) // 10))
        ax1.plot(range(len(smooth)), smooth, label=prior,
                 color=colors.get(prior, "black"), linewidth=1.5)
        losses = hist.get("loss", [])
        if losses:
            ax2.plot(range(len(losses)), losses, label=prior,
                     color=colors.get(prior, "black"), linewidth=1.5,
                     alpha=0.8)

    ax1.set_xlabel("game index")
    ax1.set_ylabel(r"moving-avg reward from trainee's perspective")
    ax1.set_title("Self-play training curves\n(smoothed over 10% of total)")
    ax1.axhline(0.0, color="#888", linestyle=":", linewidth=1)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    ax2.set_xlabel("optimiser step")
    ax2.set_ylabel("REINFORCE loss")
    ax2.set_title("Loss trajectory")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-games", type=int, default=400)
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--step-every", type=int, default=8)
    ap.add_argument("--refresh-every", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--teacher-phase", type=int, default=200,
                    help="first N games play vs ca_combo_v2 for dense signal")
    ap.add_argument("--seed", type=int, default=20260418)
    ap.add_argument("--quick", action="store_true",
                    help="60 games/variant for dev iteration")
    args = ap.parse_args()

    if args.quick:
        args.total_games = 60
        args.max_moves = 100
        args.teacher_phase = 60  # all teacher for quick mode

    out = _run(
        total_games=args.total_games,
        max_moves=args.max_moves,
        step_every=args.step_every,
        refresh_every=args.refresh_every,
        temperature=args.temperature,
        lr=args.lr,
        teacher_phase=args.teacher_phase,
        seed=args.seed,
    )

    fig = Path("figures") / "fig_nca_train_curves.png"
    _plot(out, str(fig))
    print(f"\n[saved] {fig}")

    print("\n── Training summary ──")
    for prior, hist in out.items():
        r = hist["reward"]
        tail = r[-max(10, len(r) // 5):]
        dec = sum(hist["decisive"][-max(10, len(r) // 5):]) / max(
            1, len(r) // 5)
        print(f"  {prior:>16s}  final mean_R={sum(tail)/len(tail):+.2f}  "
              f"final_decisive={dec:.2f}  "
              f"wall={hist['_wall_time']:.1f}s")


if __name__ == "__main__":
    main()
