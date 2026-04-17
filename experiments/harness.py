"""
Parallel match harness for HexGo agents.

Runs matchups and round-robins in parallel using multiprocessing.Pool.
Top-level factories + top-level worker so mp can dispatch them.

Outputs:
  - MatchResults dataclass (wins / draws / unfinished / mean length / Wilson CI)
  - JSON save to results/
  - PNG figures: win-rate heatmap, length distribution

Usage:
  from experiments.harness import run_matchup, run_round_robin, default_registry
  r = run_matchup("random", "ca_combo", n_games=100, parallelism=8, seed=0)
  r.save("results/random_vs_combo.json")
  r.plot_winrate_heatmap("figures/rr_winrate.png")
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

# worktree shim — see CLAUDE.md > "Worktree gotcha"
_REAL_HEXGO = Path(r"C:\Users\Leon\Desktop\Psychograph\hexgo")
if _REAL_HEXGO.exists() and str(_REAL_HEXGO) not in sys.path:
    sys.path.insert(0, str(_REAL_HEXGO))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine import HexGame, EisensteinGreedyAgent, RandomAgent
from engine.agents import ForkAwareAgent, PotentialGradientAgent, ComboAgent, MirrorAgent
from engine.ca_policy import (
    make_greedy_ca,
    make_fork_aware_ca,
    make_potential_gradient_ca,
    make_combo_ca,
    make_combo_v2_ca,
)


# ── Agent registry ──────────────────────────────────────────────────────────
# All factories are module-level callables so mp workers can resolve them.

def _f_random() -> object:
    return RandomAgent()


def _f_greedy() -> object:
    return EisensteinGreedyAgent("greedy", defensive=True)


def _f_fork_aware() -> object:
    return ForkAwareAgent("fork_aware", alpha=2.0)


def _f_potential() -> object:
    return PotentialGradientAgent("potential")


def _f_combo() -> object:
    return ComboAgent("combo")


def _f_ca_greedy() -> object:
    return make_greedy_ca(defensive=True)


def _f_ca_fork_aware() -> object:
    return make_fork_aware_ca(alpha=2.0)


def _f_ca_potential() -> object:
    return make_potential_gradient_ca()


def _f_ca_combo() -> object:
    return make_combo_ca()


def _f_ca_combo_v2() -> object:
    return make_combo_v2_ca()


def _f_mirror() -> object:
    return MirrorAgent()


def _f_neural_ca() -> object:
    # Imported inside the factory so the mp workers pay the torch import cost
    # only once, not at module load of harness.py.
    from engine.neural_ca import NeuralCAAgent
    return NeuralCAAgent(name="neural_ca", seed=0)


def default_registry() -> dict[str, Callable[[], object]]:
    """Top-level factories keyed by short name."""
    return {
        "random":         _f_random,
        "greedy":         _f_greedy,
        "fork_aware":     _f_fork_aware,
        "potential":      _f_potential,
        "combo":          _f_combo,
        "ca_greedy":      _f_ca_greedy,
        "ca_fork_aware":  _f_ca_fork_aware,
        "ca_potential":   _f_ca_potential,
        "ca_combo":       _f_ca_combo,
        "ca_combo_v2":    _f_ca_combo_v2,
        "mirror":         _f_mirror,
        "neural_ca":      _f_neural_ca,
    }


# ── Worker function ─────────────────────────────────────────────────────────

def _play_one(args: tuple[str, str, int, int]) -> tuple[int, int]:
    """
    Run a single game. Returns (winner, move_count).

    winner: 0 = unfinished, 1 = black wins, 2 = white wins.
    """
    black_name, white_name, max_moves, seed = args
    random.seed(seed)
    reg = default_registry()
    black = reg[black_name]()
    white = reg[white_name]()

    g = HexGame()
    mc = 0
    while g.winner is None and mc < max_moves:
        agent = black if g.current_player == 1 else white
        legal = g.legal_moves()
        if not legal:
            break
        mv = agent.choose_move(g)
        # Accept any empty cell that HexGame.make() will accept, not just
        # the candidate frontier — MirrorAgent plays at -c which may be far.
        if mv in g.board:
            mv = random.choice(legal)
        if not g.make(*mv):
            mv = random.choice(legal)
            g.make(*mv)
        mc += 1
    return (g.winner or 0, mc)


# ── Wilson 95% confidence interval ──────────────────────────────────────────

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ── Result container ────────────────────────────────────────────────────────

@dataclass
class MatchResults:
    black: str
    white: str
    n_games: int
    max_moves: int
    wall_time: float
    wins_black: int = 0
    wins_white: int = 0
    unfinished: int = 0
    mean_length: float = 0.0
    lengths: list[int] = field(default_factory=list)

    def ci_black(self) -> tuple[float, float]:
        return _wilson(self.wins_black, self.n_games)

    def ci_white(self) -> tuple[float, float]:
        return _wilson(self.wins_white, self.n_games)

    def summary(self) -> str:
        lo_b, hi_b = self.ci_black()
        lo_w, hi_w = self.ci_white()
        return (
            f"{self.black:>14s} vs {self.white:<14s} | "
            f"B={self.wins_black:3d} [{lo_b:.2f},{hi_b:.2f}]  "
            f"W={self.wins_white:3d} [{lo_w:.2f},{hi_w:.2f}]  "
            f"unfin={self.unfinished:3d}  "
            f"<len>={self.mean_length:5.1f}  "
            f"({self.wall_time:.1f}s)"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci_black"] = self.ci_black()
        d["ci_white"] = self.ci_white()
        return d

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Entry points ────────────────────────────────────────────────────────────

def run_matchup(
    black: str,
    white: str,
    n_games: int = 50,
    parallelism: int = 4,
    seed: int = 0,
    max_moves: int = 200,
) -> MatchResults:
    """Run `n_games` between (black, white) in parallel. Returns MatchResults."""
    reg = default_registry()
    if black not in reg:
        raise KeyError(f"unknown agent: {black!r}")
    if white not in reg:
        raise KeyError(f"unknown agent: {white!r}")

    jobs = [(black, white, max_moves, seed + i) for i in range(n_games)]
    t0 = time.perf_counter()
    if parallelism <= 1:
        outcomes = [_play_one(j) for j in jobs]
    else:
        with mp.Pool(parallelism) as pool:
            outcomes = pool.map(_play_one, jobs)
    t1 = time.perf_counter()

    wins_b = sum(1 for w, _ in outcomes if w == 1)
    wins_w = sum(1 for w, _ in outcomes if w == 2)
    unfin = sum(1 for w, _ in outcomes if w == 0)
    lens = [n for _, n in outcomes]

    return MatchResults(
        black=black,
        white=white,
        n_games=n_games,
        max_moves=max_moves,
        wall_time=t1 - t0,
        wins_black=wins_b,
        wins_white=wins_w,
        unfinished=unfin,
        mean_length=sum(lens) / max(1, len(lens)),
        lengths=lens,
    )


def run_round_robin(
    agent_names: list[str],
    n_games: int = 30,
    parallelism: int = 4,
    seed: int = 0,
    max_moves: int = 200,
) -> dict[tuple[str, str], MatchResults]:
    """Run every ordered pair (black, white). Returns dict keyed by (black, white)."""
    out: dict[tuple[str, str], MatchResults] = {}
    for b in agent_names:
        for w in agent_names:
            r = run_matchup(b, w, n_games=n_games, parallelism=parallelism,
                            seed=seed, max_moves=max_moves)
            out[(b, w)] = r
            print("  " + r.summary())
    return out


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_winrate_heatmap(
    rr: dict[tuple[str, str], MatchResults],
    agent_names: list[str],
    path: str,
    title: str = "Black win-rate (rows=black, cols=white)",
) -> None:
    """Heatmap of Black win rate across a round-robin."""
    import numpy as np
    import matplotlib.pyplot as plt

    N = len(agent_names)
    mat = np.zeros((N, N))
    for i, b in enumerate(agent_names):
        for j, w in enumerate(agent_names):
            r = rr.get((b, w))
            if r is None or r.n_games == 0:
                mat[i, j] = float("nan")
            else:
                mat[i, j] = r.wins_black / r.n_games

    fig, ax = plt.subplots(figsize=(1.1 * N + 2, 1.0 * N + 1.5))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_yticklabels(agent_names)
    ax.set_xlabel("white")
    ax.set_ylabel("black")
    ax.set_title(title)
    for i in range(N):
        for j in range(N):
            v = mat[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if (v < 0.3 or v > 0.7) else "black",
                        fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def plot_length_distribution(
    results: list[MatchResults],
    path: str,
    title: str = "Game length distribution",
) -> None:
    """Overlayed histograms of move counts across several matchups."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        if not r.lengths:
            continue
        ax.hist(r.lengths, bins=30, alpha=0.45,
                label=f"{r.black} vs {r.white}")
    ax.set_xlabel("moves")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--black", default="random")
    ap.add_argument("--white", default="ca_combo")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--parallelism", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    r = run_matchup(args.black, args.white, n_games=args.n,
                    parallelism=args.parallelism, seed=args.seed,
                    max_moves=args.max_moves)
    print(r.summary())
    if args.save:
        r.save(args.save)
        print(f"[saved] {args.save}")


if __name__ == "__main__":
    _main()
