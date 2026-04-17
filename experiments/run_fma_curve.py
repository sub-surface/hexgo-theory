r"""
First-mover-advantage (FMA) curve across the agent ladder.

Question: does the Black (first-mover) advantage grow, saturate, or invert
as agent strength increases? The strategy-stealing argument gives a
*lower* bound $p_B \geq 0.5$ on Black's share of decisive games in
self-play, but says nothing about the *magnitude* of the advantage. A
naive expectation ("stronger play amplifies tempo") predicts $p_B$
rising along the ladder. Combo-v1 *violated* the lower bound before
the v2 opening-centre-bias fix — so we treat $p_B$ as genuinely
uncertain per-agent.

Design: for each agent A on the ladder
  A ∈ [random, greedy, fork_aware, combo, ca_combo_v2]
run n self-play games of A-B vs A-W, and report
  p_B(A)     = wins_black / (wins_black + wins_white)         [decisive only]
  d(A)       = (wins_black + wins_white) / n_games            [decisive rate]
  ± Wilson 95% CI on p_B.

This answers two sub-questions:

  (Q1) Is strategy-stealing respected pointwise?
       → p_B(A) ≥ 0.5 (lower Wilson bound) for every A.
       → falsified by any agent with hi-Wilson < 0.5.
  (Q2) Does FMA scale with strength?
       → p_B(A) monotone in A's ladder position.
       → falsified by non-monotone pattern (interesting either way).

Outputs:
  results/fma_curve.json
  figures/fig_fma_curve.png
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


# Ordered by expected strength (weakest → strongest). The ladder intentionally
# excludes neural_ca — it forces parallelism=1 (CUDA), which would dominate the
# wall-time budget with no added signal: the untrained NCA is already known to
# beat Random ~60–70% and lose all games vs Combo-v2.
LADDER = [
    "random",
    "greedy",
    "fork_aware",
    "combo",
    "ca_combo_v2",
]


def _run(n_games: int, parallelism: int, max_moves: int, seed: int) -> dict:
    print(f"\n── FMA curve ──  ladder={LADDER}  n={n_games}  "
          f"parallel={parallelism}  horizon={max_moves}\n")
    out = {}
    t0 = time.perf_counter()
    for idx, agent in enumerate(LADDER):
        # Per-agent seed offset: otherwise low-entropy agents (greedy with
        # tiebreakers) share trajectories across matchups — see the lesson
        # from run_combo_defect.py.
        r = run_matchup(agent, agent, n_games=n_games, parallelism=parallelism,
                        seed=seed + idx * 100_000, max_moves=max_moves)
        print("  " + r.summary())
        out[agent] = r.to_dict()
    out["_wall_time"] = time.perf_counter() - t0
    out["_params"] = dict(n_games=n_games, parallelism=parallelism,
                          max_moves=max_moves, seed=seed,
                          ladder=LADDER)
    return out


def _black_share(r: dict) -> tuple[float, float, float, int]:
    """(p_B, lo, hi, decisive_n) with Wilson 95% CI over decisive games only."""
    b = r["wins_black"]
    w = r["wins_white"]
    dec = b + w
    if dec == 0:
        return (0.5, 0.0, 1.0, 0)
    lo, hi = _wilson(b, dec)
    return (b / dec, lo, hi, dec)


def _decisive_rate(r: dict) -> float:
    return (r["wins_black"] + r["wins_white"]) / max(1, r["n_games"])


def _verdict(results: dict) -> str:
    lines = ["p_B = Black share of decisive games; ± Wilson 95 % CI; "
             "d = decisive fraction.\n"]
    lines.append(f"{'agent':>14s}  {'p_B':>5s}  {'lo':>5s}  {'hi':>5s}  "
                 f"{'d':>5s}  {'n_dec':>5s}")
    q1_violations = []
    share_curve = []
    for agent in LADDER:
        r = results[agent]
        pB, lo, hi, dec = _black_share(r)
        d = _decisive_rate(r)
        lines.append(f"{agent:>14s}  {pB:5.2f}  {lo:5.2f}  {hi:5.2f}  "
                     f"{d:5.2f}  {dec:5d}")
        share_curve.append(pB)
        if hi < 0.5:
            q1_violations.append(agent)

    lines.append("")
    if not q1_violations:
        lines.append("(Q1) Strategy-stealing respected pointwise: "
                     "every agent's upper Wilson ≥ 0.5.")
    else:
        lines.append(f"(Q1) Strategy-stealing VIOLATED by: "
                     f"{', '.join(q1_violations)} (upper Wilson < 0.5).")

    # Monotonicity check across non-random ladder (random self-play is noise-pinned at 0.5).
    non_random = share_curve[1:]
    is_monotone_up = all(non_random[i + 1] >= non_random[i] - 0.05
                         for i in range(len(non_random) - 1))
    is_monotone_down = all(non_random[i + 1] <= non_random[i] + 0.05
                           for i in range(len(non_random) - 1))
    if is_monotone_up and not is_monotone_down:
        lines.append("(Q2) p_B weakly monotone ↑ in strength — FMA grows.")
    elif is_monotone_down and not is_monotone_up:
        lines.append("(Q2) p_B weakly monotone ↓ in strength — FMA shrinks.")
    else:
        lines.append("(Q2) p_B non-monotone across ladder — "
                     "FMA is not a simple function of strength.")
    return "\n".join(lines)


def _plot(results: dict, path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    pBs = []
    los = []
    his = []
    ds = []
    for agent in LADDER:
        pB, lo, hi, _ = _black_share(results[agent])
        pBs.append(pB)
        los.append(lo)
        his.append(hi)
        ds.append(_decisive_rate(results[agent]))

    x = np.arange(len(LADDER))
    # Left panel: p_B with CI error bars.
    yerr_low = [pB - lo for pB, lo in zip(pBs, los)]
    yerr_high = [hi - pB for pB, hi in zip(pBs, his)]
    ax1.errorbar(x, pBs, yerr=[yerr_low, yerr_high],
                 fmt="o-", color="#222", ecolor="#888",
                 capsize=4, markersize=7, linewidth=1.5)
    ax1.axhline(0.5, color="#a22", linestyle="--", linewidth=1,
                label="strategy-stealing lower bound")
    for xi, pB in zip(x, pBs):
        ax1.text(xi, pB + 0.03, f"{pB:.2f}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LADDER, rotation=30, ha="right")
    ax1.set_ylabel(r"$p_B$ = Black share of decisive games")
    ax1.set_title("First-mover advantage vs agent strength")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # Right panel: decisive-game rate (how often play terminates before horizon).
    ax2.bar(x, ds, color="#36a")
    for xi, d in zip(x, ds):
        ax2.text(xi, d + 0.02, f"{d:.2f}", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LADDER, rotation=30, ha="right")
    ax2.set_ylabel("decisive-game fraction")
    ax2.set_title("Decisiveness vs strength\n(complements the Hamkins-echo result)")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--parallelism", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max-moves", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20260417)
    ap.add_argument("--quick", action="store_true",
                    help="small n, short horizon for fast iteration")
    args = ap.parse_args()

    if args.quick:
        args.n = 20
        args.max_moves = 120

    results = _run(args.n, args.parallelism, args.max_moves, args.seed)

    rpath = Path("results") / "fma_curve.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {rpath}")

    fig = Path("figures") / "fig_fma_curve.png"
    _plot(results, str(fig))
    print(f"[saved] {fig}")

    print("\n── Verdict ──\n" + _verdict(results))


if __name__ == "__main__":
    main()
