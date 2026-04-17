r"""
Neural-CA zoo tournament — CA-prior ablation study.

Five hex-conv NCA variants differ only in their first-layer prior (synthesis
note §7), with all higher layers randomly initialised from the same seed.
This is the **untrained-prior ablation**: measures what the inductive bias
alone buys before any policy-gradient training. The full training loop lands
in a follow-up; this experiment is the "prior is enough, before training"
baseline + the round-robin infrastructure.

Variants (from `engine.neural_ca.make_nca_variant`):
  1. nca_random             — control, no prior
  2. nca_d6_tied            — layer-0 D_6 symmetrisation
  3. nca_line_detector      — axis-pair own-stone detectors
  4. nca_erdos_selfridge    — threat-potential initialiser
  5. nca_combo              — d6 + line + erdos stacked

Plus two reference points:
  6. random                 — baseline floor
  7. ca_combo_v2            — strongest hand-crafted agent

Output is a 7x7 win-rate matrix (row = Black, col = White). The zoo champion
is the NCA variant with the highest average win-rate against the other four
NCA variants. The champion's win-rate vs ca_combo_v2 tells us whether
untrained-prior NCAs already compete with the hand-crafted ladder.

CUDA forces parallelism=1 — each NCA instance holds weights on GPU, and
mp workers would each fork a CUDA context. Wall time at n=10 per pair,
horizon=120, with ~40s/game is on the order of 12 hours. Use --quick for
dev iteration.

Outputs:
  results/nca_zoo_tournament.json
  figures/fig_nca_zoo_winrates.png
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


NCA_VARIANTS = [
    "nca_random",
    "nca_d6_tied",
    "nca_line_detector",
    "nca_erdos_selfridge",
    "nca_combo",
]
REFERENCE = ["random", "ca_combo_v2"]
ALL_AGENTS = NCA_VARIANTS + REFERENCE


def _run(n_games: int, max_moves: int, seed: int, include_reference: bool) -> dict:
    agents = ALL_AGENTS if include_reference else NCA_VARIANTS
    print(f"\n── NCA zoo tournament ──  agents={agents}  "
          f"n={n_games}  horizon={max_moves}  parallelism=1 (CUDA)\n")

    out: dict = {}
    t0 = time.perf_counter()
    pair_idx = 0
    for b in agents:
        for w in agents:
            # Allow b == w self-play — it's the FMA diagonal for each variant.
            r = run_matchup(b, w, n_games=n_games, parallelism=1,
                            seed=seed + pair_idx * 100_000,
                            max_moves=max_moves)
            print("  " + r.summary())
            out[f"{b}__vs__{w}"] = r.to_dict()
            pair_idx += 1
    out["_wall_time"] = time.perf_counter() - t0
    out["_params"] = dict(n_games=n_games, max_moves=max_moves, seed=seed,
                          agents=agents)
    return out


def _winrate(r: dict) -> tuple[float, float, float]:
    """(Black win rate, decisive rate, wilson_low on black-win-fraction)."""
    n = r["n_games"]
    if n == 0:
        return (0.5, 0.0, 0.0)
    b_rate = r["wins_black"] / n
    d_rate = (r["wins_black"] + r["wins_white"]) / n
    lo, _ = _wilson(r["wins_black"], n)
    return (b_rate, d_rate, lo)


def _champion(results: dict, agents: list[str]) -> tuple[str, dict]:
    """Identify the NCA zoo champion by average win rate across the 5 NCAs
    when it plays any seat (Black or White)."""
    nca = [a for a in agents if a.startswith("nca_")]
    score = {}
    for me in nca:
        wins_as_black = 0
        wins_as_white = 0
        n_games = 0
        for opp in nca:
            if opp == me:
                continue
            rb = results[f"{me}__vs__{opp}"]
            rw = results[f"{opp}__vs__{me}"]
            wins_as_black += rb["wins_black"]
            wins_as_white += rw["wins_white"]
            n_games += rb["n_games"] + rw["n_games"]
        score[me] = {
            "wins": wins_as_black + wins_as_white,
            "n_games": n_games,
            "rate": (wins_as_black + wins_as_white) / max(1, n_games),
        }
    champion = max(score, key=lambda k: score[k]["rate"])
    return champion, score


def _plot_matrix(results: dict, agents: list[str], path: str) -> None:
    N = len(agents)
    mat = np.full((N, N), float("nan"))
    for i, b in enumerate(agents):
        for j, w in enumerate(agents):
            r = results.get(f"{b}__vs__{w}")
            if r is None or r["n_games"] == 0:
                continue
            mat[i, j] = r["wins_black"] / r["n_games"]

    fig, ax = plt.subplots(figsize=(1.1 * N + 2, 1.0 * N + 1.5))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdBu_r")
    short = [a.replace("nca_", "") for a in agents]
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    ax.set_xlabel("white")
    ax.set_ylabel("black")
    ax.set_title("NCA zoo: Black win rate (rows=black, cols=white)")
    for i in range(N):
        for j in range(N):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if (v < 0.3 or v > 0.7) else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def _verdict(results: dict, agents: list[str]) -> str:
    champion, scores = _champion(results, agents)
    lines = [f"Zoo champion: {champion}"]
    lines.append(f"  avg win rate across other NCAs: "
                 f"{scores[champion]['rate']:.2f}  "
                 f"({scores[champion]['wins']}/{scores[champion]['n_games']})")

    lines.append("\nAll-NCA ranking (by avg win rate vs other NCAs):")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]["rate"]):
        lines.append(f"  {k:>22s}  rate={v['rate']:.2f}  "
                     f"({v['wins']}/{v['n_games']})")

    # Champion vs ca_combo_v2 (if reference included).
    if "ca_combo_v2" in agents:
        b = results.get(f"{champion}__vs__ca_combo_v2")
        w = results.get(f"ca_combo_v2__vs__{champion}")
        if b and w:
            nca_wins = b["wins_black"] + w["wins_white"]
            total = b["n_games"] + w["n_games"]
            lines.append(f"\nChampion vs ca_combo_v2: "
                         f"champion won {nca_wins}/{total}  "
                         f"({nca_wins/max(1,total):.2f})")

    # Synthesis §7 prediction check: d6_tied should rank >= line_detector and
    # erdos_selfridge individually, and combo ≈ d6_tied within noise.
    if all(a in scores for a in ["nca_d6_tied", "nca_line_detector",
                                  "nca_erdos_selfridge", "nca_combo"]):
        d6 = scores["nca_d6_tied"]["rate"]
        ld = scores["nca_line_detector"]["rate"]
        es = scores["nca_erdos_selfridge"]["rate"]
        co = scores["nca_combo"]["rate"]
        lines.append("\nSynthesis §7 prediction check "
                     "(D_6 load-bearing, tactical priors redundant):")
        if d6 >= ld - 0.05 and d6 >= es - 0.05 and abs(co - d6) < 0.10:
            lines.append("  → SUPPORTED: d6_tied competes with or beats "
                         "tactical priors; combo ≈ d6.")
        else:
            lines.append("  → NOT supported at this corpus size / training level. "
                         f"d6={d6:.2f}, line={ld:.2f}, "
                         f"erdos={es:.2f}, combo={co:.2f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="games per matchup (CUDA parallelism=1; keep small)")
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260418)
    ap.add_argument("--quick", action="store_true",
                    help="n=4 games per pair, NCA-only (skip reference agents)")
    ap.add_argument("--nca-only", action="store_true",
                    help="run only the 5x5 NCA round-robin (no reference)")
    args = ap.parse_args()

    include_reference = not (args.quick or args.nca_only)
    if args.quick:
        args.n = 4
        args.max_moves = 80

    results = _run(args.n, args.max_moves, args.seed, include_reference)

    rpath = Path("results") / "nca_zoo_tournament.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {rpath}")

    agents = ALL_AGENTS if include_reference else NCA_VARIANTS
    fig = Path("figures") / "fig_nca_zoo_winrates.png"
    _plot_matrix(results, agents, str(fig))
    print(f"[saved] {fig}")

    print("\n── Verdict ──\n" + _verdict(results, agents))


if __name__ == "__main__":
    main()
