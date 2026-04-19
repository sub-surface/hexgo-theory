"""Cross-program tournament table — ours vs Charlie's.

We can't directly play our agents against Charlie's (incompatible board
substrates: 32x32 bounded vs infinite $\\mathbb{Z}[\\omega]$), but we can
put the two tournament matrices side by side. The figure makes three
things legible:

1. **Ordering consistency.** Both programs agree that lookahead agents
   dominate greedy agents, and greedy agents dominate random.
2. **FMA inversion is everywhere.** Diagonal cells on Charlie's
   greedy-level agents are dark (Black loses); ours are shaded grey
   because "unfinished at max-ply" dominates.
3. **What we don't have.** Our ladder is missing an MCTS rung. This is
   the gap the unified agent is meant to fill.

Inputs:
    results/fma_curve.json                                 -- our diagonals
    results/mirror_agent.json                              -- mirror cross
    results/combo_defect.json                              -- combo cross
    results/neural_ca_benchmark.json                       -- neural_ca cross
    results/nca_zoo_tournament.json                        -- nca zoo cross
    ../../results/charlies-artifacts/.../tournament_results.pt

Output:
    results/cross_program_table.json
    figures/fig_cross_program_table.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parents[1]
MAIN = Path("C:/Users/Leon/Desktop/Psychograph/hexgo-theory")
CHARLIE_PT = MAIN / "results" / "charlies-artifacts" / "checkpoints" / "tournament_results.pt"
OUT_JSON = HERE / "results" / "cross_program_table.json"
OUT_FIG = HERE / "figures" / "fig_cross_program_table.png"


def _load_ours() -> dict[tuple[str, str], dict]:
    """Build a sparse (black, white) -> result dict from our json files."""
    out: dict[tuple[str, str], dict] = {}

    def add(b: str, w: str, r: dict) -> None:
        out[(b, w)] = {
            "wins_black": r["wins_black"],
            "wins_white": r["wins_white"],
            "unfinished": r["unfinished"],
            "n_games": r["n_games"],
        }

    # fma_curve: diagonals
    fma = json.loads((HERE / "results" / "fma_curve.json").read_text())
    for agent in ["random", "greedy", "fork_aware", "combo", "ca_combo_v2"]:
        add(agent, agent, fma[agent])

    # mirror_agent: random/mirror, combo_v2/mirror cross
    mirror = json.loads((HERE / "results" / "mirror_agent.json").read_text())
    for key, val in mirror.items():
        if "__" not in key:
            continue
        b, w = key.split("__vs__")
        add(b, w, val)

    # combo_defect: combo, combo_v2 cross
    defect = json.loads((HERE / "results" / "combo_defect.json").read_text())
    for key, val in defect.items():
        if "__" not in key:
            continue
        b, w = key.split("__vs__")
        add(b, w, val)

    # neural_ca_benchmark: neural_ca + random/combo_v2 cross
    neural = json.loads((HERE / "results" / "neural_ca_benchmark.json").read_text())
    for key, val in neural.items():
        if "__" not in key:
            continue
        b, w = key.split("__vs__")
        add(b, w, val)

    return out


OUR_AGENTS = ["random", "greedy", "fork_aware", "combo", "ca_combo_v2", "mirror", "neural_ca"]


def _load_charlie() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (agent list, p_black matrix, n_games matrix).

    p_black = wins_black / max(1, wins_black + wins_white). Unfinished
    shown as NaN handling downstream via n_dec.
    """
    d = torch.load(CHARLIE_PT, weights_only=False)
    agent_set: list[str] = []
    for r in d["results"]:
        if " vs " not in r["name"]:
            continue
        a, b = r["name"].split(" vs ")
        for n in (a, b):
            if n not in agent_set:
                agent_set.append(n)
    # fixed order: random, greedy, nca_greedy, lookahead, balanced_look, lookahead_oracle
    prefer = ["local_random", "oracle_greedy", "nca_greedy", "lookahead", "balanced_lookahead", "lookahead_oracle"]
    agents = [a for a in prefer if a in agent_set]
    idx = {a: i for i, a in enumerate(agents)}
    n = len(agents)
    p = np.full((n, n), np.nan)
    ndec = np.full((n, n), np.nan)
    for r in d["results"]:
        if " vs " not in r["name"]:
            continue
        a, b = r["name"].split(" vs ")
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        wB, wW = r["p0_wins"], r["p1_wins"]
        dec = wB + wW
        if dec > 0:
            p[i, j] = wB / dec
            ndec[i, j] = dec
    return agents, p, ndec


def _build_our_matrix(data: dict[tuple[str, str], dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    n = len(OUR_AGENTS)
    p = np.full((n, n), np.nan)
    ndec = np.full((n, n), np.nan)
    unfinished_frac = np.full((n, n), np.nan)
    idx = {a: i for i, a in enumerate(OUR_AGENTS)}
    for (b, w), r in data.items():
        if b not in idx or w not in idx:
            continue
        i, j = idx[b], idx[w]
        dec = r["wins_black"] + r["wins_white"]
        if dec > 0:
            p[i, j] = r["wins_black"] / dec
            ndec[i, j] = dec
        unfinished_frac[i, j] = r["unfinished"] / max(1, r["n_games"])
    return OUR_AGENTS, p, ndec, unfinished_frac


def _cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "pB",
        [
            (0.0, "#d62728"),   # red: Black loses
            (0.5, "#f7f7f7"),   # neutral
            (1.0, "#1f77b4"),   # blue: Black wins
        ],
    )


def _plot_matrix(ax, agents, p, ndec, title, unfinished=None) -> None:
    n = len(agents)
    ax.imshow(p, cmap=_cmap(), vmin=0, vmax=1, aspect="equal")
    for i in range(n):
        for j in range(n):
            val = p[i, j]
            if np.isnan(val):
                if unfinished is not None and not np.isnan(unfinished[i, j]):
                    ax.text(j, i, f"{int(unfinished[i,j]*100)}%u", ha="center", va="center",
                            fontsize=6, color="gray")
                else:
                    ax.text(j, i, "-", ha="center", va="center", color="gray")
            else:
                nd = int(ndec[i, j]) if not np.isnan(ndec[i, j]) else 0
                colour = "white" if (val < 0.25 or val > 0.75) else "black"
                ax.text(j, i, f"{val:.2f}\n(n={nd})", ha="center", va="center",
                        fontsize=7, color=colour)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([a.replace("_", "\n") for a in agents], rotation=0, fontsize=8)
    ax.set_yticklabels(agents, fontsize=8)
    ax.set_xlabel("White")
    ax.set_ylabel("Black")
    ax.set_title(title)


def main() -> None:
    ours = _load_ours()
    our_agents, our_p, our_ndec, our_unf = _build_our_matrix(ours)
    charlie_agents, charlie_p, charlie_ndec = _load_charlie()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    _plot_matrix(
        axes[0],
        our_agents,
        our_p,
        our_ndec,
        f"Ours (infinite lattice, n/cell varies)\np_black = wins_B / decisive games",
        unfinished=our_unf,
    )
    _plot_matrix(
        axes[1],
        charlie_agents,
        charlie_p,
        charlie_ndec,
        f"Charlie (32x32, n=30/cell)\np_black = wins_B / decisive games",
    )

    # shared colour bar
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    sm = plt.cm.ScalarMappable(cmap=_cmap(), norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, cax=cax, label="p_black")

    fig.suptitle(
        "Cross-program tournament matrices: ordering consistent, FMA inversion shared",
        fontsize=12,
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # dump json
    payload = {
        "ours": {
            "agents": our_agents,
            "p_black": our_p.tolist(),
            "n_decisive": our_ndec.tolist(),
            "unfinished_frac": our_unf.tolist(),
        },
        "charlie": {
            "agents": charlie_agents,
            "p_black": charlie_p.tolist(),
            "n_decisive": charlie_ndec.tolist(),
        },
        "note": (
            "Ours: infinite hex lattice, 1-2-2, win=6, matchups harvested"
            " from results/{fma_curve,mirror_agent,combo_defect,neural_ca_benchmark}.json."
            " Charlie: 32x32 board, parsed from tournament_results.pt."
            " p_black computed over decisive games only. NaN cells have"
            " n_decisive=0 (all games unfinished)."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
