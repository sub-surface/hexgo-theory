"""First-mover-advantage inversion panel.

Combines two independent self-play datasets on a single (strength, p_B)
axis:

    * our side — results/fma_curve.json, 5 rungs of the ladder {random,
      greedy, fork_aware, combo, ca_combo_v2}, n=200 games each on the
      infinite hex board with WIN_LENGTH=6 and the 1-2-2 rule.
    * charlie's side — tournament_results.pt diagonals {local_random,
      oracle_greedy, nca_greedy, lookahead, balanced_lookahead}, n=30
      games each on a 32x32 board.

Both sides agree on something that, at first glance, looks wrong: for
rollout-level agents (greedy, fork_aware, oracle_greedy, nca_greedy)
p_B collapses well below 0.5, inverting the strategy-stealing lower
bound. The inversion vanishes as soon as the agent does any lookahead
(combo, lookahead, balanced_lookahead, ca_combo_v2).

Strategy-stealing still holds in the limit - the bound is on _perfect_
play. What this panel shows is the size of the bounded-rationality gap:
short-horizon greedy policies are systematically disadvantaged when
they move first, because the extra stone they put down is a handle the
opponent's greedy reply latches onto. The gap closes once the policy
can see one ply ahead.

Output:
    results/fma_inversion_combined.json
    figures/fig_fma_inversion_panel.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parents[1]
# Charlie's artifacts live in the main checkout (not worktree-local).
MAIN_CHECKOUT = Path("C:/Users/Leon/Desktop/Psychograph/hexgo-theory")
OURS_JSON = HERE / "results" / "fma_curve.json"
CHARLIE_PT = (
    MAIN_CHECKOUT
    / "results"
    / "charlies-artifacts"
    / "checkpoints"
    / "tournament_results.pt"
)
OUT_JSON = HERE / "results" / "fma_inversion_combined.json"
OUT_FIG = HERE / "figures" / "fig_fma_inversion_panel.png"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (point, lo, hi). If n == 0, returns (nan, nan, nan).
    """
    if n == 0:
        nan = float("nan")
        return (nan, nan, nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


OUR_LADDER = [
    ("random", "random"),
    ("greedy", "greedy"),
    ("fork_aware", "fork"),
    ("combo", "combo"),
    ("ca_combo_v2", "combo_v2"),
]
CHARLIE_DIAGONALS = [
    ("local_random", "random*"),
    ("oracle_greedy", "greedy*"),
    ("nca_greedy", "nca_greedy*"),
    ("balanced_lookahead", "bal_look*"),
    ("lookahead", "lookahead*"),
]


def load_ours() -> list[dict]:
    d = json.loads(OURS_JSON.read_text())
    out = []
    for key, label in OUR_LADDER:
        row = d[key]
        wB, wW = row["wins_black"], row["wins_white"]
        n_dec = wB + wW
        p, lo, hi = wilson(wB, n_dec)
        out.append(
            {
                "source": "ours",
                "agent": key,
                "label": label,
                "n_games": row["n_games"],
                "wins_black": wB,
                "wins_white": wW,
                "unfinished": row["unfinished"],
                "n_decisive": n_dec,
                "decisive_rate": n_dec / row["n_games"],
                "p_black": p,
                "p_black_ci": [lo, hi],
            }
        )
    return out


def load_charlie() -> list[dict]:
    d = torch.load(CHARLIE_PT, weights_only=False)
    lookup = {}
    for r in d["results"]:
        name = r["name"]
        if " vs " in name:
            a, b = name.split(" vs ")
            if a == b:
                lookup[a] = r
    out = []
    for key, label in CHARLIE_DIAGONALS:
        r = lookup[key]
        wB, wW = r["p0_wins"], r["p1_wins"]
        n_dec = wB + wW
        p, lo, hi = wilson(wB, n_dec)
        out.append(
            {
                "source": "charlie",
                "agent": key,
                "label": label,
                "n_games": r["n_games"],
                "wins_black": wB,
                "wins_white": wW,
                "unfinished": r["draws"],  # charlie's "draws" = ply-cap games on 32x32
                "n_decisive": n_dec,
                "decisive_rate": n_dec / r["n_games"],
                "p_black": p,
                "p_black_ci": [lo, hi],
            }
        )
    return out


def main() -> None:
    ours = load_ours()
    charlie = load_charlie()

    # combined strength rank: we know the ordering by construction.
    # order reflects decision horizon, not raw ELO:
    #   random -> greedy -> nca_greedy -> fork_aware -> oracle_greedy
    #   -> combo -> ca_combo_v2 -> balanced_lookahead -> lookahead
    ordering = [
        ("ours", "random"),
        ("charlie", "local_random"),
        ("ours", "greedy"),
        ("charlie", "nca_greedy"),
        ("charlie", "oracle_greedy"),
        ("ours", "fork_aware"),
        ("ours", "combo"),
        ("ours", "ca_combo_v2"),
        ("charlie", "balanced_lookahead"),
        ("charlie", "lookahead"),
    ]
    by_key = {("ours", r["agent"]): r for r in ours}
    by_key.update({("charlie", r["agent"]): r for r in charlie})
    rows = [by_key[k] for k in ordering]

    combined = {
        "ours": ours,
        "charlie": charlie,
        "ordered": rows,
        "note": (
            "p_black computed over decisive games only; unfinished ="
            " reached max-ply cap. Wilson 95% CI. Charlie's data played"
            " on a 32x32 board (can wrap/clip), ours on the infinite"
            " hex lattice - the two axes are not directly calibrated,"
            " but the ordering within each dataset is."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(combined, indent=2))

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = list(range(len(rows)))
    labels = [r["label"] for r in rows]
    ps = [r["p_black"] for r in rows]
    los = [r["p_black_ci"][0] for r in rows]
    his = [r["p_black_ci"][1] for r in rows]
    colours = [
        "#4c78a8" if r["source"] == "ours" else "#f58518" for r in rows
    ]

    for i, r in enumerate(rows):
        if math.isnan(r["p_black"]):
            ax.scatter([i], [0.5], marker="x", color="gray", s=80, zorder=3)
            ax.text(i, 0.5 + 0.03, "n_dec=0", ha="center", fontsize=8, color="gray")
            continue
        p = r["p_black"]
        lo = r["p_black_ci"][0]
        hi = r["p_black_ci"][1]
        ax.errorbar(
            [i],
            [p],
            yerr=[[p - lo], [hi - p]],
            fmt="o",
            color=colours[i],
            ecolor=colours[i],
            capsize=4,
            markersize=7,
            zorder=3,
        )
        ax.text(
            i,
            hi + 0.02,
            f"{p:.2f}\n(n={r['n_decisive']})",
            ha="center",
            fontsize=8,
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(
        len(rows) - 0.4,
        0.51,
        "strategy-stealing bound (p_B >= 0.5)",
        fontsize=8,
        ha="right",
        style="italic",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("p_black  (wins_B / decisive games)")
    ax.set_xlabel("agent (ordered by decision horizon)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "First-mover-advantage inversion across two independent self-play"
        " datasets\nbounded-rationality gap: greedy policies *lose* when"
        " they move first; lookahead restores the bound"
    )

    # legend
    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8",
               markersize=8, label="ours (infinite lattice, n=200)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f58518",
               markersize=8, label="Charlie (32x32, n=30)"),
    ]
    ax.legend(handles=legend, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_JSON}")

    # console summary
    print("\n  agent                  source    n_dec   p_B   [95% CI]")
    for r in rows:
        if math.isnan(r["p_black"]):
            print(f"  {r['label']:22s} {r['source']:8s}   {r['n_decisive']:3d}     -     (all unfinished)")
        else:
            lo, hi = r["p_black_ci"]
            print(
                f"  {r['label']:22s} {r['source']:8s}   {r['n_decisive']:3d}   "
                f"{r['p_black']:.2f}  [{lo:.2f}, {hi:.2f}]"
            )


if __name__ == "__main__":
    main()
