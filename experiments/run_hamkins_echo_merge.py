"""Merge the {30, 60, 120, 240, 480} Hamkins echo sweep with the h=960
extension and re-plot on a single set of axes.

At h=480 the decisive share for combo-vs-combo was already 39/50 = 0.78
with unfinished = 11/50. The Hamkins-style prediction ("longer horizon
-> more draws because two competent players can interdict forever") would
have the unfinished share *rising* with horizon. It's doing the opposite:
decisive share grows monotonically as the horizon grows. The h=960 cell
(40/50 decisive) confirms the trend doesn't flip at longer horizons.

Output:
    results/hamkins_echo_combined.json
    figures/fig_hamkins_echo_merged.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
BASE_JSON = HERE / "results" / "hamkins_echo.json"
EXT_JSON = HERE / "results" / "hamkins_echo_h960.json"
OUT_JSON = HERE / "results" / "hamkins_echo_combined.json"
OUT_FIG = HERE / "figures" / "fig_hamkins_echo_merged.png"

MATCHUPS = ["random_vs_combo", "greedy_vs_combo", "combo_vs_combo"]
COLOURS = {
    "random_vs_combo": "#f58518",
    "greedy_vs_combo": "#4c78a8",
    "combo_vs_combo": "#54a24b",
}


def main() -> None:
    base = json.loads(BASE_JSON.read_text())
    ext = json.loads(EXT_JSON.read_text())
    combined = {}
    for key in MATCHUPS:
        rows = list(base[key]["rows"]) + list(ext[key]["rows"])
        rows.sort(key=lambda r: r["horizon"])
        combined[key] = {
            "black": base[key]["black"],
            "white": base[key]["white"],
            "rows": rows,
        }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(combined, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # panel 1: decisive share vs horizon
    ax = axes[0]
    for key in MATCHUPS:
        rows = combined[key]["rows"]
        h = np.array([r["horizon"] for r in rows])
        dec = np.array([r["black_wins"] + r["white_wins"] for r in rows])
        n = np.array([r["n_games"] for r in rows])
        share = dec / n
        ax.plot(h, share, "o-", color=COLOURS[key], label=key, linewidth=2,
                markersize=8)
        for hi, si in zip(h, share):
            ax.text(hi, si + 0.02, f"{si:.2f}", ha="center", fontsize=7, color=COLOURS[key])
    ax.set_xscale("log")
    ax.set_xlabel("horizon  (max_moves, log scale)")
    ax.set_ylabel("decisive share  (B wins + W wins) / n_games")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Decisive share grows monotonically with horizon\n"
                 "(opposite of the Hamkins-echo prediction)")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.4)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right")

    # panel 2: mean game length vs horizon, with y=x reference
    ax = axes[1]
    for key in MATCHUPS:
        rows = combined[key]["rows"]
        h = np.array([r["horizon"] for r in rows])
        mean_len = np.array([r["mean_length"] for r in rows])
        ax.plot(h, mean_len, "o-", color=COLOURS[key], label=key, linewidth=2,
                markersize=8)
    hs = np.array([30, 60, 120, 240, 480, 960])
    ax.plot(hs, hs, "k--", alpha=0.4, label="y = horizon (ran out)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("horizon  (max_moves, log scale)")
    ax.set_ylabel("mean game length  (plies, log scale)")
    ax.set_title("Mean game length saturates below horizon\n"
                 "(games end decisively long before the cap)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left")

    fig.suptitle(
        "Hamkins echo on HexGo: draws do NOT dominate at long horizons"
        " (P5 supported at h=960, n=50/matchup)",
        fontsize=12,
    )
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
