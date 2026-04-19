"""Diffraction Bragg99 histogram — existing data, new cut.

results/diffraction_p4.json has n=18 self-play games + n=18 matched
random controls at horizon=200. The original figure for P4 was a mean
+- CI bar chart. This script instead plots the full per-game
distributions, which is the presentation format that actually makes
separation between self-play and random-control legible to a reader.

The "Bragg99" statistic (see [engine/diffraction.py](../../engine/diffraction.py))
is the top-percentile peak intensity in the Fourier transform of the
occupied-cell indicator. Large Bragg99 = sharp diffraction peaks =
quasi-periodic order. The prediction (P4 in the synthesis note) is
that self-play corpora show substantially larger Bragg99 than random
controls. We already have the effect in the JSON; this figure makes the
separation visible.

Output:
    figures/fig_diffraction_bragg_histogram.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
IN_JSON = HERE / "results" / "diffraction_p4.json"
OUT_FIG = HERE / "figures" / "fig_diffraction_bragg_histogram.png"


def wilson_from_means(x: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a small sample via t/normal approx."""
    arr = np.array(x, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"),) * 3
    m = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, m - z * se, m + z * se


def main() -> None:
    d = json.loads(IN_JSON.read_text())
    sp = np.array(d["bragg_sp"])
    rand = np.array(d["bragg_rand"])
    hex_ctrl = d["bragg_hex_control"]
    n_per_game = np.array(d["N_per_game"])

    # 2-panel: left = histogram, right = per-game scatter ordered by N
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # panel 1: histogram
    ax = axes[0]
    bins = np.linspace(0, max(sp.max(), rand.max(), hex_ctrl) * 1.05, 20)
    ax.hist(rand, bins=bins, alpha=0.55, color="#f58518",
            label=f"random control (n={len(rand)})", edgecolor="black", linewidth=0.3)
    ax.hist(sp, bins=bins, alpha=0.55, color="#4c78a8",
            label=f"ca_combo_v2 self-play (n={len(sp)})", edgecolor="black", linewidth=0.3)
    ax.axvline(hex_ctrl, color="#2ca02c", linestyle="--", linewidth=2,
               label=f"hex-lattice control (Bragg99={hex_ctrl:.2f})")
    sp_m, sp_lo, sp_hi = wilson_from_means(list(sp))
    r_m, r_lo, r_hi = wilson_from_means(list(rand))
    ax.axvline(sp_m, color="#4c78a8", linestyle="-", linewidth=1, alpha=0.8)
    ax.axvline(r_m, color="#f58518", linestyle="-", linewidth=1, alpha=0.8)
    ax.set_xlabel("Bragg99 (99th-pctile FFT peak intensity)")
    ax.set_ylabel("number of games")
    ax.set_title(f"Bragg99 distribution\nself-play {sp_m:.2f} [{sp_lo:.2f},{sp_hi:.2f}]"
                 f"  vs  random {r_m:.3f} [{r_lo:.3f},{r_hi:.3f}]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # panel 2: scatter against stone count (N)
    ax2 = axes[1]
    ax2.scatter(n_per_game, rand, alpha=0.7, color="#f58518",
                label="random control", marker="s", s=40)
    ax2.scatter(n_per_game, sp, alpha=0.7, color="#4c78a8",
                label="ca_combo_v2 self-play", marker="o", s=40)
    ax2.axhline(hex_ctrl, color="#2ca02c", linestyle="--", linewidth=2,
                label=f"hex-lattice ceiling")
    ax2.set_xlabel("stones placed N (per game)")
    ax2.set_ylabel("Bragg99")
    # correlation
    if len(n_per_game) > 2:
        corr_sp = float(np.corrcoef(n_per_game, sp)[0, 1])
        corr_r = float(np.corrcoef(n_per_game, rand)[0, 1])
        ax2.set_title(f"Bragg99 vs N\ncorr(N, Bragg_sp)={corr_sp:+.2f}"
                      f"  corr(N, Bragg_rand)={corr_r:+.2f}")
    else:
        ax2.set_title("Bragg99 vs N")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "P4 diffraction signal: self-play corpora carry quasi-periodic structure"
        f" absent in matched random placements (n={len(sp)} games)",
        fontsize=11,
    )
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")
    print(f"  self-play Bragg99:   {sp_m:.3f}  (min={sp.min():.3f}, max={sp.max():.3f})")
    print(f"  random    Bragg99:   {r_m:.4f}  (min={rand.min():.4f}, max={rand.max():.4f})")
    print(f"  hex ctrl  Bragg99:   {hex_ctrl:.3f}")
    print(f"  separation (median):  sp / rand = {np.median(sp) / np.median(rand):.1f}x")


if __name__ == "__main__":
    main()
