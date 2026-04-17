"""
Diffraction test of Proposition P4: does long self-play produce Bragg peaks?

P4 (docs/theory/2026-04-17-hamkins-synthesis.md):
  The diffraction intensity over all stone positions at move 100+ of a
  Combo-v2 self-play has a Bragg-peak component exceeding 50% of total
  intensity when averaged over multiple games.

Procedure:
  1. Play `n_games` Combo-v2 self-plays to a fixed horizon (default 180 moves).
  2. For each game, drop the first `burn_in` moves and take the union of all
     remaining stone positions as a point set.
  3. Compute the 2D diffraction intensity on a k-grid via engine/diffraction.py
     (torch + CUDA).
  4. Compare Bragg-peak fraction against two controls:
       (a) same N random points in a similarly-sized disc
       (b) pure hex lattice patch of the same N

Outputs:
  - results/diffraction_p4.json         (per-game metrics + controls)
  - figures/fig_diffraction_p4.png      (mean I(k) heatmap)
  - figures/fig_diffraction_radial.png  (azimuthally averaged profile)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
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

from engine import HexGame
from engine.ca_policy import make_combo_v2_ca
from engine.diffraction import (
    axial_to_cart, diffraction_intensity, radial_profile, bragg_peak_fraction
)


def _delone_bounds(pts: np.ndarray) -> tuple[float, float]:
    """
    Return (d_min, d_max) where d_min = min pairwise distance and d_max =
    max first-neighbour (nearest-neighbour) distance across the point set.
    Proposition P5 asks both to be bounded as game length grows.
    """
    N = pts.shape[0]
    if N < 2:
        return (float("nan"), float("nan"))
    # Broadcast pairwise distances — O(N^2) in memory; fine for N ~ 200.
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    mask = ~np.eye(N, dtype=bool)
    d_min = float(dist[mask].min())
    nearest = np.where(mask, dist, np.inf).min(axis=1)
    d_max = float(nearest.max())
    return d_min, d_max


def _self_play(seed: int, horizon: int) -> HexGame:
    random.seed(seed)
    black = make_combo_v2_ca()
    white = make_combo_v2_ca()
    g = HexGame()
    while g.winner is None and len(g.move_history) < horizon:
        agent = black if g.current_player == 1 else white
        legal = g.legal_moves()
        if not legal:
            break
        mv = agent.choose_move(g)
        if mv in g.board:
            mv = random.choice(legal)
        if not g.make(*mv):
            mv = random.choice(legal)
            g.make(*mv)
    return g


def _collect_points(game: HexGame, burn_in: int) -> list[tuple[int, int]]:
    """Take all stones placed after the burn_in move index."""
    return list(game.move_history[burn_in:])


def _run(n_games: int, horizon: int, burn_in: int, grid: int,
         k_extent: float, seed: int) -> dict:
    print(f"\n── Diffraction P4 ──  n={n_games}  horizon={horizon}  "
          f"burn_in={burn_in}  grid={grid}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    bragg_sp = []          # self-play Bragg-99 scores
    bragg_rand = []        # Random-point control at matched N
    sizes = []             # |S| for each game
    dmins = []             # Delone d_min per game (P5)
    dmaxs = []             # Delone d_max per game (P5)
    mean_I = None          # running mean of I(k) heatmap

    t0 = time.perf_counter()
    for i in range(n_games):
        g = _self_play(seed + i, horizon)
        pts_axial = _collect_points(g, burn_in)
        N = len(pts_axial)
        if N < 10:
            print(f"  game {i+1}/{n_games}: only {N} stones — skipping")
            continue
        sizes.append(N)
        pts = axial_to_cart(pts_axial)
        d_min, d_max = _delone_bounds(pts)
        dmins.append(d_min)
        dmaxs.append(d_max)

        _, _, I = diffraction_intensity(pts, k_extent=k_extent, grid=grid,
                                        device=device, normalise=True)
        bpf = bragg_peak_fraction(I, threshold_q=0.99)
        bragg_sp.append(bpf)

        if mean_I is None:
            mean_I = I.clone()
        else:
            mean_I += I

        # Random-point control (same N, plane region matched to Cartesian
        # extent of self-play positions).
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        rng = np.random.default_rng(seed + 10_000 + i)
        rand_pts = rng.uniform(lo, hi, size=(N, 2)).astype(np.float32)
        _, _, Irand = diffraction_intensity(rand_pts, k_extent=k_extent,
                                            grid=grid, device=device,
                                            normalise=True)
        bragg_rand.append(bragg_peak_fraction(Irand, threshold_q=0.99))

        print(f"  game {i+1}/{n_games}: N={N:3d}  winner={g.winner}  "
              f"bragg_sp={bpf:.3f}  bragg_rand={bragg_rand[-1]:.3f}",
              flush=True)

    if not bragg_sp:
        raise RuntimeError("no games with enough stones")
    mean_I = mean_I / len(bragg_sp)
    wall = time.perf_counter() - t0

    # Hex-lattice positive control — match N to mean self-play size.
    target_N = int(np.mean(sizes))
    side = int(math.ceil(math.sqrt(target_N)))
    lat = [(q, r) for q in range(-(side // 2), side // 2 + 1)
           for r in range(-(side // 2), side // 2 + 1)][:target_N]
    lat_pts = axial_to_cart(lat)
    _, _, I_lat = diffraction_intensity(lat_pts, k_extent=k_extent,
                                         grid=grid, device=device)
    bragg_lat = bragg_peak_fraction(I_lat, threshold_q=0.99)

    print(f"\n[done] wall={wall:.1f}s  mean N={np.mean(sizes):.0f}  "
          f"bragg_sp mean={np.mean(bragg_sp):.3f}  "
          f"bragg_rand mean={np.mean(bragg_rand):.3f}  "
          f"hex-control={bragg_lat:.3f}")

    return {
        "bragg_sp":     [float(x) for x in bragg_sp],
        "bragg_rand":   [float(x) for x in bragg_rand],
        "bragg_hex_control": float(bragg_lat),
        "N_per_game":   [int(x) for x in sizes],
        "d_min":        [float(x) for x in dmins],
        "d_max":        [float(x) for x in dmaxs],
        "mean_I":       mean_I.detach().cpu().numpy().tolist(),
        "k_extent":     float(k_extent),
        "grid":         int(grid),
        "_wall_time":   float(wall),
        "_params":      dict(n_games=n_games, horizon=horizon,
                             burn_in=burn_in, grid=grid, k_extent=k_extent,
                             seed=seed, device=device),
    }


def _plot_heatmap(results: dict, path: str) -> None:
    mean_I = np.array(results["mean_I"], dtype=np.float32)
    k_extent = results["k_extent"]
    grid = results["grid"]
    fig, ax = plt.subplots(figsize=(7, 6))
    # log-scale makes the peaks stand out.
    disp = np.log10(mean_I + 1e-9)
    im = ax.imshow(disp, extent=[-k_extent, k_extent, -k_extent, k_extent],
                   origin="lower", cmap="inferno", aspect="equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title(f"Mean diffraction $\\log_{{10}} I(k)$ over "
                 f"{len(results['bragg_sp'])} Combo-v2 self-plays\n"
                 f"Bragg99={np.mean(results['bragg_sp']):.3f}  "
                 f"(random control {np.mean(results['bragg_rand']):.3f}, "
                 f"hex lattice {results['bragg_hex_control']:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def _plot_radial(results: dict, path: str) -> None:
    mean_I = torch.tensor(np.array(results["mean_I"], dtype=np.float32))
    k_extent = results["k_extent"]
    grid = results["grid"]
    centres, means = radial_profile(mean_I, grid, k_extent, n_bins=96)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centres, means, color="C0", lw=1.2, label="mean self-play I(|k|)")
    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$\langle I(|k|) \rangle$")
    ax.set_yscale("log")
    ax.set_title("Radial diffraction profile — "
                 "Bragg peaks show up as spikes at hex-lattice reciprocal radii")
    ax.legend()
    ax.grid(alpha=0.4, linestyle=":")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def _verdict(results: dict) -> str:
    sp = np.array(results["bragg_sp"])
    rd = np.array(results["bragg_rand"])
    hx = results["bragg_hex_control"]
    Ns = np.array(results["N_per_game"])

    # Stratify: "long" games reached the horizon (N at the max observed).
    N_max = Ns.max()
    long_mask = Ns >= N_max - 5
    short_mask = ~long_mask

    msg = [
        f"Combo-v2 self-play overall:  Bragg99 mean={sp.mean():.3f}  "
        f"std={sp.std():.3f}  n={len(sp)}",
        f"Random control:              Bragg99 mean={rd.mean():.3f}  "
        f"std={rd.std():.3f}",
        f"Hex-lattice control:         Bragg99 = {hx:.3f}",
        "",
        f"Stratified by game length:",
        f"  long games (N>={N_max-5}, n={long_mask.sum()}):   "
        f"Bragg99 mean={sp[long_mask].mean():.3f}  std={sp[long_mask].std():.3f}",
        f"  short games (N<{N_max-5}, n={short_mask.sum()}):  "
        f"Bragg99 mean={sp[short_mask].mean():.3f}  "
        f"std={sp[short_mask].std():.3f}" if short_mask.any() else "  short games: none",
    ]

    if sp[long_mask].mean() > 0.5:
        msg.append("→ P4 supported in long self-play: Bragg99 > 0.50 threshold.")
    elif sp[long_mask].mean() > rd.mean() + 5 * rd.std():
        msg.append("→ P4 partially supported: long-game Bragg99 well above random "
                   "baseline but below 0.50 threshold at this horizon.")
    else:
        msg.append("→ P4 falsified at current horizon: long-game spectrum "
                   "indistinguishable from random.")

    # P5 — Delone property: bounded d_min and bounded d_max.
    if "d_min" in results and results["d_min"]:
        dm = np.array(results["d_min"])
        dM = np.array(results["d_max"])
        msg += [
            "",
            f"P5 Delone bounds (all games, n={len(dm)}):",
            f"  d_min  min={dm.min():.3f}  mean={dm.mean():.3f}  max={dm.max():.3f}",
            f"  d_max  min={dM.min():.3f}  mean={dM.mean():.3f}  max={dM.max():.3f}",
        ]
        # P5 is a Delone property: d_min bounded below and d_max bounded
        # above, with both bounds independent of game length.
        # On a lattice d_min >= 1 is tautological; the substantive content is
        # that d_max does not grow with N, i.e. no arbitrarily large holes.
        sizes_arr = np.array(results["N_per_game"], dtype=np.float32)
        # Pearson correlation of d_max with N (long games should NOT have
        # larger d_max if the point set is Delone).
        if len(dM) > 2:
            corr = float(np.corrcoef(sizes_arr, dM)[0, 1])
        else:
            corr = float("nan")
        msg.append(f"  corr(N, d_max) = {corr:+.2f}")
        if dm.min() >= 0.9 and dM.max() <= 6.0 and (np.isnan(corr) or corr < 0.5):
            msg.append("→ P5 supported: Delone bounds hold, no "
                       "growth of d_max with game length.")
        else:
            msg.append(f"→ P5 equivocal: d_min_floor={dm.min():.3f}, "
                       f"d_max_ceiling={dM.max():.3f}, corr={corr:+.2f}")

    return "\n".join(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games",  type=int, default=30)
    ap.add_argument("--horizon",  type=int, default=180)
    ap.add_argument("--burn-in",  type=int, default=20,
                    help="moves to discard before measuring")
    ap.add_argument("--grid",     type=int, default=256)
    ap.add_argument("--k-extent", type=float, default=4.0 * math.pi)
    ap.add_argument("--seed",     type=int, default=20260417)
    ap.add_argument("--quick",    action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n_games = 8
        args.horizon = 100
        args.grid = 128

    results = _run(args.n_games, args.horizon, args.burn_in,
                   args.grid, args.k_extent, args.seed)

    rpath = Path("results") / "diffraction_p4.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    # Don't save the heatmap in the JSON by default — figures hold that.
    save = {k: v for k, v in results.items() if k != "mean_I"}
    with open(rpath, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n[saved] {rpath}")

    # Keep mean_I only for figures, then drop.
    _plot_heatmap(results, "figures/fig_diffraction_p4.png")
    print("[saved] figures/fig_diffraction_p4.png")
    _plot_radial(results, "figures/fig_diffraction_radial.png")
    print("[saved] figures/fig_diffraction_radial.png")

    print("\n── Verdict ──\n" + _verdict(results))


if __name__ == "__main__":
    main()
