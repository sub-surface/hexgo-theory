"""
Matplotlib / static-SVG HexGo visualisations.

Everything here is torch-free and depends only on numpy + matplotlib so it
works inside a marimo notebook.

- hex_to_xy(q, r)        axial -> cartesian (flat-top hexes)
- draw_board(game, ax)   static render of a HexGame state
- draw_potential(...)    heatmap of Erdos-Selfridge potential
- draw_forks_threats(.)  overlay of fork + threat cells
- animate_game(moves)    return list of matplotlib figures (frames) for a game
- sample_game(...)       quick helper: play one game and return the sequence
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from engine import HexGame
from engine.analysis import (
    threat_cells, fork_cells, potential_map, axis_chain_lengths,
)

# ── coordinate transform (flat-top hexes) ────────────────────────────────────

SQRT3 = math.sqrt(3)


def hex_to_xy(q: int, r: int, size: float = 1.0) -> tuple[float, float]:
    """Axial (q, r) -> cartesian. Flat-top orientation."""
    x = size * (1.5 * q)
    y = size * (SQRT3 * (r + q / 2.0))
    return x, y


def hex_polygon(q: int, r: int, size: float = 1.0) -> np.ndarray:
    """Return (6, 2) array of vertex coordinates for hex at (q, r)."""
    cx, cy = hex_to_xy(q, r, size)
    return np.array([
        (cx + size * math.cos(math.radians(60 * k)),
         cy + size * math.sin(math.radians(60 * k)))
        for k in range(6)
    ])


# ── static board rendering ──────────────────────────────────────────────────

def _all_cells_within(cells: Iterable[tuple[int, int]], pad: int = 2):
    """Yield (q, r) over a bounding region around `cells`, padded."""
    cells = list(cells)
    if not cells:
        for dq in range(-3, 4):
            for dr in range(-3, 4):
                yield (dq, dr)
        return
    qs = [c[0] for c in cells]
    rs = [c[1] for c in cells]
    for q in range(min(qs) - pad, max(qs) + pad + 1):
        for r in range(min(rs) - pad, max(rs) + pad + 1):
            yield (q, r)


def draw_board(game: HexGame,
               ax=None,
               size: float = 1.0,
               show_potential: bool = True,
               show_threats: bool = True,
               show_forks: bool = True,
               show_axes: bool = False,
               title: str | None = None):
    """Render a HexGame state. Returns the matplotlib Axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # potential heatmap over candidate cells
    if show_potential:
        pot = potential_map(game)
        if pot:
            vmax = max(pot.values()) or 1.0
            for (q, r), phi in pot.items():
                if (q, r) in game.board:
                    continue
                poly = hex_polygon(q, r, size)
                shade = plt.cm.YlOrRd(min(1.0, phi / vmax * 0.9))
                ax.fill(poly[:, 0], poly[:, 1], color=shade, alpha=0.6,
                        edgecolor="none", zorder=1)

    # empty cells in the vicinity — light grey outline
    drawn = set()
    for (q, r) in _all_cells_within(game.board.keys() if game.board else [(0,0)]):
        if (q, r) in drawn:
            continue
        drawn.add((q, r))
        if (q, r) in game.board:
            continue
        poly = hex_polygon(q, r, size)
        ax.plot(np.append(poly[:, 0], poly[0, 0]),
                np.append(poly[:, 1], poly[0, 1]),
                color="#d0d0d0", lw=0.5, zorder=2)

    # placed stones
    for (q, r), p in game.board.items():
        poly = hex_polygon(q, r, size)
        color = "#2a4a8a" if p == 1 else "#b8392b"
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.92,
                edgecolor="black", linewidth=0.9, zorder=3)
        ax.plot(np.append(poly[:, 0], poly[0, 0]),
                np.append(poly[:, 1], poly[0, 1]),
                color="black", lw=0.8, zorder=4)

    # highlight last move
    if game.move_history:
        q, r = game.move_history[-1]
        cx, cy = hex_to_xy(q, r, size)
        ax.plot(cx, cy, marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=1.2, zorder=5)

    # threats
    if show_threats:
        for p, col in ((1, "#2a4a8a"), (2, "#b8392b")):
            for (q, r) in threat_cells(game, p):
                cx, cy = hex_to_xy(q, r, size)
                ax.plot(cx, cy, marker="X", markersize=14,
                        markerfacecolor=col, markeredgecolor="white",
                        markeredgewidth=1.2, zorder=6)

    # forks
    if show_forks:
        for p, col in ((1, "#2a4a8a"), (2, "#b8392b")):
            for (q, r), k in fork_cells(game, p).items():
                cx, cy = hex_to_xy(q, r, size)
                ax.plot(cx, cy, marker="D", markersize=8,
                        markerfacecolor="none", markeredgecolor=col,
                        markeredgewidth=1.5, zorder=6)

    # axis overlay from origin
    if show_axes:
        for (dq, dr), c in zip([(1, 0), (0, 1), (1, -1)], ["#888", "#888", "#888"]):
            for s in range(-8, 9):
                cx, cy = hex_to_xy(dq * s, dr * s, size)
                ax.plot(cx, cy, marker=".", color=c, markersize=2, zorder=1.5)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def draw_move_trail(game: HexGame, ax=None, size: float = 1.0,
                    fade: bool = True):
    """Draw placed stones with a move-number annotation. Good for replays."""
    ax = draw_board(game, ax=ax, size=size,
                    show_potential=False, show_threats=False, show_forks=False)
    for i, ((q, r), p) in enumerate(zip(game.move_history, game.player_history)):
        cx, cy = hex_to_xy(q, r, size)
        ax.text(cx, cy, str(i + 1),
                ha="center", va="center", fontsize=7,
                color="white" if p == 1 else "white",
                fontweight="bold", zorder=7)
    return ax


# ── animation helpers (return a list of figures) ────────────────────────────

def replay_game(moves: Sequence[tuple[int, int]],
                step_range: tuple[int, int] | None = None,
                size: float = 1.0) -> plt.Figure:
    """Replay a game up to a given step, return a figure."""
    g = HexGame()
    lo, hi = step_range or (0, len(moves))
    for i, mv in enumerate(moves[:hi]):
        if mv in set(g.legal_moves()):
            g.make(*mv)
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_board(g, ax=ax, size=size, show_potential=True,
               show_threats=True, show_forks=True,
               title=f"move {hi} / {len(moves)}")
    plt.tight_layout()
    return fig


# ── D6 orbit panel ───────────────────────────────────────────────────────────

D6_OPS = [
    ("id",  lambda q, r: (q, r)),
    ("r60", lambda q, r: (-r, q + r)),
    ("r120",lambda q, r: (-q - r, q)),
    ("r180",lambda q, r: (-q, -r)),
    ("r240",lambda q, r: (r, -q - r)),
    ("r300",lambda q, r: (q + r, -q)),
    ("mq",  lambda q, r: (q + r, -r)),
    ("mr",  lambda q, r: (-q, q + r)),
    ("md",  lambda q, r: (-q - r, r)),
    ("mq60",lambda q, r: (r, q)),
    ("mr60",lambda q, r: (-r, -q)),
    ("md60",lambda q, r: (q, -q - r)),
]


def orbit_panel(game: HexGame, size: float = 0.9) -> plt.Figure:
    """Render the D6 orbit of a game state as a 4x3 panel."""
    fig, axs = plt.subplots(3, 4, figsize=(14, 10))
    axs = axs.flatten()
    for ax, (name, op) in zip(axs, D6_OPS):
        g2 = HexGame()
        g2.board = {op(q, r): p for (q, r), p in game.board.items()}
        g2.move_history = [op(q, r) for (q, r) in game.move_history]
        g2.player_history = list(game.player_history)
        draw_board(g2, ax=ax, size=size,
                   show_potential=False, show_threats=False,
                   show_forks=False, title=name)
    plt.tight_layout()
    return fig


# ── one-liner game sampler ──────────────────────────────────────────────────

def sample_game(agent_a: Callable, agent_b: Callable,
                max_moves: int = 150, seed: int | None = None) -> HexGame:
    import random
    if seed is not None:
        random.seed(seed)
    a = agent_a() if callable(agent_a) else agent_a
    b = agent_b() if callable(agent_b) else agent_b
    g = HexGame()
    m = 0
    while g.winner is None and m < max_moves:
        ag = a if g.current_player == 1 else b
        legal = g.legal_moves()
        if not legal:
            break
        mv = ag.choose_move(g)
        if mv not in set(legal):
            mv = random.choice(legal)
        g.make(*mv)
        m += 1
    return g
