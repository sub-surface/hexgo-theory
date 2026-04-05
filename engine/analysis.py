"""
Mathematical analysis of HexGame states.

- live_lines(game)         : all 6-cell windows that are not blocked by either player
- threat_cells(game, p)    : cells that, if placed by player p, complete a win
- fork_cells(game, p)      : cells that extend two or more distinct axes simultaneously
- potential_map(game)      : Erdos-Selfridge potential per candidate cell
- axis_chain_lengths(game) : per-cell max chain length on each axis
- pair_correlation(moves)  : g(r) — pair correlation function of the move set
"""

from __future__ import annotations
import math
from collections import defaultdict
from engine import HexGame, AXES, WIN_LENGTH


# ── Helpers ───────────────────────────────────────────────────────────────────

def _window_cells(q: int, r: int, dq: int, dr: int, length: int = WIN_LENGTH):
    return [(q + i * dq, r + i * dr) for i in range(length)]


def _all_windows(game: HexGame):
    """
    Yield (cells, axis_idx) for every 6-cell window that overlaps any placed stone.
    Only windows that touch the occupied region are considered.
    """
    seen: set[tuple] = set()
    for (sq, sr) in game.board:
        for a_idx, (dq, dr) in enumerate(AXES):
            # Walk back to the start of any window this cell could be in
            for offset in range(WIN_LENGTH):
                oq, or_ = sq - offset * dq, sr - offset * dr
                key = (a_idx, oq, or_)
                if key in seen:
                    continue
                seen.add(key)
                cells = _window_cells(oq, or_, dq, dr)
                yield cells, a_idx


# ── Public API ────────────────────────────────────────────────────────────────

def live_lines(game: HexGame) -> list[tuple[list, int]]:
    """
    Return all (cells, axis_idx) windows not blocked by either player
    (i.e., both players have at least one stone, or one player does — window is live
    as long as it is not mixed: no cell from BOTH players).
    Returns windows containing exactly one player's stones (or empty).
    """
    result = []
    for cells, a_idx in _all_windows(game):
        players_present = {game.board[c] for c in cells if c in game.board}
        if len(players_present) <= 1:  # not blocked (mixed)
            result.append((cells, a_idx))
    return result


def threat_cells(game: HexGame, player: int) -> dict[tuple, int]:
    """
    Return {cell: threat_count} — empty cells where placing player's stone
    would immediately win (cell fills the last gap in a 5-of-6 window).
    threat_count = number of such windows the cell completes.
    """
    threats: dict[tuple, int] = defaultdict(int)
    for cells, _ in _all_windows(game):
        players = {game.board[c] for c in cells if c in game.board}
        if players != {player}:
            continue
        empty = [c for c in cells if c not in game.board]
        if len(empty) == 1:
            threats[empty[0]] += 1
    return dict(threats)


def fork_cells(game: HexGame, player: int, min_chain: int = 2) -> dict[tuple, int]:
    """
    Return {cell: axis_count} for empty cells where placing would extend
    chains on two or more distinct axes (fork potential).
    axis_count = number of axes on which placing here extends a chain of length >= min_chain.
    """
    forks: dict[tuple, int] = defaultdict(int)
    for cand in game.candidates:
        if cand in game.board:
            continue
        q, r = cand
        axes_hit = 0
        for dq, dr in AXES:
            chain = 1
            for sign in (1, -1):
                nq, nr = q + sign * dq, r + sign * dr
                while game.board.get((nq, nr)) == player:
                    chain += 1
                    nq += sign * dq
                    nr += sign * dr
            if chain >= min_chain:
                axes_hit += 1
        if axes_hit >= 2:
            forks[cand] = axes_hit
    return dict(forks)


def potential_map(game: HexGame) -> dict[tuple, float]:
    """
    Erdos-Selfridge potential per candidate cell.
    phi(cell) = sum over all live windows containing cell of (1/2)^(stones_in_window)
    High potential = cell sits in many live, partially-filled windows.
    """
    cell_potential: dict[tuple, float] = defaultdict(float)
    for cells, _ in _all_windows(game):
        players = {game.board[c] for c in cells if c in game.board}
        if len(players) > 1:
            continue  # blocked window
        n_stones = sum(1 for c in cells if c in game.board)
        contrib = (0.5 ** n_stones)
        for c in cells:
            cell_potential[c] += contrib
    return dict(cell_potential)


def axis_chain_lengths(game: HexGame, player: int) -> dict[tuple, list[int]]:
    """
    For each occupied cell of `player`, return [chain_q, chain_r, chain_diag].
    Chain length = number of consecutive same-player stones along that axis through the cell.
    """
    result = {}
    for (q, r), p in game.board.items():
        if p != player:
            continue
        chains = []
        for dq, dr in AXES:
            count = 1
            for sign in (1, -1):
                nq, nr = q + sign * dq, r + sign * dr
                while game.board.get((nq, nr)) == player:
                    count += 1
                    nq += sign * dq
                    nr += sign * dr
            chains.append(count)
        result[(q, r)] = chains
    return result


def pair_correlation(moves: list[tuple[int, int]], max_r: int = 20) -> dict[int, float]:
    """
    Discrete pair correlation function g(r): for each hex distance r,
    count how many pairs of moves are at that distance, normalised by
    the expected count under a uniform random distribution.
    Returns {r: g(r)} for r in 1..max_r.
    """
    dist_counts: dict[int, int] = defaultdict(int)
    n = len(moves)
    if n < 2:
        return {}
    for i in range(n):
        q1, r1 = moves[i]
        for j in range(i + 1, n):
            q2, r2 = moves[j]
            d = (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) // 2
            if d <= max_r:
                dist_counts[d] += 1
    # Normalise: shell area at distance r in hex grid = 6r
    total_pairs = n * (n - 1) / 2
    result = {}
    for r in range(1, max_r + 1):
        shell_frac = (6 * r) / max(1, sum(6 * rr for rr in range(1, max_r + 1)))
        expected = total_pairs * shell_frac
        result[r] = dist_counts.get(r, 0) / max(1.0, expected)
    return result


def live_ap_count(game: HexGame) -> tuple[int, int]:
    """
    Count of live (unblocked) 6-windows for each player.
    Returns (p1_live, p2_live).
    """
    p1, p2 = 0, 0
    for cells, _ in _all_windows(game):
        players = {game.board[c] for c in cells if c in game.board}
        if players == {1} or not players:
            p1 += 1
        if players == {2} or not players:
            p2 += 1
    return p1, p2


def pattern_fingerprint(game: HexGame, radius: int = 2) -> dict[tuple, str]:
    """
    For each occupied cell, encode the local neighborhood of `radius` as a
    canonical string (up to D6 symmetry via sorting rotations).
    Returns {cell: fingerprint_string}.
    """
    def _encode_local(game: HexGame, cq: int, cr: int, radius: int) -> str:
        tokens = []
        for dq in range(-radius, radius + 1):
            for dr in range(max(-radius, -dq - radius), min(radius, -dq + radius) + 1):
                p = game.board.get((cq + dq, cr + dr), 0)
                tokens.append(str(p))
        return "".join(tokens)

    return {
        (q, r): _encode_local(game, q, r, radius)
        for (q, r) in game.board
    }
