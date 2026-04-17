"""
CA-policy framework: express any local-rule strategy as a CAAgent.

A strategy is a pipeline of **features** over candidate cells, combined by a
**reduce** op. Each feature returns a ``dict[cell -> float`` over the currently
legal cells; the reduce op combines features into a single score per cell;
the agent picks argmax (with optional epsilon tie-break).

A second mechanism — **priority channels** — lets us short-circuit with
ordered overrides (e.g. "if an immediate-win cell exists, play it; else fall
through to affine scoring"). ComboAgent is the canonical example.

Design goals
------------
1. Any existing agent in engine/agents.py can be re-expressed as a CAAgent
   with matching move distribution on seeded positions.
2. Adding a new idea (e.g. opening-center bias, D6-equivariance, mirror-pair)
   is a single feature or a single reduce-op swap — never a new agent class.
3. Features run on CPU with numpy-free dict semantics for now. A later
   torch/CUDA variant will vectorise the same feature signatures.

Feature signature
-----------------
    feature(game, player, opponent, legal) -> dict[(q,r): float]

where ``legal`` is a precomputed ``set[(q,r)]`` of legal candidate cells.
Features only need to return entries for cells they want to score; missing
cells default to 0.

Priority channels signature
---------------------------
    priority(game, player, opponent, legal) -> tuple[cell | None, ...]

Priority functions return either ``None`` (pass through) or a single cell
to play immediately. They are consulted in order before the scoring
pipeline runs. This is how we keep ``ComboAgent``'s "win if you can, else
block" logic compact.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable, Literal

from engine import HexGame, AXES, WIN_LENGTH

# ── Type aliases ─────────────────────────────────────────────────────────────

Cell = tuple[int, int]
Feature = Callable[[HexGame, int, int, set[Cell]], dict[Cell, float]]
Priority = Callable[[HexGame, int, int, set[Cell]], Cell | None]


# ── Window enumeration (shared helper, zero-alloc hot path) ──────────────────

def _enumerate_windows(game: HexGame):
    """
    Yield (window_cells, stones_in_window_by_player, empties) for every
    ``WIN_LENGTH``-window that touches an occupied cell. Windows that are
    mixed (both players present) are skipped; such windows cannot be live.

    Each yield:
      cells:    list[Cell] of length WIN_LENGTH
      occ:      dict[int -> int]   e.g. {1: 3} means 3 P1 stones, 0 else
      empties:  list[Cell]
    """
    board = game.board
    seen: set[tuple[int, int, int]] = set()
    for (sq, sr) in board:
        for a_idx, (dq, dr) in enumerate(AXES):
            for offset in range(WIN_LENGTH):
                oq, or_ = sq - offset * dq, sr - offset * dr
                key = (a_idx, oq, or_)
                if key in seen:
                    continue
                seen.add(key)
                cells = [(oq + i * dq, or_ + i * dr) for i in range(WIN_LENGTH)]
                occ = {1: 0, 2: 0}
                empties = []
                mixed = False
                for c in cells:
                    p = board.get(c, 0)
                    if p:
                        occ[p] += 1
                    else:
                        empties.append(c)
                if occ[1] and occ[2]:
                    continue  # blocked / mixed — skip entirely
                yield cells, occ, empties


# ── Features ─────────────────────────────────────────────────────────────────

def feat_chain_length(
    *, defensive: bool = False
) -> Feature:
    """
    Max contiguous chain length if ``player`` places at candidate cell.
    When ``defensive``, take max(own_chain, opp_chain_if_I_block_here).
    Matches EisensteinGreedyAgent.
    """
    def _chain_if_placed(board: dict, q: int, r: int, p: int) -> int:
        best = 1
        for dq, dr in AXES:
            count = 1
            for sign in (1, -1):
                nq, nr = q + sign * dq, r + sign * dr
                while board.get((nq, nr)) == p:
                    count += 1
                    nq += sign * dq
                    nr += sign * dr
            best = max(best, count)
        return best

    def feature(game, player, opponent, legal):
        board = game.board
        out: dict[Cell, float] = {}
        for (q, r) in legal:
            own = _chain_if_placed(board, q, r, player)
            if defensive:
                opp = _chain_if_placed(board, q, r, opponent)
                out[(q, r)] = float(max(own, opp))
            else:
                out[(q, r)] = float(own)
        return out

    return feature


def feat_fork_axes(*, min_chain: int = 1, scale: float = 1.0) -> Feature:
    """
    Number of axes along which placing at the candidate extends a chain of
    length ``>= min_chain``. Matches ForkAwareAgent._fork_axes.
    """
    def feature(game, player, opponent, legal):
        board = game.board
        out: dict[Cell, float] = {}
        for (q, r) in legal:
            axes_hit = 0
            for dq, dr in AXES:
                chain = 1
                for sign in (1, -1):
                    nq, nr = q + sign * dq, r + sign * dr
                    while board.get((nq, nr)) == player:
                        chain += 1
                        nq += sign * dq
                        nr += sign * dr
                if chain >= min_chain:
                    axes_hit += 1
            out[(q, r)] = scale * axes_hit
        return out

    return feature


def feat_potential(*, decay: float = 0.5, scale: float = 1.0) -> Feature:
    """
    Erdos-Selfridge potential. phi(c) = sum over live windows W containing c
    of ``decay ** n_stones(W)``. Matches analysis.potential_map on the legal
    subset and PotentialGradientAgent's cell_pot dict.
    """
    def feature(game, player, opponent, legal):
        out: dict[Cell, float] = {}
        for cells, occ, _ in _enumerate_windows(game):
            n_stones = occ[1] + occ[2]
            contrib = scale * (decay ** n_stones)
            for c in cells:
                if c in legal:
                    out[c] = out.get(c, 0.0) + contrib
        return out

    return feature


def feat_opening_center_bias(
    *, active_until_moves: int = 4, origin: Cell = (0, 0), scale: float = 1.0
) -> Feature:
    """
    While the board has fewer than ``active_until_moves`` stones, prefer cells
    near ``origin``. Score = scale * max(0, active_until_moves - hex_dist).

    Exists to fix ComboAgent's going-Black defect: the opening move has no
    potential gradient, so ComboAgent previously fell through to epsilon-noise.
    This feature gives it a definite opening-move preference without affecting
    mid-game behaviour.
    """
    def _hex_dist(a: Cell, b: Cell) -> int:
        q1, r1 = a
        q2, r2 = b
        return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) // 2

    def feature(game, player, opponent, legal):
        if len(game.board) >= active_until_moves:
            return {}
        out: dict[Cell, float] = {}
        for c in legal:
            d = _hex_dist(c, origin)
            out[c] = scale * max(0, active_until_moves - d)
        return out

    return feature


def feat_noise(*, scale: float = 1e-3) -> Feature:
    """Tiny additive noise for deterministic tie-breaking without bias."""
    def feature(game, player, opponent, legal):
        return {c: scale * random.random() for c in legal}
    return feature


# ── Priority channels ────────────────────────────────────────────────────────

def prio_immediate_win(*, player_side: Literal["own", "opp"] = "own") -> Priority:
    """
    Return a cell that fills a ``WIN_LENGTH - 1``-of-``WIN_LENGTH`` window for
    the specified side (own = immediate win; opp = block opponent's win).

    If multiple such cells exist, return the one completing the most windows
    (matches PotentialGradientAgent behaviour at lines 162-167).
    """
    def priority(game, player, opponent, legal):
        target = player if player_side == "own" else opponent
        counts: dict[Cell, int] = {}
        for cells, occ, empties in _enumerate_windows(game):
            if occ[target] == WIN_LENGTH - 1 and len(empties) == 1:
                ec = empties[0]
                if ec in legal:
                    counts[ec] = counts.get(ec, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)

    return priority


# ── The agent ────────────────────────────────────────────────────────────────

@dataclass
class CAAgent:
    """
    Reducing-CA-style agent. Score per legal cell is:

        score(c) = reduce_op({w_i * feat_i(c) for i in features})

    ``reduce_op`` is "sum" (affine / vanilla) or "max" (tropical / max-plus).

    Priority channels are consulted in order before scoring; the first one
    that returns a non-None cell is played immediately.
    """
    name: str
    features: list[tuple[Feature, float]] = field(default_factory=list)
    priorities: list[Priority] = field(default_factory=list)
    reduce: Literal["sum", "max"] = "sum"

    def choose_move(self, game: HexGame) -> Cell:
        legal = set(game.legal_moves())
        if not legal:
            return next(iter(game.candidates))  # should never happen

        player = game.current_player
        opponent = 3 - player

        # 1. Priority overrides.
        for prio in self.priorities:
            hit = prio(game, player, opponent, legal)
            if hit is not None:
                return hit

        # 2. Scoring pipeline.
        scores: dict[Cell, float] = {}
        for feat, weight in self.features:
            contrib = feat(game, player, opponent, legal)
            if self.reduce == "sum":
                for c, v in contrib.items():
                    scores[c] = scores.get(c, 0.0) + weight * v
            else:  # "max" — tropical
                for c, v in contrib.items():
                    wv = weight * v
                    if wv > scores.get(c, float("-inf")):
                        scores[c] = wv

        if not scores:
            return random.choice(list(legal))
        return max(scores, key=scores.get)


# ── Canonical re-expressions ─────────────────────────────────────────────────
#
# These are intended to be move-distribution-equivalent to the corresponding
# agents in engine/agents.py and hexgo/elo.py, given the same RNG state.
# Verified in tests/test_ca_policy.py.


def make_greedy_ca(defensive: bool = True) -> CAAgent:
    """EisensteinGreedyAgent as a CAAgent."""
    return CAAgent(
        name=f"ca_greedy_{'def' if defensive else 'off'}",
        features=[(feat_chain_length(defensive=defensive), 1.0)],
        reduce="sum",
    )


def make_fork_aware_ca(alpha: float = 2.0, defensive: bool = True,
                      min_chain: int = 1, eps: float = 0.01) -> CAAgent:
    """ForkAwareAgent as a CAAgent."""
    return CAAgent(
        name=f"ca_fork_a{int(alpha)}",
        features=[
            (feat_chain_length(defensive=defensive), 1.0),
            (feat_fork_axes(min_chain=min_chain), alpha),
            (feat_noise(scale=eps), 1.0),
        ],
        reduce="sum",
    )


def make_potential_gradient_ca(
    w_pot: float = 1.0, w_fork: float = 3.0, eps: float = 1e-3
) -> CAAgent:
    """PotentialGradientAgent as a CAAgent."""
    return CAAgent(
        name="ca_potgrad",
        priorities=[
            prio_immediate_win(player_side="own"),
            prio_immediate_win(player_side="opp"),
        ],
        features=[
            (feat_potential(decay=0.5), w_pot),
            (feat_fork_axes(min_chain=1), w_fork),
            (feat_noise(scale=eps), 1.0),
        ],
        reduce="sum",
    )


def make_combo_ca(w_pot: float = 1.0, w_fork: float = 4.0,
                  eps: float = 1e-3) -> CAAgent:
    """ComboAgent as a CAAgent."""
    return CAAgent(
        name="ca_combo",
        priorities=[
            prio_immediate_win(player_side="own"),
            prio_immediate_win(player_side="opp"),
        ],
        features=[
            (feat_potential(decay=0.5), w_pot),
            (feat_fork_axes(min_chain=1), w_fork),
            (feat_noise(scale=eps), 1.0),
        ],
        reduce="sum",
    )


def make_combo_v2_ca(w_pot: float = 1.0, w_fork: float = 4.0,
                    w_open: float = 2.0, active_until_moves: int = 4,
                    eps: float = 1e-3) -> CAAgent:
    """
    ComboAgent v2 — adds opening_center_bias to fix the going-Black defect
    identified in the Hamkins-echo sweep (2026-04-17). The opening feature
    is active only for the first few stones and is dormant thereafter, so
    mid-game behaviour is identical to make_combo_ca.
    """
    return CAAgent(
        name="ca_combo_v2",
        priorities=[
            prio_immediate_win(player_side="own"),
            prio_immediate_win(player_side="opp"),
        ],
        features=[
            (feat_potential(decay=0.5), w_pot),
            (feat_fork_axes(min_chain=1), w_fork),
            (feat_opening_center_bias(active_until_moves=active_until_moves), w_open),
            (feat_noise(scale=eps), 1.0),
        ],
        reduce="sum",
    )
