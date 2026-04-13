"""
Hand-crafted agent hierarchy for hexgo-theory.

Agent ladder (ascending strength):
  RandomAgent               (imported from hexgo elo.py)
  EisensteinGreedyAgent     (imported from hexgo elo.py) — baseline
  ForkAwareAgent            chain_score + alpha * fork_axes
  PotentialGradientAgent    Erdos-Selfridge potential + threat/fork bonuses
  ComboAgent                threat-first, then potential gradient
"""
from __future__ import annotations
import random
from engine import HexGame, AXES, WIN_LENGTH
from engine.analysis import fork_cells, threat_cells, potential_map


# ── ForkAwareAgent ─────────────────────────────────────────────────────────────

class ForkAwareAgent:
    """
    Extends EisensteinGreedy by adding a fork multiplier.

    Score = max_chain_length + alpha * axes_hit + eps * noise

    where axes_hit = number of Z[omega] axes along which placing here extends
    a chain of at least min_chain stones (the fork dimension).

    alpha=0 recovers pure greedy.  alpha=2 (default) weights fork cells heavily.
    defensive=True also scores blocking the opponent's chains/forks.
    eps: small noise to break determinism without affecting strategy.
    """
    _AXES = AXES  # [(dq, dr), ...]

    def __init__(self, name: str = "fork_aware", alpha: float = 2.0,
                 min_chain: int = 1, defensive: bool = True, eps: float = 0.01):
        self.name = name
        self.alpha = alpha
        self.min_chain = min_chain
        self.defensive = defensive
        self.eps = eps

    def choose_move(self, game: HexGame) -> tuple[int, int]:
        player   = game.current_player
        opponent = 3 - player
        best_move, best_score = None, -1.0

        for q, r in game.legal_moves():
            own_chain  = self._chain_if_placed(game, q, r, player)
            own_forks  = self._fork_axes(game, q, r, player)
            own_score  = own_chain + self.alpha * own_forks

            if self.defensive:
                opp_chain = self._chain_if_placed(game, q, r, opponent)
                opp_forks = self._fork_axes(game, q, r, opponent)
                opp_score = opp_chain + self.alpha * opp_forks
                score = max(own_score, opp_score)
            else:
                score = own_score

            score += self.eps * random.random()

            if score > best_score or best_move is None:
                best_score, best_move = score, (q, r)

        return best_move or random.choice(game.legal_moves())

    def _chain_if_placed(self, game: HexGame, q: int, r: int, player: int) -> int:
        best = 1
        for dq, dr in self._AXES:
            count = 1
            for sign in (1, -1):
                nq, nr = q + sign * dq, r + sign * dr
                while game.board.get((nq, nr)) == player:
                    count += 1
                    nq += sign * dq
                    nr += sign * dr
            best = max(best, count)
        return best

    def _fork_axes(self, game: HexGame, q: int, r: int, player: int) -> int:
        """Number of axes on which placing at (q,r) extends a chain >= min_chain."""
        axes_hit = 0
        for dq, dr in self._AXES:
            chain = 1
            for sign in (1, -1):
                nq, nr = q + sign * dq, r + sign * dr
                while game.board.get((nq, nr)) == player:
                    chain += 1
                    nq += sign * dq
                    nr += sign * dr
            if chain >= self.min_chain:
                axes_hit += 1
        return axes_hit


# ── PotentialGradientAgent ─────────────────────────────────────────────────────

class PotentialGradientAgent:
    """
    Follows the Erdos-Selfridge potential gradient.

    Score = w_pot * potential(cell)
           + w_threat_own * threat_count_own
           + w_threat_opp * threat_count_opp
           + w_fork       * fork_axes

    Single-pass window scan: potential, threats, and fork bonuses are all
    computed in one loop over candidate cells to avoid repeated _all_windows calls.
    """

    def __init__(self, name: str = "potential_gradient",
                 w_pot: float = 1.0,
                 w_threat_own: float = 100.0,
                 w_threat_opp: float = 80.0,
                 w_fork: float = 3.0,
                 eps: float = 0.001):
        self.name = name
        self.w_pot        = w_pot
        self.w_threat_own = w_threat_own
        self.w_threat_opp = w_threat_opp
        self.w_fork       = w_fork
        self.eps          = eps

    def choose_move(self, game: HexGame) -> tuple[int, int]:
        player   = game.current_player
        opponent = 3 - player
        board    = game.board
        legal    = set(game.legal_moves())

        cell_pot:    dict[tuple, float] = {}
        cell_th_own: dict[tuple, int]   = {}
        cell_th_opp: dict[tuple, int]   = {}

        seen: set[tuple] = set()
        for (sq, sr) in board:
            for a_idx, (dq, dr) in enumerate(AXES):
                for offset in range(WIN_LENGTH):
                    oq, or_ = sq - offset * dq, sr - offset * dr
                    key = (a_idx, oq, or_)
                    if key in seen:
                        continue
                    seen.add(key)
                    cells = [(oq + i*dq, or_ + i*dr) for i in range(WIN_LENGTH)]
                    players_in = {board[c] for c in cells if c in board}
                    if len(players_in) > 1:
                        continue  # blocked
                    n_stones = sum(1 for c in cells if c in board)
                    contrib  = 0.5 ** n_stones
                    empty    = [c for c in cells if c not in board]
                    for c in cells:
                        if c in legal:
                            cell_pot[c] = cell_pot.get(c, 0.0) + contrib
                    # Threat detection
                    if n_stones == WIN_LENGTH - 1 and len(empty) == 1:
                        ec = empty[0]
                        if ec in legal:
                            if players_in == {player}:
                                cell_th_own[ec] = cell_th_own.get(ec, 0) + 1
                            elif players_in == {opponent}:
                                cell_th_opp[ec] = cell_th_opp.get(ec, 0) + 1

        # Fast-exit: immediate win
        if cell_th_own:
            return max(cell_th_own, key=cell_th_own.get)
        # Fast-exit: block opponent win
        if cell_th_opp:
            return max(cell_th_opp, key=cell_th_opp.get)

        best_move, best_score = None, -1e9
        for q, r in legal:
            cell  = (q, r)
            score = self.w_pot * cell_pot.get(cell, 0.0)
            score += self.w_fork * self._fork_axes(game, q, r, player)
            score += self.eps * random.random()
            if score > best_score or best_move is None:
                best_score, best_move = score, cell

        return best_move or random.choice(list(legal))

    def _fork_axes(self, game: HexGame, q: int, r: int, player: int) -> int:
        axes_hit = 0
        board = game.board
        for dq, dr in AXES:
            chain = 1
            for sign in (1, -1):
                nq, nr = q + sign * dq, r + sign * dr
                while board.get((nq, nr)) == player:
                    chain += 1
                    nq += sign * dq
                    nr += sign * dr
            if chain >= 1:
                axes_hit += 1
        return axes_hit


# ── ComboAgent ─────────────────────────────────────────────────────────────────

class ComboAgent:
    """
    Threat-first with potential gradient tiebreaking.

    Priority order:
      1. Win immediately if possible (own threat)
      2. Block opponent's winning threat
      3. PotentialGradientAgent scoring for the rest
    """

    def __init__(self, name: str = "combo",
                 w_pot: float = 1.0, w_fork: float = 4.0):
        self.name = name
        self._pg = PotentialGradientAgent(
            name=name,
            w_pot=w_pot,
            w_threat_own=10000.0,
            w_threat_opp=8000.0,
            w_fork=w_fork,
        )

    def choose_move(self, game: HexGame) -> tuple[int, int]:
        return self._pg.choose_move(game)
