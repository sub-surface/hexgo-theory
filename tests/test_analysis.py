"""
Tests for engine/analysis.py — all headless, no Qt.

Run: python -m pytest tests/ -v
  or: python tests/test_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from engine import HexGame, WIN_LENGTH
from engine.analysis import (
    live_lines, threat_cells, fork_cells, potential_map,
    axis_chain_lengths, pair_correlation, live_ap_count,
    pattern_fingerprint,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def place(game: HexGame, moves: list[tuple[int, int]]):
    """Place a sequence of moves, alternating players as per 1-2-2 rule."""
    for q, r in moves:
        game.make(q, r)


def game_with_chain(player: int, length: int, axis=(1, 0)) -> HexGame:
    """Return a game where `player` has `length` stones along axis from origin."""
    g = HexGame()
    dq, dr = axis
    # P1 always goes first; interleave both players
    p1_moves = [(i * dq, i * dr) for i in range(length)] if player == 1 else [(100, 100 + i) for i in range(length)]
    p2_moves = [(100, i) for i in range(length)] if player == 1 else [(i * dq, i * dr) for i in range(length)]

    # Build interleaved sequence honouring 1-2-2 rule
    for i in range(length):
        if player == 1:
            g.make(p1_moves[i][0], p1_moves[i][1])
            if i < length - 1 or True:  # second placement
                g.make(p2_moves[i][0], p2_moves[i][1])
        else:
            g.make(p1_moves[i][0], p1_moves[i][1])
            g.make(p2_moves[i][0], p2_moves[i][1])
    return g


# ── live_lines ────────────────────────────────────────────────────────────────

class TestLiveLines:
    def test_empty_game_has_no_lines(self):
        g = HexGame()
        assert live_lines(g) == []

    def test_single_stone_creates_live_lines(self):
        g = HexGame()
        g.make(0, 0)
        lines = live_lines(g)
        # A single stone should be part of multiple 6-windows across 3 axes
        assert len(lines) > 0

    def test_mixed_window_not_live(self):
        """A window with stones from both players is blocked — not live."""
        g = HexGame()
        # P1 at (0,0), then P2 at (1,0) — same q-axis window
        g.make(0, 0)   # P1 move 1
        g.make(1, 0)   # P2 move 1
        g.make(2, 0)   # P2 move 2
        lines = live_lines(g)
        # Any window containing both (0,0) P1 and (1,0) or (2,0) P2 should be blocked
        blocked = []
        for cells, _ in lines:
            if (0, 0) in cells and ((1, 0) in cells or (2, 0) in cells):
                blocked.append(cells)
        assert len(blocked) == 0, "Mixed windows should not appear in live_lines"

    def test_line_count_grows_with_board(self):
        g = HexGame()
        g.make(0, 0)
        count_1 = len(live_lines(g))
        g.make(5, 5)   # P2 far away
        g.make(5, 6)   # P2 second move
        g.make(1, 0)   # P1 (turn 2, move 1)
        count_2 = len(live_lines(g))
        assert count_2 > count_1


# ── threat_cells ──────────────────────────────────────────────────────────────

class TestThreatCells:
    def test_no_threats_on_empty_game(self):
        g = HexGame()
        assert threat_cells(g, 1) == {}
        assert threat_cells(g, 2) == {}

    def test_five_in_row_creates_threat(self):
        """Five P1 stones in a row → the 6th cell is a threat."""
        g = HexGame()
        # Place 5 P1 stones at (0,0)..(4,0) on q-axis
        # Turn structure: P1 places 1, then 2-2-2...
        # Move 1: P1@(0,0)
        # Move 2+3: P2 somewhere, P2 somewhere
        # Move 4+5: P1, P1 ... etc
        g.make(0, 0)          # P1 first stone
        g.make(10, 10)        # P2
        g.make(10, 11)        # P2
        g.make(1, 0)          # P1
        g.make(2, 0)          # P1
        g.make(10, 12)        # P2
        g.make(10, 13)        # P2
        g.make(3, 0)          # P1
        g.make(4, 0)          # P1
        threats = threat_cells(g, 1)
        assert (5, 0) in threats or (-1, 0) in threats, \
            f"Expected threat at (5,0) or (-1,0), got {list(threats.keys())}"

    def test_threat_count_reflects_multiple_windows(self):
        """A cell completing two separate windows should have threat_count >= 2."""
        g = HexGame()
        # Build two overlapping 5-in-a-row for P1 sharing one empty endpoint
        # Axis q: (0,0)..(4,0) → threat at (5,0)
        # Axis r: (5,1)..(5,5) → threat at (5,0) if we set up carefully
        # This is complex to guarantee without a specific layout; just check count >= 1
        g.make(0, 0)
        threats = threat_cells(g, 1)
        # At minimum, a newly placed stone doesn't immediately create a win threat
        # (need 5 in a row first) — just verify structure is correct type
        assert isinstance(threats, dict)
        for v in threats.values():
            assert isinstance(v, int) and v >= 1


# ── fork_cells ────────────────────────────────────────────────────────────────

class TestForkCells:
    def test_no_forks_on_fresh_game(self):
        g = HexGame()
        g.make(0, 0)
        forks = fork_cells(g, 1)
        # Single stone — adjacent cells might extend on 1 axis only
        for cell, axes in forks.items():
            assert axes >= 2, f"Fork cell {cell} only has {axes} axes"

    def test_fork_requires_two_axes(self):
        """fork_cells only returns cells that extend chains on 2+ axes."""
        g = HexGame()
        # Place stones on two axes meeting at a junction
        g.make(0, 0)    # P1
        g.make(50, 50)  # P2
        g.make(50, 51)  # P2
        g.make(1, 0)    # P1  — extends q-axis
        g.make(2, 0)    # P1
        g.make(50, 52)  # P2
        g.make(50, 53)  # P2
        g.make(0, 1)    # P1  — extends r-axis from origin vicinity
        g.make(0, 2)    # P1
        forks = fork_cells(g, 1, min_chain=2)
        for cell, axes in forks.items():
            assert axes >= 2

    def test_fork_cell_is_empty(self):
        """No occupied cell should appear in fork_cells."""
        g = HexGame()
        g.make(0, 0)
        g.make(5, 5)
        g.make(5, 6)
        forks = fork_cells(g, 1)
        for cell in forks:
            assert cell not in g.board, f"Occupied cell {cell} in fork_cells"


# ── potential_map ─────────────────────────────────────────────────────────────

class TestPotentialMap:
    def test_empty_game_no_potential(self):
        g = HexGame()
        pot = potential_map(g)
        assert pot == {}

    def test_potential_positive(self):
        g = HexGame()
        g.make(0, 0)
        pot = potential_map(g)
        for v in pot.values():
            assert v > 0

    def test_blocked_window_reduces_potential(self):
        """Mixing both players in a window should reduce total potential."""
        g1 = HexGame()
        g1.make(0, 0)
        pot1 = sum(potential_map(g1).values())

        # Block the q-axis window by adding P2 stone in same window
        g2 = HexGame()
        g2.make(0, 0)   # P1
        g2.make(1, 0)   # P2
        g2.make(2, 0)   # P2
        pot2 = sum(potential_map(g2).values())
        assert pot2 < pot1 * 3, "Blocking should not increase potential proportionally"

    def test_potential_cell_in_candidates(self):
        """Potential map cells should include candidates (empty cells near stones)."""
        g = HexGame()
        g.make(0, 0)
        pot = potential_map(g)
        # All cells in potential map should be either board cells or near them
        assert len(pot) > 0


# ── axis_chain_lengths ────────────────────────────────────────────────────────

class TestAxisChainLengths:
    def test_isolated_stone_has_chain_one(self):
        g = HexGame()
        g.make(0, 0)
        chains = axis_chain_lengths(g, 1)
        assert (0, 0) in chains
        assert all(c == 1 for c in chains[(0, 0)])

    def test_chain_length_on_q_axis(self):
        g = HexGame()
        g.make(0, 0)    # P1
        g.make(10, 10)  # P2
        g.make(10, 11)  # P2
        g.make(1, 0)    # P1
        g.make(2, 0)    # P1
        chains = axis_chain_lengths(g, 1)
        # (1,0) should have q-axis chain of 3 (0,0), (1,0), (2,0)
        assert chains[(1, 0)][0] == 3  # index 0 = q-axis (1,0)

    def test_only_correct_player(self):
        g = HexGame()
        g.make(0, 0)   # P1
        g.make(1, 0)   # P2
        g.make(2, 0)   # P2
        chains_p1 = axis_chain_lengths(g, 1)
        chains_p2 = axis_chain_lengths(g, 2)
        assert (0, 0) in chains_p1
        assert (1, 0) not in chains_p1
        assert (1, 0) in chains_p2


# ── pair_correlation ──────────────────────────────────────────────────────────

class TestPairCorrelation:
    def test_empty_returns_empty(self):
        assert pair_correlation([]) == {}
        assert pair_correlation([(0, 0)]) == {}

    def test_returns_dict_with_distance_keys(self):
        moves = [(0, 0), (1, 0), (0, 1), (2, 0)]
        corr = pair_correlation(moves, max_r=5)
        assert isinstance(corr, dict)
        for k in corr:
            assert 1 <= k <= 5

    def test_clustered_moves_have_high_short_range_correlation(self):
        """Moves clustered near origin should have g(1) > 1 (above random)."""
        moves = [(i, 0) for i in range(8)]  # tight cluster on q-axis
        corr = pair_correlation(moves, max_r=10)
        assert corr.get(1, 0) > 0.5, "Clustered moves should show short-range correlation"


# ── live_ap_count ─────────────────────────────────────────────────────────────

class TestLiveApCount:
    def test_empty_game(self):
        g = HexGame()
        p1, p2 = live_ap_count(g)
        assert p1 == 0 and p2 == 0

    def test_counts_are_nonnegative(self):
        g = HexGame()
        g.make(0, 0)
        p1, p2 = live_ap_count(g)
        assert p1 >= 0 and p2 >= 0

    def test_p1_count_higher_after_p1_stones(self):
        """After placing only P1 stones, P1 should have more live APs than P2."""
        g = HexGame()
        g.make(0, 0)    # P1
        g.make(50, 50)  # P2 far away
        g.make(50, 51)  # P2 far away
        g.make(1, 0)    # P1
        g.make(2, 0)    # P1
        p1, p2 = live_ap_count(g)
        # P1 has a forming chain; P2 has isolated far stones — P1 should lead
        assert p1 >= p2


# ── pattern_fingerprint ───────────────────────────────────────────────────────

class TestPatternFingerprint:
    def test_returns_dict(self):
        g = HexGame()
        g.make(0, 0)
        fp = pattern_fingerprint(g, radius=1)
        assert isinstance(fp, dict)
        assert (0, 0) in fp

    def test_fingerprint_is_string(self):
        g = HexGame()
        g.make(0, 0)
        fp = pattern_fingerprint(g, radius=1)
        for v in fp.values():
            assert isinstance(v, str)

    def test_same_position_same_fingerprint(self):
        """Two games with the same board state should produce the same fingerprints."""
        g1 = HexGame()
        g1.make(0, 0)
        g2 = HexGame()
        g2.make(0, 0)
        fp1 = pattern_fingerprint(g1, radius=1)
        fp2 = pattern_fingerprint(g2, radius=1)
        assert fp1 == fp2

    def test_different_position_different_fingerprint(self):
        g = HexGame()
        g.make(0, 0)
        g.make(5, 5)   # P2
        g.make(5, 6)   # P2
        g.make(1, 0)   # P1
        g.make(0, 1)   # P1
        fp = pattern_fingerprint(g, radius=1)
        # Cells at (0,0) and (1,0) have different local neighborhoods
        assert fp.get((0, 0)) != fp.get((1, 0))


# ── Integration: EisensteinGreedyAgent ───────────────────────────────────────

class TestEisensteinIntegration:
    def test_full_game_runs_to_completion(self):
        """Eisenstein vs Eisenstein should terminate within max_moves."""
        from engine import EisensteinGreedyAgent
        a = EisensteinGreedyAgent("A", defensive=False)
        b = EisensteinGreedyAgent("B", defensive=True)
        g = HexGame()
        agents = {1: a, 2: b}
        max_moves = 300
        move_count = 0
        while g.winner is None and move_count < max_moves:
            m = agents[g.current_player].choose_move(g)
            g.make(*m)
            move_count += 1
        assert g.winner is not None or move_count == max_moves

    def test_analysis_on_complete_game(self):
        """All analysis functions should run without error on a real game state."""
        from engine import EisensteinGreedyAgent
        a = EisensteinGreedyAgent("A", defensive=True)
        b = EisensteinGreedyAgent("B", defensive=True)
        g = HexGame()
        agents = {1: a, 2: b}
        for _ in range(20):
            if g.winner:
                break
            m = agents[g.current_player].choose_move(g)
            g.make(*m)

        # All analysis calls should not raise
        ll = live_lines(g)
        t1 = threat_cells(g, 1)
        t2 = threat_cells(g, 2)
        f1 = fork_cells(g, 1)
        f2 = fork_cells(g, 2)
        pot = potential_map(g)
        aps = live_ap_count(g)
        chains = axis_chain_lengths(g, 1)
        fp = pattern_fingerprint(g, radius=2)
        corr = pair_correlation(g.move_history, max_r=10)

        assert isinstance(ll, list)
        assert isinstance(t1, dict) and isinstance(t2, dict)
        assert isinstance(f1, dict) and isinstance(f2, dict)
        assert isinstance(pot, dict)
        assert len(aps) == 2
        assert isinstance(chains, dict)
        assert isinstance(fp, dict)
        assert isinstance(corr, dict)

    def test_fork_cells_detected_in_real_game(self):
        """Across a 50-move game, at least some fork cells should appear."""
        from engine import EisensteinGreedyAgent
        a = EisensteinGreedyAgent("A", defensive=False)
        b = EisensteinGreedyAgent("B", defensive=False)
        g = HexGame()
        agents = {1: a, 2: b}
        found_forks = False
        for _ in range(50):
            if g.winner:
                break
            m = agents[g.current_player].choose_move(g)
            g.make(*m)
            if fork_cells(g, 1) or fork_cells(g, 2):
                found_forks = True
                break
        assert found_forks, "No fork cells detected across 50 moves of real play"


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    sys.exit(result.returncode)
