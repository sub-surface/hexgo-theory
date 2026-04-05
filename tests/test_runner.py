"""
Tests for experiments/runner.py — headless, no Qt event loop needed.
Tests the worker logic directly without threading.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from engine import HexGame, EisensteinGreedyAgent, RandomAgent
from experiments.runner import ExperimentWorker, ExperimentStats, MoveEvent, GameEvent


def _run_worker_sync(experiment: str, n_games: int = 3, step_delay_ms: int = 0,
                     agent_a_defensive: bool = False, agent_b_defensive: bool = True):
    """Run ExperimentWorker._run_inner() synchronously (no QThread) for testing."""
    move_events = []
    game_events = []
    log_lines = []
    final_stats = []

    class FakeWorker(ExperimentWorker):
        def __init__(self, *a, **kw):
            # Bypass QObject.__init__ entirely — we just want the logic
            self.experiment = kw['experiment']
            self.n_games = kw['n_games']
            self.step_delay_ms = kw['step_delay_ms']
            self.agent_a_defensive = kw['agent_a_defensive']
            self.agent_b_defensive = kw['agent_b_defensive']
            self.max_moves = kw.get('max_moves', 300)
            self._stop = False

        # Stub out Qt signals with plain callables
        class _Sig:
            def emit(self, val): pass
        move_ready = _Sig()
        game_done = _Sig()
        log_line = _Sig()
        finished = _Sig()
        error = _Sig()

    worker = FakeWorker.__new__(FakeWorker)
    worker.experiment = experiment
    worker.n_games = n_games
    worker.step_delay_ms = step_delay_ms
    worker.agent_a_defensive = agent_a_defensive
    worker.agent_b_defensive = agent_b_defensive
    worker.max_moves = 300
    worker._stop = False

    # Monkey-patch signals
    worker.move_ready = type('S', (), {'emit': lambda self, v: move_events.append(v)})()
    worker.game_done  = type('S', (), {'emit': lambda self, v: game_events.append(v)})()
    worker.log_line   = type('S', (), {'emit': lambda self, v: log_lines.append(v)})()
    worker.finished   = type('S', (), {'emit': lambda self, v: final_stats.append(v)})()
    worker.error      = type('S', (), {'emit': lambda self, v: (_ for _ in ()).throw(RuntimeError(v))})()

    worker._run_inner()

    return move_events, game_events, log_lines, final_stats[0] if final_stats else None


class TestExperimentRunner:
    def test_eis_vs_eis_completes(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=2)
        assert len(games) == 2
        assert stats is not None
        assert stats.total_games == 2

    def test_eis_vs_random_completes(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_random", n_games=2)
        assert stats.total_games == 2

    def test_move_events_emitted(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=1)
        assert len(moves) > 0

    def test_move_event_structure(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=1)
        evt = moves[0]
        assert hasattr(evt, 'game')
        assert hasattr(evt, 'move')
        assert hasattr(evt, 'player')
        assert hasattr(evt, 'threats_p1')
        assert hasattr(evt, 'threats_p2')
        assert hasattr(evt, 'forks_p1')
        assert hasattr(evt, 'forks_p2')
        assert hasattr(evt, 'potential')
        assert hasattr(evt, 'live_aps')
        assert isinstance(evt.game, HexGame)
        assert isinstance(evt.move, tuple) and len(evt.move) == 2

    def test_game_event_structure(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=1)
        evt = games[0]
        assert hasattr(evt, 'winner')
        assert hasattr(evt, 'move_count')
        assert hasattr(evt, 'move_history')
        assert hasattr(evt, 'moves_p1')
        assert hasattr(evt, 'moves_p2')
        assert hasattr(evt, 'correlation')
        assert hasattr(evt, 'pattern_counts')
        assert evt.move_count > 0

    def test_winner_is_valid(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=3)
        for g in games:
            assert g.winner in (1, 2, None)

    def test_stats_wins_sum_to_games(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=5)
        total = sum(stats.wins.get(k, 0) for k in (1, 2, 0))
        assert total == 5

    def test_pattern_freq_populated(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=2)
        assert len(stats.pattern_freq) > 0

    def test_all_move_positions_recorded(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=2)
        assert len(stats.all_move_positions) > 0

    def test_forks_count_nonnegative(self):
        moves, games, logs, stats = _run_worker_sync("fork_hunt", n_games=2)
        assert stats.total_forks_seen >= 0

    def test_move_events_have_valid_live_aps(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=1)
        for evt in moves[:10]:
            ap1, ap2 = evt.live_aps
            assert ap1 >= 0 and ap2 >= 0

    def test_log_lines_emitted(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=2)
        assert len(logs) > 0
        assert any("experiment" in l for l in logs)
        assert any("done" in l for l in logs)

    def test_move_numbers_sequential(self):
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=1)
        for i, evt in enumerate(moves):
            assert evt.move_number == i + 1

    def test_player_alternates_correctly(self):
        """Players should follow the 1-2-2 pattern."""
        moves, games, logs, stats = _run_worker_sync("eis_vs_eis", n_games=1)
        # Move 1: P1. Moves 2,3: P2. Moves 4,5: P1. etc.
        players = [e.player for e in moves[:7]]
        assert players[0] == 1, "First move must be P1"
        assert players[1] == 2 and players[2] == 2, "Moves 2,3 must be P2"
        assert players[3] == 1 and players[4] == 1, "Moves 4,5 must be P1"


class TestExperimentStats:
    def test_default_stats(self):
        s = ExperimentStats()
        assert s.total_games == 0
        assert s.total_forks_seen == 0
        assert isinstance(s.pattern_freq, dict)
        assert isinstance(s.all_move_positions, list)

    def test_wins_default_keys(self):
        s = ExperimentStats()
        assert 1 in s.wins
        assert 2 in s.wins
        assert 0 in s.wins


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    sys.exit(result.returncode)
