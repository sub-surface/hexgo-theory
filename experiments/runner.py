"""
Experiment runner — runs in a QThread, emits signals per move and per game.

Experiments:
  - EisensteinVsEisenstein
  - EisensteinVsRandom
  - ForkHunt        (enumerate fork-creating moves across many games)
  - PotentialLandscape (snapshot Erdos-Selfridge map at fixed move counts)
  - PatternCensus   (collect local pattern fingerprints across a corpus)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable

from PySide6.QtCore import QThread, Signal, QObject

from engine import HexGame, EisensteinGreedyAgent, RandomAgent
from engine.analysis import (
    live_lines, threat_cells, fork_cells, potential_map,
    live_ap_count, pair_correlation, pattern_fingerprint,
)


# ── Payloads ──────────────────────────────────────────────────────────────────

@dataclass
class MoveEvent:
    game: HexGame          # snapshot after move
    move: tuple[int, int]
    player: int
    move_number: int
    threats_p1: dict
    threats_p2: dict
    forks_p1: dict
    forks_p2: dict
    potential: dict
    live_aps: tuple[int, int]


@dataclass
class GameEvent:
    game_number: int
    winner: int | None
    move_count: int
    duration: float
    move_history: list[tuple[int, int]]   # ordered sequence of all moves
    moves_p1: list[tuple[int, int]]
    moves_p2: list[tuple[int, int]]
    correlation: dict[int, float]
    pattern_counts: dict[str, int]


@dataclass
class ExperimentStats:
    total_games: int = 0
    wins: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 0: 0})
    total_moves: int = 0
    total_forks_seen: int = 0
    pattern_freq: dict[str, int] = field(default_factory=dict)
    all_move_positions: list[tuple[int, int]] = field(default_factory=list)


# ── Worker ────────────────────────────────────────────────────────────────────

class ExperimentWorker(QObject):
    move_ready   = Signal(object)   # MoveEvent
    game_done    = Signal(object)   # GameEvent
    log_line     = Signal(str)
    finished     = Signal(object)   # ExperimentStats
    error        = Signal(str)

    def __init__(
        self,
        experiment: str,
        n_games: int,
        step_delay_ms: int,
        agent_a_defensive: bool,
        agent_b_defensive: bool,
        max_moves: int = 300,
    ):
        super().__init__()
        self.experiment = experiment
        self.n_games = n_games
        self.step_delay_ms = step_delay_ms
        self.agent_a_defensive = agent_a_defensive
        self.agent_b_defensive = agent_b_defensive
        self.max_moves = max_moves
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._run_inner()
        except Exception as e:
            self.error.emit(str(e))

    def _make_agents(self):
        exp = self.experiment
        if exp == "eis_vs_eis":
            a = EisensteinGreedyAgent("Eis-A", defensive=self.agent_a_defensive)
            b = EisensteinGreedyAgent("Eis-B", defensive=self.agent_b_defensive)
        elif exp == "eis_vs_random":
            a = EisensteinGreedyAgent("Eis-A", defensive=self.agent_a_defensive)
            b = RandomAgent()
        elif exp == "fork_hunt":
            a = EisensteinGreedyAgent("Eis-A", defensive=False)
            b = EisensteinGreedyAgent("Eis-B", defensive=True)
        elif exp == "potential_landscape":
            a = EisensteinGreedyAgent("Eis-A", defensive=self.agent_a_defensive)
            b = EisensteinGreedyAgent("Eis-B", defensive=self.agent_b_defensive)
        elif exp == "pattern_census":
            a = EisensteinGreedyAgent("Eis-A", defensive=self.agent_a_defensive)
            b = EisensteinGreedyAgent("Eis-B", defensive=self.agent_b_defensive)
        else:
            a = EisensteinGreedyAgent("Eis-A", defensive=self.agent_a_defensive)
            b = EisensteinGreedyAgent("Eis-B", defensive=self.agent_b_defensive)
        return a, b

    def _run_inner(self):
        stats = ExperimentStats()
        agent_a, agent_b = self._make_agents()
        self.log_line.emit(f"[experiment] {self.experiment} | {self.n_games} games | "
                           f"agents: {agent_a.name} vs {agent_b.name}")

        for game_idx in range(self.n_games):
            if self._stop:
                break

            game = HexGame()
            agents = {1: agent_a, 2: agent_b}
            if game_idx % 2 == 1:
                agents = {1: agent_b, 2: agent_a}

            t0 = time.perf_counter()
            move_count = 0

            while game.winner is None and move_count < self.max_moves and not self._stop:
                move = agents[game.current_player].choose_move(game)
                player = game.current_player
                game.make(*move)
                move_count += 1

                # Compute analytics on current snapshot
                snap = game.clone()
                t1 = threat_cells(snap, 1)
                t2 = threat_cells(snap, 2)
                f1 = fork_cells(snap, 1)
                f2 = fork_cells(snap, 2)
                pot = potential_map(snap)
                aps = live_ap_count(snap)

                stats.total_forks_seen += len(f1) + len(f2)

                evt = MoveEvent(
                    game=snap,
                    move=move,
                    player=player,
                    move_number=move_count,
                    threats_p1=t1,
                    threats_p2=t2,
                    forks_p1=f1,
                    forks_p2=f2,
                    potential=pot,
                    live_aps=aps,
                )
                self.move_ready.emit(evt)

                if self.step_delay_ms > 0:
                    time.sleep(self.step_delay_ms / 1000.0)

            dur = time.perf_counter() - t0
            winner = game.winner or 0
            stats.total_games += 1
            stats.wins[winner] = stats.wins.get(winner, 0) + 1
            stats.total_moves += move_count

            moves_p1 = [m for m, p in zip(game.move_history, game.player_history) if p == 1]
            moves_p2 = [m for m, p in zip(game.move_history, game.player_history) if p == 2]
            stats.all_move_positions.extend(game.move_history)

            corr = pair_correlation(game.move_history, max_r=15)
            pats = pattern_fingerprint(game, radius=2)
            pat_counts: dict[str, int] = defaultdict(int)
            for fp in pats.values():
                pat_counts[fp] += 1
            for fp, cnt in pat_counts.items():
                stats.pattern_freq[fp] = stats.pattern_freq.get(fp, 0) + cnt

            g_evt = GameEvent(
                game_number=game_idx + 1,
                winner=game.winner,
                move_count=move_count,
                duration=dur,
                move_history=list(game.move_history),
                moves_p1=moves_p1,
                moves_p2=moves_p2,
                correlation=corr,
                pattern_counts=dict(pat_counts),
            )
            self.game_done.emit(g_evt)

            w_str = f"P{winner}" if winner else "draw/timeout"
            self.log_line.emit(
                f"  game {game_idx+1:>4}/{self.n_games} | winner={w_str} | "
                f"moves={move_count} | forks seen={len(f1)+len(f2)} | {dur:.3f}s"
            )

        self.log_line.emit(
            f"[done] wins={stats.wins} | "
            f"avg_moves={stats.total_moves/max(1,stats.total_games):.1f} | "
            f"unique_patterns={len(stats.pattern_freq)}"
        )
        self.finished.emit(stats)


class ExperimentThread(QThread):
    def __init__(self, worker: ExperimentWorker):
        super().__init__()
        self.worker = worker
        self.worker.moveToThread(self)
        self.started.connect(self.worker.run)

    def stop(self):
        self.worker.stop()
        self.quit()
        self.wait(2000)
