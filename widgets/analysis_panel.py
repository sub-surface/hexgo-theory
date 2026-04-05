"""
AnalysisPanel — right-side panel showing live metrics and charts.

Sections:
  - Game stats (move #, player, winner)
  - Threat counts per player
  - Fork counts per player
  - Live AP counts per player
  - Chain length distribution (mini bar chart)
  - Pair correlation g(r) sparkline
  - Pattern type count
  - Cumulative win stats across games
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy, QScrollArea,
)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPaintEvent

from experiments.runner import MoveEvent, GameEvent, ExperimentStats


# ── Palette ───────────────────────────────────────────────────────────────────
BG       = QColor("#050a0f")
PANEL_BG = QColor("#080f18")
BORDER   = QColor("#0d1a2a")
TEXT     = QColor("#c8d4e0")
DIM      = QColor("#3a4a5a")
NAVY     = QColor("#003580")
WHITE_   = QColor("#e8e8e8")
RED      = QColor("#cc2200")
CYAN     = QColor("#00b4d8")
GREEN    = QColor("#1a7a40")


def _label(text: str, dim: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {'#3a4a5a' if dim else '#c8d4e0'}; "
        f"font-family: Consolas; font-size: 11px; padding: 0;"
    )
    return lbl


def _section(title: str) -> QLabel:
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "color: #003580; font-family: Consolas; font-size: 10px; "
        "font-weight: bold; letter-spacing: 1px; "
        "padding: 4px 0 2px 0; border-top: 1px solid #0d1a2a;"
    )
    return lbl


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: #0d1a2a;")
    f.setFixedHeight(1)
    return f


class SparklineWidget(QWidget):
    """Mini line chart for pair correlation g(r)."""

    def __init__(self, label: str = "", height: int = 50, parent=None):
        super().__init__(parent)
        self._label = label
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._data: list[float] = []
        self._baseline: float = 1.0

    def set_data(self, values: list[float], baseline: float = 1.0):
        self._data = values
        self._baseline = baseline
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), PANEL_BG)

        if not self._data:
            painter.setPen(DIM)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no data")
            painter.end()
            return

        w, h = self.width(), self.height()
        pad = 6
        inner_w = w - 2 * pad
        inner_h = h - 2 * pad
        max_v = max(max(self._data), self._baseline * 1.5, 1.0)
        n = len(self._data)

        # Baseline at g=1
        base_y = pad + inner_h * (1 - self._baseline / max_v)
        painter.setPen(QPen(DIM, 0.5, Qt.PenStyle.DashLine))
        painter.drawLine(QRectF(pad, base_y, inner_w, 0).topLeft(),
                         QRectF(pad + inner_w, base_y, 0, 0).topLeft())

        # Data line
        painter.setPen(QPen(NAVY, 1.2))
        pts = []
        for i, v in enumerate(self._data):
            x = pad + (i / max(1, n - 1)) * inner_w
            y = pad + inner_h * (1 - v / max_v)
            pts.append((x, y))

        for i in range(len(pts) - 1):
            painter.drawLine(
                QRectF(pts[i][0], pts[i][1], 0, 0).topLeft(),
                QRectF(pts[i+1][0], pts[i+1][1], 0, 0).topLeft(),
            )

        if self._label:
            painter.setPen(DIM)
            f = QFont("Consolas, Courier New, monospace")
            f.setPixelSize(11)
            painter.setFont(f)
            painter.drawText(QRectF(pad, pad, inner_w, 12),
                             Qt.AlignmentFlag.AlignLeft, self._label)

        painter.end()


class MiniBarWidget(QWidget):
    """Horizontal mini bar chart."""

    def __init__(self, height: int = 40, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._bars: list[tuple[str, float, QColor]] = []

    def set_bars(self, bars: list[tuple[str, float, QColor]]):
        self._bars = bars
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(self.rect(), PANEL_BG)
        if not self._bars:
            painter.end()
            return

        w, h = self.width(), self.height()
        max_v = max(v for _, v, _ in self._bars) or 1.0
        n = len(self._bars)
        bw = w / n
        pad_y = 4

        f = QFont("Consolas, Courier New, monospace")
        f.setPixelSize(10)
        painter.setFont(f)
        for i, (lbl, v, col) in enumerate(self._bars):
            bh = max(1.0, (v / max_v) * (h - pad_y - 12))
            x = i * bw
            painter.fillRect(
                QRectF(x + 1, h - pad_y - bh, bw - 2, bh), col
            )
            painter.setPen(DIM)
            painter.drawText(
                QRectF(x, h - 12, bw, 12),
                Qt.AlignmentFlag.AlignCenter, lbl
            )
        painter.end()


class AnalysisPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: #080f18;")
        self.setMinimumWidth(220)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Cumulative stats
        self._wins = {1: 0, 2: 0, 0: 0}
        self._game_count = 0
        self._all_forks = 0
        self._pattern_freq: dict[str, int] = defaultdict(int)

        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: #080f18;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setStyleSheet("background: #080f18;")
        lay = QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(2)

        # ── Current move ────────────────────────────────────────────────────
        lay.addWidget(_section("CURRENT MOVE"))
        self._lbl_move   = _label("—")
        self._lbl_player = _label("—")
        self._lbl_winner = _label("—")
        lay.addWidget(self._lbl_move)
        lay.addWidget(self._lbl_player)
        lay.addWidget(self._lbl_winner)

        # ── Threats ─────────────────────────────────────────────────────────
        lay.addWidget(_section("THREATS"))
        self._lbl_t1 = _label("P1: —")
        self._lbl_t2 = _label("P2: —")
        lay.addWidget(self._lbl_t1)
        lay.addWidget(self._lbl_t2)

        # ── Forks ───────────────────────────────────────────────────────────
        lay.addWidget(_section("FORK CELLS"))
        self._lbl_f1 = _label("P1: —")
        self._lbl_f2 = _label("P2: —")
        lay.addWidget(self._lbl_f1)
        lay.addWidget(self._lbl_f2)

        # ── Live APs ─────────────────────────────────────────────────────────
        lay.addWidget(_section("LIVE 6-APs"))
        self._lbl_ap1 = _label("P1: —")
        self._lbl_ap2 = _label("P2: —")
        lay.addWidget(self._lbl_ap1)
        lay.addWidget(self._lbl_ap2)

        # ── Chain lengths ────────────────────────────────────────────────────
        lay.addWidget(_section("CHAIN DIST"))
        self._chain_bars = MiniBarWidget(height=48)
        lay.addWidget(self._chain_bars)

        # ── Pair correlation ─────────────────────────────────────────────────
        lay.addWidget(_section("PAIR CORR g(r)"))
        self._corr_spark = SparklineWidget(label="r=1..15", height=55)
        lay.addWidget(self._corr_spark)

        # ── Cumulative ───────────────────────────────────────────────────────
        lay.addWidget(_section("CUMULATIVE"))
        self._lbl_games   = _label("games: 0")
        self._lbl_win1    = _label("P1 wins: 0")
        self._lbl_win2    = _label("P2 wins: 0")
        self._lbl_draws   = _label("timeout: 0")
        self._lbl_forks_c = _label("forks seen: 0")
        self._lbl_patterns= _label("unique patterns: 0")
        lay.addWidget(self._lbl_games)
        lay.addWidget(self._lbl_win1)
        lay.addWidget(self._lbl_win2)
        lay.addWidget(self._lbl_draws)
        lay.addWidget(self._lbl_forks_c)
        lay.addWidget(self._lbl_patterns)

        lay.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Public update methods ─────────────────────────────────────────────────

    def on_move(self, evt: MoveEvent):
        self._lbl_move.setText(f"move: {evt.move_number}")
        self._lbl_player.setText(f"player: P{evt.player}")
        w = evt.game.winner
        self._lbl_winner.setText(f"winner: P{w}" if w else "winner: —")

        t1 = len(evt.threats_p1)
        t2 = len(evt.threats_p2)
        self._lbl_t1.setText(f"P1: {t1}" + (" ★" if t1 > 0 else ""))
        self._lbl_t2.setText(f"P2: {t2}" + (" ★" if t2 > 0 else ""))
        self._lbl_t1.setStyleSheet(
            f"color: {'#cc2200' if t1 > 0 else '#c8d4e0'}; "
            "font-family: Consolas; font-size: 11px;"
        )
        self._lbl_t2.setStyleSheet(
            f"color: {'#cc2200' if t2 > 0 else '#c8d4e0'}; "
            "font-family: Consolas; font-size: 11px;"
        )

        f1 = len(evt.forks_p1)
        f2 = len(evt.forks_p2)
        self._lbl_f1.setText(f"P1: {f1}")
        self._lbl_f2.setText(f"P2: {f2}")
        self._lbl_f1.setStyleSheet(
            f"color: {'#00b4d8' if f1 > 0 else '#c8d4e0'}; "
            "font-family: Consolas; font-size: 11px;"
        )
        self._lbl_f2.setStyleSheet(
            f"color: {'#00b4d8' if f2 > 0 else '#c8d4e0'}; "
            "font-family: Consolas; font-size: 11px;"
        )

        ap1, ap2 = evt.live_aps
        self._lbl_ap1.setText(f"P1: {ap1}")
        self._lbl_ap2.setText(f"P2: {ap2}")

        # Chain length distribution from current board
        from engine.analysis import axis_chain_lengths
        chains1 = axis_chain_lengths(evt.game, 1)
        chains2 = axis_chain_lengths(evt.game, 2)
        dist: dict[int, int] = defaultdict(int)
        for ch_list in list(chains1.values()) + list(chains2.values()):
            dist[max(ch_list)] += 1
        bars = []
        for length in range(1, 7):
            count = dist.get(length, 0)
            col = RED if length >= 5 else (NAVY if length >= 3 else DIM)
            bars.append((str(length), count, col))
        self._chain_bars.set_bars(bars)

    def on_game(self, evt: GameEvent):
        winner = evt.winner or 0
        self._wins[winner] = self._wins.get(winner, 0) + 1
        self._game_count += 1

        self._lbl_games.setText(f"games: {self._game_count}")
        self._lbl_win1.setText(f"P1 wins: {self._wins.get(1,0)}")
        self._lbl_win2.setText(f"P2 wins: {self._wins.get(2,0)}")
        self._lbl_draws.setText(f"timeout: {self._wins.get(0,0)}")

        for fp, cnt in evt.pattern_counts.items():
            self._pattern_freq[fp] = self._pattern_freq.get(fp, 0) + cnt
        self._lbl_patterns.setText(f"unique patterns: {len(self._pattern_freq)}")

        # Pair correlation sparkline
        corr_vals = [evt.correlation.get(r, 0.0) for r in range(1, 16)]
        self._corr_spark.set_data(corr_vals, baseline=1.0)

    def on_stats(self, stats: ExperimentStats):
        self._lbl_forks_c.setText(f"forks seen: {stats.total_forks_seen}")

    def reset(self):
        self._wins = {1: 0, 2: 0, 0: 0}
        self._game_count = 0
        self._all_forks = 0
        self._pattern_freq = defaultdict(int)
