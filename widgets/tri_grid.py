"""
TriGridWidget — triangular lattice dual view of the same hex game state.

The triangular grid is the dual of the hexagonal grid:
  - Each hex cell becomes a vertex of the triangular lattice
  - Adjacent hex cells become connected vertices
  - This makes the lattice structure and axis directions explicit

Displays:
  - Vertices coloured by player occupancy
  - Edges along each of the three Z[omega] axes in distinct colours
  - Potential as vertex size
  - Fork vertices highlighted
  - The three axis directions as labelled reference lines
"""

from __future__ import annotations
import math
from typing import Optional

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QWheelEvent, QMouseEvent, QPaintEvent,
)

from engine import HexGame, AXES


BG        = QColor("#050a0f")
GRID_EDGE = QColor("#0d1a2a")
P1_COL    = QColor("#e8e8e8")
P2_COL    = QColor("#003580")
EMPTY_COL = QColor("#1a2535")
FORK_COL  = QColor("#00b4d8")

AXIS_EDGE_COLS = [
    QColor("#5a1010"),   # q-axis
    QColor("#104030"),   # r-axis
    QColor("#101050"),   # diag
]
AXIS_ACTIVE_COLS = [
    QColor("#cc3030"),
    QColor("#30aa60"),
    QColor("#3060cc"),
]


class TriGridWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(QSize(300, 300))
        self.setMouseTracking(True)

        self.game: Optional[HexGame] = None
        self.forks_p1: dict = {}
        self.forks_p2: dict = {}
        self.potential: dict = {}

        self.show_all_edges  = True
        self.show_axis_edges = True

        self._cell_size = 18.0
        self._offset = QPointF(0.0, 0.0)
        self._drag_start: Optional[QPointF] = None
        self._drag_offset_start: Optional[QPointF] = None

    def update_state(self, game: HexGame, forks_p1=None, forks_p2=None, potential=None):
        self.game = game
        self.forks_p1 = forks_p1 or {}
        self.forks_p2 = forks_p2 or {}
        self.potential = potential or {}
        self.update()

    def _axial_to_pixel(self, q: float, r: float) -> tuple[float, float]:
        # Same hex-to-pixel as hex grid (flat-top)
        s = self._cell_size
        x = s * (3/2 * q) + self._offset.x()
        y = s * (math.sqrt(3)/2 * q + math.sqrt(3) * r) + self._offset.y()
        return x, y

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BG)

        if not self.game or not self.game.board:
            painter.setPen(QColor("#1a2535"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No game loaded")
            painter.end()
            return

        self._draw_edges(painter)
        self._draw_vertices(painter)
        self._draw_axis_legend(painter)
        painter.end()

    def _draw_edges(self, painter: QPainter):
        game = self.game
        drawn: set[frozenset] = set()

        cells = set(game.board.keys()) | (game.candidates if self.show_all_edges else set())

        for (q, r) in cells:
            for a_idx, (dq, dr) in enumerate(AXES):
                for sign in (1, -1):
                    nb = (q + sign * dq, r + sign * dr)
                    edge_key = frozenset([(q, r), nb])
                    if edge_key in drawn:
                        continue
                    drawn.add(edge_key)

                    both_occupied = (q, r) in game.board and nb in game.board
                    if both_occupied:
                        col = AXIS_ACTIVE_COLS[a_idx]
                        w = 1.8
                    elif self.show_all_edges and (nb in cells):
                        col = AXIS_EDGE_COLS[a_idx]
                        w = 0.6
                    else:
                        continue

                    painter.setPen(QPen(col, w))
                    x1, y1 = self._axial_to_pixel(q, r)
                    x2, y2 = self._axial_to_pixel(*nb)
                    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _draw_vertices(self, painter: QPainter):
        game = self.game
        max_pot = max(self.potential.values(), default=1.0) or 1.0

        cells_to_draw = set(game.board.keys())
        if self.show_all_edges:
            cells_to_draw |= game.candidates

        for (q, r) in cells_to_draw:
            cx, cy = self._axial_to_pixel(q, r)
            if cx < -20 or cx > self.width() + 20:
                continue
            if cy < -20 or cy > self.height() + 20:
                continue

            player = game.board.get((q, r))
            is_fork = (q, r) in self.forks_p1 or (q, r) in self.forks_p2
            pot_val = self.potential.get((q, r), 0)

            # Vertex radius scales with potential for empty cells
            if player:
                radius = self._cell_size * 0.35
                col = P1_COL if player == 1 else P2_COL
            else:
                radius = self._cell_size * 0.12 + self._cell_size * 0.18 * (pot_val / max_pot)
                col = EMPTY_COL

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

            if is_fork:
                painter.setPen(QPen(FORK_COL, 1.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), radius + 2, radius + 2)

    def _draw_axis_legend(self, painter: QPainter):
        font = QFont("Consolas", 8)
        painter.setFont(font)
        labels = ["q-axis (1,0)", "r-axis (0,1)", "diag (1,-1)"]
        for i, (col, label) in enumerate(zip(AXIS_ACTIVE_COLS, labels)):
            y = 12 + i * 16
            painter.setPen(QPen(col, 2))
            painter.drawLine(QPointF(8, y), QPointF(28, y))
            painter.setPen(col)
            painter.drawText(QRectF(32, y - 6, 120, 12),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             label)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 0.89
        pos = event.position()
        ox = pos.x() - self._offset.x()
        oy = pos.y() - self._offset.y()
        self._cell_size = max(4.0, min(80.0, self._cell_size * factor))
        self._offset = QPointF(pos.x() - ox, pos.y() - oy)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position()
            self._drag_offset_start = QPointF(self._offset)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start is not None:
            delta = event.position() - self._drag_start
            self._offset = self._drag_offset_start + delta
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
