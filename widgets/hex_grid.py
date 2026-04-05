"""
HexGridWidget — flat-top axial hex grid renderer.

Features:
  - Stones for P1 (white) and P2 (navy accent)
  - Threat overlay: red border on cells that complete a win
  - Fork overlay: cross marker on cells extending 2+ axes
  - Potential heatmap: background fill intensity = Erdos-Selfridge potential
  - Axis line overlay: highlight live 6-windows on each of 3 axes
  - Pan (drag) + zoom (wheel)
  - Last-move marker (filled ring)
  - Configurable: toggle each overlay layer independently
"""

from __future__ import annotations
import math
from typing import Optional

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath, QWheelEvent,
    QMouseEvent, QPaintEvent,
)

from engine import HexGame, AXES


# ── Colour palette ────────────────────────────────────────────────────────────
BG          = QColor("#050a0f")
GRID_LINE   = QColor("#1a2535")
P1_FILL     = QColor("#e8e8e8")
P2_FILL     = QColor("#003580")   # navy accent
P1_TEXT     = QColor("#050a0f")
P2_TEXT     = QColor("#e8e8e8")
THREAT_COL  = QColor("#cc2200")
FORK_COL    = QColor("#00b4d8")
LAST_MOVE   = QColor("#ffdd00")
POTENTIAL_HI= QColor(0, 53, 128, 120)   # navy, semi-transparent
AXIS_COLS   = [
    QColor(180, 50, 50, 80),    # q-axis — dim red
    QColor(50, 140, 80, 80),    # r-axis — dim green
    QColor(80, 80, 180, 80),    # diag   — dim blue
]
CAND_COL    = QColor("#0d1a2a")


class HexGridWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(QSize(400, 400))
        self.setMouseTracking(True)

        # State
        self.game: Optional[HexGame] = None
        self.threats_p1: dict = {}
        self.threats_p2: dict = {}
        self.forks_p1: dict = {}
        self.forks_p2: dict = {}
        self.potential: dict = {}
        self.last_move: Optional[tuple] = None

        # Layer toggles
        self.show_potential  = True
        self.show_threats    = True
        self.show_forks      = True
        self.show_axis_lines = True
        self.show_candidates = False
        self.show_coords     = False

        # View state
        self._cell_size = 22.0
        self._offset = QPointF(0.0, 0.0)
        self._drag_start: Optional[QPointF] = None
        self._drag_offset_start: Optional[QPointF] = None

        # Font — use pixel size to avoid -1pt when Consolas unavailable
        self._font = QFont()
        self._font.setFamilies(["Consolas", "Courier New", "monospace"])
        self._font.setPixelSize(11)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_state(
        self,
        game: HexGame,
        threats_p1: dict = None,
        threats_p2: dict = None,
        forks_p1: dict = None,
        forks_p2: dict = None,
        potential: dict = None,
        last_move: tuple = None,
    ):
        self.game = game
        self.threats_p1 = threats_p1 or {}
        self.threats_p2 = threats_p2 or {}
        self.forks_p1   = forks_p1 or {}
        self.forks_p2   = forks_p2 or {}
        self.potential  = potential or {}
        self.last_move  = last_move
        self.update()

    def reset_view(self):
        self._offset = QPointF(0.0, 0.0)
        self._cell_size = 22.0
        self.update()

    def center_on_board(self):
        if not self.game or not self.game.board:
            return
        qs = [q for q, r in self.game.board]
        rs = [r for q, r in self.game.board]
        cq = (min(qs) + max(qs)) / 2
        cr = (min(rs) + max(rs)) / 2
        cx, cy = self._axial_to_pixel(cq, cr)
        w, h = self.width() / 2, self.height() / 2
        self._offset = QPointF(w - cx, h - cy)
        self.update()

    # ── Coordinate transforms ─────────────────────────────────────────────────

    def _axial_to_pixel(self, q: float, r: float) -> tuple[float, float]:
        s = self._cell_size
        x = s * (3/2 * q) + self._offset.x()
        y = s * (math.sqrt(3)/2 * q + math.sqrt(3) * r) + self._offset.y()
        return x, y

    def _pixel_to_axial(self, x: float, y: float) -> tuple[float, float]:
        s = self._cell_size
        ox, oy = x - self._offset.x(), y - self._offset.y()
        q = (2/3 * ox) / s
        r = (-1/3 * ox + math.sqrt(3)/3 * oy) / s
        return q, r

    def _hex_corners(self, cx: float, cy: float) -> list[QPointF]:
        s = self._cell_size * 0.95
        return [
            QPointF(cx + s * math.cos(math.radians(60 * i)),
                    cy + s * math.sin(math.radians(60 * i)))
            for i in range(6)
        ]

    def _hex_path(self, cx: float, cy: float) -> QPainterPath:
        pts = self._hex_corners(cx, cy)
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        path.closeSubpath()
        return path

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BG)

        if not self.game:
            painter.setPen(QColor("#1a2535"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No game loaded")
            painter.end()
            return

        self._draw_cells(painter)
        painter.end()

    def _visible_bounds(self) -> tuple[int, int, int, int]:
        """Return approximate axial (q, r) bounds visible on screen."""
        margin = 3
        s = self._cell_size
        w, h = self.width(), self.height()
        q_range = int(w / s) + margin
        r_range = int(h / s) + margin
        cq, cr = self._pixel_to_axial(w / 2, h / 2)
        return (int(cq) - q_range, int(cq) + q_range,
                int(cr) - r_range, int(cr) + r_range)

    def _cells_to_draw(self) -> set[tuple[int, int]]:
        cells = set()
        if self.game:
            cells |= set(self.game.board.keys())
            cells |= set(self.game.candidates)
        cells |= set(self.threats_p1.keys())
        cells |= set(self.threats_p2.keys())
        cells |= set(self.forks_p1.keys())
        cells |= set(self.forks_p2.keys())
        if self.show_potential:
            cells |= set(self.potential.keys())
        return cells

    def _draw_cells(self, painter: QPainter):
        game = self.game
        max_pot = max(self.potential.values(), default=1.0) or 1.0

        # Axis line windows (drawn first, under stones)
        if self.show_axis_lines and game:
            self._draw_axis_lines(painter, game)

        for (q, r) in self._cells_to_draw():
            cx, cy = self._axial_to_pixel(q, r)

            # Cull off-screen
            if cx < -self._cell_size * 2 or cx > self.width() + self._cell_size * 2:
                continue
            if cy < -self._cell_size * 2 or cy > self.height() + self._cell_size * 2:
                continue

            path = self._hex_path(cx, cy)
            player = game.board.get((q, r)) if game else None
            is_candidate = (q, r) in (game.candidates if game else set())
            is_last = (q, r) == self.last_move

            # Background fill
            if player == 1:
                painter.fillPath(path, P1_FILL)
            elif player == 2:
                painter.fillPath(path, P2_FILL)
            else:
                # Potential heatmap
                if self.show_potential and (q, r) in self.potential:
                    alpha = int(200 * self.potential[(q, r)] / max_pot)
                    col = QColor(0, 53, 128, alpha)
                    painter.fillPath(path, col)
                elif is_candidate and self.show_candidates:
                    painter.fillPath(path, CAND_COL)

            # Grid outline
            pen = QPen(GRID_LINE, 0.5)
            painter.setPen(pen)
            painter.drawPath(path)

            # Threat border
            if self.show_threats:
                if (q, r) in self.threats_p1 or (q, r) in self.threats_p2:
                    pen = QPen(THREAT_COL, 2.0)
                    painter.setPen(pen)
                    painter.drawPath(path)

            # Fork cross
            if self.show_forks and ((q, r) in self.forks_p1 or (q, r) in self.forks_p2):
                self._draw_fork_marker(painter, cx, cy)

            # Last move ring
            if is_last:
                pen = QPen(LAST_MOVE, 1.5)
                painter.setPen(pen)
                r_ring = self._cell_size * 0.45
                painter.drawEllipse(QPointF(cx, cy), r_ring, r_ring)

            # Stone label (coords or move number)
            if player and self._cell_size >= 16:
                painter.setFont(self._font)
                painter.setPen(P1_TEXT if player == 1 else P2_TEXT)
                try:
                    idx = game.move_history.index((q, r))
                    label = str(idx + 1)
                except ValueError:
                    label = f"{q},{r}" if self.show_coords else ""
                if label:
                    painter.drawText(
                        QRectF(cx - 12, cy - 8, 24, 16),
                        Qt.AlignmentFlag.AlignCenter, label
                    )

    def _draw_axis_lines(self, painter: QPainter, game: HexGame):
        from engine.analysis import live_lines
        lines = live_lines(game)
        for cells, a_idx in lines:
            n_stones = sum(1 for c in cells if c in game.board)
            if n_stones < 2:
                continue
            col = QColor(AXIS_COLS[a_idx])
            col.setAlpha(min(200, 40 + n_stones * 30))
            pen = QPen(col, max(1.0, n_stones * 0.6))
            painter.setPen(pen)
            pts = [QPointF(*self._axial_to_pixel(q, r)) for q, r in cells]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

    def _draw_fork_marker(self, painter: QPainter, cx: float, cy: float):
        s = self._cell_size * 0.3
        pen = QPen(FORK_COL, 1.5)
        painter.setPen(pen)
        painter.drawLine(QPointF(cx - s, cy - s), QPointF(cx + s, cy + s))
        painter.drawLine(QPointF(cx + s, cy - s), QPointF(cx - s, cy + s))

    # ── Interaction ───────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 0.89
        pos = event.position()
        # Zoom toward cursor
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
