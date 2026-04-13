"""
ThreatGraphWidget — network/hypergraph view of the current threat structure.

Nodes  = cells (occupied + candidate)
Edges  = cells sharing a live 6-window (co-membership in a potential winning line)
  - Edge weight = number of shared windows
  - Edge colour = axis (q/r/diag)
  - Node size    = number of live windows it belongs to (centrality)
  - Node colour  = player / empty / threat / fork

Layout: spring-force layout computed incrementally (Fruchterman-Reingold style),
running in-process (small graphs < 200 nodes are fast enough synchronously).
"""

from __future__ import annotations
import math
import random
from typing import Optional
from collections import defaultdict

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, QRectF, QSize, QTimer
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPaintEvent,
    QWheelEvent, QMouseEvent,
)

from engine import HexGame, AXES, WIN_LENGTH
from engine.analysis import live_lines


BG       = QColor("#050a0f")
P1_COL   = QColor("#e8e8e8")
P2_COL   = QColor("#003580")
EMPTY_COL= QColor("#1a2535")
THREAT_COL=QColor("#cc2200")
FORK_COL = QColor("#00b4d8")
EDGE_COLS = [
    QColor(160, 40, 40, 100),
    QColor(40, 130, 70, 100),
    QColor(60, 60, 180, 100),
]


class ThreatGraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(QSize(300, 300))

        self.game: Optional[HexGame] = None
        self.threats_p1: dict = {}
        self.threats_p2: dict = {}
        self.forks_p1: dict = {}
        self.forks_p2: dict = {}

        # Graph state
        self._nodes: dict[tuple, QPointF] = {}
        self._edges: list[tuple] = []   # [(cell_a, cell_b, axis_idx, weight)]
        self._node_sizes: dict[tuple, float] = {}

        # Layout
        self._layout_steps = 0
        self._layout_timer = QTimer(self)
        self._layout_timer.timeout.connect(self._layout_step)

        # Pan/zoom
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._drag_start: Optional[QPointF] = None
        self._drag_offset_start: Optional[QPointF] = None
        self.setMouseTracking(True)

        self._font = QFont()
        self._font.setFamilies(["Consolas", "Courier New", "monospace"])
        self._font.setPixelSize(11)

    def update_state(
        self, game: HexGame,
        threats_p1=None, threats_p2=None,
        forks_p1=None, forks_p2=None,
    ):
        self.game = game
        self.threats_p1 = threats_p1 or {}
        self.threats_p2 = threats_p2 or {}
        self.forks_p1   = forks_p1 or {}
        self.forks_p2   = forks_p2 or {}
        # Only rebuild graph topology every 4 moves to avoid thrashing layout
        move_count = len(game.move_history) if game else 0
        if not hasattr(self, '_last_rebuild_at'):
            self._last_rebuild_at = -1
        if move_count - self._last_rebuild_at >= 4 or move_count == 0:
            self._rebuild_graph()
            self._last_rebuild_at = move_count
        else:
            self.update()  # Just repaint with new threat/fork colours

    def _rebuild_graph(self):
        game = self.game
        if not game or not game.board:
            self._nodes = {}
            self._edges = []
            return

        lines = live_lines(game)

        # Only include cells with enough connectivity to be interesting
        membership: dict[tuple, int] = defaultdict(int)
        edge_count: dict[frozenset, list] = defaultdict(list)

        for cells, a_idx in lines:
            n_stones = sum(1 for c in cells if c in game.board)
            if n_stones < 1:
                continue
            for c in cells:
                membership[c] += 1
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    key = frozenset([cells[i], cells[j]])
                    edge_count[key].append(a_idx)

        # Keep top 120 nodes by membership
        top_cells = sorted(membership, key=lambda c: -membership[c])[:120]
        top_set = set(top_cells)

        # Initialise new nodes at random (preserve existing positions)
        # Guard against zero size during first paint before widget is shown
        w_px = self.width()  if self.width()  > 10 else 400
        h_px = self.height() if self.height() > 10 else 400
        cx, cy = w_px / 2, h_px / 2
        for cell in top_cells:
            if cell not in self._nodes:
                angle = random.uniform(0, 2 * math.pi)
                r_rand = random.uniform(20, min(cx, cy) * 0.8)
                self._nodes[cell] = QPointF(
                    cx + r_rand * math.cos(angle),
                    cy + r_rand * math.sin(angle),
                )

        # Remove old nodes
        for cell in list(self._nodes.keys()):
            if cell not in top_set:
                del self._nodes[cell]

        self._node_sizes = {c: min(3.0 + membership[c] * 0.25, 8.0) for c in top_cells}

        # Build edge list
        self._edges = []
        for key, axis_list in edge_count.items():
            pair = list(key)
            if pair[0] not in top_set or pair[1] not in top_set:
                continue
            a_idx = axis_list[0]
            w = len(axis_list)
            self._edges.append((pair[0], pair[1], a_idx, w))

        # Run layout
        self._layout_steps = 0
        self._layout_timer.start(16)

    def _layout_step(self):
        if not self._nodes or self._layout_steps > 80:
            self._layout_timer.stop()
            self.update()
            return

        nodes = list(self._nodes.keys())
        pos = self._nodes
        n = len(nodes)
        area = max(1.0, self.width() * self.height())
        k = math.sqrt(area / max(1, n)) * 0.6
        temp = k * max(0.1, 1.0 - self._layout_steps / 80)

        disp: dict[tuple, list] = {c: [0.0, 0.0] for c in nodes}

        # Repulsion
        for i in range(n):
            for j in range(i + 1, n):
                a, b = nodes[i], nodes[j]
                dx = pos[a].x() - pos[b].x()
                dy = pos[a].y() - pos[b].y()
                dist = math.sqrt(dx * dx + dy * dy) or 0.01
                force = k * k / dist
                fx, fy = dx / dist * force, dy / dist * force
                disp[a][0] += fx; disp[a][1] += fy
                disp[b][0] -= fx; disp[b][1] -= fy

        # Attraction along edges
        for (ca, cb, _, w) in self._edges:
            if ca not in pos or cb not in pos:
                continue
            dx = pos[ca].x() - pos[cb].x()
            dy = pos[ca].y() - pos[cb].y()
            dist = math.sqrt(dx * dx + dy * dy) or 0.01
            force = dist * dist / k * (1 + w * 0.3)
            fx, fy = dx / dist * force, dy / dist * force
            disp[ca][0] -= fx; disp[ca][1] -= fy
            disp[cb][0] += fx; disp[cb][1] += fy

        # Apply with temperature cap
        cx, cy = self.width() / 2, self.height() / 2
        margin = 30.0
        for cell in nodes:
            dx, dy = disp[cell]
            dist = math.sqrt(dx * dx + dy * dy) or 0.01
            scale = min(dist, temp) / dist
            nx = pos[cell].x() + dx * scale
            ny = pos[cell].y() + dy * scale
            nx = max(margin, min(self.width() - margin, nx))
            ny = max(margin, min(self.height() - margin, ny))
            pos[cell] = QPointF(nx, ny)

        self._layout_steps += 1
        if self._layout_steps % 8 == 0:
            self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BG)

        if not self._nodes:
            painter.setPen(QColor("#1a2535"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No game loaded")
            painter.end()
            return

        painter.translate(self._offset)
        painter.scale(self._scale, self._scale)

        # Draw edges (in scaled space — line width auto-corrected below)
        for (ca, cb, a_idx, w) in self._edges:
            if ca not in self._nodes or cb not in self._nodes:
                continue
            col = QColor(EDGE_COLS[a_idx % 3])
            col.setAlpha(min(200, 50 + w * 30))
            # Divide pen width by scale so lines stay thin regardless of zoom
            painter.setPen(QPen(col, max(0.5, w * 0.4) / self._scale))
            painter.drawLine(self._nodes[ca], self._nodes[cb])

        # Draw nodes in screen-space to avoid radii scaling with zoom.
        # We reset the transform per node and map the node position manually.
        painter.resetTransform()
        painter.setFont(self._font)

        def _to_screen(pt: QPointF) -> QPointF:
            return QPointF(
                pt.x() * self._scale + self._offset.x(),
                pt.y() * self._scale + self._offset.y(),
            )

        for cell, pt in self._nodes.items():
            sp = _to_screen(pt)
            player = self.game.board.get(cell) if self.game else None
            is_threat = cell in self.threats_p1 or cell in self.threats_p2
            is_fork   = cell in self.forks_p1   or cell in self.forks_p2
            radius = self._node_sizes.get(cell, 3.0)   # always in pixels

            if player == 1:
                col = P1_COL
            elif player == 2:
                col = P2_COL
            elif is_threat:
                col = THREAT_COL
            elif is_fork:
                col = FORK_COL
            else:
                col = EMPTY_COL

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(col))
            painter.drawEllipse(sp, radius, radius)

            if is_fork:
                painter.setPen(QPen(FORK_COL, 1.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(sp, radius + 2.5, radius + 2.5)

        painter.end()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        self._scale *= 1.12 if delta > 0 else 0.89
        self._scale = max(0.2, min(5.0, self._scale))
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
