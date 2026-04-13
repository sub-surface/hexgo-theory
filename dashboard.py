"""
HexGo Theory — Research Suite
PySide6 desktop app.

Layout (Option B — split-panel):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Left sidebar              │  Right: Lichess-style viewer               │
  │  ─────────────────────     │  ────────────────────────────────────────   │
  │  [LAUNCH]                  │  board (hex/tri/threat tabs)  │ move list   │
  │    experiment selector     │                               │ + analysis  │
  │    controls                │  ─────────────────────────────┤             │
  │  [CORPUS]                  │  [log drawer — toggle]        │             │
  │    saved games list        │                               │             │
  └─────────────────────────────────────────────────────────────────────────┘

CLI:
  python dashboard.py --run eis_vs_eis --games 50 --verbose
"""

from __future__ import annotations
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSlider, QCheckBox,
    QSplitter, QTextEdit, QSizePolicy, QSpinBox, QTabWidget,
    QListWidget, QListWidgetItem, QFrame, QScrollArea, QToolButton,
    QStackedWidget, QAbstractItemView,
)
from PySide6.QtCore import Qt, QPointF, QSize, QTimer, Signal, QObject
from PySide6.QtGui import QColor, QPalette, QFont, QIcon

from widgets.hex_grid import HexGridWidget
from widgets.tri_grid import TriGridWidget
from widgets.threat_graph import ThreatGraphWidget
from widgets.analysis_panel import AnalysisPanel
from experiments.runner import (
    ExperimentWorker, ExperimentThread, MoveEvent, GameEvent, ExperimentStats,
)

GAMES_DIR = Path(__file__).parent / "games"
GAMES_DIR.mkdir(exist_ok=True)

EXPERIMENTS = {
    "Eisenstein vs Eisenstein": "eis_vs_eis",
    "Eisenstein vs Random":     "eis_vs_random",
    "Fork Hunt (A:off D:def)":  "fork_hunt",
    "Potential Landscape":      "potential_landscape",
    "Pattern Census":           "pattern_census",
}

# ── Stylesheet ─────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget {
    background: #050a0f;
    color: #c8d4e0;
}
QSplitter::handle { background: #0d1a2a; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical   { height: 1px; }
QComboBox {
    background: #0d1a2a; color: #c8d4e0;
    border: 1px solid #1a2535; padding: 2px 6px;
    border-radius: 0;
}
QComboBox QAbstractItemView {
    background: #0d1a2a; color: #c8d4e0;
    border: 1px solid #1a2535;
    selection-background-color: #003580;
}
QPushButton {
    background: #0d1a2a; color: #c8d4e0;
    border: 1px solid #1a2535; padding: 3px 10px;
    border-radius: 0; min-width: 52px;
}
QPushButton:hover  { background: #1a2535; }
QPushButton:pressed { background: #003580; }
QPushButton:disabled { color: #1a2535; border-color: #0d1a2a; }
QPushButton#run_btn { color: #e8e8e8; border-color: #003580; }
QPushButton#run_btn:hover { background: #003580; }
QPushButton#stop_btn { color: #cc2200; border-color: #5a1010; }
QToolButton {
    background: transparent; border: none; color: #3a4a5a;
    padding: 2px 4px;
}
QToolButton:hover  { color: #c8d4e0; }
QSlider::groove:horizontal { background: #0d1a2a; height: 3px; }
QSlider::handle:horizontal {
    background: #003580; width: 10px; height: 10px; margin: -4px 0;
}
QSlider::sub-page:horizontal { background: #003580; }
QCheckBox { color: #c8d4e0; spacing: 4px; }
QCheckBox::indicator {
    width: 11px; height: 11px;
    background: #0d1a2a; border: 1px solid #1a2535;
}
QCheckBox::indicator:checked { background: #003580; }
QTextEdit {
    background: #050a0f; color: #3a6a5a;
    border: none; border-top: 1px solid #0d1a2a; padding: 4px;
}
QSpinBox {
    background: #0d1a2a; color: #c8d4e0;
    border: 1px solid #1a2535; padding: 2px 4px;
    border-radius: 0; width: 55px;
}
QListWidget {
    background: #050a0f; color: #c8d4e0;
    border: none; outline: none;
}
QListWidget::item { padding: 3px 6px; border-bottom: 1px solid #080f18; }
QListWidget::item:selected { background: #0d1a2a; color: #003580; }
QListWidget::item:hover { background: #080f18; }
QScrollBar:vertical { background: #050a0f; width: 5px; }
QScrollBar::handle:vertical { background: #1a2535; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { height: 5px; background: #050a0f; }
QScrollBar::handle:horizontal { background: #1a2535; min-width: 20px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QTabWidget::pane { border: none; background: #050a0f; }
QTabBar::tab {
    background: #080f18; color: #3a4a5a;
    border: 1px solid #0d1a2a; border-bottom: none;
    padding: 3px 10px;
}
QTabBar::tab:selected { background: #050a0f; color: #003580; }
"""

SBG = "background: #080f18;"
SECTION_HDR_STYLE = (
    "background: #080f18; color: #003580; font-weight: bold; "
    "padding: 3px 6px; border-bottom: 1px solid #0d1a2a;"
)
DIM_STYLE = "color: #3a4a5a;"
PANEL_HDR = (
    "background: #060c14; color: #003580; font-weight: bold; "
    "padding: 2px 6px; border-bottom: 1px solid #0d1a2a;"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dim(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(DIM_STYLE)
    return lbl


def _section_hdr(title: str) -> QLabel:
    lbl = QLabel(title)
    lbl.setStyleSheet(SECTION_HDR_STYLE)
    return lbl


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: #0d1a2a;")
    f.setFixedHeight(1)
    return f


def _titled(title: str, widget: QWidget) -> QWidget:
    frame = QWidget()
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    hdr = QLabel(title)
    hdr.setStyleSheet(PANEL_HDR)
    lay.addWidget(hdr)
    lay.addWidget(widget)
    return frame


# ── Left Sidebar ───────────────────────────────────────────────────────────────

class LaunchSection(QWidget):
    """Experiment launcher controls."""
    run_requested  = Signal(dict)
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SBG)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        self._exp_combo = QComboBox()
        self._exp_combo.setMinimumWidth(10)
        for name in EXPERIMENTS:
            self._exp_combo.addItem(name)
        lay.addWidget(self._exp_combo)

        row1 = QHBoxLayout()
        row1.addWidget(_dim("games"))
        self._n_games = QSpinBox()
        self._n_games.setRange(1, 100000)
        self._n_games.setValue(20)
        row1.addWidget(self._n_games)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(_dim("step ms"))
        self._delay = QSlider(Qt.Orientation.Horizontal)
        self._delay.setRange(0, 500)
        self._delay.setValue(0)
        self._delay_lbl = QLabel("0")
        self._delay_lbl.setStyleSheet(DIM_STYLE)
        self._delay_lbl.setFixedWidth(26)
        self._delay.valueChanged.connect(lambda v: self._delay_lbl.setText(str(v)))
        row2.addWidget(self._delay)
        row2.addWidget(self._delay_lbl)
        lay.addLayout(row2)

        chk_row = QHBoxLayout()
        self._def_a = QCheckBox("Def-A")
        self._def_b = QCheckBox("Def-B")
        self._def_b.setChecked(True)
        chk_row.addWidget(self._def_a)
        chk_row.addWidget(self._def_b)
        chk_row.addStretch()
        lay.addLayout(chk_row)

        btn_row = QHBoxLayout()
        self._run_btn  = QPushButton("Run")
        self._run_btn.setObjectName("run_btn")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._stop_btn)
        lay.addLayout(btn_row)

        self._run_btn.clicked.connect(self._on_run)
        self._stop_btn.clicked.connect(self.stop_requested.emit)

    def _on_run(self):
        self.run_requested.emit({
            "experiment": EXPERIMENTS[self._exp_combo.currentText()],
            "exp_label":  self._exp_combo.currentText(),
            "n_games":    self._n_games.value(),
            "delay_ms":   self._delay.value(),
            "def_a":      self._def_a.isChecked(),
            "def_b":      self._def_b.isChecked(),
        })

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)


class CorpusSection(QWidget):
    """Corpus browser — lists saved game JSON files."""
    game_selected = Signal(str)   # path string

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SBG)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        btn_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setFixedWidth(70)
        btn_row.addWidget(self._refresh_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        lay.addWidget(self._list)

        self._refresh_btn.clicked.connect(self.refresh)
        self._list.itemDoubleClicked.connect(self._on_double_click)

        self.refresh()

    def refresh(self):
        self._list.clear()
        paths = sorted(GAMES_DIR.glob("*.json"), reverse=True)
        for p in paths:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self._list.addItem(item)

    def _on_double_click(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.game_selected.emit(path)


class Sidebar(QWidget):
    """Left sidebar: collapsible Launch + Corpus sections."""
    run_requested  = Signal(dict)
    stop_requested = Signal()
    game_selected  = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet("background: #060c14;")
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        lay.addWidget(_section_hdr("LAUNCH"))
        self._launch = LaunchSection()
        lay.addWidget(self._launch)

        lay.addWidget(_sep())
        lay.addWidget(_section_hdr("CORPUS"))
        self._corpus = CorpusSection()
        self._corpus.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self._corpus)

        self._launch.run_requested.connect(self.run_requested)
        self._launch.stop_requested.connect(self.stop_requested)
        self._corpus.game_selected.connect(self.game_selected)

    def set_running(self, running: bool):
        self._launch.set_running(running)

    def refresh_corpus(self):
        self._corpus.refresh()


# ── Overlay toggles bar ────────────────────────────────────────────────────────

class OverlayBar(QWidget):
    """Thin bar of checkboxes controlling hex grid overlays."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #060c14; border-bottom: 1px solid #0d1a2a;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(10)

        self.chk_potential  = QCheckBox("Potential")
        self.chk_threats    = QCheckBox("Threats")
        self.chk_forks      = QCheckBox("Forks")
        self.chk_axis_lines = QCheckBox("Axes")
        self.chk_candidates = QCheckBox("Cands")
        self.chk_coords     = QCheckBox("Coords")

        self.chk_potential.setChecked(True)
        self.chk_threats.setChecked(True)
        self.chk_forks.setChecked(True)
        self.chk_axis_lines.setChecked(True)

        self._center_btn = QPushButton("Center")
        self._center_btn.setFixedWidth(58)

        for w in (self.chk_potential, self.chk_threats, self.chk_forks,
                  self.chk_axis_lines, self.chk_candidates, self.chk_coords,
                  self._center_btn):
            lay.addWidget(w)
        lay.addStretch()

    def connect_hex_grid(self, grid: HexGridWidget):
        self.chk_potential.toggled.connect( lambda v: setattr(grid, 'show_potential',  v) or grid.update())
        self.chk_threats.toggled.connect(   lambda v: setattr(grid, 'show_threats',    v) or grid.update())
        self.chk_forks.toggled.connect(     lambda v: setattr(grid, 'show_forks',      v) or grid.update())
        self.chk_axis_lines.toggled.connect(lambda v: setattr(grid, 'show_axis_lines', v) or grid.update())
        self.chk_candidates.toggled.connect(lambda v: setattr(grid, 'show_candidates', v) or grid.update())
        self.chk_coords.toggled.connect(    lambda v: setattr(grid, 'show_coords',     v) or grid.update())
        self._center_btn.clicked.connect(grid.center_on_board)


# ── Move list widget ───────────────────────────────────────────────────────────

class MoveListWidget(QWidget):
    """Scrollable move list; click a move to jump to it."""
    move_selected = Signal(int)   # 0-based index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #050a0f;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QLabel("MOVES")
        hdr.setStyleSheet(PANEL_HDR)
        lay.addWidget(hdr)

        self._list = QListWidget()
        self._list.setStyleSheet("QListWidget { background: #050a0f; }")
        lay.addWidget(self._list)

        self._list.itemClicked.connect(
            lambda item: self.move_selected.emit(item.data(Qt.ItemDataRole.UserRole))
        )

    def load(self, move_history: list, player_history: list):
        self._list.clear()
        for i, ((q, r), p) in enumerate(zip(move_history, player_history)):
            text = f"{i+1:>3}.  P{p}  ({q:+d},{r:+d})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            if p == 1:
                item.setForeground(QColor("#c8d4e0"))
            else:
                item.setForeground(QColor("#6699cc"))
            self._list.addItem(item)

    def highlight(self, idx: int):
        if 0 <= idx < self._list.count():
            self._list.setCurrentRow(idx)
            self._list.scrollToItem(self._list.currentItem())

    def clear(self):
        self._list.clear()


# ── Viewer (right main area) ───────────────────────────────────────────────────

class ViewerPanel(QWidget):
    """
    Lichess-style viewer:
      - Board tabs (Hex / Tri / Threat)  center
      - Move list + Analysis panel       right column
      - Log drawer                       bottom (toggle)
      - Replay controls                  top bar (shown in replay mode)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._mode = "live"    # "live" | "replay"

        # Replay state
        self._replay_moves: list[tuple[int, int]] = []
        self._replay_players: list[int] = []
        self._replay_cursor = 0
        self._replay_timer = QTimer(self)
        self._replay_timer.setInterval(300)
        self._replay_timer.timeout.connect(self._replay_step_forward)

        # Live state
        self._pending_move: MoveEvent | None = None
        self._paint_timer = QTimer(self)
        self._paint_timer.setInterval(33)   # ~30fps
        self._paint_timer.timeout.connect(self._flush_live)
        self._live_attached = True

        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Overlay toggles bar (always visible)
        self._overlay_bar = OverlayBar()
        root.addWidget(self._overlay_bar)

        # Replay controls bar (hidden in live mode)
        self._replay_bar = self._build_replay_bar()
        self._replay_bar.setVisible(False)
        root.addWidget(self._replay_bar)

        # Main splitter: [board area] | [right column]
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(1)

        # Board area
        board_area = QWidget()
        board_v = QVBoxLayout(board_area)
        board_v.setContentsMargins(0, 0, 0, 0)
        board_v.setSpacing(0)

        self._board_tabs = QTabWidget()
        self._board_tabs.setDocumentMode(True)

        self._hex_grid     = HexGridWidget()
        self._tri_grid     = TriGridWidget()
        self._threat_graph = ThreatGraphWidget()

        self._board_tabs.addTab(self._hex_grid,     "HEX GRID")
        self._board_tabs.addTab(self._tri_grid,     "TRI GRID")
        self._board_tabs.addTab(self._threat_graph, "THREAT GRAPH")
        board_v.addWidget(self._board_tabs)

        # Log drawer (vertical splitter within board area)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(120)
        self._log.setVisible(False)

        self._log_toggle = QToolButton()
        self._log_toggle.setText("▲ LOG")
        self._log_toggle.setCheckable(True)
        self._log_toggle.setStyleSheet(
            "background: #060c14; color: #3a4a5a; border-top: 1px solid #0d1a2a; "
            "padding: 2px 6px; width: 100%;"
        )
        self._log_toggle.toggled.connect(self._toggle_log)
        board_v.addWidget(self._log_toggle)
        board_v.addWidget(self._log)

        # Right column: move list + analysis
        right_col = QSplitter(Qt.Orientation.Vertical)
        right_col.setHandleWidth(1)
        right_col.setMinimumWidth(200)
        right_col.setMaximumWidth(280)

        self._move_list = MoveListWidget()
        self._analysis  = AnalysisPanel()
        right_col.addWidget(self._move_list)
        right_col.addWidget(self._analysis)
        right_col.setSizes([300, 500])

        h_split.addWidget(board_area)
        h_split.addWidget(right_col)
        h_split.setSizes([900, 240])
        h_split.setStretchFactor(0, 1)
        h_split.setStretchFactor(1, 0)

        root.addWidget(h_split)

        # Wire overlays → hex grid
        self._overlay_bar.connect_hex_grid(self._hex_grid)

        # Move list click → replay seek
        self._move_list.move_selected.connect(self._replay_seek)

    def _build_replay_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background: #060c14; border-bottom: 1px solid #0d1a2a;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(6)

        self._rp_first = QPushButton("⏮")
        self._rp_prev  = QPushButton("◀")
        self._rp_next  = QPushButton("▶")
        self._rp_last  = QPushButton("⏭")
        self._rp_play  = QPushButton("Play")
        self._rp_stop  = QPushButton("Pause")

        self._rp_speed = QSlider(Qt.Orientation.Horizontal)
        self._rp_speed.setRange(50, 1500)
        self._rp_speed.setValue(300)
        self._rp_speed.setFixedWidth(80)
        self._rp_speed.valueChanged.connect(lambda v: self._replay_timer.setInterval(v))

        self._rp_pos = QLabel("—")
        self._rp_pos.setStyleSheet(DIM_STYLE)

        self._live_btn = QPushButton("Live")
        self._live_btn.setToolTip("Switch back to live experiment view")

        for w in (self._rp_first, self._rp_prev, self._rp_next, self._rp_last,
                  self._rp_play, self._rp_stop, _dim("speed"), self._rp_speed,
                  self._rp_pos):
            lay.addWidget(w)
        lay.addStretch()
        lay.addWidget(self._live_btn)

        self._rp_first.clicked.connect(lambda: self._replay_seek(0))
        self._rp_prev.clicked.connect(self._replay_step_back)
        self._rp_next.clicked.connect(self._replay_step_forward)
        self._rp_last.clicked.connect(lambda: self._replay_seek(len(self._replay_moves)))
        self._rp_play.clicked.connect(self._replay_timer.start)
        self._rp_stop.clicked.connect(self._replay_timer.stop)
        self._live_btn.clicked.connect(self.switch_to_live)

        return bar

    def _toggle_log(self, checked: bool):
        self._log.setVisible(checked)
        self._log_toggle.setText("▼ LOG" if checked else "▲ LOG")

    # ── Mode switching ────────────────────────────────────────────────────────

    def switch_to_live(self):
        self._mode = "live"
        self._replay_timer.stop()
        self._replay_bar.setVisible(False)

    def switch_to_replay(self):
        self._mode = "replay"
        self._replay_bar.setVisible(True)

    # ── Live board ────────────────────────────────────────────────────────────

    def start_live(self):
        self._paint_timer.start()
        self._live_attached = True

    def stop_live(self):
        self._paint_timer.stop()
        self._flush_live()

    def on_move(self, evt: MoveEvent):
        self._pending_move = evt
        self._analysis.on_move(evt)

    def on_game(self, evt: GameEvent):
        self._analysis.on_game(evt)

    def on_stats(self, stats: ExperimentStats):
        self._analysis.on_stats(stats)

    def _flush_live(self):
        if self._mode != "live":
            return
        evt = self._pending_move
        if evt is None:
            return
        self._pending_move = None
        self._update_board(
            game=evt.game,
            threats_p1=evt.threats_p1,
            threats_p2=evt.threats_p2,
            forks_p1=evt.forks_p1,
            forks_p2=evt.forks_p2,
            potential=evt.potential,
            last_move=evt.move,
        )

    def _update_board(self, game, threats_p1, threats_p2, forks_p1, forks_p2,
                      potential, last_move):
        self._hex_grid.update_state(
            game=game,
            threats_p1=threats_p1, threats_p2=threats_p2,
            forks_p1=forks_p1, forks_p2=forks_p2,
            potential=potential, last_move=last_move,
        )
        # Only update inactive tabs if they're cheap
        if self._board_tabs.currentIndex() == 1:
            self._tri_grid.update_state(
                game=game, forks_p1=forks_p1, forks_p2=forks_p2, potential=potential,
            )
        if self._board_tabs.currentIndex() == 2:
            self._threat_graph.update_state(
                game=game,
                threats_p1=threats_p1, threats_p2=threats_p2,
                forks_p1=forks_p1, forks_p2=forks_p2,
            )

    # ── Replay ────────────────────────────────────────────────────────────────

    def load_game_file(self, path: str):
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:
            self.append_log(f"[error] cannot load {path}: {e}")
            return
        moves   = [tuple(m) for m in data.get("moves", [])]
        players = data.get("players", [])
        self._load_replay(moves, players)

    def load_game_event(self, evt: GameEvent):
        """Load a just-completed game into the replay viewer."""
        players = getattr(evt, 'player_history', [])
        # Reconstruct from move_history + players if player_history absent
        if not players:
            # Try to reconstruct: alternate per Eisenstein rules
            # We can't perfectly reconstruct without player_history so skip
            players = []
        self._load_replay(list(evt.move_history), players)

    def _load_replay(self, moves: list, players: list):
        self._replay_moves   = moves
        self._replay_players = players
        self._replay_cursor  = 0
        self._move_list.load(moves, players if players else [0]*len(moves))
        self.switch_to_replay()
        self._replay_seek(len(moves))   # jump to end by default

    def _replay_seek(self, idx: int):
        self._replay_timer.stop()
        n = max(0, min(idx, len(self._replay_moves)))
        self._replay_cursor = n

        from engine import HexGame
        from engine.analysis import fork_cells, potential_map, threat_cells
        g = HexGame()
        for move in self._replay_moves[:n]:
            g.make(*move)

        fk1 = fork_cells(g, 1)
        fk2 = fork_cells(g, 2)
        pot = potential_map(g)
        th1 = threat_cells(g, 1)
        th2 = threat_cells(g, 2)

        self._update_board(
            game=g,
            threats_p1=th1, threats_p2=th2,
            forks_p1=fk1, forks_p2=fk2,
            potential=pot,
            last_move=self._replay_moves[n - 1] if n > 0 else None,
        )
        self._rp_pos.setText(f"move {n}/{len(self._replay_moves)}")
        self._move_list.highlight(n - 1)

    def _replay_step_forward(self):
        if self._replay_cursor < len(self._replay_moves):
            self._replay_seek(self._replay_cursor + 1)
        else:
            self._replay_timer.stop()

    def _replay_step_back(self):
        if self._replay_cursor > 0:
            self._replay_seek(self._replay_cursor - 1)

    # ── Log ───────────────────────────────────────────────────────────────────

    def append_log(self, line: str):
        self._log.append(line)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def clear_log(self):
        self._log.clear()

    def reset_analysis(self):
        self._analysis.reset()

    def clear_move_list(self):
        self._move_list.clear()


# ── Experiment controller ──────────────────────────────────────────────────────

class ExperimentController(QObject):
    """
    Owns the QThread/worker lifecycle.
    Emits signals that both the viewer and the status bar consume.
    """
    move_ready = Signal(object)     # MoveEvent
    game_done  = Signal(object)     # GameEvent
    log_line   = Signal(str)
    finished   = Signal(object)     # ExperimentStats
    error      = Signal(str)
    running_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: ExperimentThread | None = None
        self._worker: ExperimentWorker | None = None

    def start(self, cfg: dict):
        if self._thread is not None and self._thread.isRunning():
            return
        if self._thread is not None:
            self._thread.wait()
            self._thread = None
            self._worker = None

        worker = ExperimentWorker(
            experiment=cfg["experiment"],
            n_games=cfg["n_games"],
            step_delay_ms=cfg["delay_ms"],
            agent_a_defensive=cfg["def_a"],
            agent_b_defensive=cfg["def_b"],
        )
        thread = ExperimentThread(worker)

        worker.move_ready.connect(self.move_ready)
        worker.game_done.connect(self.game_done)
        worker.log_line.connect(self.log_line)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)

        self._worker = worker
        self._thread = thread
        self.running_changed.emit(True)
        thread.start()

    def stop(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()

    def _on_finished(self, stats: ExperimentStats):
        self.finished.emit(stats)
        self.running_changed.emit(False)

    def _on_error(self, msg: str):
        self.error.emit(msg)
        self.running_changed.emit(False)

    def cleanup(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._thread.quit()
            self._thread.wait(3000)


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HexGo Theory — Research Suite")
        self.resize(1400, 880)
        self.setStyleSheet(STYLE)

        self._last_game_evt: GameEvent | None = None
        self._ctrl = ExperimentController(self)

        self._sidebar = Sidebar()
        self._viewer  = ViewerPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._viewer)
        splitter.setSizes([220, 1180])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        self.statusBar().setStyleSheet(
            "background: #060c14; border-top: 1px solid #0d1a2a; color: #3a4a5a;"
        )

        self._wire()

    def _wire(self):
        sb = self._sidebar
        v  = self._viewer
        c  = self._ctrl

        sb.run_requested.connect(self._on_run)
        sb.stop_requested.connect(c.stop)
        sb.game_selected.connect(self._on_corpus_game_selected)

        c.move_ready.connect(v.on_move)
        c.game_done.connect(self._on_game_done)
        c.log_line.connect(v.append_log)
        c.finished.connect(self._on_finished)
        c.error.connect(self._on_error)
        c.running_changed.connect(sb.set_running)
        c.running_changed.connect(self._on_running_changed)

    def _on_run(self, cfg: dict):
        self._viewer.switch_to_live()
        self._viewer.reset_analysis()
        self._viewer.clear_log()
        self._viewer.clear_move_list()
        self._viewer.append_log(f"▶ {cfg['exp_label']}  ({cfg['n_games']} games)")
        self._viewer.start_live()
        self._ctrl.start(cfg)

    def _on_running_changed(self, running: bool):
        if not running:
            self._viewer.stop_live()

    def _on_game_done(self, evt: GameEvent):
        self._last_game_evt = evt
        self._viewer.on_game(evt)
        self._save_game(evt)
        w = evt.winner
        tag = f"P{w}" if w else "timeout"
        self.statusBar().showMessage(
            f"game {evt.game_number} | winner: {tag} | moves: {evt.move_count}"
        )

    def _on_finished(self, stats: ExperimentStats):
        self._viewer.on_stats(stats)
        self._sidebar.refresh_corpus()
        w1 = stats.wins.get(1, 0)
        w2 = stats.wins.get(2, 0)
        wd = stats.wins.get(0, 0)
        self.statusBar().showMessage(
            f"done | P1: {w1}  P2: {w2}  timeout: {wd} | patterns: {len(stats.pattern_freq)}"
        )
        self._viewer.append_log("■ experiment complete")

    def _on_error(self, msg: str):
        self._viewer.append_log(f"[ERROR] {msg}")
        self.statusBar().showMessage(f"error: {msg}")

    def _on_corpus_game_selected(self, path: str):
        self._viewer.load_game_file(path)

    def _save_game(self, evt: GameEvent):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        w  = evt.winner or 0
        fname = f"{ts}_game{evt.game_number:04d}_w{w}.json"
        data = {
            "game_number":  evt.game_number,
            "winner":       evt.winner,
            "move_count":   evt.move_count,
            "duration":     evt.duration,
            "moves":        list(evt.move_history),
            "players":      getattr(evt, 'player_history', []),
            "correlation":  {str(k): v for k, v in evt.correlation.items()},
            "pattern_counts": evt.pattern_counts,
        }
        try:
            (GAMES_DIR / fname).write_text(json.dumps(data, indent=2))
        except Exception as e:
            self._viewer.append_log(f"[warn] could not save game: {e}")

    def closeEvent(self, event):
        self._ctrl.cleanup()
        event.accept()


# ── CLI headless mode ──────────────────────────────────────────────────────────

def run_headless(args):
    """Run an experiment in the terminal with rich log output. No GUI."""
    from engine import HexGame, EisensteinGreedyAgent, RandomAgent
    from engine.analysis import (
        live_lines, threat_cells, fork_cells, potential_map,
        live_ap_count, pair_correlation, pattern_fingerprint,
    )
    from collections import defaultdict

    exp_key = args.run
    n_games = args.games
    verbose = args.verbose

    agent_map = {
        "eis_vs_eis":          (EisensteinGreedyAgent("Eis-A", defensive=False),
                                EisensteinGreedyAgent("Eis-B", defensive=True)),
        "eis_vs_random":       (EisensteinGreedyAgent("Eis-A", defensive=False),
                                RandomAgent()),
        "fork_hunt":           (EisensteinGreedyAgent("Eis-A", defensive=False),
                                EisensteinGreedyAgent("Eis-B", defensive=True)),
        "potential_landscape": (EisensteinGreedyAgent("Eis-A", defensive=False),
                                EisensteinGreedyAgent("Eis-B", defensive=True)),
        "pattern_census":      (EisensteinGreedyAgent("Eis-A", defensive=False),
                                EisensteinGreedyAgent("Eis-B", defensive=True)),
    }
    agent_a, agent_b = agent_map.get(exp_key, agent_map["eis_vs_eis"])
    print(f"[hexgo] {exp_key} | {n_games} games | {agent_a.name} vs {agent_b.name}")

    wins = {1: 0, 2: 0, 0: 0}
    total_moves = 0

    for g_idx in range(n_games):
        game = HexGame()
        agents = {1: agent_a, 2: agent_b}
        if g_idx % 2 == 1:
            agents = {1: agent_b, 2: agent_a}

        t0 = time.perf_counter()
        move_count = 0

        while game.winner is None and move_count < 300:
            move = agents[game.current_player].choose_move(game)
            player = game.current_player
            game.make(*move)
            move_count += 1

            if verbose >= 2:
                f1 = fork_cells(game, 1)
                f2 = fork_cells(game, 2)
                th1 = threat_cells(game, 1)
                th2 = threat_cells(game, 2)
                print(f"  move {move_count:>3} P{player} {move}  "
                      f"threats=({len(th1)},{len(th2)})  forks=({len(f1)},{len(f2)})")

        dur = time.perf_counter() - t0
        w = game.winner or 0
        wins[w] = wins.get(w, 0) + 1
        total_moves += move_count

        if verbose >= 1:
            w_str = f"P{w}" if w else "draw/timeout"
            print(f"game {g_idx+1:>4}/{n_games}  winner={w_str}  moves={move_count}  {dur:.3f}s")

        # Save game
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_game{g_idx+1:04d}_w{w}.json"
        corr = pair_correlation(game.move_history, max_r=15)
        pats = pattern_fingerprint(game, radius=2)
        pat_counts: dict[str, int] = defaultdict(int)
        for fp in pats.values():
            pat_counts[fp] += 1
        data = {
            "game_number": g_idx + 1,
            "winner": game.winner,
            "move_count": move_count,
            "duration": dur,
            "moves": list(game.move_history),
            "players": list(game.player_history) if hasattr(game, 'player_history') else [],
            "correlation": {str(k): v for k, v in corr.items()},
            "pattern_counts": dict(pat_counts),
        }
        try:
            (GAMES_DIR / fname).write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    print(f"\n[done] wins={wins}  avg_moves={total_moves/max(1,n_games):.1f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HexGo Theory Research Suite")
    parser.add_argument("--run", metavar="EXP",
                        help="Run experiment headlessly (no GUI). "
                             "Values: eis_vs_eis, eis_vs_random, fork_hunt, "
                             "potential_landscape, pattern_census")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of games to run (default: 20)")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="-v for game summaries, -vv for per-move output")
    args = parser.parse_args()

    if args.run:
        run_headless(args)
        return

    app = QApplication(sys.argv)
    app.setApplicationName("HexGo Theory")

    app_font = QFont()
    app_font.setFamilies(["Consolas", "Courier New", "monospace"])
    app_font.setPointSize(9)
    app.setFont(app_font)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor("#050a0f"))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor("#c8d4e0"))
    palette.setColor(QPalette.ColorRole.Base,            QColor("#080f18"))
    palette.setColor(QPalette.ColorRole.Text,            QColor("#c8d4e0"))
    palette.setColor(QPalette.ColorRole.Button,          QColor("#0d1a2a"))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#c8d4e0"))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor("#003580"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#e8e8e8"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
