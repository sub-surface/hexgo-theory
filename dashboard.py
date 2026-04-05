"""
HexGo Theory Dashboard — PySide6 desktop app.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │  toolbar: experiment selector, controls, step delay, toggles     │
  ├──────────────────────────────────────────────────────────────────┤
  │  QTabWidget:                                                     │
  │    [Live]   hex / tri / threat / analysis  +  log               │
  │    [Replay] saved-game stepper                                   │
  └──────────────────────────────────────────────────────────────────┘
"""

import sys
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QLabel, QComboBox, QPushButton, QSlider, QCheckBox,
    QSplitter, QTextEdit, QSizePolicy, QSpinBox, QTabWidget,
    QFileDialog, QScrollArea,
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QColor, QPalette, QFont

from widgets.hex_grid import HexGridWidget
from widgets.tri_grid import TriGridWidget
from widgets.threat_graph import ThreatGraphWidget
from widgets.analysis_panel import AnalysisPanel
from experiments.runner import (
    ExperimentWorker, ExperimentThread, MoveEvent, GameEvent, ExperimentStats,
)


# ── Stylesheet ────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget {
    background: #050a0f;
    color: #c8d4e0;
}
QTabWidget::pane {
    border: none;
    background: #050a0f;
}
QTabBar::tab {
    background: #080f18;
    color: #3a4a5a;
    border: 1px solid #0d1a2a;
    border-bottom: none;
    padding: 3px 14px;
}
QTabBar::tab:selected {
    background: #050a0f;
    color: #003580;
    border-bottom: 1px solid #050a0f;
}
QToolBar {
    background: #080f18;
    border-bottom: 1px solid #0d1a2a;
    spacing: 4px;
    padding: 3px 6px;
}
QComboBox {
    background: #0d1a2a;
    color: #c8d4e0;
    border: 1px solid #1a2535;
    padding: 2px 6px;
    min-width: 140px;
    border-radius: 0;
}
QComboBox QAbstractItemView {
    background: #0d1a2a;
    color: #c8d4e0;
    border: 1px solid #1a2535;
    selection-background-color: #003580;
}
QPushButton {
    background: #0d1a2a;
    color: #c8d4e0;
    border: 1px solid #1a2535;
    padding: 2px 10px;
    border-radius: 0;
    min-width: 52px;
}
QPushButton:hover { background: #1a2535; }
QPushButton:pressed { background: #003580; }
QPushButton:disabled { color: #1a2535; border-color: #0d1a2a; }
QPushButton#run_btn  { color: #e8e8e8; border-color: #003580; }
QPushButton#run_btn:hover  { background: #003580; }
QPushButton#stop_btn { color: #cc2200; border-color: #5a1010; }
QSlider::groove:horizontal { background: #0d1a2a; height: 3px; }
QSlider::handle:horizontal {
    background: #003580; width: 10px; height: 10px; margin: -4px 0;
}
QSlider::sub-page:horizontal { background: #003580; }
QCheckBox { color: #c8d4e0; spacing: 4px; }
QCheckBox::indicator {
    width: 11px; height: 11px;
    background: #0d1a2a;
    border: 1px solid #1a2535;
}
QCheckBox::indicator:checked { background: #003580; }
QTextEdit {
    background: #050a0f;
    color: #3a6a5a;
    border: none;
    border-top: 1px solid #0d1a2a;
    padding: 4px;
}
QSplitter::handle { background: #0d1a2a; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical   { height: 1px; }
QSpinBox {
    background: #0d1a2a; color: #c8d4e0;
    border: 1px solid #1a2535; padding: 2px 4px;
    border-radius: 0; width: 55px;
}
QScrollBar:vertical { background: #050a0f; width: 5px; }
QScrollBar::handle:vertical { background: #1a2535; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

# Panel header: no font-size — inherits app font
PANEL_HDR = (
    "background: #080f18; color: #003580; "
    "font-weight: bold; "
    "padding: 2px 6px; border-bottom: 1px solid #0d1a2a;"
)

EXPERIMENTS = {
    "Eisenstein vs Eisenstein": "eis_vs_eis",
    "Eisenstein vs Random":     "eis_vs_random",
    "Fork Hunt (Eis A vs D)":   "fork_hunt",
    "Potential Landscape":      "potential_landscape",
    "Pattern Census":           "pattern_census",
}


def _toolbar_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "color: #3a4a5a; padding: 0 3px;"
    )
    return lbl


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


# ── Replay tab ────────────────────────────────────────────────────────────────

class ReplayTab(QWidget):
    """Load a saved game (list of moves as JSON) and step through it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._moves: list[tuple[int, int]] = []
        self._cursor = 0
        self._game = None
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._step_forward)

        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Controls bar
        ctrl = QWidget()
        ctrl.setStyleSheet("background: #080f18; border-bottom: 1px solid #0d1a2a;")
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(6, 3, 6, 3)
        ctrl_lay.setSpacing(6)

        self._load_btn  = QPushButton("Load game…")
        self._prev_btn  = QPushButton("◀")
        self._next_btn  = QPushButton("▶")
        self._play_btn  = QPushButton("Play")
        self._stop_btn2 = QPushButton("Stop")
        self._speed_sl  = QSlider(Qt.Orientation.Horizontal)
        self._speed_sl.setRange(50, 2000)
        self._speed_sl.setValue(200)
        self._speed_sl.setFixedWidth(80)
        self._speed_sl.valueChanged.connect(
            lambda v: self._timer.setInterval(v)
        )
        self._pos_lbl = QLabel("—")
        self._pos_lbl.setStyleSheet("color: #3a4a5a;")

        for w in (self._load_btn, self._prev_btn, self._next_btn,
                  self._play_btn, self._stop_btn2,
                  _toolbar_label("speed"), self._speed_sl, self._pos_lbl):
            ctrl_lay.addWidget(w)
        ctrl_lay.addStretch()
        lay.addWidget(ctrl)

        # Panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        self._hex  = HexGridWidget()
        self._tri  = TriGridWidget()
        splitter.addWidget(_titled("HEX GRID", self._hex))
        splitter.addWidget(_titled("TRI GRID", self._tri))
        splitter.setSizes([500, 500])
        lay.addWidget(splitter)

        # Wire
        self._load_btn.clicked.connect(self._load_game)
        self._prev_btn.clicked.connect(self._step_back)
        self._next_btn.clicked.connect(self._step_forward)
        self._play_btn.clicked.connect(self._timer.start)
        self._stop_btn2.clicked.connect(self._timer.stop)

    def _load_game(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load game", str(Path.home()), "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            # Accept either a list of [q, r] pairs or a dict with "moves" key
            if isinstance(data, list):
                self._moves = [tuple(m) for m in data]
            elif isinstance(data, dict):
                self._moves = [tuple(m) for m in data.get("moves", [])]
            else:
                return
            self._cursor = 0
            self._replay_to(0)
        except Exception as e:
            print(f"[replay] load error: {e}")

    def load_moves(self, moves: list[tuple[int, int]]):
        """Called from Live tab to load last completed game."""
        self._moves = list(moves)
        self._cursor = 0
        self._replay_to(0)

    def _replay_to(self, n: int):
        from engine import HexGame
        from engine.analysis import fork_cells, potential_map
        g = HexGame()
        for move in self._moves[:n]:
            g.make(*move)
        self._game = g
        self._hex.update_state(
            game=g,
            forks_p1=fork_cells(g, 1),
            forks_p2=fork_cells(g, 2),
            potential=potential_map(g),
            last_move=self._moves[n - 1] if n > 0 else None,
        )
        self._tri.update_state(
            game=g,
            forks_p1=fork_cells(g, 1),
            forks_p2=fork_cells(g, 2),
            potential=potential_map(g),
        )
        self._pos_lbl.setText(f"move {n}/{len(self._moves)}")

    def _step_forward(self):
        if self._cursor < len(self._moves):
            self._cursor += 1
            self._replay_to(self._cursor)
        else:
            self._timer.stop()

    def _step_back(self):
        if self._cursor > 0:
            self._cursor -= 1
            self._replay_to(self._cursor)


# ── Live tab ──────────────────────────────────────────────────────────────────

class LiveTab(QWidget):
    def __init__(self, replay_tab: ReplayTab, parent=None):
        super().__init__(parent)
        self._replay_tab = replay_tab
        self._thread: ExperimentThread | None = None
        self._worker: ExperimentWorker | None = None
        self._move_count = 0
        self._last_game_moves: list[tuple[int, int]] = []

        self._pending_move: MoveEvent | None = None
        self._paint_timer = QTimer(self)
        self._paint_timer.setInterval(33)
        self._paint_timer.timeout.connect(self._flush_move)

        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Toolbar
        tb_widget = QWidget()
        tb_widget.setStyleSheet("background: #080f18; border-bottom: 1px solid #0d1a2a;")
        tb_lay = QHBoxLayout(tb_widget)
        tb_lay.setContentsMargins(6, 3, 6, 3)
        tb_lay.setSpacing(6)

        tb_lay.addWidget(_toolbar_label("EXPERIMENT"))
        self._exp_combo = QComboBox()
        for name in EXPERIMENTS:
            self._exp_combo.addItem(name)
        tb_lay.addWidget(self._exp_combo)

        tb_lay.addWidget(_toolbar_label("GAMES"))
        self._n_games_spin = QSpinBox()
        self._n_games_spin.setRange(1, 10000)
        self._n_games_spin.setValue(20)
        tb_lay.addWidget(self._n_games_spin)

        tb_lay.addWidget(_toolbar_label("STEP ms"))
        self._delay_slider = QSlider(Qt.Orientation.Horizontal)
        self._delay_slider.setRange(0, 500)
        self._delay_slider.setValue(0)
        self._delay_slider.setFixedWidth(80)
        self._delay_label = QLabel("0")
        self._delay_label.setStyleSheet("color:#3a4a5a; min-width:24px;")
        self._delay_slider.valueChanged.connect(
            lambda v: self._delay_label.setText(str(v))
        )
        tb_lay.addWidget(self._delay_slider)
        tb_lay.addWidget(self._delay_label)

        self._def_a_chk = QCheckBox("Def-A")
        self._def_b_chk = QCheckBox("Def-B")
        self._def_b_chk.setChecked(True)
        tb_lay.addWidget(self._def_a_chk)
        tb_lay.addWidget(self._def_b_chk)

        # Overlay toggles
        self._chk_potential  = QCheckBox("Potential")
        self._chk_threats    = QCheckBox("Threats")
        self._chk_forks      = QCheckBox("Forks")
        self._chk_axislines  = QCheckBox("Axes")
        self._chk_candidates = QCheckBox("Cands")
        for chk in (self._chk_potential, self._chk_threats,
                    self._chk_forks, self._chk_axislines):
            chk.setChecked(True)
            tb_lay.addWidget(chk)
        tb_lay.addWidget(self._chk_candidates)

        self._center_btn = QPushButton("Center")
        self._load_replay_btn = QPushButton("→ Replay")
        self._load_replay_btn.setToolTip("Send last completed game to Replay tab")
        tb_lay.addWidget(self._center_btn)
        tb_lay.addWidget(self._load_replay_btn)

        tb_lay.addStretch()

        self._run_btn  = QPushButton("Run")
        self._run_btn.setObjectName("run_btn")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.setEnabled(False)
        tb_lay.addWidget(self._run_btn)
        tb_lay.addWidget(self._stop_btn)

        lay.addWidget(tb_widget)

        # Panels
        self._hex_grid     = HexGridWidget()
        self._tri_grid     = TriGridWidget()
        self._threat_graph = ThreatGraphWidget()
        self._analysis     = AnalysisPanel()

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setHandleWidth(1)
        top_splitter.addWidget(_titled("HEX GRID",    self._hex_grid))
        top_splitter.addWidget(_titled("TRI GRID",    self._tri_grid))
        top_splitter.addWidget(_titled("THREAT GRAPH",self._threat_graph))
        top_splitter.addWidget(self._analysis)
        top_splitter.setSizes([420, 300, 300, 220])

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(110)

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setHandleWidth(1)
        v_splitter.addWidget(top_splitter)
        v_splitter.addWidget(self._log)
        v_splitter.setSizes([760, 110])

        lay.addWidget(v_splitter)

        # Wire
        self._run_btn.clicked.connect(self._start_experiment)
        self._stop_btn.clicked.connect(self._stop_experiment)
        self._load_replay_btn.clicked.connect(self._send_to_replay)
        self._center_btn.clicked.connect(self._center_all)

        self._chk_potential.toggled.connect( lambda v: self._overlay("potential",  v))
        self._chk_threats.toggled.connect(   lambda v: self._overlay("threats",    v))
        self._chk_forks.toggled.connect(     lambda v: self._overlay("forks",      v))
        self._chk_axislines.toggled.connect( lambda v: self._overlay("axis_lines", v))
        self._chk_candidates.toggled.connect(lambda v: self._overlay("candidates", v))

    def _overlay(self, name: str, value: bool):
        setattr(self._hex_grid, f"show_{name}", value)
        self._hex_grid.update()

    def _center_all(self):
        self._hex_grid.center_on_board()
        self._tri_grid._offset.setX(self._hex_grid._offset.x())
        self._tri_grid._offset.setY(self._hex_grid._offset.y())
        self._tri_grid._cell_size = self._hex_grid._cell_size
        self._tri_grid.update()

    def _send_to_replay(self):
        if self._last_game_moves:
            self._replay_tab.load_moves(self._last_game_moves)

    # ── Experiment control ────────────────────────────────────────────────────

    def _start_experiment(self):
        # Clean up any previous (finished) thread before creating a new one
        if self._thread is not None:
            if self._thread.isRunning():
                return
            self._thread = None
            self._worker = None

        exp_name = self._exp_combo.currentText()
        exp_key  = EXPERIMENTS[exp_name]

        self._analysis.reset()
        self._move_count = 0
        self._log.clear()
        self._log.append(f"▶ {exp_name}")

        worker = ExperimentWorker(
            experiment=exp_key,
            n_games=self._n_games_spin.value(),
            step_delay_ms=self._delay_slider.value(),
            agent_a_defensive=self._def_a_chk.isChecked(),
            agent_b_defensive=self._def_b_chk.isChecked(),
        )
        thread = ExperimentThread(worker)

        worker.move_ready.connect(self._on_move)
        worker.game_done.connect(self._on_game)
        worker.log_line.connect(self._on_log)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)

        self._worker = worker
        self._thread = thread

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._paint_timer.start()
        thread.start()

    def _stop_experiment(self):
        self._paint_timer.stop()
        if self._thread:
            self._thread.stop()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_move(self, evt: MoveEvent):
        self._move_count += 1
        self._pending_move = evt
        self._analysis.on_move(evt)

    def _flush_move(self):
        evt = self._pending_move
        if evt is None:
            return
        self._pending_move = None
        self._hex_grid.update_state(
            game=evt.game,
            threats_p1=evt.threats_p1,
            threats_p2=evt.threats_p2,
            forks_p1=evt.forks_p1,
            forks_p2=evt.forks_p2,
            potential=evt.potential,
            last_move=evt.move,
        )
        self._tri_grid.update_state(
            game=evt.game,
            forks_p1=evt.forks_p1,
            forks_p2=evt.forks_p2,
            potential=evt.potential,
        )
        self._threat_graph.update_state(
            game=evt.game,
            threats_p1=evt.threats_p1,
            threats_p2=evt.threats_p2,
            forks_p1=evt.forks_p1,
            forks_p2=evt.forks_p2,
        )

    def _on_game(self, evt: GameEvent):
        self._analysis.on_game(evt)
        self._last_game_moves = list(evt.move_history)  # ordered; used by replay tab
        # Store ordered move history from the event
        w = evt.winner
        tag = f"P{w}" if w else "timeout"
        self.window().statusBar().showMessage(
            f"game {evt.game_number} | winner: {tag} | moves: {evt.move_count}"
        )

    def _on_log(self, line: str):
        self._log.append(line)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _on_finished(self, stats: ExperimentStats):
        self._paint_timer.stop()
        self._flush_move()
        self._analysis.on_stats(stats)
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._thread = None
        self._worker = None
        w1 = stats.wins.get(1, 0)
        w2 = stats.wins.get(2, 0)
        wd = stats.wins.get(0, 0)
        self.window().statusBar().showMessage(
            f"done | P1: {w1}  P2: {w2}  timeout: {wd} | patterns: {len(stats.pattern_freq)}"
        )
        self._log.append("■ experiment complete")

    def _on_error(self, msg: str):
        self._paint_timer.stop()
        self._log.append(f"[ERROR] {msg}")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._thread = None
        self._worker = None


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HexGo Theory")
        self.resize(1600, 900)
        self.setStyleSheet(STYLE)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        self._replay_tab = ReplayTab()
        self._live_tab   = LiveTab(self._replay_tab)

        tabs.addTab(self._live_tab,   "LIVE")
        tabs.addTab(self._replay_tab, "REPLAY")

        self.setCentralWidget(tabs)

        self.statusBar().setStyleSheet(
            "background: #080f18; border-top: 1px solid #0d1a2a; "
            "color: #3a4a5a;"
        )

    def closeEvent(self, event):
        t = self._live_tab._thread
        if t and t.isRunning():
            t.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("HexGo Theory")

    # Set app font explicitly at a valid point size so all widgets inherit it.
    # Never use font-size in stylesheets — let everything derive from this.
    app_font = QFont()
    app_font.setFamilies(["Consolas", "Courier New", "monospace"])
    app_font.setPointSize(9)   # safe on all DPI settings
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
