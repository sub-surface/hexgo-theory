"""
HexGo Theory Dashboard — PySide6 desktop app.

Layout:
  ┌─────────────────────────────────────────────────────────────────┐
  │  toolbar: experiment selector, controls, step delay, toggles    │
  ├────────────────┬──────────────────┬────────────────┬────────────┤
  │  HEX GRID      │  TRI GRID        │  THREAT GRAPH  │  ANALYSIS  │
  │  (pan/zoom)    │  (lattice dual)  │  (hypergraph)  │  (metrics) │
  ├────────────────┴──────────────────┴────────────────┴────────────┤
  │  log output (scrolling text)                                    │
  └─────────────────────────────────────────────────────────────────┘
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QLabel, QComboBox, QPushButton, QSlider, QCheckBox,
    QSplitter, QTextEdit, QSizePolicy, QSpinBox, QFrame,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QAction

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
    font-family: Consolas;
    font-size: 11px;
}
QToolBar {
    background: #080f18;
    border-bottom: 1px solid #0d1a2a;
    spacing: 6px;
    padding: 4px 8px;
}
QToolBar QLabel {
    color: #3a4a5a;
    font-size: 10px;
    letter-spacing: 1px;
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
    padding: 3px 12px;
    border-radius: 0;
    min-width: 60px;
}
QPushButton:hover { background: #1a2535; }
QPushButton:pressed { background: #003580; }
QPushButton:disabled { color: #1a2535; border-color: #0d1a2a; }
QPushButton#run_btn {
    color: #e8e8e8;
    border-color: #003580;
}
QPushButton#run_btn:hover { background: #003580; }
QPushButton#stop_btn {
    color: #cc2200;
    border-color: #5a1010;
}
QSlider::groove:horizontal {
    background: #0d1a2a;
    height: 3px;
}
QSlider::handle:horizontal {
    background: #003580;
    width: 10px;
    height: 10px;
    margin: -4px 0;
}
QSlider::sub-page:horizontal { background: #003580; }
QCheckBox { color: #c8d4e0; spacing: 4px; }
QCheckBox::indicator {
    width: 12px; height: 12px;
    background: #0d1a2a;
    border: 1px solid #1a2535;
}
QCheckBox::indicator:checked { background: #003580; }
QTextEdit {
    background: #050a0f;
    color: #3a6a5a;
    border: none;
    border-top: 1px solid #0d1a2a;
    font-family: Consolas;
    font-size: 10px;
    padding: 4px;
}
QSplitter::handle { background: #0d1a2a; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QSpinBox {
    background: #0d1a2a;
    color: #c8d4e0;
    border: 1px solid #1a2535;
    padding: 2px 4px;
    border-radius: 0;
    width: 60px;
}
QScrollBar:vertical {
    background: #050a0f;
    width: 6px;
}
QScrollBar::handle:vertical {
    background: #1a2535;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

EXPERIMENTS = {
    "Eisenstein vs Eisenstein": "eis_vs_eis",
    "Eisenstein vs Random":     "eis_vs_random",
    "Fork Hunt (Eis A vs D)":   "fork_hunt",
    "Potential Landscape":      "potential_landscape",
    "Pattern Census":           "pattern_census",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HexGo Theory")
        self.resize(1600, 900)
        self.setStyleSheet(STYLE)

        self._thread: ExperimentThread | None = None
        self._worker: ExperimentWorker | None = None
        self._move_count = 0

        self._build_ui()
        self._connect_signals()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Toolbar ────────────────────────────────────────────────────────
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setIconSize(QSize(14, 14))
        self.addToolBar(tb)

        tb.addWidget(self._label("EXPERIMENT"))
        self._exp_combo = QComboBox()
        for name in EXPERIMENTS:
            self._exp_combo.addItem(name)
        tb.addWidget(self._exp_combo)

        tb.addSeparator()
        tb.addWidget(self._label("GAMES"))
        self._n_games_spin = QSpinBox()
        self._n_games_spin.setRange(1, 10000)
        self._n_games_spin.setValue(20)
        tb.addWidget(self._n_games_spin)

        tb.addSeparator()
        tb.addWidget(self._label("STEP ms"))
        self._delay_slider = QSlider(Qt.Orientation.Horizontal)
        self._delay_slider.setRange(0, 500)
        self._delay_slider.setValue(0)
        self._delay_slider.setFixedWidth(100)
        self._delay_label = QLabel("0")
        self._delay_label.setFixedWidth(28)
        self._delay_slider.valueChanged.connect(
            lambda v: self._delay_label.setText(str(v))
        )
        tb.addWidget(self._delay_slider)
        tb.addWidget(self._delay_label)

        tb.addSeparator()
        self._def_a_chk = QCheckBox("Def-A")
        self._def_b_chk = QCheckBox("Def-B")
        self._def_b_chk.setChecked(True)
        tb.addWidget(self._def_a_chk)
        tb.addWidget(self._def_b_chk)

        tb.addSeparator()
        # Overlay toggles
        self._chk_potential  = QCheckBox("Potential")
        self._chk_threats    = QCheckBox("Threats")
        self._chk_forks      = QCheckBox("Forks")
        self._chk_axislines  = QCheckBox("Axis lines")
        self._chk_candidates = QCheckBox("Candidates")
        for chk in (self._chk_potential, self._chk_threats,
                    self._chk_forks, self._chk_axislines):
            chk.setChecked(True)
            tb.addWidget(chk)
        tb.addWidget(self._chk_candidates)

        tb.addSeparator()
        # View buttons
        self._center_btn = QPushButton("Center")
        self._center_btn.setFixedWidth(56)
        tb.addWidget(self._center_btn)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self._run_btn  = QPushButton("Run")
        self._run_btn.setObjectName("run_btn")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.setEnabled(False)
        tb.addWidget(self._run_btn)
        tb.addWidget(self._stop_btn)

        # ── Central area ───────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top: visualisation panels + analysis
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setHandleWidth(1)

        self._hex_grid    = HexGridWidget()
        self._tri_grid    = TriGridWidget()
        self._threat_graph= ThreatGraphWidget()
        self._analysis    = AnalysisPanel()

        # Wrap each canvas in a titled frame
        top_splitter.addWidget(self._titled("HEX GRID", self._hex_grid))
        top_splitter.addWidget(self._titled("TRI GRID", self._tri_grid))
        top_splitter.addWidget(self._titled("THREAT GRAPH", self._threat_graph))
        top_splitter.addWidget(self._analysis)

        top_splitter.setSizes([420, 340, 340, 240])

        # Bottom: log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(120)

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setHandleWidth(1)
        v_splitter.addWidget(top_splitter)
        v_splitter.addWidget(self._log)
        v_splitter.setSizes([760, 120])

        main_layout.addWidget(v_splitter)

        # Status bar
        self._status_lbl = QLabel("ready")
        self._status_lbl.setStyleSheet(
            "color: #3a4a5a; font-size: 10px; padding: 2px 8px;"
        )
        self.statusBar().addPermanentWidget(self._status_lbl, 1)
        self.statusBar().setStyleSheet(
            "background: #080f18; border-top: 1px solid #0d1a2a;"
        )

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #3a4a5a; font-size: 10px; letter-spacing: 1px; padding: 0 4px;"
        )
        return lbl

    def _titled(self, title: str, widget: QWidget) -> QWidget:
        frame = QWidget()
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        hdr = QLabel(title)
        hdr.setStyleSheet(
            "background: #080f18; color: #003580; font-family: Consolas; "
            "font-size: 10px; font-weight: bold; letter-spacing: 2px; "
            "padding: 3px 8px; border-bottom: 1px solid #0d1a2a;"
        )
        lay.addWidget(hdr)
        lay.addWidget(widget)
        return frame

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._start_experiment)
        self._stop_btn.clicked.connect(self._stop_experiment)
        self._center_btn.clicked.connect(self._hex_grid.center_on_board)

        self._chk_potential.toggled.connect(
            lambda v: self._set_overlay("potential", v))
        self._chk_threats.toggled.connect(
            lambda v: self._set_overlay("threats", v))
        self._chk_forks.toggled.connect(
            lambda v: self._set_overlay("forks", v))
        self._chk_axislines.toggled.connect(
            lambda v: self._set_overlay("axis_lines", v))
        self._chk_candidates.toggled.connect(
            lambda v: self._set_overlay("candidates", v))

    def _set_overlay(self, name: str, value: bool):
        setattr(self._hex_grid, f"show_{name}", value)
        self._hex_grid.update()

    # ── Experiment control ────────────────────────────────────────────────────

    def _start_experiment(self):
        if self._thread and self._thread.isRunning():
            return

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
        self._status_lbl.setText(f"running: {exp_name}")

        thread.start()

    def _stop_experiment(self):
        if self._thread:
            self._thread.stop()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_lbl.setText("stopped")

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_move(self, evt: MoveEvent):
        self._move_count += 1

        # Update all three canvas widgets
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
        self._analysis.on_move(evt)

    def _on_game(self, evt: GameEvent):
        self._analysis.on_game(evt)
        w = evt.winner
        tag = f"P{w}" if w else "timeout"
        self._status_lbl.setText(
            f"game {evt.game_number} | winner: {tag} | moves: {evt.move_count}"
        )

    def _on_log(self, line: str):
        self._log.append(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, stats: ExperimentStats):
        self._analysis.on_stats(stats)
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        w1 = stats.wins.get(1, 0)
        w2 = stats.wins.get(2, 0)
        wd = stats.wins.get(0, 0)
        self._status_lbl.setText(
            f"done | P1: {w1}  P2: {w2}  timeout: {wd} | "
            f"patterns: {len(stats.pattern_freq)}"
        )
        self._log.append(f"■ experiment complete")

    def _on_error(self, msg: str):
        self._log.append(f"[ERROR] {msg}")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_lbl.setText("error")

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("HexGo Theory")

    # Force dark palette at OS level too
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,      QColor("#050a0f"))
    palette.setColor(QPalette.ColorRole.WindowText,  QColor("#c8d4e0"))
    palette.setColor(QPalette.ColorRole.Base,        QColor("#080f18"))
    palette.setColor(QPalette.ColorRole.Text,        QColor("#c8d4e0"))
    palette.setColor(QPalette.ColorRole.Button,      QColor("#0d1a2a"))
    palette.setColor(QPalette.ColorRole.ButtonText,  QColor("#c8d4e0"))
    palette.setColor(QPalette.ColorRole.Highlight,   QColor("#003580"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#e8e8e8"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
