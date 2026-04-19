"""Hamkins echo extension at horizon = 960.

The existing [run_hamkins_echo.py](run_hamkins_echo.py) sweep covers
horizons {30, 60, 120, 240, 480}. At h=480 combo_vs_combo already shows
decisive games dominating (29/50 decisive), which is the *opposite* of
the Hamkins-style "long horizon induces draws" intuition. This script
checks whether the signal holds when we double the horizon: does the
decisive share keep growing, plateau, or start dropping?

Runs the same three matchups {random_vs_combo, greedy_vs_combo,
combo_vs_combo} at horizon=960, n=50 each. Writes the result to
`results/hamkins_echo_h960.json` (separate file — does not overwrite
the base sweep).

Combined figure produced by `run_hamkins_echo_merge.py`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_REAL_HEXGO = Path(r"C:\Users\Leon\Desktop\Psychograph\hexgo")
if _REAL_HEXGO.exists() and str(_REAL_HEXGO) not in sys.path:
    sys.path.insert(0, str(_REAL_HEXGO))

# Reuse the base module's helpers — avoids re-deriving the parallel harness.
from run_hamkins_echo import run_matchup  # type: ignore

RESULTS_PATH = ROOT / "results" / "hamkins_echo_h960.json"


def main() -> None:
    horizons = [960]
    n = 50
    matchups = [
        ("random", "combo"),
        ("greedy", "combo"),
        ("combo", "combo"),
    ]
    print(f"-- Hamkins echo @ h=960 --  n_per_cell={n}")
    t0 = time.time()
    out = {}
    for black, white in matchups:
        key = f"{black}_vs_{white}"
        print(f"\n[{key}]")
        out[key] = run_matchup(black, white, horizons, n)
    out["_wall_time"] = time.time() - t0
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {RESULTS_PATH}")
    print(f"[wall_time] {out['_wall_time']:.1f}s")


if __name__ == "__main__":
    main()
