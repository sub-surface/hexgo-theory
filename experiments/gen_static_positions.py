"""Generate a static-position dataset for Phase 0 of the unified agent.

This script plays `ca_combo_v2` vs `ca_combo_v2` games (our strongest
hand-crafted agent) and at each ply records the current encoded board
together with several per-cell labels computed from
[engine/analysis.py](../engine/analysis.py):

    * threat_self      -- `threat_cells(game, to_move)`
    * threat_opp       -- `threat_cells(game, 3 - to_move)`
    * fork_self        -- `fork_cells(game, to_move)` mapped to binary per cell
    * potential        -- `potential_map(game)` normalised to [0, 1]
    * winning_self     -- cells where to_move can literally complete 6-in-a-row
                          in one ply (strict subset of threat_self == 1 with
                          >=1 live window of length 6)
    * policy_target    -- the move that ca_combo_v2 actually played, as a
                          one-hot on the (H, W) encoded grid

The trunk pretrain (Phase 0b) uses these labels to teach the shared
feature stack to extract tactical structure, independent of any value
or policy signal. Charlie's cross-transfer matrix showed that a
trunk pretrained on `threat` alone transfers to `double`, `winning_cells`,
and `five`; this script produces the same family of labels but on our
infinite-lattice substrate.

Each sample is an `.npz` record:
    arr        (4, H, W) float32 -- observer.encode_position output
    labels     (5, H, W) float32 -- {threat_self, threat_opp, fork_self,
                                     potential_norm, winning_self}
    policy     (H, W) float32    -- one-hot of ca_combo_v2's move
    value      scalar float32    -- +1 if to_move ends up winning, -1 if
                                    losing, 0 if unfinished
    meta       (to_move, ply, game_idx, origin)

Output:
    data/static_positions/<split>_<idx>.npz     (8000 train / 2000 val)
    data/static_positions/manifest.json

Expected wall-time on our 2060: the batching is trivial
(pure CPU game-play + analysis), so 10k positions in ~15-30 min.

Usage:
    python experiments/gen_static_positions.py               # full 10k
    python experiments/gen_static_positions.py --quick       # 500 positions
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_REAL_HEXGO = Path(r"C:\Users\Leon\Desktop\Psychograph\hexgo")
if _REAL_HEXGO.exists() and str(_REAL_HEXGO) not in sys.path:
    sys.path.insert(0, str(_REAL_HEXGO))

import numpy as np

from engine import HexGame
from engine.analysis import (
    fork_cells,
    potential_map,
    threat_cells,
    _all_windows,
)
from engine.ca_policy import make_combo_v2_ca
from engine.observer import encode_position


OUT_DIR = ROOT / "data" / "static_positions"


def winning_cells(game: HexGame, player: int) -> dict[tuple, int]:
    """Cells where placing a `player` stone completes a 6-in-a-row.

    Stricter than threat_cells: `threat_cells` fires on any 5-of-6
    window with a single empty. This checks the classical win condition
    (any axis-aligned length-6 completed).
    """
    wins: dict[tuple, int] = {}
    for cells, _ in _all_windows(game):
        players = {game.board[c] for c in cells if c in game.board}
        if players != {player}:
            continue
        empty = [c for c in cells if c not in game.board]
        if len(empty) == 1:
            wins[empty[0]] = wins.get(empty[0], 0) + 1
    return wins


def cell_to_grid(cell: tuple, origin: tuple[int, int, int, int]) -> tuple[int, int] | None:
    """Map a (q, r) cell to (row, col) on the encoded grid."""
    q_min, r_min, H, W = origin
    q, r = cell
    col = q - q_min
    row = r - r_min
    if 0 <= row < H and 0 <= col < W:
        return (row, col)
    return None


def build_labels(
    game: HexGame,
    to_move: int,
    origin: tuple[int, int, int, int],
) -> np.ndarray:
    """Return (5, H, W) label tensor aligned to the encoded grid."""
    _, _, H, W = origin
    labels = np.zeros((5, H, W), dtype=np.float32)

    ts = threat_cells(game, to_move)
    to = threat_cells(game, 3 - to_move)
    fs = fork_cells(game, to_move)
    pot = potential_map(game)
    ws = winning_cells(game, to_move)

    # channel 0: threat_self (binary)
    for cell, count in ts.items():
        rc = cell_to_grid(cell, origin)
        if rc is not None:
            labels[0, rc[0], rc[1]] = 1.0
    # channel 1: threat_opp (binary)
    for cell, count in to.items():
        rc = cell_to_grid(cell, origin)
        if rc is not None:
            labels[1, rc[0], rc[1]] = 1.0
    # channel 2: fork_self (binary, axes_hit >= 2)
    for cell, axes in fs.items():
        rc = cell_to_grid(cell, origin)
        if rc is not None:
            labels[2, rc[0], rc[1]] = 1.0
    # channel 3: potential map normalised to [0, 1]
    if pot:
        max_pot = max(pot.values())
        if max_pot > 0:
            for cell, val in pot.items():
                rc = cell_to_grid(cell, origin)
                if rc is not None:
                    labels[3, rc[0], rc[1]] = val / max_pot
    # channel 4: winning_self
    for cell, count in ws.items():
        rc = cell_to_grid(cell, origin)
        if rc is not None:
            labels[4, rc[0], rc[1]] = 1.0

    return labels


def rollout_one_game(
    rng: np.random.Generator,
    max_moves: int = 120,
    sample_rate: float = 0.6,
) -> list[dict]:
    """Play one ca_combo_v2 self-play game and sample positions.

    sample_rate: per-ply Bernoulli probability of recording this ply
    (cuts correlation between adjacent samples without killing coverage).
    """
    # Build two combo_v2 agents -- they're stateful, so fresh each game.
    blk = make_combo_v2_ca()
    wht = make_combo_v2_ca()
    game = HexGame()
    records: list[dict] = []
    ply = 0
    while game.winner is None and ply < max_moves:
        if not game.candidates:
            break
        to_move = game.current_player
        mover = blk if to_move == 1 else wht
        try:
            move = mover.choose_move(game)
        except Exception:
            break
        if move is None or move in game.board:
            break
        record_here = rng.random() < sample_rate
        if record_here:
            arr, origin = encode_position(game, to_move=to_move, pad=4)
            labels = build_labels(game, to_move, origin)
            H, W = arr.shape[1], arr.shape[2]
            policy = np.zeros((H, W), dtype=np.float32)
            rc = cell_to_grid(move, origin)
            if rc is not None:
                policy[rc[0], rc[1]] = 1.0
            records.append({
                "arr": arr,
                "labels": labels,
                "policy": policy,
                "to_move": to_move,
                "ply": ply,
                "origin": origin,
            })
        if not game.make(*move):
            break
        ply += 1

    # Once game is resolved, backfill values.
    # winner in {1, 2, None}; value from perspective of to_move.
    if game.winner is None:
        final_value = 0.0
    else:
        final_value = 1.0  # placeholder overwritten per-record
    for rec in records:
        if game.winner is None:
            rec["value"] = 0.0
        else:
            rec["value"] = 1.0 if rec["to_move"] == game.winner else -1.0
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--target", type=int, default=10_000,
                        help="target total positions (train + val)")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        target = 500
    else:
        target = args.target

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # clean old
    for p in OUT_DIR.glob("*.npz"):
        p.unlink()
    (OUT_DIR / "manifest.json").unlink(missing_ok=True)

    rng = np.random.default_rng(args.seed)
    all_records: list[dict] = []
    games_played = 0
    t0 = time.time()
    while len(all_records) < target:
        recs = rollout_one_game(rng)
        games_played += 1
        all_records.extend(recs)
        if games_played % 10 == 0:
            elapsed = time.time() - t0
            rate = len(all_records) / max(1e-6, elapsed)
            eta = (target - len(all_records)) / max(1e-6, rate)
            print(f"  games={games_played}  positions={len(all_records)}"
                  f"  rate={rate:.1f} pos/s  eta={eta:.0f}s")
    all_records = all_records[:target]
    n_val = int(len(all_records) * args.val_frac)
    rng.shuffle(all_records)
    val_recs = all_records[:n_val]
    train_recs = all_records[n_val:]

    # write each record as its own npz (arrays have variable H, W so a
    # single big tensor would need padding; keep per-record for now)
    def _write(split: str, records: list[dict]) -> list[str]:
        paths: list[str] = []
        for i, rec in enumerate(records):
            fname = f"{split}_{i:06d}.npz"
            fp = OUT_DIR / fname
            np.savez_compressed(
                fp,
                arr=rec["arr"],
                labels=rec["labels"],
                policy=rec["policy"],
                value=np.float32(rec["value"]),
                to_move=np.int32(rec["to_move"]),
                ply=np.int32(rec["ply"]),
                origin=np.array(rec["origin"], dtype=np.int32),
            )
            paths.append(fname)
        return paths

    train_files = _write("train", train_recs)
    val_files = _write("val", val_recs)

    manifest = {
        "n_train": len(train_files),
        "n_val": len(val_files),
        "games_played": games_played,
        "wall_time_s": time.time() - t0,
        "label_channels": [
            "threat_self", "threat_opp", "fork_self",
            "potential_norm", "winning_self",
        ],
        "source": "ca_combo_v2 self-play, sample_rate=0.6, max_moves=120",
        "seed": args.seed,
        "train_files": train_files,
        "val_files": val_files,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone. n_train={len(train_files)} n_val={len(val_files)}"
          f" games={games_played} wall={manifest['wall_time_s']:.1f}s")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
