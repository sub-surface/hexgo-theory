"""Phase 0 pretrain runner for the unified agent.

Trains `engine.alphazero.UnifiedNet` on `data/static_positions/` with
multi-task supervision (policy + threat + win + fork + potential).
Value head is skipped in Phase 0 (see engine/alphazero.py docstring).

Output:
    results/az_pretrain.json          -- per-epoch history
    artifacts/checkpoints/az_pretrain.pt  -- trained model state
    figures/fig_az_pretrain.png       -- train/val curves

Usage:
    python experiments/run_az_pretrain.py           # full 40 epochs
    python experiments/run_az_pretrain.py --quick   # 3 epochs, 500 train
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from engine.alphazero import pretrain_trunk


DATA_DIR = ROOT / "data" / "static_positions"
RESULTS_JSON = ROOT / "results" / "az_pretrain.json"
CKPT_DIR = ROOT / "artifacts" / "checkpoints"
CKPT_PATH = CKPT_DIR / "az_pretrain.pt"
FIG_PATH = ROOT / "figures" / "fig_az_pretrain.png"


def plot_history(hist: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    epochs = np.array(hist["epoch"])

    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], "-", label="train", color="#4c78a8")
    vl = np.array(hist["val_loss"], dtype=float)
    mask = ~np.isnan(vl)
    ax.plot(epochs[mask], vl[mask], "o-", label="val", color="#f58518")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (combined)")
    ax.set_title("Total multi-task loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, hist["train_policy_acc"], "-", label="train pol_acc", color="#4c78a8")
    vpa = np.array(hist["val_policy_acc"], dtype=float)
    mask = ~np.isnan(vpa)
    ax.plot(epochs[mask], vpa[mask], "o-", label="val pol_acc", color="#f58518")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Policy (next-move) top-1 accuracy")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.4)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, hist["train_threat_f1"], "-", label="train threat F1", color="#4c78a8")
    ax.plot(epochs, hist["train_win_f1"], "-", label="train win F1", color="#54a24b")
    vtf = np.array(hist["val_threat_f1"], dtype=float)
    vwf = np.array(hist["val_win_f1"], dtype=float)
    mt = ~np.isnan(vtf)
    mw = ~np.isnan(vwf)
    ax.plot(epochs[mt], vtf[mt], "o-", label="val threat F1", color="#f58518")
    ax.plot(epochs[mw], vwf[mw], "s-", label="val win F1", color="#e45756")
    ax.set_xlabel("epoch")
    ax.set_ylabel("F1")
    ax.set_title("Tactical head F1 scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Phase 0 supervised trunk pretrain -- UnifiedNet"
        f" (hidden=32, depth=6)",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    if args.quick:
        args.epochs = 3
        max_train = 500
    else:
        max_train = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    print(f"data_dir = {args.data_dir}")
    t0 = time.time()

    model, history = pretrain_trunk(
        args.data_dir,
        hidden=args.hidden,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        seed=args.seed,
        max_train=max_train,
    )

    wall = time.time() - t0
    history["_wall_time"] = wall
    history["_args"] = vars(args)

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(history, indent=2))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": args.hidden,
        "depth": args.depth,
        "history": history,
    }, CKPT_PATH)
    plot_history(history, FIG_PATH)

    print(f"\n[done] wall_time={wall:.1f}s")
    print(f"  wrote {RESULTS_JSON}")
    print(f"  wrote {CKPT_PATH}")
    print(f"  wrote {FIG_PATH}")
    # final metrics
    tr_pol = history["train_policy_acc"][-1] if history["train_policy_acc"] else 0
    va_pol = next((v for v in reversed(history["val_policy_acc"]) if not (v != v)), 0)
    tr_thr = history["train_threat_f1"][-1] if history["train_threat_f1"] else 0
    va_thr = next((v for v in reversed(history["val_threat_f1"]) if not (v != v)), 0)
    tr_win = history["train_win_f1"][-1] if history["train_win_f1"] else 0
    va_win = next((v for v in reversed(history["val_win_f1"]) if not (v != v)), 0)
    print(f"  final: policy_acc tr={tr_pol:.3f} va={va_pol:.3f}")
    print(f"         threat_f1 tr={tr_thr:.3f} va={va_thr:.3f}")
    print(f"         win_f1    tr={tr_win:.3f} va={va_win:.3f}")


if __name__ == "__main__":
    main()
