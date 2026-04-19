r"""
Train a shared-trunk `StrategyObserver` on self-play corpora from every
agent in the ladder, then measure:

  (A) H_T(A) — irreducible policy cross-entropy on held-out games
  (B) MLM accuracy on held-out masked positions
  (C) Linear-probe accuracies for {threat_self, threat_opp, fork_self,
      high_potential}
  (D) S_T ladder — minimum trunk size to reach within tolerance of best
      irreducible loss (Finzi et al. 2026 two-part-MDL estimator)

This is the experimental leg of the
[docs/theory/2026-04-18-epiplexity-of-strategy.md] synthesis note: it
unifies the friend's "shared representation + MLM + linear probes" recipe
with Leon's "apply epiplexity to our strategies" request.

Outputs:
  results/strategy_observer.json   — all metrics, per corpus
  figures/fig_strategy_observer.png
    top-left:    H_T per agent (policy cross-entropy) + MLM CE
    top-right:   probe accuracies heatmap (agent × predicate)
    bottom-left: S_T vs hidden ladder per agent
    bottom-right: (|P|, H_T) Pareto scatter
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# worktree shim — see CLAUDE.md > "Worktree gotcha"
_REAL_HEXGO = Path(r"C:\Users\Leon\Desktop\Psychograph\hexgo")
if _REAL_HEXGO.exists() and str(_REAL_HEXGO) not in sys.path:
    sys.path.insert(0, str(_REAL_HEXGO))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine import RandomAgent, EisensteinGreedyAgent
from engine.agents import MirrorAgent
from engine.ca_policy import make_combo_v2_ca
from engine.observer import (
    CorpusExample,
    generate_corpus,
    train_observer,
    train_linear_probe,
    epiplexity_estimate,
    _state_dict_gzip_bytes,
)


# Ladder of agent factories. Each factory returns a fresh instance so
# generate_corpus can paint both sides without shared state.
def _agent_factories():
    out = {
        "random": RandomAgent,
        "greedy": EisensteinGreedyAgent,
        "mirror": MirrorAgent,
        "combo_v2": make_combo_v2_ca,
    }
    return out


# Predicate set for the linear-probe suite (§3.1 Head C of synthesis note).
PREDICATES = ["threat_self", "threat_opp", "fork_self", "high_potential"]


def _agent_gzip_bytes(factory) -> int:
    """Gzipped bytes of the agent's implementation file — proxy for |P_A|.

    Uses the source of the factory's __module__ as the representation.  This
    is rough but consistent across agents.
    """
    import gzip
    import inspect
    try:
        mod = inspect.getmodule(factory)
        src = inspect.getsource(mod).encode()
        return len(gzip.compress(src))
    except Exception:
        return -1


def _run(
    n_games: int,
    max_moves: int,
    epochs: int,
    hidden: int,
    depth: int,
    batch_size: int,
    ladder_hidden_sizes: tuple[int, ...],
    seed: int,
) -> dict:
    factories = _agent_factories()
    out: dict = {
        "config": {
            "n_games": n_games, "max_moves": max_moves, "epochs": epochs,
            "hidden": hidden, "depth": depth, "batch_size": batch_size,
            "ladder_hidden_sizes": list(ladder_hidden_sizes), "seed": seed,
        },
        "agents": {},
    }
    for name, fac in factories.items():
        print(f"\n── corpus for {name} (n_games={n_games}) ──")
        t0 = time.perf_counter()
        corpus = generate_corpus(
            fac, n_games=n_games, max_moves=max_moves, seed=seed, pad=4,
        )
        t_corpus = time.perf_counter() - t0
        print(f"  corpus size: {len(corpus)}  ({t_corpus:.1f}s)")

        # (A+B) Train an observer at the reference hidden/depth — H_T and MLM CE.
        t0 = time.perf_counter()
        model, hist = train_observer(
            corpus, hidden=hidden, depth=depth, epochs=epochs,
            batch_size=batch_size, seed=seed,
        )
        t_train = time.perf_counter() - t0
        print(f"  train  policy_ce_val={hist['final_policy_ce']:.3f}  "
              f"mlm_ce_val={hist['final_mlm_ce']:.3f}  ({t_train:.1f}s)")

        # (C) Linear probes over the frozen trunk.
        probe_accs: dict[str, tuple[float, float]] = {}
        for pred in PREDICATES:
            try:
                tr, vl = train_linear_probe(
                    model.trunk, corpus, predicate=pred,
                    batch_size=batch_size, epochs=3, seed=seed,
                )
                probe_accs[pred] = (tr, vl)
                print(f"    probe {pred:<16s}  train={tr:.2f}  val={vl:.2f}")
            except Exception as e:
                probe_accs[pred] = (float("nan"), float("nan"))
                print(f"    probe {pred:<16s}  FAILED: {e}")

        # (D) S_T ladder.
        t0 = time.perf_counter()
        epi = epiplexity_estimate(
            corpus, hidden_sizes=ladder_hidden_sizes, depth=depth,
            epochs=max(2, epochs // 2), seed=seed,
        )
        t_epi = time.perf_counter() - t0
        print(f"  epi  min_hidden={epi['min_hidden']}  "
              f"S_T_bytes={epi['S_T_gzip_bytes']}  ({t_epi:.1f}s)")

        out["agents"][name] = {
            "corpus_size": len(corpus),
            "t_corpus_s": t_corpus, "t_train_s": t_train, "t_epi_s": t_epi,
            "agent_source_gzip_bytes": _agent_gzip_bytes(fac),
            "history": hist,
            "probe_accs": {k: list(v) for k, v in probe_accs.items()},
            "epiplexity": {
                "hidden_losses": {str(k): v for k, v in epi["hidden_losses"].items()},
                "hidden_gzip_bytes": {str(k): v for k, v in epi["hidden_gzip_bytes"].items()},
                "reference_loss": epi["reference_loss"],
                "threshold_loss": epi["threshold_loss"],
                "min_hidden": int(epi["min_hidden"]),
                "S_T_gzip_bytes": int(epi["S_T_gzip_bytes"]),
            },
            "observer_gzip_bytes": _state_dict_gzip_bytes(model),
        }
    return out


def _plot(out: dict, path: str) -> None:
    agents = list(out["agents"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_ht, ax_heat, ax_st, ax_pareto = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    # (1) H_T per agent + MLM CE on secondary axis.
    ht = [out["agents"][a]["history"]["final_policy_ce"] for a in agents]
    mlm = [out["agents"][a]["history"]["final_mlm_ce"] for a in agents]
    x = np.arange(len(agents))
    ax_ht.bar(x - 0.18, ht, width=0.35, label=r"policy CE ($H_T$)", color="#36a")
    ax_ht.bar(x + 0.18, mlm, width=0.35, label="MLM CE", color="#c83")
    ax_ht.set_xticks(x)
    ax_ht.set_xticklabels(agents, rotation=30, ha="right")
    ax_ht.set_ylabel("cross-entropy (nats)")
    ax_ht.set_title("Observer irreducible loss by agent")
    ax_ht.legend()
    ax_ht.grid(axis="y", linestyle=":", alpha=0.5)

    # (2) Probe accuracy heatmap.
    predicates = PREDICATES
    heat = np.full((len(agents), len(predicates)), np.nan)
    for i, a in enumerate(agents):
        for j, p in enumerate(predicates):
            v = out["agents"][a]["probe_accs"].get(p, [np.nan, np.nan])[1]
            heat[i, j] = v
    im = ax_heat.imshow(heat, aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
    ax_heat.set_xticks(range(len(predicates)))
    ax_heat.set_xticklabels(predicates, rotation=30, ha="right")
    ax_heat.set_yticks(range(len(agents)))
    ax_heat.set_yticklabels(agents)
    ax_heat.set_title("Linear-probe val accuracy")
    for i in range(len(agents)):
        for j in range(len(predicates)):
            v = heat[i, j]
            if np.isnan(v):
                continue
            ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v < 0.75 else "black", fontsize=8)
    plt.colorbar(im, ax=ax_heat, fraction=0.04)

    # (3) S_T ladder — policy CE vs hidden for each agent.
    for a in agents:
        hl = out["agents"][a]["epiplexity"]["hidden_losses"]
        hs = sorted(int(k) for k in hl)
        ls = [hl[str(h)] for h in hs]
        ax_st.plot(hs, ls, marker="o", label=a)
    ax_st.set_xscale("log", base=2)
    ax_st.set_xlabel("trunk hidden width")
    ax_st.set_ylabel(r"policy CE ($H_T$ estimator)")
    ax_st.set_title("Epiplexity ladder")
    ax_st.legend(fontsize=9)
    ax_st.grid(which="both", linestyle=":", alpha=0.5)

    # (4) (|P_A|, H_T) Pareto scatter — gzip bytes of agent source vs H_T.
    for a in agents:
        g = out["agents"][a]["agent_source_gzip_bytes"]
        ht_a = out["agents"][a]["history"]["final_policy_ce"]
        if g > 0:
            ax_pareto.scatter([g], [ht_a], s=70)
            ax_pareto.annotate(a, (g, ht_a), xytext=(4, 4),
                               textcoords="offset points", fontsize=9)
    ax_pareto.set_xlabel(r"agent source gzip bytes ($|P_A|$ proxy)")
    ax_pareto.set_ylabel(r"policy CE ($H_T$)")
    ax_pareto.set_title(r"$(|P_A|, H_T)$ Pareto scatter")
    ax_pareto.grid(linestyle=":", alpha=0.5)

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=40)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ladder", type=str, default="4,8,16,32,64",
                    help="comma-separated hidden widths for epiplexity ladder")
    ap.add_argument("--seed", type=int, default=20260418)
    ap.add_argument("--quick", action="store_true",
                    help="6 games, 1 epoch — smoke test")
    args = ap.parse_args()

    if args.quick:
        args.n_games = 6
        args.max_moves = 60
        args.epochs = 1
        args.ladder = "4,8,16"

    ladder = tuple(int(x) for x in args.ladder.split(","))

    out = _run(
        n_games=args.n_games, max_moves=args.max_moves,
        epochs=args.epochs, hidden=args.hidden, depth=args.depth,
        batch_size=args.batch_size, ladder_hidden_sizes=ladder,
        seed=args.seed,
    )

    rpath = Path("results") / "strategy_observer.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    with open(rpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {rpath}")

    fpath = Path("figures") / "fig_strategy_observer.png"
    _plot(out, str(fpath))
    print(f"[saved] {fpath}")

    print("\n── Summary ──")
    for a, rec in out["agents"].items():
        ht = rec["history"]["final_policy_ce"]
        mh = rec["epiplexity"]["min_hidden"]
        st = rec["epiplexity"]["S_T_gzip_bytes"]
        print(f"  {a:>10s}  H_T={ht:.3f}  min_hidden={mh}  S_T={st} bytes")


if __name__ == "__main__":
    main()
