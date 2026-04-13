"""
Run the headline epiplexity scans for ROADMAPv2 Programmes A and E.

Outputs:
  corpora/<name>_N<size>.pkl.gz   — pickled corpora
  results/epiplexity_scan.json    — MDL measurements
  figures/fig_A_paradox1.png      — Paradox 1 (info from computation)
  figures/fig_D_scaling.png       — S_T scaling with N (Pisot spectroscope)
  figures/fig_E_pareto.png        — agent Pareto frontier

Run:
    python -X utf8 experiments/run_epiplexity_scan.py
    python -X utf8 experiments/run_epiplexity_scan.py --quick   # small sizes for dev
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from engine import (
    RandomAgent, EisensteinGreedyAgent, ForkAwareAgent,
    PotentialGradientAgent, ComboAgent,
)
from engine.epiplexity import (
    generate_corpus, measure_corpus,
    agent_program_length, agent_cross_entropy_bits,
    Corpus,
)


CORPORA_DIR = ROOT / "corpora"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for d in (CORPORA_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(exist_ok=True, parents=True)


AGENT_FACTORIES = {
    "random":     lambda: RandomAgent(),
    "greedy_off": lambda: EisensteinGreedyAgent("greedy_off", defensive=False),
    "greedy_def": lambda: EisensteinGreedyAgent("greedy_def", defensive=True),
    "fork_a2":    lambda: ForkAwareAgent("fork_a2", alpha=2.0),
    "fork_a4":    lambda: ForkAwareAgent("fork_a4", alpha=4.0),
    "potgrad":    lambda: PotentialGradientAgent("potgrad"),
    "combo":      lambda: ComboAgent("combo"),
}


def corpus_path(name: str, n: int) -> Path:
    return CORPORA_DIR / f"{name}_N{n}.pkl.gz"


def get_or_generate(name: str, n: int, seed: int = 0, force: bool = False) -> Corpus:
    p = corpus_path(name, n)
    if p.exists() and not force:
        return Corpus.load(p)
    fac = AGENT_FACTORIES[name]
    c = generate_corpus(fac, fac, n_games=n, seed=seed)
    c.save(p)
    return c


def programme_A_paradox1(n_per_agent: int, quick: bool) -> dict:
    """Compare S_T and H_T for random vs structured-agent corpora."""
    print(f"\n── Programme A: Paradox 1 scan (N={n_per_agent} games per agent) ──")
    reports = {}
    for name in ["random", "greedy_def", "fork_a4", "potgrad", "combo"]:
        t0 = time.time()
        c = get_or_generate(name, n_per_agent, seed=42)
        r = measure_corpus(c, name=name)
        reports[name] = {
            "n_games": r.n_games,
            "n_tokens": r.n_tokens,
            "markov_H_T": r.markov_H_T_bits_per_token,
            "markov_S_T_bits": r.markov_S_T_bits,
            "gzip_bpt": r.gzip_bits_per_token,
            "gzip_total_bits": r.gzip_total_bits,
        }
        print(f"  {name:12s}  H_T={r.markov_H_T_bits_per_token:5.2f} b/tok  "
              f"gzip={r.gzip_bits_per_token:5.2f} b/tok  "
              f"S_T={r.markov_S_T_bits:>7d} bits  "
              f"({time.time()-t0:.1f}s)")
    return reports


def programme_D_scaling(sizes: list[int]) -> dict:
    """S_T as a function of corpus size N — the Pisot spectroscope."""
    print(f"\n── Programme D: S_T vs N scaling scan (sizes={sizes}) ──")
    agents = ["random", "greedy_def", "combo"]
    scaling: dict[str, list[dict]] = {a: [] for a in agents}
    for name in agents:
        for n in sizes:
            t0 = time.time()
            c = get_or_generate(name, n, seed=7)
            r = measure_corpus(c, name=name)
            scaling[name].append({
                "N": n,
                "n_tokens": r.n_tokens,
                "H_T": r.markov_H_T_bits_per_token,
                "S_T_bits": r.markov_S_T_bits,
                "gzip_total_bits": r.gzip_total_bits,
                "gzip_bpt": r.gzip_bits_per_token,
            })
            print(f"  {name:12s}  N={n:>5d}  H_T={r.markov_H_T_bits_per_token:5.2f}  "
                  f"S_T={r.markov_S_T_bits:>7d} bits  gzip={r.gzip_total_bits:>7d} bits  "
                  f"({time.time()-t0:.1f}s)")
    return scaling


def programme_E_pareto(n_per_agent: int) -> dict:
    """Agent Pareto frontier in (|P|, H_T) space."""
    print(f"\n── Programme E: Pareto frontier (N={n_per_agent} games) ──")
    # evaluation corpus: ComboAgent self-play, fixed seed
    eval_corpus = get_or_generate("combo", n_per_agent, seed=2026)
    points = {}
    for name, fac in AGENT_FACTORIES.items():
        a = fac()
        prog = agent_program_length(a)
        ce = agent_cross_entropy_bits(a, eval_corpus)
        points[name] = {"prog_bytes": prog, "H_T_bits": ce}
        print(f"  {name:12s}  |P|={prog:>5d} B  H_T={ce:5.2f} bits/move")
    return points


def plot_paradox1(reports: dict, out: Path) -> None:
    names = list(reports.keys())
    H_T = [reports[n]["markov_H_T"] for n in names]
    gzip_bpt = [reports[n]["gzip_bpt"] for n in names]
    colors = ["#888", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w/2, H_T, w, label="Markov-3 observer", color=colors[:len(names)])
    ax.bar(x + w/2, gzip_bpt, w, label="gzip observer",
           color=colors[:len(names)], alpha=0.55, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("cross-entropy  (bits / move)")
    ax.set_title("Paradox 1 — random corpus is incompressible; structured play is not\n"
                 "(lower = more predictable = more structural information extracted)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_scaling(scaling: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    colors = {"random": "#888", "greedy_def": "#1f77b4", "combo": "#d62728"}
    markers = {"random": "o", "greedy_def": "s", "combo": "^"}

    for name, rows in scaling.items():
        Ns = np.array([r["N"] for r in rows])
        H = np.array([r["H_T"] for r in rows])
        gz = np.array([r["gzip_total_bits"] for r in rows])
        ax1.plot(Ns, H, marker=markers[name], color=colors[name], label=name)
        ax2.plot(Ns, gz, marker=markers[name], color=colors[name], label=name)
    ax1.set_xscale("log")
    ax1.set_xlabel("corpus size  N (games)")
    ax1.set_ylabel(r"$H_T$  (bits / move)")
    ax1.set_title("Residual unpredictability $H_T$ vs N")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("corpus size  N (games)")
    ax2.set_ylabel("total gzip bits  (log-log)")
    ax2.set_title("Corpus description length vs N  — slope ~1 means no finite program")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()

    # fit slopes on log-log
    for name, rows in scaling.items():
        Ns = np.log10([r["N"] for r in rows])
        gz = np.log10([r["gzip_total_bits"] for r in rows])
        if len(Ns) >= 2:
            slope, intercept = np.polyfit(Ns, gz, 1)
            ax2.annotate(f"{name}: slope ≈ {slope:.2f}",
                         xy=(0.55, 0.10 + 0.06 * list(scaling.keys()).index(name)),
                         xycoords="axes fraction",
                         color=colors[name], fontsize=9)

    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_pareto(points: dict, out: Path) -> None:
    names = list(points.keys())
    P = np.array([points[n]["prog_bytes"] for n in names])
    H = np.array([points[n]["H_T_bits"] for n in names])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(P, H, s=120, c="#2ca02c", edgecolor="black", linewidth=0.7)
    for x, y, n in zip(P, H, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    # Pareto frontier (lower-left hull)
    pts = sorted(zip(P, H, names))
    frontier = []
    best = float("inf")
    for x, y, n in pts:
        if y < best:
            frontier.append((x, y, n))
            best = y
    if frontier:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                "--", color="crimson", alpha=0.7, label="Pareto frontier")

    ax.set_xlabel(r"$|P|$  gzipped canonical source (bytes)")
    ax.set_ylabel(r"$H_T$  (bits / move against Combo-self-play corpus)")
    ax.set_title("Programme E — agents as time-bounded models in MDL space")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use small sizes so the scan finishes in <1 min")
    parser.add_argument("--sizes", type=int, nargs="*", default=None)
    parser.add_argument("--n_paradox", type=int, default=None)
    parser.add_argument("--n_pareto", type=int, default=None)
    args = parser.parse_args()

    if args.quick:
        sizes = [50, 100, 200]
        n_paradox = 200
        n_pareto = 100
    else:
        sizes = args.sizes or [50, 100, 200, 500, 1000]
        n_paradox = args.n_paradox or 500
        n_pareto = args.n_pareto or 200

    out = {}
    out["paradox1"] = programme_A_paradox1(n_paradox, quick=args.quick)
    out["scaling"]  = programme_D_scaling(sizes)
    out["pareto"]   = programme_E_pareto(n_pareto)

    RESULTS_PATH = RESULTS_DIR / "epiplexity_scan.json"
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {RESULTS_PATH}")

    plot_paradox1(out["paradox1"], FIGURES_DIR / "fig_A_paradox1.png")
    plot_scaling(out["scaling"],   FIGURES_DIR / "fig_D_scaling.png")
    plot_pareto(out["pareto"],     FIGURES_DIR / "fig_E_pareto.png")
    print(f"[saved] figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
