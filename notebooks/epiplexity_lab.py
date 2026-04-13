"""
Epiplexity Lab — live scratchpad for HexGo theory v2.

Run with:  marimo edit notebooks/epiplexity_lab.py

This version uses REAL corpora and real measurements via
`engine.epiplexity`. First open will generate small corpora (takes seconds).
Bigger scans use `experiments/run_epiplexity_scan.py` and the results land
in `results/epiplexity_scan.json`; the notebook picks them up automatically.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium", app_title="HexGo Epiplexity Lab")


# ── intro ────────────────────────────────────────────────────────────────────

@app.cell
def _intro():
    import marimo as mo
    mo.md(
        r"""
        # HexGo Epiplexity Lab

        Live measurement of structural information in HexGo self-play corpora.

        Each agent in `engine/agents.py` is a time-bounded probabilistic model
        in the sense of Finzi et al. 2026. Two-part MDL:

        $$\text{MDL}_T(X) \;=\; S_T(X) + H_T(X)$$

        We estimate $H_T$ with two torch-free observers:

        - **Markov-3 back-off language model** over (relative-coord) move tokens.
          $|P|$ = gzipped pickle of the transition tables.
        - **gzip observer** — practical universal compressor; gives an upper
          bound on $H(X)$ in bits / move.

        The central question: *does $S_T(\text{corpus}_N)$ grow like $\log N$
        (Pisot substitution) or like $N$ (no finite program)?*
        """
    )
    return mo,


# ── config + paths ───────────────────────────────────────────────────────────

@app.cell
def _config():
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    RESULTS = ROOT / "results" / "epiplexity_scan.json"
    FIGURES = ROOT / "figures"
    CORPORA = ROOT / "corpora"
    return CORPORA, FIGURES, RESULTS, ROOT


# ── helpers ──────────────────────────────────────────────────────────────────

@app.cell
def _helpers(CORPORA):
    import json
    from engine import (RandomAgent, EisensteinGreedyAgent, ForkAwareAgent,
                        PotentialGradientAgent, ComboAgent)
    from engine.epiplexity import (generate_corpus, measure_corpus,
                                   agent_program_length, Corpus)

    AGENT_FACTORIES = {
        "random":     lambda: RandomAgent(),
        "greedy_def": lambda: EisensteinGreedyAgent("greedy_def", defensive=True),
        "fork_a2":    lambda: ForkAwareAgent("fork_a2", alpha=2.0),
        "fork_a4":    lambda: ForkAwareAgent("fork_a4", alpha=4.0),
        "potgrad":    lambda: PotentialGradientAgent("potgrad"),
        "combo":      lambda: ComboAgent("combo"),
    }

    def get_corpus(name, n, seed=42):
        p = CORPORA / f"{name}_N{n}.pkl.gz"
        if p.exists():
            return Corpus.load(p)
        c = generate_corpus(AGENT_FACTORIES[name], AGENT_FACTORIES[name],
                            n_games=n, seed=seed)
        c.save(p)
        return c

    return (AGENT_FACTORIES, Corpus, agent_program_length, generate_corpus,
            get_corpus, json, measure_corpus)


# ── load scan results if available ───────────────────────────────────────────

@app.cell
def _load_results(RESULTS, json):
    import marimo as mo
    if RESULTS.exists():
        SCAN = json.loads(RESULTS.read_text())
        status = f"✓ loaded cached scan ({RESULTS.name})"
    else:
        SCAN = None
        status = "(no cached scan — run  `python experiments/run_epiplexity_scan.py --quick`)"
    mo.md(f"**Scan status:** {status}")
    return SCAN,


# ── A: Paradox 1 — live bar chart ────────────────────────────────────────────

@app.cell
def _paradox1_chart(SCAN, get_corpus, measure_corpus):
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    if SCAN and "paradox1" in SCAN:
        rows = SCAN["paradox1"]
    else:
        # compute a small scan inline so the notebook has SOMETHING to show
        rows = {}
        for name in ["random", "greedy_def", "combo"]:
            c = get_corpus(name, 50)
            r = measure_corpus(c, name=name)
            rows[name] = {
                "markov_H_T": r.markov_H_T_bits_per_token,
                "gzip_bpt": r.gzip_bits_per_token,
            }

    names = list(rows.keys())
    H_T = [rows[n]["markov_H_T"] for n in names]
    gz = [rows[n]["gzip_bpt"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(names))
    ax.bar(x - 0.18, H_T, 0.36, label="Markov-3 observer")
    ax.bar(x + 0.18, gz, 0.36, label="gzip observer",
           alpha=0.55, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("cross-entropy  (bits/move)")
    ax.set_title("Paradox 1 — deterministic play is compressible; random is not")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()

    mo.md(
        "## Programme A — Paradox 1\n"
        f"Below bar heights = bits / move needed to encode the corpus.\n"
        f"Large gap between `random` and structured agents is the "
        f"**computational information** that deterministic self-play generates."
    )
    return fig, rows


# ── D: scaling log-log ───────────────────────────────────────────────────────

@app.cell
def _scaling(SCAN):
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    if not (SCAN and "scaling" in SCAN):
        mo.md("*(scaling scan not yet run — `python experiments/run_epiplexity_scan.py` to populate)*")
        return (None,)

    scaling = SCAN["scaling"]
    colors = {"random": "#888", "greedy_def": "#1f77b4", "combo": "#d62728"}
    markers = {"random": "o", "greedy_def": "s", "combo": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for name, rows in scaling.items():
        Ns = np.array([r["N"] for r in rows])
        H = np.array([r["H_T"] for r in rows])
        gz = np.array([r["gzip_total_bits"] for r in rows])
        ax1.plot(Ns, H, marker=markers.get(name, "o"), color=colors.get(name), label=name)
        ax2.plot(Ns, gz, marker=markers.get(name, "o"), color=colors.get(name), label=name)
        if len(Ns) >= 2:
            slope, _ = np.polyfit(np.log10(Ns), np.log10(gz), 1)
            ax2.annotate(f"{name}: slope ≈ {slope:.2f}",
                         xy=(0.55, 0.10 + 0.06 * list(scaling).index(name)),
                         xycoords="axes fraction",
                         color=colors.get(name), fontsize=9)
    for ax in (ax1, ax2):
        ax.set_xscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend()
        ax.set_xlabel("corpus size N (games)")
    ax1.set_ylabel(r"$H_T$  bits/move");   ax1.set_title("residual unpredictability")
    ax2.set_yscale("log"); ax2.set_ylabel("gzip bits total")
    ax2.set_title("corpus description length — slope <1 hints at finite program")
    plt.tight_layout()

    mo.md(
        "## Programme D — Pisot spectroscope\n"
        "Gzip total bits vs corpus size, log-log. Slope 1.0 = linear growth "
        "(no finite program). Slope → 0 = finite substitution system."
    )
    return fig, scaling


# ── E: Pareto frontier ───────────────────────────────────────────────────────

@app.cell
def _pareto(SCAN):
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    if not (SCAN and "pareto" in SCAN):
        mo.md("*(pareto scan not yet run)*")
        return (None,)

    points = SCAN["pareto"]
    names = list(points.keys())
    P = np.array([points[n]["prog_bytes"] for n in names])
    H = np.array([points[n]["H_T_bits"] for n in names])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(P, H, s=120, c="#2ca02c", edgecolor="black", linewidth=0.7)
    for x, y, n in zip(P, H, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    # Pareto front
    pts = sorted(zip(P, H, names))
    frontier, best = [], float("inf")
    for x, y, n in pts:
        if y < best:
            frontier.append((x, y, n)); best = y
    ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
            "--", color="crimson", alpha=0.7, label="Pareto frontier")
    ax.set_xlabel(r"$|P|$ gzipped source (bytes)")
    ax.set_ylabel(r"$H_T$ bits/move on Combo-self-play corpus")
    ax.set_title("Programme E — agents as MDL points")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()

    mo.md("## Programme E — MDL Pareto frontier")
    return fig, points


# ── toy viz: hex tiling with live agents ─────────────────────────────────────

@app.cell
def _toy_hex_viz(AGENT_FACTORIES):
    import marimo as mo
    agent_pairs = {
        "random vs random": ("random", "random"),
        "greedy vs random": ("greedy_def", "random"),
        "combo vs greedy":  ("combo", "greedy_def"),
        "combo vs combo":   ("combo", "combo"),
        "potgrad vs combo": ("potgrad", "combo"),
    }
    dropdown = mo.ui.dropdown(
        options=list(agent_pairs.keys()),
        value="combo vs greedy",
        label="matchup")
    n_moves = mo.ui.slider(2, 60, value=30, label="moves to play")
    seed = mo.ui.number(0, 9999, value=2026, label="seed")
    btn = mo.ui.run_button(label="play!")
    mo.md("## Live HexGo — play a sample game and render it").callout()
    mo.hstack([dropdown, n_moves, seed, btn])
    return agent_pairs, btn, dropdown, n_moves, seed


@app.cell
def _hex_game_fig(agent_pairs, dropdown, n_moves, seed, AGENT_FACTORIES, btn):
    import matplotlib.pyplot as plt
    from engine import HexGame
    from engine.viz import draw_board
    import random as _rand

    name_a, name_b = agent_pairs[dropdown.value]
    _rand.seed(seed.value)
    a = AGENT_FACTORIES[name_a]()
    b = AGENT_FACTORIES[name_b]()
    g = HexGame()
    m = 0
    while g.winner is None and m < n_moves.value:
        ag = a if g.current_player == 1 else b
        legal = g.legal_moves()
        if not legal: break
        mv = ag.choose_move(g)
        if mv not in set(legal): mv = _rand.choice(legal)
        g.make(*mv); m += 1

    fig, ax = plt.subplots(figsize=(7, 6.3))
    draw_board(g, ax=ax, show_potential=True, show_threats=True, show_forks=True,
               title=f"{name_a}  vs  {name_b}   |   moves: {m}   |   winner: {g.winner or 'ongoing'}")
    plt.tight_layout()
    return fig, g


# ── token-entropy live plot ─────────────────────────────────────────────────

@app.cell
def _per_move_entropy(get_corpus, measure_corpus):
    """Compute per-position cross-entropy curves so we can see where structure lives."""
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    from engine.epiplexity import MarkovBackoffObserver, corpus_token_stream, _relative_tokens

    curves = {}
    for name in ["random", "greedy_def", "combo"]:
        c = get_corpus(name, 50)
        toks_train = corpus_token_stream(c)
        # simple hold-out: train on first 80% of games
        split = int(0.8 * len(c.games))
        train = corpus_token_stream(type(c)(games=c.games[:split], manifest=c.manifest))
        held = c.games[split:]
        obs = MarkovBackoffObserver(max_order=3).fit(train)
        obs.fit_weights(corpus_token_stream(type(c)(games=held, manifest=c.manifest)))
        # per-move surprise averaged across held-out games
        max_len = max((len(g.moves) for g in held), default=0)
        sums = np.zeros(max_len); counts = np.zeros(max_len)
        for g in held:
            toks = _relative_tokens(g)
            ctx: list = []
            for i, t in enumerate(toks):
                p = obs.prob(ctx, t)
                sums[i] += -math.log2(max(p, 1e-12))
                counts[i] += 1
                ctx.append(t)
        mean = sums / np.maximum(counts, 1)
        curves[name] = mean

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name, y in curves.items():
        ax.plot(y, label=name)
    ax.set_xlabel("move number")
    ax.set_ylabel("mean surprise  (bits)")
    ax.set_title("Time-resolved $H_T$ — where does structure live within a game?")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    mo.md(
        "## Time-resolved epiplexity\n"
        "Observer surprise as a function of move number. Mid-game peaks = fork cascade era; "
        "endgame troughs = forced sequences. This is a cheap version of the NOTES.md "
        "*time-resolved $S_T$* idea."
    )
    return curves, fig


# ── scratch ─────────────────────────────────────────────────────────────────

@app.cell
def _scratch():
    import marimo as mo
    mo.md(
        r"""
        ## Scratchpad

        **Next measurements to wire in:**
        - Programme B orderings (`engine/orderings.py`) — same corpus, six
          permutations, six cross-entropies. Pre-registered in ROADMAPv2 §3.
        - Programme C linear probes (`engine/probes.py`) — needs a trained
          observer with extractable representations; gzip observer is
          insufficient for this. Deferred until we optionally add a
          small torch model.
        - Bigger corpora for Programme D — need N=10^4 before the slope
          estimate becomes trustworthy.

        **Sanity checks that passed so far:**
        - `random` corpus at ≥ 10× more bits/move than `combo` corpus.
        - `combo` corpus cross-entropy < greedy cross-entropy
          (stronger agent ⇒ more compressible play ⇒ more structure).
        - Agent |P| order (gzipped source): random < greedy < combo < fork < potgrad.

        See `notebooks/NOTES.md` for technical-risk commentary.
        """
    )
    return


if __name__ == "__main__":
    app.run()
