# CLAUDE.md — onboarding for Claude agents working on hexgo-theory

You are joining a solo research project by Leon. This file is your orientation. Read it once, act on it always.

## What this project is

**hexgo-theory** is the theoretical sibling of the `hexgo` game engine (sits at `../hexgo` relative to this directory on disk — see note under "Worktree gotcha" below). The engine plays Connect-6 on the infinite hex lattice ($\mathbb{Z}[\omega]$, win = 6 stones in a row along any of 3 axes). The theoretical goal is to **characterise the structure of optimal play** — its symmetries, tiling properties, descriptive-set-theoretic complexity, and whether perfect play exhibits quasi-crystalline order.

The **final output** is a publishable paper / long-form blog post combining:
- epiplexity (time-bounded MDL, per Finzi et al. 2026) as the measurement framework
- combinatorial-game-theoretic results (strategy-stealing, pairing strategies, Hales-Jewett-style bounds)
- topological / descriptive-set-theoretic positioning (where does HexGo sit in the Borel hierarchy)
- empirical validation via self-play corpora + diffraction analysis

Everything in this repo should be either feeding that write-up or falsifying part of its thesis.

## Read this first — in order

1. **[README.md](README.md)** — the external framing and central "Pisot quasicrystal" conjecture
2. **[docs/ROADMAP.md](docs/ROADMAP.md)** — canonical 12-month plan, organised around the Finzi epiplexity paradoxes
3. **[docs/theory/](docs/theory/)** — living synthesis of individual theoretical threads (Hamkins paper synthesis, complexity positioning, etc.). Append here when you develop an idea; don't scatter theory across random files.
4. **[papers/](papers/)** — PDFs we build on. Currently: Finzi et al. (epiplexity, 2026), Hamkins–Leonessi (Infinite Hex is a draw, 2022).

## Agent expectations

### Research-mode defaults
- Ideation MUST cite real repo symbols with file:line refs — e.g. `live_lines` at [engine/analysis.py:47](engine/analysis.py:47). Vague handwaving is rejected.
- Every theoretical claim needs a **falsifiable prediction** you can tie to an experiment in `experiments/`.
- Unify narratives rather than stacking them. If your proposed direction doesn't land on the `(|P|, H_T)` MDL plane of the ROADMAP, explain why it's still worth the detour.
- Cite **Hamkins-style descriptive set theory** (Σ⁰ₙ, Π⁰ₙ, analytic, projective) where it clarifies; don't invoke it for flavour.
- Write at post-graduate level. Leon reads these synthesis notes carefully; they should be tight.

### What to build and not build
- `engine/` — game mechanics, agents, analysis helpers. Keep small and focused. Upstream game code lives in `../hexgo/`; we re-export via [engine/__init__.py](engine/__init__.py).
- `experiments/run_*.py` — one file per self-contained experiment. Must produce `results/<name>.json` and `figures/fig_<name>_*.png`. Existing examples: [run_epiplexity_scan.py](experiments/run_epiplexity_scan.py), [run_hamkins_echo.py](experiments/run_hamkins_echo.py).
- Don't create a new module just to hold one function. Prefer existing files.
- Don't add backward-compat shims, feature flags, or speculative abstractions.
- Don't write comments that restate what the code does. Comments only for *why* non-obvious decisions are the way they are.

### Experiments convention
- One entry point per experiment: `experiments/run_<topic>.py`.
- `--quick` flag for dev iteration (~1 min), full sweep as default.
- Output JSON to `results/`, PNG to `figures/` — both are tracked in git (reproducibility > repo size for this project).
- Seed everything. Reproducibility is a prerequisite for every claim.
- **Use GPU / parallelism where it helps.** Leon has a 5GB RTX 2060. For any compute that scales with corpus size — self-play batching, diffraction FFTs, tensor ops, MCTS rollouts — default to torch+CUDA. For embarrassingly parallel game-playing, use `multiprocessing.Pool` or `joblib`. Fall back to sequential CPU only when CUDA is unavailable or the problem is trivially small. Budget VRAM carefully (5GB is tight — prefer float32, small batches). Report wall time so speedups are visible.

### Worktree gotcha
`engine/__init__.py` computes the hexgo import path as `Path(__file__).parent.parent.parent / "hexgo"`. When working in a `.claude/worktrees/*` checkout, that path is wrong — the real hexgo repo is at `C:\Users\Leon\Desktop\Psychograph\hexgo`. Either run from the main checkout, or (if you must run in a worktree) prepend the real path to `sys.path` explicitly before `from engine import ...` — see the pattern in [experiments/run_hamkins_echo.py](experiments/run_hamkins_echo.py).

## Current research state (keep this updated)

**Last updated: 2026-04-17.**

### Active threads
- **Epiplexity scan** (ROADMAP Programmes A, D, E): running, infrastructure in [run_epiplexity_scan.py](experiments/run_epiplexity_scan.py). Measures S_T, H_T for random vs structured agents; tests whether corpus description length saturates (Pisot conjecture prediction).
- **Hamkins echo** ([run_hamkins_echo.py](experiments/run_hamkins_echo.py)): does draw fraction rise with horizon? Pilot says *no* — strong play is decisive, not draw-prone. Full 5×3×50 sweep currently running.
- **Descriptive complexity positioning**: HexGo payoff = $\Sigma^0_1$ open, determined by Gale–Stewart directly. Infinite Hex (Hamkins) = $\Sigma^0_7$ per Törnä. We are *below* their game in complexity, which is why finite-horizon analysis is the right tool for us.

### Pending experiments (in priority order)
1. **Diffraction spectrum of long self-play** — tests Leon's quasicrystal conjecture directly. Compute $|\sum_j e^{ik\cdot x_j}|^2$ over stone positions from a Combo-vs-Combo game at move 100+. Pure-point spectrum ⇒ Meyer set ⇒ quasicrystal.
2. **First-mover-advantage curve** vs opening ply / agent strength — falsifies or strengthens the "perfect play is a first-player win" expectation.
3. **Mirror / pairing agent** — port Hamkins §3 mirroring strategy to our axis-alignment game as `MirrorAgent` in [engine/agents.py](engine/agents.py). Predicted loss rate vs Combo < 30%.

### Known-resolved
- `ROADMAPv2.md` → `docs/ROADMAP.md`. The old v1 file is gone. If a comment or note still says "ROADMAPv2", that's a text reference, not a dead link — leave it or fix opportunistically.

## Invariants — do not violate

- `WIN_LENGTH` is asserted `== 6` deep inside the upstream engine ([../hexgo/game.py:147](../hexgo/game.py:147)). Sweeping it requires patching the assertion; don't do this without flagging it explicitly.
- The hex turn rule is **1-2-2** (P1 places 1 stone on opening; thereafter each turn = 2 placements). Agents written assuming standard 1-1 alternation will mis-play.
- Agents only need `name: str` and `choose_move(game) -> (q, r)`. Don't subclass or require a protocol — keep it flat.

## On tooling
- Always prefer editing existing files over creating new ones.
- Don't invoke skills that don't apply. `brainstorming` is for creative/UI design work; most theory questions here are answered by reading, writing, and running experiments, not by running the brainstorming flow.
- For short theoretical replies Leon prefers: (a) cite real repo symbols, (b) unify narratives, (c) end with falsifiable predictions or next experiments. See his saved preferences in `~/.claude/projects/.../memory/`.
