# HexGo Theory — Roadmap

Phased research plan with verification checkpoints.
Tests are in `tests/` and run headlessly via `.\run_tests.bat`.

---

## Phase 1 — Foundation (current)

Core analysis infrastructure and dashboard. Establish the empirical baseline.

### 1a. Engine bridge & analysis
- [x] Bridge to hexgo `game.py` and `elo.py` without copying code
- [x] `live_lines(game)` — all unblocked 6-windows
- [x] `threat_cells(game, player)` — cells completing a win
- [x] `fork_cells(game, player)` — cells extending 2+ axes simultaneously
- [x] `potential_map(game)` — Erdős-Selfridge potential per cell
- [x] `axis_chain_lengths(game, player)` — chain length per axis per stone
- [x] `pair_correlation(moves)` — g(r) spatial correlation function
- [x] `live_ap_count(game)` — live 6-APs per player
- [x] `pattern_fingerprint(game, radius)` — local neighbourhood encoding
- [ ] Verify `fork_cells` correctly identifies all D6-equivalent forks
- [ ] Verify `potential_map` is monotone in chain length (more stones → higher potential)

### 1b. Dashboard
- [x] Hex grid with potential heatmap, threat borders, fork markers, axis overlays
- [x] Triangular lattice dual view with axis-coloured edges
- [x] Threat hypergraph with spring layout
- [x] Analysis panel: threats, forks, live APs, chain bars, g(r) sparkline
- [x] Experiment runner (5 types), 30fps paint throttle, step delay slider
- [ ] Fix: game tree viewer (collapsible forcing graph, not yet built)
- [ ] Fix: "center on board" re-centers correctly after many moves
- [ ] Feature: replay mode — step through a saved game frame by frame
- [ ] Feature: export current board as SVG/PNG

### 1c. Tests
- [x] `test_analysis.py` — 25 tests covering all analysis functions
- [x] `test_runner.py` — 14 tests covering experiment worker logic headlessly
- [ ] All tests passing: `.\run_tests.bat`
- [ ] CI: add GitHub Actions workflow running tests on push

---

## Phase 2 — Fork Geometry

The central gap in `EisensteinGreedyAgent`: it scores by `max(chain)` across axes,
making it blind to fork cells. This phase characterises fork structure precisely.

### Goals
- [ ] **Fork frequency census**: run 1000 Eis-vs-Eis games, count fork cells per move,
      plot distribution. Hypothesis: forks appear from move ~10 onward.
- [ ] **Fork vs win correlation**: does the player who creates more fork cells win more often?
      Prediction: yes, fork cell count is a leading indicator of win probability.
- [ ] **Triple fork existence**: does a cell ever sit on all 3 axes simultaneously?
      Prove or disprove: in Z[ω], a cell at the intersection of 3 live 6-windows
      across all three axes is a "triple fork" — is this geometrically achievable?
- [ ] **Greedy fork blindness proof**: construct a specific game state where
      Eisenstein-greedy misses a winning fork that a fork-aware agent would take.
      Use this as a unit test `test_greedy_fork_blindness()`.
- [ ] **ForkAwareAgent**: implement a new agent that scores moves by
      `chain_score + alpha * fork_count`, find optimal alpha empirically.
      Compare ELO: ForkAware vs EisensteinGreedy.
- [ ] **Test**: `test_fork_aware_beats_greedy()` — ForkAware wins >60% over 100 games.

---

## Phase 3 — Substitution Structure

Test the quasi-crystal conjecture: does optimal play exhibit a finite substitution tiling?

### Goals
- [ ] **Pattern type census**: collect radius-2 local patterns across 10,000 games.
      Plot frequency distribution. If it converges to a power law, that's substitution evidence.
- [ ] **Forcing graph construction**: for board states within radius 5, enumerate all
      states where exactly one move (up to D6) avoids immediate loss. Build graph.
      - [ ] Check if forcing graph is finite (finitely many types up to D6)
      - [ ] If finite: extract substitution rules (which types map to which)
- [ ] **Substitution matrix**: from the forcing graph, build the |types| × |types|
      substitution matrix M where M[i][j] = how many type-j patterns a type-i pattern
      generates. Compute Perron-Frobenius eigenvalue.
      - [ ] Is the eigenvalue a Pisot number (algebraic integer, all conjugates < 1 in modulus)?
      - [ ] Candidates: tribonacci ~1.3247, golden ratio ~1.618, their products
- [ ] **Test**: `test_substitution_matrix_pisot()` — eigenvalue passes Pisot criterion.
- [ ] **Inflation constant visualisation**: if Pisot, overlay the tiling at inflation scales
      on the hex grid (cells at distances 1, λ, λ², λ³ from the first stone).

---

## Phase 4 — Spectral Analysis

Empirical diffraction spectrum — does the point set of optimal moves look like a quasi-crystal?

### Goals
- [ ] **Corpus collection**: generate 10,000 games between the strongest available agent
      (ForkAware or eventual net-guided agent). Extract all move positions as a point set.
- [ ] **2D DFT of occupied cells**: compute |F(k)|² where F is the Fourier transform of
      the indicator function of occupied cells. Plot as 2D image.
      - [ ] Pure point spectrum (sharp Bragg peaks) → quasi-crystal
      - [ ] Absolutely continuous → random/disordered
      - [ ] Mixed → intermediate
- [ ] **D6 symmetry test**: does the diffraction pattern have 6-fold rotational symmetry?
      Compute rotational symmetry score: correlation of pattern with 60° rotations.
- [ ] **Radial profile g(r)**: pair correlation function across full corpus.
      - [ ] Quasi-periodic oscillations (peaks at irrational multiples of lattice spacing) → quasi-crystalline
      - [ ] Exponential decay → liquid-like disorder
- [ ] **Test**: `test_corpus_d6_symmetry()` — rotational symmetry score > 0.8.
- [ ] **Test**: `test_pair_correlation_quasi_periodic()` — g(r) has local maxima
      at r values inconsistent with any periodic lattice.

---

## Phase 5 — Algebraic Structure

Formal connection to A₂ root system and Coxeter theory.

### Goals
- [ ] **A₂ hyperplane arrangement**: implement the arrangement of all potential 6-lines
      as hyperplanes in ℝ². Visualise chamber decomposition.
- [ ] **Coxeter word encoding**: encode optimal game paths as words in the Coxeter group
      of the A₂ arrangement (generators = reflections across hyperplanes).
      - [ ] Are optimal paths geodesic in the Cayley graph of the Coxeter group?
- [ ] **Van der Waerden density**: estimate the maximum density of a 2-colouring of Z[ω]
      that avoids monochromatic 6-APs. Lower bound = 1/6 (trivial). Upper bound = ?
- [ ] **Sublattice coset structure**: which cosets of ⟨6u₁⟩, ⟨6u₂⟩, ⟨6u₃⟩ are
      "strategically equivalent" (interchangeable under D6 without changing game value)?
- [ ] **Test**: `test_coset_equivalence()` — positions related by coset shift have
      equal Erdős-Selfridge potential (up to D6 tolerance).

---

## Phase 6 — Agent Hierarchy & Validation

Build progressively stronger hand-crafted agents as rungs on an ELO ladder,
each embodying a deeper theoretical insight.

| Agent | Key idea | Target ELO vs Eisenstein |
|-------|----------|--------------------------|
| `EisensteinGreedyAgent` | max chain length, greedy | 0 (baseline) |
| `ForkAwareAgent` | chain + fork multiplicity | +200 |
| `SubstitutionAgent` | plays to substitution tiling templates | +400 |
| `PotentialGradientAgent` | follows Erdős-Selfridge gradient | +300 |
| `CoxeterAgent` | geodesic Coxeter word strategies | TBD |

- [ ] Implement ForkAwareAgent (Phase 2 output)
- [ ] Implement PotentialGradientAgent
- [ ] ELO ladder: run round-robin of all agents, 200 games each pair
- [ ] Test: each successive agent beats previous by >55% win rate

---

## Near-Term Priorities

Based on the analysis so far, the most promising immediate direction is **Phase 2 (Fork Geometry)**:

1. The `EisensteinGreedyAgent`'s `max()` scoring is provably blind to forks — this is the
   clearest gap between greedy and perfect play, and it's small enough to close quickly.
2. A `ForkAwareAgent` that simply adds `alpha * fork_count` to the score is a one-line change
   with potentially large ELO gain — cheap to test.
3. The fork census will produce data that directly feeds Phase 3 (substitution structure),
   since fork cells are the "intersection points" of the substitution tiling.

Phase 4 (diffraction spectrum) requires a corpus large enough to get statistical signal —
this is the most computationally intensive phase and should be deferred until we have a
stronger agent to generate interesting games.

Phase 5 (algebraic structure) is the most mathematically ambitious and least time-constrained.
Work on it in parallel with Phase 2-4 as the theory develops.
