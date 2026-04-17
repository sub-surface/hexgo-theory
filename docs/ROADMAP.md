# HexGo Theory — Roadmap v2 (Epiplexity Overhaul)

> **Version:** 2.0 · canonical roadmap (v1 retired 2026-04-17)
> **Framing paper:** Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson — *From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence*, arXiv:2601.03220v2 (March 2026). Local copy: `papers/finzi-2026-epiplexity.pdf`.
> **Horizon:** 12 months, organised into four quarterly programmes with clear verification gates.
> **Thesis of v2:** the two parallel narratives of the original roadmap — *quasi-crystal / Pisot substitution* on one side, *ELO agent ladder* on the other — are the **same object** viewed through the lens of epiplexity. Every experiment in v2 outputs a point on a two-axis plot: `(program-length |P|, time-bounded cross-entropy H_T)`. ELO is the downstream task; Pisot structure is the saturation of `S_T`.

---

## 0. Reading the paper in one paragraph

Finzi et al. define, for a random variable `X` over `{0,1}^n` and a time budget `T`, the **time-bounded two-part MDL code length** of the minimising program `P*`:

```
  P*  = argmin_{P ∈ P_T} ( |P| + E[log 1/P(X)] )
  S_T(X) := |P*|                   ← epiplexity (structural content)
  H_T(X) := E[log 1/P*(X)]         ← time-bounded entropy (residual randomness)
  MDL_T(X) = S_T(X) + H_T(X)
```

Three paradoxes of classical information theory dissolve:

1. **Deterministic transforms cannot add Shannon info** — yet self-play, PRNGs, chaos clearly teach bounded learners. Resolution: they add *epiplexity*, not entropy.
2. **Factorisation order is invariant under Shannon/Kolmogorov** — yet L→R beats R→L on English text. Resolution: order changes `S_T` without changing `H`.
3. **Likelihood-maximising models are distribution-matching** — yet models can learn more structure than is "in" the generator (Game of Life). Resolution: the generator's program is short; the *unrolled* behaviour has vast epiplexity for a computationally bounded observer.

**Why this matters for HexGo.** HexGo's rules are ~200 bytes of Python. Eisenstein-greedy self-play is deterministic given a tie-break seed. So `K(corpus) ≈ |rules| + |agent| + |seed|` is tiny — yet we empirically see fork structures, threat cascades, and (conjecturally) a Pisot quasi-crystal emerge. This is paradox 3 at a human-tractable scale, where the generating process is fully specified.

**The unifying claim of v2:** the Pisot inflation constant `λ` is not a separate mathematical curio from the agent ladder. It *is* the rate at which `S_T` of a self-play corpus saturates with corpus size `N`. A finite substitution system ⟺ `S_T(corpus_N)` grows like `O(log N)` rather than `Ω(N)`. Measuring `S_T` gives a spectroscopic handle on the Pisot conjecture without first enumerating the forcing graph.

---

## 1. Guiding principles for v2

1. **Every experiment produces a point in MDL space.** Instrumentation emits `(|P|, H_T)` alongside whatever domain metric (ELO, win-rate, fork-count) it was already emitting.
2. **Paradoxes, not phases.** The organising structure is the paper's three paradoxes, with a fourth "synthesis" programme tying them to the Pisot conjecture. This is more falsifiable than phase-gated milestones.
3. **Real-time lab notebook.** A marimo notebook (`notebooks/epiplexity_lab.py`) is the shared scratchpad for all measurements; every figure in any eventual write-up should reproducibly come from a cell there.
4. **Bounded-observer agents are first-class objects, not baselines.** `ForkAwareAgent`, `PotentialGradientAgent`, `ComboAgent`, and the eventual `CoxeterAgent` are *explicit time-bounded models* `P ∈ P_T` in the paper's formalism. Their source code length is their `|P|`; their move-prediction cross-entropy is their `H_T`.
5. **Falsifiability before generalisation.** Each programme has a concrete prediction that could kill it. Listed in §8.

---

## 2. Programme A — Paradox 1: *Information created by computation* (Q1)

**Claim under test.** Deterministic Eisenstein self-play corpora contain structural information that a neural observer can exploit, even though the corpus has trivial Kolmogorov complexity.

### A.1 Deterministic corpus generation
- Implement `experiments/corpus.py::generate_corpus(agent_spec, n_games, seed)` returning a pickled list of `GameEvent` with a manifest `{agent_source_sha, seed, n_games, rules_sha}`. The manifest *is* the short program that generates the corpus — its length in bytes is the Kolmogorov upper bound.
- Generate, at minimum, the following corpora (each size `N ∈ {10², 10³, 10⁴, 10⁵}`):
  - `det_greedy` — `EisensteinGreedyAgent` deterministic (`eps=0`)
  - `noisy_greedy` — same, `eps=0.01`
  - `det_fork_a4` — `ForkAwareAgent(alpha=4, eps=0)`
  - `det_combo` — `ComboAgent` deterministic
  - `random` — `RandomAgent` (control: should have high `H_T` but low `S_T`)
  - `det_combo_selfbootstrap` — iterate: train small policy on `det_combo` corpus, self-play, retrain. 5 generations. This is the AlphaZero analogue.

### A.2 Bounded-observer learner
- A single canonical small transformer (`engine/observer_net.py`, ~300k params, 4-layer, d=64) trained to predict next move given the move-prefix. Always identical architecture; only the training corpus varies.
- Training loss curve area *above* the irreducible-loss floor is the §4 heuristic for epiplexity. Cumulative KL (teacher / student) is the rigorous version — implement both.

### A.3 Measurements
- `H_T(corpus)` ← irreducible test loss of a large-overparameterised observer (ceiling).
- `S_T(corpus) ≈ area_under_curve − irreducible_loss × n_steps`, the heuristic.
- Independently, compute `|P|` of each generating agent as the gzipped byte length of its Python source (a crude but honest upper bound on its description length).

### A.4 Verification gate (A-gate)
- `S_T(det_greedy) ≫ S_T(random)` despite similar `|P|`. Target: at least 10× difference in AUC.
- `S_T(det_combo_selfbootstrap[gen=5]) > S_T(det_combo_selfbootstrap[gen=0])` — **self-play literally creates structural information**, reproducing the paper's Paradox 1 in a hand-crafted setting. This is the publishable result of Q1.

---

## 3. Programme B — Paradox 2: *Factorisation-order asymmetry* (Q2)

**Claim under test.** The same game corpus, reordered, has different epiplexity. HexGo is the cleanest possible testbed because the reordering is geometric, not linguistic.

### B.1 Six canonical orderings of a game
For each game, produce six token streams from the same underlying move set:

1. **Temporal** — natural (1, 2, 3, …)
2. **Reverse-temporal** — (T, T-1, …, 1) — paper's R→L analogue
3. **Radial-in** — sort by `|q + ωr|` ascending
4. **Radial-out** — sort descending
5. **Axis-stratified** — all u₁ moves, then u₂, then u₃
6. **D₆-canonicalised-temporal** — rotate/reflect each game to a canonical orbit representative, then temporal

Shannon entropy of the corpus is invariant under all six. Epiplexity is not.

### B.2 Per-ordering observer training
- Train the canonical observer on each of the 6 ordered corpora (same `N = 10⁴`, same architecture, same seed).
- Report `H_T`, `S_T`, and — crucially — *downstream transfer*: does an observer trained on ordering X extract structure that transfers to predicting forks in natural temporal order?

### B.3 Predictions (pre-registered)
- `S_T(temporal) < S_T(reverse-temporal)` by at least 15% — **arrow of time for game tactics**.
- `S_T(radial-in) < S_T(radial-out)` — propagation from origin is the natural causal direction.
- `S_T(D₆-canonical-temporal) < S_T(temporal)` — removing the six-fold redundancy compresses the corpus.

### B.4 Verification gate (B-gate)
- At least two of the three predictions above hold at p < 0.01 across three independent corpus seeds. If all three fail, the bounded-observer hypothesis is wrong for this domain and the programme pivots.

---

## 4. Programme C — Paradox 3: *Likelihood matches generator, but observer exceeds it* (Q3)

**Claim under test.** The `ComboAgent`'s move distribution is a valid generator for its own self-play corpus (distribution-matching fixed point). Yet a learned observer trained on that corpus can score moves better than `ComboAgent` itself on held-out positions, because the observer extracts emergent fork-cascade structure that `ComboAgent` does not represent explicitly.

This is the HexGo analogue of "a model trained on Conway's Life outputs learns about gliders even though the rules don't mention them."

### C.1 Emergent-structure probes
Define a library of *post-hoc* structural predicates that an agent is not told about at train time:

- `is_triple_fork(cell, game)` — cell is a fork on all three axes simultaneously
- `is_forced_response(cell, game)` — cell is the unique reply to a double-threat
- `is_ladder_foot(cell, game)` — cell initiates a forced k-step ladder
- `lives_on_substitution_vertex(cell, game, tiling_hypothesis)` — cell aligns with conjectured Pisot tile vertices

### C.2 Linear-probe analysis
- Freeze the trained observer; train a single linear classifier on its internal representations to predict each probe.
- Report probe accuracy vs. a baseline observer trained on `random` corpus (which should be at chance).

### C.3 Prediction
Observers trained on `det_combo` should linearly decode `is_triple_fork` and `is_forced_response` at ≥ 85% accuracy, even though `ComboAgent` itself does not compute these predicates. That gap is *emergent structural information* — epiplexity made concrete.

### C.4 Verification gate (C-gate)
- At least 2 of the 4 probes exceed 85% linear-probe accuracy on `det_combo`-trained observers, while staying ≤ 60% on `random`-trained. This is the empirical signature that *likelihood training extracts more structure than the generator represents*.

---

## 5. Programme D — Synthesis: *Epiplexity scaling as a Pisot spectroscope* (Q4)

**Claim under test.** If perfect play admits a finite substitution tiling with Pisot inflation constant `λ`, then `S_T(corpus_N)` grows like `C + α log_λ N` for large `N`. Measuring the slope `α / log λ` gives an empirical estimate of `log λ`.

This is the most ambitious — and most original — direction, and it is the **bridge between the two original narratives**.

### D.1 Minimum-observer-size scan
- For each corpus size `N ∈ {10², 10^2.5, 10³, 10^3.5, 10⁴, 10^4.5, 10⁵}` and corpus type `T ∈ {greedy, combo, combo_selfbootstrap[gen=5]}`:
  - Find the smallest observer (by parameter count or gzipped checkpoint size) that achieves loss within ε of the irreducible `H_T`.
  - This is the empirical `S_T(corpus_N)` at budget `T`.
- Plot `S_T` vs `log N`.

### D.2 Regression regimes
- **Sublogarithmic saturation** (`S_T` flattens to a constant): strong substitution, program is finite, perfect play has bounded type-alphabet. Pisot conjecture supported.
- **Logarithmic growth** (`S_T = α log N + β`): substitution with finite inflation rate. Estimate `λ = exp(1/α)`. Compare to known Pisot candidates (tribonacci ≈ 1.3247, golden ratio ≈ 1.618, plastic number ≈ 1.3247).
- **Linear growth** (`S_T = Ω(N)`): no finite generative program; Pisot conjecture empirically refuted in this regime.

### D.3 Diffraction cross-check
- Parallel track: run the ROADMAP v1 §4 diffraction spectrum on the same corpora.
- Compare: does the `λ` estimated from epiplexity scaling match the `λ` read off diffraction peak spacings?
- **Convergence of the two independent estimates is the strongest possible empirical evidence for the Pisot conjecture** — and a genuinely novel methodological contribution.

### D.4 Verification gate (D-gate — the main result)
- `|λ_epiplexity − λ_diffraction| < 0.05` on `combo_selfbootstrap[gen=5]` corpora at `N = 10⁵`. This is a very high bar. A less strict version is an acceptable fallback: both estimates fall in the same Pisot-family interval (e.g., both in `[1.30, 1.35]`).

---

## 6. Programme E — Agent ladder rebuilt as MDL Pareto frontier (ongoing, Q1→Q4)

The existing agent ladder is reframed, not replaced.

### E.1 Agents as time-bounded models
For each agent `A`:
- `|P_A| :=` byte length of gzipped canonical source (deterministic serialisation — pin Python version).
- `H_T(A; corpus)` := cross-entropy of `A`'s softmax-scored move distribution (temperature τ = 0.5, default) against held-out corpus.
- `MDL_T(A; corpus) = |P_A| + n · H_T(A; corpus)` — the joint description length of agent + moves.

### E.2 Pareto frontier plot
- Scatter all agents in `(|P|, H_T)` space.
- Draw the Pareto frontier (lower-left convex hull).
- Overlay the observer-net as a single point (large `|P|`, small `H_T`) — it competes on predictive power but loses on parsimony.

### E.3 New agents motivated by gaps on the frontier

**Primary (2026-04-17 pivot):** five neural-CA variants with different inductive priors, trained under self-play, competing in a round-robin to *select* the strongest agent on the right-hand side of the frontier. See synthesis note [docs/theory/2026-04-17-hamkins-synthesis.md](../docs/theory/2026-04-17-hamkins-synthesis.md) §7 for the prior taxonomy. The substrate is the existing `NeuralCAAgent` ([engine/neural_ca.py](../engine/neural_ca.py)) — a hex-conv stack on torch+CUDA.

Variants:
  1. `nca_random_init` — control. Already a point on the frontier ([experiments/run_neural_ca.py](../experiments/run_neural_ca.py)).
  2. `nca_d6_tied` — weight-tie conv filters over the 12-element $D_6$ group. 12× smaller effective parameter count.
  3. `nca_line_detector` — depth-1 kernels initialised to respond to adjacent own-stone pairs along each Eisenstein axis.
  4. `nca_erdos_selfridge` — initialiser encoding the discretised $\phi(c) = \sum_L \alpha^{n_L^\text{own}} \mathbb{1}[n_L^\text{opp}=0]$ potential.
  5. `nca_combo` — priors (2) + (3) + (4) stacked.

Tournament (run via [experiments/harness.py](../experiments/harness.py)'s `run_round_robin`) gives both a round-robin win-rate matrix (the **CA-prior ablation study** — publishable on its own) and a single champion. The champion replaces `ca_combo_v2` at the top of the ladder for downstream Programme D measurements.

**Stretch (optional, was Primary in v2.0):** `SubstitutionAgent` (tied to D.2 tile templates), `CoxeterAgent` (A₂-geodesic player), `MDLEGreedy` (minimises post-move MDL under a small observer). These are worth adding as frontier points if time permits but are not on the critical path to the paper.

### E.4 Why replace hand-crafted agents with a neural zoo

The hand-crafted ladder (`random → greedy → fork_aware → combo → ca_combo_v2`) empirically plateaus: the 2026-04-17 FMA curve ([results/fma_curve.json](../results/fma_curve.json)) shows ca_combo_v2 at $p_B = 0.44$ [0.31, 0.58] — not meaningfully stronger than combo. Adding more hand-engineered features is a diminishing-returns direction. Training under self-play with architectural priors is the natural way to extend the frontier *rightward* (toward higher $|P|$, lower $H_T$) without hand-crafting every feature. Crucially, the zoo is also an **ablation experiment** — it falsifies or confirms the §7 prediction that $D_6$-equivariance is the load-bearing prior and tactical priors are not.

### E.4 ELO ↔ MDL correlation
- Compute Pearson `r` between ELO rating and negative `MDL_T`.
- Strong negative correlation validates MDL as a proxy for strategic quality. This lets us rank agents *without* running round-robin tournaments — a big compute saving.

---

## 7. Programme F — Infrastructure (throughout)

### F.1 Marimo notebook (`notebooks/epiplexity_lab.py`)
- Real-time visualisation of corpus generation, observer training, and `(|P|, H_T)` scatter.
- Cells organised to mirror programmes A–D, so any measurement can be re-run interactively.
- Initial scaffold written in this overhaul (see §10).

### F.2 Engine additions
- `engine/epiplexity.py` — code-length and entropy measurements.
  - `agent_program_length(agent) → int` (gzipped source bytes)
  - `cross_entropy(agent, corpus, temperature=0.5) → float`
  - `two_part_mdl(agent, corpus) → float`
  - `irreducible_loss(corpus, max_model_size=10_000_000) → float`
- `engine/observer_net.py` — the canonical small transformer. Keep architecture *frozen* for the year so that results are comparable across programmes.
- `engine/orderings.py` — the six canonical move-stream permutations from §3.
- `engine/probes.py` — structural predicates from §4.

### F.3 Data management
- `corpora/` directory, gitignored, holds pickled corpora with manifests.
- `results/` directory, committed, holds JSON/CSV summaries of every run.
- `checkpoints/` directory, gitignored, holds observer-net checkpoints keyed by corpus hash.

### F.4 Testing
- Unit tests for all `engine/epiplexity.py` functions in `tests/test_epiplexity.py`.
- Golden-file tests for the six orderings (ensure Shannon-entropy-preservation).
- Regression test: `MDL_T(random_agent; random_corpus)` should be within 1% of `n` — a sanity check that random-on-random truly has no structure.

---

## 8. Falsifiability — what would kill each programme

| Programme | Kill condition | Graceful exit |
|-----------|----------------|---------------|
| A (Paradox 1) | No gap between `S_T(deterministic)` and `S_T(random)` at any corpus size | Investigate whether HexGo is so constrained that greedy play is near-random in predictive structure — would itself be a negative result worth reporting |
| B (Paradox 2) | All six orderings give within-noise identical `S_T` | Reframe: board game moves are already D₆-canonical up to small corrections, so ordering effects may be weaker than in LLMs. Report as null result. |
| C (Paradox 3) | Linear probes fail on all four structural predicates | Try richer probes (MLP, not linear); if still null, the observer is not extracting the predicates we thought — valuable to know |
| D (Synthesis) | `S_T` grows linearly with `N` in all regimes | **This falsifies the Pisot conjecture empirically.** Report as major negative result; pivot to characterising the non-substitution structure |
| E (Ladder) | No correlation between MDL and ELO | Strategic quality is not captured by next-move prediction — interesting finding; reframe as evidence for look-ahead mattering beyond single-move policy |
| E.3 (NCA zoo) | Random-init NCA ties or beats all prior-initialised variants in tournament | Inductive bias adds nothing in this domain. Reports as a negative result on the §7 "symmetry is load-bearing" claim. Still provides a strongest-available agent for Programme D; the paper section becomes "tried CA priors, none beat random init — use random init going forward." |

---

## 9. Milestones and timeline

Quarters are calibrated to "one FTE-equivalent of focused hobby time." Adjust ruthlessly; the ordering matters more than the dates.

### Q1 (months 1–3) — Programme A + infrastructure
- Weeks 1–2: `engine/epiplexity.py`, `engine/observer_net.py`, marimo scaffold live and tested
- Weeks 3–6: corpus generation (A.1), canonical observer trained on all corpora
- Weeks 7–10: `S_T` measurement, selfbootstrap loop running
- Weeks 11–13: **A-gate verification**, short write-up of Paradox 1 result

### Q2 (months 4–6) — Programme B + ladder reframe
- Weeks 14–17: `engine/orderings.py`, six-ordering corpora generated
- Weeks 18–21: per-ordering observer training, significance testing
- Weeks 22–24: **B-gate verification**; Programme E Pareto plot computed, `r(MDL, ELO)` reported
- Weeks 25–26: integration — start using MDL as agent-selection heuristic in `elo_ladder.py`

### Q3 (months 7–9) — Programme C + NCA zoo training & tournament
- Weeks 27–30: `engine/probes.py`, linear-probe harness
- Weeks 31–33: **C-gate verification**; emergent-structure analysis
- Weeks 34–37: NCA zoo training (E.3 primary path) — five prior variants, each trained via self-play policy gradient on RTX 2060 until either (a) tournament-ready or (b) compute budget ($\le 24$ h / variant) exhausted. Infrastructure reuses `NeuralCAAgent` ([engine/neural_ca.py](../engine/neural_ca.py)).
- Week 38: zoo round-robin via `run_round_robin` + win-rate matrix figure. Champion replaces `ca_combo_v2` as downstream "strongest agent."
- Week 39: ladder refresh. Stretch `SubstitutionAgent` / `CoxeterAgent` only if tournament finished early.

### Q4 (months 10–12) — Programme D + synthesis
- Weeks 40–44: minimum-observer-size scan across 7 corpus sizes
- Weeks 45–48: diffraction cross-check; both `λ` estimates computed
- Weeks 49–51: **D-gate verification**
- Week 52: synthesis write-up — the Pisot-via-epiplexity paper, or the negative-result paper, whichever the data supports

---

## 10. Marimo notebook (`notebooks/epiplexity_lab.py`)

Created as part of this overhaul. Sections mirror programmes A–E. It imports the `engine/epiplexity.py` module when implemented; initial cells work on placeholder synthetic data so the notebook runs immediately.

**Purpose.** A living scratchpad. Every plot that might end up in a write-up should have a cell here that reproduces it from a corpus manifest hash. No plot should exist only inside a one-off script.

---

## 11. Relationship to ROADMAP v1

| v1 Phase | v2 Programme | Notes |
|----------|--------------|-------|
| 1 (Foundation) | F (Infrastructure) | Mostly complete; v2 adds epiplexity-layer instrumentation |
| 2 (Fork Geometry) | C.1 probes, E.3 agents | Fork-awareness is now a structural probe, not a phase goal |
| 3 (Substitution Structure) | D.2, E.3 | Substitution matrix falls out of the epiplexity scan as a by-product, not a direct enumeration |
| 4 (Spectral Analysis) | D.3 | Diffraction is now the *cross-check* that validates the epiplexity estimate of λ |
| 5 (Algebraic Structure) | E.3 (CoxeterAgent) | A₂/Coxeter work is motivated by needing parsimonious high-structure agents for the Pareto frontier |
| 6 (Agent Hierarchy) | E (ongoing) | Reframed as MDL Pareto frontier, not linear ladder |

**v1 is not deprecated — it is the grounded-in-code substrate that v2 is built on top of.** Anyone who wants to skip the epiplexity framing can still read v1 as the mathematical programme.

---

## 12. Key references (to add to `papers/`)

- Finzi et al. 2026 — *Epiplexity* (added)
- Baake & Grimm 2013 — *Aperiodic Order, Vol. 1* (Pisot substitution theory; look up PDF)
- Kenyon 1996 — *The construction of self-similar tilings*
- Grünwald 2007 — *The Minimum Description Length Principle* (MDL foundations that the paper builds on)
- Koppel 1988 — *Structure* (sophistication — the limiting case of epiplexity)
- Hutter 2005 — *Universal AI* (Solomonoff induction, context for `H_T` at unbounded `T`)
- Olsson et al. 2022 — *In-context Learning and Induction Heads* (why "circuits learned from structure transfer" — cited in Finzi et al.)

---

## 13. What "done" looks like — paper triptych

Added 2026-04-17 to make the completion criterion concrete. The final write-up is built around three load-bearing figures; everything else is supporting content.

### Figure 1 — MDL Pareto frontier in $(|P|, H_T)$ space
- Programme E.2. Scatter all agents in $(|P|, H_T)$ space with the lower-left convex hull drawn.
- Hand-crafted agents (random, greedy, fork_aware, combo, ca_combo_v2) populate the left side (small $|P|$).
- The NCA zoo champion (E.3) occupies the far right.
- Observer-net is a reference point off the frontier (huge $|P|$, lowest $H_T$).
- **Reader take-away:** structure and parsimony trade off. The frontier has distinct regimes.

### Figure 2 — Diffraction spectrum of long self-play
- Programmes P4/P5 extended: rerun on the NCA zoo champion at $N \ge 500$ stones per game, $\ge 30$ games. Compute diffraction via [engine/diffraction.py](../engine/diffraction.py).
- Pure-point component fraction (Bragg99) vs control (random placement).
- **Reader take-away:** perfect-or-near-perfect HexGo play produces a Meyer set with aperiodic order.

### Figure 3 — $S_T(N)$ vs $\log N$ with $\lambda$ fit
- Programme D-gate. On the NCA zoo champion's self-play corpus, for corpus sizes $N \in \{10^2, 10^{2.5}, \dots, 10^5\}$, compute the minimum observer-net size achieving loss within $\epsilon$ of irreducible $H_T$. Plot $S_T(N)$ vs $\log N$.
- Fit $S_T \sim \alpha \log N + \beta$; report $\lambda_\text{epiplexity} = \exp(1/\alpha)$.
- Overlay: independent $\lambda_\text{diffraction}$ from Figure 2's peak spacings.
- **Reader take-away:** two independent routes to the same Pisot inflation constant. This is the headline methodological contribution.

### Supporting content (sections, not figures)
- P1 strategy-stealing validation (combo_defect table) — one paragraph.
- P2 MirrorAgent result — one paragraph, cites Hamkins–Leonessi.
- FMA curve + decisiveness monotonicity — shows weak agents don't reach terminal positions, which is why we need strong agents for Pisot measurement. One paragraph.
- Hamkins echo (decisive not drawish) — one paragraph positioning us in the $\Sigma^0_1$ regime vs infinite Hex at $\Sigma^0_7$.

### Completion criterion
The paper ships when:
1. NCA zoo tournament has a clear champion (not necessarily a statistical dominant, but a unique $\operatorname{argmax}$ on the round-robin).
2. Figure 2 is reproducible from a single [experiments/run_*.py](../experiments/) and the champion's checkpoint.
3. Figure 3 shows either (a) $\lambda_\text{epiplexity}, \lambda_\text{diffraction}$ both inside a Pisot-family interval (golden/plastic/tribonacci), or (b) a clearly-reported null result with the Pisot conjecture explicitly falsified.
4. All P1–P5 propositions in the synthesis note have a paragraph of justification in the paper's supporting text.

Anything else is optional polish.

---

*End of roadmap v2. Last touched with the epiplexity framing in mind.*
