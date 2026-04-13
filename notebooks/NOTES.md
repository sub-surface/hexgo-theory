# Scratch notes — epiplexity × HexGo

Notes to my future self, written while digesting Finzi et al. 2026 and ROADMAPv2.
These are working thoughts, not conclusions. Expect them to age.

## The one-sentence pitch

The ELO ladder is the $|P|$ axis. The Pisot tiling is what $S_T$ saturates to.
They are one object seen from two sides. Roadmap v2 exists to make that
concrete.

## Why this project is unusually well-suited to epiplexity

Most domains where epiplexity matters (LLMs on web text, vision on natural
images) have *unknown* generating processes — you can measure $S_T$ but you
cannot compare it to ground truth. HexGo is different:

- The generating process is **~200 bytes of game rules + one deterministic agent**.
- So the Kolmogorov complexity of any self-play corpus is bounded above *by
  inspection*.
- Yet we will (I predict) see $S_T$ grow unboundedly with corpus size — at
  least until saturation at the substitution-system level.
- That gap, measured in bits, is **Paradox 3 at a human-tractable scale**.

Nobody has done this experiment cleanly. The paper waves at Life and AlphaZero
but doesn't quantify either. HexGo lets us quantify.

## Concerns and technical risks

### 1. Gzipped source is a noisy $|P|$

Real $|P|$ in the paper's definition is the prefix-free Turing machine
description length. Gzipped Python source is an upper bound, but it conflates
language verbosity with algorithmic complexity. Two agents that do identical
things might differ by 2× in gzip length if one uses comprehensions and the
other uses loops.

Mitigations:
- Normalise via `ast.unparse` before gzipping (strips comments, standardises
  whitespace).
- Report *relative* $|P|$ differences within a stylistic family.
- Eventually: reimplement the agent ladder in a minimal Lisp or lambda-calculus
  for a tighter bound — Q3 or Q4 side-project.

### 2. "Irreducible loss" is not actually irreducible

$H_T$ in the paper is the cross-entropy of the *optimal* bounded program.
Empirically we approximate it by the loss of a very large overparameterised
observer. But if the generating process has structure our big observer cannot
capture (unlikely but possible), we overestimate $H_T$ and underestimate $S_T$.

Mitigation: do the "minimum observer size" scan (§D.1) with multiple
architectures — transformer, MLP, state-space model — and verify the
irreducible-loss ceilings converge.

### 3. Self-play bootstrap loop may diverge

If the small policy we train on `det_combo` corpora is weaker than
`ComboAgent`, iterating will *lose* structure, not gain it. AlphaZero avoided
this by using MCTS at self-play time. We may need a mini-MCTS wrapping the
policy. Budget for this in Q1.

### 4. The Pisot conjecture might just be wrong

If `HexGo` admits forced winning strategies that are *not* substitution-like
(e.g., the game is secretly a trivial pairing strategy), the quasi-crystal
picture collapses. The `S_T` vs `N` scan (§D.2) is what will tell us. A
linear scaling would be a clean negative result. Don't get attached to the
positive hypothesis.

## Small experimental ideas I don't want to lose

- **Time-resolved $S_T$.** Split each game into thirds (opening / middlegame /
  endgame), measure $S_T$ per third. Prediction: peaks mid-game (fork cascade
  era), collapses in endgame (forced sequences). This is the game-phase
  analogue of the paper's data-selection arguments.
- **D6 quotient compression.** Canonicalise every game to its D6 orbit
  representative before training. If this reduces $S_T$ by exactly $\log_2 12$
  bits per game, we have numerically confirmed D6 symmetry. Any deviation is
  evidence of broken symmetry in practice (e.g., first-player bias).
- **Per-move epiplexity.** Treat each move as a token; compute the
  observer's per-token surprise. Sum surprise over a game = game-level $H_T$.
  Moves where observer surprise is high are candidates for "interesting"
  moves — a data-selection heuristic for opening books.
- **Transfer across board geometries.** Train observer on HexGo (hex),
  fine-tune on Connect6 (square). How much epiplexity transfers? This is a
  direct test of the paper's OOD-generalisation claim in a combinatorial
  setting where the *nature* of structural shift is known.

## Things the paper doesn't quite address

1. What happens when the observer is *the same architecture* as the generator?
   Their setup assumes generator and observer are different. Self-play
   bootstrap blurs this. I think §A.1.6 (`det_combo_selfbootstrap`) is the
   right experiment to probe it but I'm not 100% sure how to interpret the
   result theoretically.
2. Finite data regime. Their Def. 8 is in the limit $N \to \infty$. At
   $N = 10^5$ games we have ~3M moves — not nothing, not infinity. The
   finite-sample corrections to $S_T$ estimates may be non-trivial. Check
   Grünwald 2007 ch. 7.
3. They don't discuss *causal* orderings separately from *observed* orderings.
   For HexGo the causal arrow is clear (move $t$ causes constraints at $t+1$),
   but radial-in and radial-out are both acausal. Worth thinking about.

## Deliverables checklist (Q1)

- [x] Roadmap v2 document
- [x] Marimo notebook scaffold
- [ ] `engine/epiplexity.py` with `agent_program_length`, `cross_entropy`,
      `two_part_mdl`, `irreducible_loss`
- [ ] `engine/observer_net.py` canonical transformer
- [ ] `engine/orderings.py` six permutations
- [ ] `engine/probes.py` structural predicates
- [ ] `tests/test_epiplexity.py` including the random-on-random sanity check
- [ ] A-gate run: `det_combo` vs `random` $S_T$ comparison

## Don't lose sight of

The goal of v2 is not to prove the Pisot conjecture. The goal is to *make it
falsifiable*. Epiplexity scaling is the measurement apparatus. The apparatus
matters whether or not the hypothesis survives.
