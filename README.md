# HexGo Theory

Number-theoretic and combinatorial investigation of optimal play in HexGo — an infinite hexagonal Connect6 variant played on the Eisenstein integer ring **Z[ω]**.

The goal is to characterise the structure of perfect play: its symmetries, its tiling properties, and whether it exhibits quasi-crystalline (Penrose-like) order.

---

## The Game

HexGo is played on the infinite hex grid, identified with **Z[ω]** where ω = e^(2πi/3). Two players alternate placing stones; the turn rule is **1-2-2** (P1 places 1 stone on turn 1, both players place 2 per turn thereafter). The win condition is **6 consecutive stones along any of the three Z[ω] unit axes**:

```
u₁ = (1, 0)    q-axis
u₂ = (0, 1)    r-axis  
u₃ = (1, -1)   diagonal axis
```

A win is exactly a length-6 arithmetic progression in Z[ω] with unit step — a purely number-theoretic object.

The strategy-stealing argument proves the game cannot be a second-player win. It is almost certainly a first-player win on an infinite board.

---

## Central Conjecture

> **Perfect play in HexGo produces a quasi-crystalline pattern: aperiodic, D6-symmetric, with a substitution structure whose inflation constant is a Pisot number.**

The reasoning:

1. **No translational symmetry**: P1's opening stone at the origin breaks translation invariance. Any forced response propagates this asymmetry outward. A periodic pattern would let the opponent exploit a period vector — contradicting optimality.

2. **D6 symmetry is preserved**: The full dihedral group D6 (rotations by 60°, reflections) is a symmetry of Z[ω] and of the game rules. If the optimal strategy is unique, the occupied set after T moves has D6 symmetry radiating from the origin.

3. **Self-similarity from constraint propagation**: Each local threat at radius r generates forced responses at radius r+5 (the win-length minus 1), which generate further constraints at r+10, etc. If the number of local pattern types (up to D6) is finite, this is a substitution tiling system.

4. **Pisot property**: By the Pisot substitution theorem (Thurston-Kenyon), if the substitution matrix has a Perron-Frobenius eigenvalue that is a Pisot number, the tiling is aperiodic with pure-point diffraction spectrum — a mathematical quasi-crystal.

---

## Research Plan

### Phase 1 — Hand-Crafted Analysis

Use the `EisensteinGreedyAgent` from the main HexGo engine as a substrate:

- **Self-play of Eisenstein vs. Eisenstein**: collect game trajectories and inspect the spatial structure of occupied cells. The greedy agent is fast (no MCTS), so we can generate thousands of games quickly.

- **Van der Waerden analysis**: count, for each game state, how many potential 6-APs in Z[ω] are still "live" (not blocked). Track how this number evolves under optimal play vs. greedy play.

- **Forced-move enumeration**: enumerate all board states within radius R where only one move (up to D6) avoids immediate loss. Build the **forcing graph** — if it is finite (finitely many types up to symmetry), a substitution tiling exists.

- **Erdős-Selfridge potential landscape**: extend the existing potential function `Σ (1/2)^|line|` to a spatial map. Visualise the gradient field — this is the discrete analogue of a harmonic measure on Z[ω].

### Phase 2 — Algebraic Structure

- **Sublattice decomposition**: Z[ω] has sublattices of every index. The win condition (6-AP with unit step) partitions Z[ω] into cosets of ⟨6u₁⟩, ⟨6u₂⟩, ⟨6u₃⟩. Analyse which cosets are "strategically equivalent" under optimal play.

- **Hyperplane arrangement**: each potential 6-line defines a constraint hyperplane in the space of board states. The game navigates the **A₂ hyperplane arrangement** (root system of sl₃), which tiles the plane with 60°-symmetric chambers. Optimal play paths in this arrangement = Coxeter group words.

- **Transfer matrix**: if the forcing graph is finite, construct the substitution matrix and compute its spectrum. Look for Pisot eigenvalues. Natural candidates in the Eisenstein setting: the tribonacci constant (~1.3247, root of x³-x-1=0) or algebraic integers related to the norm form a²-ab+b².

### Phase 3 — Empirical Validation

Use self-play data from the trained HexNet (AlphaZero-style, ~1.9M params) to test the conjectures:

- **Pair correlation function** g(r): compute the two-point correlation of occupied cells. Quasi-periodic oscillations (peaks at irrational multiples of lattice spacing) → quasi-crystalline evidence.

- **Diffraction spectrum**: compute the Fourier transform of the point measure on occupied cells. Pure point spectrum → quasi-crystal. Absolutely continuous → random noise. Mixed → intermediate regime.

- **Pattern frequency analysis**: extract all local patterns of radius 2 and 3 from self-play games. If the number of distinct types (up to D6) is bounded and their frequencies converge, this is strong evidence for a substitution tiling.

---

## Tools and Frameworks

| Framework | Application |
|-----------|-------------|
| `EisensteinGreedyAgent` | Fast hand-crafted player; generates tractable game trees for combinatorial analysis |
| Erdős-Selfridge potential | Threat quantification; spatial potential landscape |
| Combinatorial Game Theory (Berlekamp-Conway-Guy) | Nim-value decomposition for late-game disconnected threat regions |
| Discrete harmonic analysis on Z[ω] | Green's function, harmonic measure, gradient flows |
| Substitution tiling theory (Thurston-Kenyon) | Pisot substitution → aperiodicity proof strategy |
| A₂ hyperplane arrangement / Coxeter theory | Algebraic structure of the forcing graph |
| Diffraction theory (Baake-Grimm) | Spectral characterisation of quasi-crystalline order |

---

## Key Questions

1. **Is the forcing graph finite?** (Finitely many local pattern types up to D6 under optimal play.)
2. **What is the inflation constant?** (Perron-Frobenius eigenvalue of the substitution matrix — is it Pisot?)
3. **What is the critical density?** (Fraction of cells occupied within radius R under optimal play as R→∞ — is it irrational?)
4. **Does the diffraction spectrum have pure point component?** (Empirical test via self-play FFT.)
5. **Is the game a first-player win?** (Strategy-stealing gives lower bound; empirical self-play gives upper bound direction.)

---

## Relation to the Main HexGo Engine

The theoretical work here is designed to feed back into the engine:

- **Better ZOI heuristics**: if the forcing graph is finite and the substitution structure is known, the Zone of Interest can be shaped by the substitution rule rather than a fixed radius.
- **Architecture priors**: if optimal play has D6 symmetry and a Pisot self-similar structure, the network architecture should reflect this (equivariant convolutions, multiscale features at Pisot-ratio scales).
- **Opening book structure**: the substitution tiling approach predicts a finite set of "canonical opening types" — the tiles of the substitution system. These become the opening book.
- **Evaluation function**: the Erdős-Selfridge potential, refined by harmonic analysis on Z[ω], gives a mathematically grounded evaluation function independent of learned weights.

---

## Repository Structure (Planned)

```
hexgo-theory/
├── README.md                  — this file
├── analysis/
│   ├── eisenstein_selfplay.py — run Eisenstein vs. Eisenstein, collect trajectories
│   ├── potential_map.py       — Erdős-Selfridge spatial potential landscape
│   ├── forcing_graph.py       — enumerate forced moves, build substitution candidate
│   └── diffraction.py        — FFT-based spectral analysis of occupied cell sets
├── algebra/
│   ├── sublattice.py          — Z[ω] sublattice decomposition and coset analysis
│   ├── hyperplane.py          — A₂ arrangement, Coxeter word encoding of game paths
│   └── substitution.py        — substitution matrix construction and Pisot test
├── notebooks/
│   └── 01_eisenstein_games.ipynb  — exploratory analysis of greedy self-play
└── docs/
    └── conjectures.md         — formal statement of conjectures with proof sketches
```

---

## References

- Baake, M., & Grimm, U. (2013). *Aperiodic Order, Vol. 1*. Cambridge University Press.
- Berlekamp, E., Conway, J., & Guy, R. (1982). *Winning Ways for your Mathematical Plays*. Academic Press.
- Berger, R. (1966). The undecidability of the domino problem. *Memoirs of the AMS*, 66.
- Erdős, P., & Selfridge, J. (1973). On a combinatorial game. *Journal of Combinatorial Theory*, 14, 298-301.
- Kenyon, R. (1996). The construction of self-similar tilings. *Geometric and Functional Analysis*, 6, 471-488.
- Robinson, R. (1971). Undecidability and nonperiodicity for tilings of the plane. *Inventiones Mathematicae*, 12, 177-209.
- Thurston, W. (1989). Groups, tilings, and finite state automata. *AMS Lectures*.
- Wu, I.-C., & Huang, D.-Y. (2006). A new family of k-in-a-row games. *ICGA Journal*, 29(1), 26-34.
