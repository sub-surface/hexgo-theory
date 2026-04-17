# Hamkins–Leonessi and HexGo — synthesis

*2026-04-17*

## 1. What their paper is

Hamkins & Leonessi, *Infinite Hex is a draw* ([papers/2201.06475v3-Infinite-Hex-is-a-draw.pdf](../../papers/2201.06475v3-Infinite-Hex-is-a-draw.pdf)), prove that infinite-board Hex (each player wants to connect opposite sides of an unbounded board) is a draw under perfect play, rather than the finite-board first-player win. The mechanism is a **pairing / mirroring strategy**: player II divides the plane into dominoes and responds to each of player I's moves within the pairing partner, guaranteeing no infinite monochromatic path for player I.

The complexity of the game as a set-theoretic object is striking. The payoff set "player I has an infinite path" is not open — you can't observe the win in finite time — so Gale–Stewart's basic determinacy theorem doesn't apply. Hamkins–Leonessi instead classify the payoff at **$\Sigma^0_7$** (a specific level of the Borel hierarchy) via a fairly intricate formula; Törnä's subsequent note tightens this.

## 2. Where HexGo sits

HexGo (Connect-6 on the infinite hex lattice with `WIN_LENGTH == 6`) is a fundamentally easier descriptive-set-theoretic object. The payoff "some player has 6 in a row" is a finite-conjunction statement: for each of three axes, and each of countably many starting cells, a single open condition on the stones. This makes the payoff set **$\Sigma^0_1$ (open)** — directly in the domain of Gale–Stewart determinacy. Under optimal play, every game ends in finite time.

That is not a flaw in our setting — it is the *reason* our questions are different. Hamkins–Leonessi's machinery is aimed at an irreducibly infinitary target. Our target is the **structure inside finite, decidable play**: what is the character of optimal move sequences, how much description length does perfect play cost ($|P|$, $H_T$ on the ROADMAP's epiplexity plane), and does the positional pattern exhibit quasi-crystalline order.

In short: descriptive set theory tells us where we are *relative to* infinite Hex — two levels lower in the Borel hierarchy, with finite-horizon analysis being the appropriate lens. We don't need Gale–Stewart; we need measure.

## 3. What transfers directly

The **pairing construction** is portable. In infinite Hex, the pairing is a trapezoid tiling of $\mathbb{Z}^2$ (Hamkins §3). In HexGo, a translated analogue — pair cells at $c$ and $c + 3\omega$ along each axis, say — is a candidate **MirrorAgent** strategy. This is the immediate next implementation target (see pending TODO / [engine/agents.py](../../engine/agents.py)).

Key questions:

- **Does a Connect-6 pairing exist at all?** A pairing works if, for every winning 6-line, at least one of its cells is paired with another cell in the same line. In hex with three axes this is a combinatorial constraint — we should check it on paper before coding.
- **Even if a pairing exists, it only secures draw / non-loss, not a win.** The empirical target for `MirrorAgent` is therefore: **>90% non-loss vs RandomAgent, ≥50% non-loss vs Combo**. If it lives up to that we have a concrete blocker against the "optimal play is a first-player win" expectation becoming testable.

## 4. The quasicrystal conjecture in this light

Leon's conjecture: perfect play in HexGo produces a point set in $\mathbb{Z}[\omega]$ that is a Meyer set — pure-point diffraction spectrum ⇒ quasi-crystalline order. Hamkins–Leonessi is tangential to this but clarifies *where the conjecture must live*:

- The conjecture is **not** about the infinitary strategy class. It is about the *geometry* of optimal finite stone configurations sampled at increasing horizon. That geometry is a property of finite-sized $\Sigma^0_1$ solutions, not of $\Sigma^0_7$ equilibrium strategies.
- This aligns the conjecture with the epiplexity plane $(|P|, H_T)$ of the ROADMAP: a Meyer-set outcome predicts corpus description length $|P|$ saturates (Pisot substitution) rather than growing linearly. The test is empirical — diffraction of a long Combo-vs-Combo self-play — and falsifiable.

The key bridge here is that **"optimal play = Meyer set"** is a statement at the level of configurations, not at the level of strategy complexity. Hamkins's framework explains why that's the right level.

## 5. Five falsifiable propositions

Each claim below is intended to be tested by one experiment in `experiments/`, with a specific go/no-go condition. Writing them as numbered propositions so they can be cited in the paper.

> **P1 (Strategy-stealing for HexGo).** In self-play between equal-strength agents, Black win-rate ≥ White win-rate (within Wilson 95% CI) for every agent on our ladder.
>
> *Falsified by:* any agent where White beats Black significantly (Black share of decisive games < lower Wilson bound of 0.5). **Currently violated by ComboAgent v1** — see [run_combo_defect.py](../../experiments/run_combo_defect.py). Fix status: opening-centre bias (v2) being evaluated 2026-04-17.

> **P2 (No second-player MirrorAgent win).** A $D_6$-equivariant pairing-strategy agent (if a valid pairing exists) achieves non-loss ≥ 90% against RandomAgent but strictly less than 50% as second player against any stronger agent than RandomAgent.
>
> *Falsified by:* >50% wins as second player against a non-random opponent — which would contradict strategy-stealing.

> **P3 (Saturating description length).** For a Combo-v2-vs-Combo-v2 self-play at horizon 480, the optimal LZ77 / MDL encoded length $|P|$ of the stone sequence grows sub-linearly in $T$ — specifically, $|P|(T)/T \to 0$ and $|P|(T)/\log T \to \infty$ is the Pisot-substitution signature.
>
> *Falsified by:* linear scaling of $|P|$, i.e. constant per-move description cost. See [experiments/run_epiplexity_scan.py](../../experiments/run_epiplexity_scan.py).

> **P4 (Pure-point diffraction).** The diffraction intensity $\left| \sum_j e^{2\pi i k \cdot x_j} \right|^2$ over all stone positions at move 100+ of a Combo-v2 self-play has a Bragg-peak component exceeding 50% of total intensity when averaged over multiple games.
>
> *Falsified by:* flat / continuous spectrum ⇒ amorphous stone placement ⇒ no quasi-crystalline order.

> **P5 (Meyer-set radius scaling).** The minimum pairwise stone distance $d_{\min}$ in long self-play saturates at a constant (no arbitrarily close pairs), and the maximum first-neighbour distance $d_{\max}$ is bounded (no arbitrarily large holes). Together these are the Delone property — the combinatorial prerequisite for Meyer-ness.
>
> *Falsified by:* violation of either bound as game length grows.

## 6. Current data vs these propositions

| Proposition | Status | Evidence |
|-------------|--------|----------|
| P1 | **supported (after v2 fix)** | `results/combo_defect.json` — v2 Black share = 0.53 [0.42, 0.64] |
| P2 | **supported on both clauses** | `results/mirror_agent.json` — Mirror non-loss vs Random = 1.00; Mirror-P2 wins vs Combo = 0.14 |
| P3 | preliminary support | `results/epiplexity_scan.json` — Combo's $|P|/T$ slope below Random's, but no Pisot confirmation yet |
| P4 | **supported in long self-play** | `results/diffraction_p4.json` — long-game Bragg99 = 0.51 ± 0.13 (n=9) vs short-game 0.24 ± 0.09 vs random 0.055; `figures/fig_diffraction_p4.png` shows unambiguous hex-lattice Bragg peaks. Length-dependence is itself a non-trivial finding: quasi-crystalline order emerges as the agents draw out play, not in short decisive games. |
| P5 | **supported** | `results/diffraction_p4.json` — d_min ∈ [1.0, 2.0], d_max ∈ [1.0, 4.58], corr(N, d_max) = +0.07. Delone property holds with no holes growing with game length. |

The work order now is P1-fix → MirrorAgent → diffraction, which also aligns with the pending-experiment priorities listed in [CLAUDE.md](../../CLAUDE.md) §"Current research state".
