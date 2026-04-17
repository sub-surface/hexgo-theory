# Epiplexity of HexGo strategies — a measurement plan

*2026-04-18*

## 1. What's being asked

Two threads converging on the same construction:

- **Friend's input.** "Shared representation not only does a great job, it does better on one of the heads." They're running *linear probes + self-supervised masked reconstruction* to test generality.
- **Leon's follow-up.** Use the epiplexity methodology (Finzi et al. 2026) to understand learnability and other details in relation to our strategies.

These are the same experiment from two angles. The friend is describing the **architecture/training recipe**; Leon is describing the **measurement framework**. Both resolve into: "train one shared small observer on different self-play corpora; read out what the observer learned via linear probes on strategic predicates." That recipe *is* the Finzi et al. $S_T$ / MDL-pair construction, and it is what ROADMAP Programme A + C already specify — we just haven't built it yet.

## 2. What this buys us

Three deliverables for the paper that nothing else produces:

1. **An agent-comparable strength metric that doesn't require playing tournaments.** For an agent $A$, $H_T(A; \text{held-out corpus})$ = cross-entropy of $A$'s softmax move distribution against held-out games — a single scalar per agent, directly comparable. Pareto-plot against $|P_A| \approx$ gzipped source bytes. ROADMAP §6 Programme E.
2. **An ordering on the learnability of play styles.** A shared observer net trained on corpus $C_A$ from each agent and evaluated (a) by its irreducible loss $H_T$ and (b) by the AUC above irreducible loss (the $S_T$ heuristic, §4.3 of the paper) tells us *which agent's play contains the most structural information* — i.e. which styles are most "learnable" given bounded compute.
3. **A discriminator for the NCA-zoo priors that doesn't require beating ca_combo_v2.** Even if all five priors end up at similar tournament win-rates after training, $S_T(\text{corpus}_{\text{nca-prior}_k})$ will split them — the prior that produces the *most structurally rich* play wins a different race, and one that matters for the Pisot conjecture.

The friend's "does better on one of the heads" comment — if I'm reading it correctly — is exactly this: a shared representation whose different heads (move-policy, masked-reconstruction, probe-for-threat) all converge to a common structural encoding. We should adopt this multi-head setup.

## 3. The concrete experiment

### 3.1 Architecture — one observer, three heads

Call it `StrategyObserver`. Inputs: (player 1 stones, player 2 stones, to-move-flag) windowed Cartesian axial board, (3, H, W). Shared trunk = stack of HexConv2d blocks ([engine/neural_ca.py:43](../../engine/neural_ca.py)) producing a (C, H, W) feature map. Three heads share the trunk:

- **Head A — next-move policy.** Linear layer over the feature map → (1, H, W) score map. Loss: categorical cross-entropy against the human / agent move distribution. This is the standard "imitate corpus" head, used for $H_T$ computation.
- **Head B — masked reconstruction.** Randomly mask 15% of stones in the input; predict their colour from the surrounding context. Loss: three-way categorical cross-entropy (empty, P1, P2) at masked positions. This is the friend's self-supervised generality head.
- **Head C — strategic probes.** *Frozen* linear classifiers on the trunk, one per probe, each predicting a single binary predicate — `is_triple_fork(c, g)`, `is_forced_response(c, g)`, `feat_potential(c, g) > threshold`, etc. Probe *accuracy* after training the trunk is the read-out of whether the corpus taught the trunk to represent that predicate internally.

Crucially: probes are trained only on linear layers *after* the trunk is frozen from A+B training. Probe accuracy is an estimator of whether the trunk learnt that predicate, not whether the trunk can be forced to learn it.

### 3.2 Corpora

Reuse the ladder we already have:

| Corpus | Agent | Source |
|---|---|---|
| `corpus_random` | `RandomAgent` | control |
| `corpus_greedy` | `EisensteinGreedyAgent` | [engine/ca_policy.py](../../engine/ca_policy.py) |
| `corpus_combo_v2` | `make_combo_v2_ca` | current strongest hand-crafted |
| `corpus_mirror` | `MirrorAgent` | [engine/agents.py](../../engine/agents.py) |
| `corpus_nca_<prior>` | trained NCA per prior | [experiments/run_nca_train.py](../../experiments/run_nca_train.py) |

Each at $N \in \{10^2, 10^3, 10^4\}$ games. Horizon 240.

### 3.3 Measurements per corpus

1. **Agent-only metrics** (no training): $|P_A|$ = gzipped Python source length; $H_T(A; \text{corpus}_A)$ = self-cross-entropy at temperature 0.5 (tiny; the agent predicts itself perfectly).
2. **Cross-play metric:** $H_T(A; \text{corpus}_B)$ — how well does $A$ predict $B$'s moves? Mutual distance matrix between agents.
3. **Shared-observer training:** train `StrategyObserver` on each corpus. Record irreducible loss (head A floor), AUC above floor (= $S_T$ heuristic), masked-reconstruction accuracy (head B — should improve monotonically with corpus structure), probe accuracies (head C — should be higher for structured corpora).
4. **Epiplexity scaling:** for each corpus, record irreducible loss (head A) vs minimum observer size to reach it. Plot $S_T$ vs $\log N$. This is ROADMAP D-gate groundwork.

### 3.4 Predictions (falsifiable)

- **P6 (agent learnability ordering).** Cross-entropy ranking of the shared observer trained on each corpus matches the tournament ranking of its source agent, up to one inversion. *Falsified by:* two or more inversions. This is the friend's "shared rep does a great job" claim made concrete in our setting.
- **P7 (probe emergence is structure-dependent).** Linear-probe accuracy on `is_forced_response` for observers trained on `corpus_combo_v2` exceeds 80%, while the same probe on `corpus_random`-trained observers stays below 55%. *Falsified by:* <75% / >60% gap, which would mean the observer can't tell the two apart at the level of tactical structure.
- **P8 (MLM-quality mirrors tactical quality).** Masked-reconstruction accuracy on held-out positions from `corpus_combo_v2` > accuracy from `corpus_random` by at least 15 percentage points. *Falsified by:* MLM accuracy insensitive to corpus quality — which would suggest the masked-prediction task is trivially solvable from local context and tells us nothing about strategy. This is the falsifiable version of "MLM generality" for a game setting.
- **P9 (NCA-prior discriminator).** Among the five NCA priors, the $D_6$-tied prior produces self-play corpora with the highest probe-C accuracies *regardless* of its tournament win-rate. *Falsified by:* combo / line_detector / erdos_selfridge having ≥10 percentage points higher probe accuracy. This would overturn the §7 symmetry-is-load-bearing thesis from the perspective of learnability, not tournament strength.

P6-P9 sit alongside P1-P5 as the second five-tuple in the falsifiability table.

## 4. What to build

New module: `engine/observer.py`.

```python
class StrategyObserver(nn.Module):
    """Shared-trunk multi-head small CNN on axial board windows.

    Heads: (A) next-move policy, (B) masked-reconstruction,
    (C) strategic-probe linear classifiers over a frozen trunk.
    """

def train_observer(corpus, *, heads=("policy", "mlm"), ...) -> dict:
    """Phase 1: train trunk + policy/MLM heads jointly.
       Phase 2: freeze trunk, train probe heads as linear-only."""

def linear_probe_accuracy(trunk, corpus, predicate) -> float:
    """Labelled-example accuracy for a single binary predicate."""

def epiplexity_estimate(corpus, *, trunk_sizes) -> dict:
    """Minimum trunk size to reach irreducible loss ± eps; returns S_T."""
```

New experiment: `experiments/run_strategy_observer.py`. Round through every corpus, train the shared observer, record all four metric families from §3.3, produce the $(|P|, H_T)$ scatter + the probe-accuracy heatmap + the $S_T$ vs $\log N$ curve.

## 5. Why this unifies the threads

The friend's recipe (shared trunk, multi-head, linear probes, MLM) and the Finzi-et-al.\ epiplexity methodology (minimum-observer-size, two-part MDL, time-bounded entropy) are the same mechanism. The friend articulates the engineering; Finzi et al.\ articulate the measurement theory. Adopting both gives us:

- **One number per agent** ($H_T$) that lets us rank without tournaments.
- **One number per agent** ($S_T$) that measures learnability / structural richness.
- **A distinguishing feature for the NCA zoo** that doesn't depend on which variant wins the tournament.
- **A direct route to the ROADMAP D-gate** — the $S_T$-vs-$\log N$ curve with a Pisot-$\lambda$ fit *is* the paper's headline.

Plan of attack: let the current NCA-training run complete, then build `engine/observer.py` and the observer experiment. This becomes the natural next milestone after the tournament.
