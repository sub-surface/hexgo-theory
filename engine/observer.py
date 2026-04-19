"""
Shared-trunk multi-head observer for HexGo strategy corpora.

This module implements the measurement architecture described in
[docs/theory/2026-04-18-epiplexity-of-strategy.md].  It unifies two threads
that land on the same construction:

  * The friend's recipe: shared HexConv trunk + (A) next-move policy head,
    (B) self-supervised masked-reconstruction head, (C) linear probes for
    strategic predicates over the frozen trunk.
  * The Finzi et al. 2026 epiplexity framework: S_T / H_T as
    trunk-size vs. irreducible-loss pair; minimum trunk size to reach loss
    floor approximates the time-bounded MDL description length.

Call surface.
  - encode_position(game_or_board, to_move, pad)  -> (array, origin)
  - mask_stones(arr, rate, rng)                   -> (masked, target, mask)
  - generate_corpus(agent_factory, n_games, ...)  -> list[CorpusExample]
  - StrategyTrunk / PolicyHead / MLMHead / LinearProbe / StrategyObserver
  - train_observer(corpus, ...)                   -> (model, history)
  - train_linear_probe(trunk, corpus, predicate)  -> (train_acc, val_acc)
  - epiplexity_estimate(corpus, hidden_sizes, ..) -> dict (S_T estimator)

The trunk uses HexConv2d from [engine/neural_ca.py] so the inductive
structure matches the NCA-zoo agents on which we will run the observer.

All tensors live on CUDA when available (RTX 2060, 5GB budget) — prefer
float32 and modest hidden widths (<=64) to stay within VRAM.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import HexGame
from engine.neural_ca import HexConv2d


# ── Encoding ──────────────────────────────────────────────────────────────────


def encode_position(
    game_or_board,
    to_move: int | None = None,
    pad: int = 4,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Build a (4, H, W) float32 array around the current stone cloud.

    Channels: [empty, own, opp, to_move_is_P1_flag].  "own" is the stones of
    the player to move; "opp" is the other player's.  The fourth channel is
    a constant plane (1.0 if to_move == 1, else 0.0) so the observer can
    distinguish P1-to-move from P2-to-move without having to infer parity
    from stone counts.

    Returns (array, (q_min, r_min, H, W)).  H == r_max-r_min+1+2*pad, likewise W.
    """
    if isinstance(game_or_board, HexGame):
        board = game_or_board.board
        player = game_or_board.current_player if to_move is None else to_move
    else:
        board = dict(game_or_board)
        if to_move is None:
            raise ValueError("to_move required when passing a raw board")
        player = to_move

    if not board:
        H = W = 2 * pad + 1
        arr = np.zeros((4, H, W), dtype=np.float32)
        arr[0] = 1.0
        arr[3] = 1.0 if player == 1 else 0.0
        return arr, (-pad, -pad, H, W)

    qs = [q for (q, _) in board]
    rs = [r for (_, r) in board]
    q_min, q_max = min(qs) - pad, max(qs) + pad
    r_min, r_max = min(rs) - pad, max(rs) + pad
    W = q_max - q_min + 1
    H = r_max - r_min + 1
    arr = np.zeros((4, H, W), dtype=np.float32)
    arr[0] = 1.0
    opponent = 3 - player
    for (q, r), p in board.items():
        c = q - q_min
        rr = r - r_min
        arr[0, rr, c] = 0.0
        if p == player:
            arr[1, rr, c] = 1.0
        elif p == opponent:
            arr[2, rr, c] = 1.0
    arr[3] = 1.0 if player == 1 else 0.0
    return arr, (q_min, r_min, H, W)


def mask_stones(
    arr: np.ndarray,
    mask_rate: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly mask ~mask_rate of occupied cells for MLM pre-training.

    Returns (masked_arr, target, mask).  target[r, c] in {0, 1, 2}
    (empty/own/opp) is the true label for masked positions; mask[r, c] is 1
    at those positions, 0 elsewhere.  For a small fraction (mask_rate/2) of
    empty cells we also include them in the mask set, so the MLM head is
    not allowed to cheat with "masked => occupied".
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = arr.shape[1], arr.shape[2]
    own = arr[1] > 0.5
    opp = arr[2] > 0.5
    empty = ~(own | opp)

    n_stones = int(own.sum() + opp.sum())
    n_empty = int(empty.sum())
    # Sample masked-stone + masked-empty sets independently.
    stone_keep = rng.random(size=(H, W)) < mask_rate
    empty_keep = rng.random(size=(H, W)) < (mask_rate * 0.3)
    mask = ((own | opp) & stone_keep) | (empty & empty_keep)

    target = np.zeros((H, W), dtype=np.int64)
    target[own] = 1
    target[opp] = 2

    masked = arr.copy()
    # Where masked, replace with "unknown" pattern: equal probability across
    # own/opp/empty on the input side (a uniform prior the trunk must disambiguate).
    masked[0][mask] = 1.0 / 3.0
    masked[1][mask] = 1.0 / 3.0
    masked[2][mask] = 1.0 / 3.0
    return masked, target, mask.astype(np.float32)


# ── Corpora ───────────────────────────────────────────────────────────────────


@dataclass
class CorpusExample:
    """A single (position, next-move) training example."""
    arr: np.ndarray                       # (4, H, W) float32
    origin: tuple[int, int, int, int]     # (q_min, r_min, H, W)
    move: tuple[int, int] | None          # (q, r) — next move in canonical coords
    to_move: int                          # 1 or 2


def generate_corpus(
    agent_factory: Callable,
    n_games: int,
    *,
    opponent_factory: Callable | None = None,
    max_moves: int = 240,
    seed: int = 0,
    pad: int = 4,
) -> list[CorpusExample]:
    """
    Roll out `n_games` of `agent_factory()` against `opponent_factory()`
    (defaults to self-play with the same agent), collecting every
    (position, next-move) pair where the player-to-move is the target agent.

    If opponent_factory is None we emit examples from both sides; this is
    the right thing for self-play corpora.
    """
    if opponent_factory is None:
        opponent_factory = agent_factory
    out: list[CorpusExample] = []
    rng = np.random.default_rng(seed)
    for g in range(n_games):
        game = HexGame()
        agent_a = agent_factory()
        agent_b = opponent_factory()
        moves = 0
        while game.winner is None and moves < max_moves:
            if not game.candidates:
                break
            player = game.current_player
            # Always record the example for the player about to move.
            arr, origin = encode_position(game, to_move=player, pad=pad)
            mover = agent_a if player == 1 else agent_b
            try:
                move = mover.choose_move(game)
            except Exception:
                break
            if move is None or move in game.board:
                break
            ok = game.make(*move)
            if not ok:
                break
            out.append(CorpusExample(arr=arr, origin=origin, move=move, to_move=player))
            moves += 1
        _ = rng.random()  # advance RNG — keeps seed semantics stable if we parallelise later
    return out


# ── Architecture ──────────────────────────────────────────────────────────────


class StrategyTrunk(nn.Module):
    """Shared HexConv stack.  Takes (B, 4, H, W) → (B, hidden, H, W)."""

    def __init__(self, in_ch: int = 4, hidden: int = 32, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_ch
        for _ in range(depth):
            layers.append(HexConv2d(c, hidden))
            layers.append(nn.ReLU(inplace=True))
            c = hidden
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyHead(nn.Module):
    """1x1 HexConv → (B, 1, H, W) score map for next-move prediction."""

    def __init__(self, hidden: int):
        super().__init__()
        self.head = HexConv2d(hidden, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats).squeeze(1)  # (B, H, W)


class MLMHead(nn.Module):
    """1x1 HexConv → (B, 3, H, W) logits over {empty, own, opp}."""

    def __init__(self, hidden: int):
        super().__init__()
        self.head = HexConv2d(hidden, 3)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)  # (B, 3, H, W)


class LinearProbe(nn.Module):
    """Single-layer linear probe over a (possibly frozen) trunk feature map.

    If per_cell is True the probe operates per-cell (output shape (B,H,W));
    else it pools the trunk features globally and predicts a scalar per example.
    """

    def __init__(self, hidden: int, per_cell: bool = False):
        super().__init__()
        self.per_cell = per_cell
        if per_cell:
            self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        else:
            self.head = nn.Linear(hidden, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.per_cell:
            return self.head(feats).squeeze(1)  # (B, H, W)
        pooled = feats.mean(dim=(2, 3))  # (B, hidden)
        return self.head(pooled).squeeze(-1)  # (B,)


class StrategyObserver(nn.Module):
    """Shared trunk + policy head + MLM head."""

    def __init__(self, hidden: int = 32, depth: int = 4):
        super().__init__()
        self.trunk = StrategyTrunk(in_ch=4, hidden=hidden, depth=depth)
        self.policy = PolicyHead(hidden)
        self.mlm = MLMHead(hidden)
        self.hidden = hidden
        self.depth = depth

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.trunk(x)
        return self.policy(f), self.mlm(f)


# ── Batching helpers ──────────────────────────────────────────────────────────


def _pad_to(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Pad a (C, h, w) array to (C, H, W) with zeros, keeping channel 0 = empty
    set to 1 in the padded region so the "empty" plane remains consistent."""
    C, h, w = arr.shape
    if h == H and w == W:
        return arr
    out = np.zeros((C, H, W), dtype=arr.dtype)
    out[0] = 1.0  # empty plane default
    # Copy original (top-left anchored).
    out[:, :h, :w] = arr
    # Preserve the to_move flag across the pad: read from the original plane-3.
    out[3] = arr[3, 0, 0]
    return out


def _batchify(
    examples: list[CorpusExample],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int, int]]]:
    """
    Collate a list of CorpusExample into padded (B,4,H,W) input + (B,H,W)
    target-index tensor encoding the next-move cell (or -1 if no move).
    """
    H = max(e.arr.shape[1] for e in examples)
    W = max(e.arr.shape[2] for e in examples)
    B = len(examples)
    xs = np.zeros((B, 4, H, W), dtype=np.float32)
    ys = np.full((B,), -1, dtype=np.int64)  # flat target cell per example
    origins = []
    for i, e in enumerate(examples):
        xs[i] = _pad_to(e.arr, H, W)
        origins.append(e.origin)
        if e.move is None:
            continue
        q_min, r_min, _, _ = e.origin
        mc = e.move[0] - q_min
        mr = e.move[1] - r_min
        if 0 <= mc < W and 0 <= mr < H:
            ys[i] = mr * W + mc
    return (
        torch.from_numpy(xs).to(device),
        torch.from_numpy(ys).to(device),
        origins,
    )


# ── Training ──────────────────────────────────────────────────────────────────


def train_observer(
    corpus: list[CorpusExample],
    *,
    hidden: int = 32,
    depth: int = 4,
    epochs: int = 6,
    batch_size: int = 32,
    lr: float = 3e-4,
    policy_weight: float = 1.0,
    mlm_weight: float = 0.5,
    mask_rate: float = 0.15,
    device: str | None = None,
    seed: int = 0,
    val_frac: float = 0.1,
) -> tuple[StrategyObserver, dict]:
    """Train a shared observer jointly on policy + MLM heads.

    Returns the trained model and a history dict with per-epoch train/val
    losses for both heads, plus a "final_policy_ce" scalar used downstream
    as H_T(A).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n = len(corpus)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    val_ids = idx[:n_val]
    train_ids = idx[n_val:]

    model = StrategyObserver(hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict = {
        "policy_ce_train": [], "policy_ce_val": [],
        "mlm_ce_train": [], "mlm_ce_val": [],
        "n_train": int(len(train_ids)), "n_val": int(len(val_ids)),
        "hidden": hidden, "depth": depth,
    }

    for ep in range(epochs):
        rng.shuffle(train_ids)
        model.train(True)
        tot_p, tot_m, seen = 0.0, 0.0, 0
        for start in range(0, len(train_ids), batch_size):
            batch_ids = train_ids[start:start + batch_size]
            batch = [corpus[i] for i in batch_ids]
            xs, ys, _ = _batchify(batch, device)

            # Build masked inputs + MLM targets.
            masked_np = np.zeros_like(xs.detach().cpu().numpy())
            target_np = np.full((xs.shape[0], xs.shape[2], xs.shape[3]), -100, dtype=np.int64)
            for i in range(xs.shape[0]):
                orig = xs[i].detach().cpu().numpy()
                m, t, mk = mask_stones(orig, mask_rate=mask_rate, rng=rng)
                masked_np[i] = m
                # Set label only at masked positions; else -100 (ignore index).
                lab = t.copy()
                lab[mk < 0.5] = -100
                target_np[i] = lab
            xmasked = torch.from_numpy(masked_np).to(device)
            mlm_target = torch.from_numpy(target_np).to(device)

            policy_logits, mlm_logits = model(xs)  # policy on clean inputs
            _, mlm_logits_masked = model(xmasked)  # MLM on masked inputs
            B, H, W = policy_logits.shape
            pl_flat = policy_logits.view(B, H * W)
            valid = ys >= 0
            if valid.any():
                policy_loss = F.cross_entropy(pl_flat[valid], ys[valid])
            else:
                policy_loss = torch.tensor(0.0, device=device)
            mlm_loss = F.cross_entropy(mlm_logits_masked, mlm_target, ignore_index=-100)

            loss = policy_weight * policy_loss + mlm_weight * mlm_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_p += float(policy_loss) * len(batch_ids)
            tot_m += float(mlm_loss) * len(batch_ids)
            seen += len(batch_ids)
        history["policy_ce_train"].append(tot_p / max(1, seen))
        history["mlm_ce_train"].append(tot_m / max(1, seen))

        # Validation.
        model.train(False)
        vp, vm, vs = 0.0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_ids), batch_size):
                batch_ids = val_ids[start:start + batch_size]
                batch = [corpus[i] for i in batch_ids]
                xs, ys, _ = _batchify(batch, device)
                # MLM on masked copies for consistent metric.
                masked_np = np.zeros_like(xs.detach().cpu().numpy())
                target_np = np.full((xs.shape[0], xs.shape[2], xs.shape[3]), -100, dtype=np.int64)
                for i in range(xs.shape[0]):
                    orig = xs[i].detach().cpu().numpy()
                    m, t, mk = mask_stones(orig, mask_rate=mask_rate, rng=rng)
                    masked_np[i] = m
                    lab = t.copy()
                    lab[mk < 0.5] = -100
                    target_np[i] = lab
                xmasked = torch.from_numpy(masked_np).to(device)
                mlm_target = torch.from_numpy(target_np).to(device)

                policy_logits, _ = model(xs)
                _, mlm_logits_masked = model(xmasked)
                B, H, W = policy_logits.shape
                pl_flat = policy_logits.view(B, H * W)
                valid = ys >= 0
                if valid.any():
                    pl = float(F.cross_entropy(pl_flat[valid], ys[valid]))
                    vp += pl * int(valid.sum())
                    vs += int(valid.sum())
                vm += float(F.cross_entropy(mlm_logits_masked, mlm_target,
                                             ignore_index=-100)) * len(batch_ids)
        history["policy_ce_val"].append(vp / max(1, vs))
        history["mlm_ce_val"].append(vm / max(1, len(val_ids)))

    history["final_policy_ce"] = history["policy_ce_val"][-1]
    history["final_mlm_ce"] = history["mlm_ce_val"][-1]
    return model, history


# ── Linear probes ─────────────────────────────────────────────────────────────


def _predicate_labels(
    game_or_board_list: list[tuple[dict, int]],
    predicate: str,
) -> np.ndarray:
    """
    Compute per-example binary labels for one of a small set of predicates.

    Supported:
      "threat_self"      — 1 if the to-move player has >=1 immediate-win cell
      "threat_opp"       — 1 if the opponent has >=1 immediate-win cell
      "fork_self"        — 1 if the to-move player has a fork (>=2 axes extending)
      "high_potential"   — 1 if max(potential_map) > threshold (top-quartile proxy)

    The inputs are (board_dict, to_move) pairs (we reconstruct a HexGame in
    the cheapest way possible for each: analysis.py walks ``game.board`` +
    ``game.candidates``).
    """
    from engine.analysis import threat_cells, fork_cells, potential_map

    out = np.zeros((len(game_or_board_list),), dtype=np.int64)
    for i, (board, to_move) in enumerate(game_or_board_list):
        g = HexGame()
        # Rebuild board + candidate set by replaying makes in any order is not
        # viable (turn parity matters); we just poke the internal state.
        g.board = dict(board)
        g.current_player = to_move
        # Re-seed candidates from neighbours of occupied cells (mirrors game.make).
        cands = set()
        DIRS = ((1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1))
        for (q, r) in g.board:
            for dq, dr in DIRS:
                nb = (q + dq, r + dr)
                if nb not in g.board:
                    cands.add(nb)
        if not cands:
            cands.add((0, 0))
        g.candidates = cands

        if predicate == "threat_self":
            out[i] = 1 if len(threat_cells(g, to_move)) > 0 else 0
        elif predicate == "threat_opp":
            out[i] = 1 if len(threat_cells(g, 3 - to_move)) > 0 else 0
        elif predicate == "fork_self":
            out[i] = 1 if len(fork_cells(g, to_move)) > 0 else 0
        elif predicate == "high_potential":
            pot = potential_map(g)
            out[i] = 1 if (max(pot.values()) if pot else 0.0) > 1.25 else 0
        else:
            raise ValueError(f"unknown predicate {predicate!r}")
    return out


def train_linear_probe(
    trunk: StrategyTrunk,
    corpus: list[CorpusExample],
    *,
    predicate: str,
    batch_size: int = 32,
    epochs: int = 4,
    lr: float = 3e-3,
    device: str | None = None,
    seed: int = 0,
    val_frac: float = 0.2,
) -> tuple[float, float]:
    """
    Train a global-pooled linear probe over a frozen trunk to predict one of
    the binary predicates in ``_predicate_labels``.  Returns (train_acc, val_acc).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Freeze the trunk — same mechanism as NeuralCAAgent.__post_init__.
    for p in trunk.parameters():
        p.requires_grad_(False)
    trunk.train(False)

    # Build labels via the predicate function.
    pairs = []
    for e in corpus:
        # Reconstruct (board, to_move) from the encoded array is painful; we
        # just reconstruct from the cell-level presence in channels 1/2.
        q_min, r_min, H, W = e.origin
        board = {}
        own = e.arr[1] > 0.5
        opp = e.arr[2] > 0.5
        for rr in range(H):
            for c in range(W):
                if own[rr, c]:
                    board[(c + q_min, rr + r_min)] = e.to_move
                elif opp[rr, c]:
                    board[(c + q_min, rr + r_min)] = 3 - e.to_move
        pairs.append((board, e.to_move))
    labels = _predicate_labels(pairs, predicate)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = len(corpus)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    val_ids = idx[:n_val]
    train_ids = idx[n_val:]

    probe = LinearProbe(hidden=trunk.hidden, per_cell=False).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    def _forward(batch_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        batch = [corpus[i] for i in batch_ids]
        xs, _, _ = _batchify(batch, device)
        with torch.no_grad():
            feats = trunk(xs)
        logits = probe(feats)
        y = torch.from_numpy(labels[batch_ids]).float().to(device)
        return logits, y

    for ep in range(epochs):
        rng.shuffle(train_ids)
        probe.train(True)
        for start in range(0, len(train_ids), batch_size):
            batch_ids = train_ids[start:start + batch_size]
            logits, y = _forward(batch_ids)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.train(False)
    with torch.no_grad():
        logits_t, y_t = _forward(train_ids)
        train_acc = float(((logits_t > 0).long() == y_t.long()).float().mean())
        logits_v, y_v = _forward(val_ids)
        val_acc = float(((logits_v > 0).long() == y_v.long()).float().mean())
    return train_acc, val_acc


# ── Epiplexity estimator ──────────────────────────────────────────────────────


def _state_dict_gzip_bytes(model: nn.Module) -> int:
    """Size in bytes of the gzipped state_dict — a proxy for |P| (description length)."""
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    raw = buf.getvalue()
    return len(gzip.compress(raw))


def epiplexity_estimate(
    corpus: list[CorpusExample],
    *,
    hidden_sizes: Iterable[int] = (4, 8, 16, 32, 64),
    depth: int = 4,
    epochs: int = 4,
    tolerance_ratio: float = 1.10,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    """
    Run ``train_observer`` at a ladder of trunk widths; record validation
    policy-CE (the H_T estimator) and the gzipped state-dict size (the |P|
    estimator).  Returns a dict with:

      hidden_losses     {hidden: final_policy_ce_val}
      hidden_gzip_bytes {hidden: state_dict_gzip_bytes}
      reference_loss    the floor — best policy-CE across widths
      threshold_loss    reference_loss * tolerance_ratio
      min_hidden        smallest width with loss <= threshold
      S_T_gzip_bytes    gzipped size at min_hidden (our S_T proxy)

    This is the MDL two-part code: "smallest model that explains the data to
    within ``tolerance_ratio`` of the best we could do at any width".
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_losses: dict[int, float] = {}
    hidden_gzip: dict[int, int] = {}
    models: dict[int, StrategyObserver] = {}
    for h in hidden_sizes:
        model, hist = train_observer(
            corpus,
            hidden=h, depth=depth, epochs=epochs,
            device=device, seed=seed,
        )
        hidden_losses[h] = hist["final_policy_ce"]
        hidden_gzip[h] = _state_dict_gzip_bytes(model)
        models[h] = model

    reference = min(hidden_losses.values())
    threshold = reference * tolerance_ratio
    min_hidden = min(
        (h for h, loss in hidden_losses.items() if loss <= threshold),
        default=max(hidden_sizes),
    )
    return {
        "hidden_losses": hidden_losses,
        "hidden_gzip_bytes": hidden_gzip,
        "reference_loss": reference,
        "threshold_loss": threshold,
        "min_hidden": min_hidden,
        "S_T_gzip_bytes": hidden_gzip[min_hidden],
    }
