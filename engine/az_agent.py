"""AlphaZero-style agent wrapping a trained UnifiedNet.

Phase 1 variant (no MCTS):
    Encode current position -> UnifiedNet -> policy logits over encoded
    grid -> mask to empty+legal cells -> softmax (optionally temperatured)
    -> argmax or sample.

Phase 2 variant (with MCTS):
    Same net but used as (prior, value) pair inside a PUCT search. NOT
    yet wired up in this file -- see engine/mcts.py for the tree.

Minimal-interface: the agent has `name: str` and
`choose_move(game) -> (q, r)`. This matches the protocol used by
engine.agents.* and the harness in experiments/harness.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from engine import HexGame
from engine.alphazero import UnifiedNet
from engine.observer import encode_position


def load_unified_net(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> UnifiedNet:
    """Load a trained UnifiedNet from a checkpoint dict."""
    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = UnifiedNet(hidden=ck["hidden"], depth=ck["depth"]).to(device)
    model.load_state_dict(ck["state_dict"])
    # Freeze -- inference-only. (Using train(False) + requires_grad_(False)
    # to match observer's freezing idiom.)
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class AlphaZeroAgent:
    """Policy-head agent (no MCTS). Sample from masked softmax."""

    def __init__(
        self,
        model: UnifiedNet,
        *,
        name: str = "az_policy",
        temperature: float = 1.0,
        device: str = "cuda",
        seed: int | None = None,
    ):
        self.model = model
        self.name = name
        self.temperature = float(temperature)
        self.device = device
        self.rng = np.random.default_rng(seed)

    def choose_move(self, game: HexGame) -> tuple[int, int]:
        arr, origin = encode_position(game, to_move=game.current_player, pad=4)
        q_min, r_min, H, W = origin

        x = torch.from_numpy(arr[None]).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        logits = out["policy"][0].cpu().numpy()  # (H, W)

        # mask: allow only (empty cells in encoded grid) AND (legal move on the infinite board)
        empty_plane = arr[0] > 0.5
        mask = np.full_like(logits, -np.inf, dtype=np.float32)

        cands = list(game.candidates)
        any_legal = False
        if cands:
            for (q, r) in cands:
                col = q - q_min
                row = r - r_min
                if 0 <= row < H and 0 <= col < W and empty_plane[row, col]:
                    mask[row, col] = 0.0
                    any_legal = True
        if not any_legal:
            # opening: play the centre of the encoded grid.
            return (0, 0)

        logits = logits + mask
        # temperature + softmax
        if self.temperature <= 0:
            flat = logits.flatten()
            best = int(np.argmax(flat))
        else:
            scaled = logits / max(1e-8, self.temperature)
            scaled = scaled - np.max(scaled)
            probs = np.exp(scaled)
            probs = probs / max(1e-12, probs.sum())
            flat = probs.flatten()
            best = int(self.rng.choice(flat.size, p=flat))

        row = best // W
        col = best % W
        q = col + q_min
        r = row + r_min
        return (int(q), int(r))


def make_az_agent(
    ckpt: str | Path = "artifacts/checkpoints/az_pretrain.pt",
    *,
    name: str = "az_policy",
    temperature: float = 0.0,
    device: str | None = None,
    seed: int | None = None,
) -> AlphaZeroAgent:
    """Convenience factory for AlphaZeroAgent."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_unified_net(ckpt, device=device)
    return AlphaZeroAgent(
        model, name=name, temperature=temperature, device=device, seed=seed,
    )
