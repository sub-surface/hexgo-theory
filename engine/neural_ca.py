"""
Neural cellular-automaton policy agent for HexGo, on torch + CUDA.

Design.
  - Board state is a (3, H, W) tensor of one-hot channels [empty, P1, P2],
    windowed around the current stone cloud with a pad margin so the NCA
    receptive field sees the full WIN_LENGTH=6 neighbourhood.
  - The "cellular automaton" is a stack of hex-convolution layers: 3x3 axial
    conv with the two unused corners zeroed, producing a true 6-neighbour
    hex kernel. Depth D gives receptive-field radius D hops — we default
    D=6 to match WIN_LENGTH.
  - Output: a single (1, H, W) score map; masked by legality and by the
    adjacency frontier, then argmax'd to choose the move.
  - Parameters are random by default (no training yet). The point of the
    untrained version is to (a) prove the GPU pipeline is live, (b) give
    us a baseline random-NN agent to measure against once training lands,
    and (c) provide a substrate for neural-CA-style evolution experiments.

This is the torch+CUDA implementation required by the
``feedback_neural_ca_torch`` user preference (2026-04-17).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


# Hex 6-neighbour mask over a 3x3 axial kernel.
# For axial coords (q, r), the six neighbours are at offsets
# (+-1,0), (0,+-1), (+1,-1), (-1,+1).  In 3x3 kernel (row=dr, col=dq) the
# two corners (r,q) = (-1,-1) and (+1,+1) are NOT neighbours.
_HEX_MASK = torch.tensor(
    [[1.0, 1.0, 0.0],
     [1.0, 1.0, 1.0],
     [0.0, 1.0, 1.0]],
    dtype=torch.float32,
)


class HexConv2d(nn.Module):
    """3x3 axial convolution with the non-hex corners masked out."""

    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        # Register the mask as a buffer so it moves with .to(device).
        self.register_buffer("mask", _HEX_MASK.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Re-apply the mask every forward (so gradient updates stay hex-shaped).
        w = self.conv.weight * self.mask[None, None, :, :]
        return nn.functional.conv2d(x, w, self.conv.bias, padding=1)


class NeuralCA(nn.Module):
    """
    Stack of hex convolutions. Input: (B, 3, H, W) — one-hot [empty, P1, P2].
    Output: (B, 1, H, W) — score map for the CURRENT player to move.

    The player-to-move is encoded by which channel counts as "own". We swap
    channels 1 and 2 on odd-parity turns so the network always sees "empty,
    me, opp" regardless of whose turn it is.
    """

    def __init__(self, n_layers: int = 6, hidden: int = 16):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for _ in range(n_layers):
            layers.append(HexConv2d(in_ch, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_ch = hidden
        layers.append(HexConv2d(in_ch, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class NeuralCAAgent:
    """
    A CAAgent-shaped adaptor around a NeuralCA. Stateless call surface:
    choose_move(game) -> (q, r).

    - Builds a Cartesian window around the current stone cloud + pad.
    - Runs the NCA forward once per move.
    - Picks the argmax over legal cells in the window.
    - Falls back to a random legal move if the window misses all legals
      (should never happen with pad >= 3).
    """
    name: str = "neural_ca"
    model: NeuralCA = field(default_factory=lambda: NeuralCA(n_layers=6, hidden=16))
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    pad: int = 6        # window pad — must be >= WIN_LENGTH so the NCA sees full context
    seed: int = 0       # torch RNG seed for reproducibility

    def __post_init__(self):
        with torch.no_grad():
            # Deterministic weights from `self.seed` (call sites can vary seed
            # to get distinct untrained policies).
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            for p in self.model.parameters():
                p.copy_(torch.normal(mean=0.0, std=0.2,
                                     size=p.shape, generator=gen))
        self.model.to(self.device)
        # Disable training-mode side effects (e.g. dropout) for inference.
        self.model.train(False)

    @torch.no_grad()
    def _encode(self, game) -> tuple[torch.Tensor, tuple[int, int, int, int], int]:
        """
        Build a (1, 3, H, W) one-hot tensor around the current stone cloud.
        Returns (tensor, (q_min, r_min, H, W), player).
        """
        player = game.current_player
        opponent = 3 - player

        if game.board:
            qs = [q for (q, _) in game.board]
            rs = [r for (_, r) in game.board]
            q_min, q_max = min(qs) - self.pad, max(qs) + self.pad
            r_min, r_max = min(rs) - self.pad, max(rs) + self.pad
        else:
            q_min, q_max = -self.pad, self.pad
            r_min, r_max = -self.pad, self.pad

        W = q_max - q_min + 1
        H = r_max - r_min + 1

        arr = np.zeros((3, H, W), dtype=np.float32)
        # Channel 0 = empty — default everything empty.
        arr[0, :, :] = 1.0
        for (q, r), owner in game.board.items():
            col = q - q_min
            row = r - r_min
            arr[0, row, col] = 0.0
            if owner == player:
                arr[1, row, col] = 1.0
            else:
                arr[2, row, col] = 1.0

        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor, (q_min, r_min, H, W), player

    @torch.no_grad()
    def choose_move(self, game) -> tuple[int, int]:
        legal = set(game.legal_moves())
        if not legal:
            return (0, 0)

        tensor, (q_min, r_min, H, W), _ = self._encode(game)
        scores = self.model(tensor)[0, 0]  # (H, W)
        scores_np = scores.detach().cpu().numpy()

        # Mask to legal cells only.
        best = None
        best_s = -float("inf")
        for (q, r) in legal:
            col = q - q_min
            row = r - r_min
            if 0 <= row < H and 0 <= col < W:
                s = float(scores_np[row, col])
                if s > best_s:
                    best_s = s
                    best = (q, r)
        if best is None:
            import random as _r
            return _r.choice(list(legal))
        return best


# ── CA-prior initialisers (synthesis §7) ─────────────────────────────────────
#
# Factory entrypoint `make_nca_variant(prior, seed)` returns a NeuralCAAgent
# whose first convolutional layer is overwritten with a hand-crafted kernel
# corresponding to one of five priors. Higher layers remain randomly
# initialised. Downstream training (self-play policy gradient, evolutionary
# search) is expected to refine these kernels; the priors are warm-starts,
# not constraints.
#
# Priors:
#   "random"          — no init beyond the default normal(0, 0.2)
#   "d6_tied"         — weight-tie within the 12-element D_6 hex symmetry group
#   "line_detector"   — first layer responds to adjacent own-stone pairs along
#                       each of the three Eisenstein axes
#   "erdos_selfridge" — first layer computes an approximation of
#                       phi(c) = sum_L alpha^{n_L^own} 1[n_L^opp == 0]
#                       over the 6-lines through c
#   "combo"           — d6_tied + line_detector + erdos_selfridge stacked

def _d6_tie_first_conv(model: NeuralCA) -> None:
    """Average the first HexConv2d's weight over the 6 rotational images of
    the 3x3 hex kernel, and over its horizontal reflection. Imposes D_6 / D_3
    approximate invariance on layer 0 only — cheap surrogate for true
    per-layer equivariance (real equivariance needs Steerable CNNs)."""
    with torch.no_grad():
        hex_conv = model.net[0]
        if not isinstance(hex_conv, HexConv2d):
            return
        w = hex_conv.conv.weight  # (out, in, 3, 3)
        # 6 rotations on the axial 3x3 kernel: enumerate by 60° rotation of
        # the 6 hex-neighbour positions. For the masked 3x3, the 6 active
        # positions form a hexagonal ring; rotating them permutes 6 values.
        ring_coords = [(0, 0), (0, 1), (1, 2), (2, 2), (2, 1), (1, 0)]  # ccw
        centre = (1, 1)
        averaged = w.clone()
        for k in range(1, 6):
            rot = w.clone()
            for i, (r0, c0) in enumerate(ring_coords):
                r1, c1 = ring_coords[(i + k) % 6]
                rot[:, :, r1, c1] = w[:, :, r0, c0]
            rot[:, :, centre[0], centre[1]] = w[:, :, centre[0], centre[1]]
            averaged = averaged + rot
        averaged = averaged / 6.0
        # Horizontal reflection (swap two opposite neighbour pairs).
        refl = averaged.clone()
        for (r0, c0), (r1, c1) in [((0, 0), (2, 2)), ((0, 1), (2, 1)),
                                   ((1, 0), (1, 2))]:
            refl[:, :, r0, c0] = averaged[:, :, r1, c1]
            refl[:, :, r1, c1] = averaged[:, :, r0, c0]
        hex_conv.conv.weight.copy_((averaged + refl) / 2.0)


def _line_detector_first_conv(model: NeuralCA) -> None:
    """Overwrite first layer's own-channel weights so each of the first three
    output channels activates on an own-stone pair along one Eisenstein axis.
    Remaining channels stay random. Three filters are enough for the three
    hex axes; the depth-2 composition gives triples."""
    with torch.no_grad():
        hex_conv = model.net[0]
        if not isinstance(hex_conv, HexConv2d):
            return
        w = hex_conv.conv.weight  # (out, in=3, 3, 3)
        out_ch = w.shape[0]
        # Axial 3x3 kernel coordinates for the six neighbours (row, col):
        # axis 0 (q, horizontal):   (1, 0) and (1, 2)
        # axis 1 (r, anti-diag):    (0, 1) and (2, 1)
        # axis 2 (q+r, main diag):  (0, 0) and (2, 2)  — but these are masked!
        # The two masked corners are (0, 2) and (2, 0). Our diagonal axis is
        # (+1,-1) / (-1,+1), i.e. coordinates (0, 2) and (2, 0) — also masked.
        # So the hex-mask hex kernel encodes two axis-aligned pairs natively
        # and one diagonal pair as (0, 0) and (2, 2).
        axis_pairs = [
            [(1, 0), (1, 2)],    # axis 0
            [(0, 1), (2, 1)],    # axis 1
            [(0, 0), (2, 2)],    # axis 2 (diagonal, corners of hex mask)
        ]
        # Zero out the first three output channels, then put +1 on the own
        # channel (index 1) at both ends of the pair for that axis.
        for ch, pairs in enumerate(axis_pairs):
            if ch >= out_ch:
                break
            w[ch].zero_()
            for (r, c) in pairs:
                w[ch, 1, r, c] = 1.0  # channel 1 = own-stone
            # Bias toward zero so an empty neighbourhood gives 0 activation.
            if hex_conv.conv.bias is not None:
                hex_conv.conv.bias[ch] = -1.5


def _erdos_selfridge_first_conv(model: NeuralCA, alpha: float = 2.0) -> None:
    """Overwrite first layer's own-channel weight for channels 3..5 (if present)
    so they compute a Erdős–Selfridge-style potential contribution for each of
    the three axes: count own-stones along the axis through the centre cell,
    zero if any opponent stone blocks. Implemented at layer 0 as a weighted
    sum of own-channel values along each axis; the opp-channel contribution
    is negative with large magnitude, so any opp stone in the receptive field
    pushes the output below 0 (ReLU kills it)."""
    with torch.no_grad():
        hex_conv = model.net[0]
        if not isinstance(hex_conv, HexConv2d):
            return
        w = hex_conv.conv.weight
        out_ch = w.shape[0]
        axis_coords = [
            [(1, 0), (1, 1), (1, 2)],       # axis 0
            [(0, 1), (1, 1), (2, 1)],       # axis 1
            [(0, 0), (1, 1), (2, 2)],       # axis 2 (diagonal)
        ]
        # Channels 3, 4, 5 host the potential per axis (if hidden >= 6).
        for offset, coords in enumerate(axis_coords):
            ch = 3 + offset
            if ch >= out_ch:
                break
            w[ch].zero_()
            for (r, c) in coords:
                w[ch, 1, r, c] = alpha          # own stone: + alpha^contrib
                w[ch, 2, r, c] = -10.0          # opp stone: massive negative
            if hex_conv.conv.bias is not None:
                hex_conv.conv.bias[ch] = 0.0


def make_nca_variant(prior: str, seed: int = 0,
                      n_layers: int = 6, hidden: int = 16) -> NeuralCAAgent:
    """Produce a NeuralCAAgent with the named prior applied.

    prior ∈ {"random", "d6_tied", "line_detector", "erdos_selfridge", "combo"}.
    """
    model = NeuralCA(n_layers=n_layers, hidden=hidden)
    agent = NeuralCAAgent(name=f"nca_{prior}", model=model, seed=seed)
    # __post_init__ has already done the random init + .to(device) + eval().
    # We now overwrite the first layer's weights in-place. The agent is still
    # on `self.device`; the helpers use `torch.no_grad()` and the agent's
    # parameters, so device placement is preserved.
    if prior == "random":
        pass
    elif prior == "d6_tied":
        _d6_tie_first_conv(agent.model)
    elif prior == "line_detector":
        _line_detector_first_conv(agent.model)
    elif prior == "erdos_selfridge":
        _erdos_selfridge_first_conv(agent.model)
    elif prior == "combo":
        _d6_tie_first_conv(agent.model)
        _line_detector_first_conv(agent.model)
        _erdos_selfridge_first_conv(agent.model)
    else:
        raise ValueError(f"unknown prior {prior!r}")
    return agent
