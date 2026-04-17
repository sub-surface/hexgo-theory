"""
GPU diffraction analyser for HexGo stone configurations.

Tests Proposition P4 (docs/theory/2026-04-17-hamkins-synthesis.md):
  the point set of stone positions in a long self-play has a Bragg-peak
  spectrum (pure-point diffraction), consistent with a Meyer set /
  quasi-crystalline structure.

Diffraction intensity of a finite point set S = {x_j} in the plane:

    I(k) = | sum_j exp(-2 pi i k . x_j) |^2

We implement this directly on the Eisenstein lattice: axial coordinates
(q, r) are converted to Cartesian (x, y) using the standard hex basis
with e_1 = (1, 0), e_2 = (1/2, sqrt(3)/2), then I(k) is evaluated on a
regular k-grid via torch on CUDA.

For |S| ~ 200 and |k-grid| = 65k, this is one (N, K) complex matmul and
runs in milliseconds on a 2060.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


# Hex basis: axial -> Cartesian.  e_1 = (1, 0); e_2 = (1/2, sqrt(3)/2).
_HEX_E2 = (0.5, math.sqrt(3.0) / 2.0)


def axial_to_cart(cells: Iterable[tuple[int, int]]) -> np.ndarray:
    """Convert iterable of (q, r) axial cells to a (N, 2) float array."""
    out = np.zeros((len(cells) if hasattr(cells, "__len__") else 0, 2),
                   dtype=np.float32)
    cs = list(cells)
    out = np.zeros((len(cs), 2), dtype=np.float32)
    for i, (q, r) in enumerate(cs):
        out[i, 0] = q + _HEX_E2[0] * r
        out[i, 1] = _HEX_E2[1] * r
    return out


def diffraction_intensity(
    points: np.ndarray,
    k_extent: float = 2.0 * math.pi,
    grid: int = 256,
    device: str | None = None,
    normalise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the diffraction intensity I(k) on a square k-grid.

    Parameters
    ----------
    points : (N, 2) ndarray of Cartesian positions (use axial_to_cart).
    k_extent : half-width of the k-grid; so k in [-k_extent, k_extent]^2.
               For lattice spacing 1 the first BZ edge is at k = pi; we
               default to 2 pi to see higher-order peaks too.
    grid : resolution per axis.
    device : "cuda", "cpu", or None (auto).
    normalise : if True, divide I by N^2 so I in [0, 1].

    Returns
    -------
    (kx, ky, I) all as torch tensors on the chosen device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N = int(points.shape[0])
    pts = torch.from_numpy(points.astype(np.float32)).to(device)  # (N, 2)

    kx_axis = torch.linspace(-k_extent, k_extent, grid, device=device)
    ky_axis = torch.linspace(-k_extent, k_extent, grid, device=device)
    kx, ky = torch.meshgrid(kx_axis, ky_axis, indexing="xy")
    k = torch.stack([kx.flatten(), ky.flatten()], dim=1)  # (K, 2)

    # phase[k, n] = 2 pi (k . x_n)
    phase = 2.0 * math.pi * (k @ pts.t())  # (K, N)
    #  S(k) = sum_n exp(-i phase[k, n])
    real = torch.cos(phase).sum(dim=1)
    imag = -torch.sin(phase).sum(dim=1)
    I = real * real + imag * imag  # (K,)
    if normalise and N > 0:
        I = I / (N * N)
    I = I.reshape(grid, grid)

    return kx, ky, I


def radial_profile(I: torch.Tensor, grid: int, k_extent: float,
                   n_bins: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged I(|k|). Returns (r_bins, mean_I)."""
    kx_axis = torch.linspace(-k_extent, k_extent, grid, device=I.device)
    ky_axis = torch.linspace(-k_extent, k_extent, grid, device=I.device)
    kx, ky = torch.meshgrid(kx_axis, ky_axis, indexing="xy")
    kr = torch.sqrt(kx * kx + ky * ky)

    kr_flat = kr.flatten().cpu().numpy()
    I_flat = I.flatten().cpu().numpy()
    bins = np.linspace(0.0, k_extent * math.sqrt(2.0), n_bins + 1)
    means = np.zeros(n_bins, dtype=np.float32)
    for b in range(n_bins):
        mask = (kr_flat >= bins[b]) & (kr_flat < bins[b + 1])
        if mask.any():
            means[b] = I_flat[mask].mean()
    centres = 0.5 * (bins[:-1] + bins[1:])
    return centres, means


def bragg_peak_fraction(I: torch.Tensor, threshold_q: float = 0.99) -> float:
    """
    Fraction of total I concentrated in the top (1 - threshold_q) of pixels.
    Bragg / pure-point ⇒ this is close to 1.  Amorphous ⇒ close to 1 - q.
    """
    I_flat = I.flatten()
    if I_flat.numel() == 0:
        return 0.0
    total = I_flat.sum()
    if total.item() == 0:
        return 0.0
    thresh = torch.quantile(I_flat, threshold_q)
    peak = I_flat[I_flat >= thresh].sum()
    return (peak / total).item()
