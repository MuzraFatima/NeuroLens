"""Empirical test of Proposition 2 (selective absorption order).

Toy setting, following Chanin et al.'s two-feature construction extended to two parents:
three orthogonal feature directions p1, p2 and c. On child tokens c fires and both parents fire
with magnitudes 1 + sigma_i * eps (eps ~ N(0, 1)). On parent-only tokens a single parent fires
alone. A plain ReLU SAE with an L1 penalty lambda is trained from a random start. Absorption of
parent i is the share of child tokens on which the latent that encodes p_i (matched on parent-only
tokens) stays silent.

Proposition 2 predicts that the parent with the smaller sigma_i^2 is absorbed first. A cell is
"informative" when the two absorption rates differ by more than ``min_gap``; agreement is measured
over informative cells only, and the number of informative cells is reported alongside it.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _toy_data(sigma1_sq: float, sigma2_sq: float, n: int, d: int, rng: np.random.Generator):
    U = np.linalg.qr(rng.standard_normal((d, 3)))[0]  # orthonormal p1, p2, c
    kind = rng.choice(3, size=n, p=[0.4, 0.3, 0.3])  # 0 = child token, 1 = p1 only, 2 = p2 only
    A = np.zeros((n, 3))
    child = kind == 0
    nc = int(child.sum())
    A[child, 0] = np.maximum(0.05, 1.0 + np.sqrt(sigma1_sq) * rng.standard_normal(nc))
    A[child, 1] = np.maximum(0.05, 1.0 + np.sqrt(sigma2_sq) * rng.standard_normal(nc))
    A[child, 2] = 1.0
    A[kind == 1, 0] = 1.0
    A[kind == 2, 1] = 1.0
    X = A @ U.T + 0.01 * rng.standard_normal((n, d))
    return X.astype(np.float32), kind, U


def run_cell(
    sigma1_sq: float,
    sigma2_sq: float,
    lam: float,
    seed: int = 0,
    n_tokens: int = 4000,
    d: int = 8,
    n_latents: int = 4,
    n_steps: int = 1500,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    X, kind, _ = _toy_data(sigma1_sq, sigma2_sq, n_tokens, d, rng)
    x = torch.from_numpy(X)
    torch.manual_seed(seed)
    W = F.normalize(torch.randn(d, n_latents), dim=0)
    W_dec = torch.nn.Parameter(W.clone())
    W_enc = torch.nn.Parameter(W.t().clone())
    b_enc = torch.nn.Parameter(torch.zeros(n_latents))
    opt = torch.optim.Adam([W_dec, W_enc, b_enc], lr=1e-2)
    for _ in range(n_steps):
        z = F.relu(x @ W_enc.t() + b_enc)
        dec = F.normalize(W_dec, dim=0)
        loss = (x - z @ dec.t()).pow(2).sum(-1).mean() + lam * z.sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = F.relu(x @ W_enc.t() + b_enc).numpy()
    # The latent for parent i is the one that fires most on tokens where p_i fires alone.
    lat1 = int(z[kind == 1].mean(0).argmax())
    lat2 = int(z[kind == 2].mean(0).argmax())
    child = kind == 0
    thr1 = 0.1 * z[kind == 1, lat1].mean()
    thr2 = 0.1 * z[kind == 2, lat2].mean()
    abs1 = float((z[child, lat1] <= thr1).mean())
    abs2 = float((z[child, lat2] <= thr2).mean())
    return {
        "sigma1_sq": sigma1_sq,
        "sigma2_sq": sigma2_sq,
        "lambda": lam,
        "seed": seed,
        "abs_p1": abs1,
        "abs_p2": abs2,
        "same_latent": float(lat1 == lat2),
    }


def run_grid(
    sigmas: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.4),
    lambdas: Sequence[float] = (0.05, 0.1, 0.2, 0.4),
    seed: int = 0,
    min_gap: float = 0.1,
    **cell_kw,
) -> Dict[str, object]:
    cells: List[Dict[str, float]] = []
    for i, s1 in enumerate(sigmas):
        for j, s2 in enumerate(sigmas):
            for k, lam in enumerate(lambdas):
                cells.append(run_cell(s1, s2, lam, seed=seed + 100 * i + 10 * j + k, **cell_kw))
    informative, agree = 0, 0
    for c in cells:
        if c["sigma1_sq"] == c["sigma2_sq"] or c["same_latent"]:
            continue
        gap = c["abs_p1"] - c["abs_p2"]
        if abs(gap) <= min_gap:
            continue
        informative += 1
        lower_is_p1 = c["sigma1_sq"] < c["sigma2_sq"]
        agree += int((gap > 0) == lower_is_p1)
    return {
        "cells": cells,
        "n_cells": len(cells),
        "n_informative": informative,
        "agreement_rate": agree / informative if informative else float("nan"),
        "min_gap": min_gap,
    }
