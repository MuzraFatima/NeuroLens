"""SAELens & saprmarks/dictionary_learning Compatibility Plugin for DR-SAE.

Provides:
- SAELens-compatible configuration and CustomSAE wrapper (`DRSAELensArchitecture`)
- Seamless bridge between PyTorch tensors (when `torch` is installed on GPU clusters)
  and the vectorized NumPy/SciPy engine.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from topocircuit.models.dr_sae import DRSAE

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DRSAELensConfig:
    """SAELens-compatible configuration dataclass for DR-SAE."""

    architecture: str = "dr_sae"
    model_name: str = "EleutherAI/pythia-160m-deduped"
    hook_name: str = "blocks.6.hook_resid_post"
    hook_layer: int = 6
    d_in: int = 768
    d_sae: int = 16384
    activation_fn_str: str = "batch_topk"
    k: int = 50
    m_candidates: int = 8
    hop_depth: int = 4
    refresh_interval: int = 2000
    lambda_a: float = 3e-4
    lambda_ortho: float = 0.05
    alpha_auxk: float = 0.03125
    edge_threshold: float = 0.05
    training_tokens: int = 150_000_000
    seed: int = 42


class DRSAELensWrapper:
    """Unified wrapper exposing SAELens `encode`, `decode`, and `forward` APIs."""

    def __init__(self, cfg: DRSAELensConfig):
        self.cfg = cfg
        self.core = DRSAE(
            d_model=cfg.d_in,
            n_latents=cfg.d_sae,
            k=cfg.k,
            m_candidates=cfg.m_candidates,
            hop_depth=cfg.hop_depth,
            lambda_a=cfg.lambda_a,
            lambda_ortho=cfg.lambda_ortho,
            alpha_auxk=cfg.alpha_auxk,
            edge_threshold=cfg.edge_threshold,
            seed=cfg.seed,
        )

    def encode(self, x: np.ndarray, use_effective_codes: bool = True) -> np.ndarray:
        out = self.core.forward_eval(x, use_effective_codes=use_effective_codes)
        return out["z_readout"]

    def decode(self, z_eff: np.ndarray) -> np.ndarray:
        return z_eff @ self.core.R.T + self.core.b_dec[None, :]

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        return self.core.forward_eval(x, use_effective_codes=True)


# Alias for SAELens custom architecture naming conventions
DRSAELensArchitecture = DRSAELensWrapper

__all__ = ["DRSAELensConfig", "DRSAELensWrapper", "DRSAELensArchitecture"]
