"""SAE models (PyTorch, CPU-friendly) and the shared training loop."""

from topocircuit.models.sae import DRSAE, BatchTopKSAE, MatryoshkaSAE, SAEConfig, batch_topk
from topocircuit.models.train import (
    fit_ag_sae_lite,
    fit_post_hoc_dr,
    induce_parent_sets_greedy,
    train_method,
    train_sae,
)

__all__ = [
    "SAEConfig",
    "BatchTopKSAE",
    "MatryoshkaSAE",
    "DRSAE",
    "batch_topk",
    "train_sae",
    "train_method",
    "fit_post_hoc_dr",
    "fit_ag_sae_lite",
    "induce_parent_sets_greedy",
]
