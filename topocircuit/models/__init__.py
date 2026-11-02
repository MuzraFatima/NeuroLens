"""Public API for topocircuit.models.

Exposes the DR-SAE and all five baseline architectures,
the SAELens/dictionary_learning compatibility wrapper, and
the shared BatchTopK activation helper.
"""

from topocircuit.models.baselines import (
    HSAE,
    AGSAELite,
    BatchTopKSAE,
    MatryoshkaSAE,
    PostHocDRSAE,
)
from topocircuit.models.dr_sae import DRSAE, batch_topk, recover_ista_codes
from topocircuit.models.saelens_plugin import (
    DRSAELensArchitecture,
    DRSAELensConfig,
    DRSAELensWrapper,
)

__all__ = [
    "DRSAE",
    "batch_topk",
    "recover_ista_codes",
    "BatchTopKSAE",
    "MatryoshkaSAE",
    "HSAE",
    "AGSAELite",
    "PostHocDRSAE",
    "DRSAELensConfig",
    "DRSAELensWrapper",
    "DRSAELensArchitecture",
]
