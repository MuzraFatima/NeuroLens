"""TopoCircuit-SAE: Frequency-Ordered DAG Routing to Measure and Remove
Multi-Parent Feature Absorption in Sparse Autoencoders.

Author: Muzra Fatima
Version: 1.0.0
"""

from topocircuit.benchmarks.mpabs import MPAbsBenchmark, MPAbsCell
from topocircuit.data.synth_generator import (
    MixedTopologyDAG,
    MixedTopologySyntheticGenerator,
)
from topocircuit.eval.circuits import SparseFeatureCircuitEvaluator
from topocircuit.models.baselines import (
    HSAE,
    AGSAELite,
    BatchTopKSAE,
    MatryoshkaSAE,
    PostHocDRSAE,
)
from topocircuit.models.dr_sae import DRSAE
from topocircuit.models.saelens_plugin import DRSAELensConfig, DRSAELensWrapper

__version__ = "1.0.0"
__author__ = "Muzra Fatima"

__all__ = [
    "DRSAE",
    "BatchTopKSAE",
    "MatryoshkaSAE",
    "HSAE",
    "AGSAELite",
    "PostHocDRSAE",
    "DRSAELensConfig",
    "DRSAELensWrapper",
    "MPAbsBenchmark",
    "MPAbsCell",
    "MixedTopologySyntheticGenerator",
    "MixedTopologyDAG",
    "SparseFeatureCircuitEvaluator",
    "__version__",
    "__author__",
]
