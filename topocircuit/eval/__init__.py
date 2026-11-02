"""Public API for topocircuit.eval.

Exports structure-recovery metrics, cross-seed reproducibility,
causal readout check, statistical tests, and the Sparse Feature
Circuit faithfulness evaluator.
"""

from topocircuit.eval.circuits import SparseFeatureCircuitEvaluator
from topocircuit.eval.metrics import (
    compute_cross_seed_reproducibility,
    evaluate_dag_structure,
    holm_bonferroni_correction,
    paired_wilcoxon_and_bootstrap,
    run_causal_readout_check,
)

__all__ = [
    "evaluate_dag_structure",
    "compute_cross_seed_reproducibility",
    "run_causal_readout_check",
    "paired_wilcoxon_and_bootstrap",
    "holm_bonferroni_correction",
    "SparseFeatureCircuitEvaluator",
]
