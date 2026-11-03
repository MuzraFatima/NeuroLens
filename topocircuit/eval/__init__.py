"""Evaluation: statistics and gates, synthetic structure recovery, and the Proposition 2 test."""

from topocircuit.eval.metrics import (
    check_gate,
    compute_fvu,
    compute_minimum_detectable_effect,
    holm_bonferroni_correction,
    paired_wilcoxon_and_bootstrap,
)

__all__ = [
    "check_gate",
    "compute_fvu",
    "compute_minimum_detectable_effect",
    "holm_bonferroni_correction",
    "paired_wilcoxon_and_bootstrap",
]
