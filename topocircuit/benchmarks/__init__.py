"""Public API for topocircuit.benchmarks.

Exports the 112-cell MPAbs benchmark suite, its per-cell dataclass,
the L2-regularized logistic regression probe, and the Exit Gate G4 checker.
"""

from topocircuit.benchmarks.mpabs import (
    LogisticRegressionProbe,
    MPAbsBenchmark,
    MPAbsCell,
    verify_exit_gate_g4,
)

__all__ = [
    "MPAbsBenchmark",
    "MPAbsCell",
    "LogisticRegressionProbe",
    "verify_exit_gate_g4",
]
