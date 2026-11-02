"""Public API for topocircuit.benchmarks.

Exports the 112-cell MPAbs benchmark suite, its per-cell dataclass,
and the L2-regularized logistic regression probe.
"""

from topocircuit.benchmarks.mpabs import (
    LogisticRegressionProbe,
    MPAbsBenchmark,
    MPAbsCell,
)

__all__ = [
    "MPAbsBenchmark",
    "MPAbsCell",
    "LogisticRegressionProbe",
]
