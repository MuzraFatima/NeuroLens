"""Public API for topocircuit.data.

Exports the mixed-topology synthetic ground-truth generator
and its associated DAG dataclass.
"""

from topocircuit.data.synth_generator import (
    MixedTopologyDAG,
    MixedTopologySyntheticGenerator,
)

__all__ = [
    "MixedTopologyDAG",
    "MixedTopologySyntheticGenerator",
]
