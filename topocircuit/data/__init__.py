"""Public API for topocircuit.data.

Exports the mixed-topology synthetic ground-truth generator
and its associated DAG dataclass.
"""

from topocircuit.data.synth_generator import (
    MixedTopologyDAG,
    MixedTopologySyntheticGenerator,
    run_synthetic_topology_sweep,
    verify_exit_gate_g2,
)

__all__ = [
    "MixedTopologyDAG",
    "MixedTopologySyntheticGenerator",
    "run_synthetic_topology_sweep",
    "verify_exit_gate_g2",
]
