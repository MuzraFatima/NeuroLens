"""Unit tests for Phase 2: Theory, Proposition 2 Derivation, and Synthetic Generator.

Verifies:
1. MixedTopologySyntheticGenerator topology and hierarchical atom orthogonality.
2. Proposition 2 selective absorption simulation across variance regimes.
3. Parameterized synthetic sweeps across multi-parent share rho.
4. Phase 2 Exit Gate G2 evaluation.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.data.synth_generator import (
    MixedTopologySyntheticGenerator,
    run_synthetic_topology_sweep,
    verify_exit_gate_g2,
)


class TestPhase2TheoryGenerator(unittest.TestCase):
    def setUp(self):
        self.seed = 42

    def test_generator_hierarchical_orthogonality(self):
        """Child residual atom must be orthogonal to all ancestor residual atoms."""
        gen = MixedTopologySyntheticGenerator(n_features=32, d_model=32, rho=0.25, seed=self.seed)
        dag = gen.dag
        for c, parents in dag.parent_sets.items():
            for p in parents:
                dot = np.abs(np.dot(dag.residual_atoms[:, c], dag.residual_atoms[:, p]))
                self.assertLess(dot, 1e-10, f"Atom orthogonality violated between child {c} and parent {p}")

    def test_proposition2_selective_absorption_order(self):
        """Lower conditional variance parent must have higher or equal absorption rate."""
        # sigma1^2 = 0.02 (low var) vs sigma2^2 = 0.25 (high var)
        cell = MixedTopologySyntheticGenerator.simulate_proposition2_cell(
            sigma1_sq=0.02,
            sigma2_sq=0.25,
            l0_penalty_lambda=0.05,
            n_tokens=3000,
            seed=self.seed,
        )
        self.assertGreater(cell["abs_rate_p1"], cell["abs_rate_p2"])
        self.assertEqual(cell["order_agrees_with_prop2"], 1.0)

    def test_run_synthetic_topology_sweep_structure(self):
        """Sweep produces valid metrics across rho values."""
        rhos = (0.0, 0.25)
        sweep = run_synthetic_topology_sweep(
            rhos=rhos, n_features=32, d_model=32, n_tokens=500, seed=self.seed
        )
        self.assertIn(0.0, sweep)
        self.assertIn(0.25, sweep)
        self.assertGreaterEqual(sweep[0.25]["n_multi_parent_nodes"], sweep[0.0]["n_multi_parent_nodes"])
        self.assertGreater(sweep[0.25]["effective_l0"], 0.0)

    def test_verify_exit_gate_g2_passing(self):
        """Exit Gate G2 passes when multi-parent structure is measurable and scales with rho."""
        sweep = {
            0.0: {"n_multi_parent_nodes": 0},
            0.25: {"n_multi_parent_nodes": 6},
            0.50: {"n_multi_parent_nodes": 14},
        }
        res = verify_exit_gate_g2(sweep)
        self.assertTrue(res["gate_passed"])
        self.assertTrue(res["multi_parent_measurable"])
        self.assertTrue(res["monotonic_scaling_verified"])

    def test_verify_exit_gate_g2_failing_when_no_multi_parents(self):
        """Exit Gate G2 fails if no multi-parent nodes exist at rho >= 0.10."""
        sweep = {
            0.0: {"n_multi_parent_nodes": 0},
            0.25: {"n_multi_parent_nodes": 0},
            0.50: {"n_multi_parent_nodes": 0},
        }
        res = verify_exit_gate_g2(sweep)
        self.assertFalse(res["gate_passed"])
        self.assertFalse(res["multi_parent_measurable"])


if __name__ == "__main__":
    unittest.main()
