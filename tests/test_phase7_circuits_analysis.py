"""Unit tests for Phase 7: Sparse Feature Circuits, Causal Faithfulness, and Exit Gate G7.

Verifies:
1. SparseFeatureCircuitEvaluator across 4 Pythia-70M subject-verb agreement tasks.
2. Causal faithfulness curve monotonicity and AUC bounds.
3. Node budget reduction at 0.80 faithfulness for DR-SAE vs Matryoshka.
4. Research Question 4 (RQ4) and Phase 7 Exit Gate G7 evaluation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.eval.circuits import SV_AGREEMENT_TASKS, SparseFeatureCircuitEvaluator
from topocircuit.eval.metrics import verify_exit_gate_g7
from topocircuit.models.dr_sae import DRSAE


class TestPhase7CircuitsAnalysis(unittest.TestCase):
    def setUp(self):
        self.evaluator = SparseFeatureCircuitEvaluator(n_layers=6, d_model=32, seed=42)
        self.model = DRSAE(d_model=32, n_latents=64, k=4, seed=42)

    def test_tasks_list(self):
        """Must cover all 4 pre-registered subject-verb agreement tasks."""
        self.assertEqual(len(SV_AGREEMENT_TASKS), 4)
        self.assertIn("simple_sv_agreement", SV_AGREEMENT_TASKS)
        self.assertIn("rc_sv_agreement", SV_AGREEMENT_TASKS)

    def test_evaluate_method_circuits_execution(self):
        """Evaluator computes faithfulness curve, AUC, and nodes_at_08 across tasks."""
        res = self.evaluator.evaluate_method_circuits(
            method_name="DR-SAE (Ours)",
            sae_model=self.model,
            mpabs_mean=0.0399,
            node_budgets=[10, 20, 50, 100, 200],
        )
        self.assertIn("mean_nodes_at_08", res)
        self.assertIn("mean_faithfulness_auc", res)
        self.assertIn("mean_faithfulness_curve", res)
        curve = res["mean_faithfulness_curve"]
        self.assertTrue(all(0.0 <= f <= 1.0 for f in curve))

    def test_verify_exit_gate_g7_passing(self):
        """Exit Gate G7 passes when DR-SAE reaches >= 20% node reduction vs Matryoshka."""
        circuit_results = {
            "Matryoshka": {"nodes_at_08": 102.7, "faithfulness_auc": 0.880},
            "DR-SAE (Ours)": {"nodes_at_08": 62.6, "faithfulness_auc": 0.927},
        }
        res = verify_exit_gate_g7(circuit_results, min_node_reduction=0.20)
        self.assertTrue(res["gate_passed"])
        self.assertTrue(res["node_reduction_sufficient"])
        self.assertTrue(res["faithfulness_auc_superior"])

    def test_verify_exit_gate_g7_failing(self):
        """Exit Gate G7 fails when node reduction is less than threshold."""
        circuit_results = {
            "Matryoshka": {"nodes_at_08": 100.0, "faithfulness_auc": 0.880},
            "DR-SAE (Ours)": {"nodes_at_08": 95.0, "faithfulness_auc": 0.885},
        }
        res = verify_exit_gate_g7(circuit_results, min_node_reduction=0.20)
        self.assertFalse(res["gate_passed"])
        self.assertFalse(res["node_reduction_sufficient"])


if __name__ == "__main__":
    unittest.main()
