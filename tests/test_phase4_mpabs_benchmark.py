"""Unit tests for Phase 4: MPAbs Benchmark, Multi-Parent Absorption Suite, and Exit Gate G4.

Verifies:
1. MPAbs benchmark contains >= 100 cells with >= 20 tokens per cell.
2. Parent logistic probes meet the >= 0.90 validation accuracy gate.
3. Multi-parent absorption and joint absorption scoring across orthographic, semantic, and case parents.
4. Phase 4 Exit Gate G4 evaluation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.benchmarks.mpabs import MPAbsBenchmark, verify_exit_gate_g4
from topocircuit.models.baselines import BatchTopKSAE


class TestPhase4MPAbsBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = MPAbsBenchmark(d_model=32, tokens_per_cell=25, seed=42)

    def test_mpabs_benchmark_cell_count_and_tokens(self):
        """Benchmark must contain at least 100 cells with >= 20 tokens each."""
        self.assertGreaterEqual(len(self.bench.cells), 100)
        for cell in self.bench.cells:
            self.assertGreaterEqual(cell.n_tokens, 20)
            self.assertEqual(len(cell.parent_ids), 3)

    def test_parent_probe_validation_accuracy_gate(self):
        """All parent probes must achieve validation accuracy >= 0.90."""
        for name, acc in self.bench.probe_val_accuracies.items():
            self.assertGreaterEqual(
                acc, 0.90, f"Probe for parent '{name}' failed the >= 0.90 accuracy gate (got {acc:.4f})"
            )

    def test_evaluate_sae_on_mpabs(self):
        """Evaluating a baseline SAE produces valid mean, joint, and per-parent absorption metrics."""
        model = BatchTopKSAE(d_model=32, n_latents=64, k=4, seed=42)
        res = self.bench.evaluate_sae_on_mpabs(model, use_effective_codes=False)
        self.assertIn("mean_mpabs", res)
        self.assertIn("mean_joint_abs", res)
        self.assertIn("first_letter_abs", res)
        self.assertIn("semantic_cat_abs", res)
        self.assertIn("case_abs", res)
        self.assertGreaterEqual(res["mean_mpabs"], 0.0)
        self.assertLessEqual(res["mean_mpabs"], 1.0)

    def test_verify_exit_gate_g4_passing(self):
        """Exit Gate G4 passes when cell count, tokens per cell, and probe accuracies satisfy criteria."""
        gate_res = verify_exit_gate_g4(self.bench)
        self.assertTrue(gate_res["gate_passed"])
        self.assertTrue(gate_res["sufficient_cells"])
        self.assertTrue(gate_res["sufficient_tokens_per_cell"])
        self.assertTrue(gate_res["probes_exceed_accuracy_gate"])


if __name__ == "__main__":
    unittest.main()
