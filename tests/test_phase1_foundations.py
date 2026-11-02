"""Unit tests for Phase 1: Foundations, Reproduction, and Exit Gate G1.

Verifies:
1. Baseline architectures (BatchTopKSAE, MatryoshkaSAE) forward passes and reconstructions.
2. Fraction of Variance Unexplained (FVU) calculation accuracy.
3. Compute estimation and token throughput profiling (10M token projection).
4. Exit Gate G1 evaluation criteria.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.eval.metrics import (
    compute_fvu,
    profile_token_throughput,
    verify_exit_gate_g1,
)
from topocircuit.models.baselines import BatchTopKSAE, MatryoshkaSAE


class TestPhase1Foundations(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.d_model = 32
        self.n_latents = 64
        self.k = 4
        self.batch_size = 64
        self.X_sample = self.rng.standard_normal((self.batch_size, self.d_model))

    def test_compute_fvu_exact_identity(self):
        """Zero error reconstruction gives FVU of 0.0."""
        fvu = compute_fvu(self.X_sample, self.X_sample)
        self.assertAlmostEqual(fvu, 0.0, places=6)

    def test_compute_fvu_mean_predictor(self):
        """Reconstructing with mean gives FVU of 1.0."""
        mean_recon = np.tile(np.mean(self.X_sample, axis=0, keepdims=True), (self.batch_size, 1))
        fvu = compute_fvu(self.X_sample, mean_recon)
        self.assertAlmostEqual(fvu, 1.0, places=5)

    def test_baseline_batchtopk_forward_and_fvu(self):
        """BatchTopK baseline runs forward pass and achieves bounded FVU."""
        model = BatchTopKSAE(d_model=self.d_model, n_latents=self.n_latents, k=self.k, seed=42)
        out = model.forward_eval(self.X_sample)
        self.assertEqual(out["x_hat"].shape, self.X_sample.shape)
        self.assertEqual(out["z_native"].shape, (self.batch_size, self.n_latents))
        fvu = compute_fvu(self.X_sample, out["x_hat"])
        self.assertIsInstance(fvu, float)
        self.assertGreater(fvu, 0.0)

    def test_baseline_matryoshka_prefix_banks(self):
        """Matryoshka baseline defines nested banks and valid forward pass."""
        model = MatryoshkaSAE(d_model=self.d_model, n_latents=self.n_latents, k=self.k, seed=42)
        self.assertEqual(len(model.banks), 4)
        out = model.forward_eval(self.X_sample)
        self.assertEqual(out["x_hat"].shape, self.X_sample.shape)

    def test_profile_token_throughput(self):
        """Throughput profiler measures positive tokens/sec and finite 10M token runtime."""
        model = BatchTopKSAE(d_model=self.d_model, n_latents=self.n_latents, k=self.k, seed=42)
        stats = profile_token_throughput(model, self.X_sample, n_trials=3)
        self.assertGreater(stats["tokens_per_second"], 100.0)
        self.assertGreater(stats["latency_ms"], 0.0)
        self.assertGreater(stats["est_sec_for_10m_tokens"], 0.0)
        self.assertGreater(stats["est_hours_for_10m_tokens"], 0.0)

    def test_verify_exit_gate_g1_passing(self):
        """Exit Gate G1 passes when FVU <= 0.16 and Matryoshka first-letter absorption is lower."""
        passing_metrics = {
            "BatchTopK": {"fvu": 0.1362, "first_letter_abs": 0.5661},
            "Matryoshka": {"fvu": 0.1318, "first_letter_abs": 0.2393},
        }
        res = verify_exit_gate_g1(passing_metrics)
        self.assertTrue(res["gate_passed"])
        self.assertTrue(res["fvu_within_expected_bound"])
        self.assertTrue(res["first_letter_hierarchy_verified"])

    def test_verify_exit_gate_g1_failing_fvu(self):
        """Exit Gate G1 fails when FVU exceeds expected bounds."""
        failing_metrics = {
            "BatchTopK": {"fvu": 0.35, "first_letter_abs": 0.5661},
            "Matryoshka": {"fvu": 0.32, "first_letter_abs": 0.2393},
        }
        res = verify_exit_gate_g1(failing_metrics)
        self.assertFalse(res["gate_passed"])
        self.assertFalse(res["fvu_within_expected_bound"])


if __name__ == "__main__":
    unittest.main()
