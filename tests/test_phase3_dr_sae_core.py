"""Unit tests for Phase 3: DR-SAE Core, Lemma 1 Acyclicity, and Exit Gate G3.

Verifies:
1. Lemma 1 frequency ordering and strict DAG acyclicity by construction.
2. Nilpotent routing series expansion: (I - A)^(-1) z == sum_{h=0}^H A^h z.
3. Hub-degree protection mechanism against edge collapse.
4. Phase 3 Exit Gate G3 evaluation.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.eval.metrics import verify_exit_gate_g3
from topocircuit.models.dr_sae import DRSAE


class TestPhase3DRSAECore(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.n_latents = 32
        self.d_model = 32
        self.k = 4
        self.hop_depth = 4
        self.sae = DRSAE(
            d_model=self.d_model,
            n_latents=self.n_latents,
            k=self.k,
            hop_depth=self.hop_depth,
            seed=self.seed,
        )

    def test_lemma1_strict_acyclicity(self):
        """Frequency-ordered candidate masking guarantees strict DAG acyclicity."""
        rng = np.random.default_rng(self.seed)
        n_tokens = 800
        Z = (rng.random((n_tokens, self.n_latents)) < 0.10).astype(np.float64)
        for i in range(self.n_latents):
            Z[: (i + 1) * 15, i] = 1.0

        self.sae.update_frequency_order_and_candidates(Z)
        self.sae.A[self.sae.candidate_mask] = 0.20
        self.assertTrue(self.sae.is_acyclic())

    def test_nilpotent_series_identity(self):
        """z_eff from series sum matches exact matrix inverse (I - A)^(-1) z."""
        A = np.zeros((self.n_latents, self.n_latents), dtype=np.float64)
        A[2, 5] = 0.35
        A[5, 11] = 0.40
        A[11, 20] = 0.25

        rng = np.random.default_rng(self.seed)
        z = rng.uniform(0.0, 2.0, size=(10, self.n_latents))

        z_eff_series = self.sae.route_ancestors(z, A_mat=A)

        eye_mat = np.eye(self.n_latents)
        inv_I_minus_A = np.linalg.inv(eye_mat - A)
        z_eff_exact = z @ inv_I_minus_A.T

        np.testing.assert_allclose(z_eff_series, z_eff_exact, atol=1e-10)

    def test_hub_cap_enforcement(self):
        """Latents exceeding maximum child fraction must have lower-weight edges pruned."""
        # Create a hub parent 0 with 25 children
        self.sae.A[0, 1:26] = 0.50
        self.sae._enforce_hub_cap(max_children_frac=0.25)
        out_degree = int(np.sum(self.sae.A[0, :] > 0))
        self.assertLessEqual(out_degree, 10)

    def test_verify_exit_gate_g3_passing(self):
        """Exit Gate G3 passes when parent-set F1 >= 0.80 and no hub latent exceeds threshold."""
        eval_metrics = {"parent_set_f1": 0.84}
        learned_parents = {i: [0] for i in range(1, 5)}
        res = verify_exit_gate_g3(
            metrics_or_dag_eval=eval_metrics,
            learned_parents=learned_parents,
            n_features=32,
            min_parent_set_f1=0.80,
            max_hub_frac=0.25,
        )
        self.assertTrue(res["gate_passed"])
        self.assertTrue(res["parent_set_f1_sufficient"])
        self.assertTrue(res["no_hub_collapse"])

    def test_verify_exit_gate_g3_failing_f1(self):
        """Exit Gate G3 fails if parent-set F1 < 0.80."""
        eval_metrics = {"parent_set_f1": 0.65}
        learned_parents = {i: [0] for i in range(1, 5)}
        res = verify_exit_gate_g3(
            metrics_or_dag_eval=eval_metrics,
            learned_parents=learned_parents,
            n_features=32,
            min_parent_set_f1=0.80,
            max_hub_frac=0.25,
        )
        self.assertFalse(res["gate_passed"])
        self.assertFalse(res["parent_set_f1_sufficient"])


if __name__ == "__main__":
    unittest.main()
