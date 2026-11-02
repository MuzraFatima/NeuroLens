"""Unit tests for Phase 5: Pilot, Pre-Registration, Minimum Detectable Effect (MDE), and Exit Gate G5.

Verifies:
1. Minimum Detectable Effect (MDE) statistical calculation.
2. Pre-registration hypotheses thresholds exceed MDEs.
3. Config schema and frozen parameters alignment.
4. Phase 5 Exit Gate G5 evaluation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.eval.metrics import (
    compute_minimum_detectable_effect,
    verify_exit_gate_g5,
)
from topocircuit.utils import load_yaml_config


class TestPhase5PreRegistration(unittest.TestCase):
    def setUp(self):
        cfg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "configs", "preregistration_v1.yaml")
        )
        self.cfg = load_yaml_config(cfg_path)

    def test_compute_minimum_detectable_effect(self):
        """MDE calculation returns positive value that shrinks with sample size sqrt(N)."""
        spread = [0.03, 0.04, 0.035, 0.045, 0.038]
        mde_100 = compute_minimum_detectable_effect(spread, n_samples=100)
        mde_400 = compute_minimum_detectable_effect(spread, n_samples=400)
        self.assertGreater(mde_100, 0.0)
        self.assertLess(mde_400, mde_100)
        # 4x samples should roughly halve the MDE
        self.assertAlmostEqual(mde_100 / mde_400, 2.0, delta=0.1)

    def test_hypotheses_thresholds_exceed_mdes(self):
        """All frozen hypotheses thresholds (H1-H4) must exceed their pilot MDE bounds."""
        mde_estimates = {
            "H2_mpabs_reduction": 0.042,
            "H3_edge_reproducibility": 0.051,
        }
        res = verify_exit_gate_g5(self.cfg["hypotheses"], mde_estimates)
        self.assertTrue(res["gate_passed"])
        self.assertTrue(res["h2_threshold_exceeds_mde"])
        self.assertTrue(res["h3_threshold_exceeds_mde"])

    def test_frozen_preregistration_metadata(self):
        """Pre-registration metadata contains frozen tag and owner."""
        project = self.cfg["project"]
        self.assertEqual(project["preregistration_tag"], "v1.0-prereg-phase5")
        self.assertEqual(project["owner"], "Muzra Fatima")


if __name__ == "__main__":
    unittest.main()
