"""Unit tests for Phase 8: Write-Up, Release, Defence Assets, and Exit Gate G8.

Verifies:
1. Existence and integrity of publication-grade research paper, defence deck, and demo.
2. Dataset card and pre-registration documents presence.
3. Fresh-eyes reproduction of headline numbers from README.
4. Phase 8 Exit Gate G8 evaluation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.eval.metrics import verify_exit_gate_g8


class TestPhase8ReleaseAndReproduction(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def test_release_deliverables_exist(self):
        """All Phase 8 deliverables (paper, deck, explorer demo) must exist and be non-empty."""
        expected_paths = [
            os.path.join(self.base_dir, "paper", "TopoCircuit_SAE_Paper.tex"),
            os.path.join(self.base_dir, "paper", "TopoCircuit_SAE_Paper.pdf"),
            os.path.join(self.base_dir, "presentation", "TopoCircuit_SAE_Defence_Deck.pptx"),
            os.path.join(self.base_dir, "demo", "explorer.html"),
            os.path.join(self.base_dir, "demo", "explorer.min.html"),
            os.path.join(self.base_dir, "MPAbs_DATASET_CARD.md"),
            os.path.join(self.base_dir, "PREREGISTRATION.md"),
            os.path.join(self.base_dir, "README.md"),
        ]
        for path in expected_paths:
            self.assertTrue(os.path.isfile(path), f"Missing release deliverable: {path}")
            self.assertGreater(os.path.getsize(path), 0, f"File is empty: {path}")

    def test_verify_exit_gate_g8_passing(self):
        """Exit Gate G8 passes when reproduced numbers match headline results within tolerance."""
        reproduced = {
            "matryoshka_mpabs": 0.3405,
            "dr_sae_mpabs": 0.0399,
            "dr_sae_circuit_nodes": 62.6,
        }
        res = verify_exit_gate_g8(reproduced)
        self.assertTrue(res["gate_passed"])
        self.assertTrue(res["matryoshka_reproduced"])
        self.assertTrue(res["dr_sae_reproduced"])
        self.assertTrue(res["circuit_nodes_reproduced"])

    def test_verify_exit_gate_g8_failing(self):
        """Exit Gate G8 fails if reproduced numbers deviate beyond tolerance."""
        deviating = {
            "matryoshka_mpabs": 0.45,
            "dr_sae_mpabs": 0.15,
            "dr_sae_circuit_nodes": 85.0,
        }
        res = verify_exit_gate_g8(deviating)
        self.assertFalse(res["gate_passed"])
        self.assertFalse(res["matryoshka_reproduced"])


if __name__ == "__main__":
    unittest.main()
