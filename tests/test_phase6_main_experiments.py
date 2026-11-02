"""Unit tests for Phase 6: Main Experiments, Headline Benchmarks, and Exit Gate G6.

Verifies:
1. Headline benchmark results table integrity across 6 methods + matched effective L0.
2. 9-variant ablation study table integrity.
3. Master JSON results payload including Wilcoxon signed-rank and bootstrap tests.
4. Primary Hypothesis H2 and Phase 6 Exit Gate G6 evaluation.
"""

import json
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.eval.metrics import verify_exit_gate_g6


class TestPhase6MainExperiments(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.main_csv = os.path.join(base_dir, "results", "tables", "main_benchmark_results.csv")
        self.ablation_csv = os.path.join(base_dir, "results", "tables", "ablation_results.csv")
        self.master_json = os.path.join(base_dir, "results", "tables", "master_results.json")

    def test_main_benchmark_table_integrity(self):
        """Headline table contains all 7 methods and valid metric bounds."""
        self.assertTrue(os.path.exists(self.main_csv))
        df = pd.read_csv(self.main_csv)
        self.assertEqual(len(df), 7)
        expected_methods = {
            "BatchTopK",
            "Matryoshka",
            "H-SAE (Tree)",
            "AG-SAE-lite",
            "Post-hoc DR",
            "DR-SAE (Matched Eff. L0)",
            "DR-SAE (Ours)",
        }
        self.assertEqual(set(df["Method"]), expected_methods)
        self.assertTrue(all(df["MPAbs_Mean_%"] >= 0.0))
        self.assertTrue(all(df["MPAbs_Mean_%"] <= 100.0))

    def test_ablation_table_integrity(self):
        """Ablation table contains all 9 component ablation variants."""
        self.assertTrue(os.path.exists(self.ablation_csv))
        df = pd.read_csv(self.ablation_csv)
        self.assertEqual(len(df), 9)
        self.assertIn("mean_mpabs", df.columns)
        self.assertIn("nodes_at_08", df.columns)

    def test_master_json_statistical_tests(self):
        """Master JSON payload contains RQ1-RQ4 data, paired Wilcoxon p-value, and Holm correction."""
        self.assertTrue(os.path.exists(self.master_json))
        with open(self.master_json, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("rq2_rq3_main_benchmarks", data)
        self.assertIn("stat_vs_matryoshka_effective_l0", data["rq2_rq3_main_benchmarks"])
        wilcoxon = data["rq2_rq3_main_benchmarks"]["stat_vs_matryoshka_effective_l0"]
        self.assertLess(wilcoxon["p_value_raw"], 0.001)
        self.assertIn("hypothesis_tests_holm_bonferroni", data)

    def test_verify_exit_gate_g6_passing(self):
        """Exit Gate G6 passes with headline DR-SAE vs Matryoshka metrics."""
        results = {
            "Matryoshka": {"mpabs_mean": 0.3405, "fvu": 0.1318},
            "DR-SAE (Ours)": {"mpabs_mean": 0.0399, "fvu": 0.1329},
            "DR-SAE (Matched Eff. L0)": {"mpabs_mean": 0.0671, "fvu": 0.1354},
        }
        res = verify_exit_gate_g6(results)
        self.assertTrue(res["gate_passed"])
        self.assertTrue(res["h2_native_l0_reduction_passed"])
        self.assertTrue(res["h2_effective_l0_reduction_passed"])
        self.assertTrue(res["fvu_fidelity_parity_passed"])


if __name__ == "__main__":
    unittest.main()
