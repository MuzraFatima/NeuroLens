"""Unit tests for topocircuit.utils helper functions."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.utils import (
    cosine_similarity_matrix,
    ensure_dir,
    format_percent,
    load_yaml_config,
    seed_everything,
    topk_mask,
    validate_preregistration_config,
)


class TestUtils(unittest.TestCase):
    def test_seed_everything_sets_seed(self):
        """Calling seed_everything twice with the same seed gives identical draws."""
        seed_everything(99)
        a = np.random.rand(10)
        seed_everything(99)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_cosine_similarity_matrix_unit_vectors(self):
        """Cosine of a column with itself must be 1.0."""
        A = np.eye(4)
        sim = cosine_similarity_matrix(A, A)
        np.testing.assert_allclose(sim, np.eye(4), atol=1e-12)

    def test_cosine_similarity_matrix_shape(self):
        """Output shape is (m, n) for (d, m) and (d, n) inputs."""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((8, 5))
        B = rng.standard_normal((8, 3))
        sim = cosine_similarity_matrix(A, B)
        self.assertEqual(sim.shape, (5, 3))

    def test_topk_mask_keeps_exactly_k(self):
        """topk_mask returns exactly k True entries per row."""
        x = np.array([[3.0, 1.0, 4.0, 1.0, 5.0], [9.0, 2.0, 6.0, 5.0, 3.0]])
        mask = topk_mask(x, k=2)
        self.assertTrue(np.all(mask.sum(axis=1) == 2))

    def test_topk_mask_selects_largest(self):
        """topk_mask selects the two largest entries per row."""
        x = np.array([[1.0, 5.0, 3.0, 7.0]])
        mask = topk_mask(x, k=2)
        # Indices 1 (5.0) and 3 (7.0) should be selected
        self.assertTrue(mask[0, 1])
        self.assertTrue(mask[0, 3])
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[0, 2])

    def test_format_percent(self):
        """format_percent converts a fraction to a percentage string."""
        self.assertEqual(format_percent(0.0671), "6.71%")
        self.assertEqual(format_percent(1.0), "100.00%")
        self.assertEqual(format_percent(0.0, decimals=0), "0%")

    def test_load_yaml_config_reads_file(self):
        """load_yaml_config parses a valid YAML file to a dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            fh.write("project:\n  name: test\n  version: 1\n")
            tmp_path = fh.name
        try:
            cfg = load_yaml_config(tmp_path)
            self.assertIsInstance(cfg, dict)
            self.assertEqual(cfg["project"]["name"], "test")
        finally:
            os.unlink(tmp_path)

    def test_load_yaml_config_missing_file_raises(self):
        """load_yaml_config raises FileNotFoundError for a missing path."""
        with self.assertRaises(FileNotFoundError):
            load_yaml_config("/nonexistent/path/config.yaml")

    def test_ensure_dir_creates_directory(self):
        """ensure_dir creates the directory and returns its absolute path."""
        with tempfile.TemporaryDirectory() as base:
            target = os.path.join(base, "sub", "nested")
            result = ensure_dir(target)
            self.assertTrue(os.path.isdir(result))
            self.assertEqual(result, os.path.abspath(target))

    def test_validate_preregistration_config_valid(self):
        """Validates that configs/preregistration_v1.yaml conforms strictly to schema."""
        cfg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "configs", "preregistration_v1.yaml")
        )
        cfg = load_yaml_config(cfg_path)
        self.assertTrue(validate_preregistration_config(cfg))
        self.assertEqual(cfg["project"]["owner"], "Muzra Fatima")

    def test_validate_preregistration_config_invalid_raises(self):
        """Validates that malformed config missing required fields raises KeyError."""
        bad_cfg = {"project": {"name": "Test"}}
        with self.assertRaises(KeyError):
            validate_preregistration_config(bad_cfg)


if __name__ == "__main__":
    unittest.main()
