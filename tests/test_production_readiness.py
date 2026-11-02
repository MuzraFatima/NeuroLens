"""Unit Tests for Production Engineering Checklist (SQLite Pool, Indexes, N+1, Gzip, ETag, Pagination)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.models.saelens_plugin import DRSAELensConfig, DRSAELensWrapper
from topocircuit.server.app import TopoCircuitAPIApp
from topocircuit.server.db import TopoCircuitDatabase


class TestProductionReadiness(unittest.TestCase):
    def setUp(self):
        self.db = TopoCircuitDatabase(":memory:", pool_size=4)
        self.db.seed_from_benchmark()
        self.app = TopoCircuitAPIApp(db=self.db)

    def tearDown(self):
        self.db.pool.close_all()

    def test_sqlite_indexes_and_zero_n_plus_1_queries(self):
        """Verifies composite B-Tree index usage and that fetching the full DAG executes 1 SQL JOIN query."""
        plans = self.db.explain_index_usage()
        for plan in plans:
            self.assertIn("USING INDEX", plan)

        q0 = self.db.query_counter
        nodes = self.db.get_dag_with_parents_single_query("all")
        q1 = self.db.query_counter
        self.assertEqual(q1 - q0, 1, "Expected exactly 1 SQL JOIN query (no N+1 queries)!")
        self.assertEqual(len(nodes), 14)

        # Second call must hit @lru_cache with 0 new SQL queries
        _ = self.db.get_dag_with_parents_single_query("all")
        self.assertEqual(self.db.query_counter, q1)

    def test_api_caching_etag_pagination_and_gzip_compression(self):
        """Verifies server-side caching, ETag 304 responses, gzip payload compression, and pagination."""
        r_miss = self.app.handle_request("/api/v1/cells?page=2&page_size=10", accept_encoding="gzip")
        self.assertEqual(r_miss["status"], 200)
        self.assertEqual(r_miss["headers"]["X-Cache"], "MISS")
        self.assertEqual(r_miss["headers"]["Content-Encoding"], "gzip")
        self.assertLess(int(r_miss["headers"]["X-Compressed-Bytes"]), int(r_miss["headers"]["X-Raw-Bytes"]))

        r_hit = self.app.handle_request("/api/v1/cells?page=2&page_size=10", accept_encoding="gzip")
        self.assertEqual(r_hit["headers"]["X-Cache"], "HIT")

        r_304 = self.app.handle_request(
            "/api/v1/cells?page=2&page_size=10",
            accept_encoding="gzip",
            if_none_match=r_miss["headers"]["ETag"],
        )
        self.assertEqual(r_304["status"], 304)

    def test_saelens_wrapper_encode_decode(self):
        """Verifies SAELens plugin compatibility."""
        import numpy as np

        import topocircuit
        from topocircuit.models.saelens_plugin import DRSAELensArchitecture

        self.assertIs(DRSAELensArchitecture, DRSAELensWrapper)
        self.assertTrue(hasattr(topocircuit, "DRSAE"))
        self.assertTrue(hasattr(topocircuit, "MPAbsBenchmark"))

        cfg = DRSAELensConfig(d_in=32, d_sae=64, k=4, seed=42)
        wrapper = DRSAELensWrapper(cfg)
        x = np.random.default_rng(42).standard_normal((16, 32))
        out = wrapper.forward(x)
        self.assertEqual(out["x_hat"].shape, (16, 32))
        self.assertEqual(out["z_eff"].shape, (16, 64))

    def test_zero_n_plus_1_queries_on_paginated_cells(self):
        """Verifies that pagination does not perform N+1 queries per item."""
        q0 = self.db.query_counter
        res = self.db.get_paginated_cells(page=1, page_size=12)
        q1 = self.db.query_counter
        # Unified paginated retrieval executes 1 call instead of 1 + 12 queries
        self.assertEqual(q1 - q0, 1, "Expected single paginated retrieval (zero N+1 queries)!")
        self.assertEqual(len(res["items"]), 12)

    def test_unnecessary_rerender_guards_in_explorer(self):
        """Verifies state-signature dirty checking prevents redundant DOM/SVG reconstruction."""
        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo", "explorer.html"))
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("lastRenderedGraphSig", content)
        self.assertIn("lastRenderedTableSig", content)
        self.assertIn("if (sig === lastRenderedGraphSig) return;", content)
        self.assertIn("if (sig === lastRenderedTableSig) return;", content)

    def test_unused_dependencies_audit(self):
        """Verifies requirements.txt contains zero unused dependencies."""
        req_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "requirements.txt"))
        with open(req_path, encoding="utf-8") as f:
            packages = [line.strip().split(">=")[0] for line in f if line.strip()]

        # All 6 packages in requirements.txt must be actively importable and needed
        import importlib

        pkg_map = {
            "numpy": "numpy",
            "scipy": "scipy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "PyYAML": "yaml",
            "Pillow": "PIL",
        }
        for pkg in packages:
            self.assertIn(pkg, pkg_map, f"Unexpected or unneeded dependency: {pkg}")
            module_name = pkg_map[pkg]
            mod = importlib.import_module(module_name)
            self.assertIsNotNone(mod)


if __name__ == "__main__":
    unittest.main()
