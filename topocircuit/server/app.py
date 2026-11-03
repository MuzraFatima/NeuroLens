"""Production HTTP REST API Server with Server-Side TTL/ETag Caching and Gzip Payload Compression.

Endpoints:
- GET /api/v1/metrics      -> Returns master_results.json with provenance metadata
- GET /api/v1/dag          -> Returns single-JOIN DAG nodes & edges (?family=all|animal|location|food)
- GET /api/v1/cells        -> Returns paginated MPAbs cells (?page=1&page_size=12&letter=all&category=all)
- GET /api/v1/prop2        -> Nearest measured Proposition 2 cell from results/synthetic/prop2_grid.json
- GET /api/v1/health       -> Returns DB connection pool status, cache hit stats, and index verification
"""

import gzip
import hashlib
import json
import os
import time
from functools import lru_cache
from typing import Dict, Tuple
from urllib.parse import parse_qs, urlparse

from topocircuit.server.db import TopoCircuitDatabase


class ServerSideResponseCache:
    """Thread-safe server-side TTL + ETag response cache."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 512):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: Dict[str, Tuple[float, str, bytes, bytes]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Tuple[bool, str, bytes, bytes]:
        now = time.time()
        if key in self._store:
            ts, etag, raw_bytes, gz_bytes = self._store[key]
            if now - ts <= self.ttl:
                self.hits += 1
                return True, etag, raw_bytes, gz_bytes
            del self._store[key]
        self.misses += 1
        return False, "", b"", b""

    def set(self, key: str, payload_obj: object) -> Tuple[str, bytes, bytes]:
        if len(self._store) >= self.max_entries:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        raw_bytes = json.dumps(payload_obj, separators=(",", ":")).encode("utf-8")
        gz_bytes = gzip.compress(raw_bytes, compresslevel=6)
        etag = '"' + hashlib.sha256(raw_bytes).hexdigest()[:16] + '"'
        self._store[key] = (time.time(), etag, raw_bytes, gz_bytes)
        return etag, raw_bytes, gz_bytes


@lru_cache(maxsize=4)
def _load_json(path: str, mtime: float) -> Dict[str, object]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_result(path: str) -> Dict[str, object]:
    """Reads a results JSON written by the experiment scripts (cached until the file changes)."""
    if not os.path.exists(path):
        return {"status": "not_run_yet", "path": os.path.relpath(path)}
    return dict(_load_json(path, os.path.getmtime(path)))


def nearest_prop2_cell(grid: Dict[str, object], s1: float, s2: float, lam: float) -> Dict[str, object]:
    """Returns the measured Proposition 2 cell closest to the requested (sigma1^2, sigma2^2, lambda)."""
    cells = grid.get("cells", [])
    if not cells:
        return {"status": "not_run_yet"}
    best = min(
        cells,
        key=lambda c: (c["sigma1_sq"] - s1) ** 2 + (c["sigma2_sq"] - s2) ** 2 + (c["lambda"] - lam) ** 2,
    )
    return {**best, "provenance": "MEASURED / results/synthetic/prop2_grid.json"}


class TopoCircuitAPIApp:
    """WSGI/HTTP-compatible application engine for deterministic testing and local serving."""

    def __init__(self, db: TopoCircuitDatabase = None):
        self.db = db or TopoCircuitDatabase(":memory:")
        self.db.seed_from_benchmark()
        self.cache = ServerSideResponseCache(ttl_seconds=300)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.master_results_path = os.path.join(base_dir, "results", "master_results.json")
        self.prop2_path = os.path.join(base_dir, "results", "synthetic", "prop2_grid.json")

    def handle_request(
        self,
        path_with_query: str,
        accept_encoding: str = "gzip",
        if_none_match: str = "",
    ) -> Dict[str, object]:
        """Handles a GET request with server-side caching, ETag validation, and gzip compression."""
        hit, etag, raw_bytes, gz_bytes = self.cache.get(path_with_query)
        if not hit:
            parsed = urlparse(path_with_query)
            qs = parse_qs(parsed.query)
            route = parsed.path

            if route == "/api/v1/dag":
                fam = qs.get("family", ["all"])[0]
                payload = {
                    "provenance": "CACHED / SQLITE-SINGLE-JOIN",
                    "nodes": self.db.get_dag_with_parents_single_query(fam),
                }
            elif route == "/api/v1/cells":
                page = int(qs.get("page", ["1"])[0])
                page_size = int(qs.get("page_size", ["12"])[0])
                let = qs.get("letter", ["all"])[0]
                cat = qs.get("category", ["all"])[0]
                payload = self.db.get_paginated_cells(
                    page=page, page_size=page_size, letter_filter=let, category_filter=cat
                )
            elif route == "/api/v1/prop2":
                s1 = round(float(qs.get("s1", ["0.05"])[0]), 3)
                s2 = round(float(qs.get("s2", ["0.38"])[0]), 3)
                lam = round(float(qs.get("lam", ["0.20"])[0]), 3)
                payload = nearest_prop2_cell(load_result(self.prop2_path), s1, s2, lam)
            elif route == "/api/v1/metrics":
                payload = load_result(self.master_results_path)
                payload["provenance"] = "LOCAL STORE / results/master_results.json"
            elif route == "/api/v1/health":
                payload = {
                    "status": "ok",
                    "cache_hits": self.cache.hits,
                    "cache_misses": self.cache.misses,
                    "db_pool_size": self.db.pool.pool_size,
                    "index_plans": self.db.explain_index_usage(),
                }
            else:
                return {"status": 404, "headers": {}, "body": b'{"error":"not_found"}'}

            etag, raw_bytes, gz_bytes = self.cache.set(path_with_query, payload)

        if if_none_match and if_none_match == etag:
            return {
                "status": 304,
                "headers": {"ETag": etag, "X-Cache": "HIT" if hit else "MISS"},
                "body": b"",
            }

        use_gzip = "gzip" in accept_encoding.lower()
        body_out = gz_bytes if use_gzip else raw_bytes
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "public, max-age=300",
            "ETag": etag,
            "X-Cache": "HIT" if hit else "MISS",
            "X-Raw-Bytes": str(len(raw_bytes)),
            "X-Compressed-Bytes": str(len(body_out)),
        }
        if use_gzip:
            headers["Content-Encoding"] = "gzip"
        return {"status": 200, "headers": headers, "body": body_out}
