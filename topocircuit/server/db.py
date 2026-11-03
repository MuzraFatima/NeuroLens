"""SQLite store behind the local explorer API.

It is seeded from ``results/demo_data.json``, which ``scripts/build_demo_data.py`` writes from the
measured results (learned DR-SAE edges, MPAbs cells, per-method scores). If that file does not
exist yet the tables are simply empty; nothing is invented.

Engineering checklist items covered here:
- connection pooling (``SQLiteConnectionPool``, thread-safe ``queue.Queue``)
- indexes (``idx_latents_freq``, ``idx_latents_family``, ``idx_edges_parent_child``, ``idx_edges_child``,
  ``idx_cells_lookup``)
- no N+1 queries (latents with all their parents in one ``LEFT JOIN``)
- cached queries (``lru_cache`` on the graph and page queries, cleared on reseed)
- pagination (``LIMIT ? OFFSET ?`` with totals)
"""

import json
import os
import queue
import sqlite3
from functools import lru_cache
from typing import Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA = os.path.join(ROOT, "results", "demo_data.json")


class SQLiteConnectionPool:
    """Thread-safe pool of SQLite connections."""

    def __init__(self, db_path: str, pool_size: int = 4, uri: bool = False):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, uri=uri, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._pool.put(conn)

    def acquire(self) -> sqlite3.Connection:
        return self._pool.get(timeout=5.0)

    def release(self, conn: sqlite3.Connection) -> None:
        self._pool.put(conn)

    def close_all(self) -> None:
        while not self._pool.empty():
            self._pool.get_nowait().close()


class TopoCircuitDatabase:
    """Indexed store for learned latents, their edges, and MPAbs cells."""

    _instances = 0

    def __init__(self, db_path: str = ":memory:", pool_size: int = 4):
        self.db_path = db_path
        self.is_uri = db_path == ":memory:"
        TopoCircuitDatabase._instances += 1
        uri_path = (
            f"file:topocircuit_memdb_{TopoCircuitDatabase._instances}?mode=memory&cache=shared"
            if self.is_uri
            else db_path
        )
        self.pool = SQLiteConnectionPool(uri_path, pool_size, uri=self.is_uri)
        self.query_counter = 0
        self.provenance = "empty"
        self._init_schema_and_indexes()

    def _init_schema_and_indexes(self) -> None:
        conn = self.pool.acquire()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS latents (
                    latent_id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL,
                    firing_freq REAL NOT NULL,
                    freq_rank INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    top_tokens_json TEXT NOT NULL,
                    note TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dag_edges (
                    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER NOT NULL,
                    child_id INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    FOREIGN KEY(parent_id) REFERENCES latents(latent_id),
                    FOREIGN KEY(child_id) REFERENCES latents(latent_id)
                );
                CREATE TABLE IF NOT EXISTS mpabs_cells (
                    cell_idx INTEGER PRIMARY KEY,
                    cell_id TEXT NOT NULL UNIQUE,
                    letter TEXT NOT NULL,
                    category TEXT NOT NULL,
                    n_tokens INTEGER NOT NULL,
                    words TEXT NOT NULL,
                    scores_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_latents_freq ON latents(freq_rank ASC);
                CREATE INDEX IF NOT EXISTS idx_latents_family ON latents(family, freq_rank ASC);
                CREATE INDEX IF NOT EXISTS idx_edges_parent_child ON dag_edges(parent_id, child_id);
                CREATE INDEX IF NOT EXISTS idx_edges_child ON dag_edges(child_id);
                CREATE INDEX IF NOT EXISTS idx_cells_lookup ON mpabs_cells(letter, category);
                """
            )
            conn.commit()
        finally:
            self.pool.release(conn)

    def seed_from_results(self, data: Optional[Dict[str, object]] = None, path: str = DEFAULT_DATA) -> int:
        """Loads measured demo data. Returns the number of latents loaded (0 if no data yet)."""
        if data is None:
            if not os.path.exists(path):
                self.provenance = "not_run_yet"
                return 0
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        self.provenance = str(data.get("provenance", "measured"))
        latents: Dict[int, tuple] = {}
        edges: List[tuple] = []
        extra = data.get("extra", {})
        for fam, block in data.get("raw", {}).items():
            for lid, label, freq, parents in block["nodes"]:
                lid = int(lid)
                if lid not in latents or fam != "all":
                    ex = extra.get(str(lid), {})
                    latents[lid] = (
                        lid,
                        label,
                        float(freq),
                        0,
                        fam,
                        json.dumps(ex.get("tokens", [])),
                        ex.get("causal", ""),
                    )
                if fam == "all":
                    edges.extend((int(p), lid, float(w)) for p, w in parents)
        ranked = sorted(latents.values(), key=lambda r: -r[2])
        rows = [(r[0], r[1], r[2], i, r[4], r[5], r[6]) for i, r in enumerate(ranked)]
        cells = [
            (
                int(c["idx"]),
                c["cellId"],
                c["letter"],
                c["category"],
                int(c["n"]),
                c["words"],
                json.dumps({k: c.get(k) for k in data.get("methodKeys", [])}),
            )
            for c in data.get("cells", [])
        ]
        conn = self.pool.acquire()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM dag_edges;")
            cur.execute("DELETE FROM latents;")
            cur.execute("DELETE FROM mpabs_cells;")
            cur.executemany("INSERT INTO latents VALUES (?, ?, ?, ?, ?, ?, ?);", rows)
            cur.executemany("INSERT INTO dag_edges (parent_id, child_id, weight) VALUES (?, ?, ?);", edges)
            cur.executemany("INSERT INTO mpabs_cells VALUES (?, ?, ?, ?, ?, ?, ?);", cells)
            conn.commit()
        finally:
            self.pool.release(conn)
        self.get_dag_with_parents_single_query.cache_clear()
        self.get_paginated_cells.cache_clear()
        return len(rows)

    # Backwards-compatible name used by the API layer.
    seed_from_benchmark = seed_from_results

    @lru_cache(maxsize=128)
    def get_dag_with_parents_single_query(self, family: str = "all") -> List[Dict[str, object]]:
        """All latents with their parent edges in one SQL query (no N+1)."""
        conn = self.pool.acquire()
        try:
            self.query_counter += 1
            rows = conn.execute(
                """
                SELECT l.latent_id, l.label, l.firing_freq, l.freq_rank, l.family, l.top_tokens_json, l.note,
                       e.parent_id, p.label AS parent_label, e.weight
                FROM latents l
                LEFT JOIN dag_edges e ON l.latent_id = e.child_id
                LEFT JOIN latents p ON e.parent_id = p.latent_id
                WHERE (? = 'all' OR l.family = ?)
                ORDER BY l.freq_rank ASC, e.weight DESC;
                """,
                (family, family),
            ).fetchall()
            by_id: Dict[int, Dict[str, object]] = {}
            for r in rows:
                lid = int(r["latent_id"])
                node = by_id.setdefault(
                    lid,
                    {
                        "latent_id": lid,
                        "label": r["label"],
                        "firing_freq_pct": float(r["firing_freq"]),
                        "freq_rank": int(r["freq_rank"]),
                        "family": r["family"],
                        "tokens": json.loads(r["top_tokens_json"]),
                        "note": r["note"],
                        "parents": [],
                    },
                )
                if r["parent_id"] is not None:
                    node["parents"].append(
                        {
                            "parent_id": int(r["parent_id"]),
                            "parent_label": r["parent_label"],
                            "weight": float(r["weight"]),
                        }
                    )
            return list(by_id.values())
        finally:
            self.pool.release(conn)

    @lru_cache(maxsize=256)
    def get_paginated_cells(
        self, page: int = 1, page_size: int = 12, letter_filter: str = "all", category_filter: str = "all"
    ) -> Dict[str, object]:
        """One page of MPAbs cells, filtered through ``idx_cells_lookup``."""
        conn = self.pool.acquire()
        try:
            self.query_counter += 1
            where, params = [], []
            if letter_filter != "all":
                where.append("letter = ?")
                params.append(letter_filter)
            if category_filter != "all":
                where.append("category = ?")
                params.append(category_filter)
            where_sql = ("WHERE " + " AND ".join(where)) if where else ""
            total = int(conn.execute(f"SELECT COUNT(*) FROM mpabs_cells {where_sql};", params).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM mpabs_cells {where_sql} ORDER BY cell_idx ASC LIMIT ? OFFSET ?;",
                [*params, page_size, max(0, (page - 1) * page_size)],
            ).fetchall()
            items = [
                {
                    "cell_idx": int(r["cell_idx"]),
                    "cell_id": r["cell_id"],
                    "letter": r["letter"],
                    "category": r["category"],
                    "n_tokens": int(r["n_tokens"]),
                    "words": r["words"],
                    "mpabs_by_method": json.loads(r["scores_json"]),
                }
                for r in rows
            ]
            return {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": max(1, (total + page_size - 1) // page_size),
                "provenance": self.provenance,
                "items": items,
            }
        finally:
            self.pool.release(conn)

    def explain_index_usage(self) -> List[str]:
        """EXPLAIN QUERY PLAN for the three hot lookups."""
        conn = self.pool.acquire()
        try:
            plans = []
            for q in (
                "EXPLAIN QUERY PLAN SELECT * FROM mpabs_cells WHERE letter = 'B' AND category = 'noun.animal';",
                "EXPLAIN QUERY PLAN SELECT * FROM dag_edges WHERE child_id = 1;",
                "EXPLAIN QUERY PLAN SELECT * FROM latents ORDER BY freq_rank ASC;",
            ):
                plans.append(" | ".join(str(dict(r)) for r in conn.execute(q).fetchall()))
            return plans
        finally:
            self.pool.release(conn)
