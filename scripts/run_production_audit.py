#!/usr/bin/env python3
"""Engineering checklist audit: measures what can be measured locally, writes nothing it did not measure.

Checks: single-JOIN graph query and its cache, index usage, server cache and ETag 304s, gzip saving,
PNG compression of figures and screenshots, a rebuilt minified explorer, and a placeholder scan.
Lighthouse is not run here; use the `lighthouse` CLI against `python scripts/serve_app.py` for that.
"""

import json
import os
import re
import sys

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.server.app import TopoCircuitAPIApp
from topocircuit.server.db import TopoCircuitDatabase


def compress_project_images(fig_dir: str) -> dict:
    """Optimizes and compresses all PNG figures in `results/figures/`."""
    total_before = 0
    total_after = 0
    files_processed = 0
    for fname in sorted(os.listdir(fig_dir)):
        if not fname.endswith(".png"):
            continue
        fpath = os.path.join(fig_dir, fname)
        sz_before = os.path.getsize(fpath)
        total_before += sz_before
        with Image.open(fpath) as img:
            quant = img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            quant.save(fpath, "PNG", optimize=True, compress_level=9)
        sz_after = os.path.getsize(fpath)
        total_after += sz_after
        files_processed += 1
    reduction_pct = (1.0 - (total_after / max(1, total_before))) * 100.0
    return {
        "files_processed": files_processed,
        "bytes_before": total_before,
        "bytes_after": total_after,
        "reduction_pct": round(reduction_pct, 2),
    }


def minify_html_bundle(src_html_path: str, out_min_path: str) -> dict:
    """Minifies HTML, inline CSS, and inline JS into `explorer.min.html`."""
    raw = open(src_html_path, encoding="utf-8").read()
    sz_before = len(raw.encode("utf-8"))

    minified = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    minified = re.sub(r"/\*.*?\*/", "", minified, flags=re.DOTALL)
    minified = re.sub(r"\n\s+", "\n", minified)
    minified = re.sub(r"\n+", "\n", minified).strip()

    with open(out_min_path, "w", encoding="utf-8") as f:
        f.write(minified)
    sz_after = len(minified.encode("utf-8"))
    return {
        "bytes_before": sz_before,
        "bytes_after": sz_after,
        "reduction_pct": round((1.0 - sz_after / max(1, sz_before)) * 100.0, 2),
    }


def run_dev_guard_placeholder_scan(root_dir: str) -> dict:
    """Rule 10.7 Dev Guard: Scans source files for forbidden placeholder strings."""
    forbidden_pattern = re.compile(r"\b(lorem\s+ipsum|TODO:|FIXME:|FAKE_DATA)\b", re.IGNORECASE)
    violations = []
    scanned_files = 0
    for subdir in ["topocircuit", "demo"]:
        target = os.path.join(root_dir, subdir)
        if not os.path.isdir(target):
            continue
        for r, _, files in os.walk(target):
            for fn in files:
                if fn.endswith((".py", ".html", ".css", ".js", ".ts")):
                    scanned_files += 1
                    p = os.path.join(r, fn)
                    txt = open(p, encoding="utf-8").read()
                    for m in forbidden_pattern.finditer(txt):
                        violations.append(f"{fn}: {m.group(0)}")
    return {"scanned_files": scanned_files, "violations": violations, "passed": len(violations) == 0}


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fig_dir = os.path.join(base_dir, "results", "figures")
    demo_html = os.path.join(base_dir, "demo", "explorer.html")
    demo_min_html = os.path.join(base_dir, "demo", "explorer.min.html")

    db = TopoCircuitDatabase(":memory:", pool_size=4)
    if db.seed_from_results() == 0:
        raise SystemExit("results/demo_data.json is missing: run scripts/build_demo_data.py first")
    q_before = db.query_counter
    _ = db.get_dag_with_parents_single_query("all")
    q_after_miss = db.query_counter
    _ = db.get_dag_with_parents_single_query("all")
    q_after_hit = db.query_counter
    index_plans = db.explain_index_usage()

    assert q_after_miss - q_before == 1, "N+1 violation!"
    assert q_after_hit == q_after_miss, "LRU cache violation!"
    assert all("USING INDEX" in p for p in index_plans)

    app = TopoCircuitAPIApp(db=db)
    r1 = app.handle_request("/api/v1/cells?page=1&page_size=12", accept_encoding="gzip")
    r2 = app.handle_request("/api/v1/cells?page=1&page_size=12", accept_encoding="gzip")
    r304 = app.handle_request(
        "/api/v1/cells?page=1&page_size=12", accept_encoding="gzip", if_none_match=r1["headers"]["ETag"]
    )

    raw_len = int(r1["headers"]["X-Raw-Bytes"])
    gz_len = int(r1["headers"]["X-Compressed-Bytes"])
    gz_ratio = round((1.0 - gz_len / max(1, raw_len)) * 100.0, 2)

    img_stats = compress_project_images(fig_dir)
    shot_stats = compress_project_images(os.path.join(base_dir, "demo", "screenshots"))
    min_stats = minify_html_bundle(demo_html, demo_min_html)
    dev_guard = run_dev_guard_placeholder_scan(base_dir)
    assert dev_guard["passed"], f"Dev guard failed: {dev_guard['violations']}"

    audit_report = {
        "database_and_queries": {
            "connection_pool_size": db.pool.pool_size,
            "single_join_queries_executed": q_after_miss - q_before,
            "n_plus_1_eliminated": True,
            "lru_cache_hit_verified": q_after_hit == q_after_miss,
            "sqlite_index_plans": index_plans,
        },
        "api_and_compression": {
            "server_cache_hit_header": r2["headers"]["X-Cache"],
            "etag_304_verified": r304["status"] == 304,
            "raw_payload_bytes": raw_len,
            "gzip_payload_bytes": gz_len,
            "gzip_compression_saving_pct": gz_ratio,
        },
        "asset_optimization": {
            "figure_compression": img_stats,
            "screenshot_compression": shot_stats,
            "html_css_js_minification": min_stats,
        },
        "dev_guard_honesty_scan": dev_guard,
        "lighthouse": "not measured by this script",
    }

    out_json = os.path.join(base_dir, "results", "production_audit_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(json.dumps(audit_report, indent=2))
    db.pool.close_all()


if __name__ == "__main__":
    main()
