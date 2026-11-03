# Engineering checklist audit

Status of the 20-point performance checklist for the local explorer and API, as of the final
release. Measured values come from `scripts/run_production_audit.py`
(`results/production_audit_report.json`). The project is not deployed anywhere, so items that only
make sense for a hosted service are marked **skipped**.

| # | Item | Status | Evidence |
|:-:|---|---|---|
| 1 | Cache API responses | Done | `Cache-Control: public, max-age=300` and ETag on every API response; `If-None-Match` returns 304 (`topocircuit/server/app.py`) |
| 2 | Load balancer | Skipped | Single local process; nothing to balance |
| 3 | Index the database | Done | 5 indexes; `EXPLAIN QUERY PLAN` shows index use for cell lookup, edge lookup and frequency ordering |
| 4 | Compress images | Done | PNG quantisation: figures −62%, screenshots −73% |
| 5 | Loading skeletons | Not needed | All explorer data ships in one static `demo/data/results.js`; there is no loading state to fill |
| 6 | Cache expensive queries | Done | `lru_cache` on the graph and page queries; a repeat call runs 0 SQL queries; cache cleared on reseed |
| 7 | N+1 database queries | Done | Latents and all their parents in one `LEFT JOIN` (asserted in `tests/test_production_readiness.py`) |
| 8 | Debounce input handlers | Done | Search box (180 ms) and both hop-depth sliders (120 ms) in `demo/explorer.html` |
| 9 | Split code into chunks | Partly | Data is split from code (`results.js`); the page itself is one ~66 KB file, small enough that further splitting was not worth it |
| 10 | Add CDN | Done | Three.js r128 from cdnjs, fonts from Google Fonts |
| 11 | Server-side caching | Done | TTL response cache with `X-Cache: HIT/MISS` |
| 12 | Paginate large lists | Done | `LIMIT/OFFSET` with totals in the API; 10 cells per page with filters in the explorer |
| 13 | Lighthouse audit | Not run | Run `lighthouse http://127.0.0.1:8000/demo/explorer.html` against `python scripts/serve_app.py` |
| 14 | Compress API payloads | Done | gzip, 80% smaller on `/api/v1/cells` |
| 15 | Unnecessary re-renders | Done | State-signature guards skip unchanged graph and table renders |
| 16 | Minify JS and CSS | Done | `demo/explorer.min.html` rebuilt from the current page (−9%, comment and whitespace removal) |
| 17 | Add lazy loading | Partly | The 3D render loop pauses when the tab is hidden; the scene itself is built on load |
| 18 | Defer non-critical scripts | Not done | The inline app script needs Three.js at parse time; deferring it would require restructuring the page |
| 19 | Unused dependencies | Done | `pandas` removed after the old pipeline was deleted; every package in `requirements.txt` is imported |
| 20 | Database connection pooling | Done | `SQLiteConnectionPool`, thread-safe queue of connections |

An earlier version of this file reported Lighthouse scores and a client-side cache that were not
present in the page it described; it has been replaced.
