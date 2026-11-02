"""Production HTTP Web and REST API Server.

Serves the TopoCircuit-SAE interactive explorer frontend alongside the SQLite-backed
/api/v1 REST endpoints with gzip payload compression, ETag caching, and CORS support.

Routes:
- GET /                     -> demo/explorer.html
- GET /demo/explorer.html   -> demo/explorer.html
- GET /api/v1/metrics       -> Master benchmark results and provenance
- GET /api/v1/dag           -> Single-JOIN composite indexed DAG graph
- GET /api/v1/cells         -> Paginated 112-cell MPAbs benchmark partition
- GET /api/v1/prop2         -> Cached Proposition 2 selective absorption simulator
- GET /api/v1/health        -> Connection pool, LRU cache stats, and SQLite index plans
- GET /results/figures/*    -> High-resolution benchmark figures
"""

import argparse
import http.server
import mimetypes
import os
import urllib.parse
import webbrowser

from topocircuit.server.app import TopoCircuitAPIApp
from topocircuit.server.db import TopoCircuitDatabase


class TopoCircuitRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Unified HTTP handler serving both static web assets and REST API endpoints."""

    api_app: TopoCircuitAPIApp = None
    repo_root: str = ""

    def __init__(self, *args, **kwargs):
        if not self.repo_root:
            self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        super().__init__(*args, directory=self.repo_root, **kwargs)

    def do_OPTIONS(self) -> None:
        """Handles CORS preflight requests."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """Routes REST API endpoints to TopoCircuitAPIApp and static files to directory server."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Route root or demo root to the primary explorer frontend
        if path in ("", "/", "/demo", "/demo/"):
            self.path = "/demo/explorer.html"
            return super().do_GET()

        # Handle favicon request gracefully without 404
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        # Route /api/v1/* requests through the REST API app
        if path.startswith("/api/v1/"):
            self._handle_api_request()
            return

        # Serve static assets with CORS enabled for font/module loading
        return super().do_GET()

    def end_headers(self) -> None:
        """Appends CORS headers to all static and API responses."""
        self._send_cors_headers()
        super().end_headers()

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, If-None-Match")

    def _handle_api_request(self) -> None:
        """Dispatches GET request to TopoCircuitAPIApp with ETag and Gzip negotiation."""
        accept_enc = self.headers.get("Accept-Encoding", "")
        if_none_match = self.headers.get("If-None-Match", "")

        response = self.api_app.handle_request(
            path_with_query=self.path,
            accept_encoding=accept_enc,
            if_none_match=if_none_match,
        )

        status = response.get("status", 200)
        self.send_response(status)

        headers = response.get("headers", {})
        for k, v in headers.items():
            self.send_header(k, str(v))
        self.end_headers()

        body = response.get("body", b"")
        if body and status != 304:
            self.wfile.write(body)

    def guess_type(self, path: str) -> str:
        """Overrides MIME type resolution to ensure .svg, .js, and .html have explicit UTF-8."""
        ctype, _ = mimetypes.guess_type(path)
        if ctype is None:
            return "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            return f"{ctype}; charset=utf-8"
        return ctype

    def log_message(self, format: str, *args) -> None:
        """Suppresses noisy per-request logging in non-debug mode."""
        if os.environ.get("TOPOCIRCUIT_DEBUG"):
            super().log_message(format, *args)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    db_pool_size: int = 4,
) -> http.server.ThreadingHTTPServer:
    """Instantiates and returns the configured threaded HTTP server."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db = TopoCircuitDatabase(":memory:", pool_size=db_pool_size)
    db.seed_from_benchmark()
    api_app = TopoCircuitAPIApp(db=db)

    TopoCircuitRequestHandler.api_app = api_app
    TopoCircuitRequestHandler.repo_root = repo_root

    server = http.server.ThreadingHTTPServer((host, port), TopoCircuitRequestHandler)
    return server


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
    browser_path: str = "",
) -> None:
    """Runs the HTTP server and optionally launches the external browser."""
    server = create_server(host=host, port=port)
    url = f"http://{host}:{port}/demo/explorer.html"
    print(f"TopoCircuit-SAE Server running at: {url}")
    print("Serving interactive explorer + REST API (Press Ctrl+C to stop)...")

    if open_browser:
        if browser_path and os.path.exists(browser_path):
            import subprocess

            subprocess.Popen([browser_path, url])
        else:
            webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoCircuit-SAE Interactive Web & REST API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Binding host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--open-browser", action="store_true", help="Launch external browser upon start")
    parser.add_argument("--browser-path", default="", help="Optional explicit path to external Chrome binary")
    args = parser.parse_args()

    serve(host=args.host, port=args.port, open_browser=args.open_browser, browser_path=args.browser_path)


if __name__ == "__main__":
    main()
