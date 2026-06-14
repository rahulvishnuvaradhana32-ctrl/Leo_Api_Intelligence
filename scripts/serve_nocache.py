#!/usr/bin/env python3
"""Tiny no-cache static server for local preview of web/ (stdlib only).

The deployed app (web_server.py) already sends no-cache headers; this gives
the same behaviour locally without needing fastapi/uvicorn, so the browser
never serves a stale page during review.

    python scripts/serve_nocache.py [port]
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, *args):  # quiet
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8012
    handler = partial(NoCacheHandler, directory=str(WEB))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"LEO preview (no-cache) at http://localhost:{port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
