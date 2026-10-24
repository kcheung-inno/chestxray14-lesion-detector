"""Serve the trained-model demo over HTTP.

Why a server rather than opening the file: onnxruntime-web cannot fetch a 122 MB
model from a file:// origin (opaque origin blocks the request), and WASM
multithreading needs cross-origin isolation headers. Same-origin HTTP removes both.

Run:  python scripts/serve_demo.py       (then open http://127.0.0.1:8777/ )
"""
import os, sys
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT = int(os.environ.get("NODULE_DEMO_PORT", "8777"))
NO_ISOLATION = os.environ.get("NODULE_DEMO_NO_ISOLATION", "") not in ("", "0", "false")
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo")
DEMO_DIR = os.path.abspath(DEMO_DIR)

MIME = {
    ".onnx": "application/octet-stream",
    ".wasm": "application/wasm",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".json": "application/json",
    ".png": "image/png",
    ".html": "text/html; charset=utf-8",
}


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable SharedArrayBuffer -> threaded WASM in onnxruntime-web.
        # Set NODULE_DEMO_NO_ISOLATION=1 to omit these, which is what a host without
        # custom-header support (e.g. GitHub Pages) would serve. Used to test whether
        # the demo degrades gracefully instead of silently breaking.
        if not NO_ISOLATION:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        return MIME.get(ext, super().guess_type(path))

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            body = b'{"service":"nodule-demo","port":%d}' % PORT
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    os.chdir(DEMO_DIR)
    print(f"serving {DEMO_DIR}")
    print(f"open  http://127.0.0.1:{PORT}/")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()