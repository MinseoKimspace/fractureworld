import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit


def serve(path, host="127.0.0.1", port=8765, open_browser=False):
    selected = Path(path).resolve(strict=True)
    content = selected.read_bytes()
    document = json.loads(content)
    if not isinstance(document, dict) or not isinstance(document.get("bodies"), list) or not document.get("frames"):
        raise ValueError("replay JSON must contain bodies and at least one frame")
    assets = files("fracture_world").joinpath("web")
    routes = {
        "/": ("text/html; charset=utf-8", assets.joinpath("index.html").read_bytes()),
        "/replay.js": ("text/javascript; charset=utf-8", assets.joinpath("replay.js").read_bytes()),
        "/data.json": ("application/json; charset=utf-8", content),
    }

    class ReplayHandler(BaseHTTPRequestHandler):
        def respond(self, include_body):
            route = routes.get(urlsplit(self.path).path)
            if route is None:
                self.send_error(404, "Unknown replay resource")
                return
            content_type, body = route
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def do_GET(self):
            self.respond(True)

        def do_HEAD(self):
            self.respond(False)

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer((host, port), ReplayHandler) as server:
        address = f"http://{host}:{server.server_address[1]}"
        print(f"Fracture World viewer: {address}", flush=True)
        if open_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
