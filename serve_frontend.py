import http.server
import socketserver
import os
import urllib.request

PORT = int(os.environ.get("FRONTEND_PORT", "6008"))
GRADIO_PORT = 8808
DIRECTORY = "/root/autodl-tmp/kralvoxCPM"

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        if self.path.startswith("/gradio-assets/"):
            asset_path = "/assets/" + self.path[len("/gradio-assets/"):].lstrip("/")
            url = f"http://localhost:{GRADIO_PORT}{asset_path}"
            req = urllib.request.Request(url)
            for key, val in self.headers.items():
                if key.lower() not in ("host", "connection"):
                    req.add_header(key, val)
            try:
                with urllib.request.urlopen(req) as resp:
                    self.send_response(resp.status)
                    for key, val in resp.getheaders():
                        if key.lower() not in ("transfer-encoding", "connection"):
                            self.send_header(key, val)
                    self.end_headers()
                    self.wfile.write(resp.read())
            except Exception as e:
                self.send_error(502, f"Asset proxy error: {e}")
        elif self.path.startswith("/gradio_api/") or self.path.startswith("/upload") or self.path.startswith("/call/"):
            # Proxy to Gradio backend
            url = f"http://localhost:{GRADIO_PORT}{self.path}"
            req = urllib.request.Request(url)
            for key, val in self.headers.items():
                if key.lower() not in ("host", "connection"):
                    req.add_header(key, val)
            try:
                with urllib.request.urlopen(req) as resp:
                    self.send_response(resp.status)
                    for key, val in resp.getheaders():
                        if key.lower() not in ("transfer-encoding", "connection"):
                            self.send_header(key, val)
                    self.end_headers()
                    self.wfile.write(resp.read())
            except Exception as e:
                self.send_error(502, f"Proxy error: {e}")
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/frontend.html"
            super().do_GET()
        else:
            super().do_GET()
    
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        
        url = f"http://localhost:{GRADIO_PORT}{self.path}"
        req = urllib.request.Request(url, data=body, method="POST")
        for key, val in self.headers.items():
            if key.lower() not in ("host", "connection", "content-length"):
                req.add_header(key, val)
        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, val)
                self.end_headers()
                # Stream the response (important for SSE)
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except Exception as e:
            self.send_error(502, f"Proxy error: {e}")

class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

with ReusableThreadingTCPServer(("", PORT), ProxyHandler) as httpd:
    print(f"Frontend server running at http://0.0.0.0:{PORT}")
    print(f"Proxying Gradio API from http://localhost:{GRADIO_PORT}")
    httpd.serve_forever()
