import sys

APP_PATH = "/root/autodl-tmp/kralvoxCPM/app.py"
FRONTEND_PATH = "/root/autodl-tmp/kralvoxCPM/frontend.html"

with open(APP_PATH, "r") as f:
    content = f.read()

# Add import for FastAPI and StaticFiles at the top (after existing imports)
if "from starlette.middleware" not in content:
    # Add starlette imports after existing imports
    insert_after = "from voxcpm.model.utils import resolve_runtime_device"
    new_imports = """from voxcpm.model.utils import resolve_runtime_device
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse"""
    content = content.replace(insert_after, new_imports)

# Add middleware class before run_demo
middleware_class = '''

class FrontendMiddleware(BaseHTTPMiddleware):
    """Serve custom frontend HTML at the root path, while keeping Gradio API intact."""
    async def dispatch(self, request, call_next):
        path = request.url.path
        # Serve frontend at root or /new-ui
        if path == "/" or path == "/new-ui":
            with open("/root/autodl-tmp/kralvoxCPM/frontend.html", "r") as f:
                html = f.read()
            return StarletteResponse(content=html, media_type="text/html")
        # Let all other requests (API, uploads, assets) pass through
        return await call_next(request)

'''

if "class FrontendMiddleware" not in content:
    content = content.replace("def run_demo(", middleware_class + "def run_demo(")

# Add middleware to the Gradio app after launch
old_launch = '''    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )'''

new_launch = '''    app = interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
        prevent_thread_lock=True,
    )
    app.app.add_middleware(FrontendMiddleware)'''

if "prevent_thread_lock" not in content:
    content = content.replace(old_launch, new_launch)

with open(APP_PATH, "w") as f:
    f.write(content)

print("app.py patched successfully")
