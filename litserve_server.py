"""
LitServe adapter for MiniCPM-o 4.5 Full-Duplex Demo
Proxies frontend requests to the gateway service.

Usage:
    # Start gateway + workers first, then:
    python litserve_server.py --port 8000 --gateway-host localhost --gateway-port 10024

    # Deploy to Lightning Cloud:
    lightning deploy litserve_server.py --cloud
"""

import argparse
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("litserve_minicpmo")


def create_app(gateway_host: str = "localhost", gateway_port: int = 10024):
    gateway_ws_base = f"ws://{gateway_host}:{gateway_port}"
    gateway_http_base = f"http://{gateway_host}:{gateway_port}"

    http_client = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal http_client
        http_client = httpx.AsyncClient(base_url=gateway_http_base, timeout=30)
        yield
        await http_client.aclose()

    app = FastAPI(title="MiniCPM-o 4.5 LitServe (proxy)", lifespan=lifespan)
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # ── Health ───────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "minicpmo-litserve-proxy"}

    # ── Root ─────────────────────────────────────────────────────
    @app.get("/")
    async def root():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "MiniCPM-o 4.5 LitServe Proxy", "docs": "/docs"}

    # ── API info ─────────────────────────────────────────────────
    @app.get("/api/info")
    async def api_info():
        return {
            "model": "MiniCPM-o 4.5 (proxied)",
            "modes": ["chat", "half_duplex", "duplex"],
            "endpoints": {
                "chat": "/ws/chat",
                "half_duplex": "/ws/half_duplex/{session_id}",
                "duplex": "/ws/duplex/{session_id}",
            },
        }

    # ── App list for frontend nav ────────────────────────────────
    @app.get("/api/apps")
    async def get_enabled_apps():
        return {
            "apps": [
                {"app_id": "turnbased", "name": "Turn-based Chat", "route": "/turnbased"},
                {"app_id": "omni", "name": "Omni Full-Duplex", "route": "/omni"},
                {"app_id": "audio_duplex", "name": "Audio Full-Duplex", "route": "/audio_duplex"},
            ]
        }

    # ── Mode pages ───────────────────────────────────────────────
    def _serve_page(page_path: str, fallback_title: str):
        full = os.path.join(static_dir, page_path)
        if os.path.exists(full):
            return FileResponse(full)
        return HTMLResponse(f"<h1>{fallback_title}</h1><p>Page not found</p>")

    @app.get("/turnbased", response_class=HTMLResponse)
    async def turnbased():
        return _serve_page("turnbased.html", "Turn-based Chat")

    @app.get("/omni", response_class=HTMLResponse)
    async def omni():
        return _serve_page(os.path.join("omni", "omni.html"), "Omni Full-Duplex")

    @app.get("/half_duplex", response_class=HTMLResponse)
    async def half_duplex():
        return _serve_page(os.path.join("half-duplex", "half_duplex.html"), "Half-Duplex Audio")

    @app.get("/audio_duplex", response_class=HTMLResponse)
    async def audio_duplex():
        return _serve_page(os.path.join("audio-duplex", "audio_duplex.html"), "Audio Full-Duplex")

    @app.get("/admin", response_class=HTMLResponse)
    async def admin():
        return _serve_page("admin.html", "Admin")

    @app.get("/realtime", response_class=HTMLResponse)
    async def realtime():
        return _serve_page(os.path.join("realtime", "realtime.html"), "Realtime API")

    @app.get("/s/{session_id}", response_class=HTMLResponse)
    async def session_viewer(session_id: str):
        return _serve_page("session-viewer.html", "Session Viewer")

    # ── WebSocket proxy ──────────────────────────────────────────
    async def _proxy_ws(client_ws: WebSocket, target: str):
        await client_ws.accept()
        try:
            async with websockets.connect(target) as server_ws:
                async def fwd_client():
                    try:
                        while True:
                            data = await client_ws.receive_text()
                            await server_ws.send(data)
                    except WebSocketDisconnect:
                        pass

                async def fwd_server():
                    try:
                        while True:
                            data = await server_ws.recv()
                            await client_ws.send_text(data)
                    except websockets.ConnectionClosed:
                        pass

                await asyncio.gather(fwd_client(), fwd_server())
        except OSError as e:
            logger.error(f"WS proxy cannot connect to gateway at {target}: {e}")
            if client_ws.client_state.name != "DISCONNECTED":
                await client_ws.close(code=1011, reason="gateway unreachable")
        except Exception as e:
            logger.error(f"WS proxy error ({target}): {e}")
            if client_ws.client_state.name != "DISCONNECTED":
                await client_ws.close(code=1011, reason=str(e))

    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket):
        await _proxy_ws(ws, f"{gateway_ws_base}/ws/chat")

    @app.websocket("/ws/half_duplex/{session_id}")
    async def ws_half_duplex(ws: WebSocket, session_id: str):
        await _proxy_ws(ws, f"{gateway_ws_base}/ws/half_duplex/{session_id}")

    @app.websocket("/ws/duplex/{session_id}")
    async def ws_duplex(ws: WebSocket, session_id: str):
        await _proxy_ws(ws, f"{gateway_ws_base}/ws/duplex/{session_id}")

    @app.websocket("/v1/realtime")
    async def ws_realtime(ws: WebSocket):
        await _proxy_ws(ws, f"{gateway_ws_base}/v1/realtime")

    # ── HTTP proxy to gateway ────────────────────────────────────
    async def _proxy_http(request: Request, path: str) -> Response:
        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        try:
            resp = await http_client.request(
                method=request.method,
                url=path,
                content=body,
                headers=headers,
                params=request.query_params,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except httpx.HTTPError:
            return Response(
                content=json.dumps({"error": "gateway unreachable"}),
                status_code=502,
                media_type="application/json",
            )

    # Catch-all: proxy remaining /api/* and other gateway endpoints
    @app.api_route("/api/admin/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_admin_api(path: str, request: Request):
        return await _proxy_http(request, f"/api/admin/{path}")

    @app.api_route("/api/assets/{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy_assets_api(path: str, request: Request):
        return await _proxy_http(request, f"/api/assets/{path}")

    @app.api_route("/api/config/{path:path}", methods=["GET", "PUT"])
    async def proxy_config_api(path: str, request: Request):
        return await _proxy_http(request, f"/api/config/{path}")

    @app.api_route("/api/queue/{path:path}", methods=["GET", "DELETE"])
    async def proxy_queue_api(path: str, request: Request):
        return await _proxy_http(request, f"/api/queue/{path}")

    @app.api_route("/api/streaming/{path:path}", methods=["GET", "POST"])
    async def proxy_streaming_api(path: str, request: Request):
        return await _proxy_http(request, f"/api/streaming/{path}")

    @app.api_route("/api/sessions/{path:path}", methods=["GET", "DELETE"])
    async def proxy_sessions_api(path: str, request: Request):
        return await _proxy_http(request, f"/api/sessions/{path}")

    @app.api_route("/api/default_ref_audio", methods=["GET", "POST"])
    async def proxy_default_ref_audio(request: Request):
        return await _proxy_http(request, "/api/default_ref_audio")

    @app.get("/status")
    async def proxy_status(request: Request):
        return await _proxy_http(request, "/status")

    @app.get("/workers")
    async def proxy_workers(request: Request):
        return await _proxy_http(request, "/workers")

    @app.api_route("/sessions", methods=["GET", "DELETE"])
    async def proxy_sessions(request: Request):
        return await _proxy_http(request, "/sessions")

    @app.api_route("/sessions/{session_id}", methods=["GET", "DELETE"])
    async def proxy_session_by_id(session_id: str, request: Request):
        return await _proxy_http(request, f"/sessions/{session_id}")

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 LitServe Proxy Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--gateway-host", type=str, default="localhost", help="Gateway host")
    parser.add_argument("--gateway-port", type=int, default=10024, help="Gateway port")
    args = parser.parse_args()

    app = create_app(args.gateway_host, args.gateway_port)

    logger.info(
        f"Starting MiniCPM-o 4.5 LitServe proxy on "
        f"{args.host}:{args.port} → {args.gateway_host}:{args.gateway_port}"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
