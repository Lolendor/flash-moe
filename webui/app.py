"""Flash-MoE WebUI — Litestar backend serving Vue 3 SPA."""

import json
from pathlib import Path
from typing import AsyncGenerator

import asyncio

import httpx
from litestar import Litestar, Request, get, post, MediaType
from litestar.response import Response, Stream
from litestar.static_files import StaticFilesConfig
from litestar.config.cors import CORSConfig

STATIC_DIR = Path(__file__).parent / "static"


@get("/", media_type=MediaType.HTML)
async def index() -> Response:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return Response(
        content=html,
        media_type=MediaType.HTML,
        headers={"Content-Type": "text/html; charset=utf-8"},
    )


@post("/api/chat")
async def chat_proxy(request: Request) -> Response:
    """Proxy non-streaming chat completions to the upstream /v1 API."""
    body = await request.json()
    base_url = body.pop("base_url", "http://localhost:8080")
    auth_header = body.pop("auth_header", "")

    body["stream"] = False

    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header

    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(url, json=body, headers=headers)
            return Response(
                content=resp.text,
                status_code=resp.status_code,
                media_type=MediaType.JSON,
            )
        except httpx.ConnectError:
            return Response(
                content=json.dumps({"error": {"message": f"Cannot connect to {url}", "type": "connection_error"}}),
                status_code=502,
                media_type=MediaType.JSON,
            )
        except Exception as e:
            return Response(
                content=json.dumps({"error": {"message": str(e), "type": "proxy_error"}}),
                status_code=500,
                media_type=MediaType.JSON,
            )


@post("/api/chat/stream")
async def chat_stream_proxy(request: Request) -> Stream:
    """Proxy streaming chat completions, forwarding SSE events.

    When the client disconnects (abort button), uvicorn cancels the async
    generator. We catch the cancellation in a finally block and close the
    upstream httpx connection so the inference server stops generating.
    """
    body = await request.json()
    base_url = body.pop("base_url", "http://localhost:8080")
    auth_header = body.pop("auth_header", "")

    body["stream"] = True

    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header

    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    async def event_generator() -> AsyncGenerator[bytes, None]:
        client = httpx.AsyncClient(timeout=300.0)
        upstream_resp = None
        try:
            upstream_resp = await client.send(
                client.build_request("POST", url, json=body, headers=headers),
                stream=True,
            )

            if upstream_resp.status_code != 200:
                error_bytes = b""
                async for chunk in upstream_resp.aiter_bytes():
                    error_bytes += chunk
                error_text = error_bytes.decode("utf-8", errors="replace")
                yield f"data: {json.dumps({'error': {'message': error_text, 'type': 'upstream_error'}})}\n\n".encode("utf-8")
                return

            # Stream raw bytes to preserve multi-byte UTF-8 sequences (emoji etc).
            # SSE lines are delimited by \n — split on byte boundaries.
            buf = b""
            async for chunk in upstream_resp.aiter_bytes():
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        yield line + b"\n\n"
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': {'message': f'Cannot connect to {url}', 'type': 'connection_error'}})}\n\n".encode("utf-8")
        except Exception as e:
            if not isinstance(e, (GeneratorExit, asyncio.CancelledError)):
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'proxy_error'}})}\n\n".encode("utf-8")
        finally:
            # Close upstream connection — critical for abort to stop inference
            if upstream_resp is not None:
                await upstream_resp.aclose()
            await client.aclose()

    return Stream(
        content=event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@get("/api/models")
async def models_proxy(request: Request) -> Response:
    """Proxy model list from upstream."""
    base_url = request.query_params.get("base_url", "http://localhost:8080")
    auth_header = request.query_params.get("auth_header", "")

    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header

    url = f"{base_url.rstrip('/')}/v1/models"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            return Response(
                content=resp.text,
                status_code=resp.status_code,
                media_type=MediaType.JSON,
            )
        except httpx.ConnectError:
            return Response(
                content=json.dumps({"error": {"message": f"Cannot connect to {url}", "type": "connection_error"}}),
                status_code=502,
                media_type=MediaType.JSON,
            )
        except Exception as e:
            return Response(
                content=json.dumps({"error": {"message": str(e), "type": "proxy_error"}}),
                status_code=500,
                media_type=MediaType.JSON,
            )


@get("/api/health")
async def health_proxy(request: Request) -> Response:
    """Proxy health check from upstream."""
    base_url = request.query_params.get("base_url", "http://localhost:8080")
    auth_header = request.query_params.get("auth_header", "")

    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header

    url = f"{base_url.rstrip('/')}/health"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            return Response(
                content=resp.text,
                status_code=resp.status_code,
                media_type=MediaType.JSON,
            )
        except httpx.ConnectError:
            return Response(
                content=json.dumps({
                    "status": "disconnected",
                    "error": f"Cannot connect to {url}",
                }),
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception as e:
            return Response(
                content=json.dumps({
                    "status": "error",
                    "error": str(e),
                }),
                status_code=200,
                media_type=MediaType.JSON,
            )


cors_config = CORSConfig(
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app = Litestar(
    route_handlers=[index, chat_proxy, chat_stream_proxy, models_proxy, health_proxy],
    cors_config=cors_config,
    static_files_config=[
        StaticFilesConfig(
            directories=[STATIC_DIR],
            path="/static",
        ),
    ],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)
