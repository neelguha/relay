"""FastAPI application factory for the relay web dashboard.

This module provides the ``create_app`` factory function, which assembles the
FastAPI application: Jinja2 templating, static file mounting, security
middleware, optional bearer-token authentication, and all route modules.

The dashboard is intended for local/internal use and does not ship a
JavaScript build step — all interactivity is handled by HTMX loaded from a
CDN link embedded in the base template.

Example::

    import uvicorn
    from relay.dashboard.web.app import create_app

    app = create_app(db_path="~/.relay/jobs.db")
    uvicorn.run(app, host="127.0.0.1", port=7860)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from relay.dashboard.web.routes import overview, jobs, costs

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject a Content-Security-Policy header on every response.

    The policy allows:
    - ``self`` for all default resource types.
    - ``https://unpkg.com`` for the HTMX CDN script.
    - ``unsafe-inline`` for styles so inline Tailwind-like classes work.

    Args:
        app: The ASGI application to wrap.
    """

    _CSP = (
        "default-src 'self'; "
        "script-src 'self' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to the response.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that passes the request to the next handler.

        Returns:
            The HTTP response with ``Content-Security-Policy`` header set.
        """
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = self._CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response


class _BearerAuthMiddleware(BaseHTTPMiddleware):
    """Enforce bearer-token authentication when a token is configured.

    If the ``relay.dashboard.auth_token`` config value is set (or the
    ``RELAY_DASHBOARD_AUTH_TOKEN`` environment variable), every request must
    carry a matching ``Authorization: Bearer <token>`` header.  Requests for
    static assets are exempted so that CSS and JS load correctly even before
    the browser sends credentials.

    Args:
        app: The ASGI application to wrap.
        token: The expected secret token string.
    """

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate the bearer token on each non-static request.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that passes the request to the next handler.

        Returns:
            The HTTP response, or a 401 Unauthorized response when the token
            is missing or incorrect.
        """
        # Let static asset requests through unconditionally.
        if request.url.path.startswith("/static"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {self._token}":
            return await call_next(request)

        return Response(
            content="Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(db_path: str) -> FastAPI:
    """Assemble and return the relay web dashboard FastAPI application.

    The returned app has:
    - Jinja2 templating pointed at the ``templates/`` directory adjacent to
      this file, with ``db_path`` stored in ``app.state``.
    - Static files mounted at ``/static``.
    - A ``Content-Security-Policy`` header injected by middleware.
    - Optional bearer-token auth (reads ``RELAY_DASHBOARD_AUTH_TOKEN`` from
      the environment, or ``relay.dashboard.auth_token`` from the relay
      config if available).
    - Overview, jobs, and costs routers included.

    Args:
        db_path: Filesystem path (may include ``~``) to the relay SQLite
            database that backs the dashboard's data queries.

    Returns:
        A fully configured :class:`fastapi.FastAPI` instance ready for an
        ASGI server such as Uvicorn.

    Example::

        app = create_app("~/.relay/jobs.db")
    """
    app = FastAPI(
        title="relay dashboard",
        description="Local monitoring dashboard for relay batch jobs.",
        docs_url=None,  # disable Swagger UI in production
        redoc_url=None,
    )

    # Store shared state so route handlers can access it.
    app.state.db_path = str(Path(db_path).expanduser().resolve())
    app.state.templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Mount static files (CSS, favicon, etc.)
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Security headers on every response.
    app.add_middleware(_SecurityHeadersMiddleware)

    # Optional bearer-token auth — checked after CSP middleware.
    auth_token = os.environ.get("RELAY_DASHBOARD_AUTH_TOKEN", "")
    if not auth_token:
        # Try to read from the relay config if available.
        try:
            from relay.config import get_config  # noqa: PLC0415

            cfg = get_config()
            auth_token = getattr(cfg.dashboard, "auth_token", "") or ""
        except Exception:  # noqa: BLE001
            pass

    if auth_token:
        app.add_middleware(_BearerAuthMiddleware, token=auth_token)

    # Include route modules.
    app.include_router(overview.router)
    app.include_router(jobs.router)
    app.include_router(costs.router)

    return app
