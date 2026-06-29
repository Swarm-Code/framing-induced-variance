"""MCP (Model Context Protocol) support.

Declare external MCP servers (stdio subprocess or HTTP) and turn them into Pydantic AI
toolsets that attach to the harness agent. The agent can then call tools exposed by
those servers exactly like local tools.

`pydantic-ai` provides the server/toolset classes; importing them requires the optional
`mcp` package. We import lazily so the rest of the harness works even if `mcp` isn't
installed — `build_mcp_toolsets` simply returns an empty list with a clear warning.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("multivac.mcp")


class MCPServerConfig(BaseModel):
    """Declarative description of one MCP server."""

    name: str
    transport: Literal["stdio", "http", "sse"] = "stdio"

    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None

    # http / sse transport
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    # prefix tool names so multiple servers don't collide
    tool_prefix: str | None = None


def _mcp_available() -> bool:
    try:
        import mcp  # noqa: F401

        return True
    except Exception:
        return False


def build_mcp_toolsets(configs: list[MCPServerConfig]) -> list[Any]:
    """Construct Pydantic AI MCP toolsets from configs.

    Returns a list suitable for `Agent(toolsets=...)`. Returns an empty list (and logs)
    if the `mcp` package is missing, so callers never crash on a missing optional dep.
    """
    if not configs:
        return []
    if not _mcp_available():
        logger.warning(
            "MCP servers configured (%s) but the `mcp` package is not installed; "
            'install with: pip install "pydantic-ai-slim[mcp]"',
            ", ".join(c.name for c in configs),
        )
        return []

    # pydantic-ai 2.x reworked the MCP API: build a transport -> FastMCPClient ->
    # MCPToolset. Fall back to the 1.x server classes if running on the old version.
    try:
        from pydantic_ai.mcp import (
            FastMCPClient,
            MCPToolset,
            SSETransport,
            StdioTransport,
        )
        from fastmcp.client.transports import StreamableHttpTransport

        return _build_v2(
            configs, MCPToolset, FastMCPClient, StdioTransport, SSETransport, StreamableHttpTransport
        )
    except ImportError:
        from pydantic_ai.mcp import (  # type: ignore[attr-defined]
            MCPServerSSE,
            MCPServerStdio,
            MCPServerStreamableHTTP,
        )

        return _build_v1(configs, MCPServerStdio, MCPServerStreamableHTTP, MCPServerSSE)


def _build_v2(
    configs, MCPToolset, FastMCPClient, StdioTransport, SSETransport, StreamableHttpTransport
) -> list[Any]:
    toolsets: list[Any] = []
    for cfg in configs:
        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(f"MCP server {cfg.name!r}: stdio transport needs `command`")
            transport = StdioTransport(
                command=cfg.command, args=cfg.args, env=cfg.env or None, cwd=cfg.cwd
            )
        elif cfg.transport == "http":
            if not cfg.url:
                raise ValueError(f"MCP server {cfg.name!r}: http needs `url`")
            transport = StreamableHttpTransport(url=cfg.url, headers=cfg.headers or None)
        elif cfg.transport == "sse":
            if not cfg.url:
                raise ValueError(f"MCP server {cfg.name!r}: sse needs `url`")
            transport = SSETransport(url=cfg.url, headers=cfg.headers or None)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown MCP transport: {cfg.transport}")

        toolset = MCPToolset(client=FastMCPClient(transport=transport, name=cfg.name))
        if cfg.tool_prefix:
            toolset = toolset.prefixed(cfg.tool_prefix)
        toolsets.append(toolset)
    return toolsets


def _build_v1(configs, MCPServerStdio, MCPServerStreamableHTTP, MCPServerSSE) -> list[Any]:
    toolsets: list[Any] = []
    for cfg in configs:
        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(f"MCP server {cfg.name!r}: stdio transport needs `command`")
            toolsets.append(
                MCPServerStdio(
                    command=cfg.command,
                    args=cfg.args,
                    env=cfg.env or None,
                    cwd=cfg.cwd,
                    tool_prefix=cfg.tool_prefix,
                )
            )
        elif cfg.transport in ("http", "sse"):
            if not cfg.url:
                raise ValueError(f"MCP server {cfg.name!r}: {cfg.transport} needs `url`")
            klass = MCPServerStreamableHTTP if cfg.transport == "http" else MCPServerSSE
            toolsets.append(
                klass(url=cfg.url, headers=cfg.headers or None, tool_prefix=cfg.tool_prefix)
            )
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown MCP transport: {cfg.transport}")
    return toolsets
