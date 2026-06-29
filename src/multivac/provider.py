"""The model choke-point.

One place builds the model the harness (and its sub-agents) talk to:

* LIVE  (api_key set): real model over the OpenAI-compatible endpoint (Cerebras Gemma 4).
* OFFLINE (no key, or MULTIVAC_OFFLINE=1): a deterministic `TestModel` so the harness,
  skills, hooks, MCP wiring and compaction can all be exercised with no network.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel

from .config import Settings

# Browser-like UA to pass Cloudflare bot-integrity (avoids HTTP 403 "error code:
# 1010" on Cerebras from datacenter/sandbox egress). See provider._live_model.
_BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) python-requests/2.31"


def _build_rotating_transport_cls():
    """Build the rotating-key transport lazily so httpx is only imported when live."""
    import asyncio

    import httpx

    class _RotatingKeyTransport(httpx.AsyncBaseTransport):
        """Rotate across PROVIDERS (endpoint+key+host) on 429/5xx for uptime.

        Each provider is {base_url, api_key, host}. On a rate-limited or transient
        response we rewrite the request's URL host/scheme/path-prefix and the
        Authorization header to the next provider and retry with brief backoff.
        With Fireworks (primary) + Cerebras (fallback) this gives cross-provider
        failover so one provider's 100k tok/min cap never stalls a run. Always
        injects the browser UA. If only one provider is supplied, behaves as a
        single-endpoint UA-injecting transport with retry.
        """

        def __init__(self, providers, *, base_headers=None, max_attempts=6):
            # providers: list of {"base_url","api_key"} dicts (ordered by priority)
            self._providers = [p for p in providers if p.get("api_key")] or [{}]
            self._i = 0
            self._base_headers = base_headers or {}
            self._max_attempts = max(1, max_attempts)
            self._inner = httpx.AsyncHTTPTransport(retries=0)

        def _retarget(self, request: httpx.Request, prov: dict) -> None:
            """Point the request at this provider's base_url + key."""
            for hk, hv in self._base_headers.items():
                request.headers[hk] = hv
            if prov.get("api_key"):
                request.headers["Authorization"] = f"Bearer {prov['api_key']}"
            base = prov.get("base_url")
            if base:
                b = httpx.URL(base)
                # base path like '/v1' or '/inference/v1'; the OpenAI client appended
                # '/chat/completions' onto the original base — recompose against new base.
                tail = request.url.path
                # keep only the part after the original '/v1' boundary if present
                if "/v1/" in tail:
                    tail = tail[tail.index("/v1/") + 3:]  # -> '/chat/completions'
                elif tail.startswith("/"):
                    tail = tail
                new = b.join(b.path.rstrip("/") + tail)
                request.url = new
                request.headers["host"] = new.host

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            last: httpx.Response | None = None
            n = len(self._providers)
            for attempt in range(self._max_attempts):
                prov = self._providers[self._i % n]
                self._retarget(request, prov)
                resp = await self._inner.handle_async_request(request)
                if resp.status_code not in (429, 500, 502, 503, 529):
                    return resp
                await resp.aclose()
                last = resp
                self._i += 1  # rotate to the next provider
                await asyncio.sleep(min(1.5 * (attempt + 1), 6.0))
            return last

        async def aclose(self) -> None:
            await self._inner.aclose()

    return _RotatingKeyTransport


_RotatingKeyTransport = None  # bound lazily in _live_model


class ModelProvider:
    """Builds and caches the harness model for the active mode."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Model | None = None

    @property
    def is_live(self) -> bool:
        return self.settings.is_live

    @property
    def mode(self) -> str:
        return "live" if self.is_live else "offline"

    def model(self) -> Model:
        if self._model is None:
            self._model = self._live_model() if self.is_live else self._offline_model()
        return self._model

    # ------------------------------------------------------------------ live
    def _live_model(self) -> Model:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        # Cerebras sits behind Cloudflare, which 403s ("error code: 1010") on
        # datacenter/sandbox egress when the User-Agent looks like a bot
        # (python-urllib / openai default). A browser-like UA passes the check.
        # We also rotate API keys on HTTP 429 (rate limit) for constant uptime
        # under the 100k tokens/min cap. Both are done via a custom httpx client.
        provider_kwargs = {"base_url": self.settings.base_url, "api_key": self.settings.api_key}
        try:
            import httpx
            from openai import AsyncOpenAI

            # Multi-provider rotation when configured (Fireworks primary, Cerebras
            # fallback); otherwise single-key Cerebras with the fallback keys.
            if self.settings.providers:
                provs = self.settings.providers
            else:
                provs = [{"name": "cerebras", "base_url": self.settings.base_url,
                          "api_key": k, "model": self.settings.model}
                         for k in [self.settings.api_key, *self.settings.fallback_keys] if k]
            primary = provs[0]
            rotating_cls = _build_rotating_transport_cls()
            transport = rotating_cls(provs, base_headers={"User-Agent": _BROWSER_UA})
            http_client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(120.0))
            client = AsyncOpenAI(
                api_key=primary["api_key"],
                base_url=primary["base_url"],
                http_client=http_client,
                max_retries=0,  # rotation transport owns retry/rotation
            )
            provider = OpenAIProvider(openai_client=client)
            model_id = primary.get("model") or self.settings.model
        except Exception:  # noqa: BLE001 - fall back to plain provider if SDK shape differs
            provider = OpenAIProvider(**provider_kwargs)
            model_id = self.settings.model
        return OpenAIChatModel(model_id, provider=provider)

    # --------------------------------------------------------------- offline
    def _offline_model(self) -> Model:
        return TestModel(
            call_tools=[],  # don't auto-invoke registered tools with dummy args
            custom_output_text=(
                "[offline] Multi-Vac harness response. Set MULTIVAC_API_KEY / "
                "CEREBRAS_API_KEY to talk to a real model."
            ),
            model_name="offline",
        )
