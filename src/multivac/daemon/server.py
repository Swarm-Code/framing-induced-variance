"""The Multi-Vac daemon server.

One asyncio process that:
* owns `DaemonState` (providers, profiles, workspaces, sessions),
* runs the `Scheduler` (loop jobs + idle compaction),
* serves newline-delimited JSON-RPC over a Unix socket to any number of thin clients.

Harness `run_sync` calls are dispatched to a thread so they never block the event loop.

RPC methods (method -> params):
  ping
  state.snapshot
  provider.add {name, base_url, model, api_key_env} | provider.remove {name} | provider.list
  profile.add {name, provider, system_prompt?, bundle?} | profile.remove {name} | profile.list
  workspace.add {name, path, profile?, bundle?} | workspace.remove {name} | workspace.list
  session.create {workspace?, profile?, title?} | session.close {id} | session.list
  session.chat {id, message}            # runs a turn (handles /commands)
  session.compact {id}
  skill.list {session} | skill.view {session, name} | skill.create {...} | skill.patch {...}
  hook.list {session}
  subagent.list {session} | subagent.add {session, name, system_prompt, description?}
  loop.add {session, prompt, interval_seconds} | loop.remove {id} | loop.list
  command.list
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path

from ..subagents import SubAgentSpec
from ..types import HookContext, HookEvent
from .commands import CommandError, default_registry, parse_interval
from .events import EventBus
from .protocol import Request, Response, daemon_home, default_socket_path
from .scheduler import Scheduler
from .state import DaemonState, Profile, Provider, Workspace


class DaemonServer:
    def __init__(self, socket_path: str | None = None, home: Path | None = None) -> None:
        self.socket_path = socket_path or default_socket_path()
        self.home = Path(home) if home else daemon_home()
        self.state = DaemonState(self.home)
        self.commands = default_registry()
        self.bus = EventBus()
        self._instrumented: set[str] = set()
        self.scheduler = Scheduler(self._run_turn, self._compact_session, home=self.home)
        self.scheduler.bind_sessions(lambda: list(self.state.registry.sessions))
        self._server: asyncio.AbstractServer | None = None
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------ lifecycle
    async def serve(self) -> None:
        sp = Path(self.socket_path)
        if sp.exists():
            sp.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=self.socket_path
        )
        self.bus.bind_loop(asyncio.get_running_loop())
        self.scheduler.start()
        loop = asyncio.get_running_loop()
        # Signal handlers are a convenience and only work on the main thread; ignore
        # failures so the daemon can also run inside a worker thread (e.g. in tests/GUI).
        import contextlib

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, ValueError, RuntimeError):
                loop.add_signal_handler(sig, self._stopping.set)
        async with self._server:
            await self._stopping.wait()
        await self.scheduler.stop()
        with __import__("contextlib").suppress(FileNotFoundError):
            sp.unlink()

    # --------------------------------------------------------------- connections
    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                try:
                    req = Request.model_validate_json(line.decode())
                except Exception as e:  # noqa: BLE001
                    writer.write(
                        (Response.failure(0, f"bad request: {e}").model_dump_json() + "\n").encode()
                    )
                    await writer.drain()
                    continue
                # Streaming method: attach turns this connection into an event stream.
                if req.method == "session.attach":
                    await self._stream_attach(req, reader, writer)
                    break
                resp = await self._dispatch(req)
                writer.write((resp.model_dump_json() + "\n").encode())
                await writer.drain()
        finally:
            writer.close()

    async def _stream_attach(
        self,
        req: Request,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Subscribe this connection to a session's events and stream them as frames."""
        session_id = req.params.get("session")  # None => all sessions
        # Ensure the session is instrumented so events flow.
        if session_id:
            try:
                self._harness(session_id)
            except (KeyError, ValueError) as e:
                writer.write((Response.failure(req.id, str(e)).model_dump_json() + "\n").encode())
                await writer.drain()
                return
        sub = self.bus.subscribe(session_id)
        writer.write(
            (Response.success(req.id, {"attached": session_id or "*"}).model_dump_json() + "\n").encode()
        )
        await writer.drain()
        try:
            while not self._stopping.is_set():
                try:
                    payload = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if reader.at_eof():
                        break
                    continue
                import json

                writer.write((json.dumps(payload) + "\n").encode())
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self.bus.unsubscribe(sub)

    async def _dispatch(self, req: Request) -> Response:
        handler = getattr(self, f"_m_{req.method.replace('.', '_')}", None)
        if handler is None:
            return Response.failure(req.id, f"unknown method: {req.method}")
        try:
            result = await handler(req.params)
            return Response.success(req.id, result)
        except (CommandError, ValueError, KeyError) as e:
            return Response.failure(req.id, str(e))
        except Exception as e:  # noqa: BLE001
            return Response.failure(req.id, f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------- harness helpers
    def _harness(self, session_id: str):
        """Get the session harness, instrumenting it for event streaming once."""
        h = self.state.harness(session_id)
        if session_id not in self._instrumented:
            self._instrument(h, session_id)
            self._instrumented.add(session_id)
        return h

    def _instrument(self, harness, session_id: str) -> None:
        """Register hooks that publish tool + sub-agent activity to the event bus."""
        bus = self.bus

        def on_pre_tool(ctx: HookContext):
            bus.publish(
                "tool_pre",
                session_id,
                tool=ctx.tool_name,
                args=ctx.tool_args,
            )
            return None

        def on_post_tool(ctx: HookContext):
            result = ctx.tool_result
            bus.publish(
                "tool_post",
                session_id,
                tool=ctx.tool_name,
                result=str(result)[:500],
                error=bool(ctx.metadata.get("error")),
            )
            return None

        harness.on([HookEvent.PRE_TOOL], on_pre_tool, name="_bus_pre_tool", priority=1)
        harness.on([HookEvent.POST_TOOL], on_post_tool, name="_bus_post_tool", priority=999)

        # Wrap sub-agent runs to emit start/end events.
        registry = harness.subagents
        original_run = registry.run

        def run_with_events(name: str, task: str, *, depth: int = 0) -> str:
            bus.publish("subagent_start", session_id, name=name, task=task[:200])
            out = original_run(name, task, depth=depth)
            bus.publish("subagent_end", session_id, name=name, output=str(out)[:500])
            return out

        registry.run = run_with_events  # type: ignore[assignment]

    async def _run_turn(self, session_id: str, message: str) -> str:
        """Run one turn, resolving /commands first. Offloaded to a thread."""
        # Slash command?
        if self.commands.is_command(message):
            name, args = self.commands.split(message)
            cmd = self.commands.get(name)
            if cmd is None:
                return f"[unknown command: /{name}]"
            if name == "help":
                return self._help_text()
            if name == "compact":
                did = await self._compact_session(session_id)
                return "[compacted]" if did else "[nothing to compact]"
            if name == "loop":
                interval, prompt = parse_interval(args)
                if not prompt:
                    return "[usage: /loop [interval] <prompt>]"
                job = self.scheduler.add_loop(session_id, prompt, interval)
                return (
                    f"[loop {job.id} scheduled every {interval}s; "
                    f"auto-expires in 3 days]"
                )
            # Other commands inject a prompt then run normally.
            message = cmd.get_prompt(args)

        self.bus.publish("turn_start", session_id, message=message[:500])
        harness = self._harness(session_id)
        result = await asyncio.to_thread(harness.chat, message)
        self.bus.publish("assistant", session_id, output=result.output)
        return result.output

    async def _compact_session(self, session_id: str) -> bool:
        try:
            harness = self._harness(session_id)
        except (KeyError, ValueError):
            return False
        did = await asyncio.to_thread(harness.maybe_compact)
        if did:
            self.bus.publish("compacted", session_id)
        return did

    def _help_text(self) -> str:
        lines = ["Available commands:"]
        for c in self.commands.all():
            hint = f" {c.arg_hint}" if c.arg_hint else ""
            lines.append(f"  /{c.name}{hint} — {c.description}")
        return "\n".join(lines)

    # ----------------------------------------------------------------- RPC methods
    async def _m_ping(self, p: dict) -> dict:
        return {"pong": True, "pid": os.getpid(), "socket": self.socket_path}

    async def _m_state_snapshot(self, p: dict) -> dict:
        return self.state.snapshot()

    async def _m_command_list(self, p: dict) -> list:
        return [
            {"name": c.name, "description": c.description, "arg_hint": c.arg_hint}
            for c in self.commands.all()
        ]

    # providers
    async def _m_provider_add(self, p: dict) -> dict:
        return self.state.add_provider(Provider(**p)).model_dump()

    async def _m_provider_remove(self, p: dict) -> dict:
        self.state.remove_provider(p["name"])
        return {"removed": p["name"]}

    async def _m_provider_list(self, p: dict) -> list:
        return [v.model_dump() for v in self.state.registry.providers.values()]

    # profiles
    async def _m_profile_add(self, p: dict) -> dict:
        return self.state.add_profile(Profile(**p)).model_dump()

    async def _m_profile_remove(self, p: dict) -> dict:
        self.state.remove_profile(p["name"])
        return {"removed": p["name"]}

    async def _m_profile_list(self, p: dict) -> list:
        return [v.model_dump() for v in self.state.registry.profiles.values()]

    # workspaces
    async def _m_workspace_add(self, p: dict) -> dict:
        return self.state.add_workspace(Workspace(**p)).model_dump()

    async def _m_workspace_remove(self, p: dict) -> dict:
        self.state.remove_workspace(p["name"])
        return {"removed": p["name"]}

    async def _m_workspace_list(self, p: dict) -> list:
        return [v.model_dump() for v in self.state.registry.workspaces.values()]

    async def _m_workspace_infer(self, p: dict) -> dict:
        """Get-or-create a workspace for the caller's cwd (auto-discovers a bundle)."""
        ws = self.state.infer_workspace(p["cwd"])
        return ws.model_dump()

    # sessions
    async def _m_session_create(self, p: dict) -> dict:
        # If a cwd is given and no workspace, infer one (cwd -> workspace + bundle).
        workspace = p.get("workspace")
        if not workspace and p.get("cwd"):
            workspace = self.state.infer_workspace(p["cwd"]).name
        meta = self.state.create_session(
            workspace=workspace,
            profile=p.get("profile"),
            title=p.get("title", ""),
        )
        return meta.model_dump()

    async def _m_session_close(self, p: dict) -> dict:
        self.state.close_session(p["id"])
        return {"closed": p["id"]}

    async def _m_session_list(self, p: dict) -> list:
        return [v.model_dump() for v in self.state.registry.sessions.values()]

    async def _m_session_chat(self, p: dict) -> dict:
        output = await self._run_turn(p["id"], p["message"])
        return {"output": output}

    async def _m_session_compact(self, p: dict) -> dict:
        did = await self._compact_session(p["id"])
        return {"compacted": did}

    # skills (scoped to a session's harness skill store)
    async def _m_skill_list(self, p: dict) -> list:
        h = self._harness(p["session"])
        return [s.model_dump() for s in h.skills.list()]

    async def _m_skill_view(self, p: dict) -> dict:
        h = self._harness(p["session"])
        return h.skills.view(p["name"]).model_dump()

    async def _m_skill_create(self, p: dict) -> dict:
        h = self._harness(p["session"])
        s = h.skills.create(p["name"], p.get("description", ""), p.get("body", ""))
        return s.model_dump()

    async def _m_skill_patch(self, p: dict) -> dict:
        h = self._harness(p["session"])
        s = h.skills.patch(p["name"], body=p.get("body"), description=p.get("description"))
        return s.model_dump()

    # hooks
    async def _m_hook_list(self, p: dict) -> list:
        h = self._harness(p["session"])
        return [
            {"name": hook.name, "events": [e.value for e in hook.events], "priority": hook.priority}
            for hook in h.hooks._hooks  # noqa: SLF001 - introspection for management UI
        ]

    # subagents
    async def _m_subagent_list(self, p: dict) -> list:
        h = self._harness(p["session"])
        return [{"name": n} for n in h.subagents.names()]

    async def _m_subagent_add(self, p: dict) -> dict:
        h = self._harness(p["session"])
        h.add_subagent(
            SubAgentSpec(
                name=p["name"],
                system_prompt=p["system_prompt"],
                description=p.get("description", ""),
            )
        )
        return {"added": p["name"]}

    # loops / cron
    async def _m_loop_add(self, p: dict) -> dict:
        job = self.scheduler.add_loop(
            p["session"], p["prompt"], int(p["interval_seconds"])
        )
        return job.model_dump()

    async def _m_loop_remove(self, p: dict) -> dict:
        return {"removed": self.scheduler.remove_loop(p["id"])}

    async def _m_loop_list(self, p: dict) -> list:
        return [j.model_dump() for j in self.scheduler.list_loops()]


def run_daemon(socket_path: str | None = None) -> None:
    server = DaemonServer(socket_path)
    asyncio.run(server.serve())
