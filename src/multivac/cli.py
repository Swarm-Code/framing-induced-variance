"""Multi-Vac CLI — a thin client over the daemon (stdlib argparse, no extra deps).

The daemon is the single execution engine; this CLI is just one UI. A future GUI would
speak the same socket. Commands:

  multivac daemon start [--foreground] | stop | status
  multivac provider add <name> --base-url --model --api-key-env | list | remove <name>
  multivac profile  add <name> --provider [--system-prompt] [--bundle] | list | remove <name>
  multivac workspace add <name> --path [--profile] [--bundle] | list | remove <name>
  multivac session  new [--workspace] [--profile] [--title] | list | close <id>
  multivac chat [--session <id>] [--workspace] [--profile]     # interactive REPL with /commands
  multivac skill    list|view|create|patch --session <id> ...
  multivac hook     list --session <id>
  multivac subagent list|add --session <id> ...
  multivac loop     list | remove <id>
  multivac tmux     attach <id>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from .daemon.client import DaemonClient, DaemonError
from .daemon.protocol import default_socket_path


def _client() -> DaemonClient:
    return DaemonClient(default_socket_path())


# ----------------------------------------------------------------------- daemon
def cmd_daemon(args: argparse.Namespace) -> int:
    if args.action == "start":
        c = _client()
        if c.is_running():
            print("daemon already running")
            return 0
        if args.foreground:
            from .daemon.server import run_daemon

            run_daemon()
            return 0
        # Spawn a detached background daemon.
        proc = subprocess.Popen(
            [sys.executable, "-m", "multivac.daemon", "--foreground"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "PYTHONPATH": _pythonpath()},
        )
        for _ in range(50):
            time.sleep(0.1)
            if c.is_running():
                print(f"daemon started (pid {proc.pid})")
                return 0
        print("daemon failed to start", file=sys.stderr)
        return 1

    if args.action == "status":
        c = _client()
        if not c.is_running():
            print("daemon: stopped")
            return 1
        info = c.call("ping")
        print(f"daemon: running (pid {info['pid']}) at {info['socket']}")
        return 0

    if args.action == "stop":
        c = _client()
        if not c.is_running():
            print("daemon: not running")
            return 0
        try:
            info = c.call("ping")
            os.kill(info["pid"], 15)
            print("daemon stopped")
        except (DaemonError, ProcessLookupError) as e:
            print(f"could not stop daemon: {e}", file=sys.stderr)
            return 1
        return 0
    return 2


def _pythonpath() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    existing = os.environ.get("PYTHONPATH", "")
    return f"{here}:{existing}" if existing else here


# ----------------------------------------------------------------- generic call
def _print(result) -> None:
    import json

    print(json.dumps(result, indent=2))


def cmd_provider(args: argparse.Namespace) -> int:
    c = _client()
    if args.action == "add":
        _print(
            c.call(
                "provider.add",
                name=args.name,
                base_url=args.base_url,
                model=args.model,
                api_key_env=args.api_key_env,
            )
        )
    elif args.action == "list":
        _print(c.call("provider.list"))
    elif args.action == "remove":
        _print(c.call("provider.remove", name=args.name))
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    c = _client()
    if args.action == "add":
        _print(
            c.call(
                "profile.add",
                name=args.name,
                provider=args.provider,
                system_prompt=args.system_prompt,
                bundle=args.bundle,
            )
        )
    elif args.action == "list":
        _print(c.call("profile.list"))
    elif args.action == "remove":
        _print(c.call("profile.remove", name=args.name))
    return 0


def cmd_workspace(args: argparse.Namespace) -> int:
    c = _client()
    if args.action == "add":
        _print(
            c.call(
                "workspace.add",
                name=args.name,
                path=os.path.abspath(args.path),
                profile=args.profile,
                bundle=args.bundle,
            )
        )
    elif args.action == "list":
        _print(c.call("workspace.list"))
    elif args.action == "remove":
        _print(c.call("workspace.remove", name=args.name))
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    c = _client()
    if args.action == "new":
        _print(
            c.call(
                "session.create",
                workspace=args.workspace,
                profile=args.profile,
                title=args.title or "",
            )
        )
    elif args.action == "list":
        _print(c.call("session.list"))
    elif args.action == "close":
        _print(c.call("session.close", id=args.id))
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    c = _client()
    if args.action == "list":
        _print(c.call("loop.list"))
    elif args.action == "remove":
        _print(c.call("loop.remove", id=args.id))
    return 0


def cmd_skill(args: argparse.Namespace) -> int:
    c = _client()
    if args.action == "list":
        _print(c.call("skill.list", session=args.session))
    elif args.action == "view":
        _print(c.call("skill.view", session=args.session, name=args.name))
    elif args.action == "create":
        _print(
            c.call(
                "skill.create",
                session=args.session,
                name=args.name,
                description=args.description or "",
                body=args.body or "",
            )
        )
    return 0


def cmd_hook(args: argparse.Namespace) -> int:
    _print(_client().call("hook.list", session=args.session))
    return 0


def cmd_subagent(args: argparse.Namespace) -> int:
    c = _client()
    if args.action == "list":
        _print(c.call("subagent.list", session=args.session))
    elif args.action == "add":
        _print(
            c.call(
                "subagent.add",
                session=args.session,
                name=args.name,
                system_prompt=args.system_prompt,
                description=args.description or "",
            )
        )
    return 0


def cmd_tmux(args: argparse.Namespace) -> int:
    from .daemon import tmux

    if args.action == "attach":
        os.execvp("tmux", tmux.attach_command(args.id))
    return 0


# -------------------------------------------------------------------------- chat
def cmd_chat(args: argparse.Namespace) -> int:
    c = _client()
    if not c.is_running():
        print("daemon not running — start it with: multivac daemon start", file=sys.stderr)
        return 1

    # Resolve the active workspace from cwd (auto-discovers a bundle) unless one given.
    workspace = args.workspace
    bundle = None
    if workspace is None:
        ws = c.call("workspace.infer", cwd=os.getcwd())
        workspace = ws["name"]
        bundle = ws.get("bundle")

    session_id = args.session
    if session_id is None:
        meta = c.call(
            "session.create",
            workspace=workspace,
            profile=args.profile,
            title="cli",
            cwd=os.getcwd(),
        )
        session_id = meta["id"]

    # Show the active context so it's obvious what loaded.
    print("Multi-Vac chat")
    print(f"  session   : {session_id}")
    print(f"  workspace : {workspace}")
    print(f"  bundle    : {bundle or '(none — using default profile)'}")
    _print_commands(c)
    print("Type a message, a /command, or /exit to quit.\n")

    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line in ("/commands", "/?"):
            _print_commands(c)
            continue
        try:
            res = c.call("session.chat", id=session_id, message=line)
            print(f"multivac > {res['output']}\n")
        except DaemonError as e:
            print(f"[error] {e}", file=sys.stderr)
    return 0


def _print_commands(c: DaemonClient) -> None:
    try:
        cmds = c.call("command.list")
    except DaemonError:
        return
    print("Slash commands:")
    for cmd in cmds:
        hint = f" {cmd['arg_hint']}" if cmd.get("arg_hint") else ""
        print(f"  /{cmd['name']}{hint:<18} {cmd['description']}")
    print()


def cmd_tui(args: argparse.Namespace) -> int:
    from .daemon.tui import run_tui

    return run_tui()


# -------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="multivac", description="Multi-Vac daemon client")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daemon", help="manage the daemon")
    d.add_argument("action", choices=["start", "stop", "status"])
    d.add_argument("--foreground", action="store_true")
    d.set_defaults(func=cmd_daemon)

    pr = sub.add_parser("provider", help="manage providers")
    pr.add_argument("action", choices=["add", "list", "remove"])
    pr.add_argument("name", nargs="?")
    pr.add_argument("--base-url", default="https://api.cerebras.ai/v1")
    pr.add_argument("--model", default="gemma-4-31b")
    pr.add_argument("--api-key-env", default="CEREBRAS_API_KEY")
    pr.set_defaults(func=cmd_provider)

    pf = sub.add_parser("profile", help="manage profiles")
    pf.add_argument("action", choices=["add", "list", "remove"])
    pf.add_argument("name", nargs="?")
    pf.add_argument("--provider", default="cerebras")
    pf.add_argument("--system-prompt", default=None)
    pf.add_argument("--bundle", default=None)
    pf.set_defaults(func=cmd_profile)

    ws = sub.add_parser("workspace", help="manage workspaces")
    ws.add_argument("action", choices=["add", "list", "remove"])
    ws.add_argument("name", nargs="?")
    ws.add_argument("--path", default=".")
    ws.add_argument("--profile", default=None)
    ws.add_argument("--bundle", default=None)
    ws.set_defaults(func=cmd_workspace)

    se = sub.add_parser("session", help="manage sessions")
    se.add_argument("action", choices=["new", "list", "close"])
    se.add_argument("id", nargs="?")
    se.add_argument("--workspace", default=None)
    se.add_argument("--profile", default=None)
    se.add_argument("--title", default=None)
    se.set_defaults(func=cmd_session)

    lp = sub.add_parser("loop", help="manage recurring loop jobs")
    lp.add_argument("action", choices=["list", "remove"])
    lp.add_argument("id", nargs="?")
    lp.set_defaults(func=cmd_loop)

    sk = sub.add_parser("skill", help="manage skills")
    sk.add_argument("action", choices=["list", "view", "create", "patch"])
    sk.add_argument("--session", required=True)
    sk.add_argument("--name", default=None)
    sk.add_argument("--description", default=None)
    sk.add_argument("--body", default=None)
    sk.set_defaults(func=cmd_skill)

    hk = sub.add_parser("hook", help="inspect hooks")
    hk.add_argument("action", choices=["list"])
    hk.add_argument("--session", required=True)
    hk.set_defaults(func=cmd_hook)

    sa = sub.add_parser("subagent", help="manage sub-agents")
    sa.add_argument("action", choices=["list", "add"])
    sa.add_argument("--session", required=True)
    sa.add_argument("--name", default=None)
    sa.add_argument("--system-prompt", default=None)
    sa.add_argument("--description", default=None)
    sa.set_defaults(func=cmd_subagent)

    tx = sub.add_parser("tmux", help="tmux integration")
    tx.add_argument("action", choices=["attach"])
    tx.add_argument("id")
    tx.set_defaults(func=cmd_tmux)

    ch = sub.add_parser("chat", help="interactive chat REPL with /commands")
    ch.add_argument("--session", default=None)
    ch.add_argument("--workspace", default=None)
    ch.add_argument("--profile", default=None)
    ch.set_defaults(func=cmd_chat)

    tu = sub.add_parser("tui", help="full-screen TUI to attach to sessions + sub-agents")
    tu.set_defaults(func=cmd_tui)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DaemonError as e:
        print(f"[daemon error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
