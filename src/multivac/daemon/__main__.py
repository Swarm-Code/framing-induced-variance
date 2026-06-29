"""`python -m multivac.daemon [--foreground]` — run the daemon server."""

from __future__ import annotations

import argparse

from .server import run_daemon


def main() -> None:
    p = argparse.ArgumentParser(prog="multivac.daemon")
    p.add_argument("--foreground", action="store_true", help="run in the foreground")
    p.add_argument("--socket", default=None, help="override the socket path")
    args = p.parse_args()
    run_daemon(args.socket)


if __name__ == "__main__":
    main()
