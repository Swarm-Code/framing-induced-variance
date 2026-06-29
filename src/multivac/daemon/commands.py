"""Slash-command registry — mirrors the Claude-Code/deconstruct command pattern.

A `Command` has a name, description, argument hint, and a `get_prompt(args)` that turns
user arguments into the text injected into the conversation. The CLI/GUI parse a leading
``/word`` and dispatch here. Some commands (``/loop``) also schedule daemon-side work.

Built-in commands:
* ``/goal <objective>``        — inject a goal-oriented execution prompt.
* ``/loop [interval] <prompt>``— schedule a recurring run (default 10m, max 3 days).
* ``/compact``                 — compact the current session now.
* ``/help``                    — list commands.

Interval grammar (from deconstruct): ``<n><unit> <text>`` where unit ∈ s|m|h|d.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_INTERVAL_RE = re.compile(r"^(\d+)?([smhd])\s+(.*)$", re.DOTALL)
_DEFAULT_INTERVAL_SECONDS = 600  # 10 minutes
_MAX_INTERVAL_SECONDS = 3 * 24 * 3600  # 3 days


def parse_interval(text: str) -> tuple[int, str]:
    """Parse ``5m do thing`` -> (300, "do thing"). No unit -> default 10m, text intact."""
    m = _INTERVAL_RE.match(text.strip())
    if not m:
        return _DEFAULT_INTERVAL_SECONDS, text.strip()
    num = int(m.group(1) or "1")
    unit = m.group(2)
    rest = m.group(3).strip()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return num * mult, rest


GOAL_PROMPT = """# Goal-Oriented Task Execution

You are executing a goal-oriented workflow to achieve a specific objective.

## Goal
{goal}

## Instructions
1. Understand the goal and break it into measurable outcomes.
2. Create a step-by-step plan (prerequisites, subtasks, obstacles).
3. Execute systematically, verifying each step before proceeding.
4. Validate success: check requirements, test the outcome, report completion.

Begin with understanding and planning."""


@dataclass
class Command:
    name: str
    description: str
    arg_hint: str
    get_prompt: Callable[[str], str]
    schedules: bool = False  # True if the command registers daemon-side recurring work


class CommandError(ValueError):
    pass


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, cmd: Command) -> None:
        self._commands[cmd.name] = cmd

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    def names(self) -> list[str]:
        return sorted(self._commands)

    def all(self) -> list[Command]:
        return [self._commands[n] for n in self.names()]

    def is_command(self, text: str) -> bool:
        return text.lstrip().startswith("/")

    def split(self, text: str) -> tuple[str, str]:
        """'/goal ship it' -> ('goal', 'ship it')."""
        body = text.lstrip()[1:]
        name, _, args = body.partition(" ")
        return name.strip(), args.strip()


def default_registry() -> CommandRegistry:
    reg = CommandRegistry()

    reg.register(
        Command(
            name="goal",
            description="Set and work toward a specific objective with systematic planning",
            arg_hint="<objective>",
            get_prompt=lambda args: _goal_prompt(args),
        )
    )
    reg.register(
        Command(
            name="loop",
            description="Run a prompt/command on a recurring interval (default 10m, max 3d)",
            arg_hint="[interval] <prompt>",
            get_prompt=lambda args: args,  # the daemon schedules; prompt is the payload
            schedules=True,
        )
    )
    reg.register(
        Command(
            name="compact",
            description="Compact the current conversation now",
            arg_hint="",
            get_prompt=lambda args: "",
        )
    )
    reg.register(
        Command(
            name="help",
            description="List available slash commands",
            arg_hint="",
            get_prompt=lambda args: "",
        )
    )
    return reg


def _goal_prompt(args: str) -> str:
    if not args.strip():
        raise CommandError("usage: /goal <objective>")
    return GOAL_PROMPT.format(goal=args.strip())
