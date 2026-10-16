from __future__ import annotations

import queue
import sys
import time

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)


def dashboard(run):
    snapshot = run.snapshot()
    table = Table(expand=True, box=None, padding=(0, 1))
    for title, width in [("Agent", 12), ("Model", 28), ("State", 10), ("Step", 4), ("Tokens", 6)]:
        table.add_column(title, width=width, overflow="ellipsis", no_wrap=True)
    colors = {"inference": "cyan", "tool": "yellow", "approval": "magenta", "done": "green", "error": "red", "limit": "yellow"}
    for name, state in snapshot["states"].items():
        model = state["model"].split("/")[-1].replace("-HauhauCS-Aggressive", "")
        table.add_row(Text(name), Text(model), Text(state["status"], style=colors.get(state["status"], "dim")),
                      str(state["step"]), str(state["tokens"]))
    notes = []
    for entry in snapshot["events"]:
        if entry.get("detail"):
            notes.append(Text(f"{entry['agent']} · {entry['detail'][:140]}"))
    route = run.route
    selection = "choosing route" if route["mode"] == "pending" else f"{route['mode']} · {route['workers']} active"
    footer = Text(f"{snapshot['seconds']:.1f}s  ·  {selection}  ·  {run.config.mode}  ·  {run.config.inference_parallel} inference slot(s)  ·  Ctrl-C cancels", style="dim")
    return Panel(Group(table, Text(""), *notes, Text(""), footer), title="Agentscape · " + run.id, border_style="cyan")


def execute_run(run, *, quiet=False):
    run.start()
    interactive = sys.stdin.isatty() and not quiet
    live = Live(dashboard(run), console=console, refresh_per_second=4, transient=False) if console.is_terminal and not quiet else None
    if live:
        live.start()
    try:
        while not run.done.wait(0.15):
            if live:
                live.update(dashboard(run))
            try:
                request = run.gate.requests.get_nowait()
            except queue.Empty:
                continue
            if live:
                live.stop()
            if interactive:
                console.print(Panel(Text(request.detail), title=f"{request.agent} requests {request.action}", border_style="yellow"))
                try:
                    request.allowed = Confirm.ask("Allow this action?", default=False, console=console)
                finally:
                    request.done.set()
            else:
                request.allowed = False
                request.done.set()
            if live:
                live.start()
    except KeyboardInterrupt:
        run.cancel.set()
        if live:
            live.stop()
        console.print("Cancelling; stopping active commands and waiting for inference to return…", style="yellow")
        # Do not exit and leave daemon threads mutating the workspace after cancellation.
        while not run.done.wait(0.2):
            try:
                request = run.gate.requests.get_nowait()
                request.allowed = False
                request.done.set()
            except queue.Empty:
                pass
    finally:
        if live:
            live.update(dashboard(run))
            live.stop()
    if not quiet:
        console.print()
        console.print(Markdown(run.answer))
        console.print(Text(f"\n{run.status} · {time.monotonic() - run.started:.1f}s · saved: {run.directory}", style="dim"))
    return run
