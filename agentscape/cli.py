from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from . import __version__
from .benchmark import benchmark
from .config import Config, ROOT, STATE, load_config, save_config
from .engine import Run
from .ollama import Ollama, OllamaError
from .ui import console, execute_run

COMMANDS = ["setup", "run", "workers", "agents", "mode", "routing", "model", "supervisor", "concurrency", "agent", "models", "pull", "stop",
            "config", "workspace", "permissions", "tools", "scope", "doctor", "bench", "runs", "show", "help", "exit", "continue"]


def ollama(config):
    return Ollama(config.ollama_url, config.inference_parallel, config.request_timeout)


def model_exists(config, name):
    if not any(m["name"] == name or m["name"] == name + ":latest" for m in ollama(config).tags()):
        raise ValueError(f"Model is not installed in Ollama: {name}. Use 'agentscape pull MODEL' first.")


def team_table(config):
    table = Table(box=None, expand=True)
    table.add_column("Agent", style="cyan")
    table.add_column("Ollama model")
    table.add_column("Role", max_width=44)
    if config.supervisor:
        table.add_row("supervisor", Text(config.supervisor), "Plan assignments + combine evidence")
    for worker in config.workers:
        table.add_row(Text(worker.name), Text(worker.model or config.model), Text(worker.role))
    return table


def show_config(config, raw=False):
    if raw:
        console.print_json(json.dumps(asdict(config)))
        return
    console.print(Panel(team_table(config), title="Your team", border_style="cyan"))
    console.print(Text(f"Workers: {config.mode} · inference slots: {config.inference_parallel} · context: {config.context} · protocol: {config.protocol}"))
    console.print(Text(f"Routing: {config.routing} · configured workers are available capacity"))
    console.print(Text(f"Commands/files: {config.permissions} · model phase unloading: {'on' if config.compact else 'off'} · thinking: {config.thinking}"))
    console.print(Text(f"Workspace: {config.workspace}\nOllama: {config.ollama_url}\nSettings: {STATE / 'config.json'}", style="dim"))


def help_text():
    console.print(Markdown("""# Agentscape
Enter a task to run your team. Use slash commands inside the environment, or `agentscape COMMAND` from your shell.

| Command | What it does |
|---|---|
| `/setup` | Guided setup: model, workers, scheduling, supervisor |
| `/workers 3` | Choose the number of workers (`/agents` also works) |
| `/mode parallel` | Run workers sequentially or in parallel |
| `/routing auto` | Choose direct answers, one agent, or a team; `single` and `team` force a route |
| `/model MODEL` | Set the default worker model |
| `/supervisor MODEL` | Set a supervisor; use `on` for the worker model or `off` |
| `/concurrency 1` | Limit simultaneous Ollama requests from this run |
| `/agent 2 --name reviewer --role "Check results" --model MODEL` | Customize one worker |
| `/models` | Show downloaded and currently loaded Ollama models |
| `/pull MODEL` | Download through Ollama |
| `/stop MODEL` | Unload a model from Ollama memory |
| `/workspace PATH` | Choose where agents read/write/run commands |
| `/permissions ask` | Execution policy: `ask`, `allow`, or `deny` |
| `/tools read,fetch,search` | Select tools; `all` or `none` also works |
| `/scope add example.com` | Restrict fetch hosts; `*.example.com` and CIDRs accepted |
| `/scope clear` | Remove fetch restrictions |
| `/config` | Show current configuration |
| `/config set context 8192` | Change context or other settings |
| `/doctor` | Check Ollama connectivity, models, tools, and paths |
| `/bench --workers 2 --supervisor MODEL` | Measure inference, tool calls, and team overhead |
| `/runs` | List saved runs |
| `/show latest` | Read a saved report |
| `/continue TASK` | Follow up using the previous run's report |
| `/exit` | Leave the environment |

Ctrl-C cancels a running task. Ctrl-D leaves the environment. Reports and tool evidence are saved automatically.
"""))


def choose_model(models, default, label="Worker model"):
    for i, model in enumerate(models, 1):
        console.print(Text(f"  {i}. {model['name']}"))
    while True:
        choice = Prompt.ask(label + " (number or full name)", default=default, console=console)
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]["name"]
        if any(m["name"] == choice for m in models):
            return choice
        console.print("Choose an installed model. Use /pull first to download another.", style="yellow")


def setup(config, quick=False):
    models = ollama(config).tags()
    if not models:
        raise ValueError("No models installed. Run 'agentscape pull MODEL', then setup again.")
    if quick:
        model_exists(config, config.model)
    else:
        console.print(Panel("Choose your team. Every model is served and managed by Ollama.", title="Agentscape setup", border_style="cyan"))
        config.model = choose_model(models, config.model)
        config.resize(IntPrompt.ask("How many workers? (1–16)", default=len(config.workers), console=console))
        config.mode = Prompt.ask("Worker scheduling", choices=["sequential", "parallel"], default=config.mode, console=console)
        config.routing = Prompt.ask("Task routing", choices=["auto", "single", "team"], default=config.routing, console=console)
        if Confirm.ask("Use a supervisor to plan and combine results?", default=bool(config.supervisor), console=console):
            config.supervisor = choose_model(models, config.supervisor or config.model, "Supervisor model")
        else:
            config.supervisor = ""
        config.inference_parallel = IntPrompt.ask("Simultaneous model requests (1 uses least memory)", default=config.inference_parallel, console=console)
        config.workspace = str(Path(Prompt.ask("Workspace", default=config.workspace, console=console)).expanduser().resolve())
        config.permissions = Prompt.ask("Shell and file-write permission", choices=["ask", "allow", "deny"], default=config.permissions, console=console)
    save_config(config)
    Path(config.workspace).mkdir(parents=True, exist_ok=True)
    console.print("Setup saved. Enter agentscape to start.", style="green")
    show_config(config)


def doctor(config):
    checks = []
    try:
        client = ollama(config)
        version = client.request("GET", "/api/version").get("version", "unknown")
        checks.append((True, f"Ollama connected: {version} at {config.ollama_url}"))
        for model in sorted({config.model, *[w.model for w in config.workers if w.model], *([config.supervisor] if config.supervisor else [])}):
            try:
                info = client.show(model)
                capabilities = info.get("capabilities", [])
                checks.append((True, f"Model installed: {model}; capabilities: {', '.join(capabilities)}"))
                if config.protocol == "native" and "tools" not in capabilities:
                    checks.append((False, "Native tool support not advertised; try config set protocol json"))
            except OllamaError as exc:
                checks.append((False, str(exc)))
    except OllamaError as exc:
        checks.append((False, f"Cannot reach Ollama. Start the Ollama app or run ollama serve. {exc}"))
    workspace = Path(config.workspace).expanduser()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        checks.append((os.access(workspace, os.W_OK), f"Workspace: {workspace}"))
    except OSError as exc:
        checks.append((False, f"Workspace error: {exc}"))
    checks.append((True, f"Agent mode: {config.mode}; local inference limit: {config.inference_parallel}"))
    checks.append((True, "Ollama server controls its own parallel slots. Raising the local limit alone does not change OLLAMA_NUM_PARALLEL."))
    checks.append((True, f"Tool policy: {config.permissions}. Host shell commands run as your user; this is not a sandbox."))
    for good, detail in checks:
        console.print(Text(("✓ " if good else "✗ ") + detail, style="green" if good else "red"))
    return all(good for good, _ in checks)


def list_models(config):
    client = ollama(config)
    loaded = {m["name"]: m for m in client.running()}
    table = Table(title="Ollama models")
    for title in ["Model", "Disk size", "Loaded", "Loaded / VRAM"]:
        table.add_column(title)
    for model in client.tags():
        running = loaded.get(model["name"])
        size = model.get("size", 0)
        memory = f"{running.get('size', 0) / 1e9:.2f} / {running.get('size_vram', 0) / 1e9:.2f} GB" if running else "—"
        table.add_row(Text(model["name"]), f"{size / 1e9:.2f} GB", "yes" if running else "no", memory)
    console.print(table)


def run_task(config, task, args=None):
    if args:
        if args.workers is not None:
            config.resize(args.workers)
        for key in ("mode", "routing", "model", "supervisor", "permissions", "workspace", "inference_parallel"):
            value = getattr(args, key, None)
            if value is not None:
                setattr(config, key, "" if key == "supervisor" and value == "off" else value)
    config.validate()
    if not task.strip():
        raise ValueError("Enter a task.")
    models = {w.model or config.model for w in config.workers}
    if config.supervisor:
        models.add(config.supervisor)
    for model in models:
        model_exists(config, model)
    run = execute_run(Run(config, task), quiet=bool(args and args.json))
    if args and args.json:
        print(json.dumps(run.summary(), ensure_ascii=False))
    return run


def saved_run(identifier):
    runs = sorted((STATE / "runs").glob("*/run.json"), reverse=True)
    if not runs:
        raise ValueError("No saved runs yet.")
    if identifier == "latest":
        return runs[0]
    matches = [p for p in runs if p.parent.name == identifier or p.parent.name.startswith(identifier)]
    if len(matches) != 1:
        raise ValueError("Run ID is missing or ambiguous. Use /runs to see IDs.")
    return matches[0]


def parser():
    result = argparse.ArgumentParser(prog="agentscape", description="A terminal team of Ollama agents. Run without arguments to enter.")
    result.add_argument("--version", action="version", version=__version__)
    sub = result.add_subparsers(dest="command")
    sub.add_parser("help")
    sub.add_parser("setup").add_argument("--quick", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("task", nargs="+")
    run.add_argument("--workers", type=int)
    run.add_argument("--mode", choices=["sequential", "parallel"])
    run.add_argument("--routing", choices=["auto", "single", "team"])
    run.add_argument("--model")
    run.add_argument("--supervisor")
    run.add_argument("--permissions", choices=["ask", "allow", "deny"])
    run.add_argument("--workspace")
    run.add_argument("--inference-parallel", type=int)
    run.add_argument("--json", action="store_true")
    for name in ["workers", "agents", "concurrency"]:
        sub.add_parser(name).add_argument("value", nargs="?", type=int)
    for name in ["mode", "routing", "model", "supervisor", "workspace", "permissions", "tools"]:
        sub.add_parser(name).add_argument("value", nargs="?")
    agent = sub.add_parser("agent")
    agent.add_argument("index", type=int, nargs="?")
    agent.add_argument("--name")
    agent.add_argument("--model")
    agent.add_argument("--role")
    sub.add_parser("models")
    sub.add_parser("pull").add_argument("model")
    sub.add_parser("stop").add_argument("model")
    conf = sub.add_parser("config")
    conf.add_argument("operation", nargs="?", choices=["set"])
    conf.add_argument("key", nargs="?")
    conf.add_argument("value", nargs="?")
    conf.add_argument("--json", action="store_true")
    scope = sub.add_parser("scope")
    scope.add_argument("operation", choices=["add", "remove", "clear"], nargs="?")
    scope.add_argument("host", nargs="?")
    sub.add_parser("doctor")
    bench = sub.add_parser("bench")
    bench.add_argument("--models", nargs="+")
    bench.add_argument("--workers", type=int, default=2)
    bench.add_argument("--supervisor", default="")
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--output", default=str(STATE / "benchmark.json"))
    sub.add_parser("runs")
    sub.add_parser("show").add_argument("id", nargs="?", default="latest")
    return result


def dispatch(args):
    config = load_config()
    command = args.command
    if command == "help":
        help_text()
    elif command == "setup":
        setup(config, args.quick)
    elif command == "run":
        run = run_task(config, " ".join(args.task), args)
        return 0 if run.status == "completed" else 1
    elif command in {"workers", "agents", "concurrency", "mode", "routing", "model", "supervisor", "workspace", "permissions", "tools"}:
        if args.value is None:
            show_config(config)
            return 0
        if command in {"workers", "agents"}:
            config.resize(args.value)
        elif command == "concurrency":
            config.inference_parallel = args.value
        elif command == "supervisor":
            config.supervisor = "" if args.value == "off" else config.model if args.value == "on" else args.value
            if config.supervisor:
                model_exists(config, config.supervisor)
        elif command == "model":
            model_exists(config, args.value)
            config.model = args.value
        elif command == "workspace":
            config.workspace = str(Path(args.value).expanduser().resolve())
        elif command == "tools":
            config.enabled_tools = ["shell", "fetch", "search", "read", "write"] if args.value == "all" else [] if args.value == "none" else args.value.split(",")
        else:
            setattr(config, command, args.value)
        save_config(config)
        show_config(config)
    elif command == "agent":
        if args.index is None:
            show_config(config)
            return 0
        if not 1 <= args.index <= len(config.workers):
            raise ValueError("Worker index out of range. Increase /workers first.")
        worker = config.workers[args.index - 1]
        for key in ("name", "model", "role"):
            value = getattr(args, key)
            if value is not None:
                if key == "model" and value not in {"default", "inherit"}:
                    model_exists(config, value)
                setattr(worker, key, "" if key == "model" and value in {"default", "inherit"} else value)
        save_config(config)
        show_config(config)
    elif command == "models":
        list_models(config)
    elif command == "pull":
        last = [""]
        def progress(item):
            status = item.get("status", "")
            percentage = int(100 * item.get("completed", 0) / item["total"]) if item.get("total") else None
            label = f"{status} {percentage}%" if percentage is not None else status
            if label != last[0]:
                console.print(Text(label))
                last[0] = label
        ollama(config).pull(args.model, progress)
    elif command == "stop":
        ollama(config).unload(args.model)
        console.print(Text(f"Unloaded {args.model}"))
    elif command == "config":
        if args.operation == "set":
            if not args.key or args.value is None:
                raise ValueError("Use config set KEY VALUE.")
            if args.key in {"workers", "agents"}:
                config.resize(int(args.value))
            else:
                if not hasattr(config, args.key) or args.key in {"enabled_tools", "scope"}:
                    raise ValueError("Unknown or list-valued setting. Use /config --json to see settings, /tools or /scope for lists.")
                current = getattr(config, args.key)
                if isinstance(current, bool):
                    if args.value.lower() not in {"true", "false", "on", "off"}:
                        raise ValueError("Use true/false or on/off.")
                    value = args.value.lower() in {"true", "on"}
                elif isinstance(current, int):
                    value = int(args.value)
                elif isinstance(current, float):
                    value = float(args.value)
                else:
                    value = args.value
                setattr(config, args.key, value)
            save_config(config)
        show_config(config, args.json)
    elif command == "scope":
        if args.operation == "clear":
            config.scope = []
        elif args.operation in {"add", "remove"}:
            if not args.host:
                raise ValueError("Supply a hostname, wildcard domain, IP, or CIDR.")
            if args.operation == "add" and args.host not in config.scope:
                config.scope.append(args.host)
            elif args.operation == "remove":
                config.scope.remove(args.host)
        save_config(config)
        console.print(Text("Fetch scope: " + (", ".join(config.scope) or "unrestricted")))
    elif command == "doctor":
        return 0 if doctor(config) else 1
    elif command == "bench":
        if not 1 <= args.workers <= 16 or not 1 <= args.repeats <= 10:
            raise ValueError("Benchmark workers must be 1–16, repeats 1–10.")
        models = args.models or [config.model]
        for model in [*models, *([args.supervisor] if args.supervisor else [])]:
            model_exists(config, model)
        result = benchmark(config, models, args.workers, args.supervisor, args.repeats, args.output, lambda text: console.print(Text(text, style="dim")))
        table = Table(title="Benchmark results")
        for title in ["Case", "Scheduling", "Seconds", "Check"]:
            table.add_column(title)
        for row in result["rows"]:
            case = row["kind"] + (" " + row["protocol"] if row.get("protocol") else "") + (" + sup" if row.get("supervisor") else "")
            table.add_row(case, row.get("mode", "—"), str(row["wall_seconds"]), "pass" if row["correct"] else "FAIL")
        console.print(table)
        console.print(Text(f"Saved: {args.output}\n{result['note']}", style="dim"))
        return 0 if all(r["correct"] for r in result["rows"]) else 1
    elif command == "runs":
        table = Table(title="Saved runs")
        for title in ["ID", "Status", "Seconds", "Task"]:
            table.add_column(title)
        for path in sorted((STATE / "runs").glob("*/run.json"), reverse=True)[:20]:
            run = json.loads(path.read_text())
            table.add_row(path.parent.name, run["status"], str(run["seconds"]), Text(run["task"][:90]))
        console.print(table)
    elif command == "show":
        path = saved_run(args.id)
        data = json.loads(path.read_text())
        console.print(Markdown(data["answer"]))
        console.print(Text(str(path.parent), style="dim"))
    return 0


def repl():
    STATE.mkdir(parents=True, exist_ok=True)
    config = load_config()
    if not (STATE / "config.json").exists():
        setup(config, quick=True)
    console.print(Panel(Text("A small terminal home for your Ollama agents.\nEnter a task · /setup configures your team · /help lists commands"),
                        title=f"Agentscape {__version__}", border_style="cyan"))
    show_config(config)
    session = PromptSession(history=FileHistory(str(STATE / "history")),
                            completer=WordCompleter(["/" + c for c in COMMANDS]), complete_while_typing=False)
    last = None
    while True:
        config = load_config()
        label = f"scape [{len(config.workers)} {config.mode[:3]}{' + sup' if config.supervisor else ''}] › "
        try:
            line = session.prompt(label).strip()
            if not line:
                continue
            if line in {"/exit", "/quit", "exit", "quit"}:
                break
            if line.startswith("/continue "):
                task = line[len("/continue "):]
                prior = last.summary() if last else json.loads(saved_run("latest").read_text())
                last = run_task(config, f"Previous task: {prior['task'][:2000]}\nPrevious result: {prior['answer'][:3500]}\n\nFollow-up task: {task}")
            elif line.startswith("/"):
                try:
                    parsed = parser().parse_args(shlex.split(line[1:]))
                    if parsed.command is None:
                        help_text()
                    else:
                        dispatch(parsed)
                except SystemExit:
                    pass
            else:
                last = run_task(config, line)
        except (KeyboardInterrupt, EOFError):
            if isinstance(sys.exc_info()[1], EOFError):
                break
            console.print("Use /exit to leave.", style="dim")
        except (ValueError, OSError, OllamaError) as exc:
            console.print(Text(str(exc), style="red"))


def main():
    try:
        args = parser().parse_args()
        if args.command is None:
            if not sys.stdin.isatty():
                console.print("Use agentscape run 'TASK' for noninteractive execution, or enter agentscape in a terminal.")
                return 2
            repl()
            return 0
        return dispatch(args)
    except (ValueError, OSError, OllamaError) as exc:
        console.print(Text(str(exc), style="red"))
        return 1
    except KeyboardInterrupt:
        console.print("Cancelled.", style="yellow")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
