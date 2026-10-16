from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("AGENTSCAPE_HOME", Path(__file__).resolve().parent.parent)).expanduser().resolve()
STATE = ROOT / ".agentscape"
ROLES = [
    "Investigate the task and gather evidence. State what is known and what still needs checking.",
    "Carry out the task using concrete checks or changes. Verify your results with evidence.",
    "Review the task critically, check assumptions and results, and identify remaining issues.",
]


@dataclass
class Worker:
    name: str
    model: str = ""  # Empty inherits the team model.
    role: str = ""


@dataclass
class Config:
    model: str = "hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M"
    workers: list[Worker] = field(default_factory=lambda: [Worker("worker-1", role=ROLES[0])])
    mode: str = "sequential"
    routing: str = "auto"
    supervisor: str = ""
    ollama_url: str = "http://localhost:11434"
    inference_parallel: int = 1
    context: int = 4096
    max_steps: int = 12
    max_tokens: int = 768
    temperature: float = 0.2
    protocol: str = "json"
    workspace: str = str(ROOT / "workspace")
    permissions: str = "ask"
    command_timeout: int = 60
    request_timeout: int = 180
    scope: list[str] = field(default_factory=list)
    compact: bool = True
    keep_alive: int = 600
    unload_after_run: bool = False
    thinking: bool = False
    enabled_tools: list[str] = field(default_factory=lambda: ["shell", "fetch", "search", "read", "write"])

    def validate(self):
        if not 1 <= len(self.workers) <= 16:
            raise ValueError("Choose between 1 and 16 workers.")
        if self.mode not in {"sequential", "parallel"}:
            raise ValueError("Mode must be sequential or parallel.")
        if self.routing not in {"auto", "single", "team"}:
            raise ValueError("Routing must be auto, single, or team.")
        if self.permissions not in {"ask", "allow", "deny"}:
            raise ValueError("Permissions must be ask, allow, or deny.")
        if self.protocol not in {"json", "native"}:
            raise ValueError("Protocol must be json or native.")
        if not 1 <= self.inference_parallel <= 16:
            raise ValueError("Inference concurrency must be between 1 and 16.")
        if not 512 <= self.context <= 131072:
            raise ValueError("Context must be between 512 and 131072 tokens.")
        if not 1 <= self.max_steps <= 100 or not 64 <= self.max_tokens <= 8192:
            raise ValueError("Steps must be 1–100 and output tokens 64–8192.")
        if not 1 <= self.command_timeout <= 3600 or not 1 <= self.request_timeout <= 3600:
            raise ValueError("Timeouts must be 1–3600 seconds.")
        if type(self.keep_alive) is not int or not -1 <= self.keep_alive <= 86400:
            raise ValueError("Model keep-alive must be -1 (indefinite), 0, or up to 86400 seconds.")
        if type(self.unload_after_run) is not bool:
            raise ValueError("unload_after_run must be true or false.")
        if not self.model.strip():
            raise ValueError("A worker model is required.")
        names = [w.name for w in self.workers]
        if len(names) != len(set(names)) or any(not n or n in {"system", "supervisor"} or not all(c.isalnum() or c in "-_" for c in n) for n in names):
            raise ValueError("Worker names must be unique and contain only letters, numbers, - or _.")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if set(self.enabled_tools) - {"shell", "fetch", "search", "read", "write"}:
            raise ValueError("Unknown tool enabled.")
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must start with http:// or https://.")

    def resize(self, n: int):
        if not 1 <= n <= 16:
            raise ValueError("Choose between 1 and 16 workers.")
        self.workers = self.workers[:n]
        while len(self.workers) < n:
            i = len(self.workers)
            name = f"worker-{i + 1}"
            while name in {w.name for w in self.workers}:
                name += "-new"
            self.workers.append(Worker(name, role=ROLES[i % len(ROLES)]))


def load_config(path: Path | None = None) -> Config:
    path = path or STATE / "config.json"
    if not path.exists():
        return Config()
    data = json.loads(path.read_text())
    data["workers"] = [Worker(**w) for w in data.get("workers", [asdict(Worker("worker-1", role=ROLES[0]))])]
    config = Config(**data)
    config.validate()
    return config


def save_config(config: Config, path: Path | None = None):
    config.validate()
    path = path or STATE / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(asdict(config), indent=2) + "\n")
    temp.chmod(0o600)
    temp.replace(path)
