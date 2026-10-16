from __future__ import annotations

import ipaddress
import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .ollama import Cancelled

DESCRIPTIONS = {
    "shell": "Run a shell command in the workspace. args: command (string), timeout (optional seconds).",
    "fetch": "Fetch an HTTP(S) page as text. args: url (string).",
    "search": "Search the web. args: query (string). Returns links; use fetch to read a source.",
    "read": "Read a workspace file or list a directory. args: path (string, '.' for root), offset (optional character index).",
    "write": "Write a UTF-8 workspace file. args: path (string), content (string).",
    "result": "Read evidence from this run, including other workers. args: id (string: a result ID, 'list', or 'latest-WORKER_NAME' to wait for a worker's evidence), offset (optional character index).",
}

PARAMETERS = {
    "shell": ({"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]),
    "fetch": ({"url": {"type": "string"}}, ["url"]),
    "search": ({"query": {"type": "string"}}, ["query"]),
    "read": ({"path": {"type": "string"}, "offset": {"type": "integer"}}, ["path"]),
    "write": ({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    "result": ({"id": {"type": "string"}, "offset": {"type": "integer"}}, ["id"]),
}


def action_schema(enabled):
    branches = []
    for name in [*enabled, "result", "finish"]:
        properties, required = PARAMETERS[name] if name != "finish" else ({}, [])
        branches.append({"type": "object", "properties": {
            "action": {"type": "string", "enum": [name]},
            "args": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
            "answer": {"type": "string"},
        }, "required": ["action", "args", "answer"], "additionalProperties": False})
    return {"anyOf": branches}


def native_tools(enabled):
    return [{"type": "function", "function": {
        "name": name, "description": DESCRIPTIONS[name],
        "parameters": {"type": "object", "properties": PARAMETERS[name][0], "required": PARAMETERS[name][1], "additionalProperties": False},
    }} for name in [*enabled, "result"]]


@dataclass
class Approval:
    agent: str
    action: str
    detail: str
    done: threading.Event = field(default_factory=threading.Event)
    allowed: bool = False


class Gate:
    def __init__(self, policy, cancel):
        self.policy, self.cancel = policy, cancel
        self.requests = queue.Queue()

    def approve(self, agent, action, detail):
        if self.policy == "allow":
            return True
        if self.policy == "deny":
            return False
        request = Approval(agent, action, detail)
        self.requests.put(request)
        while not request.done.wait(0.1):
            if self.cancel.is_set():
                raise Cancelled("Cancelled while waiting for command approval.")
        return request.allowed


class Tools:
    def __init__(self, config, run_dir, gate, cancel, event=lambda *a, **kw: None):
        self.config, self.run_dir, self.gate, self.cancel, self.event = config, run_dir, gate, cancel, event
        self.workspace = Path(config.workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.result_dir = run_dir / "evidence"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.count = 0
        self.available = {}

    def path(self, value):
        if not isinstance(value, str):
            raise ValueError("path must be a string")
        target = (self.workspace / value).resolve()
        if not target.is_relative_to(self.workspace):
            raise ValueError("File path must stay inside the workspace.")
        return target

    def allowed_url(self, url):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Use an HTTP(S) URL without embedded credentials.")
        if not self.config.scope:
            return
        host = parsed.hostname.lower().rstrip(".")
        for entry in self.config.scope:
            entry = entry.lower().rstrip(".")
            try:
                if ipaddress.ip_address(host) in ipaddress.ip_network(entry, strict=False):
                    return
            except ValueError:
                if host == entry or (entry.startswith("*.") and host.endswith(entry[1:]) and host != entry[2:]):
                    return
        raise ValueError(f"{host} is outside the configured fetch scope.")

    def _fetch(self, url):
        # Scope is checked for every redirect, not just the first URL.
        with httpx.Client(timeout=20, follow_redirects=False, headers={"User-Agent": "Agentscape/0.1"}) as client:
            for _ in range(6):
                self.allowed_url(url)
                with client.stream("GET", url) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers.get("location", ""))
                        continue
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        if self.cancel.is_set():
                            raise Cancelled("Cancelled during page fetch.")
                        size += len(chunk)
                        if size > 2_000_000:
                            raise ValueError("Page exceeds the 2 MB fetch limit.")
                        chunks.append(chunk)
                    raw = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                    if "html" in response.headers.get("content-type", "").lower():
                        soup = BeautifulSoup(raw, "html.parser")
                        for tag in soup(["script", "style", "nav", "noscript"]):
                            tag.decompose()
                        raw = soup.get_text("\n", strip=True)
                    return f"URL: {url}\nHTTP status: {response.status_code}\n\n{raw}"
        raise ValueError("Too many redirects.")

    def _search(self, query):
        if not isinstance(query, str) or not 1 <= len(query) <= 1000:
            raise ValueError("query must be a nonempty string of at most 1000 characters")
        # Optional official search service; inference remains on the configured Ollama server.
        api_key = os.environ.get("OLLAMA_API_KEY")
        if api_key:
            response = httpx.post("https://ollama.com/api/web_search", json={"query": query, "max_results": 5},
                                  headers={"Authorization": f"Bearer {api_key}"}, timeout=25)
            response.raise_for_status()
            return json.dumps(response.json(), ensure_ascii=False)
        results = []
        errors = []
        for endpoint in ("https://lite.duckduckgo.com/lite/", "https://html.duckduckgo.com/html/"):
            try:
                response = httpx.get(endpoint, params={"q": query, "kp": "-2"},
                                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20, follow_redirects=True)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.select("a.result-link, .result__a"):
                    href = urljoin(str(response.url), link.get("href", ""))
                    params = parse_qs(urlparse(href).query)
                    if "uddg" in params:
                        href = unquote(params["uddg"][0])
                    parent = link.find_parent(class_="result")
                    snippet = parent.select_one(".result__snippet") if parent else None
                    if snippet is None and link.find_parent("tr"):
                        snippet = link.find_parent("tr").find_next(class_="result-snippet")
                    results.append({"title": link.get_text(" ", strip=True), "url": href,
                                    "snippet": snippet.get_text(" ", strip=True) if snippet else ""})
                    if len(results) == 5:
                        break
                if results:
                    break
            except httpx.HTTPError as exc:
                errors.append(str(exc))
        if not results:
            raise ValueError("Search provider returned no results or a bot challenge. Fetch a known URL directly, or set OLLAMA_API_KEY to use Ollama web search. " + "; ".join(errors))
        return json.dumps(results, ensure_ascii=False)

    def _shell(self, agent, args, raw_path):
        command = args.get("command")
        if not isinstance(command, str) or not command.strip() or len(command) > 16000:
            raise ValueError("command must be a nonempty string of at most 16000 characters")
        timeout = min(int(args.get("timeout", self.config.command_timeout)), self.config.command_timeout)
        if timeout < 1:
            raise ValueError("timeout must be positive")
        self.event(agent, "approval", detail=command[:120])
        if not self.gate.approve(agent, "shell", command):
            return {"ok": False, "error": "Command execution denied by the configured permission policy."}
        self.event(agent, "tool", detail="shell: " + command[:100])
        start = time.monotonic()
        stopped = ""
        with raw_path.open("wb") as output:
            process = subprocess.Popen(command, shell=True, executable="/bin/sh", cwd=self.workspace,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                while process.poll() is None:
                    if self.cancel.is_set() or time.monotonic() - start > timeout or raw_path.stat().st_size > 2_000_000:
                        stopped = "cancelled" if self.cancel.is_set() else "timeout or output limit reached"
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                        break
                    time.sleep(0.05)
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
        if self.cancel.is_set():
            raise Cancelled("Cancelled during command execution.")
        with raw_path.open("rb") as stream:
            output = stream.read(2_000_000).decode("utf-8", errors="replace")
        return {"ok": process.returncode == 0 and not stopped, "exit_code": process.returncode,
                "seconds": round(time.monotonic() - start, 3), "error": stopped,
                "output": output[:5000], "truncated": len(output) > 5000}

    def execute(self, agent, name, args):
        if self.cancel.is_set():
            raise Cancelled("Cancelled before tool execution.")
        with self.lock:
            self.count += 1
            result_id = f"{self.count:04d}-{agent}"
        raw_path = self.result_dir / (result_id + ".txt")
        self.event(agent, "tool", detail=name)
        try:
            if name not in [*self.config.enabled_tools, "result"]:
                raise ValueError(f"Tool '{name}' is disabled or unknown.")
            if not isinstance(args, dict):
                raise ValueError("Tool args must be an object.")
            missing = [key for key in PARAMETERS[name][1] if key not in args]
            if missing:
                raise ValueError(f"{name} requires args: {', '.join(missing)}. {DESCRIPTIONS[name]}")
            if name == "shell":
                result = self._shell(agent, args, raw_path)
            else:
                if name == "fetch":
                    raw = self._fetch(args["url"])
                elif name == "search":
                    raw = self._search(args["query"])
                elif name == "read":
                    target = self.path(args.get("path", "."))
                    if target.is_dir():
                        raw = "\n".join(p.name + ("/" if p.is_dir() else "") for p in sorted(target.iterdir()))
                    else:
                        if target.stat().st_size > 2_000_000:
                            raise ValueError("File exceeds the 2 MB read limit; use shell to inspect a section.")
                        raw = target.read_text(errors="replace")
                elif name == "write":
                    target, content = self.path(args["path"]), args["content"]
                    if not isinstance(content, str) or len(content.encode()) > 256_000:
                        raise ValueError("content must be a string of at most 256 KB")
                    self.event(agent, "approval", detail="write " + str(target))
                    if not self.gate.approve(agent, "write", f"{target}\n\n{content[:2000]}"):
                        raise ValueError("File write denied by permission policy.")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content)
                    raw = f"Wrote {len(content.encode())} bytes to {target.relative_to(self.workspace)}"
                elif name == "result":
                    result_id_requested = args["id"]
                    if not isinstance(result_id_requested, str):
                        raise ValueError("id must be a string")
                    if result_id_requested == "list":
                        with self.lock:
                            raw = json.dumps(self.available, ensure_ascii=False)
                    else:
                        if result_id_requested.startswith("latest-"):
                            worker = result_id_requested[len("latest-"):]
                            if worker not in {w.name for w in self.config.workers}:
                                raise ValueError("Unknown worker name. Use the roster name, such as latest-worker-1.")
                            deadline = time.monotonic() + self.config.command_timeout
                            while True:
                                if self.cancel.is_set():
                                    raise Cancelled("Cancelled while waiting for worker evidence.")
                                with self.lock:
                                    found = [key for key, value in self.available.items() if value["agent"] == worker and value["tool"] != "result"]
                                if found:
                                    result_id_requested = found[-1]
                                    break
                                if time.monotonic() >= deadline:
                                    raise ValueError(f"No completed evidence from {worker} yet. Try result id=list, or use sequential scheduling for dependent tasks.")
                                time.sleep(0.1)
                        target = (self.result_dir / (result_id_requested + ".txt")).resolve()
                        if target.parent != self.result_dir.resolve():
                            raise ValueError("Invalid result id.")
                        with self.lock:
                            if result_id_requested not in self.available:
                                raise ValueError("Unknown or pending result ID. Use result with id='list' or id='latest-WORKER_NAME'.")
                        raw = target.read_text(errors="replace")
                offset = int(args.get("offset", 0))
                if offset < 0:
                    raise ValueError("offset must be nonnegative")
                raw_path.write_text(raw)
                result = {"ok": True, "output": raw[offset:offset + 5000], "offset": offset,
                          "total_characters": len(raw), "truncated": len(raw) > offset + 5000}
        except Cancelled:
            raise
        except (ValueError, TypeError, KeyError, OSError, httpx.HTTPError) as exc:
            result = {"ok": False, "error": str(exc)}
        if not raw_path.exists():
            raw_path.write_text(json.dumps(result, ensure_ascii=False))
        result["result_id"] = result_id
        with self.lock:
            self.available[result_id] = {"agent": agent, "tool": name, "ok": result["ok"]}
        self.event(agent, "tool_result", detail=f"{name}: {'ok' if result['ok'] else result.get('error', 'failed')}",
                   tool=name, args=args, result=result)
        return result
