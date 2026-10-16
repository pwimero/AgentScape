"""Reproducible before/after probes; run from the repository root.

python benchmarks/efficiency.py --output /tmp/agentscape-before.json --live
Uses synthetic logs and fixed arithmetic, not private workspace contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentscape.config import Config, load_config
from agentscape.engine import Run
from agentscape.ollama import Ollama, metrics


def event_probe(count=5000, size=8192):
    with tempfile.TemporaryDirectory(prefix="agentscape-events-") as scratch:
        root = Path(scratch)
        run = Run(Config(workspace=str(root / "workspace")), "Synthetic event memory probe", state_dir=root)
        expected = hashlib.sha256()
        tracemalloc.start()
        start = time.perf_counter()
        for i in range(count):
            payload = f"{i:08d}:" + "x" * size
            expected.update(payload.encode())
            run.event("system", "probe", payload=payload)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        write_seconds = time.perf_counter() - start
        actual = hashlib.sha256()
        for event in run.events:
            actual.update(event["payload"].encode())
        source_bytes = (run.directory / "events.jsonl").stat().st_size
        archive_seconds = None
        if hasattr(run.events, "archive"):
            start = time.perf_counter()
            run.events.archive()
            archive_seconds = time.perf_counter() - start
            # Check that archiving also leaves the complete journal iterable.
            archived = hashlib.sha256()
            for event in run.events:
                archived.update(event["payload"].encode())
            assert archived.digest() == expected.digest()
        stored_bytes = sum(p.stat().st_size for p in run.directory.glob("events.jsonl*"))
        return {"events": count, "payload_characters": size, "retained_python_bytes": current,
                "peak_python_bytes": peak, "write_seconds": write_seconds,
                "source_bytes": source_bytes, "stored_bytes": stored_bytes,
                "archive_seconds": archive_seconds, "all_payloads_preserved": actual.digest() == expected.digest(),
                "note": "Synthetic repetitive payloads test retention and lossless round-trip; compression ratio is not representative of arbitrary tool output."}


def live_probe(repeats):
    config = replace(load_config(), temperature=0, max_tokens=64, thinking=False)
    client = Ollama(config.ollama_url, 1, config.request_timeout)
    client.unload(config.model)
    rows = []
    for i in range(repeats + 1):
        first = []
        started = time.perf_counter()
        response = client.chat(config.model, [{"role": "user", "content":
            "Calculate 17 times 19. Return only the decimal result, with no explanation."}],
            config, on_token=lambda token: first.append(time.perf_counter()) if token and not first else None)
        answer = response["message"]["content"].strip()
        try:
            correct = Decimal(answer) == Decimal(323)
        except InvalidOperation:
            correct = False
        rows.append({"phase": "cold" if i == 0 else "warm", **metrics(response),
                     "time_to_first_content_seconds": first[0] - started if first else None,
                     "correct": correct, "answer": answer})
    loaded = [{k: m.get(k) for k in ("name", "size", "size_vram", "context_length")} for m in client.running()]
    return {"server_version": client.request("GET", "/api/version").get("version"),
            "model": config.model, "context": config.context, "rows": rows, "loaded_models": loaded,
            "note": "Fixed greedy arithmetic checks output compatibility; not a general agent quality evaluation. The first request unloads this model to measure a cold start."}


def residency_probe():
    base = replace(load_config(), temperature=0, routing="single", supervisor="", max_steps=3,
                   max_tokens=128, enabled_tools=[], keep_alive=600)
    client = Ollama(base.ollama_url, 1, base.request_timeout)
    client.unload(base.model)
    rows = []
    with tempfile.TemporaryDirectory(prefix="agentscape-residency-") as scratch:
        root = Path(scratch)
        for unload in (False, True, False, True):
            config = replace(base, workers=base.workers[:1], workspace=str(root / "workspace"), unload_after_run=unload)
            run = Run(config, "Calculate 17 times 19. Return only the numeric answer.", state_dir=root).start()
            if not run.done.wait(120):
                run.cancel.set()
                raise RuntimeError("Residency probe exceeded 120 seconds.")
            deadline = time.monotonic() + 5
            while True:
                loaded = [m for m in client.running() if m.get("name") in {base.model, base.model + ":latest"}]
                if not unload or not loaded or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            try:
                correct = Decimal(run.answer.strip()) == Decimal(323)
            except InvalidOperation:
                correct = False
            rows.append({"unload_after_run": unload, "status": run.status, "correct": correct,
                         "answer": run.answer, "wall_seconds": run.summary()["seconds"],
                         "post_run_loaded_bytes": sum(m["size"] for m in loaded), "calls": run.calls})
    return {"server_version": client.request("GET", "/api/version").get("version"), "rows": rows, "note": "Alternating resident/unload-after-run policy, same model and greedy prompt. Loaded bytes are Ollama allocation reports, not measured total system memory. Unloading can affect other clients using the same server."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", type=int, default=5000)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--residency", action="store_true", help="Compare resident and unload-after-run policies on a live model")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.events <= 100000 or not 1 <= args.repeats <= 20:
        parser.error("events must be 1–100000 and repeats 1–20")
    result = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "platform": platform.platform(), "python": platform.python_version(),
              "engine_sha256": hashlib.sha256(Path(sys.modules[Run.__module__].__file__).read_bytes()).hexdigest(),
              "events": event_probe(args.events)}
    if args.live:
        result["live"] = live_probe(args.repeats)
    if args.residency:
        result["residency"] = residency_probe()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["events"]["all_payloads_preserved"]:
        raise SystemExit(1)
    if args.live and not all(r["correct"] for r in result["live"]["rows"]):
        raise SystemExit(1)
    if args.residency and not all(r["correct"] and r["status"] == "completed" and
            (r["post_run_loaded_bytes"] == 0 if r["unload_after_run"] else r["post_run_loaded_bytes"] > 0)
            for r in result["residency"]["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
