"""Small reproducible diagnostics, not a general model intelligence ranking."""
from __future__ import annotations

import json
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .config import Worker
from .engine import Run
from .ollama import Ollama, metrics
from .profiling import ResourceSampler


PROMPT = "Calculate 17 times 19. Return only the decimal result, with no explanation."


def arithmetic_correct(answer):
    try:
        return Decimal(answer.strip()) == Decimal(323)
    except InvalidOperation:
        return False


def _benchmark(config, models, workers=2, supervisor="", repeats=1, progress=lambda _: None):
    rows = []
    client = Ollama(config.ollama_url, config.inference_parallel, config.request_timeout)
    config = replace(config, max_tokens=256, thinking=False, routing="team")
    with tempfile.TemporaryDirectory(prefix="agentscape-bench-") as scratch:
        root = Path(scratch)
        for model in models:
            progress(f"Warming {model} (excluded from inference timing)")
            client.chat(model, [{"role": "user", "content": "Reply OK."}], config)
            for mode in ("sequential", "parallel"):
                for repetition in range(repeats):
                    progress(f"{model}: {mode}, {workers} requests, repeat {repetition + 1}")
                    labels = [uuid.uuid4().hex[:8] for _ in range(workers)]
                    def sample(i):
                        response = client.chat(model, [{"role": "user", "content": PROMPT + f"\nTest ID (do not include in answer): {labels[i]}"}], config)
                        return {**metrics(response), "correct": arithmetic_correct(response["message"]["content"]),
                                "answer": response["message"]["content"]}
                    start = time.monotonic()
                    if mode == "parallel":
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            samples = list(pool.map(sample, range(workers)))
                    else:
                        samples = [sample(i) for i in range(workers)]
                    loaded = client.running()
                    rows.append({"kind": "inference", "model": model, "mode": mode, "workers": workers,
                                 "inference_slots": config.inference_parallel, "repeat": repetition + 1,
                                 "wall_seconds": round(time.monotonic() - start, 3), "samples": samples,
                                 "correct": all(s["correct"] for s in samples),
                                 "loaded_models": [{k: m.get(k) for k in ("name", "size", "size_vram", "context_length")} for m in loaded]})
            tool_config = replace(config, model=model, workers=[Worker("worker-1", role="Execute the requested command and report its output.")],
                                  supervisor="", workspace=str(root / "workspace"), permissions="allow", max_steps=4)
            for protocol in ("json", "native"):
                progress(f"{model}: {protocol} tool execution smoke test")
                run = Run(replace(tool_config, protocol=protocol), "Use the shell tool to run exactly: printf 'AGENTSCAPE_TOOL_OK\\n'. Then report the actual output.",
                          client=client, state_dir=root / "state").start()
                run.done.wait()
                executed = [e for e in run.events if e["kind"] == "tool_result" and e.get("tool") == "shell"]
                rows.append({"kind": "tool", "model": model, "protocol": protocol, "wall_seconds": round(time.monotonic() - run.started, 3),
                             "status": run.status, "correct": run.status == "completed" and any(e["result"].get("ok") and "AGENTSCAPE_TOOL_OK" in e["result"].get("output", "") for e in executed),
                             "calls": run.calls, "answer": run.answer})
            for lead_model in (["", supervisor] if supervisor else [""]):
                for mode in ("sequential", "parallel"):
                    progress(f"{model}: {workers} workers, supervisor {lead_model or 'off'} ({mode})")
                    team_config = replace(tool_config, workers=[Worker(f"worker-{i + 1}", role="Independently verify the arithmetic.") for i in range(workers)],
                                          mode=mode, supervisor=lead_model, max_tokens=512)
                    team = Run(team_config, "Independently check 17 times 19 and report the answer concisely. No tools are needed.",
                               client=client, state_dir=root / "state").start()
                    team.done.wait()
                    rows.append({"kind": "team", "model": model, "supervisor": lead_model, "mode": mode,
                                 "workers": workers, "wall_seconds": round(time.monotonic() - team.started, 3),
                                 "status": team.status, "correct": team.status == "completed" and "323" in team.answer,
                                 "calls": team.calls, "answer": team.answer})
    result = {"note": "Simple arithmetic and tool smoke tests. Warm-up excluded from inference rows. Team rows include planning and synthesis. "
                       "Loaded size/VRAM is an end-of-case Ollama snapshot, not peak system memory. Parallel workers do not imply parallel inference.",
              "context": config.context, "inference_slots": config.inference_parallel,
              "keep_alive_seconds": config.keep_alive, "unload_after_run": config.unload_after_run, "rows": rows}
    return result


def benchmark(config, models, workers=2, supervisor="", repeats=1, output=None, progress=lambda _: None, *, profile=False):
    sampler = ResourceSampler() if profile else None
    with (sampler if sampler is not None else nullcontext()):
        result = _benchmark(config, models, workers, supervisor, repeats, progress)
    if sampler is not None:
        result["resources"] = sampler.summary()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2))
    return result
