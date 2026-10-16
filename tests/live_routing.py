"""Opt-in real-model integration diagnostics: python tests/live_routing.py.

Uses only the currently configured Ollama worker model; no downloads.
"""
import json
import re
import tempfile
from dataclasses import replace
from pathlib import Path

from agentscape.config import ROOT, load_config
from agentscape.engine import Run


def main():
    base = load_config()
    base.resize(4)
    base.supervisor = base.model
    base.mode, base.routing, base.permissions = "parallel", "auto", "allow"
    base.max_tokens, base.max_steps = 512, 6
    rows = []
    with tempfile.TemporaryDirectory(prefix="agentscape-routing-") as directory:
        base.workspace = directory
        root = Path(directory)
        (root / "frontend").mkdir()
        (root / "backend").mkdir()
        (root / "frontend/config.json").write_text('{"debug":true,"environment":"production"}')
        (root / "backend/config.json").write_text('{"port":70000,"timeout_seconds":-1}')
        cases = [
            ("greeting", "hello", "direct"),
            ("fact", "What is the capital of France? Answer briefly.", "direct"),
            ("focused_tool", "Use shell to run exactly printf 'ROUTE_SINGLE_OK\\n'. Report the actual output briefly.", "single"),
            ("short_file_check", "Read frontend/config.json and backend/config.json and briefly list their keys. Do not change anything.", "single"),
            ("parallel_audits", "Perform two independent audits in parallel: (1) frontend/config.json, checking the rule that debug must be false in production; "
             "(2) backend/config.json, checking that the port must be between 1 and 65535 and timeout_seconds must be nonnegative. "
             "Keep the investigations separate and use actual file contents. Combine the findings concisely. "
             "This is an explicit request for separate parallel independent investigations. Read files only; do not change anything.", "team"),
        ]
        for name, task, expected in cases:
            config = replace(base, enabled_tools=["shell"] if name == "focused_tool" else ["read"])
            run = Run(config, task).start()
            run.done.wait()
            result = run.summary()
            passed = run.status == "completed" and run.route["mode"] == expected
            passed &= not bool(re.search(r"worker[-_ ]?\d+|supervisor", run.answer, re.I))
            if expected == "direct":
                passed &= len(run.calls) == 1 and run.tools.count == 0
            elif expected == "single":
                passed &= len(run.reports) == 1 and run.tools.count >= 1
                if name == "focused_tool":
                    passed &= "ROUTE_SINGLE_OK" in run.answer
            else:
                passed &= len(run.reports) == 2 and run.tools.count >= 2
                passed &= "70000" in run.answer and "debug" in run.answer.lower()
                passed &= not bool(re.search(r"70000.{0,30}\(valid\)", run.answer, re.I))
            row = {"case": name, "passed": bool(passed), "route": run.route, "status": run.status,
                   "wall_seconds": result["seconds"], "calls": len(run.calls), "reports": len(run.reports),
                   "tool_calls": run.tools.count, "output_tokens": sum(c["output_tokens"] for c in run.calls),
                   "answer": run.answer, "run_id": run.id, "model": base.model}
            rows.append(row)
            print(json.dumps(row), flush=True)
    output = ROOT / "benchmarks/routing.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
