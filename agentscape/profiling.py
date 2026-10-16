"""Optional sampled local memory telemetry; no third-party dependency."""
from __future__ import annotations

import os
import platform
import re
import subprocess
import threading
from pathlib import Path


def swap_used_bytes():
    try:
        if platform.system() == "Darwin":
            output = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, timeout=2)
            match = re.search(r"used\s*=\s*([\d.]+)([KMG])", output)
            return int(float(match[1]) * {"K": 1024, "M": 1024**2, "G": 1024**3}[match[2]]) if match else None
        if platform.system() == "Linux":
            values = dict(re.findall(r"^(SwapTotal|SwapFree):\s+(\d+)", Path("/proc/meminfo").read_text(), re.M))
            return (int(values["SwapTotal"]) - int(values["SwapFree"])) * 1024
    except (OSError, KeyError, subprocess.SubprocessError):
        pass
    return None


class ResourceSampler:
    def __init__(self, interval=0.5):
        if interval <= 0:
            raise ValueError("Sampling interval must be positive.")
        self.interval = interval
        self._stop = threading.Event()
        self.samples = 0
        self.peak_python_rss_bytes = None
        self.peak_ollama_rss_bytes = None
        self.errors = []

    def sample(self):
        try:
            output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss=,comm="], text=True, timeout=2)
            own = None
            processes = {}
            roots = set()
            for line in output.splitlines():
                fields = line.strip().split(None, 3)
                if len(fields) != 4:
                    continue
                pid, parent, rss, executable = fields
                pid, parent, size = int(pid), int(parent), int(rss) * 1024
                processes[pid] = (parent, size)
                if pid == os.getpid():
                    own = size
                if Path(executable).name.lower() == "ollama":
                    roots.add(pid)
            # Runners can be named llama-server; track ancestry rather than
            # assuming every runner executable is called ollama.
            while True:
                descendants = {pid for pid, (parent, _) in processes.items() if parent in roots}
                expanded = roots | descendants
                if expanded == roots:
                    break
                roots = expanded
            ollama = sum(processes[pid][1] for pid in roots)
            if own is not None:
                self.peak_python_rss_bytes = max(self.peak_python_rss_bytes or 0, own)
            self.peak_ollama_rss_bytes = max(self.peak_ollama_rss_bytes or 0, ollama)
            self.samples += 1
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            if not self.errors:
                self.errors.append(str(exc))

    def _watch(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.swap_before = swap_used_bytes()
        self.sample()
        self.thread = threading.Thread(target=self._watch, name="agentscape-memory-sampler", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()
        self.sample()
        self.swap_after = swap_used_bytes()

    def summary(self):
        return {"interval_seconds": self.interval, "samples": self.samples,
                "peak_python_rss_bytes": self.peak_python_rss_bytes,
                "peak_local_ollama_rss_bytes": self.peak_ollama_rss_bytes,
                "system_swap_before_bytes": self.swap_before, "system_swap_after_bytes": self.swap_after,
                "system_swap_growth_bytes": (self.swap_after - self.swap_before
                                             if self.swap_after is not None and self.swap_before is not None else None),
                "errors": self.errors,
                "note": "Sampled process RSS, not exact peak unified/GPU memory. Ollama RSS sums all local Ollama process trees and can double-count shared pages. Swap is system-wide and affected by other apps; remote inference memory is not measured."}
