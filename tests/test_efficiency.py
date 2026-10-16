import gzip
import json
import stat
import tempfile
import threading
import tracemalloc
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from agentscape.config import Config
from agentscape.engine import Run
from agentscape.events import EventJournal
from agentscape.ollama import Ollama, metrics
from agentscape.profiling import ResourceSampler, swap_used_bytes
from test_agentscape import FakeModel


class EventTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_bounded_memory_keeps_full_ordered_history_and_slicing(self):
        journal = EventJournal(self.root, capacity=5)
        entries = [{"n": i, "payload": "🌍" * 1000} for i in range(30)]
        for entry in entries:
            journal.append(entry)
        self.assertEqual(len(journal._tail), 5)
        self.assertEqual(len(journal), 30)
        self.assertEqual(list(journal), entries)
        for key in [slice(-5, None), slice(2, 20, 3), slice(None, None, -1), slice(25, 2), slice(0, 0)]:
            self.assertEqual(journal[key], entries[key])
        self.assertEqual(journal[0], entries[0])
        self.assertEqual(journal[-1], entries[-1])
        with self.assertRaises(IndexError):
            journal[30]
        journal.archive()
        self.assertFalse(journal.path.exists())
        self.assertEqual(list(journal), entries)
        with gzip.open(journal.compressed_path, "rt") as source:
            self.assertEqual([json.loads(x) for x in source], entries)
        self.assertEqual(stat.S_IMODE(journal.compressed_path.stat().st_mode), 0o600)
        journal.archive()  # idempotent
        with self.assertRaises(ValueError):
            journal.append({"n": 31})

    def test_run_retains_under_one_mib_after_eight_mib_of_event_payloads(self):
        run = Run(Config(workspace=str(self.root / "workspace")), "memory regression", state_dir=self.root)
        tracemalloc.start()
        try:
            for i in range(1024):
                run.event("system", "probe", payload=f"{i}:" + "x" * 8192)
            retained, _ = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(retained, 1024**2, "Event retention must stay bounded rather than growing with history.")
        self.assertEqual(len(run.events), 1024)
        self.assertEqual(sum(1 for event in run.events if event["payload"].endswith("x" * 8192)), 1024)

    def test_open_stream_survives_archiving_without_losing_events(self):
        journal = EventJournal(self.root, capacity=2)
        for i in range(20):
            journal.append({"n": i})
        stream = iter(journal)
        self.assertEqual(next(stream), {"n": 0})
        journal.archive()
        self.assertEqual(list(stream), [{"n": i} for i in range(1, 20)])

    def test_archive_failure_preserves_complete_original(self):
        journal = EventJournal(self.root)
        journal.append({"n": 1})
        with patch("agentscape.events.shutil.copyfileobj", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                journal.archive()
        self.assertEqual(list(journal), [{"n": 1}])
        self.assertFalse(journal.compressed_path.exists())
        self.assertFalse(list(self.root.glob("*.tmp")))
        journal.append({"n": 2})
        journal.archive()
        self.assertEqual(len(list(journal)), 2)

    def test_concurrent_writers_preserve_every_entry(self):
        run = Run(Config(workspace=str(self.root / "workspace")), "test", state_dir=self.root)
        threads = [threading.Thread(target=lambda n=n: [run.event("system", "test", value=(n, i)) for i in range(100)])
                   for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(run.events), 400)
        self.assertEqual(len({tuple(e["value"]) for e in run.events}), 400)
        self.assertEqual(run.snapshot()["events"], run.events[-5:])

    def test_run_can_be_benchmarked_after_archive(self):
        config = Config(model="fake", routing="single", workspace=str(self.root / "workspace"))
        run = Run(config, "test", client=FakeModel(), state_dir=self.root).start()
        self.assertTrue(run.done.wait(5))
        self.assertEqual(run.status, "completed")
        self.assertTrue(run.events.compressed_path.exists())
        self.assertTrue(any(e["kind"] == "response" for e in run.events))
        self.assertEqual(run.snapshot()["events"][-1]["kind"], "done")


    def test_failed_run_archive_retains_events_and_records_warning(self):
        config = Config(model="fake", routing="single", workspace=str(self.root / "workspace"))
        with patch.object(EventJournal, "archive", side_effect=OSError("disk full")):
            run = Run(config, "test", client=FakeModel(), state_dir=self.root).start()
            self.assertTrue(run.done.wait(5))
        self.assertTrue(run.events.path.exists())
        self.assertTrue(list(run.events))
        saved = json.loads((run.directory / "run.json").read_text())
        self.assertTrue(any("original retained" in warning for warning in saved["warnings"]))


class ResidencyTests(unittest.TestCase):
    def test_default_residency_and_opt_in_unload_preserve_output(self):
        with tempfile.TemporaryDirectory() as scratch:
            answers = []
            for unload in (False, True):
                model = FakeModel()
                config = Config(model="fake", routing="single", workspace=scratch, unload_after_run=unload)
                run = Run(config, "test", client=model, state_dir=Path(scratch)).start()
                self.assertTrue(run.done.wait(5))
                self.assertEqual(run.status, "completed")
                self.assertEqual(model.unloaded, ["fake"] if unload else [])
                answers.append(run.answer)
            self.assertEqual(answers[0], answers[1])

    def test_unload_policy_also_runs_after_inference_failure(self):
        class Offline(FakeModel):
            def chat(self, *args, **kwargs):
                raise RuntimeError("backend offline")
        with tempfile.TemporaryDirectory() as scratch:
            model = Offline()
            config = Config(model="fake", routing="single", workspace=scratch, unload_after_run=True)
            run = Run(config, "test", client=model, state_dir=Path(scratch)).start()
            self.assertTrue(run.done.wait(5))
            self.assertEqual(run.status, "partial")
            self.assertEqual(model.unloaded, ["fake"])

    def test_request_uses_configured_residency_and_exposes_timing(self):
        requests = []
        def route(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, content=json.dumps({"message": {"content": "OK"}, "done": True,
                "prompt_eval_count": 20, "prompt_eval_duration": 500000000}))
        original = httpx.Client
        with patch("agentscape.ollama.httpx.Client", side_effect=lambda **kw: original(transport=httpx.MockTransport(route), **kw)):
            response = Ollama().chat("fake", [], replace(Config(), keep_alive=42))
        self.assertEqual(requests[0]["keep_alive"], 42)
        self.assertGreaterEqual(response["time_to_first_content_seconds"], response["queue_seconds"])
        self.assertEqual(metrics(response)["prompt_tokens_per_second"], 40)
        self.assertIsNone(metrics({})["time_to_first_content_seconds"])

    def test_config_rejects_invalid_residency(self):
        for value in (-2, 86401, True, "10m", 0.5):
            with self.assertRaises(ValueError):
                replace(Config(), keep_alive=value).validate()
        for value in (-1, 0, 600):
            replace(Config(), keep_alive=value).validate()


class ProfilingTests(unittest.TestCase):
    def test_rss_and_swap_are_parsed_in_bytes(self):
        import os
        sampler = ResourceSampler()
        with patch("agentscape.profiling.subprocess.check_output", return_value=f"{os.getpid()} 1 100 /bin/python\n12 1 200 /bin/ollama\n13 12 50 /bin/llama-server\n14 999 999 /other/llama-server\n"):
            sampler.sample()
        self.assertEqual(sampler.peak_python_rss_bytes, 102400)
        self.assertEqual(sampler.peak_ollama_rss_bytes, 256000)
        with patch("agentscape.profiling.platform.system", return_value="Darwin"), patch(
                "agentscape.profiling.subprocess.check_output", return_value="total = 1024.00M used = 12.50M free = 1011.50M"):
            self.assertEqual(swap_used_bytes(), int(12.5 * 1024**2))

    def test_unavailable_telemetry_is_explicit_and_does_not_fail_run(self):
        with patch("agentscape.profiling.subprocess.check_output", side_effect=OSError("ps unavailable")):
            with ResourceSampler(interval=0.01) as sampler:
                pass
        result = sampler.summary()
        self.assertIsNone(result["peak_python_rss_bytes"])
        self.assertTrue(result["errors"])
