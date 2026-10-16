from __future__ import annotations

import json
import stat
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from agentscape.config import Config, Worker, load_config, save_config
from agentscape.engine import Run
from agentscape.ollama import Cancelled, Ollama, OllamaError
from agentscape.tools import Gate, Tools


def response(content, **extra):
    return {"message": {"role": "assistant", "content": content}, "eval_count": 5, "eval_duration": 1_000_000,
            "prompt_eval_count": 10, "done_reason": "stop", **extra}


class FakeModel:
    def __init__(self, *, barrier=None, fail_plan=False, actions=None):
        self.barrier = barrier
        self.fail_plan = fail_plan
        self.actions = list(actions or [])
        self.messages = []
        self.lock = threading.Lock()
        self.unloaded = []

    def chat(self, model, messages, config, **kwargs):
        if kwargs.get("on_start"):
            kwargs["on_start"]()
        with self.lock:
            self.messages.append((model, messages.copy()))
        system = messages[0]["content"]
        if system.startswith("Split"):
            count = kwargs["schema"]["properties"]["assignments"]["maxItems"]
            return response("{}" if self.fail_plan else json.dumps({"assignments": [f"Assignment {i}" for i in range(count)]}))
        if system.startswith("Combine"):
            return response("Combined evidence.")
        if self.barrier:
            self.barrier.wait(timeout=3)
        with self.lock:
            action = self.actions.pop(0) if self.actions else {"action": "finish", "args": {}, "answer": "Checked evidence."}
        return response(action if isinstance(action, str) else json.dumps(action))

    def unload(self, model):
        self.unloaded.append(model)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.config = Config(model="fake", workspace=str(self.root / "workspace"), permissions="allow", max_steps=4, routing="team")

    def tearDown(self):
        self.scratch.cleanup()

    def run_engine(self, config=None, client=None):
        run = Run(config or self.config, "Check the task", client=client or FakeModel(), state_dir=self.root).start()
        self.assertTrue(run.done.wait(5), "Run failed to complete")
        return run

    def test_sequential_workers_receive_previous_evidence(self):
        config = replace(self.config, workers=[Worker("one"), Worker("two")])
        model = FakeModel()
        run = self.run_engine(config, model)
        self.assertEqual(run.status, "completed")
        second = next(messages for _, messages in model.messages if messages[0]["content"].startswith("You are two"))
        self.assertIn("Previous workers' evidence", second[1]["content"])
        self.assertIn("Checked evidence", second[1]["content"])
        self.assertTrue((run.directory / "report.md").exists())

    def test_parallel_workers_reach_model_together(self):
        config = replace(self.config, workers=[Worker("one"), Worker("two")], mode="parallel")
        run = self.run_engine(config, FakeModel(barrier=threading.Barrier(2)))
        self.assertEqual(run.status, "completed")
        self.assertEqual([r.agent for r in run.reports], ["one", "two"])

    def test_supervisor_uses_selected_model_and_unloads_between_phases(self):
        config = replace(self.config, supervisor="big", workers=[Worker("one"), Worker("two")])
        model = FakeModel()
        run = self.run_engine(config, model)
        self.assertEqual(run.status, "completed")
        self.assertEqual([name for name, _ in model.messages], ["big", "fake", "fake", "big"])
        self.assertEqual(model.unloaded, ["big", "fake"])

    def test_actual_tool_evidence_returns_to_model(self):
        model = FakeModel(actions=[{"action": "shell", "args": {"command": "printf evidence"}, "answer": ""}])
        run = self.run_engine(client=model)
        self.assertEqual(run.status, "completed")
        self.assertIn("evidence", model.messages[-1][1][-1]["content"])
        self.assertEqual(run.tools.count, 1)
        self.assertEqual(len(list((run.directory / "evidence").glob("*.txt"))), 1)

    def test_malformed_action_gets_repaired(self):
        model = FakeModel(actions=["not json"])
        run = self.run_engine(client=model)
        self.assertEqual(run.status, "completed")
        self.assertTrue(any(e["kind"] == "repair" for e in run.events))

    def test_repeated_failed_tool_stops_worker(self):
        action = {"action": "read", "args": {"path": "missing"}, "answer": ""}
        run = self.run_engine(client=FakeModel(actions=[action, action, action]))
        self.assertEqual(run.status, "partial")
        self.assertEqual(run.reports[0].status, "error")

    def test_supervisor_failure_is_logged_and_falls_back(self):
        run = self.run_engine(replace(self.config, supervisor="big"), FakeModel(fail_plan=True))
        self.assertEqual(run.reports[0].status, "completed")
        self.assertEqual(run.status, "partial")
        self.assertTrue(run.warnings)
        self.assertTrue(any("Planning failed" in e.get("detail", "") for e in run.events))

    def test_native_tool_loop_preserves_tool_result_role(self):
        class NativeModel(FakeModel):
            def chat(self, model, messages, config, **kwargs):
                self.messages.append((model, messages.copy()))
                if len(self.messages) == 1:
                    return response("", message={"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": "shell", "arguments": {"command": "printf native-evidence"}}}]})
                self.asserted = messages[-1]["role"] == "tool" and "native-evidence" in messages[-1]["content"]
                return response("Verified native evidence.")
        model = NativeModel()
        run = self.run_engine(replace(self.config, protocol="native"), model)
        self.assertTrue(model.asserted)
        self.assertEqual(run.status, "completed")

    def test_compaction_keeps_assistant_with_multiple_native_results(self):
        run = Run(replace(self.config, context=512), "Task", client=FakeModel(), state_dir=self.root)
        messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Task"},
                    {"role": "assistant", "content": "old " * 1000}, {"role": "tool", "content": "old result"},
                    {"role": "assistant", "content": "latest", "tool_calls": [{}, {}]},
                    {"role": "tool", "content": "result one"}, {"role": "tool", "content": "result two"}]
        trimmed = run.trim_history(messages, ["saved evidence"])
        self.assertEqual([m["role"] for m in trimmed[-3:]], ["assistant", "tool", "tool"])

    def test_token_truncation_never_executes_partial_action(self):
        class Truncated(FakeModel):
            def chat(self, *args, **kwargs):
                return response('{"action":"shell","args":{"command":"touch bad"},"answer":""}', done_reason="length")
        run = self.run_engine(client=Truncated())
        self.assertEqual(run.status, "partial")
        self.assertEqual(run.tools.count, 0)
        self.assertFalse((Path(self.config.workspace) / "bad").exists())


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.cancel = threading.Event()
        self.config = Config(workspace=str(self.root / "workspace"), permissions="allow", command_timeout=1)
        self.tools = Tools(self.config, self.root / "run", Gate("allow", self.cancel), self.cancel)

    def tearDown(self):
        self.scratch.cleanup()

    def test_path_and_symlink_escape_rejected(self):
        self.assertFalse(self.tools.execute("one", "read", {"path": "../secret"})["ok"])
        outside = self.root / "outside"
        outside.write_text("secret")
        (self.tools.workspace / "link").symlink_to(outside)
        self.assertFalse(self.tools.execute("one", "write", {"path": "link", "content": "overwrite"})["ok"])
        self.assertEqual(outside.read_text(), "secret")

    def test_directory_swap_during_write_approval_cannot_escape(self):
        folder = self.tools.workspace / "folder"
        folder.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        def swap(*args):
            folder.rename(self.tools.workspace / "original")
            folder.symlink_to(outside, target_is_directory=True)
            return True
        with patch.object(self.tools.gate, "approve", side_effect=swap):
            result = self.tools.execute("one", "write", {"path": "folder/file.txt", "content": "escape"})
        self.assertFalse(result["ok"])
        self.assertFalse((outside / "file.txt").exists())

    def test_workspace_swap_during_write_approval_cannot_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        def swap(*args):
            self.tools.workspace.rename(self.root / "original")
            self.tools.workspace.symlink_to(outside, target_is_directory=True)
            return True
        with patch.object(self.tools.gate, "approve", side_effect=swap):
            result = self.tools.execute("one", "write", {"path": "file.txt", "content": "escape"})
        self.assertFalse(result["ok"])
        self.assertFalse((outside / "file.txt").exists())

    def test_atomic_write_does_not_modify_hard_linked_outside_file(self):
        outside = self.root / "outside.txt"
        outside.write_text("original")
        (self.tools.workspace / "linked.txt").hardlink_to(outside)
        result = self.tools.execute("one", "write", {"path": "linked.txt", "content": "replacement"})
        self.assertTrue(result["ok"])
        self.assertEqual(outside.read_text(), "original")
        self.assertEqual((self.tools.workspace / "linked.txt").read_text(), "replacement")

    def test_atomic_overwrite_preserves_executable_permissions(self):
        target = self.tools.workspace / "script.sh"
        target.write_text("original")
        target.chmod(0o755)
        result = self.tools.execute("one", "write", {"path": "script.sh", "content": "replacement"})
        self.assertTrue(result["ok"])
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)

    def test_leaf_symlink_swap_during_approval_does_not_modify_outside(self):
        target = self.tools.workspace / "file.txt"
        target.write_text("original")
        outside = self.root / "outside.txt"
        outside.write_text("outside")
        def swap(*args):
            target.unlink()
            target.symlink_to(outside)
            return True
        with patch.object(self.tools.gate, "approve", side_effect=swap):
            result = self.tools.execute("one", "write", {"path": "file.txt", "content": "replacement"})
        self.assertTrue(result["ok"])
        self.assertEqual(outside.read_text(), "outside")
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_text(), "replacement")

    def test_failed_atomic_write_preserves_file_and_cleans_temporary(self):
        target = self.tools.workspace / "file.txt"
        target.write_text("original")
        with patch("agentscape.tools.os.replace", side_effect=OSError("replace failed")):
            result = self.tools.execute("one", "write", {"path": "file.txt", "content": "replacement"})
        self.assertFalse(result["ok"])
        self.assertEqual(target.read_text(), "original")
        self.assertEqual(list(self.tools.workspace.iterdir()), [target])

    def test_nested_write_creates_directories_and_utf8_content(self):
        result = self.tools.execute("one", "write", {"path": "new/nested/file.txt", "content": "Hello 🌍"})
        self.assertTrue(result["ok"])
        self.assertEqual((self.tools.workspace / "new/nested/file.txt").read_text(), "Hello 🌍")

    def test_background_job_is_stopped_even_when_it_ignores_sigterm(self):
        result = self.tools.execute("one", "shell", {
            "command": "(trap '' TERM; touch ready; sleep 0.7; touch late) & "
                       "while [ ! -f ready ]; do sleep 0.01; done"})
        self.assertFalse(result["ok"])
        self.assertIn("Background processes", result["error"])
        time.sleep(0.8)
        self.assertFalse((self.tools.workspace / "late").exists())

    def test_cancel_stops_descendants_when_shell_exits_on_sigterm(self):
        result = []
        errors = []
        def execute():
            try:
                result.append(self.tools.execute("one", "shell", {
                    "command": "(trap '' TERM; touch ready; sleep 0.7; touch late) & wait"}))
            except Cancelled as exc:
                errors.append(exc)
        thread = threading.Thread(target=execute)
        thread.start()
        deadline = time.monotonic() + 2
        while not (self.tools.workspace / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue((self.tools.workspace / "ready").exists())
        self.cancel.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        time.sleep(0.8)
        self.assertFalse((self.tools.workspace / "late").exists())

    def test_read_large_output_has_retrievable_evidence(self):
        self.tools.execute("one", "write", {"path": "big.txt", "content": "a" * 6000 + "END"})
        result = self.tools.execute("one", "read", {"path": "big.txt"})
        self.assertTrue(result["truncated"])
        section = self.tools.execute("one", "result", {"id": result["result_id"], "offset": 6000})
        self.assertEqual(section["output"], "END")

    def test_other_worker_can_wait_for_completed_evidence(self):
        self.config.workers = [Worker("one"), Worker("two")]
        output = []
        thread = threading.Thread(target=lambda: output.append(self.tools.execute("two", "result", {"id": "latest-one"})))
        thread.start()
        time.sleep(0.1)
        self.assertTrue(thread.is_alive())
        self.tools.execute("one", "shell", {"command": "printf shared-evidence"})
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(output[0]["ok"])
        self.assertEqual(output[0]["output"], "shared-evidence")
        index = self.tools.execute("two", "result", {"id": "list"})
        self.assertIn('"agent": "one"', index["output"])

    def test_invalid_tool_arguments_return_actionable_error_and_evidence_id(self):
        result = self.tools.execute("one", "result", {})
        self.assertFalse(result["ok"])
        self.assertIn("requires args: id", result["error"])
        self.assertTrue(result["result_id"])
        result = self.tools.execute("one", "unknown", {})
        self.assertFalse(result["ok"])
        self.assertTrue(result["result_id"])

    def test_scope_checks_domains_wildcards_cidrs(self):
        self.config.scope = ["example.com", "*.lab.example", "192.168.1.0/24"]
        for url in ["https://example.com", "http://one.lab.example", "http://192.168.1.10"]:
            self.tools.allowed_url(url)
        for url in ["https://example.com.evil.test", "https://lab.example", "http://192.168.2.10", "file:///etc/passwd"]:
            with self.assertRaises(ValueError):
                self.tools.allowed_url(url)

    def test_redirect_outside_scope_is_blocked_before_request(self):
        self.config.scope = ["example.com"]
        visited = []
        def route(request):
            visited.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://outside.test"})
        original = httpx.Client
        with patch("agentscape.tools.httpx.Client", side_effect=lambda **kw: original(transport=httpx.MockTransport(route), **kw)):
            result = self.tools.execute("one", "fetch", {"url": "https://example.com"})
        self.assertFalse(result["ok"])
        self.assertEqual(len(visited), 1)

    def test_deny_policy_and_disabled_tools(self):
        self.tools.gate.policy = "deny"
        result = self.tools.execute("one", "shell", {"command": "touch should-not-exist"})
        self.assertFalse(result["ok"])
        self.assertFalse((self.tools.workspace / "should-not-exist").exists())
        self.config.enabled_tools = []
        self.assertFalse(self.tools.execute("one", "shell", {"command": "true"})["ok"])

    def test_shell_timeout_and_exit_status(self):
        result = self.tools.execute("one", "shell", {"command": "sleep 10"})
        self.assertFalse(result["ok"])
        self.assertIn("timeout", result["error"])
        result = self.tools.execute("one", "shell", {"command": "printf nope; exit 7"})
        self.assertEqual(result["exit_code"], 7)

    def test_pending_approval_cancels(self):
        gate = Gate("ask", self.cancel)
        errors = []
        def wait():
            try:
                gate.approve("one", "shell", "true")
            except Cancelled as exc:
                errors.append(exc)
        thread = threading.Thread(target=wait)
        thread.start()
        gate.requests.get(timeout=1)
        self.cancel.set()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)


class OllamaTests(unittest.TestCase):
    def test_native_stream_and_timing_are_assembled(self):
        chunks = [
            {"message": {"content": "hel"}, "done": False},
            {"message": {"content": "lo", "tool_calls": [{"function": {"name": "read", "arguments": {"path": "."}}}]},
             "done": True, "eval_count": 2, "eval_duration": 100},
        ]
        original = httpx.Client
        transport = httpx.MockTransport(lambda req: httpx.Response(200, content="\n".join(json.dumps(x) for x in chunks)))
        with patch("agentscape.ollama.httpx.Client", side_effect=lambda **kw: original(transport=transport, **kw)):
            result = Ollama().chat("fake", [], Config())
        self.assertEqual(result["message"]["content"], "hello")
        self.assertEqual(result["message"]["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(result["eval_count"], 2)

    def test_stream_without_done_is_failure(self):
        original = httpx.Client
        transport = httpx.MockTransport(lambda req: httpx.Response(200, content='{"message":{"content":"partial"}}\n'))
        with patch("agentscape.ollama.httpx.Client", side_effect=lambda **kw: original(transport=transport, **kw)):
            with self.assertRaises(OllamaError):
                Ollama().chat("fake", [], Config())

    def test_inference_slots_limit_concurrency(self):
        client = Ollama(parallel=1)
        cancel = threading.Event()
        entered = threading.Event()
        def wait():
            with client.slot(cancel):
                entered.set()
        with client.slot(cancel):
            thread = threading.Thread(target=wait)
            thread.start()
            self.assertFalse(entered.wait(0.1))
        self.assertTrue(entered.wait(1))
        thread.join()


class ConfigTests(unittest.TestCase):
    def test_roundtrip_and_validation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            config = Config()
            config.resize(3)
            config.workers[1].model = "different"
            save_config(config, path)
            result = load_config(path)
            self.assertEqual(len(result.workers), 3)
            self.assertEqual(result.workers[1].model, "different")
            with self.assertRaises(ValueError):
                result.resize(0)
            result.workers[1].name = result.workers[0].name
            with self.assertRaises(ValueError):
                result.validate()


if __name__ == "__main__":
    unittest.main()
