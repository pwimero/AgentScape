import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agentscape.config import Config, Worker
from agentscape.engine import Run
from agentscape.ui import dashboard
from test_agentscape import FakeModel, response


class RoutingModel(FakeModel):
    def __init__(self, decision, *, rewrite="The check returned OK. A second check could not be completed.", **kwargs):
        super().__init__(**kwargs)
        self.decision, self.rewrite = decision, rewrite

    def chat(self, model, messages, config, **kwargs):
        system = messages[0]["content"]
        if system.startswith("Choose the cheapest") or system.startswith("Rewrite"):
            self.messages.append((model, messages.copy()))
            if kwargs.get("on_start"):
                kwargs["on_start"]()
            return response(json.dumps(self.decision) if system.startswith("Choose") else self.rewrite)
        return super().chat(model, messages, config, **kwargs)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = Config(model="small", supervisor="big", mode="parallel", routing="auto",
                             workers=[Worker(f"worker-{i}") for i in range(1, 5)],
                             workspace=str(self.root / "workspace"), permissions="allow", max_steps=4)

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, client, config=None, task="hello"):
        run = Run(config or self.config, task, client=client, state_dir=self.root).start()
        self.assertTrue(run.done.wait(5))
        return run

    def test_greeting_is_one_call_no_tools_no_supervisor(self):
        model = RoutingModel({"mode": "direct", "workers": 1, "answer": "Hello!", "reason": "Greeting"})
        run = self.execute(model)
        self.assertEqual(run.answer, "Hello!")
        self.assertEqual(len(model.messages), 1)
        self.assertEqual([name for name, _ in model.messages], ["small"])
        self.assertEqual(run.tools.count, 0)
        self.assertEqual(run.states["supervisor"]["status"], "skipped")
        self.assertEqual(run.states["worker-4"]["status"], "skipped")
        self.assertEqual(run.summary()["route"]["mode"], "direct")

    def test_factual_question_uses_answer_from_router(self):
        model = RoutingModel({"mode": "direct", "workers": 1, "answer": "Paris.", "reason": "Known fact"})
        run = self.execute(model, task="What is the capital of France?")
        self.assertEqual(run.answer, "Paris.")
        self.assertEqual(len(run.calls), 1)

    def test_focused_tool_task_uses_one_agent_and_no_synthesis(self):
        model = RoutingModel({"mode": "single", "workers": 1, "answer": "", "reason": "One command"},
                             actions=[{"action": "shell", "args": {"command": "printf OK"}, "answer": ""}])
        run = self.execute(model, task="Run printf OK")
        self.assertEqual(run.status, "completed")
        self.assertEqual(len(run.reports), 1)
        self.assertEqual(run.tools.count, 1)
        self.assertEqual(len(model.messages), 3)  # routing + action + answer
        self.assertFalse(any(name == "big" for name, _ in model.messages))

    def test_team_selects_subset_of_available_workers(self):
        model = RoutingModel({"mode": "team", "workers": 2, "answer": "", "reason": "Two independent components"})
        run = self.execute(model, task="Investigate two independent components")
        self.assertEqual(run.status, "completed")
        self.assertEqual(len(run.reports), 2)
        self.assertEqual(run.states["worker-3"]["status"], "skipped")
        self.assertEqual([name for name, _ in model.messages].count("big"), 2)
        self.assertEqual(len(run.config.workers), 4)  # persisted capacity unchanged

    def test_forced_single_bypasses_router_and_supervisor(self):
        model = FakeModel()
        run = self.execute(model, replace(self.config, routing="single"))
        self.assertEqual(len(model.messages), 1)
        self.assertEqual(len(run.reports), 1)

    def test_forced_team_preserves_full_roster(self):
        model = FakeModel()
        run = self.execute(model, replace(self.config, routing="team"))
        self.assertEqual(len(run.reports), 4)
        self.assertEqual(len(model.messages), 6)  # plan + 4 + synthesis

    def test_invalid_router_falls_back_without_spawning_team(self):
        model = RoutingModel({"mode": "team", "workers": 99})
        run = self.execute(model)
        self.assertEqual(run.route["mode"], "single")
        self.assertEqual(len(run.reports), 1)
        self.assertEqual(len(model.messages), 2)

    def test_file_inspection_cannot_accept_fabricated_direct_answer(self):
        model = RoutingModel({"mode": "direct", "workers": 1, "answer": "I inspected config.json and it is fine.", "reason": "Simple"})
        run = self.execute(model, task="Inspect config.json for errors")
        self.assertEqual(run.route["mode"], "single")
        self.assertNotIn("it is fine", run.answer)
        self.assertTrue(any(e['kind']=='routing_fallback' for e in run.events))

    def test_synthesis_failure_keeps_diagnostics_out_of_final_answer(self):
        class Failing(FakeModel):
            def chat(self, model, messages, config, **kwargs):
                if messages[0]['content'].startswith('Combine'):
                    raise RuntimeError('supervisor worker-1 offline')
                return super().chat(model, messages, config, **kwargs)
        run = self.execute(Failing(), replace(self.config, routing='team'))
        self.assertEqual(run.status, 'partial')
        self.assertNotIn('supervisor', run.answer.lower())
        self.assertNotIn('worker', run.answer.lower())
        self.assertTrue(run.warnings)

    def test_single_reply_with_internal_attribution_is_rewritten(self):
        model = RoutingModel({"mode": "single", "workers": 1, "answer": "", "reason": "Check"},
                             actions=[{"action": "finish", "args": {}, "answer": "worker-1 returned OK. worker-2 failed."}])
        run = self.execute(model, task="Perform the checks")
        self.assertNotIn("worker", run.answer.lower())
        self.assertIn("could not be completed", run.answer)
        self.assertEqual(len(model.messages), 3)
        self.assertIn("worker-1", run.reports[0].answer)  # diagnostics preserved

    def test_failed_rewrite_does_not_dump_internal_reports(self):
        model = RoutingModel({"mode": "direct", "workers": 1, "answer": "worker-1 says hello", "reason": "Greeting"},
                             rewrite="worker-2 says hello")
        run = self.execute(model)
        self.assertEqual(run.status, "partial")
        self.assertNotIn("worker", run.answer.lower())
        self.assertTrue(run.warnings)

    def test_custom_agent_labels_are_removed(self):
        config = replace(self.config, workers=[Worker('reviewer')])
        model = RoutingModel({"mode":"single", "workers":1, "answer":"", "reason":"Check"},
                             actions=[{"action":"finish", "args":{}, "answer":"Reviewer: Returned OK."}],
                             rewrite="The check returned OK.")
        run = self.execute(model, config, task="Check the result")
        self.assertEqual(run.answer, "The check returned OK.")

    def test_dashboard_keeps_full_roster_and_skipped_agents(self):
        model = RoutingModel({"mode": "direct", "workers": 1, "answer": "Hello!", "reason": "Greeting"})
        run = self.execute(model)
        from rich.console import Console
        console = Console(record=True, width=100)
        console.print(dashboard(run))
        graphic = console.export_text()
        self.assertIn("worker-1", graphic)
        self.assertIn("worker-4", graphic)
        self.assertIn("skipped", graphic)


if __name__ == "__main__":
    unittest.main()
