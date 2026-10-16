# AgentScape

A terminal environment for teams of local Ollama agents. Configure workers, run them sequentially or in parallel, and optionally use a supervisor to plan assignments and combine their evidence.

The initial configuration expects this model to be installed in Ollama:

```text
hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M
```

No other models are downloaded automatically. Agentscape has no additional content moderation or refusal classifier. The model's own behavior still comes from its weights and prompt. All inference and model downloads/loading/unloading go through Ollama.

## Enter the environment

After running `./setup.sh`, open a new terminal and run:

```sh
agentscape
```

If your current terminal has not picked up the launcher:

```sh
export PATH="$HOME/.local/bin:$PATH"
agentscape
```

Or run `./agentscape.sh` from this folder. On a fresh installation, run `./setup.sh` once. It creates a project virtual environment, installs the CLI, places a launcher in `~/.local/bin`, and adds that directory to the shell PATH if needed. Python 3.11+ and a running Ollama server are required.

## Configure your team

Inside Agentscape:

```text
/setup
```

The wizard lets you pick an installed worker model, worker count, sequential/parallel scheduling, optional supervisor model, inference concurrency, workspace, and execution permissions.

For quick changes:

```text
/workers 3
/mode parallel
/supervisor on
/concurrency 1
```

`/supervisor on` uses the same model as the workers. `/supervisor off` disables supervision. To use a different installed model, provide its full Ollama name:

```text
/supervisor YOUR_LARGER_MODEL
/model YOUR_SMALLER_WORKER_MODEL
/agent 2 --name reviewer --role "Check findings against saved evidence"
/agent 3 --model ANOTHER_INSTALLED_MODEL
```

An individual worker model can be reset with `/agent 3 --model inherit`. Use `/models` to see model disk sizes and current loaded memory. `/pull MODEL` downloads a model through Ollama; it does not select it automatically. `/stop MODEL` unloads it without deleting it.

All slash commands also work directly in a shell:

```sh
agentscape workers 3
agentscape mode sequential
agentscape supervisor off
agentscape config
```

The initial configuration uses one worker, no supervisor, a 4096-token context, JSON actions, and one inference slot. These are modest defaults for a 16 GB machine; increase the team size when the task benefits from distinct roles.

### Automatic task routing

Routing defaults to `auto`. The configured worker count is **available capacity**, not a requirement to run every worker on every message. The default worker model makes a compact routing decision:

- **Direct:** greetings, simple questions, and explanations are answered in that same model call. No tools, supervisor planning, or extra synthesis call are used.
- **Single:** a focused tool task runs with one worker and skips supervision and synthesis.
- **Team:** work with useful independent responsibilities uses the selected number of workers, up to configured capacity. The supervisor is used only on this path, if configured.

Unused rows remain visible as `skipped` in the dashboard. Routing is a model judgment, not a guaranteed measure of task complexity; if it is unavailable or malformed, execution falls back to one worker. Override it when needed:

```text
/routing auto
/routing single
/routing team
```

The same setting can be overridden per run with `agentscape run 'TASK' --routing team`. Benchmark team cases force `team` so their configured counts are actually exercised.

Final answers use one assistant voice and omit internal agent attribution. The live dashboard and saved diagnostic logs retain agent names, routing decisions, model calls, and reports. Failed checks and uncertainty remain in the answer. A presentation repair call is used only if internal attribution appears in a draft; normal direct answers remain one model call.

Requests for files, commands, URLs, or live information are blocked from the direct-answer path when detected; the router is also instructed never to claim external observations without tools. This conservative check can send an otherwise simple question mentioning a file or URL to one agent. See the [real-model routing tests](benchmarks/routing.md).

## Give the team a task

Type a task directly at the prompt:

```text
Use shell to list the workspace, fetch https://example.com, and report the actual results.
```

The dashboard shows each agent's state, step, token counts, and recent activity. Shell commands and file writes ask for permission by default. Choose your execution policy explicitly:

```text
/permissions ask
/permissions allow
/permissions deny
```

`allow` permits shell commands and workspace writes without prompts. `deny` blocks both. These are execution controls, separate from model content. Read, fetch, and search do not require approval. In a noninteractive session, `ask` denies requested actions because no person can answer.

Agents execute host shell commands as your user, with the workspace as their working directory. **This is not an OS sandbox.** Shell commands can access files and the network outside that directory. Use a disposable VM for isolation; the entire application can run inside it, pointed at an Ollama endpoint. `read` and `write` tools enforce workspace paths. Writes use directory descriptors and atomic replacement to prevent symlink swaps from redirecting them. Concurrent writes to one file still use last-writer-wins behavior. Shell process groups are cleaned up before the tool returns; background jobs are stopped and reported as an incomplete action. Commands that deliberately detach into a new session require OS isolation to contain.

Tasks are independent by default. `/continue YOUR_FOLLOW_UP` supplies the previous task and report to the next run. It also works with the most recent saved run after restarting the terminal. Ctrl-C cancels a task; active commands are terminated. An inference cancellation is checked when Ollama sends its next stream chunk; a stalled connection can take up to the configured request timeout to return.

Run a task directly:

```sh
agentscape run 'Inspect the workspace and report its contents'
agentscape run 'Run printf "hello\n" and report its output' --permissions allow
agentscape run 'Check 17 times 19' --workers 2 --mode parallel --supervisor off
agentscape run 'Check 17 times 19' --json
```

Run flags override settings for that run without saving them. Use configuration commands to save changes.

## Tools and internet access

| Tool | Behavior |
|---|---|
| `shell` | Execute a command, capture exit status and output, enforce time/output limits |
| `fetch` | Fetch HTTP(S) text/HTML; extract readable text; check redirect hosts against scope |
| `search` | Return web links and snippets from DuckDuckGo Lite with HTML fallback |
| `read` | Read a workspace file or list a directory |
| `write` | Write a workspace file, subject to the execution policy |
| `result` | Retrieve saved evidence by ID, list available results, or wait for a named worker's output |

Tool results return at most 5000 characters to the model. Full captured output remains in the run's evidence folder (commands have a 2 MB output limit). File reads and page fetches are limited to 2 MB. `result` is always available; the other tools can be selected:

```text
/tools read,fetch,search
/tools all
/tools none
/workspace /path/to/your/project
```

Search uses public search endpoints, which can rate-limit or issue bot challenges. Failures are reported to the agent rather than replaced with invented results. Optional `OLLAMA_API_KEY` enables Ollama's hosted web search service instead; only the search query goes to that service, and model inference still uses your configured Ollama endpoint. Direct fetching needs no API key. Search and fetching retrieve text; they do not drive a browser or execute website JavaScript.

Optional fetch scope:

```text
/scope add example.com
/scope add *.example.com
/scope add 192.168.1.0/24
/scope clear
```

Scope restricts **the fetch tool's URL hostnames/IP literals**, including redirect targets. It does not filter search results, constrain shell commands, or pin DNS resolution. Use network isolation/firewall rules for a hard network boundary. An empty scope is unrestricted.

## Workers versus inference concurrency

`/mode parallel` schedules workers concurrently. `/concurrency N` caps simultaneous inference requests from **this Agentscape run**. Both settings are independent of Ollama's server configuration.

With one inference slot, parallel workers share the loaded model and take turns generating; tools can overlap while another agent generates. With more slots, Agentscape may submit multiple requests, but Ollama can still queue them if its own `OLLAMA_NUM_PARALLEL` is 1. Raising Ollama's server parallelism requires configuring/restarting the server and consumes additional context/cache memory. Agentscape deliberately does not restart your existing Ollama server or change global settings.

Same-model agents share weights on the same Ollama server; separate agents do not automatically create model copies. Multiple different resident models consume their own model memory. Compact mode unloads models not needed by the next supervisor/worker phase; it does not constrain other programs using Ollama or prevent multiple different worker models from being loaded together.

```text
/config set compact true
/config set context 4096
/config set max_steps 12
/config set max_tokens 768
/config set command_timeout 60
/config set request_timeout 180
/config set thinking false
/config set protocol json
```

JSON actions keep tool descriptions short for a small model. `/config set protocol native` uses Ollama's native function calling instead. Both protocols were tested with your model. This implementation has not been benchmarked against Pi; it was chosen to make scheduling and per-agent settings explicit with a small prompt.

## Supervisor behavior

The supervisor generates one assignment per worker before execution and combines their reports afterward. It does not run tools itself. Sequential workers receive earlier workers' reports. Parallel workers start independently with distinct assignments; they share the workspace, so conflicting edits remain possible. A supervisor is instructed to assign separate file ownership for parallel work. This is not a file-locking guarantee.

Parallel workers can inspect each other's completed tool evidence through `result` with `id="list"` or `id="latest-worker-1"` (substitute the worker's actual roster name). The latter waits up to `command_timeout` for evidence if none is ready. This waits for tool output, not the worker's final report; sequential scheduling is preferable for a strict chain of dependent tasks.

The supervisor can use a larger model than the workers, or the same model with a different role. Agentscape does not infer model capability from file size; choose models explicitly. A supervisor adds at least two model calls, so use it for tasks where planning and review justify that overhead. There is no persistent worker-to-worker messaging or automatic recursive spawning.

## Saved runs and diagnostics

```text
/runs
/show latest
/doctor
/bench --workers 2
/bench --workers 2 --supervisor YOUR_SUPERVISOR_MODEL
```

Runs are saved under `.agentscape/runs/ID/`:

- `report.md`: final report
- `run.json`: task, configuration, agent reports, timings, warnings
- `events.jsonl`: model responses, tool calls/results, state changes while running; compressed losslessly to `events.jsonl.gz` when the run finishes
- `WORKER-messages.json`: each worker's final model history
- `plan.json`: supervisor assignments, when available
- `evidence/`: captured raw tool results

`completed` means the workers returned final reports; it is not an independent guarantee that the task succeeded or that a finding is correct. `partial` preserves step-limit failures, worker errors, or supervisor fallback. Raw evidence should support any result you rely on.

Only the latest 32 events are retained in RAM. Complete history stays on disk and `run.events` streams it on demand; the dashboard reads the cached tail. Archived logs are ordinary gzip JSONL (`gzip -dc PATH/events.jsonl.gz`). If archiving fails, the original journal is retained and the run metadata records a warning. This changes log storage, not model context or model precision.

The benchmark runs arithmetic diagnostics, actual shell-tool smoke tests for both JSON and native protocols, and sequential/parallel teams with and without a supervisor. It reports wall time, tokens, model-load time, and end-of-case loaded-memory snapshots. It does not measure peak total memory or establish broad security-testing competence. See [benchmark notes](benchmarks/README.md) for the test on your current model.

Configuration lives in `.agentscape/config.json`. To add your own team-wide prompt, create `.agentscape/SYSTEM.md` (the first 4000 characters are included in worker prompts). `AGENTSCAPE_HOME` can point at a separate state directory for another profile. `ollama_url` can point at another Ollama server.

## Memory and performance

Model residency defaults to the existing ten-minute keep-alive. To release models after the entire task finishes, while retaining them between tool steps:

```sh
agentscape run 'YOUR TASK' --unload-after-run
# Or persist this policy:
agentscape config set unload_after_run true
# Revert to keeping the model warm between tasks:
agentscape config set unload_after_run false
```

`agentscape config set keep_alive 120` changes the idle interval in seconds. `-1` retains the model indefinitely; `0` unloads it after each model request and can cause repeated loading within one task. Per-run override: `--keep-alive 120`. Unloading affects the shared Ollama server and may disrupt cache reuse for other clients; leave it disabled when sharing a server or optimizing repeated-task latency. It changes residency, not weight/cache precision.

For local memory and latency diagnostics:

```sh
agentscape bench --workers 2 --repeats 2 --profile --output /tmp/agentscape-profile.json
.venv/bin/python benchmarks/efficiency.py --output /tmp/agentscape-efficiency.json --live --residency
```

The profiler samples Python and local Ollama process-tree RSS every 0.5 seconds and records system swap before/after. RSS is not exact peak unified/GPU memory, can include shared pages more than once, and does not measure remote inference. Swap changes include all applications. Call metrics now include time to first content, local semaphore wait, prompt-processing time/rate, and generation rate. A tool-only response may have no first-content time. Profiling is opt-in and adds sampling overhead.

The efficiency probe compares synthetic event retention and optionally tests a fixed local model prompt and alternating model-residency policies. The live probes unload the selected model; run them when other clients are idle. See [measured before/after results and limitations](benchmarks/efficiency.md). No inference backend, model weights, quantization, context policy, or server-wide settings are changed by these optimizations.

## Development

```sh
./setup.sh
.venv/bin/python -m unittest discover -s tests -v
# Optional integration diagnostics against your configured Ollama model:
.venv/bin/python tests/live_routing.py
```

The package is installed editable, so source changes take effect immediately. Runtime dependencies are HTTPX, Beautiful Soup, Rich, and prompt-toolkit. No Pi, LangChain, agent servers, or cloud model SDKs are required.

## Security

See [SECURITY.md](SECURITY.md) for permission controls, isolation limits, and private vulnerability reporting.
