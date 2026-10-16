# Automatic routing diagnostic — 10 October 2026

Tested with four configured workers, parallel scheduling, and an enabled supervisor. All roles used the already downloaded `hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M` through Ollama. No models were downloaded.

| Request | Selected route | Active workers | Model calls | Tool calls | Time |
|---|---|---:|---:|---:|---:|
| `hello` | Direct answer | 1 | 1 | 0 | 2.984 s |
| Capital of France | Direct answer | 1 | 1 | 0 | 1.537 s |
| Execute a specific `printf` command | Single | 1 | 3 | 1 | 4.885 s |
| Read two short config files and list their keys | Single | 1 | 4 | 2 | 6.242 s |
| Separate frontend/backend audits explicitly requested in parallel | Team | 2 of 4 | 9 | 4 | 26.669 s |

All five checks passed. Direct answers came from the routing call itself and skipped tool execution, supervisor planning, and synthesis. Focused tool tasks skipped the supervisor. The team performed actual reads and correctly reported production debug enabled, port 70000 outside the specified 1–65535 range, and a negative timeout. The final answers contained no internal worker names or supervisor references.

The dashboard retains the full roster and shows unused agents as skipped. Raw per-agent notes and route decisions remain in saved run diagnostics. The public report is separate from those diagnostics.

Times are individual end-to-end samples, not broad performance estimates. The first case includes any model startup/prefill cost. The integration test validates routing, execution, fixture findings, and final presentation; it does not establish general model factual reliability. It also verifies that reading two short files does not automatically justify a team.

Reproduce with the configured local model:

```sh
.venv/bin/python tests/live_routing.py
```

This test runs a fixed `printf` command and reads temporary fixture files. It temporarily supplies four-worker capacity for the test without changing saved team settings. [Raw results](routing.json).
