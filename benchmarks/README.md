# Local diagnostic — 10 October 2026

Tested only the already downloaded model:

`hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M`

Host: 16 GB memory, 10 CPU cores. Ollama 0.40.2. Context: 4096. Thinking disabled. Agentscape inference concurrency: 1. No models were downloaded. Both supervisor and workers used this same model.

Reproduce:

```sh
agentscape bench --workers 2 --repeats 2 \
  --supervisor hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M \
  --output benchmarks/local-model.json
```

| Case | Wall time | Diagnostic |
|---|---:|---|
| Two inference requests, sequential | 0.797 s / 0.611 s | Both returned 323 |
| Two inference requests, parallel workers / one inference slot | 0.620 s / 0.607 s | Both returned 323 |
| JSON shell-tool loop | 3.233 s | Actually executed command and reported output |
| Native shell-tool loop | 5.092 s | Actually executed command and reported output |
| Two workers, no supervisor, sequential | 13.688 s | Completed |
| Two workers, no supervisor, parallel | 12.644 s | Completed |
| Two workers + same-model supervisor, sequential | 5.958 s | Completed |
| Two workers + same-model supervisor, parallel | 6.154 s | Completed |

The JSON tool loop processed 1058 prompt tokens across two model calls; native processed 1993. Generated output was 54 versus 66 tokens, respectively. This supports using the compact JSON protocol as the initial configuration, but one prompt cannot establish general superiority.

The arithmetic inference cases show almost identical timing with one inference slot, as expected. Parallel worker scheduling does not create additional Ollama inference capacity. Team timings include planning and synthesis, and report lengths differ: the supervisor's assignments produced shorter worker reports in this example. These figures do **not** imply supervision is generally faster or that a bigger supervisor would improve results.

Ollama's end-of-case snapshot reported one loaded model, 3.24 GB loaded size / 3.24 GB VRAM, with a 4096 context. This is not peak process memory or total system memory. No separate model instance was started for each worker.

See [raw measurements](local-model.json) for token counts, load times, responses, and case metadata. Measurements were collected without competing model tests.

These are bounded arithmetic and shell-tool diagnostics, not a capability evaluation for autonomous security testing, and no comparison against Pi was run.

## Tool overlap

A separate real-model check instructed each of two workers to execute `sleep 2; printf 'TOOL_OVERLAP_OK\n'` and report the output. With one inference slot, both modes completed successfully:

| Mode | Total time | Measured command overlap |
|---|---:|---:|
| Sequential | 12.301 s | 0 s |
| Parallel | 7.045 s | 0.833 s |

This verifies tool overlap while model calls remain serialized. It does not predict speedups for workloads dominated by model generation. See [captured intervals](tool-overlap.json).
