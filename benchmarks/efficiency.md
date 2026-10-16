# Memory efficiency validation — 16 October 2026

Host: Apple M2 Pro, 16 GB unified memory. Python 3.14.6. Final live validation used Ollama 0.40.5, the existing Qwen3.5-4B Q4_K_M model, a 4096-token context and one inference slot. No models were downloaded and saved user configuration was not changed.

[Raw measurements](efficiency-results.json) retain every trial, model call and profiler field. The baseline is commit `6d872dec944d031c27b324d3c4e84cc386d87ddf`; the implementation's engine source hash is recorded in the JSON. A temporary checkout of the baseline used the same probe script and Python interpreter as the modified working tree.

## Before/after event storage

Three alternating baseline/implementation pairs each appended 5000 distinct synthetic events with 8192-character payloads. Each ran in a fresh Python process. `tracemalloc` measured Python allocations, not GPU memory or whole-process RSS. Full payloads were streamed back and hashed, including after gzip archiving.

| Median measurement | Baseline | Implementation |
|---|---:|---:|
| Retained Python allocations during event probe | 42,312,258 bytes | 277,709 bytes |
| Time to append 5000 events under tracemalloc | 0.709 s | 0.540 s |
| Complete payload round-trip | pass | pass |

Retained allocations fell **99.34%** (40.4 MiB to 0.265 MiB). Median append time fell 23.8% in this workload; timings are environment-sensitive and are not a prediction of model inference speed. Earlier individual measurements were noisier, including a slower modified run, so the table uses the explicit alternating comparison.

The journal keeps 32 recent events and streams older events from disk. Normal dashboard snapshots use the tail without reading the whole file. Explicitly asking for a full list can still allocate the full history. Unit tests also verify full event equality, Unicode, concurrent writers, slicing, archive failure recovery, and post-run access.

The final 41,339,470-byte synthetic log compressed to 231,668 bytes in 0.057 s. Its highly repetitive content is intentionally artificial: this compression ratio must not be generalized to real logs. Compression streams through a bounded buffer and leaves the source intact if it fails. Raw tool evidence and worker history files are unchanged.

## Model residency

Four real single-worker runs alternated the original resident policy and `unload_after_run=true`. All completed with the same numeric answer, `323`. Ollama reported **3,211,148,654 loaded bytes** after each resident run and **0** after each unload-after-run run.

This measures released model allocation at idle, not a reduction in active inference memory. The unload runs followed resident runs and started warm; the following resident runs started cold. Their different wall times cannot establish a policy speedup. The final direct cold probe took 3.530 s; warm probes took 0.168–0.172 s, demonstrating why unloading is opt-in. Default keep-alive remains 600 seconds.

## Live integration and telemetry

The profiled CLI benchmark passed all eight cases: four repeated sequential/parallel inference cases, both JSON and native shell-tool execution, and two-worker sequential/parallel teams. All four direct cold/warm probes also returned the expected numeric result. The benchmark now accepts equivalent numeric output such as `323.0`, rather than incorrectly treating it as a failed calculation.

Telemetry sampled Python and all local Ollama process trees, including the separate `llama-server` runner. Peak sampled Python RSS was 55,443,456 bytes; the summed local Ollama process-tree peak was 4,527,964,160 bytes. These are not exact unified-memory footprints: shared pages can be counted twice and sampling can miss short peaks. System swap grew during this run; it includes other applications, and **no swap reduction is claimed**.

No weight precision, KV-cache precision, prompts, sampling defaults, routing, or history compaction behavior was changed. There is no demonstrated inference-speed improvement from this patch. Backend changes, speculative decoding and approximate cache compression remain separate experiments.

## Reproduce

```sh
python -m unittest discover -s tests -v
python benchmarks/efficiency.py --output /tmp/efficiency.json --live --residency
agentscape bench --workers 2 --repeats 2 --profile --output /tmp/profile.json
```

Run event-only probes by omitting `--live --residency`. For an A/B comparison, run the same event-only script in clean checkouts of the baseline and modified sources using the same interpreter, alternating their order. Live probes unload the configured model and should be run without competing inference clients. No benchmark data is sent to a remote service when the configured Ollama endpoint is local.
