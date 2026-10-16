# Pretrained-model pilot: reproducible package (prepared on the development server; nothing rented)

This directory holds everything needed to run the staged pilot of
`docs/research/sylang-pilot-revised-2026-10-16.md` on a rented GPU, prepared and smoke-tested on
CPU first. **No GPU has been rented and no paid run has been started.** Every script runs with the
project's CPU virtual environment for smoke tests and with the same code on a GPU.

## Layout

| file | purpose |
|---|---|
| `contract.md` | the frozen data contract: records, the two tasks, the four renderings, splits, grader rules, decision rules |
| `data.py` | generates the frozen datasets from the v3 generator (`experiments/copy_probe/study3.py`): underlying workflow cases → four renderings × two tasks; writes `data/<version>/{train,dev,test}_*.jsonl`, `manifest.json` (seeds, sizes, split hashes, token counts) |
| `grader.py` | deterministic grader for both tasks: schema validity, exact facts, per-item flags; `test_grader.py` proves it rejects dropped pairs, swapped identifiers, invented records, wrong features, malformed JSON |
| `train_lora.py` | LoRA fine-tuning with Hugging Face Transformers + PEFT: config-driven, identical recipe for every cell, resumable checkpoints, dev-selected checkpoint, a JSON run record with tokens/updates/seconds |
| `evaluate.py` | generation-based evaluation (greedy, non-thinking), zero-shot baseline with the legend, retry policy (one retry with the format reminded), whole-request token accounting with the model's own tokenizer (system/legend, instruction, records, output; cached and uncached conventions), latency per request, cost per correct answer |
| `analyze.py` | paired per-item differences with a **cluster bootstrap over seeds then items**, the pre-registered non-inferiority rule (δ = 3 points), learning curves, tables |
| `profile_gpu.py` | stage A: measures model load time, peak memory, training examples/s, generation items/s, checkpoint size and upload time on the actual GPU, and writes `profile.json` with the projected cost of stages B and C at the quoted price |
| `ledger.py` | cost ledger: every rental's provider, instance, price, start, stop, purpose and measured spend; `verify_teardown.md` lists the provider-side checks (no running instance, no retained volume, final balance) that must be recorded before a stage is called finished |
| `smoke_cpu.sh` | end-to-end CPU smoke test with a tiny randomly-initialised Qwen3-architecture model (no weight download): data → train 20 steps → evaluate → analyze |
| `run_stage_a.sh`, `run_stage_b.sh`, `run_stage_c.sh` | the exact launch commands per stage, each wrapped in a hard `timeout` and ending with the ledger entry and the checkpoint/result upload |
| `configs/*.json` | per-cell training configs (arm, task mix, size, seed, answer format) |

## Rules carried from the study

- Identical underlying cases per seed in every arm; the answer bytes are identical across arms for the same task.
- Checkpoint selection and early stopping on dev panels only; the test panels are read once at the end of a stage.
- Every started run, failure and retry is recorded; nothing is dropped from the summary.
- Results files hold aggregates and per-item flags, never prompts or identifiers beyond synthetic ones.
- No output-format change is adopted from this pilot; the bare-answer arm is an experiment (stage C), reported, not deployed.

## Smoke test (CPU, no downloads beyond the tokenizer already in the repository's scratch)

    bash experiments/pretrained_pilot/smoke_cpu.sh   # about five minutes on four cores

## Secrets

Provider API tokens are read from environment variables (`PILOT_PROVIDER_TOKEN`, documented in
`verify_teardown.md`) and are never written to files in this repository.
