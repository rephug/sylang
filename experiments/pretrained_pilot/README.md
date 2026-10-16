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
| `train_lora.py` | LoRA fine-tuning with Hugging Face Transformers + PEFT: config-driven, identical recipe for every cell, resumable checkpoints (resume re-derives `best/` and truncates the log to the checkpoint), dev-selected checkpoint decoded by the evaluator's own rule (`common.decode_output`), a JSON run record with tokens/updates/seconds |
| `evaluate.py` | generation-based evaluation (greedy, non-thinking), zero-shot baseline with the legend, retry policy (one retry with the format reminded), whole-request token accounting with the model's own tokenizer (system/legend, instruction, records, output; cached and uncached conventions), latency per request, cost per correct answer |
| `analyze.py` | paired per-item differences with a **cluster bootstrap over seeds then items**, the pre-registered non-inferiority rule (δ = 3 points), learning curves, tables |
| `profile_gpu.py` | stage A: measures, on the `json` arm (the longest rendering, so the single measured rate is conservative for every cell), model load time, peak memory, training examples/s, generation items/s, checkpoint size and upload time (only the upload program's name, seconds and exit status are recorded), checks every plan file before measuring, and writes `profile.json` with the measured values first and then the projected cost of stages B and C at the quoted price, flagging with `!!!` any cell whose projected training or evaluation phase exceeds its `timeout_minutes` |
| `ledger.py` | cost ledger: every rental's provider, instance, price, start, stop, purpose and measured spend (`report --pending-spend` counts a running session's time against the ceiling; stage A uses it); `verify_teardown.md` lists the provider-side checks (no running instance, no retained volume, final balance) that must be recorded before a stage is called finished |
| `smoke_cpu.sh` | end-to-end CPU smoke test with a tiny randomly-initialised Qwen3-architecture model (no weight download): data → train 20 steps → evaluate → analyze |
| `run_stage_a.sh`, `run_stage_b.sh`, `run_stage_c.sh` | the exact launch commands per stage, every step under a hard `timeout`; SIGHUP/SIGINT/SIGTERM are trapped (the running step is killed and the interruption noted) and every exit ends with the ledger stop line and then the results tarball and upload; start them under `nohup`/`setsid`/tmux (header of each script) |
| `setup_gpu.sh`, `requirements-gpu.txt` | prepare a rented machine: venv (Python 3.12 or newer, `PILOT_SYSTEM_PYTHON`, checked before any pip call), weights (download only with `PILOT_ALLOW_DOWNLOAD=1`), the dataset tarball, import/GPU/template/manifest checks |
| `configs/*.json` | per-cell training configs (arm, task mix, size, seed, answer format) |

## Rules carried from the study

- Identical underlying cases per seed in every arm; the answer bytes are identical across arms for the same task.
- Checkpoint selection and early stopping on dev panels only; the test panels are read once at the end of a stage.
- Every started run, failure and retry is recorded; nothing is dropped from the summary.
- Results files hold aggregates and per-item flags, never prompts or identifiers beyond synthetic ones.
- No output-format change is adopted from this pilot; the bare-answer arm is an experiment (stage C), reported, not deployed.

## Smoke test (CPU; nothing downloaded; the tokenizer and the study packet are local directories)

    PYTHON=<venv>/bin/python PILOT_TOKENIZER=<dir with tokenizer.json> SYLANG_PACKET=<packet> \
      bash experiments/pretrained_pilot/smoke_cpu.sh   # about three minutes on four cores

Passed on the development server on 2026-10-16 (153 s): data → check → two 20-step trainings of a randomly initialised
Qwen 3 → zero-shot and trained evaluations → analysis. The verification record is §6.1 of
`docs/research/sylang-pilot-revised-2026-10-16.md`.

## Secrets

Provider API tokens live only in the owner's provider CLI and, if this session needs one, in the cloud
environment's secrets as `PILOT_PROVIDER_TOKEN` (`verify_teardown.md`); the package never reads or
writes one, and `ledger.py` refuses a line containing it. `PILOT_UPLOAD_CMD` must be credential-free
(rclone config or ssh keys): the scripts print only its first word and `profile.json` records only the
program name. `profile.json` is git-ignored; it travels in the stage tarball and is committed only after review.
