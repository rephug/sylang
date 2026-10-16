# File formats and interfaces of the pilot package

`common.py` is the code form of this page; both are frozen together with `contract.md`. Every
script takes paths and settings from the command line or a config file; nothing is hard-coded
to a machine. Environment variables: `PILOT_TOKENIZER` (directory with the Qwen 3
`tokenizer.json` and `tokenizer_config.json`), `PILOT_MODEL` (directory with the base weights, or
`tiny-random`), `SYLANG_PACKET` (the study packet, needed by `data.py` only, for the `english`
rendering), `PILOT_ALLOW_DOWNLOAD=1` (the only way any script may download weights),
`PILOT_PROVIDER_TOKEN` (the provider API token for the owner's provider CLI; the stage scripts neither read nor write it, and `ledger.py` refuses any line containing its value).

## Data: `data/<version>/`

| file | content |
|---|---|
| `legends.json` | `{"json": {arm: legend}, "bare": {arm: legend}}`: the system message per arm and answer format, 100–130 Qwen 3 tokens, arms matched within 15 tokens of each other (asserted by `data.py`) |
| `cases_dev.jsonl`, `cases_test.jsonl`, `cases_train_s<seed>.jsonl` | one underlying case per line (arm-independent), see below |
| `dev_<arm>.jsonl`, `test_<arm>.jsonl`, `train_s<seed>_<arm>.jsonl` | rendered examples, two per case (one per task), in case order: case 0 select, case 0 lookup, case 1 select, … |
| `manifest.json` | `version`, `generator` (`git_commit`; `sha256` of `data.py`, `study3.py`, `sylang_batch.py`, `language.py`; `packet_english_renderer_sha256`: sorted SHA-256 digests of the packet modules that render the `english` arm, digests only, never a path), `seeds`, `sizes` (cases per split), `counts` (examples per file), `sha256` per file (the files are byte-reproducible from the seeds), `split_hashes` (SHA-256 over each case file's case hashes), `legend_tokens` per arm and format, `token_stats` (mean prompt tokens per arm, task, split; mean answer tokens), `render_seconds` (per arm and split: `n`, `mean`, `p95`, `max` seconds of the arm's rendering per case, the contract's conversion-in), `heldout` (the wordings and feature pairs), `generated_at`. Datasets are frozen: `data.py` exits with status 1 when `data/<version>/manifest.json` exists unless `--force` is given |

The nested prefix rule: the training stream of a seed holds `train_cases` cases; a config's
`train_size` = N uses the examples whose `case_index` < N (`common.training_examples`). Dev and
test are shared by all seeds.

### Case line

```
{"case": "test-c00017", "case_index": 17, "split": "test", "panel": "iid", "k": 7,
 "records": [{"subject": "ab12", "relation": "see", "object": "zq9", "polarity": "negative",
              "tense": "past", "aspect": "simple", "evidence": "unspecified"}, ...],
 "wording": {"select": 2, "lookup": 0}, "query": {"subject": "ab12", "object": "zq9"},
 "heldout_wording": false, "heldout_pair": false}
```

`panel` is `iid` for train and dev; one of `iid`, `wording`, `pair`, `long` for test. `wording`
indexes the task's wording list (0–3 trained, 4–5 held out). `query` is the lookup pair: a uniformly drawn record of the case (the case's own random stream),
except on the `pair` panel where it is the record forced onto a held-out feature pair (the index is
still drawn, so the draw order is identical on every panel); `data.py --check` verifies that every
pair-panel query record is held out. Records sharing a
subject are adjacent (contract), and no (subject, object) pair repeats within a case.

### Example line (one per case and task, identical `answer` bytes across arms)

```
{"id": "test-c00017-select", "case": "test-c00017", "case_index": 17, "split": "test",
 "panel": "iid", "task": "select", "arm": "sylang2", "k": 7, "max_id_len": 5, "n_negative": 2,
 "wording": 2, "heldout_wording": false, "heldout_pair": false,
 "records": "ab12\tSp\tzq9\n...",                       # the rendering, ends with "\n"
 "instruction": {"json": "List [subject,object] of each negative record in order; [] if none.",
                 "bare": "..."},                           # no trailing newline
 "answer": {"json": "[[\"ab12\",\"zq9\"]]", "bare": "ab12 zq9"},
 "query": {"subject": "ab12", "object": "zq9"}}           # lookup examples only
```

Prompt assembly (`common.py`): user message = `records` + `instruction[answer_format]`; the
full prompt string is `PROMPT_TEMPLATE` with the arm's legend as the system message; the
training target is `answer[answer_format]` + `<|im_end|>`. The retry request is the same
prompt with `FORMAT_REMINDER[(task, answer_format)]` appended to the user message on a new line.

Bare answers: *select* = `subject object` pairs in record order joined by single spaces, or the
word `none`; *lookup* = `relation polarity`.

## Grader: `grader.py`

`grade(task, answer_format, expected: str, output: str) -> dict` with keys `correct` (bool),
`invalid` (bool: the output could not be parsed into the task's shape), `reason` (short
string: `ok`, `parse`, `shape`, `wrong`, `extra_text`, …). Parsing is strict: for `json` the
whole output (after stripping whitespace) must be exactly one JSON value of the right shape;
trailing prose is `invalid`. Equality is exact on the canonical value (list of pairs; object
with exactly the two keys), not on the output bytes, so `[["a", "b"]]` with spaces is accepted.

## Configs: `configs/<name>.json`

A JSON object whose keys are a subset of `common.DEFAULT_CONFIG`; `name` defaults to the file
stem. `model` is `tiny-random` or a local directory; `train_size` counts cases; `tasks` is the
task mix; `answer_format` selects the instruction/answer/legend variant; `max_steps` caps the
optimizer steps (smoke tests).

## Training: `train_lora.py --config C --out runs/<name>`

Writes `runs/<name>/`:

| file | content |
|---|---|
| `run.json` | the record below, rewritten at every checkpoint |
| `checkpoints/epoch<N>/` | PEFT adapter (`adapter_model.safetensors`, `adapter_config.json`) plus `trainer_state.pt` (optimizer, scheduler, RNG, step) for resumption |
| `best/` | copy of the dev-selected adapter; `--resume` re-derives it from `checkpoints/epoch<selected_epoch>/` (a kill may have lost the copy) |
| `train_log.jsonl` | one line per logging step: `step`, `epoch`, `loss`, `lr`, `examples_seen`, `tokens_seen`, `seconds`; on `--resume` the lines past the checkpoint's `step` (written by the killed attempt) are dropped before training continues, so no step appears twice |

`run.json`:

```
{"name": ..., "config": {...}, "status": "running|completed|failed|timeout", "error": null,
 "git_commit": ..., "started_at": ..., "finished_at": ..., "seconds": ...,
 "device": "cuda", "gpu_name": ..., "dtype": "bfloat16", "model_path": ..., "model_params": ...,
 "trainable_params": ..., "examples": {"train": N, "dropped_too_long": M, "dev_eval": D},
 "steps": S, "epochs_done": E, "examples_seen": ..., "tokens_seen": ...,
 "train_loss": [{"epoch": 1, "mean_loss": ...}, ...],
 "dev": [{"epoch": 1, "accuracy": ..., "invalid_rate": ..., "by_task": {...}, "seconds": ...}, ...],
 "selected_epoch": 2, "peak_memory_bytes": ..., "checkpoint_bytes": ..., "resumed_from": null}
```

`--resume` continues from the last complete `checkpoints/epoch<N>/`. Dev outputs are decoded with
`common.decode_output` (text before the first `<|im_end|>`, special tokens skipped), the same rule
`evaluate.py` grades by, so checkpoint selection sees exactly the text the evaluator sees. Every
exception is caught and recorded (`status: failed`) before re-raising; the stage scripts' `timeout` is
recorded by the scripts as `status: timeout` in the ledger (the process cannot record it).

## Evaluation: `evaluate.py --config C --adapter runs/<name>/best|none --split test|dev --out results/<name>`

`--adapter none` is the zero-shot baseline (legend, no training). Writes `results/<name>/`:

| file | content |
|---|---|
| `eval.json` | aggregates below |
| `items.jsonl` | one line per example; no prompt text, no outputs (synthetic ids only in `id`) |

`items.jsonl` line:

```
{"id": ..., "panel": ..., "task": ..., "k": 7, "max_id_len": 5, "n_negative": 2,
 "heldout_wording": false, "heldout_pair": false,
 "correct_first": true, "invalid_first": false, "retried": false, "correct_final": true,
 "invalid_final": false, "reason_first": "ok", "reason_final": "ok",
 "tokens": {"system": 112, "records": 60, "instruction": 18, "template": 13, "prompt_total": 203,
            "output_first": 11, "output_retry": 0, "prompt_retry": 0},
 "latency_s": 0.41, "latency_mode": "batched|single", "convert_in_s": ..., "tokenize_s": ..., "convert_out_s": ...}
```

`eval.json`:

```
{"name": ..., "config": {...}, "adapter": ..., "split": ..., "model_path": ..., "git_commit": ...,
 "device": ..., "dtype": ..., "started_at": ..., "seconds": ..., "generation": {"greedy": true,
 "max_new_tokens": {...}, "batch_size": ...}, "n_items": ...,
 "panels": {"<panel>": {"<task>": {"n": ..., "acc_first": ..., "acc_final": ..., "invalid_first": ...,
                                    "invalid_final": ..., "retry_rate": ...}, "all": {...}}, "all": {...}},
 "tokens": {"per_request_mean": {"system": ..., "records": ..., "instruction": ..., "template": ...,
                                 "prompt_total": ..., "output": ...},
            "totals": {"uncached": ..., "cached": ..., "output": ..., "requests": ..., "retries": ...}},
 "latency": {"batched_items_per_s": ..., "single": {"n": ..., "mean_s": ..., "p50_s": ..., "p95_s": ...,
                                                "by_task": {"<task>": {"n": ..., "mean_s": ..., "p50_s": ..., "p95_s": ...}}}},
 "conversion": {"in_mean_s": ..., "in_source": "...", "tokenize_mean_s": ..., "out_mean_s": ...},
 "cost": {"price_per_hour": ..., "gpu_seconds": ..., "per_correct_final": ..., "per_item": ...}}
```

`convert_in_s` is the codec time: the manifest's `render_seconds` mean for the config's arm and the
evaluated split (0.0 when the manifest has no such key); `tokenize_s` is the prompt assembly and
tokenisation wall-clock of the first request plus, when retried, of the retry (the codec is not
re-run); `convert_out_s` is the parse and grade wall-clock only; eval.json's `conversion.in_source` states that `in_mean_s` was measured at generation, not during the evaluation. Generated ids are decoded with
`common.decode_output` (the ids before the first `<|im_end|>`, special tokens skipped, the rule
`train_lora.py` uses for dev selection); output token counts are the tokens before the end token.
Immediately before the timed generation one untimed warm-up generation runs on the first request
(`generation.warmup_s`; inside `cost.gpu_seconds`, outside `seconds_generation`).
`cached` counts the system message once per 100 requests (contract); `uncached` counts every
token of every request including retries. `--price-per-hour` (default 0) fills `cost`.
`--latency-sample N` measures N items one at a time, the N split evenly over the tasks and spread
evenly within each task's items (`latency_mode: single`); the rest are batched and carry the batch's
wall-clock divided by the batch size.

## Analysis: `analyze.py --results results/*/ --out analysis/`

Groups result directories by `(model, arm, seed, train_size, answer_format, split)` from each
`eval.json`'s config, where `model` is the config's `model` when set, else the basename of the
`model_path` the evaluator recorded (the stage configs leave `model` null and `PILOT_MODEL` supplies
the weights), else `unknown`; the result records and the inventory table carry the full `model_path`,
and the warning about superseded directories prints each directory's `model_path`. Writes `analysis/summary.md` and `analysis/summary.json` with: accuracy
tables per panel and task; the contrasts `arm − english`, `sylang2g − sylang2`, `sylang2 −
json` per panel, task and train size with the 90% cluster bootstrap (seeds then items, 2,000
resamples, seed 0), the per-item-only interval, and the verdict `non-inferior | inferior |
superior | inconclusive` under δ = 3 (contract), issued only with three or more shared seeds
(`MIN_DECISION_SEEDS = 3`); with fewer the verdict is `exploratory (1 seed): no decision` or
`exploratory (2 seeds): no decision`; the secondary `correct_first` contrasts carry intervals only; learning curves (accuracy vs train size); token, latency and cost
tables; the retry rates. `--delta` and `--resamples` override the defaults and the overrides
are printed in the output.

## Profiling: `profile_gpu.py --config C --out profile.json [--price-per-hour P] [--upload-cmd '...']`

(The script is not called `profile.py` because that name shadows the standard-library module that torch imports.)

Measures, on the current machine: model load seconds, parameter count, peak memory, training
examples/s and tokens/s over `--train-steps` (default 30) steps with the config's batch, dev
generation items/s over `--gen-items` (default 64), checkpoint save seconds and bytes, and the
upload seconds if `--upload-cmd` is given (a shell command receiving the checkpoint path).
`upload` in `profile.json` is `{"program", "seconds", "returncode"}` (the first word of `--upload-cmd`); the
command line and its output are never written (they may carry credentials); on a non-zero exit the
last 500 characters of stderr are printed to the console only. Every plan file is read and checked
before anything is measured (each cell must carry `config`, `train_cases`, `dev_evals`,
`dev_eval_items`, `eval_items`, the counts numeric); a malformed plan aborts with the file and cell
named before any GPU time. The measured profile is written with `projection: null` before
projecting; if the projection fails the file is rewritten with `notes: ["projection failed: …;
measured values kept"]` and the script exits non-zero. `configs/stage_a_profile.json` profiles the
`json` arm, the longest rendering, so the single-rate projection over-estimates the other arms.

Writes `profile.json` with those numbers and, for the cell lists in `configs/plan_stage_b.json`
and `configs/plan_stage_c.json` (or the `--plan NAME=FILE` overrides), the projected seconds and
cost of each stage at the given price with a 1.5× contingency (failed runs and retries), under
`projection`. Per cell, two phases, each loading the model (`train_lora.py` and `evaluate.py` each
load it): `train_phase_s` = `load_s` + `train_s` + `dev_gen_s` + `checkpoint_s` (save seconds ×
epochs), what the training timeout covers; `eval_phase_s` = `load_s` + `test_gen_s`, what the
evaluation timeout covers; `epoch_s` = `load_s` + (`train_s` + `dev_gen_s`)/epochs + save seconds,
the time to the first checkpoint; `seconds` = `train_phase_s` + `eval_phase_s` + `upload_s`.
`exceeds_timeout` = max(`train_phase_s`, `eval_phase_s`) > `timeout_minutes` × 60 and
`epoch_exceeds_timeout` = `epoch_s` > the same; a flagged cell prints a `!!!` line (config,
projected training and evaluation minutes against `timeout_minutes`, 'raise timeout_minutes or
shrink the cell before starting the stage'), also kept in `notes`.

Plan file format (`configs/plan_stage_b.json`, `configs/plan_stage_c.json`): `{"stage": "B", "cells": [{"name", "config", "arm", "answer_format", "seed", "train_size", "train_cases", "train_examples", "epochs", "batch_size", "max_length", "optimizer_steps", "dev_evals", "dev_eval_items", "eval_items", "timeout_minutes", "optional"}, ...], "zero_shot": [{"name", "config", "arm", "answer_format", "adapter": "none", "eval_items"}, ...], "totals": {...}, "totals_with_optional": {...}}`; the profiler reads `config`, `train_cases`, `epochs`, `eval_items`, `dev_evals` and `dev_eval_items` of every cell (optional cells included, zero-shot evaluations not projected: both conservative directions are inside the contingency). Upload convention, here and in the stage scripts' `PILOT_UPLOAD_CMD`: the path is appended as the command's last argument; there is no placeholder substitution.

## Ledger: `ledger.py`

`ledger.jsonl` (append-only, committed) lines:

```
{"ts": ..., "event": "start|stop|note|balance", "provider": ..., "instance": ..., "gpu": ...,
 "price_per_hour": ..., "stage": "A|B|C", "purpose": ..., "run_names": [...],
 "measured_spend": ..., "balance_after": ..., "note": ...}
```

`ledger.py add --event ... --provider ... [...]` appends; `ledger.py report --ceiling 100`
prints the total measured spend, spend per stage, the number of completed experiments (result
directories with `eval.json`), dollars per completed experiment, and exits non-zero when the
spend or the projection in `profile.json` exceeds the ceiling. A `balance` line that carries `measured_spend` is the provider's own figure for its `stage` and overrides the stop line's elapsed-time estimate in the report; `results/smoke_*` directories (free CPU checks) are not counted as experiments; a line containing the value of `PILOT_PROVIDER_TOKEN` or an API-key-shaped string is refused.

## Stage scripts

`run_stage_a.sh`, `run_stage_b.sh`, `run_stage_c.sh`: `set -euo pipefail`; refuse to start
unless `PILOT_TOKENIZER`, `PILOT_MODEL`, `PILOT_PRICE_PER_HOUR`, `PILOT_PROVIDER`, `PILOT_INSTANCE` and `PILOT_GPU` are set (the last three label the ledger lines; optional: `PILOT_CEILING`, `PILOT_STAGE_BUDGET`, `PILOT_UPLOAD_CMD`, `STAGE_B_CURVES=1`, `STAGE_C_FILL_SEED0=1`); append a ledger
`start` line; run each cell under `timeout` with the config's `timeout_minutes`; on timeout
or failure append a ledger `note` and continue with the next cell; run the evaluation and the
analysis; append a ledger `stop` line; print the `verify_teardown.md` checklist. They never
call a provider API: starting and stopping the instance is the owner's action, recorded in the
ledger by hand with `ledger.py add`. Stage A runs `profile_gpu.py` on `configs/stage_a_profile.json` (with
`--upload-cmd "$PILOT_UPLOAD_CMD"` when set), then, unless `STAGE_A_END_TO_END=0`, trains and evaluates
the 500-case `sylang2` cell `stage_b_sylang2_s0_n500` end to end (the optional learning-curve point of
stage B, skipped there once it exists), optionally the `english` zero-shot baseline (`STAGE_A_ZERO_SHOT=1`),
and finally `ledger.py report --ceiling ${PILOT_CEILING:-100} --profile profile.json`, whose exit code the
script returns: stage B is not started on a non-zero exit. The 1.7B extension has its own configs
(`stage_d_1p7b_*`) and plan file (`plan_stage_d_1p7b.json`), projected with `profile_gpu.py --plan` after
profiling the 1.7B weights; no stage script runs it.

`smoke_cpu.sh`: removes any previous smoke dataset and generates a small one (`data.py --smoke`: 64 train cases per seed for 2
seeds, 16 dev, 32 test), builds the tiny model, trains 20 steps for two arms, evaluates
zero-shot and trained, runs the analysis; exits non-zero on any failure.
