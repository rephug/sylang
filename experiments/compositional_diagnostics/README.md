# Compositional diagnostics for Sylang

An executable, CPU-only benchmark that asks whether a reader of the four pilot renderings (controlled English,
compact JSON, `sylang2`, `sylang2g`) understands how a record's meaning is composed: tense, aspect and evidence,
several features at once, and two records bound through a shared entity. Every question comes with matched
controls (a relevant edit that must change the answer; an irrelevant edit, a consistent renaming, a grouped
reordering and added distractors that must not), so answer patterns, positions and memorised identifiers do not
earn credit.

- Specification: [SPEC.md](SPEC.md)
- What the pilot's select and lookup tasks can and cannot measure: [reports/existing_tasks.md](reports/existing_tasks.md)
- CPU baseline report (weak solvers, audits, targeted checks, oracle verification): [reports/baseline_report.md](reports/baseline_report.md)
- Rendered examples: [samples/families_sample.md](samples/families_sample.md)

Nothing here trains a model, loads weights, uses a GPU, calls an API or touches the pilot's files. The pilot's
`data.py`, `study3.py` and the public codec `sylang_batch.py` are imported read-only ([pilot_compat.py](pilot_compat.py)
supplies empty placeholders for `torch` when it is not installed; the regenerated v1 case files match the frozen
manifest's hashes, which `analyze_existing.py` checks).

## Commands

Python 3.10+ standard library only. From this directory:

```bash
./run_all.sh                                  # everything below, about 3 minutes on one CPU core

python3 -I generate.py --force                # data/cd-v0/ (seed 0; 40 dev + 100 test families per question type)
python3 -I generate.py --check                # regenerate in memory and compare every file's SHA-256
python3 -I analyze_existing.py                # reports/existing_tasks.{json,md}
python3 -I verify_oracle.py                   # reports/oracle_verification.json (about 80 s)
python3 -I baseline.py                        # reports/baseline.{json,md}; exits 1 if a targeted check fails
python3 -I make_samples.py                    # samples/families_sample.md
python3 -I -m unittest discover -s tests      # 32 tests, about 30 s
```

Same seed and configuration give identical bytes, also under a different `PYTHONHASHSEED` (tested). The generated
data are not committed (`data/` is ignored); [manifest_cd-v0.json](manifest_cd-v0.json) records the SHA-256 of every
file, which `generate.py --check` reproduces. Other sizes or seeds: `--seed`, `--dev-families`,
`--test-families`, `--version`.

## Scoring a model

For each example line of `data/cd-v0/test_<arm>.jsonl` build the prompt from `legends.json[arm]` (system), then
`records` + `instruction` (user), and record the raw output as `{"id": ..., "output": ...}` per line. Then:

```bash
python3 -I score.py --arm sylang2 --outputs outputs.jsonl --out score.json
```

It prints exact, family, relevant and robustness scores with family-cluster bootstrap intervals and the
uniform-guess chance for the same items. The weak solvers in `solvers.py` are scored through the same code.

## Files

| file | role |
|---|---|
| `schema.py` | record schema, identifier grammar, grouped ordering rule, structural signature |
| `questions.py` | question types, instructions, oracle, canonical answers |
| `families.py` | base cases, the five controlled variants, traps, balance schedule |
| `render.py` | the four renderings, their parsers, legends |
| `generate.py` | deterministic generator and `--check` |
| `answer_grader.py` | strict full-answer grader |
| `reference.py`, `verify_oracle.py` | independent reference and the exhaustive comparison |
| `solvers.py` | oracle, chance and deliberately weak solvers |
| `metrics.py`, `score.py` | metrics, bootstrap, chance; scoring of an outputs file |
| `audit.py` | answer balance, identifier and near-duplicate overlap, position and length cues |
| `baseline.py` | the CPU baseline report and the targeted checks |
| `analyze_existing.py` | the analysis of the pilot's select and lookup tasks |
| `tests/` | unit tests, including hand-checked examples |

## What a later test with the existing Qwen checkpoints could establish

Running the pilot's Qwen 3 checkpoints (the zero-shot base model and the stage B/C LoRA adapters, once they exist)
on this test split, with no further training, could establish, per arm and with intervals that respect families:

- whether a model fine-tuned only on select and lookup answers questions about tense, aspect and evidence at all,
  that is, whether those features are usable from its reading of each format without having been asked for in
  training, against the zero-shot base model as the control;
- whether its correct answers survive the controls: a high `exact` with a low `family` or `relevant` score would
  show answering from patterns that the matched edits break, and which member fails (rename, reorder, distractor,
  the subject or object decoy, the unbound link) points to the shortcut, by comparison with the weak solvers'
  signatures in the baseline report;
- whether those properties differ between `sylang2`, `sylang2g`, `json` and this benchmark's `english`, on
  identical content, by more than the bootstrap intervals (about ±5–10 points per question type at 100 families);
- whether the identifiers matter: all test identifiers are outside the pilot's training bucket, and `rename`
  checks that answers move with the identifiers rather than with memorised strings.

## What it still could not prove

- **Not language understanding in general.** The domain is a closed synthetic schema (three relations, five
  features, depth-two joins, one instruction wording per type). Passing shows that a model computes these
  particular functions over these formats; it says nothing about natural text, open vocabularies, longer
  reasoning chains or other tasks.
- **Not the absence of a skill.** The checkpoints were trained on different questions and a different answer format.
  A low score can come from not following a new instruction or output shape (watch `invalid` and `fenced`) rather
  than from not representing tense or binding; only few-shot prompting or training on these question types (which
  needs approval) would separate the two.
- **Not the mechanism.** These are behavioural tests. A perfect score excludes the shortcuts listed in
  `solvers.py` and the cues audited in the report; it does not exclude shortcuts nobody has thought of, and it does
  not show how the model represents the records internally.
- **Not a comparison with the pilot's english arm.** This benchmark's controlled English is its own renderer.
- **Not a decision rule.** No threshold or contrast here is pre-registered for the pilot's decisions; a result
  would be exploratory unless a pre-registration is written first.
