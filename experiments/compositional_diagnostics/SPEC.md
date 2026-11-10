# Compositional diagnostics benchmark: specification (cd-v0)

Status: evaluation-only benchmark, generated and checked on CPU. No model has been run on it. It is new work beside
the pretrained pilot; it does not modify the pilot, its frozen data, configs or results.

## 1. Why

The pilot's two tasks cannot test whether a model reads how a record's meaning is composed
([reports/existing_tasks.md](reports/existing_tasks.md), computed on the frozen v1 cases):

- No single-field edit of tense, aspect or evidence changes a *select* or a *lookup* answer (0 of 55,923 edits on
  the test cases). Those features are rendered in every arm and never tested; the held-out-pair panel asks only for
  relation and polarity.
- A solver that reads one character per record (the case and letter of the code's first character, or the json
  keys `r` and `neg`) scores 100% on both tasks on every panel in sylang2, sylang2g and json.
- In 78–90% of lookup queries the object identifier alone finds the queried record, so lookup accuracy cannot
  separate binding both identifiers from matching one.

These are properties of the tasks, not findings about any model. This benchmark asks questions whose answers
depend on the untested features and on binding, and pairs each question with controls.

## 2. Records and cases

The pilot's seven-field record (`subject`, `relation` see|help|contain, `object`, `polarity`, `tense`,
`aspect`, `evidence`) and its two structural invariants ([schema.py](schema.py)): no (subject, object) pair
repeats and no self-pair; **records sharing a subject are adjacent**, which is what makes the grouped format
`sylang2g` a lossless re-layout (record order = line order, then item order within a line). Identifiers are 2–5
characters of `[a-z0-9]` with a letter, never a vocabulary word. Cases have 4–13 records.

## 3. Questions ([questions.py](questions.py))

| type | asks | answer JSON | depends on |
|---|---|---|---|
| `tense` / `aspect` / `evidence` | that feature of the one record with subject X and object Y | `"past"` | the asked feature of the record bound by *both* X and Y |
| `profile` | all five features of that record | `{"relation":…,"polarity":…,"tense":…,"aspect":…,"evidence":…}` | all five features, jointly |
| `filter` | every (subject, object) whose record meets two conditions on distinct features, at least one of tense/aspect/evidence | `[["s","o"],…]`, `[]` if none | the two features, in the *same* record |
| `join` | every Z with records (X, rel_a, Y, positive, tense_a) and (Y, rel_b, Z, positive) for the same Y | `["z",…]`, `[]` if none | relation, polarity, tense, and the shared Y (bounded: exactly two records) |

One English instruction per type, identical in every arm; set answers are order-free.

## 4. Families ([families.py](families.py))

Every question comes as a family of six members built from one base case before rendering:

| member | edit | the answer must |
|---|---|---|
| `base` | – | – |
| `relevant` | one feature of one record that the answer depends on | **change** |
| `irrelevant` | one feature that it does not depend on (single-feature types: another feature *of the queried record itself*; profile: a feature of another record; filter: a third feature of a matching record; join: aspect or evidence of a hop) | stay the same |
| `rename` | every identifier replaced consistently by a fresh identifier of the same length and split | be the base answer, renamed |
| `reorder` | subject groups permuted and records permuted within groups (only orders sylang2g can express) | stay the same |
| `distractor` | one to three records added that share identifiers with the question and fail it | stay the same |

The generator verifies every expectation with the oracle (redrawing a family that fails), that `relevant` and
`irrelevant` differ from base in exactly one field of one record, that `reorder` is a grouped permutation,
`distractor` only adds records and `rename` is a bijection onto unused identifiers.

Built-in traps, so the corresponding shortcuts meet their failure case: pair questions carry a subject decoy (X,
another object) and an object decoy (another subject, Y) with a different value of the asked feature; filter
carries two half-matching records under one subject (one per condition); join carries a chain broken by the wrong
tense, a link not reached from X, and a negated hop.

Balance: the asked feature's base value (single-feature types), the number of matches (filter: 0,1,1,2,2,3) and of
chains (join: 0,1,1,2) cycle through shuffled balanced blocks; `join` relevant edits choose between breaking and
completing a chain with equal probability when both exist.

## 5. Renderings ([render.py](render.py))

`english` (controlled English written for this benchmark: one numbered sentence per record, the verb form carries
tense and aspect, an opening phrase the evidence; **not** the pilot's packet English, which is not public), `json`
(byte-identical to `study3._json_record`), `sylang2` (the public codec `sylang_batch.encode(version=2)`),
`sylang2g` (identical to `data.group_by_subject`). Every rendering is parsed back and must equal its records. The
system message per arm is `legends.json` (the pilot's arm legends, english adapted, plus a task line). Suggested
prompt: system = `legends[arm]`; user = `records` + `instruction`; target = `answer`.

## 6. Splits and leakage

Whole families are assigned to dev or test before rendering: each (split, type) has its own stream
`random.Random("cd-v0:<seed>:<split>:<type>")` and draws identifiers only from its split, dev from the pilot's dev
hash bucket and test from its test bucket (`study3.split_of`), so **no identifier is a pilot training
identifier** and dev and test share none. A family is redrawn if any member's structural signature (invariant to
renaming and reordering) matches any member of an earlier family in either split. Default size: 40 dev and 100
test families per type (240 / 600 families, 1,440 / 3,600 items per arm).

## 7. Grader ([answer_grader.py](answer_grader.py))

The whole output, after whitespace and at most one Markdown fence around all of it (counted as `fenced`, the
pilot's 2026-10-22 rule), must be one JSON value of the type's shape; prose, a missing or extra key, a duplicate
key or a duplicate set element is `invalid`. Correct only when the whole value equals the expected one: no
partial credit.

## 8. Metrics ([metrics.py](metrics.py))

Per family: `exact` (share of the six correct), `exact_base`, `family` (all six correct), `relevant` (base and
relevant both correct; their expected answers differ, so ignoring the change earns at most one), `robust_<v>`
(base and v both correct) and `robust_all`. Invariance is never credited on its own: `raw_invariance_<v>`
(outputs agree, after renaming, right or wrong) and `raw_sensitivity` are uncredited diagnostics. A constant
answerer has raw invariance 1.0 and `relevant` = `family` = 0.

Intervals: percentile bootstrap over **families** (the unit within which items are related), stratified by type
for pooled numbers, 2,000 resamples, seed 0, 95%. Chance: the exact expectation of each metric for an independent
uniform guesser over each member's answer space (3, 3, 4 values; 216 profiles; all subsets of the case's records
or identifiers); the report also gives empirical `random_guess` and `constant_majority` rows.

## 9. Weak solvers and targeted checks ([solvers.py](solvers.py), [baseline.py](baseline.py))

Constant (dev majority), uniform random, first record, last record, blind to each of the five features, subject
only, object only, feature bag (pools a subject's records), unbound join, role swap. Each is the oracle with one
defect. `TARGETS` names, for each, the metric that must expose it; `SPECIFICITY` names metrics where the defect is
harmless and must not be penalised (ignoring aspect is fine on tense questions). All pass on cd-v0
([reports/baseline_report.md](reports/baseline_report.md)) and on an independently seeded small set in the tests.

## 10. Oracle verification ([verify_oracle.py](verify_oracle.py), [reference.py](reference.py))

The reference shares no code with the oracle: records as positional tuples, each question compiled to a
conjunctive query and answered by backtracking. On three exhaustively enumerated domains (all 1,296 one-record
cases with the full feature space; all 15,360 two-record cases and all 10,240 three-record cases over reduced
feature spaces; every askable question; both record orders) oracle, reference and the solvers' defect-free path
agree on 5.6 million evaluations. A test plants two bugs in the oracle and checks the comparison catches both.
Hand-written renderings and answers: [tests/test_handchecked.py](tests/test_handchecked.py).

## 11. Known limits

- One instruction wording per type; wording robustness is not tested.
- The english arm is this benchmark's own renderer, so english results here are not comparable with the pilot's
  english arm.
- Lengths are audited in characters; no tokenizer was loaded (token counts need `PILOT_TOKENIZER`).
- Queried records sit less often at the very first or last position (6–15% first, 8–10% last, against 17% expected
  for a uniform position) because they sit inside the largest subject group; the first- and last-record solvers stay near chance.
- In the code arms an irrelevant edit changes the rendering's length less often than a relevant one; this is
  visible only when two members are compared, never inside one prompt.
- The domain is a closed synthetic schema: three relations, five features, joins of depth two.
