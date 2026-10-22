# Frozen data contract (pilot v1)

## Underlying case

A *case* is an ordered list of K records with the study's seven fields: `subject`, `relation`
(see | help | contain), `object`, `polarity` (positive | negative), `tense` (past | present |
future), `aspect` (simple | progressive | completed), `evidence` (unspecified | direct | reported |
inferred). Identifiers are 2–5 characters from `[a-z0-9]` with at least one letter, drawn by the
v3 hash split (`study3.split_of`). **Subject reuse**: with probability `p_reuse` (0.5 in pilot v1)
a record's subject repeats an earlier record's subject, so grouped renderings have something to
group and the data resemble a handoff where one actor carries several facts; objects are unique
unless reused with probability 0.2. **Record order**: a record whose subject was used before is
placed directly after the last record of that subject, so records sharing a subject are adjacent
in every arm (grouping by subject is then a pure re-layout and "record order" is unambiguous);
no (subject, object) pair repeats within a case, so every lookup query has exactly one record.
K is uniform in 1..12 for training; 13..16 is a held-out
length band. Two feature pairs are held out of training as in the study: (contain, future) and
(help, completed).

## Tasks (identical answer bytes across renderings)

1. **select** (the study's task): list the `[subject, object]` pairs of the negated records in
   record order, as a JSON list; `[]` if none.
2. **lookup** (relationship lookup, new): given a subject and an object that occur together in
   exactly one record, return that record's relation and polarity as a JSON object
   `{"relation": ..., "polarity": ...}`; the queried pair is chosen uniformly among the case's
   records, except on the held-out-pair test panel, where the query is the record carrying the
   held-out feature pair (so every pair-panel lookup tests the held-out combination). The request
   names the pair in the instruction line.

Request wordings: four trained, two held out (per task; the lookup wordings are new strings of
the same style).

## Renderings (arms)

| arm | rendering of the records |
|---|---|
| `english` | controlled English, numbered records (the packet renderer's `english` payload) |
| `json` | compact JSON array of objects with one-letter keys and defaults omitted |
| `sylang2` | header-free Sylang codes, `subject<TAB>code<TAB>object` lines (the tested compact format) |
| `sylang2g` | the improved-M candidate: header-free codes **grouped by subject** — one line per subject, `subject<TAB>code object;code object`, record order preserved by first occurrence (the single reversible change under test) |

Every arm's prompt = system message (legend for that arm, 100–130 Qwen 3 tokens, arms matched
within 15 tokens of each other, asserted by `data.py`; the grouped and ungrouped code legends are
token-identical) + user message (records in the arm's rendering, then the task's
instruction line). Output = the task's JSON answer. A second **answer format** factor
(`bare`: `subject object` pairs separated by spaces for *select*, the word `none` when there is
no negated record; `relation polarity` for *lookup*) exists only for stage C and is never adopted
by the pipeline from this pilot. File formats, prompt assembly and the retry text: `interfaces.md`.

## Splits

Split by underlying case before rendering, keyed by a hash of the case's record tuple: the same
case never appears in two splits in any rendering. Identifiers follow the v3 train/dev/test hash
split. Held out of training entirely: two wordings per task, the two feature pairs, K 13..16.
Sizes (pilot v1): train up to 8,000 cases per seed (nested prefixes 500 / 2,000 / 8,000 of one
stream per seed), dev 400 cases, test 1,000 cases (iid 480, held-out wording 160, held-out pair
160, K 13..16 200); dev and test are shared by all seeds; each case yields one example per task. `manifest.json` records every seed, size and the SHA-256 of each file, the SHA-256 digests of the
packet modules that render the `english` arm (digests only, never the packet's path) and the
generator commit; a version directory is frozen: `data.py` refuses to regenerate into an existing
`data/<version>/` unless `--force` is passed.

## Grader

Deterministic. *select*: parse JSON; must be a list of 2-element string lists; exact match of
the ordered list of pairs. *lookup*: parse JSON; must be an object with exactly `relation` and
`polarity`; exact match. Any parse failure or shape violation is a wrong answer and is counted
separately as `invalid`. **Amendment 2026-10-22, before any GPU run**: one Markdown code fence
wrapping the whole output (```` ```json … ``` ````) is removed before parsing and counted as `fenced`
(reported per arm as a rate and in output tokens); the real-weights CPU smoke showed Qwen3-0.6B
fencing every zero-shot answer, and fencing is a chat habit, not a reading error. Prose outside the
fence is still `invalid`. Per-item flags are stored. `test_grader.py` feeds corrupted outputs
(dropped pair, swapped identifiers, invented pair, reordered pairs, wrong polarity, missing key,
extra key, malformed JSON, trailing prose) and asserts each is rejected.

## Retry policy

One retry per request when the first output is invalid, with the instruction repeated and the
output format reminded; both requests are counted in tokens, latency and cost; accuracy is
reported before and after the retry; the retry rate per arm is a reported number.

## Token accounting (the model's own tokenizer)

Per request: system tokens (legend), instruction tokens, record tokens, output tokens (first
and retry); two conventions reported side by side: *uncached* (every token of every request) and
*cached* (the system message counted once per 100 requests, the provider's cache discount not
assumed). Conversion-in wall-clock is the arm's rendering time of the case's records (the codec), measured
at generation and summarised per arm and split in the manifest (`render_seconds`: mean, p95, max);
conversion-out (parser and grader) wall-clock is measured per request by the evaluator, with the
prompt assembly and tokenisation time reported separately.

## Decision rules (pre-registered; δ = 3 accuracy points)

For a contrast A − B on a panel, the primary interval is a 90% **cluster bootstrap**: resample
seeds with replacement, then items within each seed, 2,000 resamples, on the paired per-item
differences averaged over the resampled seeds. (The per-item-only interval is reported beside
it.) *Non-inferior*: lower bound > −δ. *Inferior*: upper bound < −δ. *Superior*: lower bound
> +δ. *Inconclusive*: the interval reaches below −δ without lying wholly below it. With one seed
per arm (stage B) no decision is taken; the numbers are exploratory and only size the expansion.
With three seeds (stage C) the rule applies; an inconclusive result at three seeds is reported as
such and may be followed by two more seeds for the two arms in the contrast if the ceiling
allows, after which the result stands as reported.

Diagnosing a failure (stage B and C): a format fails only if its learning curve is flat or
below the others at every size while at least one arm learns; if every arm is below 90% at
the largest training size run (8,000 items in the full plan) with training loss still falling, the
model is undertrained (more items or epochs); if every arm fails and the zero-shot baseline is also
at chance on `english`, suspect the pipeline (legend, chat template, decoding, grader) before the
model; if `english` and `json` learn and
the codes do not, and the gap shrinks with data, the result is a capacity/data question that
warrants the 1.7B run; if the gap does not shrink, it is a format result.
