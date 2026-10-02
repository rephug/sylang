# First executable baseline: 2026-10-02

The bounded codec works: **44 tests passed** and **165/165 semantic round trips
passed** across 33 fixtures and five formats. No model comprehension, inference,
training, latency improvement or monetary saving was measured.

The 33-fixture authored suite is a correctness smoke suite, not a representative
language benchmark. The two public holdout cases are visible and are not a
statistical generalization test. The same validated semantic record is used for
each format. Controlled English is deliberately templated, not arbitrary prose.

## Measured counts

These are sums of separately encoded fixture payloads or complete prompts. Each
column uses one unchanged tokenizer throughout; no cross-tokenizer ratio is used.
All punctuation, headers, feature fields and literal content are counted.

| Format | UTF-8 payload bytes | Qwen3 payload tokens | Qwen3.5 payload tokens | Qwen3 complete-prompt tokens | Qwen3.5 complete-prompt tokens |
|---|---:|---:|---:|---:|---:|
| Controlled English | 3,801 | 1,166 | 1,170 | 15,686 | 15,888 |
| Compact JSON | 8,611 | 2,245 | 2,249 | 9,505 | 9,740 |
| Named-field DSL | 6,559 | 1,912 | 1,916 | 11,548 | 11,783 |
| Experimental Prime | 6,262 | 1,872 | 1,876 | 11,409 | 11,644 |
| Experimental M | 2,408 | 1,030 | 1,034 | 12,151 | 12,386 |

M has 11.66% fewer payload tokens than controlled English under Qwen3 and 11.62%
under Qwen3.5 in this suite. That is not evidence for the historical 45–60% claim.
M also beats the verbose named-field DSL payload here; Prime and that DSL are
intentionally very similar structural controls. There is no learned morphology,
adaptive fusion, or tokenizer retraining in this result.

**JSON wins the complete-prompt comparison in this protocol.** M's payload
advantage does not offset its teaching material relative to JSON. The raw prompt
includes the common task/schema and a format-specific legend; it is not a real
model chat request. The English legend is longer than M's, so the full-prompt
English comparison is especially sensitive to instructional design. No model has
been tested for understanding any of these prompts. Output token costs, few-shot
examples and amortized/cached legend costs have not been measured.

## Fidelity findings that constrain interpretation

- All five deterministic codecs preserve every supported semantic field and
  literal code point in all 33 fixtures.
- Both unchanged tokenizers use **NFC normalization**. The `literal-decomposed`
  fixture loses codepoint-exact text fidelity in all formats. Each tokenizer thus
  preserves exact payload text for **32/33 fixtures per format**. This failure is
  reported, not fixed by normalizing away the fixture.
- `literal-special-token` produces registered special-token IDs in all formats.
  Text decoding is exact, but a future chat interface needs an explicit escaping
  policy. No unknown-token IDs were emitted.
- The headline totals above retain these diagnostics. They do not establish a
  lossless, ready-to-serve model input path.

Excluding the two named diagnostic fixtures consistently across all formats leaves
31 fixtures. Qwen3 then counts English/M payloads at **1,123/992** and JSON/M
complete prompts at **8,967/11,439**. Qwen3.5 counts them at **1,127/996** and
**9,188/11,660**, respectively. The main ordering persists; exclusion does not
establish broad semantic understanding or general-language efficiency.

The newer tokenizer offers no observed count advantage on this narrow suite.
That says nothing about its model's quality or general multilingual performance.
Unicode labels are preservation diagnostics, not translated-language benchmarks.

## Reproduction and evidence

[Full per-fixture machine-readable results](../benchmarks/results/initial-tokenizers.json)
include hashes, tokenizer configuration, raw counts, macro/micro/median/worst
ratios, development/holdout summaries, diagnostics and timings.
[The lock](../benchmarks/tokenizers.lock.json) pins both official tokenizer files
and their licenses/revisions/SHA-256 digests. The assets themselves are ignored.

Environment: Windows AMD64, Python **3.12.7**, Hugging Face `tokenizers` **0.22.2**.
The optional tokenizer wheel was fetched from official PyPI, pinned and verified;
the Windows wheel SHA-256 was
`c9ea31edff2968b44a88f97d784c2f16dc0729b8b143ed004699ebca91f05c48`.
It was installed in an isolated task environment without Hub dependencies because
only the local JSON loader is used. Effective vocabularies including added tokens:
Qwen3 **151,665**, Qwen3.5 **248,066**. These are not padded embedding row counts.

Commands actually run from the repository root (the local environment's Python
executable is abbreviated here as `python`):

```text
python tools/fetch_tokenizers.py --download
python benchmarks/build_fixtures.py
python -m unittest discover -s tests -v
python -m evaluation run --tokenizer-manifest benchmarks/tokenizers.lock.json --output benchmarks/results/initial-tokenizers.json --repetitions 5
python -m evaluation export-comprehension --output benchmarks/results/comprehension-tasks.jsonl
python -S -m evaluation run --output .cache/offline-proof.json --repetitions 1
```

The final unit tests completed in **2.409 seconds**. They include 216 exhaustive feature
combinations, random trees with seed **20261002**, independently hand-written
expected surfaces, all format pairs, Unicode and boundary cases, malformed
documents, fixture tampering, and comprehension-scorer rejection tests. The first
intermediate run encountered two stale-fixture errors while a new regression
fixture was being added; regeneration produced the final 33-case suite and all
43 tests passed. An additional independent minimal-pair regression test brought
the final passing count to 44. Standard-library-only evaluation also passed with
Python site packages disabled (`-S`). The final fixture SHA-256 is
`514bfd44281002154e3e41fc1484b92ca35a95377f46e704ee7d874af0f673e3`.

The exported **165 comprehension prompts** contain no answer keys and make no
model calls. The scorer accepts separately supplied local responses; model
comprehension remains unmeasured. See [evaluation.md](evaluation.md).

All codec process-CPU medians in this Windows run were **zero at the observable
timer granularity**. They are not zero-cost execution and provide no useful
latency estimate. Tokenizer timings are single CPU samples, not serving latency.
Use batched calibrated timings and model-level experiments before making speed
claims. Deterministic counts and hashes, rather than timings, are the reproducible
outcome of this run.

Independent technical review checked the codec, schema, fixed expected examples,
fixture integrity, tokenizer settings, source claims and the report tables.
Review found and resolved Unicode JSONL line-separator handling, implicit reserved
token policy and ambiguous timing/ratio documentation. No remaining correctness
or privacy blocker was found for this bounded experimental milestone. The limits
and unmeasured claims above remain explicit.

## Next concrete review

Review the experimental field meanings, condition scope and format legends first.
Then author a separate held-out comprehension suite and decide the literal
normalization/special-marker policy. If inference is separately approved, compare
all formats on the same frozen model, exposure and output budget. Keep Qwen3 as
the initial baseline; training and tokenizer adaptation still come later.
