# Offline evaluation of the experimental core

This scaffold compares five deterministic representations of the **same validated semantic tree**: controlled English, compact JSON, a named-field DSL, experimental Prime, and experimental M. It does not translate unrestricted English, train a model, download models, or call an inference service. The core schema defines the supported semantics; the familiar Prime/M names do not imply compatibility with older proposals.

## Run

Use Python 3.12. Run commands from the repository root. The core evaluation has no third-party dependencies:

```text
python -m unittest discover -s tests -v
python -m evaluation run --output benchmarks/results/offline.json
```

Optional tokenizer measurement requires the reviewed package pin in `requirements-tokenizers.txt` and previously downloaded, explicit local tokenizer JSON files. Asset downloading is a separate opt-in action; evaluation itself performs no network access. The checked-in lock records source URLs, immutable revisions, licenses, sizes, SHA-256 digests, and local paths. No model weights, Transformers, `trust_remote_code`, or model-specific Python code are used.

```text
python -m evaluation run --tokenizer-manifest benchmarks/tokenizers.lock.json --output benchmarks/results/tokenizers.json --repetitions 5
```

Custom fixtures must supply a matching integrity manifest:

```text
python -m evaluation --suite path/to/gold_suite.jsonl --fixture-manifest path/to/manifest.json run --output results.json
```

The optional tokenizer manifest is `{"tokenizers":[{"id":"MODEL","revision":"40-character commit hash","tokenizer_path":"relative/or/absolute/local/tokenizer.json","sha256":"FILE SHA256","license":"LICENSE","source":"SOURCE URL"}]}`. Relative paths resolve against the manifest directory. The loader verifies hashes and pinned `tokenizers==0.22.2` before using the Rust-backed `Tokenizer.from_file` loader; it never fetches missing files. Keep asset review and package installation separate from running the benchmark.

## Fixtures and scope

`benchmarks/gold_suite.jsonl` contains 33 newly authored synthetic MIT fixtures. Their provenance and exact count/size/hash are recorded in `benchmarks/manifest.json`. `python benchmarks/build_fixtures.py` regenerates both deterministically. Regeneration changes the manifest deliberately; evaluation rejects silent edits, unknown fields, duplicate IDs, and invalid semantic trees. JSONL records are separated by LF, while Unicode line separators remain literal string data.

The suite covers positive/negative contrasts; every tense/aspect combination; evidence distinctions; conditional scope and nested conditionals; quotes, backslashes and control characters; CJK/Arabic/emoji labels; composed/decomposed Unicode; literal tokenizer markers; code-like data and grammar delimiters; and minimum/maximum literal lengths. Invalid syntax/schema and depth/node limits are tested by the core unit/property tests. The evaluation tests add tampered hashes, duplicate fixtures/answers, unknown answer IDs, and incorrect/missing/invalid comprehension responses.

These are synthetic representational checks, not a representative language corpus. CJK and Arabic occur as literal entities inside the same English-based grammar; their counts cannot establish multilingual fairness. Long repeated literals and control strings are stress cases and can skew aggregate ratios. Inspect per-fixture results and tags. The two `holdout` cases are checked into this public repository and visible to developers: they are a split convention, **not an uncontaminated hidden evaluation**. Make a separately authored, frozen hidden suite before claiming generalization or using the public suite for training.

## What the output means

Every fixture/format row reports UTF-8 bytes, Unicode scalars, a deterministic payload hash, per-record gzip bytes, and exact decoded-AST equality. Equality preserves literal code points and every explicit semantic field. Gzip measures transport compression only; gzipped bytes are not model tokens and the model is not assumed to interpret compressed data. Small records include gzip overhead.

For each tokenizer independently, the harness measures:

- The bare payload token count.
- The complete instruction envelope, including the schema, format legend, reconstruction task, and payload, encoded **in one call**. Token counts are never obtained by adding separately tokenized fragments.
- Exact tokenizer encode/decode text equality, emitted unknown-token IDs, and emitted registered special-token IDs.

Each tokenizer uses `add_special_tokens=False`, `skip_special_tokens=False`, `encode_special_tokens=False`, no padding, and no truncation. The explicit special-token policy can recognize marker strings from the input as reserved IDs; those events are reported even when text decoding is exact. Such fixtures diagnose a transport hazard and must not be interpreted as safe chat messages. An eventual model interface needs a separate literal-escaping/chat-boundary policy. Both tokenizer configuration and effective vocabulary size including added tokens are included in the result metadata.

Serializer round-trip success and tokenizer text fidelity are separate outcomes. A tokenizer with NFC normalization can collapse decomposed literals even though the serializer itself preserves them. Counts from a failed tokenizer text round trip remain diagnostics, not evidence of lossless model transport. No normalization is applied to fixtures to make a result pass.

Summary ratios compare candidate format with English, compact JSON, and the DSL on the **same fixtures and tokenizer**. `micro_ratio` is the sum of candidate counts divided by the sum of baseline counts. `macro_ratio` is the unweighted mean of individual fixture ratios; median and worst (largest) ratios are also reported. A ratio below 1 means fewer tokens for that comparison. Totals and fixture denominators are explicit, and development/holdout summaries are separate. Never sum token counts across tokenizers or infer that equal token counts imply equal cost or model capability.

The instruction envelope is a fixed experimental prompt, **not** a model chat template or billable API request. The human-readable English legend is longer than the compact M legend; whole-prompt rankings depend on those authored instructions and are not an intrinsic compression property. The measured one-shot legend overhead does not establish few-shot or amortized performance. Future exposure studies should fix examples, vary instruction budget deliberately, and report the effect rather than tune one format's prompts against the evaluation set.

Codec timings are median process CPU nanoseconds over the requested repetitions after one warmup. Tokenizer timing is one process CPU sample per string, without warmup, excluding asset load time. These noisy local microbenchmarks are diagnostic only. They measure neither inference latency nor hosted pricing; end-to-end experiments must include model prefill/decode, output length, instruction/examples, retries, caching, and quality at a fixed model/task budget. Timing fields vary between runs even when payloads and counts are reproducible. There is no randomness in fixture construction, encoding, task order, or scoring; results record `seed=null` rather than implying an unused seed matters.

## Comprehension interface, without inference

Export all fixed reconstruction tasks as JSONL:

```text
python -m evaluation export-comprehension --output benchmarks/results/comprehension-tasks.jsonl
```

Each task includes a stable fixture/format identifier, split, versioned template, and the exact prompt. The identifier includes the fixture manifest hash prefix. Prompts ask for the whole semantic tree, which tests entities, relation, negation, time/aspect/evidence, and conditional scope together. Export contains no answer keys; scoring derives expected answers from the verified local gold suite. All fixtures remain public and discoverable, so this is not a secrecy mechanism.

Supply local responses as one JSON object per line with exactly `task_id` and `answer`, where `answer` is the model's parsed semantic JSON object:

```json
{"task_id":"core-comprehension-v1/HASH/plain-positive/english","answer":{"version":"core-v0.1","statement":{"kind":"pred","subject":"Ada","relation":"see","object":"a bird","polarity":"positive","tense":"present","aspect":"simple","evidence":"unspecified"}}}
```

```text
python -m evaluation score-comprehension --answers local-responses.jsonl --output benchmarks/results/comprehension-score.json
```

The scorer reports exact semantic matches, schema validity, missing responses, accuracy over all tasks, and accuracy over submitted tasks. It rejects duplicate or unknown IDs instead of silently choosing a response. Absent responses count against the all-task denominator. Store non-JSON model responses as a string `answer`; these fail schema validity. Record refusals, request failures, retries, and raw response provenance separately rather than dropping them. A perfect local reference-answer check validates the scorer and is **not a model-comprehension result**.

Before actual model comparison, pre-register the model and revision, full chat template, deterministic decoding or fixed seeds, equal exposure/examples per format, permitted retries, and independent semantic holdouts. Include schema-constrained and unconstrained outputs as different conditions; grammar validity alone does not prove faithful understanding. Add targeted minimal-pair accuracy and field/path diagnostics after this baseline. No inference run or training run is included in the current scaffold.
