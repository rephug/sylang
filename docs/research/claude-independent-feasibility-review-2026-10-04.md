# Independent technical feasibility review of Sylang

**Date:** 2026-10-04 · **Reviewer:** Claude (independent review; not an author of the reviewed code)
**Reviewed commit:** `0f7c5700c9a314b9f0b635bb4773c24caf785bf2` (branch `feat/reversible-core-evaluation`, draft PR rephug/sylang#1)
**Review branch:** `claude/festive-meitner-w5wagk` (fast-forwarded to the reviewed commit; no existing file modified)
**Companion plan:** [claude-independent-next-steps-2026-10-04.md](../plans/claude-independent-next-steps-2026-10-04.md)
**Reproduction package:** [`benchmarks/claude-review-2026-10-04/`](../../benchmarks/claude-review-2026-10-04/) (`fetch_assets.py`, `run.py`, `results.json`)

This review sits alongside the earlier
[technical review (2026-10-02)](technical-review-2026-10-02.md) and
[decisions](../decisions.md). It does not replace them. It deliberately challenges their conclusions.

## Evidence labels

| Label | Meaning |
|---|---|
| **[R]** | Reproduced in this review from the repository at the reviewed commit |
| **[M]** | New measurement made in this review (offline, deterministic, in `results.json`) |
| **[L-d]** | Literature or documentation read directly from the primary source during this review |
| **[L-i]** | Literature confirmed only through a search-engine index of the primary page (arXiv, ACL Anthology and Hugging Face could not be fetched) |
| **[A]** | Analytical argument or estimate, not a measurement |
| **[P]** | Proposed experiment, not run |

No model inference, training, fine-tuning, serving job or paid API call was made. No model weights were downloaded.

---

## 1. Verdict in brief

**No-go** for Sylang as a token-efficiency *language*. That covers the M/Prime notations as a cost-saving device, a custom tokenizer, vocabulary adaptation, and fine-tuning a model to learn the notation.

**Conditional go** only for a bounded falsification-and-salvage track. It costs about 1–3 weeks of engineering and at most a few hundred dollars of approved inference. The purpose is to close the open question with stronger baselines, then either stop the language track or repurpose the validated codec and harness.

Why:

1. **[M] The reported 11.6% saving is a property of a weak baseline, not of M.**
   - The controlled-English baseline carries a per-record `Sylang core-v0.1:` header (297 of its 1,166 Qwen3 tokens, 25%) and a mandatory "Without specified evidence," prefix.
   - Against *reversible* baselines that drop that redundancy, M uses **41% more payload tokens than concise controlled English** and **55% more than a compact word-based DSL** (Qwen3, all 33 fixtures). The ranges across seven tokenizers are 35–56% and 44–72%.
2. **[M] Cryptic one-letter codes buy nothing over ordinary words once redundancy is removed.**
   - A best-effort optimized M (defaults omitted, relation as head, one-letter codes) ties a readable DSL that uses English words: 670 vs 664 tokens on Qwen3, within ±1.4% on all seven tokenizers.
   - Words such as ` past` and ` progressive` are already single tokens. M's codes do not get their own tokens; they fuse with punctuation into pieces such as `:p`, `,g` and `+,`.
3. **[M] Complete chat prompts:** for 100 records, M is 35–36% more expensive than the readable DSL, both to read (Qwen3, full prompt) and to generate (Qwen3 tokens, output priced 5× input).
   - M beats the repository's verbose JSON once there are about 3 records per prompt, which overturns the "JSON wins the complete prompt" finding as stated. But any compact format does that.
   - M never beats concise English, the readable DSL or optimized M at any batch size.
4. **[M/L-d] There is no route through the tokenizer.**
   - Exotic glyphs cost the same or 1–3 extra tokens.
   - New tokens carry no meaning until trained. Training breaks compatibility with hosted APIs.
   - Hosted tokenizers change underneath you. Anthropic states that Claude 4.7 and later produce about 30% more tokens for the same text.
5. **[R] What does work:** the reversible codec. It is a well-tested serializer for a tiny closed schema, and the evaluation scaffold is carefully built. Both are reusable, but neither is evidence that a new language is needed.

Details, caveats and what would change this verdict are in §11.

---

## 2. Method and environment

- **Environment:** Linux, Python 3.12.3 (plus 3.11.15 for a second test run), `tokenizers==0.22.2`, `tiktoken==0.14.0`, `sentencepiece==0.2.2`. The original baseline ran on Windows with Python 3.12.7.
- **Hugging Face was blocked** by this environment's egress policy (HTTP 403 on CONNECT). The repository's `tools/fetch_tokenizers.py` therefore could not fetch the pinned Qwen tokenizer files, and the review did not route around the block.
- **Tokenizer assets** came from official vendor distributions on reachable channels, each pinned by SHA-256 in `fetch_assets.py`:
  - Alibaba's DashScope wheel (`qwen.tiktoken`, 151,643 BPE ranks)
  - Meta's `llama-models` wheel (Llama 3 and Llama 4 `tokenizer.model`)
  - Mistral's `mistral-common` wheel (Tekken v3)
  - OpenAI `o200k_base` and `cl100k_base`, bundled in the `litellm` wheel. Their SHA-256 values equal the `expected_hash` values hard-coded in OpenAI's `tiktoken`.
  - Google's public `gemma-data` bucket (Gemma 3 SentencePiece, 262,144 pieces)
- **Qwen3 was rebuilt, not downloaded.** The rebuild uses the official ranks plus the pre-tokenizer regex and NFC normalizer recorded in the repository's committed results. **[M] It reproduces all 330 committed Qwen3 counts (payload and full prompt, 33 fixtures × 5 formats) and all 330 text-fidelity flags exactly.** It is therefore a faithful substitute for counting.
- **Qwen3.5 could not be re-measured** for new variants. Its 248k vocabulary is only on Hugging Face. Its committed numbers are cited as reported.
- **Not measured:** Claude's tokenizer is not public. Anthropic provides a free `count_tokens` endpoint, but no API key was available here and none was requested. DeepSeek, Kimi and GLM tokenizers were not measured.
- **Alternative representations** were built for comparison only, in `benchmarks/claude-review-2026-10-04/variants.py`.
  - Each has a strict decoder.
  - **[M] All eight decoded back to the identical tree on 2,249 trees** (the 33 fixtures, all 216 enum combinations, and 2,000 seeded adversarial random trees with quotes, controls, combining marks, CJK, Arabic, emoji and grammar delimiters in literals). They are reversible on the same domain as M, not lossy shortcuts.
  - One reference row, `english_natural*` (unquoted prose), is *not* reversible and is marked with an asterisk wherever it appears.

---

## 3. Reproduction of the reported results

| Reported claim | Status | Notes |
|---|---|---|
| 44 tests pass | **[R] Yes** | 44/44 on Python 3.12.3 with the optional tokenizer test enabled. 43 passed and 1 skipped on 3.11.15 without `tokenizers`. |
| 165/165 semantic round trips (33 fixtures × 5 formats) | **[R] Yes** | `python -m evaluation run`. 165 comprehension prompts exported, with no answer keys. |
| M saves 11.66% (Qwen3) / 11.62% (Qwen3.5) payload tokens vs controlled English | **[R] Yes for Qwen3** (micro ratio 0.8834); **Qwen3.5 as committed** (0.8838) | Macro ratio is 0.850; worst fixture 0.990. **[M]** On other tokenizers the saving ranges from 1.4% (Gemma 3) to 11.8% (Llama 4). |
| JSON uses fewer complete-prompt tokens than M | **[R] Yes, at one record per prompt** (M/JSON = 1.278) | **[M]** This is an N=1 artifact; see §6.2. |
| Unicode normalization and reserved-token literals unresolved | **[R] Yes** | **[M]** Both are tokenizer- and transport-level, not format-level. Concrete resolutions in §5.4–5.5. |
| No comprehension, latency or monetary result | **[R] Correct** | Still true after this review. |

---

## 4. What the implementation is, and what it is not

### 4.1 What exists

- `sylang_core` is a strict, dependency-free **serializer for one small closed schema**.
- The schema covers:
  - three binary relations (`see`, `help`, `contain`)
  - four closed features: polarity (2) × tense (3) × aspect (3) × evidence (4), so 72 combinations per relation and 216 per predicate shape
  - opaque string labels
  - a binary `if` node, with depth ≤ 8 and at most 127 nodes
- Five renderers (controlled English, JSON, a named-field DSL, Prime, M) are **isomorphic encodings** of that tree. Parsing is fail-closed.
- The evaluation scaffold has integrity manifests, same-tokenizer ratios, and separation of codec, token and comprehension outcomes. It is well engineered and better than most prototypes.

### 4.2 What it is not

| Claimed or implied capability | Reality at this commit |
|---|---|
| **A language** | No open lexicon, no productive morphology, and no way to express anything outside 3 relations × 72 feature combinations. Labels are opaque strings, so all real content lives in English/Unicode literals that M passes through unchanged. |
| **A natural-language translator** | Controlled English is template output with an exact-match parser. Producing a tree from real prose needs an NLU step, in practice an LLM call that outputs JSON or M. That step is where cost and errors would live, and it is not modelled. |
| **A reasoning system** | Conditionals are preserved, not evaluated. There is no entailment, consistency checking, time anchoring or evidential semantics (the docs say so explicitly). |
| **Monosemantic / one-token-one-meaning** | **[M]** False at the token level. M opcodes are not separate tokens; e.g. Qwen3 splits `M0.1:p("Bo",h,"Cy",+,p,g,u)` into `M · 0 · . · 1 · :p · (" · Bo · ", · h · ," · Cy · ", · +, · p · ,g · ,u · )`. The model sees `,g` and `+,`, code-punctuation pieces whose pretrained associations have nothing to do with "progressive" or "positive". |

### 4.3 Growth path

To grow beyond 3 relations, the relation slot must become open: either a growing closed codebook or an arbitrary predicate string.

- **Open string:** M degenerates into a generic positional S-expression with JSON strings. That is functionally the compact DSL baseline below.
- **Closed codebook:** every new code must be taught, which grows the prompt legend linearly with vocabulary.

The historical proposal's "semantic density" (one word ≈ a phrase) collides with this: a dense lexicon is exactly what an existing model does not know. **[M]** The historical README's showcase sentence `Magbrunkan rapfensmaleg retgadyesaf` (claimed "~5–7 tokens") measures **12–14 tokens** on all seven tokenizers, against **15** for the English sentence it encodes. It splits into meaningless fragments (`Mag · br · unk · an …`).

### 4.4 Worthwhile use cases (realistic)

1. **Validated fact interchange between programs and LLMs.** Value comes from *fail-closed parsing and exact round trips* (error detection, schema enforcement), not from compression. JSON Schema plus constrained decoding already provides most of this. A compact readable DSL with a strict parser is a reasonable alternative where JSON's verbosity matters.
2. **A format-effects benchmark.** With stronger baselines, manifests and a bias-free scorer (§4.5), the harness could become a small, reusable benchmark for "does the serialization change accuracy/cost?" Recent independent work suggests this is an open, active question (§7).
3. **A semantic minimal-pair probe set.** Polarity scope, conditional scope, tense/aspect and evidentiality minimal pairs are useful diagnostics *independent of format*.
4. **Context packing** of large machine-generated fact sets. This is a capacity rather than cost argument, and a compact readable DSL does it at least as well as M (§6).

### 4.5 Defects and unsupported assumptions found

Severity: **H** invalidates a reported conclusion; **M** biases or breaks an evaluation; **L** minor.

| # | Sev. | Finding | Evidence | Suggested fix |
|---|---|---|---|---|
| D1 | **H** | **Weak English baseline inflates M's saving.** Every English record repeats `Sylang core-v0.1:` and spells the default evidence as "Without specified evidence,". | **[M]** The header is 297 of 1,166 Qwen3 English tokens (25.5%); M's header is 132 of 1,030 (12.8%). Concise English with neither is 730 tokens; M is 1,030. | Compare against concise English, a compact word DSL and minified/positional JSON (§6.1). Carry version and defaults once per batch, not per record. |
| D2 | **H** | **The complete-prompt comparison uses one record per prompt.** That maximizes legend overhead and makes "JSON wins" an artifact of N=1. | **[M]** M overtakes the repository JSON at N ≈ 2.8–4.0 on all six chat-templated tokenizers (§6.2). | Report cost curves against N, plus cached/uncached legend accounting. |
| D3 | **M** | **The comprehension task is biased toward JSON.** Every task asks for the JSON tree, so the JSON task is a pure copy, and the shared prompt already teaches the JSON output schema to all formats. | **[M]** 33 of 33 JSON payloads equal the expected answer exactly. The shared prompt is 918 characters before the legend. The English legend is 1,063 characters; JSON's is 120. | Separate reading (answer questions in natural language, or extract fields) from generation (emit format X). Never use the input format as the output format in reading tasks. |
| D4 | **M** | **One malformed model answer aborts scoring of the whole file.** `read_json` raises on any duplicate key or NaN *inside* an `answer`. | **[M]** A file with one good answer and one answer containing a duplicated `version` key exits with `Duplicate JSON key: version`. A `NaN` answer behaves the same. | Parse the envelope strictly, but score a malformed `answer` as schema-invalid for that task only, recording the raw text. |
| D5 | **M** | **A deeply nested answer crashes the scorer with a traceback.** The `RecursionError` from `json.loads` is not caught. | **[M]** A 100,000-deep array answer makes `score-comprehension` exit with status 1 and a traceback. | Catch `RecursionError` and bound the answer length before parsing. |
| D6 | **M** | **M opcodes are overloaded by position.** `s` means see and simple; `c` means contain and completed; `p` means predicate and past; `i` means if and inferred. | **[R]** Codec tables. The parser handles this correctly; models may not. | If M survives, use field-unique codes, or words. |
| D7 | **M** | **Controlled-English templates inherit natural-language ambiguity.** | `see` + progressive renders "is seeing", which commonly means *dating*. Stative verbs in progressive ("is containing") are marked or odd. Perfect is not completion (already documented). | Choose relations and templates by pilot comprehension, not by morphology. Include idiom traps in the probe set. |
| D8 | **M** | **The fixture suite cannot support aggregate efficiency claims.** | 13 of 33 fixtures are literal stress cases. 9 are `time-*` variants of `"Bo" helps "Cy"`. Labels are 1–3 characters, so structure dominates. The two "holdout" items are public (already documented). | Keep it as a codec smoke suite. Use separately generated, realistic and frozen corpora for any token or cost claim. |
| D9 | **L** | **The CLI rejects a UTF-8 BOM** ("unexpected character at 0"). | **[M]** | Strip one leading U+FEFF in the CLI only, or document the behaviour. |
| D10 | **L** | **Asset pinning is uneven.** The Qwen3.5 asset has no Git blob check. Licence notes omit the gpt-oss usage policy and the Apertus acceptable-use policy for the optional candidates. | **[R]** `tools/fetch_tokenizers.py`; **[L-i]** model cards. | Add the blob check and record licence add-ons. |
| D11 | **L** | **The tokenizer harness reproduces an unsafe serving path.** `encode_special_tokens=False` parses reserved strings in data as control tokens. | **[R]** | Keep it as a diagnostic. Measure the *safe* path too (content encoded with special-token parsing disabled), because that is what a correct server does. |

The codec itself had no correctness bug under this review's probing:
- All 54 English verb phrases are unique.
- Quoted keywords cannot impersonate structure.
- Depth and node bounds hold.
- Escaped surrogates are rejected.
- All eight new controls round-trip through the same literal tokenizer.

The defects are in the evaluation design and the claims, not in serialization.

---

## 5. Tokenizer behaviour (actual tokens, not characters)

### 5.1 Bytes are a poor proxy for tokens **[M]**

- M is 37% fewer UTF-8 bytes than controlled English (2,408 vs 3,801), but only 11.7% fewer Qwen3 tokens.
- Optimized M is 15% fewer bytes than the readable DSL (1,893 vs 2,231), yet 0.9% *more* tokens.
- Byte-level BPE vocabularies already contain frequent English words, punctuation clusters and JSON idioms (`","`, `")`, `":"`) as single tokens. Shortening a word to a letter saves bytes, not tokens.

### 5.2 What the tokens look like **[M]**

The same fact (`Bo` was helping `Cy`, unspecified evidence), counted with Qwen3 (o200k_base is identical here):

| Rendering | Tokens | Pieces |
|---|---:|---|
| Controlled English (repo) | 21 | `S·yl·ang· core·-v·0·.·1·:· Without· specified· evidence·,· "·Bo·"· was· helping· "·Cy·".` |
| M (repo) | 17 | `M·0·.·1·:p·("·Bo·",·h·,"·Cy·",·+,·p·,g·,u·)` |
| M with full words | 17 | `p·("·Bo·",·help·,"·Cy·",·+,·past·,·progress·ive·,·un·specified·)` |
| Concise English | 8 | `"·Bo·"· was· helping· "·Cy·".` |
| Compact word DSL | 8 | `help·("·Bo·","·Cy·")· past· progressive` |
| Optimized M | 8 | `h·("·Bo·","·Cy·")·/·pg` |

### 5.3 Exotic glyphs do not help **[M]**

Extra tokens per occurrence when an ASCII letter opcode is replaced by a symbol, inside `p("A",X,"B")`:

| Symbol | Qwen3 | o200k | cl100k | Llama 3 | Llama 4 | Tekken | Gemma 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `see` (English word) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `σ`, `→`, `见` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `⊕` | 0 | +1 | +2 | +1 | +1 | +2 | 0 |
| `⟦` | 0 | +2 | +1 | +1 | +1 | +2 | 0 |
| `👁` | 0 | +1 | +2 | +2 | +1 | +3 | 0 |
| U+E000 (private use) | +2 | +1 | +2 | +2 | +1 | +2 | 0 |

No symbol is cheaper than an ordinary word on any tokenizer, and several are more expensive. A symbol that happens to be one token is often rare in training, which is the under-trained-token risk documented by Land & Bartolo (EMNLP 2024, [arXiv 2405.05417](https://arxiv.org/abs/2405.05417)) **[L-i]**.

### 5.4 Unicode normalization **[M]**

- **Only the Qwen tokenizers normalize.** Their NFC normalizer (Qwen3 measured; Qwen3.5's NFC recorded in the committed metadata) changes:
  - decomposed `e+U+0301`
  - singleton canonical equivalents: Angstrom sign U+212B, Ohm sign U+2126, Kelvin sign U+212A, CJK compatibility U+F900
  - conjoining Hangul jamo
- o200k, cl100k, Llama 3, Llama 4, Tekken and Gemma 3 (byte fallback) round-trip all probes exactly. NFKC-only cases (U+FB01, fullwidth A) survive everywhere.
- **The hazard is format-independent.** It affects the literal inside English, JSON and M equally.
- **Resolution that keeps the semantic layer code-point-exact:** a *model-facing escaping policy*. Emit `\uXXXX` escapes for any literal segment where `NFC(s) ≠ s`, and for format/control characters. All formats already use JSON string escaping, so no grammar change is needed. An escaped `"café"` costs 9 Qwen3 tokens against 4 raw, and survives every tokenizer tested.
- **The alternative** is to declare NFC canonical in a new schema version. That is simpler but lossy for code and identifiers. The choice is a policy decision; the mechanism is not a research problem.

### 5.5 Reserved strings **[M]**

Which strings are control tokens depends on the tokenizer:

| Reserved string | Control token in |
|---|---|
| `<\|im_start\|>` | Qwen |
| `<\|endoftext\|>` | Qwen and OpenAI encodings |
| `<\|eot_id\|>` | Llama 3 |
| `<start_of_turn>` | Gemma 3 |
| `[INST]` | Tekken |
| `<\|start\|>` | o200k harmony |

Everywhere else these strings are ordinary 3–7-token text.

The hazard exists in every format. The standard fix sits in the serving layer: tokenize untrusted content with special-token parsing disabled (`split_special_tokens`/`encode_special_tokens=True` in Hugging Face `tokenizers`, `encode_ordinary` in tiktoken) and insert control IDs only from the template. Hosted chat APIs generally treat message text as plain text. This is not a reason to design a new notation.

### 5.6 Numbers **[M]**

Digit-splitting rules differ by tokenizer, and no serialization choice changes them:

| Number | Qwen3, Tekken, Gemma 3 (one token per digit) | o200k, Llama 3/4 (groups of up to 3 digits) |
|---|---:|---:|
| `1,250,000` | 9 tokens | 5 tokens |
| 20-digit integer | 20 tokens | 7 tokens |

Any future quantity or unit support will cost the same in every format. Re-encoding numbers (e.g. base-N) would save tokens only by damaging the model's numeracy.

### 5.7 Vocabulary and embedding compatibility **[L-i]/[A]**

**A new token has no meaning to an existing model.**
- The standard initialization (mean of existing embeddings; Hewitt 2021) is deliberately neutral.
- Making it meaningful requires gradient updates:
  - TokAlign (ACL 2025, [arXiv 2506.03523](https://arxiv.org/abs/2506.03523)) recovers performance after about 5k steps.
  - Zero-shot tokenizer transfer (NeurIPS 2024, [arXiv 2405.07883](https://arxiv.org/abs/2405.07883)) needs a per-model hypernetwork plus continued training.
  - zip2zip ([arXiv 2506.01084](https://arxiv.org/abs/2506.01084)) needs a modified model and serving stack.

**Each of these:**
- changes token IDs, which invalidates existing prefix caches
- is impossible on closed hosted APIs
- must be redone for every base-model upgrade

Shipped vocabularies also change underneath users:
- Qwen3 → Qwen3.5 went from about 151k to 248k entries (repository metadata).
- Anthropic states that **"Claude 4.7 and later models … use a newer tokenizer … approximately 30% more tokens for the same text"** **[L-d]** ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)).

A notation tuned to one tokenizer's merges is therefore a moving target.

---

## 6. End-to-end savings for identical semantic content

### 6.1 Payload decomposition: where M's saving actually comes from **[M]**

Total payload tokens over all 33 fixtures:

| Representation (all reversible unless *) | UTF-8 bytes | Qwen3 | o200k | cl100k | Llama 3 | Llama 4 | Tekken | Gemma 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Controlled English (repo) | 3,801 | 1,166 | 1,140 | 1,181 | 1,178 | 1,155 | 1,300 | 1,176 |
| JSON (repo) | 8,611 | 2,245 | 2,260 | 2,262 | 2,260 | 2,155 | 2,440 | 2,171 |
| Named-field DSL (repo) | 6,559 | 1,912 | 1,959 | 1,927 | 1,924 | 1,901 | 2,192 | 1,961 |
| Prime (repo) | 6,262 | 1,872 | 1,880 | 1,887 | 1,884 | 1,821 | 2,105 | 1,797 |
| **M (repo)** | 2,408 | **1,030** | 1,037 | 1,045 | 1,042 | 1,019 | 1,164 | 1,159 |
| M without header | 2,243 | 898 | 905 | 913 | 910 | 887 | 1,032 | 994 |
| M with full words (no header) | 3,204 | 1,014 | 1,021 | 1,029 | 1,026 | 1,003 | 1,164 | 1,028 |
| Positional JSON array | 3,872 | 1,030 | 1,045 | 1,047 | 1,045 | 1,011 | 1,189 | 1,042 |
| JSON, short keys, defaults omitted | 3,293 | 1,076 | 1,091 | 1,093 | 1,091 | 1,047 | 1,240 | 1,065 |
| Compact word DSL, all features | 3,411 | 798 | 806 | 814 | 811 | 783 | 945 | 809 |
| **Concise controlled English** | 2,311 | **730** | 737 | 745 | 742 | 721 | 865 | 743 |
| **Optimized M** (best effort) | 1,893 | **670** | 678 | 686 | 683 | 658 | 806 | 666 |
| **Compact word DSL, defaults omitted** | 2,231 | **664** | 672 | 680 | 677 | 649 | 810 | 675 |
| Natural English* (not reversible) | 2,081 | 590 | 595 | 608 | 604 | 576 | 716 | 593 |

Readings (Qwen3 unless stated):
- **The per-record header and the default-evidence prefix explain most of English's deficit.** Removing them gives English → concise English −37%. Abbreviating M's codes from full words saves 11% in the positional form, and less (3%) on Gemma 3.
- **Omitting defaults is worth 17%** (compact DSL with all features vs with defaults omitted). M as designed forbids defaults.
- **Optimized M ≈ compact word DSL on every tokenizer** (ratio 0.987–1.014). Once redundancy is gone, the remaining difference between letters and words is noise.
- Excluding the 13 literal-stress fixtures (20 plain fixtures), M is **1.98×** the compact DSL (458 vs 231) and **1.57×** concise English (458 vs 291). Short-label workloads exaggerate structure and make M look *worse*, not better.

### 6.2 Complete chat prompts, batches and legend amortization **[M]**

**Setup.**
- The system message holds a fixed task line plus a format legend. The repository's own legends are used for English, JSON and M; legends of similar specificity were written for the new controls.
- The user message holds N records and one fixed question. Each family's published generation-prompt template is applied (ChatML for Qwen, Llama 3 headers, Gemma turns, the harmony developer/user form for o200k, Tekken instruction tags, Llama 4 headers).
- Every count is a single encode of the whole templated string.
- The corpus is 200 seeded synthetic records with realistic labels such as `"Dr. Priya Raman"` and `"the quarterly report"`: 20% conditionals and depth ≤ 3. Features are either default-skewed or uniform. A second corpus uses multilingual labels (zh, ar, hi, ru, ja, es, yo, el).

Complete prompt tokens at N=100, default-skewed English corpus (tokens per record at N=200 in parentheses):

| Format | Qwen3 | o200k | Llama 3 | Llama 4 | Tekken | Gemma 3 |
|---|---:|---:|---:|---:|---:|---:|
| Natural English* | 1,525 (13.4) | 1,495 (13.1) | 1,527 (13.4) | 1,443 (12.6) | 1,516 (13.4) | 1,494 (13.2) |
| **Compact word DSL** | **1,866 (16.0)** | 1,843 (15.8) | 1,868 (16.0) | 1,829 (15.7) | 1,917 (16.5) | 1,842 (15.8) |
| Optimized M | 1,921 (16.2) | 1,898 (16.0) | 1,923 (16.2) | 1,880 (15.9) | 1,928 (16.3) | 1,801 (15.2) |
| Concise English | 2,152 (18.4) | 2,127 (18.1) | 2,154 (18.4) | 2,110 (18.0) | 2,155 (18.4) | 2,173 (18.7) |
| **M (repo legend)** | **2,541 (21.9)** | 2,517 (21.7) | 2,543 (21.9) | 2,501 (21.5) | 2,547 (21.9) | 2,873 (25.0) |
| Controlled English (repo legend) | 2,635 (21.3) | 2,612 (21.1) | 2,637 (21.3) | 2,593 (20.9) | 2,638 (21.3) | 2,648 (21.6) |
| Positional JSON array | 2,881 (25.8) | 2,905 (26.0) | 2,883 (25.8) | 2,797 (25.1) | 3,001 (26.9) | 2,910 (26.2) |
| JSON (repo legend) | 5,973 (54.2) | 5,998 (54.4) | 5,975 (54.2) | 5,675 (51.4) | 6,229 (56.5) | 5,598 (50.8) |

**When M is cheaper** (fixed overhead + N × per-record cost):

| Against | M is cheaper when |
|---|---|
| Repository JSON | N > 2.8–4.0 on every tokenizer |
| Positional JSON array | N > 19–27 (Gemma 3: N > 85–101) |
| Repository controlled English | Skewed corpus: N < ~190 (Gemma 3: N < ~32); M's legend is shorter, but it costs more per record. Uniform corpus: always (Gemma 3: N < ~43) |
| Concise English, compact DSL, optimized M | Never |

The uniform-feature and multilingual corpora give the same ordering. In the multilingual corpus, script and tokenizer dominate absolute cost; Qwen3 spends about 20–30% more per record than o200k on the same content.

The flat predicate-only batch (200 records, no conditionals) gives the same picture:
- A TOON/CSV-style table with defaults omitted (2,974 Qwen3 tokens) loses to the compact DSL (2,379).
- M (3,346) loses to both.

### 6.3 Output side, caching and quality-adjusted cost **[M]+[L-d]+[A]**

**Verified pricing structure** (Anthropic, read 2026-10-04) **[L-d]**:
- Output costs 5× input on every listed model.
- Cache reads cost 0.1× input (0.05× on Opus 5.5, 0.025× on Fable 5.1). 5-minute cache writes cost 1.25× input.
- The minimum cacheable prefix is 512–4,096 tokens depending on the model ([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)).
- OpenAI and Google pricing was confirmed only through search indexes **[L-i]**. It shows similar output/input ratios (about 5–8×) and similar cached-read discounts.

Cost per call at N=100 in *input-token equivalents*, using Qwen3 token counts and these multipliers. This is an illustration, not a bill: Claude's own tokenizer differs.

| Format | Read, uncached | Read, legend cached | Generate 100 records |
|---|---:|---:|---:|
| Natural English* | 1,540 | 1,482 | 7,365 |
| **Compact word DSL** | **1,881** | **1,773** | **8,845** |
| Optimized M | 1,936 | 1,800 | 8,996 |
| Concise English | 2,167 | 2,045 | 10,216 |
| **M (repo)** | **2,556** | **2,391** | **11,973** |
| Controlled English (repo) | 2,650 | 2,379 | 11,971 |
| Positional JSON array | 2,896 | 2,815 | 14,045 |
| JSON (repo) | 5,988 | 5,908 | 29,509 |

Consequences:
1. **Output is where tokens cost money and time.** Prefill is parallel; decoding is sequential (Splitwise, ISCA 2024; DistServe, OSDI 2024 **[L-i]**). vLLM's own documentation says prefix caching "only reduces the time of processing the queries (the prefilling phase)" **[L-i via repository docs]**. Here M emits **35% more output** than the compact DSL for the same 100 facts.
2. **Caching the legend does not rescue M.**
   - M's legend plus task prompt (about 180 tokens) is below every Anthropic minimum cacheable length. It is only cached when bundled into a larger stable prefix.
   - Even with free legends, M is more expensive per record than three readable alternatives.
3. **Retries and validation multiply cost.** Expected cost per *correct* result is roughly (tokens per attempt) ÷ P(success per attempt) plus validation, which is negligible. Codec CPU time is microseconds. **[A]** M starts 35% behind the compact DSL on generation, so it would need a *higher* success rate than a readable format just to break even. The literature suggests the opposite (§7).
4. **The NL → tree step is unaccounted for.** **[A]** Real inputs are prose. Some model must produce the tree, and that is a generation task priced at output rates. Savings from a compact notation exist only where structured facts already exist. There the question reduces to "which serialization should the model read or write", a well-studied engineering choice.
5. **Latency was not measured.** **[A]** For a fixed model, decode latency scales roughly linearly with output tokens. Input differences of a few hundred tokens change time-to-first-token by milliseconds to tens of milliseconds on modern GPUs; that is an order-of-magnitude estimate, not a measurement. No speed claim is justified without a serving experiment (plan Phase 3).

---

## 7. Modern approaches compared

| Approach | Needs | Compatibility | Evidence (source label) | Relevance to Sylang |
|---|---|---|---|---|
| **Unchanged model + compact readable format** (minified JSON, positional arrays, CSV/TOON, word DSL) | Nothing | Full; works on hosted APIs | **[M]** this review; TOON benchmarks are author-run **[L-i]**; independent agentic study finds TOON up to 18% fewer tokens at about 9 points accuracy cost ([Notation Matters, arXiv 2605.29676](https://arxiv.org/abs/2605.29676)) **[L-i]**; TOON vs JSON generation benchmark ([arXiv 2603.03306](https://arxiv.org/abs/2603.03306)) **[L-i]** | **Strongest simple baseline.** Already delivers what M was meant to, with readable words. Accuracy effects need testing per model. |
| **Prompt caching / batching** | Provider feature | Full | **[L-d]** Anthropic: 0.1× cache reads, 50% batch discount; OpenAI/Google similar **[L-i]** | Removes most repeated-legend cost without any format change. A billed coding-agent study found cutting tool-output tokens by 38.4% *raised* the bill by 6.8% through cache effects ([arXiv 2607.12161](https://arxiv.org/abs/2607.12161)) **[L-i]**. |
| **Constrained decoding** (XGrammar-2, DOMINO, JSON Schema) | Grammar or schema; open serving stack or provider structured outputs | High for JSON; custom grammars need self-hosting | XGrammar-2 v4 ([arXiv 2601.04426](https://arxiv.org/abs/2601.04426)) **[L-i]**; JSONSchemaBench ([arXiv 2501.10868](https://arxiv.org/abs/2501.10868)) **[L-i]**; "The Format Tax": the accuracy loss comes mostly from format instructions, and recent closed models show little of it ([arXiv 2604.03616](https://arxiv.org/abs/2604.03616)) **[L-i]** | Guarantees syntax, not meaning. Hosted APIs support JSON Schema natively, not M; a custom grammar is a deployment tax on M. |
| **Prompt/context compression** (LLMLingua-2, gist tokens, ICAE) | Learned compressor; often white-box | Lossy | LLMLingua-2, Findings ACL 2024 **[L-i]**; an independent ECIR 2026 study reports at most 18% end-to-end speed-up ([arXiv 2604.02985](https://arxiv.org/abs/2604.02985)) **[L-i]**; CAVEWOMAN reports input compression *raised* net cost about 1.15× while output compression cut it 1.4–2.4× ([arXiv 2606.24083](https://arxiv.org/abs/2606.24083)) **[L-i]** | Different goal (lossy). The evidence that output-side savings dominate agrees with §6.3. |
| **Concise reasoning** (Chain of Draft, Sketch-of-Thought) | Prompting | Full | Chain of Draft ([arXiv 2502.18600](https://arxiv.org/abs/2502.18600)) **[L-i]** | Shortens outputs in ordinary language with no new notation. Directly competes with any "Sylang for outputs" idea. |
| **Fine-tuning on an unchanged tokenizer** (LoRA/QLoRA) | Training data, GPUs, evaluation | Open weights only; per-model | QLoRA ([arXiv 2305.14314](https://arxiv.org/abs/2305.14314)) | Could teach M, but could equally teach the compact DSL, which needs no teaching. There is no rationale to fine-tune toward a notation that is more expensive before training. |
| **Vocabulary adaptation** (TokAlign, ZeTT, ALM, SuperBPE, zip2zip) | Training (thousands of steps to full pretraining) | Changes IDs and caches; impossible on closed APIs | **[L-i]** see §5.7; a decoder retrofit reported only up to 6% shorter sequences ([arXiv 2411.18553](https://arxiv.org/abs/2411.18553)) | Highest cost and lowest compatibility. Not justified by any measured gap. |
| **Byte-level / tokenizer-free models** (ByT5, BLT, H-Net, Bolmo) | New or converted models | New model family | BLT (ACL 2025) and H-Net report FLOP-matched parity; BLT cautions that wall-clock time may lag; ByT5 is 1.5–9.5× slower at inference than mT5 **[L-i]** | Orthogonal. They remove the tokenizer M was designed around, and none is served by the major APIs. |
| **Latent / KV-cache agent communication** (C2C, KVComm, LatentMAS) | White-box, often a trained projector per model pair | Same-family open models | **[L-i]**; independent critiques show gains are often not carried by message content | For agent-to-agent bandwidth, these dominate any text codec where they apply. Irrelevant for hosted black-box models. |

**Learning a new notation in context is expensive and brittle.**
- Grammar-book translation gains come mostly from the *parallel examples*, not the grammar descriptions (Aycock et al., ICLR 2025, [arXiv 2409.19151](https://arxiv.org/abs/2409.19151)) **[L-i]**.
- Deterministic decoding tasks degrade sharply when the target output is low-probability ("Embers of Autoregression", PNAS 2024) **[L-i]**.

Both point the same way: an existing model will read and write words it already knows more reliably than invented codes. That is a hypothesis for the plan to test, not a conclusion.

---

## 8. Correctness and generalization: what round trips do and do not prove

| Risk | Status in the codec | What a model could still get wrong | Test (plan) |
|---|---|---|---|
| **Ambiguity** | None at parse time; closed enums | Idioms ("is seeing"), positional code overloading (D6), polarity attached to the wrong clause | Minimal pairs with idiom traps; per-field error taxonomy |
| **Scope** | Explicit tree; brackets in English | Negation scope inside nested conditionals; condition/consequence swaps at depth ≥ 3 | Depth-stratified items; swap-detection probes |
| **Literals** | Code-point exact; JSON escaping | Copying long or escaped literals; quoted text acting as instructions (prompt injection) | Copy-fidelity items; injection literals; safe-tokenization serving path |
| **Unicode** | Exact in the codec | NFC tokenizers alter text; models may "fix" diacritics or confusables | Escaping policy A/B (§5.4); confusable and bidi probes |
| **Numbers and units** | Not supported (rejected) | Digit-split differences (§5.6); unit conversion; arithmetic | Out of scope until a quantity schema exists; then compare formats with numbers as literals |
| **Numerical reasoning** | None | Format-induced degradation of reasoning ("format tax") | Reason in free text, then serialize, against direct serialization |
| **Multilingual** | Literals only, inside an English-based grammar | Labels in other scripts cost more on some tokenizers; no multilingual *content* is represented | Per-script results, worst-group reporting, native-speaker review |
| **Long context** | Size bounds only | Retrieval and aggregation over thousands of records; lost-in-the-middle; legend far from payload | Needle and aggregation tasks at 8k–128k tokens per format |

**What round trips prove.** A round trip proves the *serializer* is injective and its parser is its inverse on the valid domain. It says nothing about whether a model maps the string to the same meaning. The parser and renderer were written together, so they could share a misunderstanding (e.g. "completed" = perfect) and still round-trip perfectly. Only independently authored expectations and model-based tests can catch that.

---

## 9. Recheck of the 2026-10-02 review

### 9.1 Sources, dates and claims

Primary sites (arXiv, ACL Anthology, PMLR, Hugging Face) could not be fetched from this environment. Most checks are therefore **[L-i]**, through search indexes of those primary pages. Anthropic documentation was read directly **[L-d]**.

| Item | Finding |
|---|---|
| All 17 cited works | **Exist.** Every cited version date in the earlier review's table (XGrammar-2 v4 2026-08-05; TokAlign v1 2025-06-04; Teaching Old Tokenizers v2 2026-03-23; Writing-System v1 2026-08-01; Parity-aware BPE v3 2026-07-02; BLT v1; H-Net v2; Coconut v4 2026-08-23; WeDLM v1 2025-12-28) matches the indexed submission histories. |
| PMLR `park25l`, cited as "grammar/tokenizer alignment" | **Mischaracterized.** It is "Flexible and Efficient Grammar-Constrained Decoding" (Park, Zhou, D'Antoni, ICML 2025). Aligning subword tokens with grammar terminals is a subproblem of the algorithm, not the paper's subject. |
| XGrammar-2 title | Changed in later versions. v4 is "XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs". |
| BLT, H-Net, ByT5 venues | Cited as arXiv only. They appeared at ACL 2025, ICLR 2026 (as indexed) and TACL 2022 respectively. |
| Lotz et al. (ACL 2025) | "Across model scales" means two scales (350M, 2.7B). On English, tokenizer effects were small; the differences were mostly multilingual and machine translation. |
| Grammar-constrained logical parsing (ACL 2025 Industry) | Gains concentrate in small models; larger models were sometimes better unconstrained. |
| Coconut | Underperforms chain of thought on GSM8K at GPT-2 scale (34.1 vs 42.9). |
| ByT5 | Inference 1.5–9.5× slower than mT5. |
| vLLM prefix caching | Prefill only. The earlier review states this correctly. |
| Licences | gpt-oss and Apertus carry usage/acceptable-use policies on top of Apache-2.0. |
| Qwen3-0.6B-Base `tokenizer.json` size | 7,031,645 bytes. It cannot be re-verified here because Hugging Face is blocked. **[M]** Count equivalence of the rebuilt tokenizer is strong indirect evidence that the pinned vocabulary, regex and normalizer are as recorded. |

### 9.2 Where I agree with the earlier review

- Separate codec fidelity, token counts, comprehension and economics.
- Use same-tokenizer ratios only.
- Tokenize whole prompts in one call.
- Never infer meaning from a new token.
- No training before measured need.
- Pin assets.
- Treat the public holdout as non-secret.

### 9.3 Where I disagree

1. **Baseline choice.** It compared M against controlled English and verbose named-field controls. It treated "compare M against the strongest compact baseline" as a later gate (decisions.md gate 2), but that comparison is decisive and cheap. Run now (§6), it reverses the headline: against the strongest compact baselines, M uses more tokens, not fewer.
2. **The roadmap's direction.** The six-step priority list ends in "constrained generation, then adaptation only if needed". After §6, there is no measured gap in M's favour for adaptation to close. The roadmap should end at a stop/salvage decision, not at LoRA.
3. **"Keep Qwen3 as the baseline."** A Qwen tokenizer count is a proxy. Costs that matter are billed by the serving provider's tokenizer, which can change (+30% for recent Claude models), and Qwen is the only measured family that normalizes Unicode. Measure several families, and use provider `usage` fields for any money claim.
4. **The comprehension protocol.** "Reconstruct the JSON tree" conflates reading with translation into JSON, makes the JSON task a copy (D3), and never measures *generation* of M, which is where cost savings would matter. Reading and generation must be separate experiments with format-neutral answers.
5. **Unicode and reserved tokens as open research questions.** Both have standard engineering resolutions (escaping policy; safe tokenization of untrusted content). They should not block or shape the notation.
6. **Its research framing** listed techniques (BLT, H-Net, WeDLM, Coconut) without the caveats that make them irrelevant to a black-box, text-level notation. It missed the 2025–2026 evidence most directly on point:
   - independent compact-format benchmarks (TOON vs JSON, Notation Matters)
   - format-tax studies
   - the empirical finding that token reduction does not equal cost reduction

---

## 10. What a fair evaluation needs (summary)

The full protocol, with phases, budgets and gates, is in the [plan](../plans/claude-independent-next-steps-2026-10-04.md). Core requirements:

- **Independent frozen holdouts.**
  - Authored or generated by someone other than the codec author, from a different generator.
  - Sealed by hash, with the holdout stored outside the public repository until the run.
  - Must include minimal pairs, idiom traps, depth-stratified scope items, multilingual labels and long-context sets.
- **Matched prompts and budgets.**
  - The same instruction skeleton and the same number of worked examples per format.
  - Legend length either matched or varied deliberately as a factor.
  - The same decoding parameters, `max_tokens` and retry policy.
  - The strongest compact baselines are included (minified JSON, positional JSON, compact word DSL, concise English, natural English).
- **Separate reading from generation.**
  - *Reading:* format X → answer in a format-neutral way (short natural-language answer or field extraction).
  - *Generation:* natural language or JSON → format X, validated by the strict codec.
- **Quantify failures and uncertainty.**
  - Count parse failures, schema-invalid output, semantic errors by field, refusals, timeouts and retries.
  - Use paired designs with McNemar or paired bootstrap tests, and 95% intervals.
  - Report results per model and per group, with the worst group always shown.
- **Quality-adjusted cost and latency.**
  - Billed cost per correct item, from provider `usage` fields (input, cache write, cache read, output).
  - Time to first token, total latency and p50/p95, under cold and warm caches.
  - Separate uncached and cached accounting.
- **Keep the evidence tiers distinct.** Reproduced / new measurement / literature / proposed-and-unrun, as in this document.

---

## 11. Verdict, disagreements, decisive experiments, stopping criteria

### 11.1 Verdict

| Track | Verdict | Basis |
|---|---|---|
| Sylang as a token-saving language (M/Prime notation as the product) | **No-go** | **[M]** Loses to readable compact baselines in payload, complete-prompt and generation tokens on 7/7 tokenizers. Optimized cryptic codes only tie readable words. |
| Custom tokenizer, new tokens, vocabulary adaptation | **No-go** | **[M]/[L]** Glyphs give no gain; new tokens are meaningless until trained; incompatible with hosted APIs and caches; tokenizers drift. |
| Fine-tuning to teach M | **No-go** | **[A]** The format is more expensive before training; any fine-tune could target a cheaper readable format instead. |
| Falsification-and-salvage study (plan Phases 0–2) | **Conditional go** | Cheap. Settles the only remaining question (comprehension and generation accuracy of M vs readable compact formats). Salvages the codec and harness as a format-evaluation tool. Needs explicit approval for any paid inference. |

### 11.2 Decisive experiments

1. **E1 — Token falsification on billing tokenizers [P, mostly done here].** Add Claude (free `count_tokens`, needs a key and approval), Qwen3.5, DeepSeek and current OpenAI encodings to §6 on frozen realistic corpora.
   - *Stop the notation track if* M or any Sylang-specific notation is not ≥10% cheaper in complete-prompt *and* generation tokens than the best readable compact baseline on a majority of billing tokenizers.
   - On present evidence this criterion is already met for 6 of 6 chat-templated tokenizers plus cl100k.
2. **E2 — Reading accuracy [P].** Same facts in M, compact DSL, concise English and minified JSON. Format-neutral questions about polarity, scope, tense/aspect and evidence. Two or more model families, a small and a large tier.
   - *Pass for M* only if M is non-inferior to the best readable format (margin 2 percentage points) *and* cheaper per correct answer.
3. **E3 — Generation accuracy [P].** Natural language → format, codec-validated: parse-failure rate, semantic accuracy, output tokens, retries.
   - *Pass for M* only if cost per correct record is ≥10% below the compact DSL.
4. **E4 — Serving latency [P, only if E2 and E3 pass].** A local open model on vLLM: time to first token, decode time, cold/warm caches, p95.

### 11.3 Stopping criteria

- If E1 holds (expected) **and** E2/E3 show no Sylang-specific advantage, **stop the language track permanently**. Archive M/Prime as historical, keep the codec and harness, and pivot any further work to "format selection and validation tooling".
- Stop immediately if reaching non-inferiority would require fine-tuning or vocabulary changes.
- Stop if results disagree across model families in a way that cannot be mapped to a deployment decision (i.e. no single format choice is robustly better).

### 11.4 What would change my mind

- **Accuracy.** A pre-registered, independently held-out study showing that a Sylang-specific notation is *both* non-inferior in accuracy *and* at least 10% cheaper per correct result than the best readable compact format on two or more current model families. "Cheaper" means billed tokens including legend, retries and output.
- **Scale.** A realistic workload where structure (not literals) dominates and readable compact formats are impossible or ambiguous, for example very wide records with dozens of closed-enum fields, and where M-style positional codes measurably win there.
- **Deployment.** Evidence that the production target is an open model the project will fine-tune anyway, *and* that teaching M yields higher accuracy than teaching a readable compact format at equal training cost.
- **Tokenizer.** A tokenizer-level fact that defeats §5–6, e.g. a major provider whose tokenizer makes word-based structure systematically more expensive than codes. None of the seven measured behaves this way.

---

## 12. Evidence ledger

| Statement | Tier |
|---|---|
| 44 tests; 165/165 round trips; Qwen3 11.66% payload saving vs controlled English; JSON wins at N=1 | **[R]** |
| Rebuilt Qwen3 equals committed counts (330/330) | **[M]** |
| All eight reversible controls round-trip on 2,249 trees | **[M]** |
| Payload, batch, generation, glyph, Unicode, reserved-string and number tables | **[M]** (`results.json`) |
| README showcase sentence: 12–14 tokens, not ~5–7 | **[M]** |
| Harness defects D3–D5 and D9 | **[M]** |
| Anthropic prices, cache multipliers and minimums; Claude 4.7+ tokenizer about 30% more tokens | **[L-d]** |
| All other literature and model-card statements | **[L-i]**, author-reported unless stated |
| Latency, money in dollars, comprehension accuracy, generation accuracy | **Not measured.** Estimates are labelled **[A]**; experiments are **[P]** |

### Reproduce

```sh
python -m unittest discover -s tests -v
python -m evaluation run --output .cache/offline.json --repetitions 1
pip install --only-binary=:all: tiktoken==0.14.0 sentencepiece==0.2.2
python benchmarks/claude-review-2026-10-04/fetch_assets.py --download   # explicit opt-in; pinned SHA-256
python benchmarks/claude-review-2026-10-04/run.py --output benchmarks/claude-review-2026-10-04/results.json
```

### Primary sources used

- **Read directly:**
  - Anthropic [pricing](https://platform.claude.com/docs/en/about-claude/pricing) and [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), accessed 2026-10-04
  - OpenAI `tiktoken` 0.14.0 `openai_public.py` (expected hashes)
  - Meta `llama-models` 0.3.0 tokenizer sources (regex patterns)
  - Mistral `mistral-common` 1.12.0 Tekken config
- **Index-confirmed:** the works linked inline above. The agent-assisted literature scan surfaced further 2026 preprints. Those not linked here were not independently checked and are not relied on.
