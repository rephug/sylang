# Independent technical feasibility review of Sylang

**Date:** 2026-10-04 · **Reviewer:** Claude (independent review; not an author of the reviewed code)
**Reviewed commit:** `0f7c5700c9a314b9f0b635bb4773c24caf785bf2` (branch `feat/reversible-core-evaluation`, draft PR rephug/sylang#1)
**Review branch:** `claude/festive-meitner-w5wagk` (fast-forwarded to the reviewed commit; no existing file modified)
**Companion plan:** [claude-independent-next-steps-2026-10-04.md](../plans/claude-independent-next-steps-2026-10-04.md)
**Reproduction package:** [`benchmarks/claude-review-2026-10-04/`](../../benchmarks/claude-review-2026-10-04/) (`fetch_assets.py`, `run.py`, `results.json`)
**Revision:** updated the same day after an adversarial verification pass, which searched for technical, architectural and mathematical improvements (§13). Corrections to this review's own claims are applied in place and logged in §13.4.

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
| **[V]** | Measured or derived during the §13 verification pass by two independent verifiers (one re-ran the evidence, one checked logic and value); scripts not committed unless marked "run.py" |
| **[A]** | Analytical argument or estimate, not a measurement |
| **[P]** | Proposed experiment, not run |

No model inference, training, fine-tuning, serving job or paid API call was made. No model weights were downloaded.

---

## 1. Verdict in brief

**No-go** for Sylang as a token-efficiency *language* as designed. That covers:
- the repository's M and Prime notations as a cost-saving device
- a custom tokenizer or new vocabulary tokens
- fine-tuning a model to learn the notation

**Conditional go** for a bounded falsification-and-salvage track. It costs about 2–3.5 weeks of engineering (plan Phases 0–2) and at most a few hundred dollars of approved inference.

**The one question left open** is whether an *optimized* Sylang-style code can keep accuracy while saving about 10% of tokens. That code is a fused code per fact in a tab-separated layout, `c_tsv`. It is compared against the same layout written in words, `r_tsv_p`. Offline it is knife-edge; only a comprehension test can settle it.

Why:

1. **[M] The reported 11.6% saving is a property of a weak baseline, not of M.**
   - The controlled-English baseline carries a per-record `Sylang core-v0.1:` header (297 of its 1,166 Qwen3 tokens, 25%) and a mandatory "Without specified evidence," prefix.
   - Against *reversible* baselines that drop that redundancy, M uses **41% more payload tokens than concise controlled English** and **55% more than a compact word DSL** (Qwen3, all 33 fixtures). The ranges across seven tokenizers are 35–56% and 44–72%.
   - Part of that gap is M's own per-record header (`M0.1:`). Header-free M is still +23% vs concise English and +35% vs the compact DSL, with ranges of 19–34% and 27–47% across the seven tokenizers.
   - Of the 366-token Qwen3 gap to the compact DSL: header 36%, M's no-defaults rule 37%, notation 27% **[V]**.
2. **[M] Codes vs words depends on layout. The earlier "codes only tie words" result was an artifact.**
   - An optimized M that keeps a `/` separator only ties the readable DSL. That holds for fixture payloads (±1.4%) and for complete prompts at N=100 (−6% to +3%).
   - At a **matched** delimiter-light layout, a fused per-fact code such as `hpgd` (help, past, progressive, direct) is cheaper than readable words:
     - payload: 9.8–16.2% cheaper on the English-label corpora, 8–12% on multilingual labels
     - complete prompts at N=100: 8.5–12.3% cheaper on default-skewed English labels (≥10% on 5/6 tokenizers), 12.4–14.0% with uniform features, but only 6.8–10.0% with multilingual labels
   - Even so, the codes are not atomic tokens: `hpgd` becomes `\th · pg · d`. M's own codes fuse with punctuation (`:p`, `,g`, `+,`), and the legend teaches different tokens than the payload uses **[V]**.
3. **[M] Complete chat prompts and generation.** At N=100, the repository's M costs **30–66% more** than the best readable reversible format on six chat-templated tokenizers across the three corpora (41–66% on the default-skewed English corpus). That best format is the tab-separated word layout `r_tsv_p` in every case.
   - **Generation** (Qwen3 tokens, output priced 5× input): emitting 100 facts costs 11,973 input-token equivalents in M, against 7,444 for the readable tab layout and 8,845 for the compact DSL.
   - **Against verbose JSON:** M first beats it at N = 2 records per prompt (N = 3 on Gemma 3), so "JSON wins the complete prompt" is an N = 1 artifact. Any compact format beats verbose JSON there.
   - **Offline cost gate [V]:** to tie the compact DSL on cost per correct answer, M would need 1.31–1.56× its accuracy. The retry cap does not change this. This kills the repository's M offline, without any paid test.
4. **[M/L-d/V] There is no route through the tokenizer.**
   - Exotic glyphs cost the same or 1–3 extra tokens.
   - New tokens carry no meaning until trained, and training breaks compatibility with hosted APIs.
   - Hosted tokenizers change underneath you: Anthropic states that Claude 4.7 and later produce about 30% more tokens for the same text.
   - Making M's codes atomic vocabulary entries would make M *longer* (+17% idealized) **[V]**. The structure floor is the same for every quote-delimited format.
5. **[R] What works:** the reversible codec. It is a well-tested serializer for a tiny closed schema, and the evaluation scaffold is carefully built. Both are reusable; neither shows that a new language is needed. §13 lists the verified improvements to them and to this review's own method: 60 surviving findings, merged into 49 items.

Details, caveats and what would change this verdict are in §11. The improvements are in §13.

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
- **Qwen3 was rebuilt, not downloaded.** The rebuild uses the official ranks plus the pre-tokenizer regex and NFC normalizer recorded in the repository's committed results. **[M] It reproduces all 330 committed Qwen3 counts (payload and full prompt, 33 fixtures × 5 formats) and all 330 text-fidelity flags exactly.**
  - That is count equivalence, not token-ID equivalence. Count equality on this suite is a weak certificate: a converted cl100k matched 330/330 counts but only 320/330 IDs **[V]**. Plan 1.1 re-certifies on IDs.
  - Since the update the adapter applies NFC with the Hugging Face runtime's own normalizer, removing a Unicode-version skew with Python's tables. No published count changed.
- **Qwen3.5 could not be re-measured** for new variants. Its 248k vocabulary is only on Hugging Face. Its committed numbers are cited as reported.
- **Not measured:** Claude's tokenizer is not public. Anthropic provides a free `count_tokens` endpoint, but no API key was available here and none was requested. DeepSeek, Kimi and GLM tokenizers were not measured.
- **Alternative representations** were built for comparison only, in `benchmarks/claude-review-2026-10-04/variants.py`.
  - Each has a decoder that is fail-closed on structure. Most also accept non-canonical spacing, as the repository's own decoders do, so "strict" does not mean "canonical-only" **[V]**.
  - The update added a layout-matched pair: `r_tsv_p` (tab-separated, words) and `c_tsv` (tab-separated, fused codes).
  - **[M] All ten decoded back to the identical tree on 2,249 trees** (the 33 fixtures, all 216 enum combinations, and 2,000 seeded adversarial random trees with quotes, controls, combining marks, CJK, Arabic, emoji and grammar delimiters in literals). They are reversible on the same domain as M, not lossy shortcuts.
  - One reference row, `english_natural*` (unquoted prose), is *not* reversible and is marked with an asterisk wherever it appears.

---

## 3. Reproduction of the reported results

| Reported claim | Status | Notes |
|---|---|---|
| 44 tests pass | **[R] Yes** | 44/44 on Python 3.12.3 with the optional tokenizer test enabled. 43 passed and 1 skipped on 3.11.15 without `tokenizers`. |
| 165/165 semantic round trips (33 fixtures × 5 formats) | **[R] Yes** | `python -m evaluation run`. 165 comprehension prompts exported, with no answer keys. |
| M saves 11.66% (Qwen3) / 11.62% (Qwen3.5) payload tokens vs controlled English | **[R] Yes for Qwen3** (micro ratio 0.8834); **Qwen3.5 as committed** (0.8838) | Macro ratio is 0.850; worst fixture 0.990. **[M]** On other tokenizers the saving ranges from 1.4% (Gemma 3) to 11.8% (Llama 4). |
| JSON uses fewer complete-prompt tokens than M | **[R] Yes, at one record per prompt** (M/JSON = 1.278) | **[M]** This is an N=1 artifact: with complete chat templates, M is already cheaper at N = 2 (N = 3 on Gemma 3); see §6.2. |
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
| D1 | **H** | **Weak English baseline inflates M's saving.** Every English record repeats `Sylang core-v0.1:` and spells the default evidence as "Without specified evidence,". | **[M]** The header is 297 of 1,166 Qwen3 English tokens (25.5%); M's header is 132 of 1,030 (12.8%). Concise English with neither is 730 tokens; M is 1,030, and header-free M is 898. | Compare against concise English, a compact word DSL and minified/positional JSON (§6.1). Carry version and defaults once per batch, not per record. |
| D2 | **H** | **The complete-prompt comparison uses one record per prompt.** That maximizes legend overhead and makes "JSON wins" an artifact of N=1. | **[M]** On real complete prompts, M is already cheaper than the repository JSON at N = 2 on five chat-templated tokenizers and N = 3 on Gemma 3. A linear overhead + per-record model gives break-even N ≈ 2.8–4.1 (§6.2). | Report cost curves against N, plus cached/uncached legend accounting. |
| D3 | **M** | **The comprehension task is biased toward JSON.** Every task asks for the JSON tree, so the JSON task is a pure copy, and the shared prompt already teaches the JSON output schema to all formats. | **[M]** 33 of 33 JSON payloads equal the expected answer exactly (`results.json` → `defects`). **[A]** The shared prompt text, excluding legend and payload, is 918 characters for M, about 780 of them before the legend. The English legend is 1,063 characters; JSON's is 120. | Separate reading (answer questions in natural language, or extract fields) from generation (emit format X). Never use the input format as the output format in reading tasks. |
| D4 | **M** | **One malformed model answer aborts scoring of the whole file.** `read_json` raises on any duplicate key or NaN *inside* an `answer`. | **[M]** A file with one good answer and one answer containing a duplicated `version` key exits with `Duplicate JSON key: version`. A `NaN` answer behaves the same. | Parse the envelope strictly, but score a malformed `answer` as schema-invalid for that task only, recording the raw text. |
| D5 | **M** | **A deeply nested answer crashes the scorer with a traceback.** The `RecursionError` from `json.loads` is not caught. | **[M]** A 100,000-deep array answer makes `score-comprehension` exit with status 1 and a traceback. | Catch `RecursionError` and bound the answer length before parsing. |
| D6 | **M** | **M opcodes are overloaded by position.** `s` means see and simple; `c` means contain and completed; `p` means predicate and past; `i` means if and inferred. | **[R]** Codec tables. The parser handles this correctly; models may not. | If M survives, use field-unique codes, or words. |
| D7 | **M** | **Controlled-English templates inherit natural-language ambiguity.** | `see` + progressive renders "is seeing", which commonly means *dating*. Stative verbs in progressive ("is containing") are marked or odd. Perfect is not completion (already documented). | Choose relations and templates by pilot comprehension, not by morphology. Include idiom traps in the probe set. |
| D8 | **M** | **The fixture suite cannot support aggregate efficiency claims.** | 13 of 33 fixtures are literal stress cases. 9 are `time-*` variants of `"Bo" helps "Cy"`. Labels are 1–3 characters, so structure dominates. The two "holdout" items are public (already documented). | Keep it as a codec smoke suite. Use separately generated, realistic and frozen corpora for any token or cost claim. |
| D9 | **L** | **The CLI rejects a UTF-8 BOM** ("unexpected character at 0"). | **[M]** | Strip one leading U+FEFF in the CLI only, or document the behaviour. |
| D10 | **L** | **Asset pinning is uneven.** The Qwen3.5 asset has no Git blob check. Licence notes omit the gpt-oss usage policy and the Apertus acceptable-use policy for the optional candidates. | **[R]** `tools/fetch_tokenizers.py`; **[L-i]** model cards. | Add the blob check and record licence add-ons. |
| D11 | **L** | **The tokenizer harness reproduces an unsafe serving path.** `encode_special_tokens=False` parses reserved strings in data as control tokens. | **[R]** | Keep it as a diagnostic. Measure the *safe* path too (content encoded with special-token parsing disabled), because that is what a correct server does. |

Further defects and hardening items found by the verification pass are in §13: three test-suite blind spots, inconsistent error classes, JSON Schema drift, task IDs not bound to prompt content, and integrity tests that don't pin their guarantees.

The codec itself had no correctness bug under this review's probing:
- All 54 English verb phrases are unique.
- Quoted keywords cannot impersonate structure.
- Depth and node bounds hold.
- Escaped surrogates are rejected.
- All ten comparison formats round-trip through the same literal tokenizer.

The defects are in the evaluation design and the claims, not in serialization.

---

## 5. Tokenizer behaviour (actual tokens, not characters)

### 5.1 Bytes are a poor proxy for tokens **[M]**

- M is 37% fewer UTF-8 bytes than controlled English (2,408 vs 3,801), but only 11.7% fewer Qwen3 tokens.
- Optimized M is 15% fewer bytes than the readable DSL (1,893 vs 2,231), yet 0.9% *more* tokens.
- Byte-level BPE vocabularies already contain frequent English words, punctuation clusters and JSON idioms (`","`, `")`, `":"`) as single tokens. Shortening a word to a letter saves bytes, not tokens.

### 5.2 What the tokens look like **[M]**

The same fact (`Bo` was helping `Cy`, unspecified evidence), counted with Qwen3 (`results.json` → `payload.example_pieces`). o200k_base differs in three rows:
- controlled English: it merges `Syl`, giving 20 tokens
- M with full words: it splits `uns·pecified`, still 17 tokens
- tab word layout: it does not fuse the TAB with `help`, giving 7 tokens

| Rendering | Tokens | Pieces |
|---|---:|---|
| Controlled English (repo) | 21 | `S·yl·ang· core·-v·0·.·1·:· Without· specified· evidence·,· "·Bo·"· was· helping· "·Cy·".` |
| M (repo) | 17 | `M·0·.·1·:p·("·Bo·",·h·,"·Cy·",·+,·p·,g·,u·)` |
| M with full words | 17 | `p·("·Bo·",·help·,"·Cy·",·+,·past·,·progress·ive·,·un·specified·)` |
| Concise English | 8 | `"·Bo·"· was· helping· "·Cy·".` |
| Compact word DSL | 8 | `help·("·Bo·","·Cy·")· past· progressive` |
| Optimized M | 8 | `h·("·Bo·","·Cy·")·/·pg` |
| Tab layout, words (`r_tsv_p`) | 6 | `Bo·\thelp· past· progressive·\tC·y` |
| Tab layout, fused code (`c_tsv`) | 5 | `Bo·\th·pg·\tC·y` (the TAB fuses with neighbouring characters, here even splitting the label) |

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
- **Resolution that keeps the semantic layer code-point-exact:** a *model-facing escaping policy*. All formats already use JSON string escaping, so no grammar change is needed.
  - **The guarantee must be checked on the encoded text T itself:** `NFC(T) == T` and `decode(T) == tree`.
  - The rule first proposed here ("escape a literal when NFC changes it") is **insufficient [M, run.py]**. The literal `x` + LF + U+0303 is already NFC, so that rule leaves it alone. But its JSON escape `\n` ends in a letter, Qwen's NFC composes that `n` with the combining tilde, and the encoded text then fails to parse in **all five formats**. Exhaustively over C0 controls × combining marks, the naive rule leaves 98 such unparseable cases **[V]**.
  - A greedy NFC-safe escaper is provably correct and verified exhaustively, with zero overhead on NFC text (§13.1.2). An escaped `"cafe\u0301"` costs 9 Qwen3 tokens against 4 raw and survives every tokenizer tested.
  - The policy forces MAX_TEXT_BYTES up to at least 2^19, because a maximal tree of escaped astral labels needs 393,216 bytes.
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

The hazard exists in every format. The primary fix sits in the serving layer: tokenize untrusted content with special-token parsing disabled (`split_special_tokens`/`encode_special_tokens=True` in Hugging Face `tokenizers`, `encode_ordinary` in tiktoken) and insert control IDs only from the template. Hosted chat APIs generally treat message text as plain text.

That fix is not universal **[M, run.py]**:
- Through raw SentencePiece, Gemma 3's `<start_of_turn>` and `<end_of_turn>` are USER_DEFINED pieces. They are matched mid-text with no switch to disable them (`note <start_of_turn>user` → `[14210, 236743, 105, 2364]`).
- Hugging Face added tokens flagged `special=False` are not split by `encode_special_tokens=True` **[V]**.

A cheap literal-level guard adds defence in depth: write `<` and `[` inside literals as `\u003c` and `\u005b` (§13.1.4). None of this is a reason to design a new notation.

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
| Tab layout, words (`r_tsv_p`, added in the update) | 2,015 | 586 | 612 | 604 | 603 | 594 | 743 | 633 |
| Tab layout, fused code (`c_tsv`, added in the update) | 1,654 | 552 | 565 | 570 | 569 | 548 | 688 | 609 |
| Natural English* (not reversible) | 2,081 | 590 | 595 | 608 | 604 | 576 | 716 | 593 |

Readings (Qwen3 unless stated):
- **The per-record header and the default-evidence prefix explain most of English's deficit.** Removing them gives English → concise English −37%. Abbreviating M's codes from full words saves 11% in the positional form, and less (3%) on Gemma 3.
- **Omitting defaults is worth 17%** (compact DSL with all features vs with defaults omitted). M as designed forbids defaults.
- **Header-free M** (898) is still +23% vs concise English and +35% vs the compact DSL. The ranges across tokenizers are 19–34% and 27–47%.
- **Optimized M ≈ compact word DSL on fixture payloads** (ratio 0.987–1.014). This was first read as "letters and words are equivalent". **That reading was wrong:** it is an artifact of optimized M's `/` separator.
  - At a matched tab-separated layout, the fused code `c_tsv` beats the word layout `r_tsv_p` by 4–8% on these short-label fixtures, and by 9.8–16.2% on the realistic English corpora (§6.2).
- Excluding the 13 literal-stress fixtures (20 plain fixtures), M is **1.98×** the compact DSL (458 vs 231) and **1.57×** concise English (458 vs 291). Short-label workloads exaggerate structure and make M look *worse*, not better.
- These are single-document totals. They should not be read as per-record batch costs: in a batch, pre-tokenizers glue the newline onto each record's closing punctuation, which biases standalone ratios by about 3 points **[V]**. §6.2 counts the joined blocks directly.

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
| Optimized M | 1,921 (16.25) | 1,898 (16.0) | 1,923 (16.25) | 1,880 (15.9) | 1,928 (16.3) | 1,801 (15.2) |
| Concise English | 2,152 (18.4) | 2,127 (18.1) | 2,154 (18.4) | 2,110 (18.0) | 2,155 (18.4) | 2,173 (18.7) |
| **M (repo legend)** | **2,541 (21.9)** | 2,517 (21.7) | 2,543 (21.9) | 2,501 (21.5) | 2,547 (21.9) | 2,873 (25.0) |
| Controlled English (repo legend) | 2,635 (21.3) | 2,612 (21.1) | 2,637 (21.3) | 2,593 (20.9) | 2,638 (21.3) | 2,648 (21.6) |
| Positional JSON array | 2,881 (25.8) | 2,905 (26.0) | 2,883 (25.8) | 2,797 (25.15) | 3,001 (26.9) | 2,910 (26.2) |
| JSON (repo legend) | 5,973 (54.2) | 5,998 (54.4) | 5,975 (54.2) | 5,675 (51.4) | 6,229 (56.5) | 5,598 (50.8) |
| **Tab layout, words** (`r_tsv_p`) | **1,617 (13.4)** | 1,617 (13.4) | 1,619 (13.4) | 1,606 (13.3) | 1,805 (15.2) | 1,731 (14.4) |
| **Tab layout, fused code** (`c_tsv`) | **1,446 (11.7)** | 1,418 (11.4) | 1,448 (11.7) | 1,417 (11.4) | 1,600 (13.1) | 1,584 (13.0) |

**When M is cheaper** (fixed overhead + N × per-record cost):

| Against | M is cheaper when |
|---|---|
| Repository JSON | From N = 2 on real complete prompts (N = 3 on Gemma 3). The linear model gives N > 2.8–4.1. |
| Positional JSON array | N > 19–26 (Gemma 3: N > 85–101) |
| Repository controlled English | Skewed corpus: N < ~190 (Gemma 3: N < ~32); M's legend is shorter, but it costs more per record. Uniform corpus: always (Gemma 3: N < ~43) |
| Concise English, compact DSL, optimized M, both tab layouts | Never |

The uniform-feature and multilingual corpora give the same ordering. In the multilingual corpus, script and tokenizer dominate absolute cost. Qwen3 spends 1.08× (repository JSON) to 1.38× (natural English) as many tokens per record as o200k on the same content; 7 of the 10 formats fall in 1.21–1.33×. cl100k is measured for payloads and probes only, not in these chat-templated batches.

**Layout-matched codes vs words (added in the update).** The tab-separated word layout `r_tsv_p` is the cheapest readable reversible format on every tokenizer and corpus. The fused code `c_tsv` in the same layout is cheaper still. Ratio of `c_tsv` to `r_tsv_p` at N=100 (complete prompt):

| Corpus | c_tsv / r_tsv_p | Saving | Tokenizers at ≥10% saving |
|---|---|---|---|
| Default-skewed English | 0.877–0.915 | 8.5–12.3% | 5/6 (Gemma 3 fails) |
| Uniform features | 0.860–0.876 | 12.4–14.0% | 6/6 |
| Multilingual labels | 0.900–0.932 | 6.8–10.0% | knife-edge |

This is the strongest Sylang-style result found. It turns on comprehension, which no token count can settle (§11).

The flat predicate-only batch (200 records, no conditionals) gives the same picture:
- A TOON/CSV-style table with defaults omitted (2,974 Qwen3 tokens) loses to the compact DSL (2,379).
- M (3,346) loses to both.

*Run-to-run:* every count in this section reproduces byte-identically from `run.py` (`results.json` → `batch`).

### 6.3 Output side, caching and quality-adjusted cost **[M]+[L-d]+[A]**

**Verified pricing structure** (Anthropic, read 2026-10-04) **[L-d]**:
- Output costs 5× input on every listed model.
- Cache reads cost 0.1× input (0.05× on Opus 5.5, 0.025× on Fable 5.1). 5-minute cache writes cost 1.25× input.
- The minimum cacheable prefix is 512–4,096 tokens depending on the model ([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). Every prefix measured here (46–282 tokens) is below even the smallest (512).
- OpenAI and Google pricing was confirmed only through search indexes **[L-i]**. It shows similar output/input ratios (about 5–8×) and similar cached-read discounts.

Cost per call at N=100 in *input-token equivalents*, using Qwen3 token counts and these multipliers. This is an illustration, not a bill: Claude's own tokenizer differs.

The "prefix cached" column caches only the true prefix (template + task + legend, before the records), which is 46–282 tokens here. Every such prefix is **below the smallest minimum cacheable length (512)**, so the column is **counterfactual** unless the prefix is bundled into a larger stable one. Cache writes (1.25×) are ignored.

| Format | Read, uncached | Read, prefix cached (counterfactual) | Generate 100 records |
|---|---:|---:|---:|
| Natural English* | 1,540 | 1,499 | 7,365 |
| **Tab layout, fused code** (`c_tsv`) | **1,461** | **1,318** | **6,513** |
| **Tab layout, words** (`r_tsv_p`) | **1,632** | **1,506** | **7,444** |
| Compact word DSL | 1,881 | 1,790 | 8,845 |
| Optimized M | 1,936 | 1,817 | 8,996 |
| Concise English | 2,167 | 2,062 | 10,216 |
| **M (repo)** | **2,556** | **2,408** | **11,973** |
| Controlled English (repo) | 2,650 | 2,396 | 11,971 |
| Positional JSON array | 2,896 | 2,832 | 14,045 |
| JSON (repo) | 5,988 | 5,925 | 29,509 |

Consequences:
1. **Output is where tokens cost money and time.** Prefill is parallel; decoding is sequential (Splitwise, ISCA 2024; DistServe, OSDI 2024 **[L-i]**). vLLM's own documentation says prefix caching "only reduces the time of processing the queries (the prefilling phase)" **[L-i via repository docs]**. Here M's generation cost is 1.35× the compact DSL's, 1.61× the tab word layout's and 1.84× the fused code's, for the same 100 facts.
2. **Caching the legend does not rescue M.**
   - M's cacheable prefix (164 Qwen3 tokens) is below every Anthropic minimum cacheable length. It is only cached when bundled into a larger stable prefix.
   - Even with free legends, M is more expensive per record than six reversible alternatives, including the repository's own controlled English (five on the uniform corpus).
   - With a cacheable prefix reused over Q questions, input-side format savings keep only about 21.5% of their uncached value at Q = 10 **[V]**.
   - Near the minimum-length threshold, caching policy rather than format can decide which format is cheaper **[V]**.
3. **Retries, validation and accuracy.**
   - **Closed form [V].** With parse-failure probability f, accuracy a given a parse, and costs C_s (success) and C_f (failure), cost per correct answer = (C_s + C_f·f/(1 − f))/a. This is **independent of the retry cap**; the cap changes only coverage. Codec validation costs microseconds.
   - **M must be more accurate to break even.** M-repo needs at least 1.31–1.56× the compact DSL's accuracy to tie (generation, N = 100, output 4–8× input). That is impossible once the DSL's accuracy exceeds about 0.64–0.76. This is an offline kill condition for M-repo.
   - **The fused code needs at least parity accuracy in most settings.** The required accuracy ratio to pass a 10% cost-per-correct gate is (cost ratio)/0.90; tying needs only an accuracy ratio ≥ the cost ratio.
     - Generation, English labels (cost ratio 0.855–0.900): about 0.95–1.0, i.e. essentially parity.
     - Multilingual reading (cost ratio up to 0.93): ≥ 1.03, so the code would have to be *more* accurate than words.
4. **The NL → tree step is unaccounted for.** **[A]** Real inputs are prose. Some model must produce the tree, and that is a generation task priced at output rates. Savings from a compact notation exist only where structured facts already exist. There the question reduces to "which serialization should the model read or write", a well-studied engineering choice.
5. **Latency was not measured.** **[A]** For a fixed model without constrained decoding, decode latency scales roughly linearly with sampled output tokens. Input differences of a few hundred tokens change time-to-first-token by milliseconds to tens of milliseconds on modern GPUs; that is an order-of-magnitude estimate, not a measurement.
   - **Grammar fast-forward changes the ranking [V].** Engines that fast-forward forced bytes do not sample keys and punctuation. Full canonical JSON documents then need only 0.99–1.23× header-free M's sampled steps while billing 2.54–3.03× its output tokens.
     - For statement-only JSON records (the batch rendering in §6.2), the billed ratio is 2.05–2.60 (`results.json`, N=100).
   - So billing and self-hosted latency rank formats differently. No speed claim is justified without a serving experiment (plan Phase 3).

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
  - The strongest compact baselines are included: the tab word layout `r_tsv_p` (the layout-matched control), minified JSON and concise English, with natural English for reading. Positional JSON and the compact DSL stay as offline token baselines.
- **Separate reading from generation.**
  - *Reading:* format X → answer in a format-neutral way (short natural-language answer or field extraction).
  - *Generation:* natural language or JSON → format X, validated by the strict codec.
- **Quantify failures and uncertainty.**
  - Count parse failures, schema-invalid output, semantic errors by field, refusals, timeouts and retries.
  - Use paired designs with a non-inferiority test (Tango's score test, not McNemar, which tests equality) and call-clustered variance or a cluster bootstrap, with 95% intervals. Sample sizes and the pooled design are in plan §5.2.
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
| Sylang as a token-saving language as designed (repository M/Prime notation as the product) | **No-go** | **[M]** M costs 30–66% more than the best readable reversible format in complete prompts at N=100 on 6/6 chat-templated tokenizers. In generation it costs 1.43–1.70× the best readable format (Qwen3: 1.61×; 1.35× the compact DSL; 1.84× the fused code). **[V]** Offline cost-per-correct algebra requires 1.31–1.56× the readable format's accuracy to tie. |
| Optimized Sylang-style fused code (`c_tsv`) | **Undetermined; tested in the conditional-go study** | **[M]** 8.5–14.0% cheaper than the same layout in words on English-label corpora (≥10% on 5/6 or 6/6 tokenizers), 6.8–10.0% on multilingual labels. It needs near-parity accuracy to pass. Comprehension is unmeasured, and the codes are not atomic tokens. |
| Custom tokenizer, new tokens, vocabulary adaptation | **No-go** | **[M]/[L]/[V]** Glyphs give no gain; new tokens are meaningless until trained; incompatible with hosted APIs and caches; tokenizers drift. Atomic codes make M longer, and the structure floor is format-independent. |
| Fine-tuning to teach M | **No-go** | **[A]** The format is more expensive before training; any fine-tune could target a cheaper readable format instead. |
| Falsification-and-salvage study (plan Phases 0–2) | **Conditional go** | Cheap. Settles the only remaining question: the accuracy of the best Sylang-style code against the same layout in words, for reading and generation. Salvages the codec and harness as a format-evaluation tool, and recommends a readable compact format either way. Needs explicit approval for any paid inference. |

### 11.2 Decisive experiments

1. **E1 — Token falsification on billing tokenizers [P, mostly done here].** Add Claude (free `count_tokens`, needs a key and approval), Qwen3.5, DeepSeek and current OpenAI encodings to §6 on frozen realistic corpora.
   - *Stop the notation track if* no Sylang-style notation is ≥10% cheaper in complete-prompt *and* generation tokens than the best readable reversible format **at a matched layout**, on a majority of billing tokenizers.
   - **Status:** met for the repository's M on 6/6 chat-templated tokenizers (cl100k was not batch-tested). **Not met for `c_tsv`** on English-label corpora (5/6 and 6/6 pass), but met on multilingual labels.
   - Report it per label-length stratum. The decision depends on the target workload's literal load (§13.3.1).
2. **E2 — Reading accuracy [P].** The same facts in `c_tsv` vs `r_tsv_p` (the decisive matched pair), plus minified JSON and concise English as references. The repository's M is dropped: it is already killed offline.
   - Format-neutral, answer-balanced questions about polarity, scope, tense/aspect and evidence.
   - Two or more model families, a small and a large tier, with a pooled estimand.
   - *Pass for the code* only if it is non-inferior to `r_tsv_p` (one pre-registered margin, Tango test, call-clustered variance) *and* at least 10% cheaper per correct answer (cost-per-correct ratio ≤ 0.90). The statistics are corrected in plan §5 and review §13.3.10–13.3.13.
3. **E3 — Generation accuracy [P].** Natural language → format, codec-validated under one constraint policy for every arm: parse-failure rate, semantic accuracy, output tokens, retries.
   - *Pass for the code* only if cost per correct record is ≥10% below `r_tsv_p`.
4. **E4 — Serving latency [P, only if E2 and E3 pass].** A local open model on vLLM: time to first token, decode time, cold/warm caches, p95. Report fast-forward on and off.

### 11.3 Stopping criteria

- If E2/E3 show the fused code losing more accuracy than its token saving buys, **stop the language track permanently**. That means a cost-per-correct ratio not ≤ 0.90, which is the expected outcome. Then archive M/Prime as historical, keep the codec and harness, and recommend the readable tab layout (or minified JSON with structured outputs, whichever wins the salvage comparison).
- Stop immediately if reaching non-inferiority would require fine-tuning or vocabulary changes.
- Stop if results disagree across model families in a way that cannot be mapped to a deployment decision (i.e. no single format choice is robustly better).

### 11.4 What would change my mind

- **Accuracy.** A pre-registered, independently held-out study showing that a Sylang-style code is *both* non-inferior in accuracy *and* at least 10% cheaper per correct result than the same layout in words, on two or more current model families. "Cheaper" means billed tokens including legend, retries and output. Token counts alone already favour the fused code slightly at matched layout (§6.2), so accuracy is now the deciding evidence.
- **Scale.** A realistic workload where structure (not literals) dominates, for example very wide records with dozens of closed-enum fields, and where fused codes keep accuracy.
  - The headroom ceiling (§13.3.1) bounds the possible gain at about 12–17% on the default-skewed English corpus, 19–23% with uniform features, and 7–13% on multilingual labels.
- **Deployment.** Evidence that the production target is an open model the project will fine-tune anyway, *and* that teaching the code yields higher accuracy than teaching a readable compact format at equal training cost.
- **Tokenizer.** Superseded in part by the verification pass: at a fixed layout, word-based fields *are* more expensive than fused codes on 7/7 tokenizers. What remains open is whether that survives comprehension.

---

## 12. Evidence ledger

| Statement | Tier |
|---|---|
| 44 tests; 165/165 round trips; Qwen3 11.66% payload saving vs controlled English; JSON wins at N=1 | **[R]** |
| Rebuilt Qwen3 equals committed counts (330/330; counts, not IDs) | **[M]** |
| All ten reversible controls round-trip on 2,249 trees | **[M]** |
| Payload, batch, generation, layout-matched, glyph, Unicode, reserved-string and number tables; first crossing N | **[M]** (`results.json`) |
| README showcase sentence: 12–14 tokens, not ~5–7 | **[M]** (`results.json` → `probes.readme_showcase`) |
| Harness defects D3–D5 and D9 | **[M]** (`results.json` → `defects`) |
| NFC escape fusion in all five formats; Gemma 3 raw SentencePiece matches `<start_of_turn>` | **[M]** (`results.json` → `probes`) |
| §13 improvements and corrections not marked "run.py" | **[V]**: two independent verifiers per finding; scripts not committed |
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

---

## 13. Technical, architectural and mathematical improvements

This section adds the findings from a second verification pass: 60 survived, merged into 49 numbered items. For each finding, two verifiers worked independently: one reproduced the numbers empirically, the other checked the logic and value. Where they corrected the original finding, the corrected figure is used. Findings rejected in verification are left out.

**How to read each item**
- **Evidence label.** **[V]** measured during verification (tokenizers, tests, simulation) by the two verifiers; scripts not committed. **[V; M run.py]** additionally reproduced by this review's package (`results.json`). **[A]** analytical or reasoned. **[P]** proposed and not yet run.
- **Priority.** P0: invalidates a planned measurement or gate if not fixed. P1: should land before Phase 2. P2: hygiene. P3: affects review tooling only.
- **Effort.** S under a day; M 1–3 days; L longer.

The verification scripts live in the review session's scratch space and are not committed. Deliverable 0.14 in the plan asks for the decision-relevant ones to be committed.

**What changes, in one table**

| Topic | Effect on this review's conclusions |
|---|---|
| M as designed (M-repo) | **Unchanged.** It loses to readable compact baselines at every label length tested (1–16 words) and on 7/7 tokenizers (§13.3.4). |
| "Optimized codes only tie words" (§1, §6.1, §11.1) | **Overstated.** The tie is an artifact of m_opt's layout. At a matched layout, letter codes save 8–16% of payload, so G1 for Sylang-style variants is **undetermined** and sits near the 10% threshold (§13.3.2). |
| Phase 2 statistics (first-draft plan §5.2, G2; now corrected) | **Could not support the gates as first written.** The non-inferiority margin is mismatched, G2 is conjunctive, within-call clustering is ignored, and answer priors are skewed (§13.3.10–13.3.14). |
| Harness integrity | Task IDs are not bound to prompt content, and none of 20 targeted integrity mutants is caught by the test suite (§13.1.12, §13.2.1). |
| Unicode policy (§5.4) | The proposed rule leaves 98 unparseable cases. The fix is to check NFC-invariance on the *encoded* text (§13.1.1–13.1.2). |

---

### 13.1 Technical

#### 13.1.1 NFC fuses JSON escapes with a following combining mark — P1, S, [V; M run.py]
- **Mechanism.** `json.dumps` writes C0 controls as `\b \f \n \r \t` or `\u00XX`. Each of these ends in a letter or hex digit. If a combining mark follows, Qwen's NFC composes it with that letter. For example, `x` + LF + U+0303 is written `"x\n"` followed by the tilde, and NFC turns that into `"x\ñ"`.
- **Effect.** `decode()` then raises ParseError in **all 5 formats**.
- **Scale.** Exhaustive check over 32 controls × combining marks: 31,808 pairs. Of these, 112 become unparseable and 114 change the literal silently.
- **Why §5.4's rule fails.** The rule escapes the whole literal when NFC(raw) ≠ raw. That fixes all 114 silent changes and 14 of the unparseable cases. The other 98 unparseable cases remain, because their raw literal is already NFC.
- **Why a per-character variant also fails.** Escaping only the changed scalars creates new failures: U+212B + U+0307 becomes `\u212ḃ` after NFC, which is invalid JSON.
- **Rule to adopt.** State the guarantee on the model-facing text T itself: NFC(T) == T and decode(T) == tree.
- **Severity.** The trigger is a C0 control immediately followed by a combining mark, which is essentially absent from realistic labels.
- **Pointers.** codec.py:130-131 (`_quote`), :211; review §5.4; plan 0.6.

#### 13.1.2 A greedy NFC-safe escaper, and the MAX_TEXT_BYTES consequence — P1, M, [V]+[A]
- **Rule "min".** Keep the invariant that `'"' + output` is NFC. Append ASCII characters and JSON-mandatory escapes unchanged. Append a non-ASCII scalar raw only if it is assigned (category ≠ Cn) *and* the extended prefix is still NFC. Otherwise emit `\uXXXX`, using a surrogate pair for astral scalars.
- **Verification.** Finder: exhaustive over every scalar in 6 contexts (20,017,152 strings) and over 219,358,803 interacting pairs. Independent check: 22,000 random trees × 5 formats under 4 normalizers (HF Rust NFC, Python unicodedata 15.0, unicodedata2 18.0, ucd 3.2.0). Both found 0 failures.
- **Version robustness [A].**
  - Newer Unicode: the Normalization Stability Policy keeps the output normalized.
  - Older Unicode: decompositions and ccc values are immutable, and newer characters are inert starters.
- **Cost.** Zero on NFC input, except where a C0 control sits directly before a combining or NFC_QC=Maybe character. Those escapes are necessary.
- **Fair baseline.** The relevant comparison is a *corrected* whole-literal rule: escape the whole literal with ensure_ascii when the **encoded** literal is not NFC-invariant. That rule is also correct and also costs 0 on NFC text. Greedy saves tokens only on non-NFC input. Qwen3 counts (raw → whole-literal → greedy):
  - パソコン (NFD): 3 → 25 → 10
  - 서울특별시 (NFD): 6 → 65 → 35
  - Nguyễn (NFD): 7 → 23 → 20
  - café (NFD): 4 → 9 → 9
- **Required companion change.** Any escaping policy breaks the byte limit: 128 literals × 256 escaped astral scalars × 12 B = 393,216 B, which exceeds MAX_TEXT_BYTES = 262,144. Raise the limit to ≥ 2^19; this is not optional. Pass a quote function rather than using a module-global flag.
- **Trap.** `unicodedata.decomposition('가') == ''`. A Maybe table derived from it therefore misses all 48 conjoining V/T jamo, and 'ᄀ'+'ᅡ' would stay raw and be composed.
- **Pointers.** codec.py:24, :130-131, :205-224.

#### 13.1.3 Invisible, bidi and format characters in literals — P2, M, [V]
- **Problem.** The codec emits all of these raw in every format: RLO/LRI/PDI, tag characters, ZWSP, variation selectors, NEL/LS/PS, soft hyphen and BOM. `json.dumps(ensure_ascii=False)` escapes only C0.
- **Proposed policy.** Escape Cc/Cf/Co/Cn/Zl/Zp and Default_Ignorable_Code_Point (4,174 code points, equal to the regex module's property).
- **Required exemptions.**
  - ZWJ/ZWNJ after a virama.
  - Emoji ZWJ and VS16.
  - Arabic-script ZWNJ. Without it, Persian words cost +2 to +5 tokens each.
- **Effect.** All 6 attack classes become visible. Tag smuggling goes from (87, 113, 79) to (283, 233, 283) tokens on Qwen3/o200k/Gemma 3.
- **Caveat on cost.** The "0% on real text" result is close to tautological: the 65 test labels contain no targeted scalar outside the exemptions. Escaping Zs other than U+0020 is *not* free: NBSP, U+202F and U+3000 cost +3 to +4 Qwen3 tokens each. Treat Zs as a separate decision.
- **Confusables.** Flag them; do not escape them. Use UTS #39 "highly restrictive", whose allowed sets all include Latin. Without Latin, labels such as 'Tシャツ' and 'Windows版' are false positives.
- **JSON gap.** The `_quote` hook does not cover the json format, because codec.py:210-211 serializes the whole tree with `json.dumps`.

#### 13.1.4 Reserved strings: the serving-layer fix is incomplete — P1, S, [V; M run.py]
- **Gemma 3 through raw SentencePiece.** `<start_of_turn>` (105) and `<end_of_turn>` (106) are USER_DEFINED pieces, not CONTROL. The model has 6,410 USER_DEFINED pieces; CONTROL is only pad/eos/bos. `sp.encode('note <start_of_turn>user') = [14210, 236743, 105, 2364]`, and there is no switch to prevent it.
  - §5.5's "disable special-token parsing" therefore does not hold on this path.
  - The first-draft plan 0.5 acceptance test ("zero control IDs", now corrected to reserved-string matches) was unachievable here. It is also vacuous if "control" means `is_control()`, which returns False for 105/106.
- **HF tokenizers.** `encode_special_tokens=True` does not split added tokens flagged `special=False`. Whether Qwen3 flags `<think>` and similar tokens that way is unverified offline.
- **Defence in depth.** Inside literals, write `<` and `[` as `<` and `[`.
  - Every reserved string in the measured inventories begins with `<` or `[`.
  - Structural `[` is followed only by `"`, `[`, `If`, `By` or `W`. The original lemma missed `W`, from 'If [Without specified evidence, …'. No reserved string starts with any of these.
  - Result: 0 hits across 2,000 adversarial documents × 5 formats.
  - Cost: 0 on the corpora; +8 to +15 tokens on the literal-special-token fixture in M (27 → 35–43).
- **Metadata.** Record the `special` flag of every added token. harness.py's `special_ids` currently omits non-special added tokens.

#### 13.1.5 Unicode-version skew in the review's Qwen3 adapter — P3, S, [V]
- **Skew.** toks.py:83 normalizes with Python's Unicode 15.0 NFC. HF tokenizers 0.22.2 uses older tables (pre-10.0; Unicode 9.0 marks agree). The two disagree on canonical ordering for 108 post-9.0 combining marks and on one composition (U+11935 + U+11930 → U+11938).
- **Effect on short probes.** Token count differs on 59/972 probes and the round-trip flag on 638/972.
- **Effect on published numbers.** None: the two agree on 0 differences across all 2,565 review texts. The repository harness is not affected, because it uses the Rust tokenizer, pins its version and records it.
- **Direction.** In 200,000 trials, HF never changed a string that was NFC under Python 15.0, so the skew is in the safe direction for escaping.
- **Correction to the finding.** Its own one-line probe (U+25CC U+0334 U+0D3B) prints True; only the reversed order shows the difference.
- **Fix.** Use `tokenizers.normalizers.NFC()` in the adapter. **Applied in this update** (`toks.py`); no published count changed.

#### 13.1.6 Codec regression tests miss three realistic mutants — P2, S, [V]
- **Robustness is good.** Fuzzing found no crash: 962,500 decode calls, Hypothesis, and 131,072-deep JSON on Python 3.10–3.13.
- **Surviving mutants.** These pass the full 44-test suite:
  - `expect("condition")` → `atom(("condition","consequence"))`, and the symmetric consequence mutant. DSL and Prime then silently accept `if(consequence=X,consequence=Y)` and read X as the condition.
  - Separators tested with `char.isspace()`. NBSP, U+3000 and U+2028 are then accepted.
  - Byte-limit off-by-one. The existing probe is 262,148 B, which overshoots the limit by 4 bytes, so the boundary itself is never tested.
- **Fix.** Three targeted tests:
  1. Keyword-swap rejection in DSL and Prime.
  2. NBSP, U+3000 and U+2028 rejected as separators.
  3. Exact MAX_TEXT_BYTES and MAX_TEXT_BYTES + 1, each with one 2-byte character.
- **Optional.** An exhaustive one-edit neighbourhood oracle (83,265 edits, about 7 s) kills only the scope mutants. The "12/12 killed" figure is post hoc.
- **Pointers.** codec.py:201, :241, :336, :340.

#### 13.1.7 JSON Schema: generate it, depth-unroll it, keep it portable — P1, S, [V]
- **Known gap.** The committed schema accepts depth-9 trees, lone surrogates and 255-node trees. Its `$comment` already discloses this.
- **Real risk.** Enum drift. The schema test checks only the version constant, `additionalProperties` and the predicate's required set.
- **Fix.** Generate the schema from the codec tables and test that the committed file equals the generator output.
  - Depth-unroll it: `$defs` node1..node8, with node8 predicate-only.
  - Use `anyOf`, which is exact here because `kind` is a const discriminator. llguidance refuses `oneOf` (§13.1.9).
  - Result: 0 depth or surrogate disagreements with `validate()` (jsonschema 4.26).
- **Portable surrogate pattern.** `^[^\ud800-\udfff]*$` rejects every astral label under ECMA-262 regex semantics without the `u` flag. Use `^(?:[^\ud800-\udfff]|[\ud800-\udbff][\udc00-\udfff])*$`.
- **Remaining gap.** The node cap is still inexpressible: n ≤ 127 nodes is p ≤ 64 predicates, since n = 2p − 1. MAX_DEPTH = 7 would imply it (2^7 − 1 = 127), but that needs a version bump.

#### 13.1.8 The EBNF omits the lexical rules — P2, S, [V]
- **Unstated rules.** WORD is maximal munch over `[A-Za-z][A-Za-z0-9_.-]*`, and whitespace is exactly SP/HT/CR/LF.
- **Effect.** A grammar transcribed literally from experimental-grammar.md:160-185 disagrees with `decode()` in **both** directions:
  - It over-accepts `Prime0.1pred{…}`, `Sylangcore-v0.1:`, `doesnot see`, `Bydirect` and `has notseen`.
  - It under-accepts `pred { subject = "A" ; … }`, because the EBNF fuses `pred{subject=` into one literal.
- **Fix.** Add a normative lexical section, and generate grammars with word-boundary keyword terminals.

#### 13.1.9 Constrained-decoding grammars generated from the codec tables — P1, M, [V]
- **Generator.** About 150 lines, emitting llguidance Lark grammars.
- **Exactness.**
  - On o200k: 0 over- or under-acceptances on 34,424 canonical documents and 375,604 single-edit mutants.
  - Independent check on cl100k: 7,800 positives and 310,236 targeted mutants, also 0.
  - "Exact" is evidenced for one engine and single-edit mutants only.
- **Only gap.** The 127-node cap. It binds only at depth 8.
- **Canonical label terminal.** `"(?:[^"\\\x00-\x1f]|\\["\\bfnrt]|\\u00(?:0[0-7bef]|1[0-9a-f])){1,256}"`
- **Two engine traps.**
  1. A fused `'["if",'` literal makes llguidance's greedy lexer reject **every** json_array document.
  2. llguidance refuses the committed schema's `oneOf`.
- **Size and cost.** The worst-case canonical document is 208,699 B (json), 20.4% under the byte limit. Regex-only unrolling grows exponentially: m_body reaches 51,887 characters at depth 8, against a 619-character CFG.
- **Other engines.** XGrammar and GBNF need their own differential runs. A grammar covering syntax only still admits empty strings and lone `\uD800` escapes, so run `validate()` afterwards.

#### 13.1.10 Token alignment at prompt boundaries — P2, S, [V]
- **Terminal crossing.** Canonical tokens cross grammar-terminal boundaries in every record; for example, 38–42% of JSON tokens. Byte-level engines handle this: llguidance never masked the canonical token in 1,267,875 steps. Crossing therefore matters only for home-grown per-terminal maskers.
- **Header prefill.** Cut points matter:
  - `Sylang core-v0.1:`, `Prime0.1` and `M0.1` give 0% straddle on 7/7 tokenizers.
  - Cutting after the trailing space or colon gives 100% straddle.
  - JSON straddles even at `…"statement"`, so its prefill must stop earlier or rely on engine fast-forward.
- **Batch separators.** The newline fuses with the record end (`)\n`, `}\n`, `.\n`):
  - 100% of boundaries for M, JSON and English on 6/7 tokenizers.
  - 34% for m_opt and cdsl.
  - 0% on Gemma 3.
  - Fix: put the separator inside the batch grammar instead of chaining per-record grammars with EOS.

#### 13.1.11 Whitespace drift and the m_opt decoder — P2, S, [V]
- **Drift cost.** Adding one space after `,`, `:` and `;` outside string literals changes generation tokens:

  | Format | Change |
  |---|---|
  | json | +33.5 to +38.2% |
  | json_short | +29 to +34% |
  | json_array | +25 to +30.5% |
  | m | +18.5 to +22.8% (Gemma 3 +4.4%) |
  | m_opt, cdsl | +6 to +10% |
  | dsl, prime | −2.6 to −10% |
  | English | 0% (it already spaces separators) |

- **Implication.** Drift only widens M's deficit, so it cannot rescue G1. It does move the JSON arm by about 3× the 10% decision margin.
- **Fix.** Pin canonical separators. For llguidance's JSON Schema path (`JsonCompileOptions.whitespace_flexible`, default true), set `whitespace_flexible=false`.
- **Decoder asymmetry.** `d_m_opt` rejects every whitespace drift (0/400 each). All other decoders repair it. This is a one-line fix when porting.

#### 13.1.12 Harness tests do not pin its integrity guarantees — P1, M, [V]
- **Result.** All 20 non-equivalent hand-written mutants of evaluation/harness.py and `__main__.py` survive the 44 tests; 3 positive controls are killed. Branch coverage is 83%.
- **Most consequential survivors.**
  - `encode_special_tokens` flipped. The unit-test tokenizer has `<reserved>` in its base vocabulary, so the flip is invisible to it.
  - Summary ratios computed against the format's own row. Tests only assert that the key exists.
  - Median replaced by the mean.
  - `semantic_roundtrip_exact` hard-coded True.
- **Real-tokenizer impact of the flag flip.** On Qwen3 it changes literal-special-token payloads by +17% to +35% (+7 to +10 tokens): english 22 → 29, json 50 → 60, dsl 41 → 48, prime 39 → 46, m 20 → 27. The +78% figure applies only to the bare fragment (9 → 16 tokens).
- **Exit codes.** rc = 2 is used for usage errors, tampered fixtures and a missing manifest alike, and the scorer returns 0 when every answer is wrong.
- **Near-equivalent mutants.** The byte-length and count checks are mostly redundant given the SHA pin.

#### 13.1.13 Scorer: field diagnostics, an answer envelope and confound flags — P1, M, [V]
- **Envelope.** `score_answers` (harness.py:322) accepts only `{task_id, answer}`. The raw text needed for the D4 fix and the usage needed for G2 cost have nowhere to go.
- **Field diff.** A path-aligned diff with a fixed rule order (structure → scope swap → role swap → literal [nfc/whitespace/case/content] → per-enum) recovered 165/165 injected errors with 0 false positives (independent implementation).
- **Tokenization confounds.**

  | Tokenizer | Confounded tasks | Cause |
  |---|---|---|
  | Qwen | 5/165 always | literal-decomposed is not text-exact after NFC |
  | Qwen | +5/165 on the unsafe path only | literal-special-token |
  | o200k, cl100k | 5 | literal-special-token |
  | Tekken | a *different* 5 | literal-code (`</s>`) |
  | All tokenizers (union) | 15 distinct tasks | |

- **Fix.** Add per-tokenizer visibility flags, and report accuracy with and without flagged tasks. Plan items 0.5 and 0.6 remove most of these confounds at the source.

#### 13.1.14 Export format — P2, S, [V]
- **Line separators.** The JSONL export keeps raw U+2028, U+2029 and U+0085 (5 records each). `str.splitlines()` then yields 180 fragments, of which 20 fail to parse. Fix with `ensure_ascii=True` (+1.46% bytes) or targeted escaping (+50 bytes).
- **Prefix caching.** The static prefix is only 218–440 Qwen3 tokens, below the minimum cacheable length. For plan 2b the cache boundary belongs *after* the record block, so a system=legend / user=payload split is the wrong cache boundary.
- **Format label.** `Format: <name>` is constant within each arm and is part of the treatment, not a confound.

#### 13.1.15 Packaging and CI — P2, S, [V]
- **CI.** There is no CI, although plan Phase S's acceptance criterion says "CI runs the retained tests".
- **Packaging.** There is no pyproject or entry point, so `python -m sylang_core` works only from the repository root.
- **Typing.** `mypy --strict` reports 2 cosmetic errors (codec.py:213, :446).
- **Python versions.** Tests pass on 3.10–3.13.

#### 13.1.16 Review package reproducibility — P2, S, [V]
- **Unreproducible [M] labels (fixed in this update).** Five claims carried [M] but were produced by neither run.py nor results.json: the README showcase count (true: 14/12/14/14/14/14/12) and D3, D4, D5 and D9. `run.py` now emits them (`probes.readme_showcase`, `defects`).
- **Decoder strictness.** "Strict decoders" means fail-closed on structure, not canonical-only.
  - cdsl, english_concise, m_words, json_array and the repository's own M decoder accept non-canonical spacing.
  - m_opt is the only whitespace-strict decoder.
  - This does not bias Phase 2 as planned (m_opt is not an arm), but every ported decoder should report a separate `canonical` flag.
- **Other gaps.**
  - run.py has only `--output` and records no code hashes.
  - The package imports 8 private codec symbols and uses 4 header-offset slices.
  - The `break_even()` docstring does not explain the negative-value convention stored in results.json.

#### 13.1.17 Asset licence and provenance metadata — P2, S, [V]
- **Coverage.** Licence fields exist for 2 of the 9 tokenizer assets (the two Qwen entries, Apache-2.0). The 7 review assets have SHA pins but no licence fields, contrary to the plan's risk control (plan §10, licence row).
- **Llama.** The llama-models 0.3.0 wheel ships llama3 and llama4 `tokenizer.model`. Its only licence file is a 78-byte URL, and its README says to "read and accept the license".
- **Unhashed install.** The tokenizers install is not hash-checked.
  - Windows wheel: c9ea31ed…91f05c48.
  - manylinux wheel: 369cc9fc…8d72c67 (recorded nowhere).
  - A two-hash `--require-hashes` file breaks macOS and aarch64, so list every platform wheel.
- **Breach?** None found: only counts are committed, and `.cache/` is ignored. The Llama 3 vs 3.1 output clause and the Gemma gate are unverified legal readings.

#### 13.1.18 Historical claims, measured — P2, S, [V]
- **Pooled ratio.** Over 6 same-meaning pairs, Sylang/English = 0.985–1.108 across 7 tokenizers. Individual pairs range 0.6–1.25: 'navimokas' is 0.6–0.8 and the showcase sentence 0.80–0.93. The claimed ratios are 0.2–0.625.
- **Token vs word counts.**
  - PDF Table 1's "token" counts equal its word counts (6/3, 8/5); its column headers say "(words)".
  - README word counts are 15 and 3, not 14 and 1.
- **Context arithmetic.** It is inconsistent: 1/(1 − r) gives 1.82×, 2.22× and 2.5× for r = 0.45, 0.55 and 0.60.
- **README structure.** The `#-performance-benchmarks` anchor lands on "Rigorous benchmarking confirms" with no caveat. README:133-135 restates future-tense expectations from tokenizer-design.md:81/85/89 as results.
- **PDF provenance.** The PDF is a ChatGPT deep-research export whose citations point to 5 uploaded files that are not in the repository.
- **Images.** Two images carry misleading names: tokenization-process.png actually shows "LEXICON DESIGN", and token-reduction.png shows "LEARNING SYSTEM DESIGN" and is unused.
- **Scope.** Claims about the never-built custom 8k tokenizer are unmeasurable, not contradicted. "32k–50k" was accurate for GPT-2 and LLaMA-1/2. Fold all of this into the Phase S banner.

---

### 13.2 Architectural

#### 13.2.1 Bind comprehension tasks and answers to prompt content — P0, S, [V]
- **Problem.** `task_id = TEMPLATE_VERSION/fixture_sha[:12]/fixture/format`.
- **Demonstration.** Swap two M aspect codes in legends/m.txt and change "Return only the JSON object." to "Return only YAML.". All 165 prompts change; all task IDs stay byte-identical; answers produced for the old prompts still score 165/165 with rc = 0.
- **Fix.**
  - `task_id = TEMPLATE/prompt_sha256[:16]/fixture/format`, plus a `prompt_sha256` field. The payload is inside the prompt, so this also covers legends, instructions and codec rendering.
  - A per-record `ast_sha256` over `json.dumps(ast, sort_keys=True, separators=(',',':'), ensure_ascii=False)`. This is byte-identical to RFC 8785 JCS on 33/33 fixtures and 8,000 random trees.
  - A manifest root over the full canonical records (id, split, tags, ast).
- **What not to do.** Do not hash the harness and codec sources: this is redundant and brittle.
- **Result of the minimal patch.** 44/44 tests pass, and stale answers are rejected. No answers exist yet, so now is the cheapest time.

#### 13.2.2 Separate deterministic counts from volatile fields — P1, M, [V]
- **Problem.** Two identical runs differ in 660 of 7,325 lines (timing fields only). They are equal once `*_cpu_ns` and environment are stripped.
- **Proposed layout.**
  - Canonical `counts.jsonl`: 141,736 B in 165 lines, against 347,501 B in 11,119 lines today.
  - `summary.json` and `run.json`.
  - A `verify --expected` command that byte-compares counts and names the first differing row.
- **gzip_bytes is not implementation-independent.** zlib-ng differs on 2/165 payloads (19,375 vs 19,373). Record `zlib.ZLIB_RUNTIME_VERSION`, or exclude gzip from byte identity.
- **Splits.** Compute split summaries only for n ≥ 10; holdout is n = 2 today.

#### 13.2.3 Tokenizer adapter protocol, certified on IDs — P1, M, [V]
- **Limitation.** `LocalTokenizer` requires a 40-hex revision and HF JSON under tokenizers 0.22.2. It cannot represent tiktoken, SentencePiece or API counters, so plan 1.1 needs a multi-kind lock.
- **Conversion trap (cl100k).** Oniguruma reads tiktoken's possessive `\p{N}{1,3}+` as `(\p{N}{1,3})+`.
  - Converted cl100k matches **330/330 suite counts** but only 320/330 IDs.
  - It diverges on 1,649/2,000 random 4–40-digit strings (mean 1.26×, max 1.5× tiktoken's count). Divergence starts at 6-digit runs.
  - Rewriting to `\p{N}{1,3}` or `(?>\p{N}{1,3})` gives 0/2,000 and 0/20,000 divergences.
  - Stripping every possessive is worse: 2,326/20,000.
- **Lesson.** Count equality on the suite is a weak equivalence certificate, and that applies to the review's own Qwen3 reconstruction. Certify on IDs over a stress corpus: digit runs 1–64, whitespace and CR/LF runs, U+2028/U+0085, combining marks, emoji ZWJ sequences and all reserved strings.
- **Published numbers.** None are affected.

#### 13.2.4 One error taxonomy across formats — P1, S, [V]
- **Problem.** The same fault raises a different class per format. An unknown relation or a depth-9 tree raises ParseError in english/dsl/prime/m but ValidationError in json. Plan rules keyed on exception class (one retry on parse failure; separate parse-failure vs schema-invalid counts) would therefore treat identical model errors differently by arm.
- **No structured fields.** `vars(e) == {}`. The JSON position survives only as `e.__cause__.pos`. A duplicate-key message can reach about 131,000 characters.
- **Minimal fix.** Either make text readers raise ValidationError with a field path for enum and bounds faults, or have the scorer classify by stage (decode, then `validate()`). Truncate echoed input either way.

#### 13.2.5 Single-source spec: start with a drift test — P1, S, [V]
- **Drift demonstration.** Adding relation `know` to the codec tables passes 43/44 tests, and the one failure is incidental. All 5 formats round-trip the new tree, but the committed schema rejects it, `prompt_for` and the legends omit it, and SCHEMA_VERSION is unchanged. No test reads the legends.
- **Step 1 (S effort).** A drift test asserting: codec tables == schema enums == `prompt_for` enumeration == legend code tables.
- **What a full generator achieves (prototype).**
  - A JSON Schema equal to the committed one, ignoring annotation keys.
  - Strict LALR(1) grammars for 5 surfaces that round-trip 2,249 trees.
- **Limits of the prototype.**
  - The generated M legend is **not** token-identical: 118 vs 139 Qwen3 tokens, because it omits two sentences.
  - No legend generator exists for the other formats.
- **Recommendation.** Adopt the full pipeline only if the codec survives Phase S.

#### 13.2.6 Layered API and transport policy — P2, M, [V]
- **Public API.** Header-free `encode_node`/`decode_node` (needed by plan 0.3) removes the 8 private imports and 4 offset slices.
- **Literal-policy hook.** A hook in `_quote` covers english/dsl/prime/m. JSON needs its own walk over literals.
- **Measured effect** (NFC + controls + reserved-string escaping, 33 fixtures × {m, dsl, english}):
  - Qwen3 text fidelity 96 → 99/99.
  - Reserved-string hits 3 → 0 on Qwen3, o200k, cl100k and Tekken.
  - Codec round trip stays 99/99.
  - Token cost +1.4% to +3.3%.
  - 8/37 literals change; 0/46 corpus labels change.
- **Role.** Defence in depth; the primary fix remains the safe tokenization path.

#### 13.2.7 Symmetric constraint policy for G1/G2 — P1, S, [V]+[A]
- **What is already right.** The first-draft plan ran JSON with provider structured outputs as a separate condition (plan §5.1).
- **The asymmetry.** Only JSON gets a constrained condition, and the first-draft salvage rule (plan G2) fell back to constrained JSON against unconstrained compact formats. Both are corrected in plan §5.1 and G2.
- **Engine cost is negligible.**
  - Compile plus first mask: 1.6–6.1 ms (median 2.0).
  - Per-step mask: 43–65 µs mean, p99 143–209 µs, on 100k–262k vocabularies.
- **What the "deployment tax" really is.** Hosted availability of custom grammars ([A]), not compute.
- **Uniform post-processing for every arm.** decode → canonical re-encode → node-cap check → at most one retry. Constraint becomes an explicit factor.

#### 13.2.8 Prefix-freeness and truncation — P1, S, [V]
- **Problem.** Canonical m_opt and cdsl are not prefix-free. 131/200 en_skewed records have a valid proper prefix that decodes to a *different* tree; for example, `h("Bo","Cy")/p` truncates to `h("Bo","Cy")`, which drops a field. A truncated or streamed output can therefore decode silently to the wrong tree.
- **Fixes.**
  - Head-first layout (features before the parenthesis) makes every variant prefix-free (0/163). A separate terminator is then redundant.
  - Cheaper, layout-free alternative: score every length-stopped generation as truncated.
- **Refuted claim.** Head-first does **not** reduce label-forgery exposure; it moves it to a different reader model. For canonical m vs head-first m, a prefix reader sees 10/11 forgeable cells vs 1/11, and a suffix reader 1/11 vs 10/11.
- **Token effect.** Head-first changes m_opt by −9.1% to −10.5% on 6/7 tokenizers and +4.5% on Gemma 3. That is a layout effect.

#### 13.2.9 json_short is not canonical-strict — P1, S, [V]
- **Problem.** `d_json_short` uses plain `json.loads`, so it accepts duplicate keys (last one wins) and any key order. When the label `Ada","polarity":"negative` is copied unescaped, the result decodes as polarity = negative: 36/36 key-mimic cases, against 0/36 in each of the 11 other formats.
- **Fix.** Emit enum keys first. Decode with an `object_pairs_hook` that rejects duplicates and requires a strictly increasing canonical key index.
  - Result: 0/144 forgeries.
  - Token cost: identical on 7/7 tokenizers.
- **Rule for any default-omitting format.** An optional field may be omitted only at a fixed position *before* every literal.

#### 13.2.10 Fix the model-facing injection defence before G1 — P1, S, [V]
- **Delimiter swaps are expensive and depend on format** (en_skewed, vs JSON quoting):

  | Defence | Cost |
  |---|---|
  | Guillemets | +1.3% to +23.9% |
  | ⟦⟧ | +15.6% to +97.0% |
  | Length prefix | +14.5% to +45.7% |

  Guillemets alone move m_opt/english_concise from 0.886 to 0.971 (Qwen3), 0.885 to 1.020 (o200k) and 0.814 to 0.995 (Gemma 3). Do not adopt any of these.
- **Cheap option: `"` / `\` quoting.** Existing decoders accept it.
  - Cost per escaped quote: +2.5 tokens (o200k, cl100k, Llama), about +4.3 to +4.5 (Qwen3, Tekken), about +5.0 (Gemma 3).
  - That is +0.6% to +3.8% on gold. It is 0% on the corpora only because none of their 1,008 labels contains a quote.
- **Head-first is itself a pre-registration item.** It moves m_opt/english_concise from 0.886 to 0.798 on Qwen3.
- **Sanitizer cost.** Banning C0 and C1 controls drops 3/33 gold fixtures.
- **Scoring rules.** Strict-decode the whole output, reject trailing text, never use first-record or first-object extraction, and keep the raw output.
- **Line-split forging.** A label containing `\n` forges extra records in 8/8 line-per-record formats under a full-unescape reader.
- **How not to read the prefix-forgery counts.** They are constructive (404/404 by design) and specific to one reader model. They are not a robustness ranking of formats.

#### 13.2.11 Label interning is a declared factor, not a free win — P2, M, [V]
- **Savings differ by format** (en, N = 200, mean reuse k̄ = 17; Qwen3/o200k/Gemma 3):
  - json_array: +19.1 / +18.5 / +14.4%
  - cdsl: +40.5 / +39.8 / +20.9%
  - english_concise: +43.1 / +42.4 / +39.3%
  - m_opt: +46.3 / +45.6 / +21.7%
- **Low reuse.** At k̄ = 1.8, interning every label loses 4% to 31%. A minimum-3-occurrence rule limits the worst case to −2%.
- **Realistic reuse.** On a Zipf pool of 200 names (s = 1.2, k̄ = 4.8), min-3 interning saves +30.6% on Qwen3.
- **Break-even reuse.** k* = 1 + (2r + δ)/(t − r).
  - TSV/@alias layout: 7.4.
  - Bare uppercase IDs: about 2–3.5.
- **Symmetric application still shifts ratios [A].** (S_M + L′)/(S_D + L′) rises as the shared literal cost L′ falls whenever S_M > S_D. Interning must therefore be pre-registered, reported per format, or excluded from G1.
- **Exposure.** One forged binding reaches 17–30 records in these corpora. This scales with N × labels per record / vocabulary size.

#### 13.2.12 Seeded item generator: deterministic, but two cues remain — P1, M, [V]
- **What works.** The prototype is byte-identical across hash seeds (items_sha256 0fc1505a…, 3,585,012 bytes). Balance, crossing, Cramér's V and strength-3 coverage checks all pass.
- **Three defects.**
  - **(a) Spine trees.** Every conditional tree is a spine. A reader that finds the unique innermost `if` with two leaf children scores **1.000** on all 1,824 conditional items, so outer scope navigation is never tested.
  - **(b) Lure-determined answers.** The lure and filler rules make the answer a function of the adjacent line. Cramér's V = 1.0 for k = 3 types and polarity. A lure-elimination reader scores 0.50 against a chance rate of 1/3.
  - **(c) Depth.** Maximum depth is 3, not the plan's 4.
- **Correction.** The finding's claim that the checks "caught a scope leak" is wrong. The earlier version already scored 0 at pair level on non-leftmost targets.
- **Fix.** Randomize tree shape, sibling position and lure values. Report heuristic readers on non-leftmost targets against chance.

#### 13.2.13 Code-token alignment between legend and payload — P2, S, [V]
- **Already known.** Code atomicity fails (review §4).
- **New: the legend teaches different tokens.** Payload code tokens equal the legend's tokens in only 18.9–19.5% of occurrences on the BPE tokenizers, 45.3–45.6% on Gemma 3, and 0% for m_opt. The legend teaches ` s`, ` +=`, ` p`; the payload presents `,g`, `+,`, `,u`.
- **Space-led M.** Codes separated by spaces, aspect recoded a/g/k, `i(A,B)` keeping its comma.
  - 100% atomic, legend-identical and code-monosemous on 7/7 tokenizers.
  - Identical token totals on 6 tokenizers; −10.0% on Gemma 3.
  - Round-trips 633/633.
  - In 28 paired spacing tests the spaced form was never more expensive (22 cheaper, 6 equal).
- **Caveats.** It is a new format version, and any accuracy effect is unmeasured. Record alignment as a covariate.

#### 13.2.14 Schema growth: code vs word relation identifiers — P2, S, [V]
- **Legend cost.** A code legend grows about 5 tokens per relation; a word list about 2.8–3.3.
- **Per-occurrence saving depends on layout.**
  - Call layout `REL("Ada","Bo")`: 0.60–0.98 tokens at V = 120. Break-even ≈ 680–990 records uncached, ≈ 67–100 with a cached legend.
  - Fused TAB layout: 1.53–1.72 tokens. Break-even ≈ 383–390 uncached, ≈ 38–39 cached.
- **Conclusion.** Tokens do not robustly favour word identifiers; the choice rests on the unmeasured accuracy cost of teaching codes.
- **Lint.** Feature value sets should be pairwise disjoint across fields; this enables default omission and order-tolerant parsing.
- **Lark caveat.** strict=True does not flag keyword/identifier collisions with `lexer='basic'`, or with contextual states that accept both.

---

### 13.3 Mathematical

#### 13.3.1 Additive literal/structure cost model and the headroom ceiling — P1, S, [V]+[A]
- **Decomposition.** Per-record cost T = L + S (literal + structure).
- **Substitution estimator.** S_X = T(X with every label replaced by 'X') − n_labels, and L_X = T − S_X.
  - For quote-delimited formats, L equals the standalone label sum exactly on 7/7 tokenizers.
  - In unquoted TAB layouts, L shifts by −3.5% (Tekken) to +21.7% (o200k, multilingual).
- **Saving and gate condition.** Saving of N over R = (S_R − S_N)/(L + S_R). This is exact only between formats that share literal contexts. The 10% gate is reachable only if L < L* = 9·S_R − 10·S_min.

| Corpus | Ceiling vs r_tsv_p | L* vs r_tsv_p (tokens/record) |
|---|---|---|
| en_skewed | 11.7–16.5% | 10.5–17.2 |
| en_uniform | 19.2–23.2% | 21.7–29.2 |
| multilingual | 6.8–13.2% | 9.8–15.4 |

- **Against cdsl.** L* = 22–55.
- **Hand check.** Qwen3: 9·5.13 − 10·3.56 = 10.6.
- **Label length decides.** With 2-subject labels (L ≈ 17–21), c_tsv's payload saving falls to 5.8–7.9% and fails the gate.
- **Information content.**
  - H(features) = log2 216 = 7.755 bits uniform, 6.610 bits skewed.
  - Per record: 9.47–10.94 bits ≈ 0.53–0.66 tokens.
  - About 93–94% of cdsl's S (8.4–10.5 tokens) is therefore delimiter syntax.
- **Caveats.**
  - When L differs across layouts the formula overstates. For o200k multilingual vs cdsl it predicts 28.6% where 17.9% was achieved.
  - The label-length scan is close to a tautology (S is constant by construction).
- **Fitted overhead model.** M−JSON R² = 0.999. M−English R² = 0.952 once polarity and feature terms are added.
- **Report.** Present G1 as a curve over label length, not as a single point.

#### 13.3.2 "Codes only tie words" is a layout artifact — P0, S, [V; M run.py]
- **Confound.** m_opt vs cdsl changes the code *and* the layout together: the `/` separator costs a token and breaks the `")\n` merge.
- **Matched layout.** c_tsv uses M's letter codes; in it, TAB fuses with the code cluster (`\tcp`, `\tsc`). Its field costs 0.631 tokens per predicate, against 1.957 for words (Qwen3).

| Comparison (7 tokenizers unless noted) | en_skewed | en_uniform | multilingual |
|---|---|---|---|
| Payload, c_tsv vs r_tsv_p (codes vs words, same layout) | 12.7–14.7% (Gemma 3 9.8%) | 14.3–16.2% | 8.0–12.1% |
| Complete prompt N=100 vs best readable, finder's legends | 8.8–12.9%, passes 5/6 | 6/6 | 3/6 |
| Same, independently written legends | 8.2–12.0%, passes 5/6 (Qwen3 10.3%) | 6/6 | 0/6 at N=100, 3/6 at N=200 |
| Same, with two more readable layouts added | 8.8–10.8%, passes 5/6 (knife-edge) | 12.6–13.8%, 6/6 | 2/6 |

- **m_opt vs cdsl on the corpora.** The saving ranges from −1.9% to +7.3%. The ±1.4% band holds only on the 33 fixtures.
- **Fused-head variant.** m_fused_q is 12–15% cheaper than cdsl. cdsl is not the best readable baseline, so this alone does not establish a G1 pass.
- **Reproduced by this review's package with its own legends** (`results.json` → `batch.*.layout_matched`). c_tsv vs r_tsv_p at N=100, complete prompt:
  - en_skewed: 8.5–12.3%, passes 5/6
  - en_uniform: 12.4–14.0%, passes 6/6
  - multilingual: 6.8–10.0%
  - Payload: 9.8–16.2% on the English corpora, 8.0–12.1% multilingual.
- **Conclusion.** G1 for Sylang-style variants is **undetermined** and sensitive to how hard each side is optimized. Run it against a pre-specified, equally searched readable set, with code-vs-word pairs at matched layout and a matched legend budget, and decide on G2 accuracy.

#### 13.3.3 cdsl is not the strongest readable baseline — P1, S, [V]

| Readable variant (strict decoder) | Payload saving vs cdsl |
|---|---|
| Feature-prefix cdsl (`past progressive not help("A","B")`) | 2.0–6.4% |
| r_tsv_p (TAB fields, prefix features, Polish `if`, bare labels), English corpora | 8.2–16.5% |
| r_tsv_p, multilingual | 5.5–8.9% |
| vdsl_nq, eng_bare (English corpora) | 7–12% |

- **r_tsv_p on the complete prompt** depends on N:
  - N = 100: +3.5% to +11.0%.
  - N = 200: +5.4% to +13.4%.
  - N = 10: −21% to −27%; N = 1: −52% to −60%.
  - Most of the small-N loss is legend verbosity: a 77-token legend instead of 143 gives −2.2% to +2.9% at N = 10.
- **JSON quoting** costs about 0.65 tokens per label, and 0 on Gemma 3.
- **Rule.** Define "best readable" per tokenizer and per N as the minimum over the full readable set.

#### 13.3.4 Label-length sensitivity — P2, M, [V]
- **The corpus.** 30 labels averaging 2.3 words (range 1–4).
- **M-repo against readable formats** (per-record ratio, range over 7 tokenizers):
  - vs best package-readable format (cdsl): 1.43–1.79 (k = 1) → 1.30–1.55 (k = 3) → 1.10–1.19 (k = 16).
  - vs concise English: 1.25–1.44 → 1.07–1.12.
- **m_opt vs cdsl** stays 0.95–1.02 at every k.
- **Fused-code vs best of all formats:** 0.86–1.00 (k = 1) → 0.98–1.00 (k = 16).
- **Interpretation.** M-repo loses at every k. Label length matters only for fused bare-label variants on short-label workloads.
- **Corpus quality.**
  - About 49–51% of generated predicates are type-implausible ('a cardboard box sees Ms. Okafor').
  - Make plausibility a controlled stratum rather than removing it, because implausible facts guard against answers drawn from prior knowledge.
  - The skew weights are unsourced.

#### 13.3.5 Terminator-additive cost law — P1, S, [V]
- **Law.** tokens(R1\n…\nRn) = Σ_{i<n} tokens(R_i + "\n") + tokens(R_n).
  - 0 violations in 168 corpus × format × tokenizer cells.
  - Requires that no record starts with whitespace; for o200k, Llama 4 and Tekken also that none starts with `/`.
- **Naive model error.** "Standalone + newline" overcounts per 200 records by:
  - 199 for M, JSON (Qwen3, cl100k, Llama 3) and English;
  - 171 for JSON on o200k, Llama 4 and Tekken;
  - 68–69 for cdsl;
  - 0 on Gemma 3.
- **Bias.** About 3 points: cdsl vs M-repo is −30.0% standalone but −27.0% batched (Qwen3).
- **Consequences.**
  - §6.2 counts joined blocks and is correct. §6.1 is per-document and should not be read as per-record batch cost.
  - Appending `.` to each record costs +5 tokens per 200 records on the regex BPEs, +13 on Tekken and +132 on Gemma 3.
  - Cache breakpoints placed after the terminator miss 0/199; placed before the newline, 199/199.

#### 13.3.6 Ratio summaries — P1, S, [V]+[A]
- **Arithmetic macro is direction-dependent.** By AM ≥ HM, mean(a/b) ≥ 1/mean(b/a).
  - M/English: 0.8501 vs 0.8474.
  - M/JSON: 0.4013 vs 0.3881 (1.3 points).
- **Geometric mean is exactly reciprocal:** 0.8487 and 0.3935.
- **Micro remains the billing estimand,** but report its sensitivity alongside it:
  - literal-max-length carries 26.2% of the English denominator;
  - leave-one-out micro (M/English) ranges 0.8453–0.8871, so the 11.66% headline is *diluted* by that fixture (15.5% without it);
  - non-literal fixtures give 0.825, literal fixtures 0.936;
  - the holdout summary is computed over n = 2.
- **Bootstrap.** Cluster bootstrap over 19 families gives micro [0.832, 0.938] and geo [0.829, 0.881]. On a designed smoke suite these intervals have no sampling meaning (D8). Build the machinery now; apply it to the plan 0.7 corpora.

#### 13.3.7 Cost per correct answer in closed form — P1, S, [V]+[A]
- **Model.** Attempts are independent; retries happen only on parse failure (probability f); accuracy given a parse is a; costs are C_s on success and C_f on failure.
- **Derivation.**
  - E[cost] = (1 − f^{R+1})·[C_s + C_f·f/(1 − f)].
  - P(correct) = (1 − f^{R+1})·a.
  - Cost per correct = (C_s + C_f·f/(1 − f))/a. **This does not depend on the retry cap R;** R changes only coverage, 1 − f^{R+1}.
- **Checks.** Monte Carlo agrees with the formula's 1,248.4. Correlated, item-level failures do make R matter (+1.7% from R = 0 to R = 3).
- **M-repo vs cdsl (generation, N = 100, ρ = 5).**
  - Cost ratio 1.31–1.56, flat over ρ = 4–8.
  - It needs an accuracy ratio ≥ 1.31–1.56 to tie and ≥ 1.46–1.73 to pass G2.
  - It cannot tie at all once cdsl accuracy exceeds 0.64–0.76.
  - This is an **offline kill condition**: no paid M arm is needed.
- **c_tsv vs r_tsv_p.** Cost ratio 0.855–0.900, so G2 needs an accuracy ratio ≥ 0.950–1.000 (parity on Gemma 3).
- **Gate consistency.** If a 2-point accuracy loss is expected at a_X = 0.9, G2 needs a token saving s_min = 1 − 0.9·(1 − δ/a_X) = 12.0%. G1's 10% is then necessary but not sufficient.
- **The cost endpoint is underpowered.** A one-sided 95% UCB < 0.90 with 200 calls passes a format that is truly 12% cheaper only about 40% of the time.

#### 13.3.8 Cache-aware cost model — P1, S, [V]+[A]
- **Per-question input cost.** With a cacheable prefix p (through the record block) reused over Q questions, and suffix s:
  - if p ≥ minlen: [1.25p + (Q − 1)·0.1p + Q·s]/Q = s + p·(0.1 + 1.15/Q);
  - otherwise: p + s.
- **What caching does to format savings.** Input-side savings keep 21.5% of their uncached value at Q = 10. 10% is only the limit as Q grows.
- **Threshold windows.** Without padding, the minimum-length threshold creates narrow windows of N where M-repo is cheaper per question than cdsl (Q = 10):

  | minlen | Window of N |
  |---|---|
  | 512 | 18–24 |
  | 1,024 | 38–57 (N=48: 297 vs 916) |
  | 2,048 | 82–113 |
  | 4,096 | 175–200 |

  The windows vanish under a pad-to-minlen policy and never occur at Q = 1. So near the threshold, caching policy decides cost, not format.
- **Smaller corrections.**
  - The first-draft §6.3 "legend cached" column also cached the 19-token question/assistant suffix (M: 2,391). The corrected prefix-only column gives 2,408, under 1% different.
  - The minimum cacheable length is 512–4,096 tokens depending on the model (512 on the newest Anthropic models), per the official prompt-caching page.

#### 13.3.9 Billed tokens vs forward steps under grammar fast-forward — P1, S, [V]
Engines that fast-forward forced bytes do not sample keys and punctuation. Measured on en_skewed, 7 tokenizers:

| Ratio | Billed output tokens | Sampled steps (llguidance) | Ideal decision steps |
|---|---|---|---|
| json / m_body | 2.54–3.03 | 0.99–1.23 | 1.00 |
| json / cdsl | 4.10–4.26 | 1.31–1.65 | 1.28–1.29 |
| dsl / m_body | 2.26–2.68 | 1.00–1.36 | 1.00–1.06 |
| m_body / cdsl | 1.38–1.62 | 1.32–1.35 | 1.28–1.29 |

- **Forced share of tokens:** json 71–73%, m 30–40%, cdsl 10–13%.
- **No tokenization disturbance:** 0 of 370,121 fast-forwarded tokens ended off a canonical boundary.
- **Consequence.** Billing (G1/G2) and self-hosted latency (Phase 3) rank formats differently. This qualifies §6.3 item 5 (decode latency roughly linear in output tokens).
- **Remaining M gap.** M's step disadvantage against cdsl comes from explicit defaults.
- **Unmeasured.** The mapping from steps to latency [A].

#### 13.3.10 Phase 2 non-inferiority sizing: margin mismatch — P0, S, [V]
- **Formula.** n = (z_{0.95} + z_{0.90})²·ψ/δ². The first-draft plan's 950 pairs is correct for δ = 3 points, but its G2 and review E2 used δ = 2. The plan now uses one margin (§5.2).

| Assumption (ψ = discordance) | δ = 3 pt | δ = 2 pt |
|---|---|---|
| ψ = 0.10, true difference 0 | 952 | 2,141 |
| Holm worst step (α/16) | 1,792 | 4,032 |
| ψ = 0.18 (independent errors at 90%) | 1,713 | 3,854 |
| True difference −1 point | 2,139 | 8,555 |

- **Power at the planned n.** At n = 1,000 and δ = 2: 0.639 (analytic); exact enumeration gives 0.640 (Wald) and 0.631 (Tango). Under the worst Holm step it is about 0.23.
- **Choice of test.** McNemar tests equality, not a non-inferiority null. The Wald NI test is slightly liberal (exact type I 0.051–0.053); Tango's score test gives 0.049–0.050.
- **Blinded internal pilot.** Re-estimating ψ blind after 300 pairs, with n capped at 300–4,000, holds type I at 0.050–0.053 and power at 0.88–0.89. Mean n is 572 / 952 / 1,522 for ψ = 0.06 / 0.10 / 0.16. A fixed n = 952 has power 0.74 if ψ = 0.16.

#### 13.3.11 G2 is a conjunctive (intersection-union) test — P0, S, [V]
- **What a conjunction costs.** Requiring NI in every family × tier cell against every comparator is an intersection-union test. Its power is about the product of the cell powers, and it needs no Holm.
- **Measured at n = 1,000 per cell** (ψ = 0.10, true difference 0; comparators equicorrelated at ρ = 0.5, which is exact for exchangeable arms):

  | Design | δ = 2 pt | δ = 3 pt |
  |---|---|---|
  | 4 cells × 1 comparator | 0.167 | 0.693 |
  | 4 cells × 3 comparators | 0.024 | 0.420 |
  | 4 × 3, with Holm added | 0.009 | 0.343 |

- **Per-cell sizing.** 80% / 90% overall power needs 3,369 / 3,958 pairs per cell at 2 points, and 1,498 / 1,759 at 3 points.
- **Pooled alternative.** A pooled primary estimand clustered by prompt content, an IUT over comparators at full α, and a fixed-sequence order (reading NI → reading cost → generation NI → generation cost). With the same 4,000 pairs it reaches 0.975 / 0.900 / 0.799 at design effect 1.0 / 1.4 / 1.8.
- **Consistency rule.** Make it family-level, not per-cell: strict per-cell consistency holds only 0.79 / 0.63 / 0.51 of the time.
- **Salvage comparison.** The same problem applies to the readable-compact vs JSON comparison.

#### 13.3.12 Within-call clustering — P0, S, [V]
- **Design effect.** With m = 5 questions per call, DE = 1 + (m − 1)·ICC, confirmed by simulation (1.00 / 1.20 / 1.40 / 1.80 / 2.20).
- **Naive test inflation.** True type I error at the NI boundary = 1 − Φ(1.645/√DE): 0.067 / 0.082 / 0.110 / 0.134 at ICC 0.05 / 0.1 / 0.2 / 0.3.
- **Cluster-robust fix.** Var(d̂) = K/(K − 1)·Σ_c (D_c − m_c·d̂)²/n², with D_c = n10_c − n01_c per call and t_{K−1} critical values. Type I falls to 0.053–0.056, and the DE-corrected n restores power to 0.90.
- **Generation endpoint.**
  - 600 iid records give an upper Wilson half-width of 2.05 points; 630 is needed for ±2.
  - With 20 records per call, 30/600 widens from [3.5%, 7.1%] to [3.1%, 8.0%] at ICC 0.05 and [2.1%, 11.6%] at ICC 0.3. All-or-nothing block failures give [1.2%, 19.1%].
  - Target n = 630·(1 + 19·ICC): 869 / 1,228 / 1,827 at ICC 0.02 / 0.05 / 0.1.
  - Clustering does not arise at generation N = 1.

#### 13.3.13 Answer priors can reverse the G2 verdict — P0, S, [V]+[A]
- **Model [A].** Under a read-or-guess model, observed accuracy = r + (1 − r)·g, where g is the fallback hit rate.
- **Skewed priors in today's sources.**
  - Gold majority-class rates: .875 (polarity), .725 (tense), .80 (aspect), .80 (evidence).
  - Skewed corpus: .802 / .514 / .706 / .521.
  - Copy-literal answers role questions at 0.993.

| Scenario (n = 1,000, δ = 2 pt) | Gold-like shares | Skewed corpus | Balanced |
|---|---|---|---|
| A: M reads 6 pt worse, default-biased fallback | diff +0.64 pt, P(pass) 0.79 | −1.07 pt, 0.15 | −3.90 pt, 0.000 |
| C: equal reading, baseline arm biased | −3.24 pt, 0.001 | −2.02 pt, 0.03 | 0.00 pt, 0.43–0.44 |

- **Why balance works.** Under balance, mean g = b/k + (1 − b)/k = 1/k for any bias b, and accuracy differences compress by exactly mean(1 − 1/k) = 0.65 for this question mix (0.33 under gold-like shares).
- **Estimand rule.** Post-stratifying the estimator on answer or feature weights brings the confound back (Scenario A: P(pass) 0.11–0.67). Workload-weight slots only.

#### 13.3.14 Item design, minimal pairs and duplicate runs — P1, M, [V]
- **Sampling design.**
  - At N ≈ 1,000, uniform random sampling is within 1–2% of an orthogonal design's contrast variance. Covering arrays matter only below about 150 items.
  - The skewed sampler costs 1.28–1.59× variance on rare-level contrasts; it is slightly better on common-level ones (e.g. past vs present 0.82).
- **Slot coverage.** The corpus generator puts 62.5% of predicates at the top level and none at depth 3. Stratify slots explicitly down to depth 3–4.
- **Dilution.** A format 6 points worse only on conditional targets shows up as 5.1 / 2.3 / 1.8 points under uniform / corpus / gold slot mixes. At n = 1,000 a 1.8-point deficit passes the 2-point test only about 4% of the time.
- **Minimal pairs.**
  - Within-format feature contrasts: variance ratio 1.19–1.73 (16–42% fewer items).
  - Cross-format difference-in-differences: ratio 1.04–1.32, about 0–24%, so do not cut the item budget on this basis.
  - Pair members share effects, so the gate test must cluster by pair; a naive item-level test has type I 0.058–0.093.
  - Token edit distance per feature differs by format and is nearly constant within each format (M exactly 1; json_short 3.4–5.0), so it cannot be a regression covariate across formats.
- **Duplicate runs.** With one run per item and arm, the paired variance already contains run-to-run noise. The identity ψ_between − (ψ_rr,A + ψ_rr,B)/2 = E[(p_A − p_B)²] isolates format-induced disagreement.
  - Do not pool duplicates as extra rows.
  - 100 duplicates per cell are too imprecise (4/100: [1.6%, 9.8%]); 2,000 pooled are tight (88/2,000: [3.6%, 5.4%]).
  - Runs per item k: default k = 1. The two verifiers disagree on whether cached reruns can make k = 2–3 better when format heterogeneity is small; choose from pilot estimates of Var(p_A − p_B) and the within-item variance.

#### 13.3.15 Silent corruption in M (quantifying D6) — P2, S, [V]
- **Swap test.** Swapping two enum values within one M predicate, over all 216 combinations, gives a valid but different tree in **48/2,112 swaps (2.3%)**. All 48 are relation↔aspect swaps within {s, c}: 2 ordered pairs × 24 combinations of the other fields. JSON, DSL and Prime: 0/2,160 each.
- **Free fix.** Field-unique aspect codes (m/g/k) give 0/2,160 at 0 token cost on 7/7 tokenizers, both on gold and on all 216 combinations. It breaks M0.1 spellings: 17 tests fail.
- **Random substitution.** M 3.6% vs 0.8–1.0% elsewhere with labels excluded; the aspect fix only reaches 3.2%. With labels included, JSON is 4.7% and M 4.6%.
- **English.** Deleting `not` gives a valid positive template in 21/27 negative templates. A polarity substitution is equally undetectable in every format.

#### 13.3.16 The vocabulary-extension floor — P2, S, [V]
- **Bound.** Holding literal tokenization fixed, floor = Σ tok(label) + R, where R = 2P + 1 structure runs. The floor is identical for every quote-delimited format: 2,031 Qwen3 tokens on en_skewed for six formats. The tokenizer track therefore cannot favour Sylang.
- **What layouts without new vocabulary reach.** c_tsv already gets 69–87% of M's reduction to the floor and is 1.16–3.28 tokens per predicate above it.
- **Atomic codes make M longer.**
  - Idealized position-aware atomicity: +17.4% to +17.8% (Gemma 3 +5.1%).
  - Real added-token semantics, which also split labels: +94% to +98% (English), +16% to +25% (multilingual).
- **Bound is not universal.** Layouts that put a space before each label can beat it by up to about 0.8 tokens per predicate.

#### 13.3.17 Semantics needed only for meaning-level questions — P1, S, [A]+[V]
These matter for R1 inferential keys and for prose→tree generation. They do not matter for reconstruction, R2 or transcoding, where structural equality is correct under any semantics.

- **`completed` vs `perfect`.**
  - Only the English arm defines `completed`, and it renders it as the perfect. Other legends say only `c=completed` or nothing, and batch.py:31 ("perfect means completed") contradicts experimental-grammar.md:240.
  - This covers 72/216 combinations (15.0% weighted). An interval checker finds key divergence in 56/216.
  - Fix: rename to `perfect` with a one-line gloss. It costs 0 tokens on 6/7 tokenizers, +1 on Tekken in json/dsl/prime/json_array/json_short/m_words. 13 files change; 44/44 tests pass after rebuilding fixtures.
- **English surface hazards.**
  - Tier-A flags cover 136/216 surfaces (38.4% weighted; 9/33 gold fixtures): stative progressive 48, perfect 72, present-simple `help` habitual 8, direct + future 18.
  - Use them as strata. The "32 clean" subset has no completed, unspecified or reported items, so it cannot be the primary set.
- **No stated denotation.** This is mostly a deliberate, documented silence (grammar:241-245, 285-288). The genuine inconsistencies are:
  - evaluation.md:76 says "exact semantic matches" for a structural scorer;
  - batch.py:31;
  - the English legend's "no pragmatic inference" does not exclude lexical readings.
  - The proposed formula with an existentially closed reference time is defective: it makes past-simple ≡ present-perfect for eventive relations. The reference time must be referential.
  - Either give every arm one Meaning paragraph (86 words = 109–113 tokens) or restrict R1 to field-value questions.
- **Conditionals.**
  - Over distinct atoms, 120/120 conditionals have a structurally distinct equivalent under material or strict-S5 semantics; under Stalnaker, 0/126 merge.
  - The doc example i(¬A, A) is degenerate; replace it.
  - Author G1 prose only as "If A, then B".
- **Evidence inside `if`.** 3/5 gold conditionals carry non-unspecified evidence inside a branch, and its local vs projective reading is unstated.
  - State a local-scope rule and reconcile it with any not-at-issue treatment of evidence.
  - Wrapper nodes (json +10 to +13, M +2, English +6 tokens) would be notation expansion; keep them off the critical path (plan §4, label-length strata).

---

### 13.4 Corrections to this review

All corrections below have been applied in place in §§1–12 of this document; this table is the errata log.

| Location | Statement | Correction | Basis |
|---|---|---|---|
| §1 item 1; §4.5 D1; §6.1 | "41% / 55% more" (35–56% / 44–72%) | This keeps M's per-record header but compares against header-free baselines, on the fixture suite that D8 rejects. Of the 366-token Qwen3 gap to cdsl, 132 (36%) is header, 134 (37%) is default omission and 100 (27%) is notation. Header-free M on fixtures: +23% vs concise English, +35% vs cdsl (Qwen3; 19–34% and 27–47% across 7 tokenizers). On the corpus: 19–34% and 33–58%. Under M's no-defaults rule: 3–16% vs explicit English and 9–31% vs cdsl_full. Report both. | 13.3.4; headline decomposition [V] |
| §1 item 3; §6.1 "Optimized M ≈ compact word DSL"; §11.1 "only tie" | ±1.4%, ties on every tokenizer | Holds only for fixture payload (0.987–1.014). Complete prompt at N=100: 0.978–1.030 (skewed), 0.938–1.004 (uniform), 0.987–1.034 (multilingual). Generation blocks: 0.919–1.017. At matched layout, letter codes save 8–16% of payload. | 13.3.2 |
| §11.4 "Tokenizer" | "None of the seven measured behaves this way" | At a fixed layout, word-based fields cost more than codes on 7/7 tokenizers (c_tsv vs r_tsv_p payload 9.8–14.7%, en_skewed). | 13.3.2 |
| §4.5 D2; §6.2 table | Break-even vs repo JSON "N ≈ 2.8–4.0" | Linear model: 2.83–4.08 (Gemma 3 multilingual 4.08). Actual complete prompts cross at N=2 on 5 tokenizers and N=3 on Gemma 3, so M overtakes earlier than stated. | errata [V] |
| §5.2 | "o200k_base is identical here" | o200k gives 20 tokens, the others 21. | errata [V] |
| §6.2 (multilingual paragraph) | Qwen3 "about 20–30% more per record than o200k" | 1.08 (json_repo) to 1.38 (natural); 7 of 10 formats fall in 1.21–1.33. | errata [V]; results.json |
| §11.2 E1 | "plus cl100k" | Unmeasured for complete prompts: run.py:208 excludes cl100k from batch_tables. | errata [V] |
| §6.3 "Read, legend cached" | Cached-legend figures | Counterfactual: all prefixes are below the minimum cacheable length, and the first draft also cached the 19-token suffix. Replaced by a prefix-only, flagged column. | 13.3.8; run.py |
| §6.3 item 5 | Decode latency roughly linear in output tokens | Holds for billing. Under grammar fast-forward, JSON needs 0.99–1.23× m_body's steps while billing 2.54–3.03× the tokens. | 13.3.9 |
| §6.3 item 3 | "Would need a higher success rate" | Exact condition: a_M(1−f_M)/(a_X(1−f_X)) ≥ C_M/C_X. M-repo needs ≥ 1.31–1.56× cdsl's accuracy to tie, independent of the retry cap. | 13.3.7 |
| §5.4 | Escape where NFC(s) ≠ s | Leaves 98 unparseable control+mark cases (all 5 formats). Verify NFC-invariance on the encoded text. | 13.1.1 |
| §5.5 | Disabling special-token parsing fixes reserved strings | Not on Gemma 3 via raw SentencePiece (USER_DEFINED 105/106), nor for HF added tokens flagged special=False. | 13.1.4 |
| §6.1 | Payload ratios | Correct as single-document totals. Not per-record batch costs: standalone ratios are biased by about 3 points (terminator law). | 13.3.5 |
| §2 (alternative representations) | "Each has a strict decoder" | Fail-closed on structure, not canonical-only. Only m_opt is whitespace-strict. | 13.1.16 |
| §12 ledger | Showcase, D3–D5, D9 marked [M] without a producing script | Now produced by run.py (`results.json` → `probes.readme_showcase`, `defects`); label kept as [M]. | 13.1.16 |
| §2 (Qwen3 rebuild) | Qwen3 reconstruction "validated 330/330" | Validated on counts only. Count equality is a weak equivalence certificate; converted cl100k matches 330/330 counts but 320/330 IDs. Re-certify on IDs over a stress corpus. | 13.2.3 |
