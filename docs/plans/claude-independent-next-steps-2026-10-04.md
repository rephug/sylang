# Independent next-steps plan for Sylang

**Date:** 2026-10-04 · **Author:** Claude (independent reviewer)
**Based on:** [independent feasibility review](../research/claude-independent-feasibility-review-2026-10-04.md) of commit `0f7c5700c9a314b9f0b635bb4773c24caf785bf2`

This plan is an alternative to the gates in [decisions.md](../decisions.md) and the priority table in the [2026-10-02 review](../research/technical-review-2026-10-02.md). Both are kept unchanged.

Every experiment below is **proposed and not yet run**, unless it says "done in review". Nothing here authorizes paid inference, model downloads, training or deployment. Each item that needs one is marked **APPROVAL**.

---

## 1. Principles

1. **Falsify before building.** Run the cheapest experiment that could kill the premise first. The review's offline measurements already do most of this for the token-efficiency premise.
2. **Compare against the strongest simple baseline**, not the weakest. Every arm must beat minified or positional JSON, a compact word DSL, concise English and (for reading) natural English, on the same tokenizer and the same facts.
3. **Count everything:**
   - complete chat templates
   - legends and worked examples
   - the per-record payload
   - model output
   - validation, retries and cache writes and reads
   - the natural-language → structure step, when real inputs are prose
4. **Separate reading from generation.** A reading answer must be format-neutral. Never score a format by asking the model to copy it.
5. **Freeze before measuring.** Legends, examples, prompts, decoding parameters and holdout items are frozen and hashed before any holdout run. The development split is the only place for tuning.
6. **Stop early.** Every phase ends at a gate with numeric thresholds and a named outcome for both pass and fail.

## 2. Phase overview

| Phase | Purpose | Cost | Effort (1 engineer) | Needs | Gate |
|---|---|---|---|---|---|
| **0** | Fix the evaluation; add strong baselines; freeze corpora | $0 | 3–5 days | Nothing | G0: harness acceptance tests pass |
| **1** | Offline token falsification on billing tokenizers | $0 | 2–3 days | **APPROVAL** to use a free Claude `count_tokens` key; Hugging Face access for Qwen3.5 / DeepSeek | **G1:** does any Sylang-specific notation save ≥10%? |
| **2** | Pre-registered reading + generation pilot | Hard cap **$150** (2a); optional long-context 2b capped at **$250** | 6–9 days | **APPROVAL** for paid inference; independent holdout author | **G2:** accuracy non-inferiority + cost per correct answer |
| **3** | Serving latency and long-context on an open model | GPU hours (about $50–150 rented) | 4–6 days | **APPROVAL** for model-weight download and GPU | **G3:** end-to-end quality-adjusted gain ≥10% |
| **4** | Adaptation (LoRA on an unchanged tokenizer), only if G3 passes | About $100–500 GPU, plus curation | 2–3 weeks | **APPROVAL** for training | **G4:** trained gain exceeds the trained readable control |
| **S** | Salvage and close-out (runs whenever a gate fails) | $0 | 2–4 days | Owner decision on archiving historical claims | — |

**Expected path, given the review's evidence:**
0 → 1 (G1 fails for M) → 2a (to settle accuracy and the salvage question) → S.
Phases 3–4 are listed so that the stopping rules are explicit, not because they are expected to run.

---

## 3. Phase 0: fix and harden the evaluation ($0, 3–5 days)

**Dependencies:** none. **Owner:** repository maintainer or contributor.

| ID | Deliverable | Acceptance test |
|---|---|---|
| 0.1 | **Scorer robustness** (review defects D4, D5): parse each answer line's envelope strictly; score a malformed `answer` (duplicate key, NaN, excessive nesting, non-JSON) as schema-invalid *for that task only*, and keep the raw text. | New unit tests: duplicate key, NaN and a 100k-deep array are each scored invalid while the other answers still score. No traceback. Existing 44 tests still pass. |
| 0.2 | **Format-neutral task design** (D3): replace "reconstruct JSON" with (a) reading questions answered in a closed vocabulary (`yes` / `no` / `unknown`, line numbers, field values) and (b) separate generation tasks (natural language or JSON → format X). | Test: no reading task's expected answer equals its own payload. JSON is no longer a copy task. |
| 0.3 | **Strong baselines as first-class formats:** concise controlled English, compact word DSL (defaults omitted), positional JSON array, minified JSON with short keys, and CSV for flat batches. Each has a strict parser. Port from `benchmarks/claude-review-2026-10-04/variants.py`. | Property tests: ≥2,000 seeded random trees plus all 216 enum combinations decode back exactly for every format. Canonical re-encoding is stable. |
| 0.4 | **Batch- and template-aware accounting:** cost curves for N ∈ {1, 5, 10, 25, 100, 200} records per prompt. Each family's published chat template rendered in one encode. Legend counted separately as cacheable/uncacheable against the provider's minimum cacheable length. | Unit test: the template-wrapped count equals one encode of the full string. Break-even N reported for every pair of formats. |
| 0.5 | **Safe serving-path measurement** (D11): count tokens with special-token parsing disabled for content, alongside the current diagnostic path. | Reserved-string fixtures produce zero control IDs on the safe path for every tokenizer. |
| 0.6 | **Unicode policy prototype:** a model-facing escaper that emits `\uXXXX` where `NFC(s) ≠ s` and for format/control characters, behind a flag. The semantic layer stays code-point exact. | All Unicode probes in `results.json` round-trip exactly through the Qwen3 (NFC) tokenizer when escaped. Token overhead is reported per fixture. |
| 0.7 | **Realistic corpora:** a seeded generator with realistic, multilingual and long labels; a skewed and a uniform feature distribution; conditional depth up to 4. Separate dev / holdout split. | Manifest with SHA-256 per split. The holdout hash is published, but its content is kept outside the repository until Phase 2 runs. |
| 0.8 | **Correct the baseline-results narrative (owner decision):** add a note that the 11.6% figure is against a verbose controlled English, and link the review. | Owner approval. No silent edits to historical documents. |

**Gate G0:** all acceptance tests pass, and results are byte-identical across two runs (excluding timing fields).

---

## 4. Phase 1: offline token falsification ($0, 2–3 days)

**Dependencies:** Phase 0 items 0.3, 0.4 and 0.7. **APPROVAL** before using an Anthropic API key for `count_tokens`. It is free and is not inference, but it is an external account action.

| ID | Deliverable | Acceptance test |
|---|---|---|
| 1.1 | **Billing-tokenizer coverage:** the review's seven (Qwen3, o200k, cl100k, Llama 3, Llama 4, Tekken, Gemma 3), plus Qwen3.5 and DeepSeek-V3.x (via the existing pinned Hugging Face fetcher once reachable), plus Claude via `count_tokens` for a current-tokenizer model and a pre-4.7 model. | Every asset is pinned by SHA-256 or by API model ID and date. Qwen3 count equivalence with the committed results is retained as a regression test (330/330). |
| 1.2 | **Complete-prompt and generation token tables** per format, per tokenizer and per corpus, at N ∈ {1, 10, 100}. Includes legend, worked examples (0, 2, 5 shots), template and output. | Generated by one command and committed as JSON with a methodology block. |
| 1.3 | **Cached/uncached cost curves** in input-token equivalents, using the verified Anthropic multipliers (output 5×, cache read 0.1×, 5-minute cache write 1.25×) and, as a sensitivity range, output/input ratios of 4–8×. | The table states the exact multipliers and the date they were read. No dollar figures without a price source. |

### Gate G1: the token gate

**Rule.** The Sylang-specific notation track continues only if M, or any Sylang-specific variant, is **≥10% cheaper** than the best readable compact baseline in **both** complete-prompt (N=100) and generation tokens, on a **majority** of billing tokenizers.

**Status at review time.** The review measured 7 tokenizers. Against the compact word DSL, M uses 44–72% *more* payload tokens and 33–56% more complete-prompt tokens at N=100; an optimized M only ties the DSL (±1.4%). **G1 is expected to fail.** Phase 1 exists to confirm this on Claude, Qwen3.5 and DeepSeek before the owner closes the track.

- **If G1 fails:** stop all notation design (opcode tables, alphabets, Prime/M syntax work). Continue to Phase 2a only to answer the salvage question: is a readable compact format better than minified JSON for this kind of data, in accuracy and cost? If the owner does not want that question answered, go straight to Phase S.
- **If G1 passes:** run Phase 2a with M as the primary arm.

---

## 5. Phase 2: pre-registered reading and generation pilot

**Budget.** 2a is capped at **$150**; optional 2b at **$250**. Effort is 6–9 days.

**Dependencies:** Phase 0 complete; Phase 1 tables; an independent holdout author (not the codec author); a pre-registration document committed before any holdout call. **APPROVAL** for paid inference and for the provider and models used.

### 5.1 Design

**Arms (formats):**
- minified JSON (also run with provider structured outputs as a separate condition)
- compact word DSL
- concise controlled English
- M, using the repository legend
- natural English (reading only; it is not reversible)

**Models:** at least two families × two tiers.
- Example: a small and a large hosted model from two providers.
- Optionally one open model (Phase 3 hardware) for a non-API data point.
- Model IDs and dates are pinned in the pre-registration.

**Tasks:**

| Task | What it does | Output |
|---|---|---|
| R1 reading QA | Polarity, scope (condition vs consequence), tense/aspect, evidence, role identity. Minimal pairs and idiom traps (e.g. "is seeing"). | Closed-vocabulary answers |
| R2 field extraction | Asks for specific fields of specific records | Format-neutral answers, never the input format |
| G1 generation from prose | Natural-language facts → format X, validated by the strict codec | The record in format X |
| G2 transcoding control | JSON → format X | The record in format X; isolates syntax burden from comprehension |

**Exposure:**
- The same instruction skeleton for every arm.
- Legends frozen on the dev split.
- 0-shot and 3-shot conditions with identical examples rendered per format.
- Legend length reported. A "matched legend budget" condition gives every arm the same legend length, padding with neutral text.

**Batching:** N ∈ {1, 20} records per prompt in 2a. N up to 2,000 records (8k–128k tokens) in 2b, for needle-and-aggregation questions.

**Decoding:** temperature 0 where supported, fixed `max_tokens` per task, at most one retry on a parse failure, and the same retry policy for every arm. Hosted models are not perfectly deterministic, so a 10% subsample is run twice to estimate run-to-run variance.

### 5.2 Sample size and statistics

**Per arm, per model:**
- **R1:** 1,000 paired questions. With about 10% discordant pairs and a 3-point non-inferiority margin at α = 0.05 one-sided and 90% power, roughly 950 pairs are needed. A 300-question pilot on the dev split estimates the actual discordance first.
- **G1:** 600 generated records. Enough to estimate a parse-failure rate near 5% to within ±2 points (95% Wilson interval).

**Analysis:**
- Paired McNemar test for accuracy, with a paired bootstrap for cost per correct answer.
- Holm correction across arms × models.
- 95% intervals throughout.

**Report:**
- per-field error taxonomy
- parse failures, schema-invalid output, refusals and timeouts
- retries
- per-script worst-group results for multilingual labels
- tokens taken from provider `usage` fields: input, cache write, cache read, output

### 5.3 Cost estimate for 2a (estimate, not a quote)

| Item | Estimate |
|---|---|
| Reading call | About 1k input tokens (20 records + legend + 5 questions) and about 50 output tokens |
| 1,000 questions | About 200 calls per arm per model |
| Reading scope | 5 arms × 4 model-tier cells: about 4M input and 0.2M output tokens |
| Generation scope | Similar or smaller |
| Total at mid-tier prices (e.g. $2 / $10 per MTok) | Under $50 |
| Total at top-tier prices (e.g. $10 / $50 per MTok) | Under $150 |
| Batch API discount (about 50%) | Halves these where the provider offers it |

2b (long context) is dominated by input: e.g. 128k tokens × 20 calls × 5 arms × 2 models ≈ 26M tokens. Prompt caching of the shared record block across questions should be used, and measured, to keep it under the cap.

### Gate G2: accuracy and cost per correct answer

**M continues** only if, on at least two model families:
- reading accuracy is **non-inferior** (margin 2 points) to the best readable compact arm, **and**
- billed cost per correct answer is **≥10% lower**, for reading and generation separately.

**Salvage outcome (independent of M).** Recommend the readable compact format over minified JSON only if it is non-inferior in accuracy **and** at least 20% cheaper per correct answer. Otherwise recommend JSON with provider structured outputs.

**Fail:** go to Phase S and publish the results, including the negative ones.

---

## 6. Phase 3: serving latency and long context (only if G2 passes)

**Effort:** 4–6 days. **Cost:** about $50–150 of rented GPU time.

**Dependencies:** G2 pass. **APPROVAL** for a model-weight download (an 8B-class open model) and GPU rental.

**Deliverables:**
- vLLM serving with prefix caching on and off.
- Time to first token, decode time, total latency, p50/p95/p99, and throughput.
- Cold and warm caches; batch sizes 1 and 16.
- The winning arms against the best readable baseline.
- Long-context retrieval and aggregation at 8k / 32k / 128k.

**Acceptance:**
- At least 30 repetitions per cell, with hardware, software versions and seeds recorded.
- Latency differences are reported together with accuracy, never alone.

### Gate G3

End-to-end quality-adjusted cost or latency is **≥10% better** than the best readable baseline at non-inferior accuracy. Otherwise go to Phase S.

---

## 7. Phase 4: adaptation (only if G3 passes and a residual gap remains)

**Effort:** 2–3 weeks. **Cost:** about $100–500 GPU, plus curation.

**Dependencies:** G3 pass; a residual accuracy gap that prompting (examples, legends, constrained decoding) cannot close; **APPROVAL** for training.

**Deliverables:**
- **Paired LoRA runs on an unchanged tokenizer.** Teach M versus teach the best readable format, with the same data volume, compute and evaluation. Data comes from validated tree/rendering pairs plus independently authored prose.
- **Leakage controls:** the holdout is never used for training; contamination is checked by n-gram and hash.

**Out of scope even here, unless a separate review approves it:**
- new tokens or vocabulary adaptation
- custom tokenizers
- byte-level models

The review found no tokenizer-level mechanism that would favour them, and they break compatibility with hosted APIs and prefix caches.

### Gate G4

The trained M arm beats the *trained* readable arm by at least 10% in quality-adjusted cost. Beating the untrained baseline is not enough. Otherwise go to Phase S.

---

## 8. Phase S: salvage and close-out ($0, 2–4 days)

**Deliverables:**
- **A final report** with all gate outcomes, including negative results.
- **Codec and validator retained** as a strict, reversible serializer for the bounded schema, with the readable compact DSL as its recommended surface if Phase 2 supports it.
- **The evaluation harness repackaged** as a small format-effects benchmark: manifests, strong baselines, a format-neutral scorer, and cost curves.
- **The semantic minimal-pair probe set** published as a stand-alone diagnostic.
- **Historical documents:** an owner-approved banner on the historical README/docs stating which claims were tested and how they fared. The review measured the README showcase sentence at 12–14 tokens on seven tokenizers, against a claimed ~5–7. Historical text is not deleted.

**Acceptance:** the owner signs off on the archive and banner text. CI runs the retained tests.

---

## 9. Dependencies

```text
Phase 0 ──► Phase 1 ──► G1 ──fail──► (2a salvage question, optional) ──► Phase S
                         │
                         └─pass─► Phase 2 ──► G2 ──fail──► Phase S
                                                │
                                                └─pass─► Phase 3 ──► G3 ──fail──► Phase S
                                                                       │
                                                                       └─pass─► Phase 4 ──► G4 ──► Phase S (report)
```

Approvals block the following:
- 1.1 (Claude `count_tokens` key)
- Phase 2 (paid inference)
- Phase 3 (weights and GPU)
- Phase 4 (training)
- 0.8 and Phase S banners (owner editorial decisions)

---

## 10. Risk register

| Risk | Effect | Mitigation |
|---|---|---|
| Legends or examples tuned on the evaluation items | Inflated accuracy for the tuned arm | Tune on dev only; freeze and hash before holdout |
| Holdout contamination (public repository) | Optimistic generalization | Independent author; content kept off-repository until the run; hashes published |
| Same author writes renderers and expected answers | Shared misunderstandings pass round trips | Independent expectations; semantic review of templates (e.g. perfect ≠ completion) |
| Provider tokenizer or model drift | Results stop applying | Pin model IDs and dates; use `usage` fields; re-run Phase 1 on upgrades |
| Hosted-model nondeterminism | Spurious differences | Paired design; duplicate a 10% subsample; report variance |
| Prompt injection via literals or reserved strings | Corrupted runs; security issue | Safe tokenization path; literals marked as data; injection probes reported |
| Cost overrun | Budget breach | Hard caps per phase; batch API; stop on cap |
| Licence and terms (tokenizers, model outputs) | Compliance issues | Record licences and usage policies per asset; no redistribution of model weights |
| Confirmation bias toward the project's thesis | Weak gates passed | Numeric gates fixed in advance; negative results published |

---

## 11. Disagreements with the existing plan

Details are in review §9.3.

| Existing plan | This plan |
|---|---|
| Compare M against the strongest compact baseline *after* comprehension review | Do it **first**. It is free and decisive, and was done in the review: M loses. |
| Priority ends in constrained generation, then LoRA/QLoRA adaptation | Ends at a **stop/salvage** decision. Adaptation only behind G3, and only on an unchanged tokenizer against a trained readable control. |
| Qwen3 as the primary baseline | Several families, including the actual billing tokenizers, with Claude via `count_tokens`. Qwen is the only measured family that applies NFC. |
| Comprehension = reconstruct the JSON tree | Separate format-neutral reading from codec-validated generation. JSON must not be a copy task. |
| Unicode normalization and reserved-token literals as unresolved research questions | Engineering decisions: an escaping policy and a safe serving path, prototyped in Phase 0. |
| One-record prompts for complete-prompt costs | Cost curves over N, with cached/uncached legends and output-side costs. |

## 12. Immediate next step

**Owner decision (no cost):** read review §6 and decide whether the offline evidence already closes the notation-optimization track. If yes, mark G1 as provisionally failed.

**Engineering:** start Phase 0, items 0.1–0.3:
- scorer robustness
- format-neutral tasks
- strong baselines as first-class formats, ported from `benchmarks/claude-review-2026-10-04/`

That is about 2–3 days and needs no approvals.

**In parallel, ask for the Phase 1 and Phase 2 approvals:** a Claude `count_tokens` key, and a $150 inference cap with named providers and models. Phase 2 must not start until the holdout author and the pre-registration are in place.
