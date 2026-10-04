# Independent next-steps plan for Sylang

**Date:** 2026-10-04 · **Author:** Claude (independent reviewer)
**Based on:** [independent feasibility review](../research/claude-independent-feasibility-review-2026-10-04.md) of commit `0f7c5700c9a314b9f0b635bb4773c24caf785bf2`
**Revision:** updated the same day after the review's adversarial verification pass (review §13). The statistics, gates, Phase 2 arms and budget are corrected in place, and the detailed amendments are in §13 of this plan.

This plan is an alternative to the gates in [decisions.md](../decisions.md) and the priority table in the [2026-10-02 review](../research/technical-review-2026-10-02.md). Both are kept unchanged.

Every experiment below is **proposed and not yet run**, unless it says "done in review". Nothing here authorizes paid inference, model downloads, training or deployment. Each item that needs one is marked **APPROVAL**.

---

## 1. Principles

1. **Falsify before building.** Run the cheapest experiment that could kill the premise first. The review's offline measurements already do most of this for the token-efficiency premise.
2. **Compare against the strongest simple baseline**, not the weakest. On the same tokenizer and the same facts, every arm must beat:
   - the best readable format, at minimum the tab word layout `r_tsv_p`, at a matched layout
   - minified or positional JSON, a compact word DSL and concise English (offline)
   - natural English, for reading
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
| **1** | Offline token falsification on billing tokenizers | $0 | 2–3 days | **APPROVAL** to use a free Claude `count_tokens` key; Hugging Face access for Qwen3.5 / DeepSeek | **G1:** does any Sylang-style notation save ≥10% against the same layout in words? |
| **2** | Pre-registered reading + generation pilot | Hard cap **$150** (2a, pooled design, 4 arms); optional long-context 2b capped at **$250** | 6–9 days | **APPROVAL** for paid inference; independent holdout author | **G2:** accuracy non-inferiority + cost per correct answer |
| **3** | Serving latency and long-context on an open model | GPU hours (about $50–150 rented) | 4–6 days | **APPROVAL** for model-weight download and GPU | **G3:** end-to-end quality-adjusted gain ≥10% |
| **4** | Adaptation (LoRA on an unchanged tokenizer), only if G3 passes | About $100–500 GPU, plus curation | 2–3 weeks | **APPROVAL** for training | **G4:** trained gain exceeds the trained readable control |
| **S** | Salvage and close-out (runs whenever a gate fails) | $0 | 2–4 days | Owner decision on archiving historical claims | — |

**Expected path, given the review's evidence:**
0 → 1 (G1 fails for the repository's M; undetermined, knife-edge, for the fused code `c_tsv`) → 2a (decides `c_tsv` vs `r_tsv_p` by accuracy, and answers the salvage question) → S.
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
| 0.5 | **Safe serving-path measurement** (D11): count tokens with special-token parsing disabled for content, alongside the current diagnostic path. | Zero reserved-string *matches* in content on the safe path: by token ID for tiktoken and Hugging Face tokenizers, and by piece string for SentencePiece. On Gemma 3 through raw SentencePiece, `<start_of_turn>`/`<end_of_turn>` are USER_DEFINED pieces that cannot be disabled, so this needs the literal guard (escape `<` and `[` inside literals; review §13.1.4). |
| 0.6 | **Unicode policy prototype:** a model-facing escaper: the greedy NFC-safe escaper, or a whole-literal `ensure_ascii` rule triggered when the *encoded* literal is not NFC-invariant, plus format/control-character escaping, behind a flag and judged on the encoded text T (the naive "escape where NFC(s) ≠ s" rule is insufficient; review §13.1.1). The semantic layer stays code-point exact. | Property test on the *encoded* text: NFC(T) == T and decode(T) == tree in all five formats, including C0-control + combining-mark cases (e.g. `x` + LF + U+0303), under the Hugging Face Rust NFC and Python's. MAX_TEXT_BYTES raised to ≥ 2^19 so a maximal escaped tree still encodes. Token overhead is reported per fixture. |
| 0.7 | **Realistic corpora:** a seeded generator with realistic, multilingual and long labels; a skewed and a uniform feature distribution; conditional depth up to 4. Separate dev / holdout split. | Manifest with SHA-256 per split. The holdout hash is published, but its content is kept outside the repository until Phase 2 runs. |
| 0.8 | **Correct the baseline-results narrative (owner decision):** add a note that the 11.6% figure is against a verbose controlled English, and link the review. | Owner approval. No silent edits to historical documents. |

**Gate G0:** all acceptance tests pass, and results are byte-identical across two runs (excluding timing fields).

---

## 4. Phase 1: offline token falsification ($0, 2–3 days)

**Dependencies:** Phase 0 items 0.3, 0.4 and 0.7. **APPROVAL** before using an Anthropic API key for `count_tokens`. It is free and is not inference, but it is an external account action.

| ID | Deliverable | Acceptance test |
|---|---|---|
| 1.1 | **Billing-tokenizer coverage:** the review's seven (Qwen3, o200k, cl100k, Llama 3, Llama 4, Tekken, Gemma 3), plus Qwen3.5 and DeepSeek-V3.x (via the existing pinned Hugging Face fetcher once reachable), plus Claude via `count_tokens` for a current-tokenizer model and a pre-4.7 model. | Every asset is pinned by SHA-256 or by API model ID and date. Qwen3 count equivalence with the committed results is retained as a regression test (330/330). |
| 1.2 | **Complete-prompt and generation token tables** per format, per tokenizer and per corpus, at N ∈ {1, 10, 100}. Includes legend, worked examples (0, 3 and 5 shots; 0 and 3 match Phase 2), template and output. | Generated by one command and committed as JSON with a methodology block. |
| 1.3 | **Cached/uncached cost curves** in input-token equivalents, using the verified Anthropic multipliers (output 5×, cache read 0.1×, 5-minute cache write 1.25×) and, as a sensitivity range, output/input ratios of 4–8×. | The table states the exact multipliers and the date they were read. No dollar figures without a price source. |

### Gate G1: the token gate

**Rule.** The Sylang-style notation track continues only if some Sylang-style code is **≥10% cheaper** than the same layout written in words in **both** complete-prompt (N=100) and generation tokens, on a **majority** of billing tokenizers.
- Code and words must differ only in the relation/feature field.
- The comparison must use a pre-registered readable set searched as hard as the code candidates, with matched legend budgets.
- Report it per label-length stratum and state which stratum is the target workload.
- **Threshold consistency:** the token saving consistent with G2's 10% cost-per-correct gate, when an accuracy loss of δ is tolerated at baseline accuracy a_X, is s_min = 1 − 0.9·(1 − δ/a_X). At a_X ≈ 0.9 that is about 13% for δ = 3 points (the recommended margin) and 12% for δ = 2, not 10% (review §13.3.7).

**Status at review time (6 chat-templated tokenizers; cl100k not batch-tested; reproduced by `run.py`):**
- **Failed** for the repository's M: 30–66% *more* complete-prompt tokens than the best readable reversible format at N=100 (41–66% on the default-skewed English corpus).
- **Failed** for optimized M: it only ties the compact DSL (−6% to +3%).
- **Undetermined and knife-edge** for the fused code `c_tsv` against the tab word layout `r_tsv_p`:
  - English labels, default-skewed: 8.5–12.3% cheaper (passes on 5/6)
  - uniform features: 12.4–14.0% (6/6)
  - multilingual labels: 6.8–10.0% (≥10% on only 1/6 tokenizers, so it fails the majority rule)
- Phase 1 confirms these on Claude, Qwen3.5 and DeepSeek.

**Outcomes:**
- **If G1 fails for every Sylang-style code:** stop all notation design (opcode tables, alphabets, Prime/M syntax work). Continue to Phase 2a only to answer the salvage question: which readable compact format (tab words, compact DSL, minified JSON) to recommend? If the owner does not want that answered, go straight to Phase S.
- **If G1 passes or is knife-edge for `c_tsv`:** run Phase 2a with `c_tsv` vs `r_tsv_p` as the primary pair. The repository's M is not run: it is killed offline by the cost-per-correct pre-gate (item 1.4 in §13).

---

## 5. Phase 2: pre-registered reading and generation pilot

**Budget.** 2a is capped at **$150**; optional 2b at **$250**. Effort is 6–9 days.

**Dependencies:** Phase 0 complete; Phase 1 tables; an independent holdout author (not the codec author); a pre-registration document committed before any holdout call. **APPROVAL** for paid inference and for the provider and models used.

### 5.1 Design

**Arms (formats):**
- **`c_tsv`**: tab layout, fused per-fact code. This is the Sylang candidate.
- **`r_tsv_p`**: tab layout, words. This is the matched readable control, and the primary comparator.
- minified JSON: the salvage reference. A constrained (structured-output) condition is allowed only if *every* arm is also constrained on the same engine.
- concise controlled English: a reference.
- natural English: reading only, as it is not reversible. Optional.
- The repository's M is **not** run: offline algebra already shows it would need 1.31–1.56× the readable format's accuracy.

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

### 5.2 Sample size and statistics (corrected after verification)

**One margin everywhere.** Pre-register a single non-inferiority margin δ and use it in G2, review E2 and here. The recommended primary is **δ = 3 accuracy points**.
- Passing the cost-per-correct gate already needs near-parity accuracy for `c_tsv` (review §6.3), so the cost endpoint is the binding test.
- δ = 2 needs 2,141 pairs at a discordance of ψ = 0.10 before clustering; 1,000 pairs give only 64% power.

**Pairs required** (ψ = 0.10, 90% power, one-sided α = 0.05, true difference 0; n = (z₀.₉₅ + z₀.₉₀)²·ψ·DE/δ², rounded up), scaled by the design effect DE = 1 + (m − 1)·ICC for m = 5 questions per call:

| ICC | DE | δ = 3 pt | δ = 2 pt |
|---|---|---|---|
| 0 | 1.0 | 952 | 2,141 |
| 0.05 | 1.2 | 1,142 | 2,570 |
| 0.10 | 1.4 | 1,333 | 2,998 |
| 0.20 | 1.8 | 1,713 | 3,854 |
| 0.30 | 2.2 | 2,094 | 4,711 |

**Design rules:**
- **Pooled primary estimand.** One estimand across model cells, clustered by prompt content. Requiring non-inferiority in every family × tier cell at n = 1,000 per cell would pass an exactly equivalent format only 2–17% of the time at δ = 2 and 34–69% at δ = 3 (review §13.3.11).
- **Consistency** is a family-level rule: each family's point estimate > −δ.
- **Blinded internal pilot.** After 300 holdout pairs, re-estimate ψ and the ICC without arm labels and recompute n (cap 300–4,000). The rule is pre-registered; simulated type I is 0.050–0.053 and power 0.88–0.89.
- **Generation.** Run at N = 1 record per call (no clustering): 630 records per arm *per model cell* give a ±2-point Wilson interval at a ~5% failure rate. At 20 records per call, use 630·(1 + 19·ICC) records.

**Analysis:**
- **Tango's score test** for paired non-inferiority. McNemar tests equality, not a non-inferiority null.
- **Call-level cluster-robust variance** (t with K − 1 df) or a call-resampling bootstrap; additionally cluster by pair if minimal pairs are used.
- **Intersection-union test** over the endpoints against the primary comparator `r_tsv_p` at full α, with no Holm. Use a fixed sequence: reading NI → reading cost per correct → generation NI → generation cost per correct. The other readable arms serve the salvage comparison and secondary rankings, with Holm.
- **Answer balance.** In every stratum, each closed-vocabulary answer appears equally often. Skewed answer priors can reverse the verdict (review §13.3.13).
  - State the margin on the reading-rate scale as well.
  - Weight the primary estimand by workload over *slots only*, never over answer or feature values.
- **Cost per correct** gets a log-ratio upper bound with call-level linearization or a paired call bootstrap. It is under-powered at about 200 calls, so size it separately or report it as an estimate.
- **Duplicates.** Analyse exactly one run per item and arm. Re-issue duplicates as whole calls, and estimate test-retest discordance separately.
- **95% intervals throughout.**

**Report:**
- per-field error taxonomy (path-aligned diff)
- parse failures, schema-invalid output, refusals, truncations and timeouts
- retries
- per-script worst-group results for multilingual labels
- hazard strata (stative progressive, perfect, habitual present, direct + future)
- tokens from provider `usage` fields: input, cache write, cache read, output

### 5.3 Cost estimate for 2a (estimate, not a quote; recomputed for the corrected design)

| Item | Estimate |
|---|---|
| Reading | 4 arms × 4,000 pooled questions (1,000 per model cell) at 5 questions per call: about 3,200 calls, about 3.2M input and 0.16M output tokens |
| Generation | 4 arms × 4 cells × 630 records at N = 1: about 10,000 calls, about 3.5M input and 0.3M output tokens |
| Total at mid-tier prices (e.g. $2 / $10 per MTok) | About $20 |
| Total at top-tier prices (e.g. $10 / $50 per MTok) | About $90 |
| Pilot, duplicates and reruns (× about 1.3) | Within the **$150** cap for a mixed small/large design |
| Batch API discount (about 50%) | Lowers these further where offered |

Powering every model cell separately at δ = 2 would need about 3,400–4,000 pairs per cell, i.e. roughly 4× the reading cost. That is why the pooled estimand is the default.

2b (long context) is dominated by input: e.g. 128k tokens × 20 calls × 5 arms × 2 models ≈ 26M tokens. The cacheable prefix must end *after* the record block, so questions over the same block share it. Caching should be used, and measured, to stay under the cap.

### Gate G2: accuracy and cost per correct answer

**The code (`c_tsv`) continues** only if, on the pooled primary estimand with family-level consistency:
- reading accuracy is **non-inferior** to `r_tsv_p` at the pre-registered margin, **and**
- billed cost per correct answer is **≥10% lower**, for reading and generation separately (in the fixed sequence of §5.2).

**Salvage outcome (independent of the code).** Recommend the best readable compact format over minified JSON only if it is non-inferior in accuracy **and** at least 20% cheaper per correct answer. Otherwise recommend minified JSON, with provider structured outputs as a deployment option that this study does not test.
- Compare like with like on constraint: never constrained JSON against unconstrained compact formats. Constrained conditions are tested only with every arm constrained on one self-hosted engine.

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

**Dependencies:** G3 pass, which can only come from the fused code `c_tsv`, since the repository's M is killed offline; a residual accuracy gap that prompting (examples, legends, constrained decoding) cannot close; **APPROVAL** for training.

**Deliverables:**
- **Paired LoRA runs on an unchanged tokenizer.** Teach the surviving Sylang-style code (`c_tsv`) versus teach the best readable format (`r_tsv_p`), with the same data volume, compute and evaluation. Data comes from validated tree/rendering pairs plus independently authored prose.
- **Leakage controls:** the holdout is never used for training; contamination is checked by n-gram and hash.

**Out of scope even here, unless a separate review approves it:**
- new tokens or vocabulary adaptation
- custom tokenizers
- byte-level models

The review found no tokenizer-level mechanism that would favour them, and they break compatibility with hosted APIs and prefix caches.

### Gate G4

The trained code arm beats the *trained* readable arm by at least 10% in quality-adjusted cost. Beating the untrained baseline is not enough. Otherwise go to Phase S.

---

## 8. Phase S: salvage and close-out ($0, 2–4 days)

**Deliverables:**
- **A final report** with all gate outcomes, including negative results.
- **Codec and validator retained** as a strict, reversible serializer for the bounded schema. Its recommended model-facing surface is the readable format that wins the salvage comparison (on token counts today, the tab word layout `r_tsv_p`).
- **The evaluation harness repackaged** as a small format-effects benchmark: manifests, strong baselines, a format-neutral scorer, and cost curves.
- **The semantic minimal-pair probe set** published as a stand-alone diagnostic.
- **Historical documents:** an owner-approved banner on the historical README/docs stating which claims were tested and how they fared. The review measured the README showcase sentence at 12–14 tokens on seven tokenizers, against a claimed ~5–7. Historical text is not deleted.

**Acceptance:** the owner signs off on the archive and banner text. CI runs the retained tests. The repository has no CI today, so this depends on Phase 0 item 0.13 (§13).

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
| Under-powered or mis-specified statistics | Gates pass or fail by chance | One pre-registered margin; Tango test; call-clustered variance; pooled estimand with an intersection-union test; blinded internal pilot (§5.2) |
| Answer-prior or lure shortcuts in items | A worse-reading format passes | Balanced answers per stratum; randomized tree shape and lure values; heuristic-reader checks in G0 (§13) |
| Asymmetric constraints or whitespace between arms | Arm differences reflect the engine, not the format | One constraint policy for every arm; canonical separators pinned; whole-output strict decoding |
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
| (This plan's first draft) "codes only tie words", and M as the Phase 2 arm | Corrected after verification: compare codes and words at a **matched layout**. The decisive pair is `c_tsv` vs `r_tsv_p`; M is killed offline. |

## 12. Immediate next step

**Owner decision (no cost):** read review §1, §6.2 and §13.3.2.
- Decide whether the offline evidence closes the track for the repository's M; the review recommends yes.
- Decide whether the knife-edge fused-code result is worth one accuracy study. `c_tsv` is about 8.5–14% cheaper on English-label corpora, but 6.8–10.0% on multilingual labels (≥10% on only 1/6 tokenizers).

**Engineering:** start Phase 0, about 3–5 days with no approvals needed:
- **0.1** scorer robustness and the answer envelope
- **0.2** format-neutral, answer-balanced tasks bound to prompt content
- **0.3** strong baselines as first-class formats, including `r_tsv_p` and `c_tsv`, ported from `benchmarks/claude-review-2026-10-04/`
- **0.9–0.10** the three codec regression tests and the harness integrity tests (§13)

**In parallel, ask for the Phase 1 and Phase 2 approvals:** a Claude `count_tokens` key, and a $150 inference cap with named providers and models. Phase 2 must not start until the holdout author and the pre-registration (one margin, pooled estimand, constraint policy, escaping and whitespace policy) are in place.

---
## 13. Amendments from the verification pass (2026-10-04; see review §13)

These amend §§3–8. Where §§4–5 were already corrected in place, this section gives the detailed deliverables and acceptance tests.

### Phase 0: new and amended deliverables

| ID | Deliverable | Acceptance test |
|---|---|---|
| 0.1 (amend) | **Scorer envelope and diagnostics.** The answer row requires `task_id` plus either `answer` or `raw_text`. Optional fields: `model`, `usage{input,output,cache_read,cache_write}`, `finish_reason`, `latency_ms`, `attempt`; unknown keys are rejected. Add path-aligned `field_errors` with a fixed rule order: structure → scope swap → role swap → literal (nfc/whitespace/case/content) → per-enum. Add per-tokenizer confound flags (not text-exact; contains reserved strings). | The injected-error test recovers 100% of single-field errors with 0 false positives. An NFC-normalized answer is labelled `literal:nfc_equivalent`. Accuracy is reported with and without flagged tasks. |
| 0.2 (amend) | **Bind tasks to prompts.** `task_id = TEMPLATE/prompt_sha256[:16]/fixture/arm`, plus a `prompt_sha256` field. Each record gets `ast_sha256` (sorted-key compact JSON, equal to JCS for this schema). The manifest root covers the full canonical records. | Editing one legend byte changes the affected task IDs, and stale answers are rejected with a non-zero exit. Do not hash harness or codec sources. |
| 0.2 (amend) | **Answer balance.** In every question type × slot × label-class stratum, each closed-vocabulary answer appears n/k times. The answer is independent of line, depth and nuisance fields (Cramér's V = 0). | G0 heuristic test: constant, always-default, scope-blind, copy-literal, off-by-one, innermost-if and lure-elimination readers each score ≤ 1/k + 0.02 on non-leftmost targets (pair level 0). |
| 0.3 (amend) | **Strongest readable set.** Add r_tsv_p, feature-prefix cdsl, vdsl_nq and eng_bare as readable arms, and c_tsv / m_fused as Sylang-style arms. Add layout-matched code-vs-word pairs (c_tsv vs r_tsv_p). Every decoder reports a separate `canonical` flag. `d_m_opt` gets whitespace skipping. json_short emits enum keys first and decodes with a duplicate-rejecting, order-checking hook. | Round trip on ≥ 2,000 adversarial trees plus all 216 combinations. Key-mimic labels decode to the original tree or fail; they never decode to a different tree. |
| 0.4 (amend) | **Accounting unit.** Use tokens(record + terminator). Add a per-tokenizer assertion of the terminator law. Model caching as s + p·(0.1 + 1.15/Q) when p ≥ minlen, otherwise p + s, with the cacheable prefix ending at the record block, a 1.25× write, and a pad-to-minlen sensitivity. | Law holds in every corpus × format × tokenizer cell. No cached figure for a prefix below the provider minimum. |
| 0.5 (amend) | **Safe path, per tokenizer kind.** Record the `special` flag of every added token. Gemma 3 through raw SentencePiece needs the literal guard (escape `<` and `[` inside literals). | Acceptance is restated as zero reserved-string *matches* in content: by token-ID for tiktoken/HF, and by piece string for SentencePiece, where `is_control()` is False for 105/106. |
| 0.6 (amend) | **Unicode policy.** The greedy NFC-safe escaper, or the corrected whole-literal rule, judged on encoded text. MAX_TEXT_BYTES raised to ≥ 2^19. An invisible/format-character policy with the virama, emoji and Arabic-ZWNJ exemptions. Zs handled as a separate decision. A JSON literal walk, because the `_quote` hook does not cover json. | Property test: NFC(encode(t)) == encode(t) and decode == t under HF Rust NFC and Python unicodedata, for ≥ 20,000 trees drawn from the interacting set. Regression fixtures: `x\ñ`, `x\x1ḃ`, `Å̇`, `K̀`, 'ᄀ'+'ᅡ'. A 127-node tree with escaped astral labels encodes. |
| 0.7 (amend) | **Corpora.** Zipf label reuse with distinct-label rate as a parameter; report mean reuse k̄. Label-length strata from 1 to 16 words. Plausibility as a controlled stratum. Explicit slot strata to depth 4. Randomized tree shape and sibling position. Lure values drawn independently of the answer. | Manifest records k̄, the label-length histogram and the slot shares. Depth-3 and depth-4 contrasts are estimable. |
| 0.9 (new) | **Codec hardening.** Three regression tests: keyword-swap rejection in DSL/Prime; NBSP/U+3000/U+2028 rejected as separators; exact MAX_TEXT_BYTES and MAX_TEXT_BYTES + 1 with a 2-byte character. Stage-consistent errors: text readers raise ValidationError with a field path for enum and bounds faults. A normative lexical section in experimental-grammar.md. A drift test: codec tables == schema enums == `prompt_for` == legend tables. Generated depth-unrolled `anyOf` schema with the portable surrogate pattern. | The 3 known surviving mutants are killed. The same semantic fault raises the same class in all 5 formats. The committed schema equals the generator output. Depth 9 and a lone surrogate are rejected by the schema. |
| 0.10 (new) | **Harness integrity tests.** About 15 single-guarantee tests: SHA-only tamper, special-token test without the reserved string in the base vocabulary, hand-computed summary numbers, CLI subprocess tests, fixture regeneration byte-compare. Exit codes: 0 ok, 1 measurement/round-trip failure, 2 usage, 3 integrity, 4 asset. `--fail-under` for the scorer. | All 20 listed mutants of evaluation/ are killed; 44 existing tests still pass. |
| 0.11 (new) | **Run directory.** Canonical `counts.jsonl`, `summary.json`, `run.json` (environment, tokenizers version, `zlib.ZLIB_RUNTIME_VERSION`, input hashes), optional `timing.json`. Add `python -m evaluation verify --expected <dir>`. Split statistics only for n ≥ 10. | Two runs give a byte-identical `counts.jsonl`. A corrupted count makes `verify` exit 1 and name the row. |
| 0.12 (new) | **Export.** Escape U+2028, U+2029 and U+0085 (or use `ensure_ascii=True`) in every JSONL the tools write. Optional message-structured export with `static_prefix_sha256`; cache boundary after the record block. | `str.splitlines()` on the export yields exactly one parseable record per task. |
| 0.13 (new) | **Packaging, CI, provenance.** pyproject (requires-python ≥ 3.10, `sylang` script, `py.typed`). CI on Python 3.10–3.13 running unittest and mypy --strict on sylang_core. Tokenizer lock v2 with all 9 assets (kind, pins, licence, usage policy, gated flag). `--require-hashes` with every platform wheel. | CI green. The lock lists 9/9 assets with licence fields. A tampered wheel fails the install. |
| 0.14 (new) | **Review package reproducibility.** run.py flags (`--tokenizers`, `--seed`, `--records`). Code hashes in results.json. Probes for the showcase and D3–D5/D9. Commit the decision-relevant verification scripts: L/S decomposition, terminator law, G1 layout pairs, cost-per-correct, power and cluster simulations. | Every [M] claim in the review is produced by one command. |

**Gate G0 (amend):** G0 also requires 0.2 balance, 0.9, 0.10 and 0.11. Byte identity covers `counts.jsonl` only; gzip is excluded or pinned by zlib version.

### Phase 1: amendments

- **1.1** Add a tokenizer adapter protocol (hf-json, tiktoken, sentencepiece, api-count) with kind-specific pins. Any conversion between runtimes must match the reference on token **IDs** over a stress corpus: digit runs 1–64, whitespace and CR/LF runs, U+2028/U+0085, combining marks, emoji ZWJ sequences and all reserved strings. Re-certify the Qwen3 reconstruction on IDs.
- **1.2** Add these columns to the token tables:
  - L/S decomposition by the substitution estimator;
  - ceiling h and critical load L* = 9·S_R − 10·S_min per tokenizer and corpus;
  - `structure_floor` (Σ tok(label) + R);
  - decision steps per record under fast-forward;
  - code-token legend alignment (a covariate).
  Report ratios as micro (billing) plus geometric mean, max weight share and leave-one-out range.
- **1.3** Add the cache formula from 0.4. State the provider minimum per model and the date it was read.
- **1.4 (new) Offline cost pre-gate.** Compute C_M/C_X per tokenizer (reading and generation, N = 100). Record G2-cost as failed for any arm whose required accuracy ratio (C_M/C_X)/0.9 exceeds 1/a_X at the baseline accuracy observed in the cheaper arms. On review numbers this already kills M-repo (it needs ≥ 1.46–1.73× cdsl's accuracy).

**Gate G1 (amend):**
- **Status.** Failed for M-repo and m_opt; **undetermined** for Sylang-style variants (c_tsv, m_fused). c_tsv passes 5/6 tokenizers on en_skewed by about 10–12%, and 0–3/6 on multilingual, depending on legends and the baseline set.
- **Comparison set.** Pre-register a readable baseline set searched as hard as the Sylang candidates, with matched legend budgets.
- **Matched layout.** Decide code-vs-word on pairs that differ only in the relation/feature field.
- **Pre-registered before any G1 number is seen:**
  - the quoting/escaping policy (`"`/`\` recommended; no guillemets, ⟦⟧ or length prefixes);
  - head-first or canonical field order;
  - interning (off, or the same rule for every arm, reported per format);
  - whitespace (canonical separators pinned).
- **Strata.** Report G1 per label-length stratum, and state which stratum is the target workload.
- **Threshold.** If a 2-point accuracy loss is tolerated at a_X ≈ 0.9, the token threshold consistent with G2 is about 12%, not 10%.

### Phase 2: statistics and protocol amendments

- **One margin everywhere.** Choose δ and state it in G2, E2 and §5.2. Required pairs (ψ = 0.10, 90% power, one-sided α = 0.05), scaled by DE = 1 + 4·ICC for 5 questions per call:

| ICC | DE | δ = 3 pt | δ = 2 pt |
|---|---|---|---|
| 0 | 1.0 | 952 | 2,141 |
| 0.05 | 1.2 | 1,142 | 2,570 |
| 0.10 | 1.4 | 1,333 | 2,998 |
| 0.20 | 1.8 | 1,713 | 3,854 |
| 0.30 | 2.2 | 2,094 | 4,711 |

  These n assume a true difference of 0. If M is 1 point worse, δ = 2 needs 8,555 pairs before clustering.
- **Primary test.**
  - Use Tango's score test for paired non-inferiority, not McNemar (which tests equality).
  - Use call-level cluster-robust variance with t_{K−1}, or a call-resampling bootstrap.
  - If minimal pairs are used, cluster by pair as well.
- **Blinded internal pilot.** It replaces the fixed 300-question pilot. After 300 holdout pairs, re-estimate ψ and the ICC without arm labels and recompute n (cap 300–4,000). The rule is pre-registered. Simulated type I is 0.050–0.053 and power 0.88–0.89.
- **G2 structure.**
  - A single pooled primary estimand across model cells, clustered by prompt content.
  - Intersection-union test against each readable comparator at full α, with no Holm.
  - Family-level consistency rule: each family's point estimate > −δ, not per-cell significance.
  - Fixed sequence: reading NI → reading cost per correct → generation NI → generation cost.
  - Holm only for secondary rankings.
  - The same structure applies to the salvage comparison.
- **Cost endpoint.**
  - UCB by log-delta with call-level linearization, or a paired call bootstrap.
  - Its power at K = 200 calls is only about 40% for a format that is truly 12% cheaper. Size K for the cost endpoint separately, or treat cost as estimated rather than tested.
- **Reading items.**
  - Answers are balanced within strata (0.2).
  - The margin is stated on the reading-rate scale r = (acc − 1/k)/(1 − 1/k), where 1 point of accuracy = 1/0.65 points of r for this question mix.
  - The primary estimand is workload-weighted over **slots only**, never over answer or feature values.
  - Hazard flags (stative progressive, perfect, habitual present, direct + future) and plausibility are strata, not exclusions.
  - Adversarial/injection items are excluded from the primary endpoint and scored as a secondary attack-followed rate against stored `attack_targets`.
- **Meaning-level questions.** Before legends freeze:
  - rename `completed` to `perfect` and add a gloss to every legend;
  - fix batch.py:31;
  - either give every arm one Meaning paragraph (referential reference time; local evidence scope inside `if`) or restrict R1 to field-value questions.
  - G1 prose uses only "If A, then B".
- **Generation endpoint.**
  - Block-level parse failure is a separate call-level endpoint.
  - Record-level rates use cluster-robust intervals; plan 630·(1 + 19·ICC) records (869 / 1,228 / 1,827 at ICC 0.02 / 0.05 / 0.1), or run N = 1.
  - Any length-stopped output is scored as truncated.
  - Strict-decode the whole output, reject trailing text, keep raw text.
  - Never use first-record or first-object extraction.
- **Constraint policy.** Constraint is an explicit factor: either every arm is unconstrained (validate → canonicalize → node-cap → one retry), or every arm is constrained on one self-hosted engine with generated grammars. Never compare constrained JSON against unconstrained compact arms, including in the salvage rule. Prefill only up to canonical token boundaries (`Sylang core-v0.1:`, `Prime0.1`, `M0.1`). Batch grammars contain the separator.
- **Duplicates.** Analyse exactly one run per item and arm. Re-issue duplicates as whole calls. Pool them in a logistic model to estimate test-retest discordance per arm, and report ψ_between − mean(ψ_rr) = E[(p_A − p_B)²]. Never add them as extra rows. Default to k = 1 run per item unless the pilot shows cached reruns and low format heterogeneity.
- **Budget.** Recompute it from the corrected n and design effect. Powering every model cell separately at δ = 2 would cost several times the $150 cap [A]. The pooled estimand of §5.2 (about 4,000 pooled questions per arm, 4 arms, generation at N = 1) keeps 2a near $20 at mid-tier and $90 at top-tier prices (§5.3). Record the margin, design and cap in the pre-registration.

### Phase 3: amendment
- Run every arm with grammar fast-forward on and off. Report billed output tokens and sampled forward steps as separate metrics. Do not predict self-hosted latency from G1 token ratios.

### Phase S: amendments
- The banner deliverable absorbs the historical-claims audit:
  - a measured-status table (claim, source line, status: contradicted on deployed tokenizers / unsupported / unmeasurable offline, measured value, script);
  - a README restructure, so that historical H2 sections are demoted under one caveated heading;
  - a PDF provenance note (ChatGPT deep-research export; citations to 5 files not in the repository);
  - image renames and captions;
  - links to the review and plan.
- Acceptance "CI runs the retained tests" depends on 0.13.
- A tokenizer/vocabulary track may be reopened only if a proposal states its gain over the best layout that needs no tokenizer change (c_tsv/r_tsv). The format-independent floor bounds that gain at about 1.2–3.3 tokens per predicate.
