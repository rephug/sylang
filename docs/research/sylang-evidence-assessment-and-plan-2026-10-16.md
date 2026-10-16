# Sylang: an independent assessment of the evidence, and a plan for the next US$100 and US$1,000

**Date:** 2026-10-16 · **Author:** Claude (independent assessment, analysis and planning only) · **Scope:** the code and result files on branch `claude/focused-edison-cuu64g`, the packet's codecs (read as data), the repository's Sylang Prime documents, and the §10 report in [`copy-bottleneck-diagnosis-2026-10-13.md`](copy-bottleneck-diagnosis-2026-10-13.md). Nothing in this document was run on a rented GPU, no paid call was made, no history was rewritten. The budget figures are ceilings for review, not spending.

---

## 0. The conclusions in plain English

1. **Three different things are called "Sylang", and only the narrowest one has been tested.** *Sylang Prime* is the human-readable constructed language in the repository's README and reports (agglutinative words, a custom 8,192-token tokenizer, "55–60% fewer tokens"); nothing in this repository measures it, and its numbers come from design documents, not experiments. The packet's *core-v0.1* grammar has experimental `prime` and `m` serializers for single statements; they were not trained or evaluated here either. What was tested is the *SYLANG-BATCH* record format in its `codes` style (the "compact" format: one line per record, `subject<TAB>code<TAB>object`), with and without its header, on one closed schema (three relations, four small feature sets, identifiers of 2–5 characters) and one task (list the negated records). Every result below is about that slice.

2. **On that slice, the compact format works and the evidence is solid for a tiny model trained from scratch.** A 1.1-million-parameter byte model learns header-free Sylang in 2 of 3 pre-registered seeds, is at ceiling on everything inside its training distribution, beats compact JSON and CSV on every panel (both machine formats never reach 90% in the same budget), ties controlled English on accuracy, and costs 0.63× the English tokens under the Qwen 3 tokenizer, 0.37× the prefill compute and 0.31× the training time. Removing the 35-byte header was the single most valuable change (−21% tokens against the current format; the byte model also learns it more reliably).

3. **Three things are uncertain, and they matter more than anything already shown.** (a) *No pretrained language model has seen the format.* Every learner here was trained from scratch per format, so the comparison measures how learnable each format is for a blank slate, not how a real model with strong English and JSON priors behaves. A pretrained model may prefer JSON. (b) *Learning is seed-dependent on the harder data*: 2 of 5 seeds reached 90% for both Sylang and English, so "it learns" is a rate, not a guarantee. (c) *No encoding extrapolates to record counts beyond its training range*; this was fixed by training on the range the pipeline will see, which is cheap, but it is a limit, not a solved problem.

4. **Claims that would be premature today:** that Sylang saves 55–60% of tokens (untested; the measured saving of the compact record format is 37% against controlled English and 58% against compact JSON on this schema); that the format is "better for LLMs" (no LLM has been tested); that the benefit holds for free-text or nested content (only flat seven-field records were used); that the two-token record-count line improves long inputs (suggestive on one learned seed, not established); that Sylang Prime's custom tokenizer adds anything (unmeasured).

5. **Where the next machine-efficiency wins are, measured with the deployment tokenizer and no model change.** The answer format is the largest untouched lever: the JSON list of pairs costs 24–30% more output tokens than a bare `subject object` list, and output tokens are the expensive side of every hosted model. Replacing tabs by spaces saves 6% of input tokens. Grouping records by a repeated subject saves 20–30% when subjects repeat (it saves nothing when every identifier is unique, as in the study's items). The batch codec's repeated-value dictionary, the one M feature already implemented but never exercised, *costs* tokens on short identifiers at any repetition rate and only pays with long, repeated labels (12–20 characters: −17% to −36%). Single-letter "macro" codes for frequent feature combinations save about 1%. Anything that needs new vocabulary (fusion tokens, a Sylang tokenizer) requires teaching the model new embeddings and is a different, more expensive experiment.

6. **The fastest credible next experiment is a pretrained small model, not more byte-model runs.** Fine-tune Qwen3-0.6B (and 1.7B if the budget allows) with LoRA on identical items in four arms, concise English, compact JSON, the tested compact format, and one justified improvement (space separators plus the bare answer list, each as its own controlled factor), after first measuring the zero-shot prior of the same model with a legend. Count everything per request: legend and instruction, records, output, conversion overhead, invalid outputs and one retry; report accuracy, latency and cost per correct answer. The training runs are minutes each on a rented consumer GPU, so a US$100 ceiling buys the whole design with seeds to spare; the cost is dominated by careful setup, not GPU time.

7. **The staged plan** spends the remaining budget only on questions the pilot leaves open, with written gates: realistic data (repeats, long labels, nested `if` records) and the M variants where they pay; then model scale and a second tokenizer family to test whether the advantage is tokenizer-specific; and only then custom vocabulary. The stop rule is simple: if a pretrained model does as well or better with JSON at equal cost per correct answer, Sylang's niche is from-scratch or custom-tokenizer models and the plan changes direction.

8. **Git history.** Twenty-eight early commits on the branch carried four files with the owner's synthetic fixture identifiers and a scorer file with expected answers; the current head is clean. This is fixture content, not credentials or personal data. A squash and force-push of this branch would remove those commits from the branch's reachable history and would not remove them from GitHub's object store (a Support purge does that), from clones made since 13 October, or from the pull request's event timeline. It needs the owner's decision and is not urgent.

**Recommended next action:** approve the US$100 pilot design in §4 (it is the only way to answer the pretrained-model question, which decides whether the rest of the plan is worth running), and decide the two items in §8. Nothing else should be spent before the pilot's gate is read.

---

## 1. What exactly was tested, and what the evidence supports

### 1.1 Three things called Sylang

| name | where it is defined | what it is | tested here? |
|---|---|---|---|
| **Sylang Prime** (the "fuller historical" design) | `README.md`, `docs/research/one-token-one-meaning.md`, `docs/research/Sylang Prime_ LLM Performance and Benefits Report.pdf`, `docs/tokenizer/*.md` | a constructed language for humans and models: agglutinative words (`Taru nkarisu lemai`), ~300 roots and ~30 affixes, a two-pass morphology-aware tokenizer with an 8,192-token vocabulary and mined "fusion tokens" for frequent morpheme sequences; claims 45–60% fewer tokens than English, faster inference and better accuracy | **No.** The repository holds no corpus, tokenizer, model or measurement for it; the report is a design document (its token counts are illustrative, with no tokenizer named). |
| **core-v0.1 formats** (`english`, `json`, `dsl`, `prime`, `m`) | the packet's `sylang_core/codec.py` | serializers for one closed statement tree: a seven-field predicate (subject, relation, object, polarity, tense, aspect, evidence) with optional nested `if(condition, consequence)`. `prime` is a readable keyword form, `Prime0.1 pred{subject="a";relation=help;…}`; `m` is a positional code form, `M0.1:p("a",h,"b",-,p,s,r)`, with single-letter codes for every feature including the defaults | **No.** They were used by the packet's own offline measurement harness for content-length comparisons, not trained on here. |
| **SYLANG-BATCH** (the current format) | the packet's `evaluation/batch_codec.py`; the public re-implementation `experiments/copy_probe/sylang_batch.py` | a batch framing of the same tree: a header line, then one line per statement. Two styles: `words` (`help negative past reported`) and `codes` (`Hpr`: relation letter, upper-case when negative, then letters for non-default tense/aspect/evidence). Three escaping policies. An optional **dictionary** (`intern`): labels occurring twice or more are declared once (`@i<TAB>label`) and referenced as `@i`. Nested `if` records are flattened into the line. Caps: 1,000 records, 256-scalar labels. | **Partly.** `codes` style, `selective` policy, **no dictionary**, **no `if` records**, flat predicates only. Three framings were trained: with the header (`sylang1`), without it (`sylang2`, the "compact" candidate, equal to the earlier `compact` arm), and header-free with a one-line record count (`sylang2c`, exploratory). `words` was trained once (§8) and dropped because its models read the filler, not the words. |

The "improved-M" ideas the owner asks about (repeated-value dictionaries, compact relationships, combined frequent structures) do not exist as a separate specification in the repository or the packet; the dictionary is implemented in the batch codec and the rest are design intentions. Section 2 measures each of them.

### 1.2 The task, the data and the learner

- **Task**: given K records, output the JSON list of `[subject, object]` of the negated records in record order, or `[]`. The answer depends on polarity and the two identifiers only; relation, tense, aspect and evidence are distractors. Nothing else was ever asked of a trained model (the packet's harness defines three reading questions and a generation task; only this selection task was trained).
- **Schema**: 3 relations × 2 polarities × 3 tenses × 3 aspects × 4 evidences = 216 feature combinations; two combinations are held out of training. Identifiers: letter-digit-letter in the earlier studies, 2–5 characters from `[a-z0-9]` in v3, every identifier unique within an item. Record counts K = 1..8 (v2.1), 1..12 trained and 13..16 held out (v3), 1..16 trained and 17..20 held out (the expansion). Six request wordings in v3, two held out.
- **Learner**: the packet's tiny byte model (4 layers, width 128, 1.12 M parameters, 2,048-byte context), trained from scratch per format with the packet's optimiser; with two changes found in §8–9 of the report: relative positions (ALiBi) instead of the learned position table, and a cosine learning-rate decay. **No pretrained model, no subword tokenizer, no GPU.** The Qwen 3 tokenizer was used only to count tokens.
- **Comparators**: controlled English (one sentence per record, numbered, no filler), compact JSON (one-letter keys, defaults omitted), CSV (every field spelled out). English and JSON are what a deployed model would receive; the byte model was trained on them exactly as on Sylang.

### 1.3 Demonstrated (with the failures included)

All numbers are from the committed result files and the §10 tables; "learned" means the fine iid curve reached 90%; the dev-selected checkpoint is reported.

| claim | evidence | strength |
|---|---|---|
| Removing the header makes the format cheaper and easier to learn | −21% Qwen 3 tokens against the current `codes` format over K 1..8 and −38% against English (§9.3.1); on the byte model 3/3 seeds learn instead of 2/3 and 90% is reached at 38,000 items instead of 66,000 (§9.3.4) | strong for the byte model; the token saving is arithmetic |
| Relative positions are necessary for this byte model | with the learned position table, `codes` with its header scores 0% selected-only at K ≤ 8 after 80,000 items and every learned-position model collapses when the prompt is shifted by one byte (§9.3.2); under ALiBi every arm learns | strong, but specific to this architecture |
| The compact format beats compact JSON and CSV for a from-scratch learner | v3, same items, same recipe: Sylang 2/3 seeds learn at 54,000 items; JSON 0/2 (one seed ends at 80%), CSV 0/2; paired per item Sylang is above JSON by +7 to +12 points and above CSV by +15 to +48 on every panel, 90% lower bounds above +3 (§10.6.2) | strong for the byte model; two seeds for the machine formats |
| The compact format ties controlled English on accuracy inside the training range | items to 90%: 54,000 vs 60,000–62,000; paired per item within ±3 points on iid, reorder, swap, flip, held-out wording and held-out combination (§10.6.3); over five seeds, 2/5 learned for both | moderate: the escape rate is 40–50% for both and three seeds resolve only large effects |
| The compact format is cheaper on every cost axis measured | 0.63× English tokens, 0.42× JSON (Qwen 3, K 1..12 mix); prefill 0.37× English and 0.21× JSON in FLOPs; 0.31× the training time per 80,000 items; handoff demo 0.69× process tokens and 0.39× consumer latency (§10.5, §10.6.4) | strong (arithmetic and measured), latencies relative only |
| Variable identifiers, unseen wordings, unseen feature combinations, reordered/swapped/flipped records are handled at ceiling once in the training distribution | learned seeds: 93–100% by identifier length, 96% held-out wording, 97% (95–100) held-out combination, 97–98% (94–99) reorder/swap/flip (§10.6.3) | strong |
| Identifier shape is a hard failure mode | a model trained on three-character identifiers scores 0% selected-only on every other shape while scoring 85–93% on its own (§10.4) | strong, and the most important practical finding |
| Record-count extrapolation fails for every encoding and is fixed by training on the range | Sylang 34% (30, 38), English 52% (41, 64) at K 13–16 after training on K ≤ 12; training on K ≤ 16 lifts the fully held-out set from 30% to 89% at no extra data cost (52,000 items) and moves the limit to K 17–20 (44%) (§10.6.3) | strong on one seed for the expansion |
| The codec is correct and compatible | byte-identical to the packet renderer on 3,000 random batches and every BMP code point; an adversarial review found no blocking or major defect; 16 tests (§10.3) | strong |

**Failures and deviations that belong next to the successes:**

- Seeds that never left the always-empty plateau within 80,000 items on the v3 corpus: English 0, 3, 4 (of 0–4); Sylang v2 1, 3 (seed 4 escaped late and ended at 85%); JSON 1 (seed 0 ended at 80%); CSV 0 and 1; Sylang v1 1 (of 0–2); the count-line arm 1 (of 0–1). On the easier §9 corpus one `codes` seed in three never learned and one English seed in three under the constant learning rate.
- The pilot's single-factor corpora (`k12`, `wording`) could not be judged on the pre-registered panels because those panels use variable-length identifiers (deviation 2); they were re-scored on fixed-shape panels at the final checkpoint, where `k12` also beats the combined corpus on one panel (`full`, 58% against 47%).
- Checkpoint naming omitted the corpus until mid-pilot (deviation 1); the decisive wave was queued before the pilot finished (deviation 3); extra seeds, extra `sylang1` seeds and the `sylang2c` arm were added after results were seen (deviations 4–6) and are reported as supplementary; the pre-registered compute estimate was wrong by about 2× (36 core-hours of training against a planned 20).
- The COPY-96 initialisation, the staged and mixed curricula, and the `words` style did not help (§8.5, §9.4).
- The two independent claim-vs-data passes corrected 28 numerical slips in §10 (ranges written as mean-to-max, one overstated tie); those corrections are in the committed text.

### 1.4 Uncertain

1. **Behaviour of a pretrained model.** Every learner was a blank slate. A model that already knows English and JSON may need fewer examples for those formats than for Sylang codes, may answer zero-shot with a legend, and may make different errors. This is the question the next experiment must answer before anything else is built.
2. **The escape rate.** 2 of 5 seeds at this budget for both leading formats. Whether more updates, a warm-up, or a larger model removes the plateau is unknown; it may be a property of the tiny model.
3. **Transfer to realistic content**: nested `if` records, long or repeated labels, free-text fields, more relations, a second task (lookup, feature extraction, generation) were never trained.
4. **The count cue** (a header or count line helps extrapolation to longer batches): consistent in direction on every learned seed, but the spread between seeds of one arm (42 against 83 points) is as large as the difference between arms.
5. **Latency and training-time ratios** were measured on a shared four-core CPU beside other runs; the ratios are reliable, the absolute times are not.

### 1.5 Premature

- "Sylang saves 55–60% of tokens": the measured saving is 37% against controlled English and 58% against compact JSON for this record schema under Qwen 3; the Prime figure is from a design document.
- "Sylang is better for LLMs" or "improves accuracy": no LLM has processed it in this work.
- "The format generalises": only within the trained identifier shapes, wordings, feature combinations and record counts; outside them it fails or degrades.
- "The dictionary / interning helps": it is implemented, untested, and the tokenizer arithmetic says it hurts on short identifiers (§2).
- "The count line helps long inputs": exploratory, one learned seed per arm (§6).
- Any claim about Sylang Prime's custom tokenizer, morphology or human readability: unmeasured.

---

## 2. Machine-efficiency ideas from the M design that the experiment did not use

The owner's priority is machine efficiency; readable Prime training is deferred. The useful question is which M-style ideas reduce *the cost a deployed model is billed for* (tokens in and out, and the latency they imply) without changing the tokenizer, and which ones need new vocabulary. Every number below was measured with the Qwen 3 tokenizer (`tokenizer.json`, the file the study used) on freshly generated synthetic records, identical across encodings, 48 batches per K (`experiments/copy_probe/measure_m.py`, output `results/m_design_tokens.json`; deterministic from its seeds).

### 2.1 Measured, tokenizer unchanged

Mean Qwen 3 tokens of the record body (the instruction, 17 tokens, and the answer are the same for every encoding and are excluded):

| K | english | compact JSON | CSV | **compact (tabs)** | spaces instead of tabs | + dictionary (`@i`) | grouped by subject | macro codes |
|---|---|---|---|---|---|---|---|---|
| *unique identifiers (the study's items)* | | | | | | | | |
| 4 | 64.8 | 105.6 | 69.6 | 35.1 | 33.0 | 35.1 | 34.1 | 34.7 |
| 8 | 131.1 | 211.4 | 141.3 | 72.5 | 67.9 | 72.5 | 70.0 | 72.0 |
| 16 | 268.4 | 419.3 | 282.5 | 145.0 | 136.3 | 145.0 | 140.5 | 143.6 |
| *half the records reuse an earlier subject, a fifth an earlier object* | | | | | | | | |
| 8 | 130.6 | 210.5 | 140.8 | 72.6 | 68.2 | 85.0 | 59.0 | 72.0 |
| 16 | 270.3 | 420.1 | 281.4 | 144.8 | 136.3 | 168.8 | 116.2 | 143.2 |
| *few actors: 80% subject reuse, 50% object reuse* | | | | | | | | |
| 8 | 130.9 | 210.5 | 139.9 | 71.4 | 67.1 | 87.0 | 54.0 | 70.5 |
| 16 | 268.3 | 416.7 | 280.9 | 144.6 | 135.5 | 164.4 | 102.9 | 143.2 |

With long labels (12–20 alphanumeric characters, the regime where a dictionary is meant to pay):

| K | english | compact JSON | compact (tabs) | + dictionary | grouped by subject |
|---|---|---|---|---|---|
| unique, 16 | 554.2 | 702.7 | 429.8 | 429.8 | 426.2 |
| half reuse, 16 | 558.2 | 710.5 | 432.9 | 357.3 | 335.1 |
| few actors, 16 | 558.0 | 708.0 | 435.5 | 277.6 | 286.0 |

Readings, one per idea:

1. **Repeated-value dictionary (interning), the implemented M feature.** On identifiers of 2–5 characters it *costs* tokens at every repetition rate (+17% at K = 16 with heavy reuse): a declaration line `@i<TAB>label` plus two-token references (`@`, digit) is more than repeating a two-to-four-token identifier. It pays only when labels are long: −17% at half reuse and −36% with few actors on 12–20-character labels. Rule for the pipeline: intern a label only if its token count exceeds about 3 and it occurs at least twice; the codec can decide this per batch from the tokenizer, and the decoder already accepts both spellings. **Tokenizer unchanged; decoder support exists in the packet codec; the public codec would need the dictionary implemented (it currently raises).**
2. **Compact relationships, as grouping by subject** (`subject<TAB>code obj;code obj`): no gain on unique identifiers (−3%), −20% at K = 16 with half reuse, −29% with few actors; with long labels −23% to −34%. It changes the record grammar (one line per subject, several relationships per line) and the answer still lists pairs. **Tokenizer unchanged; needs a codec extension and a decoder rule; the model must learn the grouped layout.**
3. **Combining frequent structures as macro codes** (one letter for each of the 12 most frequent feature combinations): −1%. The code grammar is already one to four single-letter tokens; there is nothing left to compress there. **Not worth a change.**
4. **Separators**: a space instead of a tab saves 6% of the record tokens at every K and every distribution (the tab is a separate token; a space merges with the following word piece). Invisible to the byte model, which was trained on tabs; a one-line change in the codec's wire format for the deployment path. **Tokenizer unchanged; a v2 wire option.**
5. **The answer format** (not an M idea, but the largest lever on the output side):

| K | JSON list of pairs | bare `subject object` pairs | saving |
|---|---|---|---|
| 4 | 17.0 | 12.0 | 29% |
| 8 | 31.0 | 23.2 | 25% |
| 12 | 47.3 | 37.0 | 22% |
| 16 | 60.9 | 46.6 | 23% |

Output tokens are billed at several times the input price by hosted models and are generated one at a time, so a quarter fewer output tokens is worth more than a quarter fewer input tokens. The lean answer is also an easier target for a small model (fewer structural bytes); the §9 breakdown showed structural bytes are learned first and identifier bytes last, so the accuracy effect should be neutral or positive, but it must be measured rather than assumed. **Tokenizer unchanged; a parser change on the receiving side.**

6. **Legend and instruction amortisation.** A pretrained model needs a legend for the codes (what `Hpr` means); the packet's legends are 60–120 tokens. For batched use the legend sits in a cached system prompt and is paid once per cache window, not per request; for single requests it is a fixed cost that erases the per-record saving below about K = 5. The pilot must count it both ways.

### 2.2 Ideas that need new vocabulary (custom tokens)

- **Fusion tokens / a Sylang tokenizer** (the Prime design): adding tokens such as whole codes with their separators, or whole Sylang words, to a model's vocabulary. Under an unchanged tokenizer the codes already cost one to four tokens per record on top of the identifiers, so the ceiling of this change on the record format is small (the identifiers dominate: 5.3 of 9.0 tokens per record are the two identifiers). For free text in Sylang Prime it would be the whole point, but it needs embedding initialisation and continued pre-training or a long fine-tune, a frozen vocabulary that every consumer must share, and it is incompatible with hosted models whose tokenizer cannot be changed. It is a research programme, not a pilot.
- **Dedicated control tokens** for record boundaries or roles (`<rec>`, `<neg>`): the same objection; the byte model's results show that plain bytes carry the roles once positions are relative.

**Recommendation for §3–4:** test the unchanged-tokenizer improvements that have a measured, distribution-independent effect (space separators; the bare answer list) as controlled factors, and keep the distribution-dependent ones (dictionary for long labels, grouping for repeated subjects) for the stage that uses realistic data, where the gain exists. Do not spend the first US$100 on custom tokens.

---

## 3. The fastest credible next experiment: a pretrained model on identical items

### 3.1 Why this and not more byte-model runs

The byte model has answered what it can: the format is learnable from scratch and is the cheapest to learn and to run. The open question that decides the project is whether a model that already speaks English and JSON gains anything from the compact format. That needs a pretrained model, a subword tokenizer, and a GPU for an afternoon, nothing larger.

### 3.2 Design (pre-registered before any run; the full text goes into the report's §11.1 when approved)

**Model.** Qwen3-0.6B as the primary learner (the deployment tokenizer family the study has used throughout; small enough that a run is minutes); Qwen3-1.7B as the second size if the pilot budget allows, to see whether the ordering holds with capacity. Both open-weight, Apache-2.0, 151k-token vocabulary. Training: LoRA (rank 16, all linear projections), bf16, sequence length ≤ 1,024, AdamW with cosine decay, two epochs; identical hyperparameters for every arm. A byte-level alternative (ByT5-small) is a stage-1 option only if the subword results are surprising.

**Items.** The v3 generator unchanged (identifiers 2–5 characters from the hash split, six wordings with two held out, two held-out feature combinations, K 1..12 trained and 13..16 held out), plus a second item family with K 1..16 trained and 17..20 held out, so the pilot reports both the in-range and the extrapolation limit. All items are fresh synthetic; nothing from the packet.

**Arms** (one rendering per arm of the same item; the answer bytes identical):

| arm | rendering | why it is in |
|---|---|---|
| `english` | controlled English, numbered records | the readable comparison, labelled as such |
| `json` | compact JSON, one-letter keys, defaults omitted | the machine-format comparison a pretrained model already knows |
| `sylang2` | header-free compact codes with tabs | the tested candidate, unchanged |
| `sylang2s` | the same with space separators | the one input-side improvement with a measured, distribution-independent effect (−6%) |

and one **output-side factor** applied to every arm: JSON list of pairs (the study's answer) versus the bare `subject object` list (−22 to −29% output tokens). Two factors, each one change at a time: 4 arms × 2 answer formats = 8 cells. Nothing else varies.

**Controls.**
- *Zero-shot prior*: before any fine-tuning, each arm is prompted with its legend (the packet's wording for English and JSON, a new one of matched length for the codes) and the instruction, on the test panels. This measures what the model brings to each format for free; without it, a fine-tuning difference cannot be attributed to the format.
- *Learning curves*: fine-tunes at 500, 2,000 and 8,000 training items per cell, three seeds for `json` and `sylang2` at every size, two seeds for the others at 8,000 and one at the smaller sizes. Items to 90% is read from the curve, as in the study.
- *Dev-selected checkpoint* (dev panels only), the same stopping and selection rules as §10.1.
- *Context check*: every sequence against the model's context; the longest JSON item at K = 20 is about 450 tokens.

**What is counted, per request** (the owner's list): legend tokens (counted once as a cached system prompt and once per request, both reported); instruction tokens; record tokens; output tokens; conversion-in and conversion-out wall-clock (the codec, microseconds); invalid outputs (unparseable, wrong shape) and the retry policy (one retry with the same prompt at temperature 0 is pointless; the policy is one retry with the output format reminded, counted as a second request); accuracy after retry; wall-clock latency per request at batch 1 and at batch 32 under vLLM; GPU-seconds per 1,000 requests; and **cost per correct answer** using the packet's `economics.cost_per_correct` convention (every request, retry and failure in the numerator, correct answers in the denominator), computed with the measured GPU cost and, for reference, with a published hosted-model token price for the same model family.

**Criteria, fixed now.**
1. An arm *learns* at a data size if dev-selected test accuracy ≥ 90% in ≥ 2 of 3 seeds.
2. The format question: at 8,000 items, paired per item on the test panels, `sylang2` vs `json`: *better* if the 90% lower bound is above +3 points, *tied* if the interval contains 0, *worse* if the upper bound is below −3; the same for items to 90%. The cost question: cost per correct answer of `sylang2` relative to `json` and to `english`, with the legend counted per request and amortised.
3. The improvement factors: `sylang2s` − `sylang2` and bare − JSON answer, paired per item; an improvement is *kept* if it is tied or better on accuracy (lower bound above −3) and cheaper by the measured margin.
4. Gate for stage 1 (§5): the compact format is tied or better than JSON on accuracy at 8,000 items **and** its cost per correct answer is at most 0.85× JSON's with the legend amortised; if it is worse than JSON by criterion 2 with the legend counted per request as well, the direction changes (§5.5).

**Deliverables.** A results file per cell (aggregates and per-item flags, no prompts), a summary table, learning curves, the cost-per-correct table under both legend conventions, the worked example of one request in all eight cells, and a one-page decision memo against the criteria. All generation and scoring code is already in the repository except the LoRA trainer and the vLLM evaluation loop (about 300 lines; written and smoke-tested on CPU before any GPU is rented).

### 3.3 What this experiment does not answer

Human readability (deferred by the owner), nested records, free text, the custom tokenizer, and any hosted model's behaviour (no paid calls; a hosted comparison is a separate decision, §8).

---

## 4. The US$100 pilot (design for review; nothing has been rented or spent)

### 4.1 In plain English

Rent one consumer GPU by the hour, with a prepaid balance that cannot overrun, for about a day of wall-clock spread over two or three sessions. Generate all data locally for free. Fine-tune a small open model (Qwen3-0.6B) eight ways on identical items, measure everything per request, and read one pre-registered gate. If the core runs finish under US$40, repeat the same design on the 1.7B model. Expected spend is US$15–35; the US$100 ceiling is a hard cap enforced by the deposit, not a target.

### 4.2 Specification

| item | choice | reason |
|---|---|---|
| Model(s) | `Qwen/Qwen3-0.6B` (core); `Qwen/Qwen3-1.7B` (extension A) | the tokenizer family of every token count in this project; Apache-2.0; open; a 0.6B LoRA run on 8,000 items is minutes |
| GPU | Runpod **Community Cloud RTX 4090 24 GB at US$0.34/h** (expectation); **Secure Cloud RTX 4090 at US$0.89/h** (ceiling, if Community stock is gone); Vast.ai RTX 4090 at about US$0.37/h on-demand as the alternative (host-set storage and bandwidth fees must be read on the offer) | cheapest 24 GB card with bf16; both models' LoRA fits at micro-batch 4–8 |
| Training method | LoRA rank 16 on all linear projections, bf16, gradient checkpointing, sequence ≤ 1,024, AdamW, cosine decay, 2 epochs, identical for every cell; dev-selected checkpoint; greedy decoding, non-thinking format | one recipe for all arms; the study's rules carried over |
| Data | generated on the server with the v3 generator: per cell 8,000 training items (sizes 500 / 2,000 / 8,000 are nested prefixes of one stream per seed), dev panels, test panels (iid, held-out wording, held-out combination, K 13–16, full, reorder/swap/flip; about 1,000 items), plus the K ≤ 16 / 17–20 family; synthetic only | free; identical records across arms; nothing from the packet |
| Repetitions | `json` and `sylang2` cells: 3 seeds × 3 sizes; `english` and `sylang2s` cells: 2 seeds at 8,000, 1 at 500 and 2,000; both answer formats; = 52 runs for 0.6B (and 52 for 1.7B in extension A) | three seeds where the pre-registered contrast is read, fewer elsewhere |
| Held-out tests | the v3 panels (identifiers, wordings, combinations, record counts never trained), read once at the dev-selected checkpoint; the zero-shot prior on the same panels before any training | the study's structure, unchanged |
| Expected runtime | 0.6B: training about 2–6 min per run (anchors: Llama-3.2-1B LoRA 2,000 items × 2 epochs in 4.2 min on an A10G; SmolLM2-1.7B 20,000 items in 8.75 min on an L40S; a 4090 is faster than an A10G), 52 runs ≈ 3.5 h; evaluation with vLLM ≈ 30 s per model ≈ 0.5 h; zero-shot ≈ 0.2 h; setup and smoke ≈ 1.5 h → **≈ 6 h**. 1.7B: ≈ 2.5× training → **≈ 12 h** | measured anchors in the appendix |
| GPU cost | 0.6B: 6 h × US$0.34 = **US$2** (US$5.4 at the Secure price); 1.7B: 12 h × US$0.34 = **US$4** (US$10.7 Secure) | |
| Storage and transfer | Runpod network volume 20 GB at US$0.07/GB-month = US$1.40 per month (kept one month); container disk negligible; Runpod charges no ingress or egress; adapters are about 10 MB each, result files kilobytes | |
| Reserve | failed pods, a re-run of any cell whose seeds disagree by more than 10 points, one extra seed per contrast cell: ×2 on everything above | |
| **Expected total** | **US$15–35** (Community) · **US$30–60** (Secure) | |
| **Ceiling** | **US$100**, enforced by prepaid deposits only (first deposit US$25, no auto-pay), a second deposit only after the core gate is read | Runpod stops pods at zero balance; Vast likewise, with storage still accruing until the grace period ends, so delete the volume at the end |

**Spending controls, in order.** (1) Prepaid credits, auto-pay off on both providers. (2) Every pod launched with a hard runtime limit (a `timeout`-wrapped command that uploads results and terminates the pod); nothing left running overnight. (3) The LoRA trainer and the evaluation loop smoke-tested on the CPU server with 20 steps of Qwen3-0.6B before any rental, so GPU time is not spent debugging. (4) All code and generated data pushed to the pod from the server; results pulled back as aggregates (per-item flags, no prompts). (5) A ledger in the results directory with every rental's start, stop, price and purpose; the pilot report quotes it.

**Deliverables.** `results/pilot_pretrained.jsonl` (one line per run), `results/pilot_pretrained_zero_shot.json`, learning curves per cell, the cost-per-correct table under both legend conventions, latency at batch 1 and 32, a worked example of one request in all eight cells, the ledger, and the decision memo against §3.2's criteria, appended to the report as §11.

**If the result is inconclusive.** Three cases are pre-defined. (a) *Everything at ceiling* (every arm ≥ 97% at 2,000 items): the task is too easy for a pretrained model; the cost comparison stands on its own, and stage 1 moves straight to the harder data. (b) *Nothing learns at 8,000 items* in any arm: a pipeline defect is more likely than a format effect; check the zero-shot prior and the legend, fix, re-run the core once within the reserve, and if it still fails, stop and report. (c) *Mixed*: the `sylang2`−`json` interval straddles the ±3 band in either direction: add two seeds to those two cells (within the reserve) and report the five-seed interval; if it still straddles, the honest answer is "tied", and stage 1 proceeds on the cost result alone.

### 4.3 What the pilot deliberately does not include

A hosted model (no paid API calls; a zero-shot reference on a hosted Qwen-family model would cost a few dollars and is listed as a separate decision in §8), a second model family (stage 2), custom tokens (stage 3), human readability, and the GBG integration (a later example only).

---

## 5. Staged expansion to US$1,000 (including the first US$100)

### 5.1 In plain English

Each stage buys one answer the previous stage could not give, and each has a written gate that must pass before the next deposit. Stage 1 asks whether the advantage survives realistic data and a second task; stage 2 whether it survives a bigger model and a different tokenizer; stage 3 is the only place custom vocabulary or a real workflow dataset is paid for. The total is capped at US$1,000 and the expected spend if every gate passes is about half of that.

### 5.2 Stages

| stage | cumulative ceiling | expected cost | what it establishes | GPU | gate to proceed |
|---|---|---|---|---|---|
| **0 — pretrained pilot** (§4) | US$100 | US$15–35 | whether a pretrained model gains from the compact format against JSON and English at equal items, and the cost per correct answer with the legend counted both ways | RTX 4090 | **G0**: `sylang2` tied or better than `json` on accuracy at 8,000 items (paired, 90% lower bound above −3) **and** cost per correct ≤ 0.85× `json` with the legend amortised. The improvement factors (space separators, bare answer) are kept or dropped by their own criterion. |
| **1 — realistic data and the M variants** | US$250 | US$40–80 | whether the gain survives repeated subjects and long labels (where grouping and the dictionary pay, §2), nested `if` records, K up to 32, and a second task (the packet's lookup and feature-extraction questions); whether the count line helps (§6, as a factor); whether the escape plateau exists for a pretrained model at all | RTX 4090 / L40S, 3 seeds per contrast | **G1**: the G0 ordering holds on the realistic distribution and on at least one other task; each M variant's measured token saving is within 5 points of the arithmetic in §2 **without** an accuracy loss (lower bound above −3); otherwise the variant is dropped, not the stage |
| **2 — scale and tokenizer family** | US$550 | US$120–200 | whether the ordering holds at Qwen3-4B and 8B (LoRA on an A100 80 GB at US$1.19–1.79/h, about 30–40 h) and on a second family with a different vocabulary (SmolLM2-1.7B, 49k tokens, Apache-2.0; or Llama-3.2-3B under its community licence) at equal items; a vLLM latency and throughput benchmark at batch 1/32 on the larger model | A100 80 GB | **G2**: the sign of the accuracy contrast and the cost ratio hold at 4B and on the second family; if the ratio holds but the accuracy contrast reverses on the second family, the result is "tokenizer-specific" and stage 3 is limited to option (a) |
| **3 — one of two** | US$1,000 | US$150–350 | (a) *custom vocabulary*: add tokens for the codes and separators (and, as a separate arm, whole feature-combination codes) with embedding initialisation and a continued fine-tune, against the unchanged tokenizer at equal items and compute; this is the only place the Prime idea of a dedicated vocabulary is tested, on the record format; or (b) *a real workflow dataset*: an end-to-end handoff (a GBG task handoff is the natural example) with a pretrained consumer, measured as §10.5 did, with a hosted-model zero-shot reference if the owner approves paid calls | H100 (a) or A100 (b) | **success** = the owner's machine-efficiency goal met with a written cost-per-correct and latency table on realistic data; the report's verdict updated |

Costs are expectations at Community/Vast prices with a ×2 reserve inside each ceiling; the ceilings sum to US$1,000.

### 5.3 When a larger corpus or a larger model is worth paying for

- **Corpus**: only when the learning curve is still rising between 2,000 and 8,000 items by more than 3 points in the arm of interest, or when a new item family is added (nesting, long labels, a second task). Flat curves mean more of the same data buys nothing; the study's §10.4 showed that targeted additions (the record-count range, the identifier shapes) move results and blind scaling does not.
- **Model**: only when the small model fails the task (below 90% in every arm at 8,000 items), when the contrast between formats is within the noise band at 0.6B and 1.7B (a larger model can separate them or erase them, both informative), or when the deployment target is larger. Never before stage 2.

### 5.4 Success criteria, fixed before any run

The project's machine-efficiency claim is **established** when all of these hold on a pretrained model: (1) accuracy tied or better than compact JSON and controlled English at equal items on in-range panels, on three seeds; (2) cost per correct answer at most 0.85× JSON with the legend amortised and at most 1.0× with the legend counted per request; (3) the result reproduced on a second tokenizer family or explained as tokenizer-specific; (4) a measured latency advantage under vLLM at batch 32 consistent with the token ratio. Anything less is reported as what it is.

### 5.5 Stop / change-direction rules

- **G0 fails on accuracy** (a pretrained model is better with JSON by the pre-registered margin): stop the pretrained line. Sylang's demonstrated niche is then from-scratch or tiny models, where the byte-model results already stand, and custom-tokenizer models, which are a research programme; the remaining budget is not spent on fine-tuning, and the owner decides between the byte-model pipeline on the existing server (free) and stage 3(a) as a standalone question.
- **G0 fails on cost only** (tied accuracy, cost ratio above 0.85× because the legend dominates at small K): the format is viable only for batched or cached use; stage 1 proceeds with the legend cached and K ≥ 8, and the pilot report says so.
- **G1 fails**: the gain does not survive realistic data; report, keep the codec and the handoff pattern for the cases that work (unique short identifiers, K inside the trained range), and stop.
- **Any stage overruns its ceiling** before its gate can be read: stop, report what was measured, and do not top up.

---

## 6. The two-token record-count line: a fair test, not an adoption

**Status.** `sylang2c` (a one-line record count before the record lines, +2.3 Qwen 3 tokens) was added after the decisive results were seen, because every arm with an explicit count cue (the v1 header's record count, English's numbered records) extrapolated to K 13–16 better than the bare format on every learned seed. One `sylang2c` seed learned (54,000 items, 50% at K 13–16 at its dev-selected checkpoint against 30–38% for the bare format) and one never left the plateau. That is a hypothesis with a direction, not a result, and the §10.6.5 tables say so.

**Why a naive test would fool us.** The final checkpoint of the learned `sylang2c` seed scores 78% at K 13–16 while its held-out-wording score falls to 45%; picking checkpoints or seeds by the long panel would "confirm" the cue by selection. The test below never touches its final panel until the end.

**Design (byte model, cheap; the same design transfers to the pretrained pilot as an added factor on two arms).**
- Arms: `sylang2` and `sylang2c`, identical items per seed; the count line is the only difference. A third arm `sylang2x`, a count line with a *wrong* count (uniformly off by ±1..3), trained and tested the same way, tells whether a model uses the number or merely the extra line; it costs one more run per seed.
- Training range K 1..12 (the `all` corpus), 80,000 items, the §10 recipe, five seeds per arm (the escape rate makes fewer seeds uninformative; a byte-model run is about 1.3 CPU-hours, so 15 runs are a night on the four cores).
- Selection: dev panels only, as before, **plus** a `dev_long` panel at K 13–14 on dev identifiers for the one decision this test needs (which checkpoint to read); the long panels on test identifiers are not read until every run is finished.
- Final test, read once: `long` at K 15–20 (never used for selection, split into 15–16, 17–18, 19–20 to show the profile), `full` at K 15–20, and the in-range panels to confirm nothing was lost.
- Criterion, fixed now: the count line *helps* if, paired per item on the learned seeds, `sylang2c` − `sylang2` on `long` (K 15–20) has a 90% lower bound above +5 points **and** `sylang2x` is not within 3 points of `sylang2c` (the model uses the number); it is *noise* if the interval contains 0 or `sylang2x` matches `sylang2c`; in-range panels must be within 3 points or the change is rejected regardless. If fewer than three seeds learn in either arm, the test is inconclusive and is reported as such, not extended until it works.
- Report the escape rate of each arm beside the result; a count line that changes the escape rate is a finding in itself.

**Cost.** About 20 CPU-hours on the existing server, no money. If the pretrained pilot is approved, the count line rides as a factor on the `sylang2` arm only (two cells), with the same panels and criterion, for a few GPU-minutes.

---

## 7. The history squash and force-push, explained precisely

**What is in the history now** (scanned with the owner's private identifier list as whole tokens; counts only, no values reproduced):

| material | where | commits | sensitivity |
|---|---|---|---|
| four text files containing the owner's synthetic fixture identifiers: `experiments/copy_probe/results/real_pipeline.jsonl` and `balanced12.jsonl` (per-example evaluation rows with identifiers), `experiments/copy_probe/results/patched_copy_worker_scores.json` (a scorer output with expected answers), and the research note's prose | the first 28 commits of the branch, `1741ff6` (2026-10-13) to `f3e9734` (2026-10-14); removed from the tree at `7081c78` | 28 of 68 | fixture content from the uploaded packet: identifier strings and scorer rows that the owner asked to keep out of shared artifacts. Not credentials, not personal data, not packet source code. |
| one synthetic identifier of the public reference replica that coincided with a packet binding identifier | commits up to `89e5e86` (2026-10-15), where it was relabelled | 46 | the same string; its presence in the replica files is a coincidence of two three-character alphabets, and the replica rows themselves are public synthetic data |
| two pre-existing images (`images/token-reduction.png`, `images/tokenization-process.png`) that match an identifier token inside their compressed bytes | every commit, including `main` before this branch | 68 | noise: byte coincidences in PNG data that predate the packet; nothing to clean |
| packet source code, credentials, paths, the owner's private strings, checkpoints | none found in any commit by the verbatim-overlap and credential scans run before each push | 0 | – |

The current head is clean: the only matches are the two images.

**What a squash and force-push would do.** Replace the branch's 68 commits by one commit (or a handful) whose tree equals the current head; push it with `--force-with-lease` to the same branch name; the pull request would then show only the new commit(s). Reviewers lose the per-commit narrative (the deviation log in the report keeps the scientific record, and the task checklist keeps the operational one).

**What it would not remove.**
- *GitHub's object store*: the old commits stay reachable by their SHA for a time (GitHub documents that unreachable objects are retained and may be served until a garbage-collection pass; removing them for good requires a request to GitHub Support with the old SHAs). The PR's own event timeline keeps references to the old commits.
- *Clones and forks*: the repository is public; anyone who cloned or forked since 2026-10-13 keeps the old history. There is no evidence either way; the branch is a draft PR on a small repository.
- *The owner's own copies*: the packet itself, the Netcup paths and the local checkouts named in the packet's provenance file, which were never in this repository.
- *The images*: nothing to remove.

**Alternatives.** `git filter-repo` (or `git filter-branch`) can rewrite only the four paths out of the 28 commits and the one identifier out of 46, keeping every commit otherwise; it is more work, produces the same force-push and the same residue on GitHub, and keeps commit granularity. A squash is simpler and is what the earlier assessment recommended as the smallest cleanup.

**Recommendation.** Because the material is fixture content rather than a secret, the cleanup is a tidiness-and-courtesy decision, not an emergency: squash when the owner is ready to lose the per-commit history, follow with a Support purge if the owner wants the SHAs gone, and do nothing about the images. Until then the draft PR stays as it is. **No history rewrite has been performed; it needs the owner's explicit approval, which this document does not have.**

---

## 8. Decisions needed from the owner, and the recommended next action

**Recommended next action.** Approve the US$100 pilot of §4 as written (model Qwen3-0.6B, Runpod prepaid, the eight cells, the pre-registered gate G0). It is the one experiment that can tell whether the project's main uncertainty, a pretrained model's response to the format, goes the project's way, and it costs about a day of work and tens of dollars. Before it runs, the trainer and evaluation loop are written and smoke-tested on the existing server at no cost; §6's count-line test can run on that server in parallel, also at no cost.

**Decisions.**

1. **Fund and run the US$100 pilot** (yes / no / change the design). Nothing is rented without this.
2. **History cleanup**: squash and force-push this branch (and optionally ask GitHub Support to purge the old commits), or leave the history as it is. The material is fixture content, not a secret; the choice is about tidiness and the per-commit record (§7).
3. **A hosted-model zero-shot reference** (a few dollars of paid calls on a Qwen-family or Claude model, with the legend) as part of the pilot report, or not. Not required for G0.
4. **The answer format**: whether the pipeline's answer may change from the JSON list of pairs to the bare `subject object` list if the pilot shows it is tied on accuracy (it is a parser change on the receiving side and the largest output-side saving measured).
5. **The count line**: run §6's free byte-model test on the server now (about 20 CPU-hours), or fold it into the pilot as a two-cell factor.

GBG remains a later integration example only; nothing above depends on it.

---

## Appendix A. Prices, sources and anchors (retrieved 2026-10-16 unless dated)

**Runpod** (official pricing page https://www.runpod.io/pricing; docs https://docs.runpod.io/pods/pricing, https://docs.runpod.io/references/billing-information). On-demand per GPU-hour, Community / Secure: RTX 4090 24 GB US$0.34 / 0.89; RTX A6000 48 GB 0.33 / 0.59; RTX 6000 Ada 48 GB 0.74 / 0.99; L40S 48 GB 0.79 / 1.09; A100 80 GB PCIe 1.19 / 1.79; A100 80 GB SXM 1.39 / 1.79; H100 80 GB PCIe 1.99 / 2.89; H100 SXM 2.69 / 3.99. Community Cloud is closed to new hosts (https://docs.runpod.io/pods/choose-a-pod), so its stock is shrinking; several Community cards showed out of stock on 2026-10-16 (getdeploying.com). Spot/interruptible pricing could not be verified for 2026 (not on the pricing page or the Pod pricing docs; a blog updated 2026-09-13 still describes it). Storage: container disk US$0.10/GB-month while running; volume disk 0.10 running, 0.20 stopped; network volume 0.07/GB-month under 1 TB; no ingress or egress fees; per-second billing; at least one hour of credit needed to deploy; pods stop at zero balance. Prepaid credits, minimum card deposit US$10, auto-pay opt-in (https://www.runpod.io/blog/manage-runpod-account-funding, 2026-10-10); default spend limit US$80/hour.

**Vast.ai** (https://vast.ai/pricing is rendered client-side and shows no numbers; docs https://docs.vast.ai/billing, https://docs.vast.ai/pricing, https://docs.vast.ai/documentation/reference/faq/rental-types). Prices are set per offer by the host: GPU US$/h billed per second while active; storage US$/GB-h billed while the instance exists, including stopped; bandwidth US$/TB for upload and download. Secondary snapshots of the cheapest on-demand listings: RTX 4090 US$0.37/h (computeprices.com, 2026-10-14), A100 80 GB PCIe 0.94, L40S 0.74, H100 SXM 3.51 (1×); a July-2026 marketplace snapshot gave RTX 4090 min–median 0.14–0.35 (usagepricing.com). Interruptible bids are not published by any tracker; Vast's own text says 50–80% below on-demand with pre-emption. Prepaid credits, minimum deposit US$5, auto top-up opt-in; at zero balance instances stop and storage keeps accruing until a grace period ends. Verification tiers: unverified, verified, datacenter.

**Alternatives.** Lambda Cloud on-demand (https://lambda.ai/pricing, ex-VAT): H100 SXM US$4.29 (1×), H100 PCIe 3.29, A100 40 GB 1.99, A6000 1.09; per-minute billing, no spend cap documented. Google Colab: Pro US$9.99/month with 100 compute units, Pro+ US$49.99 with 500 (secondary, 2026-09); measured burn about 1.2 CU/h on a T4 and 5.4 CU/h on an A100 40 GB (secondary, 2026-03); premium GPUs subject to availability. Kaggle: a floating weekly GPU quota of about 30 hours, 12-hour sessions, T4 (and formerly P100) cards (official pages are client-rendered; from search snippets). Free tiers (Lightning AI, Paperspace, Hugging Face ZeroGPU, Modal's monthly credit) exist but are too small or too restricted for a 100-run sweep; they are fine for the smoke test.

**Models** (Hugging Face model cards and `config.json`): Qwen/Qwen3-0.6B (0.6B, 151,936-token vocabulary, Apache-2.0, 32k context), Qwen/Qwen3-1.7B (1.7B, same), Qwen/Qwen2.5-0.5B and -1.5B (Apache-2.0), HuggingFaceTB/SmolLM2-360M and -1.7B (49,152-token vocabulary, Apache-2.0), meta-llama/Llama-3.2-1B (community licence, gated), google/gemma-3-1b (Gemma terms, gated, 262k vocabulary), google/byt5-small (byte-level encoder-decoder, Apache-2.0; several times slower per byte). Estimated LoRA memory in bf16 at 1,024 tokens on a 24 GB card: 0.6B about 5–8 GB at micro-batch 4; 1.7B about 9–14 GB.

**Throughput anchors.** Llama-3.2-1B LoRA, 2,000 Alpaca items × 2 epochs at 512 tokens, one A10G: 4.17 min (LitGPT config hub). SmolLM2-1.7B LoRA r=16, 20,000 SQuAD items, one epoch, one L40S: 8 min 45 s (lora-speedrun, 2026-07). Llama-3.2-3B LoRA packed 4,096 tokens on one A100: 8,210 tokens/s (torchtune). Qwen2.5-1.5B QLoRA on an 8 GB RTX 4060: 500 tokens/s (arXiv 2509.12229). No directly measured RTX 4090 figure for a ≤ 2B LoRA was found; the pilot's estimates assume 2–3× an A10G, which is conservative for a 4090.

**Not verified** (and therefore not relied on): Runpod spot availability, Vast live medians and interruptible bids, official Colab prices, Kaggle's current card roster, any RTX 4090 LoRA throughput.
