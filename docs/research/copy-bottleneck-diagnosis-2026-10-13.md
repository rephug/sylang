# Why the byte-level model formats answers but does not copy identifiers

**Date:** 2026-10-13 · **Author:** Claude (independent investigation)
**Code:** [`experiments/copy_probe/`](../../experiments/copy_probe/) · **Numbers:** [`results/SUMMARY.md`](../../experiments/copy_probe/results/SUMMARY.md) · **Patch:** [`patches/README.md`](../../experiments/copy_probe/patches/README.md)

Evidence labels:

- **[R]** Reported in the uploaded packet `sylang-source-and-results-20261013.zip` (journals, scores, reviews). Read here, not re-run.
- **[M-real]** Measured here with the project's own `tiny.py`, `decode.py` and `binding.py`, its own rows and schedules, and the trainers' exact loss, optimiser and clipping. This uses `real_pipeline.py` under PyTorch 2.6.0+cpu, the version of the native runs. The warm-start checkpoint is not in the packet, so these runs start from `tiny.build_model` under the recipe seed. That difference is itself one of the comparisons (§2.3).
- **[M-ref]** Measured with the independent reference model in `experiments/copy_probe`.
- **[I]** Inference.

No project checkpoint was read or replaced, no model was downloaded and no API was called. CPU PyTorch 2.6.0 and 2.14 were installed into scratch virtual environments. **[M-local]** below marks runs on the packet's own model and code, done locally with synthetic data (`followup.py`, `language.py`).

## Current answer (latest first)

0. **Study v2.1, K ≤ 8 (§9), three seeds.** With relative positions (ALiBi) and a cosine learning-rate decay, a header-free Sylang encoding (`compact`) reaches **100%** on every panel (unseen identifiers, unseen feature combinations, both, similar identifiers, reordered records, swap, flip) on 3 of 3 seeds, after a median of 26,000 training items (1.5 M Qwen 3 tokens); controlled English under the same recipe reaches 96–99% after 58,000 items (5.4 M tokens). The current encoding's 35-byte header costs 15 Qwen 3 tokens per request (−21% tokens when removed; `compact` is cheaper than English at every K, `codes` only from K = 2) and, for the byte model, 1.7× the training items with one seed in three never learning. The COPY-96 initialisation does not help at K ≤ 8. The held-out-combination weakness is explained (absolute positions, §9.3.2) and gone under relative positions.
1. **English vs Sylang (§8.3).**
   - **With the packet's learned absolute position table**, `codes` learns selection and English does not. The cause is that `codes` puts its fields at fixed byte offsets. The same symbols at English-like offsets (`codes_jitter`) lose most of the advantage. The failing models copy the identifier's digit but not its letters.
   - **With relative positions (ALiBi, one change)**, offsets stop mattering. Natural English then learns as fast as `codes` or faster:
     - K = 1: 16,000 items to 90%, against 40,000 (2 seeds each);
     - K = 1–4: tied per item (56,000 against 48,000, median of 2 seeds; seeds vary 2×), with `codes` using fewer bytes because its items are shorter.
   - **What survives for `codes` is compression:** 22% fewer prompt bytes and 0.74× prefill FLOPs at K = 1–4 (§8.4). At K = 1 it saves nothing: the batch header makes it 9 bytes longer.
   - **The padded-English and `words` arms are not usable.** Their models read polarity from the filler (cue-conflict 0%), so natural English is the English arm.
2. **Selection fine-tuning (§8.5).** Under the packet's recipe (learned positions, K = 1), a COPY-96 initialisation makes fresh-identifier selection learnable and reliable.
   - Natural English: 3 of 3 seeds, 96% held-out; fresh manages 1 of 3.
   - Filler English: 85% held-out; fresh and memoriser initialisations stay at 0%.
   - This is a learning benefit, not zero-shot transfer.
   - Two limits: held-out relation/tense combinations stay at 44–65% under learned positions, against 91–100% for ALiBi English; and the benefit did not carry over to K = 1–4 at LR 1e-3 (one seed).
3. **COPY (§0.2).** Breadth × initialisation interaction: only a memorising initialisation with a narrow identifier set fails. 24 or more balanced identifiers fix it from any initialisation, consistent with the native 432/432.
4. **No implementation defect (§1, §2.1).** All 11 runtime probes pass on the real code. The original binding collapse is a plateau: a 339-byte constant prompt at 192 × 4 updates.
5. **Next (§5.2).**
   - **N1:** the comparison natively, with relative positions, natural English, K up to 8, 3 seeds and a pre-set cost/accuracy rule. A token-count check under the target tokenizer comes first.
   - **N2:** COPY-96 initialisation for the 64-example selection task, scored on fresh identifiers.

---

## 0. Update after the native follow-up (same day)

### 0.1 What the native runs showed [R]

- **Warm start + 96 identifiers copies.** Warm start `efb2…`, 96 balanced `a–f` training identifiers, 192 updates × 4 rows: COPY passed 96/96 seen, 12/12 on the legacy held-out set (previously 2/12) and 108/108 on fresh held-out identifiers. NONE passed 216/216.
- **Neither checkpoint selects zero-shot.** On the original 48 selection checks, both the warm start and the new COPY checkpoint scored 0/48. Neither produced valid JSON.

### 0.2 Reconciliation: a breadth × initialisation interaction, not "the warm start" [M-local]

My earlier §1.3 attributed the 12-identifier COPY failure to the warm start. The native result shows that this is incomplete.

I reran the question as a matrix on the packet's own model (`followup.py copy-suite`, 3 seeds per cell):

- **Breadth:** 12, 24, 48 or 96 balanced training identifiers. The sets are nested and contain the legacy 12. The 96-set has every letter 16 times at every position and all 36 letter pairs for every position pair.
- **Initialisation:** fresh, or a **surrogate memoriser**. The real `efb2…` checkpoint is not available locally. The surrogate was trained on the packet's own 16-case memorisation fixture: 676/684-byte prompts (the same lengths as the original phase), 200 updates, batch 2, LR 3e-4.
- **Fixed held-out panel:** legacy 12 + 108 fresh identifiers.

| Initialisation | Train ids | Updates | Exposures per id | Legacy-12 held-out | Fresh-108 held-out | Held-out errors that are training spellings |
|---|---|---|---|---|---|---|
| surrogate memoriser | 12 | 192 | 32 | **8% (1/12 every seed)** | 12% | **73 of ~107** |
| surrogate memoriser | 24 | 192 | 16 | 100% | 99% | 1 |
| surrogate memoriser | 48 / 96 | 192 | 8 / 4 | 100% | 100% | 0 |
| fresh | 12 | 192 | 32 | 97% (92–100) | 92% (80–100) | 5 (0–12) |
| fresh | 24 / 48 / 96 | 192 | 16 / 8 / 4 | 100% | 100% | 0 |
| either | 96 | 1,536 | 32 | 100% | 100% | 0 |

Reading:

- **Only one combination fails: a memorisation-trained initialisation with a narrow identifier set.** The surrogate reproduces the native pattern: 2/12 → 1/12 at 12 identifiers, and full success at 96. Its held-out errors are overwhelmingly training spellings.
- **Breadth is the robust fix.** 24 balanced identifiers already suffice even from the memoriser.
- **The memorisation signal is far above chance.** A wrong but valid identifier falls in the training set by chance |train|/215 of the time: 5.6% at 12 identifiers. The surrogate's wrong held-out answers are training spellings in 218 of 317 cases (69%). Fresh initialisation makes far fewer errors, 27 in all, of which 15 (56%) are training spellings. Raw counts are not comparable across breadths, because the chance rate rises to 45% at 96.
- **Fresh initialisation is a weaker, sufficient fix at 12 identifiers.** It mostly copies, but one seed emits 12 training spellings.
- **Repetition is not the problem.** 96 identifiers × 32 exposures each (1,536 updates) is perfect. What drives memorisation is the narrow set, not repeated exposure to it.
- **One confound remains.** At a fixed 192 updates, breadth and exposures per identifier move together. The 96 × 1,536 arm shows high exposure is harmless at high breadth. A 12-identifier × 4-exposure arm (24 updates) learned nothing from either initialisation, because 24 updates is too few overall, so it cannot separate the two.

Fresh initialisation helped my small-data runs but is not required: **breadth matters more than initialisation.** The real-checkpoint arms of this matrix belong on the native side (§0.4).

### 0.3 Zero-shot selection failure is expected

The COPY training distribution has nothing in common with the selection task:

- 8-byte prompts against 359/367-byte prompts;
- `a–f` letters against letter-digit-letter identifiers;
- no JSON, polarity or roles.

The 0/48 says nothing about whether COPY pre-training helps *learning* selection. That is the native team's proposed next question.

### 0.4 Coordination: which runs belong where

| Question | Where | Status |
|---|---|---|
| Real warm start × 12 / 96 identifiers, extra seeds | native (needs `efb2…`) | proposed: 2 more schedule seeds per arm, to put intervals on 2/12 and 432/432 |
| Fresh vs surrogate × breadth 12/24/48/96 × exposure | local | **done** (above) |
| COPY-init vs original-init on the *original* 64-example selection task | native (needs both real checkpoints) | the native team's proposed next run; not duplicated here |
| Initialisation (fresh / COPY-pretrained / surrogate) on length-matched short-instruction selection with fresh identifiers | local | **done**: COPY-96 learns, fresh and surrogate do not (§8.5) |
| English vs Sylang, matched budget, identifiers and lengths | local | **done** for K = 1 and K = 1–4, both position schemes (§8.3) |

### 0.5 Length matching: the correction applied, and what each attempt cost

The owner pointed out that random filler alone can leave the class predictable from length. Where negation costs bytes (English "not", Sylang words "negative"), every way of equalising length moves the information somewhere else. Each variant was audited before any comparison run (`language.py audit`, held-out lookup classifiers):

| Design | What it equalises | Where the information went | Verdict |
|---|---|---|---|
| Stratified sampling within each prompt-length cell | Exact length distributions | Features: relation, tense, aspect and evidence predict polarity **70.8%** (chance 50%); evidence alone 60.6% | Rejected |
| Filler at the end of each positive record | Record and prompt length | A "`~` at end of line" flag: a filler-only regex scores **100%** on English and `words` versus 19–33% on `codes` (found by an independent review) | Rejected |
| **Filler in the negation slot** (adopted): `contain ~~~~~~~~~past` vs `contain negative past`; English `is ~~~~containing` vs `was not containing` | Record and prompt length; layout; feature distributions | A local marker inside the negation slot. That is the same position and kind of cue as `codes`' letter case (`s` vs `S`), but a different symbol from natural English | **Adopted**, with two checks (below) |

Two checks measure what the adopted filler does, instead of assuming it is harmless:

- **Cue-conflict panel.** The same test items are rendered with the filler inverted: the shorter form is natural, and the longer form carries the filler in its slot. A model that reads "not"/"negative" keeps its accuracy; one that keys on `~` collapses.
- **`english_natural` arm.** Controlled English with no filler at all. Its length does carry polarity information; it measures what the filler changes about learning.

Audit of the adopted design over 40,000 training items. Lookup classifiers were fit on one half and scored on the other; chance is 50% per record, and K alone predicts the negative count 43.7% of the time:

| Cue | english | english_natural | words | codes | codes_eqlen | codes_jitter |
|---|---|---|---|---|---|---|
| Mean prompt bytes as trained | 177.8 | 171.7 | 202.1 | 133.2 | 212.8 | 171.3 |
| Negative count from prompt length + K | 43.0% | **49.3%** | 42.8% | 43.5% | 43.0% | 42.6% |
| Record polarity from line length | 50.0% | **70.3%** | 49.9% | 49.7% | 50.0% | 50.0% |
| Record polarity from features | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| Record polarity from filler presence | 100% | 49.7% | 100% | 49.7% | 49.8% | 49.8% |

The matched arms carry no length signal. Uncontrolled English does, which is the owner's point measured.

`codes_eqlen` pads each `codes` record at line end to the width of the matching English slot line. Its total prompt length therefore tracks English, which separates a sequence-length effect from a vocabulary effect. The padding is the same for both polarities, so it carries no class information (49.8%).

Raw numbers: [`results/language_audit.json`](../../experiments/copy_probe/results/language_audit.json).

---

## 1. Diagnosis

**1. There is no implementation defect in the paths that could cause this. [M-real]**

Eleven runtime probes pass on the real model, decoder and loss under PyTorch 2.6.0:

- causal masking and visibility, on both the training path and the evaluation (fused fast) path;
- train/eval kernel parity: maximum logit difference 2.7e-5, same argmax;
- gradient reach to every prompt byte;
- parameter and optimiser coverage;
- label alignment, including the `x = seq[:-1]` convention;
- strict checkpoint round trip;
- decoder consistency;
- padding invariance.

The static review's one open item, train/eval kernel parity, is now measured.

**2. The binding failure is a plateau caused by a long constant prompt at a small budget. [M-real] [R]**

- **The collapse reproduces from a fresh initialisation.** With the real data and the original 192 updates × 4 rows, the collapse appears in 3/3 seeds: one constant answer (a single training identifier in both roles), 0.00 bits of prompt dependence, swap TV 0.00, and 100% empty answers.
- **The journals show the same plateau.** In the 8× weighted run, first-identifier loss sits at **2.26 nats** from update 80 to update 191, with 8.5% accuracy. ln 8 = 2.08 is the loss of a guess spread over the eight possible initials. The model learned *which initials exist*, not *which one is in the prompt*.
- **Any one of three changes escapes the plateau** (2/2 seeds each):
  - more updates on the same data;
  - a larger batch with LR 1e-3;
  - the same records with the instruction shortened, at the original 192 updates.

| Same data and init; one change | Seen pairs | Unseen pairs |
|---|---|---|
| none (as executed: 367-byte prompt, 192 × 4) | 0% | 0% |
| 1,536 updates | 100% | 100% |
| batch 16, LR 1e-3, 768 updates | 100% | 100% |
| **45-byte instruction in place of 339 constant bytes, 192 updates** | **100%** | **97%** |

**3. The COPY run's held-out failure needed the warm start *and* a narrow identifier set. [M-real vs R]** *(Revised in §0.2: the native 96-identifier result shows breadth alone fixes it.)*

- **Fresh initialisation copies.** The exact COPY packet (same 24 rows, schedule, optimiser and 192 updates), started from a fresh initialisation, copies held-out identifiers **12/12, 12/12 and 10/12** in three seeds. The warm-started run copied **2/12**.
- **A patched run of your own trainer confirms it.** Running `copy_train.worker` itself with the one-line initialisation change (admission stubbed) meets the packet's own strict gate: held-out COPY **12/12** with all 36 held-out bytes correct, NONE 12/12, all 10 predeclared contrasts passed, and `strict_success: true`. Training took 192 updates in 2.7 s ([`results/patched_copy_worker_scores.json`](../../experiments/copy_probe/results/patched_copy_worker_scores.json)).
- **The warm start is a memoriser.** According to the static review, it is the final state of a phase that memorised 16 long English examples.
- **Its failure pattern matches.** It emitted training spellings for 6 of the 12 held-out identifiers, each a training identifier with the same first letter: a lookup prior, not a copy rule.

**4. Neither task, as built, can show copying of *new* identifiers. [M-real]**

- **The binding data has no new identifiers to test.** It uses 8 identifiers, each fully determined by its first character. Train and test differ only in which identifiers are paired. Every binding run trained on them scores **0% on fresh identifiers**, even when it scores 100% on the eight.
- **Prompt length gives the class away.** Every positive (empty-answer) prompt is 359 bytes and every negative one is 367, so the empty-answer score carries no information.
- **Fresh training identifiers do produce copying.** Training on fresh identifiers with the short instruction gives **100% on held-out fresh identifiers** at 1,536 updates (2/2 seeds); at 768 updates 1 of 2 seeds succeeds, and at 192, 0/2.
- **The long prompt needs more.** With fresh identifiers it fails at 1,536 updates × 4 (0/2). At batch 16 / LR 1e-3 they reach 34% (6–62%) at 768 updates and **100% at 1,536 updates** (2/2).

**[I] Short version:** the model and pipeline can copy. The 192-update budget is too small for a 367-byte prompt. The warm start biases short tasks toward memorisation, and the data never asks for a new identifier.

---

## 2. Evidence

### 2.1 Runtime audit of the real code [M-real]

`python real_pipeline.py audit --source <packet>` gives identical results under PyTorch 2.6.0 and 2.14:

| Probe | Result |
|---|---|
| causality, training path and eval fast path (last 96 positions) | PASS: max change before the edit 0.0 |
| visibility at the first-answer position, both paths | PASS: all 369 earlier positions change the logits (min Δ 0.051) |
| train vs eval path on the same input | PASS: max logit difference 2.7e-5, same argmax |
| gradient reach (binding quartet, real loss) | PASS: 1,456/1,456 prompt positions receive gradient |
| parameter and optimiser coverage | PASS: no zero-grad, frozen or unoptimised tensor |
| oracle labels (`x = seq[:-1]`, labels shifted, EOS supervised) | PASS: next-token oracle 0.0; echo oracle 53.7; wrong EOS 6.3 |
| checkpoint round trip with strict load | PASS: bit-identical logits |
| decoder consistency (`decode.complete` vs teacher-forced argmax) | PASS: 4/4 prompts |
| padding invariance (positive 359 B batched with negative 367 B) | PASS: max difference ≤ 3e-5 |

### 2.2 The uploaded journals [R]

The baseline run averages subject/object byte accuracy of **0.70** from update 64 onward. That is about 0.1 on the first byte plus 1.0 on each of the two suffix bytes, which the first byte determines. Loss falls 0.331 → 0.267 → 0.248 over updates 64–79, 80–127 and 128–191: a plateau, not a slow descent.

In the weighted run, the suffix loss keeps falling (0.10 nats at the end) while the first-identifier loss stays flat at 2.26 nats. Both runs end with 24 × `[]` and 24 × the same training identifier in both roles.

In the frozen diagnostic, P(a) = 17.3% at the first identifier byte, although every initial is exactly 12.5% of the training targets. The from-scratch runs here collapse to a different identifier instead. **[I]** So the specific identifier is inherited from the warm start, while the collapse itself is not.

### 2.3 One-factor runs on the real model and data [M-real]

Mean over seeds (min–max). Batch is rows per update, in quartets; all runs are from fresh initialisation. Full table: `results/SUMMARY.md` → `real_pipeline`.

**COPY packet** (`COPY:<id>` → `<id>`, three letters `a–f`; 12 balanced training identifiers, 12 held out):

| Data | Updates | Seeds | Seen COPY | Held-out COPY | Held-out NONE |
|---|---|---|---|---|---|
| the packet's 24 rows (as executed, but fresh init) | 192 | 3 | 100% | 94% (83–100) | 100% |
| the packet's 24 rows | 1,536 | 3 | 100% | 100% | 100% |
| fresh identifier pairs each update (held-out 12 excluded) | 192 / 768 / 1,536 | 3 each | 100% | 100% | 100% |
| *warm start, as executed* **[R]** | 192 | 1 | 100% | 17% (2/12) | 100% |

**Binding** (one English record):

- **Long prompt:** the packet's 367/359-byte prompt, of which 339 bytes are constant.
- **Short prompt:** the same record without its list number, then `[] if positive, else [["subject","object"]].` (45 bytes), for 71/63 bytes in total.

| Training identifiers | Prompt | Batch × updates (LR) | Seeds | Seen-pair checks | Unseen-pair checks | Fresh-id checks | Swap TV |
|---|---|---|---|---|---|---|---|
| the 8 (as executed, but fresh init) | long | 4 × 192 (3e-4) | 3 | 0% | 0% | 0% | 0.00 |
| the 8 | long | 4 × 1,536 (3e-4) | 2 | 100% | 100% | 0% | 1.00 |
| the 8 | long | 16 × 768 (1e-3) | 2 | 100% | 100% | 0% | 1.00 |
| the 8 | short | 4 × 192 (3e-4) | 2 | 100% | 97% (94–100) | 0% | 0.76 |
| the 8 | short | 4 × 1,536 (3e-4) | 2 | 100% | 100% | 0% | 1.00 |
| fresh | long | 4 × 1,536 (3e-4) | 2 | 0% | 0% | 0% | 0.00 |
| fresh | long | 16 × 768 (1e-3) | 2 | 38% (0–75) | 47% (19–75) | 34% (6–62) | 0.93 |
| fresh | long | 16 × 1,536 (1e-3) | 2 | 100% | 100% | 100% | 1.00 |
| fresh | short | 4 × 192 (3e-4) | 2 | 0% | 0% | 0% | 0.00 |
| fresh | short | 4 × 768 (3e-4) | 2 | 50% (0–100) | 69% (38–100) | 59% (19–100) | 0.72 |
| fresh | short | 4 × 1,536 (3e-4) | 2 | 100% | 100% | 100% | 0.99 |
| *the 8, warm start, as executed* **[R]** | long | 4 × 192 (3e-4) | 1 | 0% | 0% | — | ≈ 0.0006 |

Empty-answer checks score 100% in every arm, because length gives them away. Columns are selected-answer exact match. Swap TV is measured at the first identifier byte of the unseen-pair checks.

For models trained on fresh identifiers, the "seen-pair" checks use the eight real identifiers, which are excluded from their training.

Here "fresh ids" means each quartet uses two identifiers drawn from 5,400 letter-digit-letter combinations with independent characters, excluding the eight real ones. "Fresh-id checks" are 16 selected and 16 empty prompts built from a disjoint identifier split.

### 2.4 Reference-model evidence and one correction [M-ref]

The reference model is a 2-layer, width-128 byte transformer with verified wiring. It reproduces the same signature on an approximate English replica: batch 1 × 192 updates gives a constant pair in 12/12 runs, 0.00 bits and swap TV 0.00. It also shows two further things:

- **Copying needs examples, not loss weighting.** Fresh-identifier COPY/NONE is 0% at batch 1 × 192 updates and 96–100% at 3,072. 8× first-byte weighting never rescued 192 updates.
- **Role selection over long, variable positions takes longer.** Two-fact role selection with fresh identifiers reaches 100% by 3,000 × 32 at a fixed position. With a 0–16-byte position jitter it reaches only 32% (6–56%).

**Correction to an earlier draft of this note.** My first 12-identifier arm drew identifiers at random from 6,760 combinations. In 10–12 of the 12 held-out identifiers, some character never occurred at that position in training, so 0% held-out was guaranteed regardless of copying.

With the packet's balanced 12/12 identifiers:

- **COPY/NONE:** the reference copies held-out identifiers 100% (3/3 seeds, at both 192 and 3,000 updates × 32). A small balanced pool *can* test single-identifier copying, and the packet's COPY design was sound; its failure was the initialisation.
- **Two-fact role selection:** the same balanced pool gives **0% held-out exact with 99% first-byte accuracy** (3/3 seeds, 3,000 × 32). The model copies the selected identifier's first byte and completes the rest from memorised spellings. That is the warm-started COPY run's failure pattern, and the binding run's suffix behaviour.
- **Fresh identifiers fix role selection:** 100% held-out.

So selection tasks need fresh identifiers even when the character columns are balanced.

---

## 3. Ranked explanations

| Rank | Explanation | Verdict | Decisive evidence |
|---|---|---|---|
| 1 | **Long constant prompt × small budget → pre-copy plateau** (binding) | **Confirmed** | Reproduced from scratch on the real code and data (3/3 seeds). Shortening the instruction alone fixes it at the same 192 updates (§2.3) |
| 2 | **Memorisation prior × narrow identifier set → memorisation** (COPY held-out failure) | **Confirmed as an interaction (§0.2)** | Surrogate memoriser: 1/12 at 12 ids, 100% at ≥ 24 ids. Native warm start: 2/12 at 12 ids, 432/432 at 96. Fresh init: 97% at 12 ids |
| 3 | **Data that cannot demand a new identifier, with a length-coded class** | **Confirmed as a limit on every result so far** | 0% on fresh identifiers for every model trained on the 8; 359 vs 367 bytes separates classes exactly |
| 4 | **Implementation defect** | **Excluded for the tested paths** | 11/11 runtime probes under PyTorch 2.6.0. Not tested: the native supervisor's config and path selection (`worker.py`) |
| 5 | **Architectural incapacity** | **Excluded** | The unchanged architecture learns both tasks and copies fresh identifiers |
| — | **First-byte loss weighting** | **Does not address the cause** | The weighted run stays on the same 2.26-nat plateau. Weighting changes gradient emphasis, not the information the model must learn to retrieve |

---

## 4. Patch

[`patches/README.md`](../../experiments/copy_probe/patches/README.md) describes the change to the COPY packet's `runtime/training/copy_train.py`. The ready-made diff was delivered privately, because it contains lines of unpublished source.

It adds a plan field `"initialization": "fresh"`, so the choice is bound into the approved plan digest just as the checkpoint pin is. When the field is set, the worker builds the model under the recipe seed and does not load the warm start. Everything else is unchanged: rows, schedule, optimiser, budget, guards and telemetry. The outer `worker.py` still verifies the old checkpoint file's hash, which is harmless.

The identical change applies to `binding_train.py`. There it is necessary but not sufficient (§2.3): the binding data also needs the short instruction or a larger budget. Those are data-builder changes, so they are recommended (§5) and demonstrated by `real_pipeline.py`, not patched.

The builder's caps would block the budget-based fixes. `binding.schedule` limits epochs to 12, and `tiny.validate_recipe` limits steps to 200 and batch to 4. Raising them is a policy decision for you.

---

## 5. Next two experiments

### 5.1 Status of the original proposals

| Proposal | Status |
|---|---|
| **E1**: initialisation × the unchanged COPY packet | **Answered.** Natively: warm start + 96 identifiers gives 432/432 (§0.1). Locally: the breadth × initialisation matrix (§0.2). Breadth matters more than initialisation; fresh initialisation is sufficient, but not necessary. |
| **E2**: copy *new* identifiers in selection, with no length cue | **Answered, with a twist.** With the packet's learned absolute positions, fresh-initialised English selection does not learn within 12,000 × 8 updates: it copies digits, not letters (§8.3.1). It does learn with relative positions (ALiBi) at the same settings (§8.3.2). Under the packet's own recipe (batch 4, LR 3e-4, 6,144 updates), a COPY-96 initialisation makes it learn reliably: 3 of 3 seeds for natural English, against 1 of 3 fresh (§8.5). Length matching uses slot filler, audited in §0.5. The filler itself became a shortcut (cue-conflict 0%), so the next runs use natural English. |

### 5.2 Next experiments (revised after §9)

N1 of the previous revision has been run locally (§9): relative positions, natural English, K up to 8, three seeds, pre-registered criteria. What remains is native and needs things not available here.

**N1′: the same comparison with the deployment tokenizer and real identifiers.** Count Gemma 3 tokens (gated here) for `compact`, `codes` and English on the production identifier distribution; the 3-tokens-per-identifier floor under Qwen 3 is two thirds of a `compact` record, so the byte saving overstates the token saving (§9.3.1). Success: `compact` ≥ 10% fewer tokens than `codes` over the production K mix under both tokenizers.

**N2′: adopt the three recipe changes in the packet trainer and re-run the selection suite.** (1) Drop the batch header or replace it with a 2-byte record count (`compact_count`, §9.3.1); (2) relative positions (ALiBi or RoPE) in `tiny.py`; (3) a cosine LR decay. Each is one change; §9.3.4 gives the expected effect of each. Success, 3 seeds at K ≤ 8: ≥ 97% on the §9.1 panels for `compact`, and English learning in 3/3 seeds. The COPY-96 pre-training stage can be dropped for this task (§9.3.3).

**N3: the only open generalisation question is scale.** Every held-out panel is at parity with iid once an arm learns (§9.3.4), so the next hold-outs should be structural: K beyond 8, more than one negative record of the same subject, and conditional (`if`) records, which the codec supports and this study did not use.

---

## 6. A fair English-versus-Sylang comparison (design; run in §8)

The binding builder already renders the same ASTs in two arms, `english` and `codes` (`binding.render(asts, arm)`, with an AST round-trip check). Build the comparison on that.

### 6.1 Items and controls

- **Same ASTs, same question, same answer bytes in both arms.** Only the record rendering differs. The instruction is short, identical and format-neutral. In the current prompt it is 339 of 367 bytes, which would swamp both length and learnability.
- **Independent identifiers:** fresh per item, uniform, independent characters, and identical across the two renderings of an item. A disjoint test identifier split.
- **Held-out combinations:** relation × polarity × tense combinations and conditional shapes absent from training, plus unseen identifier pairs and unseen identifiers.
- **Length audit per arm:** a length-only classifier and a bag-of-bytes classifier must sit at chance on every answer field. Add neutral filler jitter where needed.
- **Arms:**
  - English;
  - codes;
  - **codes padded to the English byte length** with neutral filler, per item: same structure, no compression;
  - a compact readable baseline, e.g. the tab-word layout from the 2026-10-04 review.

### 6.2 Training and measurement

- Same `tiny` architecture, byte tokenizer and fresh initialisation for every arm. Choose the LR per arm from {3e-4, 1e-3} on a development split; ≥ 5 seeds.
- Train to equal **items seen**, and log bytes processed, wall-clock and FLOPs. Measure prefill and decode latency per query on one machine; answers are format-neutral, so decode steps are equal across arms.

**Separating the two kinds of gain:**

- **Compression gain** = bytes (≈ FLOPs) per item, known before training. It counts only at non-inferior accuracy.
- **Learning gain** = the shift in *held-out exact accuracy vs items seen* (items needed to reach 95%), and the accuracy gap at equal items.
- **Codes vs padded codes** separates them. If the curves coincide per item, any compute saving is pure compression. If codes win per item, shorter context itself helps learning; §2.3 shows prompt length matters a great deal at small budgets.

**Decision rule (pre-register):** codes are worth pursuing only if they are non-inferior on held-out exact accuracy (within 2 points, 95% bootstrap over seeds and items) *and* at least 10% cheaper end to end than the best readable compact baseline, not only than English.

**Caveat:** from-scratch byte-model results do not transfer to pretrained BPE models, where English is privileged and code byte savings mostly vanish in tokens (2026-10-04 review).

---

## 7. Limits

- The warm-start checkpoint was not available. The initialisation comparison sets this investigation's fresh-init runs against one native warm-start run per task; E1 closes that gap.
- From-scratch runs use 1 thread here against 4 natively. With deterministic algorithms, float differences of order 1e-5 between thread counts can change individual greedy bytes, though not the plateau.
- Two seeds per binding arm. Arms that differ by 0% vs 100% are robust to that; the 768-update fresh-identifier rung is not (1/2).
- The reference-model replica approximates a written description. Where it and the real-pipeline runs differ, the real runs take precedence.
- **The language comparison (§8) uses one learning rate (1e-3) for every arm, and few seeds.** Learned positions: 1 seed per arm (K = 1 and K = 1–4). ALiBi: 2 seeds per arm. The COPY-96-start comparison: 1 seed. Selection (§8.5): 3 seeds. Seeds of the same arm differ by up to 4× in updates to 90%, so only effects larger than that are claimed: learned vs ALiBi, fixed vs variable offsets, and COPY-96 vs fresh. The English-vs-`codes` per-item difference under ALiBi at K = 1–4 is not resolved.
- **ALiBi is one relative scheme**, applied to a model trained from scratch. Other schemes (RoPE, relative biases) and pretrained models may behave differently.
- **Cost of the learned-position K = 1 English arms.** Their seeds 1–2 were not run: each costs about an hour, and seed 0 already showed no escape. The ALiBi arms got both seeds instead.
- **Study v2.1 (§9).** Three seeds per decisive arm, one per pilot; one tokenizer (Qwen 3; Gemma 3 gated); a byte model of 0.86 M parameters as used under ALiBi (1.12 M with the unused position table), trained from scratch on synthetic records with three-character identifiers; the `decay` recipe was added after the first `compact` run (logged in §9.2) and its seed 0 was the pilot that chose it (reported with and without seed 0); train seconds come from a CPU shared by 4–7 runs; the compact-vs-codes gap is carried by one `codes` seed that never escaped, so "non-inferior" is certain and "superior" is one seed's worth of evidence.
- Training seconds in §8 were measured with more worker processes than cores. Use bytes processed and benchmark FLOPs as the cost measures; only the serial benchmark's timings are clean.
- The filler in English and `words` is a synthetic device. A result on those arms holds only if their cue-conflict accuracy stays close to iid; `english_natural` is the unpadded reference.

---

## 8. English versus Sylang on the packet's model [M-local]

`language.py` runs the comparison designed in §6 on the packet's own `tiny` model, optimiser and byte tokenizer, with synthetic data and local CPU only.

### 8.1 Design

**Task.** Each item holds K predicate records. The answer is the JSON list of `[subject, object]` for every negative record, in record order, or `[]`. These are the packet's binding target bytes, identical in every arm. Two sizes:

- **K = 1** (`--suite single`): the selection task itself. Planned as 4,000 updates × 8 items, then extended to 12,000 (§8.3). A run stops at a checkpoint from update 4,000 on when that checkpoint and the previous one are both ≥ 97% on iid, dev, the held-out combination, swap and flip. Cue-conflict is not part of the rule.
- **K uniform in 1–4** (`--suite main`): multi-record selection in order. 12,000 updates × 8 items.

**Arms.** The records are rendered by the packet's own codecs:

| Arm | Rendering | Role in the comparison |
|---|---|---|
| `english` | Controlled English, with filler in the negation slot | Readable baseline, length-matched |
| `english_natural` | Controlled English, no filler | What length matching costs or gains; carries the length cue |
| `words` | Sylang batch words, with filler in the negation slot | Sylang layout with readable symbols |
| `codes` | Sylang batch codes; polarity is letter case | The compact candidate |
| `codes_eqlen` | `codes`, each record padded at line end to its English width | Same symbols as `codes` at English length: isolates sequence length |
| `codes_jitter` | `codes` with 0–3 filler bytes before the subject and 0–28 before the object, drawn from the identifiers and record number, never from polarity | Same symbols, with identifier offsets as variable as English's (4 subject and 34 object offsets, against English's 4 and 32; `codes` has 1 and 4): isolates fixed field positions. Added after the first K = 1 results (§8.3) |

**Held equal across arms:**

- the item stream: records, identifiers and their order, for each seed;
- the 68-byte instruction;
- the fresh initialisation under the same seed;
- AdamW, batch 8, LR 1e-3, number of updates;
- the evaluation panels.

**Identifiers.** Three-character letter–digit–letter identifiers are drawn per record from a 5,362-identifier training split. Every evaluated identifier comes from the disjoint split, halved into dev (715) and test (683). Each identifier is used at most once per item.

**Panels:**

- iid test;
- dev;
- a held-out (relation, tense) combination never trained: `contain` + future;
- **swap**: subject and object exchanged in every record;
- **flip**: one record's polarity inverted;
- **cue-conflict** (§0.5).

**Metrics:**

- **Exact match.** Teacher-forced exact is greedy exact. At the end of each run it is checked against the packet's decoder on 32 items.
- **Selected-only exact.**
- **The always-`[]` baseline.** 59% at K = 1 on the fixed iid panel (57 of 96 items are empty; the training stream is balanced); 28% at K = 1–4.
- **Swap margin.** log p(swapped answer) − log p(original answer), given the swapped prompt.
- **Cost per run:** items, bytes and training seconds at each checkpoint.
- **Serial benchmark** (`language.py benchmark`): per-query prefill and decode time and FLOPs, single thread.

**Learning rate.** A sweep at 2,000 updates × 8 (3e-4 against 1e-3, K = 1–4) left every arm on the plateau: 0% selected, and identifier-byte accuracy of 4–8%. It therefore carries no information about which LR suits which arm. All arms use 1e-3, fixed before the main runs. It is the rate that took fresh-identifier binding off the plateau with the long prompt in §2.3 (16 × 1,536 at 1e-3: 100%; 4 × 1,536 at 3e-4: 0%). That a single LR is fair to every arm is an assumption, recorded in §7.

**Plateau evidence already visible.** On the 2,000-update plateau, the English and `words` models already scored 0–8% on the cue-conflict panel, against 28% on iid. Their only learned behaviour, answering `[]`, keys on the filler. This is why the panel exists. A trained model must be checked on it, not assumed to read the words.

### 8.2 Pre-registered reading (written before any K = 1 or K = 1–4 result)

1. **Validity.** An arm counts only if iid exact reaches ≥ 90% in at least 2 of 3 seeds. Arms that stay on the plateau are reported as "did not learn within budget", not compared.
2. **Compression gain** is per-query bytes and FLOPs (benchmark), known before training. It counts only at non-inferior accuracy.
3. **Learning gain** is the median number of training items to reach 90% iid exact, and the accuracy gap at equal items.
4. **Two contrasts separate the gains:**
   - `codes` vs `codes_eqlen` changes only sequence length;
   - `english` vs `codes_eqlen` changes symbols and layout at about equal length.
5. **Reading the words.** An arm's cue-conflict accuracy more than 20 points below its iid accuracy means its model relies on the filler, not the negation word. Its result is then reported with that caveat.
6. **Non-inferiority (indicative at 3 seeds).** `codes` final iid exact is within 2 points of `english`, and so are held-out-combo, swap and flip.
7. **Generalisation.** Swap and flip accuracy below iid by more than 10 points means the model uses item-level shortcuts rather than roles or polarity.

### 8.3 Results

**Deviations from the plan, both made after seeing seed 0 at K = 1 and recorded here:**

1. At 4,000 updates, `english`, `english_natural` and `words` had begun to follow roles but selected nothing yet: 0% selected exact, against 100% for `codes`. Swap margin was +18, +3 and +13 nats against +69. A fixed 4,000-update budget can therefore show only "learned versus not yet". The K = 1 runs were restarted at 12,000 updates, with the same evaluation grid. The LR is constant, so a long run's update-4,000 checkpoint is the 4,000-update result.
2. `codes` and English prompts are about the same length at K = 1 (115 and 112 bytes), so the gap is not compression. The `codes` record puts both identifiers at nearly fixed byte offsets. Under learned absolute positions that is a plausible shortcut to copying. `codes_jitter` was added to test it.

A third deviation came from the results themselves. The learned-position runs pointed at the position scheme, so §8.3.2 adds a single change to the model: ALiBi in place of the learned position table.

Raw lines: `results/language_k1.jsonl`, `language.jsonl`, `language_alibi.jsonl`, `language_alibi_main.jsonl`; tables in `results/SUMMARY.md`. "Object offsets" is the number of distinct byte offsets of the first record's object over 2,000 items. **Seeds: 1 unless stated.** These are single runs with large effects, not estimates with intervals.

#### 8.3.1 The packet's model (learned absolute positions)

**K = 1**, 12,000 updates × 8 items:

| Arm | Prompt bytes | Object offsets | First update with ≥ 90% held-out exact | Selected exact at 4,000 / final | Cue-conflict |
|---|---|---|---|---|---|
| `codes` | 115 | 4 | **2,000** | 100% / 100% | 100% |
| `codes_eqlen` | 147 | 4 | 6,000 | 62% / 100% | 100% |
| `codes_jitter` | 130 | 34 | 10,000 | 0% / 100% | 100% |
| `english` | 112 | 32 | not reached | 0% / 0% | 0% |
| `english_natural` | 109 | 40 | not reached | 0% / 0% | (no filler) |
| `words` | 143 | 29 | not reached | 0% / 0% | 0% |

**K = 1–4**, 12,000 updates × 8 items:

| Arm | Selected exact | Held-out exact (always-`[]` 28%) | Swap margin, nats |
|---|---|---|---|
| `codes` | **86%** | 90% | +96 |
| `english_natural` | 30% | 50% | +78 |
| `english` | 0% | 28% | +25 |
| `words` | 0% | 28% | +34 |
| `codes_eqlen` | 0% | 28% | 0.0 |
| `codes_jitter` | 0% | 28% | +17 |

**What the failing models get wrong** (`language.py probe`, held-out selected answers):

| Model | Identifier digits | Identifier letters | Greedy sample (target → output) |
|---|---|---|---|
| `english`, K = 1 | 100% | 3% | `f1y`,`w0a` → `i1i`,`t0i` |
| `english_natural`, K = 1 | 100% | 4% | `f1y`,`w0a` → `a1a`,`a0a` |
| `words`, K = 1 | 100% | 50% | `f1y`,`w0a` → `f1y`,`n0n` |
| `codes_jitter`, K = 1–4 | 79% | 3% | `f1y`,`w0a` → `t1i`,`t0i` |

The `words` row is the clearest single observation. Its subject always sits at the same offset and is copied exactly. Its object's offset varies, and only the object's digit, the one distinctive byte, survives.

**Reading.** With a learned absolute position table, this model learns to copy a byte when the byte sits at a fixed offset, or when it is distinctive in content (the digit). It has not learned, within 12,000 updates, to copy an ordinary letter whose offset varies. The ingredients separate cleanly:

- **same symbols, variable offsets** (`codes` → `codes_jitter`): about 5× more updates at K = 1, and failure at K = 1–4;
- **same offsets, longer prompt** (`codes` → `codes_eqlen`): about 3× more updates.

English has variable offsets and letter-dense surroundings. It does not learn at K = 1 (0% selected), and at K = 1–4 only partly: natural English reaches 30% selected, filler English 0%.

#### 8.3.2 One change to the model: ALiBi instead of learned positions

`use_alibi` keeps the packet's embedding, blocks, norm and output layer, drops the position table from the forward pass, and adds per-head ALiBi distance biases to the causal mask. Checks: causality; padding and batch invariance (8e-7); train/eval parity (0.0); and the packet's decoder produces the forward pass's argmax.

**K = 1**, 12,000 updates × 8 items, with the early stop of §8.1:

| Arm | First update with ≥ 90% held-out exact (seed 0, seed 1) | Median items (MB) to 90% | Final selected exact | Cue-conflict |
|---|---|---|---|---|
| `english_natural` | 2,000, 2,000 | **16,000 (1.9 MB)** | 100% | (no filler) |
| `english` | 1,500, 4,000 | 22,000 (2.7 MB) | 100% | **0%** |
| `codes` | 4,000, 6,000 | 40,000 (5.1 MB) | 100% | 100% |
| `codes_jitter` | 4,000, 6,000 | 40,000 (5.7 MB) | 99% | 99% |
| `words` | 12,000, 3,000 | 60,000 (9.3 MB) | 90% (79–100) | **0%** |

**K = 1–4**, 12,000 updates × 8 items, 2 seeds:

| Arm | Prompt bytes | First update with ≥ 90% held-out exact (seed 0, seed 1) | Items to 90% (median) | MB to 90% (seed 0, seed 1) | Peak held-out exact | Final held-out / selected exact |
|---|---|---|---|---|---|---|
| `english_natural` | 171 | 6,000, 8,000 | 56,000 | 9.3, 12.4 | 97.7%, 100% | 93% (87–100) / 91% (82–100) |
| `codes` | 133 | 8,000, 4,000 | 48,000 | 9.9, 5.0 | 97.7%, 98.4% | 96% (95–98) / 95% (92–98) |

- **Accuracy:** both arms reach the same peak. English seed 0 ends below its peak (97.7% → 86.7% between the last two checkpoints), which is constant-LR noise.
- **Generalisation:** at the peak and final checkpoints, swap, flip and the held-out combination are within 4 points of iid in every run. While accuracy is still rising they trail iid by up to 10 points.

**Reading.**

- **Offsets stop mattering.** `codes` and `codes_jitter` learn at the same update.
- **The ranking reverses at K = 1.** Natural English needs 40% of `codes`' items and bytes, and is the most consistent arm across seeds.
- **At K = 1–4 they are tied per item within seed variation:** English 48k–64k items to 90%, `codes` 32k–64k, with the same peak accuracy. Each `codes` item is 22% shorter, so `codes` needs fewer bytes to get there: 5.0–9.9 MB against 9.3–12.4 MB. That is a compression gain at equal accuracy, not a learning gain.
- **The filler is a shortcut.** Every filler-arm model (`english` and `words`, both position schemes) scores 0% on the cue-conflict panel. These models read polarity from `~`, not from "not" or `negative`, so those arms say nothing about the languages. `english_natural` is the valid English arm. Under ALiBi it has no absolute positions from which to read prompt length, and its flip panel is 100%.

#### 8.3.3 Against the pre-registered reading (§8.2)

1. **Validity.**
   - Learned positions: only the `codes` variants are valid at K = 1, and no arm at K = 1–4 (`codes` 89.8%).
   - ALiBi: every arm is valid at K = 1 (2 of 2 seeds). Both arms are valid at K = 1–4 (2 of 2 seeds).
2. **Compression.** In natural prompts, `codes` is no shorter than English at K = 1 (115 vs 106 bytes; the batch header costs 35). At K = 1–4 it is 22% shorter (133 vs 172 bytes on average). Per-query FLOPs: §8.4.
3. **Learning gain.** Its sign depends on the position scheme. It is positive for `codes` under learned positions and negative under ALiBi. It is not a property of the language.
4. **Contrasts.** `codes` vs `codes_eqlen` shows that length costs about 3× under learned positions. `codes` vs `codes_jitter` shows that position variability costs about 5× under learned positions and nothing under ALiBi.
5. **Reading the words.** The English and `words` filler arms fail this rule. Their results are reported with that caveat, and `english_natural` replaces English in the conclusions.
6. **Non-inferiority** (under ALiBi).
   - At K = 1, `codes` matches `english_natural` on final accuracy (100% on every panel) but needs 2.5× the items.
   - At K = 1–4 (2 seeds), peak accuracy is equal: 97.7–100% for English and 97.7–98.4% for `codes`. On final accuracy `codes` is within 2 points of English or better (96% vs 93% mean). That makes it non-inferior at equal items and cheaper per item.
7. **Generalisation** (swap, flip and the held-out combination within 10 points of iid).
   - Every model at ceiling passes.
   - Learned-position `codes` at K = 1–4 is 83–91% against 90% iid, so it passes.
   - Learned-position `codes_eqlen` fails on the held-out combination (85% vs 100%).
   - ALiBi `words` had not finished learning (73% vs 92%).

#### 8.3.4 What this means for the Sylang question

- **The packet model's preference for `codes` is real but is not about the language.** It comes from fixed-width fields meeting a learned absolute position table. The same symbols at English-like offsets lose most of the advantage. A relative position scheme, used by most current language models, removes it and lets controlled English learn faster.
- **The compression gain survives, and it is modest at small K.** The batch header is fixed overhead, so the byte saving grows with K. It counts only where accuracy is non-inferior; at K = 1 under ALiBi that costs `codes` 2.5× the training items.
- **For future runs:**
  - use a relative position scheme, or report both;
  - use natural English, not padded English, as the English arm, and control length through the item mix (equal records per K) instead of through filler;
  - compare at K ≥ 2, where the header is amortised.

### 8.4 Inference cost [M-local]

`language.py benchmark`, run serially on one thread with the CPU otherwise idle. It uses 64 held-out queries per arm (16 per K) and *natural* prompts (no filler), which is the deployment cost. The weights do not affect timing.

| Arm | Prompt bytes: mean (K = 1 / 2 / 3 / 4) | Prefill FLOPs vs English | Prefill ms (median) | Decode ms (mean) |
|---|---|---|---|---|
| English (controlled, natural) | 171 (106 / 152 / 193 / 233) | 1.00 | 7.5–8.0 | 189–192 |
| `words` | 190 (132 / 174 / 212 / 244) | 1.13 | 8.1 | 209 |
| `codes` | 133 (115 / 127 / 140 / 151) | **0.74** | **5.4** | **123** |

- **Prefill.** `codes` costs 26% fewer FLOPs and about 30% less time over the K = 1–4 mix. The saving grows with K: in bytes, `codes` is 0.84× English at K = 2, 0.73× at K = 3 and 0.65× at K = 4. At K = 1, `codes` is **longer** than English (115 vs 106 bytes), because of the 35-byte batch header.
- **Decode.** The answers are identical bytes in every arm (18 bytes on average). The packet's reference decoder re-runs the whole prefix for each generated byte, so its decode time scales with prompt length (123 vs 189 ms). A KV-cached decoder would shrink that gap towards the prefill difference.
- **`codes_eqlen` and `codes_jitter`** are training-time controls. They deploy as plain `codes`.

These are byte-model numbers. Under a pretrained BPE tokenizer, the saving must be re-measured in tokens (§5.2).

### 8.5 Selection from different initialisations [M-local]

This asks the native team's question locally: does COPY pre-training help a model *learn* selection, even though it does not transfer zero-shot (§0.3)? `language.py suite --suite selection-init`.

**Setup:**

- **Task:** one length-matched English record (the `english` arm at K = 1), the 68-byte instruction, fresh identifiers, and the panels of §8.1.
- **Recipe:** the packet's, with learned positions, batch 4 and LR 3e-4, for 6,144 updates (24,576 items).
- **Initialisations, 3 seeds each:**
  - **fresh**;
  - **COPY-96**: the local fresh-init model trained on 96 balanced identifiers (§0.2);
  - **surrogate**: the local memoriser (§0.2).
- Each checkpoint's sha256 is verified at run time.

| Initialisation | Seeds reaching ≥ 90% | Held-out selected exact | Held-out identifier letters / digits | Held-out combination | Swap / flip | Cue-conflict |
|---|---|---|---|---|---|---|
| **COPY-96** | 2 of 3 (90%, 67%, 100% selected) | **85% (67–100)** | 92–100% / 97–100% | 59% (52–65) | 88% / 93% | 0% |
| fresh | 0 of 3 | 0% | 2–12% / 97–100% | 44% | 0% / 41% | 0% |
| surrogate | 0 of 3 | 0% | 1–6% / 97–100% | 44% | 0% / 41% | 0% |

All three COPY-96 runs escaped the plateau between updates 3,072 and 6,144. The fresh and surrogate runs show the digit-only failure of §8.3.1.

**The same with natural English** (no filler; `--arms english_natural`), 3 seeds each:

| Initialisation | Seeds reaching ≥ 90% | Held-out selected exact | Held-out combination | Swap / flip |
|---|---|---|---|---|
| **COPY-96** | **3 of 3** | **96% (90–100)** | 55% (50–65) | 94% / 97% |
| fresh | 1 of 3 | 28% (0, 0, 85) | 35% (27–44) | 31% / 60% |

- **The effect does not depend on the filler.** COPY-96 makes learning reliable: 3 of 3 seeds, against 1 of 3 from fresh. Fresh can learn this, but not reliably within 6,144 updates.
- **One caveat.** With learned absolute positions, a natural-English model could read polarity from prompt length, since the SEP position reveals it. The flip panel cannot separate that from reading "not".
- **Every English model that learned under learned positions is weak on the held-out (relation, tense) combination**: 44–65%, whichever initialisation. The ALiBi English models score 91–100% on it (§8.3.2). The unseen phrasing (`will be containing` and similar) moves the identifiers to offsets these models have not learned to read. That is the same absolute-position limitation, seen from another side.

**Reading:**

- **COPY pre-training on broad identifiers helps the model learn selection,** with filler English and natural English alike. That holds even though the COPY prompts share no format with this task (`a–f` identifiers, 8-byte prompts). A memorising initialisation is no better than fresh.
- **Two limits:**
  - The held-out (relation, tense) combination lags at 59%: what it learned about the record wording transfers less well than the copying.
  - Like every filler-arm model, it reads polarity from the filler (cue-conflict 0%).
- **For the native runs:** the planned COPY-init vs original-init comparison on the 64-example task should improve with COPY-96 rather than the 12-identifier COPY checkpoint. It should be scored on held-out identifiers and on a held-out combination, not on training pairs.

**COPY-96 as the starting point for the language comparison** (learned positions, K = 1–4, LR 1e-3, batch 8, seed 0):

| Arm | Fresh start | COPY-96 start |
|---|---|---|
| `codes` | 60% at update 6,000; 90% at 12,000 | **94% at update 6,000**, then 84–92% |
| `english_natural` | 30% selected at 12,000 | **0% selected** at 12,000; swap margin only +11 nats; letters 12%, digits 33% |

The COPY-96 benefit is not general. It halves the time `codes` needs at K = 1–4. It does not help natural English at K = 1–4 with LR 1e-3 and batch 8: that run is slower than fresh. Yet at K = 1 under the packet's own recipe (LR 3e-4, batch 4) it does help English. One seed cannot say which difference matters: the learning rate (which may overwrite the pre-trained copying early), or the task mix. Under the packet recipe at K = 1, COPY-96 does help natural English (above), so the learning rate and task mix are the open variables.


---

## 9. Study v2: encoding, generalisation and learning at K ≤ 8 [M-local]

### 9.1 Pre-registration (v2.1)

v2 was written on 2026-10-14 before any run longer than a 20-update smoke test, and a first wave of 16 runs was started. An adversarial review of v2 then found flaws that the runs could not repair afterwards, so the wave was stopped 45 minutes in, before any run had completed, and this v2.1 was frozen. §9.2 lists every change. Code: `language.py --suite study` at commit `469bf5d`.

**Task.** Items of K records, K uniform in 1..8 in training. Answer: the JSON list of `[subject, object]` of each negative record in record order, or `[]`. The answer depends only on polarity and the identifiers; relation, tense, aspect and evidence never change it, so the held-out panels test whether polarity reading is *invariant to an unseen frame*, not composition. Identifiers are letter-digit-letter from disjoint train (5,362), dev (715) and test (683) splits. Features: relation {see, help, contain}, tense {past, present, future}, aspect {simple, progressive, completed}, evidence {unspecified, direct, reported, inferred}; polarity 50/50 per record. **Held-out combinations**, never in training: (relation = contain, tense = future) and (relation = help, aspect = completed).

**Arms (encodings).** Main comparison with ALiBi in place of the learned position table (§8.3.2): `english_natural` (controlled English, no filler), `codes` (the current batch codes, with the 35-byte header), `compact` (`codes` without the header line; record lines byte-identical, tab-separated, so the contrast isolates the header). The deployment separator is chosen by token count without retraining: a space saves 0.23 Qwen 3 tokens per record (§9.3.1) and is invisible to the byte model. One learned-position reference: `codes`, 1 seed (the fixed-offset prediction of §8.3.1). The filler arms are excluded: their models read the filler (§8.3.2). The learned-position `english_natural` reference is not run: §8.3.1 already shows it does not learn at K = 1–4.

**Learning methods.** Baseline: fresh initialisation, mixed K from the start. Candidates, piloted at 1 seed: `copy96` initialisation, from a COPY-96 model **pre-trained under ALiBi** (`followup.py copy-pretrain --position alibi --updates 1536`; 96 balanced identifiers, 1,536 updates × 4 rows = 6,144 items and 0.08 MB, added to the arm's cost; the native 192-update recipe does not learn under ALiBi, 14/96 seen and 9/108 held-out (`results/copy96_alibi_pretrain_u192.json`), while 1,536 updates give 96/96 seen and 103/108 fresh held-out (`results/copy96_alibi_pretrain.json`)), piloted on `english_natural` and `compact`; `staged` (K = 1 only until the dev K = 1 panel reaches 97% or 3,000 updates, then mixed); `k1mix` (25% of items K = 1 throughout). The memoriser initialisation is not re-piloted: §8.5 found it no better than fresh at 3 seeds. The best candidate by criterion 4 is then run at seeds 1 and 2 on `english_natural` and `compact`.

**Recipe.** AdamW, LR 1e-3, batch 8, up to 10,000 updates (80,000 items), evaluation of every panel at the fixed grid (250 … 8,000) and of the iid panel alone every 250 updates from 1,000. Early stop when two consecutive grid checkpoints from update 4,000 on are ≥ 97% on `dev`, `dev_swap`, `dev_flip` and `dev_reorder`: the test panels never influence stopping. The **dev-selected checkpoint** (earliest grid checkpoint with the highest dev exact) is the one every headline number is read from; the final and the dev-best models are both saved. One thread per run; four runs share four cores; every sequence is checked against the 2,048-byte context.

**Panels** (fixed items, fixed seed, 32 items per K): `iid` = unseen identifiers, trained combinations; `seen` = training-split identifiers, trained combinations; `combo` = training-split identifiers, at least one held-out combination; `both` = unseen identifiers and a held-out combination (the four cells of the identifier × combination 2 × 2); `reorder` = iid items with the record order permuted (K ≥ 2, paired); `similar` = every negative record has a positive partner whose subject or object differs in one character and is itself a test-split identifier (K ≥ 2); `similar_control` = the same items with each partner's varied identifier replaced by a fresh test identifier (paired); `swap` = subject and object exchanged in every record (paired); `flip` = one record's polarity inverted (paired); `dev` and its counterfactuals for stopping and selection only. Every panel reports exact, selected-only (non-empty answers), empty-only and by K; K = 5–8 is the "long" view. Per-item flags are stored for paired analyses.

**Costs.** Items, bytes and Qwen 3 tokens seen to first reach 90% iid exact on the fine curve (a seed that never reaches it counts as infinity; the number of such seeds is reported beside every median); train seconds (caveat: shared CPU); the COPY-96 pre-training's items and bytes added to its arm; the serial inference benchmark per encoding over K = 1..8, with prefill as the cost and the reference decoder's timing labelled as such; token counts of header, records, instruction and answer under Qwen 3 (Gemma 3's tokenizer is gated and unavailable; the 3-tokens-per-identifier floor is tokenizer-specific).

**Frozen criteria.**

1. An arm or method *learns* if iid exact ≥ 90% in ≥ 2 of 3 seeds on the fine curve.
2. Efficiency: primary, the median over seeds of items to first reach 90% iid exact (fine curve); secondary, iid exact at the dev-selected checkpoint; the peak over checkpoints is reported as a descriptive column only.
3. `compact` vs `codes`: at the dev-selected checkpoint, per-item paired differences (compact − codes) on the shared iid, combo, both, similar and reorder panels, averaged over seeds, bootstrapped over items (1,000 resamples): *non-inferior* if the 90% interval's lower bound is above −3 points on every one of those panels. The token saving (criterion "worthwhile" in v2) is settled by accounting (§9.3.1: −21% vs `codes` over the mix) and is not an experimental question.
4. A learning method is an *improvement* over the baseline in the same encoding if, pairing runs by seed, items-to-90% ratios (method / baseline) are ≤ 0.67 in all three seeds or the median ratio is ≤ 0.5, **and** at the dev-selected checkpoint its selected-only exact on combo, on both and on the K = 5–8 rows of iid is within 2 points of the baseline's (paired per item). Anything smaller is reported as a tie.
5. A model is *frame-invariant* if selected-only exact on both combo and both is ≥ 85% at the dev-selected checkpoint.
6. The pilot runs (seed 0) are reused as seed 0 of the decisive runs when their configuration is unchanged. Because the method was chosen on seed 0, the decisive comparison is reported with and without seed 0, and the direction must hold on seeds 1 and 2 alone.
7. Three seeds resolve only large effects; differences under 2× in items-to-90% between arms are reported as ties.

### 9.2 Deviations

1. **2026-10-14, after 45 minutes of wave 1, no run completed.** Stopped and restarted under v2.1 after an adversarial review of v2. Changes: early stopping and checkpoint selection moved from test panels to dev panels; combo and both raised from 16 to 32 items per K; the `seen` cell added; `similar` partners restricted to test-split identifiers and the paired `similar_control` added; the iid fine curve, per-item flags, the context check and dev-best checkpoints added; criterion 4 rewritten (pairing by seed, a 2× rule, and the composition terms) and criterion 5 renamed; the COPY-96 initialisation re-based on an ALiBi pre-trained COPY-96 and piloted on `compact` as well; the memoriser pilot and the learned-position English reference dropped. The aborted runs' partial logs are not used.
2. **2026-10-14, after the first `compact` run completed.** A fourth learning candidate, `decay` (mixed K, cosine learning-rate decay from 1e-3 to 1e-4 over the run), was added because every learned run so far fluctuates by 10–25 points between grid checkpoints after its escape under the constant LR (`compact` seed 0: 95% at 4,000, 71% at 4,750, 91% at 10,000). It is piloted on `compact` seed 0 under the same rules as the other candidates; its primary criterion is still items-to-90% (criterion 4), and its effect on the dev-selected and final accuracy is reported as a secondary observation.
3. **2026-10-14, method choice after the seed-0 pilots.** On `compact` seed 0, `decay` reaches 90% iid at 26,000 items against the baseline's 32,000 (ratio 0.81: a tie under criterion 4's 2× rule) and is the only run at ceiling at K ≤ 8 (100% on iid, combo, both and similar at update 8,000, against 95% dev-selected and 91% final for the constant-LR baseline). On `english_natural` seed 0 the baseline never reaches 90%, so `staged` (60,000 items) cannot be compared to it by ratio. By the pre-registered tie-break (criterion 2's secondary measure), `decay` is the method run at seeds 1 and 2 on `english_natural` and `compact`; `decay` on `english_natural` seed 0 is added to complete its three seeds. The other candidates stay at one seed and are reported as ties.

### 9.3 Results

#### 9.3.1 Tokens under the deployment tokenizer [M-local]

`tokens.py`, Qwen 3 tokenizer (`Qwen/Qwen3-0.6B`; Gemma 3 is gated and was not available), 64 fresh synthetic items per K, the same items in every arm, no special tokens. Prompt = header + records + the 68-byte instruction (17 tokens, identical everywhere). Raw numbers: `results/tokens_qwen3.json`.

| Arm | Header | Per record | of which identifiers | K = 1 | K = 2 | K = 4 | K = 8 | Mix K = 1..8 | Bytes, mix | Cheaper than English from |
|---|---|---|---|---|---|---|---|---|---|---|
| `english_natural` | 0 | 17.0 | 6.0 (35%) | 33.9 | 51.0 | 85.1 | 152.7 | **93.3** | 253 | – |
| `words` | 15 | 11.1 | 6.0 (54%) | 43.1 | 54.2 | 76.4 | 121.0 | 82.0 | 259 | K = 3 |
| `codes` (current) | 15 | 9.1 | 6.0 (66%) | 41.1 | 50.2 | 68.6 | 105.2 | **73.2** | 157 | K = 2 |
| `compact` (tab) | 0 | 9.1 | 6.0 (66%) | 26.1 | 35.2 | 53.6 | 90.2 | **58.2** | 122 | K = 1 |
| `compact` (space) | 0 | 8.9 | 6.0 (68%) | 25.9 | 34.8 | 52.6 | 88.3 | 57.1 | 122 | K = 1 |
| `compact` + count line | 2 | 9.1 | 6.0 (66%) | 28.1 | 37.2 | 55.6 | 92.2 | 60.2 | 124 | K = 1 |

Answers are identical in every arm: 20.3 tokens on average (`[["abc","def"]]` is 10 tokens; `[]` is 1). A leaner answer format (`abc def` per negative record) would cost 6.0 tokens per record instead of 9.1; it is not used anywhere here.

Reading:

- **The header is the cost that decides small batches.** It is 15 tokens, 1.6 records' worth of `codes`. With it, `codes` costs *more* than English at K = 1 (41 vs 34) and breaks even at K = 2. Without it, `compact` is cheaper than English at every K: 23% at K = 1, 41% at K = 8, 38% over the mix.
- **Identifiers set the floor.** Three-character identifiers tokenize as three tokens each in Qwen 3, so 6 of the 9 tokens of a `codes` record are the identifiers, which no encoding of the relation can remove. The relation, tense, aspect and evidence cost 1–3 tokens in `codes` (mean 2.2) against 11 in English.
- **Separator choice is a 2% effect** (space 57.1 vs tab 58.2 over the mix). The trained `compact` arm keeps tabs, so that its record lines are byte-identical to `codes` and the training comparison isolates the header.
- **Byte and token savings differ.** Over the mix, `codes` saves 38% of the bytes but 22% of the tokens against English; `compact` saves 52% of the bytes and 38% of the tokens. The byte model's training cost follows bytes; a deployed model's cost follows tokens.

#### 9.3.2 Why unseen combinations were weak under learned positions [M-local]

`combo_analysis.py` on the K = 1 checkpoints of §8.3 and §8.5: 9 (relation, tense) combinations × 2 polarities × 64 fresh items each, test-split identifiers; the byte offset of each identifier in the rendered prompt is compared with the offsets that occur in 4,000 training-distribution renderings of the same arm; wrong answers are classified by their first wrong byte. Raw numbers: `results/combo_analysis.json`.

| Model (K = 1) | Positions | Trained combinations | Held-out combination | Held-out negatives at a **novel** answer offset | … at a **seen** offset | Held-out errors: polarity / copy / structure |
|---|---|---|---|---|---|---|
| `english_natural`, COPY-96 start, 3 seeds | learned | 93–99% | 52–59% (sel 41–56, empty 61–70) | **0%** (9 of 64 items) | 47–65% | 20–27 / 19–26 / 8–9 |
| `english_natural`, fresh, seed 2 | learned | 95% | 48% (sel 33, empty 64) | **0%** | 38% | 29 / 30 / 7 |
| `english` (filler), COPY-96, 3 seeds | learned | 80–99% | 69–75% (sel 38–50, empty 100) | **0%** | 44–58% | 7–9 / 23–31 / 0–2 |
| `codes`, `codes_jitter`, fresh | learned | 100% | 100% | (none novel) | 100% | 0 |
| `codes_eqlen`, fresh | learned | 100% | 92% (sel 84) | (none novel) | 84% | 0 / 8 / 2 |
| `english_natural`, fresh, 2 seeds | **ALiBi** | 100% | **100%** | **100%** | 100% | 0 |
| `codes`, fresh, 2 seeds | **ALiBi** | 100% | 99–100% | (none novel) | 98–100% | 0–1 |

Reading:

- **Novel positions are never right under learned positions.** A held-out English phrase ("will not be containing") puts the answer at byte positions the model has never answered from; every learned-position English model scores 0 on those items. That is the §8.3.1 mechanism again.
- **It is not the whole story.** At positions seen in training, held-out negatives still score 38–65% against 87–98% for trained combinations. Those errors split about evenly between **polarity** (the unseen wording is not recognised as a negation, and 25–39% of held-out *positives* are answered as if negative) and **copying**. The filler-English models make almost no polarity errors (the filler decides polarity) and fail only on copying.
- **Codes compose.** For `codes` the unseen combination is one new letter pair at a fixed offset; learned-position `codes` and `codes_jitter` score 100% on it. `codes_eqlen` loses 8% to copy errors, because its padding width changes with the unseen code.
- **Relative positions remove the weakness at K = 1.** Both ALiBi English models score 100% on the held-out combination, at novel and seen offsets alike. Whether that holds for K up to 8 and for the second hold-out is what §9.3.3 measures.

**Eval-time shift probe** (`language.py probe`, `results/shift_probe.jsonl`): the same K = 1 iid prompts with 0, 1, 2 or 3 newline bytes prepended, no retraining. A model that reads positions relative to the content is unaffected; one that reads absolute positions is not.

| Model (K = 1) | Positions | Shift 0 | 1 byte | 2 bytes | 3 bytes |
|---|---|---|---|---|---|
| `english_natural`, fresh | ALiBi | 100% | **100%** | **100%** | **100%** |
| `codes`, fresh | ALiBi | 100% | 53% | 1% | 1% |
| `codes`, fresh | learned | 100% | 34% | 14% | 6% |
| `codes_jitter`, fresh | learned | 100% | 63% | 51% | 52% |
| `english_natural`, COPY-96 start (2 seeds) | learned | 98–99% | 39–44% | 35–40% | 32–35% |

Two readings. Every learned-position model collapses with a one-byte shift, which is the position dependence directly observed. The ALiBi `codes` model also collapses, although the ALiBi English model does not: under ALiBi the distance to the BOS token still encodes absolute position, and the fixed-offset `codes` layout invites a solution that uses it, while English's variable offsets force a content-relative one. Fixed layouts are a position shortcut whatever the position scheme; `codes_jitter`, trained on variable offsets, is the most shift-tolerant of the learned-position models.

#### 9.3.3 Learning-method pilots at K ≤ 8, seed 0 [M-local]

All ALiBi, 10,000 updates × 8 items at most, dev-selected checkpoint, panels of §9.1. Items-to-90% is read from the fine iid curve; "censored" means 90% was never reached within 80,000 items. One seed each except where stated; these pilots chose the method for the 3-seed runs (§9.2, deviation 3) and are reported as they are.

| Arm | Method | Items to 90% iid | Dev-selected update | iid exact | iid selected | K = 5–8 | combo sel. | both sel. | similar sel. | Final iid |
|---|---|---|---|---|---|---|---|---|---|---|
| `compact` | baseline (fresh, mixed, constant LR) | 32,000 | 4,000 | 95% | 95% | 92% | 95% | 97% | 99% | 91% |
| `compact` | **decay** (cosine LR to 10%) | **26,000** | 8,000 (early stop) | **100%** | 100% | 100% | 100% | 100% | 100% | 100% |
| `compact` | COPY-96 start (ALiBi-pretrained) | censored (peak 85%) | 8,000 | 85% | 83% | 70% | 83% | 85% | 90% | 84% |
| `english_natural` | baseline | censored (peak 89%) | 8,000 | 84% | 81% | 68% | 78% | 81% | 87% | 77% |
| `english_natural` | **decay** | **58,000** | 10,000 | **96%** | 95% | 91% | 96% | 96% | 99% | 96% |
| `english_natural` | staged (K = 1 until 3,000, then mixed) | 60,000 | 10,000 | 92% | 91% | 85% | 90% | 92% | 95% | 92% |
| `english_natural` | k1mix (25% K = 1 throughout) | 46,000 | 8,000 | 90% | 89% | 81% | 81% | 93% | 92% | 84% |
| `english_natural` | COPY-96 start (ALiBi-pretrained) | censored (peak 80%) | 10,000 | 79% | 76% | 60% | 78% | 79% | 82% | 79% |
| `codes` (header) | baseline | censored (18%) | 10,000 | 18% | 4% | 2% | 2% | 2% | 3% | 18% |

Reading, by the frozen criteria:

- **No method is an improvement on speed (criterion 4).** On `compact`, `decay` reaches 90% at 0.81× the baseline's items, a tie under the 2× rule. On English the baseline is censored, so every method that reaches 90% is "< 1×"; staged, k1mix and decay are within 1.3× of each other.
- **`decay` is the only method at ceiling, on both encodings.** The constant-LR runs all fluctuate by 10–25 points between checkpoints after their escape (`compact` baseline: 95% at 4,000, 71% at 4,750, 91% at 10,000; English baseline: 84% at 8,000, 89% at 9,500, 77% at 10,000). The cosine schedule removes the fluctuation and ends at 100% (`compact`) and 96% (English), and it is the only English run above 90% at K = 5–8 (91%). That is why it went to three seeds (§9.2, deviation 3; results in §9.3.4).
- **The COPY-96 initialisation does not help at K ≤ 8.** Pre-trained under ALiBi so that its weights match the position scheme, it leads the fresh run early (English at update 4,000: 48% vs 15% iid exact, 39% vs 0% selected-only) and then falls behind on both encodings (censored on both; peaks 80% and 85%). The benefit found at K = 1 under the packet recipe (§8.5) does not transfer to the multi-record task at LR 1e-3. The pre-training cost it would add is small (6,144 items, 0.08 MB, 26 s), so cost is not the objection; it simply does not improve learning here. The memoriser start was not re-run (§8.5: no better than fresh at 3 seeds).
- **Curricula help English modestly and consistently** (staged, k1mix: 90–92% dev-selected against 84%), but less than the LR schedule, and they do not reach the ceiling.
- **The header costs `codes` the budget on this seed.** With identical record lines, `compact` reaches 90% at 32,000 items and `codes` never does (18% at 80,000). Seed 1 of `codes` did learn (98%, §9.3.4), so the header delays the escape by a seed-dependent amount rather than preventing it.

#### 9.3.4 Decisive runs: three seeds per arm [M-local]

ALiBi, K ≤ 8, 10,000 updates × 8 items at most, dev-selected checkpoint. Items, MB and Qwen 3 tokens to 90% are read from the fine iid curve (tokens = items × the arm's K = 1..8 mix mean of §9.3.1); a censored seed counts as ∞ in the median. Raw lines: `results/language_study.jsonl`; tables: `results/SUMMARY.md`.

**Encodings, baseline recipe (fresh, mixed K, constant LR 1e-3):**

| Arm | Seeds ≥ 90% | Items to 90%: median (per seed) | MB to 90% | Qwen 3 tokens to 90% | iid exact | iid selected | K = 5–8 | combo sel. | both sel. | similar sel. | reorder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `compact` | **3/3** | **38,000** (32k, 38k, 58k) | 6.0 | 2.2 M | 97% (95–99) | 96% | 94% | 95% | 96% | 98% | 95% |
| `codes` (header) | 2/3 | 66,000 (∞, 66k, 54k) | 12.7 | 4.8 M | 71% (18–98) | 66% | 64% | 65% | 66% | 67% | 68% |
| `english_natural` | 1/3 | ∞ (∞, ∞, 44k) | ∞ | ∞ | 88% (84–90) | 86% | 76% | 82% | 84% | 88% | 85% |
| `codes`, learned positions (1 seed) | 0/1 | ∞ | ∞ | ∞ | 14% | 0% | 0% | 0% | 0% | 0% | 7% |

**Learning recipe `decay` (cosine LR 1e-3 → 1e-4), fresh init, mixed K:**

| Arm | Seeds ≥ 90% | Items to 90%: median (per seed) | MB | Qwen 3 tokens | iid exact | iid selected | K = 5–8 | combo sel. | both sel. | similar sel. | reorder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `compact` + decay | **3/3** | **26,000** (26k, 32k, 26k) | 4.1 | **1.5 M** | **100%** (3 × 100) | 100% | 100% | 100% | 100% | 100% | 100% |
| `english_natural` + decay | **3/3** | 58,000 (58k, 58k, 36k) | 16.7 | 5.4 M | **97%** (96–99) | 96% | 94% | 96% | 97% | 99% | 96% |

Every run's teacher-forced exact flag agrees with the packet's greedy decoder on 32/32 items; the longest sequence was 609 bytes against a 2,048-byte context. Train time to the end of each run on the shared CPU (4–7 single-thread runs on 4 cores, so only indicative): `compact` 1,700–3,500 s, `codes` 3,100–4,100 s, English 5,300–8,800 s.

**Against the frozen criteria (§9.1):**

1. *Learns* (≥ 90% iid in ≥ 2 of 3 seeds): `compact` yes (3/3), `codes` yes (2/3), English baseline **no** (1/3; the other two peak at 89.4% and 89.8%); with `decay`, `compact` and English both 3/3.
2. *Efficiency*: `compact` needs 38,000 items to 90% against 66,000 for `codes` (1.7×, within the 2× tie rule) and against a censored English median; in bytes and tokens the gap is 2.1× and 2.2× against `codes`, because each `compact` item is shorter.
3. *Non-inferiority of `compact` to `codes`* (paired per item at the dev-selected checkpoint, 3 seeds, 1,000 bootstrap resamples): compact − codes = **+25.8** points on iid [90% interval +24.0, +27.3], +25.1 on combo [+23.3, +27.0], +26.7 on both [+25.3, +28.3], +30.5 on similar [+29.5, +31.5], +26.9 on reorder [+25.4, +28.4]. Every lower bound is far above −3: non-inferior on every panel, and in fact superior. The gap is carried by `codes` seed 0, which never escaped; on seeds 1 and 2 alone both arms sit at 96–99% and are indistinguishable. Removing the header never costs accuracy and sometimes rescues the run.
4. *Learning method*: on `compact`, `decay` / baseline items-to-90% ratios are 0.81, 0.84 and 0.45 by seed (median 0.81): **a tie on speed** by the frozen rule (≤ 0.67 in all seeds or median ≤ 0.5). On English the baseline is censored on two seeds, so the ratio is "< 1" there and 0.82 on the third: also a tie on speed as the rule is written. On the composition terms `decay` is not merely within 2 points but above the baseline everywhere (`compact`: 100% vs 94–96% on combo, both and K = 5–8; English: 94–97% vs 76–84%). What the speed rule does not capture is that `decay` turns the English baseline from "does not learn" (1/3) into "learns" (3/3) and lifts both encodings to the ceiling. §9.4 reads it accordingly.
5. *Frame-invariant* (combo and both selected ≥ 85%): every run that learned, under both recipes; `compact` + decay at 100% on both.
6. *Winner's curse*: `decay`'s advantage holds on seeds 1 and 2 alone (`compact` ratios 0.84 and 0.45; English 3/3 ≥ 90% against the baseline's 1/3, with the one comparable seed at 0.82).
7. *2× rule*: the one arm-level difference beyond 2× is bytes and tokens to 90%, `compact` against `codes` (2.1–2.2×) and against English (∞).

**Identifiers versus combinations (the 2 × 2, selected-only exact at the dev-selected checkpoint, mean of 3 seeds):**

| | Trained combinations | Held-out combination |
|---|---|---|
| Training-split identifiers | `seen`: compact 96%, codes 66%, English 83%; with decay 100% / 98% | `combo`: compact 95%, codes 65%, English 82%; with decay 100% / 96% |
| Unseen identifiers | `iid`: compact 96%, codes 66%, English 86%; with decay 100% / 96% | `both`: compact 96%, codes 66%, English 84%; with decay 100% / 97% |

Within every arm the four cells agree to within 4 points: under relative positions, neither unseen identifiers nor unseen combinations are a separate difficulty; an arm either learns the task or it does not. The per-record breakdowns (`results/breakdown_*.json`) say the same: held-out and trained records are included at the same rate (97–100%), positives are excluded at 100%, and 46 of the 47 residual errors are copy errors and one is a structure error; none is a polarity error.

**What did not work:**

- *The COPY-96 initialisation* (ALiBi-pre-trained, 6,144 items of pre-training): censored on both encodings (dev-selected 79% English, 85% `compact`), below the fresh baseline. The K = 1 benefit of §8.5 does not carry to K ≤ 8.
- *The batch header*: 2/3 seeds instead of 3/3, 1.7× the items, 2.1× the bytes, a seed-0 failure; and 15 Qwen 3 tokens per request for nothing the task needs.
- *Learned absolute positions*: `codes` at 0% selected after 80,000 items at K ≤ 8, where it learned K = 1 in 16,000.
- *Curricula* (staged, k1mix): small, consistent gains for English on one seed (90–92% vs 84%), well short of the schedule change.

#### 9.3.5 Inference cost over K = 1..8 [M-local]

`language.py benchmark --k 12345678`, serial, one thread, idle CPU, 16 natural-prompt queries per K (128 per arm); the weights do not affect timing. Prefill is the cost that scales with the encoding; the packet's reference decoder re-runs the whole prefix for every generated byte, so its decode time is reported as such, beside the analytic FLOPs of a KV-cached decoder (one position per byte plus attention over the prefix). Raw numbers: `results/language_benchmark_k8.json`.

| Arm | Prompt bytes, mix (K = 1 / 4 / 8) | Prefill ms, median (K = 1 / K = 8) | Prefill GFLOPs | vs English | Decode ms, reference decoder | KV-cached decode GFLOPs | vs English |
|---|---|---|---|---|---|---|---|
| `english_natural` | 254 (107 / 235 / 411) | 14.4 (5.0 / 31.7) | 0.574 | 1.00 | 797 | 0.0799 | 1.00 |
| `words` | 261 (135 / 244 / 391) | 14.1 (6.6 / 28.2) | 0.587 | 1.02 | 784 | 0.0800 | 1.00 |
| `codes` (header) | 157 (115 / 151 / 200) | 7.6 (5.1 / 9.9) | 0.314 | 0.55 | 347 | 0.0705 | 0.88 |
| `compact` | **122** (80 / 116 / 165) | **5.6** (4.0 / 7.5) | **0.237** | **0.41** | 270 | 0.0680 | 0.85 |

Reading:

- **Prefill follows bytes, and the header is one third of `codes`' prefill at K = 1.** `compact` costs 0.75× the prefill FLOPs of `codes` over the mix and 0.41× English's; at K = 1 the three arms are within 1.2 ms of each other, at K = 8 `compact` is 4.2× faster than English in wall-clock.
- **Decode is almost encoding-independent once cached.** The answers are the same bytes in every arm, so a KV-cached decoder spends the same per-byte work and differs only in attention over the prefix: 0.85–0.88× English for the Sylang arms. The reference decoder's 2.9× gap is an artefact of re-prefilling.
- **A cached constant prefix is cheap at inference, but not in tokens.** Prefix caching would hide most of the header's prefill cost in a server; it does not hide its 15 tokens per request in a context budget, and it did not hide its cost to the byte model's learning (§9.3.4). `words` is no cheaper than English at inference on this model.

### 9.4 Verdict: what improved, with evidence

**Improved, and recommended for the project:**

1. **Remove the batch header** (`compact`). Evidence: −21% Qwen 3 prompt tokens against the current `codes` over K = 1..8 and −38% against English, cheaper than English at every K where `codes` loses at K = 1 (§9.3.1); for the byte model, 3/3 seeds learn instead of 2/3, 90% is reached at 38,000 items instead of 66,000, and at the dev-selected checkpoint `compact` is +25 to +31 points over `codes` paired per item on every panel (§9.3.4). Nothing was lost: the record lines are byte-identical. If the record count must travel, a 2-byte count line costs 2 tokens instead of 15.
2. **Relative positions.** Evidence: with the packet's learned position table, `codes` with its header scores 0% selected-only (14% exact, the always-empty floor) at K ≤ 8 after 80,000 items (§9.3.4) and every learned-position model collapses when the prompt is shifted by one byte (§9.3.2); with ALiBi the same model learns every arm, and unseen identifiers, unseen feature combinations and their conjunction are at parity with iid.
3. **A decaying learning rate.** Evidence: the constant-LR runs fluctuate by 10–25 points between checkpoints after they escape; a cosine decay to 10% takes `compact` to 100% on all panels in 3/3 seeds at 26,000 items, and English from 1/3 seeds learning to 3/3 at 96–99% (§9.3.3–§9.3.4). By the pre-registered speed criterion it is a tie (ratios 0.45–0.84); on accuracy it is the largest single change in this study.

**Not improved, reported plainly:**

- The **COPY-96 initialisation** does not help multi-record selection: censored on both encodings at K ≤ 8, below fresh (§9.3.3). Its K = 1 benefit under the packet recipe (§8.5) is real but does not transfer. The memoriser start is no better than fresh (§8.5).
- **Curricula** (K = 1 first, or 25% K = 1 throughout) give English a few points on one seed and nothing comparable to the LR schedule.
- The **identifier floor**: three-character identifiers are 3 Qwen 3 tokens each, two thirds of a `compact` record; no encoding of the relation can go below about 3 tokens per record on top of that (§9.3.1).

**Answers to the three tracks:**

- *Real efficiency.* Bytes, tokens, training items and inference cost are four different quantities here: `compact` saves 52% of the bytes but 38% of the tokens against English, needs 0.45× the tokens to 90% of `codes`, and costs 0.41× the prefill FLOPs of English over K = 1..8 (`codes` 0.55×; §9.3.5). The byte model's cost follows bytes; a deployed model's follows tokens; neither follows the other proportionally.
- *Generalisation.* The unseen-combination weakness was a learned-position artefact (0% at never-seen answer offsets, polarity misread at seen ones, §9.3.2). Under relative positions, every hold-out (identifiers, combinations, both, similar identifiers, reordered records, swap, flip, K = 5–8) is within 5 points of iid for every seed that learns, within 4 points at the arm level for `compact` and English + decay, and at 100% for `compact` + decay (§9.3.4); the `codes` arm mean is pulled down by its one failed seed.
- *Better learning.* Of the candidates, only the LR schedule changes the outcome; the representation change (relative positions) and the encoding change (no header) matter more than any initialisation or curriculum tried.

**Worked example** (`results/example_compact_s0_k5.txt`, the `compact` model at its dev-selected checkpoint, K = 5, fresh synthetic identifiers):

```
--- English input
1. "x3x" has seen "y3d".
2. By inference, "j6e" has not contained "a0z".
3. By report, "r4a" contained "a3x".
4. By report, "p3w" has contained "j9d".
5. "e2p" is not helping "v5w".
List [subject,object] of each negative record in order; [] if none.
--- compact input (126 bytes)
x3x	sc	y3d
j6e	Cci	a0z
r4a	cpr	a3x
p3w	ccr	j9d
e2p	Hg	v5w
List [subject,object] of each negative record in order; [] if none.
expected: [["j6e","a0z"],["e2p","v5w"]]
model:    [["j6e","a0z"],["e2p","v5w"]]  [correct]
```

The same item with its records reordered and with every subject and object swapped is also answered correctly (the file shows both). With the instruction line, the English rendering of this item is 250 bytes and 97 Qwen 3 tokens; the `compact` rendering is 126 bytes and 61 tokens, of which 30 are the ten identifiers.

## 10. Study v3: toward a usable pipeline — larger record sets, variable identifiers, varied wording, machine formats, a fully held-out set [M-local]

### 10.1 Pre-registration (v3)

Written on 2026-10-15 at the start of the corpus pilot, after the harness smoke test (20 updates, all arms 0%) and before any pilot result was read; the only training output seen was the first 250-update line of three pilot runs (≤ 9% iid). Code: `study3.py`, `language.py --suite study3`, `tokens3.py`, `sylang_batch.py` (the codec), `summarize.py` (the `language_study3` section).

**Candidate recipe, carried over from §9.4:** the header-free encoding, relative positions (ALiBi) and the cosine learning-rate decay. v3 asks whether the candidate still holds when the data get harder and whether its benefit is more than brevity.

**Task.** Unchanged: the JSON list of `[subject, object]` of each negative record in record order, or `[]`. The answer bytes are identical in every arm.

**What changes in the data.**

- *Identifiers*: 2–5 characters from `[a-z0-9]` with at least one letter, drawn from the whole space and split by a hash of the identifier into train/dev/test (about 80/10/10; `study3.split_of`). The 5,362 v2 training identifiers (letter-digit-letter) are excluded from the v3 dev and test splits, so a model trained on the v2 corpus is evaluated on identifiers it never saw. Accuracy is reported by the longest identifier in the item.
- *Request wording*: each item carries one of six instruction phrasings that ask for the same answer. Four are used in training, two are held out (`study3.WORDINGS`). The byte model has to read the request, not just the records.
- *Record counts*: K uniform in 1..12 in the full corpus; K = 13..16 is never trained (`long`).
- *Held-out combinations*: as in v2, (contain, future) and (help, completed) never appear in training.

**Arms** (one item stream per seed; every arm renders the same records, order and wording):

- `english`: the packet's controlled English, **explicitly the readable comparison**, not a strawman: no filler, one sentence per record.
- `sylang2`: the header-free format (Sylang batch v2; `sylang_batch.encode(..., version=2)`), record lines byte-identical to v1.
- `sylang1`: the current format with its header line (reference, one seed).
- `json`: compact JSON, one object per record with one-letter keys and default feature values omitted, no whitespace. This is what an engineer writes to minimise bytes without inventing a code.
- `csv`: one comma-separated row per record with every field spelled out, no header: the cheapest line-oriented text format with explicit fields.

JSON and CSV are in the study so that a Sylang win cannot be explained by English being verbose: if `sylang2` beats `json` and `csv` on cost or learning, the benefit is the code, not the shortening.

**Corpus pilot** (one seed, `sylang2`, ALiBi + decay, the §9 recipe). Five corpora: `v2` (K 1..8, three-character identifiers, one wording: the §9 corpus), `k12` (+ K up to 12), `ids` (+ variable-length identifiers), `wording` (+ the four training phrasings), `all` (all three). Every run is scored on the same v3 panels, so each increase is judged on the panel that targets it and on the fully held-out panel. Rule for choosing the decisive corpus, fixed now: an increase is *useful* if, at the dev-selected checkpoint, it raises its target panel (`k12` → `long`; `ids` → the longest-identifier-4-and-5 rows of `iid`; `wording` → `wording`) by at least 10 points over `v2` without lowering `full` by more than 3 points. The decisive corpus is `all` unless `all` is more than 3 points below the best single-increase corpus on `full`, in which case the best single-increase corpus is used and the interaction is reported. Cost is reported beside every gain: items, bytes and train seconds to 90% iid and to the dev-selected checkpoint, and points on `full` per 10,000 extra items relative to `v2`. The pilot's seed-0 run on the chosen corpus is reused as seed 0 of the decisive runs (its configuration is unchanged), and the decisive direction must hold on seeds 1 and 2 alone, as in §9 criterion 6.

**Decisive runs** on the chosen corpus, same recipe for every arm (AdamW, LR 1e-3, batch 8, up to 10,000 updates, early stop from 4,000 on the dev panels only, dev-selected checkpoint, one thread per run, three runs share the four cores): `english` × 3 seeds, `sylang2` × 3 seeds, `json` × 2 seeds, `csv` × 2 seeds, `sylang1` × 1 seed. Matched conditions: identical items per seed, model, initialisation, optimiser, batch, update budget and evaluation grid; the only arm-dependent quantity is the sequence length, which is part of what is measured.

**Panels** (fixed items, seed 7, 16 per K): `iid` (test identifiers, trained wording and combinations, K 1..12); `wording` (held-out wording, 8 per K); `combo` (a held-out combination, 8 per K); `long` (K 13..16); `full` (held-out wording **and** a held-out combination **and** K 13..16: the fully held-out set); `reorder`, `swap`, `flip` (counterfactuals of `iid`); `dev` and its counterfactuals for stopping and selection only. Every panel reports exact, selected-only, empty-only, by K and by longest identifier, with per-item flags for paired analyses.

**Costs and measurements, all under matched conditions.** (1) Accuracy as above. (2) Tokens of the intended deployment tokenizer (Qwen 3): `tokens3.py` on test-split items, K 1..16, six wordings, sections header/records/instruction/answer, by K, by identifier length and by wording; the ratio to `english` and the first K at which an arm is cheaper. (3) Latency: the serial benchmark of §9.3.5 (`language.py benchmark --design v3`), prefill time and FLOPs per arm at K 1..16 on an idle core. (4) Training cost: items, bytes, Qwen 3 tokens and train seconds to 90% iid and to the dev-selected checkpoint. (5) Inference cost per request: prompt + answer tokens and prefill FLOPs. (6) The end-to-end handoff demonstration (§10.5): conversion into the format, transport tokens, consumer answer, conversion out, with wall-clock for each stage.

**Frozen criteria.**

1. An arm *learns* if iid exact ≥ 90% on the fine curve in ≥ 2 of 3 seeds (both seeds for `json` and `csv`).
2. *Benefit beyond brevity.* `sylang2` against each of `json` and `csv`, paired per item at the dev-selected checkpoint on `iid`, `wording`, `combo`, `long` and `full`, averaged over shared seeds, bootstrapped over items (1,000 resamples): `sylang2` is *better* on a panel if the 90% interval's lower bound is above +3 points; *tied* if the interval contains 0; *worse* if the upper bound is below −3. Speed: the median over seeds of items to 90% iid; a ratio ≤ 0.67 counts, anything under 2× is a tie. The token ratio is settled by accounting (criterion 5) and is not an experimental question. The verdict "Sylang's benefit goes beyond shortening English" requires `sylang2` to be better than or tied with both machine formats on every panel **and** cheaper in tokens than both.
3. *Candidate holds under harder data.* `sylang2` is *robust* if selected-only exact on `wording`, `combo` and `long` is each within 5 points of `iid`, and `full` selected-only exact is ≥ 85%, at the dev-selected checkpoint, on ≥ 2 of 3 seeds. Length extrapolation (K 13..16) is reported separately: a fall there with `iid` at ceiling is a length-generalisation limit, not a format failure, and is read for every arm.
4. *Against English.* The §9 reading carries over: items to 90% (ratio, 2× rule) and the paired per-item difference at the dev-selected checkpoint on every panel.
5. *Tokens.* Reported from `results/tokens3_qwen3.json` (test-split items, 48 per K): the §9.3.1 identifier floor applies to every arm equally, so the comparison is on the non-identifier tokens per record and on the prompt ratio over the K 1..12 mix.
6. *Targeted expansion.* After the decisive runs, the weakest cell of `sylang2` (lowest selected-only exact among `wording`, `combo`, `long`, `full` and the longest-identifier rows of `iid`, averaged over seeds) determines one targeted corpus addition, run at one seed on `sylang2` and `english`. It is *kept* if it raises that cell by ≥ 5 points without lowering `full` by more than 2 points. Blind scaling (more of the same) is not run.
7. Three seeds resolve only large effects; two seeds resolve only very large ones. Everything below those thresholds is reported as a tie.

**Compute and schedule** (4 shared cores, no paid calls; a 20-update smoke test gave 0.25–1.0 s per update depending on the arm's sequence length, and about 90 s per evaluation of the 14 panels): the pilot is five `sylang2` runs (about 1 h each, three in parallel); the decisive wave is eleven runs (about 20 core-hours, 7 h wall at three workers); the benchmark runs alone afterwards. Checkpoints and per-run logs stay in scratch; only the aggregate `.jsonl` results, the token report and the demo's aggregate measurements are committed.

### 10.2 Run plan and checklist

```
# harness (done): study3.py, language.py --suite study3, tokens3.py, summarize.py study3 section, smoke test (20 updates, 5 arms)
# tokens (done): tokens3.py --source <packet> --tokenizer <qwen3> --out results/tokens3_qwen3.json --example
# pilot (running): for c in v2 k12 ids wording all: language.py suite --suite study3 --corpus $c --arms sylang2 --seeds 1 --position alibi --curriculum decay --updates 10000 --save-dir <ckpt>/study3 --out results/study3/pilot-sylang2-$c.jsonl
# decide the corpus by the §10.1 rule; record the decision in §10.3 before launching
# decisive: --arms english,sylang2 --seeds 3 ; --arms json,csv --seeds 2 ; --arms sylang1 --seeds 1 (same corpus, recipe, grid)
# benchmark: language.py benchmark --design v3 --arms english,sylang2,sylang1,json,csv --k 1..16 --out results/language_benchmark_v3.json (idle core)
# demo: demo_handoff.py --source <packet> --tokenizer <qwen3> --checkpoint <sylang2 dev-best> --english-checkpoint <english dev-best> --out results/demo_handoff.json
# targeted expansion (criterion 6): one addition, one seed, sylang2 + english
# summarize.py results > results/SUMMARY.md ; §10.3–10.6 ; README ; PR body (draft) ; claims and privacy verification
```

**Deviations (logged as they happened).**

1. *2026-10-15, after the `v2` pilot run finished (the first to complete).* Checkpoint names omitted the corpus, so the five pilot runs (same arm, seed and K list) would have overwritten each other's saved models. The `v2` checkpoints were copied under a corpus-specific name before the next run finished, a watcher does the same for the four runs still going, and `language.py` now puts the corpus in the name of every later run. No result file was affected.
2. *Same time.* The pre-registered rule judges `k12` and `wording` on the v3 panels, whose identifiers are 2–5 characters. Corpora trained on fixed-shape identifiers (`v2`, `k12`, `wording`) are therefore out of distribution on every panel: the `v2` run scores 0% selected-only on all of them at every checkpoint, and its dev-selected checkpoint, chosen on those panels, means nothing. That makes the rule's `k12 → long` and `wording → wording` contrasts 0 against 0. Fix: those three corpora are additionally scored at their **final** checkpoint on panels built the same way (same seed) with letter-digit-letter identifiers (`eval3.py --lengths v2`), and the rule's target-panel contrasts are read there; their v3-panel numbers stay in the table as the identifier-shape result. `ids` and `all` are read on the v3 panels as planned, and the decisive corpus is still chosen by the rule's `full` clause on the v3 panels, the panels the decisive runs use.
3. *Same time, hedge.* The decisive wave's `all`-corpus runs were queued behind the pilot before the pilot finished (`all` is the rule's default) so that the four cores stay busy. If the rule picks another corpus, those runs are reported as supplementary and the wave is re-run on the chosen corpus.
4. *2026-10-15 14:15 UTC, with 7 of 11 decisive runs finished* (`sylang2` seed 0 learned and seed 1 did not; `english` seeds 1 and 2 learned and seed 0 did not; `json` seed 0 reached 80% and seed 1 did not). On this corpus the escape from the always-empty plateau within the 80,000-item budget is a seed-dependent event in every encoding, so three seeds measure the escape rate poorly. Two more seeds (3 and 4) of `sylang2` and of `english` were queued behind the pre-registered wave. The pre-registered comparison is reported on seeds 0–2 exactly as fixed in §10.1; the five-seed view is reported beside it, labelled, and the escape rate is given as learned-seeds / seeds with its small denominator. `json` and `csv` stay at two seeds (a `json` seed costs 6.4 h of CPU).

### 10.4 Corpus pilot: which data increase buys what [M-local]

One seed, `sylang2`, ALiBi + decay, 10,000 updates (80,000 items), the five corpora of §10.1. Results at the final checkpoint; the dev-selected checkpoint is the same for `ids` and `all` (update 10,000) and meaningless for the fixed-shape corpora (deviation 2). `results/language_study3_pilot.jsonl`, `results/study3_eval/`.

**On the v3 panels** (identifiers of 2–5 characters from the test split):

| corpus | iid | iid selected | wording (held out) | combo selected | long (K 13–16) | full (held out ×3) | reorder | swap | flip | iid, longest id 4 / 5 | items to 90% iid | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 (K ≤ 8, 3-char ids, 1 wording) | 7 | 0 | 10 | 0 | 0 | 0 | 3 | 0 | 5 | 13 / 4 | ∞ | 2,756 |
| k12 (+ K ≤ 12) | 7 | 0 | 10 | 0 | 0 | 0 | 3 | 0 | 5 | 13 / 4 | ∞ | 4,278 |
| ids (+ 2–5-char ids) | 66 | 64 | 71 | 56 | 2 | 3 | 60 | 66 | 66 | 67 / 65 | ∞ | 3,096 |
| wording (+ 4 wordings) | 9 | 0 | 12 | 0 | 0 | 0 | 5 | 0 | 7 | 20 / 5 | ∞ | 2,792 |
| **all** | **94** | **93** | **97** | **94** | **30** | **30** | 97 | 94 | 96 | 93 / 94 | 54,000 | 4,479 |

The non-zero "exact" of the fixed-shape corpora is the always-empty floor (items whose answer is `[]`); their selected-only exact is 0 on every panel at every checkpoint.

**On fixed-shape (letter-digit-letter) identifier panels**, same construction, final checkpoint (`eval3.py --lengths v2`):

| corpus | iid | wording (held out) | combo selected | long (K 13–16) | full | reorder | swap | flip | iid K 9–12 |
|---|---|---|---|---|---|---|---|---|---|
| v2 | 86 | 81 | 84 | 11 | 8 | 84 | 89 | 82 | 69 |
| k12 | 85 | 81 | 93 | **53** | 58 | 88 | 86 | 84 | 86 |
| ids | 66 | 61 | 74 | 8 | 2 | 65 | 66 | 68 | 47 |
| wording | 93 | **90** | 91 | 12 | 6 | 94 | 89 | 95 | 78 |
| all | 98 | 95 | 98 | 61 | 47 | 99 | 98 | 98 | 97 |

**Reading, by the pre-registered rule.**

1. *Identifier shape is the first weakness.* A model trained on three-character letter-digit-letter identifiers is at 0% selected-only on every item whose identifiers have another shape, while scoring 86–93% on its own shape. Nothing in §9 could have shown this: every §9 panel used the training shape. The `ids` increase takes the 2–5-character panels from 0 to 65–67% and is *useful* by the rule, at a price: with K ≤ 8 and one wording it escapes only after 4,000 updates and never reaches 90% in the budget (the §9 corpus reached 100% at 3,250).
2. *Record count.* `k12` raises `long` (K 13–16) from 11 to 53 points on fixed-shape panels and K 9–12 from 69 to 86: *useful*. Extrapolation stays partial: K = 16 is 38%.
3. *Wording.* The §9 model already answers unseen request wordings at 81% without ever seeing a second wording; `wording` adds 8 points (90%), below the 10-point threshold: *not useful by the rule*, borderline, one seed.
4. *Combination.* `all` is above every single increase on every panel, including the ones the single increases target, and reaches 90% iid at 54,000 items. On `full` (the rule's clause) it scores 30 against 3 for the best single increase, so **the decisive corpus is `all`**, as the rule's default and the hedge of deviation 3 assumed. Its per-K profile on the fully held-out set is 69 / 25 / 19 / 6% at K = 13 / 14 / 15 / 16: length extrapolation degrades with distance from the trained range, while everything inside the range is at 88–100% (wording 97%, combination 94%, identifiers of every length 93–100%, reorder/swap/flip 94–97%).
5. *Cost.* Train seconds are a shared-CPU caveat; in items, `all` reaches 90% iid at 54,000 against 26,000 for the §9 corpus on the §9 panels (§9.3.4): the harder corpus costs about 2× the data, and the single increases alone cost more than that (`ids` never reached 90%).

**Targeted expansion (criterion 6), fixed before any decisive result was read.** The weakest cell of `sylang2` on `all` is `long`/`full` at K ≥ 14 (6–25%). The addition is `all16`: the same corpus with K uniform in 1..16 instead of 1..12, same budget, same recipe; the held-out set then has K 13..16 inside the trained range, and a new extrapolation panel at K 17..20 (`longer`, identical construction) measures whether extrapolation improves with the longer trained range or just moves. Run at one seed on `sylang2` and `english`. Kept if it raises the K 13–16 cells of `full` by ≥ 5 points without lowering the rest of `full` by more than 2, as pre-registered; the K 17–20 numbers are reported as the new limit. No other blind increase (more items of the same kind) is run.

### 10.3 The header-free format and the codec [M-local]

`sylang_batch.py` is an independent implementation of the SYLANG-BATCH wire format, written from the format alone (letter tables, defaults and the identifier escaping were derived empirically by rendering every combination through the packet renderer as a black box; no packet source was copied). Version 1 is the existing format; `encode(..., version=1)` is byte-identical to the packet renderer on 3,000 random batches in both styles and on every BMP code point in the first and a later identifier position. Version 2 is the **opt-in header-free format**: the same record lines, no header; an optional `#sylang2` marker line costs 6 Qwen 3 tokens where the v1 header costs 15. `decode` accepts both versions and both styles and returns the records with the detected version, so every existing input keeps working; a v2 text whose first subject is literally `SYLANG-BATCH` is rejected at encode time to keep detection unambiguous. Interning, `if` records and other policies raise `NotImplementedError` rather than decoding wrongly. An independent adversarial review of the codec found no blocking or major defect (parity held on digit-only, length-1, 256-scalar, non-NFC, control-character and astral identifiers, batches of 1 and 1,000, and a 1,500-batch fuzz) and six minor ones, all fixed: header counts must now be canonical ASCII decimals (so `encode(decode(t)) == t` holds), `if` is special only as a first field, the reference's caps (1..1000 records, 1..256 scalars per identifier) are enforced in both versions so that every text this codec writes is readable by the existing tooling, a damaged v1 header or a byte-order mark is rejected instead of being read as v2, the marker's token claim is stated for a single-digit record count, and `convert --to 1 --marker` is an error. 16 tests, including the parity test, pass under the packet (`test_sylang_batch.py`).

### 10.5 A working task handoff, measured end to end [M-local]

`demo_handoff.py` runs the workflow the pipeline is for: a *producer* agent holds a batch of status records as plain dicts and hands them to a *consumer* agent with one request (list the negated records as `[subject, object]` pairs); the consumer is the packet's tiny byte model trained on the handoff format, and its answer is parsed back into pairs for a downstream tool. The abbreviation in the request ("GBG") was not defined anywhere in the project, so the demo implements a generic task handoff. Every stage is timed and every message is counted with the Qwen 3 tokenizer, the intended deployment tokenizer (no model was called; only its tokenizer). First run, with the §9 dev-selected models (K ≤ 8, three-character identifiers, one wording; 64 fresh synthetic handoffs, 8 per K; `results/demo_handoff_v2.json`, worked examples in `results/demo_handoff_v2_examples.txt`):

| format | consumer model | correct | message tokens (mean) | process tokens, 64 handoffs | ratio to English | conversion in | consumer (byte model, shared CPU) | parse out | lossless round trip |
|---|---|---|---|---|---|---|---|---|---|
| english | §9 English + decay, seed 0 | 62/64 (K = 8: 6/8) | 93.6 | 7,197 | 1.00 | 0.07 ms | 0.59 s | no parser | n/a |
| sylang2 | §9 `compact` + decay, seed 0 | **64/64** | 58.2 | 4,931 | **0.69** | 0.11 ms | 0.26 s | 0.06 ms | 64/64 |
| sylang1 | (tokens only) | – | 73.2 | 5,891 | 0.82 | 0.10 ms | – | 0.06 ms | 64/64 |
| json | (tokens only) | – | 140.2 | 10,179 | 1.41 | 0.06 ms | – | 0.03 ms | 64/64 |
| csv | (tokens only) | – | 96.3 | 7,374 | 1.02 | 0.01 ms | – | 0.01 ms | 64/64 |

Reading: the conversions into and out of Sylang are microseconds against a consumer step of hundreds of milliseconds, so the format's cost is entirely in the transport tokens and the consumer's prefill; the whole process (message + answer) costs 0.69× English in the deployment tokenizer and the byte-model consumer answers in 0.44× the time (0.30 against 0.65 GFLOP per handoff, prefill plus KV-cached decode). The Sylang messages decode back to the producer's records byte-for-byte on every handoff; compact JSON costs 1.41× English and CSV 1.02×, so neither machine format beats controlled English on tokens here, while header-free Sylang does. The consumer latencies were measured on a shared CPU during the v3 pilot and are relative, not absolute. The v3 models (variable identifiers, six wordings, K ≤ 16) re-run this demo in §10.6 once the decisive runs finish.
