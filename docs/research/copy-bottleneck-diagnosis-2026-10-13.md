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

In the weighted run, the suffix loss keeps falling (0.10 nats at the end) while the first-identifier loss stays flat at 2.26 nats. Both runs end with 24 × `[]` and 24 × `[["a0r","a0r"]]`.

In the frozen diagnostic, P(a) = 17.3% at the first identifier byte, although every initial is exactly 12.5% of the training targets. The from-scratch runs here collapse to a different identifier instead. **[I]** So the specific `a0r` is inherited from the warm start, while the collapse itself is not.

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

### 5.2 Next two experiments (revised after §8)

Both are native runs: they need the real warm start, more seeds, or the target model's tokenizer.

**N1: the language comparison where it can mean something.**

- **Arms:** natural controlled English, `codes`, and one compact readable baseline (e.g. the tab-word layout without filler), all with the same item stream.
- **Model:** the packet's `tiny` with relative positions (ALiBi, as `language.py --suite position`). If absolute positions must stay, add `codes_jitter` as a fourth arm, so a fixed-offset advantage is visible as such.
- **Tasks:** K = 1–4 and K = 4–8. The batch header is amortised only at larger K.
- **Budget:** 12,000 updates × 8, LR 1e-3 and 3e-4, 3 seeds per arm.
- **Scoring:** held-out identifiers, a held-out (relation, tense) combination, swap and flip panels; items, bytes and FLOPs to 90%; peak and final accuracy.
- **Success**, decided before running:
  - every arm reaches ≥ 90% held-out exact in ≥ 2 of 3 seeds;
  - `codes` is worth pursuing only if it is within 2 points of the best readable arm on every panel at equal FLOPs, **and** at least 10% cheaper per query in prefill FLOPs.
- **Expected from the local two-seed result (§8.3.2):**
  - `codes` passes the cost criterion at K = 1–4: 22% fewer bytes and 0.74× prefill FLOPs.
  - It is tied with natural English per item: a median of 48,000 vs 56,000 items to 90%.
  - Because each item is shorter, it needs fewer bytes to get there: a median of 7.4 vs 10.8 MB.
  - If that holds over seeds, the honest claim is "a compression gain at equal accuracy", not "easier to learn".

**N2: COPY-96 initialisation for the 64-example selection task.**

- **Arms:** the native team's planned COPY-init vs original-init comparison, using the 96-identifier COPY checkpoint (432/432), plus a fresh arm.
- **Scoring:** fresh held-out identifiers and a held-out combination, as well as the original 48 checks, which reuse the training identifiers.
- **Budget:** the packet's (batch 4, LR 3e-4), with the epoch cap raised so that 6,144 updates are allowed. Locally, COPY-96 escaped only between updates 3,072 and 6,144.
- **Success:** COPY-96-init ≥ 90% on fresh-identifier selected checks in ≥ 2 of 3 seeds, with fresh and original-init lower. A held-out combination below 70% would show the copying transferring but the record understanding not.

**Cheap check before N1 (no training): token counts.** Tokenise every arm's prompts with the tokenizer of the model Sylang is meant for. In pretrained BPE models, English is cheap and code-like strings fragment, so the 22% byte saving may shrink or reverse in tokens (2026-10-04 review). If `codes` is not ≥ 10% cheaper in tokens, N1's cost criterion fails before any training.

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

### 9.1 Pre-registration

Written 2026-10-14 before any v2 run longer than the 20-update smoke test. Deviations made after the runs started are logged in §9.2. Code: `language.py --suite study` at commit `12f0985`.

**Task.** Items of K records, K uniform in 1..8 in training. Answer: the JSON list of `[subject, object]` of each negative record in record order, or `[]`. Identifiers are letter-digit-letter from disjoint train (5,362), dev (715) and test (683) splits. Features: relation {see, help, contain}, tense {past, present, future}, aspect {simple, progressive, completed}, evidence {unspecified, direct, reported, inferred}; polarity 50/50 per record. **Held-out combinations**, never in training: (relation = contain, tense = future) and (relation = help, aspect = completed).

**Arms (encodings).** Main comparison with ALiBi in place of the learned position table (§8.3.2): `english_natural` (controlled English, no filler), `codes` (the current batch codes, with the 35-byte header), `compact` (`codes` without the header line; record lines byte-identical). Reference runs with the packet's learned positions: `english_natural` and `codes`, 1 seed. The filler arms are excluded: their models read the filler (§8.3.2).

**Learning methods.** Baseline: fresh initialisation, mixed K from the start. Candidates, piloted at 1 seed on `english_natural`: `copy96` initialisation (the COPY-96 checkpoint; under ALiBi its position table is unused), `surrogate` (memoriser) initialisation, `staged` (K = 1 only until the dev K = 1 panel reaches 97% or 3,000 updates, then mixed), `k1mix` (25% of items K = 1 throughout). The best candidate by criterion 4 is then run at 3 seeds on every encoding.

**Recipe.** AdamW, LR 1e-3, batch 8, up to 10,000 updates (80,000 items), evaluation at the fixed grid (250 … 8,000), early stop when two consecutive checkpoints from update 4,000 on are ≥ 97% on iid, dev, swap, flip and reorder. One thread per run; four runs share four cores.

**Panels** (fixed items, fixed seed; 32 items per K for iid/dev/flip, 16 per K for combo/both): `iid` = unseen identifiers, trained combinations; `combo` = training-split identifiers, at least one held-out combination; `both` = unseen identifiers and a held-out combination; `reorder` = iid items with the record order permuted (K ≥ 2, paired); `similar` = every negative record has a positive partner whose subject or object differs in one character (K ≥ 2); `swap` = subject and object exchanged in every record (paired); `flip` = one record's polarity inverted (paired); `dev` for monitoring and early stop only. Every panel reports exact, selected-only (non-empty answers), empty-only and by K; K = 5–8 is the "long" view.

**Costs.** Items, bytes and Qwen 3 tokens seen to first reach 90% iid exact; train seconds (caveat: shared CPU); COPY-96 pre-training cost added to its arm; the serial inference benchmark per encoding; token counts of header, records, instruction and answer under the Qwen 3 tokenizer (Gemma 3's is gated and unavailable here).

**Frozen criteria.**

1. An arm or method *learns* if iid exact ≥ 90% in ≥ 2 of 3 seeds at some checkpoint.
2. Efficiency: primary, the median over seeds of items to first reach 90% iid exact; secondary, peak iid exact; tertiary, final iid exact.
3. `compact` is *non-inferior* to `codes` if its mean peak iid exact and its mean selected-only exact on combo, both, similar and reorder (final checkpoint) are each within 2 points of `codes`'; it is *worthwhile* if its Qwen 3 prompt tokens are ≥ 10% below `codes`' over the K = 1..8 mix.
4. A learning method is an *improvement* if its median items-to-90% is ≥ 25% below the baseline's in the same encoding, with peak iid exact within 1 point.
5. A model *composes* if selected-only exact on both combo and both is ≥ 85%.
6. Pilot runs are reused as seed 0 of the decisive runs when their configuration is unchanged.
7. Three seeds resolve only large effects; differences under 2× in items-to-90% between arms are reported as ties.

### 9.2 Deviations

(none yet)

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
- **Identifiers set the floor.** Three-character identifiers tokenize as three tokens each in Qwen 3, so 6 of the 9 tokens of a `codes` record are the identifiers, which no encoding of the relation can remove. The relation, tense, aspect and evidence cost 1–2 tokens in `codes` against 11 in English.
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
- **It is not the whole story.** At positions seen in training, held-out negatives still score 38–65% against 87–98% for trained combinations. Those errors split about evenly between **polarity** (the unseen wording is not recognised as a negation, and 30–40% of held-out *positives* are answered as if negative) and **copying**. The filler-English models make almost no polarity errors (the filler decides polarity) and fail only on copying.
- **Codes compose.** For `codes` the unseen combination is one new letter pair at a fixed offset; learned-position `codes` and `codes_jitter` score 100% on it. `codes_eqlen` loses 8% to copy errors, because its padding width changes with the unseen code.
- **Relative positions remove the weakness at K = 1.** Both ALiBi English models score 100% on the held-out combination, at novel and seen offsets alike. Whether that holds for K up to 8 and for the second hold-out is what §9.3.3 measures.
