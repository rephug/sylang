# Why the byte-level model formats answers but does not copy identifiers

**Date:** 2026-10-13 · **Author:** Claude (independent investigation)
**Code:** [`experiments/copy_probe/`](../../experiments/copy_probe/) · **Numbers:** [`results/SUMMARY.md`](../../experiments/copy_probe/results/SUMMARY.md) · **Patch:** [`patches/README.md`](../../experiments/copy_probe/patches/README.md)

Evidence labels:

- **[R]** Reported in the uploaded packet `sylang-source-and-results-20261013.zip` (journals, scores, reviews). Read here, not re-run.
- **[M-real]** Measured here with the project's own `tiny.py`, `decode.py` and `binding.py`, its own rows and schedules, and the trainers' exact loss, optimiser and clipping. This uses `real_pipeline.py` under PyTorch 2.6.0+cpu, the version of the native runs. The warm-start checkpoint is not in the packet, so these runs start from `tiny.build_model` under the recipe seed. That difference is itself one of the comparisons (§2.3).
- **[M-ref]** Measured with the independent reference model in `experiments/copy_probe`.
- **[I]** Inference.

No project checkpoint was read or replaced, no model was downloaded and no API was called. CPU PyTorch 2.6.0 and 2.14 were installed into scratch virtual environments. **[M-local]** below marks runs on the packet's own model and code, done locally with synthetic data (`followup.py`, `language.py`).

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
- **Fresh initialisation is a weaker, sufficient fix at 12 identifiers.** It mostly copies, but one seed emits 12 training spellings.
- **Repetition is not the problem.** 96 identifiers × 32 exposures each (1,536 updates) is perfect. What drives memorisation is the narrow set, not repeated exposure to it.
- **One confound remains.** At a fixed 192 updates, breadth and exposures per identifier move together. The 96 × 1,536 arm shows high exposure is harmless at high breadth. A 12-identifier × 4-exposure arm (24 updates) learned nothing from either initialisation, because 24 updates is too few overall, so it cannot separate the two.

Fresh initialisation helped my small-data runs but is not required: **breadth matters more than initialisation.** The real-checkpoint arms of this matrix belong on the native side (§0.4).

### 0.3 Zero-shot selection failure is expected

The COPY training distribution has nothing in common with the selection task:

- 8-byte prompts against 359/367-byte prompts;
- `a–f` letters against `a0r…h7y` identifiers;
- no JSON, polarity or roles.

The 0/48 says nothing about whether COPY pre-training helps *learning* selection. That is the native team's proposed next question.

### 0.4 Coordination: which runs belong where

| Question | Where | Status |
|---|---|---|
| Real warm start × 12 / 96 identifiers, extra seeds | native (needs `efb2…`) | proposed: 2 more schedule seeds per arm, to put intervals on 2/12 and 432/432 |
| Fresh vs surrogate × breadth 12/24/48/96 × exposure | local | **done** (above) |
| COPY-init vs original-init on the *original* 64-example selection task | native (needs both real checkpoints) | the native team's proposed next run; not duplicated here |
| Initialisation (fresh / COPY-pretrained / surrogate) on length-matched short-instruction selection with fresh identifiers | local | §8 |
| English vs Sylang, matched budget, identifiers and lengths | local | §8 |

### 0.5 Length matching: the correction applied, and what it cost

The owner pointed out that random filler alone can leave the class predictable from length. My first implementation did match length distributions exactly, by stratified sampling within each prompt-length cell. A leakage audit then showed that the leak had only moved:

- **Length:** length-only accuracy was exactly 50%.
- **Features:** relation, tense, aspect and evidence predicted polarity **70.8%** of the time, against 50% by chance; evidence alone gave 60.6%. When negation costs bytes, equal lengths force the *other* content to differ between classes.

The runs therefore use **negation-length padding** instead:

- each positive record line gets filler (`~`) equal to the bytes its own negative form adds in that arm;
- features are sampled independently of polarity.

Audit over 40,000 items (`language.py audit`). Lookup classifiers were fit on one half and scored on the other:

| Cue | english | words | codes | codes_padded |
|---|---|---|---|---|
| Negative count from prompt length + K (K alone: 43.7%) | 43.0% | 42.8% | 43.5% | 42.8% |
| Record polarity from padded line length (chance 50%) | 50.0% | 49.9% | 49.7% | 49.9% |
| Record polarity from features | 50.0% | 50.0% | 50.0% | 50.0% |
| Record polarity from filler presence | 100% | 100% | 49.7% | 49.7% |

The one residual cue is deliberate and local: in English and `words`, a positive record carries filler, carrying the same information as the absence of "not". The polarity-flip panel recomputes the padding, so it tests that the model reads each record.

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

- **The collapse reproduces from a fresh initialisation.** With the real data and the original 192 updates × 4 rows, the collapse appears in 3/3 seeds: one constant answer `[["b1s","b1s"]]`, 0.00 bits of prompt dependence, swap TV 0.00, and 100% empty answers.
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
- **Its failure pattern matches.** It emitted training spellings for 6 of the 12 held-out identifiers, e.g. `cfe → cdf` and `fbb → fdd`: a lookup prior, not a copy rule.

**4. Neither task, as built, can show copying of *new* identifiers. [M-real]**

- **The binding data has no new identifiers to test.** It uses 8 identifiers (`a0r b1s … h7y`), each fully determined by its first character. Train and test differ only in which identifiers are paired. Every binding run trained on them scores **0% on fresh identifiers**, even when it scores 100% on the eight.
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

In the frozen diagnostic, P(a) = 17.3% at the first identifier byte, although every initial is exactly 12.5% of the training targets. The from-scratch runs here collapse to `b1s` instead. **[I]** So the specific `a0r` is inherited from the warm start, while the collapse itself is not.

### 2.3 One-factor runs on the real model and data [M-real]

Mean over seeds (min–max). Batch is rows per update, in quartets; all runs are from fresh initialisation. Full table: `results/SUMMARY.md` → `real_pipeline`.

**COPY packet** (`COPY:abf` → `abf`; 12 balanced training identifiers, 12 held out):

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

Both run on the native CPU in minutes.

### E1: confirm the initialisation effect on the unchanged COPY packet

Run the COPY packet, with the patch, six times:

- 3 warm-started and 3 fresh;
- seeds varied by the torch/Python seed and the schedule shuffle seed;
- otherwise identical to the run already made.

Score held-out COPY exact, per-byte accuracy and the predeclared contrasts. Each run takes under 10 s of training.

The fresh arm has already passed once here, at the recipe seed, with the patched worker (§1). E1 repeats it natively, across seeds, against warm-started runs of the same packet.

| Outcome | Reading |
|---|---|
| Fresh ≥ 11/12 held-out in ≥ 2 of 3 seeds, **and** warm ≤ 4/12 in ≥ 2 of 3 | **Initialisation confirmed as the cause.** Stop warm-starting from the memorisation checkpoint for capability probes |
| Both high | The 2/12 run was a seed outlier. Explanation 2 is withdrawn |
| Both low on native hardware | Something differs between native and these runs. Run `real_pipeline.py audit` natively and diff configurations |

### E2: copy *new* identifiers in the binding task, with no length cue

Change the binding data, not the model:

- **Fresh identifiers per quartet** from a large split with independent characters. Hold out a disjoint identifier set and keep the existing unseen-pair checks.
- **Short instruction**, identical for every row, e.g. `[] if positive, else [["subject","object"]].`
- **No length cue:** a random 0–16-byte neutral filler before the record, drawn independently of polarity, so length predicts nothing and absolute positions vary.
- **Fresh initialisation** (the patch).
- **Budget ladder** of 768, 1,536 and 3,072 updates at batch 4, LR 3e-4 (the epoch and step caps must be raised), plus one arm at batch 16 / LR 1e-3. Use 3 seeds.

Log per update: first-byte vs suffix loss, prompt dependence (bits) and swap TV on a fixed probe set.

**Success** means all of the following in 3/3 seeds at some rung of the ladder:

- ≥ 95% exact on held-out *fresh-identifier* selected prompts;
- ≥ 95% on held-out empties;
- swap TV ≥ 0.9;
- a length-only classifier at chance on the training data.

Reference points:

- short instruction, fresh identifiers, no jitter: 100% at 1,536 updates × 4 (2/2 seeds);
- long instruction, fresh identifiers: 100% at 1,536 updates × 16 with LR 1e-3 (2/2 seeds);
- the reference model with jitter: 32% at 3,000 × 32.

So the jitter rung may need the larger budget. If E2 fails at every rung while E1 passes, the remaining obstacle is position-invariant retrieval. Then compare learned absolute positions with a relative scheme, as a separate, single change.

---

## 6. A fair English-versus-Sylang comparison (after E2 passes)

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
