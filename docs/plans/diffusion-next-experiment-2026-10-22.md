# Next experiment: matched AR vs masked-diffusion screen (design, 2026-10-22)

> **Status: proposal only.** Nothing has been trained, rented or launched. This
> design accompanies the independent review in
> [`docs/research/diffusion-review-2026-10-22.md`](../research/diffusion-review-2026-10-22.md)
> (section 9). Three review agents drafted independent designs, CPU-first,
> single-GPU and pretrained-transfer; a judge agent scored them, merged them and
> re-checked their arithmetic. Neither the reviewer nor the owner has run any
> part of it. All token, FLOP and time figures are planning estimates to be
> replaced by the Stage 0 measurements. Every step that installs software,
> reserves hardware or spends money needs the owner's approval.

## Zero-cost offline steps before any training

These need no training, rental or network. Only items 12-13 touch the owner's server, for a few minutes.

1. **Correct the headline claims.** Restate M's "19.5% fewer Qwen3 tokens" (864 world) and "11.66%" (smoke suite) as **mostly version-tag effects**. Report body-only counts from `benchmarks/results/diffusion-review-checks.json`:

   | Format | Tokens per record |
   |---|---:|
   | Concise English | 7.86 |
   | Codec English body | 12.11 |
   | M body | 13.00 |
   | Tuple JSON of codes | 14.5 |
   | Tuple JSON of words | 15.83 |
   | Named JSON (with wrapper) | 46.83 |

   Body-only, M/E = 1.073 on the 864 world and 1.033 on the smoke suite.

2. **Restate the §6 gate.**
   - I has an equal-steps null, I_0 = G_M/G_E: 1.38 in passes and about 1.21 in CPU time, for M vs concise English on PARSE.
   - Decisions use I/I_0, which equals k_E/k_M in passes, and absolute within-MDM ratios.
   - This screen uses δ = 2 pp; the 1 pp margin is kept for any scaled follow-up.

3. **Avoid the harness's copy condition.** In `evaluation/harness.py` the JSON arm reconstructs its own input. Here every arm gets the identical neutral English input, and no arm's output equals or contains its input.

4. **Freeze renderers and strict parsers for E, J, M, C and F.**
   - J and M are header-free; the decoder re-adds `M0.1:` or the core-v0.1 wrapper before `sylang_core.decode`.
   - 100% round trip on 200k items per format and on all 864 meanings.
   - Assert that no version string appears anywhere.

5. **Neutral input lexicon and overlap audit.** Build the lexicon disjoint from be, did, does, had, has, have, is, not, was, will, the verb stems and E's adverbs. Add the at-most-5% non-name overlap audit.

6. **Name generator, splits and panels.**
   - Names: 1-5 Qwen3 tokens, 30% two-word, SHA-256 80/10/10 split, coverage of at least 20 training names per subword.
   - 50% quoted inputs.
   - The 3 held-out compositions.
   - Panels frozen with SHA-256 manifests.

7. **Exact-tokenization prediction table.** Per format: output and canvas lengths, I_0, per-item FLOPs, and break-even pass ratios (CPU 0.80-0.81; memory-bound 7B 0.90). Replace the 5% budget margin with measured values. Publish this as the pre-registered structural prediction.

8. **Sampler conformance in numpy against `FiniteOracle`.** Reproduce tau = 0.9 rounds of M 7.0, English 8.69 and JSON 8.83 to within ±0.05, plus EB-Sampler and fixed-k. A one-hot model must decode in 1 step.

9. **Detectability.** Recompute parser-detectable substitution rates for E, C and F with the `tools/review_diffusion_claims.substitutions` pattern. Known values: English 80.7%, JSON 39.5%, M 0%.

10. **Port rephug/sylang#3 tooling.**
    - The grader's 59 corrupted-output tests become AST-grading tests, about 60 per format.
    - `analyze.py` gains a δ parameter and t/z inflation.
    - `ledger.py` gains a CPU-hours event with a 420 core-hour ceiling.

11. **Commit the operating-characteristics simulation** (look 0 / 3 / 6 rule) and the **pre-registration file**. It holds hashes of the generator, panels and analysis code, every threshold, the A* ladder, the adaptive E_max rule and the decision order.

12. **GEMM and throughput micro-benchmark on the owner's server** (minutes; no training). If it shows under 20 GFLOP/s per process, Tier-S is selected before Stage 0.

13. **Owner installs a pinned PyTorch CPU wheel.** This needs approval; the review container cannot install it.

## Recommended design: from-scratch screen, CPU first, US$0

**Design only.** Nothing is trained, rented or launched by this document.

The design keeps the **2 x 3 factorial the owner asked for**:
- families: autoregressive (AR) and masked diffusion (MDM);
- formats: concise English (E), named JSON (J), Sylang M (M).

It adds **two aligned-code controls that run in the MDM family only**:
- C, a generic code line;
- F, the same code line with fixed-width name slots.

That makes **8 cells**. All of them use:
- identical 3.93M-parameter prefix-LM transformers;
- one deterministic English-to-AST parsing task whose input is the same text in every arm;
- exact Qwen3 tokenization;
- 3 paired seeds, with a seed-0 futility look and a pre-registered extension to 6 seeds.

The decision rests on two numbers, both measured at a matched exact-AST accuracy:
- MDM denoising passes;
- within-MDM end-to-end CPU latency.

The report's interaction ratio I is always shown next to its equal-steps null I_0, and only I/I_0 is interpreted.

Grafted ideas:
- **From the CPU design:** the C and F controls, the I_0 correction, NFE as the primary measure, the modeled 7B sign check, constant LR with EMA, the A* ladder and Tier-S.
- **From the GPU design:** the neutral input lexicon with a 5% overlap audit, names quoted in 50% of inputs, preconditions P1-P5, sampler conformance against the oracle, and t/z bootstrap inflation.
- **From the pretrained design:** a pre-registered structural prediction, seed-0 futility, retry cost counted in latency, and Dream-v0-Base-7B vs Qwen2.5-7B as the only paid next step.

### (a) Hypotheses

**H1-align (primary; field-aligned encodings in general).** Some aligned output A in {M, C, F}, decoded by the MDM, does all three of the following against both E and J:
1. It needs at most 0.80x the denoising passes at matched accuracy A*.
2. It has at most 0.90x the within-MDM end-to-end latency.
3. It loses no more than 2 percentage points (pp) of whole-AST accuracy.

**H1-M (secondary; Sylang-specific).** M satisfies H1-align and is at least 10% faster than both C and F within the MDM family, at non-inferior accuracy.

From-scratch models give M's mnemonic letters no meaning. M therefore differs from C only in delimiters, quoting and length: M is about 6 tokens per record longer.

**H0.** Format changes MDM passes at A* by less than 10%, and within-MDM latency differences follow canvas length.

**Why the §6 interaction ratio needs a correction.** I = (T_MDM,E / T_MDM,M) / (T_AR,E / T_AR,M) is above 1 under H0 whenever M is longer than E. AR pays per output token; MDM, at equal steps, does not.
- On this task the equal-steps null is I_0,NFE = G_M/G_E = 46.5/33.7 = **1.38** in passes, and I_0,time is about **1.21** in CPU time.
- Only I/I_0 is read as diffusion-specific. In passes, I/I_0 equals k_E/k_M exactly.

**Pre-registered structural prediction (an expectation, not a criterion).**
- Outputs per item: concise E 33.7 tokens, M 46.5, C 31.6, F 39.9, J 106.0.
- Break-even passes for M on CPU, where a step costs 12.0 ms for E and 13.2 ms for M with a 10.7 ms prefill: M must need **at most 0.80-0.81x E's passes** to reach T_M/T_E <= 0.90.
- In a memory-bound 7B regime, M must need at most 0.90x E's passes.
- The exact-marginal oracle with a Fast-dLLM threshold of 0.9 gives M only 0.81x codec English's rounds (7.0 vs 8.69).

**Expected outcome:** STOP, or REVISE (vs-JSON-only, R1).

### (b) Arms: 8 cells

**Families** share the network, vocabulary, initialization RNG, data order, batch and optimizer steps:
- AR is a prefix LM: the prompt attends bidirectionally, the output causally, with a KV cache.
- MDM is absorbing-state masked diffusion: the prompt is bidirectional and never masked; the output canvas is bidirectional.

**Version tags.** No sequence the model reads or writes contains a version tag. The generator asserts that `Sylang`, `core-v0.1`, `M0.1` and `Prime0.1` are absent. Decoders add the header outside the model, and that work is timed. All token counts are body-only.

| Code | Surface for one record | Out tokens/item (K 1-4) | Families |
|---|---|---:|---|
| E, concise bijective English | `Reportedly, Bumche had not seen Vondu.` (adverbs Directly / Reportedly / Presumably / none; codec `_verb` table) | 33.7 | AR, MDM |
| J, codec named-field statement, header-free | `{"kind":"pred","subject":"Bumche","relation":"see",...,"evidence":"reported"}` | 106.0 | AR, MDM |
| M, header-free M body | `p("Bumche",s,"Vondu",-,p,c,r)` | 46.5 | AR, MDM |
| C, generic code line | `Bumche s Vondu - p c r` | 31.6 | MDM only |
| F, C with every name slot padded to 5 tokens | reserved filler token, dropped when detokenizing | 39.9 | MDM only |

Token counts come from the CPU design's prototype with literal isolation. Stage 0 recomputes them under exact tokenization, and all budgets below carry a 5% margin for that.

**Why AR-C and AR-F are not trained.** For a fixed network at batch 1, AR forward passes and latency are a deterministic function of output length: prefill plus G times the per-token cost. AR NFE, I_0 and modeled AR latency for C and F are therefore computed from gold lengths and measured AR-E/AR-M step costs. The only thing lost is AR accuracy for C and F; see (j).

**Arms dropped:**
- Codec English: 35-53% longer than concise E, so it is not the strongest English.
- JSON tuples of words (JT) or of M's codes (T): C bounds them. JT is added only in R1.
- DSL and Prime.
- Any header-bearing variant.

### (c) Task and data

**Task: PARSE.** The input is 1-4 English clauses; the output is the records in order, in the cell's format. The input bytes are identical for every arm, so no arm can copy its input into its output.

**Neutral input lexicon.** It is disjoint from every output's non-name tokens, including the codec verb-phrase words be, did, does, had, has, have, is, not, was, will and the verb stems.

| Field | Input wording |
|---|---|
| Relations | notice / observe / spot; assist / aid / support; hold / include / enclose |
| Tense x aspect | `<v>s / <v>ed / shall <v>`; `keeps / kept / shall keep <v>ing`; `finishes / finished / shall finish <v>ing` |
| Negation | `never` |
| Evidence | `Firsthand,` / `As witnessed,`; `Word has it that` / `According to hearsay,`; `By deduction,` / `It follows that`; nothing for unspecified |

- Evidence phrases are placed at the front or the back of the clause.
- Clauses are joined by `. `, `; ` or `. Also, `.
- Voice is active only, because passives need is/was.

**Overlap audit (precondition).** Output tokens that are not names or punctuation and also occur in the input must be **at most 5%** for every arm.

**Name quoting.** Names are quoted in 50% of inputs, chosen by item hash. Quoted outputs (M, J) and unquoted outputs (E, C, F) each meet cross-context BPE re-segmentation on half the items. Results are reported by stratum.

**Names.** Syllable pseudo-names:
- 1 word (70%) or 2 words (30%), 1-5 Qwen3 tokens in every context;
- template and verb words excluded;
- split 80/10/10 by SHA-256(name);
- every subword of a dev or test name must appear in at least 20 training names.

**Held-out compositions** (never in training records): (contain, future, progressive), (help, past, completed), (reported, negative).

**Panels.** All are frozen to JSONL with SHA-256 manifests before Stage 0.

| Panel | Size | Contents |
|---|---|---|
| Training stream | 256k items per run | Same order in all cells for a given seed |
| dev | 1,000 | |
| curve | 500 | |
| **test_iid (primary)** | 2,000 | 500 per K |
| test_combo | 1,000 | |
| test_long | 500 | K = 5-6 |
| test_seen | 500 | |

**Leakage checks:**
- train-only names;
- a 1M-draw scan for held-out compositions;
- vocabulary and canvas tables built from training data only;
- any out-of-vocabulary token fails the build.

### (d) Models

**Network.** A pre-LayerNorm transformer, identical in both families:

| Item | Value |
|---|---|
| Width / layers / heads | d = 256, 4 layers, 4 heads |
| Feed-forward | GELU, 1,024 |
| Other | RoPE, tied embeddings, no dropout |
| Vocabulary | V = 3,072: exact Qwen3 IDs from all arms' training renderings and inputs, remapped to a compact table, plus `<bos> <sep> <eos> <mask> <fill> <pad>` |
| Parameters | 3,145,728 non-embedding + 786,432 embedding = **3.93M** in every cell |

The prompt never attends to the output, so prompt-KV caching is exact in both families.

**Objectives:**
- **MDM:** draw t ~ U(0.001, 1), stratified across the batch, and mask each canvas token with probability t. Loss = (1/t) x the sum of masked-position cross-entropy / canvas length. No time conditioning.
- **AR:** next-token cross-entropy on the output plus EOS.

**Canvas.** For each format, the 99th percentile of gold length among training items with prompt length within ±2, frozen; linear extrapolation for test_long. An overflow is scored as an error.

**Decoding** (greedy, no grammar constraints):
- **AR:** greedy until EOS or 1.5 x canvas. NFE = number of generated tokens.
- **MDM:** committed tokens are never revised. Samplers:
  - fixed-k (LLaDA low-confidence), k in {1, 2, 4, 8, 16, L};
  - threshold (Fast-dLLM), tau in {0.5, 0.7, 0.9, 0.95, 0.99};
  - entropy bound (EB-Sampler), gamma in {0.1, 0.5, 1.0} nats;
  - **semi-autoregressive block 16 at tau = 0.9**, a sensitivity setting that predicts a Dream/Fast-dLLM follow-up.
- **Validator fallback** (secondary policy): re-decode once at k = L when parsing fails, counting both costs.

**Sampler conformance (precondition).** Fed exact oracle marginals in the 864-meaning world, the threshold sampler must reproduce `benchmarks/results/diffusion-review-checks.json` at tau = 0.9 within ±0.05 rounds: M 7.0, English 8.69, JSON 8.83. A one-hot "perfect model" must decode every item in 1 step.

### (e) Training protocol

**Optimizer and threading:**
- AdamW (0.9, 0.98), weight decay 0.01, clip 1.0, fp32.
- 500-step warm-up, then **constant LR**; evaluate an EMA of the weights. This makes checkpoints comparable and continuation exact.
- Batch 32 items.
- One `torch.set_num_threads(1)` process per run, 4 runs in parallel; measured GEMM throughput is about 100 GFLOP/s per single-thread process, against about 150 for one 4-thread process.

**Stage 0 (calibration seeds 100 and up; blind to format contrasts).**
1. Code tests: round trips, mask leaks, sampler conformance, and an MDM toy copy task reaching 100% in one step.
2. A 200-step throughput benchmark.
3. LR sweep: AR/MDM x E/M x {5e-4, 1e-3, 2e-3} x 64k items, 12 runs. Choose one LR per family on the pooled E+M dev score.
4. Continue the 4 chosen-LR runs to 128k and train 4 runs of a second calibration seed to 128k.
5. **Reliability gate:** at least 7 of these 8 runs must learn. "Learned" means dev exact accuracy of at least 90%, with the MDM decoded at k = L. P(pass) is 0.81 if the per-run learn probability is 0.9, and 0.11 if it is 0.6.
6. If the gate fails:
   - one LR retry for the failing family;
   - then d = 384, 6 layers (about 3.4x compute) on Lightning L4 free credits, with owner approval;
   - otherwise outcome R6.

**Adaptive endpoint (mechanical, pre-registered).** E_max = 128k items only if all 8 gate runs:
- reach at least 95% dev accuracy at 128k;
- gained no more than 1 pp from 64k to 128k;
- changed tau = 0.9 NFE by no more than 5%.

Otherwise E_max = 256k. There is no early stopping.

**Checkpoints.**
- Equal exposure: {4k, 8k, ..., E_max} items.
- Equal compute: {25, 50, 100, 200, 375} TFLOP. These sit below the smallest per-run total, AR-E at 256k = 397 TFLOP, and are halved if E_max = 128k.

**Seeds and looks.** Seeds are paired: the same seed gives the same data order and initialization RNG in every cell.

| Look | Runs | Purpose |
|---|---|---|
| 0 | Seed 0, all 8 cells | Futility check only (X0); cannot declare success |
| 1 | Seeds 1-2, all 8 cells | First decision |
| 2 | Seeds 3-5 on AR-E, AR-M, MDM-E, MDM-M, MDM-C, plus MDM-J or MDM-F if the unresolved contrast involves them (at most 6 cells) | Final decision |

**MDM extension.** If pooled MDM-E/MDM-M dev accuracy at k = L is below 90% at E_max but rose by at least 2 pp over the previous half of training, all MDM cells (every format and seed) continue once to 2 x E_max.

**Divergence and failed runs.** On NaN, or a loss above 3x its running median for 200 steps, restart once from the last checkpoint. A second failure marks the run failed. Failed runs stay in the analysis (intent-to-train).

### (f) Measurements

**Correctness.**
- Primary: **whole-output exact**. Every line parses and validates through `sylang_core.validate`, the record count equals K, and every record is `semantic_equal` to gold.
- Also reported:
  - record-level exact accuracy;
  - per-field errors;
  - literal codepoint fidelity;
  - wrong-K, validity and overflow rates.
- Breakdowns: by panel, by K, and by quoted vs unquoted input.

**Redundancy.**
- Silent-error rate: valid but wrong outputs as a share of all outputs.
- Offline parser-detectable single-token substitution rate for E, J, M, C and F.
- Accuracy and latency under the validator fallback.

**Decoding cost per request:** NFE, prefill positions, canvas positions, padded positions and analytic FLOPs.

**Latency (actual wall clock).**

| Item | Specification |
|---|---|
| Hardware | Owner's server with training stopped, `taskset`; governor, kernel and torch/oneDNN versions logged |
| Request | Raw string → tokenize → prefill → decode → detokenize → add header → parse → validate → AST list |
| Timers | `perf_counter_ns` around the request, plus component timers |
| Items | 500 test_iid items (125 per K), the same for every cell |
| Threads | 4 threads for every seed (primary); 1 thread for seed 0 only |
| Warm-up | 30 requests |
| Order | Randomized interleaved blocks |
| Sessions | 2; per-item median across sessions |
| MDM configs | Dev-selected and tau = 0.9 (fixed k = 8 for seed 0 only) |
| Throughput | Batch 64 on test_iid |

Estimated window: 1.3-2.4 h; the reservation is 4 h or less.

**Modeled 7B sign check.** T = prefill + NFE x max(15.2 GB / 768 GB/s = 19.8 ms, 2 x 7.07e9 x positions / 77.5 TFLOP/s), i.e. an A6000-class card at 50% efficiency.

**Resource cost.**
- Training: analytic FLOPs, core-hours, wall time, peak RSS, kWh estimate, US$ (0). Training cost per usable model = run cost / learn rate.
- Inference per 1,000 correct answers: CPU-seconds, NFE and positions.
- **Legend tokens are 0 by construction.** A projection table carries the measured legend costs (E 242, M 139, J 22 Qwen3 tokens) and the break-even: M's complete prompt beats J's only when at least 4 records share one legend. The projection is not used for decisions.

### (g) Analysis

**Matched accuracy A\*.** The highest rung of {95%, 90%} reached on dev by at least 2 of 3 seeds in **each** of AR-E, AR-M, MDM-E and MDM-M, using AR greedy and MDM at k = L. Look 0 uses the rung reached by seed 0 in all four.

**Operating point.** For each MDM model, the sampler configuration with the fewest dev NFE whose dev accuracy is at least A* + 0.5 pp. It is then measured once on test. A cell that never reaches A* gets k = infinity.

**Endpoints.**
- Primary: r_k(x) = k_x/k_E and k_x/k_J for x in {M, C, F}.
- Co-primary: within-MDM latency ratios r_T.
- Reported: I_NFE and I_time with I_0, and I/I_0.
- Secondary:
  - items and FLOPs to A*;
  - learning curves against items and against FLOPs;
  - r_k at the earliest checkpoint where MDM-E reaches 90% (the uncertain regime);
  - held-out panels;
  - silent errors;
  - cost per correct answer.

**Statistics.**
- Ratios: per-seed log ratios with a seed-paired t interval (df = n − 1), **two-sided 90% at every look**. A hierarchical bootstrap (seeds, then items; 10,000 resamples; half-width inflated by t/z) is a sensitivity check.
- Accuracy differences: the rephug/sylang#3 `analyze.py` cluster bootstrap and a seed-paired t interval; the more conservative decides. δ = 2 pp at this scale; the report's 1 pp margin is kept for any scaled follow-up.
- Decisions are intersection-union tests, so no multiplicity correction is applied.

**Operating characteristics of the step rule** (simulated with this exact rule set; between-seed SD of the log step ratio 0.2 / 0.3):

| True k_M/k_E | P(success) | P(stop) | Notes |
|---:|---|---|---|
| 1.00 | 0.014 / 0.045 | 0.986 / 0.955 | Seed-0 futility fires 0.31 / 0.38 |
| 0.90 | 0.12 / 0.16 | | |
| 0.80 | 0.54 / 0.43 | | At the threshold |
| 0.67 | 0.98 / 0.85 | | Seed-0 futility fires wrongly 0.007 / 0.049 |
| 0.50 | 1.00 / 0.995 | | |

### (h) Decision criteria (numeric, pre-registered)

Decision order: preconditions → X0 → PROCEED → STOP → REVISE.

**Preconditions.** If any fails, the result is INCONCLUSIVE: fix the recipe, not the hypothesis.
- **P1:** 100% round trips on 200k items per format and on all 864 meanings; 0 out-of-vocabulary tokens; no version strings; overlap of at most 5%; sampler conformance passes.
- **P2:** Stage-0 gate of at least 7 of 8.
- **P3:** A* of at least 90%.
- **P4:** at least 2 of 3 seeds learned in each key cell.
- **P5:** latency coefficient of variation between sessions below 5%.

**PROCEED.** Licenses only the owner-approved pretrained pair, Dream-v0-Base-7B vs Qwen2.5-7B, starting with its seed-0 futility stage (about US$6-7). All of the following must hold:
- **S1 (passes):**
  - k_M/k_E ≤ 0.80 with the 90% upper bound < 1.00, and the same for k_M/k_J;
  - the point estimate is ≤ 0.90 under the non-selected tau = 0.9 and block-16 samplers.
- **S2 (latency):**
  - within-MDM CPU T_M/T_E ≤ 0.90 and T_M/T_J ≤ 0.90, each with the upper bound < 1.00;
  - p95 ratio ≤ 1.00;
  - modeled-7B ratio ≤ 0.90.
- **S3 (accuracy):**
  - M − E lower bound ≥ −2 pp on test_iid in **both** families;
  - test_combo lower bound ≥ −3 pp;
  - test_long point estimate ≥ −5 pp;
  - silent-error rate M − E: point ≤ +1 pp and upper bound ≤ +3 pp.
- S1 implies I/I_0 ≥ 1.25.

**Labels.**
- "Sylang-specific" only if T_M ≤ 0.90 x T_C and ≤ 0.90 x T_F, each with upper bound < 1.00, at non-inferior accuracy.
- Otherwise "field-aligned, generic": the follow-up must include C or a JSON code tuple, and no Sylang-specific claim is made.

**STOP.** No further diffusion work on M; publish the negative result.
- **X0 (seed-0 futility):**
  - MDM-E has learned;
  - every aligned A in {M, C, F} has seed-0 k_A/k_E ≥ 1.10;
  - no aligned format is 3 pp or more above E in accuracy.
- **X1:** for the best aligned format, the 90% lower bound of k_A/k_E is ≥ 0.90. If this holds for F as well, stop the whole "alignment helps diffusion" line.
- **X2:** the lower bound of T_MDM,M / T_MDM,E is ≥ 0.95.
- **X3:** the upper bound of M − E is < −2 pp in either family at the final exposure.
- **X4:** after look 2, neither success nor X1-X3. A ≥ 20% reduction cannot be shown at this scale.

**REVISE (the hypothesis changes).**

| | Situation | Revision |
|---|---|---|
| R1, vs JSON only | M meets S1-S3 against J but not against E | The claim becomes "compact positional encoding is cheaper than named JSON". It is Sylang-specific only if M also beats C. Add MDM-JT (3 runs, about 1 PFLOP) before any claim against tuple JSON. |
| R2, canvas eats the savings | S1 holds against E but S2 fails | Test a shorter aligned surface (C-like, with no quotes or parentheses) against the revised M only |
| R3, fixed width needed | F meets S1-S2; M and C do not | Redesign M with fixed-width literal slots or a separate literal channel; rerun F vs the revised M (6 runs or fewer) |
| R4, uncertain regime only | S1 holds at the earliest checkpoint where MDM-E reaches 90%, but not at the final exposure | The hypothesis becomes "alignment helps uncertain decoders". Next test underdetermined generation; make no latency claim. |
| R5, general compactness | I/I_0 within [0.8, 1.25] and AR-M at least as good as AR-E | Not diffusion-specific; pursue only through the AR pilot (rephug/sylang#3) |
| R6, scale cannot answer | MDM below 90% at 2 x E_max while AR is at least 95%, or the gate fails after its fixes | Inconclusive, not support for H0. Go to the pretrained pair's seed-0 stage, with owner approval. |

**Run-level stop rules.**
- The ledger refuses to continue beyond 420 core-hours.
- Throughput below 20 GFLOP/s per process switches the plan to Tier-S.
- A lost server reservation means resuming from the last checkpoint.

### (i) Compute and cost (arithmetic checked)

**FLOPs per item.** From the CPU design's per-item figures, with a 5% exact-tokenization margin:
- AR: E 1.55, J 3.44, M 1.87 GFLOP.
- MDM: E 1.59, J 3.70, M 1.99, C 1.59, F 1.77 GFLOP.
- All 8 cells: **18.38 GFLOP per item**; the 4 key cells: 7.35.

**Wall time.** Planning throughput is 25 / 35 / 45 GFLOP/s per process, with 4 processes. The prior 1.12M-parameter runs reached about 27 effective.

| Stage | PFLOP | Hours at 25 / 35 / 45 GFLOP/s |
|---|---:|---|
| Stage 0 (sweep, continuation, second seed) | 2.82 | 7.8 / 5.6 / 4.4 |
| Look 0 (seed 0, 8 cells, plus evaluation) | 5.26 | 14.6 / 10.4 / 8.1 |
| Look 1 (seeds 1-2) | 10.52 | 29.2 / 20.9 / 16.2 |
| **Core (Stage 0 + looks 0-1)** | **18.6** | **51.7 / 36.9 / 28.7** |
| Core if E_max = 128k | about 11 | about 31 / 22 / 17 |
| Look 2 (up to 6 cells x 3 seeds) | 7.53 | 20.9 / 14.9 / 11.6 |
| MDM extension | 9.08 | 25.2 / 18.0 / 14.0 |
| Worst case | 35.0 | 97.8 / 69.9 / 54.3 |
| Expected under H0 | 19.9 | about 55 at 25 |
| Seed-0 futility exit | 8.1 | 22.4 / — / 12.5 |

- The worst case is about 391 process-hours plus about 16 for latency, hence the 420 core-hour ledger ceiling.
- Latency adds a window of 4 h or less.
- Memory is under 4 GB for 4 processes; checkpoints take about 6 GB of disk.

**Money.**
- US$0 cash. Worst-case electricity is about 12 kWh, about US$1.5-3.5.
- Optional: the same code on a Lightning L4 with free credits. Tiny models are launch-bound at about 5-10 ms per step x 8,000 steps, about 1-2 minutes per run. The whole design including evaluation takes about 2-3 GPU-hours, about US$1.6-2.4 at US$0.79/h. Latency is still measured on CPU.
- The US$100 pilot ceiling and the US$1,000 cap are untouched.

**Elapsed time.**
- Engineering: 4-6 days for about 3,000 lines including tests. The CPU design's 2-3 days is optimistic.
- Compute: about 1.5-2.5 days; worst case about 4 days.
- Analysis: half a day.

**Owner approvals:**
1. Installing a pinned PyTorch CPU wheel.
2. Reserving the server for Stage 0 (half a day), then for looks 0-1.
3. Look 2 and the MDM extension.
4. Any GPU use, including free credits.
5. The paid pretrained follow-up.

### (j) What it cannot establish

- **Scale and serving:** 7B, GPU serving, Mercury, approximate caches, or AR-vs-MDM wall-clock ratios. On CPU an MDM step costs 9-20x an AR step; on a 7B GPU it costs about 1-1.5x. So those ratios are reported but carry no decision weight.
- **Pretraining:** pretrained priors, legends, and the mnemonic value of M's codes.
- **Controls:** AR accuracy for C and F, and tuple-JSON arms except through the R1 add-on.
- **Tasks:**
  - edit or transform tasks;
  - underdetermined generation, where parallel-commit conflicts matter;
  - conditionals;
  - K above 6;
  - natural input (the input lexicon is a controlled grammar);
  - Unicode.
- **Statistical resolution:** effects under about 20% in passes or 10% in latency; δ is 2 pp, not 1 pp; escape-rate differences show only when gross.

### (k) Reuse plan

**Reused from this repository:**
- `sylang_core`: `encode`, `decode`, `validate` and `semantic_equal`; `_verb` and `_VERB_LOOKUP` for E; `_CODES` for C and F.
- `evaluation/harness.load_tokenizers`, with the pinned tokenizers version and SHA.
- `evaluation/diffusion_surfaces`.
- `evaluation/diffusion_oracle.FiniteOracle` and `tools/review_diffusion_claims.py` (`threshold_rounds`, `substitutions`, `concise_english`, `tuple_json`) for sampler conformance and detectability.

**Reused from rephug/sylang#3 (`origin/claude/focused-edison-cuu64g`):**
- `pretrained_pilot/grader.py` and its 59 corrupted-output tests, ported to AST grading;
- `analyze.py`: the cluster bootstrap, plus a δ parameter and t/z inflation;
- `evaluate.py` timing and token-accounting conventions;
- `ledger.py`, plus a CPU-hours event;
- the `data.py` freeze and manifest;
- the stage scripts;
- `copy_probe/model.py` as the base network, with its noncausal and detach_kv bug-injection tests;
- `study3.split_of`.

**New code:**

| Module | Content | Lines |
|---|---|---:|
| `data.py` | Generator, lexicon, overlap audit, renderers and parsers for E, J, M, C, F | ~650 |
| `tok.py` | Tokenization and vocabulary remap | ~150 |
| `model.py` | Model changes: RoPE, prefix masks, MDM mode, cache, EMA | ~250 |
| `train.py` | Training loop | ~300 |
| `sample.py` | Samplers | ~250 |
| `evaluate.py` + `latency.py` | Scoring and latency harness | ~400 |
| `analyze.py` | Decision analysis | ~400 |
| Tests | | ~600 |

### (l) Top risks and mitigations

1. **Seeds fail to learn** (prior about 0.4). The Stage-0 gate, a larger model or L4 fallback, intent-to-train, and outcome R6.
2. **The task is too easy, so every MDM decodes in 1-2 steps.** That is itself a decisive X1. R4 keeps the uncertain-regime reading open.
3. **Tokenizer artifacts.** Exact tokenization with 50% quoted inputs. A literal-isolated 1-seed MDM-M/AR-M sensitivity run (about 1 PFLOP) is triggered only if M's name-field errors exceed E's by 2 pp or more.
4. **Copy shortcut for English.** The neutral lexicon and the 5% audit.
5. **Selection bias.** LR is chosen per family on pooled E+M; sampler budgets are chosen on dev and measured on test; non-selected samplers are reported.
6. **Mis-reading I.** Only I/I_0 is interpreted.
7. **MDM bugs** (leakage, 1/t weighting). Leak tests, the toy copy task and oracle conformance.
8. **CPU timing noise.** An exclusive window, `taskset`, interleaving, and the CV precondition. Decisions rest on NFE plus within-family latency ratios.
9. **Over-generalization.** PROCEED buys only the pretrained seed-0 stage.

## Fallback: Tier-S, a stop-or-escalate screen

**When to use it:**
- The owner's server cannot be reserved for about 3 days;
- Stage 0 measures under 20 GFLOP/s per process; or
- the owner wants an answer in about a day.

**Design.** It uses the same pipeline, panels, tokenizer, network, Stage 0, looks and statistics as the recommended design, with **5 cells**: AR x {E, M} and MDM x {E, M, C}, 3 seeds. The cost is 9.02 GFLOP per item, including the 5% margin.

| Configuration | PFLOP | Hours at 25 / 35 / 45 GFLOP/s per process x 4 |
|---|---:|---|
| Stage 0 + 3 seeds at 256k | 2.82 + 7.6 = 10.4 | 29 / 21 / 16 |
| With E_max = 128k | about 6.6 | 18 / 13 / 10 |
| Seed-0 futility exit | about 5.4 | 15 / 11 / 8 |

The cost is US$0.

**What it can establish:**
- STOP outcomes X0-X3 for M against concise English.
- M vs C within the MDM family, i.e. Sylang-specific vs generic aligned.
- I/I_0 for M vs E.
- REVISE outcomes R2, R4 and R5.

**What it cannot establish:**
1. **Anything about JSON**, the required arm: no accuracy, step or latency comparison against J in either family, and no equal-compute contrast with J. Because PROCEED requires beating J, **Tier-S can never return PROCEED**, only STOP or "escalate to the full design".
2. Outcome R1 (the M advantage holds against named JSON only), which is the expected outcome.
3. Whether fixed-width slots are what matters (no F cell), so it cannot separate R3 from STOP for the alignment line as a whole.
4. Whether JSON can be learned at all by an AR model (rephug/sylang#3 saw 0 of 2 runs learn for compact JSON).
5. AR accuracy for C.
6. Everything already listed in (j) of the main design: scale, serving, pretrained priors, legends, edits, conditionals, K above 6, effects under about 20%, δ of 2 pp rather than 1 pp.

**Micro variant.** If even Tier-S is too large, run MDM only, E/M/C x 3 seeds, about 6 PFLOP, about 17 h at 25 GFLOP/s.
- Because AR NFE equals output length, I/I_0 in passes (= k_E/k_M) is still computable.
- It **cannot establish** matched accuracy across families, AR learnability, any measured AR latency, or I in time.
- It therefore does not meet the owner's AR-vs-MDM requirement and can only trigger X0/X1.
