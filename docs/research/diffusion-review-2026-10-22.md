# Independent review: Sylang diffusion research of 2026-10-20

> **Status: draft in progress.** Adversarial verification of the [agent]
> findings and the next-experiment design (sections 9 and 12) are still
> running. Numbers marked [reproduced] or [new] are final.

Review date: **2026-10-22**. Reviewed commit:
`4a621c7ee71547d06215c86a665421dd538e2e14` (branch
`codex/sylang-diffusion-research-20261021`, draft PR rephug/sylang#4).
Reviewed material: [diffusion-sylang-2026-10-20.md](diffusion-sylang-2026-10-20.md),
[cpu-diffusion-oracle-2026-10-20.md](cpu-diffusion-oracle-2026-10-20.md), the
experiment code under `evaluation/diffusion_*.py`,
`tools/analyze_diffusion_tradeoffs.py`, the five `tests/test_diffusion_*.py`
files, and `benchmarks/results/diffusion-{oracle,tradeoffs}.json`.

This is a skeptical review requested by the project owner. It was asked to
test the reports, not to agree with them. No model was trained or run for this
review. Nothing here is a neural result, and no oracle round count is
described as inference speed.

**Evidence labels used below**

- **[reproduced]**: the reviewer re-ran the repository's own code or recomputed
  the value from the raw result files.
- **[new]**: a new computation for this review, using the repository's codec,
  pinned Qwen3 tokenizer and unmodified `FiniteOracle`. Every [new] number in
  sections 2–7 is produced by
  [`tools/review_diffusion_claims.py`](../../tools/review_diffusion_claims.py)
  (deterministic, about 6 s on a 4-core CPU) and saved in
  [`benchmarks/results/diffusion-review-checks.json`](../../benchmarks/results/diffusion-review-checks.json),
  unless marked as coming from a review agent.
- **[agent]**: computed by a review agent and then independently re-derived by
  a second, adversarial agent. Scripts are not committed; their results are
  summarised.
- **[read]**: taken from a document without recomputation.

## 1. Executive verdict

**Recommendation: revise the hypothesis.** Do not proceed with "Sylang M makes
diffusion language models faster" as currently framed. Do not stop the line of
work entirely either.

The work itself is careful. Every number in both reports reproduces exactly.
The code has no demonstrated bugs. The reports already disclaim latency,
neural and Mercury claims, and they say plainly that JSON wins the
complete-prompt comparison. The problem is what the evidence supports:

1. **The compactness premise is mostly an artifact of the version tag.** M's
   "19.5% fewer tokens than English" (864-meaning world) and "11.66% fewer"
   (33 smoke fixtures) come from the per-record version tag: English's
   `Sylang core-v0.1: ` costs 9 Qwen3 tokens and M's `M0.1:` costs 4.
   Remove only those tags and M is **7.3% longer** than English (13.00 vs
   12.11 tokens per record) and **3.3% longer** on the smoke suite (898 vs
   869). [new] M is genuinely shorter only than named-field JSON. A plain
   JSON array of the same seven values recovers most of that gap.
2. **The parallel-decoding advantage depends on the definitions used.** M's 1 adaptive
   round against English 4.69 and JSON 3.00 requires two single-token labels,
   an exactly factorized prior, an exact joint-independence oracle and two
   unfavourable scan orders for the baselines. Two-token names give M 2–3
   rounds, and dropping one of 864 meanings gives M 2.80. A confidence-threshold
   sampler, fed perfect marginals, gives M 7.00 rounds against English 8.69,
   about 19% fewer. [new]
3. **The trade-off is real, but it is redundancy against parallel
   consistency, not compactness against robustness.** M places exactly one
   token on each field. It therefore has the worst erasure recovery (7.81% vs
   English 23.94% at 50% masking) and **no ability to detect substitutions**:
   100% of in-distribution single-token substitutions in M give another valid
   meaning, against 19.3% for English. [reproduced, new]
4. **Payload savings are a small part of the end-to-end cost.** In the only
   end-to-end task the repository defines, every format must output the same
   JSON. M's payload saving is therefore 0.76% of input-plus-output tokens.
   Legends dominate complete prompts, and JSON has the shortest
   single-record prompt. [agent]
5. **No neural model of any kind has ever been trained or evaluated on M.**
   The only learned-model evidence in the project is a 1.12M-parameter byte
   model trained on a different Sylang code format. [agent, read]

What survives is narrower and not specific to Sylang. A layout with **one
token per field and fixed field widths** makes factorized parallel sampling
more coherent. It also gives shorter outputs than named-field JSON. Any
positional code has the same properties, including a JSON array of codes.

**Concrete next action:** amend the two reports' headline claims (section 10)
and pre-register the revised hypothesis. Then run the small matched experiment
in section 9, which includes a generic aligned-code control. If that
experiment cannot separate M from the generic control, stop the
Sylang-specific diffusion claim.

## 2. What was reproduced and what was only read

| Item | Status |
| --- | --- |
| 92 tests (`python -m unittest discover -s tests`) with the pinned tokenizers | **[reproduced]** 92/92 pass. With system Python and no `tokenizers` package: 90 pass, 2 skip |
| 37 oracle/fusion/experiment tests under `python3 -S` | **[reproduced]** pass |
| Full oracle run (`lexical,utf8_bytes,qwen3,qwen3.5`, 16 masks per meaning) | **[reproduced]** in 47 s. Byte-identical to `diffusion-oracle.json` except `environment.python` (3.12.3 saved vs 3.13.16 here) |
| SHA-256 `fa73a6d0…7460` of the saved oracle JSON; six source hashes | **[reproduced]** match |
| `tools/analyze_diffusion_tradeoffs.py` | **[reproduced]** byte-identical output |
| Pinned Qwen3/Qwen3.5 tokenizer assets | **[reproduced]** downloaded; size and SHA-256 match `benchmarks/tokenizers.lock.json` |
| All numeric claims in both reports (56 rows) | **[agent]** every value matches the raw data within the stated rounding |
| Vendor and preprint claims (Mercury 3, BLD, JPEG-DLM and others) | **[agent]** fetched; all sources exist with the stated dates; characterisations in section 6 |
| Neural accuracy, denoising steps, latency | Not measured anywhere in the repository. Nothing to reproduce |

## 3. Question 1: what was actually measured

| Quantity | What it is | Evidence type |
| --- | --- | --- |
| Token counts (Qwen3, Qwen3.5, lexical splitter, UTF-8 bytes) | Deterministic tokenizer counts of codec output | Measured |
| Residual entropy and Bayes recovery ceiling under masking | Exact posterior over a known finite prior. Interior mask rates are averaged by Monte Carlo over 16 masks per meaning (SEs are correct and calibrated [agent]) | Exact computation plus mask sampling |
| Matched semantic erasure (7.8125%, identical for all formats) | Computed once on native slots and copied (`evaluation/diffusion_experiment.py:293-297`). True by construction for bijective codecs | A theorem, not a measurement |
| "Rounds" (adaptive, static, one-round support) | Exact expectations for **named idealised samplers** that know the prior and, for the adaptive policy, can test exact joint independence (`evaluation/diffusion_oracle.py:251-355`) | Exact computation, sampler-specific |
| Speed-up, break-even and block tables | Arithmetic on **input payload** token counts (`docs/research/diffusion-sylang-2026-10-20.md:133-152`) | Hypothetical scenarios |
| Neural inference steps | None anywhere | Not measured |
| Real elapsed time or latency | None for any model. The only timing data in the repository are earlier Windows tokenizer/codec samples quantised to 0 or 15.6 ms, which neither report uses [agent] | Not measured |
| Mercury 3 throughput, BLD and JPEG-DLM speed-ups | Vendor and preprint claims about other systems | External |

The reports label these classes correctly in most places
(`cpu-diffusion-oracle-2026-10-20.md:3-6`, `diffusion-sylang-2026-10-20.md:4-6`,
`:111-114`, `:139`). The overstatements are smaller and are listed in
section 8: the headline "compactness trades…" sentence, analysis JSON keys
named `*_speedup` and `*_steps_break_even`
(`tools/analyze_diffusion_tradeoffs.py:68-69,90-91`), an "8.4%" threshold that
belongs to one fixture, and a 50% masking table that does not say target
length was given.

## 4. Question 2: is the English/JSON/M comparison fair?

Ranked by how much each issue moves the headline numbers.

1. **Version tag (critical).** See section 1, point 1. The codec adds
   `Sylang core-v0.1: ` to English and `M0.1:` to M (`sylang_core/codec.py:215,218`).
   In Qwen3 the English tag is 9 tokens and the M tag 4 (the colon merges into
   `:p`). That gap is larger than M's entire average saving of 4.11 tokens per
   record. [new] The tag is mandatory in the grammar, but its length is a
   naming choice that carries zero information. A batch of records would send
   it once, or not at all. Neither report says the tag accounts for more than
   the whole gap. The earlier independent review on PR rephug/sylang#2 reached
   the same conclusion on 2026-10-04 [read], and the diffusion reports do not
   cite it.

   | Mean Qwen3 tokens, 864-meaning world [new] | Mean | Range |
   | --- | ---: | --- |
   | Codec English (as reported) | 21.11 | 19–23 |
   | Codec English without the version tag | 12.11 | 10–14 |
   | Concise English (unquoted labels, no tag, default evidence unmarked) | 7.86 | 4–10 |
   | Named-field JSON (as reported) | 46.83 | 46–48 |
   | JSON array of words, e.g. `["A","see","A","positive","past","simple","unspecified"]` | 15.83 | 15–17 |
   | JSON array of M's codes, e.g. `["A","s","A","+","p","s","u"]` | 14.50 | 14–15 |
   | M (as reported) | 17.00 | 17 |
   | M without the version tag | 13.00 | 13 |

   The concise English is injective on this world only and is not a codec; it
   shows how much of "English" is template.
2. **Single-token labels (major).** The world uses only `LABELS = ("A", "B")`
   (`evaluation/diffusion_experiment.py:31`), and each is one Qwen3 token.
   That is what gives M constant length, exactly one varying token per field,
   and its 1-round result. Section 5 shows what realistic names do. The
   project's earlier learned-model work found identifier shape to be the main
   failure mode [read], and this world keeps it fixed.
3. **Sampler definition (major).** The round comparison depends on the
   sampler; see section 5.
4. **Controlled English is a verbose template (major).** It always quotes
   labels and spells out "Without specified evidence," even for the default
   value (`sylang_core/codec.py:46-51`). The grammar itself calls it a control
   serialisation (`docs/experimental-grammar.md:204-206`). A concise variant
   is 7.86 tokens, keeps the same recovery ceiling (24.58% vs 23.94% [agent])
   and needs fewer adaptive rounds (3.94 / 5.54 [agent]).
5. **Named-field JSON is not the strongest JSON baseline (major).** The
   report says so (`cpu-diffusion-oracle-2026-10-20.md:27-28`;
   `diffusion-sylang-2026-10-20.md:361-363`) but did not run the array
   control it recommends. The array of M's codes matches M's one-round result
   when length is given [agent], because the result comes from token-to-field
   alignment, which any positional code provides.
6. **Legends excluded from the payload experiments (moderate).** The legends
   are 242 Qwen3 tokens for English, 139 for M and 22 for JSON
   (`evaluation/legends/*.txt`). With the repository's own prompt builder
   (`evaluation/harness.py:107-122`), a single-record prompt is shortest in
   JSON: 266.8 tokens against M 354.0 and English 461.1 [agent]. The oracle
   gives every format a perfect, free legend.
7. **Training exposure is absent, which favours M (moderate).** The oracle
   knows each format's exact prior, including M's position-dependent codes
   (`s` means *see* as a relation and *simple* as an aspect,
   `sylang_core/codec.py:38-44`). A pretrained model knows English and JSON,
   not M. Perfect knowledge for every format removes M's main practical
   handicap.
8. **Points that hold up.** All three formats encode the same AST; the extra
   content is constant. Stopping thresholds are identical for every format
   (`evaluation/diffusion_experiment.py:174,191-193`). Masks that fall on
   constant tokens do not bias recovery under independent Bernoulli masking.
   English's erasure advantage is real and lies entirely in polarity, tense,
   aspect and evidence; subject, relation and object are equally fragile in
   every format [agent]. Qwen3.5 tokenizes all 2,592 surfaces identically to
   Qwen3, so it is consistent with Qwen3 but not an independent replication
   [agent].

## 5. Question 3: does the oracle's parallel-decoding advantage transfer?

**Why M needs one round.** In the Qwen3 M stream, 10 of 17 positions are
constant and each of the other 7 corresponds one-to-one with one semantic
field. The prior is an exact Cartesian product, so every field-aligned
position passes the exact integer test `|J∩T|·n == |J|·|T|`
(`evaluation/diffusion_oracle.py:313-314`) and all positions join round 1
[reproduced]. English has 1 of 14 varying positions aligned with a single
field, JSON 6 of 14; JSON's misaligned positions come from Qwen3 splitting
`unspecified` and `inferred` into two tokens [agent].

**The unrestricted joint oracle needs one round for every format.** That is
hard-coded at `evaluation/diffusion_experiment.py:208` and true by definition.
The report states it (`cpu-diffusion-oracle-2026-10-20.md:123-124,155-156`).
The consequence is that round counts do not measure a representation's
intrinsic parallelism. They measure how far each format's token-level joint
distribution is from a product of per-position marginals, under one sampler
class that also needs an exact dependency oracle no trained model has.

**Small perturbations change the numbers a lot.** Fixed public canvas, full
erasure, adaptive policy, forward / reverse mean rounds:

| Condition [new] | English | JSON | M |
| --- | ---: | ---: | ---: |
| As reported: labels A/B | 4.69 / 6.00 | 3.00 / 3.50 | **1.00 / 1.00** |
| Equal-length two-token labels (Priya/Fatima) | 5.69 / 6.50 | 3.00 / 3.50 | 2.00 / 2.00 |
| Unequal realistic labels (Priya/Xiomara) | 6.69 / 8.75 | 5.00 / 6.25 | 3.00 / 3.25 |
| Drop 1 of 864 meanings (total correlation 0.0017 bits) | — | — | 2.80 / 2.38 (worst 7) |

After dropping one meaning, a one-round factorized sampler is still 99.89%
valid for M. The adaptive count nearly triples only because 863 is prime and
the exact equality test can no longer pass [new, agent]. Within the same exact
policy class, a per-round information-maximising scan order gives English
2.79 and JSON 2.00 rounds, so the reported forward and reverse orders are both
pessimistic for the baselines [agent]. Over 300 random orders English ranges
2.89–6.61 rounds (median 4.04); M is 1.00 in every order [agent].

**Published samplers use marginals, not independence tests.** LLaDA unmasks a
fixed number of tokens per step and its README says quality is best when steps
equal length. Fast-dLLM commits every token above a confidence threshold
(default 0.9), plus at least one. EB-Sampler bounds the summed entropy [agent,
external]. With exact marginals in an idealised threshold sampler at τ = 0.9:

| Rounds, fixed canvas [new] | English | JSON | M |
| --- | ---: | ---: | ---: |
| Independent world | 8.69 (max 11) | 8.83 (max 10) | 7.00 (max 7) |
| Correlated world | 6.50 (max 9) | 5.83 (max 7) | 4.67 (max 5) |

M keeps a lead, but it is about 19% rather than 4.7×. A slightly different
threshold rule used by another review agent gave M 8.0 against English 10.81
(26% fewer), and an EB-Sampler analogue gave 7.0 against 9.28 [agent]. The
lead survives realistic names (7.25 against 10.08–11.13) and prior
perturbation (7.00 against 8.71) [agent]. Under LLaDA-style fixed-k decoding,
M is fully coherent in the independent world but is the *least* coherent
format in the correlated world (failure mass 0.907 against English 0.823 at
k = 4) [agent].

**Conditional generation removes the gap.** All refinement numbers model
sampling a random meaning from 9.75 bits of entropy. The intended Sylang
workloads, serialising or transforming a known meaning, are low-entropy given
the context. With all seven fields given, every format needs exactly one round
under both policies. With four given, adaptive rounds are English 2.83, JSON
1.86, M 1.00 [agent].

**Verdict on question 3: both a real effect and an artifact.** Most of the
reported magnitude, 1 against 4.69, comes from how fields, tokens, the prior
and the independence test were defined. A modest residual is real: fixed-width
fields with one token each let a marginal-only sampler commit coherently in
fewer rounds when output uncertainty is high. It is a property of field
alignment, which a code tuple also has, and the most transferable measure of
it is validity at a fixed step budget, not adaptive round counts. Whether a
trained denoiser learns accurate marginals for dense one-token codes is not
addressed by any of this.

## 6. Question 4: compactness against error recovery

**Reported numbers verified** [reproduced]. At 50% Qwen3 token masking with
the target length given, independent world:

| Format | Mean tokens | Residual entropy | Bayes recovery ceiling | Mask SE |
| --- | ---: | ---: | ---: | ---: |
| English | 21.11 | 2.930 bits | 23.94% | 0.21 pp |
| JSON | 46.83 | 3.785 bits | 12.66% | 0.13 pp |
| M | 17.00 | 4.888 bits | 7.81% | 0.11 pp |

Exact enumeration gives 23.71%, 12.68% and exactly 5/64 = 7.8125%; the saved
Monte Carlo values lie within 1.1 SE [agent]. M's value is exactly
semantic-field erasure: with one varying token per field,
E[1/m] = ∏ᵢ((1−p) + p/kᵢ) = (3/4)³(2/3)³(5/8) = 5/64.

**The "19.5% fewer tokens" is not a compactness result.** See section 4.
Without the version tag, M is longer than English in both corpora. M is
compact against named-field JSON (64% fewer output tokens in this world), and
about 10–18% shorter than a JSON array (14.50 and 15.83 tokens against 13.00
for tag-free M) [new].

**Where English's advantage comes from.** Almost all of it is redundancy
across tokens. English spreads polarity, tense, aspect and evidence over
several tokens (`will not have contained`, `By inference,`). Giving the true
length adds only about 0.045 bits at p = 0.5; it matters more at high masking
[agent]. To leave English with as much residual entropy as M has at p = 0.5,
about 75% of English tokens must be masked [agent]. At an equal number of
*visible* tokens English also recovers more than M for every budget below about
14 tokens [agent].

**Token count does not determine robustness.** JSON is the longest Qwen3
format, but 37 of its tokens are constant. Lexical-splitter JSON (83 positions)
has exactly M's 5/64 ceiling, and with UTF-8 bytes JSON (178 positions)
recovers 53.67% [agent]. Robustness is set by how many independently erased
units carry each field.

**Substitutions are not covered by the reports.** The experiments use only
erasures, where corrupted positions are flagged. A sampler's actual mistakes
are wrong commitments. Replace one token with another token seen at that
position [new]:

| Format | Substitutions | Silent: another valid meaning | Detectable: invalid |
| --- | ---: | ---: | ---: |
| English | 25,128 | 19.3% | **80.7%** |
| JSON | 11,664 | 60.5% | 39.5% |
| M | 10,368 | 100.0% | **0.0%** |

Every M field combination is grammatical, so a validator cannot catch a wrong
polarity, tense, aspect or evidence code. That cuts against the reports'
hypothesised "validate, constrain and selectively repair" benefit
(`diffusion-sylang-2026-10-20.md:88-89,229-234`) for exactly the fields that
matter. This is an idealised error model; real model error distributions are
unmeasured.

**Limitations of the trade-off evidence.** Uniform synthetic priors; Bayes
ceilings, not model accuracy; independent Bernoulli erasures rather than real
error processes; two one-token labels; one fixed predicate shape; gold length
in the headline table. The correlated-world token-masking results, computed
but not reported, favour English more strongly: 93.6% against M 73.2% at 50%
masking [agent].

## 7. Question 5: data, statistics, leakage and reproducibility

- **Reproducibility is excellent.** See section 2. One gap: the hashed source
  list (`evaluation/diffusion_experiment.py:300-302`) leaves out
  `evaluation/harness.py`, which loads and pins the tokenizers.
- **The 864-meaning world is a designed population, not a sample.** The 19.5%
  has no sampling variance; its sensitivity is to design choices (tag, labels,
  codes). JSON length is exactly additive in the seven field values, so the
  world encodes about 19 per-value length facts plus boilerplate [agent].
- **The 11.66% smoke-suite figure has no uncertainty statement.** A bootstrap
  over the 33 authored fixtures puts the saving at 7.3–16.6%; clustering by
  fixture family gives 7.9–18.9% [agent]. The fixtures are authored, so this is
  a sensitivity range, not population inference.
- **The "actual suite" quadratic proxy (1.084×) and its 8.4% threshold come
  from one fixture.** `literal-max-length` supplies 78% of the summed squared
  English lengths. Without it the proxy is 1.40× [agent]. The report names the
  fixture (`diffusion-sylang-2026-10-20.md:146-150`) but still contrasts the
  number with the uniform scenario.
- **Leakage.** No oracle path uses hidden gold beyond the declared length
  condition [agent]. The fixed canvas reveals almost all length information at
  intermediate mask rates (97.7% of English's 1.975 bits at p = 0.5) but this
  does not drive the 50% table: exact English recovery is 23.62% on the fixed
  canvas against 23.71% with gold length [agent].
- **Tests are sound where they exist but leave headline numbers unprotected.**
  Mutation testing killed 8 of 19 single-point mutants. The survivors include
  standard-error formula changes, removing fixed-canvas padding in
  `run_experiments`, and inverting the tradeoff tool's block speed-up. No test
  calls `run_experiments()` or `analyze()`, so the 19.5%, the 50% table, the
  round table and the 11.66% are protected only by reproduction, which
  succeeds today [agent]. The matched-semantic test is tautological:
  replacing every surface with arbitrary unique streams gives the same result
  [agent].
- **Independence and correlation.** The report's "independent" world has no
  correlated examples by construction. Its "correlated" world has 18 meanings
  driven by three latent variables, and its 16-mask estimates carry 1–2 pp SEs
  [agent]. Neither world says anything about generalisation, because no
  learning happens.

## 8. Question 6: the end-to-end workflow

| Stage | Measured? | Notes |
| --- | --- | --- |
| Natural language → semantic tree | No | No translator exists (`README.md:9`; `diffusion-sylang-2026-10-20.md:47-49`). ESTIMATE [agent]: an LLM translation call reads at least the M legend (139 tokens) and task envelope (~196) and writes M (~31 tokens per smoke record), roughly two orders of magnitude more than the ~4-token saving per record |
| Tree → text (codec) | Yes, by this review | 17–140 µs per record for encode plus decode; English is slowest because its decoder re-sorts candidate phrases (`sylang_core/codec.py:299-308`). Codec CPU time only [agent] |
| Legend and instructions | Counted, not amortised | Fixed overhead per request: English 440, JSON 220, M 337 Qwen3 tokens, repeated in all 33 prompts [agent] |
| Chat template, system prompt | No | `evaluation/harness.py:277` |
| Model input and output tokens | Input counted; output not | The only defined task asks every format to output the same JSON (`evaluation/harness.py:120-121`), so output savings are zero and the JSON arm is a copy [agent] |
| Denoising passes, latency | No | Oracle rounds are idealised sampler counts |
| Decoding and validation | Code exists | Fail-closed codec and validator (`sylang_core/codec.py:396-430`) |
| Error handling and retries | No | No repair or retry code; failure rates unmeasured |
| Rendering back for people | Template only | Controlled English with tag and quotes; fluent prose would need a model |

In the measured protocol, M's payload saving against English is **0.76% of
input-plus-output tokens** (smoke suite) and 0.81% in the 864-meaning world.
With equal legends, an M failure rate only 0.76–0.81 percentage points higher
than English's would erase the saving [agent]. The complete-prompt work
proxies in the tradeoff tool (M against English 1.291× linear, 1.655×
quadratic) are 96% legend length; by the same proxy JSON is fastest
(1.650× / 2.663× against English), but the tool's console output prints only the
M-against-English line [agent]. The reports do not quote those proxies as
speed-ups.

Two results point the other way and are worth keeping. If four or more records
share one legend, M's complete prompt beats JSON's. Emitting M instead of
named-field JSON would cut *output* tokens by 54% on the smoke suite and 64% in
the 864-meaning world; a JSON array of codes gets most of that saving [agent].
Neither has been tested with a model.

## 9. Proposed next experiment

*To be completed.*

## 10. Methodological flaws ranked by severity

| Severity | Flaw | Where | Effect |
| --- | --- | --- | --- |
| Critical | Payload savings against English come from the per-record version tag; tag-free M is longer | `cpu-diffusion-oracle-2026-10-20.md:71-73`; `diffusion-sylang-2026-10-20.md:58,135,146`; `sylang_core/codec.py:215,218` | Removes the compactness premise against English and invalidates the speed-up rows built on r = 1030/1166 |
| Major | M's 1-round and 100% one-round support results require one-token labels and an exactly factorized prior | `evaluation/diffusion_experiment.py:31`; `cpu-diffusion-oracle-2026-10-20.md:9-10,121,147` | Headline parallel-sampling result does not survive realistic names or tiny prior changes |
| Major | Adaptive rounds use an exact joint-independence oracle and two pessimistic baseline scan orders; no published sampler is compared | `evaluation/diffusion_oracle.py:251-355`; `cpu-diffusion-oracle-2026-10-20.md:126-156` | 1 against 4.69 shrinks to roughly 7 against 8.7 under a marginal-based sampler |
| Major | Baselines are a verbose English template and named-field JSON; the recommended array control was not run | `sylang_core/codec.py:46-51,209-211`; `diffusion-sylang-2026-10-20.md:361-363` | Credits Sylang with gains a generic positional code also gives |
| Major | Recovery analysed only for erasures; substitutions, the errors a sampler makes, are silent in M | `cpu-diffusion-oracle-2026-10-20.md:42-48` | Hides M's loss of error detection |
| Major | No learned-model evidence on M; prior work on PR rephug/sylang#2 and rephug/sylang#3 is not cited | both reports | The case rests entirely on oracle and tokenizer arithmetic |
| Moderate | Only defined end-to-end task outputs identical JSON for all formats; payload saving < 1% of request | `evaluation/harness.py:107-122` | Payload savings do not translate into end-to-end savings |
| Moderate | All refinement results use full erasure (unconditional generation) | `evaluation/diffusion_experiment.py:174-230` | Gap vanishes when the meaning is determined by context |
| Moderate | No uncertainty for 11.66%; quadratic proxy rests on one fixture | `diffusion-sylang-2026-10-20.md:58,146-150` | Precision overstated |
| Moderate | Headline numbers not protected by tests; 11 of 19 mutants survive | `tests/test_diffusion_*.py` | Regressions would be caught only by reproduction |
| Minor | Headline sentence credits "compactness"; matched-semantic result presented as a finding; 50% table does not say length is given; correlated token-masking results omitted; Qwen3.5 not independent; "output" added to Mercury's "1,068 tokens per second"; BLD cited as "directly supports" while BLD reports harder joint generation and slower batch-1 latency than ELF-L; LLaDA GUIDELINES.md has no inference section | various | Presentation |

## 11. Strongest arguments for and against continuing

**For continuing**

- The computational work is sound, reproducible and honestly caveated. It is a
  good base for a real experiment.
- Field-aligned, fixed-width encodings have a modest but robust coherence
  advantage for marginal-based parallel samplers: about 19–26% fewer rounds
  with perfect marginals, surviving realistic names and prior perturbation
  [new, agent].
- M's output is 54–64% shorter than named-field JSON, the de facto format for
  structured output, and that is exactly where diffusion decoding spends its
  passes.
- Dream-v0-7B, an open masked-diffusion model, uses a tokenizer identical to
  the repository's pinned Qwen3 tokenizer, so the token counts apply directly
  to a real dLLM, and Dream and Qwen2.5-7B form a matched pair with shared
  initialisation [agent, external].
- In the project's only learned-model study, trained from scratch, a compact
  Sylang code tied English in range and clearly beat compact JSON and CSV
  [read, agent].

**Against continuing as framed**

- Against English, M is not more compact once the version tag is removed.
- The parallel-decoding headline depends on one-token labels, an exact
  product prior and an exact independence oracle.
- M gives up redundancy and all substitution detection.
- Payload savings are under 1% of a measured request; legends and output
  format dominate.
- Every advantage that survives is shared by a generic positional code, so
  none is yet specific to Sylang.
- At small scale, masked diffusion needs roughly 14–16× the compute of AR to
  match validation loss [agent, external], so a fair diffusion experiment is
  more expensive than the report implies, and the report gives no budget.
- No model has ever been trained or evaluated on M.

## 12. Recommendation

*To be completed.*

## Appendix: reproducing this review

```sh
python3 -m venv .cache/diffusion-venv
.cache/diffusion-venv/bin/python -m pip install --only-binary=:all: --no-deps -r requirements-tokenizers.txt
python3 tools/fetch_tokenizers.py --download
.cache/diffusion-venv/bin/python -m unittest discover -s tests
.cache/diffusion-venv/bin/python -m evaluation.diffusion_experiment \
  --tokenizers lexical,utf8_bytes,qwen3,qwen3.5 --samples-per-state 16 \
  --output .cache/oracle-repro.json
.cache/diffusion-venv/bin/python tools/review_diffusion_claims.py \
  --output benchmarks/results/diffusion-review-checks.json
```

Compare `.cache/oracle-repro.json` with `benchmarks/results/diffusion-oracle.json`;
only `environment.python` should differ.
