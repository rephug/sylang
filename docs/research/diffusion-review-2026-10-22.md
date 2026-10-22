# Independent review: Sylang diffusion research of 2026-10-20

> **Status: draft in progress.** Sections 9 and 12 (next experiment and
> recommendation) are still being finalised. Everything else has been through
> adversarial verification.

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
described as inference speed or latency.

**How the review was done.** The reviewer read both reports and the code,
reran the tests and the full experiment, and wrote a reproduction script for
the most important new numbers. Eight review agents then each took one
question (what was measured, fairness, oracle transfer, compactness against
recovery, statistics and code, end-to-end cost, external sources, prior
project evidence). A second, independent agent re-derived every finding with
its own code and tried to refute it. None of the 103 findings was refuted; 60
were corrected, mostly in precision or severity, and the corrected versions are
used here.

**Evidence labels**

- **[reproduced]**: the reviewer reran the repository's own code or recomputed
  the value from the raw result files.
- **[new]**: a new computation using the repository's codec, pinned Qwen3
  tokenizer and unmodified `FiniteOracle`, produced by
  [`tools/review_diffusion_claims.py`](../../tools/review_diffusion_claims.py)
  (deterministic, about 6 s on a 4-core CPU) and saved in
  [`benchmarks/results/diffusion-review-checks.json`](../../benchmarks/results/diffusion-review-checks.json).
- **[agent]**: computed by a review agent and independently re-derived by a
  verifying agent. Those scripts are not committed; this document reports their
  results.
- **[read]**: taken from a document or external source without recomputation.

## 1. Executive verdict

**Recommendation: revise the hypothesis.** Do not proceed with "Sylang M makes
diffusion language models more efficient" as currently framed. Do not stop
entirely either: a narrower, testable hypothesis survives.

The work is careful and honest. Every number in both reports reproduces
exactly, the code has no demonstrated bugs, and the reports already disclaim
latency, neural and Mercury claims and state that JSON wins the
complete-prompt comparison. The problems are in what the evidence is taken to
show.

1. **M is not more compact than English once the baselines are fair.** The
   reported savings (19.5% in the 864-meaning world, 11.66% on the 33 smoke
   fixtures) are a near-constant 4.1 tokens per record. The per-record version
   tags account for 97–122% of that: English's `Sylang core-v0.1: ` is 9 Qwen3
   tokens, M's `M0.1:` is 4. Removing only the tags, M is 7.3% *longer* than
   the codec's English (13.00 vs 12.11 tokens per record) and 3.3% longer on the
   smoke suite. Stripping every constant token instead makes M 10% shorter, so
   the sign depends on the convention. Against a concise English, M is longer
   under every convention tested; on the smoke suite by 41% (1,030 vs 730
   tokens). The report's own linear break-even scenario then gives 0.71×, not
   1.13×. [new, agent]
2. **The parallel-decoding headline is mostly definitional, but a real residual
   survives.** M's 1 adaptive round against English 4.69 needs single-token
   labels, an exactly factorized prior and an exact joint-independence test no
   real sampler has. Two-token names give M 2–3 rounds; dropping one of 864
   meanings gives 2.80. What survives: with fields that are truly independent,
   M is the only format that stays fully valid under aggressive parallel
   decoding (100% valid at a 0.5 confidence threshold, where English is 1.9%
   valid). When fields are correlated, the same layout makes one-step decoding
   fail (2.1% valid). This property comes from one token per field and fixed
   field widths, not from anything specific to Sylang. [new, agent]
3. **M gives up all redundancy.** Its erasure recovery equals pure field
   erasure (7.81% against English 23.94% at 50% masking), and none of its
   in-distribution single-token substitutions can be detected (100% give another
   valid meaning; English flags 80.7%). [reproduced, new]
4. **End-to-end savings are small and partly untested.** The repository's
   comprehension harness asks every format to reconstruct the same JSON, which
   makes the JSON arm a copy task. Under that protocol, M's payload saving is
   0.76–0.81% of input-plus-output tokens per uncached single-record request,
   and 4.0–6.1% if the shared instructions are prefix-cached. Legends dominate
   complete prompts. [agent]
5. **No neural model has been trained or evaluated on M anywhere in this
   repository or its branches.** The only learned evidence is a 1.12M-parameter
   byte model trained on a different Sylang code, which tied English under the
   study's own pre-registered rule and beat compact JSON. [agent, read]

**What to test instead.** A layout with one token per field and fixed field
widths, which Sylang M is one instance of, may let a masked-diffusion decoder
commit more positions per step at equal accuracy, and may shorten structured
output against named-field JSON. Both claims need a generic aligned-code
control, so that a result can be attributed to Sylang specifically or not.

**Concrete next action:** see section 12.

## 2. What was reproduced and what was only read

| Item | Status |
| --- | --- |
| 92 tests (`python -m unittest discover -s tests`) with the pinned tokenizers | **[reproduced]** 92/92 pass. With system Python and no `tokenizers` package: 90 pass, 2 skip |
| 37 oracle/fusion/experiment tests under `python3 -S` | **[reproduced]** pass |
| Full oracle run (`lexical,utf8_bytes,qwen3,qwen3.5`, 16 masks per meaning) | **[reproduced]** in 47 s. Identical to `diffusion-oracle.json` except `environment.python` (3.12.3 saved vs 3.13.16 here) |
| Standard-library-only oracle run | **[agent]** reproduces the lexical, byte, native and fusion sections in 23 s |
| SHA-256 `fa73a6d0…7460` of the saved oracle JSON; six recorded source hashes | **[reproduced]** match |
| `tools/analyze_diffusion_tradeoffs.py` | **[reproduced]** byte-identical output |
| Pinned Qwen3/Qwen3.5 tokenizer assets | **[reproduced]** downloaded; size and SHA-256 match `benchmarks/tokenizers.lock.json` |
| Every numeric claim in both reports (56 rows) | **[agent]** all match the raw data within the stated rounding; one citation is off by one line |
| Vendor and preprint sources | **[agent]** all exist with the stated dates; characterisations in sections 5 and 10 |
| Neural accuracy, denoising steps, latency | Not measured anywhere in the repository; nothing to reproduce |

## 3. Question 1: what was actually measured

| Quantity | What it is | Evidence type |
| --- | --- | --- |
| Token counts (Qwen3, Qwen3.5, lexical splitter, UTF-8 bytes) | Deterministic tokenizer counts of codec output | Measured |
| Residual entropy and Bayes recovery ceiling under masking | Exact posterior over a known finite prior. Interior mask rates are averaged by Monte Carlo over 16 masks per meaning; the standard errors are correct and well calibrated against exact enumeration [agent] | Exact computation plus mask sampling |
| Matched semantic erasure (7.8125%, identical for all formats) | Computed once on native slots and copied into the results (`evaluation/diffusion_experiment.py:293-297`) | A theorem for bijective codecs, not a measurement |
| "Rounds" (adaptive, static, one-round support) | Exact expectations for **named idealised samplers** that know the prior; the adaptive policy can also test exact joint independence (`evaluation/diffusion_oracle.py:251-355`) | Exact computation, specific to the sampler |
| Speed-up, break-even and block tables | Arithmetic on **input payload** token counts (`docs/research/diffusion-sylang-2026-10-20.md:133-152`) | Hypothetical scenarios |
| Neural inference steps | None anywhere | Not measured |
| Model latency or elapsed time | None. The only timing data are process-CPU samples of tokenization and codec calls from an earlier Windows run, quantised to 0 or 15.6 ms and already disclaimed (`docs/baseline-results.md:107-111`); neither report uses them | Not measured |
| Mercury 3 throughput, BLD and JPEG-DLM speed-ups | Vendor and preprint claims about other systems | External |

The reports label these classes correctly in most places
(`cpu-diffusion-oracle-2026-10-20.md:3-6`, `diffusion-sylang-2026-10-20.md:4-6`,
`:111-114`, `:139`). The smaller overstatements are listed in section 10. The
largest is the headline sentence "compactness trades erasure robustness
against parallel consistency" (`cpu-diffusion-oracle-2026-10-20.md:8`), which
credits compactness with effects the data tie to token-to-field alignment.
Lexical-splitter JSON, 83 positions long, behaves exactly like M on both
erasure and rounds because it also has one varying unit per field [agent].
Where alignment *is* held fixed, as in the native-slot fusion arms, compactness
does matter, but only when a fused unit encloses a dependency.

## 4. Question 2: is the English/JSON/M comparison fair?

Ranked by how much each issue moves the headline numbers.

**4.1 Version tags and template English (major).** The codec writes
`Sylang core-v0.1: ` before English and `M0.1:` before M
(`sylang_core/codec.py:215,218`). Each core-v0.1 document holds exactly one
statement (`docs/experimental-grammar.md:32`), so the tag is per record by
design. Its length is a naming choice that carries no information.
`docs/baseline-results.md:16` discloses that headers are counted, and
`diffusion-sylang-2026-10-20.md:293` proposes measuring them separately, but
neither report quantifies the effect. The earlier independent review on PR
rephug/sylang#2 had already concluded that the 11.6% saving is "a property of a
weak baseline, not of M" [read]. The diffusion reports do not cite it, and
their branch's own decision gate asks to "compare M against the strongest
compact baseline, not only English" (`docs/decisions.md:42`).

| Mean Qwen3 tokens per record, 864-meaning world | Mean | Range | Source |
| --- | ---: | --- | --- |
| Codec English (as reported) | 21.11 | 19–23 | [new] |
| Codec English, version tag removed | 12.11 | 10–14 | [new] |
| Concise English, labels quoted (lossless for any label) | 10.61 | 6–13 | [agent] |
| Concise English, labels unquoted (injective on this world only) | 7.86 | 4–10 | [new] |
| Named-field JSON (as reported) | 46.83 | 46–48 | [new] |
| JSON array of words, e.g. `["A","see","A","positive","past","simple","unspecified"]` | 15.83 | 15–17 | [new] |
| JSON array of M's codes, e.g. `["A","s","A","+","p","s","u"]` | 14.50 | 14–15 | [new] |
| M (as reported) | 17.00 | 17 | [new] |
| M, version tag removed | 13.00 | 13 | [new] |

"Concise English" here means: no tag, the default evidence value left
unmarked, and the other three as *Directly*, *Reportedly* and *Presumably*.
For example, `A saw A.` and `Presumably, B will not have contained B.`

On the 33 smoke fixtures: with tags, English 1,166 and M 1,030; tags removed,
869 and 898; concise English 730 [new, agent]. The "M is shorter" direction
against tag-free English depends on the boilerplate convention: removing every
constant edge token instead of only the tag gives M 10.00 against English
11.11. Against concise English, M is longer under every convention tested
(+1.4% to +56.5%) [agent]. M is clearly shorter than named-field JSON, and 10–18%
shorter than the JSON arrays once its tag is removed.

**4.2 Single-token labels (major).** The world uses only `LABELS = ("A", "B")`
(`evaluation/diffusion_experiment.py:31`), each one Qwen3 token. That is what
gives M a constant length, exactly one varying token per field, and its 1-round
result. 38 of the 80 label occurrences in the project's own smoke fixtures are
multi-token in M, including every label in both holdout fixtures [agent].
Section 5 shows the effect. Earlier learned-model work in the project found
identifier shape to be the main failure mode [read].

**4.3 Named-field JSON is not the strongest JSON baseline (major).** The report
says so (`cpu-diffusion-oracle-2026-10-20.md:27-28`;
`diffusion-sylang-2026-10-20.md:361-363`) but did not run the array control it
recommends. The array of M's codes needs one round when the length is given,
like M. On the fixed canvas it needs two, only because Qwen3 merges `","+` but
not `","-`, which makes its length depend on polarity [agent].

**4.4 Training exposure is absent, which favours M (moderate).** The oracle
knows each format's exact prior, including M's position-dependent codes (`s`
is *see* as a relation and *simple* as an aspect, `sylang_core/codec.py:38-44`).
A pretrained model knows English and JSON well and M not at all.

**4.5 Legends are excluded from payload experiments (minor; disclosed).** The
legends are 242 Qwen3 tokens for English, 139 for M and 22 for JSON. With the
repository's prompt builder (`evaluation/harness.py:100-123`), a single-record
prompt in the 864-meaning world is shortest in JSON: 266.8 tokens against M
354.0 and English 461.1. M needs about four records per shared legend to beat
JSON [agent].

**4.6 What holds up.** All three formats encode the same AST and differ only
in constant content. Stopping thresholds are identical for every format
(`evaluation/diffusion_experiment.py:174,191-193`). Masks that land on constant
tokens do not bias recovery. English's erasure advantage is genuine; section 6
explains where it comes from. Qwen3.5 tokenizes all 2,592 surfaces with
identical segmentation to Qwen3, so it is consistent with Qwen3 but not an
independent replication [agent].

## 5. Question 3: does the oracle's parallel-decoding advantage transfer?

**Why M needs one round.** In the Qwen3 M stream, 10 of 17 positions are
constant. Each of the other 7 corresponds one-to-one with a semantic field. The
prior is an exact Cartesian product, so every field-aligned position passes the
exact integer test `|J∩T|·n == |J|·|T|` (`evaluation/diffusion_oracle.py:313-314`)
and all of them join round 1 [reproduced]. English has 1 of 14 varying
positions aligned with a single field and JSON 6 of 14. JSON's misalignment
comes from Qwen3 splitting `progressive`, `unspecified` and `inferred` [agent].
Lexical and UTF-8-byte M also take one round, and Qwen3 M reproduces the
hypothetical one-slot-per-field native arm exactly, in both worlds [agent].

**The unrestricted joint oracle.** It needs one round for every format. That is
hard-coded at `evaluation/diffusion_experiment.py:208` and true by definition,
and the report says so (`cpu-diffusion-oracle-2026-10-20.md:123-124,155-156`).
The consequence is that round counts do not measure a representation's
intrinsic parallelism. They measure how far each format's token-level joint
distribution is from a product of per-position marginals, for one sampler class.

**The magnitude is fragile.** Fixed public canvas, full erasure, adaptive
policy, forward/reverse mean rounds:

| Condition | English | JSON | M |
| --- | ---: | ---: | ---: |
| As reported: labels A/B [reproduced] | 4.69 / 6.00 | 3.00 / 3.50 | **1.00 / 1.00** |
| Same exact policy, per-round information-maximising order [agent] | 2.79 | 2.00 (optimal) | 1.00 |
| Equal-length two-token labels, Priya/Fatima [new] | 5.69 / 6.50 | 3.00 / 3.50 | 2.00 / 2.00 |
| Unequal-length labels, Priya/Xiomara [new] | 6.69 / 8.75 | 5.00 / 6.25 | 3.00 / 3.25 |
| Drop 1 of 864 meanings (total correlation 0.0017 bits) [new] | — | — | 2.80 / 2.38 |

After dropping one meaning, sampling every M position at once from its marginal
is still 99.89% valid. The adaptive count nearly triples only because 863 is
prime and the exact equality test can no longer pass [new, agent]. With 43
random meanings dropped, M typically needs more forward rounds than English
(median 6.01 against 5.82 over 30 draws) [agent]. Over 300 random scan orders
English ranges from 2.93 to 5.72 rounds; the reported 6.00 reverse figure is
above all of them [agent]. M stays lowest or tied in every label world, and its
absolute forward lead under the report's policy is unchanged at 3.69 rounds;
only the ratio shrinks [agent].

**Samplers that use marginals.** Published dLLM samplers do not test
independence. LLaDA unmasks a fixed number of tokens per step and reports its
best quality when steps equal length. Fast-dLLM commits every token above a
confidence threshold (default 0.9) plus at least one. EB-Sampler bounds summed
entropy [agent, external]. Fed exact marginals on the same fixed canvases:

| Idealised sampler, fixed canvas | English | JSON | M |
| --- | --- | --- | --- |
| Threshold τ = 0.9, independent world [new] | 8.69 rounds, 100% valid | 8.83, 100% | 7.00, 100% |
| Threshold τ = 0.9, correlated world [new] | 6.50, 100% | 5.83, 100% | 4.67, 100% |
| Threshold τ = 0.7, independent world [agent] | 6.39, 76.4% valid | 5.40, 50.9% | 7.00, 100% |
| Threshold τ = 0.5, independent world [agent] | 1.07, 1.9% valid | 1.35, 10.2% | 5.00, 100% |
| One step, every position from its marginal (LLaDA with steps = 1), independent [reproduced] | 0.002% valid | 0.72% | 100% |
| One step, correlated world [reproduced] | ≈ 0% | 0.015% | 2.1% |
| Fixed four tokens per step, correlated world: failure mass [agent] | 82.3% | 69.6% | **90.7%** |

Two cautions apply to the τ = 0.9 row. Under this uniform prior every uncertain
marginal is at most 0.5, so any τ above 0.5 commits exactly one field per
round. M's 7.00 is simply its field count, and the English and JSON excess is
the cost of materialising multi-token fragments. These counts are as
definitional as the adaptive ones. Rule details also matter: a stricter
Fast-dLLM variant gives 10.81 / 11.67 / 8.00 [agent].

With realistic unequal names (Alice/Bartholomew) at τ = 0.9, English is 97.3%
valid and JSON 77.5%, while M stays 100% valid at 7.25 rounds [agent].

**Conditional generation shrinks the gap.** All refinement numbers model
sampling a random meaning from 9.75 bits of entropy. When some fields are given
by the context, the gap shrinks with the remaining uncertainty. With four of
seven fields given, adaptive rounds are English 2.83, JSON 1.86, M 1.00; with
six given, 1.66, 1.29 and 1.00. With all seven given, every format needs one
round by definition [agent].

**Verdict on question 3: both.** The reported magnitude, 1 against 4.69, comes
largely from how fields, tokens, the prior and the independence test were
defined. A narrower effect is real. When fields are independent, one token per
field with fixed widths lets a marginal-only sampler commit many fields at once
without incoherence, and only M keeps that property under aggressive
thresholds. When fields are correlated, the same layout makes parallel commits
fail more often than in English or JSON. That is the report's own fusion
insight (`cpu-diffusion-oracle-2026-10-20.md:210-214`) applied to surface
tokens. The property belongs to field alignment, which any positional code can
have. Whether a trained denoiser learns accurate marginals for dense one-token
codes is untested.

## 6. Question 4: compactness against error recovery

**Reported numbers verified** [reproduced]. At 50% Qwen3 token masking with
the target length given, independent world:

| Format | Mean tokens | Residual entropy | Bayes recovery ceiling | Mask SE |
| --- | ---: | ---: | ---: | ---: |
| English | 21.11 | 2.930 bits | 23.94% | 0.21 pp |
| JSON | 46.83 | 3.785 bits | 12.66% | 0.13 pp |
| M | 17.00 | 4.888 bits | 7.81% | 0.11 pp |

Exact enumeration gives 2.937 / 3.797 / 4.877 bits and 23.71% / 12.68% /
exactly 5/64 = 7.8125%. The Monte Carlo values lie within about 1 SE [agent].
M's value is exactly semantic-field erasure. With one varying token per field,
E[1/m] = ∏ᵢ((1−p) + p/kᵢ) = (3/4)³(2/3)³(5/8) = 5/64.

**The "19.5% fewer tokens" is not evidence of compactness against English.**
See section 4.1. M's absolute saving is a near-constant 4.1 tokens per record
(2–9 per smoke fixture). The percentage is larger in the 864-meaning world only
because its one-character labels make records short; on the 306-token
`literal-max-length` fixture M saves 3 tokens, about 1% [agent].

**Where English's advantage comes from.** Of English's 1.94-bit advantage at
p = 0.5, 1.77 bits come from lower marginal entropy in polarity, tense, aspect
and evidence, which English spreads over several tokens (`will not have
contained`, `By inference,`). The remaining 0.17 bits are residual dependence
among fields. Subject, relation and object are equally exposed in every format.
Revealing the true length is worth at most 0.065 bits at p = 0.5, though it
matters at high masking [agent]. For English to be left with as much residual
entropy as M has at p = 0.5, about 71–75% of its tokens must be masked [agent].
At an equal number of *visible* tokens, English recovers more than M up to 13
visible tokens with the length given (12 on the fixed canvas), and M recovers
more beyond that [agent].

**Token count does not determine robustness.** Qwen3 JSON is the longest
format, but about 37 of its 47 tokens are constant. Lexical-splitter JSON (83
positions) has exactly M's 5/64 ceiling. With UTF-8 bytes, JSON (178 positions)
recovers about 54% and beats English [agent]. Robustness is set by how many
independently erased units carry each field.

**Substitutions are not covered by the reports (moderate).** The experiments
use only erasures, where corrupted positions are flagged. A sampler's errors
are wrong commitments. Under an idealised model, one token is replaced with
another token seen at that position [new]:

| Format | Substitutions | Silent (another valid meaning) | Detectable (invalid) |
| --- | ---: | ---: | ---: |
| English | 25,128 | 19.3% | **80.7%** |
| JSON | 11,664 | 60.5% | 39.5% |
| M | 10,368 | 100.0% | **0.0%** |

Restricting English to substitutions within the same syntactic role still
leaves 78.6% detectable [agent]. This is the mirror image of M's 100% one-step
validity. When every field combination is grammatical, independent samples
are always valid, and a wrong polarity, tense, aspect or evidence code is never
caught by a validator. Detection does not raise the Bayes ceiling, but it is
what a remasking or repair sampler relies on. That bears on the report's
"validate, constrain, and selectively repair" hypothesis
(`diffusion-sylang-2026-10-20.md:93-94`) for value fields. Structural roles and
scope can still be checked.

**Limitations.** Uniform synthetic priors; Bayes ceilings, not model accuracy;
independent Bernoulli erasures rather than real error processes; two one-token
labels; one fixed predicate shape; target length given in the headline table.
The correlated-world token-masking results were computed but not reported, and
they favour English more strongly: 93.6% against M's 73.2% at 50% masking
[agent].

## 7. Question 5: data, statistics, leakage and reproducibility

- **Reproducibility is excellent** (section 2). One gap: the hashed source list
  (`evaluation/diffusion_experiment.py:300-302`) leaves out
  `evaluation/harness.py`, which loads and pins the tokenizers.
- **The 864-meaning world is a designed population, not a sample.** The 19.5%
  has no sampling variance; it is sensitive to design choices (tag, labels,
  codes). JSON length is exactly additive in the seven field values [agent].
  The correlated world has 18 meanings from three latent variables, and its
  16-mask estimates carry 1–2 pp standard errors [agent]. Neither world says
  anything about generalisation, because nothing is learned.
- **The 11.66% smoke-suite figure has no uncertainty statement.** A bootstrap
  over the 33 authored fixtures gives 7.3–16.6%, and resampling by fixture
  family gives 7.9–18.9% [agent]. Because the fixtures are authored, this is a
  sensitivity range, not population inference.
- **The quadratic proxy (1.084×) and its 8.4% threshold are dominated by one
  fixture.** `literal-max-length` supplies 78% of the summed squared English
  lengths; without it the proxy is 1.40× [agent]. The report names the fixture
  (`diffusion-sylang-2026-10-20.md:146-150`) but still contrasts the number with
  the uniform scenario as if it were a property of the suite.
- **Leakage.** No oracle path uses hidden gold beyond the declared length
  condition [agent]. The fixed canvas reveals almost all length information at
  intermediate mask rates, which the report records, but this does not drive
  the 50% table [agent]. The report switches length conditions between adjacent
  results: one-round support uses the given length, rounds use the fixed
  canvas. With the length given in the correlated world, English beats M on
  reverse rounds (2.72 against 3.67) and on one-round support (8.8% against
  2.1%) [agent].
- **Tests are sound where they exist but leave the headline numbers
  unprotected.** Mutation testing killed 8 of 19 single-point mutants. The
  survivors include standard-error changes, removing fixed-canvas padding in
  `run_experiments`, and inverting the trade-off tool's block speed-up. No test
  calls `run_experiments()` or `analyze()`, so the 19.5%, the 50% table, the
  round table and the 11.66% are protected only by reproduction, which succeeds
  today [agent]. The matched-semantic test cannot fail: replacing every surface
  with arbitrary unique streams gives the same result [agent].

## 8. Question 6: the end-to-end workflow

The design note plans input-only, output-only and both-side conditions
(`diffusion-sylang-2026-10-20.md:328-330`). Only the input-comprehension
condition exists in code: every format must be reconstructed as the same JSON
(`evaluation/harness.py:120-121`). In that condition the JSON arm is a copy
task, so any future comprehension result on it is confounded in JSON's favour
[agent].

| Stage | Measured? | Notes |
| --- | --- | --- |
| Natural language → semantic tree | No | No translator exists (`README.md:9`; `diffusion-sylang-2026-10-20.md:47-49`). The cost applies to every structured format equally, so it matters only against sending raw natural language, which was never measured |
| Tree → text (codec) | Yes, by this review | 17–140 µs per record for encode plus decode (codec CPU time only). English is slowest because its decoder re-sorts candidate phrases (`sylang_core/codec.py:299-308`) [agent] |
| Legend and instructions | Counted, not amortised | Fixed overhead per request: English 440, JSON 220, M 337 Qwen3 tokens (legend plus a shared 198-token envelope), repeated in all 33 prompts [agent] |
| Chat template, system prompt | No | `evaluation/harness.py:277` |
| Model input and output tokens | Input counted; output not | In the existing protocol the minimal output is the same JSON for every format |
| Denoising passes, latency | No | Oracle rounds are idealised sampler counts |
| Decoding and validation | Code exists | Fail-closed codec and validator (`sylang_core/codec.py:396-430`) |
| Error handling and retries | No | No repair or retry code; failure rates unmeasured |
| Rendering back for people | Template only | Controlled English with tag and quotes; fluent prose would need a model |

Under the existing protocol (one record per request, minimal JSON output),
M's payload saving over English is 0.76% of input-plus-output tokens on the
smoke suite and 0.81% in the 864-meaning world. If the shared envelope is
prefix-cached, it is 4.0% and 6.1% [agent]. The trade-off tool's complete-prompt
work proxies for M against English (1.29× linear, 1.66× quadratic) are 96%
legend length. By the same proxy JSON comes out best (1.65× / 2.66× against
English). The saved JSON records this, but the console summary prints only the
M-against-English line, and the reports never quote these proxies as speed-ups
[agent].

Two results point the other way. Emitting M instead of named-field JSON would
cut *output* tokens by 54% on the smoke suite and 64% in the 864-meaning world;
a JSON array of codes gets most of that saving (M/array 0.83–0.89) [agent]. And
if four or more records share one legend, M's complete prompt beats JSON's,
although the codec currently rejects multi-record documents [agent]. Neither
has been tested with a model.

## 9. Proposed next experiment

*To be completed.*

## 10. Methodological flaws ranked by severity

Severities are after adversarial verification. None of these invalidates the
reports' computations; they change what the computations can be taken to show.

| Severity | Flaw | Where | Effect |
| --- | --- | --- | --- |
| Major | Compactness against English rests on version-tag length and a verbose template English; against concise English, M is longer and the linear break-even scenario reverses (0.71×) | `cpu-diffusion-oracle-2026-10-20.md:71-73`; `diffusion-sylang-2026-10-20.md:58,135,146`; `sylang_core/codec.py:215,218`; `docs/decisions.md:42` | Removes the density premise against English |
| Major | The 1-round and 100% results require one-token labels; the headline sentence never says so | `evaluation/diffusion_experiment.py:31`; `cpu-diffusion-oracle-2026-10-20.md:9-10,121,147` | Headline parallel-sampling result does not survive realistic names |
| Major | Adaptive rounds depend on an exact product prior and an exact independence test, and the baselines get two pessimistic scan orders | `evaluation/diffusion_oracle.py:251-355`; `cpu-diffusion-oracle-2026-10-20.md:126-156` | 1 against 4.69 becomes 1 against 2.79 with a better order, and 2.80 for M when one meaning is dropped |
| Major | Baselines are named-field JSON and a verbose English template; the array control the report recommends was not run | `sylang_core/codec.py:46-51,209-211`; `diffusion-sylang-2026-10-20.md:361-363` | Credits Sylang with gains any positional code also gives |
| Moderate | Robustness is analysed for erasures only; substitutions in M are always silent | `cpu-diffusion-oracle-2026-10-20.md:42-48` | Hides M's loss of error detection |
| Moderate | Headline credits compactness for effects that follow token-to-field alignment | `cpu-diffusion-oracle-2026-10-20.md:8-13` | Wrong variable named in the conclusion |
| Moderate | No learned-model evidence on M; prior reviews (rephug/sylang#2, rephug/sylang#3) and the branch's own decision gates are not engaged, including the separate approval required for diffusion runtimes (`docs/decisions.md:51-53`) | both reports | The case rests on oracle and tokenizer arithmetic alone |
| Moderate | The only implemented end-to-end task makes the JSON arm a copy; output-side conditions are planned but not built | `evaluation/harness.py:107-122`; `diffusion-sylang-2026-10-20.md:328-330` | Payload savings do not show up end to end; a future comprehension result would be confounded |
| Moderate | Headline numbers are not protected by tests; 11 of 19 mutants survive | `tests/test_diffusion_*.py` | Regressions would be caught only by reproduction |
| Minor | No uncertainty for 11.66%; quadratic proxy dominated by one fixture; Monte Carlo values quoted where exact ones are cheap; Qwen3.5 is not an independent replication; 50% table caption omits the length condition; correlated token-masking and gold-length round results not reported | various | Precision overstated |
| Minor | External citations: "output" added to Mercury 3's "1,068 tokens per second"; BLD's explicit compression-difficulty trade-off (grouped sequential generation) not mentioned; LLaDA `GUIDELINES.md` has no inference section; the Mercury 3 API resets out-of-range temperature to 1 and has no seed, which the pilot protocol does not guard against | `diffusion-sylang-2026-10-20.md:17,74-82,255-260,337-342` | Presentation; pilot design |
| Minor | Proposed programme has no budget, model sizes or GPU-hours, and is not sequenced against the existing pretrained-pilot gate on rephug/sylang#3 | `diffusion-sylang-2026-10-20.md:324-401` | Planning |

## 11. Strongest arguments for and against continuing

**For continuing**

- The computational work is sound, reproducible and honestly caveated, which
  makes it a good base for a real experiment.
- When fields are independent, a layout with one token per field and fixed
  widths is the only one that stays fully valid under aggressive marginal-based
  parallel decoding in this oracle. It keeps that lead with realistic names,
  where English and JSON start producing invalid output at τ = 0.9 [new,
  agent].
- M's output is 54–64% shorter than named-field JSON, the usual format for
  structured output, and output is where diffusion decoding spends its passes.
- Dream-v0-7B, an open masked-diffusion model initialised from Qwen2.5-7B,
  uses a tokenizer identical to the repository's pinned Qwen3 asset: 2,976 test
  strings tokenize identically [agent, external]. The token counts here are
  therefore exact Dream counts, and Dream and Qwen2.5-7B form a matched
  diffusion/AR pair with shared initialisation. Dream's context is 2,048
  tokens.
- In the project's only learned-model study, trained from scratch, a compact
  Sylang code reached 90% accuracy in fewer training items than English
  (54,000 against 60,000–62,000; a tie under the study's own pre-registered
  rule) and clearly beat compact JSON and CSV [read, agent].

**Against continuing as framed**

- Against English, M is not more compact once the version tag is removed or a
  concise English is used.
- The parallel-decoding headline depends on one-token labels, an exact product
  prior and an exact independence test. With correlated fields, M's one-token
  packing makes parallel commits fail more often than in English or JSON.
- M gives up redundancy and all detection of value substitutions.
- Payload savings are under 1% of a measured uncached request and about 4–6%
  with prefix caching; legends and output format dominate.
- Every advantage that survives is shared by a generic positional code, so
  none is yet specific to Sylang.
- At small scale, compute-matched studies find masked diffusion needs roughly
  14–16× the compute of AR to match validation loss, although diffusion gains
  when data repeat, as in a finite synthetic grammar [agent, external]. A fair
  diffusion experiment therefore costs more than the report implies, and the
  report gives no budget.
- No model in this repository has been trained or evaluated on M.

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
  --output .cache/diffusion-review-checks.json
```

`.cache/oracle-repro.json` should match `benchmarks/results/diffusion-oracle.json`
except for `environment.python`. `.cache/diffusion-review-checks.json` should be
identical to `benchmarks/results/diffusion-review-checks.json`.
