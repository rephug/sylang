# Independent review: Sylang diffusion research of 2026-10-20

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
review. Nothing here is a neural result. Oracle "rounds" are counts for
idealised samplers that know the exact prior; they are never inference speed,
denoising steps of a trained model, or latency.

**How the review was done.**

1. The reviewer read both reports and the code, reran the tests and the full
   experiment, and wrote a reproduction script for the new numbers.
2. Eight review agents each took one question: what was measured, fairness,
   oracle transfer, compactness against recovery, statistics and code,
   end-to-end cost, external sources, and prior project evidence.
3. For each agent, a second agent re-derived the findings with its own code
   and tried to refute them. None of the 103 findings was refuted; 60 were
   corrected, mostly in precision or severity. The corrected versions are used
   here.
4. Two further agents checked the finished draft: one for facts and
   citations, one for fairness to the original reports. Their corrections are
   applied.

**Evidence labels**

- **[reproduced]**: the reviewer reran the repository's own code or recomputed
  the value from the raw result files.
- **[new]**: a new computation using the repository's codec, pinned Qwen3
  tokenizer and unmodified `FiniteOracle`. These come from
  [`tools/review_diffusion_claims.py`](../../tools/review_diffusion_claims.py),
  which is deterministic and runs in about 6 s on a 4-core CPU. Its output is
  saved in
  [`benchmarks/results/diffusion-review-checks.json`](../../benchmarks/results/diffusion-review-checks.json).
- **[agent]**: computed by a review agent. Most such values were independently
  re-derived by a verifying agent. A few were introduced by a verifier and
  computed only once: the 30-draw drop-43 medians, the Alice/Bartholomew
  threshold validity, and the visible-token crossover. Their scripts are not
  committed.
- **[read]**: taken from a document or external source without recomputation.

## 1. Executive verdict

**Recommendation: revise the hypothesis.** Do not proceed on the reports'
current framing. In that framing, M's density (19.5% and 11.66% fewer tokens
than controlled English) and its oracle round counts are read as support for
"Sylang could reduce the work of diffusion language modeling"
(`diffusion-sylang-2026-10-20.md:8-9`). Do not stop the line entirely either:
a narrower, testable hypothesis survives.

The work is careful and honest. Every number in both reports reproduces within
its stated rounding, and the code has no demonstrated bugs. The reports
already disclaim latency, neural and Mercury claims, say that density alone
does not imply fewer passes (`diffusion-sylang-2026-10-20.md:9-11`), and say
that JSON wins the complete-prompt comparison. The problems are in what the
evidence is taken to show.

1. **M is not more compact than a concise English.** The reported savings are
   a near-constant 4.1 tokens per record, and the per-record version tags
   account for 97–122% of that. English's `Sylang core-v0.1: ` is 9 Qwen3
   tokens; M's `M0.1:` is 4. [new, agent]
   - **Tags removed:** M is 7.3% longer than the codec's English (13.00 vs
     12.11 tokens per record) and 3.3% longer on the 33 smoke fixtures.
   - **Constant leading and trailing tokens removed instead:** M is 10% shorter
     (10.00 vs 11.11). So against the codec's templated English, the sign
     depends on the convention.
   - **Against a concise English:** M is longer under every convention tested.
     On the smoke suite it is 23% longer with both tags removed (898 vs 730
     tokens) and 41% longer if M keeps its tag. The report's same-pass linear
     work scenario then gives 0.81× (tags removed) or 0.71× (M tagged), not
     1.13×.
2. **The parallel-decoding headline is mostly definitional, but a residual
   survives.** M's 1 adaptive round against English's 4.69 rests on two
   single-token labels, an exactly factorized prior and an exact
   joint-independence test that no real sampler has.
   - **Multi-token names:** M needs 2 or 3 rounds. English and JSON also
     degrade, so M's absolute lead (3.69 rounds) is unchanged.
   - **One of 864 meanings dropped:** M needs 2.80 adaptive rounds, although
     sampling all of M at once is still 99.89% valid.
   - **What survives**, in the exact-marginal oracle (not a model result):
     with independent fields, M is the only format that stays fully valid
     under aggressive confidence-threshold decoding. At τ = 0.5 it is 100%
     valid, against English 1.9%. With correlated fields the picture depends on
     the sampler. M is still the most valid format under one-step sampling and
     under thresholds of 0.5 and 0.7, but it fails most often under fixed steps
     of 2, 3, 4 or 6 tokens.
   - This property comes from field alignment, which any positional code can
     have. [new, agent]
3. **M gives up all redundancy.** The erasure half of this is the oracle
   report's own headline: M's Bayes recovery ceiling equals pure field erasure
   (7.81% against English 23.94% at 50% masking). The new half concerns
   substitutions. Under an idealised single-token substitution model, none of
   M's in-distribution substitutions can be detected; 100% give another valid
   meaning, whereas English flags 80.7%. Detection does not raise the Bayes
   ceiling, but remasking and repair samplers depend on it. [reproduced, new]
4. **End-to-end savings come from legends, not payloads, and are untested.**
   - The repository's comprehension harness asks every format to reconstruct
     the same JSON, which makes the JSON arm a copy task.
   - Under that protocol, with minimal JSON output assumed, M's complete
     request is 19.7% smaller than English's and 22.5% larger than JSON's.
     Both gaps come from legend length.
   - The payload part of M's saving over English is 0.76–0.81% of
     input-plus-output tokens per uncached single-record request, or 4.0–6.1%
     if the shared instructions are prefix-cached.
   - The reports claim no request-level saving. [agent]
5. **No neural model has been trained or evaluated on M anywhere in this
   repository or its branches.** The reports disclose that no learned model
   exists. The only learned evidence in the project is a 1.12M-parameter byte
   model trained on a different Sylang code. That code tied English under the
   study's own pre-registered rule and beat compact JSON. [agent, read]

**What to test instead.** A field-aligned layout, in which each semantic field
occupies its own token positions, may let a masked-diffusion decoder commit
more positions per step at equal accuracy. It may also shorten structured
output compared with named-field JSON. Sylang M is one such layout, but it is
fixed-width only when its labels and codes are single tokens. Both claims need
a generic aligned-code control before any credit goes to Sylang. The reports
already call for this design: the AR-versus-masked-diffusion × English/JSON/M
factorial with tuple/array and generic AST-ID controls
(`diffusion-sylang-2026-10-20.md:358-367`), and a trained-model test of token
alignment (`cpu-diffusion-oracle-2026-10-20.md:256-261`). The revision concerns
the claims and the baselines (concise English, version tags, the interaction
ratio's null value), not the direction of the next step.

**Concrete next action:** see section 12.

## 2. What was reproduced and what was only read

| Item | Status |
| --- | --- |
| 92 tests (`python -m unittest discover -s tests`) with the pinned tokenizers | **[reproduced]** 92/92 pass. With system Python and no `tokenizers` package: 90 pass, 2 skip |
| 37 oracle/fusion/experiment tests under `python3 -S` | **[reproduced]** pass |
| Full oracle run (`lexical,utf8_bytes,qwen3,qwen3.5`, 16 masks per meaning) | **[reproduced]** in 47 s. Identical to `diffusion-oracle.json` except `environment.python` (3.12.3 saved, 3.13.16 here) |
| Standard-library-only oracle run | **[agent]** reproduces the lexical, byte, native and fusion sections in 23 s |
| SHA-256 `fa73a6d0…7460` of the saved oracle JSON; six recorded source hashes | **[reproduced]** match |
| `tools/analyze_diffusion_tradeoffs.py` | **[reproduced]** byte-identical output |
| Pinned Qwen3/Qwen3.5 tokenizer assets | **[reproduced]** downloaded; size and SHA-256 match `benchmarks/tokenizers.lock.json` |
| Every numeric claim in both reports (56 rows) | **[agent]** all match the raw data within the stated rounding. One statement (`cpu-diffusion-oracle-2026-10-20.md:154-155`) holds only on the fixed canvas |
| Vendor and preprint sources | **[agent]** all exist with the stated dates; characterisations in sections 5 and 10 |
| Neural accuracy, denoising steps, latency | Not measured anywhere in the repository, so there is nothing to reproduce |

## 3. Question 1: what was actually measured

| Quantity | What it is | Evidence type |
| --- | --- | --- |
| Token counts (Qwen3, Qwen3.5, lexical splitter, UTF-8 bytes) | Deterministic tokenizer counts of codec output | Measured |
| Residual entropy and Bayes recovery ceiling under masking | Exact posterior over a known finite prior. Interior mask rates are averaged by Monte Carlo over 16 masks per meaning; the standard errors are correct and well calibrated against exact enumeration [agent] | Exact computation plus mask sampling |
| Matched semantic erasure (7.8125%, identical for all formats) | Computed once on native slots and copied into the results (`evaluation/diffusion_experiment.py:293-297`) | True by construction for bijective codecs, not a measurement |
| "Rounds" (adaptive, static, one-round support) | Exact expectations for **named idealised samplers** that know the prior. The adaptive policy can also test exact joint independence (`evaluation/diffusion_oracle.py:251-355`) | Exact computation, specific to the sampler |
| Speed-up, break-even and block tables | Arithmetic on **input payload** token counts (`diffusion-sylang-2026-10-20.md:133-152`) | Hypothetical scenarios |
| Neural inference steps | None anywhere | Not measured |
| Model latency or elapsed time | None. The only timing data are process-CPU samples of tokenizer and codec calls from an earlier Windows run, quantised to 0 or 15.6 ms and already disclaimed (`docs/baseline-results.md:107-111`). Neither report uses them | Not measured |
| Mercury 3 throughput, BLD and JPEG-DLM speed-ups | Vendor and preprint claims about other systems | External |

The reports label these classes correctly in most places
(`cpu-diffusion-oracle-2026-10-20.md:3-6`; `diffusion-sylang-2026-10-20.md:4-6,111-114,139`).
The smaller overstatements are listed in section 10.

The most visible one is the lead sentence "compactness trades erasure
robustness against parallel consistency" (`cpu-diffusion-oracle-2026-10-20.md:8`).
For the English/JSON/M surfaces, the measured effects follow token-to-field
alignment, not length. Lexical-splitter JSON, 83 positions long, behaves
exactly like M on both erasure and rounds, because it also has one varying unit
per field. The report states this itself (`:158-162`), and its next sentence
names "M's token boundaries" (`:9-10`); the lead sentence is still broader
than the evidence. Where alignment *is* held fixed, as in the native-slot
fusion arms, compactness matters in two ways:
- it helps parallel consistency only when a fused unit encloses a dependency;
- it also changes whole-record erasure recovery (7.81% → 19.01% → 28.26% at
  50% slot erasure in the independent world) without changing residual
  entropy [agent].

## 4. Question 2: is the English/JSON/M comparison fair?

Ranked by how much each issue moves the headline numbers.

### 4.1 Version tags and template English (major)

The codec writes `Sylang core-v0.1: ` before English and `M0.1:` before M
(`sylang_core/codec.py:215,218`). Each core-v0.1 document holds exactly one
statement (`docs/experimental-grammar.md:32`), so the tag is per record by
design. Its length is a naming choice that carries no information.

What the project already says:
- `docs/baseline-results.md:16` discloses that headers are counted.
- `docs/baseline-results.md:10` and the report tables call the baseline
  "Controlled English", a deliberate template.
- `diffusion-sylang-2026-10-20.md:293` proposes measuring headers separately.

What is still missing: neither report quantifies the effect, and
`cpu-diffusion-oracle-2026-10-20.md:71` drops the "controlled" qualifier.
The earlier independent review on PR rephug/sylang#2 had concluded that the
11.6% saving is "a property of a weak baseline, not of M" [read]. That PR is
not an ancestor of the reviewed commit, which partly explains why the reports
do not cite it. The branch's own decision gate, however, asks to "compare M
against the strongest compact baseline, not only English"
(`docs/decisions.md:42-44`).

| Mean Qwen3 tokens per record, 864-meaning world | Mean | Range | Source |
| --- | ---: | --- | --- |
| Codec English (as reported) | 21.11 | 19–23 | [new] |
| Codec English, version tag removed | 12.11 | 10–14 | [new] |
| Concise English (rephug/sylang#2 renderer: quoted labels, codec evidence phrases, default evidence unmarked) | 10.86 | — | [agent] |
| Concise English, quoted labels, *Directly/Reportedly/Presumably* (lossless for any label) | 10.61 | 6–13 | [agent] |
| Concise English, unquoted labels (injective on this world only) | 7.86 | 4–10 | [new] |
| Named-field JSON (as reported) | 46.83 | 46–48 | [new] |
| JSON array of words, e.g. `["A","see","A","positive","past","simple","unspecified"]` | 15.83 | 15–17 | [new] |
| JSON array of M's codes, e.g. `["A","s","A","+","p","s","u"]` | 14.50 | 14–15 | [new] |
| M (as reported) | 17.00 | 17 | [new] |
| M, version tag removed | 13.00 | 13 | [new] |

The figures reproduce: 1 − 17.00/21.11 = 19.47% [reproduced].

On the 33 smoke fixtures [new, agent]:
- with tags, English 1,166 and M 1,030;
- with tags removed, 869 and 898;
- concise English (the rephug/sylang#2 renderer), 730.

Against the tag-free codec English, the sign of the difference depends on the
boilerplate convention, as section 1 shows. Against concise English, M is
longer under every convention tested:
- +1.4% to +56.5% against the rephug/sylang#2 renderer [agent];
- larger margins against the *Directly/Reportedly/Presumably* variants above.

M is clearly shorter than named-field JSON. With no version marker on either
side, it is 10–18% shorter than the JSON arrays.

### 4.2 Single-token labels (major)

The world uses only `LABELS = ("A", "B")` (`evaluation/diffusion_experiment.py:31`),
and each is one Qwen3 token. That is what gives M a constant length, exactly
one varying token per field, and its 1-round result. The labels are stated
(`cpu-diffusion-oracle-2026-10-20.md:24`), and "two ASCII labels" is listed as
a limit (`evaluation/diffusion_experiment.py:323`). The headline never says
the result depends on them.

In the project's own smoke fixtures, 38 of the 80 label occurrences are
multi-token in M, including every label in both holdout fixtures [agent].
Section 5 shows the effect. The project's earlier learned-model work found
identifier shape and record-count extrapolation to be the main failure modes
[read].

### 4.3 Named-field JSON is not the strongest JSON baseline (major)

The reports say so and call for tuple/array and generic AST-ID controls
before any Sylang-specific attribution
(`cpu-diffusion-oracle-2026-10-20.md:27-28`; `diffusion-sylang-2026-10-20.md:361-365`).
Those controls were not run. An array of M's codes needs one round when the
length is given, like M. On the fixed canvas it needs two, only because Qwen3
merges `","+` but not `","-`, so the array's length depends on polarity
[agent].

### 4.4 Training exposure is absent, which favours M (moderate; partly disclosed)

The oracle knows each format's exact prior, including M's position-dependent
codes: `s` means *see* as a relation and *simple* as an aspect
(`sylang_core/codec.py:38-44`). A pretrained model knows English and JSON well
and M not at all. The design note calls the oracle "an analytical upper bound,
not a learned-model result" that cannot estimate learning difficulty
(`diffusion-sylang-2026-10-20.md:315-319`). It proposes equal-exposure
training and cipher controls (`:366-373`). It does not say that the oracle
thereby removes M's unfamiliarity handicap.

### 4.5 Legends are excluded from payload experiments (minor; disclosed)

The legends are 242 Qwen3 tokens for English, 139 for M and 22 for JSON. With
the repository's prompt builder (`evaluation/harness.py:100-123`), a
single-record prompt in the 864-meaning world is shortest in JSON: 266.8
tokens, against M 354.0 and English 461.1. M needs about four records per
shared legend to beat JSON [agent].

### 4.6 What holds up

- All three formats encode the same AST and differ only in constant content.
- Stopping thresholds are identical for every format
  (`evaluation/diffusion_experiment.py:174,191-193`).
- Masks that land on constant tokens do not bias recovery.
- English's erasure advantage is genuine; section 6 explains where it comes
  from.
- Qwen3.5 segments all 2,592 surfaces exactly as Qwen3 does. The report does
  not claim an independent replication and correctly attributes its small
  differences to separate mask seeds (`cpu-diffusion-oracle-2026-10-20.md:77-79`)
  [agent].

## 5. Question 3: does the oracle's parallel-decoding advantage transfer?

**Why M needs one round.** In the Qwen3 M stream, 10 of 17 positions are
constant. Each of the other 7 corresponds one-to-one with a semantic field.
The prior is an exact Cartesian product, so every field-aligned position
passes the exact integer test `|J∩T|·n == |J|·|T|`
(`evaluation/diffusion_oracle.py:313-314`), and all of them join round 1
[reproduced].

By contrast, English has 1 of 14 varying positions aligned with a single
field, and JSON 6 of 14. JSON's misalignment comes from Qwen3 splitting
`progressive`, `unspecified` and `inferred` [agent]. Lexical and UTF-8-byte M
also take one round, and Qwen3 M exactly reproduces the hypothetical
one-slot-per-field native arm in both worlds [agent].

**The unrestricted joint oracle.** It needs one round for every format. That
is hard-coded at `evaluation/diffusion_experiment.py:208`, true by
definition, and stated by the report (`cpu-diffusion-oracle-2026-10-20.md:123-124,155-156`).
The consequence is that round counts do not measure a representation's
intrinsic parallelism. They measure how far each format's token-level joint
distribution is from a product of per-position marginals, for one sampler
class.

**The magnitude is fragile.** Fixed public canvas, full erasure, adaptive
policy, forward/reverse mean rounds:

| Condition | English | JSON | M |
| --- | ---: | ---: | ---: |
| As reported: labels A/B [reproduced] | 4.69 / 6.00 | 3.00 / 3.50 | **1.00 / 1.00** |
| Same exact policy, per-round information-maximising order [agent] | 2.79 | 2.00 (optimal) | 1.00 |
| Equal-length two-token labels, Priya/Fatima [new] | 5.69 / 6.50 | 3.00 / 3.50 | 2.00 / 2.00 |
| Unequal-length labels, Priya/Xiomara (2 and 3 tokens) [new] | 6.69 / 8.75 | 5.00 / 6.25 | 3.00 / 3.25 |
| Drop 1 of 864 meanings (total correlation 0.0017 bits) [new] | — | — | 2.80 / 2.38 |

M stays lowest or tied in every label world. Its absolute forward lead under
the report's policy is unchanged at 3.69 rounds; only the ratio shrinks
[agent]. What does not survive multi-token labels is the exact 1-round and
100% one-step result.

After one meaning is dropped, sampling every M position at once is still
99.89% valid. The adaptive count nearly triples only because 863 is prime and
the exact equality test can no longer pass [new, agent]. With 43 random
meanings dropped, M typically needs more forward rounds than English: median
6.01 against 5.82 over 30 draws [agent].

Scan order matters for the baselines. In two independent samples of 300
random scan orders, English ranged from about 2.9 to 5.7–6.6 rounds, with a
median near 4.05 [agent]. The reported forward 4.69 is above about 85% of
random orders, and the reverse 6.00 is above the 95th percentile (5.00–5.08).
JSON's forward order is at the median of random orders; only its reverse order
is pessimistic. M is 1.00 in every order.

**Samplers that use marginals.** Published dLLM samplers do not test
independence [agent, external]:
- LLaDA unmasks a fixed number of tokens per step and reports its best quality
  when steps equal length.
- Fast-dLLM commits every token above a confidence threshold (default 0.9),
  plus at least one.
- EB-Sampler bounds the summed entropy of the tokens it commits.

Fed exact marginals on the same fixed canvases, idealised versions give the
following. Rounds are means over valid outputs.

| Idealised sampler, fixed canvas | English | JSON | M |
| --- | --- | --- | --- |
| Threshold τ = 0.9, independent world [new] | 8.69 rounds, 100% valid | 8.83, 100% | 7.00, 100% |
| Threshold τ = 0.7, independent world [new] | 7.44, 76.4% valid | 8.68, 50.9% | 7.00, 100% |
| Threshold τ = 0.5, independent world [new] | 3.87, 1.9% valid | 4.00, 10.2% | 5.00, 100% |
| Threshold τ = 0.9, correlated world [new] | 6.50, 100% | 5.83, 100% | 4.67, 100% |
| Threshold τ = 0.7, correlated world [new] | 5.70, 51.8% | 5.68, 50.9% | 4.67, 100% |
| Threshold τ = 0.5, correlated world [new] | 2.17, 4.7% | 2.00, 4.5% | 3.50, 50% |
| One step, every position from its marginal (LLaDA with steps = 1), independent [reproduced] | 0.002% valid | 0.72% | 100% |
| One step, correlated world [reproduced] | ≈ 0% | 0.015% | 2.1% |
| Fixed four tokens per step, correlated world: failure mass [agent] | 82.3% | 69.6% | **90.7%** |

Notes on reading the table:
- **Why the τ = 0.9 counts are definitional.** Under this uniform prior every
  semantic field's marginal is at most 0.5. At τ = 0.9 the sampler therefore
  commits one uncertain position per round in every format. In M that is one
  field per round, so M's 7.00 is simply its field count, and the English and
  JSON excess is the cost of materialising multi-token fragments. These counts
  are as definitional as the adaptive ones.
- **Why lower thresholds break English and JSON.** Some English and JSON token
  positions have higher marginals: up to 0.94 and 0.83 at full erasure, from
  padding and shared fragments [new]. Lowering τ lets those formats commit
  several positions at once and produce invalid output.
- **Counting failed draws differently.** If a failed decode is counted at the
  round it fails, English and JSON appear to finish sooner (independent world,
  τ = 0.5: 1.07 and 1.35 rounds), but only by stopping on invalid output
  [agent].
- **Rule details matter.** A stricter Fast-dLLM variant gives 10.81 / 11.67 /
  8.00 at τ = 0.9 [agent]. In the correlated world, fixed steps of 2, 3, 4 or
  6 tokens make M fail more often than English or JSON; at 8 tokens per step M
  fails least [agent].
- **Realistic names.** With Alice/Bartholomew at τ = 0.9, English is 97.3%
  valid and JSON 77.5%, while M stays 100% valid at 7.25 rounds. With
  Priya/Xiomara all three formats stay valid [agent].

**Conditional generation shrinks the gap.** All refinement numbers model
sampling a random meaning from 9.75 bits of entropy. When the context gives
some fields, the gap shrinks with the remaining uncertainty [agent]:
- four of seven fields given: English 2.83, JSON 1.86, M 1.00 adaptive rounds;
- six given: 1.66, 1.29 and 1.00;
- all seven given: every format needs one round, by definition.

**Verdict on question 3: transfer is not established.** The oracle advantage
is partly definitional and partly real, and no trained model has tested it.

- **Definitional.** The reported 1-against-4.69 comes largely from how fields,
  tokens, the prior and the independence test were defined.
- **Real, in the oracle.** When fields are independent, one token per field
  with fixed widths lets a marginal-only sampler commit many fields at once
  without incoherence. In this oracle only M keeps that property under
  aggressive thresholds. With correlated fields, every format loses one-step
  validity. M is still the most valid format under thresholds of 0.5 and 0.7,
  but the least valid under fixed steps of 2, 3, 4 or 6 tokens. That is the report's own fusion
  insight (`cpu-diffusion-oracle-2026-10-20.md:210-214`) applied to surface
  tokens.
- **Not Sylang-specific.** The property belongs to field alignment, which any
  positional code can have.
- **Untested.** Whether a trained denoiser learns accurate marginals for dense
  one-token codes.

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
M's value is exactly semantic-field erasure: with one varying token per field,
E[1/m] = ∏ᵢ((1−p) + p/kᵢ) = (3/4)³(2/3)³(5/8) = 5/64.

**The "19.5% fewer tokens" reproduces but is not evidence of compactness
against English.** 1 − 17.00/21.11 = 19.47% [reproduced]; M is 17 tokens for
every meaning, and English ranges from 19 to 23. See section 4.1 for the
version tags. M's absolute saving is a near-constant 4.1 tokens per record
(2–9 per smoke fixture). The percentage is larger in the 864-meaning world only
because its one-character labels keep records short. On the 306-token
`literal-max-length` fixture, M saves 3 tokens, about 1% [agent].

**Where English's advantage comes from** [agent]:
- **Size and source.** English's 1.94-bit advantage at p = 0.5 is 1.77 bits of
  lower marginal entropy in polarity, tense, aspect and evidence, plus 0.17
  bits of residual dependence among fields. English spreads those four fields
  over several tokens (`will not have contained`, `By inference,`).
- **Fields with no advantage.** Subject, relation and object are equally
  exposed in every format.
- **Length.** Revealing the true length is worth at most 0.065 bits at
  p = 0.5, though it matters at high masking.
- **Matched information.** For English to keep only as much residual entropy
  as M does at p = 0.5, about 71–75% of its tokens must be masked.
- **Matched visible tokens.** English recovers more than M up to 13 visible
  tokens with the length given (12 on the fixed canvas), and M recovers more
  beyond that.

**Token count does not determine robustness.** Qwen3 JSON is the longest
format, but about 37 of its 47 tokens are constant. Lexical-splitter JSON (83
positions) has exactly M's 5/64 ceiling. With UTF-8 bytes, JSON (178
positions) recovers about 54% and beats English [agent]. Robustness is set by
how many independently erased units carry each field.

**Substitutions are not covered by the reports (moderate).** The experiments
use only erasures, where corrupted positions are flagged. A sampler's errors
are wrong commitments, which are substitutions. Under an idealised model,
each test replaces one token with another token seen at that position [new]:

| Format | Substitutions | Silent (another valid meaning) | Detectable (invalid) |
| --- | ---: | ---: | ---: |
| English | 25,128 | 19.3% | **80.7%** |
| JSON | 11,664 | 60.5% | 39.5% |
| M | 10,368 | 100.0% | **0.0%** |

Restricting English to substitutions within the same syntactic role still
leaves 78.6% detectable [agent].

This is the mirror image of M's 100% one-step validity. When every field
combination is grammatical, independent samples are always valid, and a wrong
polarity, tense, aspect or evidence code is never caught by a validator.
Detection does not raise the Bayes ceiling, but remasking and repair samplers
rely on it. That bears on the "validate, constrain, and selectively repair"
hypothesis (`diffusion-sylang-2026-10-20.md:93-94`) for value fields;
structural roles and scope can still be checked. The design note already warns
that parser success and high confidence do not establish meaning (`:233-234,242-243`).

**Limitations of this evidence.**
- Uniform synthetic priors.
- Bayes ceilings, not model accuracy.
- Independent Bernoulli erasures rather than real error processes.
- Two one-token labels and one fixed predicate shape.
- Target length given in the headline table.
- The correlated-world token-masking results were computed but not reported.
  They favour English more strongly: 93.6% against M's 73.2% at 50% masking
  [agent].

## 7. Question 5: data, statistics, leakage and reproducibility

- **Reproducibility is excellent** (section 2). One gap: the hashed source
  list (`evaluation/diffusion_experiment.py:300-302`) leaves out
  `evaluation/harness.py`, which loads and pins the tokenizers.
- **The 864-meaning world is a designed population, not a sample.**
  - The 19.5% has no sampling variance; it is sensitive to design choices
    (tag, labels, codes).
  - JSON length is exactly additive in the seven field values [agent].
  - The correlated world has 18 meanings driven by three latent variables,
    and its 16-mask estimates carry 1–2 pp standard errors [agent].
  - Neither world says anything about generalisation, because nothing is
    learned.
- **The 11.66% smoke-suite figure has no uncertainty statement (moderate).**
  A bootstrap over the 33 authored fixtures gives 7.3–16.6%; resampling by
  fixture family gives 7.9–18.9% [agent]. Because the fixtures are authored,
  this is a sensitivity range, not population inference.
- **One fixture dominates the quadratic proxy.** `literal-max-length` supplies
  78% of the summed squared English lengths, and without it the proxy is
  1.40× instead of 1.084× [agent]. The report names the fixture
  (`diffusion-sylang-2026-10-20.md:146-150`) but still contrasts the 8.4%
  threshold with the uniform scenario as if it were a property of the suite.
- **Leakage.**
  - No oracle path uses hidden gold beyond the declared length condition
    [agent].
  - The fixed canvas reveals almost all length information at intermediate
    mask rates. The report records this, and it does not drive the 50% table
    [agent].
  - The report switches length conditions between adjacent results: one-round
    support is given with the length known, rounds on the fixed canvas. With
    the length given in the correlated world, English beats M on reverse
    rounds (2.72 against 3.67) and one-round support (8.8% against 2.1%).
  - In the independent world the switch is conservative toward M. On the
    fixed canvas, one-round support falls to 0.0022% for English and 0.72% for
    JSON, while M stays at 100%. Because the given length leaks 1.975 bits to English, the fixed
    canvas is the fairer condition for rounds [agent].
- **Tests are sound where they exist but leave the headline numbers
  unprotected.**
  - Mutation testing killed 8 of 19 single-point mutants. The 11 survivors
    include standard-error changes, removing fixed-canvas padding in
    `run_experiments`, and inverting the trade-off tool's block speed-up.
  - No test calls `run_experiments()` or `analyze()`. The 19.5%, the 50%
    table, the round table and the 11.66% are therefore protected only by
    reproduction, which succeeds today [agent].
  - The matched-semantic test cannot fail: replacing every surface with
    arbitrary unique streams gives the same result [agent].

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
| Tree → text (codec) | Yes, by this review | 17–140 µs per record for encode plus decode, codec CPU time only. English is slowest because its decoder re-sorts candidate phrases (`sylang_core/codec.py:299-308`) [agent] |
| Legend and instructions | Counted, not amortised | Fixed overhead per request: English 440, JSON 220, M 337 Qwen3 tokens. That is the legend plus a shared 198-token envelope, repeated in all 33 prompts [agent] |
| Chat template, system prompt | No | `evaluation/harness.py:277` |
| Model input and output tokens | Input counted; output not | In the existing protocol the minimal output is the same JSON for every format |
| Denoising passes, latency | No | Oracle rounds are idealised sampler counts |
| Decoding and validation | Code exists | Fail-closed codec and validator (`sylang_core/codec.py:396-430`) |
| Error handling and retries | No | No repair or retry code; failure rates unmeasured |
| Rendering back for people | Template only | Controlled English with tag and quotes; fluent prose would need a model |

All figures below are modelled from token counts of the authored prompts, with
a minimal JSON output assumed [agent]:
- **Whole request.** M's request is 19.7% smaller than English's (14,396
  against 17,931 Qwen3 tokens on the smoke suite) and 22.5% larger than JSON's
  (11,750). Both gaps come from legend length.
- **Payload alone.** M's payload saving over English is 0.76% of
  input-plus-output tokens on the smoke suite and 0.81% in the 864-meaning
  world, per uncached single-record request. If the shared envelope is
  prefix-cached, it is 4.0% and 6.1%.
- **Work proxies.** The trade-off tool's complete-prompt proxies for M against
  English (1.29× linear, 1.66× quadratic) are 96% legend length. By the same
  proxy, JSON comes out best (1.65× and 2.66× against English). The saved JSON
  records this, but the console summary prints only the M-against-English
  line. The reports never quote these proxies as speed-ups.

Two results point the other way, and neither has been tested with a model:
- **Output length.** Emitting M instead of named-field JSON would cut *output*
  tokens by 54% on the smoke suite and 64% in the 864-meaning world. A JSON
  array of codes gets most of that saving: with a version element in the array
  and M's tag kept, M/array is 0.83–0.89 [agent]; with no version marker on
  either side, it is 0.90 [new].
- **Shared legends.** If four or more records share one legend, M's complete
  prompt beats JSON's, although the codec currently rejects multi-record
  documents [agent].

## 9. Proposed next experiment

The full pre-registerable design is in
[`docs/plans/diffusion-next-experiment-2026-10-22.md`](../plans/diffusion-next-experiment-2026-10-22.md).
It was produced as follows:
- Three review agents drafted independent designs: CPU-only, a single rented
  GPU, and pretrained Dream-7B against Qwen2.5-7B.
- A judge agent scored them, merged them and re-checked their arithmetic.
- Two critics then checked the merged design.

This section summarises it. Figures are planning estimates from those agents,
to be replaced by the plan's Stage 0 measurements.

**Smallest decisive experiment: a matched from-scratch screen, CPU first,
US$0.**
- **Most favourable setting for M.** From-scratch training gives English and
  JSON no pretraining advantage.
- **Full control.** It controls training exposure and compute end to end.
- **No GPU needed.** A rented GPU adds little for 4M-parameter models, whose
  batch-1 latency is dominated by launch overhead.
- **Paid follow-up.** The pretrained pair is kept as the only paid next step.

**Hypotheses**

- **H1-align (primary).** Some field-aligned output, decoded by a
  masked-diffusion model (MDM), meets all three conditions below against both
  concise English and named JSON at matched exact-AST accuracy. The candidates
  are M, a generic code line C, and C with fixed-width name slots F.
  - at most 0.80× the denoising passes;
  - at most 0.90× the within-family latency;
  - no more than 2 percentage points of accuracy lost.
- **H1-M (Sylang-specific).** M meets H1-align and is at least 10% faster
  than both C and F.
- **H0.** Format changes passes at matched accuracy by less than 10%, and
  latency differences follow output length.

**Arms: 8 cells.** All cells share the same 3.93M-parameter prefix-LM
transformer (d = 256, 4 layers). They also share exact Qwen3 token IDs
remapped to a compact vocabulary, data order and step count.

| Family | Formats |
| --- | --- |
| Autoregressive (AR) | concise English E, named JSON J, M |
| Masked diffusion (MDM) | E, J, M, C, F |

- C is a generic code line, for example `Bumche s Vondu - p c r`.
- F is C with every name slot padded to fixed width.
- No version tag appears anywhere the model reads or writes. Decoders re-add
  tags outside the model, and that time is counted.
- AR-C and AR-F are not trained, because AR passes equal output length.
- Tuple JSON is bounded by C and added only in the R1 follow-up.

**Task and data**

- **Task: PARSE.** The input is 1–4 English clauses; the output is the same
  records in the cell's format. Every arm gets the identical input.
- **Neutral input wording.** The input lexicon shares no words with any
  output: `notice`/`assist`/`hold`, `kept`, `never`, `Word has it that`, and
  so on. An audit requires at most 5% overlap of non-name tokens, so no arm
  can copy its input.
- **Names.** Pseudo-names of 1–5 Qwen3 tokens (one word in 70%, two in 30%),
  quoted in half the inputs, split by hash into train, dev and test.
- **Held-out compositions.** Three combinations never appear in training.
- **Test panels.**

  | Panel | Size |
  | --- | ---: |
  | iid (primary) | 2,000 |
  | unseen compositions | 1,000 |
  | longer records (K = 5–6) | 500 |

**Training exposure and compute**

- 128k or 256k items per run, chosen by a pre-registered rule.
- Checkpoints at equal exposure (items) and at equal compute (25–375 TFLOP).
- Paired seeds in three looks:
  - seed 0 is a futility check only;
  - seeds 1–2 give the first decision;
  - seeds 3–5 are added only for unresolved contrasts.
- A calibration gate requires at least 7 of 8 runs to learn. In earlier
  project work, only 2 of 5 seeds learned.

**Measurements**

- **Correctness.** Whole-output exact AST through `sylang_core` (primary),
  per-field errors, silent errors and validity.
- **Decoding work.** Denoising passes and positions processed.
- **Latency.** Actual wall-clock time per request on the owner's server,
  covering tokenize, prefill, decode, detokenize, re-adding the tag, parse and
  validate. Measured interleaved over two sessions; p50 and p95 reported.
- **Samplers.** Fixed tokens per step (LLaDA-style), confidence threshold
  (Fast-dLLM-style), entropy bound, and semi-autoregressive blocks of 16.
- **Resource cost.** Training FLOPs, core-hours, energy, and cost per 1,000
  correct answers.

**Analysis**

- **Matched accuracy A\*.** The higher of 95% or 90% that is reached by 2 of 3
  seeds in each of AR-E, AR-M, MDM-E and MDM-M.
- **Ratios.** Per-seed log ratios with two-sided 90% seed-paired t intervals.
  A hierarchical bootstrap is a sensitivity check.
- **Accuracy differences.** The rephug/sylang#3 cluster bootstrap and a
  seed-paired t interval; the more conservative of the two decides.
- **Interaction ratio.** The report's I = (T_MDM,E / T_MDM,M) / (T_AR,E / T_AR,M)
  exceeds 1 even with no diffusion-specific effect whenever M's output is
  longer. AR passes scale with output length; MDM passes at equal steps do
  not. On this task its null value is I₀ = 1.38 in passes for M against
  concise English, so only I/I₀ is interpreted.
- **Simulated decision rule.** With a between-seed SD of the log step ratio of
  0.2 or 0.3, P(success) is:

  | True k_M/k_E | P(success), SD 0.2 | SD 0.3 |
  | --- | ---: | ---: |
  | 1.00 (no effect) | 0.014 | 0.045 |
  | 0.80 (at the threshold) | 0.54 | 0.43 |
  | 0.67 (1.5× fewer passes) | 0.98 | 0.85 |

**Decision order.** Preconditions are checked first; then seed-0 futility,
PROCEED, STOP and REVISE. If any precondition fails, the result is
INCONCLUSIVE: fix the recipe, not the hypothesis. The preconditions are:
- round trips pass;
- no version strings appear;
- input overlap is at most 5%;
- sampler conformance with `FiniteOracle`;
- the Stage-0 gate passes (at least 7 of 8 runs learn);
- A\* is at least 90%;
- the latency coefficient of variation is below 5%.

**Success (PROCEED).** All of the following must hold.

| Criterion | Requirement |
| --- | --- |
| Passes | k_M/k_E ≤ 0.80 and k_M/k_J ≤ 0.80, each with the 90% upper bound below 1.00 |
| Passes, robustness | At most 0.90 under the non-selected samplers |
| Latency | Within-MDM T_M/T_E ≤ 0.90 and T_M/T_J ≤ 0.90, each with the upper bound below 1.00 |
| Latency tail | p95 ratio at most 1.00 |
| Latency at scale | Modelled 7B ratio at most 0.90 |
| Accuracy, iid | M − E lower bound at least −2 pp, in both families |
| Accuracy, unseen compositions | Lower bound at least −3 pp |
| Accuracy, longer records | Point estimate at least −5 pp |
| Silent errors | Excess point estimate at most +1 pp; upper bound at most +3 pp |

Meeting the pass criterion implies I/I₀ ≥ 1.25.

A result is labelled Sylang-specific only if M is also at least 10% faster
than C and F. Otherwise it is "field-aligned, generic". PROCEED licenses only
the owner-approved seed-0 stage of Dream-v0-Base-7B against Qwen2.5-7B, about
US$6–7 on one 48 GB GPU.

If C or F meets the success criteria against both E and J but M does not, the
outcome is STOP for M, and H1-align is recorded as supported for the generic
code. Any follow-up on C or F is a new, non-Sylang study that needs its own
owner decision.

**Stop.** Any one of these ends further diffusion work on M, and the negative
result is published:

| Condition | Rule |
| --- | --- |
| Seed-0 futility | MDM-E has learned, every aligned format needs at least 1.10× English's passes, and none gains 3 pp or more in accuracy |
| No pass reduction | The best aligned format's 90% lower bound of k/k_E is at least 0.90. If that also holds for F, stop the whole alignment line |
| No latency reduction | The lower bound of T_M/T_E within MDM is at least 0.95 |
| Accuracy loss | The upper bound of M − E is below −2 pp in either family |
| Unresolved | After the extension seeds, neither success nor any stop condition holds |

**Revise the hypothesis**

| If | Then |
| --- | --- |
| M beats named JSON but not English | The claim becomes "compact positional output beats named JSON". It is Sylang-specific only if M also beats C. Add a tuple-JSON arm |
| Passes improve but latency does not | Test a shorter aligned surface |
| Only F meets the pass and latency criteria | Redesign M with fixed-width literal slots |
| The benefit appears only early in training | Test underdetermined generation; make no latency claim |
| I/I₀ lies in [0.8, 1.25] and AR-M is at least as good as AR-E | General compactness, not a diffusion effect. Pursue it through the AR pilot on rephug/sylang#3 |
| The MDM never learns, even after the pre-registered fixes | Inconclusive, not support for H0. The pretrained pair needs a separate owner decision |

**Pre-registered expectation.** On this task, M's output is 38% longer than
concise English: 46.5 against 33.7 tokens per item. With the plan's CPU costs
(12.0 ms per English step, 13.2 ms per M step, 10.7 ms prefill), M must cut
passes to about 0.91× of English's to break even on within-MDM latency, and to
0.80–0.81× to meet the 0.90 latency criterion.

For comparison only: in the 864-meaning A/B world, the idealised τ = 0.9
exact-marginal sampler needs 0.81× the rounds of the codec's English (7.00
against 8.69). Those are oracle rounds, not passes of a trained model and not
latency.

The expected outcome is STOP, or the "beats named JSON only" revision.

**Cost and hardware**

- **Compute.** About 18.6 PFLOP for the core stages: 29–52 h on four
  single-thread processes at 25–45 GFLOP/s each. The worst case is about 35
  PFLOP (54–98 h). Latency measurement adds at most 4 h.
- **Engineering.** 4–6 days, about 3,000 lines including tests.
- **Money.** US$0. The same code could optionally run on free Lightning L4
  credits, about 2–3 GPU-hours.
- **Fit.** This review container cannot run the experiment: it has no
  PyTorch, and installing packages was outside the review's scope. The
  owner's 4-core server can run it once a pinned PyTorch CPU wheel is
  installed.
- **Owner approvals needed:** the install, the server reservation, the
  extension stages, any GPU use, and the paid follow-up.

**Smaller fallback (Tier-S): US$0, 16–29 h of compute, after the same 4–6
days of engineering.** It has five cells: AR × {E, M} and MDM × {E, M, C},
each with 3 seeds, about 10 PFLOP.

- **It can establish:**
  - the stop decisions;
  - whether M beats the generic code C, i.e. whether any effect is
    Sylang-specific or generic.
- **It cannot establish:**
  - anything about JSON, so it can never return PROCEED;
  - the "beats named JSON only" revision, which is the expected outcome;
  - whether an AR model can learn JSON at all (rephug/sylang#3 saw 0 of 2 runs
    learn compact JSON);
  - whether fixed width matters, because there is no F;
  - AR accuracy for C.

An even smaller MDM-only variant (about 17 h) cannot measure AR accuracy or
latency, so it does not meet the AR-against-diffusion requirement.

**What even the full screen cannot establish**

- 7B behaviour, GPU serving, and Mercury latency.
- AR-against-MDM wall-clock ratios. On CPU an MDM step costs 9–20× an AR
  step; at 7B on a GPU it costs about 1–1.5×.
- Pretrained priors, legend costs, and the mnemonic value of M's codes.
- Edit tasks, underdetermined generation, conditionals, record counts above 6,
  natural input, and Unicode.
- Effects smaller than about 20% in passes or 10% in latency, and an accuracy
  margin as tight as 1 pp.

## 10. Methodological flaws ranked by severity

Severities are after adversarial verification. None of these invalidates the
reports' computations. They change what the computations can be taken to show.

| Severity | Flaw | Where | Effect |
| --- | --- | --- | --- |
| Major | Compactness against English rests on version-tag length and a templated English. The template status is disclosed, but `cpu-diffusion-oracle-2026-10-20.md:71` drops the "controlled" qualifier. Against concise English, M is longer under every convention, and the same-pass linear work scenario reverses (0.81× tags removed; 0.71× M tagged) | `cpu-diffusion-oracle-2026-10-20.md:71-73`; `diffusion-sylang-2026-10-20.md:58,135,146`; `sylang_core/codec.py:215,218`; `docs/decisions.md:42-44` | Removes the density premise against English. No posterior, recovery ceiling or round count changes, because the tags are constant. M's tag-free fixed canvas is still 13 positions against English's 14 |
| Major | The exact 1-round and 100% figures need single-token labels. The labels {A, B} are stated and listed as a limit, but the headline never says it depends on them | `evaluation/diffusion_experiment.py:31,323`; `cpu-diffusion-oracle-2026-10-20.md:9-10,24,121,147` | The exact result does not survive multi-token names (M needs 2–3 rounds). M's ranking does: it stays lowest or tied, with its forward lead unchanged at 3.69 rounds |
| Major | Adaptive rounds depend on an exact product prior and an exact independence test. English's two scan orders and JSON's reverse order are pessimistic. The report says the policy is greedy, not optimal, and that a joint oracle needs one round, but does not quantify the effect | `evaluation/diffusion_oracle.py:251-355`; `cpu-diffusion-oracle-2026-10-20.md:126-156` | 1 against 4.69 becomes 1 against 2.79 with a better order, and M needs 2.80 when one meaning is dropped. One-step M validity stays at 99.89%, so the brittleness is in the round count, not in M's approximate parallelism |
| Major | The JSON baseline is named-field JSON. The tuple/array and generic AST-ID controls that the report requires before any Sylang-specific attribution were not run | `sylang_core/codec.py:209-211`; `diffusion-sylang-2026-10-20.md:361-365` | M's advantages over JSON cannot yet be separated from those of any positional code |
| Moderate | The proposed interaction gate (I > 1) passes on length alone: I exceeds 1 with no diffusion-specific effect whenever M's output is longer, with I₀ = 1.38 in passes on the planned task [new; derived in the plan, not adversarially verified] | `diffusion-sylang-2026-10-20.md:391-401` | Interpret I/I₀, not I |
| Moderate | Robustness is analysed for erasures only. Under an idealised substitution model, value errors in M are never detectable; this is the dual of the reported 100% one-round support | `cpu-diffusion-oracle-2026-10-20.md:42-48`; `diffusion-sylang-2026-10-20.md:93-94` | M's loss of error detection for value fields is not quantified |
| Moderate | The oracle gives every format its exact prior, so M's lack of training exposure is not costed. The report calls the oracle an upper bound that cannot estimate learning difficulty | `diffusion-sylang-2026-10-20.md:315-319`; `sylang_core/codec.py:38-44` | Comparisons are optimistic for M relative to any pretrained model |
| Moderate | The lead sentence credits compactness for surface-arm effects that follow token-to-field alignment. The next sentence and the body concede the point | `cpu-diffusion-oracle-2026-10-20.md:8-13,158-162,213-214` | The summary names the wrong variable for the surface arms |
| Moderate | The reports disclose that no learned model exists, but they do not engage the branch's own decision gates: compare against the strongest compact baseline, and get separate approval for diffusion runtimes. Prior reviews rephug/sylang#2 and #3 are not cited; neither is an ancestor of the reviewed commit, which partly explains that | `docs/decisions.md:42-44,51-53`; both reports | The case rests on oracle and tokenizer arithmetic without the project's existing gates |
| Moderate | The only implemented end-to-end task makes the JSON arm a copy task. The output-side conditions are planned but not built | `evaluation/harness.py:107-122`; `diffusion-sylang-2026-10-20.md:328-330` | Payload savings do not show up end to end, and a future comprehension result would be confounded |
| Moderate | No uncertainty is given for 11.66% (fixture bootstrap 7.3–16.6%) | `diffusion-sylang-2026-10-20.md:58` | Precision overstated |
| Moderate | Headline numbers are not protected by tests; 11 of 19 mutants survive | `tests/test_diffusion_*.py` | Regressions would be caught only by reproduction |
| Minor | Other precision gaps: one fixture dominates the quadratic proxy; Monte Carlo values are quoted where exact ones are cheap; the 50% table caption omits the length condition, which is stated just before the table (`:53-54`); the correlated token-masking and given-length round results are not reported | various | Precision overstated |
| Minor | External citations. Mercury 3's "1,068 tokens per second" is reported as "output tokens". BLD's explicit compression-difficulty trade-off (grouped sequential generation) is not mentioned. LLaDA's `GUIDELINES.md` has no inference section. The Mercury 3 API resets out-of-range temperatures to 1 and has no seed parameter, which the pilot protocol does not guard against | `diffusion-sylang-2026-10-20.md:17,74-82,255-260,337-342` | Presentation; pilot design |
| Minor | The proposed programme gives no budget, model sizes or GPU-hours (it defers sizing until scheduling, `:355-356`) and is not sequenced against the existing pretrained-pilot gate on rephug/sylang#3 | `diffusion-sylang-2026-10-20.md:324-401` | Planning |

## 11. Strongest arguments for and against continuing

**For continuing**

- The computational work is sound, reproducible and honestly caveated. It is a
  good base for a real experiment, and the reports already propose its shape.
- In this oracle, with independent fields, a layout with one token per field
  and fixed widths is the only one that stays fully valid under aggressive
  marginal-based parallel decoding. When fields are correlated it is also the
  most valid format under confidence thresholds of 0.5 and 0.7; at 0.9 all
  three formats are fully valid. M keeps its lead with the
  unequal-length names tested. With Alice/Bartholomew at τ = 0.9, English and
  JSON start producing invalid output (97.3% and 77.5% valid), while M does
  not [new, agent].
- M's output is 54–64% shorter than named-field JSON, the usual format for
  structured output, and output is where diffusion decoding spends its passes.
- Dream-v0-7B is an open masked-diffusion model initialised from Qwen2.5-7B.
  It uses the same vocabulary, merges and pre-tokenizer as the repository's
  pinned Qwen3 asset, plus two added special tokens. On a reconstruction of
  the Dream-v0-Instruct-7B tokenizer, 2,976 test strings tokenize identically
  [agent, external].
  - The token counts here therefore carry over to a real dLLM.
  - Dream and Qwen2.5-7B form a matched diffusion/AR pair with shared
    initialisation.
  - Dream's context is 2,048 tokens.
- In the project's only learned-model study, trained from scratch, a compact
  Sylang code reached 90% accuracy in fewer training items than English
  (54,000 against 60,000–62,000). That is a tie under the study's own
  pre-registered rule. The code clearly beat compact JSON and CSV [read,
  agent].

**Against continuing as framed**

- **Compactness.** Against a concise English, M is not more compact under any
  convention tested. Against the codec's English with only the version tags
  removed, M is 7.3% longer, though it is 10% shorter if constant leading and
  trailing tokens are stripped instead.
- **Parallel decoding.** The headline depends on one-token labels, an exact
  product prior and an exact independence test. With correlated fields, M
  loses exact one-step validity. Under fixed steps of 2, 3, 4 or 6 tokens it
  fails more often than English or JSON.
- **Error detection.** M gives up redundancy and all detection of value
  substitutions.
- **End-to-end share.** Payload savings are under 1% of an uncached
  single-record request's input-plus-output tokens, and about 4–6% with
  prefix caching. Those figures come from token counts of the authored prompt,
  with output length assumed, not measured. Legends and output format
  dominate.
- **Specificity.** Every advantage that survives is shared by a generic
  positional code, so none is yet specific to Sylang.
- **Compute.** At small scale, compute-matched studies find masked diffusion
  needs roughly 14–16× the compute of AR to match validation loss, although
  diffusion gains when data repeat, as in a finite synthetic grammar [agent,
  external]. A fair diffusion experiment therefore needs an explicit compute
  budget, which the report does not give; it defers sizing until an
  experiment is scheduled (`diffusion-sylang-2026-10-20.md:355-356`).
- **No model evidence.** No model in this repository has been trained or
  evaluated on M.

## 12. Recommendation

**Revise the hypothesis.**

- **Old framing.** "Sylang could reduce the work of diffusion language
  modeling" (`diffusion-sylang-2026-10-20.md:8-9`). It was supported by:
  - M's 19.5% and 11.66% density over controlled English;
  - the 1-against-4.69 adaptive rounds;
  - the 1.132× linear scenario;
  - the summary "compactness trades erasure robustness against parallel
    consistency" (`cpu-diffusion-oracle-2026-10-20.md:8`).

  The report itself says density alone does not imply fewer passes or faster
  serving (`diffusion-sylang-2026-10-20.md:9-11`).
- **Revised hypothesis.** This is H1-align in section 9. A field-aligned output
  encoding (M, a generic code line C, or C with fixed-width name slots F) lets
  a masked-diffusion decoder reach matched exact-AST accuracy, losing no more
  than 2 percentage points, with:
  - at most 0.80× the passes of both concise English and named JSON;
  - at most 0.90× their within-MDM latency.

  Meeting the pass criterion implies I/I₀ ≥ 1.25. An I/I₀ between 0.8 and 1.25
  is read as general compactness, not a diffusion effect. Sylang M gets credit
  only if it is also at least 10% faster than C and F (H1-M).
- **Stop quoting** 19.5% and 11.66% as compactness against English, and
  1 round against 4.69 as a parallel-decoding advantage.

**Concrete next action.** Steps 1–2 can start now: they need no installs,
training, spending, merges or deployments. Step 3 trains models on the owner's
CPU at US$0 cash, and starts only after the owner explicitly approves the
install and each server reservation. Step 4 may license paid work, again only
with owner approval.

1. **Correct the reports.** Add an erratum note at the top of both reports
   that points to this review. This review has not edited them. The erratum
   should cover:
   - the version-tag split;
   - the label and prior sensitivity of the rounds table;
   - the dependence on the sampler and on the exact independence test;
   - the substitution blind spot;
   - the I₀ correction to the proposed AR-against-diffusion gate.
2. **Freeze the pre-registration.** Complete the offline steps in the
   [plan](../plans/diffusion-next-experiment-2026-10-22.md):
   - renderers and strict parsers for E, J, M, C and F;
   - the neutral-input generator;
   - panels frozen with hashes;
   - sampler conformance against `FiniteOracle`;
   - the analysis code and every threshold.
3. **Run the screen, with the owner's approval.** Install a pinned PyTorch CPU
   wheel on the server and reserve the machine for about three days. Run the
   throughput benchmark, then Stage 0 and looks 0–1.
4. **Act on the pre-registered outcome:**
   - **STOP:** close the diffusion line for M and publish the negative result.
   - **REVISE:** follow the matching row of the table in section 9.
   - **PROCEED:** run the owner-approved seed-0 stage of Dream-v0-Base-7B
     against Qwen2.5-7B, about US$6–7.
   - **INCONCLUSIVE** (a precondition fails, including the Stage-0 gate): fix
     the recipe, not the hypothesis. If the MDM still does not learn after the
     pre-registered fixes, the pretrained pair needs a separate owner
     decision; it does not follow automatically.

**Why run a screen that is expected to stop.** The pre-registered expectation
is STOP, or the "beats named JSON only" revision. The prior learn rate (2 of 5
seeds) also makes an inconclusive gate plausible. If the owner does not want
to spend 4–6 engineering days and about two days of server time on an
expected negative, stopping the diffusion line now is a defensible
alternative. The screen's value is that it would be the project's first
trained-model test of this question, at US$0 cash.

**Hold until step 4:**
- **The paid Mercury 3 pilot and any GPU rental.** Both also need the separate
  approval that `docs/decisions.md:51-53` requires for diffusion runtimes.
- **Sequencing with the pretrained AR pilot on rephug/sylang#3.** That pilot
  answers a different question: does a pretrained model read Sylang codes as
  well as JSON or English? Its own plan says its gate should be read before
  other spending. The two lines are compatible.

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
