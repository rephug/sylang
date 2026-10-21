# CPU oracle experiments: density, masking, fusion, and refinement

Date: **2026-10-20**. These are executed finite-world experiments, with exact
conditional probabilities and explicitly sampled mask expectations. No neural
model was trained or run. Oracle rounds measure named idealized sampling
policies, not Mercury refinement steps or GPU latency.

**Finding:** compactness trades erasure robustness against parallel consistency.
In the independent-field world, M's token boundaries permit exact parallel
sampling with fewer positions. English retains more clues under equal token
masking. Fusion helps when it contains a dependency; it can also remove useful
redundancy. Matched semantic observations give identical recoverability across
all three reversible representations.

## Experiment and evidence

- [Machine-readable results](../../benchmarks/results/diffusion-oracle.json)
- [Runner](../../evaluation/diffusion_experiment.py)
- [Finite posterior/refinement engine](../../evaluation/diffusion_oracle.py)
- [Exact fusion controls](../../evaluation/diffusion_fusion.py)
- [Lossless tokenizer adapters](../../evaluation/diffusion_surfaces.py)

The complete independent universe has **864 equally likely meanings**:
subject and object each in `{A,B}`; three relations; two polarities; three tenses;
three aspects; four evidence annotations. Its entropy is **9.7548875 bits**.
Every record uses the current core-v0.1 codec, and English, JSON, and M pass
exact text/token/AST round trips. JSON is canonical minified named-field JSON,
not a minimum-length tuple encoding. There are no legends, instructions, or
chat wrappers in these payload experiments.

A second, deliberately correlated prior has **18 equally likely meanings**.
Choose subject index `s`, relation index `r`, and tense index `t` independently;
set object index to `1-s`, polarity to `(s+r)%2`, aspect to `t`, and evidence to
`(s+r+t)%4`. This introduces known dependencies and redundant clues. It is an
artificial stress test, not a model of natural language. Its entropy is 4.169925 bits.

Four tokenizations are compared: the two pinned Qwen3/Qwen3.5 tokenizer assets,
a lossless lexical splitter, and UTF-8 bytes. Only the Qwen streams are existing
model token IDs. The other two are explicitly surrogate controls. No tokenizer
was retrained or altered. Hypothetical native semantic slots are a separate arm.

The main predicate experiments use independent Bernoulli erasures at probabilities
`0, .25, .5, .75, .9, 1`. For surface strings, enumerate every source meaning
and sample **16 masks per meaning** at each interior rate, with seed 20261020.
Each resulting posterior is exact; only its expectation over surface masks is
Monte Carlo. Endpoints, semantic-unit masks, fusion controls, and refinement
distributions are enumerated exactly. Reported mask standard errors exclude
uncertainty about real-world priors and generalization.
The exact binary controls use the same rates except `.9`.

## Recoverability under token erasure

For a visible token observation, the oracle considers every prior-supported
meaning with the observed length and matching visible positions. Because
erasure likelihood is equal for matching candidates, the posterior is uniform.
If `m` meanings remain, entropy is `log2(m)`, the Bayes ceiling for recovering
the original is `1/m`, and unique recovery occurs only when `m=1`.

At **50% Qwen3 token masking** in the independent world:

| Encoding | Mean payload tokens | Residual semantic entropy | Bayes original-recovery ceiling |
| --- | ---: | ---: | ---: |
| Controlled English | 21.11 | 2.930 bits | 23.94% |
| Minified JSON | 46.83 | 3.785 bits | 12.66% |
| Sylang M | 17.00 | 4.888 bits | 7.81% |

Recovery estimates have mask standard errors of approximately 0.21, 0.13,
and 0.11 percentage points respectively. This is recovery of a randomly hidden
original from surviving clues, not language-model comprehension accuracy.

M uses **19.5% fewer tokens than English** on this generated universe. That
differs from the earlier 11.7% smoke-suite result because the distribution and
literal lengths differ. Token density is workload-dependent.

Equal token masking leaves the longer encodings more surviving tokens and
redundant/overlapping clues. It does not hold semantic information constant.
The Qwen3.5 density results agree here; small differences between its mask
estimates and Qwen3's arise under separately seeded mask draws and do not
establish a tokenizer or model quality difference.

### Length is observed information

In the independent world, the Qwen3 token count alone reveals **1.975 bits for
English**, **1.459 bits for JSON**, and **0 bits for M**. With all content
masked but the true length given, English therefore starts with an advantage.

The fixed-canvas control pads every stream to the public maximum length for
that encoding and masks PAD exactly like content. At full erasure, all three
then return to **9.7548875 bits** and a **1/864 original-recovery ceiling**.
Visible PAD can still reveal length at intermediate rates; this is recorded
rather than treated as absent information. Padding also removes some possible
length-dependent computation savings.

### Matched semantic erasure

Reveal the same semantic field paths and values, removing surface text and
length clues. Exhaustively enumerate all 128 field masks. English, JSON, and M
have identical posteriors and recovery metrics, as their codecs are bijections
over the same supported meanings. At 50% field erasure, the independent-world
entropy is **4.8774438 bits** and original-recovery ceiling **7.8125%**.

This control shows that a change in spelling alone does not create information.
Any advantage must come from computational cost, learned inductive bias, noise
placement, dependency structure, or the representation of a task's information.

## Refinement experiments

Recovering a hidden original and generating a posterior-consistent completion
are different tasks. Self-generated commitments add no evidence about which
ambiguous original was hidden. Extra rounds can improve completion consistency;
they cannot exceed the observation's Bayes original-recovery ceiling.

The independent-marginal sampler predicts all positions from exact marginal
distributions in one round. Those individually correct predictions can disagree
with each other. The result reports both mass on the finite posterior's support
and total variation from the exact posterior. Support consistency is stronger
than grammar validity in the correlated prior: an unsupported record can still
be valid core-v0.1 syntax.

For full erasure with the correct Qwen3 output length given, one-round support
mass in the independent world is **7.76% English, 43.15% JSON, and 100% M**.
This is a property of the declared sampler and known prior, not a neural-model
success rate. An unrestricted joint posterior oracle completes any format in
one round, which demonstrates why these counts are sampler-specific.

The main adaptive sampler chooses all currently deterministic positions and
greedily adds uncertain positions only when their **joint** posterior exactly
factorizes. It samples those positions independently, conditions the next
round on the sampled commitments, and repeats. It tests full joint independence,
not merely pairwise independence. When a prediction makes remaining positions
deterministic, a subsequent prediction round still materializes them.

Forward and reverse position orders expose scheduling sensitivity. The policy
has exact final posterior fidelity, but its round count is greedy rather than
globally optimal. Fixed contiguous/interleaved schedules and rejection sampling
are additional controls in the JSON results. Their minimum tested budgets must
not be read as the number of steps a trained diffusion model needs.

For the **fully erased fixed Qwen3 canvas**, English allocates 23 positions,
JSON 48, and M 17. Target-specific length is hidden. Exact adaptive round counts
are:

| Prior | Encoding | Forward order, mean (maximum) rounds | Reverse order, mean (maximum) rounds |
| --- | --- | ---: | ---: |
| Independent 864 | English | 4.69 (6) | 6.00 (8) |
| Independent 864 | JSON | 3.00 (3) | 3.50 (4) |
| Independent 864 | M | 1.00 (1) | 1.00 (1) |
| Correlated 18 | English | 4.78 (5) | 4.94 (6) |
| Correlated 18 | JSON | 2.00 (2) | 4.83 (6) |
| Correlated 18 | M | 2.00 (2) | 3.67 (4) |

These counts describe the exact greedy policy, not an optimal sampler or
trained model. M's one-round result disappears when semantic fields depend on
each other. In the forward correlated case, JSON and M require the same rounds
despite their different lengths. An unrestricted joint oracle uses one round
for every row.

Tokenization also changes the result: **lexical JSON needs one round in the
independent world**, despite its 83-position canvas. Its field values remain
atomic under that splitter. With Qwen tokens, fragmentation and position shifts
introduce additional dependencies. This is evidence to study token/meaning
alignment, rather than attributing the round reduction to density alone.

All outputs use the same fixed public one-predicate topology; shape and length
prediction are not tested. Detailed distributions are stored under each arm's
`refinement_fixed_canvas` field. Neither the fixed-canvas comparison nor the
gold-length condition estimates a deployed model's length prediction cost.

## Fusion and redundancy

Native-slot ablations regroup the seven semantic fields into seven separate
slots, four slots with a 72-state grammatical-feature bundle, or three slots
with a 216-state relation/feature bundle. These are hypothetical atomic IDs,
not newly trained Qwen tokens.

Under the independent prior and 50% slot erasure:

| Native representation | Slots | Observed vocabulary | Residual entropy | Original-recovery ceiling |
| --- | ---: | ---: | ---: | ---: |
| Separate fields | 7 | 19 | 4.8774 bits | 7.8125% |
| Fuse polarity/tense/aspect/evidence | 4 | 79 | 4.8774 bits | 19.0104% |
| Fuse relation and all grammatical features | 3 | 220 | 4.8774 bits | 28.2552% |

Expected residual entropy is unchanged in this independent-factor control.
Fusion makes erasures more all-or-nothing, changing exact whole-record recovery
without increasing average retained information. All three native encodings
already admit exact one-round independent sampling in this world; fusion adds
no round reduction there. It does reduce positions, while increasing vocabulary
size and the number of possible values each fused prediction must distinguish.
Embedding/head cost and learnability are unmeasured.

In the correlated 18-state prior, fusion can hide several mutually informative
fields at once. At 50% slot masking, residual entropy rises from **0.673 bits**
for separate fields to **1.355** and **1.710 bits** for the two fusion arms.
Fusion therefore does not uniformly improve erasure robustness.

### Exact binary dependency controls

Let a fair semantic bit be represented by two equal fields `(Z,Z)`. Compare
separate slots, one fused slot, and duplicated separate slots (four copies).
Exhaustively enumerate source states, every mask, every sampler output, and
all allowed independent-group policies:

| Representation | Residual entropy at 50% erasure | Full-erasure one-round support | Exact independent-group rounds at full erasure |
| --- | ---: | ---: | ---: |
| Two separate equal fields | 0.25 bits | 50% | 2 |
| One fused pair | 0.50 bits | 100% | 1 |
| Four redundant copies | 0.0625 bits | 12.5% | 2 |

This establishes both mechanisms in the same experiment: fusion can halve the
conditional sampling rounds when it encloses a dependency, while redundancy
improves recovery from erasure. Fusing only a pair in a three-field parity
constraint still requires two independent-group rounds. Dependency boundaries
matter; density by itself is insufficient.

The binary oracle optimizes worst-case rounds within its exact
posterior-independent group policy class. This optimality claim applies only
to those tiny controls. The main 864-state surface experiment uses the greedy
policy described above. Neither is a lower bound for all possible joint or
learned diffusion samplers.

## Reproduce and interpret

Using the optional runtime/assets already pinned by the repository:

```sh
python3 -m venv .cache/diffusion-venv
.cache/diffusion-venv/bin/python -m pip install --only-binary=:all: --no-deps -r requirements-tokenizers.txt
python3 tools/fetch_tokenizers.py --download
.cache/diffusion-venv/bin/python -m evaluation.diffusion_experiment \
  --tokenizers lexical,utf8_bytes,qwen3,qwen3.5 --samples-per-state 16 \
  --output benchmarks/results/diffusion-oracle.json
.cache/diffusion-venv/bin/python -m unittest discover -s tests -v
```

For a standard-library-only run, omit installation/download and run
`python3 -m evaluation.diffusion_experiment --samples-per-state 16 --output .cache/oracle-stdlib.json`.
That run includes lexical/byte/native controls and excludes Qwen token IDs.
Asset fetching is separate; the experiment runner itself never uses the network.
Results record source/corpus hashes, tokenizer revisions, mask seed, settings,
and the Python version. There are no timestamps or CPU timing claims.

All **92 tests passed** in the pinned-tokenizer environment. The tests include
analytic entropy/recovery cases, exact codec/token transport,
fixed-canvas leakage controls, a pairwise-independent parity trap, biased
rejection sampling, correct final materialization counts, and native fusion
invariants. Work proxies `K*N` and `K*N²` omit output heads, oracle computation,
training, literal decoding, validation, and hardware effects.

A second complete run produced byte-identical JSON. Its SHA-256 is
`fa73a6d0ee7ab78e201cf771bf53d7205c7174c041b56cbf13d5ad8a3d1d7460`.
All six recorded source hashes match the executed files. The 37 oracle,
fusion, and experiment tests also passed with Python site packages disabled
(`python3 -S`), separately verifying the standard-library computational core.

These experiments justify testing semantic token alignment and dependency-aware
fusion in a trained model. They also identify failure modes to preserve in that
test: redundancy loss, output-length assistance, scheduling sensitivity, and
mistaking consistent generation for recovery of absent information. The next
step is a small matched training experiment with equal-exposure/equal-compute
learning curves and exact semantic task scoring.
