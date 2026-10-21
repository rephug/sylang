# Sylang with diffusion language models

Research date: **2026-10-20**. This note separates repository measurements,
published results, mathematical examples, and proposed experiments. No model
inference or training was performed for this investigation. The accompanying
offline analysis reuses saved tokenizer counts; it does not measure diffusion.

**Assessment:** Sylang could reduce the work of diffusion language modeling,
especially if the denoiser operates directly on compact semantic units. Density
alone does not imply fewer denoising passes, better reasoning, or faster serving.
The research objective should be **more semantically correct work per second at
matched quality**, with learning curves showing how that changes with training.

## 1. What Mercury 3 establishes

Inception's [October 19 announcement](https://www.inceptionlabs.ai/blog/introducing-mercury-3)
reports 1,068 output tokens/second and a 1M-token context. These are vendor claims
for Mercury 3's own representation and serving system, not measurements of
Sylang. Its announcement does not specify the tokenizer, denoising schedule,
block sizes, or enough architecture and benchmark conditions to predict a
Sylang speedup.

The earlier [Mercury technical report, June 2025](https://arxiv.org/html/2506.17298v1)
describes Transformer denoisers, parallel refinement, a denoising training loss,
and a proprietary engine with dynamic batching, paging, and custom kernels.
That is useful context, **not a specification of Mercury 3's internals**.

Current [streaming documentation](https://docs.inceptionlabs.ai/capabilities/streaming)
describes refined text blocks and `diffusing=true` streams of intermediate full
text. We can inspect visible revisions in a later API pilot, but network events
are not guaranteed to correspond one-to-one with neural forward passes. The
[chat API](https://docs.inceptionlabs.ai/api-reference/chat/create-a-chat-completion)
documents usage counts and sampling controls; the reviewed public interface
does not document replacing the tokenizer or directly selecting a denoising
step count. [Reasoning effort](https://docs.inceptionlabs.ai/capabilities/reasoning-efforts)
is a separate control, not an exposed refinement-step setting. Pin it explicitly.

Consequently, Mercury can test whether a frozen commercial diffusion model
understands and benefits from Sylang prompts. Open training code is needed to
test the stronger native-representation hypothesis.

## 2. What Sylang currently supplies

The executable [core-v0.1 grammar](../experimental-grammar.md) represents three
relations, explicit polarity/tense/aspect/evidence, opaque Unicode labels, and
bounded conditional trees. English, JSON, DSL, Prime, and M serialize the same
validated semantic tree. M is a compact deterministic serialization. There is
no trained Sylang tokenizer, complete natural-language translator, or learned
diffusion model in this repository.

The [saved baseline](../baseline-results.md) contains 33 authored smoke fixtures:

| Same unchanged Qwen3 tokenizer | Controlled English | M | M / English |
| --- | ---: | ---: | ---: |
| Bare payload tokens | 1,166 | 1,030 | 0.88336 |
| Complete experimental prompt tokens | 15,686 | 12,151 | 0.77464 |

M saves **11.66% of payload tokens** here. Qwen3.5 gives a similar 11.62%.
Compact JSON requires 9,505 complete-prompt Qwen3 tokens, versus M's 12,151:
**JSON wins that full-prompt comparison**. Legends differ in length, so these
prompt rankings are properties of the authored protocol. They are not intrinsic
language efficiency. Output tokens, learned comprehension, and inference speed
were not measured. Historical 45–60% savings remain unverified.

These are Qwen tokenizer counts, not Mercury tokenizer counts. Two fixtures
diagnose transport problems: NFC normalization changes decomposed literal text;
literal reserved markers can become special token IDs. Retain those results as
diagnostics, and report the same excluded subset across every format when
computing an additional transport-clean comparison. Exact codec round trips do
not establish an exact model-token transport path.

## 3. Where a diffusion benefit could come from

In masked discrete diffusion, training corrupts token sequences and teaches a
model to predict clean tokens. Generation starts with masked positions and
reveals or revises groups of predictions across repeated passes. Some samplers
only unmask; others remask or revise. Block diffusion denoises within successive
blocks while retaining completed context. Continuous latent diffusion is a
different option: denoise compressed vectors and decode them into text later.
[LLaDA guidelines](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md),
[MDLM](https://arxiv.org/abs/2406.07524),
[Block Diffusion](https://arxiv.org/abs/2503.09573).

Possible advantages, all hypotheses for Sylang:

- **Less repeated work:** a shorter actual denoising canvas reduces positions
  processed on each pass, including feed-forward layers and attention.
- **More meaning per block/context:** a fixed token block or context window can
  cover more semantic records; fewer output blocks may need sequential decoding.
- **Fewer dependencies between positions:** an atomic symbol could jointly
  encode a small dependent bundle that would otherwise need coordinated token
  predictions. It may avoid incompatible parallel choices.
- **More useful constraints:** explicit roles, scope, and field types could
  make partial structures easier to validate, constrain, and selectively repair.
- **Fewer surface choices:** canonical serialization removes optional wording
  when the desired output is an exact semantic record.

The first two benefits can also help autoregressive models. A claim of a
*special synergy with diffusion* requires an AR-versus-diffusion control.

### Cost and break-even calculation

Let `N` be the positions actually processed, `K` the forward-pass count, `V` the
effective output vocabulary, and `O` encoding, scheduling, and other overhead:

```text
T = O + K C(N, V)
C(N, V) ≈ h + a N d² + b N² d + c N V d
```

This is an explanatory full-attention Transformer work model, not a latency
predictor. `h` represents costs that do not shrink with sequence length;
projection over only masked positions changes the last term. GPU occupancy,
weight movement, precision, and kernels determine wall time.

For identical models/vocabularies, negligible request and per-pass fixed costs
(`O≈0`, `h≈0`), uniform compression
`r=N_M/N_E`, and quadratic share `q` of the baseline modeled variable work:

```text
c = C_M / C_E = (1-q) r + q r²
speedup = 1 / [(K_M/K_E) c]
M is faster if K_M/K_E < 1/c
```

With a common per-request overhead fraction `f` of baseline latency, the illustrative ratio is
`T_M/T_E = f + (1-f)(K_M/K_E)c`. Extra conversion, failed outputs, and retries
add costs beyond this expression. If the output head changes, recompute its
cost rather than reusing the same coefficients. Per-pass fixed work recurs
with `K`: if its fraction of baseline pass cost is `u=h/C_E`, replace `c` by
`u+(1-u)[(1-q)r+qr²]`. Do not count that work as per-request overhead.

| Hypothetical uniform canvas reduction | Same-pass linear speedup | Same-pass attention-only speedup |
| --- | ---: | ---: |
| 11.66% (`r=1030/1166`) | 1.132× | 1.282× |
| 40% (`r=0.60`) | 1.667× | 2.778× |
| 50% (`r=0.50`) | 2.000× | 4.000× |

These columns are limiting scenarios, not bounds on observed serving speed.
At the current payload ratio, just over **13.2% more passes erases the linear
benefit** before extra conversion costs. The attention-only threshold is 28.2%.
The 1.282× number assumes every sequence has that same ratio: for the actual
heterogeneous suite, use `sum(N_E_i²)/sum(N_M_i²)`, not the square of aggregate
token totals. The offline tool computes that distinction explicitly.

For the actual Qwen3 fixture lengths, the calculation is **1.132× for linear
work but only 1.084× for quadratic work**: the maximum-length literal fixture
has English/M lengths of 306/303 tokens and dominates the squared sum. Thus
the actual suite's attention-only pass-inflation threshold is about **8.4%**,
not the uniform example's 28.2%. With hypothetical padded blocks, English/M
counts are **48/45 at B=32, 37/37 at B=64, and 35/35 at B=128**. These reuse
input payload lengths as scenarios; they do not measure output block counts.

Further conditions can remove the expected gain:

- Compressing only output gives `N_M=P+rG`, not `r(P+G)`, if the full clean
  prompt `P` is recomputed alongside output `G`. Compressing only input does not
  reduce output blocks. With a cached prefix, use the actual block/prefix costs.
- A fixed 512-position output allocation can cost the same for a 450-token
  answer and a 300-token answer. Shorter emitted text must lead to smaller
  allocations, early stopping, or fewer blocks to save that work.
- With fixed padded block size `B`, block count is `ceil(G/B)`. Moving from
  60 to 53 tokens at `B=64` still uses one block; moving from 70 to 62 removes
  one. Fixed semantic block size is a different experiment.
- Fewer KV positions can reduce cache memory; model weights do not shrink.
  Less arithmetic does not guarantee less wall time on an underfilled GPU.
- A larger fused vocabulary can increase projection cost and data sparsity.
  Masking illegal IDs improves validity but does not automatically avoid the
  full output matrix multiplication.

### Density preserves information and changes the noise problem

For a bijection between supported semantic records and canonical encodings,
the document entropy is unchanged: `H(encode(S))=H(S)`. Fewer tokens redistribute
that information. This does not force any particular conditional entropy at a
noise level, number of neural passes, or generalization ability.

A model-free counterexample makes the tradeoff precise. Let `Z` be a fair bit;
encode it as `(Z,Z)` or compactly as `(Z)`. Mask positions independently with
probability `p`:

```text
Expected residual semantic entropy:
  redundant encoding: p² bits
  compact encoding:   p bits
```

At 50% masking, compact coding leaves 0.5 unknown semantic bits versus 0.25.
Removing redundancy makes this denoising problem harder at equal mask fraction.
But when both redundant copies are unknown, sampling their exact marginals
independently produces inconsistent copies half the time; a joint compact
symbol always represents a valid bit. Fusion can improve joint consistency
while making each mask erase more information. Neither example predicts a
trained model's wall time. A fair noise comparison needs both equal-token-mask
and matched-semantic-erasure conditions.

## 4. Technical implementations worth testing

| Route | Concrete mechanism | Main uncertainty |
| --- | --- | --- |
| Existing tokenizer + M surface | Prompt or fine-tune an ordinary dLLM on exact M strings; retain its token IDs. | M punctuation/opcodes can fragment; unfamiliarity and legends can overwhelm savings. |
| Native semantic token stream | Assign IDs to structural operators and field values; use subwords/bytes or exact copy paths for open literals. Add or adapt embeddings and output head, then train with denoising loss. | The model must learn these IDs. A lexer token is not an existing model token. |
| Typed AST denoising | Represent nodes/slots with role and type embeddings; mask semantic fields/subtrees; predict shape/length as well as content. | Typed slots may need more positions; correct tree shape must not be supplied only to Sylang as hidden assistance. |
| Semantic block diffusion | Denoise a predicate or subtree in parallel; cache completed blocks; choose adaptive stopping at semantic boundaries. | Dependencies crossing blocks and repeated scope/label checks may require additional passes. |
| Compressed latent diffusion | Encode one or several semantic records into vectors; denoise a shorter latent sequence; decode to a validated tree or M. | Latent reconstruction can lose distinctions; encoder/decoder work and decoder serialization count toward latency. |

The most defensible design starts with **factorized semantic fields plus an
exact literal path**, then selectively fuses frequent dependent bundles. The
current polarity/tense/aspect/evidence product has `2×3×3×4=72` states; including
relation gives 216. This is a manageable experimental fusion family, unlike a
token for every possible sentence. Compare separate fields, 72-state bundles,
and 216-state bundles, including held-out combinations and fallback behavior.
No fixed historical 8,192-token vocabulary is justified by the present evidence.

Current M opcodes are also position-dependent: for example `s` can mean `see`
or `simple` in different fields. A native typed-token design must explicitly
choose whether field types disambiguate shared IDs or each meaning gets a
distinct ID. Neither choice proves neuron monosemanticity.

Specific interventions:

1. **Field-aware corruption.** Mask a complete field or AST subtree, not just
   a random punctuation fragment. Include independent token masking as the
   baseline. Change the training corruption and sampling rules together;
   an arbitrary new schedule on a frozen model is not an equivalent model.
2. **Structure/content stages.** Predict node types, edges, and scope first,
   then fill labels and grammatical fields. Ablate this against joint sampling;
   an early structural error may otherwise constrain every later prediction.
3. **Joint constraints and repair.** A partial sequence with holes must admit
   a valid completion. Ordinary left-to-right grammar masks do not solve
   arbitrary-order infilling. [Constrained diffusion with CFGs](https://arxiv.org/abs/2508.10111)
   supplies a relevant method. Add core bounds and literal checks separately.
   Parser success does not establish intended meaning; repair may need adjacent
   nodes or an entire dependent subtree, not just the locally invalid token.
4. **Literal copying and optional external dictionaries.** Keep exact names,
   Unicode, and code outside lossy latent reconstruction where possible. A
   dictionary/pointer scheme would be a new experimental transport layer, not
   an existing feature of core-v0.1. Count side-table tokens, resolution work,
   collisions, unseen-label fallback, and codepoint fidelity.
5. **Calibrated early stopping.** Track confidence and stability per semantic
   field, especially polarity and scope; freeze supported spans only when
   calibrated on holdouts. Measure false commitment and remasking. High
   confidence and grammatical validity do not prove semantic correctness.

Do not apply an autoregressive model with its causal mask removed and call
that a trained diffusion baseline. Denoising training or a validated conversion
recipe is required.

### Relevant published work and limits

- [Fast-dLLM](https://arxiv.org/abs/2505.22618) studies confidence-aware parallel
  decoding and approximate caching. It makes token dependence a practical issue:
  Sylang fusion could remove some dependencies or lower confidence and hurt
  parallelism. Both need measurement.
- [BLD, October 7, 2026](https://arxiv.org/abs/2610.09311) denoises 64 continuous
  block states for 1,024 surface tokens, then uses parallel local AR branches.
  This directly supports shortening the repeatedly processed state sequence.
  Its [full evaluation](https://arxiv.org/html/2610.09311v1) has tokenizer,
  precision, training-budget, and quality differences: greater throughput does
  not establish uniformly lower single-request latency or matched-quality gains.
- [JPEG-DLM, September 27, 2026](https://arxiv.org/abs/2609.33698) jointly learns
  compression, flow modeling, and decoding, reporting about 2.3× ELF throughput
  at a 0.5 compression rate. This is a recent continuous-latent preprint whose
  abstract promises code; it is not evidence for discrete Sylang or a ready
  replication package. Joint learnability matters as much as compression.
- [Beyond Text Compression, ACL 2025](https://aclanthology.org/2025.acl-long.1546/)
  finds compression alone an incomplete tokenizer-quality proxy. Its smaller
  model methodology motivates inexpensive screening; it does not guarantee
  that small synthetic Sylang results transfer to frontier models.

## 5. Tests before substantial training

### A. Offline work accounting — executable now

Run the standard-library tool from the repository root:

```sh
python3 tools/analyze_diffusion_tradeoffs.py --output benchmarks/results/diffusion-tradeoffs.json
```

The [tool](../../tools/analyze_diffusion_tradeoffs.py) and
[saved analysis](../../benchmarks/results/diffusion-tradeoffs.json)
reanalyze `initial-tokenizers.json` using paired per-fixture lengths,
distributions, linear/quadratic work proxies, and fixed-block rounding. It
reports source hashes and an additional subset excluding both transport
diagnostics. These are **input/payload length scenarios**, not measured output
canvases, model comprehension, denoising passes, or serving latency. Do not
concatenate the independent fixtures and pretend they were one attention matrix.

Next, create a separate frozen generator-based suite with more tree sizes,
literal lengths, and semantic compositions. Preserve the current core bounds.
Use independently authored labels/templates and unseen combinations; the public
33 fixtures remain smoke tests. Measure headers, legends, chat wrappers, output
targets, dictionary side tables, and Unicode transport separately.

### B. Exact small-state oracle — proposed, CPU-only

Update: the first bounded predicate-world and binary controls have now been
[implemented and run](cpu-diffusion-oracle-2026-10-20.md). The broader subtree,
corpus-prior, and topology experiments below remain proposals.

Enumerate a finite corpus of supported ASTs with a declared prior. Render each
format, corrupt it, and compute the exact posterior over candidate ASTs by
enumeration. Measure posterior semantic entropy, recoverability, dependencies
between masked fields, and consistency of parallel marginal samples. Begin
with the binary duplication example, then a finite predicate/subtree universe.

Use three corruption comparisons: independent token masks, the same hidden
semantic field paths across formats, and masks calibrated to similar posterior
entropy. Run both uniform and explicitly correlated priors. Report the prior;
an oracle with uniform independent fields can hide the dependencies that matter
in real tasks. Token-to-field alignment needs declared handling of tokens that
cross field boundaries. Revealed delimiters and tree shape are information too.

An oracle using the full candidate universe, gold length/shape, or all possible
answers is an analytical upper bound, not a learned-model result. Partition
oracle knowledge from training/evaluation and never use hidden gold structure
to supply only one representation with scaffolding. This experiment can expose
bad corruption choices; it cannot estimate GPU latency or learning difficulty.
For ambiguous masks, score against the exact posterior or Bayes recovery
ceiling: failure to reproduce a randomly drawn hidden original can be
intrinsically unavoidable, even for a perfect denoiser.

### C. Frozen-model pilot — proposed, no tokenizer changes

Use Mercury 3 and/or an available open diffusion model with identical underlying
tasks. Compare English, JSON, DSL/Prime, and M with preregistered instruction and
example budgets. Run input-only, output-only, and both-side encoding conditions.
First reconstruct the exact AST into the same JSON output to isolate input
comprehension; separately generate each target representation and validate it.

Use field extraction and precisely specified transformations (swap ordered
roles, change one feature, preserve conditional branches) with deterministic
answers. Core-v0.1 does not implement unrestricted inference, truth evaluation,
or program execution: those would require separately specified semantics.

For Mercury, record actual API token usage and model ID/date, reasoning effort,
sampling settings, cache state, concurrent load, request failures, and retries.
Use ordinary streaming for timing; reserve diffusing streams for qualitative
revision analysis because instrumentation adds traffic. Record time to first
stable usable content and full validated answer. Pin or record endpoint
revisions where available; commercial endpoints can change.

A poor frozen-model M result diagnoses distribution shift and prompt overhead.
It cannot rule out a well-trained native representation. A good result proves
usefulness on that interface, not the benefit of replacing the tokenizer.

## 6. Training experiments that distinguish the hypotheses

Start with [MDLM's released training/baseline framework](https://github.com/kuleshov-group/mdlm)
or [BD3-LMs](https://github.com/kuleshov-group/bd3lms), at a small configurable
scale. [LLaDA](https://github.com/ML-GSAI/LLaDA) and
[Dream](https://github.com/DreamLM/Dream) are later open-model adaptation paths;
their checkpoints and runtimes do not imply matching Mercury serving behavior.
Pin source commits, checkpoints, tokenizers, dependencies, and sampler settings
when an experiment is actually scheduled.

The first clean factorial is **AR versus masked diffusion × English, compact
JSON, and M**, with equivalent semantic examples, model capacity, and tuning
budgets. Keep DSL/Prime as structural controls where resources allow. The current
JSON baseline is minified named-field JSON, not the shortest possible JSON
encoding; add a reversible tuple/array control before attributing compactness
to Sylang-specific design. Add native
semantic IDs afterward, with a generic equally compact AST-ID control: this
separates compression and structure from Sylang-specific surface choices.
Frequency-matched renaming/cipher controls can probe reliance on pretrained
symbol meanings. Report total and non-embedding parameter counts when V changes.

Run two complementary training comparisons:

- **Equal semantic exposure:** same records and task distribution, so shorter
  encodings do not get extra facts merely by packing more examples per batch.
- **Equal compute:** matched measured training FLOPs/GPU budget, allowing
  compressed encodings to spend saved computation on more semantic examples.

Equal token count alone answers neither question. Plot semantic accuracy and
quality-matched serving latency against both cumulative compute and examples,
with logarithmically spaced checkpoints and multiple seeds (start with three).
Do not conclude that native Sylang must eventually win: test whether the
learning curves improve, plateau below the baseline, or cross it. Expand model
size and data diversity only after a useful effect or a clear bottleneck appears.

Ablate token versus semantic masking; fixed versus adaptive steps; constraints
on/off; ordinary versus semantic block boundaries; literal copy on/off; and
factorized versus fused features. Use paired tasks at several dependency depths
and exact-length versus predicted-length settings. Gold output length or tree
shape is a separately labeled oracle condition. Evaluate unseen lengths and
compositions within the supported grammar, not just new random samples of
training templates.

To measure a diffusion-specific interaction at a fixed quality target, compare:

```text
I = (T_diffusion,English / T_diffusion,M)
    / (T_AR,English / T_AR,M)
```

`I>1` suggests an additional diffusion interaction only when uncertainty,
capacity, training exposure, sampling budgets, and serving conditions support
the comparison. Faster M in both families still establishes useful general
representation efficiency.

## 7. Measurement and decision gates

Primary outcomes: exact whole-AST success, polarity/scope/role error rates,
literal codepoint fidelity, and **correct tasks/second**. Also report p50/p95
end-to-end latency, cold/warm legend/cache conditions, time to usable output,
forward passes, active/padded positions per pass, block counts, memory, energy
if instrumented, and codec/constraint/translation/retry costs. Tokens/second is
secondary because different tokens carry different amounts of meaning.

Report token-weighted and per-record results, by tree size, dependency depth,
label length, and Unicode category. Separate syntax validity from semantic
accuracy. Use paired confidence intervals over independently generated task
families; repeated seeds on one template do not create independent semantic
evidence. Perplexities with different token units are not directly comparable;
compare exact semantic tasks and, where tractable, likelihood per semantic
record/field under the same distribution.

Proposed preregistration gates, subject to task requirements:

1. **Transport:** deterministic codecs remain exact; diagnose and resolve model
   token normalization/reserved markers before claiming lossless transport.
2. **Comprehension:** held-out whole-AST accuracy is within a predeclared margin
   (for example 1 percentage point), with separate limits on polarity and scope
   errors. Size the suite for the margin; the current smoke suite is insufficient.
3. **Speed:** at that quality threshold, an example meaningful target is at
   least 10% lower end-to-end latency with paired uncertainty excluding no
   improvement, without worse p95 or uncounted retries.
4. **Training value:** show equal-exposure and equal-compute learning curves;
   require repeatability and a benefit on unseen compositions before scaling.
5. **Diffusion interaction:** make the AR comparison before attributing the
   improvement specifically to diffusion.

The best near-term program is offline accounting and oracle analysis, followed
by a small matched training experiment. The strongest longer-term hypothesis
is **a denoiser trained on compact, structured semantic units with exact literal
handling**, rather than assuming that a denser prompt automatically accelerates
an existing commercial model.
