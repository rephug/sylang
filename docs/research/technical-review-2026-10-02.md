# Sylang technical review: executable semantics before model adaptation

Research cutoff: **2026-10-02**. This is an authored technical assessment, not a
validation of historical performance claims or a complete Sylang specification.
Only public primary sources are linked. No private source documents are reproduced.

Keep **Qwen3 as the baseline**, compare Qwen3.5 tokenizer assets, and build the
semantic contract before optimizing spellings or training models. Newer research
offers useful evaluation and generation techniques, but does not establish any
particular percentage saving, latency gain, or comprehension improvement for Sylang.

## Priority and decision trace

| Priority | Decision | Reason and acceptance gate |
| --- | --- | --- |
| 1 | Freeze a bounded semantic AST and executable validators. | Every supported distinction must survive each representation; reject unsupported input instead of guessing. |
| 2 | Curate fixtures before selecting compact spellings. | Include independent expected structures, minimal semantic pairs, malformed inputs, and Unicode edge cases; keep a held-out set. |
| 3 | Implement deterministic renderers and parsers. | Require semantic round trips, canonicalization stability, resource bounds, and negative tests. |
| 4 | Measure unchanged Qwen3 and Qwen3.5 tokenizers. | Compare equivalent content, pin assets and software, and expose regressions as well as savings. |
| 5 | Prepare and review a comprehension evaluation. | Test interpretation and generation separately; establish an accuracy baseline before requesting inference resources. |
| 6 | Consider constrained generation, then adaptation only if needed. | A measured failure or cost bottleneck must justify the added model/runtime/training complexity. |

The selected experimental contract is **core-v0.1**, isolated from broader language
design proposals: predicates `see`, `help`, and `contain`; explicit polarity, tense,
aspect, and evidence; and bounded `if` structures. English, JSON, a conventional DSL,
readable Prime, and compact M encode the same AST. Here English means a controlled
template language, not general natural-language understanding. Prime and M names
identify experimental representations, not complete implementations of a larger spec.

M is a reversible serialization of the supported AST. It is not an optimizer allowed
to delete distinctions. A human-facing language, arbitrary prose translation,
unbounded reasoning, and a new tokenizer are outside this milestone. Implementation
status and measured results belong in the accompanying test/evaluation reports;
this memo makes no claim that a proposed experiment has already run.

## Evidence that changes the near-term plan

**Compression alone is an incomplete objective.** Lotz et al., ACL 2025, compare
tokenizers across model scales and find that text compression is an incomplete
predictor of downstream performance, especially in multilingual settings. Their
results support evaluating several tokenizer properties and downstream tasks;
they do not transfer a numeric performance prediction to Sylang.
[Beyond Text Compression](https://aclanthology.org/2025.acl-long.1546/).

**Grammar constraints are useful after the grammar exists.** Published logical
parsing experiments find benefits from constrained decoding. XGrammar-2 adds
dynamic structure dispatch, compilation reuse, and token-mask optimizations; its
authors report low serving overhead in their evaluated systems. A context-free
grammar guarantees membership in its language, not that an output expresses the
intended fact. Keep semantic validation and answer checking outside decoding.
[Logical parsing study](https://aclanthology.org/2025.acl-industry.34/),
[XGrammar-2, v4](https://arxiv.org/abs/2601.04426v4),
[grammar/tokenizer alignment](https://proceedings.mlr.press/v267/park25l.html).

**Tokenizer adaptation changes the model interface.** TokAlign aligns vocabularies,
rearranges embeddings, and progressively fine-tunes; its reported recovery involves
training. Continued BPE training and leaf-based pruning address inefficient or
unreachable vocabulary additions. These are published techniques worth revisiting
if unchanged tokenizers fail a measured need, not cost-free replacements for them.
Do not count a newly added symbol as one token and infer that an existing model
already knows its meaning.
[TokAlign](https://arxiv.org/abs/2506.03523v1),
[Teaching Old Tokenizers New Words, v2](https://arxiv.org/abs/2512.03989v2).

An August 2026 writing-system adaptation preprint further separates BPE merge
reachability from model quality. Its Ukrainian improvements coexist with degradation
on an evaluated neighboring Cyrillic language aggregate. It explicitly leaves
embedding initialization, continued pretraining, and downstream quality to later
experiments. Preserved token IDs and lower token counts do not prove preserved
behavior; exact-segmentation and multilingual regression audits would be necessary.
[Writing-System-Level Tokenizer Adaptation](https://arxiv.org/abs/2608.00582v1).

**Multilingual evaluation needs group-level results.** Parity-aware BPE, ACL 2026,
changes the merge objective to improve cross-language equality while measuring the
compression tradeoff. Language-variation research also shows task-dependent effects.
For Sylang, preservation of a few Unicode literals demonstrates codec behavior only.
Claims about multilingual efficiency or understanding need meaning-aligned language
fixtures, independent translation review, and per-language and worst-group results.
[Parity-aware BPE, v3](https://arxiv.org/abs/2508.04796v3),
[Tokenization is Sensitive to Language Variation](https://aclanthology.org/2025.findings-acl.572/).

## Techniques to retain on the watchlist

| Technique | What the primary evidence supports | Relevance and limitation |
| --- | --- | --- |
| Byte Latent Transformer (BLT) | Learned byte processing with entropy-based patches; author experiments compare training/inference FLOPs and robustness. | A different trained architecture. The paper explicitly cautions that wall-clock implementation efficiency may lag token-based systems. No drop-in Qwen tokenizer or Sylang speed prediction. |
| H-Net | Jointly learned dynamic chunking and hierarchical sequence modeling, with compute/data-controlled comparisons. | A model architecture and training method, not a reversible interchange format. The paper does not directly benchmark against BLT. |
| ByT5 | Byte-to-byte modeling offers an established tokenizer-free comparison. | Raw bytes avoid fixed subword vocabulary boundaries but change sequence lengths and model computation; counting bytes does not reproduce a byte model's cost. |
| Coconut | Continuous hidden-state reasoning, with results on selected reasoning tasks. | Hidden states do not supply an externally readable, portable, exact semantic codec. Retain an explicit AST. |
| WeDLM | Diffusion-style parallel decoding with standard causal attention and prefix caching; authors compare with optimized autoregressive serving. | A different model/runtime experiment, orthogonal to representation compression. Speedups depend on workload and deployment. |
| LLMLingua-2 | Learned extractive prompt compression with measured downstream and latency tradeoffs. | A possible future lossy comparator; extraction cannot establish exact invertibility or replace Prime/M round-trip tests. |

Sources: [BLT, v1](https://arxiv.org/abs/2412.09871v1),
[BLT limitations](https://arxiv.org/html/2412.09871#S9),
[H-Net, v2](https://arxiv.org/abs/2507.07955v2),
[ByT5](https://arxiv.org/abs/2105.13626),
[Coconut, v4](https://arxiv.org/abs/2412.06769v4),
[WeDLM, v1](https://arxiv.org/abs/2512.22737v1),
[LLMLingua-2, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.57/).

WeDLM has a public paper, [official implementation](https://github.com/tencent/WeDLM),
and [released model card](https://huggingface.co/tencent/WeDLM-8B-Instruct).
This review verifies the December 28, 2025 arXiv release, not a peer-reviewed venue
acceptance. Model-card speed claims are vendor/author reports, not independent
Sylang measurements. No WeDLM runtime, weights, or training are adopted here.

## Tokenizer-only baseline

| Candidate | Role | License | Listed `tokenizer.json` size |
| --- | --- | --- | --- |
| [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) | Primary baseline | Apache-2.0 | About 7.03 MB |
| [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68) | Newer within-family comparison | Apache-2.0 | About 12.8 MB |
| [swiss-ai/Apertus-8B-2509](https://huggingface.co/swiss-ai/Apertus-8B-2509/tree/main) | Optional broader multilingual comparison | Apache-2.0 | About 17.1 MB |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/tree/main) | Optional different tokenizer family | Apache-2.0 | About 27.9 MB |

The first two are selected; optional candidates are not download or inference
commitments. Sizes describe individual assets, not model weights, and are rounded
repository listings. Base and instruction-tuned checkpoints must be identified
precisely; similarly named tokenizer files are not assumed byte-identical.
Padded model embedding dimensions are not tokenizer vocabulary measurements.

Review license and size before fetching explicit allowlisted assets. Pin repository
revision and SHA-256, record actual byte sizes and software versions, and load the
local tokenizer JSON with the reviewed Hugging Face `tokenizers` library. Do not
execute repository code, enable `trust_remote_code`, fetch weights, or clone a
whole model repository. Record vocabulary size, normalizer, pre-tokenizer, added
tokens, and special-token settings. Disable truncation and padding for counts.
Check exact text encode/decode preservation with special-token skipping disabled.

## Measurement contract and full-stack accounting

1. Derive all five representations from one independently validated AST. Require
   exact AST round trips and canonical re-encoding before counting a case as valid.
2. Compare each representation with English, JSON, and DSL under the **same**
   tokenizer and settings. Publish per-fixture results, totals, median, and worst
   cases. Never use different tokenizers for numerator and denominator of a
   claimed representation saving; keep cross-tokenizer observations separate.
3. Report payload counts and full instruction/legend/payload counts separately.
   Tokenize complete prompts: summing independently tokenized fragments can miss
   merges at their boundaries. Charge dictionary/setup overhead and identify
   whether it is repeated per request, shared, or cached.
4. Report UTF-8 bytes and codec CPU time separately from tokens. Compressed archive
   bytes are transport/storage costs, not tokens presented to a language model.
5. Keep semantic round trips, parse validity, tokenizer text preservation, model
   comprehension accuracy, and end-to-end latency as separate outcomes.
6. For later inference, record model/revision, prompt template, output budget,
   decoding settings/seeds, hardware, repetitions, and cold/warm cache conditions.
   Include preparation, encoding, prefill, decoding, validation, retries, and
   recovery costs; report tail latency and failure rate, not just a mean.
7. Any monetary estimate needs explicit current input/output/cache prices and
   workloads. It must include retries and output length. Counts alone establish
   neither money saved nor faster inference.

Prefix caching is a useful independent baseline: shared prefixes can avoid repeated
prefill computation, but do not accelerate output decoding by themselves. A stable
representation legend can therefore have different cold and warm costs. This is a
future measurement requirement, not a server feature implemented by the scaffold.
[Official vLLM explanation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).

## Comprehension gate, hypotheses, and unresolved decisions

Prepare an offline request/expected-answer interface for role identification,
polarity, tense/aspect/evidence, and conditional scope. Include minimal pairs and
held-out compositions. Evaluate reading and generating each representation
separately under the same model, with zero-shot and fixed few-shot conditions.
Charge the format-teaching prompt. Review semantic judgments independently of the
encoder; self-consistent parser/renderer bugs can otherwise pass round trips.

Before training, freeze held-out tests and define acceptable accuracy/error bounds.
If a measured comprehension gap remains, validated AST/representation pairs could
support supervised adaptation using an unchanged tokenizer and LoRA/QLoRA. QLoRA
reduces training memory; it does not establish correctness, eliminate training cost,
or make a tiny synthetic test set sufficient. Distillation would require a verified
teacher, provenance, validation, and leakage controls. No training is authorized by
this research recommendation. [QLoRA](https://arxiv.org/abs/2305.14314v1).

Hypotheses to test, not conclusions: M may reduce repeated structural overhead;
Prime may require fewer teaching examples; common ASCII spellings may be easier
for existing tokenizers than exotic glyphs. Optimize only on a development split
and retest on frozen fixtures. No savings target is an acceptance result.

Explicit decisions still needed before expanding the core: reference/coreference
semantics; modal and evidential interpretation; tense/aspect composition; conditional
scope and nesting limits; quantity/unit/number representation; Unicode normalization
versus exact literal preservation; error recovery; version migration; and domain
coverage. Stable normalization must not silently erase a meaningful distinction.
Any extension needs new positive, negative, and comprehension cases first.

## Evidence dates and limits

Version dates below are no later than the research cutoff; publication status is
only stated where verified from the primary paper or proceedings page. Results
remain scoped to the authors' experimental settings and are not reproduced here.

| Source | Version/publication used |
| --- | --- |
| Beyond Text Compression; language variation; logical parsing | ACL 2025 proceedings linked above |
| Grammar/tokenizer alignment | ICML 2025 / PMLR volume 267 |
| TokAlign | arXiv v1, 2025-06-04; ACL 2025 |
| Continued BPE adaptation | arXiv v2, 2026-03-23; Findings EACL 2026 |
| Writing-system adaptation | arXiv v1, 2026-08-01; workshop poster acceptance reported, non-archival |
| Parity-aware BPE | arXiv v3, 2026-07-02; ACL 2026 |
| XGrammar-2 | arXiv v4, 2026-08-05; ACM CAIS 2026 |
| BLT; H-Net | arXiv v1, 2024-12-13; arXiv v2, 2025-07-15, respectively |
| Coconut | arXiv v4, 2026-08-23; accepted COLM 2025 |
| WeDLM | arXiv v1, 2025-12-28; venue acceptance not verified |
| ByT5; QLoRA; LLMLingua-2 | 2021 original preprint; 2023 original preprint; Findings ACL 2024 |

Repository/model-card links and live documentation can change after review; pinned
asset manifests and generated reports are the reproducibility record. This is a
targeted technical review, not an exhaustive literature survey. Public results
cannot establish Sylang's general-language coverage, learnability, or accuracy.
The next concrete review is the core contract, fixture independence, deterministic
test results, and paired tokenizer measurements, followed by a costed comprehension
evaluation proposal. No paid inference, model training, or serving job is implied.
