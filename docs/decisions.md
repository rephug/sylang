# Experimental baseline decisions

Status: proposed implementation baseline, 2026-10-02. This is a small executable
research milestone, not ratification of a complete language specification.

| Decision | Rationale | Revisit when |
|---|---|---|
| Use a versioned `core-v0.1` semantic record. | Historical proposals contain incompatible rules and incomplete tables; an explicit small contract is testable. | A reviewed grammar extension has new gold and rejection fixtures. |
| Keep readable Prime as the semantic inspection layer and deterministic M as its compact encoding. | Conversion must preserve every supported field and scope without a learned translator. | Never relax exact reversibility within a version. |
| Defer L1, open-ended natural-language parsing, quantification, quantities/units, modality, causatives and discourse reference. | Silently approximating these would fabricate semantic fidelity. | Each construct has defined semantics and independent tests. |
| Use opaque quoted Unicode literals, preserving code points. | Names, code and foreign text cannot safely be lowercased or stripped of accents. | Normalization is made a separate opt-in, versioned semantic policy. |
| Keep a closed relation and feature inventory; reject unsupported values. | Unknown symbols must not silently change meaning. | An extension includes unambiguous mappings in every format. |
| Use fixed bounded opcodes in M; no adaptive fusion table. | There is no frequency corpus or held-out evidence justifying fusion. M symbols are not claimed to be individual model tokens. | A frozen, licensed corpus permits a held-out ablation with setup cost included. |
| Compare unchanged Qwen3 first, then Qwen3.5. | Native tokenizer assets provide a cheap baseline without embedding changes or model code. | Other reviewed tokenizer-only assets add a useful, separately reported comparison. |
| Evaluate tooling before training. | A codec bug or unsupported savings premise must be found before model adaptation. | Frozen fidelity tests and comprehension experiments identify a specific gap worth training. |
| Separate codec fidelity, token counts, model comprehension and end-to-end economics. | None is a substitute for another. | The project has independently measured all four. |

## Evidence boundaries

The older repository documents and illustrations preserve research ideas. Their
claims about 45–60% savings, learning time, perfect morphological alignment,
neuron monosemanticity, accuracy and latency are not validated by the recovered
repository. Historical examples are not gold fixtures for this implementation.
This milestone authors new fixtures under the repository's MIT license.

The current implementation specifies one trial serialization. Its label does not
mean compatibility with historical Prime v2.1 or M v0.1. Structural syntax and
literal payloads have separate rules; there is no inherited claim of a universal
21-character alphabet. Legacy morphology, full lexicon governance and conversion
tables remain unresolved.

Private design evidence was reviewed for technical constraints. Original private
documents, conversation transcripts and private-source links are not included in
the public implementation or fixtures. The public research memo cites public
primary sources only.

## Gates for the next review

1. Freeze the subset and gold fixtures after reviewing field meanings, condition
   scope, unknown-value rejection and codepoint preservation. Require all exact
   round trips and rejection tests to pass.
2. Inspect per-fixture measurements under each unchanged tokenizer. Include the
   shared teaching material and all separators, headers and literals. Compare M
   against the strongest compact baseline, not only English.
3. Review an offline comprehension protocol before authorizing inference: matched
   semantic questions, fixed examples, held-out compositions and output checks.
   A generated prompt or answer key is not an evaluated model result.
4. If inference is later authorized, measure accuracy, failures and retries with
   the same model and budget across formats. Report cold/warm cache conditions,
   prompt preparation, prefill/decode time, output tokens and translation costs.
5. Only propose an adaptation experiment after those results. Vocabulary changes,
   LoRA, distillation, tokenizer-free models and diffusion runtimes require their
   own controlled comparisons and separate execution approval.

## Accounting to use later

For a deployment, compute total cost from its actual input, cached-input and output
token prices, plus translation/validation/retry work. Compare total costs between
the same task and quality target. A payload-token ratio does not measure dollars
or latency, and a CPU codec timing does not measure LLM inference.

Report representation ratios within each tokenizer. Do not divide English tokens
from one tokenizer by M tokens from another and call that language compression.
Do not count a reversible serialized syntax as proof that a model understands it.
