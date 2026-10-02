# Experimental core-v0.1 grammar

This is a proposed executable subset for evaluating Sylang. It is **not** an
implementation of the complete historical Prime language, a final alphabet, or
a general natural-language translator. The versioned semantic tree below is the
contract for this experiment. Prime is its readable semantic representation;
M is a bounded reversible serialization of the same tree. The human-facing L1
language remains deferred.

The Python implementation is dependency-free. Its public API is:

```python
from sylang_core import (
    SCHEMA_VERSION, FORMATS, validate, encode, decode,
    canonicalize, semantic_equal, SylangError,
)

# validate(ast) -> None, or ValidationError
# encode(ast, format) -> canonical str
# decode(text, format) -> validated dict, or ParseError/ValidationError
# canonicalize(text, format) -> encode(decode(text, format), format)
# semantic_equal(left, right) -> bool; both inputs must first be valid
# FORMATS = ("english", "json", "dsl", "prime", "m")
```

`SylangError` is the common `ValueError` subclass. Parsing consumes the whole
document. There are no guessed defaults, ignored fields, unknown-opcode escapes,
automatic version migrations, or arbitrary code execution.

## Semantic tree

Every document has exactly two fields: `version: "core-v0.1"` and `statement`.
The statement is a predicate or a conditional. This example declares every
predicate field:

```json
{
  "version": "core-v0.1",
  "statement": {
    "kind": "pred",
    "subject": "Mira",
    "relation": "see",
    "object": "lamp",
    "polarity": "positive",
    "tense": "present",
    "aspect": "simple",
    "evidence": "direct"
  }
}
```

| Predicate field | Supported values and scope |
| --- | --- |
| `kind` | Exactly `pred` |
| `subject`, `object` | Opaque, nonempty Unicode scalar strings; distinct ordered roles |
| `relation` | `see`, `help`, `contain`; three binary relations |
| `polarity` | `positive`, `negative`; applies to this predicate only |
| `tense` | `past`, `present`, `future`; explicit grammatical category |
| `aspect` | `simple`, `progressive`, `completed`; explicit grammatical category |
| `evidence` | `unspecified`, `direct`, `reported`, `inferred`; annotation on this predicate |

A conditional is exactly:

```text
{"kind":"if","condition":NODE,"consequence":NODE}
```

Its branches are ordered and their scope is explicit. Reversing the branches,
moving a negative polarity, changing an evidence tag, or changing a literal
changes the tree. There is no conditional-level negation, conjunction,
quantification, confidence number, implicit subject, cross-reference or variable
binding in this version. The implementation preserves conditional structure;
it does not evaluate truth, establish logical equivalence, or prove an
antecedent.

Entity labels are lexical values, not globally resolved entity identities. For
example, `"null"` is a valid string label, while JSON `null` is not a label.
Case, leading whitespace, combining marks and code points are significant.
`"é"` and `"e\u0301"` remain distinct. Two labels can happen to name the same
real-world entity without this implementation knowing that fact.

The machine-readable [JSON Schema](../schema/core-v0.1.schema.json) describes the
closed object structure. The executable validator additionally enforces scalar
Unicode, total node count and depth. Passing only a generic JSON Schema checker
is therefore insufficient to establish validity under the full contract.

## Bounds and literal encoding

| Limit | Value |
| --- | --- |
| Tree depth | At most 8; the statement root has depth 1 |
| Total nodes | At most 127, counting predicates and conditionals |
| Subject/object length | 1–256 Unicode scalar values each |
| Encoded input | At most 262,144 UTF-8 bytes per document |
| Numeric semantic values | None; integers, fractions, exponents and nonfinite values are rejected |

Every literal in every format uses JSON string quoting. Quotes, backslashes and
control characters are escaped. Valid Unicode scalar characters, including
supplementary-plane characters, are preserved without normalization. Raw and
escaped unpaired surrogate code points are rejected; an escaped high/low
surrogate pair representing one scalar is accepted. The 256 KiB input bound
accommodates the maximum allowed tree even when all label characters expand
to six-character JSON escapes.

ASCII spaces, tabs, carriage returns and line feeds may occur between tokens.
They do not occur inside a token unless part of a quoted literal. The canonical
encoders use the exact spacing shown below. JSON object key order and escape
spelling can vary on input; duplicate JSON keys are rejected. DSL and Prime use
the fixed field order shown here. Quoted strings cannot impersonate structural
keywords or opcodes.

## Five equivalent representations

These are all canonical encodings of the example tree above. Version markers
are mandatory, including for one-predicate documents. `Prime0.1` and `M0.1`
explicitly denote the `core-v0.1` experimental schema, not a historical release.

Controlled English:

```text
Sylang core-v0.1: By direct observation, "Mira" sees "lamp".
```

Compact JSON:

```json
{"version":"core-v0.1","statement":{"kind":"pred","subject":"Mira","relation":"see","object":"lamp","polarity":"positive","tense":"present","aspect":"simple","evidence":"direct"}}
```

Generic field-labelled DSL:

```text
core("core-v0.1",pred(subject="Mira",relation=see,object="lamp",polarity=positive,tense=present,aspect=simple,evidence=direct))
```

Prime:

```text
Prime0.1 pred{subject="Mira";relation=see;object="lamp";polarity=positive;tense=present;aspect=simple;evidence=direct}
```

M:

```text
M0.1:p("Mira",s,"lamp",+,n,s,d)
```

Prime retains readable labels so reviewers can inspect role, scope and category
changes. M replaces those fixed field labels with positions and closed opcode
tables. Neither serialization claims an opcode will occupy one model token or
have a learned meaning in an existing model. No frequency-based fusion,
dictionary induction, custom tokenizer or trained embedding is used.

The two field-labelled formats deliberately form conservative controls. Any
measured advantage over those verbose formats must also be compared with the
concise controlled English baseline and with full prompt/legend overhead. This
small grammar is not a representative natural-language benchmark.

## Syntax summary

`S` means a JSON-quoted string; `R`, `P`, `T`, `A`, `E` mean the relation,
polarity, tense, aspect and evidence enum names above. Optional token whitespace
is omitted from these productions. The JSON production is exactly the semantic
tree, with arbitrary object key order and no additional fields.

```ebnf
english-document = "Sylang core-v0.1:", english-node ;
english-node = english-evidence, S, english-verb, S, "."
             | "If", "[", english-node, "]", "then", "[", english-node, "]", "." ;

dsl-document = "core", "(", '"core-v0.1"', ",", dsl-node, ")" ;
dsl-node = "pred", "(", "subject=", S, ",relation=", R, ",object=", S,
           ",polarity=", P, ",tense=", T, ",aspect=", A, ",evidence=", E, ")"
         | "if(condition=", dsl-node, ",consequence=", dsl-node, ")" ;

prime-document = "Prime0.1", prime-node ;
prime-node = "pred{subject=", S, ";relation=", R, ";object=", S,
             ";polarity=", P, ";tense=", T, ";aspect=", A, ";evidence=", E, "}"
           | "if{condition=", prime-node, ";consequence=", prime-node, "}" ;

m-document = "M0.1:", m-node ;
m-node = "p(", S, ",", relation-code, ",", S, ",", polarity-code,
         ",", tense-code, ",", aspect-code, ",", evidence-code, ")"
       | "i(", m-node, ",", m-node, ")" ;
```

M codes are interpreted only in their fixed field positions:

| Position | Field | Codes |
| --- | --- | --- |
| 1 | Subject | JSON string |
| 2 | Relation | `s` see; `h` help; `c` contain |
| 3 | Object | JSON string |
| 4 | Polarity | `+` positive; `-` negative |
| 5 | Tense | `p` past; `n` present; `f` future |
| 6 | Aspect | `s` simple; `g` progressive; `c` completed |
| 7 | Evidence | `u` unspecified; `d` direct; `r` reported; `i` inferred |

`p(...)` is a predicate; `i(condition,consequence)` is a conditional. There are
no optional fields, overloaded arities, numeric IDs or unspecified extensions.

## Controlled English contract

Only the generated templates are recognized. This is an independently readable
control serialization, **not** the deferred L1 language or an NLP parser.
Entity labels are always quoted and treated as singular grammatical units.
Evidence prefixes are exactly:

| Evidence | Prefix |
| --- | --- |
| unspecified | `Without specified evidence,` |
| direct | `By direct observation,` |
| reported | `By report,` |
| inferred | `By inference,` |

The `see` forms define the template. `help` substitutes
`help/helps/helped/helping/helped`; `contain` substitutes
`contain/contains/contained/containing/contained`.

| Tense / aspect | Positive | Negative |
| --- | --- | --- |
| present / simple | sees | does not see |
| past / simple | saw | did not see |
| future / simple | will see | will not see |
| present / progressive | is seeing | is not seeing |
| past / progressive | was seeing | was not seeing |
| future / progressive | will be seeing | will not be seeing |
| present / completed | has seen | has not seen |
| past / completed | had seen | had not seen |
| future / completed | will have seen | will not have seen |

For example, this preserves negative polarity, future tense, completed aspect
and reported evidence:

```text
Sylang core-v0.1: By report, "A" will not have helped "B".
M0.1:p("A",h,"B",-,f,c,r)
```

The enum name `completed` is represented by perfect auxiliaries as an explicit
experimental convention. English perfect aspect does not universally entail
real-world completion. The prototype has no event calculus, temporal reference
binding, speaker model or theory of evidential truth. Those semantic questions
remain open; reversible rendering does not resolve them.

Bracketed conditionals preserve recursive scope, including antecedent negation:

```text
Sylang core-v0.1: If [By direct observation, "Mira" does not see "lamp".] then [By direct observation, "Mira" sees "lamp".].
M0.1:i(p("Mira",s,"lamp",-,n,s,d),p("Mira",s,"lamp",+,n,s,d))
```

## Running and checking the core

From the repository root:

```sh
python -m unittest discover -s tests -v
python -m sylang_core --from m --to prime --input example.m
```

The CLI accepts one UTF-8 document from a file or standard input and emits one
canonical document plus a newline. Errors go to standard error with exit status
2. The codec API itself does not add a trailing newline.

The unit suite includes independently written expected surfaces for every
format, a negative future/perfect example, scoped conditionals, all 216
predicate enum combinations, synthetic Unicode and escape cases, 150 generated
trees with seed `20261002`, every pair of format translations, and malformed or
out-of-bound input. Boundary tests exercise maximum depth, node count, literal
length and escaped document expansion. All fixtures in these tests are newly
authored and covered by the repository's MIT license.

The acceptance property is `decode(encode(ast, format), format) == ast` for this
valid tree domain. Canonicalization is a separate operation: accepted whitespace
or escape spelling may change without changing the tree. `semantic_equal`
ignores dictionary key order and nothing else; it does not collapse reordered
conditionals, Unicode normalization variants or logically equivalent formulas.

## Decisions deliberately left open

- The complete Prime vocabulary, historical alphabets, affix spelling and L1
  surface design. This subset does not reconcile them by silently inventing a
  complete language.
- Broader semantics: reference frames, identities, quantifiers, plural entities,
  argument types, units/numbers, speaker/time anchoring, temporal/aspect logic
  and evidential scope. Additions require a versioned schema and independent
  examples before introducing compact codes.
- Whether positional M codes help a particular tokenizer or model. Select
  candidates using development workloads, then freeze the choices before
  measuring them on held-out evaluation data. Character length alone is
  insufficient to assess tokenizer behavior.
- Model learnability and semantic comprehension. Exact codec round trips prove
  representation fidelity within this domain; they do not prove a model can
  understand, infer from, or reliably generate the representation. The separate
  [evaluation protocol](evaluation.md) includes an offline task/export interface
  for later comprehension checks without an inference run in this milestone.
- Whether any future tokenizer adaptation or training is justified. The next
  evidence comes from baselines, error analysis and held-out comprehension,
  before any training proposal.
