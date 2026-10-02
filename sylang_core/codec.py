"""Strict, deterministic serializers for the proposed core-v0.1 tree.

No tokenizer, model, network call, source execution, or Unicode normalization is
involved. See docs/experimental-grammar.md for the complete supported subset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "core-v0.1"
FORMATS = ("english", "json", "dsl", "prime", "m")
RELATIONS = ("see", "help", "contain")
POLARITIES = ("positive", "negative")
TENSES = ("past", "present", "future")
ASPECTS = ("simple", "progressive", "completed")
EVIDENCES = ("unspecified", "direct", "reported", "inferred")
MAX_DEPTH = 8  # The statement root has depth one.
MAX_NODES = 127
MAX_LITERAL_SCALARS = 256
MAX_TEXT_BYTES = 262_144

_PRED_FIELDS = (
    "subject", "relation", "object", "polarity", "tense", "aspect", "evidence"
)
_ENUMS = {
    "relation": RELATIONS,
    "polarity": POLARITIES,
    "tense": TENSES,
    "aspect": ASPECTS,
    "evidence": EVIDENCES,
}
# Codes have meaning only in their documented field positions. Their spelling
# makes no promise about any model's tokenizer or learned representations.
_CODES = {
    "relation": {"see": "s", "help": "h", "contain": "c"},
    "polarity": {"positive": "+", "negative": "-"},
    "tense": {"past": "p", "present": "n", "future": "f"},
    "aspect": {"simple": "s", "progressive": "g", "completed": "c"},
    "evidence": {"unspecified": "u", "direct": "d", "reported": "r", "inferred": "i"},
}
_DECODE_CODES = {key: {v: k for k, v in table.items()} for key, table in _CODES.items()}
_EVIDENCE_ENGLISH = {
    "unspecified": "Without specified evidence,",
    "direct": "By direct observation,",
    "reported": "By report,",
    "inferred": "By inference,",
}
_INFLECTIONS = {
    "see": ("sees", "saw", "seeing", "seen"),
    "help": ("helps", "helped", "helping", "helped"),
    "contain": ("contains", "contained", "containing", "contained"),
}


class SylangError(ValueError):
    """Base class for invalid input to this experimental core."""


class ValidationError(SylangError):
    """The tree violates the closed schema or its resource bounds."""


class ParseError(SylangError):
    """The source is malformed, out of bounds, or outside the grammar."""


def _keys(value: Any, expected: set[str], path: str) -> None:
    if type(value) is not dict:
        raise ValidationError(f"{path}: expected an object")
    if set(value) != expected:
        raise ValidationError(f"{path}: expected exactly fields {', '.join(sorted(expected))}")


def _literal(value: Any, path: str) -> None:
    if type(value) is not str or not 1 <= len(value) <= MAX_LITERAL_SCALARS:
        raise ValidationError(f"{path}: expected 1..{MAX_LITERAL_SCALARS} Unicode scalars")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValidationError(f"{path}: unpaired surrogate code points are forbidden")


def validate(ast: Any) -> None:
    """Validate the exact AST shape without mutation, defaults, or coercions.

    Object key order is immaterial. All node fields are required. Entity labels
    are opaque Unicode strings; no NFC/NFKC normalization or case folding occurs.
    """
    _keys(ast, {"version", "statement"}, "document")
    if type(ast["version"]) is not str or ast["version"] != SCHEMA_VERSION:
        raise ValidationError(f"document.version: expected {SCHEMA_VERSION!r}")
    count = 0

    def visit(node: Any, depth: int, path: str) -> None:
        nonlocal count
        count += 1
        if depth > MAX_DEPTH or count > MAX_NODES:
            raise ValidationError(f"{path}: maximum depth or node count exceeded")
        if type(node) is not dict or type(node.get("kind")) is not str:
            raise ValidationError(f"{path}: expected a node with a string kind")
        if node["kind"] == "pred":
            _keys(node, {"kind", *_PRED_FIELDS}, path)
            _literal(node["subject"], f"{path}.subject")
            _literal(node["object"], f"{path}.object")
            for name, allowed in _ENUMS.items():
                if type(node[name]) is not str or node[name] not in allowed:
                    raise ValidationError(f"{path}.{name}: expected one of {allowed}")
        elif node["kind"] == "if":
            _keys(node, {"kind", "condition", "consequence"}, path)
            visit(node["condition"], depth + 1, f"{path}.condition")
            visit(node["consequence"], depth + 1, f"{path}.consequence")
        else:
            raise ValidationError(f"{path}.kind: unknown node kind")

    visit(ast["statement"], 1, "document.statement")


def _ordered_node(node: dict[str, Any]) -> dict[str, Any]:
    if node["kind"] == "pred":
        return {"kind": "pred", **{name: node[name] for name in _PRED_FIELDS}}
    return {
        "kind": "if",
        "condition": _ordered_node(node["condition"]),
        "consequence": _ordered_node(node["consequence"]),
    }


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _verb(relation: str, polarity: str, tense: str, aspect: str) -> str:
    present, past, progressive, participle = _INFLECTIONS[relation]
    neg = polarity == "negative"
    if aspect == "simple":
        if tense == "present":
            return "does not " + relation if neg else present
        if tense == "past":
            return "did not " + relation if neg else past
        return "will " + ("not " if neg else "") + relation
    if aspect == "progressive":
        auxiliary = {"present": "is", "past": "was", "future": "will"}[tense]
        return auxiliary + (" not" if neg else "") + (" be " if tense == "future" else " ") + progressive
    auxiliary = {"present": "has", "past": "had", "future": "will"}[tense]
    return auxiliary + (" not" if neg else "") + (" have " if tense == "future" else " ") + participle


_VERB_LOOKUP = {
    _verb(relation, polarity, tense, aspect): (relation, polarity, tense, aspect)
    for relation in RELATIONS
    for polarity in POLARITIES
    for tense in TENSES
    for aspect in ASPECTS
}


def _render(node: dict[str, Any], format: str) -> str:
    if node["kind"] == "if":
        condition = _render(node["condition"], format)
        consequence = _render(node["consequence"], format)
        if format == "english":
            return f"If [{condition}] then [{consequence}]."
        if format == "dsl":
            return f"if(condition={condition},consequence={consequence})"
        if format == "prime":
            return f"if{{condition={condition};consequence={consequence}}}"
        return f"i({condition},{consequence})"
    if format == "english":
        verb = _verb(node["relation"], node["polarity"], node["tense"], node["aspect"])
        return f"{_EVIDENCE_ENGLISH[node['evidence']]} {_quote(node['subject'])} {verb} {_quote(node['object'])}."
    values = {
        name: _quote(node[name]) if name in ("subject", "object") else node[name]
        for name in _PRED_FIELDS
    }
    if format == "dsl":
        return "pred(" + ",".join(f"{name}={values[name]}" for name in _PRED_FIELDS) + ")"
    if format == "prime":
        return "pred{" + ";".join(f"{name}={values[name]}" for name in _PRED_FIELDS) + "}"
    return "p(" + ",".join(
        values[name] if name in ("subject", "object") else _CODES[name][node[name]]
        for name in _PRED_FIELDS
    ) + ")"


def _check_format(format: str) -> None:
    if type(format) is not str or format not in FORMATS:
        raise ValidationError(f"format: expected one of {FORMATS}")


def _check_text(text: str) -> None:
    if type(text) is not str:
        raise ParseError("source must be a string")
    if len(text) > MAX_TEXT_BYTES:
        raise ParseError("source exceeds the byte limit")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ParseError("source contains an unpaired surrogate") from error
    if size > MAX_TEXT_BYTES:
        raise ParseError("source exceeds the UTF-8 byte limit")


def encode(ast: Any, format: str) -> str:
    """Return the deterministic canonical spelling for a validated tree."""
    _check_format(format)
    validate(ast)
    ordered = {"version": SCHEMA_VERSION, "statement": _ordered_node(ast["statement"])}
    if format == "json":
        result = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    else:
        body = _render(ordered["statement"], format)
        result = {
            "english": lambda: f"Sylang {SCHEMA_VERSION}: {body}",
            "dsl": lambda: f"core({_quote(SCHEMA_VERSION)},{body})",
            "prime": lambda: f"Prime0.1 {body}",
            "m": lambda: f"M0.1:{body}",
        }[format]()
    _check_text(result)
    return result


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")
_PUNCTUATION = frozenset("(){}[],:;=+-.")
_STRING_DECODER = json.JSONDecoder()


def _tokens(text: str) -> list[_Token]:
    tokens = []
    position = 0
    while position < len(text):
        char = text[position]
        if char in " \t\r\n":
            position += 1
            continue
        if char == '"':
            try:
                value, end = _STRING_DECODER.raw_decode(text, position)
            except (json.JSONDecodeError, RecursionError) as error:
                raise ParseError(f"invalid quoted string at character {position}") from error
            tokens.append(_Token("string", value, position))
            position = end
            continue
        match = _WORD.match(text, position)
        if match:
            tokens.append(_Token("word", match.group(), position))
            position = match.end()
            continue
        if char in _PUNCTUATION:
            tokens.append(_Token("punctuation", char, position))
            position += 1
            continue
        raise ParseError(f"unexpected character at {position}")
    return tokens


class _Reader:
    def __init__(self, text: str):
        self.tokens = _tokens(text)
        self.index = 0
        self.nodes = 0

    def error(self, message: str) -> ParseError:
        location = self.tokens[self.index].position if self.index < len(self.tokens) else "end"
        return ParseError(f"{message} at character {location}")

    def expect(self, value: str) -> None:
        if self.index >= len(self.tokens):
            raise self.error(f"expected {value!r}")
        token = self.tokens[self.index]
        if token.kind == "string" or token.value != value:
            raise self.error(f"expected {value!r}")
        self.index += 1

    def atom(self, allowed: tuple[str, ...] | dict[str, str]) -> str:
        if self.index >= len(self.tokens):
            raise self.error("expected a known field value")
        token = self.tokens[self.index]
        if token.kind == "string" or token.value not in allowed:
            raise self.error("unknown field value or opcode")
        self.index += 1
        return token.value

    def string(self) -> str:
        if self.index >= len(self.tokens) or self.tokens[self.index].kind != "string":
            raise self.error("expected a JSON-quoted string")
        result = self.tokens[self.index].value
        self.index += 1
        return result

    def phrase(self, phrases: dict[str, Any]) -> Any:
        for phrase in sorted(phrases, key=len, reverse=True):
            expected = _tokens(phrase)
            actual = self.tokens[self.index:self.index + len(expected)]
            if len(actual) == len(expected) and all(
                a.kind == b.kind and a.value == b.value for a, b in zip(actual, expected)
            ):
                self.index += len(expected)
                return phrases[phrase]
        raise self.error("unrecognized controlled-English phrase")

    def node(self, format: str, depth: int = 1) -> dict[str, Any]:
        self.nodes += 1
        if depth > MAX_DEPTH or self.nodes > MAX_NODES:
            raise self.error("maximum depth or node count exceeded")
        if format == "english":
            return self.english_node(depth)
        if format == "m":
            kind = self.atom(("p", "i"))
            self.expect("(")
            if kind == "i":
                condition = self.node(format, depth + 1)
                self.expect(",")
                consequence = self.node(format, depth + 1)
                self.expect(")")
                return {"kind": "if", "condition": condition, "consequence": consequence}
            values: dict[str, Any] = {"kind": "pred"}
            for index, name in enumerate(_PRED_FIELDS):
                if index:
                    self.expect(",")
                values[name] = self.string() if name in ("subject", "object") else _DECODE_CODES[name][self.atom(_DECODE_CODES[name])]
            self.expect(")")
            return values
        kind = self.atom(("pred", "if"))
        opening, separator, closing = ("(", ",", ")") if format == "dsl" else ("{", ";", "}")
        self.expect(opening)
        if kind == "if":
            self.expect("condition")
            self.expect("=")
            condition = self.node(format, depth + 1)
            self.expect(separator)
            self.expect("consequence")
            self.expect("=")
            consequence = self.node(format, depth + 1)
            self.expect(closing)
            return {"kind": "if", "condition": condition, "consequence": consequence}
        values = {"kind": "pred"}
        for index, name in enumerate(_PRED_FIELDS):
            if index:
                self.expect(separator)
            self.expect(name)
            self.expect("=")
            values[name] = self.string() if name in ("subject", "object") else self.atom(_ENUMS[name])
        self.expect(closing)
        return values

    def english_node(self, depth: int) -> dict[str, Any]:
        if self.index < len(self.tokens) and self.tokens[self.index].kind == "word" and self.tokens[self.index].value == "If":
            self.expect("If")
            self.expect("[")
            condition = self.node("english", depth + 1)
            self.expect("]")
            self.expect("then")
            self.expect("[")
            consequence = self.node("english", depth + 1)
            self.expect("]")
            self.expect(".")
            return {"kind": "if", "condition": condition, "consequence": consequence}
        evidence = self.phrase({v: k for k, v in _EVIDENCE_ENGLISH.items()})
        subject = self.string()
        relation, polarity, tense, aspect = self.phrase(_VERB_LOOKUP)
        object_ = self.string()
        self.expect(".")
        return {
            "kind": "pred", "subject": subject, "relation": relation,
            "object": object_, "polarity": polarity, "tense": tense,
            "aspect": aspect, "evidence": evidence,
        }

    def finish(self) -> None:
        if self.index != len(self.tokens):
            raise self.error("unexpected trailing source")


def _no_numbers(value: str) -> Any:
    raise ParseError("numeric values are not supported by core-v0.1")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ParseError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def decode(text: str, format: str) -> dict[str, Any]:
    """Parse a complete document and validate it; never guess or ignore fields."""
    _check_format(format)
    _check_text(text)
    if format == "json":
        try:
            ast = json.loads(
                text, object_pairs_hook=_unique_keys, parse_int=_no_numbers,
                parse_float=_no_numbers, parse_constant=_no_numbers,
            )
        except (json.JSONDecodeError, RecursionError) as error:
            raise ParseError("invalid or excessively nested JSON") from error
    else:
        reader = _Reader(text)
        if format == "english":
            reader.expect("Sylang")
            reader.expect(SCHEMA_VERSION)
            reader.expect(":")
        elif format == "dsl":
            reader.expect("core")
            reader.expect("(")
            if reader.string() != SCHEMA_VERSION:
                raise ParseError("unknown schema version")
            reader.expect(",")
        elif format == "prime":
            reader.expect("Prime0.1")
        else:
            reader.expect("M0.1")
            reader.expect(":")
        ast = {"version": SCHEMA_VERSION, "statement": reader.node(format)}
        if format == "dsl":
            reader.expect(")")
        reader.finish()
    validate(ast)
    return {"version": SCHEMA_VERSION, "statement": _ordered_node(ast["statement"])}


def canonicalize(text: str, format: str) -> str:
    """Normalize accepted surface spelling; preserve every semantic field."""
    return encode(decode(text, format), format)


def semantic_equal(left: Any, right: Any) -> bool:
    """Compare valid trees, ignoring object key order only.

    This is syntactic semantic-tree equality, not logical equivalence or a test
    that a model or human has understood the represented assertion.
    """
    validate(left)
    validate(right)
    return left == right
