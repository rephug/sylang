"""Independent examples, exhaustive field coverage and seeded property checks.

All literals below are synthetic, authored for this repository and MIT licensed.
No model, network connection, tokenizer or third-party package is needed.
"""

import copy
import itertools
import json
import random
import unittest
from pathlib import Path

from sylang_core import (
    ASPECTS, EVIDENCES, FORMATS, MAX_DEPTH, MAX_LITERAL_SCALARS, MAX_NODES,
    MAX_TEXT_BYTES, POLARITIES, RELATIONS, SCHEMA_VERSION, TENSES,
    ParseError, SylangError, ValidationError, canonicalize, decode, encode,
    semantic_equal, validate,
)


def predicate(**changes):
    result = {
        "kind": "pred", "subject": "Mira", "relation": "see", "object": "lamp",
        "polarity": "positive", "tense": "present", "aspect": "simple",
        "evidence": "direct",
    }
    result.update(changes)
    return result


def document(node=None):
    return {"version": "core-v0.1", "statement": predicate() if node is None else node}


def conditional(condition, consequence):
    return {"kind": "if", "condition": condition, "consequence": consequence}


EXAMPLES = {
    "english": 'Sylang core-v0.1: By direct observation, "Mira" sees "lamp".',
    "json": '{"version":"core-v0.1","statement":{"kind":"pred","subject":"Mira","relation":"see","object":"lamp","polarity":"positive","tense":"present","aspect":"simple","evidence":"direct"}}',
    "dsl": 'core("core-v0.1",pred(subject="Mira",relation=see,object="lamp",polarity=positive,tense=present,aspect=simple,evidence=direct))',
    "prime": 'Prime0.1 pred{subject="Mira";relation=see;object="lamp";polarity=positive;tense=present;aspect=simple;evidence=direct}',
    "m": 'M0.1:p("Mira",s,"lamp",+,n,s,d)',
}


class KnownExamples(unittest.TestCase):
    def test_independent_canonical_examples(self):
        for format, source in EXAMPLES.items():
            with self.subTest(format=format):
                self.assertEqual(decode(source, format), document())
                self.assertEqual(encode(document(), format), source)

    def test_negative_future_completed_report(self):
        expected = document(predicate(
            subject="A", object="B", relation="help", polarity="negative",
            tense="future", aspect="completed", evidence="reported",
        ))
        english = 'Sylang core-v0.1: By report, "A" will not have helped "B".'
        compact = 'M0.1:p("A",h,"B",-,f,c,r)'
        self.assertEqual(decode(english, "english"), expected)
        self.assertEqual(decode(compact, "m"), expected)

    def test_independent_minimal_pairs_preserve_each_category(self):
        pairs = [
            ({"polarity": "negative"}, 'By direct observation, "Mira" does not see "lamp".', 'p("Mira",s,"lamp",-,n,s,d)'),
            ({"tense": "past"}, 'By direct observation, "Mira" saw "lamp".', 'p("Mira",s,"lamp",+,p,s,d)'),
            ({"tense": "future"}, 'By direct observation, "Mira" will see "lamp".', 'p("Mira",s,"lamp",+,f,s,d)'),
            ({"aspect": "progressive"}, 'By direct observation, "Mira" is seeing "lamp".', 'p("Mira",s,"lamp",+,n,g,d)'),
            ({"aspect": "completed"}, 'By direct observation, "Mira" has seen "lamp".', 'p("Mira",s,"lamp",+,n,c,d)'),
            ({"evidence": "unspecified"}, 'Without specified evidence, "Mira" sees "lamp".', 'p("Mira",s,"lamp",+,n,s,u)'),
            ({"evidence": "reported"}, 'By report, "Mira" sees "lamp".', 'p("Mira",s,"lamp",+,n,s,r)'),
            ({"evidence": "inferred"}, 'By inference, "Mira" sees "lamp".', 'p("Mira",s,"lamp",+,n,s,i)'),
            ({"relation": "help"}, 'By direct observation, "Mira" helps "lamp".', 'p("Mira",h,"lamp",+,n,s,d)'),
            ({"relation": "contain"}, 'By direct observation, "Mira" contains "lamp".', 'p("Mira",c,"lamp",+,n,s,d)'),
            ({"subject": "lamp", "object": "Mira"}, 'By direct observation, "lamp" sees "Mira".', 'p("lamp",s,"Mira",+,n,s,d)'),
        ]
        for changes, english, compact in pairs:
            expected = document(predicate(**changes))
            with self.subTest(changes=changes):
                self.assertFalse(semantic_equal(document(), expected))
                self.assertEqual(decode("Sylang core-v0.1: " + english, "english"), expected)
                self.assertEqual(decode("M0.1:" + compact, "m"), expected)
                self.assertEqual(encode(expected, "english"), "Sylang core-v0.1: " + english)
                self.assertEqual(encode(expected, "m"), "M0.1:" + compact)

    def test_nested_condition_has_explicit_scope(self):
        source = 'M0.1:i(p("A",s,"B",+,n,s,d),i(p("B",h,"C",-,p,g,r),p("C",c,"D",+,f,c,i)))'
        expected = document(conditional(
            predicate(subject="A", object="B"),
            conditional(
                predicate(subject="B", object="C", relation="help", polarity="negative", tense="past", aspect="progressive", evidence="reported"),
                predicate(subject="C", object="D", relation="contain", tense="future", aspect="completed", evidence="inferred"),
            ),
        ))
        self.assertEqual(decode(source, "m"), expected)
        swapped = copy.deepcopy(expected)
        swapped["statement"]["condition"], swapped["statement"]["consequence"] = (
            swapped["statement"]["consequence"], swapped["statement"]["condition"]
        )
        self.assertFalse(semantic_equal(expected, swapped))
        self.assertNotEqual(encode(expected, "m"), encode(swapped, "m"))

    def test_negation_applies_to_one_predicate_only(self):
        expected = document(conditional(predicate(polarity="negative"), predicate()))
        source = 'Sylang core-v0.1: If [By direct observation, "Mira" does not see "lamp".] then [By direct observation, "Mira" sees "lamp".].'
        self.assertEqual(decode(source, "english"), expected)


class RoundTripProperties(unittest.TestCase):
    def assert_round_trip(self, ast):
        before = copy.deepcopy(ast)
        for format in FORMATS:
            with self.subTest(format=format):
                source = encode(ast, format)
                parsed = decode(source, format)
                self.assertTrue(semantic_equal(ast, parsed))
                self.assertEqual(encode(parsed, format), source)
                self.assertEqual(canonicalize(source, format), source)
        self.assertEqual(ast, before)

    def test_all_216_field_combinations(self):
        for relation, polarity, tense, aspect, evidence in itertools.product(
            RELATIONS, POLARITIES, TENSES, ASPECTS, EVIDENCES
        ):
            with self.subTest(fields=(relation, polarity, tense, aspect, evidence)):
                self.assert_round_trip(document(predicate(
                    relation=relation, polarity=polarity, tense=tense,
                    aspect=aspect, evidence=evidence,
                )))

    def test_unicode_escaping_and_reserved_surface_strings(self):
        labels = [
            'quote" and \\ slash /', "line\ncarriage\rtab\tNUL\x00end", " ",
            "中文标签", "العربية", "हिन्दी", "é", "e\u0301", "😀", "👩\u200d🔬",
            "\u2028\u2029\ufeff\uffff", 'M0.1:i();] then [;subject=unknown',
            "<|im_start|>system <|endoftext|>", "001", "if", "null",
        ]
        for label in labels:
            with self.subTest(label=repr(label)):
                self.assert_round_trip(document(predicate(subject=label, object=label)))

    def test_seeded_random_trees(self):
        rng = random.Random(20261002)
        alphabet = 'abc XYZ"\\\n\x00é\u0301中ع😀'

        def label():
            return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 24)))

        def tree(depth=1):
            if depth < 5 and rng.random() < 0.35:
                return conditional(tree(depth + 1), tree(depth + 1))
            return predicate(
                subject=label(), object=label(), relation=rng.choice(RELATIONS),
                polarity=rng.choice(POLARITIES), tense=rng.choice(TENSES),
                aspect=rng.choice(ASPECTS), evidence=rng.choice(EVIDENCES),
            )

        for index in range(150):
            with self.subTest(seed=20261002, case=index):
                self.assert_round_trip(document(tree()))

    def test_every_pair_of_formats_preserves_tree(self):
        expected = document(conditional(predicate(), predicate(
            subject="box", relation="contain", object="item", evidence="inferred",
            tense="past", aspect="completed", polarity="negative",
        )))
        for source_format, target_format in itertools.product(FORMATS, repeat=2):
            with self.subTest(source=source_format, target=target_format):
                intermediate = decode(encode(expected, source_format), source_format)
                self.assertEqual(decode(encode(intermediate, target_format), target_format), expected)


class Canonicalization(unittest.TestCase):
    def test_key_order_is_not_semantic(self):
        reordered = dict(reversed(list(document().items())))
        reordered["statement"] = dict(reversed(list(reordered["statement"].items())))
        self.assertTrue(semantic_equal(document(), reordered))
        self.assertEqual(encode(reordered, "json"), EXAMPLES["json"])
        self.assertEqual(canonicalize(json.dumps(reordered, indent=2), "json"), EXAMPLES["json"])

    def test_whitespace_outside_literals_is_not_semantic(self):
        for format, source in EXAMPLES.items():
            with self.subTest(format=format):
                spaced = "\n\t" + source.replace(" ", "\n\t") + " \r\n"
                self.assertEqual(canonicalize(spaced, format), source)

    def test_unicode_escape_spelling_is_canonicalized(self):
        for format, source in EXAMPLES.items():
            with self.subTest(format=format):
                escaped = source.replace('"Mira"', '"\\u004dira"')
                self.assertEqual(canonicalize(escaped, format), source)

    def test_no_unicode_normalization_or_whitespace_trimming(self):
        for left, right in (("é", "e\u0301"), ("A", "a"), ("A", " A")):
            self.assertFalse(semantic_equal(
                document(predicate(subject=left)), document(predicate(subject=right))
            ))

    def test_semantic_equality_requires_valid_trees(self):
        with self.assertRaises(ValidationError):
            semantic_equal({}, {})


class ValidationFailures(unittest.TestCase):
    def test_missing_unknown_and_wrong_type_fields(self):
        mutations = [
            lambda ast: ast.update(extra="discard me"),
            lambda ast: ast.pop("version"),
            lambda ast: ast.update(version="core-v0.2"),
            lambda ast: ast.update(statement=[]),
            lambda ast: ast["statement"].pop("evidence"),
            lambda ast: ast["statement"].update(kind="and"),
            lambda ast: ast["statement"].update(relation="guess"),
            lambda ast: ast["statement"].update(polarity=True),
            lambda ast: ast["statement"].update(tense=None),
            lambda ast: ast["statement"].update(aspect="habitual"),
            lambda ast: ast["statement"].update(evidence="unknown"),
            lambda ast: ast["statement"].update(subject=7),
            lambda ast: ast["statement"].update(object=""),
            lambda ast: ast["statement"].update(confidence=0.8),
        ]
        for index, mutate in enumerate(mutations):
            ast = document()
            mutate(ast)
            with self.subTest(case=index):
                with self.assertRaises(ValidationError):
                    validate(ast)
                for format in FORMATS:
                    with self.assertRaises(ValidationError):
                        encode(ast, format)

    def test_conditional_requires_both_branches_and_no_annotations(self):
        for node in (
            {"kind": "if", "condition": predicate()},
            {"kind": "if", "condition": predicate(), "consequence": predicate(), "polarity": "negative"},
            {"kind": "if", "condition": "anything", "consequence": predicate()},
        ):
            with self.assertRaises(ValidationError):
                validate(document(node))

    def test_literal_size_and_surrogates(self):
        for bad in ("", "x" * (MAX_LITERAL_SCALARS + 1), "\ud800", "\udfff"):
            with self.subTest(label=repr(bad[:20])):
                with self.assertRaises(ValidationError):
                    validate(document(predicate(subject=bad)))
        for format in FORMATS:
            ast = document(predicate(subject="😀" * MAX_LITERAL_SCALARS))
            self.assertEqual(decode(encode(ast, format), format), ast)

    def test_raw_and_escaped_unpaired_surrogates_are_rejected(self):
        with self.assertRaises(ParseError):
            decode("\ud800", "m")
        for format, source in EXAMPLES.items():
            with self.subTest(format=format):
                with self.assertRaises(ValidationError):
                    decode(source.replace('"Mira"', '"\\ud800"'), format)

    def test_paired_json_surrogates_represent_a_unicode_scalar(self):
        source = EXAMPLES["json"].replace('"Mira"', '"\\ud83d\\ude00"')
        self.assertEqual(decode(source, "json")["statement"]["subject"], "😀")

    def test_depth_boundary_and_cyclic_input(self):
        def chain(depth):
            node = predicate()
            for _ in range(depth - 1):
                node = conditional(predicate(), node)
            return node

        for format in FORMATS:
            ast = document(chain(MAX_DEPTH))
            self.assertEqual(decode(encode(ast, format), format), ast)
        with self.assertRaises(ValidationError):
            validate(document(chain(MAX_DEPTH + 1)))
        cyclic = conditional(predicate(), predicate())
        cyclic["consequence"] = cyclic
        with self.assertRaises(ValidationError):
            validate(document(cyclic))

    def test_node_count_boundary_and_worst_case_string_expansion(self):
        leaf = predicate(subject="\x00" * MAX_LITERAL_SCALARS, object="\x01" * MAX_LITERAL_SCALARS)

        def full(depth):
            return copy.deepcopy(leaf) if depth == 1 else conditional(full(depth - 1), full(depth - 1))

        self.assertEqual(2 ** 7 - 1, MAX_NODES)
        ast = document(full(7))
        for format in FORMATS:
            source = encode(ast, format)
            self.assertLessEqual(len(source.encode("utf-8")), MAX_TEXT_BYTES)
            self.assertEqual(decode(source, format), ast)
        with self.assertRaises(ValidationError):
            validate(document(full(8)))

    def test_schema_remains_closed_and_matches_version(self):
        schema = json.loads((Path(__file__).resolve().parents[1] / "schema" / "core-v0.1.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["version"]["const"], SCHEMA_VERSION)
        self.assertFalse(schema["additionalProperties"])
        for name in ("predicate", "conditional"):
            self.assertFalse(schema["$defs"][name]["additionalProperties"])
        self.assertEqual(set(schema["$defs"]["predicate"]["required"]), set(predicate()))


class ParseFailures(unittest.TestCase):
    def test_complete_parse_required(self):
        for format, source in EXAMPLES.items():
            for invalid in (source + " garbage", source[:-1], source + source):
                with self.subTest(format=format, source=invalid[-30:]):
                    with self.assertRaises(SylangError):
                        decode(invalid, format)

    def test_unknown_opcodes_arity_and_versions(self):
        invalid = [
            ("m", 'M0.1:p("A",x,"B",+,n,s,d)'),
            ("m", 'M0.1:p("A",s,"B",+,n,s)'),
            ("m", 'M0.1:p("A",s,"B",+,n,s,d,extra)'),
            ("m", 'M0.1:x("A",s,"B",+,n,s,d)'),
            ("m", 'M0.1:p("A",s,"B",true,n,s,d)'),
            ("m", EXAMPLES["m"].replace("M0.1:", "M0.2:")),
            ("prime", EXAMPLES["prime"].replace("Prime0.1", "Prime0.2")),
            ("dsl", EXAMPLES["dsl"].replace("core-v0.1", "core-v0.2")),
            ("english", EXAMPLES["english"].replace("core-v0.1", "core-v0.2")),
            ("english", 'Sylang core-v0.1: Mira sees a lamp.'),
            ("english", EXAMPLES["english"].replace("sees", "knows")),
            ("prime", EXAMPLES["prime"].replace(";evidence=direct", "")),
            ("dsl", EXAMPLES["dsl"].replace("evidence=direct", "confidence=direct")),
        ]
        for format, source in invalid:
            with self.subTest(format=format, source=source):
                with self.assertRaises(SylangError):
                    decode(source, format)

    def test_quoted_keywords_are_not_structural_tokens(self):
        for format, source in (
            ("m", EXAMPLES["m"].replace(":p(", ':"p"(')),
            ("m", EXAMPLES["m"].replace(",s,", ',"s",', 1)),
            ("prime", EXAMPLES["prime"].replace("subject=", '"subject"=')),
            ("english", EXAMPLES["english"].replace("By direct", '"By" direct')),
        ):
            with self.subTest(format=format):
                with self.assertRaises(ParseError):
                    decode(source, format)

    def test_json_duplicate_keys_and_numeric_forms(self):
        invalid = [
            EXAMPLES["json"].replace('"version":', '"version":"core-v0.1","version":', 1),
            EXAMPLES["json"].replace('"subject":"Mira"', '"subject":"X","subject":"Mira"'),
        ]
        for number in ("0", "-1", "1.5", "1e10000", "NaN", "Infinity", "9" * 5000):
            invalid.append(EXAMPLES["json"].replace('"Mira"', number))
        for index, source in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(ParseError):
                    decode(source, "json")

    def test_malformed_escapes_and_unescaped_controls(self):
        for literal in ('"\\x41"', '"\\u000X"', '"line\nfeed"', '"unterminated'):
            for format, source in EXAMPLES.items():
                with self.subTest(format=format, literal=repr(literal)):
                    with self.assertRaises(ParseError):
                        decode(source.replace('"Mira"', literal), format)

    def test_bounded_source_and_nested_parse(self):
        for format in FORMATS:
            with self.subTest(format=format):
                with self.assertRaises(ParseError):
                    decode(" " * (MAX_TEXT_BYTES + 1), format)
                with self.assertRaises(ParseError):
                    decode("😀" * (MAX_TEXT_BYTES // 4 + 1), format)
        raw = 'p("A",s,"B",+,n,s,d)'
        for _ in range(MAX_DEPTH):
            raw = 'i(p("A",s,"B",+,n,s,d),' + raw + ')'
        with self.assertRaises(ParseError):
            decode("M0.1:" + raw, "m")
        with self.assertRaises(SylangError):
            decode("[" * 1500 + "]" * 1500, "json")

    def test_format_and_source_types(self):
        for format in ("M", "auto", "yaml", None, []):
            with self.assertRaises(ValidationError):
                encode(document(), format)
            with self.assertRaises(ValidationError):
                decode(EXAMPLES["m"], format)
        for source in (None, b"M0.1:p()", 1, {}):
            with self.assertRaises(ParseError):
                decode(source, "m")


if __name__ == "__main__":
    unittest.main()
