"""Tests for grader.py: every corruption of contract.md is rejected, correct answers pass in both formats.

    python -I experiments/pretrained_pilot/test_grader.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grader  # noqa: E402

SELECT_JSON = '[["ab12","zq9"],["x9","yy88"]]'
SELECT_BARE = 'ab12 zq9 x9 yy88'
LOOKUP_JSON = '{"relation":"help","polarity":"negative"}'
LOOKUP_BARE = 'help negative'


class GraderTestCase(unittest.TestCase):
    def ok(self, task, fmt, expected, output):
        result = grader.grade(task, fmt, expected, output)
        self.assertEqual(result, {'correct': True, 'invalid': False, 'reason': 'ok'}, (task, fmt, output))

    def wrong(self, task, fmt, expected, output):
        """A valid answer with a different value: correct False, invalid False."""
        result = grader.grade(task, fmt, expected, output)
        self.assertEqual(result, {'correct': False, 'invalid': False, 'reason': 'wrong'}, (task, fmt, output))

    def invalid(self, task, fmt, expected, output, reason=None):
        """An output that is not an answer of the task's shape: correct False, invalid True."""
        result = grader.grade(task, fmt, expected, output)
        self.assertFalse(result['correct'], (task, fmt, output))
        self.assertTrue(result['invalid'], (task, fmt, output, result))
        self.assertNotIn(result['reason'], ('ok', 'wrong'))
        if reason is not None:
            self.assertEqual(result['reason'], reason, (task, fmt, output, result))


class SelectJson(GraderTestCase):
    task, fmt, expected = 'select', 'json', SELECT_JSON

    def test_correct(self):
        self.ok(self.task, self.fmt, self.expected, self.expected)

    def test_whitespace_only_differences_accepted(self):
        for output in ('[["ab12", "zq9"], ["x9", "yy88"]]', ' [["ab12","zq9"],["x9","yy88"]]\n',
                       '[\n  ["ab12", "zq9"],\n  ["x9", "yy88"]\n]', '\t[["ab12","zq9"],\n["x9","yy88"]]  '):
            self.ok(self.task, self.fmt, self.expected, output)

    def test_dropped_pair(self):
        self.wrong(self.task, self.fmt, self.expected, '[["ab12","zq9"]]')
        self.wrong(self.task, self.fmt, self.expected, '[["x9","yy88"]]')

    def test_swapped_identifiers(self):
        self.wrong(self.task, self.fmt, self.expected, '[["zq9","ab12"],["x9","yy88"]]')
        self.wrong(self.task, self.fmt, self.expected, '[["ab12","yy88"],["x9","zq9"]]')

    def test_invented_pair(self):
        self.wrong(self.task, self.fmt, self.expected, '[["ab12","zq9"],["x9","yy88"],["q1","w2"]]')
        self.wrong(self.task, self.fmt, self.expected, '[["q1","w2"],["ab12","zq9"],["x9","yy88"]]')

    def test_reordered_pairs(self):
        self.wrong(self.task, self.fmt, self.expected, '[["x9","yy88"],["ab12","zq9"]]')

    def test_empty_list_versus_expected_pairs(self):
        self.wrong(self.task, self.fmt, self.expected, '[]')

    def test_malformed_json(self):
        for output in ('[["ab12","zq9"],["x9","yy88"]', '[["ab12","zq9"],["x9","yy88"]]]', "[['ab12','zq9']]",
                       '[["ab12","zq9"],,["x9","yy88"]]', '[["ab12","zq9"],["x9","yy88"],]', '[[ab12,zq9]]'):
            self.invalid(self.task, self.fmt, self.expected, output)
        self.invalid(self.task, self.fmt, self.expected, '[["ab12","zq9"],["x9","yy88"]', 'parse')

    def test_trailing_prose(self):
        self.invalid(self.task, self.fmt, self.expected, '[["ab12","zq9"],["x9","yy88"]] are the negated records.',
                     'extra_text')
        self.invalid(self.task, self.fmt, self.expected, '[["ab12","zq9"],["x9","yy88"]]\nDone.', 'extra_text')
        self.invalid(self.task, self.fmt, self.expected, '[["ab12","zq9"],["x9","yy88"]] []', 'extra_text')

    def test_leading_prose(self):
        self.invalid(self.task, self.fmt, self.expected, 'The negated records are [["ab12","zq9"],["x9","yy88"]]',
                     'extra_text')
        self.invalid(self.task, self.fmt, self.expected, 'Answer:\n[["ab12","zq9"],["x9","yy88"]]')
        self.invalid(self.task, self.fmt, self.expected, '```json\n[["ab12","zq9"],["x9","yy88"]]\n```')

    def test_empty_output(self):
        for output in ('', '   ', '\n\n', None):
            self.invalid(self.task, self.fmt, self.expected, output, 'empty')

    def test_bare_answer_given_to_json_format(self):
        self.invalid(self.task, self.fmt, self.expected, 'ab12 zq9 x9 yy88', 'parse')
        self.invalid(self.task, self.fmt, '[]', 'none', 'parse')

    def test_wrong_shape(self):
        for output in ('["ab12","zq9"]', '[["ab12","zq9","x9"]]', '[["ab12"],["zq9"]]', '{"ab12":"zq9"}',
                       '[["ab12",1],["x9","yy88"]]', '[[["ab12","zq9"]]]', '"[[\\"ab12\\",\\"zq9\\"]]"', '[[]]',
                       '[{"subject":"ab12","object":"zq9"}]', 'null', '42'):
            self.invalid(self.task, self.fmt, self.expected, output, 'shape')

    def test_empty_answer(self):
        self.ok(self.task, self.fmt, '[]', '[]')
        self.ok(self.task, self.fmt, '[]', '[ ]\n')
        self.wrong(self.task, self.fmt, '[]', '[["ab12","zq9"]]')
        self.invalid(self.task, self.fmt, '[]', 'none', 'parse')
        self.invalid(self.task, self.fmt, '[]', '[[]]', 'shape')
        self.invalid(self.task, self.fmt, '[]', '', 'empty')
        self.invalid(self.task, self.fmt, '[]', '[] none', 'extra_text')

    def test_expected_must_be_canonical(self):
        with self.assertRaises(ValueError):
            grader.grade(self.task, self.fmt, 'none', '[]')


class SelectBare(GraderTestCase):
    task, fmt, expected = 'select', 'bare', SELECT_BARE

    def test_correct(self):
        self.ok(self.task, self.fmt, self.expected, self.expected)

    def test_whitespace_only_differences_accepted(self):
        for output in ('ab12 zq9  x9 yy88', ' ab12 zq9 x9 yy88\n', 'ab12 zq9\nx9 yy88', '\tab12\tzq9 x9 yy88  '):
            self.ok(self.task, self.fmt, self.expected, output)

    def test_dropped_pair(self):
        self.wrong(self.task, self.fmt, self.expected, 'ab12 zq9')

    def test_swapped_identifiers(self):
        self.wrong(self.task, self.fmt, self.expected, 'zq9 ab12 x9 yy88')

    def test_invented_pair(self):
        self.wrong(self.task, self.fmt, self.expected, 'ab12 zq9 x9 yy88 q1 w2')

    def test_reordered_pairs(self):
        self.wrong(self.task, self.fmt, self.expected, 'x9 yy88 ab12 zq9')

    def test_none_versus_expected_pairs(self):
        self.wrong(self.task, self.fmt, self.expected, 'none')

    def test_odd_token_count(self):
        self.invalid(self.task, self.fmt, self.expected, 'ab12 zq9 x9', 'shape')
        self.invalid(self.task, self.fmt, self.expected, 'ab12', 'shape')

    def test_malformed(self):
        for output in ('ab12,zq9,x9,yy88', 'ab12 zq9; x9 yy88', '[ab12 zq9] [x9 yy88]', 'AB12 zq9 x9 yy88',
                       'ab12-zq9 x9-yy88', '"ab12" "zq9" "x9" "yy88"'):
            self.invalid(self.task, self.fmt, self.expected, output, 'parse')

    def test_trailing_prose(self):
        self.invalid(self.task, self.fmt, self.expected, 'ab12 zq9 x9 yy88.')
        self.invalid(self.task, self.fmt, self.expected, 'ab12 zq9 x9 yy88 (the negated records)')
        self.invalid(self.task, self.fmt, self.expected, 'ab12 zq9 x9 yy88 are negated')

    def test_leading_prose(self):
        self.invalid(self.task, self.fmt, self.expected, 'Answer: ab12 zq9 x9 yy88')
        self.invalid(self.task, self.fmt, self.expected, 'The negated records are ab12 zq9 x9 yy88')

    def test_empty_output(self):
        for output in ('', ' \n', None):
            self.invalid(self.task, self.fmt, self.expected, output, 'empty')

    def test_json_answer_given_to_bare_format(self):
        self.invalid(self.task, self.fmt, self.expected, SELECT_JSON, 'parse')
        self.invalid(self.task, self.fmt, self.expected, '[["ab12", "zq9"], ["x9", "yy88"]]', 'parse')
        self.invalid(self.task, self.fmt, 'none', '[]', 'parse')

    def test_none_answer(self):
        self.ok(self.task, self.fmt, 'none', 'none')
        self.ok(self.task, self.fmt, 'none', ' none\n')
        self.wrong(self.task, self.fmt, 'none', 'ab12 zq9')
        self.invalid(self.task, self.fmt, 'none', 'None', 'parse')
        self.invalid(self.task, self.fmt, 'none', 'none none', 'shape')
        self.invalid(self.task, self.fmt, 'none', 'none ab12 zq9', 'shape')
        self.invalid(self.task, self.fmt, 'none', 'none.', 'parse')
        self.invalid(self.task, self.fmt, 'none', '', 'empty')

    def test_expected_must_be_canonical(self):
        with self.assertRaises(ValueError):
            grader.grade(self.task, self.fmt, '[]', 'none')


class LookupJson(GraderTestCase):
    task, fmt, expected = 'lookup', 'json', LOOKUP_JSON

    def test_correct(self):
        self.ok(self.task, self.fmt, self.expected, self.expected)

    def test_whitespace_and_key_order_accepted(self):
        for output in ('{"relation": "help", "polarity": "negative"}', ' {"relation":"help","polarity":"negative"}\n',
                       '{\n "relation": "help",\n "polarity": "negative"\n}',
                       '{"polarity":"negative","relation":"help"}'):
            self.ok(self.task, self.fmt, self.expected, output)

    def test_wrong_polarity(self):
        self.wrong(self.task, self.fmt, self.expected, '{"relation":"help","polarity":"positive"}')

    def test_wrong_relation(self):
        self.wrong(self.task, self.fmt, self.expected, '{"relation":"see","polarity":"negative"}')
        self.wrong(self.task, self.fmt, self.expected, '{"relation":"contain","polarity":"negative"}')

    def test_values_swapped(self):
        self.wrong(self.task, self.fmt, self.expected, '{"relation":"negative","polarity":"help"}')

    def test_missing_key(self):
        self.invalid(self.task, self.fmt, self.expected, '{"relation":"help"}', 'shape')
        self.invalid(self.task, self.fmt, self.expected, '{"polarity":"negative"}', 'shape')
        self.invalid(self.task, self.fmt, self.expected, '{}', 'shape')

    def test_extra_key(self):
        self.invalid(self.task, self.fmt, self.expected,
                     '{"relation":"help","polarity":"negative","subject":"ab12"}', 'shape')
        self.invalid(self.task, self.fmt, self.expected, '{"relation":"help","polarity":"negative","tense":"past"}',
                     'shape')

    def test_duplicate_key(self):
        self.invalid(self.task, self.fmt, self.expected, '{"relation":"see","relation":"help","polarity":"negative"}',
                     'shape')

    def test_wrong_shape(self):
        for output in ('[["ab12","zq9"]]', '["help","negative"]', '{"relation":1,"polarity":"negative"}',
                       '{"relation":["help"],"polarity":"negative"}', '"help negative"', 'null',
                       '{"Relation":"help","Polarity":"negative"}'):
            self.invalid(self.task, self.fmt, self.expected, output, 'shape')

    def test_malformed_json(self):
        for output in ('{"relation":"help","polarity":"negative"', "{'relation':'help','polarity':'negative'}",
                       '{"relation":"help","polarity":"negative",}', '{relation:help,polarity:negative}',
                       '{"relation":"help";"polarity":"negative"}'):
            self.invalid(self.task, self.fmt, self.expected, output)
        self.invalid(self.task, self.fmt, self.expected, '{"relation":"help","polarity":"negative"', 'parse')

    def test_trailing_prose(self):
        self.invalid(self.task, self.fmt, self.expected, '{"relation":"help","polarity":"negative"} is the record.',
                     'extra_text')
        self.invalid(self.task, self.fmt, self.expected, '{"relation":"help","polarity":"negative"}\n{}',
                     'extra_text')

    def test_leading_prose(self):
        self.invalid(self.task, self.fmt, self.expected, 'The record is {"relation":"help","polarity":"negative"}',
                     'extra_text')
        self.invalid(self.task, self.fmt, self.expected, '```json\n{"relation":"help","polarity":"negative"}\n```')

    def test_empty_output(self):
        for output in ('', '\n', None):
            self.invalid(self.task, self.fmt, self.expected, output, 'empty')

    def test_bare_answer_given_to_json_format(self):
        self.invalid(self.task, self.fmt, self.expected, 'help negative', 'parse')
        self.invalid(self.task, self.fmt, self.expected, 'relation help polarity negative', 'parse')

    def test_expected_must_be_canonical(self):
        with self.assertRaises(ValueError):
            grader.grade(self.task, self.fmt, 'help negative', LOOKUP_JSON)


class LookupBare(GraderTestCase):
    task, fmt, expected = 'lookup', 'bare', LOOKUP_BARE

    def test_correct(self):
        self.ok(self.task, self.fmt, self.expected, self.expected)

    def test_whitespace_only_differences_accepted(self):
        for output in ('help  negative', ' help negative\n', 'help\nnegative', '\thelp negative  '):
            self.ok(self.task, self.fmt, self.expected, output)

    def test_wrong_polarity(self):
        self.wrong(self.task, self.fmt, self.expected, 'help positive')

    def test_wrong_relation(self):
        self.wrong(self.task, self.fmt, self.expected, 'see negative')

    def test_order_matters(self):
        self.wrong(self.task, self.fmt, self.expected, 'negative help')

    def test_missing_word(self):
        self.invalid(self.task, self.fmt, self.expected, 'help', 'shape')
        self.invalid(self.task, self.fmt, self.expected, 'negative', 'shape')

    def test_extra_word(self):
        self.invalid(self.task, self.fmt, self.expected, 'help negative past', 'shape')
        self.invalid(self.task, self.fmt, self.expected, 'relation help polarity negative', 'shape')

    def test_malformed(self):
        for output in ('help, negative', 'help/negative', 'help-negative', 'Help Negative', '"help" "negative"',
                       'help: negative'):
            self.invalid(self.task, self.fmt, self.expected, output, 'parse')

    def test_trailing_prose(self):
        self.invalid(self.task, self.fmt, self.expected, 'help negative.')
        self.invalid(self.task, self.fmt, self.expected, 'help negative is the answer')

    def test_leading_prose(self):
        self.invalid(self.task, self.fmt, self.expected, 'Answer: help negative')
        self.invalid(self.task, self.fmt, self.expected, 'The relation is help negative')

    def test_empty_output(self):
        for output in ('', '  ', None):
            self.invalid(self.task, self.fmt, self.expected, output, 'empty')

    def test_json_answer_given_to_bare_format(self):
        self.invalid(self.task, self.fmt, self.expected, LOOKUP_JSON, 'parse')
        self.invalid(self.task, self.fmt, self.expected, '{"relation": "help", "polarity": "negative"}', 'parse')

    def test_expected_must_be_canonical(self):
        with self.assertRaises(ValueError):
            grader.grade(self.task, self.fmt, LOOKUP_JSON, 'help negative')


class Arguments(unittest.TestCase):
    def test_unknown_task_or_format(self):
        with self.assertRaises(ValueError):
            grader.grade('count', 'json', '[]', '[]')
        with self.assertRaises(ValueError):
            grader.grade('select', 'xml', '[]', '[]')


if __name__ == '__main__':
    unittest.main(verbosity=1)
