"""Hand-checked examples: the renderings and answers below were written by hand, not by the code under test.

    python -I -m unittest discover -s experiments/compositional_diagnostics/tests
"""
import unittest

import _path  # noqa: F401
import questions as Q
import reference
import render
import schema

R = schema.record
CASE = [
    R('aa', 'see', 'bb', 'positive', 'past', 'simple', 'direct'),
    R('aa', 'help', 'cc', 'negative', 'present', 'progressive', 'unspecified'),
    R('bb', 'contain', 'cc', 'positive', 'future', 'completed', 'reported'),
    R('dd', 'see', 'bb', 'positive', 'past', 'simple', 'inferred'),
]

RENDERED = {
    'english': ('1. By direct observation, "aa" saw "bb".\n'
                '2. "aa" is not helping "cc".\n'
                '3. By report, "bb" will have contained "cc".\n'
                '4. By inference, "dd" saw "bb".\n'),
    'json': ('[{"s":"aa","r":"see","t":"past","e":"direct","o":"bb"},'
             '{"s":"aa","r":"help","neg":true,"a":"progressive","o":"cc"},'
             '{"s":"bb","r":"contain","t":"future","a":"completed","e":"reported","o":"cc"},'
             '{"s":"dd","r":"see","t":"past","e":"inferred","o":"bb"}]\n'),
    'sylang2': 'aa\tspd\tbb\naa\tHg\tcc\nbb\tcfcr\tcc\ndd\tspi\tbb\n',
    'sylang2g': 'aa\tspd bb;Hg cc\nbb\tcfcr cc\ndd\tspi bb\n',
}

ANSWERS = [
    ('tense', {'subject': 'aa', 'object': 'bb'}, '"past"'),
    ('tense', {'subject': 'bb', 'object': 'cc'}, '"future"'),
    ('aspect', {'subject': 'aa', 'object': 'cc'}, '"progressive"'),
    ('aspect', {'subject': 'dd', 'object': 'bb'}, '"simple"'),
    ('evidence', {'subject': 'dd', 'object': 'bb'}, '"inferred"'),
    ('evidence', {'subject': 'aa', 'object': 'cc'}, '"unspecified"'),
    ('profile', {'subject': 'bb', 'object': 'cc'},
     '{"relation":"contain","polarity":"positive","tense":"future","aspect":"completed","evidence":"reported"}'),
    ('filter', {'conditions': [['tense', 'past'], ['relation', 'see']]}, '[["aa","bb"],["dd","bb"]]'),
    ('filter', {'conditions': [['polarity', 'negative'], ['aspect', 'progressive']]}, '[["aa","cc"]]'),
    ('filter', {'conditions': [['evidence', 'reported'], ['tense', 'past']]}, '[]'),
    ('filter', {'conditions': [['aspect', 'simple'], ['polarity', 'positive']]}, '[["aa","bb"],["dd","bb"]]'),
    ('join', {'x': 'aa', 'rel_a': 'see', 'tense_a': 'past', 'rel_b': 'contain'}, '["cc"]'),
    ('join', {'x': 'dd', 'rel_a': 'see', 'tense_a': 'past', 'rel_b': 'contain'}, '["cc"]'),
    ('join', {'x': 'aa', 'rel_a': 'help', 'tense_a': 'present', 'rel_b': 'contain'}, '[]'),  # hop 1 is negated
    ('join', {'x': 'aa', 'rel_a': 'see', 'tense_a': 'future', 'rel_b': 'contain'}, '[]'),    # wrong tense
    ('join', {'x': 'aa', 'rel_a': 'see', 'tense_a': 'past', 'rel_b': 'see'}, '[]'),          # bb sees nothing
]


class HandChecked(unittest.TestCase):
    def test_renderings(self):
        for arm, text in RENDERED.items():
            self.assertEqual(render.render(arm, CASE), text, arm)
            self.assertEqual(render.parse(arm, text), CASE, arm)

    def test_answers(self):
        for qtype, params, expected in ANSWERS:
            self.assertEqual(Q.answer(qtype, CASE, params), expected, (qtype, params))
            self.assertEqual(reference.reference_answer(qtype, CASE, params), expected, (qtype, params))

    def test_instruction_text(self):
        self.assertEqual(Q.instruction('tense', {'subject': 'aa', 'object': 'bb'}),
                         'What is the tense of the record whose subject is "aa" and whose object is "bb"? Answer '
                         'with one JSON string: "past", "present" or "future".')
        self.assertIn('"aa" sees Y with tense past', Q.instruction('join', ANSWERS[11][1]))

    def test_absent_pair_is_refused(self):
        with self.assertRaises(AssertionError):
            Q.answer('tense', CASE, {'subject': 'bb', 'object': 'aa'})
        with self.assertRaises(ValueError):
            reference.reference_answer('tense', CASE, {'subject': 'bb', 'object': 'aa'})


if __name__ == '__main__':
    unittest.main()
