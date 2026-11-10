"""Renderings agree with the pilot's code paths and invert exactly; the grader rejects every corruption."""
import random
import unittest

import _path  # noqa: F401
import answer_grader as G
import pilot_compat
import render
import schema


def random_case(rng, k=6):
    pool = schema.IdentifierPool('test', rng)
    used, records = set(), []
    while len(records) < k:
        s = rng.choice([r['subject'] for r in records]) if records and rng.random() < 0.4 else pool.fresh(used)
        o = pool.fresh(used)
        f = {x: rng.choice(schema.VALUES[x]) for x in schema.FEATURES}
        records.append(schema.record(s, f['relation'], o, f['polarity'], f['tense'], f['aspect'], f['evidence']))
    return schema.group_normalise(records)


class Renderings(unittest.TestCase):
    def test_round_trip_and_pilot_identity(self):
        rng = random.Random(1)
        st3, data = pilot_compat.study3(), pilot_compat.pilot_data()
        for _ in range(300):
            case = random_case(rng, rng.randint(1, 12))
            for arm in render.ARMS:
                self.assertEqual(render.parse(arm, render.render(arm, case)), case)
            self.assertEqual(render.render('json', case), '[' + ','.join(st3._json_record(r) for r in case) + ']\n')
            self.assertEqual(render.render('sylang2g', case), data.group_by_subject(render.render('sylang2', case)))

    def test_every_feature_combination_round_trips_in_english(self):
        for r in schema.RELATIONS:
            for p in schema.POLARITIES:
                for t in schema.TENSES:
                    for a in schema.ASPECTS:
                        for e in schema.EVIDENCES:
                            case = [schema.record('ab', r, 'cd', p, t, a, e)]
                            for arm in render.ARMS:
                                self.assertEqual(render.parse(arm, render.render(arm, case)), case)

    def test_render_refuses_non_adjacent_subjects(self):
        R = schema.record
        with self.assertRaises(AssertionError):
            render.render('sylang2g', [R('ab', 'see', 'cd'), R('ef', 'see', 'cd'), R('ab', 'help', 'gh')])

    def test_identifiers_are_never_vocabulary(self):
        rng = random.Random(2)
        pool, used = schema.IdentifierPool('dev', rng), set()
        ids = [pool.fresh(used) for _ in range(2000)]
        self.assertFalse(set(ids) & schema.RESERVED)
        self.assertTrue(all(schema.id_split(i) == 'dev' for i in ids))


class Grader(unittest.TestCase):
    def assertGrade(self, qtype, expected, output, correct, invalid, reason=None):
        g = G.grade(qtype, expected, output)
        self.assertEqual((g['correct'], g['invalid']), (correct, invalid), (qtype, output, g))
        if reason:
            self.assertEqual(g['reason'], reason, (qtype, output))

    def test_single(self):
        self.assertGrade('tense', '"past"', '"past"', True, False)
        self.assertGrade('tense', '"past"', ' "past"\n', True, False)
        self.assertGrade('tense', '"past"', '```json\n"past"\n```', True, False)
        self.assertGrade('tense', '"past"', '"present"', False, False, 'wrong')
        self.assertGrade('tense', '"past"', 'past', False, True, 'parse')
        self.assertGrade('tense', '"past"', '"past" because', False, True, 'extra_text')
        self.assertGrade('tense', '"past"', '["past"]', False, True, 'shape')
        self.assertGrade('tense', '"past"', '', False, True, 'empty')
        self.assertGrade('tense', '"past"', 'The answer is "past"', False, True, 'extra_text')

    def test_profile_needs_every_field(self):
        exp = '{"relation":"see","polarity":"negative","tense":"past","aspect":"simple","evidence":"direct"}'
        self.assertGrade('profile', exp, exp.replace(',', ', '), True, False)
        reordered = '{"tense":"past","relation":"see","aspect":"simple","polarity":"negative","evidence":"direct"}'
        self.assertGrade('profile', exp, reordered, True, False)
        for field, other in (('relation', 'help'), ('polarity', 'positive'), ('tense', 'future'),
                             ('aspect', 'completed'), ('evidence', 'reported')):
            import json
            d = json.loads(exp)
            d[field] = other
            self.assertGrade('profile', exp, json.dumps(d), False, False, 'wrong')
        self.assertGrade('profile', exp, '{"relation":"see","polarity":"negative","tense":"past","aspect":"simple"}',
                         False, True, 'shape')
        self.assertGrade('profile', exp, exp[:-1] + ',"extra":"x"}', False, True, 'shape')
        self.assertGrade('profile', exp, exp[:-1] + ',"tense":"past"}', False, True, 'shape')  # duplicate key

    def test_sets(self):
        exp = '[["aa","bb"],["dd","bb"]]'
        self.assertGrade('filter', exp, '[["dd","bb"],["aa","bb"]]', True, False)   # order is free
        self.assertGrade('filter', exp, '[["aa","bb"]]', False, False, 'wrong')        # dropped pair
        self.assertGrade('filter', exp, '[["aa","bb"],["dd","bb"],["x1","y2"]]', False, False, 'wrong')  # invented
        self.assertGrade('filter', exp, '[["bb","aa"],["dd","bb"]]', False, False, 'wrong')  # swapped roles
        self.assertGrade('filter', exp, '[["aa","bb"],["aa","bb"],["dd","bb"]]', False, True, 'duplicate')
        self.assertGrade('filter', exp, '[["aa","bb","dd"]]', False, True, 'shape')
        self.assertGrade('filter', '[]', '[]', True, False)
        self.assertGrade('filter', '[]', 'none', False, True, 'parse')
        self.assertGrade('join', '["cc"]', '["cc"]', True, False)
        self.assertGrade('join', '["cc"]', '["cc","dd"]', False, False, 'wrong')
        self.assertGrade('join', '["cc","dd"]', '["dd","cc"]', True, False)
        self.assertGrade('join', '["cc"]', '["cc","cc"]', False, True, 'duplicate')
        self.assertGrade('join', '["cc"]', '"cc"', False, True, 'shape')
        self.assertGrade('join', '[]', '[] (none)', False, True, 'extra_text')


if __name__ == '__main__':
    unittest.main()
