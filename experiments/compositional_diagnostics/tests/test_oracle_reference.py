"""The oracle agrees with the independent reference; the comparison is able to catch a planted bug."""
import itertools
import random
import unittest

import _path  # noqa: F401
import questions as Q
import reference
import verify_oracle as V


class OracleReference(unittest.TestCase):
    def test_d1_complete(self):
        """Every one-record case and every question (the full D1 domain of verify_oracle.py)."""
        result = V.compare(([r] for r in V.records_of(V.FULL)), V.questions_of(V.FULL))
        self.assertEqual(result['n_mismatch'], 0, result['mismatches'])

    def test_d2_sample(self):
        """A seeded sample of D2 (verify_oracle.py runs it completely)."""
        cases = list(V.cases_of(V.records_of(V.D2), 2))
        sample = random.Random(0).sample(cases, 1500)
        result = V.compare(sample, V.questions_of(V.D2))
        self.assertEqual(result['n_mismatch'], 0, result['mismatches'])

    def test_planted_bugs_are_caught(self):
        """Mutants of the oracle (join ignoring polarity of hop 2; filter reading only one condition) must disagree
        with the reference somewhere on the small domain."""
        original = Q.oracle

        def mutant_join(qtype, records, params):
            if qtype == 'join':
                hop1 = [r['object'] for r in records if r['subject'] == params['x'] and r['relation'] == params['rel_a']
                        and r['polarity'] == 'positive' and r['tense'] == params['tense_a']]
                return sorted({r['object'] for r in records if r['subject'] in hop1 and r['relation'] == params['rel_b']})
            return original(qtype, records, params)

        def mutant_filter(qtype, records, params):
            if qtype == 'filter':
                (f1, v1), _ = params['conditions']
                return sorted([r['subject'], r['object']] for r in records if r[f1] == v1)
            return original(qtype, records, params)

        cases = list(itertools.islice(V.cases_of(V.records_of(V.D3), 3), 3000))
        for mutant in (mutant_join, mutant_filter):
            Q.oracle = mutant
            try:
                result = V.compare(cases, V.questions_of(V.D3, kinds=('filter', 'join')))
            finally:
                Q.oracle = original
            self.assertGreater(result['n_mismatch'], 0, mutant.__name__)


if __name__ == '__main__':
    unittest.main()
