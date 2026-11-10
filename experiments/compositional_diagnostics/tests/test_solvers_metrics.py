"""Every deliberately weak solver is caught by its targeted metric and not penalised where its defect is
harmless; invariance alone earns no credit; the bootstrap is deterministic and brackets the point estimate."""
import json
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
import baseline
import generate
import metrics
import score
import solvers


class SolversAndMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        files, manifest = generate.build('cd-test', 11, 10, 40)
        out = Path(cls.tmp.name) / 'cd-test'
        generate.write(out, files, manifest, force=True)
        run = baseline.run_solvers(out)
        cls.rows = run['rows']['json']
        cls.outputs = run['outputs']
        cls.identical = run['identical_across_arms']
        cls.results = {name: {'summary': metrics.summarise(metrics.family_values(cls.rows, outs), with_ci=False)}
                       for name, outs in cls.outputs.items()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_identical_across_arms(self):
        self.assertTrue(all(self.identical.values()), self.identical)

    def test_targets_and_specificity(self):
        failed = [c for c in baseline.check(self.results) if not c['pass']]
        self.assertEqual(failed, [])

    def test_constant_answer_is_invariant_but_not_credited(self):
        s = self.results['constant_majority']['summary']['all']
        for v in metrics.INVARIANCE:
            self.assertEqual(s[f'raw_invariance_{v}']['mean'], 1.0)
            self.assertLessEqual(s[f'robust_{v}']['mean'], s['exact_base']['mean'] + 1e-12)
        self.assertEqual(s['relevant']['mean'], 0.0)
        self.assertEqual(s['family']['mean'], 0.0)
        self.assertEqual(s['raw_sensitivity']['mean'], 0.0)

    def test_missing_outputs_are_invalid(self):
        result = score.score(self.rows, {}, with_ci=False)
        self.assertEqual(result['summary']['all']['exact']['mean'], 0.0)
        self.assertEqual(result['summary']['all']['invalid']['mean'], 1.0)
        self.assertEqual(result['n_missing_outputs'], len(self.rows))

    def test_bootstrap_deterministic_and_brackets(self):
        fv = metrics.family_values(self.rows, self.outputs['subject_only'])
        a = metrics.summarise(fv, resamples=300)
        b = metrics.summarise(fv, resamples=300)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        for scope in a.values():
            for e in scope.values():
                self.assertLessEqual(e['lo'], e['mean'] + 1e-12)
                self.assertGreaterEqual(e['hi'], e['mean'] - 1e-12)

    def test_chance_matches_random_guesser_order_of_magnitude(self):
        chance = metrics.summarise(metrics.chance_family_values(self.rows), metrics=metrics.CREDITED, with_ci=False)
        rand = self.results['random_guess']['summary']
        for q in ('tense', 'aspect', 'evidence'):
            self.assertAlmostEqual(rand[q]['exact']['mean'], chance[q]['exact']['mean'], delta=0.12)

    def test_solver_list_is_complete(self):
        self.assertEqual(set(self.outputs), set(solvers.SOLVERS))


if __name__ == '__main__':
    unittest.main()
