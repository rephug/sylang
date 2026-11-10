"""The pilot v1 cases regenerate exactly, and the field-dependence claims hold on them."""
import unittest

import _path  # noqa: F401
import analyze_existing as AE


class ExistingTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.cases, cls.verified = AE.v1_cases()

    def test_v1_hashes(self):
        self.assertEqual(self.verified, {'cases_dev.jsonl': True, 'cases_test.jsonl': True})

    def test_tense_aspect_evidence_never_change_an_answer(self):
        fd = AE.field_dependence(self.data, self.cases['dev'])
        for f in ('tense', 'aspect', 'evidence'):
            self.assertEqual(fd[f]['select_changed'], 0)
            self.assertEqual(fd[f]['lookup_changed'], 0)
        self.assertEqual(fd['polarity']['select_changed_share'], 1.0)
        self.assertEqual(fd['relation']['select_changed'], 0)
        self.assertEqual(fd['relation']['lookup_changed_share_on_queried_record'], 1.0)

    def test_one_character_solvers_are_perfect(self):
        for arm, panels in AE.surface_solvers(self.data, self.cases['dev']).items():
            for p in panels.values():
                self.assertEqual((p['select_accuracy'], p['lookup_accuracy']), (1.0, 1.0), arm)


if __name__ == '__main__':
    unittest.main()
