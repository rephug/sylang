"""Determinism, split hygiene and the declared expectation of every family member."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
import families as F
import generate
import questions as Q
import schema

SMALL = dict(version='cd-test', seed=3, dev_families=6, test_families=10)


class Generator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.manifest = generate.build(**SMALL)
        cls.families = [json.loads(l) for l in cls.files['families.jsonl'].decode().splitlines()]

    def test_same_seed_same_bytes_in_a_fresh_interpreter(self):
        """A new process with a different string-hash seed reproduces every file hash."""
        with tempfile.TemporaryDirectory() as tmp:
            for hashseed in ('0', '12345'):
                out = Path(tmp) / hashseed
                env = dict(os.environ, PYTHONHASHSEED=hashseed)
                subprocess.run([sys.executable, '-I', str(_path.ROOT / 'generate.py'), '--out', str(out),
                                '--version', SMALL['version'], '--seed', str(SMALL['seed']),
                                '--dev-families', str(SMALL['dev_families']),
                                '--test-families', str(SMALL['test_families'])], check=True, env=env,
                               capture_output=True)
                written = json.loads((out / SMALL['version'] / 'manifest.json').read_text())
                self.assertEqual(written['sha256'], self.manifest['sha256'])

    def test_other_seed_other_data(self):
        files, _ = generate.build(SMALL['version'], SMALL['seed'] + 1, SMALL['dev_families'], SMALL['test_families'])
        self.assertNotEqual(files['families.jsonl'], self.files['families.jsonl'])

    def test_expectations_hold(self):
        for fam in self.families:
            members = {m['variant']: m for m in fam['members']}
            self.assertEqual(set(members), set(F.VARIANTS))
            base = Q.oracle(fam['qtype'], members['base']['records'], members['base']['params'])
            for v, m in members.items():
                schema.check_case(m['records'])
                value = Q.oracle(fam['qtype'], m['records'], m['params'])
                self.assertEqual(Q.canonical(fam['qtype'], value), m['answer'])
                if v == 'relevant':
                    self.assertNotEqual(value, base)
                elif v == 'rename':
                    self.assertEqual(value, Q.rename_value(fam['qtype'], base, m['mapping']))
                elif v != 'base':
                    self.assertEqual(value, base, (fam['family'], v))

    def test_irrelevant_features_never_matter(self):
        """Exhaustively: editing any feature outside questions.relevant_features, on any record, keeps the answer."""
        for fam in self.families:
            for m in fam['members']:
                rel = Q.relevant_features(fam['qtype'], m['params'])
                base = Q.answer(fam['qtype'], m['records'], m['params'])
                for i in range(len(m['records'])):
                    for f in schema.FEATURES:
                        if f in rel:
                            continue
                        for v in schema.VALUES[f]:
                            edited = [dict(r) for r in m['records']]
                            edited[i][f] = v
                            self.assertEqual(Q.answer(fam['qtype'], edited, m['params']), base)

    def test_splits_are_disjoint_before_rendering(self):
        ids = {'dev': set(), 'test': set()}
        sigs = {'dev': set(), 'test': set()}
        for fam in self.families:
            for m in fam['members']:
                ids[fam['split']] |= set(schema.entities(m['records']))
                sigs[fam['split']].add(schema.structural_signature(m['records'], m['params']))
        self.assertFalse(ids['dev'] & ids['test'])
        self.assertFalse(sigs['dev'] & sigs['test'])
        self.assertTrue(all(schema.id_split(i) == s for s in ids for i in ids[s]))

    def test_signature_is_invariant_to_rename_and_reorder(self):
        for fam in self.families:
            m = {x['variant']: x for x in fam['members']}
            s = schema.structural_signature(m['base']['records'], m['base']['params'])
            self.assertEqual(s, schema.structural_signature(m['rename']['records'], m['rename']['params']))
            self.assertEqual(s, schema.structural_signature(m['reorder']['records'], m['reorder']['params']))

    def test_rendered_rows_match_families(self):
        for split in ('dev', 'test'):
            for arm in ('english', 'json', 'sylang2', 'sylang2g'):
                rows = [json.loads(l) for l in self.files[f'{split}_{arm}.jsonl'].decode().splitlines()]
                self.assertEqual(len(rows), 6 * len(Q.QTYPES) * SMALL[f'{split}_families'])
                self.assertTrue(all(r['records'].endswith('\n') for r in rows))

    def test_design_schedule_is_balanced(self):
        import random
        sched = F.design_schedule('tense', 30, random.Random(0))
        self.assertEqual(sorted(sched.count(v) for v in schema.TENSES), [10, 10, 10])


if __name__ == '__main__':
    unittest.main()
