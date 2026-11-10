"""Exhaustive check of the oracle against the independent reference (``reference.py``) on small domains.

    python -I verify_oracle.py [--out reports/oracle_verification.json]

Three enumerable domains over the entities aa, bb, cc (all six ordered pairs, no self-pairs):

* D1: every one-record case with the full feature space (6 pairs x 216 features = 1,296 cases) and every
  question that can be asked of it: the four pair questions for every ordered pair (an absent pair must make
  both implementations refuse), every filter with two conditions on distinct features (178 ordered
  condition pairs), every join (3 x 3 x 3 x 3 = 81);
* D2: every two-record case over a two-value-per-feature space (relation see|help, polarity both, tense
  past|future, aspect simple|completed, evidence direct|reported; 192 records), every question over that space;
* D3: every three-record case over relation see|help, polarity both, tense past|present (aspect and evidence
  at their defaults; 48 records): every join and every filter over those three features.

Each case is also evaluated in reversed record order (answers are order-free). A third implementation, the
weak-solver module's defect-free path (``solvers.flawed`` with no defect), is checked too.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import questions as Q  # noqa: E402
import reference  # noqa: E402
import schema  # noqa: E402
import solvers  # noqa: E402

ENTITIES = ('aa', 'bb', 'cc')
PAIRS = [(s, o) for s in ENTITIES for o in ENTITIES if s != o]
FULL = dict(schema.VALUES)
D2 = {'relation': ('see', 'help'), 'polarity': ('positive', 'negative'), 'tense': ('past', 'future'),
      'aspect': ('simple', 'completed'), 'evidence': ('direct', 'reported')}
D3 = {'relation': ('see', 'help'), 'polarity': ('positive', 'negative'), 'tense': ('past', 'present'),
      'aspect': ('simple',), 'evidence': ('unspecified',)}


def records_of(space: dict) -> list[dict]:
    out = []
    for s, o in PAIRS:
        for values in itertools.product(*(space[f] for f in schema.FEATURES)):
            f = dict(zip(schema.FEATURES, values))
            out.append(schema.record(s, f['relation'], o, f['polarity'], f['tense'], f['aspect'], f['evidence']))
    return out


def questions_of(space: dict, kinds=('pair', 'filter', 'join')) -> list[tuple[str, dict]]:
    out = []
    if 'pair' in kinds:
        for s, o in PAIRS:
            for qtype in Q.PAIR_QTYPES:
                out.append((qtype, {'subject': s, 'object': o}))
    if 'filter' in kinds:
        fields = [f for f in schema.FEATURES if len(space[f]) > 1]
        for f1, f2 in itertools.permutations(fields, 2):
            for v1 in space[f1]:
                for v2 in space[f2]:
                    out.append(('filter', {'conditions': [[f1, v1], [f2, v2]]}))
    if 'join' in kinds:
        for x in ENTITIES:
            for ra in space['relation']:
                for t in space['tense']:
                    for rb in space['relation']:
                        out.append(('join', {'x': x, 'rel_a': ra, 'tense_a': t, 'rel_b': rb}))
    return out


def cases_of(pool: list[dict], k: int):
    for combo in itertools.combinations(pool, k):
        pairs = [(r['subject'], r['object']) for r in combo]
        if len(set(pairs)) == k:
            yield list(combo)


def compare(cases, qs) -> dict:
    n = refused = 0
    mismatches = []
    answers = {}
    for case in cases:
        for order in (case, case[::-1]):
            for qtype, params in qs:
                try:
                    a = Q.answer(qtype, order, params)
                except AssertionError:
                    a = None
                try:
                    b = reference.check_json(reference.reference_answer(qtype, order, params))
                except ValueError:
                    b = None
                c = Q.canonical(qtype, solvers.flawed(qtype, order, params)) if a is not None else None
                n += 1
                if a is None:
                    refused += 1
                if a != b or a != c:
                    if len(mismatches) < 10:
                        mismatches.append({'case': order, 'qtype': qtype, 'params': params, 'oracle': a,
                                           'reference': b, 'solver_oracle': c})
                if a is not None:
                    answers.setdefault(qtype, set()).add(a)
    return {'evaluations': n, 'refused_absent_pair': refused, 'mismatches': len(mismatches) and mismatches,
            'n_mismatch': len(mismatches), 'distinct_answers': {k: len(v) for k, v in sorted(answers.items())}}


def run() -> dict:
    t0 = time.perf_counter()
    report = {}
    pool1 = records_of(FULL)
    report['D1'] = {'cases': len(pool1), **compare(([r] for r in pool1), questions_of(FULL))}
    pool2 = records_of(D2)
    c2 = list(cases_of(pool2, 2))
    report['D2'] = {'cases': len(c2), **compare(c2, questions_of(D2))}
    pool3 = records_of(D3)
    c3 = list(cases_of(pool3, 3))
    report['D3'] = {'cases': len(c3), **compare(c3, questions_of(D3, kinds=('filter', 'join')))}
    report['all_agree'] = all(report[d]['n_mismatch'] == 0 for d in ('D1', 'D2', 'D3'))
    report['seconds'] = round(time.perf_counter() - t0, 1)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--out', default=str(HERE / 'reports' / 'oracle_verification.json'))
    args = p.parse_args(argv)
    report = run()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    seconds = report.pop('seconds')
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({d: {k: report[d][k] for k in ('cases', 'evaluations', 'n_mismatch')} for d in ('D1', 'D2', 'D3')}),
          f'all_agree={report["all_agree"]} ({seconds}s)')
    return 0 if report['all_agree'] else 1


if __name__ == '__main__':
    sys.exit(main())
