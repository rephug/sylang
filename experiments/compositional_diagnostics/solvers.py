"""Deliberately weak solvers, each with one named defect, plus the oracle and two chance-level solvers.

Every solver reads the *rendered* records of its arm (``render.parse``), never the generator's structures, and
the question's structured parameters (a solver is weak at reading records, not at reading the question).
``solve(name, row, arm) -> output string``.

| solver | defect |
|---|---|
| ``oracle`` | none (sanity upper bound; every score must be 1.0) |
| ``random_guess`` | uniform guess over the answer space (seeded by the item id) |
| ``constant_majority`` | the most frequent dev answer of the question type, always (fitted on dev) |
| ``first_record`` / ``last_record`` | answers from the first / last record in the rendering, whatever the question names |
| ``ignore_<field>`` (tense, aspect, evidence, polarity, relation) | cannot see that field: a condition on it always holds; asked for it, answers its default value (``see`` for relation) |
| ``subject_only`` / ``object_only`` | pair questions: takes the first record with the named subject / object, ignoring the other identifier |
| ``feature_bag`` | filter: a record qualifies when each condition holds for *some* record of the same subject (features pooled across a subject's records) |
| ``unbound_join`` | join: any positive link record ``(W, rel_b, Z)`` counts once a valid first hop exists, whether or not W is that hop's Y |
| ``role_swap`` | reads every record with subject and object exchanged |

Where a defect does not apply to a question type the solver behaves as the oracle, so each solver isolates one
shortcut.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

import questions as Q
import render
import schema

FIELD_GUESS = {'relation': 'see', **schema.DEFAULTS}
IGNORE = ('tense', 'aspect', 'evidence', 'polarity', 'relation')
SOLVERS = ('oracle', 'random_guess', 'constant_majority', 'first_record', 'last_record',
           *(f'ignore_{f}' for f in IGNORE), 'subject_only', 'object_only', 'feature_bag', 'unbound_join',
           'role_swap')


def _value_of(qtype, r):
    if r is None:
        return {k: FIELD_GUESS[k] for k in Q.PROFILE_KEYS} if qtype == 'profile' else FIELD_GUESS[Q.SINGLE_FIELD[qtype]]
    return {k: r[k] for k in Q.PROFILE_KEYS} if qtype == 'profile' else r[Q.SINGLE_FIELD[qtype]]


def flawed(qtype: str, records: list[dict], params: dict, *, blind: str | None = None, pair: str = 'both',
           bag: bool = False, unbound: bool = False) -> object:
    """The oracle with optional defects; returns an answer value."""
    def holds(r, field, value):
        return True if field == blind else r[field] == value

    if qtype in Q.PAIR_QTYPES:
        if pair == 'both':
            hits = [r for r in records if r['subject'] == params['subject'] and r['object'] == params['object']]
        elif pair == 'subject':
            hits = [r for r in records if r['subject'] == params['subject']]
        else:
            hits = [r for r in records if r['object'] == params['object']]
        value = _value_of(qtype, hits[0] if hits else None)
        if blind is not None:
            if qtype == 'profile':
                value = dict(value)
                value[blind] = FIELD_GUESS[blind]
            elif Q.SINGLE_FIELD[qtype] == blind:
                value = FIELD_GUESS[blind]
        return value
    if qtype == 'filter':
        (f1, v1), (f2, v2) = params['conditions']
        if not bag:
            return sorted([r['subject'], r['object']] for r in records if holds(r, f1, v1) and holds(r, f2, v2))
        out = []
        for r in records:
            mine = [s for s in records if s['subject'] == r['subject']]
            if (holds(r, f1, v1) or holds(r, f2, v2)) and any(holds(s, f1, v1) for s in mine) \
                    and any(holds(s, f2, v2) for s in mine):
                out.append([r['subject'], r['object']])
        return sorted(out)
    if qtype == 'join':
        hop1 = [r['object'] for r in records if r['subject'] == params['x'] and holds(r, 'relation', params['rel_a'])
                and holds(r, 'polarity', 'positive') and holds(r, 'tense', params['tense_a'])]
        links = [r for r in records if holds(r, 'relation', params['rel_b']) and holds(r, 'polarity', 'positive')]
        if unbound:
            return sorted({r['object'] for r in links}) if hop1 else []
        return sorted({r['object'] for r in links if r['subject'] in hop1})
    raise ValueError(qtype)


def _positional(qtype, record, params):
    if qtype in Q.PAIR_QTYPES:
        return _value_of(qtype, record)
    if qtype == 'filter':
        return [[record['subject'], record['object']]]
    return [record['object']]


def _rng(row_id: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(row_id.encode()).digest()[:8], 'big'))


def _random_value(qtype, records, rng):
    if qtype in Q.SINGLE_FIELD:
        return rng.choice(schema.VALUES[Q.SINGLE_FIELD[qtype]])
    if qtype == 'profile':
        return {k: rng.choice(schema.VALUES[k]) for k in Q.PROFILE_KEYS}
    if qtype == 'filter':
        return [[r['subject'], r['object']] for r in records if rng.random() < 0.5]
    return [e for e in schema.entities(records) if rng.random() < 0.5]


class Solvers:
    def __init__(self, dev_rows: list[dict] | None = None):
        self.majority: dict[str, str] = {}
        if dev_rows:
            counts: dict[str, Counter] = {}
            for row in dev_rows:
                counts.setdefault(row['qtype'], Counter())[row['answer']] += 1
            for qtype, c in counts.items():
                top = max(c.values())
                self.majority[qtype] = min(a for a, n in c.items() if n == top)

    def solve(self, name: str, row: dict, arm: str) -> str:
        qtype, params = row['qtype'], row['params']
        records = render.parse(arm, row['records'])
        if name == 'oracle':
            value = flawed(qtype, records, params)
        elif name == 'random_guess':
            value = _random_value(qtype, records, _rng(row['id']))
        elif name == 'constant_majority':
            return self.majority[qtype]
        elif name == 'first_record':
            value = _positional(qtype, records[0], params)
        elif name == 'last_record':
            value = _positional(qtype, records[-1], params)
        elif name.startswith('ignore_'):
            value = flawed(qtype, records, params, blind=name[len('ignore_'):])
        elif name == 'subject_only':
            value = flawed(qtype, records, params, pair='subject')
        elif name == 'object_only':
            value = flawed(qtype, records, params, pair='object')
        elif name == 'feature_bag':
            value = flawed(qtype, records, params, bag=True)
        elif name == 'unbound_join':
            value = flawed(qtype, records, params, unbound=True)
        elif name == 'role_swap':
            swapped = [dict(r, subject=r['object'], object=r['subject']) for r in records]
            value = flawed(qtype, swapped, params, pair='both')
        else:
            raise ValueError(name)
        return Q.canonical(qtype, value)
