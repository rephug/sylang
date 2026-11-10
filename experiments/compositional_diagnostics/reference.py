"""An independent reference answerer, used only to verify ``questions.oracle``.

It shares no code with the oracle. Records are positional tuples in ``FIELD_ORDER``; every question is
compiled into a conjunctive query (a list of atoms, each a tuple of constants and ``?variables``, plus output
variables), and the query is answered by brute-force backtracking over all records for every atom. The
answer is then put in the canonical JSON form by its own serialiser.
"""
from __future__ import annotations

import json

FIELD_ORDER = ('subject', 'relation', 'object', 'polarity', 'tense', 'aspect', 'evidence')
SLOT = {name: i for i, name in enumerate(FIELD_ORDER)}


def to_tuples(records) -> list[tuple]:
    return [tuple(r[name] for name in FIELD_ORDER) for r in records]


def _var(term) -> bool:
    return isinstance(term, str) and term.startswith('?')


def _atom(**fixed) -> tuple:
    """A 7-slot pattern: given slots are constants or ``?vars``, the others are wildcards (None)."""
    pattern = [None] * len(FIELD_ORDER)
    for name, term in fixed.items():
        pattern[SLOT[name]] = term
    return tuple(pattern)


def compile_query(qtype: str, params: dict):
    """(atoms, output variables, kind)."""
    if qtype in ('tense', 'aspect', 'evidence'):
        return [_atom(subject=params['subject'], object=params['object'], **{qtype: '?v'})], ('?v',), 'unique'
    if qtype == 'profile':
        names = ('relation', 'polarity', 'tense', 'aspect', 'evidence')
        return ([_atom(subject=params['subject'], object=params['object'], **{n: '?' + n for n in names})],
                tuple('?' + n for n in names), 'unique')
    if qtype == 'filter':
        fixed = {'subject': '?s', 'object': '?o'}
        for field, value in params['conditions']:
            fixed[field] = value
        return [_atom(**fixed)], ('?s', '?o'), 'set'
    if qtype == 'join':
        return ([_atom(subject=params['x'], relation=params['rel_a'], object='?y', polarity='positive',
                       tense=params['tense_a']),
                 _atom(subject='?y', relation=params['rel_b'], object='?z', polarity='positive')], ('?z',), 'set')
    raise ValueError(qtype)


def _solve(atoms, facts, binding):
    if not atoms:
        yield dict(binding)
        return
    first, rest = atoms[0], atoms[1:]
    for fact in facts:
        new = dict(binding)
        ok = True
        for term, value in zip(first, fact):
            if term is None:
                continue
            if _var(term):
                if term in new and new[term] != value:
                    ok = False
                    break
                new[term] = value
            elif term != value:
                ok = False
                break
        if ok:
            yield from _solve(rest, facts, new)


def reference_answer(qtype: str, records, params: dict) -> str:
    atoms, outputs, kind = compile_query(qtype, params)
    facts = to_tuples(records)
    rows = {tuple(b[v] for v in outputs) for b in _solve(atoms, facts, {})}
    if kind == 'unique':
        if len(list(_solve(atoms, facts, {}))) != 1:
            raise ValueError('the queried pair does not identify exactly one record')
        (row,) = rows
        if qtype == 'profile':
            keys = ('relation', 'polarity', 'tense', 'aspect', 'evidence')
            return '{' + ','.join(f'"{k}":"{v}"' for k, v in zip(keys, row)) + '}'
        return f'"{row[0]}"'
    if qtype == 'filter':
        return '[' + ','.join(f'["{s}","{o}"]' for s, o in sorted(rows)) + ']'
    return '[' + ','.join(f'"{z}"' for (z,) in sorted(rows)) + ']'


def check_json(text: str) -> str:
    """Round-trip through json to make sure the hand-built string is valid JSON."""
    json.loads(text)
    return text
