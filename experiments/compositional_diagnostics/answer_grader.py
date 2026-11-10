"""Full-answer grader for the six question types.

``grade(qtype, expected, output) -> {'correct', 'invalid', 'reason', 'fenced'}``. The whole output, after
stripping whitespace and at most one Markdown code fence around all of it (the pilot's 2026-10-22 amendment,
counted in ``fenced``), must be exactly one JSON value of the question's shape:

| qtype | shape |
|---|---|
| tense, aspect, evidence | a JSON string |
| profile | an object with exactly the keys relation, polarity, tense, aspect, evidence, string values |
| filter | a list of 2-element lists of strings, no pair twice |
| join | a list of strings, no string twice |

The answer is correct only when the *whole* value equals the expected one: every field of a profile, every
pair of a filter answer (as a set: order is free, duplicates are a shape error) and every Z of a join answer.
There is no partial credit. Reasons: ``ok``, ``wrong`` (right shape, other value); invalid: ``empty``,
``parse`` (not JSON), ``extra_text`` (a value plus other text), ``shape``, ``duplicate``.
"""
from __future__ import annotations

import json
import re

import questions as Q

_FENCE = re.compile(r'\A```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?```\Z', re.S)


class _DuplicateKey(ValueError):
    pass


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKey()
    return dict(pairs)


_DECODER = json.JSONDecoder(object_pairs_hook=_no_duplicate_keys)


def unfence(text: str) -> tuple[str, bool]:
    m = _FENCE.match(text.strip())
    return (m.group(1).strip(), True) if m else (text.strip(), False)


def _shape(qtype, value):
    if qtype in Q.SINGLE_FIELD:
        return (value, 'ok') if isinstance(value, str) else (None, 'shape')
    if qtype == 'profile':
        if not (isinstance(value, dict) and sorted(value) == sorted(Q.PROFILE_KEYS)
                and all(isinstance(v, str) for v in value.values())):
            return None, 'shape'
        return tuple(value[k] for k in Q.PROFILE_KEYS), 'ok'
    if qtype == 'filter':
        if not (isinstance(value, list) and all(isinstance(p, list) and len(p) == 2
                                                and all(isinstance(x, str) for x in p) for p in value)):
            return None, 'shape'
        pairs = [tuple(p) for p in value]
        if len(set(pairs)) != len(pairs):
            return None, 'duplicate'
        return frozenset(pairs), 'ok'
    if qtype == 'join':
        if not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            return None, 'shape'
        if len(set(value)) != len(value):
            return None, 'duplicate'
        return frozenset(value), 'ok'
    raise ValueError(qtype)


def parse(qtype: str, text) -> tuple:
    """(canonical value, 'ok') or (None, reason)."""
    if qtype not in Q.QTYPES:
        raise ValueError(qtype)
    text, _ = unfence(text if isinstance(text, str) else '')
    if not text:
        return None, 'empty'
    try:
        value, end = _DECODER.raw_decode(text)
    except _DuplicateKey:
        return None, 'shape'
    except ValueError:
        for opener in '["{':
            at = text.find(opener)
            while at > 0:
                try:
                    json.JSONDecoder().raw_decode(text, at)
                    return None, 'extra_text'
                except ValueError:
                    at = text.find(opener, at + 1)
        return None, 'parse'
    if text[end:].strip():
        return None, 'extra_text'
    return _shape(qtype, value)


def grade(qtype: str, expected: str, output) -> dict:
    want, reason = parse(qtype, expected)
    if want is None:
        raise ValueError(f'expected answer is not a valid {qtype} answer ({reason}): {expected!r}')
    fenced = unfence(output if isinstance(output, str) else '')[1]
    got, reason = parse(qtype, output)
    if got is None:
        return {'correct': False, 'invalid': True, 'reason': reason, 'fenced': fenced}
    if got == want:
        return {'correct': True, 'invalid': False, 'reason': 'ok', 'fenced': fenced}
    return {'correct': False, 'invalid': False, 'reason': 'wrong', 'fenced': fenced}


def canonical_output(qtype: str, output):
    """The parsed value of an output (hashable) or None: used for the uncredited raw-invariance diagnostic."""
    value, _ = parse(qtype, output)
    return value
