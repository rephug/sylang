"""The record schema, identifier grammar and the grouped-format ordering rule.

A record has the pilot's seven fields (``pretrained_pilot/contract.md``). A *case* is an ordered list of records
that obeys the pilot's two structural invariants:

1. no (subject, object) pair occurs twice and no record has subject == object;
2. records sharing a subject are adjacent (the rule that makes ``sylang2g``, one line per subject, a lossless
   re-layout: record order is recovered as line order, then item order within a line).

Every case produced or transformed by this package is checked with ``check_case``.
"""
from __future__ import annotations

import hashlib
import random

RELATIONS = ('see', 'help', 'contain')
POLARITIES = ('positive', 'negative')
TENSES = ('past', 'present', 'future')
ASPECTS = ('simple', 'progressive', 'completed')
EVIDENCES = ('unspecified', 'direct', 'reported', 'inferred')
FEATURES = ('relation', 'polarity', 'tense', 'aspect', 'evidence')
VALUES = {'relation': RELATIONS, 'polarity': POLARITIES, 'tense': TENSES, 'aspect': ASPECTS, 'evidence': EVIDENCES}
DEFAULTS = {'polarity': 'positive', 'tense': 'present', 'aspect': 'simple', 'evidence': 'unspecified'}
FIELDS = ('subject', 'relation', 'object', 'polarity', 'tense', 'aspect', 'evidence')

ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789'
LETTERS = ALPHABET[:26]
ID_LENGTHS = (2, 3, 4, 5)
# Words that may not be identifiers: the pilot's reserved words, every feature value, and every word the
# controlled-English renderer or the instructions use, so that no identifier can be read as vocabulary.
RESERVED = frozenset({
    'if', 'none', 'not', 'by', 'and', 'or', 'the', 'is', 'was', 'has', 'had', 'will', 'be', 'been', 'have',
    'does', 'did', 'do', 'sees', 'saw', 'seen', 'seeing', 'helps', 'helped', 'helping', 'contains', 'contained',
    'containing', 'report', 'direct', 'some', 'that', 'same', 'which', 'what', 'z', 'y', 'x',
    *RELATIONS, *POLARITIES, *TENSES, *ASPECTS, *EVIDENCES,
})


def is_identifier(text: str) -> bool:
    return (isinstance(text, str) and 2 <= len(text) <= 5 and all(c in ALPHABET for c in text)
            and any(c in LETTERS for c in text) and text not in RESERVED)


def record(subject, relation, obj, polarity='positive', tense='present', aspect='simple', evidence='unspecified'):
    return {'subject': subject, 'relation': relation, 'object': obj, 'polarity': polarity, 'tense': tense,
            'aspect': aspect, 'evidence': evidence}


def check_case(records: list[dict]) -> None:
    """Raise AssertionError when a case violates the schema or the two structural invariants."""
    assert records, 'empty case'
    pairs = set()
    for r in records:
        assert tuple(sorted(r)) == tuple(sorted(FIELDS)), r
        for f in FEATURES:
            assert r[f] in VALUES[f], (f, r)
        assert is_identifier(r['subject']) and is_identifier(r['object']), r
        assert r['subject'] != r['object'], r
        pair = (r['subject'], r['object'])
        assert pair not in pairs, f'repeated pair {pair}'
        pairs.add(pair)
    assert subjects_adjacent(records), 'records sharing a subject are not adjacent'


def subjects_adjacent(records: list[dict]) -> bool:
    seen, previous = set(), None
    for r in records:
        s = r['subject']
        if s != previous:
            if s in seen:
                return False
            seen.add(s)
            previous = s
    return True


def groups(records: list[dict]) -> list[list[dict]]:
    """Subject groups in order (requires ``subjects_adjacent``)."""
    out: list[list[dict]] = []
    for r in records:
        if out and out[-1][0]['subject'] == r['subject']:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def group_normalise(records: list[dict]) -> list[dict]:
    """Stable re-layout: each subject's records moved directly after that subject's first record."""
    order: dict[str, list[dict]] = {}
    for r in records:
        order.setdefault(r['subject'], []).append(r)
    return [r for rs in order.values() for r in rs]


def entities(records: list[dict]) -> list[str]:
    """Identifiers in order of first occurrence (subject before object within a record)."""
    out: dict[str, None] = {}
    for r in records:
        out.setdefault(r['subject'], None)
        out.setdefault(r['object'], None)
    return list(out)


def id_split(identifier: str) -> str:
    """The pilot's v3 identifier hash split (``study3.split_of``): ``train`` | ``dev`` | ``test``.
    Benchmark dev identifiers are drawn from the pilot's dev bucket and test identifiers from its test
    bucket, so no benchmark identifier is a pilot training identifier."""
    import pilot_compat
    return pilot_compat.study3().split_of(identifier)


class IdentifierPool:
    """Fresh identifiers of one split, drawn from a ``random.Random`` stream."""

    def __init__(self, split: str, rng: random.Random):
        assert split in ('dev', 'test'), split
        self.split, self.rng = split, rng
        self._split_cache: dict[str, str] = {}

    def _split_of(self, identifier: str) -> str:
        if identifier not in self._split_cache:
            self._split_cache[identifier] = id_split(identifier)
        return self._split_cache[identifier]

    def fresh(self, used: set, length: int | None = None) -> str:
        while True:
            n = self.rng.choice(ID_LENGTHS) if length is None else length
            candidate = ''.join(self.rng.choice(ALPHABET) for _ in range(n))
            if candidate in used or not is_identifier(candidate) or self._split_of(candidate) != self.split:
                continue
            used.add(candidate)
            return candidate


def structural_signature(records: list[dict], params: dict | None = None) -> str:
    """A hash invariant to renaming identifiers and to reordering records: the sorted multiset of per-record
    rows (the five features plus the out- and in-degree of the record's subject and object), plus the
    question's non-identifier parameters. Two cases that are the same up to renaming and reordering always
    collide; unrelated cases rarely do. Used conservatively to keep near-duplicate families out of two
    splits (a collision is redrawn, even if it is a false positive)."""
    out_deg: dict[str, int] = {}
    in_deg: dict[str, int] = {}
    for r in records:
        out_deg[r['subject']] = out_deg.get(r['subject'], 0) + 1
        in_deg[r['object']] = in_deg.get(r['object'], 0) + 1
    rows = sorted((*(r[f] for f in FEATURES), out_deg.get(r['subject'], 0), in_deg.get(r['subject'], 0),
                   out_deg.get(r['object'], 0), in_deg.get(r['object'], 0)) for r in records)
    p = None
    if params is not None:
        ids = set(entities(records))
        p = sorted((k, v) for k, v in params.items() if not (isinstance(v, str) and v in ids)
                   and k not in ('subject', 'object', 'x'))
    return hashlib.sha256(repr((rows, p)).encode()).hexdigest()
