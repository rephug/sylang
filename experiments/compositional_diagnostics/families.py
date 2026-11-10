"""Matched test families: one base case and five controlled variants per question.

A family is built in one ``random.Random`` stream and one identifier split, before anything is rendered. Members:

| variant | edit | the answer must |
|---|---|---|
| ``base`` | none | (reference) |
| ``relevant`` | one feature of one record that the question depends on | change |
| ``irrelevant`` | one feature of one record that the question does not depend on (for the single-feature questions: a *different feature of the queried record itself*) | stay the same |
| ``rename`` | every identifier replaced, consistently, by a fresh identifier of the same length | be the base answer renamed |
| ``reorder`` | the subject groups permuted and the records permuted within each group (only orders the grouped format can express: records sharing a subject stay adjacent) | stay the same |
| ``distractor`` | one or two records added that share identifiers with the question but do not satisfy it | stay the same |

The generator checks every declared expectation with the oracle and redraws the family when one fails, and
checks that ``relevant`` and ``irrelevant`` differ from the base in exactly one field of one record, that
``distractor`` only adds records, that ``reorder`` is a permutation and ``rename`` a bijection.

Built-in traps (so that the weak solvers in ``solvers.py`` meet the case they cannot handle):

* pair questions: a *subject decoy* (same subject, other object) and an *object decoy* (other subject, same
  object), each with a different value of the asked feature, so neither identifier alone finds the record;
* filter: two half-matching records that share one subject, one meeting each condition, so a solver that pools
  features across a subject's records accepts them;
* join: a chain broken by the wrong tense, a link record whose subject is not reached from x (unbound), and a
  negated link, so ignoring tense, ignoring the shared Y or ignoring polarity each adds a wrong Z.
"""
from __future__ import annotations

import random

import questions as Q
import schema

VARIANTS = ('base', 'relevant', 'irrelevant', 'rename', 'reorder', 'distractor')
EXPECT = {'base': 'reference', 'relevant': 'changes', 'irrelevant': 'same', 'rename': 'renamed', 'reorder': 'same',
          'distractor': 'same'}
P_SUBJECT_REUSE = 0.35
P_OBJECT_REUSE = 0.2


# Balanced designs, cycled in shuffled blocks per (split, question type): the asked feature's value of the base
# record (single-feature types), the number of matching records (filter), the number of chains (join).
DESIGN_CYCLE = {'tense': schema.TENSES, 'aspect': schema.ASPECTS, 'evidence': schema.EVIDENCES, 'profile': (None,),
                'filter': (0, 1, 1, 2, 2, 3), 'join': (0, 1, 1, 2)}


def design_schedule(qtype: str, n: int, rng: random.Random) -> list:
    out = []
    while len(out) < n:
        block = list(DESIGN_CYCLE[qtype])
        rng.shuffle(block)
        out += block
    return out[:n]


class Redraw(Exception):
    """The family could not be built as designed; draw a new one from the same stream."""


class Builder:
    def __init__(self, rng: random.Random, pool: schema.IdentifierPool):
        self.rng, self.pool = rng, pool
        self.used: set = set()
        self.records: list[dict] = []

    def fresh(self, length=None) -> str:
        return self.pool.fresh(self.used, length)

    def features(self, **forced) -> dict:
        out = {f: self.rng.choice(schema.VALUES[f]) for f in schema.FEATURES}
        out.update(forced)
        return out

    def other(self, field: str, value: str) -> str:
        return self.rng.choice([v for v in schema.VALUES[field] if v != value])

    def pairs(self) -> set:
        return {(r['subject'], r['object']) for r in self.records}

    def add(self, subject, obj, **features) -> dict:
        assert subject != obj and (subject, obj) not in self.pairs(), (subject, obj)
        f = self.features(**features)
        r = schema.record(subject, f['relation'], obj, f['polarity'], f['tense'], f['aspect'], f['evidence'])
        self.records.append(r)
        return r

    def filler(self, reject=None) -> dict:
        """A random record; ``reject(features)`` True redraws the features."""
        subjects = list(dict.fromkeys(r['subject'] for r in self.records))
        ids = schema.entities(self.records)
        for _ in range(100):
            subject = self.rng.choice(subjects) if subjects and self.rng.random() < P_SUBJECT_REUSE else None
            obj = None
            if self.rng.random() < P_OBJECT_REUSE and ids:
                obj = self.rng.choice(ids)
            subject = subject or self.fresh()
            obj = obj or self.fresh()
            if subject == obj or (subject, obj) in self.pairs():
                continue
            f = self.features()
            if reject is not None and reject(f):
                continue
            return self.add(subject, obj, **f)
        raise Redraw('no filler')

    def layout(self) -> list[dict]:
        """A uniformly random order of the subject groups, each group in a uniformly random internal order."""
        grouped = schema.group_normalise(self.records)
        gs = [self.rng.sample(g, len(g)) for g in schema.groups(grouped)]
        self.rng.shuffle(gs)
        return [r for g in gs for r in g]


# ---------------------------------------------------------------------------
# helpers on cases

def find(records, subject, obj) -> int:
    return next(i for i, r in enumerate(records) if r['subject'] == subject and r['object'] == obj)


def edited(records, index, field, value) -> list[dict]:
    out = [dict(r) for r in records]
    assert out[index][field] != value
    out[index][field] = value
    return out


def insert_record(rng, records, new) -> list[dict]:
    """Insert a record keeping subject groups adjacent: inside its subject's group at a random slot, or as a new
    group at a random group boundary."""
    out = [dict(r) for r in records]
    slots = [i for i, r in enumerate(out) if r['subject'] == new['subject']]
    if slots:
        position = rng.randint(slots[0], slots[-1] + 1)
    else:
        boundaries = [0] + [i for i in range(1, len(out) + 1) if i == len(out) or out[i]['subject'] != out[i - 1]['subject']]
        position = rng.choice(boundaries)
    out.insert(position, dict(new))
    return out


def reorder(rng, records) -> list[dict]:
    gs = [list(g) for g in schema.groups(records)]
    for _ in range(50):
        order = gs[:]
        rng.shuffle(order)
        order = [rng.sample(g, len(g)) for g in order]
        out = [dict(r) for g in order for r in g]
        if out != records:
            return out
    raise Redraw('no distinct grouped order')


def rename(rng, pool, records, used: set) -> tuple[list[dict], dict]:
    mapping = {}
    for e in schema.entities(records):
        mapping[e] = pool.fresh(used, len(e))
    out = [dict(r, subject=mapping[r['subject']], object=mapping[r['object']]) for r in records]
    return out, mapping


def single_edit(a, b):
    """(index, field) when b differs from a in exactly one field of one record, else None."""
    if len(a) != len(b):
        return None
    diffs = [(i, f) for i, (x, y) in enumerate(zip(a, b)) for f in schema.FIELDS if x[f] != y[f]]
    return diffs[0] if len(diffs) == 1 else None


# ---------------------------------------------------------------------------
# base cases and their edits, per question type

def _pair_family(b: Builder, qtype: str, design=None):
    rng = b.rng
    k = rng.randint(4, 8)
    x, y = b.fresh(), b.fresh()
    if qtype == 'profile':
        b.add(x, y)
        target = b.records[-1]

        def decoy_features():
            fields = rng.sample(Q.PROFILE_KEYS, 2)
            return {f: b.other(f, target[f]) for f in fields}
    else:
        field = Q.SINGLE_FIELD[qtype]
        b.add(x, y, **({field: design} if design is not None else {}))
        target = b.records[-1]

        def decoy_features():
            return {field: b.other(field, target[field])}
    b.add(x, b.fresh(), **decoy_features())          # subject decoy
    b.add(b.fresh(), y, **decoy_features())          # object decoy
    while len(b.records) < k:
        b.filler()
    records = b.layout()
    params = {'subject': x, 'object': y}
    t = find(records, x, y)

    def relevant():
        f = Q.SINGLE_FIELD.get(qtype) or rng.choice(Q.PROFILE_KEYS)
        return edited(records, t, f, b.other(f, records[t][f]))

    def irrelevant():
        if qtype == 'profile':  # every feature of the target matters: edit another record
            i = rng.choice([j for j in range(len(records)) if j != t])
            f = rng.choice(schema.FEATURES)
            return edited(records, i, f, b.other(f, records[i][f]))
        f = rng.choice([g for g in schema.FEATURES if g != Q.SINGLE_FIELD[qtype]])
        return edited(records, t, f, b.other(f, records[t][f]))

    def distractor():
        f = decoy_features()
        new1 = schema.record(x, f.get('relation', rng.choice(schema.RELATIONS)), b.fresh())
        for g in schema.FEATURES:
            new1[g] = f.get(g, rng.choice(schema.VALUES[g]))
        out = insert_record(rng, records, new1)
        f = decoy_features()
        new2 = schema.record(b.fresh(), 'see', y)
        for g in schema.FEATURES:
            new2[g] = f.get(g, rng.choice(schema.VALUES[g]))
        return insert_record(rng, out, new2)

    return records, params, relevant, irrelevant, distractor


def _filter_family(b: Builder, design=None):
    rng = b.rng
    f1 = rng.choice(Q.DEPENDENT)
    f2 = rng.choice([f for f in Q.FILTER_FIELDS if f != f1])
    v1, v2 = rng.choice(schema.VALUES[f1]), rng.choice(schema.VALUES[f2])
    conditions = [[f1, v1], [f2, v2]]
    rng.shuffle(conditions)
    params = {'conditions': conditions}
    n_match = design if design is not None else rng.choices((0, 1, 2, 3), weights=(15, 35, 30, 20))[0]
    k = rng.randint(max(5, n_match + 3), 9)

    def matches(f):
        return f[f1] == v1 and f[f2] == v2

    for _ in range(n_match):
        subjects = list(dict.fromkeys(r['subject'] for r in b.records))
        s = rng.choice(subjects) if subjects and rng.random() < P_SUBJECT_REUSE else b.fresh()
        b.add(s, b.fresh(), **{f1: v1, f2: v2})
    bag = b.fresh()  # the feature-bag trap: one subject, one record per condition
    b.add(bag, b.fresh(), **{f1: v1, f2: b.other(f2, v2)})
    b.add(bag, b.fresh(), **{f1: b.other(f1, v1), f2: v2})
    while len(b.records) < k:
        b.filler(reject=matches)
    records = b.layout()

    def full(i):
        return records[i][f1] == v1 and records[i][f2] == v2

    def half(i):
        return (records[i][f1] == v1) != (records[i][f2] == v2)

    def relevant():
        shrink = [i for i in range(len(records)) if full(i)]
        grow = [i for i in range(len(records)) if half(i)]
        if shrink and (not grow or rng.random() < 0.5):
            i = rng.choice(shrink)
            f, v = rng.choice(((f1, v1), (f2, v2)))
            return edited(records, i, f, b.other(f, v))
        i = rng.choice(grow)
        f, v = (f2, v2) if records[i][f1] == v1 else (f1, v1)
        return edited(records, i, f, v)

    def irrelevant():
        candidates = [i for i in range(len(records)) if full(i)] or [i for i in range(len(records)) if half(i)]
        i = rng.choice(candidates)
        f = rng.choice([g for g in schema.FEATURES if g not in (f1, f2)])
        return edited(records, i, f, b.other(f, records[i][f]))

    def distractor():
        subjects = list(dict.fromkeys(r['subject'] for r in records))
        s = rng.choice(subjects)
        meet = rng.choice((1, 2))
        forced = {f1: v1, f2: b.other(f2, v2)} if meet == 1 else {f1: b.other(f1, v1), f2: v2}
        f = b.features(**forced)
        new = schema.record(s, f['relation'], b.fresh(), f['polarity'], f['tense'], f['aspect'], f['evidence'])
        return insert_record(rng, records, new)

    return records, params, relevant, irrelevant, distractor


def _join_family(b: Builder, design=None):
    rng = b.rng
    x = b.fresh()
    rel_a, rel_b = rng.choice(schema.RELATIONS), rng.choice(schema.RELATIONS)
    tense_a = rng.choice(schema.TENSES)
    params = {'x': x, 'rel_a': rel_a, 'tense_a': tense_a, 'rel_b': rel_b}
    n_chain = design if design is not None else rng.choices((0, 1, 2), weights=(15, 55, 30))[0]
    chains = []
    for _ in range(n_chain):
        y, z = b.fresh(), b.fresh()
        b.add(x, y, relation=rel_a, polarity='positive', tense=tense_a)
        b.add(y, z, relation=rel_b, polarity='positive')
        chains.append((y, z))
    y1 = b.fresh()                                   # broken by tense
    b.add(x, y1, relation=rel_a, polarity='positive', tense=b.other('tense', tense_a))
    b.add(y1, b.fresh(), relation=rel_b, polarity='positive')
    b.add(b.fresh(), b.fresh(), relation=rel_b, polarity='positive')   # unbound link
    if chains:                                       # negated link from a valid Y
        b.add(chains[0][0], b.fresh(), relation=rel_b, polarity='negative')
    else:                                            # negated first hop
        y2 = b.fresh()
        b.add(x, y2, relation=rel_a, polarity='negative', tense=tense_a)
        b.add(y2, b.fresh(), relation=rel_b, polarity='positive')
    k = rng.randint(max(6, len(b.records)), 10)
    while len(b.records) < k:
        b.filler()
    records = b.layout()

    def hop1(i):
        r = records[i]
        return r['subject'] == x and r['relation'] == rel_a

    def relevant():
        shrink, grow = [], []
        for y, z in chains:
            shrink += [(find(records, x, y), 'tense'), (find(records, y, z), 'relation'),
                       (find(records, y, z), 'polarity'), (find(records, x, y), 'relation')]
        for i, r in enumerate(records):
            if hop1(i) and r['polarity'] == 'positive' and r['tense'] != tense_a:
                grow.append((i, 'tense->'))
            if hop1(i) and r['polarity'] == 'negative' and r['tense'] == tense_a:
                grow.append((i, 'polarity->'))
            if r['relation'] == rel_b and r['polarity'] == 'negative' and r['subject'] in [y for y, _ in chains]:
                grow.append((i, 'polarity->'))
        rng.shuffle(shrink)
        rng.shuffle(grow)
        options = shrink + grow if (shrink and rng.random() < 0.5) or not grow else grow + shrink
        base_answer = Q.oracle('join', records, params)
        for i, what in options:
            if what == 'tense->':
                out = edited(records, i, 'tense', tense_a)
            elif what == 'polarity->':
                out = edited(records, i, 'polarity', 'positive')
            elif what == 'polarity':
                out = edited(records, i, 'polarity', 'negative')
            else:
                out = edited(records, i, what, b.other(what, records[i][what]))
            if Q.oracle('join', out, params) != base_answer:
                return out
        raise Redraw('no relevant join edit')

    def irrelevant():
        candidates = [find(records, x, y) for y, _ in chains] + [find(records, y, z) for y, z in chains]
        candidates = candidates or [i for i in range(len(records)) if hop1(i)]
        i = rng.choice(candidates)
        f = rng.choice(('aspect', 'evidence'))
        return edited(records, i, f, b.other(f, records[i][f]))

    def distractor():
        f = b.features(relation=rel_b, polarity='positive')
        new = schema.record(b.fresh(), rel_b, b.fresh(), 'positive', f['tense'], f['aspect'], f['evidence'])
        out = insert_record(rng, records, new)
        y3 = b.fresh()
        f = b.features(relation=b.other('relation', rel_a), polarity='positive', tense=tense_a)
        out = insert_record(rng, out, schema.record(x, f['relation'], y3, 'positive', tense_a, f['aspect'], f['evidence']))
        f = b.features(relation=rel_b, polarity='positive')
        return insert_record(rng, out, schema.record(y3, rel_b, b.fresh(), 'positive', f['tense'], f['aspect'],
                                                     f['evidence']))

    return records, params, relevant, irrelevant, distractor


def build_family(qtype: str, rng: random.Random, pool: schema.IdentifierPool, design=None) -> dict:
    """One family (arm-independent): ``{'qtype', 'params', 'members': [{variant, records, params, answer,
    expect, mapping?}]}``; raises ``Redraw`` when the design cannot be met."""
    b = Builder(rng, pool)
    if qtype in Q.PAIR_QTYPES:
        records, params, relevant, irrelevant, distractor = _pair_family(b, qtype, design)
    elif qtype == 'filter':
        records, params, relevant, irrelevant, distractor = _filter_family(b, design)
    elif qtype == 'join':
        records, params, relevant, irrelevant, distractor = _join_family(b, design)
    else:
        raise ValueError(qtype)
    schema.check_case(records)
    base_value = Q.oracle(qtype, records, params)
    members = [{'variant': 'base', 'records': records, 'params': params}]
    members.append({'variant': 'relevant', 'records': relevant(), 'params': params})
    members.append({'variant': 'irrelevant', 'records': irrelevant(), 'params': params})
    renamed, mapping = rename(rng, pool, records, b.used)
    members.append({'variant': 'rename', 'records': renamed, 'params': Q.rename_params(params, mapping),
                    'mapping': mapping})
    members.append({'variant': 'reorder', 'records': reorder(rng, records), 'params': params})
    members.append({'variant': 'distractor', 'records': distractor(), 'params': params})
    for m in members:
        schema.check_case(m['records'])
        m['expect'] = EXPECT[m['variant']]
        value = Q.oracle(qtype, m['records'], m['params'])
        m['answer'] = Q.canonical(qtype, value)
        ok = {'base': True,
              'relevant': value != base_value,
              'irrelevant': value == base_value,
              'rename': value == Q.rename_value(qtype, base_value, mapping),
              'reorder': value == base_value,
              'distractor': value == base_value}[m['variant']]
        if not ok:
            raise Redraw(f'{m["variant"]} expectation failed')
    # structural checks of the edits
    by = {m['variant']: m['records'] for m in members}
    for v in ('relevant', 'irrelevant'):
        if single_edit(records, by[v]) is None:
            raise AssertionError(f'{v} is not a single-field edit')
    edit_field = single_edit(records, by['irrelevant'])[1]
    if qtype != 'profile' and edit_field in Q.relevant_features(qtype, params):
        raise AssertionError('irrelevant edit touched a relevant feature')
    key = lambda r: tuple(r[f] for f in schema.FIELDS)
    assert sorted(map(key, by['reorder'])) == sorted(map(key, records)) and by['reorder'] != records
    assert set(map(key, records)) <= set(map(key, by['distractor'])) and len(by['distractor']) > len(records)
    assert len(set(mapping.values())) == len(mapping) and not set(mapping.values()) & set(mapping)
    return {'qtype': qtype, 'params': params, 'members': members}
