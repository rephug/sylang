"""Question types, their instruction text, canonical answers and the oracle.

Six question types. ``params`` is the question's structured form; ``instruction(qtype, params)`` its English text,
identical in every arm; ``oracle(qtype, records, params)`` the answer value; ``canonical(qtype, value)`` the
answer string (sorted lists, so the canonical string of a set answer is unique).

| qtype | params | answer value | answer JSON |
|---|---|---|---|
| ``tense`` | subject, object | the tense of the one record with that pair | ``"past"`` |
| ``aspect`` | subject, object | its aspect | ``"completed"`` |
| ``evidence`` | subject, object | its evidence | ``"reported"`` |
| ``profile`` | subject, object | all five features of that record | ``{"relation":..,"polarity":..,"tense":..,"aspect":..,"evidence":..}`` |
| ``filter`` | conditions: two (field, value) on distinct features, at least one of tense/aspect/evidence | the set of (subject, object) of records meeting both conditions | ``[["a1","b2"],...]`` sorted, ``[]`` if none |
| ``join`` | x, rel_a, tense_a, rel_b | the set of Z with records (x, rel_a, Y, positive, tense_a) and (Y, rel_b, Z, positive) for the same Y | ``["c3",...]`` sorted, ``[]`` if none |

The pair questions name a pair that occurs in exactly one record (the generator guarantees it; the oracle
asserts it). ``join`` is bounded: exactly two records, chained through one shared entity.
"""
from __future__ import annotations

import json

import schema

QTYPES = ('tense', 'aspect', 'evidence', 'profile', 'filter', 'join')
PAIR_QTYPES = ('tense', 'aspect', 'evidence', 'profile')
SINGLE_FIELD = {'tense': 'tense', 'aspect': 'aspect', 'evidence': 'evidence'}
PROFILE_KEYS = ('relation', 'polarity', 'tense', 'aspect', 'evidence')
FILTER_FIELDS = ('relation', 'polarity', 'tense', 'aspect', 'evidence')
DEPENDENT = ('tense', 'aspect', 'evidence')

# The features each question type's answer depends on (beyond identifiers and binding): a change to any other
# feature of any record cannot change the answer. Used by the generator to choose irrelevant edits and checked
# exhaustively by the tests.
def relevant_features(qtype: str, params: dict) -> tuple[str, ...]:
    if qtype in SINGLE_FIELD:
        return (SINGLE_FIELD[qtype],)
    if qtype == 'profile':
        return PROFILE_KEYS
    if qtype == 'filter':
        return tuple(f for f, _ in params['conditions'])
    if qtype == 'join':
        return ('relation', 'polarity', 'tense')
    raise ValueError(qtype)


def _q(identifier: str) -> str:
    return f'"{identifier}"'


def _choices(values) -> str:
    quoted = [f'"{v}"' for v in values]
    return ', '.join(quoted[:-1]) + ' or ' + quoted[-1]


def instruction(qtype: str, params: dict) -> str:
    if qtype in SINGLE_FIELD:
        field = SINGLE_FIELD[qtype]
        return (f'What is the {field} of the record whose subject is {_q(params["subject"])} and whose object is '
                f'{_q(params["object"])}? Answer with one JSON string: {_choices(schema.VALUES[field])}.')
    if qtype == 'profile':
        return (f'Describe the record whose subject is {_q(params["subject"])} and whose object is '
                f'{_q(params["object"])}. Answer with one JSON object with exactly the keys relation, polarity, '
                f'tense, aspect and evidence, each with its value.')
    if qtype == 'filter':
        (f1, v1), (f2, v2) = params['conditions']
        return (f'List the subject and object of every record whose {f1} is {v1} and whose {f2} is {v2}. Answer '
                f'with one JSON list of [subject, object] pairs in any order, [] if there is none.')
    if qtype == 'join':
        return (f'Find every Z for which some Y exists such that one record says {_q(params["x"])} '
                f'{params["rel_a"]}s Y with tense {params["tense_a"]} and another record says that same Y '
                f'{params["rel_b"]}s Z. Only positive records count; aspect and evidence do not matter. Answer '
                f'with one JSON list of the Z identifiers in any order, [] if there is none.')
    raise ValueError(qtype)


def _the_record(records: list[dict], subject: str, obj: str) -> dict:
    matches = [r for r in records if r['subject'] == subject and r['object'] == obj]
    assert len(matches) == 1, (subject, obj, len(matches))
    return matches[0]


def oracle(qtype: str, records: list[dict], params: dict):
    """The answer value (str, dict, list of [s, o] or list of ids) of a question on a case."""
    if qtype in PAIR_QTYPES:
        r = _the_record(records, params['subject'], params['object'])
        if qtype == 'profile':
            return {k: r[k] for k in PROFILE_KEYS}
        return r[SINGLE_FIELD[qtype]]
    if qtype == 'filter':
        (f1, v1), (f2, v2) = params['conditions']
        return sorted([r['subject'], r['object']] for r in records if r[f1] == v1 and r[f2] == v2)
    if qtype == 'join':
        hop1 = [r['object'] for r in records if r['subject'] == params['x'] and r['relation'] == params['rel_a']
                and r['polarity'] == 'positive' and r['tense'] == params['tense_a']]
        zs = {r['object'] for r in records for y in hop1 if r['subject'] == y and r['relation'] == params['rel_b']
              and r['polarity'] == 'positive'}
        return sorted(zs)
    raise ValueError(qtype)


def canonical(qtype: str, value) -> str:
    if qtype == 'profile':
        return json.dumps({k: value[k] for k in PROFILE_KEYS}, separators=(',', ':'))
    if qtype in ('filter', 'join'):
        return json.dumps(sorted(value), separators=(',', ':'))
    return json.dumps(value)


def answer(qtype: str, records: list[dict], params: dict) -> str:
    return canonical(qtype, oracle(qtype, records, params))


def rename_value(qtype: str, value, mapping: dict):
    """The answer value after renaming identifiers with ``mapping`` (only set answers carry identifiers)."""
    if qtype == 'filter':
        return sorted([mapping[s], mapping[o]] for s, o in value)
    if qtype == 'join':
        return sorted(mapping[z] for z in value)
    return value


def rename_params(params: dict, mapping: dict) -> dict:
    out = {}
    for k, v in params.items():
        out[k] = mapping[v] if k in ('subject', 'object', 'x') else v
    return out


def answer_space_size(qtype: str, records: list[dict]) -> int:
    """Size of the answer space a uniform guesser draws from (the chance baseline): the feature values for the
    single-feature types, 216 profiles, every subset of the case's records (filter) or of its identifiers (join)."""
    if qtype in SINGLE_FIELD:
        return len(schema.VALUES[SINGLE_FIELD[qtype]])
    if qtype == 'profile':
        return 3 * 2 * 3 * 3 * 4
    if qtype == 'filter':
        return 2 ** len(records)
    if qtype == 'join':
        return 2 ** len(schema.entities(records))
    raise ValueError(qtype)
