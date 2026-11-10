"""The four renderings (arms) of a case, their legends, and a parser per arm that inverts the rendering.

| arm | rendering | implementation |
|---|---|---|
| ``english`` | controlled English, one numbered sentence per record | ``english()`` below (written for this benchmark; the pilot's english arm comes from the non-public study packet, which is not used here) |
| ``json`` | compact JSON array, one-letter keys, defaults omitted | ``json_records()``, byte-identical to ``copy_probe/study3._json_record`` (asserted by the tests) |
| ``sylang2`` | header-free Sylang codes, ``subject<TAB>code<TAB>object`` | the public codec ``copy_probe/sylang_batch.encode(records, version=2)`` |
| ``sylang2g`` | codes grouped by subject, ``subject<TAB>code object;code object`` | ``grouped()``, the same rule as ``pretrained_pilot/data.group_by_subject`` (asserted by the tests) |

Every rendering is checked to parse back to exactly the records it was made from (``parse``), so the four arms
carry the same information, in the same record order. The legends follow the pilot's (``data.ARM_LEGEND``) with the
English legend adapted to this renderer and the task legend replaced (the answer formats differ).
"""
from __future__ import annotations

import json
import re

import pilot_compat
import schema

ARMS = ('english', 'json', 'sylang2', 'sylang2g')

# ---------------------------------------------------------------------------
# controlled English

_FORMS = {  # relation -> (3rd-person present, past, past participle, present participle)
    'see': ('sees', 'saw', 'seen', 'seeing'),
    'help': ('helps', 'helped', 'helped', 'helping'),
    'contain': ('contains', 'contained', 'contained', 'containing'),
}
_EVIDENCE_PHRASE = {'unspecified': '', 'direct': 'By direct observation, ', 'reported': 'By report, ',
                    'inferred': 'By inference, '}


def verb_phrase(relation: str, polarity: str, tense: str, aspect: str) -> str:
    pres3, past, part, ing = _FORMS[relation]
    neg = polarity == 'negative'
    base = relation
    if aspect == 'simple':
        if tense == 'past':
            return f'did not {base}' if neg else past
        if tense == 'present':
            return f'does not {base}' if neg else pres3
        return f'will not {base}' if neg else f'will {base}'
    if aspect == 'progressive':
        aux = {'past': 'was', 'present': 'is', 'future': 'will'}[tense]
        if tense == 'future':
            return f'will not be {ing}' if neg else f'will be {ing}'
        return f'{aux} not {ing}' if neg else f'{aux} {ing}'
    # completed
    if tense == 'future':
        return f'will not have {part}' if neg else f'will have {part}'
    aux = 'had' if tense == 'past' else 'has'
    return f'{aux} not {part}' if neg else f'{aux} {part}'


_VERB_TABLE = {verb_phrase(r, p, t, a): (r, p, t, a) for r in schema.RELATIONS for p in schema.POLARITIES
               for t in schema.TENSES for a in schema.ASPECTS}
assert len(_VERB_TABLE) == 54, 'verb phrases must be distinct'
_ENGLISH_LINE = re.compile(r'(\d+)\. (By direct observation, |By report, |By inference, )?"([a-z0-9]+)" (.+) "([a-z0-9]+)"\.')
_PHRASE_EVIDENCE = {v: k for k, v in _EVIDENCE_PHRASE.items()}


def english(records: list[dict]) -> str:
    lines = []
    for i, r in enumerate(records, 1):
        lines.append(f'{i}. {_EVIDENCE_PHRASE[r["evidence"]]}"{r["subject"]}" '
                     f'{verb_phrase(r["relation"], r["polarity"], r["tense"], r["aspect"])} "{r["object"]}".')
    return '\n'.join(lines) + '\n'


def parse_english(text: str) -> list[dict]:
    out = []
    for i, line in enumerate(text.split('\n')[:-1], 1):
        m = _ENGLISH_LINE.fullmatch(line)
        if not m or int(m.group(1)) != i:
            raise ValueError(f'not a controlled-English record line: {line!r}')
        relation, polarity, tense, aspect = _VERB_TABLE[m.group(4)]
        out.append(schema.record(m.group(3), relation, m.group(5), polarity, tense, aspect,
                                 _PHRASE_EVIDENCE[m.group(2) or '']))
    return out


# ---------------------------------------------------------------------------
# compact JSON (study3._json_record)

def json_record(r: dict) -> str:
    fields = {'s': r['subject'], 'r': r['relation']}
    if r['polarity'] == 'negative':
        fields['neg'] = True
    for key, name in (('t', 'tense'), ('a', 'aspect'), ('e', 'evidence')):
        if r[name] != schema.DEFAULTS[name]:
            fields[key] = r[name]
    fields['o'] = r['object']
    return json.dumps(fields, separators=(',', ':'))


def json_records(records: list[dict]) -> str:
    return '[' + ','.join(json_record(r) for r in records) + ']\n'


def parse_json(text: str) -> list[dict]:
    out = []
    for item in json.loads(text):
        out.append(schema.record(item['s'], item['r'], item['o'], 'negative' if item.get('neg') else 'positive',
                                 item.get('t', 'present'), item.get('a', 'simple'), item.get('e', 'unspecified')))
    return out


# ---------------------------------------------------------------------------
# Sylang codes, ungrouped and grouped

def sylang2(records: list[dict]) -> str:
    return pilot_compat.sylang_batch().encode(records, version=2)


def grouped(sylang2_text: str) -> str:
    groups: dict[str, list[str]] = {}
    for line in sylang2_text.split('\n')[:-1]:
        subject, code, obj = line.split('\t')
        groups.setdefault(subject, []).append(f'{code} {obj}')
    return ''.join(f'{subject}\t{";".join(items)}\n' for subject, items in groups.items())


def ungroup(text: str) -> str:
    lines = []
    for line in text.split('\n')[:-1]:
        subject, items = line.split('\t')
        for item in items.split(';'):
            code, obj = item.split(' ')
            lines.append(f'{subject}\t{code}\t{obj}')
    return '\n'.join(lines) + '\n'


def _decode(text: str) -> list[dict]:
    records, _ = pilot_compat.sylang_batch().decode(text)
    return [schema.record(r['subject'], r['relation'], r['object'], r['polarity'], r['tense'], r['aspect'],
                          r['evidence']) for r in records]


# ---------------------------------------------------------------------------

def render(arm: str, records: list[dict]) -> str:
    schema.check_case(records)
    if arm == 'english':
        text = english(records)
    elif arm == 'json':
        text = json_records(records)
    elif arm == 'sylang2':
        text = sylang2(records)
    elif arm == 'sylang2g':
        text = grouped(sylang2(records))
    else:
        raise ValueError(arm)
    assert text.isascii() and text.endswith('\n'), arm
    return text


def parse(arm: str, text: str) -> list[dict]:
    if arm == 'english':
        return parse_english(text)
    if arm == 'json':
        return parse_json(text)
    if arm == 'sylang2':
        return _decode(text)
    if arm == 'sylang2g':
        return _decode(ungroup(text))
    raise ValueError(arm)


_CODE_LEGEND = ('Code: relation letter s see, h help, c contain, uppercase S, H, C when negated; then optional tense '
                'p past, f future; aspect g progressive, c completed; evidence d direct, r reported, i inferred; '
                'absent means present, simple, unspecified.')
ARM_LEGEND = {
    'english': ('Records are numbered English sentences, one per line, in record order. Each says that a subject '
                'in double quotes sees, helps or contains an object in double quotes; "not" marks a negated '
                'record; the verb form carries the tense (past, present, future) and the aspect (simple, '
                'progressive "is seeing", completed "has seen"); an opening phrase (By direct observation, '
                'By report, By inference) gives the evidence; without it the evidence is unspecified.'),
    'json': ('Records are a JSON array of objects in record order. Keys: s subject, r relation (see, help, '
             'contain), neg true if negated, t tense (past, future), a aspect (progressive, completed), e evidence '
             '(direct, reported, inferred), o object. Default values are omitted: positive, present, simple, '
             'unspecified.'),
    'sylang2': 'Records are lines of subject TAB code TAB object, one record per line, in record order. ' + _CODE_LEGEND,
    'sylang2g': ('Lines are subject TAB code object;code object, one line per subject, in record order. '
                 + _CODE_LEGEND),
}
TASK_LEGEND = ('Answer the question about the records with one JSON value of the requested shape and nothing else. '
               'A record is negated when its polarity is negative.')
LEGENDS = {arm: ARM_LEGEND[arm] + ' ' + TASK_LEGEND for arm in ARMS}
