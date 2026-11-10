"""Audits of a generated benchmark: answer balance, identifier and near-duplicate overlap, position and length cues.

``audit(data_dir) -> dict``; ``baseline.py`` writes it into the report. Every number is computed from the files
on disk, not from the generator's internals.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import families as F  # noqa: E402
import questions as Q
import render
import schema


def _rows(data_dir: Path, split: str, arm: str) -> list[dict]:
    return [json.loads(line) for line in (data_dir / f'{split}_{arm}.jsonl').read_text().splitlines()]


def answer_balance(rows: list[dict]) -> dict:
    out = {}
    for qtype in Q.QTYPES:
        mine = [r for r in rows if r['qtype'] == qtype]
        answers = Counter(r['answer'] for r in mine)
        entry = {'n': len(mine), 'distinct_answers': len(answers),
                 'majority_share': round(max(answers.values()) / len(mine), 4)}
        if qtype in Q.SINGLE_FIELD:
            entry['values'] = {json.loads(a): n for a, n in sorted(answers.items())}
            entry['uniform_share'] = round(1 / len(schema.VALUES[Q.SINGLE_FIELD[qtype]]), 4)
        if qtype in ('filter', 'join'):
            sizes = Counter(r['answer_size'] for r in mine)
            entry['answer_size'] = dict(sorted(sizes.items()))
            entry['empty_share'] = round(sizes.get(0, 0) / len(mine), 4)
        if qtype == 'profile':
            for f in Q.PROFILE_KEYS:
                c = Counter(json.loads(r['answer'])[f] for r in mine)
                entry[f'{f}_values'] = dict(sorted(c.items()))
        by_variant = {}
        for v in F.VARIANTS:
            c = Counter(r['answer'] for r in mine if r['variant'] == v)
            by_variant[v] = round(max(c.values()) / sum(c.values()), 4)
        entry['majority_share_by_variant'] = by_variant
        out[qtype] = entry
    return out


def identifiers(rows_by_split: dict) -> dict:
    ids = {}
    for split, rows in rows_by_split.items():
        s = set()
        for r in rows:
            for rec in render.parse(r['arm'], r['records']):
                s.add(rec['subject'])
                s.add(rec['object'])
        ids[split] = s
    buckets = {split: Counter(schema.id_split(i) for i in sorted(s)) for split, s in ids.items()}
    import analyze_existing
    _, v1, _ = analyze_existing.v1_cases()
    v1_ids = {split: {r[f] for c in cases for r in c['records'] for f in ('subject', 'object')}
              for split, cases in v1.items()}
    return {'n_identifiers': {k: len(v) for k, v in ids.items()},
            'shared_with_pilot_v1_cases': {f'{mine}_vs_v1_{theirs}': len(ids[mine] & v1_ids[theirs])
                                           for mine in ids for theirs in v1_ids},
            'dev_test_overlap': len(ids['dev'] & ids['test']),
            'pilot_hash_bucket': {k: dict(v) for k, v in buckets.items()},
            'pilot_training_bucket_identifiers': sum(v.get('train', 0) for v in buckets.values())}


def near_duplicates(families_path: Path) -> dict:
    sigs = defaultdict(set)
    for line in families_path.read_text().splitlines():
        fam = json.loads(line)
        for m in fam['members']:
            sigs[fam['split']].add(schema.structural_signature(m['records'], m['params']))
    return {'signatures': {k: len(v) for k, v in sigs.items()}, 'cross_split_overlap': len(sigs['dev'] & sigs['test'])}


def positions(rows: list[dict]) -> dict:
    """Where the answer-bearing records sit: for pair questions the queried record's index as a fraction of
    K-1, in quarters; for every type the share of items whose answer uses the first / the last record."""
    out = {}
    for qtype in Q.QTYPES:
        mine = [r for r in rows if r['qtype'] == qtype]
        quarters = Counter()
        first = last = 0
        for r in mine:
            pos = r['answer_positions']
            first += 0 in pos
            last += (r['k'] - 1) in pos
            if qtype in Q.PAIR_QTYPES:
                frac = pos[0] / (r['k'] - 1)
                quarters[min(3, int(frac * 4))] += 1
        entry = {'uses_first_record': round(first / len(mine), 4), 'uses_last_record': round(last / len(mine), 4)}
        if quarters:
            entry['target_position_quarters'] = [quarters[i] for i in range(4)]
            entry['k_mean'] = round(sum(r['k'] for r in mine) / len(mine), 2)
        out[qtype] = entry
    return out


def _bucket(n: int, width: int) -> int:
    return n // width


def length_cues(dev: list[dict], test: list[dict], arm: str) -> dict:
    """Can the answer be predicted from surface size alone? A lookup table (question type, length bucket of the
    rendered records, K) -> majority answer is fitted on dev and scored on test, against the question type's
    dev majority; and the mean absolute change of the rendering's length for relevant vs irrelevant edits."""
    width = 20

    def key(r):
        return (r['qtype'], _bucket(len(r['records']), width), r['k'])

    table, by_type = defaultdict(Counter), defaultdict(Counter)
    for r in dev:
        table[key(r)][r['answer']] += 1
        by_type[r['qtype']][r['answer']] += 1
    out = {}
    for qtype in Q.QTYPES:
        mine = [r for r in test if r['qtype'] == qtype]
        hit_len = hit_major = 0
        for r in mine:
            c = table.get(key(r)) or by_type[qtype]
            hit_len += c.most_common(1)[0][0] == r['answer']
            hit_major += by_type[qtype].most_common(1)[0][0] == r['answer']
        base = {r['family']: len(r['records']) for r in mine if r['variant'] == 'base'}
        delta = {v: [abs(len(r['records']) - base[r['family']]) for r in mine if r['variant'] == v]
                 for v in ('relevant', 'irrelevant')}
        out[qtype] = {'length_lookup_accuracy': round(hit_len / len(mine), 4),
                      'majority_accuracy': round(hit_major / len(mine), 4),
                      'mean_abs_length_change_relevant': round(sum(delta['relevant']) / len(delta['relevant']), 2),
                      'mean_abs_length_change_irrelevant': round(sum(delta['irrelevant']) / len(delta['irrelevant']), 2),
                      'share_relevant_edits_changing_length': round(
                          sum(d > 0 for d in delta['relevant']) / len(delta['relevant']), 4),
                      'share_irrelevant_edits_changing_length': round(
                          sum(d > 0 for d in delta['irrelevant']) / len(delta['irrelevant']), 4)}
    return out


def audit(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    rows = {s: {a: _rows(data_dir, s, a) for a in render.ARMS} for s in ('dev', 'test')}
    return {
        'answer_balance': {s: answer_balance(rows[s]['json']) for s in ('dev', 'test')},
        'identifiers': identifiers({s: rows[s]['json'] for s in ('dev', 'test')}),
        'near_duplicates': near_duplicates(data_dir / 'families.jsonl'),
        'positions': {s: positions(rows[s]['json']) for s in ('dev', 'test')},
        'length_cues': {a: length_cues(rows['dev'][a], rows['test'][a], a) for a in render.ARMS},
        'mean_record_chars': {a: {s: round(sum(len(r['records']) for r in rows[s][a]) / len(rows[s][a]), 1)
                                  for s in ('dev', 'test')} for a in render.ARMS},
    }


if __name__ == '__main__':
    import sys
    print(json.dumps(audit(Path(sys.argv[1])), indent=1))
