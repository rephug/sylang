"""What the pilot's select and lookup tasks can and cannot measure, computed on the frozen v1 dev and test cases.

    python -I analyze_existing.py [--out reports/existing_tasks.json --md reports/existing_tasks.md]

The v1 case files are not in the repository; they are regenerated here with the pilot's own generator
(``pretrained_pilot/data.py``: ``file_plan`` and ``generate_cases``, imported read-only through
``pilot_compat``) and compared with the ``split_hashes`` that ``pretrained_pilot/data_v1_manifest.json``
recorded when the frozen dataset was made. The answers are computed with the pilot's own ``select_answers`` and
``lookup_answers`` and the renderings with the pilot's codec calls (``sylang_batch.encode(version=2)``,
``data.group_by_subject``, ``study3._json_record``). The ``english`` arm needs the non-public study packet
and is not rendered here; the field-dependence results do not depend on the rendering.

Everything here is a property of the *tasks and data*. None of it says what a trained model did or did not
learn: a task that does not depend on a feature cannot show that a model reads it, and equally cannot show
that it does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pilot_compat  # noqa: E402

FEATURES = ('relation', 'polarity', 'tense', 'aspect', 'evidence')
VALUES = {'relation': ('see', 'help', 'contain'), 'polarity': ('positive', 'negative'),
          'tense': ('past', 'present', 'future'), 'aspect': ('simple', 'progressive', 'completed'),
          'evidence': ('unspecified', 'direct', 'reported', 'inferred')}


def v1_cases():
    data = pilot_compat.pilot_data()
    manifest = json.loads(pilot_compat.V1_MANIFEST.read_text())
    plan = data.file_plan('v1', [0], 8000, 400, 1000)  # dev, test, then train (not generated here)
    seen: set = set()
    out, verified = {}, {}
    for entry in plan[:2]:
        cases, _ = data.generate_cases(entry, seen)
        name = f'cases_{entry["split"]}.jsonl'
        digest = hashlib.sha256(''.join(data.case_hash(c['records']) for c in cases).encode()).hexdigest()
        verified[name] = digest == manifest['split_hashes'][name]
        out[entry['split']] = cases
    return data, out, verified


def field_dependence(data, cases) -> dict:
    """For every record, every feature and every other value of it: does the select / lookup answer change?"""
    counts = {f: {'edits': 0, 'select_changed': 0, 'lookup_changed': 0, 'lookup_edits_on_queried': 0,
                  'lookup_changed_on_queried': 0} for f in FEATURES}
    for case in cases:
        records, query = case['records'], case['query']
        sel, look = data.select_answers(records)['json'], data.lookup_answers(records, query)['json']
        for i, r in enumerate(records):
            queried = (r['subject'], r['object']) == (query['subject'], query['object'])
            for f in FEATURES:
                for v in VALUES[f]:
                    if v == r[f]:
                        continue
                    edited = [dict(x) for x in records]
                    edited[i][f] = v
                    c = counts[f]
                    c['edits'] += 1
                    c['select_changed'] += data.select_answers(edited)['json'] != sel
                    changed = data.lookup_answers(edited, query)['json'] != look
                    c['lookup_changed'] += changed
                    if queried:
                        c['lookup_edits_on_queried'] += 1
                        c['lookup_changed_on_queried'] += changed
    return {f: {**c, 'select_changed_share': round(c['select_changed'] / c['edits'], 4),
                'lookup_changed_share_on_queried_record': round(c['lookup_changed_on_queried'] / c['lookup_edits_on_queried'], 4)}
            for f, c in counts.items()}


def binding(data, cases) -> dict:
    """Does the lookup need both identifiers? Heuristics that read one identifier, by panel."""
    out = {}
    for panel in ('iid', 'wording', 'pair', 'long'):
        mine = [c for c in cases if c['panel'] == panel]
        if not mine:
            continue
        subj_unique = obj_unique = either = subj_ok = obj_ok = 0
        for c in mine:
            q, records = c['query'], c['records']
            want = data.lookup_answers(records, q)['json']
            by_s = [r for r in records if r['subject'] == q['subject']]
            by_o = [r for r in records if r['object'] == q['object']]
            subj_unique += len(by_s) == 1
            obj_unique += len(by_o) == 1
            either += len(by_s) == 1 or len(by_o) == 1
            fmt = lambda r: json.dumps({'relation': r['relation'], 'polarity': r['polarity']}, separators=(',', ':'))
            subj_ok += fmt(by_s[0]) == want
            obj_ok += fmt(by_o[0]) == want
        n = len(mine)
        out[panel] = {'n': n, 'subject_alone_unique': round(subj_unique / n, 4),
                      'object_alone_unique': round(obj_unique / n, 4), 'one_identifier_suffices': round(either / n, 4),
                      'first_record_with_subject_accuracy': round(subj_ok / n, 4),
                      'first_record_with_object_accuracy': round(obj_ok / n, 4)}
    return out


def _select_from_codes(text: str, grouped: bool) -> str:
    pairs = []
    for line in text.split('\n')[:-1]:
        if grouped:
            subject, items = line.split('\t')
            for item in items.split(';'):
                code, obj = item.split(' ')
                if code[0].isupper():
                    pairs.append([subject, obj])
        else:
            subject, code, obj = line.split('\t')
            if code[0].isupper():
                pairs.append([subject, obj])
    return json.dumps(pairs, separators=(',', ':'))


def _lookup_from_codes(text: str, grouped: bool, q: dict) -> str:
    letters = {'s': 'see', 'h': 'help', 'c': 'contain'}
    for line in text.split('\n')[:-1]:
        if grouped:
            subject, items = line.split('\t')
            entries = [(subject, *reversed(item.split(' '))) for item in items.split(';')]
        else:
            subject, code, obj = line.split('\t')
            entries = [(subject, obj, code)]
        for s, o, code in entries:
            if (s, o) == (q['subject'], q['object']):
                first = code[0]  # only the first character is read: tense, aspect, evidence letters never are
                return json.dumps({'relation': letters[first.lower()],
                                   'polarity': 'negative' if first.isupper() else 'positive'}, separators=(',', ':'))
    return '{}'


def _select_from_json(text: str) -> str:
    return json.dumps([[r['s'], r['o']] for r in json.loads(text) if r.get('neg')], separators=(',', ':'))


def _lookup_from_json(text: str, q: dict) -> str:
    for r in json.loads(text):
        if (r['s'], r['o']) == (q['subject'], q['object']):
            return json.dumps({'relation': r['r'], 'polarity': 'negative' if r.get('neg') else 'positive'},
                              separators=(',', ':'))
    return '{}'


def surface_solvers(data, cases) -> dict:
    """Solvers that read one character (codes) or two keys (json) per record and never tense, aspect or
    evidence, scored with the pilot's own grader on every panel."""
    sb, st3 = pilot_compat.sylang_batch(), pilot_compat.study3()
    grader = pilot_compat.pilot_grader()
    out = {}
    for arm in ('sylang2', 'sylang2g', 'json'):
        per = {}
        for c in cases:
            records = c['records']
            if arm == 'json':
                text = '[' + ','.join(st3._json_record(r) for r in records) + ']\n'
                sel, look = _select_from_json(text), _lookup_from_json(text, c['query'])
            else:
                text = sb.encode(records, version=2)
                if arm == 'sylang2g':
                    text = data.group_by_subject(text)
                sel = _select_from_codes(text, arm == 'sylang2g')
                look = _lookup_from_codes(text, arm == 'sylang2g', c['query'])
            p = per.setdefault(c['panel'], {'n': 0, 'select': 0, 'lookup': 0})
            p['n'] += 1
            p['select'] += grader.grade('select', 'json', data.select_answers(records)['json'], sel)['correct']
            p['lookup'] += grader.grade('lookup', 'json', data.lookup_answers(records, c['query'])['json'], look)['correct']
        out[arm] = {panel: {'n': p['n'], 'select_accuracy': round(p['select'] / p['n'], 4),
                            'lookup_accuracy': round(p['lookup'] / p['n'], 4)} for panel, p in sorted(per.items())}
    return out


def answer_baselines(data, dev, test) -> dict:
    lookups = Counter(data.lookup_answers(c['records'], c['query'])['json'] for c in dev)
    majority = lookups.most_common(1)[0][0]
    out = {'dev_lookup_majority': majority}
    for panel in ('iid', 'wording', 'pair', 'long'):
        mine = [c for c in test if c['panel'] == panel]
        sel_empty = sum(data.select_answers(c['records'])['json'] == '[]' for c in mine)
        look = Counter(data.lookup_answers(c['records'], c['query'])['json'] for c in mine)
        rel = Counter(json.loads(a)['relation'] for a in look.elements())
        out[panel] = {'n': len(mine), 'select_constant_empty_accuracy': round(sel_empty / len(mine), 4),
                      'lookup_dev_majority_accuracy': round(look[majority] / len(mine), 4),
                      'lookup_panel_majority_share': round(look.most_common(1)[0][1] / len(mine), 4),
                      'lookup_relation_distribution': dict(sorted(rel.items())),
                      'lookup_chance_uniform': round(1 / 6, 4)}
    return out


def md_report(r: dict) -> str:
    fd = r['field_dependence']
    lines = ['# What the existing select and lookup tasks can measure', '',
             'Generated by `analyze_existing.py` from the frozen pilot v1 dev (400) and test (1,000) cases, regenerated '
             'with the pilot\'s own generator and verified against `data_v1_manifest.json` '
             f'(dev hash match: {r["v1_hash_verified"]["cases_dev.jsonl"]}, test hash match: '
             f'{r["v1_hash_verified"]["cases_test.jsonl"]}).', '',
             'These are properties of the tasks and the data, not of any model. A task whose answer does not depend '
             'on a feature cannot show that a model reads that feature, and it cannot show that the model does not.',
             '', '## 1. Which features the answers depend on', '',
             'Every record of every test case, every feature, every other value of that feature: the share of these '
             'single-field edits that change the answer (the pilot\'s own `select_answers` / `lookup_answers`).', '',
             '| feature | single-field edits | select answer changes | lookup answer changes (edits of the queried record) |',
             '|---|---|---|---|']
    for f in FEATURES:
        lines.append(f'| {f} | {fd[f]["edits"]:,} | {fd[f]["select_changed_share"]:.1%} | '
                     f'{fd[f]["lookup_changed_share_on_queried_record"]:.1%} |')
    lines += ['', 'So *select* depends on polarity, the subject and object identifiers and record order only; *lookup* on '
              'relation, polarity and the binding of the two named identifiers to one record. Neither answer depends '
              'on tense, aspect or evidence in any case: those three features are carried by every rendering but '
              'never tested. In particular the held-out-pair panel, whose queried record carries (contain, future) '
              'or (help, completed), asks only for relation and polarity: it can show whether an unseen co-occurrence '
              'in the input disturbs reading relation and polarity, not whether the model composes tense or aspect '
              'with relation.', '',
              '## 2. One character per record solves both tasks', '',
              'In the codes a record\'s polarity is the case of the first letter of the middle field and its relation '
              'is that letter. Solvers that read only that character (or only the json keys `r` and `neg`) and '
              'never a tense, aspect or evidence letter, scored with the pilot\'s grader:', '',
              '| arm | panel | n | select | lookup |', '|---|---|---|---|---|']
    for arm, panels in r['surface_solvers'].items():
        for panel, p in panels.items():
            lines.append(f'| {arm} | {panel} | {p["n"]} | {p["select_accuracy"]:.1%} | {p["lookup_accuracy"]:.1%} |')
    lines += ['', 'Perfect scores here mean a perfect model score is *compatible* with reading one character per record. '
              'It does not mean a model uses that shortcut.', '',
              '## 3. Lookup binding: is the second identifier needed?', '',
              '| panel | n | subject alone is unique | object alone is unique | one identifier suffices | '
              'first record with the subject gives the right answer | first record with the object gives the right answer |',
              '|---|---|---|---|---|---|---|']
    for panel, b in r['binding'].items():
        lines.append(f'| {panel} | {b["n"]} | {b["subject_alone_unique"]:.1%} | {b["object_alone_unique"]:.1%} | '
                     f'{b["one_identifier_suffices"]:.1%} | {b["first_record_with_subject_accuracy"]:.1%} | '
                     f'{b["first_record_with_object_accuracy"]:.1%} |')
    lines += ['', 'With objects mostly unique (object reuse 0.2), the object identifier alone finds the queried record in '
              'most cases, so lookup accuracy cannot separate binding both identifiers from matching one.', '',
              '## 4. Answer-pattern baselines', '',
              '| panel | n | select: always `[]` | lookup: dev majority | lookup: panel majority share | lookup relation distribution |',
              '|---|---|---|---|---|---|']
    for panel in ('iid', 'wording', 'pair', 'long'):
        a = r['answer_baselines'][panel]
        lines.append(f'| {panel} | {a["n"]} | {a["select_constant_empty_accuracy"]:.1%} | '
                     f'{a["lookup_dev_majority_accuracy"]:.1%} | {a["lookup_panel_majority_share"]:.1%} | '
                     f'{a["lookup_relation_distribution"]} |')
    lines += ['', 'On the pair panel the queried relation is always contain or help (the held-out pairs fix it), so a '
              'solver that never answers "see" is favoured there relative to iid.', '',
              '## 5. The existing counterfactual probes', '',
              'The pilot v1 has no counterfactual panel (its panels are iid, held-out wording, held-out pair, K 13-16). '
              'The copy-probe study has several: `copy_probe/language.py` builds `reorder` (records permuted), '
              '`swap` (`swap_roles`: subject and object exchanged in every record), `flip` (`flip_one`: one polarity '
              'flipped) and `similar` / `similar_control` (a positive partner whose identifier differs from a negative '
              'record\'s in one character, against a fresh-identifier control); `copy_probe/train.py` feeds '
              '`probes.counterfactual_probe` a `swap` pair and a `replace` pair (one answer identifier renamed) and '
              'measures how the first answer byte\'s distribution moves. All of them act on the select task, so they '
              'probe record order, roles, polarity and identifier copying. None edits tense, aspect, evidence or '
              'relation; the only edit that must leave the answer set unchanged is `reorder`, and there the answer\'s '
              'order changes; `replace` renames a single identifier and scores the first byte, not a whole answer '
              'under a consistent renaming.', '',
              '## What follows', '',
              'Select and lookup measure identifier copying, polarity detection, relation reading and (weakly) '
              'pair binding. They cannot measure reading tense, aspect or evidence, combining features, or binding '
              'entities across records. `experiments/compositional_diagnostics` adds questions that need exactly '
              'those, with matched controls.', '']
    return '\n'.join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--out', default=str(HERE / 'reports' / 'existing_tasks.json'))
    p.add_argument('--md', default=str(HERE / 'reports' / 'existing_tasks.md'))
    args = p.parse_args(argv)
    data, cases, verified = v1_cases()
    if not all(verified.values()):
        print('regenerated v1 case files do not match the frozen manifest:', verified)
        return 1
    report = {'v1_hash_verified': verified, 'stubbed_modules': pilot_compat.STUBBED,
              'field_dependence': field_dependence(data, cases['test']),
              'binding': binding(data, cases['test']),
              'surface_solvers': surface_solvers(data, cases['test']),
              'answer_baselines': answer_baselines(data, cases['dev'], cases['test'])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    Path(args.md).write_text(md_report(report))
    print(f'wrote {args.out} and {args.md}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
