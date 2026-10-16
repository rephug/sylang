"""Why do learned-position models fail the held-out (relation, tense) combination?

    python combo_analysis.py --source <packet> --out results/combo_analysis.json <ckpt.pt> ...

Checkpoints are the files saved by ``language.py`` (``<arm>-<init>-k1-...-s<seed>[-alibi].pt``).
Only K=1 items are used. For every checkpoint:

1. **Per-cell accuracy.** 9 (relation, tense) combinations x 2 polarities, ``--per-cell``
   items each, aspect and evidence uniform, test-split identifiers, one fixed seed; exact
   accuracy from ``language.teacher_forced``; the held-out combination is marked. Selected
   (negative record) and empty (positive record) accuracy are reported separately.
2. **Positional hypothesis.** For every rendered prompt the byte offsets of the subject and
   object identifiers are found with ``str.index`` and compared with the offsets that occur in
   ``--train-items`` training-distribution renderings of the same arm (``ItemStream('train')``,
   which never contains the held-out combination). With K=1 and a fixed instruction, the object
   offset, the record length and the absolute position of the answer are the same quantity up
   to a constant in the English arms, so "object offset seen in training" = "answer position
   seen in training". Accuracy is split by seen/novel for the held-out combination and, as a
   control, for the trained combinations. The split by prompt length (= answer position) is
   reported as well because it differs from the object-offset split in ``codes_eqlen``.
3. **Error type.** The first wrong answer byte (teacher forced) classifies a wrong answer as
   ``polarity`` (byte 0 or 1: ``[]`` versus ``[[``), ``copy`` (an identifier byte) or
   ``structure`` (quote, comma, bracket, EOS).

Only aggregates are written: counts, accuracies, offsets and lengths. No identifier is stored.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import language  # noqa: E402
from followup import Runtime  # noqa: E402

COMBOS = [(r, t) for r in language.RELATIONS for t in language.TENSES]
POLARITIES = ('negative', 'positive')


# ---------------------------------------------------------------------------
# Items and renderings


def parse_name(path: str) -> dict:
    name = Path(path).name
    m = re.match(r'^(?P<arm>.+?)-(?P<init>fresh|copy96|surrogate)-k(?P<k>\d+)-.*-s(?P<seed>\d+)(?P<rest>.*)\.pt$', name)
    if not m:
        raise ValueError(f'cannot parse checkpoint name {name}')
    return dict(file=name, arm=m['arm'], init=m['init'], k=m['k'], seed=int(m['seed']),
                position='alibi' if name.endswith('-alibi.pt') else 'learned')


def cell_items(seed: int, per_cell: int) -> list[dict]:
    """``per_cell`` single-record items for every (relation, tense, polarity) cell.

    Aspect and evidence are uniform; subject and object are distinct test-split identifiers."""
    import random
    ids = language.ItemStream('test', seed, k_values=(1,)).ids
    rng = random.Random(seed)
    out = []
    for relation, tense in COMBOS:
        for polarity in POLARITIES:
            for _ in range(per_cell):
                subject, obj = rng.sample(ids, 2)
                statement = language._statement(subject, obj, relation, polarity, tense,
                                                rng.choice(language.ASPECTS), rng.choice(language.EVIDENCES))
                out.append(dict(relation=relation, tense=tense, polarity=polarity,
                                held_out=(relation, tense) == language.HELD_OUT_COMBO,
                                statement=statement))
    return out


def geometry(prompt: str, statement: dict) -> dict:
    """Byte offsets of the identifiers in a K=1 prompt, plus record and prompt lengths."""
    parts = prompt.split('\n')
    assert parts[-1] == '' and parts[-2] + '\n' == language.INSTRUCTION, 'K=1 prompt layout'
    record = parts[-3]
    base = prompt.index(record)
    return dict(subject_offset=base + record.index(statement['subject']),
                object_offset=base + record.index(statement['object']),
                record_length=len(record), prompt_length=len(prompt))


def render(renderer: language.Renderer, arm: str, item: dict) -> dict:
    prompt = renderer.prompt(arm, [item['statement']])
    return dict(item, prompt=prompt, target=renderer.target([item['statement']]),
                **geometry(prompt, item['statement']))


def training_profile(renderer: language.Renderer, arm: str, seed: int, n: int) -> dict:
    """Offsets and lengths that occur in ``n`` training-distribution K=1 renderings."""
    stream = language.ItemStream('train', seed, k_values=(1,))
    object_offsets, subject_offsets, lengths, prompt_lengths = Counter(), Counter(), Counter(), Counter()
    length_polarity = defaultdict(set)   # record length -> polarities it occurs with
    offset_polarity = defaultdict(set)   # object offset -> polarities it occurs with
    for _ in range(n):
        statement = stream.item(1)[0]
        assert (statement['relation'], statement['tense']) != language.HELD_OUT_COMBO
        g = geometry(renderer.prompt(arm, [statement]), statement)
        object_offsets[g['object_offset']] += 1
        subject_offsets[g['subject_offset']] += 1
        lengths[g['record_length']] += 1
        prompt_lengths[g['prompt_length']] += 1
        length_polarity[g['record_length']].add(statement['polarity'])
        offset_polarity[g['object_offset']].add(statement['polarity'])
    return dict(n=n, object_offsets=object_offsets, subject_offsets=subject_offsets,
                record_lengths=lengths, prompt_lengths=prompt_lengths,
                length_polarity=length_polarity, offset_polarity=offset_polarity)


def length_table(renderer: language.Renderer, arm: str, profile: dict) -> dict:
    """Per (relation, tense, polarity): the object offsets of all 12 aspect x evidence
    renderings (placeholder 3-byte identifiers), each marked seen/novel against the profile."""
    out = {}
    for relation, tense in COMBOS:
        for polarity in POLARITIES:
            offsets = Counter()
            for aspect in language.ASPECTS:
                for evidence in language.EVIDENCES:
                    s = language._statement('a1b', 'c2d', relation, polarity, tense, aspect, evidence)
                    offsets[geometry(renderer.prompt(arm, [s]), s)['object_offset']] += 1
            seen = profile['object_offsets']
            out[f'{relation}/{tense}/{polarity}'] = dict(
                held_out=(relation, tense) == language.HELD_OUT_COMBO,
                object_offsets={str(o): dict(variants=c, seen_in_training=o in seen,
                                             training_polarities=sorted(profile['offset_polarity'].get(o, ())))
                                for o, c in sorted(offsets.items())},
                novel_offsets=sorted(o for o in offsets if o not in seen))
    return out


# ---------------------------------------------------------------------------
# Model evaluation


def argmax_answers(rt: Runtime, model, rows: list[dict], batch: int = 32) -> list[list[int]]:
    """The argmax token at every answer position (teacher forced), same batching as
    language.teacher_forced."""
    import torch
    from real_pipeline import collate_rows
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            inputs, labels = collate_rows(rt.tiny, chunk, rt.recipe['context'])
            argmax = model(inputs).argmax(-1)
            for i in range(len(chunk)):
                out.append(argmax[i][labels[i] != -100].tolist())
    return out


def classify_error(row: dict, result: dict, predicted: list[int], eos: int) -> dict | None:
    """Type of the first wrong answer byte, or None when the answer is exact."""
    if result['exact']:
        return None
    hits, target = result['hits'], result['target']
    first = hits.index(False)
    text = row['target']
    pairs = json.loads(text)
    subject_pos, object_pos = set(), set()
    if pairs:
        s, o = pairs[0]
        i = text.index(s)
        subject_pos = set(range(i, i + len(s)))
        j = text.index(o, i + len(s))
        object_pos = set(range(j, j + len(o)))
    pred = chr(predicted[first]) if predicted[first] < 256 else 'EOS'
    category = ('eos' if pred == 'EOS' else 'bracket' if pred in '[]' else 'quote' if pred == '"' else
                'comma' if pred == ',' else 'digit' if pred.isdigit() else 'letter' if pred.isalpha() else 'other')
    if first <= 1:
        kind = 'polarity'
        detail = ('answered_empty' if row['polarity'] == 'negative' and pred == ']' else
                  'answered_nonempty' if row['polarity'] == 'positive' and pred == '[' else 'other')
    elif first in subject_pos:
        kind, detail = 'copy', 'subject'
    elif first in object_pos:
        kind = 'copy'
        guess = ''.join(chr(p) if p < 256 else '?' for p in predicted[min(object_pos):max(object_pos) + 1])
        detail = 'object_is_subject' if guess == pairs[0][0] else 'object'
    else:
        kind, detail = 'structure', ('eos' if target[first] == eos else 'punctuation')
    return dict(kind=kind, detail=detail, first_wrong=first, predicted=category)


def accuracy(flags: list[bool]) -> dict:
    return dict(n=len(flags), exact=round(sum(flags) / len(flags), 4) if flags else None)


def evaluate(rt: Runtime, info: dict, rows: list[dict], profile: dict, path: str) -> dict:
    model = rt.load(rt.build(0), path).eval()
    if info['position'] == 'alibi':
        language.use_alibi(model, rt.recipe['heads'], rt.tiny.PAD)
    started = time.perf_counter()
    results = language.teacher_forced(rt, model, rows)
    predicted = argmax_answers(rt, model, rows)
    seconds = time.perf_counter() - started
    errors = [classify_error(r, res, p, rt.tiny.EOS) for r, res, p in zip(rows, results, predicted)]
    exact = [res['exact'] for res in results]

    # 1. per cell
    cells = {}
    for key in sorted({(r['relation'], r['tense'], r['polarity']) for r in rows}):
        idx = [i for i, r in enumerate(rows) if (r['relation'], r['tense'], r['polarity']) == key]
        kinds = Counter(errors[i]['kind'] for i in idx if errors[i])
        cells['/'.join(key)] = dict(held_out=rows[idx[0]]['held_out'], **accuracy([exact[i] for i in idx]),
                                    errors=dict(sorted(kinds.items())),
                                    record_lengths=dict(sorted(Counter(rows[i]['record_length'] for i in idx).items())))

    def group(pred) -> dict:
        idx = [i for i, r in enumerate(rows) if pred(r)]
        sel = [exact[i] for i in idx if rows[i]['polarity'] == 'negative']
        emp = [exact[i] for i in idx if rows[i]['polarity'] == 'positive']
        return dict(**accuracy([exact[i] for i in idx]), selected=accuracy(sel), empty=accuracy(emp))

    summary = dict(held_out=group(lambda r: r['held_out']), trained=group(lambda r: not r['held_out']))

    # 2. positional hypothesis: accuracy by whether the offset / length occurred in training
    def split(pred, feature: str, seen: Counter) -> dict:
        idx = [i for i, r in enumerate(rows) if pred(r)]
        out = {}
        for name, cond in (('seen', lambda v: v in seen), ('novel', lambda v: v not in seen)):
            chosen = [i for i in idx if cond(rows[i][feature])]
            out[name] = accuracy([exact[i] for i in chosen])
            out[name]['values'] = sorted({rows[i][feature] for i in chosen})
            out[name]['errors'] = dict(sorted(Counter(errors[i]['kind'] for i in chosen if errors[i]).items()))
        return out

    offset_split, length_split = {}, {}
    for name, held, polarity in (('held_out_negative', True, 'negative'), ('held_out_positive', True, 'positive'),
                                 ('trained_negative', False, 'negative'), ('trained_positive', False, 'positive')):
        pred = lambda r, h=held, p=polarity: r['held_out'] == h and r['polarity'] == p
        offset_split[name] = split(pred, 'object_offset', profile['object_offsets'])
        length_split[name] = split(pred, 'prompt_length', profile['prompt_lengths'])

    # Length as a polarity cue (matters for english_natural): the training polarities of each length.
    def length_class(r):
        pols = profile['length_polarity'].get(r['record_length'], set())
        if not pols:
            return 'novel'
        if pols == {r['polarity']}:
            return 'seen_same_polarity_only'
        if r['polarity'] not in pols:
            return 'seen_other_polarity_only'
        return 'seen_both_polarities'

    cue = {}
    for held in (True, False):
        by = defaultdict(list)
        for i, r in enumerate(rows):
            if r['held_out'] == held:
                by[f"{r['polarity']}/{length_class(r)}"].append(exact[i])
        cue['held_out' if held else 'trained'] = {k: accuracy(v) for k, v in sorted(by.items())}

    # Within-phrase control: the verb phrase is fixed by (relation, tense, aspect, polarity); the evidence
    # prefix only shifts it. Accuracy per (aspect, evidence) for the held-out negatives separates a
    # novel offset from a novel phrase.
    phrase = {}
    for aspect in language.ASPECTS:
        for evidence in language.EVIDENCES:
            idx = [i for i, r in enumerate(rows) if r['held_out'] and r['polarity'] == 'negative'
                   and r['statement']['aspect'] == aspect and r['statement']['evidence'] == evidence]
            if not idx:
                continue
            offsets = sorted({rows[i]['object_offset'] for i in idx})
            phrase[f'{aspect}/{evidence}'] = dict(**accuracy([exact[i] for i in idx]), object_offsets=offsets,
                                                 offset_seen=all(o in profile['object_offsets'] for o in offsets))

    # 3. error types
    error_split = {}
    for held in (True, False):
        kinds, details, first, predicted = Counter(), Counter(), Counter(), Counter()
        for r, e in zip(rows, errors):
            if r['held_out'] == held and e:
                kinds[e['kind']] += 1
                details[f"{e['kind']}/{e['detail']}"] += 1
                first[str(e['first_wrong'])] += 1
                predicted[f"{e['first_wrong']}:{e['predicted']}"] += 1
        error_split['held_out' if held else 'trained'] = dict(
            wrong=sum(kinds.values()), kinds=dict(sorted(kinds.items())), details=dict(sorted(details.items())),
            first_wrong_index=dict(sorted(first.items(), key=lambda kv: int(kv[0]))),
            first_wrong_predicted=dict(sorted(predicted.items(), key=lambda kv: int(kv[0].split(':')[0]))))

    return dict(**info, eval_seconds=round(seconds, 1), n=len(rows), summary=summary, cells=cells,
                by_object_offset=offset_split, by_prompt_length=length_split,
                by_length_polarity_cue=cue, held_out_negative_by_aspect_evidence=phrase, errors=error_split)


# ---------------------------------------------------------------------------
# Report


def pct(a: dict) -> str:
    return '-' if a['exact'] is None else f"{100 * a['exact']:.0f}% ({a['n']})"


def markdown(results: list[dict]) -> str:
    head = ('| arm | init | seed | positions | trained exact (sel / emp) | held-out exact (sel / emp) '
            '| held-out neg: offset seen | held-out neg: offset novel | trained neg: offset seen | trained neg: offset novel '
            '| held-out errors (pol / copy / struct) |')
    lines = [head, '|' + '---|' * 11]
    for r in results:
        s, h = r['summary']['trained'], r['summary']['held_out']
        o = r['by_object_offset']
        e = r['errors']['held_out']['kinds']
        lines.append(f"| {r['arm']} | {r['init']} | {r['seed']} | {r['position']} "
                     f"| {100 * s['exact']:.0f}% ({100 * s['selected']['exact']:.0f} / {100 * s['empty']['exact']:.0f}) "
                     f"| {100 * h['exact']:.0f}% ({100 * h['selected']['exact']:.0f} / {100 * h['empty']['exact']:.0f}) "
                     f"| {pct(o['held_out_negative']['seen'])} | {pct(o['held_out_negative']['novel'])} "
                     f"| {pct(o['trained_negative']['seen'])} | {pct(o['trained_negative']['novel'])} "
                     f"| {e.get('polarity', 0)} / {e.get('copy', 0)} / {e.get('structure', 0)} |")
    return '\n'.join(lines)


def pooled(results: list[dict], key: str, group: str, bucket: str) -> dict:
    n = sum(r[key][group][bucket]['n'] for r in results)
    hits = sum(round(r[key][group][bucket]['exact'] * r[key][group][bucket]['n']) for r in results
               if r[key][group][bucket]['exact'] is not None)
    return dict(n=n, exact=round(hits / n, 4) if n else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-cell', type=int, default=64)
    parser.add_argument('--train-items', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('checkpoints', nargs='+')
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    warnings.filterwarnings('ignore', message='Support for mismatched src_key_padding_mask')
    rt = Runtime(args.source)
    renderer = language.Renderer(rt)
    items = cell_items(args.seed, args.per_cell)
    infos = [parse_name(p) for p in args.checkpoints]
    arms = sorted({i['arm'] for i in infos})

    profiles, rows_by_arm, tables = {}, {}, {}
    for arm in arms:
        started = time.perf_counter()
        profiles[arm] = training_profile(renderer, arm, args.seed, args.train_items)
        rows_by_arm[arm] = [render(renderer, arm, it) for it in items]
        tables[arm] = length_table(renderer, arm, profiles[arm])
        p = profiles[arm]
        print(f'{arm}: {args.train_items} training renderings in {time.perf_counter() - started:.1f}s; '
              f'{len(p["object_offsets"])} distinct object offsets, {len(p["subject_offsets"])} subject offsets, '
              f'{len(p["prompt_lengths"])} prompt lengths', flush=True)

    results = []
    for info, path in zip(infos, args.checkpoints):
        r = evaluate(rt, info, rows_by_arm[info['arm']], profiles[info['arm']], path)
        results.append(r)
        s, h = r['summary']['trained'], r['summary']['held_out']
        o = r['by_object_offset']['held_out_negative']
        print(f"{info['file']}: trained {s['exact']:.3f} held-out {h['exact']:.3f} "
              f"(sel {h['selected']['exact']:.3f} emp {h['empty']['exact']:.3f}); held-out negatives by object offset: "
              f"seen {pct(o['seen'])} novel {pct(o['novel'])}; {r['eval_seconds']}s", flush=True)

    arm_info = {}
    for arm in arms:
        p = profiles[arm]
        held_rows = [r for r in rows_by_arm[arm] if r['held_out']]
        arm_info[arm] = dict(
            training_items=p['n'],
            training_object_offsets=dict(sorted((str(k), v) for k, v in p['object_offsets'].items())),
            training_subject_offsets=dict(sorted((str(k), v) for k, v in p['subject_offsets'].items())),
            training_prompt_lengths=sorted(p['prompt_lengths']),
            training_object_offset_range=[min(p['object_offsets']), max(p['object_offsets'])],
            held_out_items_with_novel_object_offset=dict(
                negative=sum(r['object_offset'] not in p['object_offsets'] for r in held_rows if r['polarity'] == 'negative'),
                positive=sum(r['object_offset'] not in p['object_offsets'] for r in held_rows if r['polarity'] == 'positive'),
                per_polarity_n=len(held_rows) // 2),
            held_out_items_with_novel_prompt_length=dict(
                negative=sum(r['prompt_length'] not in p['prompt_lengths'] for r in held_rows if r['polarity'] == 'negative'),
                positive=sum(r['prompt_length'] not in p['prompt_lengths'] for r in held_rows if r['polarity'] == 'positive')),
            mean_prompt_length=round(statistics.fmean(r['prompt_length'] for r in rows_by_arm[arm]), 1),
            offsets_by_cell=tables[arm])

    learned = [r for r in results if r['position'] == 'learned' and r['arm'].startswith('english')]
    alibi = [r for r in results if r['position'] == 'alibi']
    pooled_summary = {}
    for name, subset in (('english_learned', learned), ('alibi', alibi)):
        if subset:
            pooled_summary[name] = {
                group: {bucket: pooled(subset, 'by_object_offset', group, bucket) for bucket in ('seen', 'novel')}
                for group in ('held_out_negative', 'trained_negative', 'held_out_positive', 'trained_positive')}
            kinds = Counter()
            for r in subset:
                kinds.update(r['errors']['held_out']['kinds'])
            pooled_summary[name]['held_out_error_kinds'] = dict(sorted(kinds.items()))

    output = dict(schema='sylang-combo-analysis-v1', seed=args.seed, per_cell=args.per_cell,
                  held_out_combo=list(language.HELD_OUT_COMBO), torch=torch.__version__,
                  arms=arm_info, pooled=pooled_summary, checkpoints=results, table_markdown=markdown(results))
    text = json.dumps(output, indent=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + '\n')
    print(markdown(results))
    print(f'wrote {args.out} ({len(text)} bytes)')


if __name__ == '__main__':
    main()
