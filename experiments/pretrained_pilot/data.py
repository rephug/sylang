"""Frozen datasets for the pretrained-model pilot (contract.md, interfaces.md).

    python -I data.py --version v1 --seeds 0,1,2 --train-cases 8000 --dev-cases 400 --test-cases 1000
    python -I data.py --smoke [--out DIR]        # seeds 0,1; 64 train cases per seed, 16 dev, 32 test
    python -I data.py --check [--out DIR]        # re-grade every answer file against itself, verify hashes
    python -I data.py ... --force                # replace an existing <out>/<version>; refused without it (frozen)

Environment: ``PILOT_TOKENIZER`` (Qwen 3 tokenizer directory, token counts) and ``SYLANG_PACKET`` (the study
packet, whose renderer produces the ``english`` arm; the manifest records the SHA-256 digests of the packet modules
that render it, never their path or text). Generation reuses the v3 generator of
``experiments/copy_probe`` (``study3.py``: hash split, identifier alphabet, select wordings, json record
rendering; ``language.py``: the held-out feature pairs and the packet renderer; ``sylang_batch.py``: the
header-free Sylang codes).

Underlying cases (contract.md). K records, K uniform in 1..12 (13..16 on the ``long`` panel). Identifiers are
2-5 characters of ``[a-z0-9]`` with at least one letter, accepted when ``study3.split_of`` puts them in the
file's split (``if`` and ``none`` are reserved: ``if`` opens an ``if`` record in the wire format and ``none`` is
the bare empty answer). Subject reuse: when an earlier record exists, with probability ``P_REUSE`` = 0.5 the
subject is one of the earlier subjects (uniform among distinct subjects) and the record is placed directly after
the last record of that subject; otherwise the subject is fresh and the record goes last. Object reuse: with
probability ``P_OBJECT_REUSE`` = 0.2 the object is one of the case's earlier identifiers (uniform, excluding the
record's subject and any identifier that would repeat a (subject, object) pair); otherwise fresh. The ``pair``
panel forces one record (uniform index in generation order) onto a held-out feature pair and draws the others
freely; every other file forbids held-out pairs. The lookup query is a uniform record index, except on the pair
panel where it is the forced held-out record (the index is still drawn); the wording indices are uniform over 0..3
(trained) or 4..5 (the ``wording`` panel). A case whose record tuple was already
generated anywhere in this run is redrawn (``duplicate_redraws`` in the manifest).

Random streams: one ``random.Random`` per file, seeded with the string ``"<version>:<split>:<panel>:<seed>"``
(``v1:train:iid:0``, ``v1:dev:iid:shared``, ``v1:test:iid:shared``, ``v1:test:wording:shared``, ...; the test
file is the concatenation of its four panel streams in the order iid, wording, pair, long). Draw order per case:
k, the forced index (pair panel), then per record in generation order: relation, tense, aspect, evidence,
polarity (repeated until the held-out rule holds), the subject-reuse coin and subject, the object-reuse coin and
object (a fresh identifier draws its length, then its characters, rejecting until accepted); then the query
index (drawn on every stream; the ``pair`` panel ignores it and queries the forced held-out record), the select
wording and the lookup wording. Dev and test are shared by every seed; the training stream
of a seed is one list of cases, so ``case_index < N`` is the nested prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # python -I does not add the script's directory
    sys.path.insert(0, str(HERE))

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('RAYON_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

HERE = Path(__file__).resolve().parent
COPY_PROBE = HERE.parent / 'copy_probe'
for _path in (str(COPY_PROBE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common  # noqa: E402
import grader  # noqa: E402
import language as L  # noqa: E402
import study3  # noqa: E402
import sylang_batch  # noqa: E402

VERSION = 'v1'
P_REUSE = 0.5
P_OBJECT_REUSE = 0.2
K_TRAIN = study3.K_TRAIN  # 1..12
K_LONG = study3.K_LONG  # 13..16
ID_LENGTHS = study3.ID_LENGTHS
RESERVED_IDENTIFIERS = frozenset({'if', 'none'})
HELDOUT_DESIGN = 'v2'  # language.HELD_OUT['v2']: (contain, future) and (help, completed)
TEST_PANEL_FRACTIONS = {'iid': 0.48, 'wording': 0.16, 'pair': 0.16, 'long': 0.20}
N_TRAINED_WORDINGS = 4
SMOKE = {'seeds': (0, 1), 'train_cases': 64, 'dev_cases': 16, 'test_cases': 32}
RECORD_FIELDS = ('subject', 'relation', 'object', 'polarity', 'tense', 'aspect', 'evidence')
LEGEND_TOKENS = (100, 130)
LEGEND_SPREAD = 15

# ---------------------------------------------------------------------------
# Request wordings: 0-3 trained, 4-5 held out; the bare wording is the json wording with its format phrase changed.

SELECT_WORDINGS = {
    'json': tuple(w.rstrip('\n') for w in study3.WORDINGS['train'] + study3.WORDINGS['heldout']),
    'bare': (
        'List subject object of each negative record in order, space-separated; none if none.',
        'Return the subject object pairs of the negated records, in order and space-separated, or none.',
        'Which records are negative? Answer with the subject and object of each, space-separated, in record '
        'order, none if none.',
        'Output the subject and object of every negated record as space-separated subject object pairs, in '
        'order; output none when there is none.',
        'Give the negative records as space-separated subject object pairs in their original order, or none if '
        'there are none.',
        'For each negated record, in order, emit subject object separated by a space; emit none if no record is '
        'negated.',
    ),
}
LOOKUP_WORDINGS = {
    'json': (
        'What is the relation between "{subject}" and "{object}"? Answer with a JSON object with the keys '
        'relation and polarity.',
        'Give the relation and polarity of the record with subject "{subject}" and object "{object}" as a JSON '
        'object {"relation": ..., "polarity": ...}.',
        'Look up the record where "{subject}" is the subject and "{object}" is the object; return '
        '{"relation": ..., "polarity": ...} as JSON.',
        'Which relation links "{subject}" to "{object}", and is it positive or negative? Reply with a JSON object '
        'with the keys relation and polarity.',
        'Find the record about "{subject}" and "{object}" and report its relation and polarity as a JSON object '
        'with the keys relation and polarity.',
        'For the pair "{subject}" and "{object}", state the relation and the polarity of their record in JSON '
        'with the keys relation and polarity.',
    ),
    'bare': (
        'What is the relation between "{subject}" and "{object}"? Answer with the relation and the polarity '
        'separated by a space.',
        'Give the relation and polarity of the record with subject "{subject}" and object "{object}" as two '
        'words: relation polarity.',
        'Look up the record where "{subject}" is the subject and "{object}" is the object; return relation '
        'polarity as two space-separated words.',
        'Which relation links "{subject}" to "{object}", and is it positive or negative? Reply with the relation '
        'and the polarity separated by one space.',
        'Find the record about "{subject}" and "{object}" and report its relation and polarity as two words '
        'separated by a space.',
        'For the pair "{subject}" and "{object}", state the relation and the polarity of their record as two '
        'words with a space between them.',
    ),
}
WORDINGS = {'select': SELECT_WORDINGS, 'lookup': LOOKUP_WORDINGS}
for _task, _per_format in WORDINGS.items():
    for _fmt, _list in _per_format.items():
        assert len(_list) == 6 and all(w.isascii() and not w.endswith('\n') for w in _list), (_task, _fmt)

# ---------------------------------------------------------------------------
# Legends (system messages): one per arm and answer format, 100-130 Qwen 3 tokens, arms matched within 15.

_CODE_LEGEND = ('Code: relation letter s see, h help, c contain, uppercase S, H, C when negated; then optional tense '
                'p past, f future; aspect g progressive, c completed; evidence d direct, r reported, i inferred; '
                'absent means present, simple, unspecified.')
ARM_LEGEND = {
    'english': ('Records are numbered English sentences, one per line, in record order. Each says that a subject '
                'in double quotes sees, helps or contains an object in double quotes; "not" marks a negated '
                'record; the verb form carries the tense and aspect; an opening phrase (By direct observation, '
                'By report, By inference) gives the evidence when it is known.'),
    'json': ('Records are a JSON array of objects in record order. Keys: s subject, r relation (see, help, '
             'contain), neg true if negated, t tense (past, future), a aspect (progressive, completed), e evidence '
             '(direct, reported, inferred), o object. Default values are omitted: positive, present, simple, '
             'unspecified.'),
    'sylang2': 'Records are lines of subject TAB code TAB object, one record per line, in record order. ' + _CODE_LEGEND,
    'sylang2g': 'Lines are subject TAB code object;code object, one line per subject, in record order. ' + _CODE_LEGEND,
}
TASK_LEGEND = {
    'json': ('Select task: output a JSON list of [subject,object] pairs of the negated records in record order, '
             'or [] if none. Lookup task: output {"relation":"...","polarity":"..."}. Nothing else.'),
    'bare': ('Select task: output subject object of each negated record in record order, space-separated, or '
             'none. Lookup task: output relation polarity, space-separated. Nothing else.'),
}
LEGENDS = {fmt: {arm: ARM_LEGEND[arm] + ' ' + TASK_LEGEND[fmt] for arm in common.ARMS} for fmt in common.ANSWER_FORMATS}


def legend_tokens(tok) -> dict:
    """``{format: {arm: tokens}}``; asserts the 100-130 range and the 15-token match between arms."""
    counts = {fmt: {arm: common.count_tokens(tok, text) for arm, text in arms.items()} for fmt, arms in LEGENDS.items()}
    for fmt, per_arm in counts.items():
        low, high = min(per_arm.values()), max(per_arm.values())
        assert LEGEND_TOKENS[0] <= low and high <= LEGEND_TOKENS[1], (fmt, per_arm)
        assert high - low <= LEGEND_SPREAD, (fmt, per_arm)
    return counts


# ---------------------------------------------------------------------------
# Underlying cases

def case_hash(records) -> str:
    return hashlib.sha256(json.dumps([[r[f] for f in RECORD_FIELDS] for r in records]).encode()).hexdigest()


class CaseStream:
    """Cases of one file (one ``random.Random`` stream): see the module docstring for the draw order."""

    def __init__(self, split: str, seed_key: str, k_values, heldout_pair: bool = False, heldout_wording: bool = False):
        self.split, self.seed_key, self.rng = split, seed_key, random.Random(seed_key)
        self.k_values = tuple(k_values)
        self.heldout_pair, self.heldout_wording = heldout_pair, heldout_wording
        self.wording_indices = (tuple(range(N_TRAINED_WORDINGS, 6)) if heldout_wording
                                else tuple(range(N_TRAINED_WORDINGS)))
        self._forced_record = None  # the pair panel's forced held-out record of the last ``records`` call

    def identifier(self, used: set) -> str:
        """As study3.ItemStream3.identifier: 2-5 characters of [a-z0-9], at least one letter, in this split."""
        while True:
            length = self.rng.choice(ID_LENGTHS)
            candidate = ''.join(self.rng.choice(study3.ALPHABET) for _ in range(length))
            if (not any(c in study3.LETTERS for c in candidate) or candidate in used
                    or candidate in RESERVED_IDENTIFIERS or study3.split_of(candidate) != self.split):
                continue
            used.add(candidate)
            return candidate

    def features(self, held_out: bool | None) -> dict:
        """Relation, polarity, tense, aspect, evidence; ``held_out`` True forces a held-out pair, False forbids one."""
        while True:
            relation, tense, aspect, evidence = (self.rng.choice(L.RELATIONS), self.rng.choice(L.TENSES),
                                                 self.rng.choice(L.ASPECTS), self.rng.choice(L.EVIDENCES))
            polarity = self.rng.choice(('positive', 'negative'))
            statement = L._statement('', '', relation, polarity, tense, aspect, evidence)
            if held_out is not None and L.is_held_out(statement, HELDOUT_DESIGN) != held_out:
                continue
            return {'relation': relation, 'polarity': polarity, 'tense': tense, 'aspect': aspect, 'evidence': evidence}

    def records(self, k: int) -> list[dict]:
        forced = self.rng.randrange(k) if self.heldout_pair else None
        self._forced_record = None
        records: list[dict] = []
        used: set = set()
        order: list[str] = []  # identifiers in order of first occurrence
        pairs: set = set()
        for i in range(k):
            held_out = (True if i == forced else None) if forced is not None else False
            features = self.features(held_out)
            subjects = list(dict.fromkeys(r['subject'] for r in records))
            if subjects and self.rng.random() < P_REUSE:
                subject = self.rng.choice(subjects)
            else:
                subject = self.identifier(used)
                order.append(subject)
            obj = None
            if self.rng.random() < P_OBJECT_REUSE:
                candidates = [x for x in order if x != subject and (subject, x) not in pairs]
                if candidates:
                    obj = self.rng.choice(candidates)
            if obj is None:
                obj = self.identifier(used)
                order.append(obj)
            pairs.add((subject, obj))
            record = {'subject': subject, 'relation': features['relation'], 'object': obj,
                      'polarity': features['polarity'], 'tense': features['tense'], 'aspect': features['aspect'],
                      'evidence': features['evidence']}
            if i == forced:  # the subject-adjacent insertion moves records, so the generation index is not the list index
                self._forced_record = record
            last = max((j for j, r in enumerate(records) if r['subject'] == subject), default=None)
            records.insert(len(records) if last is None else last + 1, record)
        check_records(records, self.split, self.heldout_pair)
        return records

    def case(self) -> dict:
        k = self.rng.choice(self.k_values)
        records = self.records(k)
        index = self.rng.randrange(k)  # drawn on every stream: the documented draw order is the same everywhere
        query = self._forced_record if self.heldout_pair else records[index]
        assert query is not None and (not self.heldout_pair or L.is_held_out(query, HELDOUT_DESIGN)), query
        wording = {'select': self.rng.choice(self.wording_indices), 'lookup': self.rng.choice(self.wording_indices)}
        return {'k': k, 'records': records, 'wording': wording,
                'query': {'subject': query['subject'], 'object': query['object']},
                'heldout_wording': self.heldout_wording, 'heldout_pair': self.heldout_pair}


def check_records(records: list[dict], split: str, heldout_pair: bool) -> None:
    """The contract's invariants of one case."""
    pairs = [(r['subject'], r['object']) for r in records]
    assert len(set(pairs)) == len(pairs), 'repeated (subject, object) pair'
    for r in records:
        for field in ('subject', 'object'):
            identifier = r[field]
            assert 2 <= len(identifier) <= 5 and all(c in study3.ALPHABET for c in identifier), identifier
            assert any(c in study3.LETTERS for c in identifier), identifier
            assert identifier not in RESERVED_IDENTIFIERS and study3.split_of(identifier) == split, (identifier, split)
        assert r['subject'] != r['object'], r
    first_last: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        first_last.setdefault(r['subject'], [i, i])[1] = i
    for subject, (first, last) in first_last.items():
        assert all(records[j]['subject'] == subject for j in range(first, last + 1)), 'subject group not adjacent'
    held = [L.is_held_out(r, HELDOUT_DESIGN) for r in records]
    assert any(held) if heldout_pair else not any(held), (heldout_pair, held)


def panel_sizes(n_test: int) -> dict:
    sizes = {panel: int(n_test * fraction) for panel, fraction in TEST_PANEL_FRACTIONS.items()}
    sizes['iid'] += n_test - sum(sizes.values())
    return sizes


def file_plan(version: str, seeds, train_cases: int, dev_cases: int, test_cases: int) -> list[dict]:
    """One entry per case file: split, seed (train only) and its panels (name, size, stream settings)."""
    plan = [{'split': 'dev', 'seed': None,
             'panels': [{'panel': 'iid', 'n': dev_cases, 'seed_key': f'{version}:dev:iid:shared', 'k': K_TRAIN}]}]
    sizes = panel_sizes(test_cases)
    plan.append({'split': 'test', 'seed': None, 'panels': [
        {'panel': 'iid', 'n': sizes['iid'], 'seed_key': f'{version}:test:iid:shared', 'k': K_TRAIN},
        {'panel': 'wording', 'n': sizes['wording'], 'seed_key': f'{version}:test:wording:shared', 'k': K_TRAIN,
         'heldout_wording': True},
        {'panel': 'pair', 'n': sizes['pair'], 'seed_key': f'{version}:test:pair:shared', 'k': K_TRAIN,
         'heldout_pair': True},
        {'panel': 'long', 'n': sizes['long'], 'seed_key': f'{version}:test:long:shared', 'k': K_LONG},
    ]})
    for seed in seeds:
        plan.append({'split': 'train', 'seed': seed, 'panels': [
            {'panel': 'iid', 'n': train_cases, 'seed_key': f'{version}:train:iid:{seed}', 'k': K_TRAIN}]})
    return plan


def generate_cases(entry: dict, seen_hashes: set) -> tuple[list[dict], int]:
    """The case lines of one file (interfaces.md), in panel order; redraws a case seen anywhere before."""
    split, seed = entry['split'], entry['seed']
    prefix = f'{split}-c' if seed is None else f'{split}-s{seed}-c'
    cases, redraws = [], 0
    for panel in entry['panels']:
        stream = CaseStream(split, panel['seed_key'], panel['k'], heldout_pair=panel.get('heldout_pair', False),
                            heldout_wording=panel.get('heldout_wording', False))
        for _ in range(panel['n']):
            while True:
                case = stream.case()
                digest = case_hash(case['records'])
                if digest not in seen_hashes:
                    break
                redraws += 1
            seen_hashes.add(digest)
            index = len(cases)
            cases.append({'case': f'{prefix}{index:05d}', 'case_index': index, 'split': split, 'panel': panel['panel'],
                          'k': case['k'], 'records': case['records'], 'wording': case['wording'],
                          'query': case['query'], 'heldout_wording': case['heldout_wording'],
                          'heldout_pair': case['heldout_pair']})
    return cases, redraws


# ---------------------------------------------------------------------------
# Renderings and answers

def statements(records: list[dict]) -> list[dict]:
    return [L._statement(r['subject'], r['object'], r['relation'], r['polarity'], r['tense'], r['aspect'], r['evidence'])
            for r in records]


def group_by_subject(sylang2_text: str) -> str:
    """sylang2g: one line per subject in order of first occurrence, ``subject<TAB>code object;code object``."""
    groups: dict[str, list[str]] = {}
    for line in sylang2_text.split('\n')[:-1]:
        subject, code, obj = line.split('\t')
        groups.setdefault(subject, []).append(f'{code} {obj}')
    return ''.join(f'{subject}\t{";".join(items)}\n' for subject, items in groups.items())


def render(arm: str, records: list[dict], renderer) -> str:
    if arm == 'english':
        text = renderer.payload('english', statements(records))
    elif arm == 'json':
        text = '[' + ','.join(study3._json_record(r) for r in records) + ']\n'
    elif arm == 'sylang2':
        text = sylang_batch.encode(records, version=2)
    elif arm == 'sylang2g':
        text = group_by_subject(sylang_batch.encode(records, version=2))
    else:
        raise ValueError(arm)
    assert text.isascii() and text.endswith('\n'), arm
    return text


def select_answers(records: list[dict]) -> dict:
    pairs = [[r['subject'], r['object']] for r in records if r['polarity'] == 'negative']
    return {'json': json.dumps(pairs, separators=(',', ':')),
            'bare': ' '.join(f'{s} {o}' for s, o in pairs) if pairs else grader.NONE_WORD}


def lookup_answers(records: list[dict], query: dict) -> dict:
    matches = [r for r in records if r['subject'] == query['subject'] and r['object'] == query['object']]
    assert len(matches) == 1, query
    record = matches[0]
    return {'json': json.dumps({'relation': record['relation'], 'polarity': record['polarity']}, separators=(',', ':')),
            'bare': f"{record['relation']} {record['polarity']}"}


def instructions(task: str, index: int, query: dict | None = None) -> dict:
    out = {}
    for fmt in common.ANSWER_FORMATS:
        text = WORDINGS[task][fmt][index]
        if task == 'lookup':
            text = text.replace('{subject}', query['subject']).replace('{object}', query['object'])
        out[fmt] = text
    return out


def render_summary(times: list[float]) -> dict:
    """Mean, 95th percentile and maximum of the per-case rendering wall-clock, in seconds."""
    if not times:
        return {'n': 0, 'mean': None, 'p95': None, 'max': None}
    ordered = sorted(times)
    return {'n': len(times), 'mean': sum(times) / len(times), 'p95': ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            'max': ordered[-1]}


def examples_of(case: dict, arm: str, renderer, render_times: list | None = None) -> list[dict]:
    """The two example lines of a case in one arm: select, then lookup (interfaces.md). The arm's
    rendering wall-clock (the contract's conversion-in) is appended to ``render_times``; it is
    summarised in the manifest, not written to the example line, so the files stay byte-reproducible."""
    records = case['records']
    t0 = time.perf_counter()
    rendering = render(arm, records, renderer)
    if render_times is not None:
        render_times.append(time.perf_counter() - t0)
    out = []
    for task in common.TASKS:
        example = {'id': f"{case['case']}-{task}", 'case': case['case'], 'case_index': case['case_index'],
                   'split': case['split'], 'panel': case['panel'], 'task': task, 'arm': arm, 'k': case['k'],
                   'max_id_len': max(len(r[f]) for r in records for f in ('subject', 'object')),
                   'n_negative': sum(r['polarity'] == 'negative' for r in records),
                   'wording': case['wording'][task], 'heldout_wording': case['heldout_wording'],
                   'heldout_pair': case['heldout_pair'], 'records': rendering}
        if task == 'select':
            example['instruction'] = instructions('select', case['wording']['select'])
            example['answer'] = select_answers(records)
        else:
            example['instruction'] = instructions('lookup', case['wording']['lookup'], case['query'])
            example['answer'] = lookup_answers(records, case['query'])
            example['query'] = dict(case['query'])
        out.append(example)
    return out


# ---------------------------------------------------------------------------
# Token statistics (batched; identical numbers to common.token_breakdown, which is asserted on a sample)

def _count_many(tok, texts: list[str]) -> list[int]:
    if not texts:
        return []
    return [len(ids) for ids in tok(texts, add_special_tokens=False)['input_ids']]


def breakdowns(tok, legend: str, examples: list[dict], answer_format: str) -> list[dict]:
    system = common.count_tokens(tok, legend)
    records = _count_many(tok, [e['records'] for e in examples])
    instruction = _count_many(tok, [e['instruction'][answer_format] for e in examples])
    total = _count_many(tok, [common.prompt_text(legend, common.user_text(e, answer_format)) for e in examples])
    return [{'system': system, 'records': r, 'instruction': i, 'template': t - system - r - i, 'prompt_total': t}
            for r, i, t in zip(records, instruction, total)]


class TokenStats:
    PARTS = ('system', 'records', 'instruction', 'template', 'prompt_total')

    def __init__(self, tok):
        self.tok = tok
        self.prompt = defaultdict(lambda: defaultdict(float))  # (fmt, arm, task, split) -> sums
        self.answer = defaultdict(lambda: defaultdict(float))  # (fmt, task, split) -> sums
        self.verified = 0

    def add(self, examples: list[dict], arm: str, split: str) -> None:
        for fmt in common.ANSWER_FORMATS:
            legend = LEGENDS[fmt][arm]
            parts = breakdowns(self.tok, legend, examples, fmt)
            for i in (0, len(examples) // 2, len(examples) - 1):  # the batched numbers equal the reference's
                assert parts[i] == common.token_breakdown(self.tok, legend, examples[i], fmt), (arm, fmt, i)
                self.verified += 1
            answers = _count_many(self.tok, [e['answer'][fmt] for e in examples])
            for example, part, n_answer in zip(examples, parts, answers):
                key = (fmt, arm, example['task'], split)
                self.prompt[key]['n'] += 1
                for name in self.PARTS:
                    self.prompt[key][name] += part[name]
                akey = (fmt, example['task'], split)
                self.answer[akey]['n'] += 1
                self.answer[akey]['tokens'] += n_answer

    def summary(self) -> dict:
        out = {fmt: {'prompt': {}, 'answer': {}} for fmt in common.ANSWER_FORMATS}
        for (fmt, arm, task, split), sums in self.prompt.items():
            cell = out[fmt]['prompt'].setdefault(arm, {}).setdefault(task, {})
            cell[split] = {'n': int(sums['n']), **{name: round(sums[name] / sums['n'], 2) for name in self.PARTS}}
        for (fmt, task, split), sums in self.answer.items():
            out[fmt]['answer'].setdefault(task, {})[split] = {'n': int(sums['n']),
                                                              'mean': round(sums['tokens'] / sums['n'], 2)}
        # Means over every split (train counts every seed), per arm and task: the headline numbers.
        for fmt in common.ANSWER_FORMATS:
            table = {}
            for arm in common.ARMS:
                table[arm] = {}
                for task in list(common.TASKS) + ['all']:
                    keys = [k for k in self.prompt if k[0] == fmt and k[1] == arm and (task == 'all' or k[2] == task)]
                    n = sum(self.prompt[k]['n'] for k in keys)
                    table[arm][task] = round(sum(self.prompt[k]['prompt_total'] for k in keys) / n, 2) if n else None
            out[fmt]['prompt_mean_by_arm'] = table
        return out


def print_table(summary: dict) -> None:
    for fmt in common.ANSWER_FORMATS:
        print(f'mean prompt tokens ({fmt} answer format, all splits):')
        print(f'  {"arm":9} {"select":>8} {"lookup":>8} {"all":>8}')
        for arm, row in summary[fmt]['prompt_mean_by_arm'].items():
            print(f'  {arm:9} ' + ' '.join(f'{row[t]:8.1f}' if row[t] is not None else f'{"-":>8}'
                                          for t in ('select', 'lookup', 'all')))


# ---------------------------------------------------------------------------
# Generation

def packet_renderer_files(rt) -> list[Path]:
    """The packet modules that render the ``english`` arm: the runtime's data module and the module behind
    ``data.SCREEN`` (a module object in the study packet; otherwise the module defining its class), plus the file
    defining ``SCREEN.encode_english`` when it is another one."""
    screen = rt.data.SCREEN
    screen_module = screen if isinstance(screen, types.ModuleType) else sys.modules[type(screen).__module__]
    files = {Path(rt.data.__file__), Path(screen_module.__file__)}
    code = getattr(getattr(screen.encode_english, '__func__', screen.encode_english), '__code__', None)
    if code is not None and Path(code.co_filename).is_file():
        files.add(Path(code.co_filename))
    return sorted(p.resolve() for p in files)


def packet_renderer_digests(rt) -> list[str]:
    """SHA-256 digests (sorted; never a path) of ``packet_renderer_files``: the manifest's identity of the renderer."""
    return sorted(common.sha256_file(p) for p in packet_renderer_files(rt))


def refuse_frozen(out_dir: Path, force: bool) -> None:
    """Datasets are frozen: an existing ``<out>/<version>/manifest.json`` is replaced only with ``--force``."""
    existing = out_dir / 'manifest.json'
    if not existing.exists() or force:
        return
    with open(existing, encoding='utf-8') as f:
        old = json.load(f)
    raise SystemExit(f'{existing} exists: dataset version {old.get("version")!r} generated at {old.get("generated_at")} '
                     f'by generator commit {(old.get("generator") or {}).get("git_commit")}; datasets are frozen: '
                     f'choose a new --version or pass --force to replace it')


def generate(args) -> None:
    from followup import Runtime  # the packet runtime: its data module renders the english arm
    out_dir = Path(args.out) / args.version
    refuse_frozen(out_dir, args.force)
    packet = os.environ.get(common.PACKET_ENV)
    if not packet:
        raise SystemExit(f'set {common.PACKET_ENV} to the study packet directory (english rendering)')
    clock = common.Stopwatch()
    tok = common.load_tokenizer()
    common.check_template(tok)
    legend_counts = legend_tokens(tok)
    rt = Runtime(packet)
    renderer = L.Renderer(rt)
    packet_digests = packet_renderer_digests(rt)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    train_cases, dev_cases, test_cases = args.train_cases, args.dev_cases, args.test_cases
    if args.smoke:
        seeds, train_cases, dev_cases, test_cases = list(SMOKE['seeds']), SMOKE['train_cases'], SMOKE['dev_cases'], SMOKE['test_cases']
    plan = file_plan(args.version, seeds, train_cases, dev_cases, test_cases)
    render_times: dict = {}  # arm -> split -> rendering seconds per case (conversion-in, contract.md)

    common.write_json(out_dir / 'legends.json', LEGENDS)
    files = ['legends.json']
    counts, cases_per_file, split_hashes, rng_seeds = {}, {}, {}, {}
    seen_hashes: set = set()
    redraws = 0
    stats = TokenStats(tok)
    for entry in plan:
        split, seed = entry['split'], entry['seed']
        started = time.perf_counter()
        cases, n_redraws = generate_cases(entry, seen_hashes)
        redraws += n_redraws
        cases_name = f'cases_{split}.jsonl' if seed is None else f'cases_train_s{seed}.jsonl'
        common.write_jsonl(out_dir / cases_name, cases)
        files.append(cases_name)
        cases_per_file[cases_name] = len(cases)
        split_hashes[cases_name] = hashlib.sha256(''.join(case_hash(c['records']) for c in cases).encode()).hexdigest()
        for panel in entry['panels']:
            rng_seeds[f'{cases_name}:{panel["panel"]}'] = panel['seed_key']
        for arm in common.ARMS:
            timings = render_times.setdefault(arm, {}).setdefault(split, [])
            examples = [example for case in cases for example in examples_of(case, arm, renderer, timings)]
            path = common.examples_path(args.version, split, arm, seed)
            path = out_dir / path.name
            common.write_jsonl(path, examples)
            files.append(path.name)
            counts[path.name] = len(examples)
            stats.add(examples, arm, split)
        print(f'{cases_name}: {len(cases)} cases, {len(common.ARMS)} arms, {time.perf_counter() - started:.1f}s')

    answers_identical_across_arms(out_dir, args.version, plan)
    summary = stats.summary()
    manifest = {
        'version': args.version,
        'generated_at': common.now_iso(),
        'generator': {'git_commit': common.git_commit(), 'sha256': {
            'data.py': common.sha256_file(HERE / 'data.py'),
            'study3.py': common.sha256_file(COPY_PROBE / 'study3.py'),
            'sylang_batch.py': common.sha256_file(COPY_PROBE / 'sylang_batch.py'),
            'language.py': common.sha256_file(COPY_PROBE / 'language.py')},
                      'packet_english_renderer_sha256': packet_digests},
        'seeds': seeds,
        'smoke': bool(args.smoke),
        'sizes': {'train': train_cases, 'dev': dev_cases, 'test': test_cases, 'test_panels': panel_sizes(test_cases),
                  'cases_per_file': cases_per_file},
        'counts': counts,
        'sha256': {name: common.sha256_file(out_dir / name) for name in files},
        'split_hashes': split_hashes,
        'cases_disjoint': True,
        'duplicate_redraws': redraws,
        'parameters': {'p_reuse': P_REUSE, 'p_object_reuse': P_OBJECT_REUSE, 'k_train': list(K_TRAIN),
                       'k_long': list(K_LONG), 'id_lengths': list(ID_LENGTHS), 'reserved_identifiers': sorted(RESERVED_IDENTIFIERS),
                       'trained_wording_indices': list(range(N_TRAINED_WORDINGS))},
        'rng': {'scheme': 'random.Random("<version>:<split>:<panel>:<seed|shared>") per file and panel; see data.py',
                'seeds': rng_seeds},
        'wordings': {task: {fmt: list(per_fmt) for fmt, per_fmt in per_task.items()} for task, per_task in WORDINGS.items()},
        'heldout': {'wordings': {task: {fmt: list(per_fmt[N_TRAINED_WORDINGS:]) for fmt, per_fmt in per_task.items()}
                                 for task, per_task in WORDINGS.items()},
                    'wording_indices': list(range(N_TRAINED_WORDINGS, 6)),
                    'feature_pairs': [list(values) for _, values in L.HELD_OUT[HELDOUT_DESIGN]],
                    'k_long': list(K_LONG)},
        'legend_tokens': legend_counts,
        'legend_token_range': list(LEGEND_TOKENS),
        'token_stats': summary,
        'render_seconds': {arm: {split: render_summary(times) for split, times in per_split.items()}
                           for arm, per_split in render_times.items()},
        'token_stats_verified_against_common': stats.verified,
        'seconds': round(clock.seconds(), 1),
    }
    assert str(Path(packet).resolve()) not in json.dumps(manifest), 'the packet path must not appear in the manifest'
    common.write_json(out_dir / 'manifest.json', manifest)
    print(f'wrote {len(files) + 1} files to {out_dir} in {clock.seconds():.1f}s (duplicate redraws: {redraws})')
    print('legend tokens:', json.dumps(legend_counts))
    print_table(summary)


def answers_identical_across_arms(out_dir: Path, version: str, plan: list[dict]) -> None:
    for entry in plan:
        answers = None
        for arm in common.ARMS:
            rows = common.read_jsonl(out_dir / common.examples_path(version, entry['split'], arm, entry['seed']).name)
            these = [(r['id'], r['answer']['json'], r['answer']['bare']) for r in rows]
            assert answers is None or these == answers, (entry, arm)
            answers = these


# ---------------------------------------------------------------------------
# --check: hashes, self-grading, invariants

def check(args) -> int:
    out_dir = Path(args.out) / args.version
    with open(out_dir / 'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    problems: list[str] = []

    def problem(text: str) -> None:
        problems.append(text)
        print('PROBLEM:', text)

    for name, digest in manifest['sha256'].items():
        path = out_dir / name
        if not path.exists():
            problem(f'{name}: missing')
        elif common.sha256_file(path) != digest:
            problem(f'{name}: sha256 differs from the manifest')
    with open(out_dir / 'legends.json', encoding='utf-8') as f:
        legends = json.load(f)
    if legends != LEGENDS:
        problem('legends.json differs from the legends in data.py')
    if os.environ.get(common.TOKENIZER_ENV):
        tok = common.load_tokenizer()
        counts = {fmt: {arm: common.count_tokens(tok, text) for arm, text in arms.items()} for fmt, arms in legends.items()}
        if counts != manifest['legend_tokens']:
            problem(f'legend token counts differ from the manifest: {counts}')
    else:
        print(f'note: {common.TOKENIZER_ENV} not set; legend token counts not re-verified')
    if os.environ.get(common.PACKET_ENV):
        from followup import Runtime
        digests = packet_renderer_digests(Runtime(os.environ[common.PACKET_ENV]))
        recorded = (manifest.get('generator') or {}).get('packet_english_renderer_sha256')
        if recorded != digests:
            problem(f'packet english renderer digests differ from the manifest: packet {digests}, manifest {recorded}')
    else:
        print(f'note: {common.PACKET_ENV} not set; packet english renderer digests not re-verified')

    graded = 0
    cases_files = sorted(name for name in manifest['sha256'] if name.startswith('cases_'))
    for cases_name in cases_files:
        cases = common.read_jsonl(out_dir / cases_name)
        split = cases[0]['split']
        seed = None if split != 'train' else int(cases_name[len('cases_train_s'):-len('.jsonl')])
        seen_pairs: set = set()
        for index, case in enumerate(cases):
            try:
                check_records(case['records'], split, case['heldout_pair'])
            except AssertionError as error:
                problem(f'{cases_name} {case["case"]}: {error}')
            if case['case_index'] != index:
                problem(f'{cases_name}: case_index {case["case_index"]} at line {index}')
            k_values = K_LONG if case['panel'] == 'long' else K_TRAIN
            if case['k'] != len(case['records']) or case['k'] not in k_values:
                problem(f'{cases_name} {case["case"]}: k {case["k"]} outside {k_values}')
            indices = range(N_TRAINED_WORDINGS, 6) if case['heldout_wording'] else range(N_TRAINED_WORDINGS)
            if any(case['wording'][task] not in indices for task in common.TASKS):
                problem(f'{cases_name} {case["case"]}: wording indices {case["wording"]} outside {list(indices)}')
            if (case['panel'] == 'wording') != case['heldout_wording'] or (case['panel'] == 'pair') != case['heldout_pair']:
                problem(f'{cases_name} {case["case"]}: panel/held-out flags disagree')
            query = (case['query']['subject'], case['query']['object'])
            queried = [r for r in case['records'] if (r['subject'], r['object']) == query]
            if len(queried) != 1:
                problem(f'{cases_name} {case["case"]}: query does not match exactly one record')
            elif case['heldout_pair'] and not L.is_held_out(queried[0], HELDOUT_DESIGN):
                problem(f'{cases_name} {case["case"]}: pair-panel query record is not a held-out feature pair')
            digest = case_hash(case['records'])
            if digest in seen_pairs:
                problem(f'{cases_name} {case["case"]}: duplicate case')
            seen_pairs.add(digest)
        answers = None
        for arm in common.ARMS:
            name = common.examples_path(args.version, split, arm, seed).name
            rows = common.read_jsonl(out_dir / name)
            if manifest['counts'].get(name) != len(rows) or len(rows) != 2 * len(cases):
                problem(f'{name}: {len(rows)} examples for {len(cases)} cases (manifest says {manifest["counts"].get(name)})')
                continue
            for i, row in enumerate(rows):
                case, task = cases[i // 2], common.TASKS[i % 2]
                if row['id'] != f'{case["case"]}-{task}' or row['case_index'] != case['case_index'] or row['task'] != task \
                        or row['arm'] != arm or row['wording'] != case['wording'][task]:
                    problem(f'{name} line {i}: header fields disagree with {cases_name}')
                if not row['records'].endswith('\n') or any(row['instruction'][f].endswith('\n') for f in common.ANSWER_FORMATS):
                    problem(f'{name} {row["id"]}: records must end with a newline and the instruction must not')
                expected = select_answers(case['records']) if task == 'select' else lookup_answers(case['records'], case['query'])
                if row['answer'] != expected:
                    problem(f'{name} {row["id"]}: answer differs from the case records')
                for fmt in common.ANSWER_FORMATS:
                    result = grader.grade(task, fmt, row['answer'][fmt], row['answer'][fmt])
                    graded += 1
                    if not (result['correct'] and not result['invalid'] and result['reason'] == 'ok'):
                        problem(f'{name} {row["id"]} {fmt}: self-grade {result}')
            these = [(r['id'], r['answer']) for r in rows]
            if answers is not None and these != answers:
                problem(f'{name}: answers differ from the other arms')
            answers = these
    print(f'checked {len(manifest["sha256"])} hashes, {len(cases_files)} case files, {graded} self-graded answers: '
          f'{"OK" if not problems else f"{len(problems)} problems"}')
    return 1 if problems else 0


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--version', default=VERSION)
    parser.add_argument('--seeds', default='0,1,2', help='comma-separated training seeds')
    parser.add_argument('--train-cases', type=int, default=8000, help='cases per training seed')
    parser.add_argument('--dev-cases', type=int, default=400)
    parser.add_argument('--test-cases', type=int, default=1000, help='split iid 48%%, wording 16%%, pair 16%%, long 20%%')
    parser.add_argument('--out', default=str(common.DATA_DIR), help='data directory; files go to <out>/<version>/')
    parser.add_argument('--smoke', action='store_true', help=f'seeds {SMOKE["seeds"]}, {SMOKE["train_cases"]} train cases '
                                                             f'per seed, {SMOKE["dev_cases"]} dev, {SMOKE["test_cases"]} test')
    parser.add_argument('--check', action='store_true', help='verify an existing dataset instead of generating')
    parser.add_argument('--force', action='store_true', help='replace an existing <out>/<version> (datasets are frozen; '
                                                             'without this flag an existing manifest.json is refused)')
    args = parser.parse_args(argv)
    if args.check:
        return check(args)
    generate(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
