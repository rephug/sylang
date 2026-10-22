"""Design v3 of the language comparison: larger record sets, variable-length identifiers,
varied request wording, machine-format comparators and a fully held-out set.

Used through ``language.py --suite study3``; this module holds the data design only.

**Records** are the same seven-field predicates as before (subject, relation, object,
polarity, tense, aspect, evidence; see ``language.py``). What changes:

- **Identifiers** are 2–5 characters from ``[a-z0-9]``, at least one letter, drawn from the
  whole space and split by hash into train / dev / test (``split_of``). The v2 training
  identifiers (letter-digit-letter) are excluded from the v3 dev and test splits, so a model
  trained on the v2 corpus is also evaluated on identifiers it never saw.
- **Request wording**: each item carries one of several instruction phrasings. Four are
  used in training (``WORDINGS['train']``); two are held out (``WORDINGS['heldout']``). All ask
  for the same answer: the JSON list of ``[subject, object]`` of the negative records in
  record order, or ``[]``.
- **Record counts**: K uniform in 1..12 in the full corpus; the ``long`` panel has K = 13..16,
  never seen in training.
- **Held-out combinations** as in v2: (contain, future) and (help, completed).

**Arms** (renderings of the same item; the answer bytes are identical in every arm):

- ``english``: the packet's controlled English, explicitly the readable comparison.
- ``sylang2``: Sylang batch codes without the header (the opt-in header-free v2 format,
  identical record lines to the current ``codes`` format).
- ``sylang1``: the current format with its header line (reference).
- ``json``: compact JSON, one object per record with one-letter keys and default feature
  values omitted, e.g. ``{"s":"k4m","r":"contain","neg":true,"t":"past","o":"f9b"}``.
- ``csv``: one comma-separated row per record with every field spelled out, no header:
  ``k4m,contain,negative,past,progressive,direct,f9b``.
- ``sylang2c``: ``sylang2`` with a one-line record count first (``3``), an exploratory arm that tests
  whether an explicit count cue helps extrapolation to unseen record counts (doc §10.6.3).

**Panels** (fixed items; identifiers from the test split unless stated): ``iid`` (trained
wording, trained combinations, K 1..12); ``wording`` (held-out wording); ``combo`` (a
held-out combination); ``long`` (K 13..16); ``full`` (held-out wording, a held-out
combination and K 13..16 together: the fully held-out set); ``reorder``, ``swap`` and
``flip`` (counterfactuals of iid); ``dev`` and its counterfactuals for early stopping and
checkpoint selection only. Accuracy is also reported by the longest identifier in the item.

**Corpus variants for the pilot** (what a model is trained on): ``v2`` = K 1..8, three-character
identifiers, one wording (the v2.1 corpus); ``k12`` adds K up to 12; ``ids`` adds variable-length
identifiers; ``wording`` adds the four training phrasings; ``all`` combines the three.
"""

from __future__ import annotations

import hashlib
import json
import random

import language as L

ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789'
LETTERS = ALPHABET[:26]
ID_LENGTHS = (2, 3, 4, 5)
K_TRAIN = tuple(range(1, 13))
K_LONG = (13, 14, 15, 16)
K_LONGER = (17, 18, 19, 20)  # extrapolation panel for the all16 corpus (targeted expansion, §10.4)

WORDINGS = {
    'train': (
        L.INSTRUCTION,
        'Return the [subject,object] pairs of the negated records, in order, or [].\n',
        'Which records are negative? Answer with a JSON list of [subject,object] in record order, [] if none.\n',
        'Output the subject and object of every negated record as [subject,object] pairs, in order; output [] when there is none.\n',
    ),
    'heldout': (
        'Give the negative records as a JSON list of [subject,object] pairs in their original order, or [] if there are none.\n',
        'For each negated record, in order, emit [subject,object]; emit [] if no record is negated.\n',
    ),
}

ARMS3 = ('english', 'sylang2', 'sylang1', 'json', 'csv', 'sylang2c')  # sylang2c: header-free + a record-count line (exploratory)
DEFAULTS = {'tense': 'present', 'aspect': 'simple', 'evidence': 'unspecified'}

CORPORA = {
    'v2': dict(k=tuple(range(1, 9)), lengths=None, wordings=(L.INSTRUCTION,)),
    'k12': dict(k=K_TRAIN, lengths=None, wordings=(L.INSTRUCTION,)),
    'ids': dict(k=tuple(range(1, 9)), lengths=ID_LENGTHS, wordings=(L.INSTRUCTION,)),
    'wording': dict(k=tuple(range(1, 9)), lengths=None, wordings=WORDINGS['train']),
    'all': dict(k=K_TRAIN, lengths=ID_LENGTHS, wordings=WORDINGS['train']),
    'all16': dict(k=tuple(range(1, 17)), lengths=ID_LENGTHS, wordings=WORDINGS['train']),  # targeted expansion: K 1..16
}


class Item(list):
    """A list of statements that also carries the request wording."""

    def __init__(self, statements, instruction: str = L.INSTRUCTION):
        super().__init__(statements)
        self.instruction = instruction


def split_of(identifier: str) -> str:
    """Hash split over the whole identifier space; v2 training identifiers are never dev/test."""
    bucket = hashlib.sha256(('v3:' + identifier).encode()).digest()[0] % 10
    if bucket in (0, 1) and identifier in V2_TRAIN:
        return 'train'
    return {0: 'dev', 1: 'test'}.get(bucket, 'train')


def _v2_train_ids() -> set:
    import tasks
    return set(tasks.FreshIds('train').ids)


V2_TRAIN = _v2_train_ids()


class ItemStream3:
    """Items for design v3. ``lengths=None`` draws v2-style letter-digit-letter identifiers
    from the v2 splits; otherwise identifiers of the given lengths from the full space."""

    def __init__(self, split: str, seed: int, k_values=K_TRAIN, lengths=ID_LENGTHS,
                 wordings=WORDINGS['train'], combos: str = 'train'):
        self.split, self.rng = split, random.Random(seed)
        self.k_values, self.lengths, self.wordings, self.combos = tuple(k_values), lengths, tuple(wordings), combos
        self.v2 = None if lengths else L.ItemStream(split, seed, design='v2')

    def identifier(self, used: set) -> str:
        if self.v2 is not None:
            while True:
                candidate = self.rng.choice(self.v2.ids)
                if candidate not in used:
                    used.add(candidate)
                    return candidate
        while True:
            length = self.rng.choice(self.lengths)
            candidate = ''.join(self.rng.choice(ALPHABET) for _ in range(length))
            if not any(c in LETTERS for c in candidate) or candidate in used or split_of(candidate) != self.split:
                continue
            used.add(candidate)
            return candidate

    def record(self, used: set, held_out: bool | None, polarity: str | None = None) -> dict:
        while True:
            features = (self.rng.choice(L.RELATIONS), self.rng.choice(L.TENSES), self.rng.choice(L.ASPECTS),
                        self.rng.choice(L.EVIDENCES))
            statement = L._statement('', '', features[0], polarity or self.rng.choice(('positive', 'negative')),
                                     *features[1:])
            if held_out is not None and L.is_held_out(statement, 'v2') != held_out:
                continue
            subject = self.identifier(used)
            obj = self.identifier(used)
            return dict(statement, subject=subject, object=obj)

    def item(self, k: int | None = None, wording: str | None = None) -> Item:
        k = k or self.rng.choice(self.k_values)
        used: set = set()
        if self.combos == 'train':
            records = [self.record(used, False) for _ in range(k)]
        else:
            forced = self.rng.randrange(k)
            records = [self.record(used, True if i == forced else None) for i in range(k)]
        return Item(records, wording or self.rng.choice(self.wordings))


# ---------------------------------------------------------------------------
# Renderings


def _json_record(s: dict) -> str:
    fields = {'s': s['subject'], 'r': s['relation']}
    if s['polarity'] == 'negative':
        fields['neg'] = True
    for key, name in (('t', 'tense'), ('a', 'aspect'), ('e', 'evidence')):
        if s[name] != DEFAULTS[name]:
            fields[key] = s[name]
    fields['o'] = s['object']
    return json.dumps(fields, separators=(',', ':'))


def render(renderer: L.Renderer, arm: str, item) -> str:
    """The prompt of ``item`` in ``arm``: the records, then the item's request wording."""
    instruction = getattr(item, 'instruction', L.INSTRUCTION)
    statements = list(item)
    if arm == 'english':
        body = renderer.payload('english', statements)
    elif arm == 'sylang1':
        body = renderer.payload('codes', statements)
    elif arm == 'sylang2':
        body = renderer.payload('codes', statements).split('\n', 1)[1]
    elif arm == 'sylang2c':  # the §9.4 variant: no header, a one-line record count, then the record lines
        body = f'{len(statements)}\n' + renderer.payload('codes', statements).split('\n', 1)[1]
    elif arm == 'json':
        body = '[' + ','.join(_json_record(s) for s in statements) + ']\n'
    elif arm == 'csv':
        body = ''.join(','.join(s[f] for f in ('subject', 'relation', 'polarity', 'tense', 'aspect', 'evidence',
                                               'object')) + '\n' for s in statements)
    else:
        raise ValueError(arm)
    text = body + instruction
    assert text.isascii()
    return text


def _wrap(statements, item) -> Item:
    return Item(statements, item.instruction)


def panels(k_values, per_k: int, lengths=ID_LENGTHS, seed: int = 7, k_long=K_LONG, k_longer=None) -> dict:
    """Design-v3 panels on test-split identifiers (dev for the dev panels). ``k_long`` is the
    held-out record-count range of ``long`` and ``full``; ``k_longer`` adds a ``longer`` panel."""
    def stream(split, offset, **kw):
        return ItemStream3(split, seed + offset, lengths=lengths, **kw)

    test = stream('test', 0, k_values=k_values)
    wording = stream('test', 1, k_values=k_values, wordings=WORDINGS['heldout'])
    combo = stream('test', 2, k_values=k_values, combos='held_out')
    long = stream('test', 3, k_values=k_long)
    full = stream('test', 4, k_values=k_long, wordings=WORDINGS['heldout'], combos='held_out')
    dev = stream('dev', 5, k_values=k_values)
    half = max(8, per_k // 2)
    out = {
        'iid': [test.item(k) for k in k_values for _ in range(per_k)],
        'wording': [wording.item(k) for k in k_values for _ in range(half)],
        'combo': [combo.item(k) for k in k_values for _ in range(half)],
        'long': [long.item(k) for k in k_long for _ in range(per_k)],
        'full': [full.item(k) for k in k_long for _ in range(per_k)],
        'dev': [dev.item(k) for k in k_values for _ in range(per_k)],
    }
    if k_longer:
        longer = stream('test', 6, k_values=k_longer)
        out['longer'] = [longer.item(k) for k in k_longer for _ in range(per_k)]
    rng = random.Random(seed + 8)
    out['reorder'] = [_wrap(L.reorder(it, rng), it) for it in out['iid'] if len(it) >= 2]
    out['swap'] = [_wrap(L.swap_roles(it), it) for it in out['iid'] if any(s['polarity'] == 'negative' for s in it)]
    out['flip'] = [_wrap(L.flip_one(it, rng), it) for it in out['iid']]
    dev_rng = random.Random(seed + 9)
    out['dev_reorder'] = [_wrap(L.reorder(it, dev_rng), it) for it in out['dev'] if len(it) >= 2]
    out['dev_swap'] = [_wrap(L.swap_roles(it), it) for it in out['dev'] if any(s['polarity'] == 'negative' for s in it)]
    out['dev_flip'] = [_wrap(L.flip_one(it, dev_rng), it) for it in out['dev']]
    return out


STOP_PANELS = ('dev', 'dev_swap', 'dev_flip', 'dev_reorder')


def longest_identifier(statements) -> int:
    return max(len(s[f]) for s in statements for f in ('subject', 'object'))
