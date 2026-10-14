"""English-versus-Sylang comparison on the project's tiny byte model.

    python language.py audit --source <packet>
    python language.py suite --source <packet> --suite lr --out results/language_lr.jsonl
    python language.py choose-lr --lr-results results/language_lr.jsonl --out results/language_lr_choice.json
    python language.py suite --source <packet> --suite main --lr-json results/language_lr_choice.json \
        --out results/language.jsonl
    python language.py suite --source <packet> --suite single --updates 4000 --out results/language_k1.jsonl
    python language.py suite --source <packet> --suite position --arms english,words,codes,codes_jitter \
        --updates 12000 --out results/language_alibi.jsonl
    python language.py suite --source <packet> --suite study --k 12345678 --arms english_natural,codes,compact --position alibi --curriculum mixed --updates 10000 --out results/language_study.jsonl
    python language.py suite --source <packet> --suite selection-init --copy-init <copy96.pt> \
        --surrogate <surrogate.pt> --out results/selection_init.jsonl
    python language.py probe --source <packet> --out results/language_probe.jsonl <ckpt-dir>/*.pt
    python language.py benchmark --source <packet> --out results/language_benchmark.json

**Task.** Each item holds K predicate records (K uniform in 1..4 by default). The
answer is the JSON list of ``[subject, object]`` for every negative record, in
record order, or ``[]``. It is the packet's binding target text, with identical
bytes in every arm.

**Arms.** Renderers are imported at run time from ``<packet>/fixture-authoring-source``:

- ``english``: controlled English (``data.SCREEN.encode_english``).
- ``words``: Sylang batch-v1 words with selective escaping; the same tab layout as codes.
- ``codes``: Sylang batch-v1 codes with selective escaping.
- ``codes_eqlen``: codes with each record line padded at its end to that record's English line width. It has the same symbols as codes at about English length; its prompts are 35 bytes longer because of the batch header.
- ``codes_jitter``: codes with polarity-independent filler before the subject (0-3 bytes) and before the object (0-28 bytes), so identifier offsets vary about as much as in English. It separates fixed field positions from the symbols themselves.
- ``compact``: codes without the batch header line. The header is implied by convention; the record lines are byte-identical to codes.
- ``english_natural``: controlled English with no filler at all. Here negation adds bytes, so length carries polarity information. This arm measures what the filler changes.

Contrasts:

- **english vs codes**: the headline comparison.
- **codes vs codes_eqlen**: same symbols, different length, so it isolates compression.
- **english vs codes_eqlen**: format and symbols at about equal record length.
- **words vs codes**: readable versus coded symbols in the same layout.
- **codes vs codes_jitter**: same symbols and similar length, fixed versus variable identifier positions.

**Fairness controls.**

- **Identical inputs except the rendering.** Every arm sees the same item stream (records, identifiers, order) for a given seed, the same instruction, model, initialisation, optimiser, batch and number of updates. The learning rate is the same for every arm (``--lr``), or is chosen per arm on the dev split (``--suite lr`` then ``choose-lr``) when that sweep separates the candidates.
- **Slot-placed negation filler.** Where an arm spells negation with extra bytes (English "not", words "negative"), the positive form carries filler (``~``) *in the negation slot*, e.g. ``contain~~~~~~~~~ past`` against ``contain negative past``. Positive and negative records therefore have identical length and layout and differ only inside the slot, as codes records differ only in the case of one letter. Record and prompt lengths are independent of polarity, and features are sampled independently of polarity. ``audit`` reports held-out accuracy for every length cue.
- **Held-out identifiers** (a test split disjoint from training, halved into dev and test), a **held-out (relation, tense) combination**, and counterfactual **swap** (subject/object in every record) and **flip** (one record's polarity) panels.
- **Cue-conflict panel.** The same items are rendered with the filler inverted: the shorter polarity form is left natural, and the longer form gets the filler in its slot. A model that reads "not"/"negative" keeps its accuracy; a model that relies on the filler loses it. Arms without filler render this panel exactly as iid.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from followup import Runtime  # noqa: E402

ARMS = ('english', 'english_natural', 'words', 'codes', 'codes_eqlen', 'codes_jitter', 'compact')
INSTRUCTION = 'List [subject,object] of each negative record in order; [] if none.\n'
FILLER = '~'
BASE_ARM = {'codes_eqlen': 'codes', 'codes_jitter': 'codes', 'compact': 'codes',
            'english_natural': 'english'}  # arm -> codec rendering
FILLER_ARMS = ('english', 'words')  # arms whose positive records carry slot filler
JITTER = (3, 28)  # codes_jitter: up to 3 filler bytes before the subject, up to 28 before the object's tab
RELATIONS = ('see', 'help', 'contain')
TENSES = ('past', 'present', 'future')
ASPECTS = ('simple', 'progressive', 'completed')
EVIDENCES = ('unspecified', 'direct', 'reported', 'inferred')
HELD_OUT_COMBO = ('contain', 'future')  # (relation, tense) never in training records
# Feature combinations never seen in training, by design version. 'v1' is the original
# single (relation, tense) hold-out; 'v2' adds a (relation, aspect) hold-out.
HELD_OUT = {
    'v1': ((('relation', 'tense'), ('contain', 'future')),),
    'v2': ((('relation', 'tense'), ('contain', 'future')), (('relation', 'aspect'), ('help', 'completed'))),
}
CURRICULA = ('mixed', 'staged', 'k1mix')
STAGE_CAP = 3000  # staged: K=1 only until the dev K=1 panel reaches 97%, or this many updates


def is_held_out(statement: dict, design: str = 'v1') -> bool:
    return any(tuple(statement[f] for f in fields) == values for fields, values in HELD_OUT[design])


def _statement(subject, obj, relation, polarity, tense, aspect, evidence):
    return {'kind': 'pred', 'subject': subject, 'relation': relation, 'object': obj, 'polarity': polarity,
            'tense': tense, 'aspect': aspect, 'evidence': evidence}


def _ast(statement):
    return {'version': 'core-v0.1', 'statement': statement}


def _with_polarity(statement, polarity):
    return dict(statement, polarity=polarity)


def slot_fill(positive: str, negative: str) -> tuple[str, str]:
    """Pad the shorter of the two polarity forms inside the slot where they differ.

    The common prefix and suffix are kept; the differing middle of the shorter
    form is extended with FILLER to the longer middle's length.
    """
    prefix = 0
    while prefix < min(len(positive), len(negative)) and positive[prefix] == negative[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(positive), len(negative)) - prefix
           and positive[-1 - suffix] == negative[-1 - suffix]):
        suffix += 1
    pos_mid = positive[prefix:len(positive) - suffix]
    neg_mid = negative[prefix:len(negative) - suffix]
    width = max(len(pos_mid), len(neg_mid))
    pos_mid = pos_mid + FILLER * (width - len(pos_mid))
    neg_mid = neg_mid + FILLER * (width - len(neg_mid))
    return (positive[:prefix] + pos_mid + positive[len(positive) - suffix:],
            negative[:prefix] + neg_mid + negative[len(negative) - suffix:])


def _insert_slot_filler(longer: str, shorter: str, gap: int) -> str:
    """Insert ``gap`` FILLER bytes into ``longer`` at the end of the slot where it
    differs from ``shorter`` (the same place slot_fill pads the shorter form)."""
    prefix = 0
    while prefix < min(len(longer), len(shorter)) and longer[prefix] == shorter[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(longer), len(shorter)) - prefix and longer[-1 - suffix] == shorter[-1 - suffix]:
        suffix += 1
    cut = len(longer) - suffix
    return longer[:cut] + FILLER * gap + longer[cut:]


def _jitter(line: str, statement: dict, number: int) -> str:
    """Polarity-independent filler that moves the identifiers about as much as English does.

    The amounts are a deterministic function of the identifiers and the record number,
    never of polarity, so they carry no class information."""
    digest = hashlib.sha256(f"{statement['subject']}|{statement['object']}|{number}".encode()).digest()
    before_subject = digest[0] % (JITTER[0] + 1)
    before_object = digest[1] % (JITTER[1] + 1)
    cut = line.rindex('\t')
    return FILLER * before_subject + line[:cut] + FILLER * before_object + line[cut:]


class Renderer:
    """Arm renderings with slot-placed negation filler, using the packet's codecs."""

    def __init__(self, rt: Runtime):
        self.screen = rt.data.SCREEN
        self.target_text = rt.data.target_text

    def payload(self, arm: str, statements: list[dict]) -> str:
        """The codec's natural rendering (no filler)."""
        asts = [_ast(s) for s in statements]
        if arm == 'english':
            return self.screen.encode_english(asts)
        return self.screen.encode_batch(asts, arm, False, 'selective')

    def lines(self, arm: str, statements: list[dict]) -> tuple[list[str], list[str]]:
        """(header lines, one line per record), each without its newline."""
        parts = self.payload(arm, statements).split('\n')
        assert parts[-1] == '', 'payload must end with a newline'
        parts = parts[:-1]
        header, records = parts[: len(parts) - len(statements)], parts[len(parts) - len(statements):]
        assert len(header) == (0 if arm == 'english' else 1), (arm, header)
        return header, records

    def record_line(self, arm: str, statement: dict, number: int, mode: str = 'slot') -> str:
        """One record line as record ``number`` of its item.

        mode 'slot':     the shorter polarity form is padded with FILLER in the negation slot;
        mode 'natural':  the codec's line, no filler;
        mode 'conflict': filler inverted: the form that 'slot' would pad is left natural and
                         the other form receives the same amount of filler in its slot.
        """
        def rendered(polarity):
            # Placeholder records before it reproduce the item's numbering exactly.
            before = [_statement(f'q{i}x', f'r{i}y', 'see', 'positive', 'present', 'simple', 'unspecified')
                      for i in range(number - 1)]
            _, records = self.lines(arm, before + [_with_polarity(statement, polarity)])
            return records[-1]

        natural = {p: rendered(p) for p in ('positive', 'negative')}
        if mode == 'natural':
            return natural[statement['polarity']]
        filled = dict(zip(('positive', 'negative'), slot_fill(natural['positive'], natural['negative'])))
        assert len(filled['positive']) == len(filled['negative'])
        if mode == 'slot':
            return filled[statement['polarity']]
        assert mode == 'conflict'
        own, other = statement['polarity'], 'negative' if statement['polarity'] == 'positive' else 'positive'
        gap = len(filled[other]) - len(natural[other])  # filler the other form carries in 'slot' mode
        if gap > 0:  # this is the longer form: it receives the filler instead
            return _insert_slot_filler(natural[own], natural[other], gap)
        return natural[own]  # this is the shorter form (or both are equal): no filler

    def prompt(self, arm: str, statements: list[dict], conflict: bool = False) -> str:
        base = BASE_ARM.get(arm, arm)
        mode = 'natural' if arm == 'english_natural' else 'conflict' if conflict else 'slot'
        header, _ = self.lines(base, statements)
        out = []
        for number, statement in enumerate(statements, 1):
            line = self.record_line(base, statement, number, mode)
            if arm == 'codes_eqlen':
                width = len(self.record_line('english', statement, number, 'slot'))
                line = line + FILLER * max(0, width - len(line))
            elif arm == 'codes_jitter':
                line = _jitter(line, statement, number)
            out.append(line)
        if arm == 'compact':  # the batch header is implied by convention, not sent
            header = []
        text = ''.join(h + '\n' for h in header) + ''.join(line + '\n' for line in out) + INSTRUCTION
        assert text.isascii()
        return text

    def natural_prompt(self, arm: str, statements: list[dict]) -> str:
        base = BASE_ARM.get(arm, arm)
        payload = self.payload(base, statements)
        if arm == 'compact':
            payload = payload.split('\n', 1)[1]
        return payload + INSTRUCTION

    def target(self, statements: list[dict]) -> str:
        return self.target_text([[s['subject'], s['object']] for s in statements if s['polarity'] == 'negative'])


RESUME_KEYS = ('arm', 'init', 'init_sha256', 'k', 'updates', 'batch', 'lr', 'seed')
POSITIONS = ('learned', 'alibi')  # learned: the packet's absolute position table (default)


def _overrides(config: dict) -> dict:
    """The identifying settings of a run. ``position`` appears only when it is not the
    default, so results written before the option existed keep their keys."""
    out = {k: config[k] for k in RESUME_KEYS}
    if config.get('position', 'learned') != 'learned':
        out['position'] = config['position']
    if config.get('design', 'v1') != 'v1':
        out['design'] = config['design']
    if config.get('curriculum', 'mixed') != 'mixed':
        out['curriculum'] = config['curriculum']
    return out


def use_alibi(model, heads: int, pad: int):
    """Replace the learned absolute positions with ALiBi distance biases: one change.

    The token embedding, blocks, final norm and output layer are the packet's own
    modules; the position table is left in place but unused. Attention gets a per-head
    additive bias of -slope * (query - key) on top of the causal mask."""
    import torch

    slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / heads) for h in range(heads)])

    def forward(ids):
        batch, size = ids.shape
        index = torch.arange(size, device=ids.device)
        distance = (index[:, None] - index[None, :]).clamp(min=0).float()
        bias = -slopes.to(ids.device)[:, None, None] * distance
        future = torch.triu(torch.ones(size, size, dtype=torch.bool, device=ids.device), diagonal=1)
        bias = bias.masked_fill(future, float('-inf'))
        mask = bias.repeat(batch, 1, 1)  # (batch * heads, L, L), index = n * heads + h
        padding = ids == pad
        hidden = model.tokens(ids)
        # The fused inference fast path mishandles per-head float masks (NaN with padding);
        # use the standard path for this model only and restore the global flag.
        fast = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            for layer in model.blocks:
                hidden = layer(hidden, src_mask=mask, src_key_padding_mask=padding)
        finally:
            torch.backends.mha.set_fastpath_enabled(fast)
        return model.output(model.norm(hidden))

    model.forward = forward
    return model


def _file_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identifiers(split: str) -> list[str]:
    """train: the FreshIds train split; dev/test: disjoint halves of the FreshIds test split."""
    import tasks
    if split == 'train':
        return tasks.FreshIds('train').ids
    half = {'dev': 0, 'test': 1}[split]
    return [i for i in tasks.FreshIds('test').ids
            if hashlib.sha256(('dev/test:' + i).encode()).digest()[0] % 2 == half]


class ItemStream:
    """Items with independently sampled records; identical across arms for a seed."""

    def __init__(self, split: str, seed: int, k_values=(1, 2, 3, 4), combos: str = 'train', design: str = 'v1'):
        self.ids = _identifiers(split)
        self.id_set = set(self.ids)
        self.rng = random.Random(seed)
        self.k_values = tuple(k_values)
        self.combos = combos
        self.design = design

    def record(self, used: set, held_out: bool | None, polarity: str | None = None) -> dict:
        """One record whose identifiers are unused in the item. ``held_out`` True forces a
        held-out feature combination, False forbids one, None allows either."""
        while True:
            features = (self.rng.choice(RELATIONS), self.rng.choice(TENSES), self.rng.choice(ASPECTS),
                        self.rng.choice(EVIDENCES))
            statement = _statement('', '', features[0], polarity or self.rng.choice(('positive', 'negative')),
                                   *features[1:])
            if held_out is not None and is_held_out(statement, self.design) != held_out:
                continue
            subject, obj = self.rng.sample(self.ids, 2)
            if subject in used or obj in used:
                continue
            used.update((subject, obj))
            return dict(statement, subject=subject, object=obj)

    def item(self, k: int | None = None) -> list[dict]:
        k = k or self.rng.choice(self.k_values)
        used: set = set()
        if self.combos == 'train':
            return [self.record(used, False) for _ in range(k)]
        forced = self.rng.randrange(k)  # at least one record uses a held-out combination
        return [self.record(used, True if i == forced else None) for i in range(k)]

    def similar_item(self, k: int) -> list[dict]:
        """K >= 2 records: every negative record has a positive partner whose subject or
        object differs from the negative's in exactly one character."""
        assert k >= 2
        used: set = set()
        records = []
        for _ in range(max(1, k // 2)):
            negative = self.record(used, False, 'negative')
            partner = self.record(used, False, 'positive')
            role = self.rng.choice(('subject', 'object'))
            partner[role] = self._one_char_variant(negative[role], used)
            records += [negative, partner]
        while len(records) < k:
            records.append(self.record(used, False, 'positive'))
        self.rng.shuffle(records)
        return records

    def _one_char_variant(self, identifier: str, used: set) -> str:
        letters, digits = 'abcdefghijklmnopqrstuvwxyz', '0123456789'
        while True:
            i = self.rng.randrange(len(identifier))
            pool = digits if identifier[i].isdigit() else letters
            new = identifier[:i] + self.rng.choice(pool.replace(identifier[i], '')) + identifier[i + 1:]
            if new not in used:
                used.add(new)
                return new


def swap_roles(statements):
    return [dict(s, subject=s['object'], object=s['subject']) for s in statements]


def flip_one(statements, rng: random.Random):
    i = rng.randrange(len(statements))
    flipped = [dict(s) for s in statements]
    flipped[i]['polarity'] = 'positive' if flipped[i]['polarity'] == 'negative' else 'negative'
    return flipped


def eval_panels(k_values, per_k: int, seed: int = 7) -> dict[str, list[list[dict]]]:
    """Fixed panels with ``per_k`` items for every K (held-out combo: half as many)."""
    test = ItemStream('test', seed)
    combo = ItemStream('test', seed + 1, combos='held_out')
    dev = ItemStream('dev', seed + 3)
    panels = {
        'iid': [test.item(k) for k in k_values for _ in range(per_k)],
        'held_out_combo': [combo.item(k) for k in k_values for _ in range(max(8, per_k // 2))],
        'dev': [dev.item(k) for k in k_values for _ in range(per_k)],
    }
    rng = random.Random(seed + 2)
    panels['swap'] = [swap_roles(it) for it in panels['iid'] if any(s['polarity'] == 'negative' for s in it)]
    panels['flip'] = [flip_one(it, rng) for it in panels['iid']]
    panels['cue_conflict'] = panels['iid']  # rendered with the filler cue inverted (see run)
    return panels


def reorder(statements, rng: random.Random):
    """The same records in a different order (K >= 2); the answer order follows."""
    while True:
        permuted = rng.sample(statements, len(statements))
        if permuted != statements:
            return permuted


def study_panels(k_values, per_k: int, design: str = 'v2', seed: int = 7, filler: bool = False) -> dict:
    """Design-v2 panels. Identifier and combination hold-outs are separated:
    iid = unseen identifiers, trained combinations; combo = training-split identifiers,
    a held-out combination; both = unseen identifiers and a held-out combination.
    reorder, similar, swap and flip are counterfactual or adversarial variants."""
    test = ItemStream('test', seed, design=design)
    combo = ItemStream('train', seed + 1, combos='held_out', design=design)
    both = ItemStream('test', seed + 4, combos='held_out', design=design)
    dev = ItemStream('dev', seed + 3, design=design)
    similar = ItemStream('test', seed + 5, design=design)
    half = max(8, per_k // 2)
    panels = {
        'iid': [test.item(k) for k in k_values for _ in range(per_k)],
        'combo': [combo.item(k) for k in k_values for _ in range(half)],
        'both': [both.item(k) for k in k_values for _ in range(half)],
        'dev': [dev.item(k) for k in k_values for _ in range(per_k)],
        'similar': [similar.similar_item(k) for k in k_values if k >= 2 for _ in range(per_k)],
    }
    rng = random.Random(seed + 2)
    panels['reorder'] = [reorder(it, rng) for it in panels['iid'] if len(it) >= 2]
    panels['swap'] = [swap_roles(it) for it in panels['iid'] if any(s['polarity'] == 'negative' for s in it)]
    panels['flip'] = [flip_one(it, rng) for it in panels['iid']]
    if filler:
        panels['cue_conflict'] = panels['iid']
    return panels


# ---------------------------------------------------------------------------
# Audit: can anything other than reading the records predict the answer?


def _heldout_lookup(pairs: list[tuple], groups: list | None = None) -> float:
    """Fit feature -> majority label on the first half and score the second half.
    Unseen features back off to the majority within the same group (e.g. K)."""
    half = len(pairs) // 2
    groups = groups or [0] * len(pairs)
    table, by_group = defaultdict(Counter), defaultdict(Counter)
    for (feature, label), group in zip(pairs[:half], groups[:half]):
        table[(group, feature)][label] += 1
        by_group[group][label] += 1
    hits = 0
    for (feature, label), group in zip(pairs[half:], groups[half:]):
        counts = table[(group, feature)] or by_group[group]
        hits += counts.most_common(1)[0][0] == label
    return hits / (len(pairs) - half)


def audit(args) -> dict:
    rt = Runtime(args.source)
    renderer = Renderer(rt)
    stream = ItemStream('train', 20261013)
    items = [stream.item() for _ in range(args.items)]
    n_negative = [sum(s['polarity'] == 'negative' for s in it) for it in items]
    ks = [len(it) for it in items]
    report = {}
    for arm in ARMS:
        prompts = [renderer.prompt(arm, it) for it in items]
        naturals = [renderer.natural_prompt(arm, it) for it in items]
        base = BASE_ARM.get(arm, arm)
        line_cues = defaultdict(list)
        for it, prompt in zip(items, prompts):
            header, _ = renderer.lines(base, it)
            n_header = 0 if arm == 'compact' else len(header)
            lines = prompt.split('\n')[n_header:n_header + len(it)]
            for s, line in zip(it, lines):
                line_cues['line_length'].append((len(line), s['polarity']))
                line_cues['filler_count'].append((line.count(FILLER), s['polarity']))
                line_cues['features'].append(((s['relation'], s['tense'], s['aspect'], s['evidence']), s['polarity']))
        report[arm] = dict(
            mean_prompt_bytes=round(statistics.fmean(len(p) for p in prompts), 1),
            mean_natural_bytes=round(statistics.fmean(len(p) for p in naturals), 1),
            mean_natural_bytes_by_k={k: round(statistics.fmean(len(p) for p, kk in zip(naturals, ks) if kk == k), 1)
                                     for k in sorted(set(ks))},
            n_negative_from_k_only=round(_heldout_lookup([(0, n) for n in n_negative], ks), 4),
            n_negative_from_k_and_length=round(_heldout_lookup([(len(p), n) for p, n in zip(prompts, n_negative)],
                                                               ks), 4),
            record_polarity_from={name: round(_heldout_lookup(pairs), 4) for name, pairs in line_cues.items()},
        )
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    return report


# ---------------------------------------------------------------------------
# Training and scoring


def run(config: dict) -> dict:
    import torch

    torch.set_num_threads(config.get('threads', 1))
    rt = Runtime(config['source'])
    renderer = Renderer(rt)
    seed = 20261013 + config['seed']  # for loaded inits the seed varies only the data order
    model = rt.build(seed)
    if config['init'] in ('copy96', 'surrogate'):
        path = config['copy_init'] if config['init'] == 'copy96' else config['surrogate']
        if _file_sha(path) != config['init_sha256']:
            raise ValueError(f'initial checkpoint {path} does not match {config["init_sha256"]}')
        rt.load(model, path)
    elif config['init'] != 'fresh':
        raise ValueError(config['init'])
    if config.get('position', 'learned') == 'alibi':
        use_alibi(model, rt.recipe['heads'], rt.tiny.PAD)
    elif config.get('position', 'learned') != 'learned':
        raise ValueError(config['position'])
    optimizer = rt.optimizer(model, config['lr'])
    design = config.get('design', 'v1')
    curriculum = config.get('curriculum', 'mixed')
    if curriculum not in CURRICULA:
        raise ValueError(curriculum)
    stream = ItemStream('train', seed, k_values=tuple(config['k']), design=design)
    per_k = max(32, 96 // len(config['k']))
    if design == 'v1':
        raw_panels = eval_panels(config['k'], per_k)
        stop_panels = ('iid', 'dev', 'held_out_combo', 'swap', 'flip')
    else:
        raw_panels = study_panels(config['k'], per_k, design, filler=config['arm'] in FILLER_ARMS)
        stop_panels = ('iid', 'dev', 'swap', 'flip', 'reorder')
    panels = {name: [dict(prompt=renderer.prompt(config['arm'], it, conflict=(name == 'cue_conflict')),
                          target=renderer.target(it), statements=it)
                     for it in items]
              for name, items in raw_panels.items()}
    checkpoints = sorted(set(config['eval_at']) | {config['updates']})
    history, seconds, bytes_seen, items_seen = [], 0.0, 0, 0
    stage_switch = None  # staged curriculum: the update at which K=1-only training ended
    for step in range(1, config['updates'] + 1):
        if curriculum == 'staged' and stage_switch is None:
            items = [stream.item(1) for _ in range(config['batch'])]
        elif curriculum == 'k1mix':
            items = [stream.item(1 if stream.rng.random() < 0.25 else None) for _ in range(config['batch'])]
        else:
            items = [stream.item() for _ in range(config['batch'])]
        rows = [dict(id=str(i), prompt=renderer.prompt(config['arm'], it), target=renderer.target(it))
                for i, it in enumerate(items)]
        started = time.perf_counter()
        rt.step(model, optimizer, rows)
        seconds += time.perf_counter() - started
        bytes_seen += sum(len(r['prompt']) + len(r['target']) + 3 for r in rows)  # + BOS, SEP, EOS
        items_seen += len(rows)
        if step in checkpoints:
            history.append(dict(step=step, items_seen=items_seen, bytes_seen=bytes_seen,
                                train_seconds=round(seconds, 2), **score(rt, model, panels)))
            h = history[-1]
            print(f"[{config['arm']} s{config['seed']} {config.get('position', 'learned')} {curriculum} "
                  f"{config['init']}] step {step} items {items_seen} train_s {seconds:.0f} "
                  f"iid {h['iid']['exact']} sel {h['iid']['selected']} dev {h['dev']['exact']} "
                  + ' '.join(f"{n}_sel {h[n]['selected']}" for n in ('combo', 'both', 'similar') if n in h),
                  flush=True)
            if curriculum == 'staged' and stage_switch is None and (
                    history[-1]['dev']['by_k'].get(1, 0) >= 0.97 or step >= STAGE_CAP):
                stage_switch = step
                history[-1]['stage_switch'] = step
            if config.get('early_stop') and step >= config['early_stop'] and _at_ceiling(history[-2:], stop_panels):
                break  # two checkpoints in a row at the ceiling: later updates cannot change the reading
    # Agreement between the teacher-forced exact flag and the packet's real greedy decoder.
    rows = panels['iid'][:32]
    flags = [r['exact'] for r in teacher_forced(rt, model, rows)]
    decoded = [rt.generate(model, r['prompt'], len(r['target']) + 2) == r['target'] for r in rows]
    history[-1]['decoder_agreement'] = dict(agree=sum(a == b for a, b in zip(flags, decoded)), n=len(rows))
    saved = None
    if config.get('save_dir'):  # keep the final model (scratch only; never committed)
        o = _overrides(config)
        name = (f"{o['arm']}-{o['init']}-k{''.join(map(str, o['k']))}-u{o['updates']}-b{o['batch']}"
                f"-lr{o['lr']:g}-s{o['seed']}" + (f"-{o['position']}" if 'position' in o else '')
                + (f"-{o['curriculum']}" if 'curriculum' in o else '') + (f"-{o['design']}" if 'design' in o else '')
                + '.pt')
        raw = rt.state_bytes(model)
        Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
        (Path(config['save_dir']) / name).write_bytes(raw)
        saved = dict(file=name, sha256=hashlib.sha256(raw).hexdigest())
    return dict(overrides=_overrides(config), saved=saved,
                torch=torch.__version__, history=history,
                byte_roles=byte_role_accuracy(rt, model, panels['iid']),
                mean_prompt_bytes=statistics.fmean(len(r['prompt']) for r in panels['iid']))


def teacher_forced(rt: Runtime, model, rows: list[dict], batch: int = 32) -> list[dict]:
    """One batched forward per chunk: per row, whether greedy decoding would emit
    exactly target+EOS (argmax equals the target at every answer position, which is
    equivalent for greedy decoding), the target's total log-probability, and the
    per-position argmax hits."""
    import torch
    from real_pipeline import collate_rows

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            inputs, labels = collate_rows(rt.tiny, chunk, rt.recipe['context'])
            logp = model(inputs).log_softmax(-1)
            argmax = logp.argmax(-1)
            for i in range(len(chunk)):
                mask = labels[i] != -100
                target = labels[i][mask]
                hits = (argmax[i][mask] == target)
                out.append(dict(exact=bool(hits.all()),
                                logprob=float(logp[i][mask].gather(1, target[:, None]).sum()),
                                hits=hits.tolist(), target=target.tolist()))
    return out


def _at_ceiling(last_two: list[dict], panels=('iid', 'dev', 'held_out_combo', 'swap', 'flip'),
                level: float = 0.97) -> bool:
    return len(last_two) == 2 and all(h[panel]['exact'] >= level for h in last_two for panel in panels)


def score(rt: Runtime, model, panels: dict) -> dict:
    out = {}
    for name, rows in panels.items():
        exact = [r['exact'] for r in teacher_forced(rt, model, rows)]
        selected = [e for e, r in zip(exact, rows) if r['target'] != '[]']
        empty = [e for e, r in zip(exact, rows) if r['target'] == '[]']
        by_k = defaultdict(list)
        for e, r in zip(exact, rows):
            by_k[len(r['statements'])].append(e)
        out[name] = dict(exact=round(sum(exact) / len(rows), 4), n=len(rows),
                         selected=round(sum(selected) / len(selected), 4) if selected else None,
                         n_selected=len(selected),
                         empty=round(sum(empty) / len(empty), 4) if empty else None, n_empty=len(empty),
                         always_empty_baseline=round(len(empty) / len(rows), 4),
                         by_k={k: round(sum(v) / len(v), 4) for k, v in sorted(by_k.items())})
    # Swap margin over the whole answer: log p(swapped target | swapped prompt)
    # minus log p(original target | swapped prompt). Positive means roles are followed.
    originals = [r for r in panels['iid'] if r['target'] != '[]']
    swapped = panels['swap']
    right = teacher_forced(rt, model, swapped)
    wrong = teacher_forced(rt, model, [dict(s, target=o['target']) for o, s in zip(originals, swapped)])
    margins = [a['logprob'] - b['logprob'] for a, b in zip(right, wrong)]
    out['swap_margin'] = dict(mean=round(statistics.fmean(margins), 3),
                              positive=round(sum(m > 0 for m in margins) / len(margins), 4), n=len(margins))
    return out


def byte_role_accuracy(rt: Runtime, model, rows: list[dict]) -> dict:
    """Teacher-forced next-byte accuracy by role of the answer byte."""
    hits, totals = Counter(), Counter()
    for result in teacher_forced(rt, model, rows):
        for hit, t in zip(result['hits'], result['target']):
            ch = chr(t) if t < 128 else ''
            role = ('eos' if t == rt.tiny.EOS else 'quote' if ch == '"' else
                    'structure' if ch in '[],' else 'identifier_digit' if ch.isdigit() else 'identifier_letter')
            totals[role] += 1
            hits[role] += hit
    return {role: round(hits[role] / totals[role], 4) for role in sorted(totals)}


def probe(args) -> None:
    """Answer-byte accuracy by role for saved checkpoints, plus greedy samples.

    The arm and K are read from the file name written by ``--save-dir``."""
    import torch

    torch.set_num_threads(1)
    rt = Runtime(args.source)
    renderer = Renderer(rt)
    out = []
    for path in sorted(args.checkpoints):
        name = Path(path).name
        arm = name.split('-fresh-')[0].split('-copy96-')[0].split('-surrogate-')[0]
        k_values = [int(c) for c in name.split('-k')[1].split('-')[0]]
        model = rt.load(rt.build(0), path).eval()
        if '-alibi' in name:
            use_alibi(model, rt.recipe['heads'], rt.tiny.PAD)
        items = eval_panels(k_values, max(32, 96 // len(k_values)))['iid']
        rows = [dict(prompt=renderer.prompt(arm, it), target=renderer.target(it)) for it in items]
        selected = [r for r in rows if r['target'] != '[]']
        samples = [dict(target=r['target'], greedy=rt.generate(model, r['prompt'], len(r['target']) + 2))
                   for r in selected[:args.samples]]
        record = dict(file=name, sha256=_file_sha(path), arm=arm, k=k_values,
                      byte_roles_selected=byte_role_accuracy(rt, model, selected), samples=samples)
        out.append(record)
        print(json.dumps(record), flush=True)
    if args.out:
        Path(args.out).write_text(''.join(json.dumps(r) + '\n' for r in out))


# ---------------------------------------------------------------------------
# Serial inference-cost benchmark (weights do not affect timing)


def _flops(length: int, layers: int, width: int, vocab: int) -> float:
    """Approximate forward FLOPs for one full sequence of ``length`` positions."""
    return length * (layers * (24 * width * width) + 2 * width * vocab) + layers * 4 * width * length * length


def benchmark(args) -> dict:
    import torch

    torch.set_num_threads(1)
    rt = Runtime(args.source)
    renderer = Renderer(rt)
    model = rt.build(0).eval()
    width, layers, vocab = rt.recipe['width'], rt.recipe['layers'], rt.tiny.VOCAB
    stream = ItemStream('test', 11)
    items = [stream.item(k) for k in (1, 2, 3, 4) for _ in range(args.items_per_k)]
    report = {}
    for arm in ARMS:
        rows = [(renderer.natural_prompt(arm, it), renderer.target(it), len(it)) for it in items]
        prefill, decode, flops_prefill, flops_decode = [], [], [], []
        with torch.no_grad():
            for prompt, target, _ in rows:
                ids = [rt.tiny.BOS] + rt.tiny.byte_ids(prompt) + [rt.tiny.SEP]
                answer = rt.tiny.byte_ids(target) + [rt.tiny.EOS]
                for _ in range(2):  # warm-up
                    model(torch.tensor([ids]))
                times = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    model(torch.tensor([ids]))
                    times.append(time.perf_counter() - started)
                prefill.append(statistics.median(times))
                started = time.perf_counter()  # reference decoding: one full forward per generated byte
                for i in range(len(answer)):
                    model(torch.tensor([ids + answer[:i]]))
                decode.append(time.perf_counter() - started)
                flops_prefill.append(_flops(len(ids), layers, width, vocab))
                flops_decode.append(sum(_flops(len(ids) + i, layers, width, vocab) for i in range(len(answer))))
        report[arm] = dict(
            mean_prompt_bytes=round(statistics.fmean(len(p) for p, _, _ in rows), 1),
            mean_prompt_bytes_by_k={k: round(statistics.fmean(len(p) for p, _, kk in rows if kk == k), 1)
                                    for k in (1, 2, 3, 4)},
            mean_answer_bytes=round(statistics.fmean(len(t) for _, t, _ in rows), 1),
            prefill_ms_median=round(1000 * statistics.median(prefill), 3),
            decode_ms_mean=round(1000 * statistics.fmean(decode), 2),
            prefill_gflops_mean=round(statistics.fmean(flops_prefill) / 1e9, 4),
            decode_gflops_mean=round(statistics.fmean(flops_decode) / 1e9, 4),
        )
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    return report


# ---------------------------------------------------------------------------
# Suites


# A fixed evaluation grid, so runs of different lengths share their early checkpoints. A run
# with early_stop ends at a checkpoint from that update on when it and the previous checkpoint
# reach 97% on iid, dev, held_out_combo, swap and flip (cue_conflict is not part of the rule).
GRID = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000]


def suite_configs(name: str, args) -> list[dict]:
    base = dict(source=args.source, copy_init=args.copy_init, surrogate=args.surrogate, init=args.init,
                save_dir=args.save_dir,
                init_sha256=None, k=[1, 2, 3, 4], batch=8)
    arms = [a for a in ARMS if not args.arms or a in args.arms.split(',')]
    shas = {'copy96': _file_sha(args.copy_init) if args.copy_init else None,
            'surrogate': _file_sha(args.surrogate) if args.surrogate else None}
    if args.init != 'fresh' and not shas.get(args.init):
        raise SystemExit(f'--init {args.init} needs --copy-init or --surrogate')
    base['init_sha256'] = shas.get(args.init)
    if name == 'lr':
        return [dict(base, arm=arm, lr=lr, seed=0, updates=args.updates,
                     eval_at=[args.updates // 4, args.updates // 2, args.updates])
                for lr, arm in itertools.product((3e-4, 1e-3), arms)]
    if name == 'main':
        choice = json.loads(Path(args.lr_json).read_text()) if args.lr_json else {}
        # The first K=1-4 runs were evaluated at updates/8, /4, /2 and the end; at 12,000 updates
        # those points are a subset of GRID, which later runs use with early stopping.
        return [dict(base, arm=arm, seed=seed, lr=choice.get(arm, args.lr), updates=args.updates,
                     early_stop=4000, eval_at=[u for u in GRID if u < args.updates])
                for seed, arm in itertools.product(range(args.first_seed, args.seeds), arms)]  # arms interleaved
    if name == 'single':  # one record per item: the selection task itself, in every arm
        return [dict(base, arm=arm, seed=seed, k=[1], lr=args.lr, updates=args.updates, early_stop=4000,
                     eval_at=[u for u in GRID if u < args.updates])
                for seed, arm in itertools.product(range(args.first_seed, args.seeds), arms)]
    if name == 'position':  # the same task with ALiBi in place of learned absolute positions
        k = [int(c) for c in args.k]
        return [dict(base, arm=arm, seed=seed, k=k, position='alibi', lr=args.lr, updates=args.updates,
                     early_stop=4000, eval_at=[u for u in GRID if u < args.updates])
                for seed, arm in itertools.product(range(args.first_seed, args.seeds), arms)]
    if name == 'study':  # design v2: K up to 8, two held-out combinations, the full panel set
        k = [int(c) for c in args.k]
        return [dict(base, arm=arm, seed=seed, k=k, design='v2', position=args.position,
                     curriculum=args.curriculum, lr=args.lr, updates=args.updates, early_stop=4000,
                     eval_at=[u for u in GRID if u < args.updates])
                for seed, arm in itertools.product(range(args.first_seed, args.seeds), arms)]
    if name == 'selection-init':
        inits = ['fresh'] + (['copy96'] if args.copy_init else []) + (['surrogate'] if args.surrogate else [])
        selection_arms = args.arms.split(',') if args.arms else ['english']
        return [dict(base, arm=arm, init=init, init_sha256=shas.get(init), k=[1], batch=4, lr=3e-4, seed=seed,
                     updates=args.updates,
                     eval_at=[args.updates // 16, args.updates // 8, args.updates // 4, args.updates // 2,
                              args.updates])
                for seed, arm, init in itertools.product(range(args.seeds), selection_arms, inits)]
    raise ValueError(name)


def suite(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.dumps(json.loads(l)['overrides'], sort_keys=True) for l in out.read_text().splitlines() if l}
    todo = [c for c in suite_configs(args.suite, args)
            if json.dumps(_overrides(c), sort_keys=True) not in done]
    print(f'{len(todo)} runs to do, {len(done)} done', flush=True)
    errors = out.with_suffix('.errors.jsonl')
    with ProcessPoolExecutor(max_workers=args.workers) as pool, out.open('a') as handle:
        futures = {pool.submit(run, c): c for c in todo}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:  # keep the other runs; record the failure separately
                with errors.open('a') as err:
                    err.write(json.dumps({'overrides': _overrides(futures[future]),
                                          'error': repr(error)}) + '\n')
                print('FAILED', _overrides(futures[future]), repr(error), flush=True)
                continue
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            last = result['history'][-1]
            panels = [k for k, v in last.items() if isinstance(v, dict) and 'exact' in v]
            print(json.dumps(result['overrides']), 'step', last['step'], 'selected', last['iid']['selected'],
                  ' '.join(f"{name} {last[name]['exact']}" for name in panels), flush=True)


def choose_lr(args) -> None:
    """Pick each arm's LR by final dev-panel exact accuracy (ties: the smaller LR)."""
    best = {}
    for line in Path(args.lr_results).read_text().splitlines():
        r = json.loads(line)
        o, dev = r['overrides'], r['history'][-1]['dev']['exact']
        if o['arm'] not in best or (dev, -o['lr']) > (best[o['arm']][0], -best[o['arm']][1]):
            best[o['arm']] = (dev, o['lr'])
    Path(args.out).write_text(json.dumps({arm: lr for arm, (_, lr) in best.items()}, indent=2) + '\n')
    print(json.dumps({arm: dict(lr=lr, dev=dev) for arm, (dev, lr) in best.items()}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--source', required=True)
    a.add_argument('--items', type=int, default=40000)
    a.add_argument('--out')
    s = sub.add_parser('suite')
    s.add_argument('--source', required=True)
    s.add_argument('--suite', choices=('lr', 'main', 'single', 'position', 'study', 'selection-init'), required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--workers', type=int, default=4)
    s.add_argument('--seeds', type=int, default=3)
    s.add_argument('--first-seed', type=int, default=0, help='run seeds first-seed .. seeds-1')
    s.add_argument('--updates', type=int, default=4000)
    s.add_argument('--lr', type=float, default=1e-3)
    s.add_argument('--lr-json')
    s.add_argument('--copy-init')
    s.add_argument('--surrogate')
    s.add_argument('--arms', help='comma-separated subset of ' + ','.join(ARMS))
    s.add_argument('--init', default='fresh', choices=('fresh', 'copy96', 'surrogate'),
                   help='initial weights for the lr/main/single/position suites')
    s.add_argument('--k', default='1', help="record counts for --suite position/study, e.g. 1, 1234 or 12345678")
    s.add_argument('--position', default='alibi', choices=POSITIONS, help='--suite study: position scheme')
    s.add_argument('--curriculum', default='mixed', choices=CURRICULA, help='--suite study: training curriculum')
    s.add_argument('--save-dir', help='write each final model here (outside the repository)')
    c = sub.add_parser('choose-lr')
    c.add_argument('--lr-results', required=True)
    c.add_argument('--out', required=True)
    pr = sub.add_parser('probe')
    pr.add_argument('--source', required=True)
    pr.add_argument('--out')
    pr.add_argument('--samples', type=int, default=4)
    pr.add_argument('checkpoints', nargs='+')
    b = sub.add_parser('benchmark')
    b.add_argument('--source', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--items-per-k', type=int, default=16)
    b.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    {'audit': audit, 'suite': suite, 'choose-lr': choose_lr, 'probe': probe, 'benchmark': benchmark}[args.command](args)


if __name__ == '__main__':
    main()
