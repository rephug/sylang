"""Follow-up experiments after the native 96-identifier COPY result (2026-10-13).

    python followup.py surrogate --source <packet> --out <dir>
    python followup.py copy-suite --source <packet> --surrogate <dir>/surrogate.pt --out results/copy_matrix.jsonl
    python followup.py select-suite --source <packet> --surrogate <dir>/surrogate.pt \
        --copy-init <dir>/copy96.pt --out results/selection.jsonl

The model, decoder, data renderer and COPY curriculum are imported or read at
run time from the unpacked packet (``<packet>/fixture-authoring-source`` and
``<packet>/copy/source``); none of that source or data is stored here. Every
run uses the packet's tiny architecture and the trainers' optimiser, loss and
clipping. Nothing reads or writes the project's own checkpoints.

**Surrogate warm start.** The project's warm start (``efb2…``) is not available
locally. ``surrogate`` trains a stand-in with the packet's own memorisation
fixture (16 English cases, 200 updates, batch 2, LR 3e-4, the generic tiny
recipe), from a fresh initialisation. It approximates "a model that has
memorised long English prompts". It is not the project's checkpoint, whose
history was longer, and its results must be confirmed natively.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from real_pipeline import collate_rows, make_optimizer, train_step  # noqa: E402

ALPHABET = 'abcdef'
SHORT_INSTRUCTION = '[] if positive, else [["subject","object"]].\n'
RELATIONS = ('see', 'help', 'contain')
TENSES = ('past', 'present', 'future')
ASPECTS = ('simple', 'progressive', 'completed')
EVIDENCES = ('unspecified', 'direct', 'reported', 'inferred')


# ---------------------------------------------------------------------------
# Runtime imports from the packet


class Runtime:
    """The packet's tiny/decode/data modules plus the binding trainer's recipe."""

    def __init__(self, source: str):
        root = Path(source) / 'fixture-authoring-source'
        if str(root) not in sys.path:
            sys.path.append(str(root))  # appended; the packet's data module inserts its own root
        from training import binding_train, data, decode, tiny
        self.tiny, self.decode, self.data = tiny, decode, data
        self.recipe = dict(binding_train.RECIPE)
        self.source = Path(source)

    def build(self, seed: int):
        import torch
        torch.manual_seed(seed)
        random.seed(seed)
        return self.tiny.build_model(self.recipe).cpu()

    def batch(self, rows):
        return collate_rows(self.tiny, rows, self.recipe['context'])

    def step(self, model, optimizer, rows) -> float:
        """One update with the packet workers' objective, optimiser and clipping."""
        return train_step(model, optimizer, *self.batch(rows))

    def optimizer(self, model, lr: float = 3e-4):
        return make_optimizer(model, lr)

    def generate(self, model, prompt: str, max_new: int):
        out = self.decode.complete(model, prompt, backend='reference', context=self.recipe['context'],
                                   max_new_tokens=max_new, max_seconds=60)
        return out['answer'] if out['status'] == 'complete' else None

    @staticmethod
    def state_bytes(model) -> bytes:
        import torch
        buffer = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
        return buffer.getvalue()

    def load(self, model, path: str):
        import torch
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        return model


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Surrogate memorisation warm start


def surrogate(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    rt = Runtime(args.source)
    recipe = json.loads((rt.source / 'fixture-authoring-source/training/tiny-recipe.json').read_text())
    cases = _jsonl(rt.source / 'fixture-authoring-source/training/data-memorization-v1/cases.jsonl')
    rows = [dict(id=c['id'], prompt=rt.data.render(c['asts'], 'english'),
                 target=rt.data.target_text(rt.data.interpret(c['asts'], c['query']))) for c in cases]
    model = rt.build(recipe['seed'])
    optimizer = rt.optimizer(model, recipe['learning_rate'])
    sampler = random.Random(recipe['seed'])
    losses = [rt.step(model, optimizer, [sampler.choice(rows) for _ in range(recipe['batch_size'])])
              for _ in range(recipe['steps'])]
    exact = sum(rt.generate(model, r['prompt'], 200) == r['target'] for r in rows)
    raw = rt.state_bytes(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'surrogate.pt').write_bytes(raw)
    meta = dict(kind='surrogate-memoriser', sha256=_sha(raw), bytes=len(raw), rows=len(rows),
                prompt_bytes=sorted({len(r['prompt'].encode()) for r in rows}),
                recipe={k: recipe[k] for k in ('seed', 'steps', 'batch_size', 'learning_rate')},
                first_last_loss=[losses[0], losses[-1]], memorised_exact=exact, torch=torch.__version__)
    (out / 'surrogate.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta))
    return meta


# ---------------------------------------------------------------------------
# COPY: initialisation x identifier breadth x exposure


def _balanced(pool: list[str], base: list[str], size: int, rng: random.Random, need_pairs: bool,
              attempts: int = 20000) -> list[str]:
    """``base`` plus identifiers from ``pool`` so that every letter occurs exactly
    size/6 times at every position; with ``need_pairs`` all 36 ordered letter
    pairs must also occur for every pair of positions (the native 96-set's
    property)."""
    per = size // len(ALPHABET)
    for _ in range(attempts):
        chosen = list(base)
        counts = [Counter(s[p] for s in chosen) for p in range(3)]
        candidates = [s for s in pool if s not in chosen]
        rng.shuffle(candidates)
        for s in candidates:
            if len(chosen) == size:
                break
            if all(counts[p][s[p]] < per for p in range(3)):
                chosen.append(s)
                for p in range(3):
                    counts[p][s[p]] += 1
        if len(chosen) != size:
            continue
        if need_pairs and any(len({(s[p], s[q]) for s in chosen}) < 36
                              for p in range(3) for q in range(3) if p != q):
            continue
        return sorted(chosen)
    raise RuntimeError(f'no balanced set of size {size}')


def copy_sets(source: Path, seed: int = 20261013) -> dict:
    curriculum = json.loads((source / 'copy/source/CURRICULUM.json').read_text())
    legacy_train, legacy_heldout = list(curriculum['train_identifiers']), list(curriculum['heldout_identifiers'])
    universe = [''.join(p) for p in itertools.product(ALPHABET, repeat=3)]
    rng = random.Random(seed)
    eligible = [s for s in universe if s not in legacy_heldout]
    p96 = _balanced(eligible, legacy_train, 96, rng, need_pairs=True)
    s24 = _balanced(p96, legacy_train, 24, rng, need_pairs=False)
    s48 = _balanced(p96, s24, 48, rng, need_pairs=False)
    fresh108 = sorted(set(universe) - set(p96) - set(legacy_heldout))
    assert len(fresh108) == 108 and set(legacy_train) <= set(s24) <= set(s48) <= set(p96)
    return {'12': sorted(legacy_train), '24': s24, '48': s48, '96': p96,
            'heldout_legacy12': sorted(legacy_heldout), 'heldout_fresh108': fresh108}


def _copy_row(identifier: str, copy: bool) -> dict:
    opcode = 'COPY' if copy else 'NONE'
    return dict(id=f'{identifier}-{opcode}', prompt=f'{opcode}:{identifier}', target=identifier if copy else '')


def _copy_schedule(ids: list[str], updates: int, rng: random.Random) -> list[list[dict]]:
    """Passes over the identifiers; each update holds 2 identifiers x {COPY, NONE}."""
    schedule = []
    while len(schedule) < updates:
        order = list(ids)
        rng.shuffle(order)
        for a, b in zip(order[0::2], order[1::2]):
            rows = [_copy_row(a, True), _copy_row(a, False), _copy_row(b, True), _copy_row(b, False)]
            rng.shuffle(rows)
            schedule.append(rows)
    return schedule[:updates]


def copy_run(config: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    rt = Runtime(config['source'])
    sets = copy_sets(rt.source)
    train_ids = sets[str(config['breadth'])]
    seed = 20261013 + config['seed']
    model = rt.build(seed)
    if config['init'] == 'surrogate':
        rt.load(model, config['surrogate'])
    elif config['init'] != 'fresh':
        raise ValueError(config['init'])
    optimizer = rt.optimizer(model)
    started = time.monotonic()
    losses = [rt.step(model, optimizer, rows)
              for rows in _copy_schedule(train_ids, config['updates'], random.Random(seed + 1))]
    seconds = time.monotonic() - started
    final = {}
    train_set = set(train_ids)
    for name, ids in (('seen', train_ids), ('heldout_legacy12', sets['heldout_legacy12']),
                      ('heldout_fresh108', sets['heldout_fresh108'])):
        copies = [rt.generate(model, f'COPY:{s}', 4) for s in ids]
        nones = [rt.generate(model, f'NONE:{s}', 4) for s in ids]
        wrong = [c for c, s in zip(copies, ids) if c != s]
        final[name] = dict(
            n=len(ids),
            copy_exact=sum(c == s for c, s in zip(copies, ids)),
            none_exact=sum(n == '' for n in nones),
            byte_correct=[sum(bool(c) and len(c) > p and c[p] == s[p] for c, s in zip(copies, ids)) for p in range(3)],
            wrong_that_are_training_ids=sum(c in train_set for c in wrong),
            wrong=len(wrong),
        )
    raw = rt.state_bytes(model)
    if config.get('save'):
        Path(config['save']).parent.mkdir(parents=True, exist_ok=True)
        Path(config['save']).write_bytes(raw)
    return dict(overrides={k: config[k] for k in ('init', 'breadth', 'updates', 'seed')},
                exposures_per_row=config['updates'] * 2 // config['breadth'],
                train_seconds=round(seconds, 2), loss_first_last=[losses[0], losses[-1]],
                checkpoint_sha256=_sha(raw), final=final)


def copy_configs(source: str, surrogate_path: str | None):
    inits = ['fresh'] + (['surrogate'] if surrogate_path else [])
    for init, seed in itertools.product(inits, range(3)):
        for breadth, updates in ((12, 192), (24, 192), (48, 192), (96, 192), (12, 24), (96, 1536)):
            yield dict(source=source, surrogate=surrogate_path, init=init, breadth=breadth,
                       updates=updates, seed=seed)


# ---------------------------------------------------------------------------
# Selection: one English record, varied features and identifiers, matched lengths


def _ids(split: str) -> list[str]:
    import tasks  # this directory: letter-digit-letter identifiers, stable 80/20 split
    return tasks.FreshIds(split).ids


def _pred(subject, obj, relation, polarity, tense, aspect, evidence):
    return {'version': 'core-v0.1', 'statement': {
        'kind': 'pred', 'subject': subject, 'relation': relation, 'object': obj, 'polarity': polarity,
        'tense': tense, 'aspect': aspect, 'evidence': evidence}}


class SelectionData:
    """SUPERSEDED by ``language.py`` (negation-length padding). Kept because the
    diagnosis note reports its leak: exact length matching by stratified sampling
    made the record features 71% predictive of polarity.

    Short-instruction selection with explicit length matching.

    Prompt: the packet's controlled-English record (``data.SCREEN.encode_english``)
    followed by ``SHORT_INSTRUCTION``. Target: the packet's binding target text,
    ``[]`` if positive, else ``[["subject","object"]]``.

    Length matching: candidates are bucketed by exact prompt byte length, and
    every bucket keeps the same number of positive and negative items. Prompt
    length therefore carries no information about the class.
    """

    def __init__(self, rt: Runtime, split: str, seed: int, held_out_combos=(('contain', 'future'),)):
        self.rt = rt
        self.ids = _ids(split)
        self.rng = random.Random(seed)
        self.held_out = set(held_out_combos)
        self.buckets: dict[int, dict[str, list]] = defaultdict(lambda: {'positive': [], 'negative': []})

    def render(self, ast) -> dict:
        record = self.rt.data.SCREEN.encode_english([ast])
        s = ast['statement']
        target = self.rt.data.target_text([] if s['polarity'] == 'positive' else [[s['subject'], s['object']]])
        return dict(prompt=record + SHORT_INSTRUCTION, target=target, ast=ast,
                    polarity=s['polarity'], subject=s['subject'], object=s['object'])

    def candidate(self, combos: str = 'train', polarity: str | None = None) -> dict:
        while True:
            subject, obj = self.rng.sample(self.ids, 2)
            relation, tense = self.rng.choice(RELATIONS), self.rng.choice(TENSES)
            if (combos == 'train') == ((relation, tense) in self.held_out):
                continue
            ast = _pred(subject, obj, relation, polarity or self.rng.choice(('positive', 'negative')),
                        tense, self.rng.choice(ASPECTS), self.rng.choice(EVIDENCES))
            return self.render(ast)

    def matched_pair(self, combos: str = 'train') -> list[dict]:
        """A positive and a negative item with identical prompt byte length."""
        while True:
            row = self.candidate(combos)
            length = len(row['prompt'].encode())
            bucket = self.buckets[length]
            bucket[row['polarity']].append(row)
            if bucket['positive'] and bucket['negative']:
                return [bucket['positive'].pop(), bucket['negative'].pop()]

    def swapped(self, row: dict) -> dict:
        s = dict(row['ast']['statement'])
        s['subject'], s['object'] = s['object'], s['subject']
        return self.render({'version': 'core-v0.1', 'statement': s})

    def flipped(self, row: dict) -> dict:
        s = dict(row['ast']['statement'])
        s['polarity'] = 'positive' if s['polarity'] == 'negative' else 'negative'
        return self.render({'version': 'core-v0.1', 'statement': s})


def length_only_accuracy(rows: list[dict]) -> float:
    """Best accuracy of any classifier that sees only the prompt length (in-sample)."""
    by_length = defaultdict(Counter)
    for r in rows:
        by_length[len(r['prompt'].encode())][r['polarity']] += 1
    return sum(max(c.values()) for c in by_length.values()) / len(rows)


def selection_run(config: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    rt = Runtime(config['source'])
    seed = 20261013 + config['seed']
    train = SelectionData(rt, 'train', seed)
    model = rt.build(seed)
    if config['init'] == 'surrogate':
        rt.load(model, config['surrogate'])
    elif config['init'] == 'copy96':
        rt.load(model, config['copy_init'])
    elif config['init'] != 'fresh':
        raise ValueError(config['init'])
    optimizer = rt.optimizer(model, config['lr'])

    # Fixed evaluation panels (independent of the training seed).
    panel_rng = 7
    ev_in = SelectionData(rt, 'test', panel_rng)
    ev_out = SelectionData(rt, 'test', panel_rng + 1)
    panels = {
        'heldout_ids': [r for _ in range(32) for r in ev_in.matched_pair('train')],
        'heldout_ids_and_combo': [r for _ in range(16) for r in ev_out.matched_pair('heldout')],
    }
    panels['swapped'] = [ev_in.swapped(r) for r in panels['heldout_ids'] if r['polarity'] == 'negative']
    panels['flipped'] = [ev_in.flipped(r) for r in panels['heldout_ids']]

    seen_rows: list[dict] = []
    checkpoints = sorted(set(config['eval_at']) | {config['updates']})
    history, started = [], time.monotonic()
    for step in range(1, config['updates'] + 1):
        rows = []
        for _ in range(config['batch'] // 2):
            rows += train.matched_pair('train')
        seen_rows += rows
        rt.step(model, optimizer, rows)
        if step in checkpoints:
            history.append(dict(step=step, train_seconds=round(time.monotonic() - started, 1),
                                **_score_selection(rt, model, panels)))
    return dict(overrides={k: config[k] for k in ('init', 'updates', 'batch', 'lr', 'seed')},
                length_only_accuracy_train=length_only_accuracy(seen_rows),
                length_only_accuracy_eval=length_only_accuracy(panels['heldout_ids']),
                prompt_bytes=[min(len(r['prompt'].encode()) for r in seen_rows),
                              max(len(r['prompt'].encode()) for r in seen_rows)],
                history=history)


def _score_selection(rt: Runtime, model, panels: dict) -> dict:
    import torch

    out = {}
    for name, rows in panels.items():
        answers = [rt.generate(model, r['prompt'], 32) for r in rows]
        exact = [a == r['target'] for a, r in zip(answers, rows)]
        selected = [e for e, r in zip(exact, rows) if r['polarity'] == 'negative']
        empty = [e for e, r in zip(exact, rows) if r['polarity'] == 'positive']
        out[name] = dict(exact=sum(exact) / len(rows),
                         selected=sum(selected) / max(1, len(selected)),
                         empty=sum(empty) / max(1, len(empty)),
                         distinct=len(set(map(str, answers))))
    # Swap TV at the first identifier byte (teacher forced), negatives only.
    negatives = [r for r in panels['heldout_ids'] if r['polarity'] == 'negative']
    tv = []
    model.eval()
    with torch.no_grad():
        for r, s in zip(negatives, panels['swapped']):
            def first_byte(row):
                ids = [rt.tiny.BOS] + rt.tiny.byte_ids(row['prompt']) + [rt.tiny.SEP] + rt.tiny.byte_ids(row['target'])[:3]
                return model(torch.tensor([ids]))[0, -1].softmax(-1)
            tv.append(float(0.5 * (first_byte(r) - first_byte(s)).abs().sum()))
    out['swap_tv'] = sum(tv) / len(tv)
    return out


def selection_configs(source: str, surrogate_path: str | None, copy_path: str | None):
    inits = ['fresh'] + (['copy96'] if copy_path else []) + (['surrogate'] if surrogate_path else [])
    for init, seed in itertools.product(inits, range(3)):
        yield dict(source=source, surrogate=surrogate_path, copy_init=copy_path, init=init,
                   updates=3072, batch=4, lr=3e-4, seed=seed, eval_at=[192, 384, 768, 1536, 3072])


# ---------------------------------------------------------------------------
# Suite runner


def _suite(configs: list[dict], run, out: Path, workers: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.dumps(json.loads(l)['overrides'], sort_keys=True) for l in out.read_text().splitlines() if l}

    def key(c):
        return json.dumps({k: c[k] for k in run_keys[run]}, sort_keys=True)

    todo = [c for c in configs if key(c) not in done]
    print(f'{len(todo)} runs to do, {len(done)} done', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, out.open('a') as handle:
        futures = [pool.submit(run, c) for c in todo]
        for future in as_completed(futures):
            result = future.result()
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            print(json.dumps(result['overrides']), flush=True)


run_keys = {copy_run: ('init', 'breadth', 'updates', 'seed'),
            selection_run: ('init', 'updates', 'batch', 'lr', 'seed')}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('surrogate')
    s.add_argument('--source', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--threads', type=int, default=4)
    c = sub.add_parser('copy-suite')
    c.add_argument('--source', required=True)
    c.add_argument('--surrogate')
    c.add_argument('--out', default=str(HERE / 'results' / 'copy_matrix.jsonl'))
    c.add_argument('--workers', type=int, default=4)
    c.add_argument('--save-copy96', help='save the fresh-init 96-identifier seed-0 model here')
    t = sub.add_parser('select-suite')
    t.add_argument('--source', required=True)
    t.add_argument('--surrogate')
    t.add_argument('--copy-init')
    t.add_argument('--out', default=str(HERE / 'results' / 'selection.jsonl'))
    t.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.command == 'surrogate':
        surrogate(args)
    elif args.command == 'copy-suite':
        configs = list(copy_configs(args.source, args.surrogate))
        if args.save_copy96:
            for config in configs:
                if (config['init'], config['breadth'], config['updates'], config['seed']) == ('fresh', 96, 192, 0):
                    config['save'] = args.save_copy96
        _suite(configs, copy_run, Path(args.out), args.workers)
    else:
        _suite(list(selection_configs(args.source, args.surrogate, args.copy_init)), selection_run,
               Path(args.out), args.workers)


if __name__ == '__main__':
    main()
