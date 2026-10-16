"""Run the probes and one-factor training variants on the project's own tiny
byte model, using its own source and data (not stored in this repository).

    python real_pipeline.py audit --source <packet>
    python real_pipeline.py train --source <packet> --task copy --data fresh --updates 768 --seed 0
    python real_pipeline.py suite --source <packet> --workers 4 --output results/real_pipeline.jsonl

``<packet>`` is the unpacked results bundle containing ``baseline/source``
(binding run: ``runtime/training``, ``rows.jsonl``, ``plan.json``) and
``copy/source`` (``train.jsonl``, ``evaluation.jsonl``, ``SCHEDULE.json``).
Only ``training.tiny``, ``training.decode``, ``training.binding`` and
``training.binding_train`` (for its ``RECIPE``) are imported, read-only. Models start from ``tiny.build_model`` (the warm-start
checkpoint is not needed or used); the optimiser, loss and clipping repeat
``binding_train.worker`` / ``copy_train.worker`` exactly.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # probes.py, tasks.py
RECIPE: dict = {}  # filled from the packet's binding_train.RECIPE at import time
COPY_ALPHABET = 'abcdef'


def _import_runtime(source: Path):
    runtime = str(source / 'baseline' / 'source' / 'runtime')
    if runtime not in sys.path:
        sys.path.append(runtime)  # appended: never shadows the standard library
    from training import binding, binding_train, decode, tiny
    RECIPE.clear()
    RECIPE.update(binding_train.RECIPE)
    return tiny, decode, binding


def target_loss(logits, labels):
    """Mean cross-entropy over supervised positions (labels != -100), the
    objective the packet's binding and copy workers optimise."""
    import torch
    per_position = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none')
    return per_position.view_as(labels)[labels != -100].mean()


# Optimiser settings recorded in the packet's run plans (fresh AdamW state).
ADAMW = dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
CLIP_NORM = 1.0


def make_optimizer(model, lr: float):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=lr, **ADAMW)


def collate_rows(tiny, rows, context: int):
    """Right-pad the packet's ``tiny.example`` encodings: PAD inputs, -100 labels."""
    import torch
    from torch.nn.utils.rnn import pad_sequence
    pairs = [tiny.example(r, context) for r in rows]
    inputs = pad_sequence([torch.tensor(x) for x, _ in pairs], batch_first=True, padding_value=tiny.PAD)
    labels = pad_sequence([torch.tensor(y) for _, y in pairs], batch_first=True, padding_value=-100)
    return inputs, labels


def train_step(model, optimizer, inputs, labels) -> float:
    import torch
    model.train()
    optimizer.zero_grad()
    loss = target_loss(model(inputs), labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Data


class CopyData:
    """COPY:xyz -> xyz / NONE:xyz -> '' (copy/source), plus a fresh-identifier variant."""

    def __init__(self, source: Path):
        root = source / 'copy' / 'source'
        self.train = _jsonl(root / 'train.jsonl')
        self.evaluation = _jsonl(root / 'evaluation.jsonl')
        rows_by_id = {row['id']: row for row in self.train}
        steps = json.loads((root / 'SCHEDULE.json').read_text())
        self.schedule = [[rows_by_id[i] for i in step['row_ids']] for step in steps]
        heldout = {r['identifier'] for r in self.evaluation if r['split'] == 'heldout'}
        self.fresh_pool = [''.join(p) for p in itertools.product(COPY_ALPHABET, repeat=3) if ''.join(p) not in heldout]

    @staticmethod
    def row(identifier: str, copy: bool) -> dict:
        opcode = 'COPY' if copy else 'NONE'
        return dict(id=f'fresh-{identifier}-{opcode.lower()}', prompt=f'{opcode}:{identifier}',
                    target=identifier if copy else '', split='train', identifier=identifier)

    def fresh_quartet(self, rng: random.Random) -> list[dict]:
        a, b = rng.sample(self.fresh_pool, 2)
        rows = [self.row(a, True), self.row(a, False), self.row(b, True), self.row(b, False)]
        rng.shuffle(rows)
        return rows

    def cohorts(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in self.evaluation:
            kind = 'copy' if r['target'] else 'none'
            out.setdefault(f"{'seen' if r['split'] == 'train' else 'heldout'}_{kind}", []).append(r)
        return out


class BindingData:
    """One English record; [] if positive, [["subject","object"]] if negative (baseline/source)."""

    def __init__(self, source: Path, binding, short: bool = False):
        root = source / 'baseline' / 'source'
        self.binding = binding
        self.short = short
        rows = [r for r in _jsonl(root / 'rows.jsonl') if r['arm'] == 'english']
        plan = json.loads((root / 'plan.json').read_text())
        rows_by_semantic_id = {row['semantic_id']: row for row in rows}
        sample = next(r for r in rows if r['target'] != '[]')
        x, y = self.identifiers(sample)
        record = f'1. "{x}" does not see "{y}".'
        self.prefix, self.suffix = sample['prompt'].split(record)
        self.real_ids = {i for r in rows for i in self.identifiers(r)}
        self.schedule = [[self._view(rows_by_semantic_id[s]) for s in batch] for batch in plan['schedule']]
        self.evaluation = [self._view(rows_by_semantic_id[s]) for s in plan['evaluation_ids']]
        # Fresh identifiers: letter-digit-letter, independent characters, minus the real ones.
        import tasks  # this directory
        self.fresh_train = [i for i in tasks.FreshIds('train').ids if i not in self.real_ids]
        self.fresh_test = [i for i in tasks.FreshIds('test').ids if i not in self.real_ids]

    @staticmethod
    def identifiers(row: dict) -> tuple[str, str]:
        line = row['prompt'].split('Records:\n')[1].split('\n')[0]
        parts = line.split('"')
        return parts[1], parts[3]

    def make(self, x: str, y: str, negative: bool, split: str = 'train') -> dict:
        record = f'1. "{x}" does not see "{y}".' if negative else f'1. "{x}" sees "{y}".'
        target = self.binding.canonical([[x, y]]) if negative else '[]'
        if self.short:
            prompt = f'{record[3:]}\n[] if positive, else [["subject","object"]].\n'
        else:
            prompt = self.prefix + record + self.suffix
        return dict(id=f'{split}-{x}-{y}-{"neg" if negative else "pos"}', prompt=prompt, target=target,
                    split=split, subject=x, object=y)

    def _view(self, row: dict) -> dict:
        x, y = self.identifiers(row)
        made = self.make(x, y, row['target'] != '[]', row['split'])
        if not self.short:
            assert made['prompt'] == row['prompt'] and made['target'] == row['target'], row['id']
        return made

    def fresh_quartet(self, rng: random.Random, pool: list[str] | None = None) -> list[dict]:
        x, y = rng.sample(pool or self.fresh_train, 2)
        rows = [self.make(x, y, True), self.make(x, y, False), self.make(y, x, True), self.make(y, x, False)]
        rng.shuffle(rows)
        return rows

    def cohorts(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in self.evaluation:
            kind = 'selected' if r['target'] != '[]' else 'empty'
            out.setdefault(f"{'seen' if r['split'] == 'train' else 'unseen_pair'}_{kind}", []).append(r)
        rng = random.Random(7)
        fresh = [q for _ in range(8) for q in self.fresh_quartet(rng, self.fresh_test)]
        out['fresh_id_selected'] = [r for r in fresh if r['target'] != '[]']
        out['fresh_id_empty'] = [r for r in fresh if r['target'] == '[]']
        return out


# ---------------------------------------------------------------------------
# Training and evaluation, mirroring binding_train.worker / copy_train.worker


def _first_byte_distributions(model, tiny, rows, torch, offset: int):
    """Teacher-forced next-byte distribution at target index ``offset``."""
    out = []
    with torch.no_grad():
        model.eval()
        for r in rows:
            ids = [tiny.BOS] + tiny.byte_ids(r['prompt']) + [tiny.SEP] + tiny.byte_ids(r['target'])[:offset]
            out.append(model(torch.tensor([ids]))[0, -1].softmax(-1))
    return torch.stack(out)


def _js_bits(p, torch) -> float:
    def entropy(q):
        return -(q * q.clamp_min(1e-30).log2()).sum(-1)
    return float(entropy(p.mean(0)) - entropy(p).mean())


def run(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    tiny, decode, binding = _import_runtime(Path(args.source))
    if args.task == 'copy':
        data = CopyData(Path(args.source))
        first_offset, max_new = 0, 4
    else:
        data = BindingData(Path(args.source), binding, short=args.data in ('short', 'short_fresh'))
        first_offset, max_new = 3, 32
    fresh = args.data in ('fresh', 'short_fresh')
    seed = RECIPE['seed'] + args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    model = tiny.build_model(RECIPE).cpu()
    optimizer = make_optimizer(model, args.lr)
    quartets_per_update = args.batch // 4
    started = time.monotonic()
    history = []
    for step in range(args.updates):
        rows = []
        for k in range(quartets_per_update):
            if fresh:
                rows += data.fresh_quartet(rng)
            else:
                rows += data.schedule[(step * quartets_per_update + k) % len(data.schedule)]
        loss = train_step(model, optimizer, *collate_rows(tiny, rows, RECIPE['context']))
        if (step + 1) % max(1, args.updates // 8) == 0:
            history.append(dict(step=step + 1, loss=round(loss, 4)))
    train_seconds = time.monotonic() - started

    final = {}
    for name, rows in data.cohorts().items():
        outputs = [decode.complete(model, r['prompt'], backend='reference', context=2048,
                                   max_new_tokens=max_new, max_seconds=60) for r in rows]
        answers = [o['answer'] if o['status'] == 'complete' else None for o in outputs]
        exact = [a == r['target'] for a, r in zip(answers, rows)]
        counts = Counter(map(str, answers))
        entry = dict(n=len(rows), exact=sum(exact) / len(rows), distinct_answers=len(counts),
                     mode_answer=min(counts, key=lambda a: (-counts[a], a)))  # ties: smallest string
        if rows[0]['target'] not in ('', '[]'):  # selected/COPY cohorts only
            dists = _first_byte_distributions(model, tiny, rows, torch, first_offset)
            gold = torch.tensor([r['target'].encode()[first_offset] for r in rows])
            entry['first_byte_acc'] = float((dists.argmax(-1) == gold).float().mean())
            entry['first_byte_js_bits'] = _js_bits(dists, torch)
            if args.task == 'binding':
                swapped = [data.make(r['object'], r['subject'], True, r['split']) for r in rows]
                swapped_dists = _first_byte_distributions(model, tiny, swapped, torch, first_offset)
                entry['swap_tv'] = float(0.5 * (dists - swapped_dists).abs().sum(-1).mean())
        final[name] = entry
    return dict(overrides=dict(task=args.task, data=args.data, updates=args.updates, batch=args.batch,
                               lr=args.lr, seed=args.seed),
                torch=torch.__version__, train_seconds=round(train_seconds, 1), history=history, final=final)


# ---------------------------------------------------------------------------
# Audit: mechanical probes on the real code


def audit(args) -> list:
    import torch

    import probes

    tiny, decode, binding = _import_runtime(Path(args.source))
    data = BindingData(Path(args.source), binding)
    quartet = data.schedule[0]
    torch.manual_seed(0)
    model = tiny.build_model(RECIPE)
    with torch.no_grad():  # sharpen the untrained argmax so decoding is context sensitive
        model.output.weight.mul_(30)

    def collate(rows):
        inputs, labels = collate_rows(tiny, rows, RECIPE['context'])
        return {'input_ids': inputs, 'labels': labels}

    def loss_fn(logits, batch):  # binding_train.worker objective
        return target_loss(logits, batch['labels'])

    def train_path(ids):
        model.train()
        return model(ids)

    def eval_path(ids):  # decode.complete: eval mode + no_grad (fused fast path)
        model.eval()
        with torch.no_grad():
            return model(ids)

    selected = collate([r for r in quartet if r['target'] != '[]'])
    query = int((selected['labels'][0] != -100).nonzero().min())
    context = selected['input_ids'][:1, : query + 1]
    results = []
    for name, path in (('train', train_path), ('eval', eval_path)):
        r = probes.causality_probe(lambda ids: path(ids).detach(), context[:, -96:], tiny.VOCAB)
        r.name = f'causality[{name} path, last 96 positions]'
        results.append(r)
        r = probes.visibility_probe(lambda ids: path(ids).detach(), context, query, tiny.VOCAB)
        r.name = f'visibility[{name} path]'
        results.append(r)
    a, b = train_path(context).detach(), eval_path(context)
    diff = float((a - b).abs().max())
    results.append(probes.ProbeResult('train_vs_eval_path', diff < 1e-4, {
        'max_logit_diff': diff, 'same_argmax_at_query': bool(a[0, -1].argmax() == b[0, -1].argmax())}))
    batch = collate(quartet)
    model.train()
    results.append(probes.gradient_reach_probe(model, model.tokens, batch, loss_fn, train_path))
    results.append(probes.parameter_gradient_probe(model, loss_fn(train_path(batch['input_ids']), batch),
                                                   make_optimizer(model, 3e-4)))
    full = [x + [y[-1]] for x, y in (tiny.example(r, RECIPE['context']) for r in quartet)]  # x = seq[:-1]
    results.append(probes.oracle_label_probe(loss_fn, batch, tiny.VOCAB, tiny.PAD, sequences=full))

    def save(m):
        buffer = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, buffer)
        buffer.seek(0)
        return buffer

    def load(m, buffer):
        m.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)

    def build_fresh():
        torch.manual_seed(1)
        return tiny.build_model(RECIPE)

    model.eval()
    results.append(probes.checkpoint_roundtrip_probe(model, build_fresh, save, load, context))
    results.append(probes.decoder_consistency_probe(
        lambda prompt: decode.complete(model, prompt, backend='reference', context=2048,
                                       max_new_tokens=6, max_seconds=60)['generated_ids'],
        lambda ids: train_path(ids).detach(),
        lambda prompt: [tiny.BOS] + tiny.byte_ids(prompt) + [tiny.SEP],
        [r['prompt'] for r in quartet]))
    by_length = sorted(quartet, key=lambda r: len(r['prompt']))
    results.append(probes.padding_invariance_probe(collate, lambda b: train_path(b['input_ids']).detach(), by_length))
    print(f'torch {torch.__version__}')
    for r in results:
        print(r)
    return results


# ---------------------------------------------------------------------------
# Suite


def suite_configs():
    for seed in range(3):
        yield dict(task='copy', data='fixed', updates=192, batch=4, lr=3e-4, seed=seed)  # the run as executed
        yield dict(task='copy', data='fresh', updates=192, batch=4, lr=3e-4, seed=seed)
        yield dict(task='copy', data='fresh', updates=768, batch=4, lr=3e-4, seed=seed)
        yield dict(task='copy', data='fresh', updates=1536, batch=4, lr=3e-4, seed=seed)
        yield dict(task='copy', data='fixed', updates=1536, batch=4, lr=3e-4, seed=seed)
        yield dict(task='binding', data='fixed', updates=192, batch=4, lr=3e-4, seed=seed)  # the run as executed
    for seed in range(2):
        yield dict(task='binding', data='fixed', updates=1536, batch=4, lr=3e-4, seed=seed)
        yield dict(task='binding', data='fixed', updates=768, batch=16, lr=1e-3, seed=seed)
        yield dict(task='binding', data='fresh', updates=1536, batch=4, lr=3e-4, seed=seed)
        yield dict(task='binding', data='fresh', updates=768, batch=16, lr=1e-3, seed=seed)
        yield dict(task='binding', data='short', updates=192, batch=4, lr=3e-4, seed=seed)
        yield dict(task='binding', data='short', updates=1536, batch=4, lr=3e-4, seed=seed)
        yield dict(task='binding', data='short_fresh', updates=1536, batch=4, lr=3e-4, seed=seed)
        # Budget bounds for copying genuinely new identifiers.
        yield dict(task='binding', data='short_fresh', updates=192, batch=4, lr=3e-4, seed=seed)
        yield dict(task='binding', data='short_fresh', updates=768, batch=4, lr=3e-4, seed=seed)
        yield dict(task='binding', data='fresh', updates=1536, batch=16, lr=1e-3, seed=seed)


def _suite_one(source: str, config: dict) -> dict:
    args = argparse.Namespace(source=source, threads=1, **config)
    return run(args)


def suite(args) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.dumps(json.loads(l)['overrides'], sort_keys=True) for l in out.read_text().splitlines() if l}
    configs = [c for c in suite_configs() if json.dumps(c, sort_keys=True) not in done]
    print(f'{len(configs)} runs to do, {len(done)} done', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, out.open('a') as handle:
        futures = [pool.submit(_suite_one, args.source, c) for c in configs]
        for future in as_completed(futures):
            result = future.result()
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            print(json.dumps(result['overrides']), json.dumps({k: round(v['exact'], 3) for k, v in result['final'].items()}),
                  flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--source', required=True)
    t = sub.add_parser('train')
    t.add_argument('--source', required=True)
    t.add_argument('--task', choices=('copy', 'binding'), required=True)
    t.add_argument('--data', choices=('fixed', 'fresh', 'short', 'short_fresh'), default='fixed')
    t.add_argument('--updates', type=int, default=192)
    t.add_argument('--batch', type=int, default=4, help='rows per update; a multiple of 4 (quartets)')
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--threads', type=int, default=4)
    s = sub.add_parser('suite')
    s.add_argument('--source', required=True)
    s.add_argument('--workers', type=int, default=4)
    s.add_argument('--output', default=str(HERE / 'results' / 'real_pipeline.jsonl'))
    args = p.parse_args()
    if args.command == 'audit':
        raise SystemExit(0 if all(r.passed for r in audit(args)) else 1)
    if args.command == 'train':
        if args.batch % 4:
            p.error('--batch must be a multiple of 4')
        print(json.dumps(run(args)))
    else:
        suite(args)


if __name__ == '__main__':
    main()
