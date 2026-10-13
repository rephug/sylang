"""English-versus-Sylang comparison on the project's tiny byte model.

    python language.py audit --source <packet>
    python language.py suite --source <packet> --suite pilot --out results/language_pilot.jsonl
    python language.py suite --source <packet> --suite main --out results/language.jsonl
    python language.py suite --source <packet> --suite selection-init --copy-init <copy96.pt> \
        --surrogate <surrogate.pt> --out results/selection_init.jsonl

**Task.** Each item holds K predicate records (K uniform in 1..4 by default). The
answer is the JSON list of ``[subject, object]`` for every negative record, in
record order, or ``[]``. It is the packet's binding target text, with identical
bytes in every arm.

**Arms.** Renderers are imported at run time from ``<packet>/fixture-authoring-source``:

- ``english``: controlled English (``data.SCREEN.encode_english``).
- ``words``: Sylang batch-v1 words with selective escaping; the same tab layout as codes.
- ``codes``: Sylang batch-v1 codes with selective escaping.
- ``codes_padded``: codes with each record line padded to that record's ``words`` line length. It has the same symbols as codes and the same length and layout as words.

Paired contrasts:

- **codes vs codes_padded** isolates length (compression).
- **words vs codes_padded** isolates the symbols at equal length.
- **english vs words** compares prose with the tabular layout.

**Fairness controls.**

- **Identical inputs except the rendering.** Every arm sees the same item stream (records, identifiers, order) for a given seed, the same instruction, model, initialisation, optimiser, learning rate, batch and number of updates.
- **Negation-length padding.** Each positive record line carries filler (``~``) equal to the bytes its negative form adds in that arm. Record and prompt lengths are then independent of polarity, while features are sampled independently of polarity. Exact length-distribution matching by stratified sampling was rejected: it made features 71% predictive of polarity (see the diagnosis note). The filler is a local, per-record marker carrying the same information as the negation word; ``audit`` reports every residual cue.
- **Held-out identifiers** (a disjoint split) and a **held-out (relation, tense) combination**.
- **Counterfactual panels:** subject/object swap in every record, and a single-record polarity flip.
"""

from __future__ import annotations

import argparse
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

from followup import Runtime  # noqa: E402

ARMS = ('english', 'words', 'codes', 'codes_padded')
INSTRUCTION = 'List [subject,object] of each negative record in order; [] if none.\n'
FILLER = '~'
RELATIONS = ('see', 'help', 'contain')
TENSES = ('past', 'present', 'future')
ASPECTS = ('simple', 'progressive', 'completed')
EVIDENCES = ('unspecified', 'direct', 'reported', 'inferred')
HELD_OUT_COMBO = ('contain', 'future')  # (relation, tense) never in training records


def _statement(subject, obj, relation, polarity, tense, aspect, evidence):
    return {'kind': 'pred', 'subject': subject, 'relation': relation, 'object': obj, 'polarity': polarity,
            'tense': tense, 'aspect': aspect, 'evidence': evidence}


def _ast(statement):
    return {'version': 'core-v0.1', 'statement': statement}


class Renderer:
    """Arm renderings with negation-length padding, using the packet's codecs."""

    def __init__(self, rt: Runtime):
        self.screen = rt.data.SCREEN
        self.target_text = rt.data.target_text

    def _payload(self, arm: str, statements: list[dict]) -> str:
        asts = [_ast(s) for s in statements]
        if arm == 'english':
            return self.screen.encode_english(asts)
        return self.screen.encode_batch(asts, arm, False, 'selective')

    def _lines(self, arm: str, statements: list[dict]) -> tuple[list[str], list[str]]:
        """(header lines, one line per record), each without its newline."""
        lines = self._payload(arm, statements).split('\n')
        assert lines[-1] == '', 'payload must end with a newline'
        lines = lines[:-1]
        header = lines[: len(lines) - len(statements)]
        records = lines[len(header):]
        assert len(records) == len(statements) and len(header) == (0 if arm == 'english' else 1)
        return header, records

    def _negation_bytes(self, arm: str, statement: dict) -> int:
        if statement['polarity'] == 'negative':
            return 0
        negative = dict(statement, polarity='negative')
        _, (pos,) = self._lines(arm, [statement])
        _, (neg,) = self._lines(arm, [negative])
        return len(neg.encode()) - len(pos.encode())

    def prompt(self, arm: str, statements: list[dict]) -> str:
        base = 'words' if arm == 'codes_padded' else arm
        header, records = self._lines('codes' if arm == 'codes_padded' else base, statements)
        padded = []
        for statement, line in zip(statements, records):
            if arm == 'codes_padded':
                _, (words_line,) = self._lines('words', [statement])
                width = len(words_line.encode()) + self._negation_bytes('words', statement)
            else:
                width = len(line.encode()) + self._negation_bytes(arm, statement)
            gap = width - len(line.encode())
            assert gap >= 0, (arm, line)
            padded.append(line + FILLER * gap)
        return ''.join(h + '\n' for h in header) + ''.join(p + '\n' for p in padded) + INSTRUCTION

    def target(self, statements: list[dict]) -> str:
        return self.target_text([[s['subject'], s['object']] for s in statements if s['polarity'] == 'negative'])


class ItemStream:
    """Items with independently sampled records; identical across arms for a seed."""

    def __init__(self, split: str, seed: int, k_values=(1, 2, 3, 4), combos: str = 'train'):
        import tasks
        self.ids = tasks.FreshIds(split).ids
        self.rng = random.Random(seed)
        self.k_values = tuple(k_values)
        self.combos = combos

    def record(self, used: set, held_out: bool | None) -> dict:
        while True:
            relation, tense = self.rng.choice(RELATIONS), self.rng.choice(TENSES)
            is_held_out = (relation, tense) == HELD_OUT_COMBO
            if held_out is not None and is_held_out != held_out:
                continue
            subject, obj = self.rng.sample(self.ids, 2)
            if subject in used or obj in used:
                continue
            used.update((subject, obj))
            return _statement(subject, obj, relation, self.rng.choice(('positive', 'negative')), tense,
                              self.rng.choice(ASPECTS), self.rng.choice(EVIDENCES))

    def item(self) -> list[dict]:
        k = self.rng.choice(self.k_values)
        used: set = set()
        if self.combos == 'train':
            return [self.record(used, False) for _ in range(k)]
        # 'held_out': at least one record uses the held-out combination.
        forced = self.rng.randrange(k)
        return [self.record(used, True if i == forced else None) for i in range(k)]


def swap_roles(statements):
    return [dict(s, subject=s['object'], object=s['subject']) for s in statements]


def flip_one(statements, rng: random.Random):
    i = rng.randrange(len(statements))
    flipped = [dict(s) for s in statements]
    flipped[i]['polarity'] = 'positive' if flipped[i]['polarity'] == 'negative' else 'negative'
    return flipped


def eval_panels(seed: int = 7, n: int = 96) -> dict[str, list[list[dict]]]:
    iid = ItemStream('test', seed)
    combo = ItemStream('test', seed + 1, combos='held_out')
    panels = {'iid': [iid.item() for _ in range(n)], 'held_out_combo': [combo.item() for _ in range(n // 2)]}
    rng = random.Random(seed + 2)
    panels['swap'] = [swap_roles(it) for it in panels['iid'] if any(s['polarity'] == 'negative' for s in it)]
    panels['flip'] = [flip_one(it, rng) for it in panels['iid']]
    return panels


# ---------------------------------------------------------------------------
# Audit: what can predict the answer without reading the records?


def _heldout_lookup_accuracy(pairs: list[tuple], majority_key=None) -> float:
    """Fit a lookup classifier (feature value -> majority label) on the first half,
    score it on the second half; unseen feature values fall back to the overall
    majority of the first half."""
    half = len(pairs) // 2
    table = defaultdict(Counter)
    overall = Counter()
    for feature, label in pairs[:half]:
        table[feature][label] += 1
        overall[label] += 1
    default = overall.most_common(1)[0][0]
    test = pairs[half:]
    hits = sum((table[f].most_common(1)[0][0] if table[f] else default) == label for f, label in test)
    return hits / len(test)


def audit(args) -> dict:
    """Held-out accuracy of classifiers that see only a cue, not the records' text."""
    rt = Runtime(args.source)
    renderer = Renderer(rt)
    stream = ItemStream('train', 20261013)
    items = [stream.item() for _ in range(args.items)]
    report = {}
    for arm in ARMS:
        prompts = [renderer.prompt(arm, it) for it in items]
        natural = [renderer._payload('codes' if arm == 'codes_padded' else arm, it) + INSTRUCTION for it in items]
        n_negative = [sum(s['polarity'] == 'negative' for s in it) for it in items]
        record_cues = defaultdict(list)
        for it, prompt in zip(items, prompts):
            header, _ = renderer._lines('codes' if arm == 'codes_padded' else arm, it)
            lines = prompt.split('\n')[len(header):len(header) + len(it)]
            for st, line in zip(it, lines):
                record_cues['padded_line_length'].append((len(line.encode()), st['polarity']))
                record_cues['features'].append(((st['relation'], st['tense'], st['aspect'], st['evidence']),
                                                st['polarity']))
                record_cues['filler_present'].append((FILLER in line, st['polarity']))
        report[arm] = dict(
            mean_prompt_bytes=round(sum(len(p.encode()) for p in prompts) / len(prompts), 1),
            mean_natural_bytes=round(sum(len(p.encode()) for p in natural) / len(natural), 1),
            mean_prompt_bytes_by_k={k: round(sum(len(p.encode()) for it, p in zip(items, prompts) if len(it) == k)
                                             / max(1, sum(len(it) == k for it in items)), 1) for k in (1, 2, 3, 4)},
            n_negative_from_length_and_k=round(_heldout_lookup_accuracy(
                [((len(it), len(p.encode())), n) for it, p, n in zip(items, prompts, n_negative)]), 4),
            n_negative_from_k_only=round(_heldout_lookup_accuracy(
                [(len(it), n) for it, n in zip(items, n_negative)]), 4),
            record_polarity_from={name: round(_heldout_lookup_accuracy(pairs), 4)
                                  for name, pairs in record_cues.items()},
        )
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Training


def run(config: dict) -> dict:
    import torch

    torch.set_num_threads(config.get('threads', 1))
    rt = Runtime(config['source'])
    renderer = Renderer(rt)
    seed = 20261013 + config['seed']
    model = rt.build(seed)
    if config['init'] == 'copy96':
        rt.load(model, config['copy_init'])
    elif config['init'] == 'surrogate':
        rt.load(model, config['surrogate'])
    elif config['init'] != 'fresh':
        raise ValueError(config['init'])
    optimizer = rt.optimizer(model, config['lr'])
    stream = ItemStream('train', seed, k_values=tuple(config['k']))
    panels = {name: [dict(prompt=renderer.prompt(config['arm'], it), target=renderer.target(it), statements=it)
                     for it in items if len(it) in config['k']]
              for name, items in eval_panels().items()}
    checkpoints = sorted(set(config['eval_at']) | {config['updates']})
    history, seconds, bytes_seen, items_seen = [], 0.0, 0, 0
    for step in range(1, config['updates'] + 1):
        items = [stream.item() for _ in range(config['batch'])]
        rows = [dict(id=str(i), prompt=renderer.prompt(config['arm'], it), target=renderer.target(it))
                for i, it in enumerate(items)]
        started = time.perf_counter()
        rt.step(model, optimizer, rows)
        seconds += time.perf_counter() - started
        bytes_seen += sum(len(r['prompt'].encode()) + len(r['target'].encode()) + 3 for r in rows)
        items_seen += len(rows)
        if step in checkpoints:
            history.append(dict(step=step, items_seen=items_seen, bytes_seen=bytes_seen,
                                train_seconds=round(seconds, 2), **score(rt, model, panels)))
    return dict(overrides={k: config[k] for k in ('arm', 'init', 'k', 'updates', 'batch', 'lr', 'seed')},
                torch=torch.__version__, history=history,
                inference=inference_cost(rt, model, panels['iid']))


def score(rt: Runtime, model, panels: dict) -> dict:
    import torch

    out = {}
    for name, rows in panels.items():
        answers = [rt.generate(model, r['prompt'], 64) for r in rows]
        exact = [a == r['target'] for a, r in zip(answers, rows)]
        by_k = defaultdict(list)
        for e, r in zip(exact, rows):
            by_k[len(r['statements'])].append(e)
        empty = [e for e, r in zip(exact, rows) if r['target'] == '[]']
        out[name] = dict(exact=round(sum(exact) / len(rows), 4), n=len(rows),
                         empty=round(sum(empty) / max(1, len(empty)), 4),
                         by_k={k: round(sum(v) / len(v), 4) for k, v in sorted(by_k.items())})
    # Swap TV at the first identifier byte of the answer (teacher forced).
    model.eval()
    tv = []
    with torch.no_grad():
        originals = [r for r in panels['iid'] if r['target'] != '[]']
        for original, swapped in zip(originals, panels['swap']):
            dists = []
            for row in (original, swapped):
                ids = ([rt.tiny.BOS] + rt.tiny.byte_ids(row['prompt']) + [rt.tiny.SEP]
                       + rt.tiny.byte_ids(row['target'])[:3])
                dists.append(model(torch.tensor([ids]))[0, -1].softmax(-1))
            tv.append(float(0.5 * (dists[0] - dists[1]).abs().sum()))
    out['swap_tv'] = round(sum(tv) / max(1, len(tv)), 4)
    return out


def inference_cost(rt: Runtime, model, rows: list[dict]) -> dict:
    """Prefill time over the prompt and decode steps, measured on this machine."""
    import torch

    model.eval()
    prefill = []
    with torch.no_grad():
        for r in rows:
            ids = torch.tensor([[rt.tiny.BOS] + rt.tiny.byte_ids(r['prompt']) + [rt.tiny.SEP]])
            started = time.perf_counter()
            model(ids)
            prefill.append(time.perf_counter() - started)
    return dict(mean_prompt_bytes=sum(len(r['prompt'].encode()) for r in rows) / len(rows),
                mean_answer_bytes=sum(len(r['target'].encode()) for r in rows) / len(rows),
                mean_prefill_ms=round(1000 * sum(prefill) / len(prefill), 3))


def suite_configs(name: str, args) -> list[dict]:
    base = dict(source=args.source, copy_init=args.copy_init, surrogate=args.surrogate, init='fresh',
                k=[1, 2, 3, 4], batch=8, lr=1e-3)
    configs = []
    if name == 'pilot':
        for arm, lr in itertools.product(('english', 'codes'), (3e-4, 1e-3)):
            configs.append(dict(base, arm=arm, lr=lr, seed=0, updates=4000, eval_at=[500, 1000, 2000, 4000]))
    elif name == 'main':
        for arm, seed in itertools.product(ARMS, range(args.seeds)):
            configs.append(dict(base, arm=arm, seed=seed, lr=args.lr, updates=args.updates,
                                eval_at=[args.updates // 8, args.updates // 4, args.updates // 2, args.updates]))
    elif name == 'selection-init':
        inits = ['fresh'] + (['copy96'] if args.copy_init else []) + (['surrogate'] if args.surrogate else [])
        for init, seed in itertools.product(inits, range(args.seeds)):
            configs.append(dict(base, arm='english', init=init, k=[1], batch=4, lr=3e-4, seed=seed,
                                updates=3072, eval_at=[192, 384, 768, 1536, 3072]))
    else:
        raise ValueError(name)
    return configs


def suite(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ('arm', 'init', 'k', 'updates', 'batch', 'lr', 'seed')
    done = set()
    if out.exists():
        done = {json.dumps(json.loads(l)['overrides'], sort_keys=True) for l in out.read_text().splitlines() if l}
    todo = [c for c in suite_configs(args.suite, args)
            if json.dumps({k: c[k] for k in keys}, sort_keys=True) not in done]
    print(f'{len(todo)} runs to do, {len(done)} done', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, out.open('a') as handle:
        for future in as_completed([pool.submit(run, c) for c in todo]):
            result = future.result()
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            last = result['history'][-1]
            print(json.dumps(result['overrides']), 'iid', last['iid']['exact'], 'combo',
                  last['held_out_combo']['exact'], 'swap', last['swap']['exact'], 'flip', last['flip']['exact'],
                  flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--source', required=True)
    a.add_argument('--items', type=int, default=20000)
    s = sub.add_parser('suite')
    s.add_argument('--source', required=True)
    s.add_argument('--suite', choices=('pilot', 'main', 'selection-init'), required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--workers', type=int, default=4)
    s.add_argument('--seeds', type=int, default=3)
    s.add_argument('--updates', type=int, default=4000)
    s.add_argument('--lr', type=float, default=1e-3)
    s.add_argument('--copy-init')
    s.add_argument('--surrogate')
    args = p.parse_args()
    if args.command == 'audit':
        audit(args)
    else:
        suite(args)


if __name__ == '__main__':
    main()
