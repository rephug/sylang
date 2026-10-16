"""Generation-based evaluation of one cell of the pretrained-model pilot (``interfaces.md``,
section "Evaluation").

    python evaluate.py --config configs/<cell>.json --adapter runs/<name>/best --split test --out results/<name>
    python evaluate.py --config configs/<cell>.json --adapter none --split test --out results/zeroshot_<arm>

What happens, in order:

1. the config, tokenizer (``common.check_template`` asserts the chat format), base model and,
   unless ``--adapter none``, the PEFT adapter are loaded (model load time is excluded from the
   cost clock);
2. the split's examples of the config's arm are read and restricted to the config's tasks
   (``--limit N`` keeps N examples, taken round-robin over the panels in file order so a smoke
   test touches every panel);
3. every prompt is assembled and tokenised (``tokenize_s`` is that wall-clock; ``convert_in_s``
   is the codec time: the example's rendering wall-clock ``render_s`` recorded by data.py, 0.0
   when the data file predates that key), then, after one untimed warm-up generation on the
   first request (CUDA/cuBLAS initialisation; the generation clock is reset afterwards), decoded
   greedily in batches with left padding, stopping at ``<|im_end|>``; ``--latency-sample N`` runs
   N items (spread evenly within each task's items: the files interleave the tasks, so an even
   stride over the whole list would pick one task only) one at a time instead, which gives the
   single-request latency distribution; batched items carry the batch wall-clock divided by the
   batch size;
4. outputs are decoded with ``common.decode_output`` (the ids before the first ``<|im_end|>``,
   special tokens skipped: the rule train_lora.py uses for dev selection) and graded with
   ``grader.grade`` (``convert_out_s`` is the parse + grade wall-clock only; decoding is
   model-side and outside that clock); every item whose first output is ``invalid`` gets exactly
   one retry request (``common.retry_user_text``; its tokenisation is added to ``tokenize_s``,
   the codec is not re-run), run and graded the same way; both requests are counted in tokens,
   latency and cost, and ``*_final`` reflects the retry;
5. ``items.jsonl`` (one line per example; synthetic ids, flags and numbers only) and
   ``eval.json`` (the aggregates) are written to ``--out``.

Token accounting: ``common.token_breakdown`` for the prompt parts; output tokens are the
generated tokens before the end token (all of them when the model never stops).  The retry
request's prompt is counted whole (``prompt_retry``) and its output as ``output_retry``.
``tokens.totals.uncached`` is every token (prompt + output) of every request including retries;
``cached`` counts the system message once per 100 requests instead of once per request.

Nothing here writes prompt text or model output to the results directory.  ``--dump-outputs``
writes the raw outputs with their grades to a separate file for debugging (default off).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # python -I does not add the script's directory
    sys.path.insert(0, str(HERE))

import common
from common import (FORMAT_REMINDER, IM_END, Stopwatch, check_template, count_tokens, examples_path, git_commit,
                    load_adapter, load_config, load_legends, load_model, load_tokenizer, now_iso, read_jsonl,
                    resolve_device, resolve_dtype, resolve_model_path, retry_user_text, token_breakdown, user_text,
                    write_json, write_jsonl)

CACHE_WINDOW = 100   # the cached convention: the system message is counted once per this many requests
TOKEN_PARTS = ('system', 'records', 'instruction', 'template', 'prompt_total')


# ---------------------------------------------------------------------------
# Command line

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='configs/<cell>.json (arm, tasks, answer_format, data_version, ...)')
    p.add_argument('--adapter', required=True, help='runs/<name>/best (a PEFT adapter directory) or "none" for zero-shot')
    p.add_argument('--split', required=True, choices=('test', 'dev'))
    p.add_argument('--out', required=True, help='results/<name>: eval.json and items.jsonl are written here')
    p.add_argument('--price-per-hour', type=float, default=0.0, help='instance price; fills eval.json "cost"')
    p.add_argument('--latency-sample', type=int, default=0, metavar='N',
                   help='run N items one at a time (latency_mode: single) for the single-request latency')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='auto', help='auto | cpu | cuda | cuda:N')
    p.add_argument('--limit', type=int, default=None, metavar='N',
                   help='smoke tests: keep N examples, round-robin over the panels in file order')
    p.add_argument('--threads', type=int, default=None, help='torch CPU threads (and single-threaded tokenisation)')
    p.add_argument('--dump-outputs', default=None, metavar='PATH',
                   help='debugging only: write every raw output with its grade to this JSONL file (never the results dir)')
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Example selection

def select_examples(rows: list[dict], tasks, limit: int | None) -> list[dict]:
    rows = [r for r in rows if r['task'] in tasks]
    if limit is None or limit >= len(rows):
        return rows
    by_panel: dict[str, list[dict]] = {}
    for r in rows:
        by_panel.setdefault(r['panel'], []).append(r)
    queues = list(by_panel.values())
    chosen = []
    while len(chosen) < limit:
        for q in queues:
            if q and len(chosen) < limit:
                chosen.append(q.pop(0))
    order = {id(r): i for i, r in enumerate(rows)}
    chosen.sort(key=lambda r: order[id(r)])
    return chosen


def latency_sample_indices(tasks: list[str], n_sample: int) -> set[int]:
    """Indices of the N single-request latency items, stratified by task: ``tasks`` is the item
    list's task per index; the N are split as evenly as possible over the tasks (in order of
    first appearance, the remainder to the first ones) and each task's share is spread evenly
    over that task's own items.  (The files interleave the tasks, so an even stride over the whole
    list would pick one task only.)"""
    positions: dict[str, list[int]] = {}
    for i, task in enumerate(tasks):
        positions.setdefault(task, []).append(i)
    n_sample = max(0, min(n_sample, len(tasks)))
    if n_sample == 0:
        return set()
    share = {task: 0 for task in positions}
    remaining = n_sample
    while remaining:   # round-robin over the tasks; terminates because n_sample <= len(tasks)
        for task, pos in positions.items():
            if remaining and share[task] < len(pos):
                share[task] += 1
                remaining -= 1
    chosen: set[int] = set()
    for task, pos in positions.items():
        m = share[task]
        chosen.update(pos[int(j * len(pos) / m)] for j in range(m))
    return chosen


# ---------------------------------------------------------------------------
# Generation

class Generator:
    """Greedy decoding with left padding, stopping at <|im_end|>; returns generated tokens only."""

    def __init__(self, model, tok, device: str, max_new_tokens: dict, batch_size: int):
        import torch
        self.torch = torch
        self.model, self.tok, self.device = model, tok, device
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.eos_id = tok.convert_tokens_to_ids(IM_END)
        self.pad_id = tok.pad_token_id
        self.generation_seconds = 0.0
        self.batched_seconds = 0.0
        self.batched_requests = 0
        tok.padding_side = 'left'

    def _sync(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()

    def _generate(self, id_lists: list[list[int]], task: str) -> tuple[list[list[int]], float]:
        width = max(len(ids) for ids in id_lists)
        input_ids = self.torch.tensor([[self.pad_id] * (width - len(ids)) + ids for ids in id_lists],
                                      dtype=self.torch.long, device=self.device)
        attention = self.torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in id_lists],
                                      dtype=self.torch.long, device=self.device)
        self._sync()
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(input_ids=input_ids, attention_mask=attention,
                                      max_new_tokens=self.max_new_tokens[task], do_sample=False, num_beams=1,
                                      temperature=None, top_p=None, top_k=None,
                                      eos_token_id=self.eos_id, pad_token_id=self.pad_id, use_cache=True)
        self._sync()
        seconds = time.perf_counter() - t0
        self.generation_seconds += seconds
        gen = out[:, input_ids.shape[1]:].tolist()
        results = []
        for row in gen:   # the tokens before the end token: the output token count, and what decode_output decodes
            if self.eos_id in row:
                row = row[:row.index(self.eos_id)]
            results.append(row)
        return results, seconds

    def run(self, requests: list[dict]) -> None:
        """``requests``: dicts with ``ids`` (prompt token ids), ``task``, ``mode`` (batched|single).
        Fills ``output_ids`` and ``latency_s`` in place.  Batched requests are grouped by task
        and sorted by prompt length so a batch pads little; order of the list is irrelevant."""
        singles = [r for r in requests if r['mode'] == 'single']
        for r in singles:
            outs, seconds = self._generate([r['ids']], r['task'])
            r['output_ids'], r['latency_s'] = outs[0], seconds
        batched = [r for r in requests if r['mode'] == 'batched']
        for task in sorted({r['task'] for r in batched}):
            group = sorted((r for r in batched if r['task'] == task), key=lambda r: len(r['ids']))
            for start in range(0, len(group), self.batch_size):
                chunk = group[start:start + self.batch_size]
                outs, seconds = self._generate([r['ids'] for r in chunk], task)
                self.batched_seconds += seconds
                self.batched_requests += len(chunk)
                for r, o in zip(chunk, outs):
                    r['output_ids'], r['latency_s'] = o, seconds / len(chunk)


# ---------------------------------------------------------------------------
# Aggregation (pure functions; analyze.py and the self-tests reuse them)

def _rate(items, key):
    return (sum(1 for it in items if it[key]) / len(items)) if items else None


def panel_block(items: list[dict]) -> dict:
    return {'n': len(items), 'acc_first': _rate(items, 'correct_first'), 'acc_final': _rate(items, 'correct_final'),
            'invalid_first': _rate(items, 'invalid_first'), 'invalid_final': _rate(items, 'invalid_final'),
            'retry_rate': _rate(items, 'retried')}


def panel_stats(items: list[dict]) -> dict:
    """``{panel: {task: block, 'all': block}, 'all': {task: block, 'all': block}}``."""
    panels = sorted({it['panel'] for it in items}, key=lambda p: (common.TEST_PANELS + (p,)).index(p))
    tasks = sorted({it['task'] for it in items}, key=lambda t: (common.TASKS + (t,)).index(t))
    out = {}
    for panel in panels + ['all']:
        sub = [it for it in items if panel == 'all' or it['panel'] == panel]
        out[panel] = {task: panel_block([it for it in sub if it['task'] == task]) for task in tasks}
        out[panel]['all'] = panel_block(sub)
    return out


def token_stats(items: list[dict], system_tokens: int) -> dict:
    """Per-request means over every request (first + retry) and the two total conventions."""
    n_items = len(items)
    n_retries = sum(1 for it in items if it['retried'])
    n_requests = n_items + n_retries
    sums = {k: 0 for k in TOKEN_PARTS + ('output',)}
    for it in items:
        t = it['tokens']
        for k in TOKEN_PARTS:
            sums[k] += t[k]
        sums['output'] += it['tokens']['output_first']
        if it['retried']:
            sums['system'] += system_tokens
            sums['records'] += t['records']
            sums['instruction'] += t['prompt_retry'] - system_tokens - t['records'] - t['template']
            sums['template'] += t['template']
            sums['prompt_total'] += t['prompt_retry']
            sums['output'] += t['output_retry']
    uncached = sums['prompt_total'] + sums['output']
    cached = uncached - sums['system'] + system_tokens * math.ceil(n_requests / CACHE_WINDOW)
    return {'per_request_mean': {k: (sums[k] / n_requests if n_requests else None) for k in sums},
            'totals': {'uncached': uncached, 'cached': cached, 'output': sums['output'],
                       'requests': n_requests, 'retries': n_retries, 'system_per_request': system_tokens}}


def _percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _single_block(values: list[float]) -> dict:
    return {'n': len(values), 'mean_s': (sum(values) / len(values)) if values else None,
            'p50_s': _percentile(values, 0.5), 'p95_s': _percentile(values, 0.95)}


def latency_stats(single_first_latency: list[tuple[str, float]], gen: Generator) -> dict:
    """``single_first_latency``: (task, seconds) of every single-request first request; the block
    carries the distribution over all of them and per task (the sample is stratified by task)."""
    tasks = sorted({t for t, _ in single_first_latency}, key=lambda t: (common.TASKS + (t,)).index(t))
    single = {**_single_block([s for _, s in single_first_latency]),
              'by_task': {t: _single_block([s for task, s in single_first_latency if task == t]) for t in tasks}}
    return {'batched_items_per_s': (gen.batched_requests / gen.batched_seconds) if gen.batched_seconds else None,
            'batched_requests': gen.batched_requests, 'batched_seconds': gen.batched_seconds, 'single': single}


# ---------------------------------------------------------------------------
# Main

def main(argv=None) -> None:
    args = parse_args(argv)
    if args.threads:
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from grader import grade  # builder A's grader.py, next to this file
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    clock_total = Stopwatch()
    started_at = now_iso()

    config = load_config(args.config)
    fmt, arm, tasks = config['answer_format'], config['arm'], list(config['tasks'])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- model and tokenizer (excluded from the cost clock)
    tok = load_tokenizer()
    check_template(tok)
    device = resolve_device(args.device)
    dtype = resolve_dtype(config['dtype'], device)
    model_path = resolve_model_path(config)
    model = load_model(model_path, device, dtype, trainable=False)
    adapter = 'none' if args.adapter.lower() == 'none' else args.adapter
    if adapter != 'none':
        model = load_adapter(model, adapter)
    seconds_model_load = clock_total.seconds()
    print(f'[evaluate] model {model_path} adapter {adapter} device {device} dtype {dtype} '
          f'loaded in {seconds_model_load:.1f}s', file=sys.stderr)

    # --- data
    legend = load_legends(config['data_version'])[fmt][arm]
    system_tokens = count_tokens(tok, legend)
    rows = read_jsonl(examples_path(config['data_version'], args.split, arm))
    examples = select_examples(rows, tasks, args.limit)
    if not examples:
        raise SystemExit(f'no examples for arm {arm!r} tasks {tasks} in split {args.split!r}')
    single_idx = latency_sample_indices([ex['task'] for ex in examples], args.latency_sample)
    single_by_task = {t: sum(1 for i in single_idx if examples[i]['task'] == t) for t in tasks}
    print(f'[evaluate] {len(examples)} examples ({args.split}, {arm}, {fmt}); {len(single_idx)} single-request '
          f'{single_by_task}, the rest batched by {args.batch_size}', file=sys.stderr)

    clock_eval = Stopwatch()   # the cost clock: everything after model load
    gen = Generator(model, tok, device, config['max_new_tokens'], args.batch_size)
    dump = open(args.dump_outputs, 'w', encoding='utf-8') if args.dump_outputs else None

    # --- first requests: assemble + tokenise (tokenize_s), generate, decode, parse + grade (convert_out_s);
    #     convert_in_s is the codec time, the rendering wall-clock data.py records per example as render_s
    items, requests = [], []
    for i, ex in enumerate(examples):
        t0 = time.perf_counter()
        prompt = common.prompt_text(legend, user_text(ex, fmt))
        ids = tok(prompt, add_special_tokens=False)['input_ids']
        tokenize = time.perf_counter() - t0
        parts = token_breakdown(tok, legend, ex, fmt)
        assert parts['prompt_total'] == len(ids), (parts, len(ids))
        mode = 'single' if i in single_idx else 'batched'
        requests.append({'ids': ids, 'task': ex['task'], 'mode': mode, 'index': i})
        items.append({'id': ex['id'], 'panel': ex['panel'], 'task': ex['task'], 'k': ex['k'],
                      'max_id_len': ex['max_id_len'], 'n_negative': ex['n_negative'],
                      'heldout_wording': ex['heldout_wording'], 'heldout_pair': ex['heldout_pair'],
                      'correct_first': None, 'invalid_first': None, 'retried': False,
                      'correct_final': None, 'invalid_final': None, 'reason_first': None, 'reason_final': None,
                      'tokens': {**parts, 'output_first': 0, 'output_retry': 0, 'prompt_retry': 0},
                      'latency_s': 0.0, 'latency_mode': mode, 'convert_in_s': float(ex.get('render_s') or 0.0),
                      'tokenize_s': tokenize, 'convert_out_s': 0.0})

    # --- warm-up: one untimed generation on the first request, so CUDA/cuBLAS initialisation inflates
    #     neither the first single-request latency nor the batched throughput; the generation clock is reset
    t0 = time.perf_counter()
    gen._generate([requests[0]['ids']], requests[0]['task'])
    warmup_s = time.perf_counter() - t0
    gen.generation_seconds = 0.0
    print(f'[evaluate] warm-up generation {warmup_s:.2f}s (untimed; generation clock reset)', file=sys.stderr)
    gen.run(requests)

    def grade_request(req, ex, which):
        it = items[req['index']]
        text = common.decode_output(tok, req['output_ids'])   # model-side decoding, outside the parser clock
        t0 = time.perf_counter()
        result = grade(ex['task'], fmt, common.answer_text(ex, fmt), text)
        it['convert_out_s'] += time.perf_counter() - t0
        it['latency_s'] += req['latency_s']
        it['tokens'][f'output_{which}'] = len(req['output_ids'])
        if dump is not None:
            dump.write(json.dumps({'id': ex['id'], 'task': ex['task'], 'request': which, 'output': text,
                                          **result, 'output_tokens': len(req['output_ids'])}) + '\n')
        return result

    single_first_latency = []
    retry_requests = []
    for req in requests:
        ex, it = examples[req['index']], items[req['index']]
        r = grade_request(req, ex, 'first')
        it['correct_first'], it['invalid_first'], it['reason_first'] = bool(r['correct']), bool(r['invalid']), r['reason']
        it['correct_final'], it['invalid_final'], it['reason_final'] = it['correct_first'], it['invalid_first'], it['reason_first']
        if req['mode'] == 'single':
            single_first_latency.append((ex['task'], req['latency_s']))
        if it['invalid_first']:
            t0 = time.perf_counter()
            prompt = common.prompt_text(legend, retry_user_text(ex, fmt))
            ids = tok(prompt, add_special_tokens=False)['input_ids']
            it['tokenize_s'] += time.perf_counter() - t0   # the codec is not re-run for a retry
            it['retried'] = True
            it['tokens']['prompt_retry'] = len(ids)
            retry_requests.append({'ids': ids, 'task': ex['task'], 'mode': req['mode'], 'index': req['index']})

    # --- the single retry per invalid first answer (contract: retry policy)
    if retry_requests:
        print(f'[evaluate] {len(retry_requests)} retries', file=sys.stderr)
        gen.run(retry_requests)
        for req in retry_requests:
            ex, it = examples[req['index']], items[req['index']]
            r = grade_request(req, ex, 'retry')
            it['correct_final'], it['invalid_final'], it['reason_final'] = bool(r['correct']), bool(r['invalid']), r['reason']
    if dump is not None:
        dump.close()

    seconds_eval = clock_eval.seconds()

    # --- aggregates
    panels = panel_stats(items)
    tokens = token_stats(items, system_tokens)
    n_correct_final = sum(1 for it in items if it['correct_final'])
    dollars = args.price_per_hour * seconds_eval / 3600.0
    cost = {'price_per_hour': args.price_per_hour, 'gpu_seconds': seconds_eval,
            'seconds_total': seconds_eval, 'seconds_generation': gen.generation_seconds,
            'total': dollars, 'per_item': dollars / len(items),
            'per_correct_final': (dollars / n_correct_final) if n_correct_final else None}
    report = {
        'name': out_dir.name, 'config': config, 'adapter': adapter, 'split': args.split,
        'model_path': str(model_path), 'git_commit': git_commit(), 'device': device, 'dtype': str(dtype).replace('torch.', ''),
        'started_at': started_at, 'finished_at': now_iso(), 'seconds': clock_total.seconds(),
        'seconds_model_load': seconds_model_load, 'limit': args.limit,
        'generation': {'greedy': True, 'max_new_tokens': {t: config['max_new_tokens'][t] for t in tasks},
                       'batch_size': args.batch_size, 'stop': IM_END, 'padding': 'left', 'thinking': False,
                       'warmup_s': warmup_s},
        'n_items': len(items), 'panels': panels, 'tokens': tokens,
        'latency': latency_stats(single_first_latency, gen),
        'conversion': {'in_mean_s': sum(it['convert_in_s'] for it in items) / len(items),        # codec (render_s)
                       'tokenize_mean_s': sum(it['tokenize_s'] for it in items) / len(items),    # prompt assembly + tokeniser
                       'out_mean_s': sum(it['convert_out_s'] for it in items) / len(items)},     # parser + grader
        'cost': cost,
    }
    write_jsonl(out_dir / 'items.jsonl', items)
    write_json(out_dir / 'eval.json', report)
    a = panels['all']['all']
    print(f"[evaluate] {out_dir}: n={a['n']} acc_first={a['acc_first']:.3f} acc_final={a['acc_final']:.3f} "
          f"invalid_first={a['invalid_first']:.3f} retry_rate={a['retry_rate']:.3f} "
          f"tokens uncached={tokens['totals']['uncached']} cached={tokens['totals']['cached']} "
          f"eval {seconds_eval:.1f}s (generation {gen.generation_seconds:.1f}s)", file=sys.stderr)


if __name__ == '__main__':
    main()
