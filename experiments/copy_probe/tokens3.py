"""Qwen 3 token accounting for the design-v3 arms (``study3.py``).

    python -I tokens3.py --source <packet> --tokenizer <tokenizer.json> \
        --out results/tokens3_qwen3.json [--items-per-k 48] [--seed 11] [--example]

Same method as ``tokens.py`` (one tokenisation of the whole prompt, tokens attributed to
sections by their start offset, checked against a per-section tokenisation), applied to the
v3 items: identifiers of 2–5 characters from ``[a-z0-9]``, K = 1..16 (13..16 is the held-out
``long`` range), and the six request wordings (four trained, two held out) in equal shares.
Items come from ``study3.ItemStream3('test', ...)``: fresh synthetic identifiers only.

Arms: ``english`` (controlled English), ``sylang2`` (header-free Sylang v2), ``sylang1``
(Sylang v1 with its header line), ``json`` (compact JSON, defaults omitted) and ``csv``
(one row per record, every field spelled out). Sections: ``header`` (sylang1 only),
``records``, ``instruction`` (the item's wording), ``prompt`` (their sum) and ``answer``
(the JSON list of ``[subject, object]`` pairs; identical in every arm).

Reported per arm: means per K, for the uniform K = 1..12 and K = 1..16 mixes, per record,
by identifier length (the item's longest identifier) and by wording; the per-K margin against
``english`` and the first K at which each arm is cheaper. Only aggregates are written.
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import language  # noqa: E402
import study3  # noqa: E402
from followup import Runtime  # noqa: E402
from tokens import Counter, _fit, _limit_threads, _mean  # noqa: E402

ARMS = study3.ARMS3
K_VALUES = tuple(range(1, 17))
MIXES = {'mix_1_12': tuple(range(1, 13)), 'mix_1_16': K_VALUES, 'mix_13_16': study3.K_LONG}
SECTIONS = ('header', 'records', 'instruction', 'prompt', 'answer')
ALL_WORDINGS = study3.WORDINGS['train'] + study3.WORDINGS['heldout']


def sections(renderer: language.Renderer, arm: str, item) -> dict:
    prompt = study3.render(renderer, arm, item)
    instruction = item.instruction
    body = prompt[:len(prompt) - len(instruction)]
    header = body.split('\n', 1)[0] + '\n' if arm == 'sylang1' else ''
    assert prompt == header + body[len(header):] + instruction
    return {'header': header, 'records': body[len(header):], 'instruction': instruction, 'prompt': prompt}


def measure(counter: Counter, renderer: language.Renderer, arm: str, item) -> dict:
    parts = sections(renderer, arm, item)
    prompt = parts['prompt']
    bounds, pos = {}, 0
    for name in ('header', 'records', 'instruction'):
        bounds[name] = (pos, pos + len(parts[name]))
        pos += len(parts[name])
    assert pos == len(prompt)
    tokens = {'header': 0, 'records': 0, 'instruction': 0}
    for start, _ in counter.offsets(prompt):
        for name, (lo, hi) in bounds.items():
            if lo <= start < hi:
                tokens[name] += 1
                break
        else:
            raise AssertionError('token outside the prompt')
    tokens['prompt'] = sum(tokens.values())
    assert tokens['prompt'] == counter.tokens(prompt)
    standalone = {name: counter.tokens(parts[name]) for name in ('header', 'records', 'instruction')}
    answer = renderer.target(list(item))
    statements = list(item)
    return {
        'k': len(statements),
        'longest_id': study3.longest_identifier(statements),
        'wording': ALL_WORDINGS.index(item.instruction),
        'bytes': {'header': len(parts['header']), 'records': len(parts['records']),
                  'instruction': len(parts['instruction']), 'prompt': len(prompt), 'answer': len(answer)},
        'tokens': dict(tokens, answer=counter.tokens(answer)),
        'identifier_tokens': sum(counter.tokens(s['subject']) + counter.tokens(s['object']) for s in statements),
        'identifier_bytes': sum(len(s['subject']) + len(s['object']) for s in statements),
        'negatives': sum(s['polarity'] == 'negative' for s in statements),
        'section_mismatch': int(any(standalone[s] != tokens[s] for s in standalone)),
    }


def aggregate(rows: list[dict]) -> dict:
    out = {'n_items': len(rows),
           'bytes': {s: _mean(r['bytes'][s] for r in rows) for s in SECTIONS},
           'tokens': {s: _mean(r['tokens'][s] for r in rows) for s in SECTIONS}}
    out['tokens_per_record'] = _mean(r['tokens']['records'] / r['k'] for r in rows)
    out['bytes_per_record'] = _mean(r['bytes']['records'] / r['k'] for r in rows)
    out['identifier_tokens_per_record'] = _mean(r['identifier_tokens'] / r['k'] for r in rows)
    out['identifier_token_fraction_of_records'] = _mean(r['identifier_tokens'] / r['tokens']['records'] for r in rows)
    out['answer_tokens_nonempty'] = _mean(r['tokens']['answer'] for r in rows if r['negatives'])
    return out


def items(source: str, seed: int, per_k: int) -> list:
    """Test-split items: for each K, ``per_k`` items cycling through the six wordings."""
    stream = study3.ItemStream3('test', seed, k_values=K_VALUES, wordings=ALL_WORDINGS)
    out = []
    for k in K_VALUES:
        for i in range(per_k):
            out.append(stream.item(k, ALL_WORDINGS[i % len(ALL_WORDINGS)]))
    return out


def report(counter: Counter, renderer: language.Renderer, panel: list) -> dict:
    arms = {}
    for arm in ARMS:
        rows = [measure(counter, renderer, arm, it) for it in panel]
        by_k = {str(k): aggregate([r for r in rows if r['k'] == k]) for k in K_VALUES}
        arms[arm] = {
            'by_k': by_k,
            **{name: aggregate([r for r in rows if r['k'] in ks]) for name, ks in MIXES.items()},
            'by_longest_id': {str(n): aggregate([r for r in rows if r['longest_id'] == n])
                              for n in sorted({r['longest_id'] for r in rows})},
            'by_wording': {str(w): {'instruction_tokens': _mean(r['tokens']['instruction'] for r in rows if r['wording'] == w),
                                    'instruction_bytes': _mean(r['bytes']['instruction'] for r in rows if r['wording'] == w),
                                    'held_out': w >= len(study3.WORDINGS['train'])}
                           for w in range(len(ALL_WORDINGS))},
            'section_mismatches': sum(r['section_mismatch'] for r in rows),
        }
    margins = {}
    for arm in ARMS:
        if arm == 'english':
            continue
        ref = [arms['english']['by_k'][str(k)]['tokens']['prompt'] for k in K_VALUES]
        own = [arms[arm]['by_k'][str(k)]['tokens']['prompt'] for k in K_VALUES]
        margin = {str(k): round(r - o, 3) for k, r, o in zip(K_VALUES, ref, own)}
        a_ref, b_ref = _fit(list(K_VALUES), ref)
        a_own, b_own = _fit(list(K_VALUES), own)
        margins[arm] = {'first_k_cheaper_than_english': next((k for k in K_VALUES if margin[str(k)] > 0), None),
                        'margin_by_k_english_minus_arm': margin,
                        'ratio_mix_1_12': round(arms[arm]['mix_1_12']['tokens']['prompt']
                                                / arms['english']['mix_1_12']['tokens']['prompt'], 4),
                        'ratio_mix_1_16': round(arms[arm]['mix_1_16']['tokens']['prompt']
                                                / arms['english']['mix_1_16']['tokens']['prompt'], 4),
                        'linear_fit_tokens_per_record': {'english': round(b_ref, 3), arm: round(b_own, 3)}}
    return {'arms': arms, 'against_english': margins}


def markdown(rep: dict) -> str:
    arms = rep['arms']
    head = ['arm', 'K=1', 'K=4', 'K=8', 'K=12', 'K=16', 'mix 1–12 tok', 'mix 1–16 tok', 'ratio vs english (1–12)',
            'tok/record', 'id tok/record', 'instruction tok', 'bytes (1–12)', 'answer tok (non-empty)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for arm, r in arms.items():
        ratio = rep['against_english'].get(arm, {}).get('ratio_mix_1_12', 1.0)
        lines.append('| ' + ' | '.join([
            arm, *(f"{r['by_k'][str(k)]['tokens']['prompt']:.1f}" for k in (1, 4, 8, 12, 16)),
            f"{r['mix_1_12']['tokens']['prompt']:.1f}", f"{r['mix_1_16']['tokens']['prompt']:.1f}", f"{ratio:.2f}",
            f"{r['mix_1_12']['tokens_per_record']:.2f}", f"{r['mix_1_12']['identifier_tokens_per_record']:.2f}",
            f"{r['mix_1_12']['tokens']['instruction']:.1f}", f"{r['mix_1_12']['bytes']['prompt']:.0f}",
            f"{r['mix_1_12']['answer_tokens_nonempty']:.1f}"]) + ' |')
    lines.append('')
    lines.append('Tokens per record by the item\'s longest identifier (mix 1–16):')
    lengths = sorted(next(iter(arms.values()))['by_longest_id'])
    lines.append('| arm | ' + ' | '.join(f'longest id {n}' for n in lengths) + ' |')
    lines.append('|---|' + '---|' * len(lengths))
    for arm, r in arms.items():
        lines.append(f'| {arm} | ' + ' | '.join(f"{r['by_longest_id'][n]['tokens_per_record']:.2f}" for n in lengths) + ' |')
    return '\n'.join(lines)


def print_example(counter: Counter, renderer: language.Renderer, seed: int) -> None:
    stream = study3.ItemStream3('test', seed + 1000, k_values=(4,))
    item = stream.item(4, study3.WORDINGS['train'][1])
    while not any(s['polarity'] == 'negative' for s in item):
        item = stream.item(4, study3.WORDINGS['train'][1])
    print('\n## Example item (synthetic identifiers), K=4')
    for arm in ARMS:
        row = measure(counter, renderer, arm, item)
        print(f"\n### {arm}: {row['bytes']['prompt']} bytes, {row['tokens']['prompt']} tokens "
              f"(header {row['tokens']['header']}, records {row['tokens']['records']}, "
              f"instruction {row['tokens']['instruction']})")
        print(repr(study3.render(renderer, arm, item)))
    answer = renderer.target(list(item))
    print(f"\n### answer: {repr(answer)} -> {counter.tokens(answer)} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--items-per-k', type=int, default=48)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args()
    _limit_threads()
    counter = Counter(args.tokenizer)
    renderer = language.Renderer(Runtime(args.source))
    panel = items(args.source, args.seed, args.items_per_k)
    rep = report(counter, renderer, panel)
    rep['config'] = {'tokenizer_sha256_prefix': counter.sha256, 'vocab_size': counter.vocab_size,
                     'items_per_k': args.items_per_k, 'seed': args.seed, 'k_values': list(K_VALUES),
                     'wordings': len(ALL_WORDINGS), 'identifier_lengths': list(study3.ID_LENGTHS)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n')
    print(markdown(rep))
    if args.example:
        print_example(counter, renderer, args.seed)


if __name__ == '__main__':
    main()
