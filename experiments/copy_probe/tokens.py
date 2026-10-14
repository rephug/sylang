"""Token accounting for the deployment tokenizer (Qwen 3) on the language-comparison prompts.

    python -I tokens.py --source <packet> --tokenizer <tokenizer.json> \
        --out results/tokens_qwen3.json [--items-per-k 64] [--seed 11] [--example]

The byte model of ``language.py`` sees bytes; a deployed model sees sub-word tokens, so
byte savings and token savings are different quantities. This script renders the same
synthetic items (``language.ItemStream('test', seed, k_values=(K,))``, K = 1..8) in every
arm and counts, per arm and K, the bytes and tokens of each prompt section:

- ``header``: the Sylang batch header line (``SYLANG-BATCH<TAB>1<TAB>codes<TAB>selective<TAB>0<TAB>N``,
  35 bytes for N < 10); English has none.
- ``records``: all record lines, one per statement.
- ``instruction``: the fixed trailing instruction line (68 bytes), identical in every arm.
- ``prompt``: header + records + instruction, exactly ``Renderer.natural_prompt``.
- ``answer``: ``Renderer.target`` (the JSON list of ``[subject, object]`` of the negative records).

Arms: ``english_natural`` (controlled English, no filler), ``codes`` (current Sylang codes),
``words`` (Sylang words), and four compact variants derived from the codes rendering by text
transformation of its record lines (the instruction line is kept in all of them):

- ``compact_tab``: no header, records ``subj<TAB>code<TAB>obj``.
- ``compact_space``: no header, records ``subj code obj``.
- ``compact_comma``: no header, records ``subj,code,obj``.
- ``compact_count``: a one-line header holding only the record count (``3``), then tab records.

Sections are attributed from one tokenisation of the whole prompt using token offsets, so
the section counts always sum to the prompt count; a second, independent per-section
tokenisation is compared with it and any mismatch is reported (the pre-tokeniser never
merges across a newline, so none is expected). Identifier tokens are counted by tokenising
the identifier strings alone (``tokens(subject) + tokens(object)`` per record). Means are
reported per K, for the uniform K = 1..8 mix and for the K = 1..4 mix, with the break-even K
(first K at which an arm's mean prompt is cheaper than ``english_natural``) in tokens and in
bytes, both empirically and from a linear fit tokens(K) = a + b K. As a note only, the
tokens of a leaner answer format (``subj obj`` pairs, space separated, on one line) are
counted per negative record; it is not used anywhere else.

Only aggregate numbers are written to ``--out``. ``--example`` prints one synthetic item's
renderings to stdout (synthetic letter-digit-letter identifiers only); nothing from the
packet's own identifier sets is read or printed.
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import language  # noqa: E402
from followup import Runtime  # noqa: E402

BASE_ARMS = ('english_natural', 'codes', 'words')
COMPACT_SEPARATOR = {'compact_tab': '\t', 'compact_space': ' ', 'compact_comma': ',', 'compact_count': '\t'}
ARMS = BASE_ARMS + tuple(COMPACT_SEPARATOR)
DESCRIPTION = {
    'english_natural': 'controlled English, no filler (Renderer.natural_prompt)',
    'codes': 'current Sylang codes with the 35-byte batch header',
    'words': 'Sylang words with the 35-byte batch header',
    'compact_tab': 'codes records subj<TAB>code<TAB>obj, no header',
    'compact_space': 'codes records "subj code obj", no header',
    'compact_comma': 'codes records "subj,code,obj", no header',
    'compact_count': 'codes tab records with a one-line header holding only the record count',
}
SECTIONS = ('header', 'records', 'instruction', 'prompt', 'answer')
K_VALUES = tuple(range(1, 9))
MIXES = {'mix_1_8': tuple(range(1, 9)), 'mix_1_4': (1, 2, 3, 4)}


def _limit_threads() -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # torch is not needed here; the Runtime import may or may not pull it in
        pass


class Counter:
    """Byte and token counts with the deployment tokenizer."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer
        self.path = str(path)
        self.tokenizer = Tokenizer.from_file(self.path)
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]  # prefix: provenance only
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.cache: dict[str, int] = {}

    def tokens(self, text: str) -> int:
        if text == '':
            return 0
        if text not in self.cache:
            self.cache[text] = len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        return self.cache[text]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).offsets)


def render(renderer: language.Renderer, arm: str, statements: list[dict]) -> dict:
    """header text, record lines (each with its newline), instruction and the whole prompt."""
    if arm in BASE_ARMS:
        base = language.BASE_ARM.get(arm, arm)
        header_lines, record_lines = renderer.lines(base, statements)
        header = ''.join(h + '\n' for h in header_lines)
        records = [line + '\n' for line in record_lines]
        prompt = header + ''.join(records) + language.INSTRUCTION
        assert prompt == renderer.natural_prompt(arm, statements), arm
    else:
        _, record_lines = renderer.lines('codes', statements)
        fields = [line.split('\t') for line in record_lines]
        assert all(len(f) == 3 for f in fields), 'codes record lines are subject<TAB>code<TAB>object'
        separator = COMPACT_SEPARATOR[arm]
        records = [separator.join(f) + '\n' for f in fields]
        header = f'{len(statements)}\n' if arm == 'compact_count' else ''
        prompt = header + ''.join(records) + language.INSTRUCTION
    assert prompt.isascii()
    return {'header': header, 'records': records, 'instruction': language.INSTRUCTION, 'prompt': prompt}


def lean_answer(statements: list[dict]) -> str:
    """The leaner answer format noted in the task: 'subj obj' pairs, space separated, one line."""
    return ' '.join(f"{s['subject']} {s['object']}" for s in statements if s['polarity'] == 'negative')


def measure(counter: Counter, renderer: language.Renderer, arm: str, statements: list[dict]) -> dict:
    parts = render(renderer, arm, statements)
    prompt = parts['prompt']
    bounds = {'header': (0, len(parts['header']))}
    bounds['records'] = (bounds['header'][1], bounds['header'][1] + sum(len(r) for r in parts['records']))
    bounds['instruction'] = (bounds['records'][1], len(prompt))
    assert bounds['instruction'][1] - bounds['instruction'][0] == len(parts['instruction'])
    # Primary: one tokenisation of the whole prompt, each token attributed by its start offset.
    tokens = {'header': 0, 'records': 0, 'instruction': 0}
    for start, _ in counter.offsets(prompt):
        for section, (lo, hi) in bounds.items():
            if lo <= start < hi:
                tokens[section] += 1
                break
        else:
            raise AssertionError('token outside the prompt')
    tokens['prompt'] = sum(tokens.values())
    assert tokens['prompt'] == counter.tokens(prompt)
    # Independent check: tokenise each section on its own.
    standalone = {'header': counter.tokens(parts['header']),
                  'records': sum(counter.tokens(r) for r in parts['records']),
                  'instruction': counter.tokens(parts['instruction'])}
    mismatch = int(any(standalone[s] != tokens[s] for s in standalone))
    answer = renderer.target(statements)
    negatives = sum(s['polarity'] == 'negative' for s in statements)
    lean = lean_answer(statements)
    identifier_tokens = sum(counter.tokens(s['subject']) + counter.tokens(s['object']) for s in statements)
    identifier_bytes = sum(len(s['subject']) + len(s['object']) for s in statements)
    return {
        'k': len(statements),
        'bytes': {'header': len(parts['header']), 'records': sum(len(r) for r in parts['records']),
                  'instruction': len(parts['instruction']), 'prompt': len(prompt), 'answer': len(answer)},
        'tokens': dict(tokens, answer=counter.tokens(answer)),
        'identifier_tokens': identifier_tokens,
        'identifier_bytes': identifier_bytes,
        'negatives': negatives,
        'lean_answer_tokens': counter.tokens(lean),
        'lean_answer_bytes': len(lean),
        'section_mismatch': mismatch,
    }


def _mean(values) -> float:
    values = list(values)
    return round(statistics.fmean(values), 3) if values else None


def aggregate(rows: list[dict]) -> dict:
    """Means over items; records are per item, so per-record values divide by K."""
    out = {'n_items': len(rows),
           'bytes': {s: _mean(r['bytes'][s] for r in rows) for s in SECTIONS},
           'tokens': {s: _mean(r['tokens'][s] for r in rows) for s in SECTIONS}}
    out['tokens_per_record'] = _mean(r['tokens']['records'] / r['k'] for r in rows)
    out['bytes_per_record'] = _mean(r['bytes']['records'] / r['k'] for r in rows)
    out['identifier_tokens_per_record'] = _mean(r['identifier_tokens'] / r['k'] for r in rows)
    out['identifier_bytes_per_record'] = _mean(r['identifier_bytes'] / r['k'] for r in rows)
    out['identifier_token_fraction_of_records'] = _mean(r['identifier_tokens'] / r['tokens']['records'] for r in rows)
    out['identifier_byte_fraction_of_records'] = _mean(r['identifier_bytes'] / r['bytes']['records'] for r in rows)
    out['non_identifier_tokens_per_record'] = round(out['tokens_per_record'] - out['identifier_tokens_per_record'], 3)
    empty = [r for r in rows if r['negatives'] == 0]
    nonempty = [r for r in rows if r['negatives'] > 0]
    out['answer'] = {
        'n_empty': len(empty), 'n_nonempty': len(nonempty),
        'tokens_mean': _mean(r['tokens']['answer'] for r in rows),
        'tokens_empty_mean': _mean(r['tokens']['answer'] for r in empty),
        'tokens_nonempty_mean': _mean(r['tokens']['answer'] for r in nonempty),
        'bytes_nonempty_mean': _mean(r['bytes']['answer'] for r in nonempty),
        'negatives_mean': _mean(r['negatives'] for r in rows),
    }
    return out


def _fit(ks: list[int], ys: list[float]) -> tuple[float, float]:
    """Least-squares a + b*K."""
    mk, my = statistics.fmean(ks), statistics.fmean(ys)
    b = sum((k - mk) * (y - my) for k, y in zip(ks, ys)) / sum((k - mk) ** 2 for k in ks)
    return my - b * mk, b


def break_even(arms: dict, reference: str = 'english_natural') -> dict:
    """First K at which an arm's mean prompt is cheaper than the reference, in tokens and bytes."""
    out = {}
    for arm in arms:
        if arm == reference:
            continue
        out[arm] = {}
        for unit in ('tokens', 'bytes'):
            ref = [arms[reference]['by_k'][str(k)][unit]['prompt'] for k in K_VALUES]
            own = [arms[arm]['by_k'][str(k)][unit]['prompt'] for k in K_VALUES]
            margin = {str(k): round(r - o, 3) for k, r, o in zip(K_VALUES, ref, own)}  # > 0: arm cheaper
            first = next((k for k in K_VALUES if margin[str(k)] > 0), None)
            a_ref, b_ref = _fit(list(K_VALUES), ref)
            a_own, b_own = _fit(list(K_VALUES), own)
            fit_k = (a_own - a_ref) / (b_ref - b_own) if b_ref > b_own else None
            out[arm][unit] = {
                'first_k_cheaper': first,
                'cheaper_at_every_k': all(margin[str(k)] > 0 for k in K_VALUES),
                'linear_fit': {'reference_a_b': [round(a_ref, 3), round(b_ref, 3)],
                               'arm_a_b': [round(a_own, 3), round(b_own, 3)],
                               'break_even_k': round(fit_k, 3) if fit_k is not None else None},
                'margin_by_k_reference_minus_arm': margin,
            }
    return out


def markdown_table(arms: dict) -> str:
    head = ['arm'] + [f'K={k} tok' for k in K_VALUES] + ['mix1-8 tok', 'mix1-4 tok', 'mix1-8 bytes', 'answer tok']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for arm, report in arms.items():
        cells = [arm] + [f"{report['by_k'][str(k)]['tokens']['prompt']:.1f}" for k in K_VALUES]
        cells += [f"{report['mix_1_8']['tokens']['prompt']:.1f}", f"{report['mix_1_4']['tokens']['prompt']:.1f}",
                  f"{report['mix_1_8']['bytes']['prompt']:.1f}", f"{report['mix_1_8']['tokens']['answer']:.2f}"]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def per_record_table(arms: dict, breaks: dict) -> str:
    head = ['arm', 'header tok', 'header bytes', 'tok/record', 'bytes/record', 'id tok/record', 'id frac',
            'break-even K tok', 'break-even K bytes', 'fit K tok']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for arm, report in arms.items():
        mix = report['mix_1_8']
        b = breaks.get(arm)
        cells = [arm, f"{mix['tokens']['header']:.1f}", f"{mix['bytes']['header']:.1f}", f"{mix['tokens_per_record']:.2f}",
                 f"{mix['bytes_per_record']:.2f}", f"{mix['identifier_tokens_per_record']:.2f}",
                 f"{mix['identifier_token_fraction_of_records']:.2f}",
                 '-' if b is None else str(b['tokens']['first_k_cheaper']),
                 '-' if b is None else str(b['bytes']['first_k_cheaper']),
                 '-' if b is None or b['tokens']['linear_fit']['break_even_k'] is None
                 else f"{b['tokens']['linear_fit']['break_even_k']:.2f}"]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def print_example(counter: Counter, renderer: language.Renderer, seed: int) -> None:
    """One synthetic item (K=3) in every arm, with its token split; stdout only."""
    stream = language.ItemStream('test', seed + 1000, k_values=(3,))
    statements = stream.item(3)
    while not any(s['polarity'] == 'negative' for s in statements):
        statements = stream.item(3)
    print('\n## Example item (synthetic identifiers), K=3')
    for arm in ARMS:
        parts = render(renderer, arm, statements)
        row = measure(counter, renderer, arm, statements)
        print(f"\n### {arm}: {row['bytes']['prompt']} bytes, {row['tokens']['prompt']} tokens "
              f"(header {row['tokens']['header']}, records {row['tokens']['records']}, "
              f"instruction {row['tokens']['instruction']})")
        print(repr(parts['prompt']))
        encoded = counter.tokenizer.encode(parts['prompt'], add_special_tokens=False)
        print('tokens:', encoded.tokens)
    answer = renderer.target(statements)
    print(f"\n### answer: {repr(answer)} -> {counter.tokens(answer)} tokens; "
          f"lean {repr(lean_answer(statements))} -> {counter.tokens(lean_answer(statements))} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='packet directory (passed to followup.Runtime)')
    parser.add_argument('--tokenizer', required=True, help='tokenizer.json of the deployment tokenizer')
    parser.add_argument('--out', required=True, help='JSON report path')
    parser.add_argument('--items-per-k', type=int, default=64)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--example', action='store_true', help='print one synthetic item in every arm')
    args = parser.parse_args()

    _limit_threads()
    counter = Counter(args.tokenizer)
    renderer = language.Renderer(Runtime(args.source))

    items = {}
    for k in K_VALUES:
        stream = language.ItemStream('test', args.seed, k_values=(k,))
        items[k] = [stream.item(k) for _ in range(args.items_per_k)]
    rows = {arm: {k: [measure(counter, renderer, arm, it) for it in items[k]] for k in K_VALUES} for arm in ARMS}

    arms = {}
    for arm in ARMS:
        report = {'description': DESCRIPTION[arm], 'by_k': {str(k): aggregate(rows[arm][k]) for k in K_VALUES}}
        for name, ks in MIXES.items():
            report[name] = aggregate([r for k in ks for r in rows[arm][k]])
        arms[arm] = report
    breaks = break_even(arms)

    all_rows = [r for arm in ARMS for k in K_VALUES for r in rows[arm][k]]
    nonempty = [r for k in K_VALUES for r in rows['codes'][k] if r['negatives'] > 0]  # answers are arm independent
    sample = language.ItemStream('test', args.seed, k_values=(1,)).item(1)[0]
    lean_note = {
        'format': "space-separated 'subj obj' pairs on one line, e.g. 'h0z r5f' per negative record",
        'example_tokens': counter.tokens('h0z r5f'),
        'tokens_per_negative_record_mean': _mean(r['lean_answer_tokens'] / r['negatives'] for r in nonempty),
        'bytes_per_negative_record_mean': _mean(r['lean_answer_bytes'] / r['negatives'] for r in nonempty),
        'json_tokens_per_negative_record_mean': _mean(r['tokens']['answer'] / r['negatives'] for r in nonempty),
        'json_bytes_per_negative_record_mean': _mean(r['bytes']['answer'] / r['negatives'] for r in nonempty),
        'json_empty_answer_tokens': counter.tokens('[]'),
        'json_empty_answer_bytes': 2,
        'single_identifier_tokens': counter.tokens(sample['subject']),
        'n_nonempty_items': len(nonempty),
        'used_elsewhere': False,
    }
    report = {
        'tokenizer': {'name': 'Qwen3 (Qwen/Qwen3-0.6B tokenizer.json)', 'file': Path(counter.path).name,
                      'sha256_prefix16': counter.sha256, 'vocab_size': counter.vocab_size,
                      'add_special_tokens': False},
        'config': {'split': 'test', 'seed': args.seed, 'items_per_k': args.items_per_k, 'k_values': list(K_VALUES),
                   'mixes': {name: list(ks) for name, ks in MIXES.items()},
                   'instruction_bytes': len(language.INSTRUCTION),
                   'instruction_tokens': counter.tokens(language.INSTRUCTION),
                   'identical_items_across_arms': True},
        'arms': arms,
        'break_even_vs_english_natural': breaks,
        'notes': {
            'lean_answer_format': lean_note,
            'section_check': {'n_prompts': len(all_rows),
                              'standalone_vs_offset_mismatches': sum(r['section_mismatch'] for r in all_rows)},
            'gemma3': 'Gemma 3 tokenizer is gated on the Hub and was unavailable; only Qwen 3 is counted.',
            'content': 'aggregate numbers only; synthetic identifiers are not stored',
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    print(f"Qwen3 token accounting: {args.items_per_k} items per K, K=1..8, seed {args.seed}, "
          f"instruction {report['config']['instruction_tokens']} tokens / {len(language.INSTRUCTION)} bytes\n")
    print(markdown_table(arms))
    print('\nPer-record and break-even (mix K=1..8; break-even against english_natural, first K cheaper):\n')
    print(per_record_table(arms, breaks))
    print(f"\nlean answer note: {lean_note['tokens_per_negative_record_mean']} tokens per negative record "
          f"(JSON: {lean_note['json_tokens_per_negative_record_mean']}); '[]' = {lean_note['json_empty_answer_tokens']} token")
    print(f"section check mismatches: {report['notes']['section_check']['standalone_vs_offset_mismatches']} "
          f"of {len(all_rows)} prompts")
    if args.example:
        print_example(counter, renderer, args.seed)


if __name__ == '__main__':
    main()
