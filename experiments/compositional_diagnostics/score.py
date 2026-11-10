"""Score model outputs on one split and arm.

    python -I score.py --data data/cd-v0 --split test --arm sylang2 --outputs outputs.jsonl [--out score.json]

``outputs.jsonl``: one line per example, ``{"id": ..., "output": "<the model's raw text>"}``; an example with no
line is scored as an empty (invalid) output. Prints the credited metrics with 95% family-cluster bootstrap
intervals, the uncredited diagnostics and the uniform-guess chance for the same items.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metrics  # noqa: E402


def load_rows(data_dir: Path, split: str, arm: str) -> list[dict]:
    return [json.loads(line) for line in (Path(data_dir) / f'{split}_{arm}.jsonl').read_text().splitlines()]


def score(rows: list[dict], outputs: dict, with_ci: bool = True) -> dict:
    fvals = metrics.family_values(rows, outputs)
    return {'summary': metrics.summarise(fvals, with_ci=with_ci),
            'chance': metrics.summarise(metrics.chance_family_values(rows), metrics=metrics.CREDITED, with_ci=False),
            'n_items': len(rows), 'n_missing_outputs': sum(r['id'] not in outputs for r in rows)}


def format_table(result: dict) -> str:
    s, c = result['summary'], result['chance']
    lines = [f'{"scope":9} ' + ' '.join(f'{m:>18}' for m in metrics.CREDITED)]
    for scope in s:
        cells = []
        for m in metrics.CREDITED:
            e = s[scope][m]
            ci = f'[{e["lo"]:.2f},{e["hi"]:.2f}]' if 'lo' in e else ''
            cells.append(f'{e["mean"]:.3f}{ci:>13}')
        lines.append(f'{scope:9} ' + ' '.join(f'{x:>18}' for x in cells))
        lines.append(f'{"  chance":9} ' + ' '.join(f'{c[scope][m]["mean"]:>18.2e}' for m in metrics.CREDITED))
    return '\n'.join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--data', default=str(HERE / 'data' / 'cd-v0'))
    p.add_argument('--split', default='test', choices=('dev', 'test'))
    p.add_argument('--arm', required=True, choices=('english', 'json', 'sylang2', 'sylang2g'))
    p.add_argument('--outputs', required=True)
    p.add_argument('--out')
    args = p.parse_args(argv)
    rows = load_rows(Path(args.data), args.split, args.arm)
    outputs = {}
    for line in Path(args.outputs).read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            outputs[item['id']] = item['output']
    result = score(rows, outputs)
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
