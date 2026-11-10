"""Deterministic generator of the compositional diagnostics benchmark.

    python -I generate.py                                  # data/cd-v0/ with the default config
    python -I generate.py --out DIR --version V --seed S --dev-families N --test-families M
    python -I generate.py --check [--out DIR --version V]  # regenerate in memory, compare every file hash

Splits are assigned to whole families before anything is rendered: each (split, question type) has its own
``random.Random("<version>:<seed>:<split>:<qtype>")`` stream and draws identifiers only from its split (dev
identifiers from the pilot's dev hash bucket, test identifiers from its test bucket). A family whose
structural signature (``schema.structural_signature``: invariant to renaming and reordering) matches any member
of any earlier family, in either split, is redrawn, so no near-duplicate crosses splits. Python's string
hashing is never used for iteration order, so the same seed and configuration reproduce identical bytes.

Files in ``<out>/<version>/``: ``families.jsonl`` (arm-independent families), ``<split>_<arm>.jsonl`` (one
rendered example per family member), ``legends.json``, ``manifest.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import families as F  # noqa: E402
import questions as Q  # noqa: E402
import render  # noqa: E402
import schema  # noqa: E402

VERSION = 'cd-v0'
DEFAULTS = {'seed': 0, 'dev_families': 40, 'test_families': 100}  # per question type
SPLITS = ('dev', 'test')
SOURCES = ('generate.py', 'families.py', 'questions.py', 'render.py', 'schema.py')
MAX_REDRAWS = 200


def generate_families(version: str, seed: int, dev_families: int, test_families: int) -> tuple[list[dict], dict]:
    sizes = {'dev': dev_families, 'test': test_families}
    seen: set = set()
    out, redraws = [], {}
    for split in SPLITS:
        for qtype in Q.QTYPES:
            rng = random.Random(f'{version}:{seed}:{split}:{qtype}')
            pool = schema.IdentifierPool(split, rng)
            schedule = F.design_schedule(qtype, sizes[split], random.Random(f'{version}:{seed}:{split}:{qtype}:design'))
            n_redraw = 0
            for index in range(sizes[split]):
                for _ in range(MAX_REDRAWS):
                    try:
                        family = F.build_family(qtype, rng, pool, schedule[index])
                    except F.Redraw:
                        n_redraw += 1
                        continue
                    signatures = {schema.structural_signature(m['records'], m['params']) for m in family['members']}
                    if signatures & seen:
                        n_redraw += 1
                        continue
                    seen |= signatures
                    break
                else:
                    raise RuntimeError(f'{split} {qtype}: {MAX_REDRAWS} redraws in a row')
                family['family'] = f'{split}-{qtype}-{index:04d}'
                family['split'] = split
                out.append(family)
            redraws[f'{split}:{qtype}'] = n_redraw
    return out, redraws


def positions(qtype: str, records: list[dict], params: dict) -> list[int]:
    """Indices of the records the answer is read from (the queried record; the matching records; both hops of
    every chain): used by the position audit."""
    if qtype in Q.PAIR_QTYPES:
        return [F.find(records, params['subject'], params['object'])]
    if qtype == 'filter':
        (f1, v1), (f2, v2) = params['conditions']
        return [i for i, r in enumerate(records) if r[f1] == v1 and r[f2] == v2]
    out = []
    for i, r in enumerate(records):
        if r['subject'] == params['x'] and r['relation'] == params['rel_a'] and r['polarity'] == 'positive' \
                and r['tense'] == params['tense_a']:
            for j, s in enumerate(records):
                if s['subject'] == r['object'] and s['relation'] == params['rel_b'] and s['polarity'] == 'positive':
                    out += [i, j]
    return sorted(set(out))


def example_rows(family: dict, arm: str) -> list[dict]:
    rows = []
    qtype = family['qtype']
    for m in family['members']:
        text = render.render(arm, m['records'])
        assert render.parse(arm, text) == m['records'], (family['family'], m['variant'], arm)
        value = json.loads(m['answer'])
        rows.append({
            'id': f'{family["family"]}-{m["variant"]}', 'family': family['family'], 'split': family['split'],
            'qtype': qtype, 'variant': m['variant'], 'expect': m['expect'], 'arm': arm,
            'records': text, 'instruction': Q.instruction(qtype, m['params']), 'answer': m['answer'],
            'params': m['params'], 'k': len(m['records']),
            'answer_size': len(value) if isinstance(value, list) else 1,
            'answer_positions': positions(qtype, m['records'], m['params']),
        })
        if 'mapping' in m:
            rows[-1]['mapping'] = m['mapping']
    return rows


def _jsonl(rows) -> bytes:
    buf = io.StringIO()
    for row in rows:
        buf.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
    return buf.getvalue().encode()


def build(version: str, seed: int, dev_families: int, test_families: int) -> tuple[dict, dict]:
    """``({filename: bytes}, manifest)``; nothing is written."""
    families, redraws = generate_families(version, seed, dev_families, test_families)
    files = {'families.jsonl': _jsonl(families),
             'legends.json': (json.dumps(render.LEGENDS, indent=1, sort_keys=True) + '\n').encode()}
    counts = {}
    for split in SPLITS:
        for arm in render.ARMS:
            rows = [row for fam in families if fam['split'] == split for row in example_rows(fam, arm)]
            name = f'{split}_{arm}.jsonl'
            files[name] = _jsonl(rows)
            counts[name] = len(rows)
    # identical answers, ids and instructions in every arm
    for split in SPLITS:
        keyed = {arm: [(r['id'], r['answer'], r['instruction']) for r in map(json.loads, files[f'{split}_{arm}.jsonl'].splitlines())]
                 for arm in render.ARMS}
        assert all(keyed[a] == keyed[render.ARMS[0]] for a in render.ARMS), split
    manifest = {
        'version': version,
        'config': {'seed': seed, 'dev_families_per_qtype': dev_families, 'test_families_per_qtype': test_families,
                   'qtypes': list(Q.QTYPES), 'variants': list(F.VARIANTS), 'arms': list(render.ARMS)},
        'families': {s: sum(f['split'] == s for f in families) for s in SPLITS},
        'counts': counts,
        'redraws': redraws,
        'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
        'generator_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCES},
        'codec_sha256': {'copy_probe/sylang_batch.py': hashlib.sha256(
            (HERE.parent / 'copy_probe' / 'sylang_batch.py').read_bytes()).hexdigest()},
    }
    return files, manifest


def write(out_dir: Path, files: dict, manifest: dict, force: bool) -> None:
    if (out_dir / 'manifest.json').exists() and not force:
        raise SystemExit(f'{out_dir}/manifest.json exists; pass --force to replace it (or choose --version)')
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--out', default=str(HERE / 'data'))
    p.add_argument('--version', default=VERSION)
    p.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    p.add_argument('--dev-families', type=int, default=DEFAULTS['dev_families'], help='per question type')
    p.add_argument('--test-families', type=int, default=DEFAULTS['test_families'], help='per question type')
    p.add_argument('--check', action='store_true', help='regenerate in memory and compare with the files on disk')
    p.add_argument('--force', action='store_true')
    args = p.parse_args(argv)
    out_dir = Path(args.out) / args.version
    if args.check:
        recorded = json.loads((out_dir / 'manifest.json').read_text())
        cfg = recorded['config']
        files, manifest = build(recorded['version'], cfg['seed'], cfg['dev_families_per_qtype'],
                                cfg['test_families_per_qtype'])
        problems = [n for n, h in manifest['sha256'].items() if recorded['sha256'].get(n) != h]
        problems += [n for n in manifest['sha256'] if (out_dir / n).exists()
                     and hashlib.sha256((out_dir / n).read_bytes()).hexdigest() != manifest['sha256'][n]]
        print(f'checked {len(manifest["sha256"])} files: ' + ('OK' if not problems else f'DIFFER: {sorted(set(problems))}'))
        return 1 if problems else 0
    files, manifest = build(args.version, args.seed, args.dev_families, args.test_families)
    write(out_dir, files, manifest, args.force)
    print(f'wrote {len(files) + 1} files to {out_dir}: {manifest["families"]} families, redraws {manifest["redraws"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
