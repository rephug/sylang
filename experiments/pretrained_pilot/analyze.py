"""Analysis of the pilot's evaluation results (``interfaces.md``, section "Analysis";
decision rules in ``contract.md``).

    python analyze.py --results results/*/ --out analysis/ [--delta 3] [--resamples 2000] [--bootstrap-seed 0] [--level 0.90]

Every result directory (an ``eval.json`` with its ``items.jsonl``) is keyed by
``(model, arm, seed, train_size, answer_format, split)`` from its config; a zero-shot baseline
(adapter ``none``) has train size 0.  The model label is the config's ``model`` when set, else
the basename of the evaluation's ``model_path`` (the stage configs leave ``model`` null and
``PILOT_MODEL`` supplies the weights), else ``unknown``; every result record and the inventory
table carry the full ``model_path``, and the warning about superseded directories prints it.  Per (split, model, answer format, train size) the
contrasts ``arm − english`` (every other arm), ``sylang2g − sylang2`` and ``sylang2 − json`` are
computed on the paired per-item differences of ``correct_final`` (and, secondarily,
``correct_first``) for every panel and task including ``all``, over the seeds that both arms
share, on the items (ids) present in both arms in every shared seed.

Intervals (accuracy points):

* cluster bootstrap (primary): draw ``n_seeds`` seeds with replacement, then within each drawn
  seed draw its items with replacement; the statistic is the mean over the drawn seeds of the
  mean paired difference; ``--resamples`` resamples from a ``numpy.random.default_rng``
  seeded with ``--bootstrap-seed`` (a fresh generator per interval, so results do not depend
  on evaluation order); the interval is the central ``--level`` percentile range;
* per-item-only: every item averaged over the seeds, then items resampled with replacement.

Verdict under δ (``--delta``) on the cluster interval of ``correct_final``: superior (lower > +δ),
non-inferior (lower > −δ), inferior (upper < −δ), inconclusive otherwise.  The rule applies at
``MIN_DECISION_SEEDS`` (3, contract) shared seeds or more; with fewer the verdict is
``exploratory (1 seed): no decision`` or ``exploratory (2 seeds): no decision`` and the intervals
are printed as exploratory numbers.  The secondary ``correct_first`` contrasts carry intervals
only, no verdict.

Also written: accuracy tables per panel and task, learning curves (accuracy vs train size per
arm, mean over seeds with min/max), retry and invalid rates, token, latency, conversion and
cost tables, and an inventory of every result directory read.  Output: ``summary.md`` and
``summary.json`` under ``--out``; the settings (and any override of the defaults) are printed.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # python -I does not add the script's directory
    sys.path.insert(0, str(HERE))

import numpy as np

import common
from common import now_iso, read_jsonl, write_json

DEFAULTS = {'delta': 3.0, 'resamples': 2000, 'bootstrap_seed': 0, 'level': 0.90}
EXPLORATORY = 'exploratory (1 seed): no decision'
MIN_DECISION_SEEDS = 3   # contract: the δ rule applies at three seeds; fewer shared seeds take no decision
METRICS = ('correct_final', 'correct_first')
CHUNK = 200   # resamples per vectorised chunk (memory bound: CHUNK × seeds × items floats)


# ---------------------------------------------------------------------------
# Command line

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', nargs='+', required=True,
                   help='result directories (each holding eval.json), globs, or a parent directory of them')
    p.add_argument('--out', required=True, help='directory for summary.md and summary.json')
    p.add_argument('--delta', type=float, default=DEFAULTS['delta'], help='non-inferiority margin in accuracy points')
    p.add_argument('--resamples', type=int, default=DEFAULTS['resamples'])
    p.add_argument('--bootstrap-seed', type=int, default=DEFAULTS['bootstrap_seed'])
    p.add_argument('--level', type=float, default=DEFAULTS['level'], help='interval level, e.g. 0.90')
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Loading

def find_result_dirs(patterns) -> list[Path]:
    found = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for m in matches:
            path = Path(m)
            if (path / 'eval.json').is_file():
                found.append(path)
            elif path.is_dir():
                found.extend(sorted(p.parent for p in path.glob('*/eval.json')))
            else:
                print(f'[analyze] warning: {pattern!r} matches no result directory', file=sys.stderr)
    seen, out = set(), []
    for p in found:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            out.append(p)
    return out


def result_key(ev: dict) -> tuple:
    cfg = ev['config']
    train_size = 0 if str(ev.get('adapter', 'none')).lower() == 'none' else int(cfg['train_size'])
    # the stage configs leave `model` null (PILOT_MODEL supplies the weights): label by the weights' directory
    model = cfg.get('model') or Path(str(ev.get('model_path') or '')).name or 'unknown'
    return (model, cfg['arm'], int(cfg['seed']), train_size, cfg['answer_format'], ev['split'])


def load_results(dirs: list[Path]) -> list[dict]:
    results = []
    for d in dirs:
        with open(d / 'eval.json', encoding='utf-8') as f:
            ev = json.load(f)
        items = read_jsonl(d / 'items.jsonl') if (d / 'items.jsonl').is_file() else []
        model, arm, seed, train_size, fmt, split = result_key(ev)
        results.append({'path': str(d), 'name': ev.get('name', d.name), 'eval': ev, 'items': items,
                        'model': model, 'model_path': ev.get('model_path'), 'arm': arm, 'seed': seed, 'train_size': train_size,
                        'answer_format': fmt, 'split': split, 'adapter': ev.get('adapter'),
                        'n_items': ev.get('n_items', len(items)), 'started_at': ev.get('started_at'), 'used': True})
    by_key = defaultdict(list)
    for r in results:
        by_key[(r['model'], r['arm'], r['seed'], r['train_size'], r['answer_format'], r['split'])].append(r)
    for key, group in by_key.items():
        if len(group) > 1:
            group.sort(key=lambda r: (r['started_at'] or '', r['path']))
            for r in group[:-1]:
                r['used'] = False
            print(f'[analyze] warning: {len(group)} result directories share the key {key}; using the latest '
                  f'({group[-1]["path"]}, model_path {group[-1]["model_path"]}), ignoring '
                  + ', '.join(f'{r["path"]} (model_path {r["model_path"]})' for r in group[:-1]), file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Bootstrap and verdict

def percentile_interval(stats: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(stats, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


def cluster_bootstrap(D: np.ndarray, resamples: int, level: float, seed: int) -> tuple[float, float]:
    """``D``: seeds × items paired differences.  Seeds with replacement, then items within each
    drawn seed; statistic = mean over drawn seeds of the mean difference."""
    S, n = D.shape
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples)
    done = 0
    while done < resamples:
        r = min(CHUNK, resamples - done)
        seed_draw = rng.integers(0, S, size=(r, S))
        item_draw = rng.integers(0, n, size=(r, S, n))
        vals = D[seed_draw[:, :, None], item_draw]
        stats[done:done + r] = vals.mean(axis=2).mean(axis=1)
        done += r
    return percentile_interval(stats, level)


def item_bootstrap(d: np.ndarray, resamples: int, level: float, seed: int) -> tuple[float, float]:
    """``d``: per-item differences already averaged over seeds; items with replacement."""
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples)
    done = 0
    while done < resamples:
        r = min(CHUNK * 5, resamples - done)
        idx = rng.integers(0, n, size=(r, n))
        stats[done:done + r] = d[idx].mean(axis=1)
        done += r
    return percentile_interval(stats, level)


def verdict(lower: float, upper: float, delta: float, n_seeds: int) -> str:
    """The pre-registered decision on the cluster interval, taken only with ``MIN_DECISION_SEEDS``
    or more shared seeds (contract: the δ rule applies at three seeds); with fewer no decision is
    taken and the string says how many seeds there were."""
    if n_seeds < MIN_DECISION_SEEDS:
        return EXPLORATORY if n_seeds <= 1 else f'exploratory ({n_seeds} seeds): no decision'
    if lower > delta:
        return 'superior'
    if lower > -delta:
        return 'non-inferior'
    if upper < -delta:
        return 'inferior'
    return 'inconclusive'


# ---------------------------------------------------------------------------
# Contrasts

def contrast_pairs(arms: set[str]) -> list[tuple[str, str]]:
    pairs = [(arm, 'english') for arm in common.ARMS if arm != 'english' and arm in arms and 'english' in arms]
    for a, b in (('sylang2g', 'sylang2'), ('sylang2', 'json')):
        if a in arms and b in arms:
            pairs.append((a, b))
    return pairs


def subsets(items_by_id: dict) -> dict:
    """``{(panel, task): [ids]}`` including the ``all`` margins, in a fixed order."""
    panels = sorted({it['panel'] for it in items_by_id.values()}, key=lambda p: (common.TEST_PANELS + (p,)).index(p))
    tasks = sorted({it['task'] for it in items_by_id.values()}, key=lambda t: (common.TASKS + (t,)).index(t))
    out = {}
    for panel in panels + ['all']:
        for task in tasks + ['all']:
            out[(panel, task)] = [i for i, it in items_by_id.items()
                                  if (panel == 'all' or it['panel'] == panel) and (task == 'all' or it['task'] == task)]
    return out


def compute_contrasts(cells: dict, args) -> list[dict]:
    """``cells``: {(split, model, fmt, train_size, arm): {seed: result}}."""
    groups = defaultdict(set)
    for (split, model, fmt, train_size, arm) in cells:
        groups[(split, model, fmt, train_size)].add(arm)
    rows = []
    for group in sorted(groups):
        split, model, fmt, train_size = group
        for a, b in contrast_pairs(groups[group]):
            seeds_a, seeds_b = cells[(split, model, fmt, train_size, a)], cells[(split, model, fmt, train_size, b)]
            seeds = sorted(set(seeds_a) & set(seeds_b))
            if not seeds:
                print(f'[analyze] {group}: {a} − {b}: no shared seed ({sorted(seeds_a)} vs {sorted(seeds_b)}); skipped',
                      file=sys.stderr)
                continue
            per_seed = {}
            common_ids = None
            for s in seeds:
                ia = {it['id']: it for it in seeds_a[s]['items']}
                ib = {it['id']: it for it in seeds_b[s]['items']}
                ids = set(ia) & set(ib)
                if len(ids) != len(ia) or len(ids) != len(ib):
                    print(f'[analyze] {group} seed {s}: {a} − {b}: pairing by id keeps {len(ids)} of '
                          f'{len(ia)}/{len(ib)} items', file=sys.stderr)
                per_seed[s] = (ia, ib)
                common_ids = ids if common_ids is None else common_ids & ids
            if not common_ids:
                print(f'[analyze] {group}: {a} − {b}: no item shared by every seed; skipped', file=sys.stderr)
                continue
            ref = {i: per_seed[seeds[0]][0][i] for i in common_ids}
            for (panel, task), ids in subsets(ref).items():
                ids = sorted(ids)
                if not ids:
                    continue
                row = {'split': split, 'model': model, 'answer_format': fmt, 'train_size': train_size,
                       'contrast': f'{a} - {b}', 'a': a, 'b': b, 'panel': panel, 'task': task,
                       'n_seeds': len(seeds), 'seeds': seeds, 'n_items': len(ids)}
                for metric in METRICS:
                    D = np.array([[float(per_seed[s][0][i][metric]) - float(per_seed[s][1][i][metric]) for i in ids]
                                  for s in seeds])
                    acc_a = float(np.mean([[per_seed[s][0][i][metric] for i in ids] for s in seeds])) * 100
                    acc_b = float(np.mean([[per_seed[s][1][i][metric] for i in ids] for s in seeds])) * 100
                    point = float(D.mean(axis=1).mean()) * 100
                    c_lo, c_hi = cluster_bootstrap(D, args.resamples, args.level, args.bootstrap_seed)
                    i_lo, i_hi = item_bootstrap(D.mean(axis=0), args.resamples, args.level, args.bootstrap_seed)
                    row[metric] = {'acc_a': acc_a, 'acc_b': acc_b, 'diff_points': point,
                                   'cluster': {'lower': c_lo * 100, 'upper': c_hi * 100},
                                   'item': {'lower': i_lo * 100, 'upper': i_hi * 100}}
                    if metric == 'correct_final':   # the secondary metric carries intervals only, no verdict
                        row[metric]['verdict'] = verdict(c_lo * 100, c_hi * 100, args.delta, len(seeds))
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Tables from eval.json

def _mean(values):
    values = [v for v in values if v is not None]
    return (float(np.mean(values)) if values else None)


def accuracy_rows(cells: dict) -> list[dict]:
    rows = []
    for (split, model, fmt, train_size, arm), by_seed in sorted(cells.items()):
        panels = {}
        for r in by_seed.values():
            for panel, tasks in r['eval']['panels'].items():
                for task, block in tasks.items():
                    panels.setdefault((panel, task), []).append(block)
        for (panel, task), blocks in sorted(panels.items(), key=lambda kv: ((common.TEST_PANELS + ('all', kv[0][0])).index(kv[0][0]),
                                                                            (common.TASKS + ('all', kv[0][1])).index(kv[0][1]))):
            finals = [b['acc_final'] for b in blocks if b['acc_final'] is not None]
            rows.append({'split': split, 'model': model, 'answer_format': fmt, 'train_size': train_size, 'arm': arm,
                         'panel': panel, 'task': task, 'n_seeds': len(blocks), 'seeds': sorted(by_seed),
                         'n_items_mean': _mean([b['n'] for b in blocks]),
                         'acc_final_mean': _mean(finals) , 'acc_final_min': (min(finals) if finals else None),
                         'acc_final_max': (max(finals) if finals else None),
                         'acc_first_mean': _mean([b['acc_first'] for b in blocks]),
                         'invalid_first_mean': _mean([b['invalid_first'] for b in blocks]),
                         'invalid_final_mean': _mean([b['invalid_final'] for b in blocks]),
                         'retry_rate_mean': _mean([b['retry_rate'] for b in blocks])})
    return rows


def learning_curves(acc_rows: list[dict]) -> list[dict]:
    curves = defaultdict(list)
    for r in acc_rows:
        curves[(r['split'], r['model'], r['answer_format'], r['arm'], r['panel'], r['task'])].append(r)
    out = []
    for (split, model, fmt, arm, panel, task), pts in sorted(curves.items()):
        pts.sort(key=lambda r: r['train_size'])
        out.append({'split': split, 'model': model, 'answer_format': fmt, 'arm': arm, 'panel': panel, 'task': task,
                    'points': [{'train_size': r['train_size'], 'n_seeds': r['n_seeds'], 'acc_final_mean': r['acc_final_mean'],
                                'acc_final_min': r['acc_final_min'], 'acc_final_max': r['acc_final_max'],
                                'acc_first_mean': r['acc_first_mean']} for r in pts]})
    return out


def resource_rows(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        ev = r['eval']
        tok, lat, conv, cost = ev.get('tokens', {}), ev.get('latency', {}), ev.get('conversion', {}), ev.get('cost', {})
        prm, tot, single = tok.get('per_request_mean', {}), tok.get('totals', {}), lat.get('single') or {}
        rows.append({'name': r['name'], 'path': r['path'], 'split': r['split'], 'model': r['model'], 'arm': r['arm'],
                     'seed': r['seed'], 'train_size': r['train_size'], 'answer_format': r['answer_format'],
                     'n_items': r['n_items'], 'used': r['used'],
                     'tokens_system': prm.get('system'), 'tokens_records': prm.get('records'),
                     'tokens_instruction': prm.get('instruction'), 'tokens_template': prm.get('template'),
                     'tokens_prompt_total': prm.get('prompt_total'), 'tokens_output': prm.get('output'),
                     'total_uncached': tot.get('uncached'), 'total_cached': tot.get('cached'),
                     'total_output': tot.get('output'), 'requests': tot.get('requests'), 'retries': tot.get('retries'),
                     'batched_items_per_s': lat.get('batched_items_per_s'), 'single_n': single.get('n'),
                     'single_mean_s': single.get('mean_s'), 'single_p50_s': single.get('p50_s'),
                     'single_p95_s': single.get('p95_s'),
                     'convert_in_mean_s': conv.get('in_mean_s'), 'convert_out_mean_s': conv.get('out_mean_s'),
                     'price_per_hour': cost.get('price_per_hour'), 'gpu_seconds': cost.get('gpu_seconds'),
                     'seconds_generation': cost.get('seconds_generation'), 'cost_total': cost.get('total'),
                     'cost_per_item': cost.get('per_item'), 'cost_per_correct_final': cost.get('per_correct_final'),
                     'device': ev.get('device'), 'dtype': ev.get('dtype'), 'batch_size': ev.get('generation', {}).get('batch_size')})
    return rows


def resource_means(rows: list[dict]) -> list[dict]:
    numeric = [k for k in rows[0] if k not in ('name', 'path', 'split', 'model', 'arm', 'seed', 'train_size',
                                              'answer_format', 'used', 'device', 'dtype', 'batch_size')] if rows else []
    groups = defaultdict(list)
    for r in rows:
        if r['used']:
            groups[(r['split'], r['model'], r['answer_format'], r['train_size'], r['arm'])].append(r)
    out = []
    for (split, model, fmt, train_size, arm), rs in sorted(groups.items()):
        row = {'split': split, 'model': model, 'answer_format': fmt, 'train_size': train_size, 'arm': arm, 'n_seeds': len(rs)}
        for k in numeric:
            row[k] = _mean([r[k] for r in rs])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Markdown

def f_pct(x, digits=1):
    return '–' if x is None else f'{100 * x:.{digits}f}'


def f_pts(x, digits=1):
    return '–' if x is None else f'{x:+.{digits}f}'


def f_num(x, digits=1):
    if x is None:
        return '–'
    return f'{x:.{digits}f}' if isinstance(x, float) else str(x)


def md_table(headers, rows) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def group_title(split, model, fmt, train_size=None) -> str:
    size = '' if train_size is None else f', train_size={train_size}' + (' (zero-shot)' if train_size == 0 else '')
    return f'split={split}, model={model}, answer_format={fmt}{size}'


def contrast_table(rows: list[dict], metric: str, with_verdict: bool = True) -> str:
    headers = ['contrast', 'panel', 'task', 'seeds', 'items', 'acc A', 'acc B', 'diff (pts)',
               'cluster CI', 'per-item CI'] + (['verdict'] if with_verdict else [])
    body = []
    for r in rows:
        m = r[metric]
        body.append([r['contrast'], r['panel'], r['task'], r['n_seeds'], r['n_items'], f"{m['acc_a']:.1f}", f"{m['acc_b']:.1f}",
                     f_pts(m['diff_points']), f"[{m['cluster']['lower']:+.1f}, {m['cluster']['upper']:+.1f}]",
                     f"[{m['item']['lower']:+.1f}, {m['item']['upper']:+.1f}]"] + ([m['verdict']] if with_verdict else []))
    return md_table(headers, body)


def write_summary_md(path: Path, settings: dict, results: list[dict], acc_rows, contrasts, curves, res_rows, res_means):
    out = [f'# Pilot analysis summary\n\nGenerated {settings["generated_at"]}.\n']
    out.append('## Settings\n')
    out.append(md_table(['setting', 'value', 'default'],
                        [[k, settings[k], DEFAULTS[k]] for k in ('delta', 'resamples', 'level', 'bootstrap_seed')]))
    if settings['overrides']:
        out.append('\n**Overrides of the pre-registered defaults:** ' +
                   ', '.join(f'{k} = {v} (default {DEFAULTS[k]})' for k, v in settings['overrides'].items()) + '\n')
    out.append('\nVerdicts use the cluster-bootstrap interval of `correct_final` under δ: superior (lower > +δ), '
               'non-inferior (lower > −δ), inferior (upper < −δ), inconclusive otherwise; with fewer than three '
               f'shared seeds no decision is taken (`exploratory`; the rule applies at {MIN_DECISION_SEEDS} seeds, '
               'contract). Intervals are the central percentile range of the resamples, in accuracy points.\n')

    out.append('\n## Result directories\n')
    out.append(md_table(['name', 'split', 'model', 'model_path', 'arm', 'seed', 'train_size', 'answer_format', 'adapter', 'items',
                         'started', 'used'],
                        [[r['name'], r['split'], r['model'], r['model_path'], r['arm'], r['seed'], r['train_size'], r['answer_format'],
                          r['adapter'], r['n_items'], r['started_at'], 'yes' if r['used'] else 'no (superseded)'] for r in results]))

    out.append('\n## Contrasts and verdicts (`correct_final`, after the retry)\n')
    groups = defaultdict(list)
    for r in contrasts:
        groups[(r['split'], r['model'], r['answer_format'], r['train_size'])].append(r)
    if not groups:
        out.append('No contrast could be formed (an `english` arm or a shared seed is missing).\n')
    for key in sorted(groups):
        out.append(f'\n### {group_title(*key)}\n')
        out.append(contrast_table(groups[key], 'correct_final'))
    out.append('\n## Contrasts on `correct_first` (before the retry; secondary: intervals only, no verdict)\n')
    for key in sorted(groups):
        out.append(f'\n### {group_title(*key)}\n')
        out.append(contrast_table(groups[key], 'correct_first', with_verdict=False))

    out.append('\n## Accuracy per panel and task (`acc_final`, % mean over seeds; min–max when more than one seed)\n')
    acc_groups = defaultdict(list)
    for r in acc_rows:
        acc_groups[(r['split'], r['model'], r['answer_format'], r['train_size'])].append(r)
    for key in sorted(acc_groups):
        rows = acc_groups[key]
        cols = sorted({(r['panel'], r['task']) for r in rows},
                      key=lambda pt: ((common.TEST_PANELS + ('all', pt[0])).index(pt[0]), (common.TASKS + ('all', pt[1])).index(pt[1])))
        arms = sorted({r['arm'] for r in rows}, key=lambda a: (common.ARMS + (a,)).index(a))
        body = []
        for arm in arms:
            cell = {(r['panel'], r['task']): r for r in rows if r['arm'] == arm}
            line = [arm, next(iter(cell.values()))['n_seeds']]
            for pt in cols:
                r = cell.get(pt)
                if r is None or r['acc_final_mean'] is None:
                    line.append('–')
                elif r['n_seeds'] > 1:
                    line.append(f"{f_pct(r['acc_final_mean'])} ({f_pct(r['acc_final_min'])}–{f_pct(r['acc_final_max'])})")
                else:
                    line.append(f_pct(r['acc_final_mean']))
            body.append(line)
        out.append(f'\n### {group_title(*key)}\n')
        out.append(md_table(['arm', 'seeds'] + [f'{p}:{t}' for p, t in cols], body))

    out.append('\n## Learning curves (`acc_final` % vs train size; mean over seeds, min–max, seeds)\n')
    curve_groups = defaultdict(list)
    for c in curves:
        if c['panel'] == 'all' or c['task'] == 'all':
            curve_groups[(c['split'], c['model'], c['answer_format'], c['panel'], c['task'])].append(c)
    for key in sorted(curve_groups, key=lambda k: (k[0], k[1], k[2], (common.TEST_PANELS + ('all', k[3])).index(k[3]),
                                                   (common.TASKS + ('all', k[4])).index(k[4]))):
        cs = curve_groups[key]
        sizes = sorted({p['train_size'] for c in cs for p in c['points']})
        body = []
        for c in sorted(cs, key=lambda c: (common.ARMS + (c['arm'],)).index(c['arm'])):
            pts = {p['train_size']: p for p in c['points']}
            line = [c['arm']]
            for s in sizes:
                p = pts.get(s)
                line.append('–' if p is None else f"{f_pct(p['acc_final_mean'])} ({f_pct(p['acc_final_min'])}–{f_pct(p['acc_final_max'])}; {p['n_seeds']})")
            body.append(line)
        out.append(f'\n### {group_title(key[0], key[1], key[2])}, panel={key[3]}, task={key[4]}\n')
        out.append(md_table(['arm'] + [f'size {s}' for s in sizes], body))

    out.append('\n## Retry and invalid rates (% mean over seeds, all panels)\n')
    body = []
    for r in acc_rows:
        if r['panel'] == 'all':
            body.append([r['split'], r['model'], r['answer_format'], r['train_size'], r['arm'], r['task'], r['n_seeds'],
                         f_pct(r['retry_rate_mean']), f_pct(r['invalid_first_mean']), f_pct(r['invalid_final_mean']),
                         f_pct(r['acc_first_mean']), f_pct(r['acc_final_mean'])])
    out.append(md_table(['split', 'model', 'format', 'train_size', 'arm', 'task', 'seeds', 'retry rate', 'invalid first',
                         'invalid final', 'acc first', 'acc final'], body))

    out.append('\n## Tokens (mean per request over every request incl. retries; totals summed over the evaluation; mean over seeds)\n')
    out.append(md_table(['split', 'model', 'format', 'train_size', 'arm', 'seeds', 'system', 'records', 'instruction', 'template',
                         'prompt', 'output', 'total uncached', 'total cached', 'total output', 'requests', 'retries'],
                        [[r['split'], r['model'], r['answer_format'], r['train_size'], r['arm'], r['n_seeds'],
                          f_num(r['tokens_system']), f_num(r['tokens_records']), f_num(r['tokens_instruction']),
                          f_num(r['tokens_template']), f_num(r['tokens_prompt_total']), f_num(r['tokens_output']),
                          f_num(r['total_uncached'], 0), f_num(r['total_cached'], 0), f_num(r['total_output'], 0),
                          f_num(r['requests'], 0), f_num(r['retries'], 0)] for r in res_means]))

    out.append('\n## Latency and conversion (mean over seeds)\n')
    out.append(md_table(['split', 'model', 'format', 'train_size', 'arm', 'seeds', 'batched items/s', 'single n', 'single mean s',
                         'single p50 s', 'single p95 s', 'convert in ms', 'convert out ms'],
                        [[r['split'], r['model'], r['answer_format'], r['train_size'], r['arm'], r['n_seeds'],
                          f_num(r['batched_items_per_s'], 2), f_num(r['single_n'], 0), f_num(r['single_mean_s'], 3),
                          f_num(r['single_p50_s'], 3), f_num(r['single_p95_s'], 3),
                          f_num(None if r['convert_in_mean_s'] is None else 1000 * r['convert_in_mean_s'], 3),
                          f_num(None if r['convert_out_mean_s'] is None else 1000 * r['convert_out_mean_s'], 3)] for r in res_means]))

    out.append('\n## Cost (per evaluation; mean over seeds; `--price-per-hour` of each evaluation)\n')
    out.append(md_table(['split', 'model', 'format', 'train_size', 'arm', 'seeds', 'price/h', 'gpu seconds', 'generation seconds',
                         'total $', '$ per item', '$ per correct (final)'],
                        [[r['split'], r['model'], r['answer_format'], r['train_size'], r['arm'], r['n_seeds'],
                          f_num(r['price_per_hour'], 3), f_num(r['gpu_seconds'], 1), f_num(r['seconds_generation'], 1),
                          f_num(r['cost_total'], 5), f_num(r['cost_per_item'], 7), f_num(r['cost_per_correct_final'], 7)]
                         for r in res_means]))

    out.append('\n## Per-result resources (every result directory read)\n')
    out.append(md_table(['name', 'arm', 'seed', 'train_size', 'items', 'requests', 'retries', 'uncached', 'cached', 'output',
                         'batched items/s', 'gpu s', 'device', 'dtype', 'batch'],
                        [[r['name'], r['arm'], r['seed'], r['train_size'], r['n_items'], f_num(r['requests'], 0),
                          f_num(r['retries'], 0), f_num(r['total_uncached'], 0), f_num(r['total_cached'], 0),
                          f_num(r['total_output'], 0), f_num(r['batched_items_per_s'], 2), f_num(r['gpu_seconds'], 1),
                          r['device'], r['dtype'], r['batch_size']] for r in res_rows]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(out), encoding='utf-8')


# ---------------------------------------------------------------------------
# Main

def main(argv=None) -> None:
    args = parse_args(argv)
    settings = {'delta': args.delta, 'resamples': args.resamples, 'level': args.level, 'bootstrap_seed': args.bootstrap_seed,
                'generated_at': now_iso()}
    settings['overrides'] = {k: settings[k] for k in DEFAULTS if settings[k] != DEFAULTS[k]}
    print(f"[analyze] delta={args.delta} resamples={args.resamples} level={args.level} bootstrap_seed={args.bootstrap_seed}"
          + (f"  OVERRIDES of the defaults: {settings['overrides']}" if settings['overrides'] else '  (pre-registered defaults)'))

    dirs = find_result_dirs(args.results)
    if not dirs:
        raise SystemExit('no result directory found')
    results = load_results(dirs)
    cells: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in results:
        if r['used']:
            cells[(r['split'], r['model'], r['answer_format'], r['train_size'], r['arm'])][r['seed']] = r
    print(f'[analyze] {len(results)} result directories, {len(cells)} cells', file=sys.stderr)

    contrasts = compute_contrasts(cells, args)
    acc_rows = accuracy_rows(cells)
    curves = learning_curves(acc_rows)
    res_rows = resource_rows(results)
    res_means = resource_means(res_rows)

    out_dir = Path(args.out)
    summary = {'settings': settings, 'git_commit': common.git_commit(),
               'results': [{k: r[k] for k in ('name', 'path', 'split', 'model', 'model_path', 'arm', 'seed', 'train_size',
                                               'answer_format', 'adapter', 'n_items', 'started_at', 'used')} for r in results],
               'contrasts': contrasts, 'accuracy': acc_rows, 'learning_curves': curves,
               'resources': res_rows, 'resource_means': res_means}
    write_json(out_dir / 'summary.json', summary)
    write_summary_md(out_dir / 'summary.md', settings, results, acc_rows, contrasts, curves, res_rows, res_means)

    # console: the verdict table (task = all per panel, and per task over all panels)
    print('\nVerdicts (correct_final):')
    shown = [r for r in contrasts if r['task'] == 'all' or r['panel'] == 'all']
    if shown:
        header = f"{'group':<44} {'contrast':<20} {'panel':<8} {'task':<7} {'seeds':>5} {'items':>6} {'diff':>6} {'cluster CI':>16} {'item CI':>16}  verdict"
        print(header)
        for r in shown:
            m = r['correct_final']
            g = f"{r['split']}/{r['model']}/{r['answer_format']}/size {r['train_size']}"
            print(f"{g:<44} {r['contrast']:<20} {r['panel']:<8} {r['task']:<7} {r['n_seeds']:>5} {r['n_items']:>6} "
                  f"{m['diff_points']:>+6.1f} [{m['cluster']['lower']:+6.1f}, {m['cluster']['upper']:+6.1f}] "
                  f"[{m['item']['lower']:+6.1f}, {m['item']['upper']:+6.1f}]  {m['verdict']}")
    else:
        print('  none (no arm pair with an english baseline or a shared seed)')
    print(f'\n[analyze] wrote {out_dir / "summary.md"} and {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
