#!/usr/bin/env python
"""Generate every config and plan file of the pilot from one table, so they stay consistent.

    python configs/gen_configs.py                      # rewrite configs/*.json and the plan files
    python configs/gen_configs.py --stage-c-train-size 8000   # the 8,000-case stage-C variant (§3.3)
    python configs/gen_configs.py --stage-c-epochs 3          # the 3-epoch stage-C variant
    python configs/gen_configs.py --batch-size 8 --grad-accum 2   # same effective batch, half the memory

Cells: stage A (the json-arm profile, the longest rendering), stage B (four arms x seed 0 x 2,000
cases, optional 500/8,000 curve points for english and sylang2), stage C (seeds 1-2 of every arm,
the bare answer format for english and sylang2 x seeds 0-2), stage D (the Qwen3-1.7B extension:
sylang2 and json x seeds 0-2, run only under the capacity trigger of contract.md), the CPU smoke
configs (tiny random model) and the real-weights CPU smoke configs. Every config is loaded back
with common.load_config and its recipe values are asserted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
if str(PILOT) not in sys.path:
    sys.path.insert(0, str(PILOT))
import common  # noqa: E402

TEST_CASES = 1000                 # contract: 1,000 test cases shared by every seed
TASKS = len(common.TASKS)         # two tasks -> two examples per case
EVAL_ITEMS = TEST_CASES * TASKS   # 2,000 eval items per evaluation

GPU_RECIPE = {                    # the one recipe shared by every GPU cell
    'model': None,                # null: take $PILOT_MODEL on the rented machine
    'data_version': 'v1',
    'epochs': 2,
    'batch_size': 16,
    'grad_accum': 1,
    'lr': 2e-4,
    'max_length': 1024,
    'dev_eval_examples': 200,
    'timeout_minutes': 90,
}
SMOKE_RECIPE = {'model': 'tiny-random', 'stage': 'smoke', 'data_version': 'smoke', 'train_size': 64, 'seed': 0,
                'batch_size': 4, 'max_steps': 20, 'epochs': 1, 'dev_eval_examples': 8, 'max_length': 512,
                'timeout_minutes': 20}
SMOKE_REAL_RECIPE = {'model': None, 'stage': 'smoke-real', 'data_version': 'smoke', 'train_size': 64, 'seed': 0,
                     'batch_size': 1, 'grad_accum': 4, 'max_steps': 20, 'epochs': 1, 'dev_eval_examples': 8,
                     'max_length': 512, 'timeout_minutes': 120, 'lr': 2e-4}


def cell(stage: str, arm: str, seed: int, train_size: int, answer_format: str = 'json', prefix: str | None = None,
         recipe: dict | None = None) -> dict:
    fmt = '' if answer_format == 'json' else f'_{answer_format}'
    name = f"stage_{prefix or stage.lower()}_{arm}{fmt}_s{seed}_n{train_size}"
    cfg = {'name': name, 'stage': stage, 'arm': arm, 'answer_format': answer_format, 'train_size': train_size, 'seed': seed}
    cfg.update(recipe or GPU_RECIPE)
    return cfg


def plan_entry(cfg: dict, optional: bool = False, note: str | None = None) -> dict:
    steps_per_epoch = -(-cfg['train_size'] * TASKS // (cfg['batch_size'] * cfg.get('grad_accum', 1)))
    entry = {'name': cfg['name'], 'config': f"configs/{cfg['name']}.json", 'arm': cfg['arm'],
             'answer_format': cfg['answer_format'], 'seed': cfg['seed'], 'train_size': cfg['train_size'],
             'train_cases': cfg['train_size'], 'train_examples': cfg['train_size'] * TASKS, 'epochs': cfg['epochs'],
             'batch_size': cfg['batch_size'], 'max_length': cfg['max_length'],
             'optimizer_steps': steps_per_epoch * cfg['epochs'], 'dev_evals': cfg['epochs'],
             'dev_eval_items': cfg['dev_eval_examples'], 'eval_items': EVAL_ITEMS,
             'timeout_minutes': cfg['timeout_minutes'], 'optional': optional}
    if note:
        entry['note'] = note
    return entry


def zero_shot_entry(cfg: dict, prefix: str = 'zeroshot') -> dict:
    return {'name': f"{prefix}_{cfg['arm']}_{cfg['answer_format']}", 'config': f"configs/{cfg['name']}.json",
            'arm': cfg['arm'], 'answer_format': cfg['answer_format'], 'adapter': 'none', 'eval_items': EVAL_ITEMS}


def totals(entries: list[dict], zero_shots: list[dict]) -> dict:
    return {'cells': len(entries), 'train_examples': sum(e['train_examples'] for e in entries),
            'train_example_passes': sum(e['train_examples'] * e['epochs'] for e in entries),
            'optimizer_steps': sum(e['optimizer_steps'] for e in entries),
            'dev_eval_items': sum(e['dev_evals'] * e['dev_eval_items'] for e in entries),
            'eval_items': sum(e['eval_items'] for e in entries) + sum(z['eval_items'] for z in zero_shots)}


def plan(stage: str, cells: list, zero_shots: list[dict], description: str) -> dict:
    entries = [plan_entry(c, opt, note) for c, opt, note in cells]
    required = [e for e in entries if not e['optional']]
    return {'stage': stage, 'description': description,
            'format': ('cells[]: one training cell each: train_cases (nested prefix of the seed stream), '
                       'train_examples (= train_cases x 2 tasks), epochs, batch_size, optimizer_steps, '
                       'dev_evals (= epochs; one dev generation pass per epoch end), dev_eval_items, '
                       'eval_items (two tasks x 1,000 test cases), timeout_minutes, optional (true: run only '
                       'when STAGE_B_CURVES=1). zero_shot[]: generation-only baselines of eval_items each. '
                       'totals: sums over the required cells; totals_with_optional: over every cell.'),
            'test_cases': TEST_CASES, 'tasks': list(common.TASKS), 'cells': entries, 'zero_shot': zero_shots,
            'totals': totals(required, zero_shots), 'totals_with_optional': totals(entries, zero_shots)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage-c-train-size', type=int, default=2000, help='cases per stage-C cell (default 2000; §3.3 variant 8000)')
    p.add_argument('--stage-c-epochs', type=int, default=2, help='epochs per stage-C cell (default 2; §3.5 variant 3)')
    p.add_argument('--batch-size', type=int, default=GPU_RECIPE['batch_size'], help='micro-batch of every GPU cell')
    p.add_argument('--grad-accum', type=int, default=GPU_RECIPE['grad_accum'], help='accumulation steps (effective batch = batch x accum)')
    p.add_argument('--out', default=str(HERE), help='configs directory')
    args = p.parse_args(argv)
    GPU_RECIPE['batch_size'], GPU_RECIPE['grad_accum'] = args.batch_size, args.grad_accum
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    configs: dict[str, dict] = {}

    for arm in ('sylang2', 'english'):                      # CPU smokes
        configs[f'smoke_{arm}'] = {'name': f'smoke_{arm}', 'arm': arm, **SMOKE_RECIPE}
        configs[f'smoke_real_{arm}'] = {'name': f'smoke_real_{arm}', 'arm': arm, **SMOKE_REAL_RECIPE}

    cfg = cell('A', 'json', 0, 2000); cfg['name'] = 'stage_a_profile'   # stage A: the longest rendering
    configs['stage_a_profile'] = cfg

    b_cells = []
    for arm in common.ARMS:
        c = cell('B', arm, 0, 2000); configs[c['name']] = c; b_cells.append((c, False, None))
    for arm in ('english', 'sylang2'):
        for n in (500, 8000):
            c = cell('B', arm, 0, n); configs[c['name']] = c; b_cells.append((c, True, 'learning curve; run when STAGE_B_CURVES=1'))
    b_zero = [zero_shot_entry(configs[f'stage_b_{arm}_s0_n2000']) for arm in common.ARMS]

    c_recipe = dict(GPU_RECIPE, epochs=args.stage_c_epochs)
    c_cells = []
    for arm in common.ARMS:
        for seed in (1, 2):
            c = cell('C', arm, seed, args.stage_c_train_size, recipe=c_recipe); configs[c['name']] = c
            c_cells.append((c, False, 'seed 0 of this arm is stage B'))
    for arm in ('english', 'sylang2'):
        for seed in (0, 1, 2):
            c = cell('C', arm, seed, args.stage_c_train_size, answer_format='bare', recipe=c_recipe); configs[c['name']] = c
            c_cells.append((c, False, 'answer-format factor (experiment only; never adopted)'))
    c_zero = [zero_shot_entry(configs[f'stage_c_{arm}_bare_s0_n{args.stage_c_train_size}']) for arm in ('english', 'sylang2')]

    d_cells = []
    for arm in ('sylang2', 'json'):
        for seed in (0, 1, 2):
            c = cell('D-1.7B', arm, seed, 2000, prefix='d_1p7b'); configs[c['name']] = c
            d_cells.append((c, False, '1.7B extension (contract: capacity question); PILOT_MODEL must point at the Qwen3-1.7B weights'))
    d_zero = [zero_shot_entry(configs[f'stage_d_1p7b_{arm}_s0_n2000'], prefix='zeroshot_1p7b') for arm in ('sylang2', 'json')]

    for name, cfg in configs.items():
        assert cfg['name'] == name
        common.write_json(out / f'{name}.json', cfg)
    common.write_json(out / 'plan_stage_b.json', plan('B', b_cells, b_zero,
        'Stage B: four arms at seed 0 and 2,000 training cases with the zero-shot baseline per arm; '
        'optionally the english and sylang2 learning-curve points at 500 and 8,000 cases.'))
    common.write_json(out / 'plan_stage_c.json', plan('C', c_cells, c_zero,
        f'Stage C: seeds 1 and 2 of every arm at {args.stage_c_train_size:,} cases and {args.stage_c_epochs} epochs '
        '(seed 0 comes from stage B), and the bare answer format for english and sylang2 at seeds 0-2; '
        'zero-shot baselines for the bare format.'))
    common.write_json(out / 'plan_stage_d_1p7b.json', plan('D-1.7B', d_cells, d_zero,
        'Qwen3-1.7B extension, run only under the capacity trigger of contract.md: sylang2 and json x seeds 0-2 x '
        '2,000 cases, same recipe as stage C. Project it with profile_gpu.py --plan stage_d_1p7b=plan_stage_d_1p7b.json '
        'after profiling the 1.7B weights.'))

    for path in sorted(out.glob('*.json')):                 # every config loads and carries the recipe
        if path.name.startswith('plan_'):
            json.loads(path.read_text()); continue
        loaded = common.load_config(path)
        assert loaded['name'] == path.stem, (path, loaded['name'])
        if path.stem.startswith('stage_'):
            assert loaded['model'] is None and loaded['data_version'] == 'v1' and loaded['lr'] == 2e-4
            assert loaded['batch_size'] == args.batch_size and loaded['grad_accum'] == args.grad_accum
            assert loaded['max_length'] == 1024 and loaded['dev_eval_examples'] == 200 and loaded['max_steps'] is None
        elif path.stem.startswith('smoke_real_'):
            assert loaded['model'] is None and loaded['data_version'] == 'smoke' and loaded['batch_size'] == 1
        else:
            assert loaded['model'] == 'tiny-random' and loaded['data_version'] == 'smoke' and loaded['max_steps'] == 20
    print(f'{len(configs)} configs + 3 plan files written to {out}')


if __name__ == '__main__':
    main()
