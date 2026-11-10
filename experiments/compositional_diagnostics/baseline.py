"""CPU-only baseline report: every weak solver on every arm of the test split, the audits, and the targeted checks.

    python -I baseline.py [--data data/cd-v0] [--out reports/baseline.json --md reports/baseline_report.md]

Each solver parses the rendered records of the arm it is run on. Because every rendering is lossless and
order-preserving, a solver's outputs must be identical in all four arms; this is asserted, so the report shows
one row per solver (with the arm-identity check) rather than four copies. ``TARGETS`` lists, for every
shortcut, the metric that must expose it; ``SPECIFICITY`` lists metrics that must *not* penalise a defect where
the defect is harmless (a solver that ignores aspect is entitled to full marks on tense questions).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit as A  # noqa: E402
import metrics  # noqa: E402
import questions as Q  # noqa: E402
import render  # noqa: E402
import score as S  # noqa: E402
import solvers as SV  # noqa: E402

# (solver, scope, metric, max allowed value): the targeted test that catches the shortcut.
TARGETS = [
    ('constant_majority', 'all', 'relevant', 0.0),
    ('constant_majority', 'all', 'family', 0.0),
    ('random_guess', 'all', 'family', 0.01),
    ('first_record', 'all', 'family', 0.10),
    ('last_record', 'all', 'family', 0.10),
    ('ignore_tense', 'tense', 'relevant', 0.0),
    ('ignore_tense', 'join', 'family', 0.10),
    ('ignore_aspect', 'aspect', 'relevant', 0.0),
    ('ignore_evidence', 'evidence', 'relevant', 0.0),
    ('ignore_polarity', 'join', 'family', 0.10),
    ('ignore_polarity', 'profile', 'family', 0.60),
    ('ignore_relation', 'join', 'family', 0.10),
    ('ignore_relation', 'profile', 'family', 0.50),
    *((s, q, 'family', 0.35) for s in ('subject_only', 'object_only') for q in Q.PAIR_QTYPES),
    ('feature_bag', 'filter', 'exact_base', 0.0),
    ('unbound_join', 'join', 'family', 0.05),
    ('role_swap', 'all', 'family', 0.0),
]
# (solver, scope, metric, min required value): the defect is irrelevant there, so no penalty is allowed.
SPECIFICITY = [
    ('oracle', 'all', m, 1.0) for m in metrics.CREDITED
] + [
    ('ignore_aspect', 'tense', 'family', 1.0), ('ignore_aspect', 'evidence', 'family', 1.0),
    ('ignore_aspect', 'join', 'family', 1.0), ('ignore_evidence', 'tense', 'family', 1.0),
    ('ignore_evidence', 'join', 'family', 1.0), ('ignore_tense', 'aspect', 'family', 1.0),
    ('ignore_polarity', 'tense', 'family', 1.0), ('ignore_relation', 'evidence', 'family', 1.0),
    ('subject_only', 'filter', 'family', 1.0), ('feature_bag', 'join', 'family', 1.0),
    ('unbound_join', 'filter', 'family', 1.0),
]


def run_solvers(data_dir: Path, split: str = 'test', names=SV.SOLVERS) -> dict:
    dev = S.load_rows(data_dir, 'dev', 'json')
    sv = SV.Solvers(dev)
    rows = {arm: S.load_rows(data_dir, split, arm) for arm in render.ARMS}
    outputs, identical = {}, {}
    for name in names:
        per_arm = {arm: {r['id']: sv.solve(name, r, arm) for r in rows[arm]} for arm in render.ARMS}
        identical[name] = all(per_arm[a] == per_arm['json'] for a in render.ARMS)
        outputs[name] = per_arm['json']
    return {'rows': rows, 'outputs': outputs, 'identical_across_arms': identical, 'majority': sv.majority}


def check(results: dict) -> list[dict]:
    out = []
    for solver, scope, metric, bound in TARGETS:
        v = results[solver]['summary'][scope][metric]['mean']
        out.append({'kind': 'target', 'solver': solver, 'scope': scope, 'metric': metric, 'bound': f'<= {bound}',
                    'value': round(v, 4), 'oracle': results['oracle']['summary'][scope][metric]['mean'],
                    'pass': v <= bound + 1e-12})
    for solver, scope, metric, bound in SPECIFICITY:
        v = results[solver]['summary'][scope][metric]['mean']
        out.append({'kind': 'specificity', 'solver': solver, 'scope': scope, 'metric': metric, 'bound': f'>= {bound}',
                    'value': round(v, 4), 'pass': v >= bound - 1e-12})
    return out


def build_report(data_dir: Path, resamples: int = metrics.RESAMPLES) -> dict:
    t0 = time.perf_counter()
    run = run_solvers(data_dir)
    rows = run['rows']['json']
    results = {}
    for name, outs in run['outputs'].items():
        fvals = metrics.family_values(rows, outs)
        results[name] = {'summary': metrics.summarise(fvals, resamples=resamples)}
    chance = metrics.summarise(metrics.chance_family_values(rows), metrics=metrics.CREDITED, with_ci=False)
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    verification = HERE / 'reports' / 'oracle_verification.json'
    return {
        'data': {'version': manifest['version'], 'config': manifest['config'], 'families': manifest['families'],
                 'sha256': manifest['sha256']},
        'solvers': results, 'chance_uniform': chance,
        'identical_across_arms': run['identical_across_arms'], 'majority_answers': run['majority'],
        'checks': check(results), 'audit': A.audit(data_dir),
        'oracle_verification': json.loads(verification.read_text()) if verification.exists() else None,
        'seconds': round(time.perf_counter() - t0, 1),
    }


def _f(e, ci=True):
    if ci and 'lo' in e:
        return f'{e["mean"]:.2f} [{e["lo"]:.2f}, {e["hi"]:.2f}]'
    return f'{e["mean"]:.2f}'


def md(report: dict) -> str:
    sol, chance = report['solvers'], report['chance_uniform']
    cfg = report['data']['config']
    L = ['# Compositional diagnostics: CPU baseline report', '',
         f'Data `{report["data"]["version"]}` (seed {cfg["seed"]}; {report["data"]["families"]["test"]} test and '
         f'{report["data"]["families"]["dev"]} dev families, six members each; '
         f'`families.jsonl` sha256 `{report["data"]["sha256"]["families.jsonl"][:16]}…`). No model was run: every '
         'row is a deterministic program (`solvers.py`). Intervals are 95% percentile bootstraps over families '
         '(stratified by question type for "all"), 2,000 resamples, seed 0.', '',
         'Every solver reads the rendered records of each arm and produced identical outputs in all four arms '
         f'(english, json, sylang2, sylang2g): {all(report["identical_across_arms"].values())}. The table is therefore '
         'one row per solver.', '',
         '## Pooled test results (all question types)', '',
         '| solver | exact | family | relevant | robust_all | robust_irrelevant | robust_rename | robust_reorder | robust_distractor | raw_invariance_reorder (uncredited) |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for name in SV.SOLVERS:
        a = sol[name]['summary']['all']
        L.append(f'| {name} | {_f(a["exact"])} | {_f(a["family"])} | {_f(a["relevant"])} | {_f(a["robust_all"])} | '
                 f'{_f(a["robust_irrelevant"], False)} | {_f(a["robust_rename"], False)} | '
                 f'{_f(a["robust_reorder"], False)} | {_f(a["robust_distractor"], False)} | '
                 f'{a["raw_invariance_reorder"]["mean"]:.2f} |')
    c = chance['all']
    L.append(f'| *uniform-guess chance* | {c["exact"]["mean"]:.3f} | {c["family"]["mean"]:.1e} | '
             f'{c["relevant"]["mean"]:.3f} | {c["robust_all"]["mean"]:.1e} | {c["robust_irrelevant"]["mean"]:.3f} | '
             f'{c["robust_rename"]["mean"]:.3f} | {c["robust_reorder"]["mean"]:.3f} | '
             f'{c["robust_distractor"]["mean"]:.3f} | – |')
    L += ['', 'The constant answerer is perfectly invariant in its raw outputs (last column) and still scores 0 on '
          '`relevant` and `family`: invariance is credited only together with correct answers.', '',
          '## Per question type: family consistency (all six members correct)', '',
          '| solver | ' + ' | '.join(Q.QTYPES) + ' |', '|---|' + '---|' * len(Q.QTYPES)]
    for name in SV.SOLVERS:
        s = sol[name]['summary']
        L.append(f'| {name} | ' + ' | '.join(_f(s[q]['family']) for q in Q.QTYPES) + ' |')
    L.append('| *chance* | ' + ' | '.join(f'{chance[q]["family"]["mean"]:.1e}' for q in Q.QTYPES) + ' |')
    L += ['', '## Per question type: exact accuracy over all members', '',
          '| solver | ' + ' | '.join(Q.QTYPES) + ' |', '|---|' + '---|' * len(Q.QTYPES)]
    for name in SV.SOLVERS:
        s = sol[name]['summary']
        L.append(f'| {name} | ' + ' | '.join(_f(s[q]['exact'], False) for q in Q.QTYPES) + ' |')
    L.append('| *chance* | ' + ' | '.join(f'{chance[q]["exact"]["mean"]:.3f}' for q in Q.QTYPES) + ' |')
    L += ['', '## Targeted checks', '',
          'Each shortcut must be exposed by the metric named for it (target), and must not be penalised where it '
          'is harmless (specificity).', '', '| kind | solver | scope | metric | bound | value | pass |',
          '|---|---|---|---|---|---|---|']
    for ch in report['checks']:
        if ch['solver'] == 'oracle':
            continue
        L.append(f'| {ch["kind"]} | {ch["solver"]} | {ch["scope"]} | {ch["metric"]} | {ch["bound"]} | {ch["value"]:.3f} | '
                 f'{"yes" if ch["pass"] else "**NO**"} |')
    oracle_ok = all(ch['pass'] for ch in report['checks'] if ch['solver'] == 'oracle')
    L += ['', f'Oracle scores 1.0 on every credited metric: {oracle_ok}. All checks pass: '
          f'{all(ch["pass"] for ch in report["checks"])}.', '']
    ov = report['oracle_verification']
    if ov:
        L += ['## Oracle verification', '',
              '| domain | cases | evaluations (oracle vs reference vs solver path, both record orders) | mismatches |',
              '|---|---|---|---|']
        for d in ('D1', 'D2', 'D3'):
            L.append(f'| {d} | {ov[d]["cases"]:,} | {ov[d]["evaluations"]:,} | {ov[d]["n_mismatch"]} |')
        L += ['', 'Domains are defined in `verify_oracle.py`; hand-checked examples are in `tests/test_handchecked.py`.', '']
    au = report['audit']
    L += ['## Audits (test split unless noted)', '', '### Answer balance', '',
          '| qtype | majority share | per-value counts or answer-set sizes | empty share |', '|---|---|---|---|']
    for q, e in au['answer_balance']['test'].items():
        detail = e.get('values') or e.get('answer_size') or f'{e["distinct_answers"]} distinct profiles'
        L.append(f'| {q} | {e["majority_share"]:.3f} | {detail} | {e.get("empty_share", "–")} |')
    idn, nd = au['identifiers'], au['near_duplicates']
    L += ['', '### Identifier and near-duplicate overlap', '',
          f'- identifiers: dev {idn["n_identifiers"]["dev"]:,}, test {idn["n_identifiers"]["test"]:,}; '
          f'shared between dev and test: {idn["dev_test_overlap"]}; in the pilot\'s training hash bucket: '
          f'{idn["pilot_training_bucket_identifiers"]} (dev ids are all in the pilot\'s dev bucket, test ids in its '
          'test bucket). Identifiers that also occur in the pilot\'s frozen v1 cases (same buckets, so overlap is possible): '
          + ', '.join(f'{k} {v}' for k, v in idn['shared_with_pilot_v1_cases'].items()) + '.',
          f'- structural signatures (invariant to renaming and reordering) of every member: dev {nd["signatures"]["dev"]:,}, '
          f'test {nd["signatures"]["test"]:,}, shared across splits: {nd["cross_split_overlap"]}.', '',
          '### Position cues', '',
          '| qtype | answer uses the first record | answer uses the last record | queried record position by quarter of the case |',
          '|---|---|---|---|']
    for q, e in au['positions']['test'].items():
        L.append(f'| {q} | {e["uses_first_record"]:.3f} | {e["uses_last_record"]:.3f} | '
                 f'{e.get("target_position_quarters", "–")} |')
    L += ['', '### Length cues', '',
          'A lookup table (question type, rendered-length bucket of 20 characters, K) -> majority answer, fitted on dev '
          'and scored on test, against the question type\'s dev majority; and how often a relevant or an irrelevant '
          'edit changes the rendering\'s length (a within-family signal a model never sees, since items are '
          'presented one at a time).', '',
          '| arm | qtype | length lookup | majority | relevant edits change length | irrelevant edits change length |',
          '|---|---|---|---|---|---|']
    for arm, per in au['length_cues'].items():
        for q, e in per.items():
            L.append(f'| {arm} | {q} | {e["length_lookup_accuracy"]:.3f} | {e["majority_accuracy"]:.3f} | '
                     f'{e["share_relevant_edits_changing_length"]:.2f} | {e["share_irrelevant_edits_changing_length"]:.2f} |')
    L += ['', f'Mean rendered characters per test item: ' +
          ', '.join(f'{a} {v["test"]}' for a, v in au['mean_record_chars'].items()) + '.', '']
    return '\n'.join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--data', default=str(HERE / 'data' / 'cd-v0'))
    p.add_argument('--out', default=str(HERE / 'reports' / 'baseline.json'))
    p.add_argument('--md', default=str(HERE / 'reports' / 'baseline_report.md'))
    p.add_argument('--resamples', type=int, default=metrics.RESAMPLES)
    args = p.parse_args(argv)
    report = build_report(Path(args.data), args.resamples)
    seconds = report.pop('seconds')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    Path(args.md).write_text(md(report))
    failed = [c for c in report['checks'] if not c['pass']]
    print(f'wrote {args.out} and {args.md} in {seconds}s; checks failed: {len(failed)}')
    for c in failed:
        print('  FAIL', c)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
