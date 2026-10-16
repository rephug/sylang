#!/usr/bin/env python
"""Cost ledger of the pretrained-model pilot (``interfaces.md``, section "Ledger").

``ledger.jsonl`` next to this file is append-only and committed. One JSON object per line:

    {"ts": ..., "event": "start|stop|note|balance", "provider": ..., "instance": ..., "gpu": ...,
     "price_per_hour": ..., "stage": "A|B|C", "purpose": ..., "run_names": [...],
     "measured_spend": ..., "balance_after": ..., "note": ...}

Events and their required fields:

    start    provider, instance, gpu, price_per_hour, stage   (the owner started an instance)
    stop     start's fields + measured_spend                  (the instance was stopped/terminated;
                                                               the stage scripts write it with
                                                               elapsed hours x price as an estimate)
    balance  balance_after                                    (the provider's balance after teardown;
                                                               measured_spend here is the provider's
                                                               own figure and overrides the estimate
                                                               of the stop lines of the same stage)
    note     note                                             (anything else: a failed cell, a timeout)

Every event accepts note, purpose and run_names. Nothing here calls a provider API, and a line
that looks like it carries a secret (the value of PILOT_PROVIDER_TOKEN, a bearer token, an API
key) is refused: tokens live only in the environment and the provider CLI's own keychain.

    ledger.py add --event start --provider runpod --instance pod-1a2b --gpu "RTX 4090" \\
        --price-per-hour 0.34 --stage B --purpose "stage B cells"
    ledger.py add --event stop  ... --measured-spend 1.23
    ledger.py add --event balance --provider runpod --balance-after 23.77 --stage B --measured-spend 1.31
    ledger.py add --event note --note "cell stage_b_json_s0_n2000 timed out" --run-names stage_b_json_s0_n2000
    ledger.py report --ceiling 100 [--profile profile.json]
    ledger.py selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER_PATH = HERE / 'ledger.jsonl'
RESULTS_DIR = HERE / 'results'

EVENTS = ('start', 'stop', 'note', 'balance')
STAGES = ('A', 'B', 'C')
FIELDS = ('ts', 'event', 'provider', 'instance', 'gpu', 'price_per_hour', 'stage', 'purpose',
          'run_names', 'measured_spend', 'balance_after', 'note')
REQUIRED = {
    'start': ('provider', 'instance', 'gpu', 'price_per_hour', 'stage'),
    'stop': ('provider', 'instance', 'gpu', 'price_per_hour', 'stage', 'measured_spend'),
    'balance': ('balance_after',),
    'note': ('note',),
}
TOKEN_ENV = 'PILOT_PROVIDER_TOKEN'
# Shapes of secrets that must never be written to the ledger.
SECRET_PATTERNS = (
    re.compile(r'(?i)\bbearer\s+\S{8,}'),
    re.compile(r'(?i)\b(?:sk|hf|rp|ak|pk|tok|key|token|secret|api[_-]?key)[_-][A-Za-z0-9]{16,}'),
    re.compile(r'(?i)\b(?:token|secret|password|api[_-]?key)\s*[:=]\s*\S{8,}'),
    re.compile(r'[A-Za-z0-9+/=_\-]{48,}'),
)


class LedgerError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_ts(value: str) -> str:
    """Accept an ISO-8601 UTC timestamp; return it normalised to ``YYYY-MM-DDTHH:MM:SSZ``."""
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise LedgerError(f'bad timestamp {value!r}: {exc}') from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _money(name: str, value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise LedgerError(f'{name} must be a number, got {value!r}') from None
    if out != out or out < 0:
        raise LedgerError(f'{name} must be a non-negative number, got {value!r}')
    return round(out, 6)


def secret_check(line: dict) -> None:
    text = json.dumps(line)
    token = os.environ.get(TOKEN_ENV, '')
    if len(token) >= 8 and token in text:
        raise LedgerError(f'refusing to write a line containing the value of {TOKEN_ENV}')
    for field in ('provider', 'instance', 'gpu', 'purpose', 'note'):
        value = line.get(field)
        if isinstance(value, str):
            for pattern in SECRET_PATTERNS:
                if pattern.search(value):
                    raise LedgerError(f'{field} looks like it contains a secret ({pattern.pattern!r}); '
                                      f'tokens belong only in the environment ({TOKEN_ENV}), never in the ledger')
    for name in line.get('run_names') or []:
        if not re.fullmatch(r'[A-Za-z0-9._\-]{1,80}', name):
            raise LedgerError(f'run name {name!r} is not a plain run name')


def make_line(event: str, *, ts: str | None = None, provider=None, instance=None, gpu=None,
              price_per_hour=None, stage=None, purpose=None, run_names=None, measured_spend=None,
              balance_after=None, note=None) -> dict:
    """Build and validate one ledger line (keys in the documented order, absent fields null)."""
    if event not in EVENTS:
        raise LedgerError(f'event must be one of {EVENTS}, got {event!r}')
    line = {
        'ts': parse_ts(ts) if ts else now_iso(),
        'event': event,
        'provider': provider or None,
        'instance': instance or None,
        'gpu': gpu or None,
        'price_per_hour': None if price_per_hour in (None, '') else _money('price_per_hour', price_per_hour),
        'stage': stage or None,
        'purpose': purpose or None,
        'run_names': list(run_names) if run_names else None,
        'measured_spend': None if measured_spend in (None, '') else _money('measured_spend', measured_spend),
        'balance_after': None if balance_after in (None, '') else _money('balance_after', balance_after),
        'note': note or None,
    }
    missing = [f for f in REQUIRED[event] if line[f] is None]
    if missing:
        raise LedgerError(f'event {event!r} requires {missing}')
    if line['stage'] is not None and line['stage'] not in STAGES:
        raise LedgerError(f'stage must be one of {STAGES}, got {line["stage"]!r}')
    for field in ('provider', 'instance', 'gpu', 'purpose', 'note'):
        if line[field] is not None and not isinstance(line[field], str):
            raise LedgerError(f'{field} must be a string')
        if isinstance(line[field], str) and ('\n' in line[field] or not line[field].strip()):
            raise LedgerError(f'{field} must be one non-empty line')
    if line['run_names'] is not None and not all(isinstance(n, str) for n in line['run_names']):
        raise LedgerError('run_names must be strings')
    secret_check(line)
    return line


def validate_line(obj, lineno: int) -> dict:
    if not isinstance(obj, dict):
        raise LedgerError(f'line {lineno}: not a JSON object')
    unknown = sorted(set(obj) - set(FIELDS))
    if unknown:
        raise LedgerError(f'line {lineno}: unknown fields {unknown}')
    if 'event' not in obj or 'ts' not in obj:
        raise LedgerError(f'line {lineno}: ts and event are required')
    kwargs = {k: v for k, v in obj.items() if k != 'event'}
    try:
        return make_line(obj['event'], **kwargs)
    except LedgerError as exc:
        raise LedgerError(f'line {lineno}: {exc}') from None


def dump_line(line: dict) -> str:
    return json.dumps({k: line[k] for k in FIELDS}, separators=(',', ':'), ensure_ascii=True)


def read_ledger(path: Path) -> list[dict]:
    """Every line validated; a corrupt ledger is an error, not a warning."""
    if not path.exists():
        return []
    lines = []
    with open(path, encoding='utf-8') as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise LedgerError(f'line {lineno}: not JSON ({exc})') from None
            lines.append(validate_line(obj, lineno))
    return lines


def append_line(path: Path, line: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(dump_line(line) + '\n')


# ---------------------------------------------------------------------------
# Report

def completed_experiments(results_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Result directories holding an ``eval.json``: (trained, zero-shot, unreadable)."""
    trained, zero_shot, broken = [], [], []
    if not results_dir.is_dir():
        return trained, zero_shot, broken
    for d in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        path = d / 'eval.json'
        if not path.is_file() or d.name.startswith('smoke_'):
            continue  # smoke_cpu.sh results are free CPU checks, not experiments
        try:
            with open(path, encoding='utf-8') as f:
                adapter = json.load(f).get('adapter')
        except (OSError, ValueError):
            broken.append(d.name)
            continue
        if adapter in (None, 'none', ''):
            zero_shot.append(d.name)
        else:
            trained.append(d.name)
    return trained, zero_shot, broken


def _cost_of(obj) -> float | None:
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        for key in ('cost', 'cost_usd', 'usd', 'projected_cost', 'total_cost', 'dollars'):
            if isinstance(obj.get(key), (int, float)) and not isinstance(obj[key], bool):
                return float(obj[key])
    return None


def read_projection(profile_path: Path) -> dict:
    """``profile.json['projection']``: per-stage and total projected cost.

    Accepted shapes (profile.py, builder C): ``projection.stages.<stage>.cost`` or
    ``projection.<stage>.cost`` (stage keys ``B``/``C`` or ``stage_b``/``stage_c``), with the total
    under ``projection.total.cost``, ``projection.total_cost`` or ``projection.total`` (a number);
    without a total the stage costs are summed. Returns ``{'stages': {stage: cost}, 'total': cost}``.
    """
    with open(profile_path, encoding='utf-8') as f:
        profile = json.load(f)
    proj = profile.get('projection') if isinstance(profile, dict) else None
    if not isinstance(proj, dict):
        raise LedgerError(f'{profile_path}: no "projection" object')
    stages: dict[str, float] = {}
    source = proj.get('stages') if isinstance(proj.get('stages'), dict) else proj
    for key, value in source.items():
        stage = str(key).upper().replace('STAGE_', '').replace('STAGE', '').strip('_ ')
        if stage in STAGES:
            cost = _cost_of(value)
            if cost is not None:
                stages[stage] = cost
    total = None
    for key in ('total', 'total_cost', 'cost'):
        if key in proj:
            total = _cost_of(proj[key])
            if total is not None:
                break
    if total is None:
        if not stages:
            raise LedgerError(f'{profile_path}: projection holds neither per-stage costs nor a total')
        total = sum(stages.values())
    return {'stages': stages, 'total': total, 'contingency': proj.get('contingency')}


def report(path: Path, ceiling: float, profile: Path | None = None, results_dir: Path | None = None,
           out=None) -> int:
    out = out or sys.stdout
    results_dir = results_dir or RESULTS_DIR
    lines = read_ledger(path)
    counts = {e: sum(1 for ln in lines if ln['event'] == e) for e in EVENTS}
    print(f'ledger: {path} ({len(lines)} lines: ' + ', '.join(f'{v} {k}' for k, v in counts.items()) + ')',
          file=out)

    # Spend per stage: the stop lines' estimate, and the provider's figure from balance lines.
    estimate = {s: 0.0 for s in STAGES}
    provider = {s: 0.0 for s in STAGES}
    provider_seen = {s: False for s in STAGES}
    unstaged_provider = 0.0
    for ln in lines:
        if ln['event'] == 'stop':
            estimate[ln['stage']] += ln['measured_spend']
        elif ln['event'] == 'balance' and ln['measured_spend'] is not None:
            if ln['stage'] is None:
                unstaged_provider += ln['measured_spend']
            else:
                provider[ln['stage']] += ln['measured_spend']
                provider_seen[ln['stage']] = True
    used = {s: (provider[s] if provider_seen[s] else estimate[s]) for s in STAGES}
    total_estimate = sum(estimate.values())
    total_provider = sum(provider.values()) + unstaged_provider
    total_used = sum(used.values()) + unstaged_provider
    print(f'measured spend (stop lines, elapsed x price):   ${total_estimate:9.2f}', file=out)
    print(f'provider-reported spend (balance lines):         ${total_provider:9.2f}', file=out)
    print(f'spend counted against the ceiling:               ${total_used:9.2f}', file=out)
    print('spend per stage:   stage   stop-lines   provider   counted', file=out)
    for s in STAGES:
        prov = f'{provider[s]:9.2f}' if provider_seen[s] else '        -'
        print(f'                   {s:5}   {estimate[s]:9.2f}   {prov}   {used[s]:9.2f}', file=out)
    if unstaged_provider:
        print(f'                   (none)          -   {unstaged_provider:9.2f}   {unstaged_provider:9.2f}', file=out)

    trained, zero_shot, broken = completed_experiments(results_dir)
    n = len(trained)
    print(f'completed experiments (results/*/eval.json with a trained adapter): {n}'
          f'  (+ {len(zero_shot)} zero-shot baselines)', file=out)
    if broken:
        print(f'  unreadable eval.json in: {", ".join(broken)}', file=out)
    per = f'${total_used / n:.2f}' if n else 'n/a (no completed experiment)'
    print(f'dollars per completed experiment: {per}', file=out)

    balances = [ln for ln in lines if ln['event'] == 'balance']
    if balances:
        last = balances[-1]
        print(f'last recorded balance: ${last["balance_after"]:.2f} at {last["ts"]}'
              + (f' ({last["provider"]})' if last['provider'] else ''), file=out)
    else:
        print('last recorded balance: none recorded', file=out)

    # Open start events: a stop matches the latest unmatched start of the same provider/instance.
    open_starts: list[dict] = []
    for ln in lines:
        if ln['event'] == 'start':
            open_starts.append(ln)
        elif ln['event'] == 'stop':
            for i in range(len(open_starts) - 1, -1, -1):
                if (open_starts[i]['provider'], open_starts[i]['instance']) == (ln['provider'], ln['instance']):
                    del open_starts[i]
                    break
            else:
                print(f'warning: stop at {ln["ts"]} ({ln["provider"]}/{ln["instance"]}) has no matching start',
                      file=out)
    print(f'open start events without a matching stop: {len(open_starts)}', file=out)
    for ln in open_starts:
        print(f'  !!! INSTANCE MAY STILL BE BILLING: start at {ln["ts"]} {ln["provider"]}/{ln["instance"]} '
              f'({ln["gpu"]}, ${ln["price_per_hour"]}/h, stage {ln["stage"]}) has no stop line', file=out)
    stops = [ln for ln in lines if ln['event'] == 'stop']
    if stops:
        last_stop_index = max(i for i, ln in enumerate(lines) if ln['event'] == 'stop')
        if not any(ln['event'] == 'balance' for ln in lines[last_stop_index + 1:]):
            print(f'  warning: the last stop ({stops[-1]["ts"]}, stage {stops[-1]["stage"]}) is not followed by a '
                  f'balance line: teardown not recorded (verify_teardown.md)', file=out)

    exceeded = []
    if total_used > ceiling:
        exceeded.append(f'measured spend ${total_used:.2f} exceeds the ceiling ${ceiling:.2f}')
    if total_estimate > ceiling and total_estimate > total_used:
        exceeded.append(f'stop-line estimate ${total_estimate:.2f} exceeds the ceiling ${ceiling:.2f}')
    if profile is not None:
        proj = read_projection(profile)
        stopped = {ln['stage'] for ln in stops}
        remaining = sum(c for s, c in proj['stages'].items() if s not in stopped)
        cont = f' (x{proj["contingency"]} contingency included)' if proj.get('contingency') else ''
        stage_text = ', '.join(f'stage {s} ${c:.2f}' for s, c in sorted(proj['stages'].items())) or 'no per-stage costs'
        print(f'projection ({profile}): {stage_text}; total ${proj["total"]:.2f}{cont}', file=out)
        print(f'  projected cost of stages without a stop line yet: ${remaining:.2f}; '
              f'measured + projected remaining: ${total_used + remaining:.2f}', file=out)
        if proj['total'] > ceiling and (remaining > 0 or not proj['stages']):
            exceeded.append(f'projected total ${proj["total"]:.2f} exceeds the ceiling ${ceiling:.2f}')
        elif proj['total'] > ceiling:
            print(f'  (the projected total ${proj["total"]:.2f} exceeded the ceiling, but every projected stage has '
                  f'a stop line: the measured spend decides)', file=out)
        if total_used + remaining > ceiling:
            exceeded.append(f'measured ${total_used:.2f} + projected remaining ${remaining:.2f} exceeds the '
                            f'ceiling ${ceiling:.2f}')
    if exceeded:
        print(f'ceiling ${ceiling:.2f}: EXCEEDED', file=out)
        for why in exceeded:
            print(f'  !!! {why}', file=out)
        return 1
    print(f'ceiling ${ceiling:.2f}: OK (${ceiling - total_used:.2f} left)', file=out)
    return 0


# ---------------------------------------------------------------------------
# CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--path', type=Path, default=LEDGER_PATH, help=f'ledger file (default {LEDGER_PATH})')
    sub = p.add_subparsers(dest='command', required=True)

    a = sub.add_parser('add', help='append one validated line')
    a.add_argument('--event', required=True, choices=EVENTS)
    a.add_argument('--provider')
    a.add_argument('--instance', help='the provider\'s instance/pod/app id (never a token)')
    a.add_argument('--gpu')
    a.add_argument('--price-per-hour', dest='price_per_hour')
    a.add_argument('--stage', choices=STAGES)
    a.add_argument('--purpose')
    a.add_argument('--run-names', dest='run_names', nargs='*')
    a.add_argument('--measured-spend', dest='measured_spend')
    a.add_argument('--balance-after', dest='balance_after')
    a.add_argument('--note')
    a.add_argument('--ts', help='ISO-8601 UTC timestamp (default: now); for lines recorded after the fact')
    a.add_argument('--dry-run', action='store_true', help='validate and print the line; append nothing')

    r = sub.add_parser('report', help='spend, completed experiments, open instances, ceiling check')
    r.add_argument('--ceiling', type=float, default=100.0, help='USD (default 100)')
    r.add_argument('--profile', type=Path, help='profile.json from profile.py (its "projection" is checked)')
    r.add_argument('--results-dir', dest='results_dir', type=Path, default=RESULTS_DIR)

    sub.add_parser('selftest', help='run the self-checks against a temporary ledger')
    return p


def cmd_add(args) -> int:
    line = make_line(args.event, ts=args.ts, provider=args.provider, instance=args.instance, gpu=args.gpu,
                     price_per_hour=args.price_per_hour, stage=args.stage, purpose=args.purpose,
                     run_names=args.run_names, measured_spend=args.measured_spend,
                     balance_after=args.balance_after, note=args.note)
    read_ledger(args.path)  # refuse to append to a corrupt ledger
    if args.dry_run:
        print('dry run (not appended): ' + dump_line(line))
        return 0
    append_line(args.path, line)
    print(f'appended to {args.path}: ' + dump_line(line))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'add':
            return cmd_add(args)
        if args.command == 'report':
            return report(args.path, args.ceiling, args.profile, args.results_dir)
        if args.command == 'selftest':
            return selftest()
    except LedgerError as exc:
        print(f'ledger.py: {exc}', file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------
# Self-checks (a temporary ledger, the CLI exercised through ``main``)

def selftest() -> int:
    import contextlib
    import io

    def run(*argv) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                rc = main(list(argv))
            except SystemExit as exc:  # argparse rejected the command line
                rc = exc.code if isinstance(exc.code, int) else 2
        return rc, buf.getvalue()

    checks = 0

    def check(cond: bool, what: str) -> None:
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(f'selftest failed: {what}')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ledger = tmp / 'ledger.jsonl'
        results = tmp / 'results'
        base = ['--path', str(ledger)]
        start = ['--provider', 'runpod', '--instance', 'pod-1', '--gpu', 'RTX 4090', '--price-per-hour', '0.34']

        rc, out = run(*base, 'add', '--event', 'note', '--note', 'ledger opened', '--dry-run')
        check(rc == 0 and 'dry run' in out and not ledger.exists(), 'dry run appends nothing')
        rc, out = run(*base, 'add', '--event', 'note', '--note', 'ledger opened')
        check(rc == 0 and ledger.exists(), 'note appended')
        rc, out = run(*base, 'add', '--event', 'start', '--provider', 'runpod', '--stage', 'A')
        check(rc == 2 and 'requires' in out, 'start without instance/gpu/price is refused')
        rc, out = run(*base, 'add', '--event', 'stop', *start, '--stage', 'A')
        check(rc == 2 and 'measured_spend' in out, 'stop without measured_spend is refused')
        rc, out = run(*base, 'add', '--event', 'balance', '--provider', 'runpod')
        check(rc == 2 and 'balance_after' in out, 'balance without balance_after is refused')
        rc, out = run(*base, 'add', '--event', 'start', *start, '--stage', 'D')
        check(rc == 2, 'unknown stage is refused')
        rc, out = run(*base, 'add', '--event', 'start', *start, '--stage', 'A', '--price-per-hour', '-1')
        check(rc == 2, 'negative price is refused')
        rc, out = run(*base, 'add', '--event', 'note', '--note', 'Authorization: Bearer abcdefghijklmnop')
        check(rc == 2 and 'secret' in out, 'bearer token in a note is refused')
        rc, out = run(*base, 'add', '--event', 'note', '--note', 'key hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456')
        check(rc == 2, 'hf_ token is refused')
        os.environ[TOKEN_ENV] = 'ZZtestsecret99'
        try:
            rc, out = run(*base, 'add', '--event', 'note', '--note', 'token ZZtestsecret99 leaked')
            check(rc == 2 and TOKEN_ENV in out, f'the value of {TOKEN_ENV} is refused')
        finally:
            del os.environ[TOKEN_ENV]

        rc, out = run(*base, 'add', '--event', 'start', *start, '--stage', 'A', '--purpose', 'profile',
                      '--ts', '2026-10-20T10:00:00Z')
        check(rc == 0, 'start appended')
        rc, out = run(*base, 'report', '--ceiling', '100', '--results-dir', str(results))
        check(rc == 0 and 'open start events without a matching stop: 1' in out
              and 'INSTANCE MAY STILL BE BILLING' in out, 'open start is flagged loudly')
        rc, out = run(*base, 'add', '--event', 'stop', *start, '--stage', 'A', '--measured-spend', '0.51',
                      '--run-names', 'stage_a_profile', '--ts', '2026-10-20T11:30:00Z')
        check(rc == 0, 'stop appended')
        rc, out = run(*base, 'report', '--results-dir', str(results))
        check(rc == 0 and 'open start events without a matching stop: 0' in out, 'start matched by stop')
        check('not followed by a balance line' in out, 'missing balance after stop is flagged')
        check('dollars per completed experiment: n/a' in out, 'no experiments yet')
        rc, out = run(*base, 'add', '--event', 'balance', '--provider', 'runpod', '--stage', 'A',
                      '--balance-after', '24.40', '--measured-spend', '0.60', '--ts', '2026-10-20T11:40:00Z')
        check(rc == 0, 'balance appended')

        # Results: two trained, one zero-shot, one without eval.json.
        for name, adapter in (('stage_b_json_s0_n2000', 'runs/stage_b_json_s0_n2000/best'),
                              ('stage_b_sylang2_s0_n2000', 'runs/stage_b_sylang2_s0_n2000/best'),
                              ('zeroshot_english_json', 'none')):
            (results / name).mkdir(parents=True)
            (results / name / 'eval.json').write_text(json.dumps({'adapter': adapter, 'split': 'test'}))
        (results / 'stage_b_english_s0_n2000').mkdir()
        rc, out = run(*base, 'add', '--event', 'start', *start, '--stage', 'B', '--ts', '2026-10-21T10:00:00Z')
        rc, out = run(*base, 'add', '--event', 'stop', *start, '--stage', 'B', '--measured-spend', '2.00',
                      '--ts', '2026-10-21T13:00:00Z')
        rc, out = run(*base, 'add', '--event', 'balance', '--balance-after', '22.00',
                      '--ts', '2026-10-21T13:10:00Z')  # no provider figure: stage B keeps its estimate
        rc, out = run(*base, 'report', '--ceiling', '100', '--results-dir', str(results))
        check(rc == 0, 'report under the ceiling exits 0')
        check('completed experiments (results/*/eval.json with a trained adapter): 2  (+ 1 zero-shot' in out,
              f'completed experiments counted: {out}')
        check('spend counted against the ceiling:               $     2.60' in out, f'provider figure overrides stage A: {out}')
        check('dollars per completed experiment: $1.30' in out, f'dollars per experiment: {out}')
        check('last recorded balance: $22.00' in out, 'last balance printed')
        rc, out = run(*base, 'report', '--ceiling', '2.5', '--results-dir', str(results))
        check(rc == 1 and 'EXCEEDED' in out, 'spend over the ceiling exits non-zero')

        profile = tmp / 'profile.json'
        profile.write_text(json.dumps({'projection': {'contingency': 1.5, 'stages': {'B': {'cost': 3.0},
                                                      'C': {'cost': 7.0}}, 'total': {'cost': 10.0}}}))
        rc, out = run(*base, 'report', '--ceiling', '100', '--profile', str(profile), '--results-dir', str(results))
        check(rc == 0 and 'stage C $7.00' in out and 'projected remaining: $9.60' in out,
              f'projection read; stage B already stopped so only C remains: {out}')
        rc, out = run(*base, 'report', '--ceiling', '9', '--profile', str(profile), '--results-dir', str(results))
        check(rc == 1 and 'projected total $10.00 exceeds' in out, 'projection over the ceiling exits non-zero')
        rc, out = run(*base, 'report', '--ceiling', '9.5', '--profile', str(profile), '--results-dir', str(results))
        check(rc == 1 and 'projected remaining $7.00 exceeds' in out, 'measured + remaining over the ceiling')
        profile.write_text(json.dumps({'projection': {'stage_b': {'cost_usd': 1.0}, 'stage_c': {'cost_usd': 2.0}}}))
        rc, out = run(*base, 'report', '--profile', str(profile), '--results-dir', str(results))
        check(rc == 0 and 'total $3.00' in out, 'alternative projection shape summed')
        profile.write_text(json.dumps({'measurements': {}}))
        rc, out = run(*base, 'report', '--profile', str(profile), '--results-dir', str(results))
        check(rc == 2 and 'projection' in out, 'profile without a projection is an error')

        # Every written line re-validates and keeps the documented key order.
        for raw in ledger.read_text().splitlines():
            check(list(json.loads(raw)) == list(FIELDS), 'key order')
        n_lines = len(read_ledger(ledger))
        check(n_lines == 7, f'7 lines expected, {n_lines} found')
        with open(ledger, 'a') as f:
            f.write('{"ts": "2026-01-01T00:00:00Z", "event": "stop"}\n')
        rc, out = run(*base, 'report', '--results-dir', str(results))
        check(rc == 2 and 'line 8' in out, 'corrupt line is reported with its number')
        rc, out = run(*base, 'add', '--event', 'note', '--note', 'after corruption')
        check(rc == 2, 'no append to a corrupt ledger')

    print(f'ledger.py selftest: {checks} checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
