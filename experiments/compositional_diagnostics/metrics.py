"""Item, family and counterfactual metrics with family-cluster bootstrap intervals and chance baselines.

Per family (six members: base, relevant, irrelevant, rename, reorder, distractor):

| metric | per-family value | what it rewards |
|---|---|---|
| ``exact`` | mean correctness of the six members | full-answer accuracy |
| ``exact_base`` | base correct | accuracy on unperturbed cases |
| ``family`` | all six correct | consistency across the whole family |
| ``relevant`` | base **and** relevant correct | responding correctly to a change that matters; the two expected answers differ, so a solver that ignores the change can get at most one of them |
| ``robust_<v>`` (v = irrelevant, rename, reorder, distractor) | base **and** v correct | resisting a change that should not matter, credited only when both answers are *correct* |
| ``robust_all`` | base and all four invariance variants correct | |

Being invariant is never credited by itself: ``raw_invariance_<v>`` (the parsed outputs agree, after renaming for
``rename``, right or wrong) and ``raw_sensitivity`` (the outputs differ on base vs relevant) are reported only as
uncredited diagnostics, so that a constant answerer's perfect raw invariance is visible and not rewarded.

Uncertainty: every metric is a mean of per-family values; the interval is a percentile bootstrap over families
(the unit within which items are related), stratified by question type for pooled numbers, 2,000 resamples,
seed 0, 95%. Chance: the exact expected value of each metric for a uniform guesser over each member's answer
space (``questions.answer_space_size``), guessing members independently.
"""
from __future__ import annotations

import random
from collections import defaultdict

import answer_grader
import questions as Q

INVARIANCE = ('irrelevant', 'rename', 'reorder', 'distractor')
CREDITED = ('exact', 'exact_base', 'family', 'relevant', *(f'robust_{v}' for v in INVARIANCE), 'robust_all')
DIAGNOSTIC = ('invalid', 'raw_sensitivity', *(f'raw_invariance_{v}' for v in INVARIANCE))
RESAMPLES = 2000
LEVEL = 0.95


def family_values(rows: list[dict], outputs: dict) -> dict:
    """``{family: {'qtype', metric: value}}`` from example rows and ``{id: output}``."""
    by_family: dict[str, dict] = defaultdict(dict)
    for row in rows:
        out = outputs.get(row['id'], '')
        g = answer_grader.grade(row['qtype'], row['answer'], out)
        by_family[row['family']][row['variant']] = (row, g, answer_grader.canonical_output(row['qtype'], out))
    result = {}
    for fam, members in by_family.items():
        assert set(members) == {'base', 'relevant', *INVARIANCE}, fam
        qtype = members['base'][0]['qtype']
        ok = {v: m[1]['correct'] for v, m in members.items()}
        parsed = {v: m[2] for v, m in members.items()}
        vals = {'qtype': qtype,
                'exact': sum(ok.values()) / len(ok),
                'exact_base': float(ok['base']),
                'family': float(all(ok.values())),
                'relevant': float(ok['base'] and ok['relevant']),
                'robust_all': float(ok['base'] and all(ok[v] for v in INVARIANCE)),
                'invalid': sum(m[1]['invalid'] for m in members.values()) / len(members),
                'raw_sensitivity': float(parsed['base'] is not None and parsed['relevant'] is not None
                                         and parsed['base'] != parsed['relevant'])}
        mapping = members['rename'][0].get('mapping')
        for v in INVARIANCE:
            vals[f'robust_{v}'] = float(ok['base'] and ok[v])
            a, b = parsed['base'], parsed[v]
            if v == 'rename' and a is not None and mapping:
                a = _rename_parsed(qtype, a, mapping)
            vals[f'raw_invariance_{v}'] = float(a is not None and a == b)
        result[fam] = vals
    return result


def _rename_parsed(qtype, value, mapping):
    if qtype == 'filter':
        return frozenset((mapping.get(s, s), mapping.get(o, o)) for s, o in value)
    if qtype == 'join':
        return frozenset(mapping.get(z, z) for z in value)
    return value


def chance_family_values(rows: list[dict]) -> dict:
    """Expected per-family metric values of an independent uniform guesser."""
    import render
    by_family: dict[str, dict] = defaultdict(dict)
    for row in rows:
        records = render.parse(row['arm'], row['records'])
        by_family[row['family']][row['variant']] = (row['qtype'], 1.0 / Q.answer_space_size(row['qtype'], records))
    out = {}
    for fam, members in by_family.items():
        p = {v: m[1] for v, m in members.items()}
        qtype = members['base'][0]
        prod = 1.0
        for v in p.values():
            prod *= v
        vals = {'qtype': qtype, 'exact': sum(p.values()) / len(p), 'exact_base': p['base'], 'family': prod,
                'relevant': p['base'] * p['relevant']}
        inv = p['base']
        for v in INVARIANCE:
            vals[f'robust_{v}'] = p['base'] * p[v]
            inv *= p[v]
        vals['robust_all'] = inv
        out[fam] = vals
    return out


def _mean(values):
    return sum(values) / len(values) if values else float('nan')


def summarise(fvals: dict, metrics=CREDITED + DIAGNOSTIC, resamples: int = RESAMPLES, seed: int = 0,
              with_ci: bool = True) -> dict:
    """``{scope: {metric: {'mean', 'lo', 'hi', 'n_families'}}}`` for every question type and ``all``."""
    strata: dict[str, list[dict]] = defaultdict(list)
    for fam in sorted(fvals):
        strata[fvals[fam]['qtype']].append(fvals[fam])
    metrics = [m for m in metrics if all(m in v for vs in strata.values() for v in vs)]
    scopes = {q: {q: strata[q]} for q in Q.QTYPES if q in strata}
    scopes['all'] = dict(strata)
    rng = random.Random(seed)
    out = {}
    for scope, groups in scopes.items():
        point = {m: _weighted(groups, m) for m in metrics}
        boots = {m: [] for m in metrics}
        if with_ci:
            for _ in range(resamples):
                sample = {q: [vs[rng.randrange(len(vs))] for _ in vs] for q, vs in groups.items()}
                for m in metrics:
                    boots[m].append(_weighted(sample, m))
        res = {}
        for m in metrics:
            entry = {'mean': point[m], 'n_families': sum(len(v) for v in groups.values())}
            if with_ci:
                b = sorted(boots[m])
                entry['lo'] = b[int((1 - LEVEL) / 2 * len(b))]
                entry['hi'] = b[min(len(b) - 1, int((1 + LEVEL) / 2 * len(b)))]
            res[m] = entry
        out[scope] = res
    return out


def _weighted(groups: dict, metric: str) -> float:
    """Mean over families with every stratum weighted by its number of families (equal here by design)."""
    total = sum(len(v) for v in groups.values())
    return sum(f[metric] for vs in groups.values() for f in vs) / total
