"""Qwen 3 token counts for candidate 'improved-M' encodings versus the tested compact format, English,
compact JSON and CSV, on identical synthetic records (doc: sylang-evidence-assessment-and-plan-2026-10-16.md §2).

    python -I measure_m.py --source <packet> --tokenizer <qwen3 tokenizer.json> --out results/m_design_tokens.json

Record distributions: unique identifiers (the study's items), workflow-like reuse (half the records reuse an
earlier subject, a fifth an earlier object), few actors (80% / 50%), and the same three with 12-20-character
labels. Encodings, tokenizer unchanged: sylang2 (tabs), sylang2_sp (spaces), m_intern (the batch codec's
dictionary: '@i<TAB>label' lines for labels occurring twice or more, '@i' references), m_group (same subject
and code merged), m_subj (one line per subject), m_macro (the 12 most frequent middle codes as single letters),
english, json, csv. The answer (JSON list of pairs) and the bare 'subject object' list are counted separately.
Only aggregates are written; identifiers are fresh synthetic strings.
"""
import argparse

import json, random, sys, statistics
from collections import Counter
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import language as L, study3
from followup import Runtime
from tokens import Counter as Tok

ARGS = argparse.ArgumentParser(); ARGS.add_argument('--source', required=True); ARGS.add_argument('--tokenizer', required=True); ARGS.add_argument('--out', required=True)
ARGS = ARGS.parse_args()
rt = Runtime(ARGS.source); R = L.Renderer(rt); T = Tok(ARGS.tokenizer)
INSTR = study3.WORDINGS['train'][0]

def records(k, rng, p_reuse=0.0, p_reuse_obj=0.0):
    stream = study3.ItemStream3('test', rng.randrange(10**6), k_values=(k,))
    used = set(); out = []
    for i in range(k):
        rec = stream.record(used, False)
        if out and rng.random() < p_reuse:
            rec = dict(rec, subject=rng.choice(out)['subject'])
        if out and rng.random() < p_reuse_obj:
            rec = dict(rec, object=rng.choice(out)['object'])
        out.append(rec)
    return out

def code(s):
    return R.payload('codes', [s]).split('\n')[1].split('\t')[1]

def enc_sylang2(recs, sep='\t'):
    return ''.join(f"{s['subject']}{sep}{code(s)}{sep}{s['object']}\n" for s in recs)

def enc_intern(recs):
    counts = Counter([s['subject'] for s in recs] + [s['object'] for s in recs])
    order = []
    for s in recs:
        for lab in (s['subject'], s['object']):
            if counts[lab] >= 2 and lab not in order:
                order.append(lab)
    ref = {lab: f'@{i}' for i, lab in enumerate(order)}
    head = ''.join(f"@{i}\t{lab}\n" for i, lab in enumerate(order))
    return head + ''.join(f"{ref.get(s['subject'], s['subject'])}\t{code(s)}\t{ref.get(s['object'], s['object'])}\n" for s in recs)

def enc_group(recs):
    lines, key_order, groups = [], [], {}
    for s in recs:
        key = (s['subject'], code(s))
        if key not in groups:
            groups[key] = []; key_order.append(key)
        groups[key].append(s['object'])
    return ''.join(f"{k[0]}\t{k[1]}\t{','.join(groups[k])}\n" for k in key_order)

def enc_subj(recs):
    order, groups = [], {}
    for s in recs:
        if s['subject'] not in groups:
            groups[s['subject']] = []; order.append(s['subject'])
        groups[s['subject']].append(f"{code(s)} {s['object']}")
    return ''.join(f"{subj}\t{';'.join(groups[subj])}\n" for subj in order)

MACRO = None
def enc_macro(recs, table):
    return ''.join(f"{s['subject']}\t{table.get(code(s), code(s))}\t{s['object']}\n" for s in recs)

def enc_json(recs):
    return '[' + ','.join(study3._json_record(s) for s in recs) + ']\n'
def enc_csv(recs):
    return ''.join(','.join(s[f] for f in ('subject','relation','polarity','tense','aspect','evidence','object')) + '\n' for s in recs)
def enc_english(recs):
    return R.payload('english', [L._statement(*(s[f] for f in ('subject','object','relation','polarity','tense','aspect','evidence'))) for s in recs])

def answer(recs):
    return R.target([L._statement(*(s[f] for f in ('subject','object','relation','polarity','tense','aspect','evidence'))) for s in recs])

def run(label, p_reuse, p_reuse_obj, ks=(4, 8, 12, 16), n=48, seed=5):
    rng = random.Random(seed)
    # macro table: 12 most frequent codes over a large sample -> single uppercase-free letters not used by the code grammar
    sample = [code(s) for k in ks for _ in range(200) for s in records(k, rng)]
    freq = [c for c, _ in Counter(sample).most_common(12)]
    letters = [ch for ch in 'abdefjklmnoqtuvwxyz' if ch not in set('shcpfgcdri')][:12]
    table = {c: l for c, l in zip(freq, letters)}
    rows = {}
    for k in ks:
        acc = {name: [] for name in ('english','json','csv','sylang2','sylang2_sp','m_intern','m_group','m_subj','m_macro','answer','instruction')}
        for _ in range(n):
            recs = records(k, rng, p_reuse, p_reuse_obj)
            texts = dict(english=enc_english(recs), json=enc_json(recs), csv=enc_csv(recs), sylang2=enc_sylang2(recs),
                         sylang2_sp=enc_sylang2(recs, ' '), m_intern=enc_intern(recs), m_group=enc_group(recs),
                         m_subj=enc_subj(recs), m_macro=enc_macro(recs, table))
            for name, text in texts.items():
                assert text.isascii()
                acc[name].append(T.tokens(text))
            acc['answer'].append(T.tokens(answer(recs))); acc['instruction'].append(T.tokens(INSTR))
        rows[k] = {name: round(statistics.fmean(v), 1) for name, v in acc.items()}
    print(f"\n### {label} (p_reuse_subject={p_reuse}, p_reuse_object={p_reuse_obj}); mean Qwen 3 tokens of the record body, {n} batches per K")
    names = ('english','json','csv','sylang2','sylang2_sp','m_intern','m_group','m_subj','m_macro','answer','instruction')
    print('| K | ' + ' | '.join(names) + ' |'); print('|---|' + '---|' * len(names))
    for k in ks:
        print(f'| {k} | ' + ' | '.join(str(rows[k][nm]) for nm in names) + ' |')
    return rows, table



def lean_answer(recs):
    return ' '.join(f"{s['subject']} {s['object']}" for s in recs if s['polarity'] == 'negative')


def run_long(p_reuse, p_reuse_obj, length=(12, 20), ks=(4, 8, 16), n=48, seed=9):
    """Long labels (12-20 alphanumeric characters): where a dictionary can pay."""
    rng = random.Random(seed)
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
    rows = {}
    for k in ks:
        acc = {name: [] for name in ('english', 'json', 'sylang2', 'm_intern', 'm_subj', 'answer', 'lean_answer')}
        for _ in range(n):
            base = records(k, rng, p_reuse, p_reuse_obj)
            mapping = {}
            for s in base:
                for f in ('subject', 'object'):
                    if s[f] not in mapping:
                        mapping[s[f]] = ''.join(rng.choice(alphabet) for _ in range(rng.randint(*length)))
            recs = [dict(s, subject=mapping[s['subject']], object=mapping[s['object']]) for s in base]
            acc['english'].append(T.tokens(enc_english(recs))); acc['json'].append(T.tokens(enc_json(recs)))
            acc['sylang2'].append(T.tokens(enc_sylang2(recs))); acc['m_intern'].append(T.tokens(enc_intern(recs)))
            acc['m_subj'].append(T.tokens(enc_subj(recs)))
            acc['answer'].append(T.tokens(answer(recs))); acc['lean_answer'].append(T.tokens(lean_answer(recs)))
        rows[k] = {name: round(statistics.fmean(v), 1) for name, v in acc.items()}
    print(f"\n### long labels {length[0]}-{length[1]} chars (p_reuse_subject={p_reuse}, p_reuse_object={p_reuse_obj})")
    names = ('english', 'json', 'sylang2', 'm_intern', 'm_subj', 'answer', 'lean_answer')
    print('| K | ' + ' | '.join(names) + ' |'); print('|---|' + '---|' * len(names))
    for k in ks:
        print(f'| {k} | ' + ' | '.join(str(rows[k][nm]) for nm in names) + ' |')
    return rows


def run_answers(ks=(4, 8, 12, 16), n=48, seed=11):
    rng = random.Random(seed)
    print('\n### answer formats (unique identifiers): JSON list of pairs vs lean "subj obj" pairs')
    print('| K | answer (JSON) | lean answer | negatives per batch |'); print('|---|---|---|---|')
    out = {}
    for k in ks:
        a, b, neg = [], [], []
        for _ in range(n):
            recs = records(k, rng)
            a.append(T.tokens(answer(recs))); b.append(T.tokens(lean_answer(recs))); neg.append(sum(s['polarity'] == 'negative' for s in recs))
        out[k] = (round(statistics.fmean(a), 1), round(statistics.fmean(b), 1), round(statistics.fmean(neg), 2))
        print(f'| {k} | {out[k][0]} | {out[k][1]} | {out[k][2]} |')
    return out


if __name__ == '__main__':
    r1, table = run('unique identifiers (the study distribution)', 0.0, 0.0)
    r2, _ = run('repeated subjects (workflow-like)', 0.5, 0.2)
    r3, _ = run('heavily repeated (few actors)', 0.8, 0.5)
    print('\nmacro table (12 most frequent middle codes -> one letter):', table)
    OUT = {'unique': r1, 'repeated': r2, 'heavy': r3, 'macro_table': table}
    rl1 = run_long(0.0, 0.0)
    rl2 = run_long(0.5, 0.2)
    rl3 = run_long(0.8, 0.5)
    ra = run_answers()
    OUT.update({'long_unique': rl1, 'long_repeated': rl2, 'long_heavy': rl3, 'answers': ra})
    json.dump(OUT, open(ARGS.out, 'w'), indent=1)
