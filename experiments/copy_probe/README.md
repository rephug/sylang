# Copy probe: can a small byte-level model learn to copy identifiers?

Diagnostics for the finding that the project's tiny byte model learned answer
*formatting* but did not select identifiers from its input.

- **Findings and next experiments:** [`docs/research/copy-bottleneck-diagnosis-2026-10-13.md`](../../docs/research/copy-bottleneck-diagnosis-2026-10-13.md)
- **All measured numbers:** [`results/SUMMARY.md`](results/SUMMARY.md); the raw per-run JSON lines are in `results/*.jsonl`.

This directory holds no checkpoints and downloads no model weights. It needs CPU PyTorch; the rest of the repository does not.

| File | Purpose |
|---|---|
| `real_pipeline.py` | Runs the probes and one-factor training variants on the **project's own** `tiny.py`, `decode.py` and `binding.py` and its own data, read from an unpacked results packet passed with `--source` |
| `followup.py` | COPY initialisation × identifier breadth × exposure matrix and the surrogate memorisation warm start, on the packet's model (`--source`) |
| `language.py` | English-vs-Sylang comparison: `english` / `english_natural` / `words` / `codes` / `codes_eqlen` / `codes_jitter` arms rendered by the packet's codecs, filler in the negation slot, held-out identifiers and combinations, swap/flip/cue-conflict panels, leakage `audit`, byte-role `probe`, an ALiBi position option (`--suite position`), COPY-96/surrogate starts (`--init`, `--suite selection-init`), the design-v2 `study` suite (K ≤ 8, `compact` header-free arm, two held-out combinations, combo/both/reorder/similar panels, `staged`/`k1mix` curricula), a worked `example`, inference `benchmark` |
| `probes.py` | Model-agnostic checks: causality, visibility, gradient reach, parameter gradients, oracle labels, checkpoint round trip, decoder consistency, padding invariance, prompt dependence, counterfactual swap/replace |
| `check_pipeline.py` | Runs every mechanical probe on a model and prints PASS/FAIL; a template for other trainers |
| `model.py` | Independent reference: a 2-layer causal byte transformer, with three injectable defects used only to test the probes |
| `tasks.py` | Reference tasks: repeat-copy control, COPY/NONE, two-fact role selection, balanced and fresh identifier sources, and an approximate replica of the English regime |
| `train.py`, `run_suite.py`, `summarize.py` | Reference training, the sweeps behind the diagnosis, and the summary tables |
| `test_probes.py` | Each probe passes on the reference model and fails on the defect it targets |
| `tokens.py` | Qwen 3 token accounting of every arm and K: header, records, instruction and answer tokens and bytes, break-even K, compact variants |
| `combo_analysis.py` | Why learned-position models fail the held-out combination: per-combination accuracy, answer-offset seen/novel split, error types by first wrong byte |
| `patches/` | Description of the fresh-initialisation option for the project's COPY packet |

## Run

```sh
python -m venv .venv && .venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
cd experiments/copy_probe
../../.venv/bin/python -m unittest test_probes -v                      # ~2 s
../../.venv/bin/python real_pipeline.py audit --source <packet>         # 11 probes on the real code
../../.venv/bin/python real_pipeline.py train --source <packet> --task binding --data short --updates 192
../../.venv/bin/python real_pipeline.py suite --source <packet> --workers 4  # ~1 h on 4 cores
../../.venv/bin/python run_suite.py matrix --workers 4                  # reference sweeps
SYLANG_COPY_CURRICULUM=<packet>/copy/source/CURRICULUM.json ../../.venv/bin/python run_suite.py balanced12
../../.venv/bin/python followup.py surrogate --source <packet> --out <ckpt-dir>
../../.venv/bin/python followup.py copy-suite --source <packet> --surrogate <ckpt-dir>/surrogate.pt --save-copy96 <ckpt-dir>/copy96.pt
../../.venv/bin/python language.py audit --source <packet> --out results/language_audit.json
../../.venv/bin/python language.py suite --source <packet> --suite main --seeds 1 --updates 12000 --out results/language.jsonl
../../.venv/bin/python language.py suite --source <packet> --suite single --seeds 3 --updates 12000 --out results/language_k1.jsonl
../../.venv/bin/python language.py suite --source <packet> --suite position --k 1 --seeds 2 --updates 12000 --out results/language_alibi.jsonl
../../.venv/bin/python language.py suite --source <packet> --suite position --k 1234 --arms english_natural,codes --seeds 2 --updates 12000 --out results/language_alibi_main.jsonl
../../.venv/bin/python language.py suite --source <packet> --suite selection-init --copy-init <ckpt-dir>/copy96.pt --surrogate <ckpt-dir>/surrogate.pt --updates 6144 --out results/selection_init.jsonl
../../.venv/bin/python language.py suite --source <packet> --suite main --init copy96 --copy-init <ckpt-dir>/copy96.pt --arms english_natural,codes --seeds 1 --updates 12000 --out results/language_copy96.jsonl
../../.venv/bin/python language.py suite --source <packet> --suite study --k 12345678 --arms english_natural,codes,compact --position alibi --curriculum mixed --updates 10000 --seeds 3 --save-dir <ckpt-dir>/study --out results/language_study.jsonl
../../.venv/bin/python language.py probe --source <packet> <ckpt-dir>/language/*.pt   # needs --save-dir on the suites
../../.venv/bin/python followup.py copy-pretrain --source <packet> --position alibi --updates 1536 --out <ckpt-dir>/copy96_alibi.pt   # ALiBi COPY-96 init
../../.venv/bin/python language.py suite --source <packet> --suite study --k 12345678 --arms english_natural,compact --position alibi --init copy96 --copy-init <ckpt-dir>/copy96_alibi.pt --updates 10000 --out results/language_study_copy96.jsonl
../../.venv/bin/python language.py example --source <packet> --checkpoint <ckpt-dir>/study/compact-fresh-k12345678-u10000-b8-lr0.001-s0-alibi-v2.pt --k 4
../../.venv/bin/python language.py breakdown --source <packet> --checkpoint <ckpt-dir>/study/compact-fresh-k12345678-u10000-b8-lr0.001-s0-alibi-v2-devbest.pt --out results/breakdown_compact_s0.json
../../.venv/bin/python combo_analysis.py --source <packet> --out results/combo_analysis.json <ckpt-dir>/selection_init/*.pt <ckpt-dir>/language_alibi/*.pt
../../.venv/bin/python tokens.py --source <packet> --tokenizer <qwen3-tokenizer.json> --out results/tokens_qwen3.json
../../.venv/bin/python language.py benchmark --source <packet> --k 12345678 --out results/language_benchmark_k8.json   # on an idle CPU
../../.venv/bin/python tokens3.py --source <packet> --tokenizer <qwen3-tokenizer.json> --out results/tokens3_qwen3.json --example   # design v3 arms, K 1..16
../../.venv/bin/python language.py suite --source <packet> --suite study3 --corpus all --arms sylang2 --position alibi --curriculum decay --updates 10000 --seeds 1 --save-dir <ckpt-dir>/study3 --out results/study3/pilot-sylang2-all.jsonl   # corpus pilot, one per corpus
../../.venv/bin/python language.py suite --source <packet> --suite study3 --corpus <chosen> --arms english,sylang2 --position alibi --curriculum decay --updates 10000 --seeds 3 --save-dir <ckpt-dir>/study3 --out results/study3/decisive.jsonl
../../.venv/bin/python language.py benchmark --source <packet> --design v3 --k 1,2,4,8,12,16 --out results/language_benchmark_v3.json   # on an idle CPU
SYLANG_PACKET=<packet> ../../.venv/bin/python -c "import sys; sys.path.insert(0, '.'); import unittest; unittest.main(module='test_sylang_batch', argv=['t'])"   # codec tests incl. packet parity
../../.venv/bin/python demo_handoff.py --source <packet> --tokenizer <qwen3-tokenizer.json> --design v2 --k 12345678 --checkpoint sylang2=<ckpt-dir>/study/compact-fresh-k12345678-u10000-b8-lr0.001-s0-alibi-decay-v2-devbest.pt --checkpoint english=<ckpt-dir>/study/english_natural-fresh-k12345678-u10000-b8-lr0.001-s0-alibi-decay-v2-devbest.pt --out results/demo_handoff_v2.json --examples results/demo_handoff_v2_examples.txt
../../.venv/bin/python language.py suite --source <packet> --suite study3 --corpus all --arms json,csv --position alibi --curriculum decay --updates 10000 --seeds 2 --save-dir <ckpt-dir>/study3 --out results/study3/decisive-machine.jsonl   # + sylang1 --seeds 1; targeted expansion: --corpus all16
../../.venv/bin/python eval3.py --source <packet> --checkpoint <ckpt-dir>/study3/<run>.pt --lengths v2 --out results/study3_eval/<run>.json   # score a saved v3 model on fixed-shape identifiers (or --lengths v3, --k-long 13-16 --longer 17-20)
../../.venv/bin/python demo_handoff.py --source <packet> --tokenizer <qwen3-tokenizer.json> --design v3 --k 1,2,3,4,5,6,7,8,9,10,11,12 --checkpoint sylang2=<ckpt-dir>/study3/sylang2-...-s2-alibi-decay-v3-all-devbest.pt --checkpoint english=<ckpt-dir>/study3/english-...-s2-alibi-decay-v3-all-devbest.pt --checkpoint json=... --checkpoint sylang1=... --out results/demo_handoff_v3.json --examples results/demo_handoff_v3_examples.txt
../../.venv/bin/python summarize.py results > results/SUMMARY.md
```

## Sylang batch codec (v1 with header, v2 header-free)

`sylang_batch.py` is an independent implementation of the SYLANG-BATCH wire format written from the
format alone (letter tables, defaults and escaping derived empirically; the packet parity test checks
byte identity on 3,000 random batches). `encode(records, version=1)` is the existing format; `version=2`
is the **opt-in header-free format**: identical record lines, no header line (`marker=True` adds a
`#sylang2` first line for 6 Qwen 3 tokens instead of the header's 15). `decode(text)` accepts both and
returns the records plus `{version, style, records, marker}`, so existing inputs keep working.

```sh
printf '%s\n' '{"subject":"k4m","relation":"contain","polarity":"negative","tense":"past","object":"f9b"}' | python -I sylang_batch.py encode --version 2
python -I sylang_batch.py decode < batch.txt            # v1 or v2 in, JSONL records out
python -I sylang_batch.py convert --to 2 < batch_v1.txt  # re-encode in the other version
```

Not supported (raise): interning (`@` references, N > 0), `if` records, policies other than `selective`.

## Design v3 (`study3.py`) and the handoff demo

`language.py --suite study3` trains the five arms `english`, `sylang2`, `sylang1`, `json`, `csv` on the
same items (identifiers of 2–5 characters from a hash split, six request wordings of which two are held
out, K 1..12 trained and 13..16 held out) and scores the panels `iid`, `wording`, `combo`, `long`, `full`
(everything held out at once) and the counterfactuals; `--corpus` selects the training corpus variant
for the pilot. `tokens3.py` counts Qwen 3 tokens for the same arms. `demo_handoff.py` is a working
producer-to-consumer task handoff measured at every stage (conversion in, transport tokens, the trained
byte model's answer, conversion out, lossless round trip for the machine formats); the doc's §10 has
the pre-registration and results.

`<packet>` is the unpacked `sylang-source-and-results-20261013` directory. It must contain `baseline/source/{runtime/training,rows.jsonl,plan.json}` and `copy/source/{train.jsonl,evaluation.jsonl,SCHEDULE.json}`.

## Pointing the probes at another trainer

`check_pipeline.check_pipeline` and `real_pipeline.audit` take callables. Each adapter must call the codebase's *own* code path:

- the training forward pass;
- the exact loss;
- the real collate function;
- the evaluation decoder;
- the real checkpoint loader.

Run them on a fresh initialisation and on the evaluated checkpoint.

For the oracle-label probe, pass `sequences=` when inputs omit the final target token (`x = seq[:-1]`).

Gradient reach is the probe that finds stop-gradient defects. Forward probes, optimiser coverage and per-parameter gradient checks all miss them: in the `detach_kv` defect, the fused QKV weight still receives gradient through Q.
