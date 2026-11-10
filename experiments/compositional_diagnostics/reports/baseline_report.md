# Compositional diagnostics: CPU baseline report

Data `cd-v0` (seed 0; 600 test and 240 dev families, six members each; `families.jsonl` sha256 `d593274041bc4314…`). No model was run: every row is a deterministic program (`solvers.py`). Intervals are 95% percentile bootstraps over families (stratified by question type for "all"), 2,000 resamples, seed 0.

Every solver reads the rendered records of each arm and produced identical outputs in all four arms (english, json, sylang2, sylang2g): True. The table is therefore one row per solver.

## Pooled test results (all question types)

| solver | exact | family | relevant | robust_all | robust_irrelevant | robust_rename | robust_reorder | robust_distractor | raw_invariance_reorder (uncredited) |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| random_guess | 0.15 [0.14, 0.16] | 0.00 [0.00, 0.00] | 0.04 [0.02, 0.06] | 0.00 [0.00, 0.01] | 0.04 | 0.06 | 0.05 | 0.05 | 0.16 |
| constant_majority | 0.22 [0.19, 0.24] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.22 [0.19, 0.25] | 0.22 | 0.22 | 0.22 | 0.22 | 1.00 |
| first_record | 0.19 [0.17, 0.21] | 0.01 [0.00, 0.01] | 0.09 [0.07, 0.11] | 0.04 [0.02, 0.05] | 0.18 | 0.18 | 0.05 | 0.12 | 0.28 |
| last_record | 0.17 [0.15, 0.19] | 0.01 [0.00, 0.01] | 0.07 [0.04, 0.09] | 0.03 [0.02, 0.05] | 0.16 | 0.16 | 0.05 | 0.11 | 0.29 |
| ignore_tense | 0.55 [0.52, 0.57] | 0.47 [0.45, 0.49] | 0.47 [0.45, 0.49] | 0.54 [0.51, 0.56] | 0.54 | 0.54 | 0.54 | 0.54 | 1.00 |
| ignore_aspect | 0.70 [0.68, 0.73] | 0.63 [0.61, 0.65] | 0.63 [0.61, 0.65] | 0.70 [0.67, 0.73] | 0.70 | 0.70 | 0.70 | 0.70 | 1.00 |
| ignore_evidence | 0.68 [0.66, 0.71] | 0.63 [0.61, 0.66] | 0.63 [0.61, 0.66] | 0.68 [0.66, 0.71] | 0.68 | 0.68 | 0.68 | 0.68 | 1.00 |
| ignore_polarity | 0.71 [0.69, 0.73] | 0.68 [0.65, 0.70] | 0.68 [0.65, 0.70] | 0.69 [0.67, 0.72] | 0.69 | 0.69 | 0.69 | 0.69 | 1.00 |
| ignore_relation | 0.80 [0.78, 0.82] | 0.67 [0.65, 0.69] | 0.80 [0.78, 0.82] | 0.67 [0.65, 0.69] | 0.83 | 0.83 | 0.83 | 0.67 | 1.00 |
| subject_only | 0.65 [0.62, 0.67] | 0.44 [0.42, 0.47] | 0.64 [0.60, 0.67] | 0.45 [0.42, 0.47] | 0.64 | 0.64 | 0.48 | 0.55 | 0.67 |
| object_only | 0.66 [0.64, 0.69] | 0.42 [0.40, 0.44] | 0.67 [0.64, 0.70] | 0.42 [0.40, 0.44] | 0.67 | 0.67 | 0.49 | 0.54 | 0.66 |
| feature_bag | 0.83 [0.83, 0.83] | 0.83 [0.83, 0.83] | 0.83 [0.83, 0.83] | 0.83 [0.83, 0.83] | 0.83 | 0.83 | 0.83 | 0.83 | 1.00 |
| unbound_join | 0.87 [0.86, 0.88] | 0.83 [0.83, 0.83] | 0.83 [0.83, 0.83] | 0.88 [0.86, 0.89] | 0.88 | 0.88 | 0.88 | 0.88 | 1.00 |
| role_swap | 0.22 [0.19, 0.24] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.22 [0.19, 0.25] | 0.22 | 0.22 | 0.22 | 0.22 | 1.00 |
| *uniform-guess chance* | 0.155 | 5.0e-04 | 0.047 | 1.5e-03 | 0.047 | 0.047 | 0.047 | 0.047 | – |

The constant answerer is perfectly invariant in its raw outputs (last column) and still scores 0 on `relevant` and `family`: invariance is credited only together with correct answers.

## Per question type: family consistency (all six members correct)

| solver | tense | aspect | evidence | profile | filter | join |
|---|---|---|---|---|---|---|
| oracle | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| random_guess | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| constant_majority | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| first_record | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| last_record | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| ignore_tense | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.28 [0.19, 0.37] | 0.54 [0.44, 0.64] | 0.00 [0.00, 0.00] |
| ignore_aspect | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.31 [0.22, 0.40] | 0.47 [0.37, 0.57] | 1.00 [1.00, 1.00] |
| ignore_evidence | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.26 [0.18, 0.35] | 0.54 [0.44, 0.64] | 1.00 [1.00, 1.00] |
| ignore_polarity | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.35 [0.26, 0.45] | 0.70 [0.61, 0.79] | 0.00 [0.00, 0.00] |
| ignore_relation | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.26 [0.18, 0.35] | 0.75 [0.66, 0.83] | 0.00 [0.00, 0.00] |
| subject_only | 0.14 [0.08, 0.21] | 0.15 [0.08, 0.22] | 0.17 [0.10, 0.25] | 0.20 [0.12, 0.28] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| object_only | 0.14 [0.07, 0.21] | 0.13 [0.07, 0.20] | 0.11 [0.05, 0.18] | 0.13 [0.07, 0.20] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| feature_bag | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| unbound_join | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| role_swap | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| *chance* | 1.4e-03 | 1.4e-03 | 2.4e-04 | 9.8e-15 | 8.1e-11 | 2.2e-17 |

## Per question type: exact accuracy over all members

| solver | tense | aspect | evidence | profile | filter | join |
|---|---|---|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| random_guess | 0.34 | 0.30 | 0.24 | 0.00 | 0.01 | 0.00 |
| constant_majority | 0.33 | 0.33 | 0.25 | 0.00 | 0.15 | 0.24 |
| first_record | 0.33 | 0.32 | 0.22 | 0.14 | 0.04 | 0.06 |
| last_record | 0.29 | 0.29 | 0.23 | 0.10 | 0.08 | 0.01 |
| ignore_tense | 0.33 | 1.00 | 1.00 | 0.34 | 0.55 | 0.05 |
| ignore_aspect | 1.00 | 0.33 | 1.00 | 0.40 | 0.47 | 1.00 |
| ignore_evidence | 1.00 | 1.00 | 0.25 | 0.31 | 0.54 | 1.00 |
| ignore_polarity | 1.00 | 1.00 | 1.00 | 0.47 | 0.70 | 0.08 |
| ignore_relation | 1.00 | 1.00 | 1.00 | 0.29 | 0.76 | 0.77 |
| subject_only | 0.52 | 0.45 | 0.46 | 0.47 | 1.00 | 1.00 |
| object_only | 0.48 | 0.54 | 0.45 | 0.50 | 1.00 | 1.00 |
| feature_bag | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| unbound_join | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.23 |
| role_swap | 0.33 | 0.33 | 0.25 | 0.00 | 0.15 | 0.24 |
| *chance* | 0.333 | 0.333 | 0.250 | 0.005 | 0.011 | 0.001 |

## Targeted checks

Each shortcut must be exposed by the metric named for it (target), and must not be penalised where it is harmless (specificity).

| kind | solver | scope | metric | bound | value | pass |
|---|---|---|---|---|---|---|
| target | constant_majority | all | relevant | <= 0.0 | 0.000 | yes |
| target | constant_majority | all | family | <= 0.0 | 0.000 | yes |
| target | random_guess | all | family | <= 0.01 | 0.000 | yes |
| target | first_record | all | family | <= 0.1 | 0.005 | yes |
| target | last_record | all | family | <= 0.1 | 0.005 | yes |
| target | ignore_tense | tense | relevant | <= 0.0 | 0.000 | yes |
| target | ignore_tense | join | family | <= 0.1 | 0.000 | yes |
| target | ignore_aspect | aspect | relevant | <= 0.0 | 0.000 | yes |
| target | ignore_evidence | evidence | relevant | <= 0.0 | 0.000 | yes |
| target | ignore_polarity | join | family | <= 0.1 | 0.000 | yes |
| target | ignore_polarity | profile | family | <= 0.6 | 0.350 | yes |
| target | ignore_relation | join | family | <= 0.1 | 0.000 | yes |
| target | ignore_relation | profile | family | <= 0.5 | 0.260 | yes |
| target | subject_only | tense | family | <= 0.35 | 0.140 | yes |
| target | subject_only | aspect | family | <= 0.35 | 0.150 | yes |
| target | subject_only | evidence | family | <= 0.35 | 0.170 | yes |
| target | subject_only | profile | family | <= 0.35 | 0.200 | yes |
| target | object_only | tense | family | <= 0.35 | 0.140 | yes |
| target | object_only | aspect | family | <= 0.35 | 0.130 | yes |
| target | object_only | evidence | family | <= 0.35 | 0.110 | yes |
| target | object_only | profile | family | <= 0.35 | 0.130 | yes |
| target | feature_bag | filter | exact_base | <= 0.0 | 0.000 | yes |
| target | unbound_join | join | family | <= 0.05 | 0.000 | yes |
| target | role_swap | all | family | <= 0.0 | 0.000 | yes |
| specificity | ignore_aspect | tense | family | >= 1.0 | 1.000 | yes |
| specificity | ignore_aspect | evidence | family | >= 1.0 | 1.000 | yes |
| specificity | ignore_aspect | join | family | >= 1.0 | 1.000 | yes |
| specificity | ignore_evidence | tense | family | >= 1.0 | 1.000 | yes |
| specificity | ignore_evidence | join | family | >= 1.0 | 1.000 | yes |
| specificity | ignore_tense | aspect | family | >= 1.0 | 1.000 | yes |
| specificity | ignore_polarity | tense | family | >= 1.0 | 1.000 | yes |
| specificity | ignore_relation | evidence | family | >= 1.0 | 1.000 | yes |
| specificity | subject_only | filter | family | >= 1.0 | 1.000 | yes |
| specificity | feature_bag | join | family | >= 1.0 | 1.000 | yes |
| specificity | unbound_join | filter | family | >= 1.0 | 1.000 | yes |

Oracle scores 1.0 on every credited metric: True. All checks pass: True.

## Oracle verification

| domain | cases | evaluations (oracle vs reference vs solver path, both record orders) | mismatches |
|---|---|---|---|
| D1 | 1,296 | 733,536 | 0 |
| D2 | 15,360 | 3,932,160 | 0 |
| D3 | 10,240 | 983,040 | 0 |

Domains are defined in `verify_oracle.py`; hand-checked examples are in `tests/test_handchecked.py`.

## Audits (test split unless noted)

### Answer balance

| qtype | majority share | per-value counts or answer-set sizes | empty share |
|---|---|---|---|
| tense | 0.340 | {'future': 204, 'past': 201, 'present': 195} | – |
| aspect | 0.333 | {'completed': 200, 'progressive': 200, 'simple': 200} | – |
| evidence | 0.253 | {'direct': 152, 'inferred': 149, 'reported': 149, 'unspecified': 150} | – |
| profile | 0.025 | 133 distinct profiles | – |
| filter | 0.152 | {0: 91, 1: 199, 2: 198, 3: 101, 4: 11} | 0.1517 |
| join | 0.245 | {0: 147, 1: 282, 2: 158, 3: 13} | 0.245 |

### Identifier and near-duplicate overlap

- identifiers: dev 3,846, test 9,259; shared between dev and test: 0; in the pilot's training hash bucket: 0 (dev ids are all in the pilot's dev bucket, test ids in its test bucket). Identifiers that also occur in the pilot's frozen v1 cases (same buckets, so overlap is possible): dev_vs_v1_dev 329, dev_vs_v1_test 0, test_vs_v1_dev 0, test_vs_v1_test 1191.
- structural signatures (invariant to renaming and reordering) of every member: dev 960, test 2,400, shared across splits: 0.

### Position cues

| qtype | answer uses the first record | answer uses the last record | queried record position by quarter of the case |
|---|---|---|---|
| tense | 0.148 | 0.080 | [141, 148, 183, 128] |
| aspect | 0.133 | 0.090 | [144, 140, 179, 137] |
| evidence | 0.063 | 0.102 | [100, 177, 158, 165] |
| profile | 0.128 | 0.095 | [135, 165, 161, 139] |
| filter | 0.223 | 0.258 | – |
| join | 0.217 | 0.200 | – |

### Length cues

A lookup table (question type, rendered-length bucket of 20 characters, K) -> majority answer, fitted on dev and scored on test, against the question type's dev majority; and how often a relevant or an irrelevant edit changes the rendering's length (a within-family signal a model never sees, since items are presented one at a time).

| arm | qtype | length lookup | majority | relevant edits change length | irrelevant edits change length |
|---|---|---|---|---|---|
| english | tense | 0.303 | 0.325 | 0.87 | 0.96 |
| english | aspect | 0.323 | 0.333 | 0.89 | 1.00 |
| english | evidence | 0.262 | 0.250 | 1.00 | 0.96 |
| english | profile | 0.022 | 0.000 | 0.91 | 0.95 |
| english | filter | 0.068 | 0.152 | 0.91 | 0.95 |
| english | join | 0.128 | 0.245 | 0.95 | 0.93 |
| json | tense | 0.307 | 0.325 | 1.00 | 0.95 |
| json | aspect | 0.313 | 0.333 | 1.00 | 0.95 |
| json | evidence | 0.282 | 0.250 | 0.84 | 1.00 |
| json | profile | 0.005 | 0.000 | 0.99 | 0.98 |
| json | filter | 0.073 | 0.152 | 0.98 | 0.94 |
| json | join | 0.158 | 0.245 | 1.00 | 0.93 |
| sylang2 | tense | 0.297 | 0.325 | 0.63 | 0.23 |
| sylang2 | aspect | 0.357 | 0.333 | 0.68 | 0.33 |
| sylang2 | evidence | 0.252 | 0.250 | 0.50 | 0.30 |
| sylang2 | profile | 0.007 | 0.000 | 0.38 | 0.42 |
| sylang2 | filter | 0.108 | 0.152 | 0.45 | 0.38 |
| sylang2 | join | 0.220 | 0.245 | 0.27 | 0.55 |
| sylang2g | tense | 0.327 | 0.325 | 0.63 | 0.23 |
| sylang2g | aspect | 0.353 | 0.333 | 0.68 | 0.33 |
| sylang2g | evidence | 0.235 | 0.250 | 0.50 | 0.30 |
| sylang2g | profile | 0.008 | 0.000 | 0.38 | 0.42 |
| sylang2g | filter | 0.088 | 0.152 | 0.45 | 0.38 |
| sylang2g | join | 0.215 | 0.245 | 0.27 | 0.55 |

Mean rendered characters per test item: english 299.4, json 487.4, sylang2 92.3, sylang2g 81.6.
