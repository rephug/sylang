#!/usr/bin/env bash
# Regenerates every committed artefact of the compositional diagnostics benchmark on a CPU (about 3 minutes).
set -euo pipefail
cd "$(dirname "$0")"
python3 -I generate.py --force                 # data/cd-v0/ (default config: seed 0, 40 dev and 100 test families per type)
python3 -I generate.py --check                 # regenerate in memory, compare every file hash
cp data/cd-v0/manifest.json manifest_cd-v0.json
python3 -I analyze_existing.py                 # reports/existing_tasks.{json,md}
python3 -I verify_oracle.py                    # reports/oracle_verification.json
python3 -I baseline.py                         # reports/baseline.json, reports/baseline_report.md
python3 -I make_samples.py                     # samples/families_sample.md
python3 -I -m unittest discover -s tests
