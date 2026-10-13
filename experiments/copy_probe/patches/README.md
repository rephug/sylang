# Fresh-initialisation option for the project's COPY packet

The packet's training source is not in this public repository. The ready-made
diff was therefore delivered privately to the project owner and is not
published here. The change is small enough to describe:

1. **Plan validation.** In the COPY trainer's `validate`, accept a new plan field `"initialization": "fresh"` as an alternative to the warm-start checkpoint pin. Reject any other value. The field becomes part of the approved plan digest, so a fresh run cannot be launched under a warm-start approval, or the reverse.
2. **Worker: checkpoint handling.** When the field is `"fresh"`, skip the checkpoint size and hash checks, the read and the `load_state_dict`. The model is then exactly `tiny.build_model(RECIPE)` under the seed the worker already sets.
3. **Worker: provenance.** Record `initialized_from: "fresh"` instead of the checkpoint hash in `training.json`.

Rows, schedule, optimiser, budget, guards, telemetry and scoring are unchanged.

The launcher's outer `worker.py` still verifies the warm-start file's hash before calling the worker. This is harmless, because the file is checked but not loaded.

## Evidence

The diff was applied to the packet's own trainer and run here with only the native admission check stubbed. That check needs the guardian, the mutex and `prctl`. The run, at 192 updates and batch 4, passed the packet's strict gate:

- held-out COPY 12/12, with all 36 held-out bytes correct;
- NONE 12/12;
- all 10 contrasts passed.

See [`../results/patched_copy_worker_scores.json`](../results/patched_copy_worker_scores.json) and the diagnosis note, §1 and §4.

Native follow-up results reported by the owner show that a warm start does *not* prevent copying once training uses 96 balanced identifiers. Fresh initialisation is one sufficient fix for the 12-identifier design, not a requirement (diagnosis note, §1).

The same three edits apply to the binding trainer. On their own they do not fix the binding task, which also needs a shorter instruction or a larger budget.
