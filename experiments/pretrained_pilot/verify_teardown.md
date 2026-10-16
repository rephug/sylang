# Teardown checklist (provider side; recorded before any stage is called finished)

A stage of the pilot is **not finished** when its script prints its last line. It is finished when
the owner has done every item below on the provider's side and the ledger holds the `balance` line
that proves it. The stage scripts (`run_stage_*.sh`) never call a provider API: starting and stopping
the machine is the owner's action, and this page is what "stopping" means.

Two facts that this checklist exists for:

- **A training timeout or a finished script is not proof that billing stopped.** The script's
  `timeout` ends a process; the instance keeps billing until the provider confirms it is gone.
- **A stopped instance can still bill for storage.** Every provider below charges for disks,
  volumes, snapshots or studios that outlive the GPU session; only deletion ends it.

## Generic checklist (every provider, every stage)

Copy this block into the stage's notes and tick each line with the time and the evidence.

- [ ] **The results are off the machine.** The stage script packaged `results/`, `analysis/`,
      `runs/*/best`, `runs/*/run.json`, `runs/*/train_log.jsonl`, `profile.json` and `ledger.jsonl`
      into `runs/_outbox/stage_<x>_<ts>.tar.gz` and either uploaded it (`PILOT_UPLOAD_CMD`) or told
      you to copy it. Verify the copy on the development server (`tar tzf`, then the
      `results/*/eval.json` count) **before** the instance is destroyed; afterwards it is gone.
- [ ] **No running instance in the console.** Open the provider's instance/pod/app list in the
      web console: nothing of this account is running, pending, paused or "stopped but retained"
      unless listed below with its storage price.
- [ ] **No running instance via the CLI or API.** The console can lag; repeat the check with the
      provider's own CLI (commands below) and read the empty list.
- [ ] **No retained volume, disk, snapshot, studio or image** unless it is listed in the stage
      notes with its size in GB and its monthly price, and a date by which it will be deleted
      (adapters are ~10 MB each and the results are kilobytes: there is no reason to keep a
      GPU-side disk beyond the stage).
- [ ] **The final balance is recorded** in the ledger from the provider's billing page, with the
      provider's own charge for the session (this corrects the script's `elapsed × price` estimate):

      python ledger.py add --event balance --provider <provider> --stage <A|B|C> \
          --balance-after <USD left on the account> --measured-spend <USD the provider charged for this session> \
          --note "billing page <date> <time UTC>: instance gone, no volumes"

      and `python ledger.py report --ceiling 100` prints `open start events without a matching
      stop: 0` and no `INSTANCE MAY STILL BE BILLING` line.
- [ ] **The API token is revoked or rotated** after the stage (every provider lets you delete the
      key in its settings page), and the shell that held `PILOT_PROVIDER_TOKEN` is closed.
- [ ] **A second look the next day**: the billing page shows no charge dated after the teardown
      time. Providers post storage and threshold charges with a delay; a surprise here goes into
      the ledger as a `note` and a corrected `balance` line.

## Secrets rule

Provider tokens live only in the environment variable `PILOT_PROVIDER_TOKEN` of the owner's shell
(and in the provider CLI's own keychain, where the CLI stores it after `login`). They are **never**
pasted into a chat, **never** written to a file in this repository, **never** put into
`ledger.jsonl` (`ledger.py` refuses a line that contains the value of `PILOT_PROVIDER_TOKEN` or
anything shaped like a bearer token or API key), never passed on a command line that a shell
history or a process list can show, and never given to the stage scripts, which do not read it.
`PILOT_UPLOAD_CMD` must be credential-free (use rclone's config or ssh keys), because it runs
through a shell and its program name is recorded in `profile.json`.
When a token has been exposed anywhere, revoke it first and investigate second.

## Modal

- Modal bills per second of container run time plus storage; a **Workspace budget and spend
  limit** can be set at `/settings/usage` (Settings → Usage). Set the limit to the stage's share of
  the ceiling before the first deployment; this is the only hard stop Modal gives.
- **Volumes bill above the free tier**; an adapter or results volume left behind costs money every
  month. `modal volume list` must be empty (or listed in the notes with GB and price), and
  `modal volume delete <name>` removes one.
- Functions can be made **preemptible** (cheaper, can be interrupted). A preempted training run is
  what the `--resume` path of `train_lora.py` is for; the ledger note records the preemption.
- Stop everything: `modal app list` shows the deployed/ephemeral apps; `modal app stop <app-id>`
  stops one; repeat `modal app list` until nothing runs. A detached run (`modal run --detach`)
  keeps billing after your terminal closes: look for it explicitly.
- Record the balance from Settings → Usage (credits used this period), then revoke the token at
  Settings → API tokens.

## Lightning AI

- A **Studio auto-sleeps after 10 minutes only if nothing is running** inside it; a process that
  survived your SSH session (a `nohup`, a `tmux`, a stuck evaluator) keeps the GPU attached and
  billing. Stop the Studio explicitly in the console (Studio → Stop) and confirm that the status is
  "Stopped", then check `lightning` CLI output or the Studios page for anything still "Running".
- **Storage above 10 GB bills daily**, including while the Studio is stopped. A stopped Studio with
  the model weights (1.5 GB for the 0.6B, 4.1 GB for the 1.7B), the venv (several GB with the CUDA
  torch) and the data can exceed it: delete the Studio when the stage's results are safely copied,
  or record its size and daily price in the stage notes.
- Check the **usage page** (Billing → Usage) for the GPU-hours and storage charged for the session,
  record them with the balance line, and revoke the API key under Settings → Keys.

## Thunder Compute

- Thunder Compute has **no Stop feature**: an instance either runs (and bills) or is deleted. To
  pause work the sequence is **snapshot → delete → restore**; "billing stops immediately when an
  instance is deleted".
- **Snapshots bill $0.05/GB-month until deleted.** A snapshot of a machine with the venv and the
  weights is tens of GB; delete it as soon as the restore is no longer needed, and list any kept
  snapshot with its GB and monthly price in the stage notes.
- Charges are **threshold charges with no hard cap**: the card is charged whenever usage crosses a
  threshold, and there is no spend limit to set. The only cap is the owner's discipline: the stage
  script's `timeout` values, the ledger, and this checklist done the same day.
- Verify in the console that the instance list is empty and the snapshot list is empty (or
  accounted for); then record the balance and revoke the key.

## Runpod

- **Stopped pods still bill the volume disk at $0.20/GB-month** (a running pod's volume is
  $0.10/GB-month); a pod that is "stopped" to save GPU money is still a monthly charge. The only
  state that bills nothing is **terminated**.
- **Network volumes bill $0.07/GB-month** independently of any pod; the pilot's 20 GB network
  volume is $1.40/month and is deleted when the pilot report is written (listed until then with
  that price). `runpodctl` or the console's Storage page shows them.
- **Terminate** the pod (not stop), confirm in the console that Pods is empty, confirm with
  `runpodctl get pod` (empty), **then check the Billing page**: the per-second GPU charge, the
  disk charges and the current balance. Record the balance line from that page.
- Keep auto-pay off (prepaid credits only), so a mistake here costs at most the balance.
  Revoke the API key under Settings → API Keys.

## Vast.ai

- **Storage bills every second the instance exists in any state**: stopped instances keep paying
  the host's storage rate (set per offer, on the instance card), and the "stop" button therefore
  does not end spending. **Destroy, not stop**: Instances → Destroy, then confirm the list is empty
  in the console and with `vastai show instances` (empty).
- Read the offer's storage and bandwidth prices before renting; record them in the ledger start
  line's `note` so the balance line can be reconciled.
- At zero balance Vast stops the instance but storage keeps accruing until the grace period ends,
  so an exhausted balance is not a teardown either.
- Check Billing for the final charges and the balance, record the balance line, delete the API key
  under Account → API Keys.

## When the owner cannot confirm

If the console or the CLI cannot be reached, the stage is **not finished**: write a `note` line
(`ledger.py add --event note --note "teardown unconfirmed: <reason>"`), keep the instance id, and
repeat the check as soon as possible. A `start` line without a `stop` line is flagged by
`ledger.py report` on every run until it is resolved.
