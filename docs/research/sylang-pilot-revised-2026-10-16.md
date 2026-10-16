# Sylang pretrained-model pilot, revised (2026-10-16)

A comparison of the evidence assessment of 2026-10-16 (`sylang-evidence-assessment-and-plan-2026-10-16.md`, "the assessment") with the second-opinion plan "Sylang: a fast, evidence-led development plan" (16 October 2026, "the second opinion"), and one revised, staged pilot that replaces the assessment's §3–§4. Prepared on the development server. **Nothing has been rented, downloaded from a model hub, or spent; Git history is unchanged; no output format has been adopted.** The owner's priority order applies throughout: machine efficiency and an improved Sylang M first, readable Prime later.

Sections: 0 plain-English summary · 1 where the two plans agree, where they differ and why · 2 the seven concerns, answered with mechanisms · 3 the revised pilot (stages, cells, decision rules, budget) · 4 provider choice · 5 owner setup steps · 6 what is prepared on the server · 7 the next approval.

---

## 0. The summary in plain English

The two plans want the same experiment: fine-tune one small open model four ways on identical synthetic cases (compact Sylang codes, an improved-M candidate, controlled English, compact JSON), grade every output with one deterministic grader, count whole requests with the model's own tokenizer, and judge on correct work per dollar and per second. They differ on *what the first test case is*, *how much to run before looking*, *how to read the result*, and *where to run it*.

What this revision does, in one sentence each:

- **Keeps the study's record schema and adds one second task** (relationship lookup) rather than switching to a GBG task-status schema now; the GBG schema is the integration stage's job, after the format question has an answer.
- **Stages the sweep**: profile the whole cycle on the GPU first (stage A), then one matched four-arm comparison with one seed and one answer format (stage B, exploratory by rule), then three seeds and the answer-format factor (stage C), with the decision rule written before any run.
- **Replaces "tied" with a non-inferiority rule** (δ = 3 points, 90% cluster bootstrap over seeds then items), under which an interval that reaches both acceptable and unacceptable differences is *inconclusive*, never a pass.
- **Treats the improved-M candidate as one reversible change** (grouping records by subject), chosen because the pilot data have repeated subjects, where grouping is the only M idea with a measured token gain (−20% to −29% at K = 16); the dictionary costs tokens on short identifiers and macro codes save 1%.
- **Makes every runtime and cost figure a measurement**: the package profiles load, training, generation, checkpointing and upload on the actual GPU and projects the later stages from that, with failed runs and retries inside a 1.5× contingency and a US$100 all-in ceiling.
- **Counts the whole request**: legend (system message), instruction, records, chat-template overhead, output, the retry, parsing and conversion wall-clock, under a cached and an uncached convention side by side, converted to latency and cost per correct task.
- **Recommends a free-credit provider first**: Lightning AI's L4 Studio (5 signup credits cover stages A and B; 25 more after adding a card cover stage C), with Thunder Compute's RTX A6000 at US$0.35 per hour as the paid fallback; Runpod only on an actual quote.

**Status.** The reproducible package (`experiments/pretrained_pilot/`) is written and smoke-tested on the CPU server with a randomly initialised Qwen 3 of the real architecture (section 6 has the exact state). **The next approval needed** is in section 7: stage A + B on Lightning's free credits (no payment method), with the Qwen3-0.6B weights downloaded on the rented machine only.

---

## 1. Where the two plans agree, where they differ, and why

### 1.1 Agreement (no change needed)

Freeze a contract before training; test the evaluator with deliberately corrupted outputs; one JSON target and one grader for every rendering; split by underlying case and hold out wordings, feature combinations and a length band; audit complete requests with the unchanged tokenizer; profile before committing GPU money; report every seed, failure and retry; distinguish equal-items from equal-compute comparisons; no custom vocabulary and no Prime training in this pilot; GBG integration later, behind a flag, and never a prerequisite. The second opinion's reading of the evidence (two of five seeds learned for both English and Sylang; 93/96 and 92/96 on the handoff demo; the 10,275 vs 14,927 transport tokens are the v2.1 demo's figure, the v3 demo's process-token ratio is 0.69×; the history question) matches the assessment's §1 and §7.

### 1.2 Disagreements, with reasons

1. **The first-use contract.** The second opinion freezes a *GBG* contract (task status, dependencies, acceptance facts, contradictory updates, nested dependencies, distracting text) as the pilot's schema. I disagree for the pilot: that schema does not exist in any codec yet, nesting and free text are exactly the features the compact codes have not been designed for, and a new schema breaks the only paired evidence the project has (three-seed byte-model results and the v3 handoff demo on the seven-field record). The pilot's question is narrower and answerable in days: *does a pretrained model read the compact codes as well as JSON and English, and what does that cost per correct answer?* The GBG schema is the right next contract **after** that answer, and the package's codec and grader are written so that a second schema is an additional data module, not a rewrite. The second task the pilot adds (lookup of one record's relation and polarity by its identifier pair) is chosen because it exercises reading a specific record rather than filtering the list, which is where the byte model failed first (identifier copying), and because it costs nothing: the same cases carry both tasks.
2. **How much of M to change at once.** The second opinion asks for several reversible M improvements, tested individually. The assessment measured the candidates under the Qwen 3 tokenizer (§2.1): the dictionary *costs* 14–29% on identifiers of 2–5 characters and pays only on 12–20-character labels; macro codes save 1%; spaces instead of tabs save 6% of record tokens at every distribution; grouping by subject saves 3% on unique subjects and 20–29% at K = 16 when half the records reuse a subject. Only grouping has a model-quality question worth a GPU (the model must learn a different line grammar). So the improved-M candidate is grouping alone, on data with repeated subjects, and the pilot reports what it adds over the current compact format. Spaces and the dictionary are tokenizer-arithmetic results: they go into the codec as options when the deployment path needs them, each still subject to a short accuracy check at that time, not into this pilot's cell count.
3. **The gate.** The second opinion's gate is "improved M no more than two percentage points below the best of the other three, and at least a 15% cost or latency improvement". A point estimate within two points can sit inside an interval that also contains a ten-point loss, so the bound has to be on the interval, not the estimate. The pilot's rule (section 3.4) is non-inferiority with δ = 3 on a 90% cluster-bootstrap interval, an *inconclusive* verdict when the interval straddles −δ, and the 15% whole-request cost criterion under both cache conventions. The assessment's earlier wording ("tied") is withdrawn.
4. **Dataset size.** 5,000–20,000 training cases is more than a pilot needs to learn the shape of the learning curve; the package uses nested prefixes of 500, 2,000 and 8,000 cases (two examples per case) so the curve is read from three points per arm and the size for stage C is chosen from the curve, not assumed.
5. **QLoRA.** Unnecessary for a 0.6B or 1.7B model on a 24–48 GB card (bf16 LoRA fits with room), and quantisation would be one more variable between arms. Plain LoRA in bf16, verified by stage A on the real card.
6. **Provider and prices.** The second opinion quotes Runpod's RTX 4090 at US$0.89 (the Secure Cloud price; Community is US$0.34 and shrinking) and treats Runpod/Vast as the venue. Section 4 re-checks eight options against today's official pages; the pilot's 16 GPU-hours cost US$0 out of pocket on two providers' free credits and under US$7 on the cheapest paid one.
7. **Scope of "M".** The second opinion's full M (broader semantic mappings, fused forms, custom vocabulary) is a separately costed later ablation in both plans; the pilot tests the compact-codes family only. No disagreement, stated here so that no "M gain" is read into the pilot's result.

### 1.3 Where the assessment was wrong, corrected here

The assessment's §3–§4 proposed 54 runs before any profile, a single task, "tied" as a verdict, runtime figures derived from anchors on other GPUs, and a cost estimate (US$15–35) without a measured cycle. All five are replaced by sections 2–3.

---

## 2. The seven concerns, answered with mechanisms

| # | concern | what the revision does | where |
|---|---|---|---|
| 1 | one task is too narrow | a second task, *lookup* (relation and polarity of one named record), on the same cases, graded by the same grader; the pilot's result is stated for exactly these two tasks on this schema (section 3.6 says what success does and does not establish) | `contract.md` §Tasks; `data.py`; §3.6 |
| 2 | the 54-run sweep was unstaged | stage A profiles load, training, generation, checkpoint and upload on the real GPU and projects B and C; stage B is one seed per arm with one answer format and is exploratory by rule; stage C adds seeds and the answer-format factor; expansion criteria and decision rules fixed here before any run | §3.2–3.4; `profile.py`; `configs/plan_stage_*.json` |
| 3 | a 0.6B failure must not end the direction | a written diagnosis rule separates format failure (one arm flat while others learn), capacity (codes lag, gap shrinks with data), training defect (zero-shot English at chance, or every arm fails) and undertraining (every arm below 90% with loss still falling), each with its own next step; the 1.7B run is triggered by the capacity case | `contract.md` §Decision rules; §3.5 |
| 4 | an interval across acceptable and unacceptable differences is not a tie | non-inferiority with δ = 3 on a 90% interval from a cluster bootstrap that resamples seeds first and items within seeds; one seed gives no verdict; inconclusive is a reported outcome with a bounded follow-up (two more seeds, once) | `contract.md`; `analyze.py` |
| 5 | runtime and cost were unmeasured | stage A measures every phase of the cycle on the actual card; the ledger records every rental with start, stop, price and measured spend; failed runs and retries are in the 1.5× contingency; a stopped instance is verified through the provider's console and CLI before a stage is called finished ("a timeout is not proof billing stopped") | `profile.py`; `ledger.py`; `verify_teardown.md` |
| 6 | whole-request savings | per request: legend, instruction, records, chat-template overhead, output, retry prompt and retry output tokens; parse and conversion wall-clock; cached (legend once per 100 requests) and uncached totals side by side; latency batched and single; cost per correct task at the quoted price | `evaluate.py`; `interfaces.md` §Evaluation |
| 7 | M choices must follow the intended data | pilot data: half the records reuse an earlier subject, a fifth an earlier object (handoff-like); the candidate is grouping by subject, the one idea with a measured gain on that distribution; the dictionary is excluded by its measured cost on short labels; the lean answer stays a stage-C factor and is not adopted | `contract.md` §Underlying case, §Renderings; §1.2 item 2 |

---

## 3. The revised pilot

### 3.1 Fixed elements (the contract)

The full text is `experiments/pretrained_pilot/contract.md`; the file formats are in `interfaces.md`. In brief: the study's seven-field records with identifiers of 2–5 characters from the v3 hash split; subject reuse 0.5 and object reuse 0.2 with records of one subject adjacent; K 1..12 trained, 13..16 held out; two feature pairs held out; two tasks (*select*: the negated records' pairs in order; *lookup*: relation and polarity of one named pair) with four trained and two held-out wordings each; four arms (`english`, `json`, `sylang2`, `sylang2g` = grouped by subject); a per-arm legend of 100–130 Qwen 3 tokens, matched within 15 tokens; splits by case hash, dev and test shared across seeds; train prefixes 500 / 2,000 / 8,000 cases per seed, dev 400 cases, test 1,000 cases in four panels (iid 480, held-out wording 160, held-out pair 160, K 13..16 200); strict grader; one retry with the format reminded, both requests counted; greedy, non-thinking decoding.

**Model.** `Qwen/Qwen3-0.6B` (Apache-2.0, the tokenizer family of every count in this project), LoRA rank 16 on all linear projections, bf16, 2 epochs, AdamW, warm-up then cosine decay, batch 16, sequence ≤ 1,024 tokens, dev-selected epoch (dev accuracy by generation on 200 dev examples). `Qwen/Qwen3-1.7B` only under the trigger in §3.5.

### 3.2 Stages, cells and what each is for

| stage | cells | purpose | GPU-hours budgeted (expected) | decision it enables |
|---|---|---|---|---|
| **A — profile** | setup; `profile.py` on `sylang2` at 2,000 cases (30 training steps, 64 generated items, one checkpoint, one upload); one real cell at 500 cases end to end | measure load time, memory, examples/s, items/s, checkpoint size and upload time on the real card; project B and C at the quoted price; prove the launch path | 1.0 (0.5) | go/no-go for B on measured cost; all later figures are replaced by measurements |
| **B — matched comparison, exploratory** | 4 arms × seed 0 × 2,000 cases (JSON answer); 4 zero-shot baselines (legend, no training); if the projection allows, `english` and `sylang2` at 500 and 8,000 cases for the curve | one paired comparison on identical cases with every number from §2 recorded; size the seed variance that stage C must overcome; find pipeline faults before they cost three seeds | 3.0 (≈ 1.0) | expansion criteria §3.3; no format verdict by rule |
| **C — decision** | 4 arms × seeds 1–2 × 2,000 cases (seed 0 from B); answer-format factor `bare` on `english` and `sylang2` × 3 seeds; zero-shot baselines for the bare legend | the pre-registered verdicts (§3.4) on three seeds; the answer-format question as a reported factor, not an adoption | 12.0 (≈ 3.0) | the format verdict, the cost ratio, the 1.7B trigger |
| **1.7B extension** (only under §3.5) | `sylang2` and `json` × 3 seeds × 2,000 cases on Qwen3-1.7B | whether the codes-vs-JSON gap is a capacity question | within the remaining ceiling (≈ 2.5× a 0.6B cell) | closes the capacity question |

The budgeted hours are deliberately above the expectation (which rests on published anchors for other GPUs, not on a measurement); stage A replaces both columns with measured numbers before stage B is approved. Nothing in B or C starts until `profile.py`'s projection, with contingency, fits inside the remaining ceiling.

### 3.3 Expansion criteria (B → C, C → 1.7B) and stop rules

B → C proceeds only if all hold: (1) every B cell completed inside 1.5× its projected time and the measured spend is at most 35% of the ceiling; (2) the pipeline is sane: the zero-shot `english` baseline is above chance on *select* and the trained `english` arm reaches ≥ 90% on the iid panel at 2,000 cases, or its training loss is still falling at the end of epoch 2 (then C runs at 8,000 cases if the projection allows, else with 3 epochs); (3) no grader or data defect was found (the `--check` pass and the invalid-rate review). If the curve for `english` or `sylang2` rises by more than 3 points between 2,000 and 8,000 cases, stage C uses 8,000. If B shows every arm at or above 97% on every panel, the task is at ceiling for a pretrained model: C still runs (the cost and latency comparison stands on its own) and the next data family is the harder one (K to 32, long labels), reported as such.

Stop rules: the ceiling is reached before C can be read (stop, report what was measured, no top-up); B shows a pipeline defect that a second B run does not cure (stop and report); the owner withdraws the approval.

### 3.4 Decision rules (pre-registered; applied at stage C only)

For each contrast, on each test panel and task, paired per item within each seed: a 90% cluster-bootstrap interval of the mean accuracy difference in points (seeds resampled, then items within seeds; 2,000 resamples; bootstrap seed 0), with the per-item-only interval beside it. With δ = 3: **non-inferior** if the lower bound is above −3; **inferior** if the upper bound is below −3; **superior** if the lower bound is above +3; **inconclusive** otherwise. An inconclusive result at three seeds may be followed, once, by two more seeds on the two arms in the contrast if the ceiling allows; after that it stands as reported. Stage B's single seed gives no verdict.

The contrasts, in order of importance for the owner's priority:

1. `sylang2g − sylang2` (what the improved-M candidate adds over the current compact format): kept as the candidate only if non-inferior on every panel and cheaper by the measured margin on the pilot distribution (the §2.1 arithmetic predicts −20% record tokens at K = 16).
2. `sylang2 − json` and `sylang2 − english` (the format question): the project's machine-efficiency direction is alive if the codes are non-inferior to both on the iid panel and inferior on none of the held-out panels.
3. The cost question, for every arm against `json` and `english`: cost per correct answer (all requests and retries in the numerator) at the quoted GPU price, under the cached and the uncached legend conventions; whole-request latency batched and single. The second opinion's 15% improvement threshold is adopted as the *engineering* bar for the next stage: a format that is non-inferior but saves less than 15% of whole-request cost or latency in both conventions is reported as "not worth a format change at this K".
4. `bare − json` answer format on `english` and `sylang2` (stage C): reported; nothing is adopted from it.

### 3.5 Reading a failure (so that a 0.6B result does not end the direction by accident)

| observation | reading | next step |
|---|---|---|
| zero-shot `english` at chance on *select*, or every trained arm fails | a pipeline defect (legend, chat template, decoding, grader) before any model claim | fix, re-run one B cell, no new money beyond stage B's budget |
| every arm below 90% at the largest size with training loss still falling | undertrained | 3 epochs or 8,000 cases within C's budget, then re-read |
| `english` and `json` learn, the codes do not, and the gap **shrinks** with data | a capacity/data question | the 1.7B extension on `sylang2` and `json` |
| `english` and `json` learn, the codes do not, and the gap does **not** shrink | a format result for this model size | reported as such; the pretrained line continues only at a larger model if the owner wants the question answered there, else the byte-model niche stands |
| one arm flat at every size while others learn | that arm's format failed | reported; no rescue |

### 3.6 What a success would and would not establish

A success (contrasts 1–2 non-inferior on three seeds, the cost ratio under both conventions) establishes: that a 0.6B pretrained model fine-tuned on identical cases reads the header-free compact codes, grouped or not, as accurately as compact JSON and controlled English on in-distribution cases and on unseen identifiers, unseen wordings, unseen feature pairs and unseen record counts; the measured whole-request token, latency and cost differences for this schema, these two tasks and this tokenizer; and whether grouping by subject is worth its grammar change on handoff-like data.

It would **not** establish: human readability (deferred by the owner); results on real GBG workflows, nested dependencies or free text; other model families, larger models or hosted models; custom-token vocabularies; record counts beyond 16; robustness to adversarial or quoted text inside records; production savings, which depend on the deployment's cache behaviour and request mix. Each of these is a later stage or a separate decision.

### 3.7 Budget: US$100 all-in

| line | allowance | note |
|---|---|---|
| stage A | US$5 | 1 GPU-hour at any listed price, plus the first storage month |
| stage B | US$15 | 3 GPU-hours budgeted |
| stage C | US$30 | 12 GPU-hours budgeted |
| 1.7B extension | US$25 | only under §3.5 |
| storage, transfer, snapshots, failed runs and retries beyond the 1.5× contingency | US$25 | the ledger's `report --ceiling 100` fails the stage scripts when exceeded |

On the recommended provider the whole plan is inside free credits; on the paid fallback the sixteen budgeted GPU-hours are US$5.60 plus about US$1 of snapshot storage. The ceiling is therefore an envelope for failures, not an expectation; the expectation is US$0–7.

---

## 4. Provider: recommendation, fallback, and dollars per completed experiment

Re-checked on 2026-10-16 from official pricing and billing pages (Lightning AI's pages are JavaScript-only and were read through a rendering proxy; figures marked accordingly); the page-by-page record with quotations, source URLs and the list of what could not be verified is `gpu-provider-recheck-2026-10-16.md` in this directory. Workloads: A = 1 GPU-hour, B = 3, C = 12 (budgeted, not expected); 20 GB of storage for one month.

| provider / GPU | rate | A | B | C | storage | total before credits | out of pocket after free credits |
|---|---|---|---|---|---|---|---|
| **Lightning AI**, L4 24 GB Studio [rendered] | US$0.79/h, per second | 0.79 | 2.37 | 9.48 | US$1 (10 GB free, then US$0.10/GB-month) | **US$17.64** | **US$0** with 5 signup + 25 card credits (one-off, expire in 12 months); **US$12.64** with the 5 signup credits only |
| **Thunder Compute**, RTX A6000 48 GB | US$0.35/h, per minute, 6 vCPU/48 GB RAM/100 GB disk included | 0.35 | 1.05 | 4.20 | US$1 (snapshot US$0.05/GB-month) | **US$6.60** | US$6.60 (no credits; prepaid credit or card) |
| Modal, L4 24 GB function | US$0.80/h GPU + about US$0.22/h for 2 cores and 16 GiB | 1.02 | 3.06 | 12.26 | US$0 (1 TiB free) | US$16.34 | **US$0** inside the US$30 monthly Starter credit (card required for GPU; Workspace budget and spend limit settable) |
| Runpod, RTX 4090 Community / Secure | US$0.34 / 0.89/h | 0.34 / 0.89 | 1.02 / 2.67 | 4.08 / 10.68 | US$1.40 (network volume) | US$6.84 / 15.64 | same (prepaid, no credits; Community closed to new hosts, stock a daily matter) |
| Vast.ai, RTX 4090 | about US$0.43/h (tracker, not a live quote) | 0.43 | 1.29 | 5.16 | host-set, about US$2 | about US$9 + bandwidth | same (prepaid) |
| Salad, RTX 4090 | US$0.16–0.33/h | 0.16–0.33 | 0.48–0.99 | 1.92–3.96 | none; external object store needed | US$2.56–5.28 + the container and checkpoint-recovery work | same |
| Colab Pro, L4 | US$9.99/month for 100 compute units; about 1.7 CU/h (community measurement) | ≈ 1.7 CU | ≈ 5 CU | ≈ 21 CU | Drive | US$9.99 minimum | US$9.99 |
| NVIDIA Build | inference endpoints only | – | – | – | – | not applicable | – |

**Recommendation: Lightning AI, L4 Studio.** It is a plain Linux VM with SSH and a terminal, so the package runs unchanged; Studios sleep after 10 minutes of inactivity unless a process is running, which protects against an idle instance; billing is per second; and the five signup credits (no card, a phone number is required) cover stages A and B as budgeted (US$3.16 plus storage inside the 10 GB free tier). Adding a card grants 25 more credits, which cover stage C. Two cautions the owner should know before choosing: the documentation conflicts on the free tier's session limits (the pricing table says L4 sessions are unlimited, a footnote says free Studios restart every 4 hours) and on whether interruptible machines are open to free accounts; every cell in the package is wrapped in a 90-minute timeout with resumable checkpoints, so a 4-hour restart costs at most one cell. If the account's console shows fewer credits than the pages promise, the owner reports the number and section 7's approval is re-sized before anything starts.

**Fallback: Thunder Compute, RTX A6000 48 GB at US$0.35 per hour.** The cheapest paid option that is also a plain VM (CLI `tnr`, SSH, PyTorch pre-installed); 48 GB makes the 1.7B extension trivial. Cautions: there is no Stop, so the workflow is snapshot → delete → restore (snapshots US$0.05/GB-month until deleted) and the ledger's `stop` line is written only after the delete; billing has threshold charges rather than a hard cap, so the pilot's cap is the prepaid credit the owner loads (US$20 is enough for everything here); the compatibility page excludes CUDA managed memory, MPS and GPU profilers, none of which the package uses.

**Second fallback: Modal** (US$30 monthly credit; a hard Workspace budget). It needs the trainer wrapped as a Modal function with a Volume and tolerates preemption only through retries; about two hours of adaptation that the package does not include yet, so it is the choice only if both VMs fail the owner's eligibility check.

**Not recommended for this pilot**: Runpod and Vast unless an actual quote on the day beats Thunder (the second opinion's US$0.89 figure is Runpod's Secure price); Salad (the checkpoint-recovery plumbing outweighs the saving for minute-long runs); Colab (accelerator access is tier-gated and a notebook workflow does not fit a scripted sweep); NVIDIA Build (inference only).

**Dollars per completed experiment** (an experiment = one trained cell evaluated on the test set, two tasks): at Lightning's rate and the budgeted hours, stage B's four cells cost US$0.59 each and stage C's fourteen cells US$0.68 each; at Thunder's rate, US$0.26 and US$0.30. The `ledger.py report` command recomputes this from measured spend and the number of result directories after each stage.

---

## 5. Owner setup steps (accounts, payment, secure access, approvals)

Rule for every step: **passwords, private keys and API tokens never appear in chat, in commit messages, in repository files or in the ledger.** Tokens that this session needs go into the cloud environment's settings (title-bar environment menu → Edit → network secrets / API credentials, or an environment variable named `PILOT_PROVIDER_TOKEN`); SSH private keys stay in the key store of the machine that connects. The stage scripts read `PILOT_PROVIDER_TOKEN` from the environment only if a provider CLI needs it; they never write it anywhere.

### 5.1 Lightning AI (recommended; stages A and B with no payment method)

1. Create the account at lightning.ai (e-mail and a phone number; no card). In *Settings → Billing* confirm the credit balance shows 5 credits (1 credit = US$1) and that a single L4 GPU is selectable for a new Studio. If either differs from the pages, note the actual numbers.
2. Create one Studio from the default image, CPU first. In *Settings → SSH* add the public key generated on the development server (`ssh-keygen -t ed25519 -f ~/.ssh/pilot_lightning`); the private key never leaves that server.
3. Upload the dataset tarball produced on the server (`data/v1.tar.gz`, about 15 MB) and the package (`git clone` of the branch, or an upload of `experiments/pretrained_pilot/`).
4. In the Studio terminal run `bash experiments/pretrained_pilot/setup_gpu.sh` with `PILOT_ALLOW_DOWNLOAD=1` set for that command only; it creates the virtual environment, installs the pinned versions and downloads `Qwen/Qwen3-0.6B` (about 1.2 GB, Apache-2.0) into the Studio's disk. Then switch the Studio to the L4.
5. Export `PILOT_TOKENIZER`, `PILOT_MODEL`, `PILOT_PRICE_PER_HOUR=0.79`, `PILOT_PROVIDER=lightning`, `PILOT_INSTANCE=<studio name>`, `PILOT_GPU=L4` and run `bash experiments/pretrained_pilot/run_stage_a.sh`. It writes `profile.json`, the ledger lines and the projection; stop the Studio (the console's Stop button; the per-second billing ends) and read the projection before stage B.
6. Stage B only after the written go in section 7 is confirmed against the projection: start the Studio, `run_stage_b.sh`, copy `runs/*/run.json`, `results/*` and `profile.json` back to the server (`scp` or the Studio's download), stop the Studio, record the console's usage figure with `ledger.py add --event balance`, and walk through `verify_teardown.md`.
7. Stage C: add a card in *Billing* (25 further credits), then `run_stage_c.sh` as above. Storage: keep the Studio under 10 GB (the model, the data and the adapters are about 2 GB); delete the Studio when the pilot report is written.

### 5.2 Thunder Compute (fallback)

1. Create the account, add a payment method, and load prepaid credit of US$20 (the pilot's practical cap on this provider; threshold charges, not a hard cap, otherwise apply).
2. Install the `tnr` CLI on the development server, log in with the token stored as `PILOT_PROVIDER_TOKEN` in the environment settings (never pasted into chat), create an RTX A6000 instance with the default 100 GB disk, and connect with `tnr connect`.
3. Steps 3–6 of §5.1 with `PILOT_PRICE_PER_HOUR=0.35`, `PILOT_PROVIDER=thunder`, `PILOT_GPU=A6000`. At the end of every session: upload results, snapshot if the next stage is approved (US$0.05/GB-month), **delete** the instance (there is no Stop), confirm in the console and with `tnr status` that nothing is running, and record the balance.

### 5.3 Who runs what

All paid actions (creating accounts, adding payment methods, starting or deleting machines) are the owner's. The stage scripts can be run by the owner in the Studio terminal, or by this session over SSH from the development server if the environment's network policy allows the provider host and the SSH private key is placed in the environment's secrets; in both cases the commands are the three `run_stage_*.sh` lines and the copy-back, nothing else. This session never holds a provider token in chat.

### 5.4 Approvals, in order

1. Stage A + B on Lightning's free credits (no money), including the model download on the rented machine (section 7).
2. Stage C after the stage-B readout is reviewed (adds a card for the 25 credits; still US$0 if the credits appear as documented).
3. Only if credits fall short: the Thunder fallback with a US$20 prepaid load inside the US$100 ceiling.
4. The 1.7B extension, only under §3.5, inside the ceiling.
5. Separate, unchanged from the assessment's §8: the history squash (not needed for the pilot), a hosted-model zero-shot reference (paid calls, not needed), adoption of any output format (not before validation and a separate decision).

---

## 6. What is prepared on the development server (no rental, no paid call, no model-hub download)

The package `experiments/pretrained_pilot/` (README there) contains the frozen contract and interfaces, the shared module, the data generator with its manifest, the strict grader with its corrupted-output tests, the LoRA trainer with resumable checkpoints and a dev-selected adapter, the generation-based evaluator with the retry policy and whole-request accounting, the analysis with the cluster bootstrap and the pre-registered verdicts, the profiler, the cost ledger with the ceiling check, the teardown checklist, the stage launch scripts with hard timeouts, the GPU setup script with pinned versions, the per-cell configs and plan files, and a CPU smoke test that exercises the whole cycle with a randomly initialised Qwen 3 of the real architecture and the real tokenizer (no weights downloaded). Section 6.1 records the smoke-test result and the measured CPU-side figures.

### 6.1 Verification record (development server, 4 CPU cores, 2026-10-16)

- **Grader**: `test_grader.py`, 59 cases over both tasks and both answer formats (dropped, swapped, invented and reordered pairs; wrong polarity or relation; missing, extra and duplicate keys; malformed JSON; leading and trailing prose; empty output; each format's answer given to the other; `[]` against `none`), all rejected as specified; whitespace and key-order differences accepted. Passes.
- **Data**: the full single-seed dataset (8,000 train cases, 400 dev, 1,000 test; four arms; 76 MB) generates in 117 s and passes `data.py --check` (hashes, invariants, 150,400 self-graded answers, answer identity across arms); a second generation is byte-identical. Realised subject reuse 0.50, object reuse 0.13 (coin 0.2, limited by the no-repeated-pair rule); K uniform; held-out pairs only in the `pair` panel, held-out wordings only in `wording`, K 13–16 only in `long`.
- **Legends** (Qwen 3 tokens; JSON answer format / bare): english 118 / 107, json 116 / 105, sylang2 121 / 110, sylang2g 127 / 116; spread 11 within each format.
- **Whole-prompt tokens on the real data** (legend + records + instruction + template, mean over all splits, JSON answer format): english 272, json 336, sylang2 225, sylang2g 221; mean answer tokens 10 for *lookup* and 26–32 for *select*. The grouped candidate saves tokens only in the record part (about −15% of record tokens at the mean K of 6.5), which is why the whole-request accounting matters.
- **CPU smoke test** (`smoke_cpu.sh`, a randomly initialised 9.8 M-parameter Qwen 3 with the real tokenizer, nothing downloaded): data → check → train 20 steps (sylang2, then english) → zero-shot and trained evaluation on 32 test items each → analysis; passes in 153 s. The run records, evaluation files, item flags and the analysis tables have every key of `interfaces.md`; the retry policy fired on every item (a random model emits nothing valid), both requests were counted, and the cached total equals the uncached total minus the repeated legends. The analysis reports `exploratory (1 seed): no decision`, as the rule requires.
- **Trainer**: kill-and-resume reproduces the uninterrupted run exactly (same permutation, same token count); a failing run records `status: failed` with the traceback. The bf16 and CUDA paths are written to the spec but untested here (no GPU); stage A exercises them before any experiment.
- **Scripts**: the three stage scripts and the smoke script pass `bash -n` and shellcheck; exercised against stub components for timeouts (exit 124), failures, idempotent re-runs, the budget guard, the ledger start/stop lines on every exit path and the ceiling check; `ledger.py selftest` passes 39 checks, including the refusal of any line that contains the provider token.
- **Independent review**: an adversarial code review of the package (five lenses, every finding refuted or confirmed by a second reader) and a claim-by-claim verification of this document were run; their confirmed findings and the corrections made are listed in the commit messages of the branch.

### 6.2 Constraints honoured

Git history unchanged (no squash, no force-push); no paid hosted-model calls; no rental; no paid training; no new output format adopted (the bare answer is a stage-C factor only); the session is not archived; private packet material stays out of every artefact (identifiers in the package are synthetic; the packet is read only by `data.py` for the English rendering, on the server).

---

## 7. The exact next approval needed

**Approve stage A + B of §3.2 on Lightning AI's free credits, with these terms:** Qwen3-0.6B weights downloaded on the rented machine only; the dataset generated on the server and uploaded; stage A run first and stopped; stage B started only if `profile.py`'s projection with contingency is at most 35% of the US$100 ceiling and the owner confirms the Studio's credit balance; no card added before the stage-B readout; the Studio stopped after each stage and `verify_teardown.md` completed; everything recorded in `ledger.jsonl`. Expected out-of-pocket cost: **US$0**; ceiling: US$100 all-in; the one thing the owner must do before anything runs is create the account and report the credit balance and GPU eligibility the console actually shows.

Two smaller nods, optional: (a) a real-weights CPU smoke of the package on the development server (a 1.2 GB download of Qwen3-0.6B onto the server; it removes the last untested step before GPU time, under the earlier "no large downloads" rule it needs a yes); (b) the free byte-model count-line test of the assessment's §6 on the server's idle cores, independent of the pilot.
