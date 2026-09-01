# Mentor Demo — 5-Terminal Manual Run (Real DAIC-WOZ Multimodal Data)

Rewritten 2026-08-23 to reflect every fix landed in the `fixes/pipeline-correctness`
branch (Fix A through Fix E5, Step 16's LR decay) — the previous version of this file
(`MENTOR_DEMO_RUNBOOK.pre_fixes.bak`) predates all of them and will show stale
expected values if you follow it. Uses `PIPELINE_MODE=multimodal`, which reads real
DAIC-WOZ text + wav2vec2 audio + DenseNet video features from
`dataset_build\daic_records_multimodal_participant_only.parquet` (186 rows, Fix A:
participant-only turns, the interviewer "Ellie" text is filtered out) — **not** the
toy `videos\sample.mp4` fixture, and **not** the old `daic_records_multimodal.parquet`
(188 rows, includes interviewer text — superseded).

**Current configuration this demo runs against** (all confirmed live in this branch):
- Text: MentalBERT, `max_len=512` (Fix A), text encoder **frozen** (Fix B — only
  `audio_encoder` + `vision_encoder` + `fusion` train; BERT's 109,482,240 params never
  move, are never DP-noised, never uploaded)
- Training: `lr=1e-4`, `epochs=10` (Fix E4), `LR_DECAY=0.7` (Step 16 — decays lr on
  rounds after the first; **this demo only runs round 1, so LR_DECAY has no visible
  effect here** — `LR_DECAY**0 = 1` — it matters starting round 2, which this basic
  demo doesn't reach)
- DP: Gaussian, `clip_norm=0.85` (Fix E4 recalibration — was 1.0, then 0.15, now
  0.85), `noise_multiplier=1.0`, `δ=1e-5`, `ε=5.302585` (RDP, single composition)
- Aggregation: trimmed_mean, `trim_ratio=0.1` (still the Rust orchestrator's
  hardcoded default — see disclosures below)

Open 5 PowerShell terminals in VS Code, all starting from:
```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
```

**One important disclosure to make to your mentor if asked:** every client run in this
demo shares **one enrolled device identity** (this machine has exactly one TPM-backed
key). Reaching the 3-update aggregation trigger means running the client three times
sequentially under that same device — it demonstrates the full mechanical pipeline
(training → DP → encryption → TPM signing → upload → trimmed-mean aggregation → global
model persistence) correctly and on real data, but it is not three independent physical
devices. No code change can fix this without a second machine or a second signed device
identity. **See "Honest disclosures" at the end of this file for three more, established
in this session's investigation — read them before your mentor asks.**

---

## TERMINAL 1 — MongoDB server start (+ fresh-state reset)

```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"

# Make sure the Windows MongoDB service (stale C: path) isn't already squatting the port
Get-Service -Name MongoDB -ErrorAction SilentlyContinue | Select-Object Name, Status, StartType
# If Status=Running: Stop-Service MongoDB -Force  (needs Administrator)

New-Item -ItemType Directory -Path "D:\MongoDB\log" -Force | Out-Null
Start-Process -FilePath "C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe" `
    -ArgumentList '--dbpath','D:\MongoDB\data','--port','27017','--bind_ip','127.0.0.1','--logpath','D:\MongoDB\log\mongod.log','--logappend' `
    -WindowStyle Hidden
Start-Sleep -Seconds 3

# Confirm it's up and pointed at the correct (D:) data directory
& ".venv\Scripts\python.exe" -c "import pymongo; c=pymongo.MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=5000); print(c.admin.command('ping')); print(c.admin.command('getCmdLineOpts')['parsed']['storage'])"

# Wipe both project databases to a clean, empty state (does NOT touch the unrelated 'libraryDB' database)
& ".venv\Scripts\python.exe" "scripts\reset_all_federated_dbs.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_db_reset.log"
```
Expected: `{'ok': 1.0}`, `dbPath: D:\MongoDB\data`, and the reset script prints
`[OK] All collections in all project databases are empty.`

Captured to `trainer_outputs\demo_db_reset.log` (same `Tee-Object` pattern as
Terminal 4's client runs below) so `scripts\render_session.py` can render the
[1/7] Database block from a file instead of console-only output. Like every
other `Tee-Object` capture in this runbook, this file is UTF-16LE with a BOM
(PowerShell's default) - any reader must decode it explicitly, not assume
UTF-8.

Leave this terminal open for the whole session.

**⚠ If you are re-running the demo (not the very first run today), read this first:**
the Rust orchestrator caches round state in memory at startup and does **not**
re-read MongoDB while it keeps running. If you reset the database (this terminal)
while an *already-running* orchestrator (Terminal 2) is still up, the orchestrator
keeps believing whatever round state it last saw — new client submissions can land in
the wrong round, or you'll see `Round state=Aggregating — nothing to submit this
cycle` even though the database is empty. **The reset in this terminal must always be
followed by killing and restarting Terminal 2's orchestrator process**, every time,
not just on the very first run. The correct order, every time you want a clean run:
DB reset (here) → orchestrator (re)start (Terminal 2) → enroll (Terminal 3) → clients
(Terminal 4). Never reset without also restarting the orchestrator afterward.

---

## TERMINAL 2 — Rust server start

```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-\server\orchestration_agent"
$env:MONGO_URI = "mongodb://localhost:27017"
$env:MONGO_DATABASE = "federated_multimodal"
$env:PATH = "C:\Program Files\Git\mingw64\bin;" + $env:PATH   # openssl, needed for device enrollment CSR signing

# IMPORTANT: use cmd.exe for redirection, not PowerShell's native > (writes UTF-16, breaks OTP parsing)
& "$env:WINDIR\System32\cmd.exe" /c "cargo run --release > `"D:\Download D\BE PIPELINE\Capstone-\trainer_outputs\demo_orchestrator.log`" 2>&1"
```
This blocks in the foreground — that's expected, it's your server. Watch for (a few
seconds after `cargo run` starts, since the binary is already built):
```
MongoDB connected: ... (database=federated_multimodal)
[DEV] Enrollment OTP: XXXXXX (valid for 10 minutes)
[RECOVERY] Round 1: verified_updates=0, global_model=absent
[SERVER] Running in mTLS mode on 0.0.0.0:50051
```
If instead you see `[RECOVERY] Round 1: verified_updates=N` for some N>0 right after a
reset you just ran, something is wrong — the reset in Terminal 1 either didn't run or
ran against the wrong database. Re-check `$env:MONGO_DATABASE` in *this* terminal
matches `federated_multimodal`, the same name Terminal 1's reset script wipes.

**`$env:MONGO_DATABASE` gotcha**: this variable has to be set correctly in *every*
process that talks to Mongo directly — it is not a global system setting. The
orchestrator (this terminal) and the client (Terminal 4, via `PIPELINE_MODE`) already
handle this correctly as shown below. It only bites you if you go beyond this basic
demo and run one of the standalone diagnostic scripts from `scripts/` (e.g.
`aggregate_offline.py`, used in this session's Step 13/14/17 investigation) directly
in a fresh shell without exporting `MONGO_DATABASE=federated_multimodal` first — those
scripts default to a database named `federated` (singular, no `_multimodal` suffix)
and will silently look in the wrong place if you forget it. Not needed for this basic
demo's 5 terminals; only relevant if your mentor asks you to go further.

The OTP is valid 10 minutes — move to Terminal 3 promptly.

---

## TERMINAL 3 — Connection check + device enrollment

```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"

# Port + cert sanity checks
Get-NetTCPConnection -LocalPort 50051 -ErrorAction SilentlyContinue   
Test-Path "server\orchestration_agent\certs\ca.pem"
Test-Path "server\orchestration_agent\certs\server.pem"

# Device enrollment (required every time Terminal 1's reset wipes the 'devices' collection)
& ".venv\Scripts\python.exe" enroll_step5.py "D:\Download D\BE PIPELINE\Capstone-\trainer_outputs\demo_orchestrator.log" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_enroll.log"
```
Expected tail: `ENROLLMENT COMPLETE`, with a `client.pem` fingerprint printed. This
single exchange (RequestEnrollment → OTP → EnrollDevice → signed cert) **is** your live
connection check: it proves mTLS, gRPC, and MongoDB are all wired together correctly.

Captured to `trainer_outputs\demo_enroll.log`, same `Tee-Object`/UTF-16LE
pattern as Terminal 1's reset capture above - `scripts\render_session.py`'s
[3/7] Enrollment block reads this file. No changes to `enroll_step5.py`
itself.

Verify the cert if you want a second, independent check:
```powershell
& "C:\Program Files\Git\mingw64\bin\openssl.exe" verify -CAfile "server\orchestration_agent\certs\ca.pem" "$env:USERPROFILE\.federated\keys\client.pem"
```

**Troubleshooting: enrollment OTP timeout.** `enroll_step5.py` reads the OTP
from whichever orchestrator log PATH is passed to it as an argument, and
gives up after 15 seconds if the OTP doesn't appear AFTER the file position
it recorded the moment it started (so it never matches a stale OTP already
sitting earlier in the file). Three common causes, in order of likelihood:
1. **Terminal 2 is writing to a different log filename than the one you
   passed to `enroll_step5.py`** — check the path in both commands character
   for character; a typo or a leftover path from a previous session is the
   usual culprit.
2. **The orchestrator has been running for more than 10 minutes** — the OTP
   it printed at startup has expired. Restart Terminal 2's orchestrator to
   get a fresh one, then enroll immediately.
3. **A database reset happened without the mandatory orchestrator restart**
   (see Terminal 1's warning above) — the orchestrator's in-memory OTP may
   no longer correspond to what `RequestEnrollment` expects.

Fix, every time: stop the orchestrator (Terminal 2, Ctrl+C or
`Stop-Process`), reset (Terminal 1), restart the orchestrator (Terminal 2),
**watch the log for the `[DEV] Enrollment OTP` line to actually appear**
before doing anything else, confirm it's the same log path you're about to
pass to `enroll_step5.py`, then enroll promptly.

---

## TERMINAL 4 — client 1 run, client 2 run, client 3 run

Aggregation requires **3 accepted updates** (`round.updates.len() >= 3`). Run all three
sequentially, in this same terminal, waiting for each to print
`Run-once pipeline complete` before starting the next.

**Three modes, pick one before you start** (Step 20/21 added modes B and C —
sharding stays **default OFF** in the code itself; these are runbook-level env
vars only, nothing in `trainer_mentalbert_privacy.py` changed):

| | What it demonstrates | Trade-off |
|---|---|---|
| **A — Full** (default, no env vars) | The pipeline mechanics end-to-end | Higher F1, but all 3 "clients" train on the identical 149-record split — **not federation**, one dataset submitted three times |
| **B — Sharded** | Genuine federation — each client gets its own disjoint data | Lower F1 (~15 positive examples per client) — see the honest disclosure below before you show this |
| **C — Non-IID** | Genuine client heterogeneity, the precondition for testing FedProx | Same data-volume cost as B, plus a deliberately skewed split |

Run **all three clients in the SAME mode** — mixing modes within one round
produces a meaningless aggregate (the clients wouldn't even agree on how the
data was supposed to be divided). Start with Mode A — it matches every prior
measurement in this project and is the safer default to walk through first.

**Physician-feedback prompt — read before running, applies to every mode.** The
supervised training path prompts once per record for an optional physician
correction (`PHYSICIAN FEEDBACK LOOP`), and it does this for **all 186 records
in the parquet, every client run** — not just ones missing a label (sharding
only changes which of those 186 get *trained on*; the prompt loop itself
always runs over all 186). Every record in this corpus already carries a real
PHQ-8 ground-truth label (Fix E1), so a blank response at every prompt is
exactly correct — it keeps the real label. Piping a single blank line (as the
old runbook did with `"" | ...`) is **not enough** and will crash with an
`EOFError` partway through the first client's run. Create the blank-stdin file
once, before whichever mode you run:
```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
$env:PATH = "C:\Program Files\Git\mingw64\bin;" + $env:PATH
$env:PYTHONUTF8 = "1"

# One-time: 260 blank lines, comfortably more than the 186 prompts needed
if (-not (Test-Path "trainer_outputs\demo_blank_stdin.txt")) {
    1..260 | ForEach-Object { "" } | Out-File -FilePath "trainer_outputs\demo_blank_stdin.txt" -Encoding utf8
}
```

### MODE A — Full (default, no env vars)

```powershell
# Client 1
Get-Content "trainer_outputs\demo_blank_stdin.txt" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run1.log"

# Client 2
Get-Content "trainer_outputs\demo_blank_stdin.txt" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run2.log"

# Client 3 — this one triggers aggregation automatically on the server ~10-20s after it finishes
Get-Content "trainer_outputs\demo_blank_stdin.txt" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run3.log"
```

**Expected values (Mode A) — use these to tell a good run from a bad one:**

| Quantity | Expected value | Where it appears in the log |
|---|---|---|
| Audio dim | 154 (wav2vec2) | `[MULTIMODAL] audio encoder` section |
| Video dim | 84 (DenseNet) | `[MULTIMODAL] video encoder` section |
| Total model parameters | 109,763,494 | `MODEL CONFIGURATION` |
| **Trainable** parameters | **281,254** | `FIX B — TEXT ENCODER FREEZE` (the number that actually gets trained/DP-noised/uploaded — not the 109.7M total) |
| Frozen parameters | 109,482,240 | same section |
| Round-1 effective learning rate | 1e-4 (0.0001) | `STEP 16 — ROUND-AWARE LR DECAY` (round_id=1 → decay factor 1.0, unaffected) |
| Partition mode | `full` (`CLIENT_SHARD_ID`/`CLIENT_N_SHARDS` unset — every client trains on all 149) | `STEP 20 — CLIENT DATA PARTITION` |
| Delta L2 norm (pre-clip, pre-DP) | ~0.55 – 0.68 (measured: ~0.63) | `DELTA / SAFETY CLAMP` → "Delta L2 norm before clamp" |
| Safety clamp engagement | should NOT engage | no `[SAFETY-CLAMP] ... ENGAGED` line anywhere in the log |
| DP clipping applied | **False** (0.55–0.68 is below the 0.85 clip threshold) | `STEP 1 - CLIPPING` |
| DP noise norm (added) | ~450 | `STEP 2 - NOISE` |
| Epsilon (ε) | **5.302585** exactly | `STEP 4 - PRIVACY ACCOUNTING` |
| Encrypted upload size | **~1.44 MB** (1,506,992 bytes) | `SECURE TRANSPORT` → "Payload size" |
| Chunks | 2 (1 MB chunk size) | same section |
| Final status | `Round 1 update submitted`, `Overall E2E status: SUCCESS` | end of run |
| Wall time per client | **~60–90 seconds** (typically ~70s) | not printed directly — time the run yourself |

If trainable params show 109M+ instead of 281,254, `FREEZE_TEXT_ENCODER` isn't
active — check the environment. If delta L2 is near 0 or the run trains in a handful
of steps, `epochs`/`lr` fell back to old defaults — check `SUPERVISED_EPOCHS` /
`SUPERVISED_LR` aren't being overridden to something stale in your shell.

### MODE B — Sharded (genuine federation)

Each client trains on its own disjoint, class-balanced ~50-record shard —
`CLIENT_SHARD_ID` (0, 1, or 2) selects which shard; `CLIENT_N_SHARDS=3` and
`CLIENT_SHARD_SEED=20240` must be identical across all three so they compute
the same partition and each takes a genuinely different, non-overlapping
piece of it (`stratified_shard()`, `trainer_mentalbert_privacy.py`).

**⚠ PowerShell env vars persist for the rest of the session.** `$env:CLIENT_SHARD_ID
= "0"` stays set until you change it or close the terminal — it does **not**
reset between commands. You must set `CLIENT_SHARD_ID` again, explicitly,
**immediately before each client's invocation**. If you forget to update it
before client 2, client 2 will silently reuse client 1's shard — both will
train on the same ~50 records, the third client will get a different shard,
and the round will not be a genuine 3-way partition even though it looks
identical to one in the logs.

```powershell
# Client 1 — shard 0
$env:CLIENT_SHARD_ID = "0"
$env:CLIENT_N_SHARDS = "3"
$env:CLIENT_SHARD_SEED = "20240"
Get-Content "trainer_outputs\demo_blank_stdin.txt" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run1.log"

# Client 2 — shard 1 (CLIENT_SHARD_ID MUST be changed here, every time)
$env:CLIENT_SHARD_ID = "1"
Get-Content "trainer_outputs\demo_blank_stdin.txt" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run2.log"

# Client 3 — shard 2 (CLIENT_SHARD_ID MUST be changed here, every time)
$env:CLIENT_SHARD_ID = "2"
Get-Content "trainer_outputs\demo_blank_stdin.txt" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run3.log"
```

**Expected values (Mode B)** — measured in Step 20 Run A
(`docs/IMPLEMENTATION_NOTES.md`, "Step 20" section), reproduced exactly by
this seed/n_shards combination:

| Client (shard) | Shard size | Class counts (pos/neg) | Achieved positive ratio | Delta L2 |
|---|---|---|---|---|
| 0 | 50 | 15 / 35 | 0.3000 | ~0.33 |
| 1 | 50 | 15 / 35 | 0.3000 | ~0.37 |
| 2 | 49 | 14 / 35 | 0.2857 | ~0.34 |

Look for `STEP 20 — CLIENT DATA PARTITION` in each client's log — `Partition
mode: iid`, and the shard/size/ratio matching its row above. **The three delta
L2 values should DIFFER from each other** (unlike Mode A, where all three
clients see the same 149 records) — that difference is the check that
sharding actually took effect, not just that the env vars were accepted.

### MODE C — Non-IID (optional, for showing genuine client drift)

Same as Mode B, plus `CLIENT_NONIID_ALPHA=0.5` set alongside the other three
vars for all three clients. Instead of each shard tracking the corpus's own
~30% positive ratio, each shard's share of each class is drawn from a
Dirichlet(alpha) distribution — a low alpha concentrates a class into fewer
shards, producing genuinely different client distributions rather than three
same-ratio samples of the same data. This is, as of this session, **the only
path to genuine client heterogeneity anywhere in this system** — and the
precondition for FedProx (its proximal term has nothing to correct without
real non-IID divergence between clients) to ever be meaningfully testable
here, not just mechanically runnable.

```powershell
# Client 1 — shard 0, non-IID
$env:CLIENT_SHARD_ID = "0"
$env:CLIENT_N_SHARDS = "3"
$env:CLIENT_SHARD_SEED = "20240"
$env:CLIENT_NONIID_ALPHA = "0.5"
Get-Content "trainer_outputs\demo_blank_stdin.txt" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run1.log"

# Client 2 — shard 1 (CLIENT_SHARD_ID MUST be changed here, every time)
$env:CLIENT_SHARD_ID = "1"
Get-Content "trainer_outputs\demo_blank_stdin.txt" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run2.log"

# Client 3 — shard 2 (CLIENT_SHARD_ID MUST be changed here, every time)
$env:CLIENT_SHARD_ID = "2"
Get-Content "trainer_outputs\demo_blank_stdin.txt" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run3.log"
```

**Expected values (Mode C, alpha=0.5)** — measured in Step 20 Run B
(`docs/IMPLEMENTATION_NOTES.md`, "Step 20" section). **The exact split depends
on the alpha and seed** — this table is what `alpha=0.5, seed=20240` produces;
a different alpha or seed will produce a different (still deterministic,
still reproducible) skew:

| Client (shard) | Shard size | Class counts (pos/neg) | Achieved positive ratio | Delta L2 |
|---|---|---|---|---|
| 0 | 73 | 11 / 62 | 0.1507 | ~0.374 |
| 1 | 40 | 1 / 39 | 0.0250 | ~0.265 |
| 2 | 36 | 32 / 4 | 0.8889 | ~0.275 |

Client 1 here is nearly all-negative, client 2 nearly all-positive — genuinely
different distributions, not three samples of the same ~30/70 mix. Look for
`Partition mode: non-iid` and `Dirichlet alpha: 0.5` in each client's log.

**Switching modes in the same terminal**: these env vars persist. Either open
a fresh terminal for a different mode, or explicitly clear them first:
```powershell
Remove-Item Env:CLIENT_SHARD_ID, Env:CLIENT_N_SHARDS, Env:CLIENT_SHARD_SEED, Env:CLIENT_NONIID_ALPHA -ErrorAction SilentlyContinue
```

**Honest disclosure — read this before demonstrating Mode B or C.** Modes B
and C produce a **lower** F1 than Mode A, not a higher or comparable one:
Step 20's measured aggregated F1 was **0.0000** (Mode B/C) vs. **0.6667**
(Mode A, the identical-partition baseline, Step 14 ARM 1 round 1) — because
each client gets only ~15 positive examples instead of 44. Step 21's offline
epoch sweep tested epochs in {10, 20, 30, 50} on a single shard and found
**no epoch count recovers non-degenerate F1** — this is a **corpus-size
floor, not a tuning problem**. The arithmetic: 55 positive participants in
the whole 186-row corpus, 44 of them land in the 149-record train split (11
held out in the global eval 37), leaving ~15 positives per client once split
three ways — no architecture or training-schedule choice changes that.
**Therefore: Mode A is the better demo of the pipeline; Mode B/C is the
honest demonstration of federation. They show different things, and both are
worth running** — just don't present Mode B/C's F1 as a regression to be
explained away, or Mode A's F1 as evidence of genuine federated learning.
Also expected, not a fault: **every per-client delta L2 in Mode B and C falls
outside the Fix E4 calibration range [0.625, 0.729]** (`clip_norm=0.85` was
calibrated against 149-record training; ~50-record shards produce
systematically smaller deltas) — documented in
`docs/IMPLEMENTATION_NOTES.md`'s Step 20 section, not retuned.

---

## TERMINAL 5 — global model update display

Start this **before or during** Terminal 4's client runs, so it live-streams the
aggregation the moment the 3rd update lands:

```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
Get-Content "trainer_outputs\demo_orchestrator.log" -Wait -Tail 0 |
    Select-String -Pattern "Aggregation starting|Aggregation status|Round \d+ complete|GLOBAL MODEL PERSISTENCE|Target round|Computed hash|Persistence action|Round \d+ ready"
```
Expected output, appearing live once the 3rd client update is accepted:
```
Aggregation starting — round=1 updates=3 algorithm=trimmed_mean trim_ratio=0.1
Aggregation status   : PASS
Round 1 complete — 3 updates aggregated
              GLOBAL MODEL PERSISTENCE
Target round        : 2
Computed hash       : <sha256>
Persistence action  : CREATED
Round 2 ready — state=Collecting
```

After that appears, press Ctrl+C to stop the tail, then run this one-shot MongoDB
confirmation (the definitive, database-level proof for your mentor):
```powershell
$py = @'
import pymongo
c = pymongo.MongoClient("mongodb://localhost:27017")
db = c["federated_multimodal"]
print("model_updates:", db.model_updates.count_documents({}))
print("receipts:", db.receipts.count_documents({}))
print("global_models:", db.global_models.count_documents({}))
for doc in db.global_models.find():
    print("global_models record:", {k: v for k, v in doc.items() if k != "_id"})
for doc in db["fs.files"].find({"filename": {"$regex": "global_model"}}):
    print("GridFS file:", doc.get("filename"), doc.get("length"), doc.get("_id"))
'@
$py | & ".venv\Scripts\python.exe" -
```
Expected: `model_updates: 3`, `receipts: 3`, `global_models: 1`, and the printed
`global_models` record's `file_id` matches the printed GridFS `_id` — proof the
aggregated global model is durably persisted. The GridFS file's `length` should be
**~1.4–1.5 MB** (three 281,254-parameter updates aggregated, not the old ~439 MB —
that figure was from before Fix B froze the text encoder and stopped transmitting it).

---

## Clean shutdown (after the demo)

```powershell
# Terminal 2: Ctrl+C, or from any terminal:
Get-Process orchestrator -ErrorAction SilentlyContinue | Stop-Process -Force

# MongoDB (graceful, no Administrator needed):
$py = @'
import pymongo
c = pymongo.MongoClient("mongodb://localhost:27017")
try:
    c.admin.command("shutdown")
except Exception:
    pass
'@
$py | & ".venv\Scripts\python.exe" -
```

---

## Honest disclosures

What to say if your mentor asks "is this real data?" — yes for the content (real
DAIC-WOZ interview transcripts, participant turns only, + pre-extracted wav2vec2 audio
embeddings + DenseNet video embeddings, frozen in
`dataset_build\daic_records_multimodal_participant_only.parquet`). Four honest caveats,
established across this branch's investigation — worth having ready, not something to
wait to be asked about individually:

1. **One physical device, three sequential submissions.** All 3 aggregated updates in
   this demo come from one enrolled device on one machine (see the top of this file).
   The full security/privacy/aggregation mechanics run correctly and on real data, but
   this is not independent multi-institution federation.

2. **Warm-start does not currently work correctly across rounds ("Defect A" and
   "Defect B", `docs/IMPLEMENTATION_NOTES.md`, "Step 12 findings").** The aggregator
   persists an *averaged delta* (a small correction, not model weights), and the
   client's warm-start code loads it via `load_state_dict(..., strict=False)` as if
   it *were* absolute weights — correct only when fed real weights, silently wrong
   when fed a delta. Separately, no two client processes share a common random
   initialisation unless `GLOBAL_INIT_SEED` is explicitly set (this demo does not set
   it, matching production behaviour) — so even a correctly-loaded warm-start would be
   averaging deltas computed from different starting points. **This basic 5-terminal
   demo only ever runs round 1** (no warm-start involved, `global_model_available` is
   false), so neither defect is visible here — they only manifest starting round 2,
   which this demo doesn't reach. Both are documented, understood, and intentionally
   not fixed in this branch (see "Fix F" below).

3. **Trimmed-mean aggregation, at this demo's 3-client scale, is not doing what its
   name implies.** `_aggregate_tensor()`'s trimmed-mean (`trim_ratio=0.1`) degenerates
   at n=3 to keeping exactly 1 of 3 sorted values per coordinate — a coordinate-wise
   *median*-of-3, not an average, and not a Byzantine-robust aggregator at this client
   count (median-of-3 is no more attack-resistant than mean-of-3 here). This session's
   investigation (`docs/IMPLEMENTATION_NOTES.md`, Step 13) measured that plain mean
   aggregation denoises measurably better at n=3 (delta L2 260.25 vs. 301.80 on
   identical data) with no robustness given up — but making that the live default
   requires changing a hardcoded literal at
   `server/orchestration_agent/src/grpc/server.rs:1499`, a Rust change intentionally
   **not made** in this branch (see "Fix F" below).

4. **A single client's local metrics (if you look at `metrics.json` or the
   `EVALUATION METRICS` section of a client's log) are LOCAL and PRE-DP.** They
   measure this one client's held-out accuracy/F1/etc. *before* DP noise, *before*
   aggregation — they say nothing about the aggregated global model's utility. This
   session ran that measurement properly for the first time (Steps 12–17,
   `docs/IMPLEMENTATION_NOTES.md`): reconstructing the true aggregated global model
   and evaluating it on held-out data. Headline result: **at this branch's DP
   configuration (noise_multiplier=1.0) and 3-client scale, aggregated utility
   collapses to a degenerate, saturated prediction — not a graceful
   privacy/utility tradeoff** — and a follow-up fix for a separate multi-round
   training divergence (Step 16's LR decay) does not recover it; if anything it
   *worsens* the per-round noise-to-signal ratio (452x → 1503x over 5 rounds, Step
   17), because decay shrinks the signal while the DP noise floor stays fixed. If
   your mentor wants to see that measurement, it isn't part of this 5-terminal demo —
   point them at `docs/IMPLEMENTATION_NOTES.md`'s Step 12–17 sections and
   `scripts/evaluate_global_model.py` / `scripts/run_step14_multiround.py`.

**Known, understood, deliberately not started in this branch ("Fix F" and friends)** —
correctness work stopped after Step 17; these four are documented, not blocking, and
should not be started without a fresh scoping conversation:
- Defect A (aggregator persists a delta; client loads it as absolute weights)
- Defect B (no shared initialisation base across clients)
- A real cumulative-epsilon accountant across rounds (today's cumulative figure is a
  naive additive upper bound only — 26.5 at 5 rounds noise_multiplier=1.0, not a tight
  sequential RDP composition)
- `server.rs:1499`: `"mode": "trimmed_mean"` → `"mode": "mean"` (Rust, requires a
  scoped Rust change this investigation deliberately did not make)

DP (Gaussian, `clip_norm=0.85`, `noise_multiplier=1.0`, `ε=5.302585`, `δ=1e-5`),
AES-GCM encryption, TPM-backed ECDSA device signing, mTLS transport, and trimmed-mean
aggregation (`trim_ratio=0.1`) are all live and unmodified by this session for the
purposes of this basic demo — only the training regime (Fix A/B/E1–E5, Step 16) and
the diagnostic/evaluation tooling (Steps 12–17) changed.
