# Manual E2E Runbook

## 1. Purpose

This runbook reproduces, by hand in VS Code PowerShell terminals, the live mechanical
end-to-end pipeline that was verified working on this machine:

```
MongoDB (D:\MongoDB\data)
  -> Rust orchestrator (mTLS, port 50051)
  -> Python client (runtime.federated_client)
  -> LDA preprocessing (ffmpeg + Whisper ASR fallback)
  -> MentalBERT local training
  -> Differential Privacy (Gaussian, eps = 5.302585092994046)
  -> AES-GCM encryption
  -> chunked upload (gRPC, up to 1 GiB)
  -> MongoDB / GridFS acceptance
  -> trimmed-mean aggregation (after 3 accepted updates)
  -> global model persisted to GridFS + `global_models` record
```

**Read §21 before you run anything or show this to anyone.** This is an *engineering*
demonstration of the pipeline mechanics — it does not use genuine DAIC-WOZ data and does
not demonstrate independent multi-device federation.

---

## 2. Current Environment

| Item | Value |
|---|---|
| Repository | `D:\Download D\BE PIPELINE\Capstone-` |
| OS | Windows, PowerShell |
| Python venv | `D:\Download D\BE PIPELINE\Capstone-\.venv` |
| Venv Python | `D:\Download D\BE PIPELINE\Capstone-\.venv\Scripts\python.exe` |
| MongoDB version | 8.0.23 |
| MongoDB data dir | `D:\MongoDB\data` (migrated off C: — see §4) |
| MongoDB port | `127.0.0.1:27017` |
| MongoDB binary | `C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe` |
| Rust orchestrator dir | `D:\Download D\BE PIPELINE\Capstone-\server\orchestration_agent` |
| Orchestrator config | `server\orchestration_agent\config\orchestrator.toml` |
| Orchestrator port | `0.0.0.0:50051` |
| Orchestrator binary (after build) | `server\orchestration_agent\target\release\orchestrator.exe` |
| TLS | mTLS enabled, certs at `server\orchestration_agent\certs\{ca,server}.{pem,key}` |
| Upload cap | `MAX_UPDATE_BYTES = 1024 * 1024 * 1024` (1 GiB — already patched in `server.rs`) |
| Client identity home | `%USERPROFILE%\.federated` (or a redirected home — see §4/§18) |
| Client cert | `%USERPROFILE%\.federated\keys\client.pem` (already enrolled) |
| Test input | `videos\sample.mp4` (repo fixture — **not** DAIC-WOZ) |

---

## 3. One-Time Prerequisites

Run these **once**. If a check already passes, skip the install.

```powershell
cd "D:\Download D\BE PIPELINE\Capstone-"

# --- MongoDB 8.0.23 binary present? ---
Test-Path "C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe"
# Expected: True. If False: winget install --id MongoDB.Server -e --version 8.0.23

# --- Rust / cargo ---
cargo --version
# Expected: cargo 1.9x.x or similar. If missing: install via rustup.rs

# --- protoc (only needed if you ever rebuild the Rust server) ---
Get-Command protoc -ErrorAction SilentlyContinue
# If missing AND you need to rebuild: winget install --id Google.Protobuf -e

# --- OpenSSL (Git bundles it) ---
Test-Path "C:\Program Files\Git\mingw64\bin\openssl.exe"
# Expected: True (comes with Git for Windows)

# --- ffmpeg ---
Get-Command ffmpeg -ErrorAction SilentlyContinue
# If missing: winget install --id Gyan.FFmpeg -e

# --- Python venv + key packages ---
& ".venv\Scripts\python.exe" -c "import grpc, pydantic, torch, transformers; print('grpc', grpc.__version__); print('pydantic', pydantic.VERSION); print('torch', torch.__version__)"
# Expected: grpc 1.83.0, pydantic 2.11.7, torch 2.8.0+cu128
# If pydantic missing: & ".venv\Scripts\python.exe" -m pip install pydantic==2.11.7

# --- Device identity already provisioned? ---
Test-Path "$env:USERPROFILE\.federated\tpm\device_pubkey.pem"
Test-Path "$env:USERPROFILE\.federated\keys\client.pem"
# Expected: both True (enrollment already completed on this machine).
# If False, see §9.

# --- MentalBERT already staged for the client? ---
Test-Path "$env:USERPROFILE\.federated\models\mentalbert\pytorch_model.bin"
# If False, see §12.
```

**Do not** re-run `winget install` for anything that already checks out True/present above.

---

## 4. Important Warning About C: vs D:

MongoDB's data directory was moved from `C:\Program Files\MongoDB\Server\8.0\data` to
**`D:\MongoDB\data`** because C: ran low on free space (below MongoDB's own 500 MiB
write-safety threshold). The **old C: copy still exists and was never deleted** — do not
delete it without deciding so deliberately.

There is also a Windows **Service** named `MongoDB` (`StartType: Automatic`) that still
points at the **old C: path** (its `mongod.cfg` under `Program Files` could not be edited
without Administrator rights). If that service is ever running, it will occupy port 27017
with the **stale, empty-of-your-work C: data**, not your real D: data. Always check for
this before starting your own instance (§7 includes the check).

The client's own working directory (`~/.federated/data/secure_store`) can also fill C: —
if you hit `MemoryError`/`OSError: No space left on device` during a run, see §18's
"MemoryError during encryption" entry, which redirects the client's home to D: for that
one process via `$env:USERPROFILE`.

---

## 5. Terminal Layout

### TERMINAL 1 — MongoDB + Rust Orchestrator
Starts both long-running services. Stays open for the whole session.

### TERMINAL 2 — Client: Enrollment (if needed) + First Update
Runs enrollment once if `client.pem` doesn't exist, then the client's first `run-once`.

### TERMINAL 3 — Client: Second and Third Updates (same device) + Verification
Runs `run-once` **twice more**, sequentially, in the **same terminal**, using the
**same enrolled device identity** as Terminal 2. Also used for MongoDB verification
queries between runs.

> **This machine has exactly one `FederatedDeviceKey` / one device identity
> (`~/.federated/tpm/device_pubkey.pem`).** Running the client multiple times — even in
> different terminals — always presents the **same** `device_id` to the server. The
> 3-update aggregation trigger (`round.updates.len() >= 3` in `server.rs`) does not check
> for distinct devices, so three same-device runs will mechanically trigger it. **This is
> not three independent federated clients** — do not describe it that way in any report.

---

## 6. Preflight Checks

Run in any terminal before starting anything.

```powershell
cd "D:\Download D\BE PIPELINE\Capstone-"

# Working directory
Get-Location
# Expected: D:\Download D\BE PIPELINE\Capstone-

# venv exists
Test-Path ".venv\Scripts\python.exe"
# Expected: True

# MongoDB reachable + correct dbPath (only meaningful once it's started, §7)
& ".venv\Scripts\python.exe" -c "import pymongo; c=pymongo.MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=3000); print(c.admin.command('ping')); print(c.admin.command('getCmdLineOpts')['parsed']['storage'])"
# Expected: {'ok': 1.0}  and  {'dbPath': 'D:\\MongoDB\\data'}
# If dbPath shows the C: path instead, see §18 "MongoDB service still configured for old C: dbPath"

# Port 50051 status (should be free before you start the server)
Get-NetTCPConnection -LocalPort 50051 -ErrorAction SilentlyContinue
# Expected: no output (free) before starting; Listen state after §8

# TLS files present
Test-Path "server\orchestration_agent\certs\ca.pem"
Test-Path "server\orchestration_agent\certs\server.pem"
Test-Path "server\orchestration_agent\certs\server.key"
# Expected: all True

# Device identity
Test-Path "$env:USERPROFILE\.federated\tpm\device_pubkey.pem"
Test-Path "$env:USERPROFILE\.federated\keys\client.pem"
# Expected: both True

# Python import safety (grpc from site-packages, agents from installer/runtime)
& ".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, r'D:\Download D\BE PIPELINE\Capstone-'); sys.path.append(r'D:\Download D\BE PIPELINE\Capstone-\installer\runtime'); import grpc; print('grpc:', grpc.__file__); print('ssl_channel_credentials:', hasattr(grpc,'ssl_channel_credentials')); import agents, runtime.pipeline; print('IMPORTS OK')"
# Expected:
#   grpc: ...\.venv\Lib\site-packages\grpc\__init__.py   (NOT installer\runtime\grpc)
#   ssl_channel_credentials: True
#   IMPORTS OK

# ffmpeg / openssl on PATH for THIS terminal
Get-Command ffmpeg -ErrorAction SilentlyContinue
Get-Command openssl -ErrorAction SilentlyContinue
# If either is missing, see §7/§8/§10 for the exact PATH additions

# Required model/config files
Test-Path "$env:USERPROFILE\.federated\configs\local_config.yaml"
Test-Path "$env:USERPROFILE\.federated\models\mentalbert\pytorch_model.bin"
# Expected: both True (one-time prep, §12)

# Test input staged
Test-Path "videos\sample.mp4"
Test-Path "$env:USERPROFILE\.federated\data\input\sample.mp4"
# First should be True (repo fixture). Second must be True before running the client — see §11.
```

---

## 7. Start MongoDB

**No Administrator rights required** — this starts `mongod.exe` directly (not via the
Windows Service, which still points at the old C: path).

```powershell
# TERMINAL 1

# 0. Make sure the auto-started Windows Service (if running) is not squatting on the port
Get-Service -Name MongoDB -ErrorAction SilentlyContinue | Select-Object Name, Status, StartType
# If Status is "Running": that is the STALE C:-path service. Stopping it needs Administrator
# rights (Stop-Service will fail otherwise). If you have an elevated terminal available, run
# `Stop-Service MongoDB -Force` there first. If you don't, and port 27017 is already occupied
# by mongod.exe, see step 1's fallback.

Get-NetTCPConnection -LocalPort 27017 -ErrorAction SilentlyContinue
# If this shows a listener and you can't stop the service, you cannot proceed to step 1 —
# stop here and get Administrator access, or use a different port for a throwaway test.

# 1. Start mongod directly against D:\MongoDB\data
New-Item -ItemType Directory -Path "D:\MongoDB\log" -Force | Out-Null
& "C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe" `
    --dbpath "D:\MongoDB\data" `
    --port 27017 `
    --bind_ip 127.0.0.1 `
    --logpath "D:\MongoDB\log\mongod.log" `
    --logappend
```

Leave this running in the foreground of Terminal 1 (or add `-run_in_background`-style job
control of your choice). Open a **second pane/split** in Terminal 1, or verify from
Terminal 2/3:

```powershell
& "D:\Download D\BE PIPELINE\Capstone-\.venv\Scripts\python.exe" -c "import pymongo; c=pymongo.MongoClient('mongodb://localhost:27017'); print(c.admin.command('ping')); print(c.admin.command('getCmdLineOpts')['parsed']['storage'])"
```
Expected: `{'ok': 1.0}` and `dbPath: D:\MongoDB\data`.

---

## 8. Start Rust Orchestrator

Run this **in a new PowerShell tab/pane within Terminal 1**, alongside `mongod` (or after
backgrounding it with your own job-control approach — the important thing is both stay
running for the session).

```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-\server\orchestration_agent"

$env:MONGO_URI = "mongodb://localhost:27017"
$env:PATH = "C:\Program Files\Git\mingw64\bin;" + $env:PATH   # openssl for EnrollDevice CSR signing

# IMPORTANT: use cmd.exe for redirection, NOT PowerShell's native `>`/`*>>`.
# PowerShell redirection writes UTF-16LE, which enroll_step5.py (§9) cannot parse for the OTP.
& "$env:WINDIR\System32\cmd.exe" /c "cargo run --release > `"D:\Download D\BE PIPELINE\Capstone-\trainer_outputs\orchestrator_server.log`" 2>&1"
```

This blocks in the foreground (that's fine — it's your long-running server). To verify from
another terminal:

```powershell
Get-NetTCPConnection -LocalPort 50051 -ErrorAction SilentlyContinue
# Expected: State = Listen

Get-Content "D:\Download D\BE PIPELINE\Capstone-\trainer_outputs\orchestrator_server.log" -Tail 15
# Expected lines (in order):
#   INFO orchestrator: MongoDB connected: mongodb://localhost:27017
#   INFO orchestrator: MongoDB indexes ensured
#   [DEV] Enrollment OTP: XXXXXX (valid for 10 minutes)
#   INFO orchestrator::grpc::server: hydrate: resumed N already-verified update(s)...  (only if MongoDB already has verified receipts for round 1)
#   INFO orchestrator::grpc::server: [SERVER] mTLS mode — binding to 0.0.0.0:50051
#   [SERVER] Running in mTLS mode on 0.0.0.0:50051
```
The OTP is valid for 10 minutes — proceed to enrollment (§9) promptly if you need a fresh one.

---

## 9. Device Enrollment

**Skip this section entirely if `%USERPROFILE%\.federated\keys\client.pem` already exists**
(check in §6) — the certificate is a standalone signed cert independent of server restarts,
and the device is already recorded in MongoDB's `devices` collection (also independent of
server restarts, since it's durable, not in-memory).

Only re-enroll if `client.pem` is missing or you intentionally want a fresh device identity.

```powershell
# TERMINAL 2
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
& ".venv\Scripts\python.exe" enroll_step5.py "D:\Download D\BE PIPELINE\Capstone-\trainer_outputs\orchestrator_server.log"
```

Expected output, in order:
```
[1] Device pubkey loaded (177 bytes) from ...\device_pubkey.pem
[2] Generating RSA-2048 client key and CSR...
[TLS]  Server-TLS channel ready -> 127.0.0.1:50051
[4] Sending RequestEnrollment...
[4] Accepted — fingerprint: <16 hex chars>
[5] Waiting for OTP in server log (...)...
[5] OTP extracted: <6 digits>
[6] Sending EnrollDevice (OTP=<6 digits>)...
[7] client.pem written -> ...\client.pem (~1480 bytes)
ENROLLMENT COMPLETE
```

Verify the cert:
```powershell
& "C:\Program Files\Git\mingw64\bin\openssl.exe" verify -CAfile "server\orchestration_agent\certs\ca.pem" "$env:USERPROFILE\.federated\keys\client.pem"
# Expected: ...client.pem: OK
& "C:\Program Files\Git\mingw64\bin\openssl.exe" x509 -in "$env:USERPROFILE\.federated\keys\client.pem" -noout -subject -issuer -dates
# Expected: subject=CN=federated-device, issuer=CN=Federated-Root-CA, valid 1 year
```

---

## 10. Client Environment

This is the exact setup that avoids the gRPC shadowing bug (`installer\runtime\grpc`
otherwise shadows the real `grpcio` package if put on `sys.path` the naive way).

**Create a small, reusable launcher** (one-time — write this file once):

```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
@'
import sys
import os

REPO_ROOT = r"D:\Download D\BE PIPELINE\Capstone-"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)                              # runtime.* resolves here
sys.path.append(os.path.join(REPO_ROOT, "installer", "runtime"))  # agents.* — APPENDED, not prepended

os.environ.setdefault("FED_SERVER", "127.0.0.1:50051")
sys.argv = ["federated_client", "run-once"]

import runpy
runpy.run_module("runtime.federated_client", run_name="__main__")
'@ | Set-Content -Path "run_client.py" -Encoding utf8
```

Why this order matters: `sys.path.insert(0, REPO_ROOT)` puts the repo root first so
`runtime.pipeline` resolves to the real source. `installer\runtime` is **appended to the
end**, after the venv's `site-packages` is already on `sys.path` — so when `import grpc`
runs (triggered from inside `runtime.grpc_client`), Python finds the real `grpcio` package
in `site-packages` first and never reaches `installer\runtime\grpc\` at all. **Never** use
a plain `$env:PYTHONPATH = "...\installer\runtime"` — `PYTHONPATH` entries are *prepended*,
which recreates the shadowing bug.

**Per-run environment** (set these in Terminal 2 and Terminal 3 before every `run_client.py`
invocation):

```powershell
$env:PATH = "C:\Program Files\Git\mingw64\bin;" + $env:PATH   # openssl (harmless if client doesn't need it, cheap to set)
$env:PATH = "<ffmpeg bin dir found via Get-Command ffmpeg>;" + $env:PATH   # only if ffmpeg isn't already on PATH
$env:PYTHONUTF8 = "1"    # avoids UnicodeEncodeError on non-ASCII characters in trainer print() output
```

---

## 11. Input Data

The live pipeline's `mode="session"` LDA step globs `*.mp4` **directly** inside
`~/.federated/data/input/` (no subdirectory recursion). The repository's own test fixture
`videos\sample.mp4` is what was used for the verified run.

**This is NOT DAIC-WOZ data.** No raw DAIC-WOZ session recordings exist on this machine —
only the pre-extracted, frozen `dataset_build\daic_records_multimodal.parquet`, which this
LDA code path cannot consume (see `README.md` and the project's own audit history for why).

Stage it (one-time per machine, or repeat if `~/.federated` was ever wiped/redirected):
```powershell
New-Item -ItemType Directory -Path "$env:USERPROFILE\.federated\data\input" -Force | Out-Null
Copy-Item "videos\sample.mp4" "$env:USERPROFILE\.federated\data\input\sample.mp4" -Force
```

Verify:
```powershell
Test-Path "$env:USERPROFILE\.federated\data\input\sample.mp4"
# Expected: True
```

---

## 12. Model/Runtime Preparation

**One-time setup** (skip anything already present, per §3/§6):

```powershell
# local_config.yaml — the LDA/DP/encryption operational config
New-Item -ItemType Directory -Path "$env:USERPROFILE\.federated\configs" -Force | Out-Null
Copy-Item "installer\runtime\configs\local_config.yaml" "$env:USERPROFILE\.federated\configs\local_config.yaml" -Force

# MentalBERT — copy the already-downloaded, SHA-pinned snapshot (do NOT re-download)
$snap = "$env:USERPROFILE\.cache\huggingface\hub\models--mental--mental-bert-base-uncased\snapshots\24809aa822c76639760d0d934742d1b42f89942f"
New-Item -ItemType Directory -Path "$env:USERPROFILE\.federated\models\mentalbert" -Force | Out-Null
Copy-Item "$snap\*" "$env:USERPROFILE\.federated\models\mentalbert\" -Recurse -Force
# If $snap does not exist, the model was never downloaded on this machine — run any script
# that loads `mental/mental-bert-base-uncased` once via transformers to populate the HF cache
# first (do not point the client at an unpinned revision).

# Device signer binary + TPM key (skip if device_pubkey.pem already exists, §3/§6)
New-Item -ItemType Directory -Path "$env:USERPROFILE\.federated\bin" -Force | Out-Null
Copy-Item "installer\runtime\windows_signer.exe" "$env:USERPROFILE\.federated\bin\windows_signer.exe" -Force
New-Item -ItemType Directory -Path "$env:USERPROFILE\.federated\tpm" -Force | Out-Null
& "$env:USERPROFILE\.federated\bin\windows_signer.exe" --init
& "$env:USERPROFILE\.federated\bin\windows_signer.exe" --pubkey "$env:USERPROFILE\.federated\tpm\device_pubkey.pem"
```

Verify:
```powershell
Test-Path "$env:USERPROFILE\.federated\configs\local_config.yaml"
Test-Path "$env:USERPROFILE\.federated\models\mentalbert\pytorch_model.bin"
Test-Path "$env:USERPROFILE\.federated\tpm\device_pubkey.pem"
# Expected: all True
```

**Note on OpenFace/openSMILE:** `local_config.yaml` references OpenFace and openSMILE
binaries under `~/.federated/deps/windows/...` that are **not installed on this machine**.
This is fine — the config validator logs warnings and the pipeline falls back gracefully
(audio/video get processed at reduced fidelity; ASR still runs via the HF Whisper fallback).
Do not treat these warnings as errors.

---

## 13. Execute Client Update

```powershell
# TERMINAL 2 (first run) — set env vars from §10, then:
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
"" | & ".venv\Scripts\python.exe" "run_client.py" *>&1 | Tee-Object -FilePath "trainer_outputs\client_run.log"
```

The leading `"" |` feeds one blank line to stdin for the interactive physician-feedback
prompt (`Enter corrected PHQ (or Enter to keep):`) — pressing Enter accepts the model's own
prediction rather than manually overriding it, a valid built-in path through that prompt,
not a workaround. If you ever stage more than one input record, pipe more blank lines:
`(1..10 | ForEach-Object { "" }) | & ".venv\Scripts\python.exe" "run_client.py"`.

Expected log stages, in order:
```
[pipeline] Running LDA...
[pipeline] LDA done in ~17-20s (or longer on first run if Whisper needs to download)
[pipeline] Running trainer (mode=supervised)...
  -- Physician feedback loop --
  Model PHQ: <value>  (class prob positive: <value>)
  Enter corrected PHQ (or Enter to keep):
[pipeline] Trainer done in ~12-18s
[pipeline] Applying DP noise...
[pipeline] DP done: L2 before=... after=... eps=5.302585
[pipeline] Finalizing encryption...
[pipeline] Streaming update to server...
[pipeline] Streaming 585049324 bytes in 558 chunks (sha256=...)
[pipeline] Upload complete — server_handle=...
[pipeline] Submitting receipt to server...
[TPM] Opened existing key 'FederatedDeviceKey'
[OK] Signed 72 bytes
[pipeline] ✅ Round 1 update submitted
Run-once pipeline complete ✓
```

If you see `grpc._channel._InactiveRpcError ... NOT_FOUND ... "already submitted"` right
after `Upload complete`, that is a benign client-side retry artifact — see §18. Check
MongoDB (§15) to confirm whether the update actually landed; it usually already did.

---

## 14. Three-Update Aggregation

Aggregation requires **3 accepted updates** in the current round
(`round.updates.len() >= 3`, checked in `server.rs`'s `submit_receipt` handler — a pure
count, with no per-device distinctness check).

**On this machine there is exactly one enrolled device identity.** Reaching 3 accepted
updates means running the client **three times sequentially**, all under that same
`device_id`. This is what actually happened during the verified run — do not attempt to
simulate "three independent clients" by any other means (separate terminals do not create
separate identities; the identity is tied to one Windows CNG key,
`FederatedDeviceKey`, shared by every process under this Windows account).

```powershell
# TERMINAL 3 — same env vars as §10, then run twice more, one after another:
Set-Location "D:\Download D\BE PIPELINE\Capstone-"

"" | & ".venv\Scripts\python.exe" "run_client.py" *>&1 | Tee-Object -FilePath "trainer_outputs\client_run_2.log"
# wait for "Run-once pipeline complete" before continuing

"" | & ".venv\Scripts\python.exe" "run_client.py" *>&1 | Tee-Object -FilePath "trainer_outputs\client_run_3.log"
```

After the 3rd accepted update, aggregation fires **automatically and asynchronously** on
the server (detached via `tokio::spawn` — it does not block the 3rd client's own response).
Watch the server log (Terminal 1) for:
```
Round 1 complete — 3 updates aggregated
Global model for round 2 stored in GridFS (hash=...)
Round 2 created — state=Collecting
```
This typically completes 30-60 seconds after the 3rd receipt is accepted.

**State explicitly for any report/demo:** *these three updates come from one enrolled
device identity on one machine — this is a mechanical exercise of the aggregation trigger,
not a demonstration of independent multi-client federated learning.*

---

## 15. Verification After Each Update

Run after each `run_client.py` invocation:

```powershell
& "D:\Download D\BE PIPELINE\Capstone-\.venv\Scripts\python.exe" -c "
import pymongo
c = pymongo.MongoClient('mongodb://localhost:27017')
db = c['federated']
print('model_updates:', db.model_updates.count_documents({}))
print('receipts:', db.receipts.count_documents({}))
for doc in db.model_updates.find():
    print('  update: round=%s size=%s verified=%s' % (doc.get('round_id'), doc.get('size_bytes'), doc.get('verified')))
for doc in db.receipts.find():
    print('  receipt: round=%s eps=%s verified=%s' % (doc.get('round_id'), doc.get('epsilon_spent'), doc.get('verified')))
"
```
Expected after run N: `model_updates: N`, `receipts: N`, each entry `size_bytes: 585049324`
(text-only architecture, single input record), `verified: True`, `eps: 5.302585092994046`,
`round_id: 1`.

---

## 16. Verify Aggregation

**Do not rely on log lines alone** — confirm directly in MongoDB after the 3rd update:

```powershell
& "D:\Download D\BE PIPELINE\Capstone-\.venv\Scripts\python.exe" -c "
import pymongo
c = pymongo.MongoClient('mongodb://localhost:27017')
db = c['federated']
print('model_updates:', db.model_updates.count_documents({}))
print('receipts:', db.receipts.count_documents({}))
print('global_models:', db.global_models.count_documents({}))
print()
for doc in db.global_models.find():
    print('global_models record:', doc)
print()
for doc in db['fs.files'].find({'filename': {'\$regex': 'global_model'}}):
    print('GridFS file:', doc)
"
```

Cross-check that `global_models.file_id` **matches** the `_id` of the GridFS
`global_model_round_1.pt` entry, and that `global_models.round_id == 2` (the code stores
the aggregated model under `round_id + 1`, so the *next* round's `GetRound` call sees
`global_model_available = true`).

---

## 17. Expected Successful End State

```
model_updates      = 3      (all round_id=1, size_bytes=585049324, verified=True)
receipts            = 3      (all round_id=1, eps=5.302585092994046, verified=True)
global_models        = 1      (round_id=2, file_id references a real GridFS file, model_hash present)
fs.files             contains a "global_model_round_1.pt" entry, length ≈ 438,786,913 bytes
Round 1              = Complete (in-memory; visible via server log "Round 1 complete — 3 updates aggregated")
Round 2               = Collecting (auto-created; server log "Round 2 created — state=Collecting")
Frozen artifacts      = all 8 pinned SHAs unchanged (see §20)
```

---

## 18. Troubleshooting

| Symptom | Cause | Fix | One-time or per-session? |
|---|---|---|---|
| `grpc` has no `ssl_channel_credentials` | Wrong `grpc` module resolved (shadowed) | Use `run_client.py` from §10, never plain `$env:PYTHONPATH` | Structural — use the launcher every time |
| `grpc` resolves from `installer\runtime` instead of site-packages | `installer\runtime` was **prepended** to `sys.path` (e.g. via `PYTHONPATH` env var) | Same as above — append, never prepend | Structural |
| `No module named 'agents'` | `installer\runtime` not on `sys.path` at all | Use `run_client.py` (§10) | Structural |
| `No module named 'pydantic'` | Documented dependency (`requirements.txt`) not installed in venv | `& ".venv\Scripts\python.exe" -m pip install pydantic==2.11.7` | One-time |
| `No module named 'cv2'` | Optional OpenCV dependency for some video-path fallbacks, not required for the verified path | `& ".venv\Scripts\python.exe" -m pip install opencv-python` if you hit this; otherwise ignore | One-time, only if needed |
| Warnings for missing `librosa`/`webrtcvad`/`pyannote`/`whisper`/`mediapipe` | Optional feature-extraction libraries, not installed | **Expected and non-fatal** — the pipeline uses documented fallbacks (HF Whisper for ASR, etc.). No action needed | N/A |
| `openssl not found` from PowerShell (server-side `certificate signing failed`) | The Rust server shells out to `openssl` for `EnrollDevice`; PowerShell's PATH doesn't include Git's `mingw64\bin` by default | `$env:PATH = "C:\Program Files\Git\mingw64\bin;" + $env:PATH` before starting the server (§8) | Per-session |
| `protoc` missing (only during a Rust rebuild) | Needed by `tonic-build` for `.proto` codegen | `winget install --id Google.Protobuf -e`, then set `$env:PROTOC` to the installed `protoc.exe` path before `cargo build` | One-time |
| `ffmpeg` missing (audio extraction from video fails) | Not installed | `winget install --id Gyan.FFmpeg -e`, add its `bin` dir to PATH | One-time install, per-session PATH |
| Server log is UTF-16 / `enroll_step5.py` times out waiting for OTP | PowerShell's native `>`/`*>>` redirection writes UTF-16LE; the OTP-parsing regex expects UTF-8 | Use the `cmd.exe /c "... > file 2>&1"` pattern from §8, never bare PowerShell redirection for this log | Per-session |
| `certificate signing failed` (client enrollment) | Same root cause as the openssl-PATH item above | Fix the server's PATH (§8) and restart it, then re-enroll | Per-session |
| MongoDB `Error code 14031 (OutOfDiskSpace)` on upload | C: drive too full for MongoDB's 500 MiB write-safety minimum | Already fixed — MongoDB now runs against `D:\MongoDB\data` (§4/§7). If it recurs, check `Get-PSDrive D` for free space, not C: | One-time migration, already done |
| `RESOURCE_EXHAUSTED: update exceeds 500 MB maximum` | Old application-level cap | Already fixed — `MAX_UPDATE_BYTES = 1024*1024*1024` in `server.rs` (already built into the current binary) | One-time, already done |
| `MemoryError` during `json.dumps()` in `encrypt_write()` (client-side, trainer or DP stage) | Low system RAM (this machine has 7.69 GB total) combined with a nearly-full `%USERPROFILE%` drive | Close non-essential apps (browsers etc.) to free RAM before running; if C: is also low, redirect the client's home for one run: `$env:USERPROFILE = "D:\federated_home"` (copy `~/.federated` there first) so secure_store writes land on D: | Per-session, situational |
| Client reports `NOT_FOUND ... "already submitted"` right after a successful-looking upload | `grpc_client.py`'s retry wrapper resent `SubmitReceipt` after a timeout, even though the first attempt had already succeeded server-side | Benign — check MongoDB (§15) before assuming failure; the real receipt is usually already there | N/A — informational |
| Aggregation never completes even though `model_updates` reaches 3 (no `Round N complete` log line, no error either) | `run_aggregation()` was previously `.await`ed synchronously inside the triggering RPC; a client-side timeout/retry could drop that future mid-flight, orphaning an aggregation that had already produced a valid result | Already fixed — aggregation now runs via detached `tokio::spawn`, and a startup hydration step resumes any already-verified-but-unaggregated round from MongoDB automatically on server start | One-time, already done (built into current binary) |
| MongoDB `getCmdLineOpts` shows `dbPath: C:\Program Files\MongoDB\Server\8.0\data` instead of D: | The Windows `MongoDB` Service (StartType: Automatic) auto-started and is squatting on port 27017 with stale C: data | `Get-Service MongoDB` to confirm; stop it (`Stop-Service MongoDB -Force`, needs Administrator) before starting your own `mongod.exe` against D: per §7 | Per-session, only if the service auto-started (e.g. after a reboot) |
| Server or MongoDB port already in use when starting | A previous instance is still running (possibly from an earlier session) | `Get-NetTCPConnection -LocalPort 50051` / `27017`, then `Stop-Process -Id <OwningProcess> -Force` if it's your own stale process (not the Windows service — see above) | Per-session |

---

## 19. Clean Shutdown

Does **not** delete any data.

```powershell
# Stop the client: it's a run-once process, it exits on its own. No action needed
# unless you want to interrupt a run in progress (Ctrl+C in that terminal).

# Stop the Rust orchestrator (Terminal 1, or from any terminal by PID):
Get-NetTCPConnection -LocalPort 50051 -ErrorAction SilentlyContinue | Select-Object OwningProcess
Stop-Process -Id <OwningProcess-from-above> -Force

# Stop MongoDB gracefully (does NOT require Administrator — uses MongoDB's own shutdown command):
& "D:\Download D\BE PIPELINE\Capstone-\.venv\Scripts\python.exe" -c "
import pymongo
c = pymongo.MongoClient('mongodb://localhost:27017')
try:
    c.admin.command('shutdown')
except Exception:
    pass  # connection drop is expected on a clean shutdown
"
```
Verify both are down:
```powershell
Get-NetTCPConnection -LocalPort 50051,27017 -ErrorAction SilentlyContinue
# Expected: no output
```

---

## 20. Evidence Collection

```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"

# Server log
Get-Content "trainer_outputs\orchestrator_server.log" -Tail 100 | Out-File "trainer_outputs\evidence_server_log.txt"

# MongoDB full state
& ".venv\Scripts\python.exe" -c "
import pymongo, json
c = pymongo.MongoClient('mongodb://localhost:27017')
db = c['federated']
out = {
    'model_updates': list(db.model_updates.find({}, {'_id': 0})),
    'receipts': list(db.receipts.find({}, {'_id': 0})),
    'global_models': list(db.global_models.find({}, {'_id': 0})),
    'devices': db.devices.count_documents({}),
}
print(json.dumps(out, indent=2, default=str))
" | Out-File "trainer_outputs\evidence_mongodb_state.json"

# Git state
git status --porcelain | Out-File "trainer_outputs\evidence_git_status.txt"
git diff --stat -- server\orchestration_agent\src\grpc\server.rs | Out-File -Append "trainer_outputs\evidence_git_status.txt"

# Frozen artifact SHA verification (project's own validator, read-only)
& ".venv\Scripts\python.exe" phase22_end_to_end\validate_phase22.py --allow-cpu | Out-File "trainer_outputs\evidence_frozen_sha_check.txt"
```

---

## 21. Important Scientific/Integrity Disclosures

**State these explicitly in any report, demo narration, or write-up that uses this runbook's output:**

1. The live E2E demonstrated here uses `videos\sample.mp4`, the repository's own test
   fixture — **not** genuine DAIC-WOZ raw recordings. No raw DAIC-WOZ session data exists
   on this machine; only the pre-extracted, frozen parquet exists, and the live client's
   LDA path cannot consume that parquet.
2. The three accepted updates all came from **one enrolled device identity**
   (`~/.federated/tpm/device_pubkey.pem`, one Windows CNG key). This is **not** a genuine
   three-independent-device federated-learning experiment.
3. DP parameters (Gaussian, clip=1.0, noise multiplier=1.0, ε=5.302585092994046, δ=1e-5),
   the training recipe, the encryption scheme (AES-GCM), and the aggregation algorithm
   (trimmed-mean, trim_ratio=0.1) are all **unmodified** by anything in this runbook.
4. Do not modify frozen datasets, `PHASE_22_DESIGN.md`, or any scientific parameter merely
   to make a demo pass. If the pipeline fails, diagnose and report — do not silently
   loosen a constraint that changes what's being measured.

---

## 22. Complete Copy-Paste Quick Run

Assumes §3, §11, §12 one-time prep are already done and `client.pem` already exists.

### TERMINAL 1
```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
Get-Service -Name MongoDB -ErrorAction SilentlyContinue | Select-Object Status
# If Status=Running, stop it first (needs Administrator): Stop-Service MongoDB -Force

New-Item -ItemType Directory -Path "D:\MongoDB\log" -Force | Out-Null
& "C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe" --dbpath "D:\MongoDB\data" --port 27017 --bind_ip 127.0.0.1 --logpath "D:\MongoDB\log\mongod.log" --logappend
```
```powershell
# new tab/pane in Terminal 1
Set-Location "D:\Download D\BE PIPELINE\Capstone-\server\orchestration_agent"
$env:MONGO_URI = "mongodb://localhost:27017"
$env:PATH = "C:\Program Files\Git\mingw64\bin;" + $env:PATH
& "$env:WINDIR\System32\cmd.exe" /c "cargo run --release > `"D:\Download D\BE PIPELINE\Capstone-\trainer_outputs\orchestrator_server.log`" 2>&1"
```

### TERMINAL 2
```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
$env:PATH = "C:\Program Files\Git\mingw64\bin;" + $env:PATH
$env:PYTHONUTF8 = "1"

# Only if client.pem is missing:
# & ".venv\Scripts\python.exe" enroll_step5.py "D:\Download D\BE PIPELINE\Capstone-\trainer_outputs\orchestrator_server.log"

"" | & ".venv\Scripts\python.exe" "run_client.py" *>&1 | Tee-Object -FilePath "trainer_outputs\client_run_1.log"
```

### TERMINAL 3
```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
$env:PATH = "C:\Program Files\Git\mingw64\bin;" + $env:PATH
$env:PYTHONUTF8 = "1"

"" | & ".venv\Scripts\python.exe" "run_client.py" *>&1 | Tee-Object -FilePath "trainer_outputs\client_run_2.log"
"" | & ".venv\Scripts\python.exe" "run_client.py" *>&1 | Tee-Object -FilePath "trainer_outputs\client_run_3.log"

# Verify:
& ".venv\Scripts\python.exe" -c "
import pymongo
c = pymongo.MongoClient('mongodb://localhost:27017')
db = c['federated']
print('model_updates:', db.model_updates.count_documents({}))
print('receipts:', db.receipts.count_documents({}))
print('global_models:', db.global_models.count_documents({}))
for doc in db.global_models.find(): print(doc)
"
```

**Reminder:** all three client runs share one device identity on one machine, and the input
is the repo's test fixture, not DAIC-WOZ. See §21.
