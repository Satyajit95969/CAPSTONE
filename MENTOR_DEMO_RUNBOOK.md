# Mentor Demo — 5-Terminal Manual Run (Real DAIC-WOZ Multimodal Data)

Validated end-to-end on this machine on 2026-08-21. Uses `PIPELINE_MODE=multimodal`,
which reads real DAIC-WOZ text + wav2vec2 audio + DenseNet video features from the
repo's frozen `dataset_build\daic_records_multimodal.parquet` (188 rows) — **not** the
toy `videos\sample.mp4` fixture. This is real research data, not a placeholder.

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
identity.

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
& ".venv\Scripts\python.exe" "scripts\reset_all_federated_dbs.py"
```
Expected: `{'ok': 1.0}`, `dbPath: D:\MongoDB\data`, and the reset script prints
`[OK] All collections in all project databases are empty.`

Leave this terminal open for the whole session.

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
The OTP is valid 10 minutes — move to Terminal 3 promptly.

---

## TERMINAL 3 — Connection check + device enrollment

```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"

# Port + cert sanity checks
Get-NetTCPConnection -LocalPort 50051 -ErrorAction SilentlyContinue   # expect State=Listen
Test-Path "server\orchestration_agent\certs\ca.pem"
Test-Path "server\orchestration_agent\certs\server.pem"

# Device enrollment (required every time Terminal 1's reset wipes the 'devices' collection)
& ".venv\Scripts\python.exe" enroll_step5.py "D:\Download D\BE PIPELINE\Capstone-\trainer_outputs\demo_orchestrator.log"
```
Expected tail: `ENROLLMENT COMPLETE`, with a `client.pem` fingerprint printed. This
single exchange (RequestEnrollment → OTP → EnrollDevice → signed cert) **is** your live
connection check: it proves mTLS, gRPC, and MongoDB are all wired together correctly.

Verify the cert if you want a second, independent check:
```powershell
& "C:\Program Files\Git\mingw64\bin\openssl.exe" verify -CAfile "server\orchestration_agent\certs\ca.pem" "$env:USERPROFILE\.federated\keys\client.pem"
```

---

## TERMINAL 4 — client 1 run, client 2 run, client 3 run

Aggregation requires **3 accepted updates** (`round.updates.len() >= 3`). Run all three
sequentially, in this same terminal, waiting for each to print
`Run-once pipeline complete` before starting the next:

```powershell
Set-Location "D:\Download D\BE PIPELINE\Capstone-"
$env:PATH = "C:\Program Files\Git\mingw64\bin;" + $env:PATH
$env:PYTHONUTF8 = "1"

# Client 1
"" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run1.log"

# Client 2
"" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run2.log"

# Client 3 — this one triggers aggregation automatically on the server ~30-45s after it finishes
"" | & ".venv\Scripts\python.exe" "run_client_multimodal.py" *>&1 | Tee-Object -FilePath "trainer_outputs\demo_client_run3.log"
```
Each run prints a full report: real DAIC-WOZ participant text sample, audio dim=154
(wav2vec2), video dim=84 (DenseNet), 109.7M-parameter fusion model, DP noise (Gaussian,
ε=5.302585), AES-GCM encryption, TPM ECDSA signing, and `✅ Round 1 update submitted`.
Each run takes roughly 40-90 seconds.

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
aggregated global model (real multimodal fusion weights, ~439 MB) is durably persisted.

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

## What to say if your mentor asks "is this real data?"

Yes for the content (real DAIC-WOZ interview transcripts + pre-extracted wav2vec2 audio
embeddings + DenseNet video embeddings, frozen in `dataset_build\daic_records_multimodal.parquet`),
with the one honest caveat above: all 3 aggregated updates come from one enrolled device
on one machine, so this demonstrates the pipeline's full security/privacy/aggregation
mechanics correctly, not independent multi-institution federation. DP (Gaussian,
clip=1.0, noise multiplier=1.0, ε=5.302585, δ=1e-5), AES-GCM encryption, TPM-backed
ECDSA device signing, mTLS transport, and trimmed-mean aggregation (trim_ratio=0.1) are
all live and unmodified.
