# PROJECT KNOWLEDGE DOCUMENT
## MediProof / BE-Major-Project — Privacy-Preserving Federated Mental-Health Detection

**Generated:** 2026-08-01 · **Basis:** full repository read (source, config, proto, phase reports, git history, artifacts)
**Convention:** **[E]** = verified in this repo · **[I]** = inference · **[?]** = uncertain / not verifiable from the repo

---

# 1. PROJECT SUMMARY

## 1.1 What it actually is

This is **not a web application**. It is a **research systems project**: a privacy-preserving **federated learning pipeline** for detecting depression severity (PHQ-8) from clinical interview data (DAIC-WOZ), where the raw data never leaves the client device.

It is composed of:
- a **Python client-side agent chain** (preprocess → train → differentially-privatise → encrypt → upload),
- a **Rust gRPC orchestration server** that coordinates federated rounds over mTLS,
- a **Python aggregator** invoked as a server-side subprocess,
- **MongoDB + GridFS** as the persistence and blob layer,
- **TPM-backed hardware identity** and a **tamper-evident HMAC receipt chain** for auditability.

## 1.2 The problem it solves

Mental-health data (interview transcripts, audio, video) is among the most sensitive data that exists. Centralised ML on it is legally and ethically fraught. This project's thesis is:

> Train a depression-detection model across many devices **without any device's raw data ever leaving it**, while producing a **cryptographically verifiable audit trail** proving that (a) each update came from a specific attested device, (b) the exact bytes uploaded are the ones the receipt covers, and (c) a bounded, accounted amount of privacy budget (ε) was spent.

Only **model deltas** — clipped, noised, encrypted — cross the network.

## 1.3 The project's actual current state (critical to understand)

The project has passed through two eras, and **conflating them is the single most common mistake**:

| Era | What it was | Status |
|---|---|---|
| **Version 1 (Phases 1–9.5)** | Reconstruct the senior's implementation and run it end-to-end unmodified | ✅ **COMPLETE and FROZEN** |
| **Version 2 (Phases 10–12)** | Controlled, single-factor experiments to make the science actually work | 🔄 **IN PROGRESS — currently at Phase 11.6, blocked on a Colab GPU run** |

[E] The infrastructure works. The **science does not yet**. Phase 10/11 established with evidence that the model has **never detected a single depressed participant** (P/R/F1 = 0.0 in every run), that both output heads collapsed to the label prior, and that the "federated" object contains no DAIC data at all. See §14.

---

# 2. TECH STACK

| Layer | Technology | Where |
|---|---|---|
| **Orchestration server** | Rust 2021, tokio 1.37, tonic 0.11 (gRPC + TLS), prost 0.12 | `server/orchestration_agent/` |
| **Transport** | gRPC over mutual TLS, with client- and server-streaming RPCs | `proto/orchestrator.proto` |
| **Persistence** | MongoDB 8.0, database `federated`, GridFS (`fs` bucket) | driver `mongodb` 2.8 (Rust), `pymongo` (Python) |
| **Client runtime** | Python 3.11, grpcio | `runtime/` |
| **ML** | PyTorch 2.8.0+cpu, HuggingFace `transformers` **pinned 4.44.0**, scikit-learn, pandas, pyarrow | trainers |
| **Model backbone** | `mental/mental-bert-base-uncased` (**gated** on HF — requires accepted licence + `huggingface-cli login`) | Track B/C |
| **Crypto (Python)** | `cryptography` — AES-GCM, HKDF-SHA256, HMAC-SHA256 | `centralized_secure_store.py`, `centralised_receipts.py` |
| **Crypto (Rust)** | `p256`/`ecdsa` (P-256 verify), `sha2`, `hmac` | `receipts.rs`, `crypto.rs` |
| **Hardware identity** | Windows Platform Crypto Provider via `windows_signer.exe` (Rust); Linux `tpm2-tools` | `installer/windows_signer/`, `runtime/tpm_guard.py` |
| **Concurrency (server)** | `dashmap` 5.5 lock-free maps; `tokio::task::spawn_blocking` for the aggregator subprocess | `state.rs`, `server.rs` |
| **Media tooling** | OpenFace (video AUs), openSMILE eGeMAPS + wav2vec2 (audio), spaCy NER, Whisper ASR | `LDA/app/pipelines/` |
| **Dataset** | DAIC-WOZ (`D:\datasets\DAIC-WOZ`, 189 archives, 188 usable) | staged into `data/`, `data_norm/`, `labels/` |
| **Training compute** | Google Colab GPU is the **official** environment; local machine is CPU-only | `colab_package/`, `colab_baseline_cv/` |

**There is no React/Next.js, no REST API, no JWT/session auth, no SQL database, no Redis.** [E] Verified — no such files exist.

---

# 3. FOLDER STRUCTURE & RESPONSIBILITIES

```
BE-Major-Project/
│
├── LDA/                          Local Data Agent — client-side multimodal preprocessing
│   └── app/
│       ├── main.py               ★ preprocess(PreprocessRequest) — the LDA entry function
│       └── pipelines/
│           ├── text.py           spaCy PII scrub (PERSON/GPE/ORG), embeddings, sentiment
│           ├── audio.py          wav2vec2, openSMILE eGeMAPS, prosody, WebRTC VAD
│           ├── video.py          OpenFace AUs/gaze/pose, face blurring
│           └── session_processor.py   A/V timeline sync, diarization
│
├── trainer_agent/                TRACK B — the federated-runtime trainer
│   ├── trainer_mentalbert_privacy.py  ★ orchestrate(); reads LDA session rows, emits encrypted delta
│   ├── model.py, trainer.py, utils.py, app.py, dummy_session.py
│
├── trainer_mentalbert_daic.py    ★ TRACK C — the frozen DAIC/Colab trainer (SHA 65b1902e…a230b)
├── create_dp_comparison.py       ★ TRACK A — DP-sweep orchestrator; produces local_probe_*.pt
│
├── dp_agent/dp_agent.py          ★ L2-clip + 6 noise mechanisms + real RDP ε accountant
├── enc_agent/enc_agent.py        AES-GCM finalisation; Fernet/KMS/CKKS/SMPC modes (stubs)
├── centralized_secure_store.py   ★ AES-GCM at-rest store, HKDF per-agent keys
├── centralised_receipts.py       ★ HMAC-SHA256 signed audit receipts
│
├── runtime/                      PRODUCTION CLIENT (the real federated client)
│   ├── federated_client.py       ★ ENTRY POINT — daemon | run-once
│   ├── pipeline.py               ★ run_pipeline() — the 8-step federated cycle
│   ├── grpc_client.py            mTLS / server-TLS dual-channel stub factory
│   ├── tpm_guard.py              TPM sign / unseal / pubkey
│   ├── daemon.py, capture.py, idle.py, offline_queue.py, logging_config.py
│   ├── runtime_guard.py, self_destruct.py, validate_deps.py, config_validator.py
│   └── grpc/orchestrator_pb2*.py ✅ CURRENT generated proto stubs
│
├── server/
│   ├── orchestration_agent/      RUST gRPC SERVER
│   │   ├── src/main.rs           ★ entry: Mongo connect, indexes, OTP, serve
│   │   ├── src/grpc/server.rs    ★ all 7 RPC handlers + run_aggregation (1203 lines)
│   │   ├── src/state.rs          OrchestratorState (DashMap); seeds Round 1
│   │   ├── src/round.rs          Round, RoundState, UpdateMeta, AggregationReceipt
│   │   ├── src/otp.rs            6-digit OTP, 600s expiry, 5-fail lockout
│   │   ├── src/receipts.rs       PEM→SPKI→EC-point→ECDSA P-256 verify
│   │   ├── src/{config,crypto,identity,errors,ledger,pubsub}.rs
│   │   ├── proto/orchestrator.proto   ★ THE CONTRACT
│   │   ├── config/orchestrator.toml   addr + TLS paths
│   │   └── certs/                 CA + server cert/key + gen_certs.sh
│   └── aggregator_agent/
│       ├── aggregator.py         ★ decrypt → key-by-key FedAvg → GridFS upload
│       └── core/                 server-side copies of SecureStore + receipts
│
├── installer/                    Install + hardening
│   ├── installer_core.py, installer_gui.py
│   ├── fs/                       component installers (OpenFace, ffmpeg, openSMILE, spaCy…)
│   ├── security/                 integrity.py, anti_debug.py, tpm_seal.py, self_destruct.py…
│   ├── windows_signer/           Rust ECDSA P-256 signer (TPM-backed)
│   └── runtime/                  MIRROR of runtime/ copied on install
│
├── client_agent/                 ⚠ LEGACY — DO NOT USE (old proto, Ed25519, enc_uri)
│
├── colab_package/                Phase 9.5 Colab bundle (notebook + trainer copy)
├── colab_baseline_cv/            Phase 11.6 CV harness — run_baseline_cv.py, aggregate_baseline_cv.py
│
├── Dataset prep helpers (all written post-handover):
│   ├── extract_daic_woz.py       unzip 189 *_P.zip archives → ./data
│   ├── stage_labels.py           AVEC CSVs → ./labels/Detailed_PHQ8_Labels.csv (fixes label-col hazard)
│   ├── normalize_transcripts.py  tab-separated → clean comma CSVs → ./data_norm
│   ├── build_daic_records.py     → PLAINTEXT ./daic_records.parquet (Track C input)
│   └── verify_daic_records.py    GATE: must print [PASS] before Colab
│
├── Validation harnesses (post-handover):
│   ├── enroll_step5.py           drives enrollment, reads OTP from server log
│   ├── round0_step6_validate.py  drives one full upload+receipt (LOCAL_UPDATE_SRC hardcoded)
│   ├── step7_download_validate.py verifies DownloadGlobalModel + strict load
│   └── gap2_test.py              8 consistency tests for global-model loading
│
├── Reports (the project's scientific record):
│   ├── ARCHITECTURE.md           Phase-3-era reference (⚠ its "blockers" are now FIXED)
│   ├── PHASE_8B_EXECUTION_REPORT.md      ┐
│   ├── PHASE_9_BASELINE_EVALUATION_REPORT.md   │ FROZEN baseline — never modify
│   ├── PHASE_9.5_COMPLETE_DATASET_REPORT.md    ┘
│   ├── PHASE_10_ROADMAP.md       ★ the strategic authority for all remaining work
│   └── PHASE_11_*.md             evaluation protocol, diagnostics, Baseline-CV
│
└── Outputs: trainer_outputs/, secure_store/, receipts/, explain_logs/, plots/, sess-*/
```

---

# 4. COMPLETE ARCHITECTURE

## 4.1 The two halves

```
┌──────────────────────── CLIENT DEVICE ────────────────────────┐
│  runtime/federated_client.py  (entry)                          │
│    ├─ IntegrityWatcher (5-min file-hash monitor)               │
│    ├─ runtime_guard()  (TPM unseal + anti-debug)               │
│    ├─ device_id = SHA-256(TPM ECDSA P-256 pubkey)              │
│    └─ run_pipeline()                                           │
│         LDA ─► Trainer ─► DP Agent ─► Enc Agent ─► upload      │
│         (all intermediate artifacts AES-GCM encrypted at rest) │
└───────────────────────────┬────────────────────────────────────┘
                            │  gRPC / mutual TLS  :50051
┌───────────────────────────▼────────────────────────────────────┐
│  Rust Orchestrator (server/orchestration_agent)                │
│    RPCs: RequestEnrollment · EnrollDevice · GetRound           │
│          UploadUpdate(stream) · SubmitReceipt                  │
│          DownloadGlobalModel(stream) · RegisterDevice(dep.)    │
│    ├─ mTLS enforced on every op RPC                            │
│    ├─ per-chunk SHA-256 verification                           │
│    ├─ ECDSA P-256 receipt signature verification               │
│    ├─ HMAC receipt chain                                       │
│    ├─ ε budget ceiling per round                               │
│    └─ at ≥3 updates ─► spawn_blocking ─► aggregator.py         │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│  MongoDB `federated` + GridFS                                  │
│    devices · model_updates · receipts · global_models · fs.*   │
└────────────────────────────────────────────────────────────────┘
```

## 4.2 The three trainer tracks — **the most important architectural fact**

[E] There are **three independent trainers** with **incompatible input contracts**. They were never designed to share files. Confusing them is the #1 source of wasted effort.

| | **Track A** | **Track B** | **Track C** |
|---|---|---|---|
| **File** | `create_dp_comparison.py` | `trainer_agent/trainer_mentalbert_privacy.py` | `trainer_mentalbert_daic.py` |
| **Role** | DP-mechanism sweep + linear probe | The **federated runtime** trainer | The **DAIC/Colab** research trainer |
| **Input** | embedding parquet from `format_daic_to_lda.py` (`.parquet.enc`) | LDA session rows (encrypted jsonl manifest) | **PLAINTEXT** `daic_records.parquet` (`text` + `phq_score`) |
| **Model** | sklearn linear probe (~296k params) | `MultiModalModel` (MentalBERT) | `MultiModalModel` (MentalBERT, ~109M params) |
| **Encryption** | SecureStore | SecureStore + `master.key` | `SecureStoreFallback` = **plaintext**, no real key |
| **Output** | `trainer_outputs/local_probe_{base,rag,vector_rag}.pt` | encrypted delta → DP → upload | `mentalbert_privacy_subset.pt` (438 MB) + delta + reports |
| **Called by** | manual CLI | `runtime/pipeline.py` | manual CLI / Colab / `run_baseline_cv.py` |

[E] **The federated round actually runs on Track A's `local_probe_base.pt`**, not on the DAIC model. Three independent blockers prevent Track C from being federated:
1. `DPAgent.process_local_update` flattens the whole delta into one vector and L2-clips to 1.0 — at ~109M params the Gaussian noise has norm ≈ √d, giving ~10⁴:1 noise-to-signal.
2. `round0_step6_validate.py` **hardcodes** `LOCAL_UPDATE_SRC = trainer_outputs/local_probe_base.pt`.
3. 438 MB × 3 uploads exceeds `SubmitReceipt`'s 15 s client timeout.

[E] And `local_probe_base.pt` was fitted on `sample_texts/sample1.txt` — a **public demo file with no DAIC data in it**. Its metrics are accuracy 1.0 / P/R/F1 0.0 — the signature of a single-class corpus.

---

# 5. DATA FLOW (end to end)

## 5.1 The federated cycle — `runtime/pipeline.py::run_pipeline()`

```
1. GetRound(device_id)                       → abort unless state == "Collecting"
2. if global_model_available:
     DownloadGlobalModel(round_id)           → stream chunks
        · verify SHA-256 per chunk
        · verify SHA-256 of full model on last chunk
        · save → ~/.federated/data/global_models/global_roundN.pt
3. LDA preprocess(mode="session")            → encrypted parquet + manifest URI
        · _validate_lda_output(): requires session_id/artifact_manifest/receipts/count, count>0
4. Trainer orchestrate(mode="supervised", epochs=1, [global_model_path])
        · warm-start via load_state_dict(strict=True) if global model present
        · delta = after − before, then apply_safety_to_delta(clamp 1e-3, global L2 ≤ 1.0)
        · _validate_trainer_output(): local_update_uri must be file:// and exist
5. DPAgent(clip_norm=1.0, noise_multiplier=1.0, mechanism="gaussian")
        · decrypt → flatten → L2-clip → add noise → unflatten → re-encrypt
        · epsilon_spent = real RDP bound (single composition, T=1)
        · guard: eps must be finite and ≤ MAX_EPS_VALUE (10.0)
6. EncryptionAgent(mode="aes").process_dp_update(receipt_uri)   → final URI
7. UploadUpdate  (client-streaming, 1 MB chunks, per-chunk SHA-256)
        → server stores in GridFS, returns server_handle = ObjectId
8. SubmitReceipt
        msg       = device_id ‖ round_id(BE8) ‖ payload_hash
        signature = TPM ECDSA P-256 DER over msg
        Receipt(payload_hash, epsilon_spent, signature,
                enc_handle=server_handle, scheme="AES-GCM-DP-ECDSA")
```

## 5.2 Server-side round lifecycle

```
Startup            → state.rs seeds Round 1 {Collecting, epsilon_max = 100.0}
UploadUpdate       → mTLS ✓, enrolled ✓, Collecting ✓, chunk hashes ✓,
                     sequential order ✓, ≤500 MB ✓ → GridFS
                     → model_updates{verified:false}
SubmitReceipt      → mTLS ✓, eps>0 ✓, enc_handle is 24-hex ObjectId ✓
                     → look up devices.pubkey_pem → ECDSA verify
                     → cross-check payload_hash vs stored upload hash
                     → model_updates{verified:true}
                     → epsilon_spent + submitted ≤ epsilon_max
                     → HMAC chain link → receipts collection
                     → round.updates.push(UpdateMeta)
                     → if round.updates.len() >= 3 → run_aggregation()
run_aggregation    → build job JSON (gridfs_ids), release DashMap lock
                     → spawn_blocking(../../.venv/Scripts/python ../aggregator_agent/aggregator.py)
                     → parse stdout {aggregated_uri, gridfs_file_id, model_hash}
                     → round N → Complete
                     → INSERT global_models{round_id: N+1, file_id, model_hash}
                     → CREATE round N+1 {Collecting}
```

## 5.3 Aggregation (`server/aggregator_agent/aggregator.py`)

```
for each update:
  GridFS.get(ObjectId) → raw bytes
  → _decrypt_secure_store_envelope()   JSON {nonce, ct} → HKDF(master.key) → AES-GCM
  → torch.load() → Dict[str, Tensor]   (raw tensors are REJECTED with TypeError)
_aggregate_state_dicts():
  · identical sorted key sets across clients   (else ValueError)
  · identical per-key shapes                    (else ValueError)
  · float params → stack → _aggregate_tensor() → cast back to original dtype
  · non-float buffers → must be byte-identical; client 0's value used
_aggregate_tensor(): mean | trimmed_mean (default, trim 0.1) | coordinate_median
→ torch.save(state_dict) → ./aggregated_round_N.pt + GridFS.put()
→ stdout JSON
```

**Note on trimmed_mean at N=3:** `lower = max(1, int(0.1·3)) = 1`, `upper = 2` → keeps exactly one middle value ⇒ **it is the coordinate-wise median**, not a mean. [E] Confirmed empirically in Phase 4 Step 7.

## 5.4 Dataset prep (Track C) — the proven sequence

```
python extract_daic_woz.py                                  # 189 zips → ./data
python stage_labels.py                                      # → ./labels/Detailed_PHQ8_Labels.csv
python normalize_transcripts.py                             # tab → comma → ./data_norm
python build_daic_records.py --data-dir ./data_norm --out ./daic_records.parquet
python verify_daic_records.py ./daic_records.parquet        # GATE: must be [PASS]
→ upload ONLY trainer_mentalbert_daic.py + daic_records.parquet to Colab
→ pip install transformers==4.44.0
→ python trainer_mentalbert_daic.py --mode supervised --input ./daic_records.parquet \
      --device cuda --epochs 3 --batch-size 8 --binarize
```

---

# 6. API FLOW (gRPC)

Contract: `server/orchestration_agent/proto/orchestrator.proto`. **This file is the source of truth** — the Rust server and the Python stubs are both generated from it.

| RPC | Kind | mTLS | Purpose |
|---|---|---|---|
| `RequestEnrollment` | unary | ❌ (only exception) | Get an OTP; server prints it to the operator console |
| `EnrollDevice` | unary | ❌ | Present OTP → server signs CSR → returns client cert |
| `GetRound` | unary | ✅ | Current round metadata + `global_model_available` |
| `UploadUpdate` | **client-stream** | ✅ | Stream encrypted bytes → GridFS → `server_handle` |
| `SubmitReceipt` | unary | ✅ | ECDSA-signed proof of what was uploaded + ε spent |
| `DownloadGlobalModel` | **server-stream** | ✅ | Stream aggregated model with hash verification |
| `RegisterDevice` | unary | ✅ | ⚠ **DEPRECATED** — no DB write, issues no real cert |

**Two fields carry the entire security design:**
- `UpdateChunk.chunk_hash` — SHA-256 of that chunk. The server rejects any mismatch. Detects transit corruption/tampering *per chunk*.
- `Receipt.enc_handle` — a **server-side GridFS ObjectId**, never a client file path. The old `enc_uri` (a client path the server could never open) was the original security vulnerability the streaming redesign fixed.

**Round selection is dynamic** [E]: `get_round` picks `max(id)` among `Collecting` rounds, falling back to `max(id)` overall. So clients migrate to round N+1 with no server restart.

---

# 7. AUTHENTICATION FLOW

There is **no user login**. Authentication is **device identity**, in four layers:

### Layer 1 — Hardware identity (TPM)
`windows_signer.exe` holds an ECDSA P-256 key named `FederatedDeviceKey` in the Microsoft Platform Crypto Provider (Linux: `tpm2_createprimary`/`tpm2_create`). The private key **cannot be extracted**.
`device_id = SHA-256(device_pubkey)`.
[E] Any failure in `sign_message()` or `unseal_master_secret()` triggers `self_destruct`.

### Layer 2 — Enrollment ceremony (two-phase OTP)
```
B1  RequestEnrollment(pubkey, csr, info)
      → fingerprint = SHA-256(pubkey)[:8] hex
      → 6-digit OTP, 600 s TTL, single-use, printed to operator console
      → pending_enrollments[fingerprint] = (pubkey, csr)
      ← EnrollmentRequestAck{accepted, device_fingerprint}
    (operator conveys the OTP out-of-band)
B2  EnrollDevice(token, pubkey, csr)
      → consume_otp_from(token, peer_addr)   -- 5 failures ⇒ 5-min lockout per peer
      → openssl x509 -req  (CA-signs the CSR, 365 days)
      → devices upsert {device_id, pubkey_pem, enrolled_at, last_seen, peer_addr}
      ← EnrollResponse{ok, client_cert}
    → client saves ~/.federated/keys/client.pem
```

### Layer 3 — Mutual TLS (transport)
[E] `require_client_cert()` rejects any request without a peer cert. TLS is configured with `client_auth_optional(true)` so *enrollment* clients (no cert yet) can connect at the TLS layer, but **every operational RPC re-checks at the application layer**.
[E] `grpc_client.py` deliberately **omits** `ssl_target_name_override` and `default_authority` — hostname verification is enforced. If you get `SSL_ERROR_SSL`, the server cert's SAN doesn't match your connect address; regenerate with `gen_certs.sh <SERVER_IP>`.

### Layer 4 — Per-message signature (application)
```
msg = device_id(32B) ‖ round_id(8B BE) ‖ payload_hash(32B)
sig = TPM ECDSA-P256 DER over msg
```
Server (`receipts.rs`): PEM → strip headers → base64 → DER (91 B) → skip 26-B SPKI header → 65-B EC point (must start `0x04`) → `VerifyingKey::from_sec1_bytes` → `Signature::from_der` → verify.

This binds each receipt to **that device**, **that round**, and **exactly those bytes**.

---

# 8. DATABASE SCHEMA — MongoDB `federated`

### `devices`
| Field | Type | Notes |
|---|---|---|
| `device_id` | string | hex SHA-256 of pubkey · **unique index** |
| `pubkey_pem` | string | used for every ECDSA verification |
| `enrolled_at` / `last_seen` | BsonDateTime | `last_seen` updated on every `GetRound` |
| `peer_addr` | string | |

### `model_updates`
| Field | Type | Notes |
|---|---|---|
| `device_id`, `round_id`, `session_id` | | |
| `payload_hash` | string | hex SHA-256 of all uploaded bytes |
| `file_id` | ObjectId | → GridFS |
| `verified` | bool | `false` until `SubmitReceipt` passes every check |
| `verified_at`, `upload_time`, `size_bytes` | | |

Index: `(file_id, device_id, round_id)`

### `receipts`
| Field | Type | Notes |
|---|---|---|
| `device_id`, `round_id`, `payload_hash`, `epsilon_spent`, `signature`, `enc_handle`, `scheme`, `timestamp` | | |
| `verified` | bool | always `true` (only stored if all checks passed) |
| `hmac_chain` | string | `HMAC(key, prev ‖ "\|" ‖ payload_hash_hex)`, genesis = `"genesis"` |

Index: `(round_id, _id: -1)` — for chain-tail lookup.

### `global_models`
| Field | Type | Notes |
|---|---|---|
| `round_id` | int64 | **unique index** — ⚠ source of the collision bug (see §14) |
| `file_id` | ObjectId | → GridFS |
| `model_hash` | string | hex SHA-256 |

[E] Written for **round N+1** after round N aggregates.

### GridFS `fs.files` / `fs.chunks`
Encrypted model updates and aggregated global models. Always referenced by ObjectId; **client-supplied paths are never used**.

### Relationships
```
devices ──1:N──► model_updates ──file_id──► GridFS
   └────1:N──► receipts ──hmac_chain──► (self-referential chain)
global_models ──file_id──► GridFS
```

---

# 9. DEPENDENCY MAP

## 9.1 Client import graph
```
runtime/federated_client.py   [ENTRY]
  ├─ runtime/logging_config.py            (MetricsCollector, HealthReporter)
  ├─ runtime/runtime_guard.py ─► installer/security/{anti_debug,tpm_seal,integrity}.py
  ├─ runtime/tpm_guard.py     ─► installer/windows_signer.exe · runtime/self_destruct.py
  ├─ runtime/grpc_client.py   ─► runtime/grpc/orchestrator_pb2_grpc.py · tpm_guard · self_destruct
  ├─ runtime/daemon.py        ─► runtime/idle.py · runtime/capture.py · runtime/pipeline.py
  └─ runtime/pipeline.py      [HUB]
        ├─ agents/lda/main.py            (= LDA/app/main.py)
        │     └─ pipelines/{text,audio,video,session_processor}.py
        │     └─ centralized_secure_store.py · centralised_receipts.py
        ├─ agents/trainer/trainer_mentalbert_privacy.py   (Track B)
        ├─ agents/dp/dp_agent.py         ─► SecureStore · CentralReceiptManager
        ├─ agents/enc/enc_agent.py       ─► SecureStore · CentralReceiptManager
        ├─ core/centralized_secure_store.py
        └─ runtime/grpc/orchestrator_pb2.py
```
⚠ **Path duality:** `pipeline.py` imports `agents.lda.main` / `core.centralized_secure_store` — the **installed** layout (`~/.federated/`). In the repo the same files live at `LDA/app/main.py` and `centralized_secure_store.py`. `installer/runtime/` is a **mirror** of `runtime/`. Editing one without the other is a classic bug source.

## 9.2 Server call graph
```
main.rs → config.rs → mongodb → _ensure_indexes() → otp::generate_otp()
        → pubsub::start() → grpc::server::serve()
              └─ Service{state, cfg, mongo, receipt_chain_key}
                    ├─ crypto::hash_bytes · identity::derive_device_id
                    ├─ otp::{generate_otp_for, consume_otp_from}
                    ├─ receipts::verify (p256)
                    ├─ round::{Round, RoundState, UpdateMeta, AggregationReceipt}
                    └─ run_aggregation() ─subprocess─► aggregator.py
                                                          ├─ pymongo/gridfs
                                                          ├─ cryptography (HKDF+AESGCM)
                                                          └─ torch
```

## 9.3 Reusable components
| Component | Used by |
|---|---|
| `SecureStore` | LDA, trainer(B), DP, Enc, aggregator — **4 copies exist** |
| `CentralReceiptManager` | every agent — **3 copies exist** |
| `MultiModalModel` / `MultiModalDataset` / `collate_batch` | Track B and Track C (separate definitions) |
| `runtime/grpc/orchestrator_pb2*.py` | pipeline, grpc_client, all validation harnesses |
| `fold_manifest.json` | every Phase 12 experiment — **must stay frozen** for paired comparisons |

---

# 10. IMPORTANT FILES — deep reference

### `runtime/pipeline.py` — the client hub
- **Purpose:** the 8-step federated cycle.
- **Internals:** validates LDA and trainer outputs by schema; streams 1 MB chunks with per-chunk SHA-256; computes `payload_hash` over the **actual bytes streamed**; reads real ε from the DP agent.
- **Depends on:** LDA, Track-B trainer, DPAgent, EncryptionAgent, SecureStore, gRPC stubs, `tpm_guard.sign_message`.
- **Used by:** `federated_client.py`, `daemon.py`. Its `_stream_update` + receipt logic is **copied verbatim** into `round0_step6_validate.py`.
- **Common mistakes:** changing `CHUNK_SIZE` (must stay ≤ server `CHUNK_SIZE_MAX` = 4 MB); altering the `msg` byte layout (breaks ECDSA verify server-side); making the ε fallback silent again; putting a file path back into `enc_handle`.

### `server/orchestration_agent/src/grpc/server.rs` — the server core
- **Purpose:** all 7 RPCs + `run_aggregation`.
- **Internals:** `require_client_cert` on every op RPC; per-chunk hash + sequential-index checks; 500 MB cap; ECDSA verify; ε ceiling; HMAC chain; `spawn_blocking` for the subprocess; round N+1 creation.
- **Common mistakes:** importing `tokio::io::AsyncWriteExt` alongside the `futures` traits (GridFS streams implement **futures**' traits — ambiguity picks the wrong bound); building in `--release` (unused-import warnings are errors — use `cargo build` debug); changing the aggregator subprocess path (it is **relative to `server/orchestration_agent/`**, the CWD the binary runs from); holding a DashMap lock across an `.await`.

### `server/aggregator_agent/aggregator.py`
- **Purpose:** decrypt + structure-preserving FedAvg + publish to GridFS.
- **Internals:** in-memory AES-GCM envelope decryption with HKDF candidate list; fail-loud key/shape validation; pure-PyTorch aggregation (no numpy float64 promotion).
- **Common mistakes:** reintroducing a `np.frombuffer` fallback (it silently aggregates **ciphertext**); running it with a different `master.key` than the clients; changing `trim_ratio` without realising N=3 makes it a median.

### `dp_agent/dp_agent.py`
- **Purpose:** L2-clip + noise + real RDP ε.
- **Internals:** `_rdp_to_dp` minimises `α/(2σ²) + ln(1/δ)/(α−1)` over α ∈ [2,256]. With nm=1.0, clip=1.0, δ=1e-5 ⇒ **ε = 5.302585092994046**. Non-Gaussian mechanisms and `none` return `math.inf` (a deliberate signal, not a bug).
- **⚠ Critical design limit:** it flattens the **entire** state dict into one vector and clips to `clip_norm`. Noise norm scales as **√d** while the clip is dimension-independent ⇒ **SNR = 1/√d**. At the probe's ~296k params: L2-before 11.34 → clipped 1.0 → L2-after ≈ **543**. That is a **543:1** noise-to-signal ratio.
- **Common mistakes:** assuming the plaintext fallback (`try decrypt_read, except: raw bytes`) is a bug — it is intentional, and it is why a plaintext delta is accepted.

### `centralized_secure_store.py`
- **Purpose:** AES-GCM at rest, HKDF per-agent keys.
- **Internals:** key = `HKDF-SHA256(master_key, info=f"{agent}:{context}")` where `context` = the **parent directory name**, with a special-case rewrite for `local_updates`. Envelope = `{"nonce": b64, "ct": b64}`.
- **⚠ Common mistakes:** changing `agent` or the directory layout **silently changes the derived key** and makes existing files undecryptable. The root-level version writes the *legacy* envelope (no `agent`/`context` fields), which is why the aggregator has a candidate-list fallback.

### `trainer_mentalbert_daic.py` (Track C) — **FROZEN**
- SHA-256 `65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b`. `run_baseline_cv.py` **aborts** if it changes.
- Self-contained: `MultiModalModel`, `SmallMLP`, `FusionHead`, `MultiModalDataset` all inside. Uses `SecureStoreFallback` (plaintext) — **not** the real SecureStore.
- Label rule: `label = 1 if phq_val > 10.0` — recomputed inside `MultiModalDataset`, so a `label` column in the input file is **ignored**.
- Loss: `CrossEntropy(label) + 0.5 · MSE(phq)`, grad-clip 1.0, AdamW, `max_length=128` hardcoded.
- Encoders are **conditionally constructed**: `audio_encoder = SmallMLP(...) if audio_dim > 0 else None`. On text-only data the condition never fires ⇒ the multimodal path is dead code.
- ⚠ `from transformers import AdamW` — **removed in transformers ≥ 4.46**. This is the sole reason for the 4.44.0 pin.

### `colab_baseline_cv/run_baseline_cv.py` (newest code)
- Imports the frozen trainer, **never edits it**; verifies its SHA at startup; **refuses to run on CPU** unless `--allow-cpu` (to keep Baseline-CV a one-factor change vs the GPU-trained Phase 9.5 baseline).
- Only change vs 9.5: the fold partition from `fold_manifest.json`, and `val_dataset=None` (train on the full 4/5 partition, since CV replaces the internal 10 % holdout).

### `format_daic_to_lda.py` (original senior code)
- ⚠ `MAX_PARTICIPANTS = 20` hardcoded; `MODEL_NAME = "mental/mental-bert-base-uncased"` hardcoded.
- [E] Its outputs were **byte-identical** across the 113 and 188 phases — the LDA stage is conclusively inert with respect to any reported metric.

---

# 11. BUSINESS-LOGIC SEPARATION

| Concern | Where it lives |
|---|---|
| **UI logic** | Only `installer/installer_gui.py` (a Tkinter install wizard). **There is no application UI.** Operator interaction = the OTP printed to the server console. |
| **"Business" / domain logic** | Depression detection: `MultiModalModel` + `FusionHead` (2-class + PHQ regression); the `phq > 10.0` binarisation; safety policy `apply_safety_to_delta` (clamp 1e-3, global L2 ≤ 1.0). |
| **Privacy logic** | `dp_agent/dp_agent.py` — clipping, 6 mechanisms, `_rdp_to_dp` ε accountant. |
| **API logic** | `proto/orchestrator.proto` (contract) · `src/grpc/server.rs` (handlers) · `runtime/grpc_client.py` (channels) · `runtime/pipeline.py` (call sequencing). |
| **Database logic** | `server.rs` (Mongo CRUD, GridFS write, index use) · `main.rs::_ensure_indexes` · `aggregator.py` (GridFS read/write). |
| **Authentication logic** | `otp.rs` · `receipts.rs` · `identity.rs` · `require_client_cert` · `runtime/tpm_guard.py` · `enroll_step5.py`. |
| **Validation logic** | `_validate_lda_output` / `_validate_trainer_output` (pipeline) · chunk-hash/order/size checks (server) · `_aggregate_state_dicts` key+shape checks · `verify_daic_records.py` · `runtime/config_validator.py`. |
| **Integrity / anti-tamper** | `installer/security/integrity.py` (static guard + 5-min watcher) · `anti_debug.py` · `self_destruct.py`. |

---

# 12. FEATURES THAT CURRENTLY EXIST

**Working and verified [E]:**
- Full end-to-end federated round on real DAIC transcripts (executed 26 Jun and 28 Jun).
- Two-phase OTP enrollment with rate limiting; CA-signed client certs.
- mTLS on every operational RPC, with hostname verification enforced.
- Chunked streaming upload/download with per-chunk **and** whole-object SHA-256.
- TPM-backed ECDSA P-256 receipt signatures, verified server-side.
- HMAC receipt chain in MongoDB (independently re-verified from genesis).
- AES-GCM at-rest encryption for every intermediate artifact; 492 receipt JSONs as a historical trail.
- Genuine RDP ε accounting (ε = 5.302585/update), tracked against a per-round ceiling.
- Structure-preserving FedAvg (trimmed-mean / mean / coordinate-median) with fail-loud validation.
- Round progression: round N completes → global model published for N+1 → round N+1 created.
- Global-model download + `load_state_dict(strict=True)` round-trip with per-tensor equality verified.
- Complete DAIC prep chain producing a verified 188-record parquet.
- Frozen participant-level stratified 5-fold manifest + trivial-baseline control arm.

**Present but inert / stubbed:**
- `enc_agent` Fernet / KMS-envelope / HE-CKKS / SMPC modes — only `aes` is wired.
- `ledger.rs` — appends to a flat file; not integrated.
- `pubsub.rs` — logs round info every 10 s; no real notification.
- `runtime/offline_queue.py` — exists, not integrated into the pipeline.
- Audio and video encoders — constructed only if dims > 0; **the condition has never fired**.
- `client_agent/` — legacy, incompatible.

---

# 13. WHAT IS ORIGINAL vs WHAT WAS ADDED

Determined from file mtimes (handover date **2026-04-29**) cross-checked against git history and the phase reports.

## 13.1 Original senior code (Ritik Shetty) — 2026-04-29
`README.md` · `LDA/app/pipelines/*` · `trainer_agent/*` · `enc_agent/*` · `dp_agent/{__init__,run_demo_single_process}.py` · `centralised_receipts.py` · `client_agent/*` (legacy) · `installer/**` · `runtime/*` (most) · `server/orchestration_agent/src/{main,config,crypto,identity,errors,ledger,pubsub,otp,receipts,round}.rs` · `proto/orchestrator.proto` · `Cargo.toml` · `format_daic_to_lda.py` · `trainer_mentalbert_daic.py` · `standalone_trainer_mentalbert_privacy.py` · `configs/local_config.yaml` · `plot_dp_comparison.py` · `regenerate_proto.py` · `client.py` · `wave2vec.py`

## 13.2 Modified after handover
| File | Date | Change |
|---|---|---|
| `centralized_secure_store.py`, `LDA/app/main.py`, `create_dp_comparison.py` | Jun 16 | Phase 3 reconstruction |
| `runtime/federated_client.py` | Jun 17 | sys.path FIX-A/B/C/D |
| `dp_agent/dp_agent.py` | Jun 18 | **Blocker 1** — added `_rdp_to_dp` + `epsilon_spent` |
| `runtime/pipeline.py`, `runtime/self_destruct.py` | Jun 18 | real-ε handling |
| `server/.../state.rs` | Jun 18 | round seeding |
| `server/.../grpc/server.rs` | Jun 19 | **Blockers 2,3,4** — `global_models` insert, `spawn_blocking`, round N+1, dynamic round selection, venv subprocess path |
| `server/aggregator_agent/aggregator.py` | Jun 19 | **GAP 1 + GAP 3** — real envelope decryption (FIX-AGG-4), structure-preserving FedAvg (FIX-AGG-5) |
| `trainer_agent/trainer_mentalbert_privacy.py` | Jun 19 | **GAP 2** — `_load_global_model(strict=True)` warm start |
| `server/.../config/orchestrator.toml` | Jun 19 | TLS toggling during enrollment |

## 13.3 Files created after handover (all mine)
**Dataset prep:** `extract_daic_woz.py` · `stage_labels.py` · `normalize_transcripts.py` · `build_daic_records.py` · `verify_daic_records.py`
**Validation harnesses:** `enroll_step5.py` · `round0_step6_validate.py` · `step7_download_validate.py` · `gap2_test.py`
**Phase 11.6:** `colab_baseline_cv/run_baseline_cv.py` · `aggregate_baseline_cv.py` + 2 docs
**Colab packaging:** `colab_package/*`
**Documentation:** `ARCHITECTURE.md` · `PHASE_8B` · `PHASE_9` · `PHASE_9.5` · `PHASE_10_ROADMAP` · all `PHASE_11_*`
**Ops:** `clear_fed.js` · `ca_plus_avg.pem` / `avg_root.pem`

## 13.4 [?] Uncertain
- Whether the senior's own repo had a longer git history — this repo's root commit contains only `README.md`, so the 2026-04-29 tree arrived as a bulk import. I cannot see pre-handover history.
- Which of the `FIX-SERVER-*` / `FIX-PIPELINE-*` comment blocks were authored by the senior versus written during reconstruction. They are present in the Apr-29 files, so **[I]** they are the senior's own security-fix annotations from an earlier hardening pass.

---

# 14. TECHNICAL DEBT, BUGS, AND RISKS

## 14.1 The six Critical scientific defects (Phase 10)
These are **silent** — nothing crashes, every gate returns `[PASS]`, the crypto verifies.

| # | Defect | Root cause | Evidence |
|---|---|---|---|
| 1 | **Classifier collapsed to the class prior** | RC-1 | P/R/F1 = 0.0 in every run; P(pos) = 0.365 at a 32.7 % prior, 0.20 at a 23.9 % prior — the output **is** the prior |
| 2 | **Regression head is near-constant** | RC-1 | `pred_phq` spread **0.06 on a 0–23 scale**; worse than the mean predictor (6.6547 vs 6.3882) |
| 3 | **Evaluation on 11 then 18 samples** | RC-2 | one flipped prediction = 5.6 % accuracy; ROC-AUC 95 % CI [0.250, 0.787] |
| 4 | **Audio and video contribute exactly 0.0** | RC-3 | all 189 archives contain audio+video; the pipeline discards both **three stages before the model** |
| 5 | **DP destroys the update (543:1)** | RC-4 | clip is dimension-independent, noise scales √d ⇒ SNR = 1/√d; 543² ≈ 296k params |
| 6 | **The federated object contains no DAIC signal** | RC-5 | global model derives from `sample_texts/sample1.txt` |

> **The decisive asymmetry:** RC-6 and RC-7 (engineering defects) produce **28 % of the weakness count and 0 % of the Critical severity**. A team triaging by visible symptoms would fix exactly the wrong half.

## 14.2 Engineering bugs and hazards
| Issue | Severity | Detail |
|---|---|---|
| **Round state is volatile** | High (operational) | `OrchestratorState::new()` always seeds Round 1; **nothing rehydrates from Mongo**. A mid-round restart orphans uploads; a fresh start collides with `global_models`' unique index on `round_id`. Only documented recovery = destructive wipe of 5 collections. |
| **`round.updates.len() >= 3` is hardcoded** | Medium | `server.rs:675`. Counts *updates*, not *distinct devices* — one device uploading 3× satisfies it. |
| **Only one TPM identity per host** | Medium | `windows_signer.exe` hardcodes key name `FederatedDeviceKey`. True multi-device federation needs separate machines/VMs/vTPMs sharing one `master.key`. |
| **15 s `SubmitReceipt` timeout** | Medium | Hardcoded in `round0_step6_validate.py`; the 3rd upload triggers synchronous aggregation inside that window. |
| **Label-source drift** | Medium (silent) | `full_test_split.csv` renamed `PHQ8_Score` → `PHQ_Score`; `stage_labels.py` now harvests 141 instead of 189 labels, **silently dropping 47** — 19 of them newly added. |
| **Extractor AppleDouble defect** | Medium (silent) | A first-match heuristic picked a 4 KB macOS `._` sidecar over the real transcript. Caught **only** because its bytes broke UTF-8 decoding. A cleanly-decoding sidecar would have produced a garbage record silently. |
| **Filename-casing hazard** | Low (fixed) | `extract_daic_woz.py` writes `_TRANSCRIPT.csv`; `format_daic_to_lda.py` reads `_Transcript.csv`. Fine on Windows, **breaks on Colab/Linux**. `build_daic_records.py` does case-insensitive lookup. |
| **AVEC label-column ordering** | Medium (fixed) | Header order is `PHQ8_Binary` **before** `PHQ8_Score`, so `next(c if "PHQ" in c)` picks the **binary** column. `stage_labels.py` emits only `Participant_ID,PHQ8_Score`. |

## 14.3 Code smells and duplication
- **`SecureStore` exists in 4 places**, `CentralReceiptManager` in 3, `MultiModalModel` in 3, gRPC stubs in 3 (`runtime/`, `installer/runtime/`, `client_agent/` — the last one **stale**). Divergence is a live risk.
- `installer/runtime/` is a **full mirror** of `runtime/` — every edit must be made twice or the installed client drifts.
- Spelling inconsistency: `centralised_receipts.py` (British) vs `centralized_secure_store.py` (American) — both are imported by exact name.
- Legacy `_apply_aggregation` (numpy) survives in `aggregator.py`, dead on the standard path.
- Repo root holds ~20 one-off scripts and 17 `sess-*` output dirs alongside real modules.
- `create_dp_comparison.py` is ~1400 lines in one file.

## 14.4 Security concerns
| Concern | Note |
|---|---|
| **`RECEIPT_CHAIN_KEY` is optional** | If unset, an ephemeral random key is generated — the chain becomes unverifiable across restarts. Development uses `00112233445566778899aabbccddeeff`; production needs a secrets manager. **This is Blocker 6 — still open.** |
| **The aggregator holds the master key** | It decrypts every client update. This is exactly what HE-CKKS/SMPC would eliminate — both are stubs. |
| **`master.key` is a plain base64 file** | At `~/.federated/data/secure_store/master.key`. No KMS, no TPM sealing of this particular key. |
| **CA private key sits next to the server** | `certs/ca.key` — the server signs CSRs with it in-process via `openssl`. |
| **Server-side `openssl` shell-out** | `EnrollDevice` shells out; depends on `openssl` on PATH. |
| **`enable_tls=false` mode exists** | Intentional for local dev, but it disables the entire mTLS layer. |
| **`self_destruct` is aggressive** | A transient gRPC failure can trigger key wiping. |
| **`weights_only=False` on `torch.load`** | Necessary for these payloads, but it deserialises arbitrary pickles — the aggregator does it on client-supplied data **after** decryption (mitigated by AES-GCM authentication, not eliminated). |

## 14.5 Performance and scalability
- Aggregation is **synchronous within the 3rd `SubmitReceipt` call**. Now on `spawn_blocking` (so the Tokio worker is free), but the client still waits.
- `upload_update` buffers the **whole** upload in RAM (`all_bytes: Vec<u8>`, up to 500 MB) before writing to GridFS.
- The aggregator loads **all** client state dicts into memory simultaneously and stacks them.
- Round state is a `DashMap` in a single process — no horizontal scaling, no leader election.
- Per-device lifetime ε is not persisted; only per-round in-memory accumulation.
- `epsilon_max = 100.0` with ε = 5.3/update means the guard would not fire until ~round 19 — **it has never constrained anything**.

## 14.6 Better design alternatives (for discussion, not committed work)
- **Per-layer or per-tensor DP clipping** instead of one global flatten — directly attacks the √d problem.
- **Persist round state in MongoDB** and rehydrate on startup — removes the destructive-wipe recovery.
- **Make the aggregation threshold and trim ratio config-driven**, and count *distinct devices*.
- **Stream uploads directly into GridFS** instead of buffering.
- **Single shared package** for `SecureStore`/receipts/proto stubs, installed rather than copied.
- **Carry audio/video into the record schema** — the encoders already exist; the work is in the data pipeline, not the model.

---

# 15. CURRENT PROJECT STATUS

| Phase | Status |
|---|---|
| 1–5 (reconstruction, enrollment, round 0) | ✅ Complete |
| 6 (DAIC prep + Colab bundle) | ✅ Complete |
| 8 / 8B (full federated round on real DAIC) | ✅ Complete |
| 9 (baseline evaluation, 113 participants) | ✅ **FROZEN** |
| 9.5 (complete dataset, 188 participants, Colab GPU) | ✅ **FROZEN** |
| 10 (root-cause analysis + Version 2 roadmap) | ✅ Complete |
| 11.1 / 11.2 (Exp 0a / 0b diagnostics) | ✅ Complete |
| 11.3 (diagnostic synthesis) | ✅ Complete |
| 11.4 / 11.5 (evaluation protocol + validation) | ✅ Complete |
| **11.6 (Baseline-CV)** | 🔄 **Training-free half done and real; model half `[PENDING-GPU]`** |
| 12 (Exps 3–12) | ⬜ Not started — gated on 11.6 |

### The frozen baseline numbers (cite these, never recompute them)
| Quantity | Value |
|---|---|
| Dataset | 188 records · 45 pos / 143 neg · 440 excluded (corrupt) · 487 recovered |
| Split | train 170 / val 18, `np.random.seed(42)`, val_split 0.1 |
| **Regression MAE** | **6.654737075169881** |
| **Accuracy** | **0.7222222222222222** (= the majority rate) |
| **Precision / Recall / F1** | **0.0 / 0.0 / 0.0** |
| **Modality text / audio / video** | **2.3365 / 0.0 / 0.0** |
| Val predictions | 18 rows, all class 0, P(pos) 0.199–0.214, pred_phq 5.03–5.09 |
| ε per update / per round | 5.302585092994046 / 15.9078 |
| DP | Gaussian, clip 1.0, nm 1.0, δ 1e-5 · L2-before 11.3448 · L2-after ≈ 543 |
| Global model | round_id 2 · hash `626dfc2a…d1b6` · 1,184,989 B · **from demo corpus** |
| Federation | 1 device · 3 sequential uploads · ≈ 45 s |

### Baseline-CV control arm (real, computed)
| Metric | Value |
|---|---|
| Mean-predictor MAE | **4.9135 ± 0.404** |
| Median-predictor MAE | 4.9090 ± 0.214 |
| ROC-AUC (constant ranker) | **0.500 ± 0.000** |
| PR-AUC (constant) | **0.2394 ± 0.004** (= prevalence) |
| Balanced accuracy / MCC | 0.500 / 0.000 |
| Accuracy (majority) | 0.7606 ± 0.004 |
| Partition-noise floor | regression **SD ≈ 0.40 MAE** |

### The single next action
Run `colab_baseline_cv/run_baseline_cv.py --repeat 1..R` on **Colab GPU** (R ≥ 5), then `aggregate_baseline_cv.py`. This fills §3.2/§4.2/§5/§6/§7 of `PHASE_11_BASELINE_CV.md` and unblocks Phase 12.

---

# 16. FUTURE SCOPE

**Immediate (Phase 12, in roadmap order — one factor per experiment):**
| Exp | Change | Success criterion |
|---|---|---|
| 3 | Convergence (stopping criterion, not 3 fixed epochs) | loss reaches criterion; MAE improves. P/R/F1 may stay 0.0 — that is informative |
| 4 | Decision rule (calibrated operating point, not argmax 0.5) | **Recall > 0 and F1 > 0** — the first non-zero classification in the project's history. MAE must be unchanged |
| 5 | Imbalance-aware objective | Recall / F1 / PR-AUC improve over Baseline-CV **and** Exp 4 |
| 6 | Loss-term rebalancing (CE vs 0.5·MSE) | classification improves **and/or** prediction variance widens |
| 7 | Multimodal activation | `modality_ablation.json`: audio and/or video becomes non-zero; text stays materially positive |

**Later:** dimension-aware DP (Exp 8) · compact DAIC-derived update (Exp 9) · federate a DAIC-derived object (Exp 10) · non-IID multi-client (Exp 11) · ε–utility curve (Exp 12).

**Explicitly OUT of scope** (Roadmap §8.2 — do not "helpfully" fix these): round-state volatility · the LDA 20-participant cap · the 15 s timeout · wall-clock instrumentation as a workstream · unpinning `transformers` · "fixing" global-model hash nondeterminism (**DP requires fresh noise — that is a correctness property**) · recovering participant 440 · amending the frozen reports · bundling multiple changes into one experiment.

**Engineering-hygiene track (alongside, not inside, Version 2):** the two silent data-integrity hazards — label-source drift and the AppleDouble defect. Both returned zero performance, both were caught only by luck, and every Version 2 experiment inherits the same staging path.

---

# 17. HIGH-RISK FILES — modify with extreme care

| Tier | File | Why |
|---|---|---|
| 🔴 **Never modify** | `trainer_mentalbert_daic.py` | SHA-pinned frozen artifact; `run_baseline_cv.py` aborts on change; invalidates the entire baseline |
| 🔴 | `PHASE_8B/9/9.5` reports | Declared FROZEN by Roadmap §2. Historical record |
| 🔴 | `trainer_outputs/baseline_cv/fold_manifest.json` | Every Phase 12 experiment reuses these folds for **paired** comparison |
| 🔴 | `proto/orchestrator.proto` | Changing it requires regenerating Rust **and** Python stubs in 3 locations, and rebuilding the server |
| 🟠 **High risk** | `server/.../src/grpc/server.rs` | All security enforcement. Needs recompile + restart; restart wipes in-memory round state |
| 🟠 | `centralized_secure_store.py` | Any change to `agent`/`context`/HKDF makes **all existing encrypted data undecryptable** |
| 🟠 | `dp_agent/dp_agent.py` | Changes the privacy guarantee and the ε the server accepts |
| 🟠 | `server/aggregator_agent/aggregator.py` | A silent fallback here aggregates garbage without any error |
| 🟠 | `runtime/pipeline.py` | The `msg` byte layout must match `receipts.rs` exactly or every signature fails |
| 🟡 **Care** | `runtime/tpm_guard.py`, `self_destruct.py` | Failures wipe keys |
| 🟡 | `runtime/grpc_client.py` | Re-adding hostname overrides silently disables MITM protection |
| 🟡 | `runtime/*` ↔ `installer/runtime/*` | Mirror pair — edit both |
| 🟡 | `stage_labels.py`, `build_daic_records.py` | Silent-corruption hazards live here |
| ⚪ **Do not use** | `client_agent/**` | Legacy: old proto, Ed25519 (server expects ECDSA P-256), `enc_uri` file path, no `UploadUpdate`, hardcoded ε=1.0 |

---

# 18. GLOSSARY

**Classes / structs**
- `PreprocessRequest` — LDA input: `mode`, `inputs{}`, `config_uri`.
- `SecureStore` — AES-GCM at-rest store; `encrypt_write(uri, bytes)` / `decrypt_read(uri)`.
- `CentralReceiptManager` — HMAC-SHA256 receipt signer; `create_receipt` / `sign` / `verify` / `write_receipt`.
- `DPAgent` — clip + noise + ε; `process_local_update()`.
- `EncryptionAgent` — finalises the encrypted update; `process_dp_update()`.
- `AggregatorAgent` — `aggregate_updates` / `_aggregate_state_dicts` / `_aggregate_tensor` / `run_job`.
- `MultiModalModel` — BERT (CLS-pooled) + optional audio MLP + optional vision MLP → `FusionHead`.
- `FusionHead` — `fc1 → ReLU → Dropout` → `classifier` (2-class), `phq_mu`, `phq_logsigma`.
- `SmallMLP` — `Linear → ReLU → Dropout → Linear → ReLU`, out_dim 128.
- `MultiModalDataset` / `collate_batch` — tokenise (max_len 128), pad audio/video, derive label.
- `Service` (Rust) — holds `state`, `cfg`, `mongo`, `receipt_chain_key`; implements all 7 RPCs.
- `OrchestratorState` — DashMaps: `devices`, `rounds`, `enrollment_tokens`, `pending_enrollments`.
- `Round` / `RoundState{Open,Collecting,Aggregating,Complete}` / `UpdateMeta` / `AggregationReceipt`.

**Functions worth knowing by name**
- `run_pipeline()` — the client federated cycle.
- `_stream_update()` — chunked upload + payload hash.
- `_download_global_model()` — verified streaming download.
- `_rdp_to_dp()` — the ε accountant.
- `compute_state_delta()` / `apply_safety_to_delta()` — delta + safety policy.
- `_load_global_model()` — strict warm-start (Track B).
- `run_aggregation()` (Rust) — subprocess + publish + round N+1.
- `require_client_cert()` / `compute_chain_hmac()` — the two server-side security primitives.
- `derive_device_id()` — SHA-256 of pubkey.
- `consume_otp_from()` — OTP validation with rate limiting.
- `pem_to_verifying_key()` — PEM → SPKI → EC point.
- `read_parquet_records()` / `infer_dims()` — trainer input loading and dim inference.

**Domain terms**
- **PHQ-8** — 8-item depression questionnaire, 0–24. Binarised at **> 10.0** in Track C.
- **DAIC-WOZ** — Distress Analysis Interview Corpus, Wizard-of-Oz. Ellie = the virtual interviewer.
- **LDA** — here, **Local Data Agent** (*not* Latent Dirichlet Allocation).
- **RDP** — Rényi Differential Privacy; converted to (ε, δ)-DP.
- **Track A / B / C** — the three trainers (§4.2).
- **Baseline-CV** — the original trainer re-measured under 5-fold CV; the **only** valid quantitative comparator for Version 2.
- **RC-1…RC-7** — the seven root causes from Phase 10.2.
- **[E] / [H] / [I]** — evidence / hypothesis / interpretation. This project's reports use them rigorously; keep the discipline.

---

*Compiled from a full read of the repository on 2026-08-01. Evidence-backed claims are marked [E]; inferences [I]; open questions [?]. No source file was modified in producing this document.*
