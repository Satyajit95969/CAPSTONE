# MediProof — Architecture Reference

**Version:** Phase 3 reconstruction  
**System:** Privacy-Preserving Federated Learning for Mental Health Misinformation Detection  
**Stack:** Python (agents, runtime) · Rust (orchestrator) · gRPC (transport) · MongoDB (persistence)

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [Component Responsibilities](#2-component-responsibilities)
3. [Runtime Flow](#3-runtime-flow)
4. [Enrollment Flow](#4-enrollment-flow)
5. [Federated Learning Flow](#5-federated-learning-flow)
6. [gRPC Communication Flow](#6-grpc-communication-flow)
7. [Rust Orchestrator Responsibilities](#7-rust-orchestrator-responsibilities)
8. [Security Architecture](#8-security-architecture)
9. [Aggregation Pipeline](#9-aggregation-pipeline)
10. [MongoDB Collections and Relationships](#10-mongodb-collections-and-relationships)
11. [Current Reconstruction Status](#11-current-reconstruction-status)
12. [Legacy Components](#12-legacy-components)
13. [Current Blockers](#13-current-blockers)
14. [Future Work](#14-future-work)

---

## 1. Directory Structure

```
BE-Major-Project/
│
├── LDA/                                    # Local Data Agent — client-side preprocessing
│   ├── app/
│   │   ├── main.py                         # FastAPI entry point; PreprocessRequest handler
│   │   └── pipelines/
│   │       ├── text.py                     # PII scrubbing, NER, BERT embeddings
│   │       ├── audio.py                    # wav2vec2, openSMILE eGeMAPS, prosody
│   │       ├── video.py                    # OpenFace AU extraction, face blurring
│   │       └── session_processor.py        # Multimodal sync, speaker diarization
│   ├── configs/local_config.yaml           # LDA config schema
│   └── requirements.txt
│
├── trainer_agent/                          # Model training orchestrator
│   ├── trainer_mentalbert_privacy.py       # MentalBERT trainer (autonomous/supervised/RL)
│   ├── model.py                            # MultiModalModel architecture
│   ├── trainer.py                          # Training loop utilities
│   ├── utils.py                            # Helper functions
│   ├── dummy_session.py                    # Synthetic data generator for testing
│   ├── app.py                              # FastAPI wrapper
│   └── requirements.txt
│
├── dp_agent/                               # Differential Privacy Agent
│   ├── dp_agent.py                         # L2-clip + 6 noise mechanisms
│   └── run_demo_single_process.py
│
├── enc_agent/                              # Encryption Agent
│   ├── enc_agent.py                        # AES-GCM (complete), HE-CKKS + SMPC (stubs)
│   └── run_demo_single_process.py
│
├── client_agent/                           # ⚠ LEGACY — see Section 12
│   ├── client_agent.py                     # Old client flow; outdated proto
│   └── grpc/
│       ├── orchestrator_pb2.py             # OLD generated proto (no streaming RPCs)
│       └── orchestrator_pb2_grpc.py        # OLD generated gRPC stub
│
├── server/
│   ├── orchestration_agent/                # Rust gRPC orchestration server
│   │   ├── src/
│   │   │   ├── main.rs                     # Entry: MongoDB init, OTP bootstrap, serve
│   │   │   ├── config.rs                   # TOML config loader
│   │   │   ├── crypto.rs                   # SHA-256 hash, constant-time comparison
│   │   │   ├── errors.rs                   # OrchestratorError enum
│   │   │   ├── identity.rs                 # derive_device_id (SHA-256 of pubkey)
│   │   │   ├── ledger.rs                   # Minimal flat-file ledger (stub)
│   │   │   ├── otp.rs                      # 6-digit OTP, 10-min expiry, rate-limit
│   │   │   ├── pubsub.rs                   # Round notification stub
│   │   │   ├── receipts.rs                 # ECDSA P-256 signature verification
│   │   │   ├── round.rs                    # RoundState, Round, UpdateMeta, AggregationReceipt
│   │   │   ├── state.rs                    # OrchestratorState (DashMap), round 1 init
│   │   │   └── grpc/
│   │   │       ├── mod.rs                  # tonic::include_proto! + pub mod server
│   │   │       └── server.rs               # All 7 RPC handlers (~1100 lines)
│   │   ├── proto/orchestrator.proto        # Canonical gRPC service definition
│   │   ├── config/orchestrator.toml        # Server address + TLS paths
│   │   ├── certs/gen_certs.sh              # Certificate generation script
│   │   ├── build.rs                        # tonic_build proto compilation
│   │   └── Cargo.toml                      # Rust dependencies
│   │
│   └── aggregator_agent/                   # Python secure aggregation
│       ├── aggregator.py                   # mean / trimmed_mean / coordinate_median
│       └── core/
│           ├── centralized_secure_store.py # Server-side SecureStore reference
│           └── centralized_receipts.py     # Server-side receipt manager reference
│
├── runtime/                                # Production client runtime (installed path)
│   ├── federated_client.py                 # Main entry: daemon | run-once
│   ├── pipeline.py                         # Full pipeline: LDA→Trainer→DP→Enc→Upload
│   ├── grpc_client.py                      # mTLS dual-channel gRPC stub factory
│   ├── daemon.py                           # Continuous background daemon loop
│   ├── capture.py                          # Session capture orchestrator
│   ├── tpm_guard.py                        # TPM sign + unseal + pubkey (Win/Linux)
│   ├── runtime_guard.py                    # Runtime integrity gates
│   ├── self_destruct.py                    # Secure self-destruct mechanism
│   ├── idle.py                             # System idle detection
│   ├── offline_queue.py                    # Offline update queueing
│   ├── logging_config.py                   # MetricsCollector, HealthReporter
│   ├── validate_deps.py
│   ├── config_validator.py
│   └── grpc/
│       ├── orchestrator_pb2.py             # ✅ Current generated proto
│       └── orchestrator_pb2_grpc.py        # ✅ Current generated gRPC stub
│
├── installer/                              # Installation and security bootstrapping
│   ├── fs/                                 # Component installers (OpenFace, ffmpeg, etc.)
│   ├── security/
│   │   ├── anti_debug.py                   # Anti-debugging checks
│   │   ├── deps_windows.py                 # Windows-specific dependencies
│   │   ├── integrity.py                    # Code integrity verification + watcher
│   │   ├── runtime_guard.py                # Runtime guard implementation
│   │   ├── self_destruct.py                # Self-destruct implementation
│   │   ├── tpm_attestation.py              # TPM attestation
│   │   ├── tpm_seal.py                     # TPM data sealing
│   │   └── windows_runtime.py              # Windows-specific runtime
│   ├── windows_signer/                     # Rust Windows ECDSA P-256 signing utility
│   │   ├── src/main.rs
│   │   └── Cargo.toml
│   ├── runtime/                            # Mirror of runtime/ (copied on install)
│   │   └── grpc/
│   │       ├── orchestrator_pb2.py         # ✅ Current (same as runtime/grpc/)
│   │       └── orchestrator_pb2_grpc.py
│   ├── installer_core.py
│   └── installer_gui.py
│
├── centralized_secure_store.py             # Root-level SecureStore (AES-GCM, HKDF)
├── centralised_receipts.py                 # Root-level HMAC receipt manager
├── configs/
│   └── local_config.yaml                   # LDA configuration
├── requirements.txt                        # Root Python dependencies
├── requirements.lock.txt                   # Locked dependency versions
└── README.md
```

**Installed runtime layout** (`~/.federated/`):

```
~/.federated/
├── bin/
│   ├── federated-client                    # Symlink → runtime/federated_client.py
│   └── windows_signer.exe                  # Windows ECDSA signer (Win only)
├── venv/                                   # Python virtual environment
├── runtime/                                # Installed runtime package
├── installer/                              # Installed installer package
├── agents/                                 # Installed agent packages
│   ├── lda/, trainer/, dp/, enc/
├── core/                                   # Installed core (SecureStore, receipts)
├── keys/
│   ├── ca.pem                              # Server CA certificate
│   ├── client.pem                          # Client certificate (post-enrollment)
│   └── client.key                          # Client private key (post-enrollment)
├── tpm/
│   ├── device_pubkey.pem                   # Device ECDSA P-256 public key
│   ├── device.ctx                          # TPM key context (Linux)
│   └── sealed_secret.ctx                   # TPM-sealed master secret (Linux)
├── data/
│   ├── secure_store/                       # AES-GCM encrypted data store
│   ├── input/                              # Session capture input
│   └── global_models/                      # Downloaded global models
├── configs/
│   └── local_config.yaml
└── state/
    ├── runtime.lock                        # Daemon single-instance lock
    └── daemon.heartbeat                    # Heartbeat timestamp
```

---

## 2. Component Responsibilities

### Local Data Agent (LDA)

Runs entirely on the client device. Accepts a `PreprocessRequest` specifying a processing mode (`session`, `batch`, `text`, `continuous`) and input paths.

- **Video pipeline** — OpenFace for facial action unit extraction, face blurring, gaze tracking
- **Audio pipeline** — wav2vec2 embeddings, openSMILE eGeMAPS features, prosody analysis, WebRTC VAD
- **Text pipeline** — spaCy NER-based PII scrubbing (PERSON, GPE, ORG), sentence-transformer embeddings, sentiment analysis
- **Session processor** — aligns audio/video timelines, handles speaker diarization

All outputs are encrypted with AES-GCM via the shared `SecureStore` before being written to disk. No cleartext data is written. Each modality produces a signed HMAC receipt.

Returns: `session_id`, `artifact_manifest` URI (encrypted JSONL), per-modality `receipt` URIs, row `count`.

### Trainer Agent

Fine-tunes a multimodal mental health detection model on locally preprocessed data.

**Model architecture:**
```
MultiModalModel
  ├── BERT text encoder         (768-dim hidden, MentalBERT weights)
  ├── Audio MLP encoder         (audio_dim → 128-dim)
  ├── Vision MLP encoder        (vision_dim → 128-dim)
  └── FusionHead
        ├── Concatenate all modalities
        └── Dense head → classification + PHQ regression
```

**Training modes:**
- `autonomous` — inference only; no weight updates
- `supervised` — physician-corrected fine-tuning; produces delta state dict
- `rl` — REINFORCE-style online updates with clinician reward

**Safety policy (built-in, non-negotiable):**
- Per-parameter absolute delta clamp (`max_param_change = 1e-3`)
- Global delta L2 norm scaling (`max_global_delta_norm = 1.0`)
- Gradient clipping (norm ≤ 1.0)
- Conservative learning rate (1e-3), small batch sizes (8–32)

Returns: `local_update_uri` (encrypted `.pt` delta state dict), metrics JSON, explainability logs.

### Differential Privacy Agent

Applies differential privacy noise to the trainer's delta before it leaves the device.

**Processing steps:**
1. Decrypt the delta state dict from `SecureStore`
2. Flatten all parameters into a single vector
3. L2-clip: if `‖delta‖₂ > clip_norm`, scale down by `clip_norm / ‖delta‖₂`
4. Add noise: `scale = noise_multiplier × sensitivity`
5. Unflatten back to state dict
6. Re-encrypt and write to `SecureStore`
7. Create HMAC-signed receipt with `l2_norm_before`, `l2_norm_after`, `clip_applied`

**Supported mechanisms:** `gaussian` · `laplace` · `uniform` · `exponential` · `student_t` · `none`

Returns: `update_uri`, `receipt_uri`, `l2_norm_before`, `l2_norm_after`.

> **Known gap:** `epsilon_spent` is not returned. See [Section 13](#13-current-blockers).

### Encryption Agent

Validates the DP receipt and finalises the encrypted update for transmission.

**Modes:** `aes` (complete) · `fernet` (partial) · `kms_envelope` (AWS KMS, mocked locally) · `he_ckks` (stub) · `smpc` (placeholder)

Returns: finalized update URI in a receipt envelope.

### Rust Orchestrator

See [Section 7](#7-rust-orchestrator-responsibilities) for full detail.

### Aggregator Agent

Runs server-side, invoked as a subprocess by the Rust orchestrator after a round threshold is reached.

Reads bytes from MongoDB GridFS (preferred) or a canonical `SecureStore` path. Applies robust aggregation over the decrypted parameter vectors.

**Aggregation modes:** `mean` · `trimmed_mean` (default, 10% trim ratio) · `coordinate_median`

> **Known gap:** The aggregated model is not written back to MongoDB `global_models`. See [Section 13](#13-current-blockers).

### SecureStore (`centralized_secure_store.py`)

Shared AES-GCM at-rest encryption layer used by every agent.

- Master key: 32 random bytes stored at `secure_store/master.key` (base64-encoded)
- Per-agent key derivation via HKDF-SHA256 with the agent name as context
- Storage format: JSON envelope `{"nonce": <b64>, "ciphertext": <b64>}`
- API: `encrypt_write(uri, bytes)` → URI; `decrypt_read(uri)` → bytes

### Receipt Manager (`centralised_receipts.py`)

Produces HMAC-SHA256 signed audit receipts for every operation.

- Each receipt carries: `agent`, `session_id`, `operation`, `params`, `outputs`, `timestamp`, `uuid`, `hmac`
- Receipts are written as JSON files under the configured `receipts/` directory
- HMAC key is derived from the master key so receipts are cryptographically bound to the installation

---

## 3. Runtime Flow

The production client entry point is `runtime/federated_client.py`. It supports two modes:

```
federated_client.py [--daemon | --run-once]
  │
  ├── setup_logging()
  ├── IntegrityWatcher.start()          -- file-hash monitoring every 5 minutes
  ├── runtime_guard()                   -- TPM unseal + anti-debug + integrity check
  ├── get_device_pubkey()               -- read ECDSA P-256 pubkey from TPM
  │     └── device_id = SHA-256(pubkey)
  ├── create_grpc_stub(SERVER_ADDR)
  │     ├── Attempt full mTLS (ca.pem + client.pem + client.key)
  │     └── Fall back to server-TLS only if client cert absent (pre-enrollment)
  ├── RegisterDevice(CSR) [best-effort]
  │
  ├── [run-once] run_pipeline(stub, device_id, master_secret)
  └── [daemon]   daemon_loop(stub, device_id, master_secret)
                   ├── wait_until_idle()
                   ├── capture_session(duration_s=300)
                   ├── run_pipeline(stub, device_id, master_secret, session_dir)
                   └── sleep(3600)
```

**`run_pipeline` steps:**

```
1. GetRound RPC
   └── Abort if round.state ≠ "Collecting"

2. DownloadGlobalModel RPC  [if round.global_model_available]
   ├── Stream ModelChunk[]
   ├── Verify SHA-256 per chunk
   ├── Verify SHA-256 of full model on last chunk
   └── Save → ~/.federated/data/global_models/global_roundN.pt

3. LDA preprocessing  (PreprocessRequest, mode="session")
   ├── video/audio/text → encrypted parquet files
   └── Returns: artifact_manifest URI

4. Trainer  (orchestrate, mode="supervised", epochs=1)
   ├── Load parquet from SecureStore
   ├── Fine-tune on local data (global model loaded if available)
   └── Returns: local_update_uri (encrypted delta .pt)

5. DP noise  (DPAgent, mechanism="gaussian", clip_norm=1.0, noise_mult=1.0)
   ├── Decrypt delta, flatten, L2-clip, add Gaussian noise, unflatten
   ├── Re-encrypt, write receipt
   └── Returns: update_uri, receipt_uri, l2_norm_before, l2_norm_after

6. Encryption  (EncryptionAgent, mode="aes")
   └── Returns: finalized encrypted update URI

7. UploadUpdate RPC  (client-streaming)
   ├── Read encrypted bytes from update file
   ├── Split into 1 MB chunks
   ├── Each chunk: chunk_index, total_chunks, data, SHA-256(data)
   ├── Server verifies each chunk hash, stores in GridFS
   └── Returns: UploadAck.server_handle (GridFS ObjectId)

8. SubmitReceipt RPC
   ├── payload_hash = SHA-256(all uploaded bytes)
   ├── msg = device_id ‖ round_id_BE8 ‖ payload_hash
   ├── signature = TPM.sign(msg)   [ECDSA P-256 DER]
   └── Receipt(device_id, round_id, payload_hash, epsilon_spent,
              signature, enc_handle=server_handle, scheme, nonce)
```

---

## 4. Enrollment Flow

Enrollment establishes a device's cryptographic identity with the server. It is a two-phase OTP ceremony.

### Phase B1 — Request Enrollment

```
Client                                    Server
  │                                          │
  │── RequestEnrollment ───────────────────▶ │
  │   (device_pubkey, csr, device_info)      │
  │                                          ├── fingerprint = SHA-256(pubkey)[:8 bytes hex]
  │                                          ├── OTP = random 6-digit, stored with 600s TTL
  │                                          ├── pending_enrollments[fingerprint] = (pubkey, csr)
  │                                          ├── Print OTP to operator console
  │◀─ EnrollmentRequestAck ──────────────── │
  │   (accepted=true, device_fingerprint)    │
```

The operator communicates the OTP to the user out-of-band (e.g., verbally or via secure admin channel).

### Phase B2 — Complete Enrollment

```
Client                                    Server
  │                                          │
  │── EnrollDevice ──────────────────────▶  │
  │   (enrollment_token, device_pubkey, csr) │
  │                                          ├── consume_otp_from(token, peer_addr)
  │                                          │     ├── Check rate-limit (5 failures → 5-min lockout)
  │                                          │     ├── Check not used, not expired
  │                                          │     └── Mark as used
  │                                          ├── derive_device_id = SHA-256(device_pubkey)
  │                                          ├── openssl x509 -req (sign CSR with CA key)
  │                                          ├── MongoDB upsert:
  │                                          │     devices { device_id, pubkey_pem,
  │                                          │               enrolled_at, peer_addr }
  │                                          ├── state.devices[device_id] = pubkey
  │                                          └── Remove from pending_enrollments
  │◀─ EnrollResponse ─────────────────────  │
  │   (ok=true, client_cert PEM bytes)       │
  │                                          │
  └── Save client_cert → ~/.federated/keys/client.pem
```

**OTP security properties:**
- 6-digit random integer (100,000–999,999)
- 600-second (10-minute) expiry
- Single-use (marked `used=true` on consumption)
- Rate-limited: 5 failed attempts → 5-minute lockout per peer address
- Expired OTPs purged on each generation and consumption attempt

---

## 5. Federated Learning Flow

### Server-side round lifecycle

```
Startup
  └── Round 1 pre-seeded: { id=1, state=Collecting, epsilon_max=1.0 }

Per client (device must be enrolled):
  1. GetRound RPC
     └── Returns current round metadata + global_model_available flag

  2. UploadUpdate RPC (streaming)
     ├── Reject unenrolled devices
     ├── Reject if round.state ≠ Collecting
     ├── Verify per-chunk SHA-256
     ├── Enforce 500 MB cap (DoS protection)
     ├── Store bytes in MongoDB GridFS
     └── Insert model_updates doc { device_id, round_id, payload_hash,
                                     file_id, verified=false, size_bytes }

  3. SubmitReceipt RPC
     ├── Validate mTLS client cert
     ├── Validate enc_handle is a valid GridFS ObjectId
     ├── Reject epsilon_spent ≤ 0
     ├── Look up device pubkey_pem from MongoDB
     ├── Verify ECDSA-P256 signature over (device_id ‖ round_id_BE8 ‖ payload_hash)
     ├── Cross-verify payload_hash against stored upload hash
     ├── Mark model_updates doc as verified=true
     ├── Enforce round epsilon ceiling: accumulated + submitted ≤ epsilon_max
     ├── HMAC-chain receipt into MongoDB receipts collection
     ├── Push UpdateMeta { device_id, enc_uri=GridFS_ObjectId, scheme, nonce }
     └── Trigger aggregation when updates.len() ≥ 3

Aggregation (triggered at ≥ 3 updates):
  ├── Round state → Aggregating
  ├── Spawn Python subprocess: server/aggregator_agent/aggregator.py
  │   └── Reads updates from GridFS, applies trimmed_mean, saves .npy
  └── Round state → Complete
```

### Client-side round interaction

Clients poll `GetRound` each cycle. A client participates only if:
- `round_meta.state == "Collecting"`
- The device is enrolled (server enforces this at `UploadUpdate`)
- The device has local session data to train on

After submitting, clients optionally call `DownloadGlobalModel` at the start of the next round to initialise from the aggregated weights rather than random weights.

---

## 6. gRPC Communication Flow

### Service definition (`proto/orchestrator.proto`)

```protobuf
service Orchestrator {
  rpc RegisterDevice       (CSR)                   returns (Certificate);        // deprecated
  rpc RequestEnrollment    (EnrollmentRequest)      returns (EnrollmentRequestAck);
  rpc EnrollDevice         (EnrollRequest)          returns (EnrollResponse);
  rpc GetRound             (DeviceId)               returns (RoundMetadata);
  rpc UploadUpdate         (stream UpdateChunk)     returns (UploadAck);          // client-streaming
  rpc SubmitReceipt        (Receipt)                returns (Ack);
  rpc DownloadGlobalModel  (RoundRequest)           returns (stream ModelChunk);  // server-streaming
}
```

### Key message fields

**`UpdateChunk`** — one chunk of the encrypted model update stream:

| Field | Type | Purpose |
|-------|------|---------|
| `session_id` | string | Client-generated correlation ID |
| `round_id` | uint64 | Must match an active Collecting round |
| `device_id` | bytes | SHA-256 of device pubkey |
| `chunk_index` | uint64 | 0-based, sequential; server rejects gaps |
| `total_chunks` | uint64 | Declared upfront; verified at stream end |
| `data` | bytes | Encrypted update bytes for this chunk |
| `chunk_hash` | bytes | SHA-256(`data`) — verified by server per chunk |

**`Receipt`** — submitted after a successful `UploadUpdate`:

| Field | Type | Purpose |
|-------|------|---------|
| `device_id` | bytes | SHA-256 of device pubkey |
| `round_id` | uint64 | Must match an active Collecting round |
| `payload_hash` | bytes | SHA-256 of ALL bytes uploaded (32 bytes) |
| `epsilon_spent` | double | Must be > 0; must come from real RDP accountant |
| `signature` | bytes | ECDSA-P256 DER over (`device_id` ‖ `round_id_BE8` ‖ `payload_hash`) |
| `enc_handle` | string | GridFS ObjectId from `UploadAck.server_handle` — NOT a file path |
| `scheme` | string | Encryption scheme identifier (e.g. `"AES-GCM-DP-ECDSA"`) |
| `nonce` | string | Optional nonce metadata |

**`ModelChunk`** — one chunk of the global model download stream:

| Field | Type | Purpose |
|-------|------|---------|
| `chunk_index` | uint64 | 0-based index |
| `total_chunks` | uint64 | Total chunks declared by server |
| `data` | bytes | Model bytes for this chunk |
| `chunk_hash` | bytes | SHA-256(`data`) — verified by client |
| `model_hash` | bytes | SHA-256 of full model; present on final chunk only |

### Channel setup

Two distinct TLS channels are used:

| Channel | When used | Credentials |
|---------|-----------|-------------|
| Server-TLS only | Pre-enrollment; installer | `ca.pem` (server CA) only |
| Full mTLS | All operational calls | `ca.pem` + `client.pem` + `client.key` |

The `create_grpc_stub()` function in `runtime/grpc_client.py` attempts mTLS first and falls back to server-TLS if the client certificate is not yet installed. The server enforces mTLS for all endpoints except `RequestEnrollment` (which the client calls before it has a cert).

**TLS hostname verification is enforced** — the server certificate must have a Subject Alternative Name (SAN) matching the connection address. No override options are set. Use `certs/gen_certs.sh <SERVER_IP>` to regenerate certificates with the correct SAN.

---

## 7. Rust Orchestrator Responsibilities

Source: `server/orchestration_agent/src/`

### Module map

| Module | Responsibility |
|--------|---------------|
| `main.rs` | Entry point: logging, config load, MongoDB connect, index creation, OTP bootstrap, pubsub start, gRPC serve |
| `config.rs` | Loads `config/orchestrator.toml` → `Config { server: { addr, enable_tls }, tls: { ca_cert, ca_key, server_cert, server_key } }` |
| `state.rs` | `OrchestratorState`: `DashMap` for devices, rounds, enrollment_tokens, pending_enrollments; seeds Round 1 at startup |
| `round.rs` | `RoundState` enum (`Open` · `Collecting` · `Aggregating` · `Complete`), `Round`, `UpdateMeta`, `AggregationReceipt` |
| `identity.rs` | `derive_device_id(pubkey) → Vec<u8>`: SHA-256 of raw pubkey bytes |
| `otp.rs` | `generate_otp_for()`, `consume_otp_from()`: OTP store with expiry + rate-limiter |
| `receipts.rs` | `verify(pubkey_pem, msg, sig_der)`: PEM → DER → EC point → `VerifyingKey::from_sec1_bytes` → ECDSA verify |
| `crypto.rs` | `hash_bytes(data) → [u8;32]` (SHA-256), `ct_eq(a, b) → bool` (constant-time) |
| `errors.rs` | `OrchestratorError` enum used across modules |
| `ledger.rs` | `append(entry)`: appends bytes to `ledger.log`; minimal stub |
| `pubsub.rs` | `start(state)`: spawns Tokio task logging round info every 10 seconds; stub |
| `grpc/mod.rs` | `tonic::include_proto!("orchestrator")` + re-exports `server` module |
| `grpc/server.rs` | `Service` struct + `Orchestrator` trait impl for all 7 RPCs; `serve()` bootstrap |

### gRPC handler summary

**`upload_update` (streaming)**
- Enforces mTLS on first chunk
- Validates device enrollment via MongoDB `devices` collection
- Validates round state is `Collecting`
- Verifies SHA-256 of each chunk
- Rejects out-of-order chunks and total_chunks mismatch
- Accumulates up to 500 MB; rejects beyond
- Stores in MongoDB GridFS via `db.gridfs_bucket()`
- Records `model_updates` document with `verified=false`
- Returns `UploadAck { ok, server_handle: GridFS ObjectId }`

**`submit_receipt`**
- Enforces mTLS
- Validates all required fields and epsilon > 0
- Looks up `pubkey_pem` from MongoDB `devices`
- Verifies ECDSA-P256 DER signature
- Validates `enc_handle` is a parseable 24-hex GridFS ObjectId
- Cross-verifies `payload_hash` against the hash computed during upload
- Marks `model_updates` record as `verified=true`
- Enforces per-round epsilon budget ceiling
- Computes HMAC chain link and stores receipt in MongoDB `receipts`
- Pushes `UpdateMeta` into in-memory `round.updates`
- Triggers `run_aggregation` at threshold (≥ 3 updates)

**`download_global_model` (streaming)**
- Enforces mTLS
- Looks up `global_models` collection by `round_id`
- Reads model bytes from GridFS by stored `file_id`
- Splits into 1 MB chunks, each with SHA-256
- Includes full-model SHA-256 on final chunk
- Returns `Pin<Box<dyn Stream<...>>>` (required by tonic 0.11)

**`request_enrollment`**
- No mTLS required (client has no cert yet)
- Generates per-device OTP, stores pending enrollment
- Displays OTP to operator on stdout

**`enroll_device`**
- Validates OTP via `consume_otp_from` (rate-limited)
- Signs CSR with `openssl x509`
- Upserts device record into MongoDB `devices`
- Returns signed client certificate

**`get_round`**
- Enforces mTLS
- Updates `last_seen` timestamp in MongoDB
- Returns `RoundMetadata` for round 1
- Includes `global_model_available` flag (checks MongoDB `global_models`)

**`run_aggregation`** (internal, not an RPC)
- Serialises the round's update list (gridfs_id, scheme, nonce) as JSON
- Spawns `python3 server/aggregator_agent/aggregator.py` synchronously (blocking)
- Parses stdout as JSON `{ aggregated_uri, num_updates, mode }`
- Advances round to `Complete`

### Dependencies (`Cargo.toml`)

| Crate | Version | Purpose |
|-------|---------|---------|
| `tokio` | 1.37 | Async runtime |
| `tonic` | 0.11 | gRPC framework (TLS feature) |
| `prost` | 0.12 | Protobuf codegen runtime |
| `mongodb` | 2.8 | MongoDB async driver (tokio-runtime) |
| `bson` | 2.10 | BSON serialisation (chrono-0_4 feature) |
| `p256` | 0.13 | ECDSA P-256 |
| `ecdsa` | 0.16 | ECDSA signature types (pem feature) |
| `sha2` | 0.10 | SHA-256 |
| `hmac` | 0.12 | HMAC-SHA256 for receipt chaining |
| `dashmap` | 5.5 | Lock-free concurrent hash map |
| `rand` | 0.8 | OTP generation + ephemeral key fallback |
| `rcgen` | 0.13 | X.509 certificate generation |
| `futures` | 0.3 | `AsyncReadExt` / `AsyncWriteExt` / `StreamExt` (GridFS compatibility) |

---

## 8. Security Architecture

### 8.1 TPM (Trusted Platform Module)

The device's long-term ECDSA P-256 key pair is managed by the TPM to prevent key extraction.

| Operation | Windows | Linux |
|-----------|---------|-------|
| Key generation | `windows_signer.exe --keygen` | `tpm2_createprimary` + `tpm2_create` |
| Sign message | `windows_signer.exe --sign` (stdin → stdout) | `tpm2_sign -s ecdsa -g sha256` |
| Read public key | `windows_signer.exe --pubkey <path>` | `tpm2_readpublic` → PEM |
| Seal master secret | `installer/security/tpm_seal.py` | `tpm2_seal` bound to PCR values |
| Unseal master secret | Read from `~/.federated/secrets/master.bin` | `tpm2_unseal -c sealed_secret.ctx` |

The public key PEM is written to `~/.federated/tpm/device_pubkey.pem` and submitted as the CSR during enrollment. The server stores it in MongoDB and uses it to verify every subsequent ECDSA signature.

Any failure in `sign_message()` or `unseal_master_secret()` triggers `self_destruct`.

### 8.2 Mutual TLS (mTLS)

Every operational gRPC endpoint requires a valid client certificate:

```
┌────────────────────────────┐     mTLS      ┌──────────────────────────────┐
│  Client                    │ ◀──────────▶  │  Server                      │
│  cert: client.pem          │               │  cert: server.pem            │
│  key:  client.key          │               │  key:  server.key            │
│  CA:   ca.pem              │               │  CA:   ca.pem                │
└────────────────────────────┘               └──────────────────────────────┘
```

- Server CA signs all certificates (via `openssl x509 -req` in `EnrollDevice`)
- `require_client_cert()` rejects any request missing a valid peer certificate
- No `ssl_target_name_override` or `default_authority` options — hostname verification is enforced
- `RequestEnrollment` is the only RPC accessible without a client certificate

Certificate authority setup: `bash server/orchestration_agent/certs/gen_certs.sh <SERVER_IP>`

### 8.3 Integrity Monitoring

Two layers of integrity monitoring run on the client:

**Static integrity guard** (`installer/security/integrity.py`, `integrity_guard()`)
- Computes SHA-256 hashes of critical runtime files at install time
- Called at module import time by `client_agent.py`, `dp_agent.py`, and `enc_agent.py`
- Any hash mismatch halts execution immediately

**Dynamic integrity watcher** (`IntegrityWatcher` in `installer/security/integrity.py`)
- Runs as a background thread
- Checks file hashes every 300 seconds
- After `max_violations` (default: 2) failures, triggers `self_destruct`
- Started unconditionally in `federated_client.py` before any other operation

### 8.4 Self-Destruct

`runtime/self_destruct.py` provides `trigger_self_destruct(reason: str)`.

Invoked when:
- TPM signing or unsealing fails
- gRPC channel cannot be established
- Integrity violations exceed the threshold
- Anti-debugging checks detect a debugger

Action: securely wipes sensitive files from `~/.federated/` (keys, secrets, sealed contexts) and terminates the process. Designed to limit data exposure if the device is compromised.

### 8.5 Receipt Chain

Every operation in the pipeline produces a signed receipt. On the server, receipts form a tamper-evident HMAC chain stored in MongoDB.

**Client-side receipts** (HMAC-SHA256, Python):
- Generated by `CentralReceiptManager` for each agent operation
- HMAC key derived from master key via HKDF
- Stored as JSON files under `receipts/`

**Server-side receipt chain** (HMAC-SHA256, Rust):
- Each `SubmitReceipt` call links to the previous receipt via:
  ```
  hmac_chain[n] = HMAC-SHA256(key, hmac_chain[n-1] ‖ "|" ‖ payload_hash_hex)
  ```
- Genesis link uses the literal string `"genesis"` as the previous value
- Any insertion, deletion, or reordering of receipts breaks all subsequent chain links
- HMAC key loaded from `RECEIPT_CHAIN_KEY` environment variable (hex-encoded 32 bytes)
- If unset, an ephemeral random key is generated with a warning — receipts are NOT verifiable across restarts without this key set

### 8.6 ECDSA Signatures

Each submitted receipt is signed by the device's TPM-backed ECDSA P-256 key.

**Canonical message format:**
```
msg = device_id (32 bytes, SHA-256 of pubkey)
    ‖ round_id  (8 bytes, big-endian uint64)
    ‖ payload_hash (32 bytes, SHA-256 of all uploaded bytes)
```

**Server verification** (`receipts.rs`):
1. Decode PEM `-----BEGIN PUBLIC KEY-----` block
2. Base64-decode to DER (SPKI format, 91 bytes for P-256)
3. Skip 26-byte SPKI header, extract 65-byte EC point (`0x04 ‖ X[32] ‖ Y[32]`)
4. `VerifyingKey::from_sec1_bytes(ec_point)`
5. `Signature::from_der(sig_bytes)`
6. `verifying_key.verify(msg, &signature)`

This binds each receipt cryptographically to: the specific device, the specific round, and the exact bytes that were uploaded.

### 8.7 Anti-Debugging

`installer/security/anti_debug.py` performs platform-specific checks at startup to detect attached debuggers. On detection, it triggers `self_destruct`.

---

## 9. Aggregation Pipeline

```
Rust Orchestrator (run_aggregation)
  │
  ├── Collect UpdateMeta[] from round.updates
  │   Each entry has: { gridfs_id, scheme, nonce }
  │
  ├── Serialise job JSON → stdin
  │   {
  │     "round_id": N,
  │     "mode": "trimmed_mean",
  │     "trim_ratio": 0.1,
  │     "updates": [
  │       { "gridfs_id": "<ObjectId>", "scheme": "...", "nonce": null },
  │       ...
  │     ]
  │   }
  │
  └── Spawn: python3 server/aggregator_agent/aggregator.py

Python Aggregator (AggregatorAgent)
  │
  ├── For each update:
  │   ├── Fetch bytes from MongoDB GridFS by ObjectId
  │   ├── Attempt torch.load → state dict or tensor
  │   │   └── Fallback: interpret as raw float32 numpy array
  │   └── Flatten all tensors into 1D float32 vector
  │
  ├── Shape normalisation: trim all vectors to minimum length if shapes differ
  │
  ├── Stack into matrix: shape (num_updates, param_count)
  │
  └── Apply aggregation:
      ├── mean:              np.mean(arr, axis=0)
      ├── trimmed_mean:      sort per-coordinate, trim 10% from each end, mean
      └── coordinate_median: np.median(arr, axis=0)

Output
  ├── Save aggregated vector → ./aggregated_round_N.npy
  └── Return JSON: { round_id, aggregated_uri, num_updates, mode }

Rust (post-aggregation)
  ├── Parse stdout JSON
  ├── Store aggregated_uri in round.upload_uri
  ├── Set round.state → Complete
  └── Set aggregation_receipt { round_id, num_updates, mode, aggregated_uri }
```

**Security properties:**
- Path traversal protection: `enc_path` inputs are validated to be inside `_CANONICAL_ROOT` (`~/.federated/data/secure_store`)
- GridFS ObjectId validation: arbitrary strings rejected with `ObjectId(gridfs_id)` parse check
- All decryption uses the canonical `SecureStore` root — same `master.key` as the agents that encrypted the data

---

## 10. MongoDB Collections and Relationships

Database name: `federated`

### `devices`

Stores enrolled device identities.

| Field | Type | Description |
|-------|------|-------------|
| `device_id` | string | Hex-encoded SHA-256 of device pubkey (unique index) |
| `pubkey_pem` | string | PEM-encoded ECDSA P-256 public key (used for signature verification) |
| `enrolled_at` | BsonDateTime | Enrollment timestamp |
| `last_seen` | BsonDateTime | Updated on every `GetRound` call |
| `peer_addr` | string | IP:port of enrolling client |

### `model_updates`

Tracks each uploaded model update.

| Field | Type | Description |
|-------|------|-------------|
| `device_id` | string | References `devices.device_id` |
| `round_id` | int64 | Round this update belongs to |
| `session_id` | string | Client-generated session correlation ID |
| `payload_hash` | string | Hex-encoded SHA-256 of all uploaded bytes |
| `file_id` | ObjectId | GridFS file ID for the encrypted update bytes |
| `upload_time` | BsonDateTime | Upload timestamp |
| `verified` | bool | `false` until `SubmitReceipt` validates signature + hash |
| `verified_at` | BsonDateTime | Set when verified |
| `size_bytes` | int64 | Total bytes uploaded |

Compound index: `(file_id, device_id, round_id)`

### `receipts`

Tamper-evident HMAC-chained audit log of all verified submissions.

| Field | Type | Description |
|-------|------|-------------|
| `device_id` | string | References `devices.device_id` |
| `round_id` | int64 | Round this receipt covers |
| `payload_hash` | string | Hex-encoded SHA-256 of uploaded bytes |
| `epsilon_spent` | double | Privacy budget consumed by this update |
| `signature` | string | Hex-encoded ECDSA DER signature from device |
| `enc_handle` | string | GridFS ObjectId of the encrypted update |
| `scheme` | string | Encryption scheme identifier |
| `timestamp` | BsonDateTime | Receipt creation time |
| `verified` | bool | Always `true` (only stored if all checks pass) |
| `hmac_chain` | string | HMAC-SHA256 chain link for this receipt |

Index: `(round_id, _id: -1)` for efficient chain tail lookup.

### `global_models`

Stores aggregated global models for download by clients.

| Field | Type | Description |
|-------|------|-------------|
| `round_id` | int64 | Round this model was produced for (unique index) |
| `file_id` | ObjectId | GridFS file ID of the serialised model |
| `model_hash` | string | Hex-encoded SHA-256 of full model bytes |

> **Note:** This collection is read by `DownloadGlobalModel` but is not currently populated by the aggregation pipeline. See [Section 13, Blocker 2](#13-current-blockers).

### GridFS (`fs.files` / `fs.chunks`)

Used by MongoDB's GridFS layer to store large binary objects (encrypted model updates and, in future, aggregated global models). Files are referenced by `ObjectId` throughout the system — client-supplied file paths are never used.

### Relationships

```
devices ──────────────── (1:N) ──────────────── model_updates
   │                                                  │
   │ (device_id)                                      │ (file_id → GridFS)
   │                                                  │
   └──────────────────── (1:N) ──────────────── receipts
                                                      │
                                                      │ (hmac_chain → prev receipt)
                                                      └── self-referential chain

global_models ─────────────────────────────── (file_id → GridFS)
```

---

## 11. Current Reconstruction Status

### Rust Orchestrator

| Component | Status | Notes |
|-----------|--------|-------|
| `main.rs` | ✅ Complete | MongoDB init, indexes, OTP bootstrap, serve |
| `config.rs` | ✅ Complete | |
| `crypto.rs` | ✅ Complete | |
| `errors.rs` | ✅ Complete | |
| `state.rs` | ✅ Complete | Round 1 pre-seeded |
| `round.rs` | ✅ Complete | All structs and state enum |
| `identity.rs` | ✅ Complete | |
| `otp.rs` | ✅ Complete | Rate-limiting, expiry, per-device |
| `receipts.rs` | ✅ Complete | PEM → SPKI → EC point → ECDSA verify |
| `grpc/server.rs` | ✅ Complete | All 7 RPCs implemented with security fixes |
| `ledger.rs` | ⚠ Stub | Appends to flat file only; not integrated with MongoDB or receipts |
| `pubsub.rs` | ⚠ Stub | Logs round info every 10s; no actual notification mechanism |

### Python Client Runtime

| Component | Status | Notes |
|-----------|--------|-------|
| `runtime/pipeline.py` | ✅ Complete | All 6 security fixes applied |
| `runtime/grpc_client.py` | ✅ Complete | mTLS dual-channel |
| `runtime/federated_client.py` | ✅ Complete | daemon / run-once modes |
| `runtime/daemon.py` | ✅ Complete | Idle detection, capture, heartbeat, lock |
| `runtime/tpm_guard.py` | ✅ Complete | Windows + Linux, ECDSA P-256 |
| `runtime/grpc/orchestrator_pb2.py` | ✅ Current | Matches proto definition |
| `runtime/self_destruct.py` | ✅ Complete | |
| `runtime/logging_config.py` | ✅ Complete | MetricsCollector, HealthReporter |
| `runtime/capture.py` | ✅ Complete | |
| `runtime/idle.py` | ✅ Complete | |

### Python Agents

| Component | Status | Notes |
|-----------|--------|-------|
| LDA text pipeline | ✅ Complete | |
| LDA audio pipeline | ✅ Complete | |
| LDA video pipeline | ✅ Complete | |
| Trainer (supervised mode) | ✅ Complete | |
| Trainer (autonomous / RL modes) | ✅ Complete | |
| DP Agent (all 6 mechanisms) | ✅ Mechanically complete | Missing `epsilon_spent` in return dict |
| Encryption Agent (AES-GCM) | ✅ Complete | |
| Encryption Agent (HE-CKKS) | ⚠ Stub | Pyfhel import present; pipeline not wired |
| Encryption Agent (SMPC) | ⚠ Placeholder | No implementation |
| Aggregator (core aggregation) | ✅ Complete | |
| Aggregator → MongoDB write | ✗ Missing | Aggregated model not stored in `global_models` |
| SecureStore | ✅ Complete | |
| Receipt Manager | ✅ Complete | |

### Security Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| Integrity guard (static) | ✅ Complete | |
| Integrity watcher (dynamic) | ✅ Complete | |
| Anti-debug checks | ✅ Complete | |
| Self-destruct | ✅ Complete | |
| TPM seal/unseal (Linux) | ✅ Complete | |
| TPM seal/unseal (Windows) | ✅ Complete | |
| windows_signer (ECDSA sign/pubkey) | ✅ Complete | |
| Installer core | ✅ Complete | |
| TLS certificate generation | ⚠ Manual | `gen_certs.sh` exists; certs not checked in |

---

## 12. Legacy Components

The following components exist in the repository for historical reasons and must not be used in the current system.

### `client_agent/` (entire directory)

This was the original client-side implementation, written against the first version of the proto. It has been superseded by `runtime/pipeline.py`.

**Do not use `client_agent/client_agent.py` because:**

1. **Old proto** — `client_agent/grpc/orchestrator_pb2.py` was generated from an older version of `orchestrator.proto`. It is missing `UpdateChunk`, `UploadAck`, `RoundRequest`, `ModelChunk` messages; missing `EnrollmentRequest*`, `EnrollRequest`, `EnrollResponse` messages; and missing the `UploadUpdate` and `DownloadGlobalModel` RPCs.

2. **`enc_uri` vs `enc_handle`** — The old `Receipt` message has field 6 named `enc_uri` (a local file path). The current server requires field 6 to be `enc_handle` (a GridFS ObjectId). Any receipt submitted from `client_agent.py` will be rejected by the server with `"enc_handle is required"`.

3. **Incompatible signature algorithm** — `client_agent.py` generates an `Ed25519PrivateKey` and signs with `key.sign(msg)`. The server's `receipts.rs` expects ECDSA P-256 DER signatures. An Ed25519 signature will fail `Signature::from_der` immediately.

4. **No `UploadUpdate` call** — `client_agent.py` skips the streaming upload step entirely and submits a receipt with `enc_uri` set to a local file path. The server cannot access client-side file paths; this path was the original security vulnerability that the streaming redesign fixed.

5. **Hardcoded `epsilon_spent=1.0`** — Uses a hardcoded value instead of reading from the DP agent.

**The current client path is:** `runtime/federated_client.py` → `runtime/pipeline.py`

### `RegisterDevice` RPC

The `RegisterDevice(CSR) → Certificate` RPC in the proto is deprecated. It performs no MongoDB write and issues no real certificate. New clients must use `RequestEnrollment` + `EnrollDevice` to obtain a valid certificate.

---

## 13. Current Blockers

The following issues prevent the system from operating end-to-end in production.

---

### Blocker 1 — `dp_agent.py` does not return `epsilon_spent`

**Affected:** `dp_agent/dp_agent.py` → `runtime/pipeline.py`

`DPAgent.process_local_update()` returns:
```python
{ "receipt", "receipt_uri", "update_uri", "l2_norm_before", "l2_norm_after" }
```

The key `epsilon_spent` is absent. `runtime/pipeline.py` falls back to `1.0` with a warning:
```
[pipeline] DP agent did not return epsilon_spent — using fallback 1.0.
```

The real epsilon must be computed by an RDP (Rényi Differential Privacy) accountant based on `clip_norm`, `noise_multiplier`, `num_steps`, and `delta`. Until this is wired in, the system never enforces accurate per-device privacy budgets.

**Fix required:** Integrate an RDP accountant (e.g. `dp_accounting` from Google) into `DPAgent.process_local_update()` and add `"epsilon_spent"` to the return dict.

---

### Blocker 2 — Aggregated model not written to MongoDB `global_models`

**Affected:** `server/aggregator_agent/aggregator.py` → `server/grpc/server.rs:run_aggregation` → `GetRound` → `DownloadGlobalModel`

`aggregator.py` saves the aggregated parameter vector as a local `.npy` file and returns:
```json
{ "aggregated_uri": "file:///absolute/path/aggregated_round_1.npy" }
```

The Rust server stores this path in `round.upload_uri` but never writes a document to the `global_models` MongoDB collection. As a result:

- `GetRound` always returns `global_model_available = false`
- `DownloadGlobalModel` always returns `Status::not_found("no global model available for this round")`
- Clients always initialise from random weights, never from the aggregated global model
- The federated learning feedback loop is broken

**Fix required:** After aggregation, the aggregated model must be serialised, stored in GridFS, its `SHA-256` computed, and a `global_models` document inserted:
```json
{ "round_id": N, "file_id": ObjectId, "model_hash": "<hex>" }
```

---

### Blocker 3 — `run_aggregation` blocks the Tokio runtime thread

**Affected:** `server/grpc/server.rs:run_aggregation`

`run_aggregation` is called synchronously inside an `async` gRPC handler. It calls `Command::new("python3")...wait_with_output()`, which blocks the calling Tokio worker thread for the entire duration of aggregation.

During aggregation, the server cannot process other gRPC requests on that thread. Under load this can starve the runtime.

**Fix required:** Wrap the subprocess call in `tokio::task::spawn_blocking` and `.await` the result.

---

### Blocker 4 — Only Round 1 ever exists; no round progression

**Affected:** `server/grpc/server.rs:get_round`, `state.rs`

`GetRound` unconditionally fetches `state.rounds.get(&1)`. Round 1 is the only round that exists. Once it transitions to `Complete`, clients get `"Collecting" = false` permanently and never participate again.

There is no mechanism to:
- Advance to round N+1 after round N completes
- Notify clients that a new round has started
- Initialise new rounds with updated epsilon budgets or model versions

**Fix required:** Implement a round progression system that creates round N+1 after round N aggregates and makes the global model available.

---

### Blocker 5 — TLS certificates not present

**Affected:** Server startup, all client connections

The certificates required by `config/orchestrator.toml` (`certs/ca.pem`, `certs/ca.key`, `certs/server.pem`, `certs/server.key`) are not checked into the repository. The server will fail to start with `enable_tls=true` until they are generated.

**Fix required:** Run `bash server/orchestration_agent/certs/gen_certs.sh <SERVER_IP>`, then copy `certs/ca.pem` to `installer/runtime/keys/ca.pem` for distribution to clients.

---

### Blocker 6 — `RECEIPT_CHAIN_KEY` not set in production

**Affected:** `server/grpc/server.rs:Service::new`

If `RECEIPT_CHAIN_KEY` is not set, the server generates an ephemeral random 32-byte key. Receipts chained with this key cannot be verified after a server restart because the key changes. The audit trail is functionally broken across restarts.

**Fix required:** Set `RECEIPT_CHAIN_KEY` to a stable hex-encoded 32-byte key in the server's environment. Store it in a secrets manager (Vault, AWS Secrets Manager, etc.) — never in config files or source code.

---

## 14. Future Work

The following items are not currently implemented and represent planned capabilities.

### 14.1 Real RDP Epsilon Accounting

Replace the hardcoded `epsilon_spent=1.0` fallback with a proper Rényi Differential Privacy accountant. The accountant should track the cumulative privacy loss across all training steps in a session and compute a tight epsilon bound per `(delta, noise_multiplier, clip_norm, num_steps)` tuple.

Suggested library: `dp-accounting` (Google). The computed epsilon must be attached to the DP agent's return dict and propagated to the `Receipt` submitted to the server.

### 14.2 Round Lifecycle Management

Implement full round progression:
- After a round completes aggregation, automatically create the next round
- Persist round configuration (epsilon budget, model version, aggregation mode) in MongoDB
- Notify enrolled clients of new rounds (requires implementing `pubsub.rs`)

### 14.3 Global Model Feedback Loop

Complete the aggregation → distribution pipeline:
- Aggregator serialises the aggregated parameter vector back to a state dict
- Store it in MongoDB GridFS
- Insert a `global_models` document
- Clients then receive and apply the global model at the start of each round

### 14.4 Pub-Sub Round Notifications

`pubsub.rs` is currently a stub. A real notification mechanism should push round state changes to subscribed clients (e.g., a `WatchRound` server-streaming RPC), so clients do not need to poll `GetRound`.

### 14.5 Ledger Integration

`ledger.rs` currently appends raw bytes to a flat log file. It should be integrated with the MongoDB `receipts` chain to provide a queryable, verifiable audit trail rather than a write-only flat file.

### 14.6 Homomorphic Encryption (HE-CKKS)

`enc_agent.py` includes Pyfhel imports for CKKS-scheme homomorphic encryption. Completing this would allow the server to aggregate encrypted updates without decrypting them, eliminating the need to trust the aggregator with the master key.

### 14.7 Secure Multi-Party Computation (SMPC)

The `smpc` mode in `enc_agent.py` is a placeholder. Implementing SMPC-based secret sharing would allow the aggregator to combine updates from N parties without any single party (including the server) seeing individual updates.

### 14.8 Multi-Round Epsilon Budget Tracking

The current implementation tracks epsilon per round in memory only. A persistent per-device epsilon budget tracker should accumulate epsilon across all rounds in MongoDB and reject a device's participation once its lifetime budget is exhausted.

### 14.9 Dynamic Aggregation Threshold

The aggregation trigger (`updates.len() >= 3`) is hardcoded in `server.rs`. This should be configurable per round in `orchestrator.toml` or in the `global_models` / round configuration document.

### 14.10 Offline Queue

`runtime/offline_queue.py` exists but its integration into the pipeline is incomplete. When the server is unreachable, the encrypted update and receipt should be persisted to the offline queue and replayed automatically when connectivity is restored.

### 14.11 Windows Signer Completion

`installer/windows_signer/src/main.rs` implements `--sign` and `--pubkey` operations. The `--keygen` flow (generating and persisting a key in the Windows certificate store or TPM) should be completed and tested end-to-end on Windows.

### 14.12 End-to-End Integration Tests

No integration tests currently exist that exercise the full client→server→aggregator→client loop. Tests should cover:
- Enrollment ceremony (B1 + B2)
- Full pipeline run with a real (minimal) model
- Streaming upload with chunk hash verification
- Receipt chain integrity across multiple devices
- Round progression after aggregation threshold
