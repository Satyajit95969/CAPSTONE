# CLAUDE CODE PROJECT CONTEXT
### Multi-Agent Privacy Orchestrated Framework for Secure Multimodal AI

> **Give this file to Claude Code together with the repository.** It is derived exclusively from the official B.E. project report (Black Book + published IEEE paper). Nothing here is invented. Where the source document contradicts itself, both readings are given and the conflict is flagged — **Claude Code must not silently resolve a flagged conflict by editing code.**

---

## Project Overview

A **multi-agent federated learning framework** that acts as a privacy-orchestration layer for multimodal AI. Its only job is to protect the data lifecycle for a downstream multimodal model. The demonstrator task is **binary depression classification (PHQ-8 ≥ 10) on the DAIC-WOZ clinical interview corpus**, fusing audio, video and text.

These are **software agents / microservices**, not LLM agents. There is **no LLM agent loop, no tool calling, no prompt engineering** anywhere in this project. The only transformer components are BERT/MentalBERT encoders plus an optional RAG retrieval layer over embeddings. Do not introduce agentic-LLM patterns.

## Core Objective

Prove that rigorous, multi-layered, *auditable* privacy and security can be co-designed with competitive federated learning performance on sensitive clinical multimodal data — while satisfying:

1. **Data minimisation** — raw user data never leaves the client device in an unprotected or reconstructable form.
2. **Formal privacy** — (ε,δ)-DP via DP-SGD with Rényi DP accounting, enforced by a server-side hard ceiling.
3. **Cryptographic protection** — AES-256-GCM at rest and in transit, HKDF per-agent key isolation, ECDSA-P256 signing, mTLS 1.3.
4. **System-level security** — TPM attestation, SHA3-256 Merkle integrity tree, inotify tamper detection, self-destruct.
5. **Auditability** — HMAC-chained, append-only, tamper-evident receipt ledger producing GDPR/HIPAA evidence.

## Required Architecture

Four hierarchical layers:

```
EDGE LAYER      capture (FFmpeg + DirectShow/V4L2/AVFoundation), VAD segmentation,
                silence-placeholder fallback, FFmpeg SHA-256 verified at install

CLIENT LAYER    Local Data Agent → Trainer Agent → DP Agent → Encryption Agent
                + Runtime Security Layer (runtime_guard, IntegrityWatcher, canaries)
                + SecureStore + Receipt Manager
                (RAG LAYER, optional: Vector Index Manager ↔ Vector Database)

SERVER LAYER    Orchestration Agent (Rust/Tokio/Tonic gRPC)
                Aggregator Agent (Python)
                Key Management Agent (KMS/HSM)
                Audit Agent
                MongoDB (+GridFS)

ARTIFACTS       Model Registry (versioned) · Object Store (S3/MinIO, planned)
                · Ledger DB (QLDB / Hyperledger Fabric, optional)
```

**Hard architectural rules — do not violate:**
1. **Only the Encryption Agent has external network egress.** All other client agents use internal event-driven message queues.
2. **Port 50051 = enrollment (server-TLS only). Port 50052 = operational (full mTLS, client certificate mandatory).** Never merge them; the separation exists to prevent enrollment RPC replay.
3. **The server never decrypts individual client updates.** Aggregation happens over Bonawitz-masked / encrypted representations.
4. **Receipts are computed over ciphertext, never over plaintext**, to keep PII out of the audit subsystem.
5. **No plaintext gradient ever leaves the client device.**
6. **Clip before noise. Redact before embed. Blur before feature-extract.**

## Required Components

| Component | Must exist | Key responsibilities |
| --- | --- | --- |
| Local Data Agent | ✔ | PII redaction (spaCy NER: PERSON/GPE/ORG + regex), face detect + Gaussian blur, eGeMAPS/OpenFace/MentalBERT feature extraction, session timestamp alignment, AES-GCM encryption, manifest + HMAC receipts, zero plaintext buffers |
| SecureStore | ✔ | AES-256-GCM, HKDF-SHA256 per-agent/per-context keys, binary format `version‖agent‖ctx‖12-byte nonce‖ciphertext` + IV/tag/checksum, path-traversal prevention, header bounds checks |
| Receipt Manager | ✔ | Signed JSON receipts, nonce freshness/replay prevention, HMAC chaining |
| Trainer Agent | ✔ | HMAC verify → decrypt → warm-start → local train (autonomous/supervised/reinforcement) → metrics → explainability → delta with safety bounds |
| DP Agent | ✔ | Clip to C, add noise (6 mechanisms), RDP→(ε,δ) accounting, `epsilon_spent`, DP receipt, comparison CSVs + plots |
| Encryption Agent | ✔ | Modular schemes (AES-GCM default / KMS envelope mock / HE-CKKS prototype), chunked upload with per-chunk SHA-256, offline encrypted retry queue, **sole external talker** |
| Runtime Security Layer | ✔ | TPM verification, debugger absence, SHA3-256 Merkle over `bin/ runtime/ agents/ core/ installer/security/`, inotify + periodic verify, canaries, `max_violations=1` self-destruct |
| Orchestration Agent | ✔ | Two-phase TPM enrollment (6-digit OTP, 600 s, 5 attempts), CSR signing, round scheduling, model distribution, ε ceiling enforcement, manifests + audit logs, `max_concurrent_sessions=4` |
| Aggregator Agent | ✔ | ECDSA verification, mask removal, 4 strategies × 5 FL algorithms, parameter-by-parameter aggregation, re-encrypt + GridFS write |
| Key Management Agent | ✔ | Symmetric/asymmetric key generation + rotation, HKDF session keys, root-of-trust, TPM-sealed/DPAPI master key |
| Audit Agent | ✔ | Receipt collection, HMAC chain verification, ε/δ + key-usage + encryption validation, audit reports, GDPR dashboard, DP certificate |
| Vector Index Manager + Vector DB | optional | RAG modes `base`/`rag`/`vector_rag`, `--rag-k`, retrieval latency + top-k metrics |
| Attack testbeds | ✔ | Membership inference, gradient inversion, poisoning/label-flip, BYPASS-1..10 integrity attacks |
| Evaluation suite | ✔ | Seeds=42 propagated, σ/C/E sweeps, JSON logs, hashed+timestamped artifacts, `MetricsCollector`, `HealthReporter`, `config_validator.py` |

## Required Technologies

**Required by the document:** Python (3.10/3.11 — both stated), PyTorch, HuggingFace Transformers + **MentalBERT** (fallback `bert-base-uncased`), scikit-learn, NumPy/Pandas/SciPy, Matplotlib, **openSMILE (eGeMAPS v02)**, librosa, torchaudio, WebRTC VAD, **OpenFace 2.0**, OpenCV (Haar cascade), **FFmpeg**, **spaCy**, Whisper or HF ASR, **AES-256-GCM / HKDF-SHA256 / HMAC-SHA256 / ECDSA-P256 / SHA-256 / SHA3-256** (`cryptography`, PyCryptodome), **TPM 2.0** (tpm2-tools) / Windows CNG + DPAPI, **Rust** (Tokio, Tonic gRPC; `windows_signer.rs`), **gRPC + TLS 1.3 + mTLS + cert pinning**, **MongoDB + GridFS**, **Parquet/PyArrow**, **FastAPI + Uvicorn**, Docker, Git.

**Optional / prototype / planned per the document:** TensorFlow (alternative to PyTorch), Flask (alternative to FastAPI), Opacus (alternative to the custom DP module), HE-CKKS (prototype), BFV/Shamir/SPDZ (named in the Abstract only), AWS KMS (**mock**), S3/MinIO (**planned**), QLDB/Hyperledger Fabric (**optional**), vector database (**product never named**).

**Not specified:** frontend framework, message-broker product, CI/CD platform, deployment target.

## Core User Flows

**Flow 1 — Device onboarding (once).** Installer verifies third-party binary hashes → provisions ECDSA P-256 key in TPM (PCRs 0/2/4/7) → writes SHA3-256 baseline with a one-time token, TPM-signs it, sets it immutable, destroys the token → device sends TPM pubkey + CSR to :50051 → orchestrator issues OTP (6-digit, 600 s, 5 attempts) → CSR signed → client certificate issued.

**Flow 2 — Federated training round.** `runtime_guard` passes → mTLS connect on :50052 → receive global model → LDA preprocess+encrypt → Trainer warm-start + local train → DP clip+noise+account → assert `Σεᵢ ≤ ε_max` → Encryption Agent AES-GCM + chunked upload (1 MB, per-chunk SHA-256) → ECDSA-signed receipt with real payload hash / real ε / real GridFS handle → `SubmitReceipt` → Aggregator verifies signatures, aggregates without decrypting individuals, applies strategy, writes `M_{t+1}` → Audit Agent appends chained receipt → repeat.

**Flow 3 — Research/benchmark run.**
```bash
python3 create_dp_comparison.py --lda-mode session --input-type text_dir \
  --input-path ./sample_texts --modes base rag vector_rag \
  --epochs 20 --lr 1e-3 --use-bert --device cuda --rag-k 3
```
Produces `secure_store/`, `trainer_outputs/`, `explain_logs/`, `dp_noise_mechanism_comparison_*.csv`, `dp_comparison_all_modes.csv`, `plots/`.

**Flow 4 — Audit/compliance.** Audit Agent gathers receipts → verifies HMAC authenticity + chain linkage → validates ε/δ, key usage, encryption → emits audit report, GDPR dashboard data, DP certificate with `meets_ε≤8` verdict.

**Flow 5 — Client entrypoint.** `python .\bin\federated-client --run-once` → logging init → integrity watcher start (interval 120 s, `max_violations=1`) → runtime guard → prompt for server address `host:port`.

## Data Flow

```
raw A/V/text (device only)
  → redacted/blurred features (in-memory, zeroed after)
  → AES-256-GCM encrypted Parquet in SecureStore + manifest + HMAC receipt
  → decrypt-to-secure-memory → local training → Δθ (in-process only)
  → clip(C) + noise(σ) → (ε,δ)-bounded Δθ̃ + epsilon_spent
  → AES-256-GCM ciphertext Cᵢ + ECDSA receipt
  → mTLS 1.3 :50052, 1 MB chunks, per-chunk SHA-256
  → MongoDB GridFS
  → masked aggregation (server never sees individual Δθ̃ᵢ)
  → M_{t+1} → Model Registry → redistributed to clients
  ⊥ every step emits a receipt → HMAC-chained append-only ledger
```

**Reference CSV schema (fixed):**
`mechanism, noise_multiplier, accuracy, precision, recall, f1, mae, distortion_ratio, silhouette_score, rag_mean_latency`

**Receipt fields:** `device_id, round_id, payload_hash (SHA-256), epsilon_spent, enc_handle (GridFS ObjectId), signature (ECDSA-P256 over device_id‖round_id_BE8‖payload_hash), hmac (includes predecessor chain value), timestamp`.

## AI/ML Flow

**Feature extraction:**
- Audio: openSMILE eGeMAPS v02 → 88 LLDs → mean+std pooling → **176-dim** (⚠ Ch.7 says 78-dim)
- Video: OpenFace 2.0 → 17 AUs + gaze + head pose → mean+std pooling → **70-dim** (⚠ Ch.7 says 98-dim)
- Text: MentalBERT `[CLS]` 768-dim per **participant** utterance → session mean-pool → **768-dim**

**Model — DepressionNet (late fusion):** `ŷ = f_fusion(f_a(a), f_v(v), f_t(t))`, each branch `ReLU(W₂·Drop(BN(ReLU(W₁x))))` → 64-dim → concat **192** → fusion head **192→128→2** → cross-entropy, dropout 0.25. Labels binarised at **PHQ-8 ≥ 10**. ⚠ Ch.6 states `fusion_in = 768+128+128 = 1024` — a different geometry. **Read the code; do not "fix" to match a table.**

**DP:** flatten Δw → ℓ₂ → if ℓ₂>C: `g ← g·C/(ℓ₂+ε_num)` → `g̃ ← g + N(0, σ²C²I)` → unflatten → `ε ← RDP_to_DP(σ,C,δ)`.
RDP: `ε_RDP(α)=α/(2σ²)`, subsample `q=B/n`, compose over T, `ε(δ)=min_{α≥2}[ε_RDP(α)+log(1/δ)/(α−1)]`, α∈[2,256].
Mechanisms: gaussian (default) · laplace · uniform · exponential · student_t · none.

**FL algorithms:** FedAvg · FedProx (µ=0.01) · FedAdam (η_s=1e-3, β1=0.9, β2=0.999) · FedYogi (η_s=1e-2, τ=1e-3) · SCAFFOLD (control variates, η=1e-3 SGD).
**Aggregations:** weighted mean · trimmed mean (β=0.1) · coordinate-wise median · Krum (f=⌊K/5⌋, **requires K≥5**).

**Hyperparameters — two conflicting sets in the source:** E=5, B=8 (agreed); LR 1e-3 vs 5e-4; σ default 0.5 vs 1.1; rounds 20 vs 30; fusion hidden 256 vs 64; dropout 0.2 vs 0.25; classification threshold 0.4; δ 1e-5 vs 1/N_total. **The config file is ground truth.**

**Documented FIX-IDs (bugs already found and fixed — verify they are still fixed):** FIX-6 (noise scaled by batch size) · FIX-7 (seed propagation for train/test alignment) · FIX-10 (warm-start support) · FIX-13 (selective encoder reinit) · FIX-14 (hospital-style client grouping, fixes single-class gradient collapse) · FIX-15 (Adam LR for heterogeneous data) · FIX-16 (threshold 0.4 for minority class) · FIX-SS-1 (path traversal) · FIX-SS-3 (header bounds) · FIX-CRYPTO-3 (HKDF per-agent isolation) · FIX-DP-1 (real ε accounting) · FIX-PIPELINE-2/3/4/5 (real GridFS handle / real ε / real payload hash / per-chunk SHA-256) · FIX-QUEUE-1 (encrypted offline queue).

## Database / Data Model

**Entities (ER):** `USER` (device owner/participant) → `LOCAL_DATA_AGENT` → `RECEIPT_MANAGER`; `LOCAL_DATA_AGENT` → `SECURE_STORE` ← `RECEIPT_MANAGER`. **No attributes/keys/cardinalities are specified anywhere in the source.**

**MongoDB collections:** device records (device ID, TPM public key, issued cert, enrollment state) · model updates in **GridFS** · receipts with HMAC chaining · global model documents.

**On-disk layout:**
```
~/.federated/{logs/, models/mentalbert/, data/secure_store/master.key}
secure_store/  trainer_outputs/{local_probe_base.pt, metrics.json}
explain_logs/<mode>_probe_explain_<timestamp>.txt  plots/
dp_noise_mechanism_comparison_*.csv  dp_comparison_all_modes.csv
manifests.csv  audit_logs.csv  baseline.sha256  .canaries/*.dat  local_config.yaml
```

**Dataset:** DAIC-WOZ (written "DIAC-WOZ" once — typo). Ch.7: 40 participants (11 depressed / 29 not), 4 stratified "hospital" clients, 10 held out (5/class). Paper: each patient is a client, 20% held out. ⚠ Mutually exclusive. **Never commit dataset files.**

## API / Service Requirements

**gRPC (Rust orchestrator, Tokio + Tonic):**
- `:50051` — enrollment, server-TLS only. Accepts TPM pubkey + CSR, OTP issue/verify, returns signed client certificate.
- `:50052` — operational, full mTLS, client cert mandatory, TLS 1.3, certificate pinning.
- Known RPCs: global-model download · streamed update upload `_stream_update(stub, device_id, round_id, update_uri) → (server_handle, payload_hash)` · `SubmitReceipt(receipt) → ack.ok`.
- **No `.proto` is given in the document** — read it from the repo.

**FastAPI (client agents):** `PreprocessRequest(mode, inputs) → {artifact_manifest, ...}`. **No routes/verbs/status codes specified.**

**Internal calls:** `LDA.preprocess()` · `Trainer.train()` / `trainer_orchestrate(input_path, mode, global_model_path)` · `DPAgent.process_update(delta)` / `process_local_update(uri, noise_multiplier)` · `EncAgent.encrypt_receipt()` · `write_baseline()` (one-time) · `_tpm_verify_baseline()` · `CanaryMonitor.check()` · `_path_is_within()` · `verify_cert_pin()` · `_rdp_to_dp()` · `trimmed_mean_aggregate()` · `build_hospital_clients()` · `_load_partial_checkpoint()` · `_selective_reinit_encoders()`.

**Named source files (from Ch.6/7 — expect these in the repo):** `integrity.py` · `military_security.py` · `tpm_attestation.py` · `tpm_seal.py` · `windows_signer.rs` · `centralized_secure_store.py` · `create_dp_comparison.py` · `config_validator.py` · `bin/federated-client` · `local_config.yaml`.

## Security Requirements

Full list in the companion analysis (§17). The non-negotiables:

- AES-256-GCM everywhere at rest; HKDF-SHA256 per-agent/per-context key isolation via distinct `info` fields.
- TLS 1.3 mutual auth on :50052 + certificate pinning; **insecure fallback is dev-only and must be gated**.
- Hardware-backed ECDSA P-256 device identity in TPM; PCR binding 0/2/4/7; Windows non-exportable CNG key.
- OTP enrollment: 6 digits, 600 s expiry, 5-attempt rate limit per device.
- Master key TPM-sealed (Linux) / DPAPI (Windows) — **operators must relocate it to TPM/HSM before production**.
- SHA3-256 Merkle baseline: write-once (one-time token + `INSTALL_LOCK`), TPM-signed, immutable (`chattr +i`).
- IntegrityWatcher: inotify sub-100 ms **plus** randomised periodic verification; `max_violations=1` → self-destruct.
- Canary files, debugger-absence check, `_path_is_within()` realpath containment, timing-safe `hmac.compare_digest`, receipt nonce freshness.
- ECDSA signature verification on every submission; unsigned/invalid rejected.
- HMAC-chained append-only receipt ledger; insertion/deletion/reorder detectable.
- Server-side hard ε ceiling; round aborted on violation.
- Byzantine defense: trimmed mean / median / Krum + anomaly detection + client reputation.
- Supply chain: SHA-256 verification of all third-party binaries at install; SLSA provenance in CI.

**Known unresolved security items, stated by the document itself — verify, report, do not paper over:**
1. FFmpeg SHA-256 hash for Windows is **a placeholder** — "a critical, unresolved security item" and "a supply-chain attack vector".
2. Master key default path is a plain filesystem location.
3. Research summary reports **`sig_ok=False` across 160 receipts** while `chain_ok=True` — ECDSA verification is failing.
4. **ε ≤ 8 (HIPAA) is `VIOLATED`** at the default noise multiplier; the Black Book NFR of **ε ≤ 2.0** is far out of reach at measured values (37.96–149.76).

## Important Constraints — do NOT change or violate

1. **Do not move network egress out of the Encryption Agent** or add a second external talker on the client.
2. **Do not merge ports 50051 and 50052**, and do not relax the mandatory client certificate on 50052.
3. **Do not add per-client decryption in the aggregator.** If it already exists, report it as a D-SEC deviation rather than deleting it silently.
4. **Do not reorder** clip→noise, redact→embed, or blur→feature-extract.
5. **Do not compute receipts over plaintext.**
6. **Do not replace real ε accounting with a constant**, and do not change ε formulas to make targets pass.
7. **Do not use `strict=False`** for checkpoint loading; keep name+shape filtering with selective reinit.
8. **Do not weaken** `max_violations=1`, the write-once baseline token, or the immutability flag.
9. **Do not remove `none` from the DP mechanism list** — it is the utility baseline for every results table.
10. **Do not change the DP comparison CSV header** — downstream plots and tables depend on it.
11. **Do not "harmonise" conflicting documented values** (dimensions, hyperparameters, ε) by editing code to match one table. Report the conflict.
12. **Do not commit DAIC-WOZ data, keys, certificates, `master.key`, or `.canaries/`.**
13. **Do not rewrite working subsystems.** First task is understanding and auditing, not refactoring.
14. **Do not introduce LLM-agent patterns** (tool loops, prompt chains) — "agent" here means microservice.
15. **Do not delete the research harness** (`create_dp_comparison.py`) even if a production orchestrator exists; the document names it as the current orchestration logic.

## Known Ambiguities (source document contradicts itself)

| # | Conflict | Resolution guidance |
| --- | --- | --- |
| 1 | §4.4 Alg.3 says the server **decrypts** each `G'ᵢ`; Paper §III-C + Alg.4 say the server **never decrypts** individual updates | Treat Paper/Alg.4 as normative. Report which the code does. **Highest severity.** |
| 2 | Abstract promises CKKS/BFV, Shamir, SPDZ, Bonawitz secure aggregation; Ch.4 Future Extensions and Ch.8.3 list HE/SMPC/secure-sum aggregation as **not yet done** | Determine what exists; report the gap |
| 3 | Backend = FastAPI/Flask (Ch.3) vs Rust/Tonic gRPC orchestrator (Paper) | Likely both, at different boundaries |
| 4 | Orchestrator = `create_dp_comparison.py` (Ch.4) vs a Rust service (Paper) | Likely two orchestration paths; audit both |
| 5 | Audio 176 vs 78; visual 70 vs 98; fusion 192 vs 1024 | Read actual tensor shapes |
| 6 | LR 1e-3 vs 5e-4; σ 0.5 vs 1.1; rounds 20 vs 30; hidden 256 vs 64; dropout 0.2 vs 0.25 | Config file is ground truth |
| 7 | ε = 56.32 vs 149.76/62.75/39.85 vs 84.9453 vs certificate 37.9592 | Re-verify the accountant |
| 8 | Privacy target ε ≤ 2.0 (NFR) vs ε ≤ 8 (HIPAA) vs measured 37–150 | A stated requirement is currently failing |
| 9 | 4 hospital clients vs 40 patient-clients; Krum needs K≥5 yet Krum results are reported with 4 clients | Read the partitioning function |
| 10 | Test split "10 patients" (25%) vs "20% held out" | Read the split code |
| 11 | Intra-client transport = "message queues" vs "APIs or secure data stores" vs "mTLS/pubsub" | Read the code; **no broker is named — do not assume one** |
| 12 | Python 3.11 vs 3.10 | Use the testbed value (3.10) for reproduction |
| 13 | TPM signing +45 ms vs 5–15 ms | Benchmark artifact only |
| 14 | DP mechanism conclusions reverse between Fig.19–22 and Fig.23 (Gaussian best vs Exponential best) | Re-derive from CSVs; narrative text is unreliable |
| 15 | Poisoning results identical across clean/poisoned/defended (0.667) and hypothesis "NOT SUPPORTED" | Verify the attack is actually applied |
| 16 | Subsampling amplification gain reported as 0.0% | Verify subsampled RDP is used |
| 17 | Vector database product never named | Read the code; **do not invent one** |
| 18 | Frontend required by NFR-09 + Fig.09 dashboard, but the web audit dashboard is a Future Extension | Do not assume a frontend must exist |

---

# CLAUDE CODE — INITIAL PROJECT AUDIT

**Your first task is to understand and audit, not to rewrite.** Preserve working functionality. Change code only when (a) it is required to satisfy a documented requirement, or (b) it fixes a *verified* defect. Produce findings with file:line evidence.

### Step 1 — Repository structure
Map the full tree (excluding `node_modules`, `.venv`, data). Identify top-level modules and how they map onto the four layers (Edge / Client / Server / Artifacts). Note anything that maps to nothing in the documented architecture, and any documented component with no corresponding directory.

### Step 2 — Technologies actually used
Read `requirements.txt` / `pyproject.toml` / `Cargo.toml` / `package.json` / `Dockerfile` / lockfiles. Produce an actual-vs-documented technology table (§ "Required Technologies"). Flag: missing required deps, unexpected heavyweight deps, and version drift (esp. Python 3.10 vs 3.11, PyTorch — requirements.txt:239 pins `torch==2.8.0`, verified installed as `2.8.0+cu128` in `.venv` as of 2026-08-26; this line previously said "2.6.0", which was stale and did not match the repo).

### Step 3 — Layer identification
Locate concretely: capture module, LDA, SecureStore, Receipt Manager, Trainer, DP Agent, Encryption Agent, runtime security, orchestrator (Rust and/or `create_dp_comparison.py`), aggregator, KMA, audit agent, vector index. For each, record path + main entry symbol.

### Step 4 — Module inventory
For every module: purpose, public entry points, inputs, outputs, dependencies, and which of the §20 requirement IDs it is responsible for.

### Step 5 — Trace the main user workflows
Trace end-to-end, by reading call chains (not assuming): (a) device enrollment, (b) a training round, (c) the `create_dp_comparison.py` benchmark run, (d) audit/compliance report generation, (e) `bin/federated-client --run-once`. Record the actual call order and flag any deviation from the documented order — especially **clip-before-noise**, **redact-before-embed**, **blur-before-OpenFace**, **encrypt-before-send**, **HMAC-over-ciphertext**.

### Step 6 — Trace data flows
For each hop in the §"Data Flow" diagram, confirm the protection applied. Specifically verify: nothing raw crosses the process/network boundary; plaintext buffers are zeroed; only the Encryption Agent imports network libraries.

### Step 7 — Configuration & environment
Read `local_config.yaml`, `config_validator.py`, `.env*`, cert/CA material paths, Docker/compose files. Record: the configured `ε_max`, default `noise_multiplier`, `clip_norm`, rounds, LR, hidden dims, threshold, `max_concurrent_sessions`, watcher interval, `max_violations`, master key path, dev-insecure-transport flag. **Report any secret, key, certificate or credential committed to the repo as a P0 finding.**

### Step 8 — Database schemas & models
Enumerate MongoDB collections/documents actually written, GridFS usage, indexes, and the receipt document shape. Compare to §"Database / Data Model". Note that the source specifies no field-level schema — document what the code defines.

### Step 9 — APIs & integrations
Extract the `.proto` (if present) and every FastAPI route. Verify dual-port binding, TLS configs, mandatory client cert on 50052, cert pinning, and that OTP expiry/attempt limits match (600 s / 5). List all external integrations and their real vs mock status (AWS KMS, S3/MinIO, ledger DB, vector DB, HuggingFace).

### Step 10 — AI/ML implementation
Verify: actual feature dimensions; DepressionNet layer geometry; participant-turn filtering; the six DP mechanisms; the clip/noise order and `ε_num` guard; batch-size noise scaling; that `_rdp_to_dp()` is real and data-dependent (**not a constant**); the five FL algorithms' update rules; the four aggregation strategies; Krum's K≥5 guard; partial checkpoint loading + selective reinit; stratified client grouping; seed propagation; threshold 0.4.

### Step 11 — Requirement comparison
Walk every requirement **R001–R082** in the companion analysis. Assign ✅ / 🟡 / ❌ / ⚠️ / ❓ with file:line evidence. Do not mark ✅ without evidence that the code path is reachable from a real flow.

### Step 12 — Missing functionality
List every ❌. Rank by whether it breaks a core claim (data minimisation, formal privacy, server blindness, auditability) versus a peripheral feature.

### Step 13 — Partial functionality
List every 🟡 with the exact unmet sub-clause (missing mode, hardcoded value, no error handling, only reachable from a research script, etc.).

### Step 14 — Architectural deviations
List every ⚠️. Classify D-SEC / D-ARCH / D-SPEC / D-DOC. For D-DOC (code right, document wrong), recommend a documentation correction rather than a code change.

### Step 15 — Bugs & suspicious implementations
Prioritise these documented red flags first:
- **`sig_ok=False` on 160 receipts** — check the ECDSA signing byte layout (`device_id ‖ round_id_BE8 ‖ payload_hash`), key provenance, and the verification side.
- Aggregator performing per-client decryption (contradicts server-blindness).
- ε returned as a constant / not varying with σ, q, T; amplification gain of 0.0%.
- Poisoning attack producing identical clean/poisoned/defended metrics — attack may not be applied.
- Placeholder values in receipts (`payload_hash`, `epsilon_spent`, `enc_handle`).
- FFmpeg Windows hash placeholder.
- `strict=False` checkpoint loads; blanket encoder reinit.
- `==` comparisons on MACs/tokens; `startswith` path checks instead of realpath containment.
- Any `verify=False`, disabled TLS verification, or ungated insecure fallback.
- Krum invoked with K<5.

### Step 16 — Dead / unused code
Identify unreferenced modules, duplicated orchestration paths, and abandoned experiment scripts. **Report only — do not delete** without confirming they are not the documented research harness.

### Step 17 — Security & configuration issues
Committed secrets/keys/certs; master key path; permissive CORS; debug mode; unpinned dependencies; missing input validation at API boundaries; overly broad file permissions; logs that could contain PII.

### Step 18 — Missing error handling
Check every row of the failure-flow table in the companion analysis (§12.2): capture fallback, parser robustness, transformers fallback, checkpoint mismatch, ε-ceiling abort, unsigned rejection, chunk corruption, offline queue, TLS failure, aggregator crash recovery, Byzantine inputs, integrity violations, config validation, explainability fallback.

### Step 19 — Testing gaps
Confirm existence and coverage of the five Ch.6 categories: unit (data loading, model/weight transfer, DP mechanisms, crypto primitives), integration (LDA→Trainer→DP→Encryption→Upload→Aggregation), system (network simulation, scalability, fault injection), security penetration (BYPASS-1..10, ATTACK-FS5, ATTACK-NET1), performance benchmarking. Flag any Ch.6 claim with no corresponding test.

### Step 20 — Final implementation audit report
Produce a report with:
1. **Executive summary** — what exists, what doesn't, top 5 risks.
2. **Requirement matrix** — R001–R082 with status + evidence.
3. **Security findings** — P0/P1/P2, each with impact and remediation.
4. **Architecture deviation register** — D-SEC/D-ARCH/D-SPEC/D-DOC.
5. **Reproducibility assessment** — can the documented results (Tables 13–16, Figs. 17–26) be regenerated? Which cannot, and why?
6. **Prioritised remediation plan** — smallest changes that close the highest-severity documented gaps first, explicitly preserving working functionality.
7. **Documentation corrections** — where the report is wrong and the code is right.

**Do not begin remediation until the audit report has been reviewed.**

---

# Master Project Completeness Checklist

### Architecture
- [ ] Four layers present and separable: Edge / Client / Server / Artifacts
- [ ] Each of the 9+ agents is an independently deployable module (NFR-08)
- [ ] Only the Encryption Agent has external network egress
- [ ] Intra-client communication uses the documented event-driven mechanism
- [ ] Client and server codebases are cleanly separated
- [ ] Optional RAG layer is genuinely optional (system runs in `base` mode without it)

### Frontend
- [ ] Determine whether any UI exists (NFR-09 requires one; the web audit dashboard is a Future Extension)
- [ ] If a GDPR/audit dashboard exists: shows receipts, ε, key usage, exceptions, violations
- [ ] If none exists: record as ❌ against NFR-09 / Fig.09, not as an assumed gap to fill

### Backend
- [ ] FastAPI/Uvicorn agent APIs present with request models
- [ ] Rust/Tonic gRPC orchestrator present, or its absence recorded
- [ ] Dual-port binding: 50051 (server TLS) and 50052 (mTLS, client cert mandatory)
- [ ] `create_dp_comparison.py` research harness present and runnable
- [ ] `bin/federated-client` entrypoint with `--run-once`
- [ ] `max_concurrent_sessions = 4` enforced
- [ ] Server-side ε ceiling assertion aborts the round

### Database
- [ ] MongoDB connection + device records collection
- [ ] GridFS storage for client updates and global models
- [ ] Receipts collection with HMAC chain values
- [ ] Global model documents / Model Registry versioning
- [ ] Encrypted Parquet artifacts + manifests (URI, timestamp, checksum)
- [ ] Object store (S3/MinIO) — real or documented as planned
- [ ] Ledger DB (QLDB/Hyperledger) — real or documented as optional/absent

### Authentication
- [ ] Two-phase TPM enrollment implemented
- [ ] ECDSA P-256 key provisioned in TPM (Linux tpm2-tools, PCRs 0/2/4/7)
- [ ] Windows path: `windows_signer.exe`, CNG NCrypt, non-exportable key
- [ ] OTP: 6 digits, 600 s expiry, 5-attempt rate limit per device
- [ ] CSR signed by CA; client certificate issued and required on 50052
- [ ] Certificate pinning enforced; insecure fallback gated to dev only
- [ ] (Record: no human-user authentication is specified in the source)

### APIs
- [ ] `.proto` present and matching the documented RPCs
- [ ] Streamed upload: 1 MB chunks with per-chunk SHA-256
- [ ] `SubmitReceipt` returns an ack and is verified server-side
- [ ] Payload hash cross-verified against the streaming-upload hash
- [ ] FastAPI routes documented (source specifies none — record what exists)

### AI/ML
- [ ] Audio pipeline → eGeMAPS v02, 88 LLDs, mean+std pooled (record actual dim)
- [ ] Video pipeline → OpenFace 17 AUs + gaze + pose, mean+std pooled (record actual dim)
- [ ] Text pipeline → MentalBERT `[CLS]` 768-dim, participant turns only, session mean-pooled
- [ ] MentalBERT fallback to `bert-base-uncased` works
- [ ] DepressionNet late-fusion geometry matches the code (record which of 192 / 1024 is real)
- [ ] Labels binarised at PHQ-8 ≥ 10; classification threshold 0.4
- [ ] Five FL algorithms implemented with correct update rules
- [ ] Four aggregation strategies implemented; Krum guards K≥5
- [ ] Six DP mechanisms including `none`
- [ ] Clip-before-noise with `ε_num` guard; noise scaled by batch size
- [ ] Real RDP accountant over α∈[2,256]; ε varies with σ, q, T
- [ ] Warm-start + partial checkpoint load + selective reinit (no `strict=False`)
- [ ] Delta safety bounds enforced before encryption
- [ ] Stratified client grouping guarantees mixed labels per client
- [ ] Seeds (42) propagated through dataloaders/workers

### Agents
- [ ] Local Data Agent: redaction, blur, extraction, session alignment, encryption, receipts, manifest, buffer zeroing
- [ ] Trainer Agent: three modes, HMAC verify, metrics, explainability, delta
- [ ] DP Agent: mechanisms, accounting, receipts, CSVs, plots
- [ ] Encryption Agent: three schemes, chunked upload, offline encrypted queue
- [ ] Runtime Security Layer: guard, Merkle tree, inotify + periodic, canaries, self-destruct
- [ ] Orchestration Agent: enrollment, rounds, policy, manifests
- [ ] Aggregator Agent: signature verification, mask removal, strategies, memory-safe aggregation
- [ ] Key Management Agent: generation, rotation, HKDF, root-of-trust, sealed master key
- [ ] Audit Agent: chain verification, ε/δ validation, reports, DP certificate

### Data Processing
- [ ] PII redaction covers PERSON, GPE, ORG (+ regex)
- [ ] Face blur applied before OpenFace
- [ ] Session Processor aligns modalities by timestamp
- [ ] `load_patient()` robust to `,` `;` `\t`, missing features, empty dirs, odd IDs
- [ ] Encrypted Parquet + manifest emitted per run
- [ ] DP comparison CSV header exactly as specified
- [ ] Plots generated to `plots/`

### Security
- [ ] AES-256-GCM at rest; HKDF per-agent/per-context keys with distinct `info`
- [ ] Encrypted file header format + bounds checking
- [ ] Receipts computed over ciphertext only
- [ ] ECDSA-P256 receipt signatures verify (**investigate `sig_ok=False`**)
- [ ] HMAC chain detects insertion/deletion/reordering
- [ ] SHA3-256 Merkle baseline: write-once token destroyed, TPM-signed, immutable
- [ ] IntegrityWatcher: inotify + randomised periodic; `max_violations=1`
- [ ] Canary monitoring; debugger-absence check
- [ ] `_path_is_within()` realpath containment; symlink escape rejected
- [ ] Timing-safe digest comparison everywhere
- [ ] Nonce freshness / replay prevention on receipts
- [ ] Byzantine anomaly detection + client reputation
- [ ] Third-party binary hashes verified — **FFmpeg Windows hash is NOT a placeholder**
- [ ] Master key path is TPM/HSM-backed (not a plain file) in production config
- [ ] No secrets, keys, certs or dataset files committed

### Error Handling
- [ ] Silence placeholder on capture failure
- [ ] Graceful `transformers` / MentalBERT fallbacks
- [ ] Checkpoint shape-mismatch handling
- [ ] ε-ceiling violation aborts the round
- [ ] Unsigned/invalid submissions rejected
- [ ] Chunk corruption detected
- [ ] Offline queue retries with exponential backoff, encrypted at rest
- [ ] TLS/cert-pin failures rejected
- [ ] Aggregator crash → state reset to `Collecting`
- [ ] Byzantine/NaN inputs absorbed
- [ ] Config validation at startup
- [ ] Explainability fallback text logs

### Testing
- [ ] Unit: data loading, embedder fallback, dataset grouping, weight transfer, DP mechanisms, RDP conversion, SecureStore, receipts, TPM integration
- [ ] Integration: LDA → Trainer → DP → Encryption → Upload → Receipt → Aggregation
- [ ] System: network simulation (TLS failures, 100–500 ms RTT), scalability, fault injection
- [ ] Security penetration: BYPASS-1..10, ATTACK-FS5, ATTACK-NET1 — all mitigated
- [ ] Performance benchmarks reproduce Tables 11 and 16
- [ ] Attack testbeds reproduce MIA / gradient-inversion / poisoning numbers

### Deployment
- [ ] Dockerfile(s) build and run
- [ ] Installer performs security provisioning (baseline, TPM key, binary hashes)
- [ ] CI exists with SLSA provenance guidance
- [ ] Cross-platform: Ubuntu 22.04 LTS and Windows 11
- [ ] (Record: no cloud/k8s/IaC target is specified in the source)

### Documentation
- [ ] README covering setup, enrollment, running a round, running the benchmark
- [ ] Configuration reference for `local_config.yaml`
- [ ] Security operator guide (master key relocation, FFmpeg hash population)
- [ ] Architecture doc matching the four-layer design
- [ ] Results reproduction instructions
- [ ] Documented-vs-implemented deviation register

### Project-Specific Requirements
- [ ] Raw data never leaves the client (verified by code path, not assertion)
- [ ] Server never decrypts individual client updates (verified by absence of per-client decrypt)
- [ ] Bonawitz masking with dropout resilience
- [ ] Privacy budget target reconciled: ε ≤ 2.0 (NFR) vs ε ≤ 8 (HIPAA) vs measured 37–150
- [ ] All four Abstract deliverables present: agent microservice codebase · reproducible evaluation suite · attack testbeds · compliance evidence generation
- [ ] Explainability artifacts produced every run
- [ ] RAG modes `base` / `rag` / `vector_rag` all runnable
- [ ] GDPR privacy-by-design principles demonstrable: data minimisation, purpose limitation, integrity & confidentiality, accountability, transparency, user control
- [ ] Scalability claim (100+ clients) tested or explicitly recorded as untested

---

# CURRENT IMPLEMENTATION REALITY (verified run, 2026-08-21)

These observed facts OVERRIDE the project document wherever they conflict.
Do not change working code to match the document — the document is stale.

## Verified working
- MongoDB + Rust orchestrator (Tonic gRPC) + mTLS with real CA-signed certs
- Two-phase enrollment with OTP; `client.pem` issued and verifiable against `ca.pem`
- gRPC chunked streaming upload, 1 MB chunks, SHA-256 per payload
- TPM-backed ECDSA P-256 device signing (key `FederatedDeviceKey`)
- AES-GCM SecureStore envelope
- Trimmed-mean aggregation, GridFS global model persistence
- End state confirmed: model_updates=3, receipts=3, global_models=1

## Actual dimensions (document says 176/70/192 — that is WRONG)
- audio (wav2vec2)  = 154
- video (DenseNet)  = 84
- text (MentalBERT) = 768
- fusion input      = 1024
- total parameters  = 109,763,494
- dataset: `dataset_build/daic_records_multimodal.parquet`, 188 rows

## Actual DP config (document says eps 37-150 — that is stale)
- mechanism gaussian, clip_norm 1.0, noise_multiplier 1.0
- eps = 5.302585 at delta = 1e-5
- accounting: RDP (Mironov 2017), single composition T=1

## Known defects to be fixed (do not treat as intended behaviour)
1. DP noise norm 10350.57 vs delta norm 0.0062 — noise swamps signal by ~1.7e6x
2. Clipping never fires (delta 0.0062 << clip 1.0)
3. Text pipeline is embedding the virtual interviewer "Ellie", not the participant
4. Local eval is single-class: accuracy 1.0 with precision/recall/f1 all 0.0
5. Training batch size is 1, not the configured 8
6. Warm-start never exercised — every round logs "No global model"
7. Privacy accounting is per-round only; no cumulative eps across the 26 recorded rounds
8. No explainability artifacts produced (`explain_logs/` empty)

## Environment
- Windows, PowerShell, `.venv\Scripts\python.exe`
- Repo root: `D:\Download D\BE PIPELINE\Capstone-`
- Secure store: `C:\Users\satya\.federated\data\secure_store`
- openssl via `C:\Program Files\Git\mingw64\bin`
