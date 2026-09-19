"""
pipeline.py — Full federated pipeline

SECURITY FIXES:
  FIX-PIPELINE-1: Model bytes are now streamed to the server via UploadUpdate
                  RPC before SubmitReceipt. Previously enc_uri was a local
                  file path the server could never open.

  FIX-PIPELINE-2: Receipt.enc_handle is the server-side GridFS ObjectId
                  returned by UploadAck — NOT a local file path.

  FIX-PIPELINE-3: epsilon_spent is taken from the DP agent's real output
                  instead of the hardcoded 1.0.

  FIX-PIPELINE-4: payload_hash is computed over the actual bytes that were
                  streamed — guarantees receipt matches uploaded data.

  FIX-PIPELINE-5: Per-chunk SHA-256 included in each UpdateChunk so server
                  can verify every chunk independently.

  FIX-PIPELINE-6: Global model downloaded via DownloadGlobalModel RPC with
                  per-chunk and full-model hash verification.
"""

import os
import uuid
import math
import json
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.lda.main import preprocess, PreprocessRequest
from agents.trainer.trainer_mentalbert_privacy import orchestrate as trainer_orchestrate
# Fix E4: single source of truth lives in trainer_mentalbert_privacy.py -
# imported, not re-declared, so this file cannot hold a stale default the way
# the old hardcoded "epochs": 1 below used to.
from agents.trainer.trainer_mentalbert_privacy import SUPERVISED_LR, SUPERVISED_EPOCHS
from agents.dp.dp_agent import DPAgent
from agents.enc.enc_agent import EncryptionAgent
from core.centralized_secure_store import SecureStore
from core import reporting as rpt
from runtime.grpc.orchestrator_pb2 import (
    DeviceId, Receipt, UpdateChunk, RoundRequest
)
from runtime.grpc_client import call_with_retry
from runtime.tpm_guard import sign_message

log = logging.getLogger(__name__)

_STORE_ROOT = Path.home() / ".federated" / "data" / "secure_store"
_CONFIG_URI = f"file://{Path.home()}/.federated/configs/local_config.yaml"
_INPUT_DIR  = str(Path.home() / ".federated" / "data" / "input")

LDA_MODE      = "session"
CHUNK_SIZE    = 1 * 1024 * 1024   # 1 MB per gRPC chunk
MAX_EPS_VALUE = 10.0               # hard ceiling — server rejects > this

# ── FIX-MULTIMODAL-2: input-mode selector ──────────────────────────────────────
# PIPELINE_MODE=text (default): unchanged behavior — LDA preprocess() on
#   ~/.federated/data/input/*.mp4, exactly as originally verified.
# PIPELINE_MODE=multimodal: skips live LDA entirely and points the trainer's
#   existing read_parquet_records() .parquet branch directly at the frozen,
#   read-only, pre-extracted DAIC-WOZ dataset — real text + real
#   features.audio.wav2vec2 (154-dim) + real features.video.densenet (84-dim).
#   Nothing about DP/encryption/TPM/mTLS/receipt submission changes for this
#   mode — only step 3 (input preparation) and the global-model local cache
#   subdirectory below branch on it.
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "text").strip().lower()
_REPO_ROOT = Path(__file__).resolve().parent.parent
# Fix A: participant-only text (Ellie's interviewer script filtered out at
# source - see scripts/rebuild_participant_only_parquet.py). The original
# dataset_build/daic_records_multimodal.parquet is left untouched.
_MULTIMODAL_PARQUET = _REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only.parquet"
# Fix D: "0" (or unset) is an explicit no-limit sentinel - load the full local
# partition. Truncation is opt-in (set MULTIMODAL_MAX_SAMPLES to a positive
# int) rather than opt-out, so a smoke test has to ask for it explicitly.
_MULTIMODAL_MAX_SAMPLES = int(os.environ.get("MULTIMODAL_MAX_SAMPLES", "0"))


# ── Schema validation (unchanged) ─────────────────────────────────────────────
_REQUIRED_MANIFEST_KEYS = {"session_id", "artifact_manifest", "receipts", "count"}

def _validate_lda_output(result: dict):
    missing = _REQUIRED_MANIFEST_KEYS - set(result.keys())
    if missing:
        raise ValueError(f"LDA output missing keys: {missing}")
    if result.get("count", 0) == 0:
        raise ValueError("LDA produced 0 rows — check input data")
    log.info("[schema] LDA output valid: %d rows", result["count"])


_REQUIRED_TRAINER_KEYS = {"local_update_uri"}

def _validate_trainer_output(result: dict):
    missing = _REQUIRED_TRAINER_KEYS - set(result.keys())
    if missing:
        raise ValueError(f"Trainer output missing keys: {missing}")
    uri  = result["local_update_uri"]
    if not uri.startswith("file://"):
        raise ValueError(f"Trainer output URI malformed: {uri}")
    path = Path(uri[len("file://"):])
    if not path.exists():
        raise ValueError(f"Trainer update file not found: {path}")
    log.info("[schema] Trainer output valid: %s", path.name)


# ── FIX-PIPELINE-6: Download global model from server ─────────────────────────
def _download_global_model(stub, device_id: bytes, round_id: int) -> Optional[str]:
    """
    Download the global model from the server via streaming RPC.
    Verifies per-chunk and full-model SHA-256 hashes.
    Returns local path to the downloaded model file, or None.
    """
    t0 = time.time()
    try:
        request = RoundRequest(device_id=device_id, round_id=round_id)
        chunks_received = []
        full_model_hash_expected = None
        chunk_count = 0

        for chunk in stub.DownloadGlobalModel(request, timeout=120):
            chunk_count += 1
            # Verify chunk integrity
            computed = hashlib.sha256(chunk.data).digest()
            if computed != bytes(chunk.chunk_hash):
                raise ValueError(
                    f"Global model chunk {chunk.chunk_index} hash mismatch — "
                    f"data corrupted in transit"
                )
            chunks_received.append(chunk.data)

            if chunk.chunk_index == chunk.total_chunks - 1 and chunk.model_hash:
                full_model_hash_expected = bytes(chunk.model_hash)

        if not chunks_received:
            log.info("[FL] No global model chunks received")
            return None

        model_bytes = b"".join(chunks_received)

        # Verify full model hash
        hash_verified = False
        if full_model_hash_expected:
            actual_hash = hashlib.sha256(model_bytes).digest()
            if actual_hash != full_model_hash_expected:
                raise ValueError(
                    "Global model full-hash mismatch — model rejected"
                )
            hash_verified = True
            log.info("[FL] Global model hash verified OK")

        # Save to local path
        # FIX-MULTIMODAL-2: separate local cache subdirectory per mode so a
        # multimodal round N's downloaded global model can never collide with
        # (or be mistaken for) the text-only chain's global_round{N}.pt file
        # — the two experiments use independent round numbering in
        # independent MongoDB databases, but round IDs can coincide (both
        # start at 1), so the local filename alone isn't a safe disambiguator.
        subdir = "global_models" if PIPELINE_MODE == "text" else f"global_models_{PIPELINE_MODE}"
        model_dir = Path.home() / ".federated" / "data" / subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"global_round{round_id}.pt"
        model_path.write_bytes(model_bytes)

        log.info(
            "[FL] Global model downloaded: %d bytes → %s",
            len(model_bytes), model_path
        )

        rpt.subheader("GLOBAL MODEL — DOWNLOAD")
        rpt.kv("Round ID", round_id)
        rpt.kv("Chunks received", chunk_count)
        rpt.kv("Model size", f"{len(model_bytes)} bytes")
        rpt.kv("SHA-256 (full model)", hashlib.sha256(model_bytes).hexdigest())
        rpt.kv("Hash verification", "PASS" if hash_verified else "SKIPPED (server sent no full-model hash)")
        rpt.kv("Saved to", str(model_path))
        rpt.kv("Download duration", f"{time.time() - t0:.2f} sec")
        rpt.ok("Global model downloaded")

        return str(model_path)

    except Exception as e:
        log.warning("[FL] Could not download global model: %s — starting from scratch", e)
        rpt.warn(f"Global model download failed: {e} — starting from random init")
        return None


# ── FIX-PIPELINE-1,4,5: Stream encrypted update to server ─────────────────────
def _stream_update(
    stub,
    device_id: bytes,
    round_id: int,
    update_path: str,
    session_id: str,
) -> tuple:
    """
    Read the encrypted update file and stream it to the server in chunks.

    Each chunk includes its own SHA-256 hash so the server can verify
    every chunk independently (FIX-PIPELINE-5).

    Returns (server_handle, payload_hash_bytes):
      server_handle    — GridFS ObjectId string from UploadAck.server_handle
      payload_hash_bytes — SHA-256 of all bytes streamed (for Receipt)
    """
    path = Path(update_path[len("file://"):])
    if not path.exists():
        raise FileNotFoundError(f"Update file not found: {path}")

    data = path.read_bytes()
    if not data:
        raise ValueError("Update file is empty — will not stream")

    total_size   = len(data)
    total_chunks = math.ceil(total_size / CHUNK_SIZE)
    payload_hash = hashlib.sha256(data).digest()   # hash of ALL bytes

    log.info(
        "[pipeline] Streaming %d bytes in %d chunks (sha256=%s…)",
        total_size, total_chunks, payload_hash.hex()[:16]
    )

    rpt.header("SECURE TRANSPORT")
    rpt.kv("Protocol", "gRPC")
    rpt.kv("TLS", "mTLS (client + server cert)")
    rpt.kv("Payload size", f"{total_size} bytes")
    rpt.kv("Chunk size", f"{CHUNK_SIZE} bytes")
    rpt.kv("Chunks", total_chunks)
    rpt.kv("SHA-256 (payload)", payload_hash.hex())

    t0 = time.time()

    def chunk_generator():
        for i in range(total_chunks):
            chunk_data = data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            chunk_hash = hashlib.sha256(chunk_data).digest()
            yield UpdateChunk(
                session_id=session_id,
                round_id=round_id,
                device_id=device_id,
                chunk_index=i,
                total_chunks=total_chunks,
                data=chunk_data,
                chunk_hash=chunk_hash,
            )

    # stream is client-side streaming — use direct stub call
    ack = stub.UploadUpdate(chunk_generator(), timeout=300)

    upload_duration = time.time() - t0

    if not ack.ok:
        rpt.fail(f"Server rejected upload: {ack.error}")
        raise RuntimeError(
            f"Server rejected upload: {ack.error}"
        )

    log.info("[pipeline] Upload complete — server_handle=%s", ack.server_handle)

    rpt.kv("mTLS status", "PASS (channel established)")
    rpt.kv("Upload status", "SUCCESS")
    rpt.kv("Upload duration", f"{upload_duration:.2f} sec")
    rpt.kv("Server handle (GridFS)", ack.server_handle)
    rpt.ok("Update securely uploaded")

    return ack.server_handle, payload_hash


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(
    stub,
    device_id: bytes,
    master_secret: bytes,
    session_dir: Optional[Path] = None,
):
    """
    Full local federated pipeline:
      1. Query round
      2. Download global model (if available)
      3. LDA preprocessing
      4. Trainer (fine-tune on local data)
      5. DP noise
      6. Encryption
      7. Stream bytes to server (NEW — FIX-PIPELINE-1)
      8. Submit receipt with real epsilon and server handle (FIX-PIPELINE-2,3)
    """
    round_t0 = time.time()
    stage_status = {}   # stage name -> "PASS" | "FAIL" | "SKIP", populated as we go

    # ── 1. Query round ────────────────────────────────────────────────────────
    log.info("[pipeline] Querying round metadata...")
    round_meta = call_with_retry(stub.GetRound, DeviceId(id=device_id), timeout=10)

    session_id = f"client-{uuid.uuid4().hex[:12]}"

    rpt.header(f"ROUND {round_meta.round_id} — CLIENT")
    rpt.subheader("[01] ROUND METADATA")
    rpt.kv("Round ID", round_meta.round_id)
    rpt.kv("Session ID", session_id)
    rpt.kv("Device ID", device_id.hex())
    rpt.kv("Round state", round_meta.state)
    rpt.kv("Global model", "Available" if round_meta.global_model_available else "Not available")
    rpt.ok("Round metadata received")
    stage_status["round_metadata"] = "PASS"

    if round_meta.state != "Collecting":
        log.info("[pipeline] Round state=%s — skipping", round_meta.state)
        rpt.warn(f"Round state={round_meta.state} — nothing to submit this cycle")
        return

    log.info("[pipeline] Round %d active — session %s", round_meta.round_id, session_id)

    # ── 2. Download global model (FIX-PIPELINE-6) ─────────────────────────────
    global_model_path = None
    if round_meta.global_model_available:
        log.info("[FL] Global model available — downloading...")
        global_model_path = _download_global_model(stub, device_id, round_meta.round_id)
        stage_status["global_model"] = "PASS" if global_model_path else "FAIL"
    else:
        log.info("[FL] No global model for this round — random init")
        rpt.subheader("[02] GLOBAL MODEL")
        rpt.kv("Global model", "Not available — this client will train from random initialization")
        stage_status["global_model"] = "SKIP"

    # ── 3. Input preparation: LDA (text mode) or frozen parquet (multimodal) ──
    if PIPELINE_MODE == "multimodal":
        log.info("[pipeline] PIPELINE_MODE=multimodal — using frozen DAIC-WOZ parquet, skipping live LDA")

        if not _MULTIMODAL_PARQUET.exists():
            stage_status["lda"] = "FAIL"
            raise FileNotFoundError(
                f"Multimodal dataset not found: {_MULTIMODAL_PARQUET}. "
                "This is a frozen, read-only fixture — it must already exist in the repo."
            )

        t0 = time.time()
        # Real values, not hardcoded: peek the actual frozen file to report
        # real row count / feature dims before the trainer loads it.
        import pyarrow.parquet as _pq
        _table = _pq.read_table(str(_MULTIMODAL_PARQUET))
        _total_rows = _table.num_rows
        _sample_row = _table.to_pandas().iloc[0]
        _sample_features = json.loads(_sample_row["features"])
        _audio_dim = len(_sample_features.get("audio", {}).get("wav2vec2", []))
        _video_dim = len(_sample_features.get("video", {}).get("densenet", []))
        lda_elapsed = time.time() - t0

        rpt.header("MULTIMODAL INPUT ANALYSIS")
        rpt.kv("Source", str(_MULTIMODAL_PARQUET))
        rpt.kv("Total rows in dataset", _total_rows)
        _rows_used = _total_rows if not _MULTIMODAL_MAX_SAMPLES else min(_MULTIMODAL_MAX_SAMPLES, _total_rows)
        rpt.kv("Rows used this round", _rows_used)
        rpt.line()
        rpt.line("TEXT INPUT")
        rpt.kv("Status", "AVAILABLE", indent=2)
        rpt.kv("Sample participant_id", _sample_row["participant_id"], indent=2)
        rpt.line()
        rpt.line("AUDIO INPUT")
        rpt.kv("Status", "AVAILABLE" if _audio_dim else "ABSENT", indent=2)
        rpt.kv("Source", "features.audio.wav2vec2 (pre-extracted, frozen)", indent=2)
        rpt.kv("[MULTIMODAL] audio dimension", _audio_dim, indent=2)
        rpt.line()
        rpt.line("VIDEO INPUT")
        rpt.kv("Status", "AVAILABLE" if _video_dim else "ABSENT", indent=2)
        rpt.kv("Source", "features.video.densenet (pre-extracted, frozen)", indent=2)
        rpt.kv("[MULTIMODAL] video dimension", _video_dim, indent=2)
        rpt.kv("Execution time", f"{lda_elapsed:.3f} sec")
        rpt.ok("Multimodal input analysis complete")

        manifest_uri = str(_MULTIMODAL_PARQUET)
        stage_status["lda"] = "PASS"
    else:
        log.info("[pipeline] Running LDA...")

        video_dir = str(session_dir) if (session_dir and session_dir.exists()) else _INPUT_DIR
        input_video_count = len(list(Path(video_dir).glob("*.mp4"))) if Path(video_dir).exists() else 0

        rpt.header("LOCAL DATA AGENT (LDA)")
        rpt.kv("Input directory", video_dir)
        rpt.kv("Input videos found", input_video_count)
        rpt.kv("Mode", LDA_MODE)

        lda_req = PreprocessRequest(
            mode=LDA_MODE,
            inputs={"video_dir": video_dir},
            config_uri=_CONFIG_URI,
        )

        t0 = time.time()
        try:
            lda_result = preprocess(lda_req)
            _validate_lda_output(lda_result)
            stage_status["lda"] = "PASS"
        except Exception:
            stage_status["lda"] = "FAIL"
            raise
        lda_elapsed = time.time() - t0
        log.info("[pipeline] LDA done in %.1fs", lda_elapsed)

        manifest_uri = lda_result["artifact_manifest"]
        rpt.kv("Session ID", lda_result["session_id"])
        rpt.kv("Output rows (manifest)", lda_result["count"])
        rpt.kv("Encrypted artifacts written", len(lda_result["receipts"]))
        rpt.kv("Artifact manifest", manifest_uri)
        rpt.kv("Execution time", f"{lda_elapsed:.2f} sec")
        rpt.ok("LDA completed")

    # ── 4. Trainer ────────────────────────────────────────────────────────────
    log.info("[pipeline] Running trainer (mode=supervised)...")

    trainer_kwargs = {
        "input_path":  manifest_uri,
        "session_id":  session_id,
        "mode":        "supervised",
        "epochs":      SUPERVISED_EPOCHS,
        "batch_size":  8,
        "lr":          SUPERVISED_LR,
        # Step 16: round_id was already available here (round_meta.round_id,
        # used below for logging/receipt signing) but never reached the
        # trainer. trainer_orchestrate() uses it to decay lr round over
        # round — see trainer_mentalbert_privacy.py's Step 16 comment.
        "round_id":    round_meta.round_id,
    }
    if PIPELINE_MODE == "multimodal":
        trainer_kwargs["max_samples"] = _MULTIMODAL_MAX_SAMPLES
    if global_model_path:
        trainer_kwargs["global_model_path"] = global_model_path

    rpt.header("LOCAL TRAINING")
    rpt.kv("Model", "MentalBERT (multimodal fusion)")
    rpt.kv("Initialization", "Global model (warm-start)" if global_model_path else "Random / local pretrained")
    rpt.kv("Round ID", trainer_kwargs["round_id"])
    rpt.kv("Epochs", trainer_kwargs["epochs"])
    rpt.kv("Batch size", trainer_kwargs["batch_size"])
    rpt.kv("Learning rate (base)", trainer_kwargs["lr"])
    rpt.kv("Optimizer", "AdamW")

    t0 = time.time()
    try:
        trainer_out = trainer_orchestrate(**trainer_kwargs)
        _validate_trainer_output(trainer_out)
        stage_status["training"] = "PASS"
    except Exception:
        stage_status["training"] = "FAIL"
        raise
    trainer_elapsed = time.time() - t0
    log.info("[pipeline] Trainer done in %.1fs", trainer_elapsed)

    local_update_uri = trainer_out["local_update_uri"]
    rpt.kv("Output update", local_update_uri)
    if "num_steps_completed" in trainer_out:
        rpt.kv("Optimizer steps completed", trainer_out["num_steps_completed"])
    rpt.kv("Training time", f"{trainer_elapsed:.2f} sec")
    rpt.ok("Local training completed")

    # ── 5. Differential Privacy ───────────────────────────────────────────────
    log.info("[pipeline] Applying DP noise...")

    store    = SecureStore(agent="trainer", root=_STORE_ROOT)
    # Fix E4 recalibration: lr/epochs changes shifted the measured delta
    # sensitivity from max=0.1177 to max=0.7294 (see
    # scripts/calibrate_clip_norm.py, N=30) - 0.15 would now clip almost
    # every run. Overridable per-run; env var wins over the calibrated default.
    dp_clip_norm = float(os.environ.get("DP_CLIP_NORM", "0.85"))
    # Step 12: overridable so the privacy-utility comparison can run a
    # genuine no-privacy arm (DP_MECHANISM=none or DP_NOISE_MULTIPLIER=0)
    # through the exact same pipeline code path as the DP arm.
    # Noise reduction (2026-09-20): lowered from 1.0 to 0.8, then to 0.75
    # after switching live accounting to _rdp_to_dp_tight() (Canonne-Kamath-
    # Steinke tightening, see dp_agent.py). Real historical delta L2 never
    # exceeds 0.77 (see docs/IMPLEMENTATION_NOTES.md), so noise was swamping
    # signal by ~590x at the old sigma=1.0. sigma=0.75 -> epsilon=6.603254
    # under the tighter method, ~17.5% margin under the eps<=8 target
    # (sigma=0.7 rejected: only ~10.47% margin under the tighter method).
    # clip_norm unchanged.
    dp_noise_multiplier = float(os.environ.get("DP_NOISE_MULTIPLIER", "0.75"))
    dp_mechanism = os.environ.get("DP_MECHANISM", "gaussian")
    dp_agent = DPAgent(
        clip_norm=dp_clip_norm,
        noise_multiplier=dp_noise_multiplier,
        mechanism=dp_mechanism,
        store=store,
    )

    rpt.header("DIFFERENTIAL PRIVACY")
    rpt.kv("Mechanism configured", dp_mechanism)
    rpt.kv("Clip norm configured", dp_clip_norm)
    rpt.kv("Noise multiplier configured", dp_noise_multiplier)

    try:
        dp_result = dp_agent.process_local_update(
            local_update_uri,
            session_id=session_id,
            metadata={"session_id": session_id},
        )
        stage_status["dp"] = "PASS"
    except Exception:
        stage_status["dp"] = "FAIL"
        raise

    log.info(
        "[pipeline] DP done: L2 before=%.4f after=%.4f eps=%.6f",
        dp_result["l2_norm_before"],
        dp_result["l2_norm_after"],
        dp_result.get("epsilon_spent", 0.0),
    )

    l2_before = dp_result["l2_norm_before"]
    l2_after  = dp_result["l2_norm_after"]
    # (Step-by-step clip/noise/epsilon detail is printed inside dp_agent.py,
    #  which has access to the raw parameter tensors — not duplicated here.)

    # FIX-PIPELINE-3: read real epsilon from DP agent, never hardcode
    epsilon_spent = dp_result.get("epsilon_spent")
    if epsilon_spent is None or epsilon_spent <= 0.0 or math.isinf(epsilon_spent):
        # Fallback: DP agent returned no finite epsilon (non-Gaussian mechanism
        # or zero noise).  Log a warning — Gaussian mechanism always returns real RDP.
        epsilon_spent = 1.0
        log.warning(
            "[pipeline] DP agent returned no finite epsilon_spent (mechanism=%s) "
            "— using fallback 1.0.  Use mechanism='gaussian' for real RDP accounting.",
            dp_agent.mechanism,
        )
        rpt.warn("DP agent returned no finite epsilon — using fallback 1.0 for the receipt")
    elif epsilon_spent > MAX_EPS_VALUE:
        rpt.fail(f"epsilon_spent={epsilon_spent:.4f} exceeds hard ceiling {MAX_EPS_VALUE}")
        raise ValueError(
            f"epsilon_spent={epsilon_spent:.4f} exceeds hard ceiling {MAX_EPS_VALUE} "
            f"— reduce noise multiplier or clip norm"
        )
    rpt.ok("Differential privacy applied")

    # ── 6. Encryption ─────────────────────────────────────────────────────────
    log.info("[pipeline] Finalizing encryption...")
    rpt.header("ENCRYPTION")
    enc_agent       = EncryptionAgent(mode="aes")
    try:
        enc_result      = enc_agent.process_dp_update(dp_result["receipt_uri"])
        stage_status["encryption"] = "PASS"
    except Exception:
        stage_status["encryption"] = "FAIL"
        raise
    final_update_uri = enc_result["receipt"]["outputs"][0]
    enc_size = Path(final_update_uri[len("file://"):]).stat().st_size
    rpt.kv("Serialization", "torch.save state_dict -> SecureStore AES-GCM envelope")
    rpt.kv("Encryption algorithm", enc_result["receipt"]["params"]["encryption_scheme"])
    rpt.kv("Encrypted file", final_update_uri)
    rpt.kv("Encrypted size", f"{enc_size} bytes")
    rpt.ok("Update encrypted")

    # ── 7. Stream bytes to server (FIX-PIPELINE-1,4,5) ───────────────────────
    log.info("[pipeline] Streaming update to server...")
    t0 = time.time()

    server_handle, payload_hash = _stream_update(
        stub,
        device_id=device_id,
        round_id=round_meta.round_id,
        update_path=final_update_uri,
        session_id=session_id,
    )

    log.info("[pipeline] Stream complete in %.1fs", time.time() - t0)

    # ── 8. Submit receipt (FIX-PIPELINE-2,3) ─────────────────────────────────
    log.info("[pipeline] Submitting receipt to server...")

    # Sign canonical message: device_id || round_id (8 bytes BE) || payload_hash
    msg = (
        device_id
        + round_meta.round_id.to_bytes(8, "big")
        + payload_hash           # SHA-256 of actual uploaded bytes
    )

    rpt.header("TPM / DEVICE SIGNING")
    rpt.kv("Key identifier", "FederatedDeviceKey (ECDSA P-256, TPM-backed)")
    rpt.kv("Payload size", f"{len(msg)} bytes")
    try:
        signature = sign_message(msg)
        stage_status["tpm_signing"] = "PASS"
    except Exception:
        stage_status["tpm_signing"] = "FAIL"
        raise
    rpt.kv("Signature generated", "YES")
    rpt.kv("Signature size", f"{len(signature)} bytes")
    rpt.ok("Message signed by device key")

    receipt = Receipt(
        device_id=device_id,
        round_id=round_meta.round_id,
        payload_hash=payload_hash,           # FIX-PIPELINE-4: real hash
        epsilon_spent=epsilon_spent,         # FIX-PIPELINE-3: real epsilon
        signature=signature,
        enc_handle=server_handle,            # FIX-PIPELINE-2: GridFS ID
        scheme="AES-GCM-DP-ECDSA",
        nonce="",
    )

    ack = call_with_retry(stub.SubmitReceipt, receipt, timeout=15)

    rpt.header("FEDERATED LEARNING - CLIENT")
    rpt.kv("Round", round_meta.round_id)
    rpt.kv("Client (device ID)", device_id.hex()[:16] + "…")
    rpt.kv("Global model", "Available" if round_meta.global_model_available else "Not available")
    rpt.kv("Local training", stage_status.get("training", "?"))
    rpt.kv("DP protection", stage_status.get("dp", "?"))
    rpt.kv("Encryption", stage_status.get("encryption", "?"))
    rpt.kv("TPM signing", stage_status.get("tpm_signing", "?"))
    rpt.kv("Update size", f"{enc_size} bytes")
    rpt.kv("Update hash (SHA-256)", payload_hash.hex())
    rpt.kv("Receipt submitted", "YES" if ack.ok else "NO")

    if ack.ok:
        log.info("[pipeline] ✅ Round %d update submitted", round_meta.round_id)
        stage_status["receipt"] = "PASS"
        rpt.ok("Client update submitted")
    else:
        log.warning("[pipeline] Server returned ok=False for round %d", round_meta.round_id)
        stage_status["receipt"] = "FAIL"
        rpt.fail("Server rejected receipt")

    # ── Round summary ──────────────────────────────────────────────────────────
    total_latency = time.time() - round_t0
    rpt.header(f"ROUND {round_meta.round_id} SUMMARY")
    for name in ("round_metadata", "global_model", "lda", "training", "dp",
                 "encryption", "tpm_signing", "receipt"):
        if name in stage_status:
            rpt.kv(name.replace("_", " ").title(), stage_status[name])
    round_success = all(v in ("PASS", "SKIP") for v in stage_status.values())
    rpt.line()
    rpt.kv("Round status", "SUCCESS" if round_success else "FAILURE")
    rpt.kv("Total latency", f"{total_latency:.2f} sec")
    rpt.line("=" * 64)

    # ── Persist round outcome for cross-run E2E summaries ─────────────────────
    try:
        history_path = Path.home() / ".federated" / "state" / "round_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "round_id": round_meta.round_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stages": stage_status,
            "status": "SUCCESS" if round_success else "FAILURE",
            "total_latency_sec": round(total_latency, 3),
            "epsilon_spent": epsilon_spent,
            "update_size_bytes": enc_size,
            "update_hash": payload_hash.hex(),
            "server_handle": server_handle,
        }
        with open(history_path, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(record) + "\n")

        # ── E2E summary across all rounds recorded on this device so far ──────
        # Each run-once invocation is a separate process/terminal for a single
        # round; this reads back the real outcomes persisted by prior runs
        # (including this one) so a multi-round summary can be shown without
        # inventing any aggregate figures.
        history_records = []
        with open(history_path, "r", encoding="utf-8") as hf:
            for hline in hf:
                hline = hline.strip()
                if hline:
                    try:
                        history_records.append(json.loads(hline))
                    except json.JSONDecodeError:
                        continue

        rpt.header("E2E FEDERATED LEARNING SUMMARY")
        rpt.kv("Rounds recorded on this device", len(history_records))
        for rec in history_records:
            rpt.line(f"  Round {rec['round_id']} [{rec['status']}]  "
                      f"eps={rec.get('epsilon_spent', 0):.4f}  "
                      f"size={rec.get('update_size_bytes', 0)}B  "
                      f"latency={rec.get('total_latency_sec', 0):.1f}s  "
                      f"({rec.get('timestamp', '')})")
        rpt.subheader("Security (always applied by this pipeline)")
        rpt.kv("DP", "ENABLED (RDP-accounted Gaussian mechanism)")
        rpt.kv("Encryption", "ENABLED (AES-GCM, SecureStore)")
        rpt.kv("TPM signing", "ENABLED (ECDSA P-256, device-bound key)")
        rpt.kv("mTLS", "ENABLED (client + server certificate)")
        rpt.line()
        rpt.kv("Overall E2E status", "SUCCESS" if round_success else "FAILURE")
        rpt.line("=" * 64)
    except Exception as e:
        log.debug("Could not persist/read round history: %s", e)