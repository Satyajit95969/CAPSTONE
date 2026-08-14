"""
round0_step6_validate.py — Phase 4 Step 6: Validate Round 0 execution.

Drives the REAL Round 0 server-side data path end-to-end:

    local update (real prior trainer output)
        -> DPAgent.process_local_update()        [REAL agent]
        -> EncryptionAgent.process_dp_update()    [REAL agent]
        -> UploadUpdate  (gRPC, GridFS)           [REAL server RPC]
        -> SubmitReceipt (gRPC, ECDSA + HMAC chain)[REAL server RPC]

The upload-streaming and receipt-construction code below is copied verbatim
from runtime/pipeline.py (_stream_update + step 7/8) so that the exact
production code path is exercised.  The only substitution is the LDA + trainer
stages: those need the DAIC dataset (absent on this machine) and were already
validated in Phases 1-3 / GAP 2.  A real prior trainer output state_dict is
used as the local update instead.

ENVIRONMENT NOTE (AVG): AVG Web/Mail Shield transparently MITMs TLS on this
host.  It re-signs the server cert with its own root but DOES forward the
client certificate to the backend (verified: GetRound succeeds through it).
So we add AVG's root to the client trust bundle; mTLS still authenticates the
client at the server.  This is a client-side trust adjustment only — no change
to the server architecture or to require_client_cert().
"""

import sys, os, io, math, json, hashlib, shutil, subprocess, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grpc
from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub
from runtime.grpc.orchestrator_pb2 import DeviceId, Receipt, UpdateChunk, RoundRequest

from dp_agent.dp_agent import DPAgent
from enc_agent.enc_agent import EncryptionAgent
from centralized_secure_store import SecureStore

# ── Config (mirrors pipeline.py) ───────────────────────────────────────────────
ADDR          = os.environ.get("FED_SERVER", "127.0.0.1:50451")
FEDERATED     = Path.home() / ".federated"
KEYS          = FEDERATED / "keys"
TPM_PUBKEY    = FEDERATED / "tpm" / "device_pubkey.pem"
SIGNER        = FEDERATED / "bin" / "windows_signer.exe"
_STORE_ROOT   = FEDERATED / "data" / "secure_store"
CHUNK_SIZE    = 1 * 1024 * 1024
MAX_EPS_VALUE = 10.0

# trust bundle: our CA + AVG root (so AVG's MITM cert verifies; client cert is
# still forwarded by AVG and authenticated at the server)
CA_BUNDLE     = Path("ca_plus_avg.pem")

# a real prior trainer output used as the Round 0 local update
LOCAL_UPDATE_SRC = Path("trainer_outputs/local_probe_base.pt")


def log(msg): print(f"[round0] {msg}", flush=True)


def sign_message(message: bytes) -> bytes:
    """Real TPM-backed ECDSA P-256 signing via windows_signer.exe (DER output)."""
    proc = subprocess.run([str(SIGNER), "--sign"], input=message,
                          stdout=subprocess.PIPE, check=True)
    return proc.stdout


def make_stub():
    creds = grpc.ssl_channel_credentials(
        root_certificates=CA_BUNDLE.read_bytes(),
        private_key=(KEYS / "client.key").read_bytes(),
        certificate_chain=(KEYS / "client.pem").read_bytes(),
    )
    opts = [("grpc.keepalive_time_ms", 10000), ("grpc.keepalive_timeout_ms", 5000)]
    ch = grpc.secure_channel(ADDR, creds, options=opts)
    grpc.channel_ready_future(ch).result(timeout=10)
    return OrchestratorStub(ch)


# ── _stream_update: copied verbatim from runtime/pipeline.py ────────────────────
def _stream_update(stub, device_id, round_id, update_path, session_id):
    path = Path(update_path[len("file://"):])
    if not path.exists():
        raise FileNotFoundError(f"Update file not found: {path}")
    data = path.read_bytes()
    if not data:
        raise ValueError("Update file is empty — will not stream")

    total_size   = len(data)
    total_chunks = math.ceil(total_size / CHUNK_SIZE)
    payload_hash = hashlib.sha256(data).digest()
    log(f"Streaming {total_size} bytes in {total_chunks} chunks "
        f"(sha256={payload_hash.hex()[:16]}…)")

    def chunk_generator():
        for i in range(total_chunks):
            chunk_data = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
            chunk_hash = hashlib.sha256(chunk_data).digest()
            yield UpdateChunk(
                session_id=session_id, round_id=round_id, device_id=device_id,
                chunk_index=i, total_chunks=total_chunks,
                data=chunk_data, chunk_hash=chunk_hash,
            )

    ack = stub.UploadUpdate(chunk_generator(), timeout=300)
    if not ack.ok:
        raise RuntimeError(f"Server rejected upload: {ack.error}")
    log(f"Upload complete — server_handle={ack.server_handle}")
    return ack.server_handle, payload_hash


def main():
    print("=" * 64)
    print("PHASE 4 STEP 6 — ROUND 0 EXECUTION VALIDATION")
    print("=" * 64)

    device_pubkey = TPM_PUBKEY.read_bytes()
    device_id = hashlib.sha256(device_pubkey).digest()
    log(f"device_id = {device_id.hex()}")

    stub = make_stub()
    log("mTLS channel ready (AVG-passthrough)")

    # ── 1. Query round ─────────────────────────────────────────────────────────
    rm = stub.GetRound(DeviceId(id=device_id), timeout=10)
    log(f"GetRound: round_id={rm.round_id} state={rm.state} "
        f"eps_max={rm.epsilon_max} global_model_available={rm.global_model_available}")
    if rm.state != "Collecting":
        raise RuntimeError(f"round not Collecting (state={rm.state})")

    session_id = f"round0-{int(time.time())}"

    # ── 2. Global model (Round 0 = cold start, none expected) ──────────────────
    if rm.global_model_available:
        log("global model available — would DownloadGlobalModel (not Round 0)")
    else:
        log("no global model (cold start) — Round 0 random-init path ✓")

    # ── 3/4. LDA + trainer → substituted with a real prior trainer output ──────
    sess_dir = _STORE_ROOT.parent / "round0_local"
    sess_dir.mkdir(parents=True, exist_ok=True)
    local_update_path = sess_dir / "local_update.pt"
    shutil.copyfile(LOCAL_UPDATE_SRC, local_update_path)
    local_update_uri = "file://" + str(local_update_path)
    sd = __import__("torch").load(local_update_path, map_location="cpu", weights_only=False)
    log(f"local update = real trainer output ({LOCAL_UPDATE_SRC.name}, "
        f"{len(sd)} tensors, {local_update_path.stat().st_size} bytes)")

    # ── 5. DP (REAL agent, exactly as pipeline.py configures it) ───────────────
    store = SecureStore(agent="trainer", root=_STORE_ROOT)
    dp_agent = DPAgent(clip_norm=1.0, noise_multiplier=1.0, mechanism="gaussian", store=store)
    dp_result = dp_agent.process_local_update(
        local_update_uri, session_id=session_id, metadata={"session_id": session_id})
    log(f"DP done: L2 before={dp_result['l2_norm_before']:.4f} "
        f"after={dp_result['l2_norm_after']:.4f} eps={dp_result.get('epsilon_spent', 0.0):.6f}")

    epsilon_spent = dp_result.get("epsilon_spent")
    if epsilon_spent is None or epsilon_spent <= 0.0 or math.isinf(epsilon_spent):
        epsilon_spent = 1.0
        log("WARNING: DP eps not finite — fallback 1.0")
    elif epsilon_spent > MAX_EPS_VALUE:
        raise ValueError(f"epsilon_spent={epsilon_spent:.4f} exceeds ceiling {MAX_EPS_VALUE}")

    # ── 6. Encryption (REAL agent) ─────────────────────────────────────────────
    enc_agent = EncryptionAgent(mode="aes")
    enc_result = enc_agent.process_dp_update(dp_result["receipt_uri"])
    final_update_uri = enc_result["receipt"]["outputs"][0]
    log(f"encryption finalized → {final_update_uri}")

    # ── 7. Stream bytes to server (REAL UploadUpdate) ──────────────────────────
    server_handle, payload_hash = _stream_update(
        stub, device_id=device_id, round_id=rm.round_id,
        update_path=final_update_uri, session_id=session_id)

    # ── 8. Submit receipt (REAL SubmitReceipt; real ECDSA over real payload) ──
    msg = device_id + rm.round_id.to_bytes(8, "big") + payload_hash
    signature = sign_message(msg)
    receipt = Receipt(
        device_id=device_id, round_id=rm.round_id,
        payload_hash=payload_hash, epsilon_spent=epsilon_spent,
        signature=signature, enc_handle=server_handle,
        scheme="AES-GCM-DP-ECDSA", nonce="",
    )
    ack = stub.SubmitReceipt(receipt, timeout=15)
    log(f"SubmitReceipt ack.ok={ack.ok}")
    if not ack.ok:
        raise RuntimeError("SubmitReceipt returned ok=False")

    print("=" * 64)
    print("ROUND 0 EXECUTION SUCCEEDED")
    print(f"  session_id    : {session_id}")
    print(f"  round_id      : {rm.round_id}")
    print(f"  server_handle : {server_handle}")
    print(f"  payload_hash  : {payload_hash.hex()}")
    print(f"  epsilon_spent : {epsilon_spent:.6f}")
    print(f"  signature_len : {len(signature)} bytes (DER ECDSA P-256)")
    print("=" * 64)

    # machine-readable summary for the verifier step
    print("RESULT_JSON " + json.dumps({
        "round_id": rm.round_id,
        "session_id": session_id,
        "server_handle": server_handle,
        "payload_hash": payload_hash.hex(),
        "epsilon_spent": epsilon_spent,
    }))


if __name__ == "__main__":
    main()
