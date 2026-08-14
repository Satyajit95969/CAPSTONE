"""
step7_download_validate.py — Phase 4 Step 7: validate DownloadGlobalModel for round 2.

Exercises the ORIGINAL client download path:

    GetRound (round 2, global_model_available=True)
        -> DownloadGlobalModel(round_id=2)        [REAL server RPC, streaming]
        -> per-chunk SHA-256 verification          [verbatim from pipeline.py]
        -> full-model SHA-256 verification         [verbatim from pipeline.py]
        -> torch.load()                            [loadability]
        -> model.load_state_dict(strict=True)      [structural correctness, GAP-3 contract]

The _download_global_model() body below is copied verbatim from
runtime/pipeline.py (FIX-PIPELINE-6) so the exact production verification code
runs.  The only addition is the dummy architecture used to prove load_state_dict
— it mirrors the substitute trainer output (fc1: 768->384, fc2: 384->1).
"""

import os, hashlib, math
from pathlib import Path

import grpc
import torch
import torch.nn as nn

from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub
from runtime.grpc.orchestrator_pb2 import DeviceId, RoundRequest

ADDR      = os.environ.get("FED_SERVER", "127.0.0.1:50051")
FEDERATED = Path.home() / ".federated"
KEYS      = FEDERATED / "keys"
TPM_PUBKEY = FEDERATED / "tpm" / "device_pubkey.pem"
CA_BUNDLE = Path("ca_plus_avg.pem")


def log(m): print(f"[step7-dl] {m}", flush=True)


def make_stub():
    creds = grpc.ssl_channel_credentials(
        root_certificates=CA_BUNDLE.read_bytes(),
        private_key=(KEYS / "client.key").read_bytes(),
        certificate_chain=(KEYS / "client.pem").read_bytes(),
    )
    ch = grpc.secure_channel(ADDR, creds,
                             options=[("grpc.keepalive_time_ms", 10000),
                                      ("grpc.keepalive_timeout_ms", 5000)])
    grpc.channel_ready_future(ch).result(timeout=10)
    return OrchestratorStub(ch)


# ── verbatim from runtime/pipeline.py _download_global_model (FIX-PIPELINE-6) ──
def _download_global_model(stub, device_id: bytes, round_id: int):
    request = RoundRequest(device_id=device_id, round_id=round_id)
    chunks_received = []
    full_model_hash_expected = None
    n_chunks = 0

    for chunk in stub.DownloadGlobalModel(request, timeout=120):
        computed = hashlib.sha256(chunk.data).digest()
        if computed != bytes(chunk.chunk_hash):
            raise ValueError(
                f"Global model chunk {chunk.chunk_index} hash mismatch — data corrupted")
        chunks_received.append(chunk.data)
        n_chunks += 1
        log(f"chunk {chunk.chunk_index}/{chunk.total_chunks-1} verified "
            f"({len(chunk.data)} B, sha256={computed.hex()[:16]}…)")
        if chunk.chunk_index == chunk.total_chunks - 1 and chunk.model_hash:
            full_model_hash_expected = bytes(chunk.model_hash)

    if not chunks_received:
        raise ValueError("No global model chunks received")

    model_bytes = b"".join(chunks_received)

    if full_model_hash_expected:
        actual = hashlib.sha256(model_bytes).digest()
        if actual != full_model_hash_expected:
            raise ValueError("Global model full-hash mismatch — model rejected")
        log(f"FULL-MODEL HASH VERIFIED: {actual.hex()}")
    else:
        raise ValueError("Server did not provide a full model_hash in the final chunk")

    return model_bytes, n_chunks


def main():
    print("=" * 64)
    print("PHASE 4 STEP 7 — DownloadGlobalModel(round 2) VALIDATION")
    print("=" * 64)

    device_id = hashlib.sha256(TPM_PUBKEY.read_bytes()).digest()
    stub = make_stub()

    rm = stub.GetRound(DeviceId(id=device_id), timeout=10)
    log(f"GetRound: round_id={rm.round_id} state={rm.state} "
        f"global_model_available={rm.global_model_available}")
    if rm.round_id != 2:
        raise RuntimeError(f"expected round 2 active, got {rm.round_id}")
    if not rm.global_model_available:
        raise RuntimeError("global_model_available is False — aggregation did not publish a model")

    model_bytes, n_chunks = _download_global_model(stub, device_id, round_id=2)
    log(f"downloaded {len(model_bytes)} bytes in {n_chunks} chunk(s)")

    # ── loadability ──
    import io
    sd = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=False)
    log(f"torch.load() OK — type={type(sd).__name__}, keys={list(sd.keys())}")

    # ── load_state_dict on the matching dummy architecture ──
    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, 384)
            self.fc2 = nn.Linear(384, 1)

    model = Probe()
    missing, unexpected = [], []
    res = model.load_state_dict(sd, strict=True)
    # strict=True raises on mismatch; if we reach here it succeeded
    log("model.load_state_dict(strict=True) OK — no missing/unexpected keys")

    # sanity: parameter equality after load
    for k, v in model.state_dict().items():
        assert torch.equal(v, sd[k]), f"param {k} not loaded faithfully"
    log("post-load parameter equality verified for all 4 tensors")

    print("=" * 64)
    print("DOWNLOAD + LOAD VALIDATION SUCCEEDED")
    print(f"  round_id          : 2")
    print(f"  bytes             : {len(model_bytes)}")
    print(f"  full_model_sha256 : {hashlib.sha256(model_bytes).hexdigest()}")
    print(f"  keys              : {list(sd.keys())}")
    print("=" * 64)


if __name__ == "__main__":
    main()
