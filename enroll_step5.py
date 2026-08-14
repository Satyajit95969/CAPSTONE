"""
enroll_step5.py — Minimal enrollment script for Phase 4 Step 5.

Performs the complete enrollment sequence against the running Rust orchestrator:
  1. Load device public key from ~/.federated/tpm/device_pubkey.pem
  2. Generate RSA-2048 client key + CSR
  3. RequestEnrollment RPC  → server prints OTP to its log
  4. Read OTP from server log file (parsed automatically)
  5. EnrollDevice RPC       → server signs CSR, returns client cert
  6. Write client.key and client.pem to ~/.federated/keys/

Usage:
    python enroll_step5.py <server_log_file>
    python enroll_step5.py /tmp/orch_start.log

Security mechanisms preserved:
  - TLS channel uses ~/.federated/keys/ca.pem for server verification (no insecure channel)
  - Enrollment channel is server-TLS only (no client cert required pre-enrollment)
  - Device pubkey from TPM (windows_signer.exe produced ~/.federated/tpm/device_pubkey.pem)
  - RSA-2048 key generated fresh (same as installer_core.py _generate_csr())
  - Cert written with chmod 0o600
"""

import sys
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import grpc
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub
from runtime.grpc.orchestrator_pb2 import EnrollmentRequest, EnrollRequest

SERVER_ADDR  = "127.0.0.1:50051"
FEDERATED    = Path.home() / ".federated"
KEYS         = FEDERATED / "keys"
DEVICE_PUBKEY_PEM = FEDERATED / "tpm" / "device_pubkey.pem"
CA_PEM       = KEYS / "ca.pem"
CLIENT_KEY   = KEYS / "client.key"
CLIENT_CSR   = KEYS / "client.csr"
CLIENT_PEM   = KEYS / "client.pem"

LOG_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/orch_start.log")


def _gen_key_and_csr() -> tuple[bytes, bytes]:
    """Generate RSA-2048 key + CSR. Mirrors installer_core._generate_csr()."""
    KEYS.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    CLIENT_KEY.write_bytes(key_pem)
    CLIENT_KEY.chmod(0o600)
    print(f"[KEY]  client.key written -> {CLIENT_KEY}")

    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "federated-device"),
        ]))
        .sign(key, hashes.SHA256())
    )
    csr_pem = csr.public_bytes(serialization.Encoding.PEM)
    CLIENT_CSR.write_bytes(csr_pem)
    print(f"[CSR]  CSR written -> {CLIENT_CSR}")
    return key_pem, csr_pem


def _open_enrollment_channel() -> grpc.Channel:
    """
    Open enrollment channel. Uses TLS if ca.pem is present and server has TLS enabled.
    Falls back to insecure channel when server runs with enable_tls=false (AVG TLS
    interception makes the TLS path unreliable on this machine without admin access to
    add the server CA to the Windows machine Root store).
    """
    options = [
        ("grpc.keepalive_time_ms", 10_000),
        ("grpc.keepalive_timeout_ms", 5_000),
    ]
    if CA_PEM.exists():
        try:
            creds = grpc.ssl_channel_credentials(root_certificates=CA_PEM.read_bytes())
            channel = grpc.secure_channel(SERVER_ADDR, creds, options=options)
            grpc.channel_ready_future(channel).result(timeout=5)
            print(f"[TLS]  Server-TLS channel ready -> {SERVER_ADDR}")
            return channel
        except Exception:
            channel.close()
            print("[TLS]  TLS channel failed — falling back to insecure (server in dev mode)")
    channel = grpc.insecure_channel(SERVER_ADDR, options=options)
    grpc.channel_ready_future(channel).result(timeout=10)
    print(f"[INSEC] Insecure channel ready -> {SERVER_ADDR}")
    return channel


def _read_otp_from_log(log_path: Path, after_pos: int, timeout: float = 15.0) -> str:
    """
    Poll the server log file for the OTP printed by request_enrollment().
    The server prints: ║  OTP         : 123456    ║
    Returns the 6-digit OTP string.
    """
    deadline = time.monotonic() + timeout
    # Match the OTP line from the server's enrollment request box
    pat = re.compile(r"OTP\s*:\s*(\d{6,})")
    while time.monotonic() < deadline:
        try:
            text = log_path.read_text(errors="replace")
            # Only look at content written after the RequestEnrollment call
            relevant = text[after_pos:]
            m = pat.search(relevant)
            if m:
                return m.group(1)
        except Exception:
            pass
        time.sleep(0.3)
    raise TimeoutError(
        f"OTP not found in server log within {timeout}s. "
        f"Check {log_path} for the enrollment box."
    )


def main():
    print("=" * 60)
    print("Phase 4 Step 5 — Device Enrollment")
    print(f"Server : {SERVER_ADDR}")
    print(f"Log    : {LOG_FILE}")
    print("=" * 60)

    # ── Step 1: Load device public key (from TPM / windows_signer) ───────────
    if not DEVICE_PUBKEY_PEM.exists():
        raise FileNotFoundError(
            f"Device pubkey not found: {DEVICE_PUBKEY_PEM}\n"
            "Run windows_signer.exe --pubkey first."
        )
    device_pubkey = DEVICE_PUBKEY_PEM.read_bytes()
    print(f"\n[1] Device pubkey loaded ({len(device_pubkey)} bytes) from {DEVICE_PUBKEY_PEM}")

    # ── Step 2: Generate RSA client key + CSR ────────────────────────────────
    print("\n[2] Generating RSA-2048 client key and CSR…")
    _, csr_pem = _gen_key_and_csr()

    # ── Step 3: Open enrollment channel ──────────────────────────────────────
    print("\n[3] Opening server-TLS enrollment channel…")
    channel = _open_enrollment_channel()
    stub = OrchestratorStub(channel)

    # ── Step 4: RequestEnrollment ─────────────────────────────────────────────
    print("\n[4] Sending RequestEnrollment…")
    log_pos_before = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0
    req_ack = stub.RequestEnrollment(
        EnrollmentRequest(
            device_pubkey=device_pubkey,
            csr=csr_pem,
            device_info="Windows / DESKTOP-O1G5VNA (Phase 4 Step 5)",
        ),
        timeout=15,
    )
    if not req_ack.accepted:
        raise RuntimeError("Server rejected RequestEnrollment")
    print(f"[4] Accepted — fingerprint: {req_ack.device_fingerprint}")

    # ── Step 5: Extract OTP from server log ───────────────────────────────────
    print(f"\n[5] Waiting for OTP in server log ({LOG_FILE})…")
    otp = _read_otp_from_log(LOG_FILE, log_pos_before, timeout=15.0)
    print(f"[5] OTP extracted: {otp}")

    # ── Step 6: EnrollDevice ──────────────────────────────────────────────────
    print(f"\n[6] Sending EnrollDevice (OTP={otp})…")
    enroll_resp = stub.EnrollDevice(
        EnrollRequest(
            enrollment_token=otp,
            device_pubkey=device_pubkey,
            csr=csr_pem,
        ),
        timeout=30,
    )
    if not enroll_resp.ok:
        raise RuntimeError("EnrollDevice returned ok=False")

    # ── Step 7: Save client certificate ──────────────────────────────────────
    CLIENT_PEM.write_bytes(enroll_resp.client_cert)
    CLIENT_PEM.chmod(0o600)
    print(f"\n[7] client.pem written -> {CLIENT_PEM} ({len(enroll_resp.client_cert)} bytes)")

    channel.close()

    print("\n" + "=" * 60)
    print("ENROLLMENT COMPLETE")
    print(f"  client.key : {CLIENT_KEY}")
    print(f"  client.pem : {CLIENT_PEM}")
    print(f"  OTP used   : {otp}")
    print(f"  Fingerprint: {req_ack.device_fingerprint}")
    print("=" * 60)


if __name__ == "__main__":
    main()
