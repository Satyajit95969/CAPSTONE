"""
aggregator.py

SECURITY FIXES:
  FIX-AGG-1: SecureStore root is now the canonical path
              (~/.federated/data/secure_store) matching all other agents.
              Previously it was "./secure_store" (relative CWD), which means
              the aggregator was using a completely different master.key than
              the agents that encrypted the data — decryption would always fail
              or silently use wrong keys.

  FIX-AGG-2: enc_uri input is now validated to be either a GridFS ObjectId
              (preferred, from UploadUpdate flow) or a canonical secure store
              path. Arbitrary file paths from clients are rejected to prevent
              path traversal attacks.

  FIX-AGG-3: When operating in GridFS mode, the aggregator fetches bytes from
              MongoDB using the ObjectId — it no longer relies on file paths
              that clients submitted.

  FIX-AGG-4 (GAP 1): _decrypt_from_gridfs() now properly decrypts the
              SecureStore AES-GCM envelope before calling torch.load().
              Previously it called torch.load() directly on ciphertext (which
              raised, then silently fell back to np.frombuffer on the ciphertext
              bytes), causing every aggregation round to operate on garbage.
              The silent np.frombuffer fallback has been removed from both
              _decrypt_from_gridfs() and _decrypt_from_store() — decryption
              failures now raise ValueError with a diagnostic message.

  FIX-AGG-5 (GAP 3): Structure-preserving federated averaging.
              Previously _decrypt_from_gridfs() flattened the decrypted
              state dict into a 1-D numpy vector (discarding all key names
              and shapes), aggregation averaged the flat vector, and run_job()
              saved a bare 1-D torch.Tensor — clients calling
              model.load_state_dict() on the downloaded global model would
              always fail with a type error.

              Now:
              - _decrypt_from_gridfs() and _decrypt_from_store() return
                Dict[str, torch.Tensor] (the full state dict, structure intact).
              - aggregate_updates() collects state dicts and dispatches to
                _aggregate_state_dicts(), which validates that all clients
                share identical key sets and per-key tensor shapes (raises
                ValueError on any mismatch — no silent truncation).
              - _aggregate_tensor() performs mean/trimmed_mean/median in
                pure PyTorch float32, avoiding the numpy float64 promotion
                bug present in the old np.mean() call.
              - Non-floating buffers (e.g. position_ids int64) are verified
                to be identical across clients and taken from client 0.
              - run_job() saves the aggregated state dict as a .pt file and
                uploads it to GridFS — clients can torch.load() and call
                model.load_state_dict() directly.
"""

import base64
import os
import io
import json
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union

from server.aggregator_agent.core import reporting as rpt
# CRITICAL: this process's stdout is a machine-parsed contract — the Rust
# orchestrator reads exactly one JSON line from it (serde_json::from_slice).
# All structured reporting below must go to stderr, never stdout.
rpt.set_default_stream(sys.stderr)

# Canonical paths — must match _CANONICAL_ROOT in centralized_secure_store.py
_FEDERATED_BASE  = Path.home() / ".federated"
_CANONICAL_ROOT  = _FEDERATED_BASE / "data" / "secure_store"
_GLOBAL_KEY_PATH = _CANONICAL_ROOT / "master.key"   # shared master key

# MongoDB connection string — set via environment variable, never hardcoded
_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
# FIX-MULTIMODAL-3: database name, from MONGO_DATABASE env var (defaults to
# "federated" — matches the Rust orchestrator's own default in main.rs).
# The Rust orchestrator spawns this subprocess inheriting its own process
# environment, so whatever MONGO_DATABASE it was started with is already
# visible here — this was the missing half of the database-isolation fix;
# the Rust side alone was not enough because this file independently
# hardcoded "federated" in two places (GridFS read + GridFS write).
_MONGO_DATABASE = os.environ.get("MONGO_DATABASE", "federated")


class AggregatorAgent:
    """
    AggregatorAgent
    ---------------
    Receives encrypted DP-updates (via GridFS ObjectId or canonical path),
    decrypts them using the shared SecureStore, and performs robust aggregation.

    Aggregation modes: mean | trimmed_mean | coordinate_median
    """

    def __init__(
        self,
        mode: str = "trimmed_mean",
        trim_ratio: float = 0.1,
        decrypt_callback=None,
    ):
        self.mode         = mode
        self.trim_ratio   = trim_ratio
        self._decrypt_cb  = decrypt_callback or self._default_decrypt

    def _default_decrypt(
        self,
        gridfs_id: Optional[str],
        enc_path:  Optional[str],
        scheme:    str,
        nonce:     Optional[str],
    ) -> np.ndarray:
        """
        Decrypt an update.

        Priority:
          1. If gridfs_id is set, fetch from MongoDB GridFS (preferred).
          2. If enc_path is set, decrypt from canonical SecureStore path.
          3. Reject anything else.
        """
        # FIX-AGG-3: GridFS path (set by UploadUpdate flow)
        if gridfs_id:
            return self._decrypt_from_gridfs(gridfs_id)

        # FIX-AGG-2: Canonical path validation — reject client-supplied paths
        if enc_path:
            return self._decrypt_from_store(enc_path, scheme)

        raise ValueError("Neither gridfs_id nor enc_path provided")

    # ── FIX-AGG-4: in-memory SecureStore AES-GCM decryption ──────────────────

    def _decrypt_secure_store_envelope(self, raw_bytes: bytes) -> bytes:
        """
        Decrypt a SecureStore AES-GCM envelope from raw bytes in memory.

        The client pipeline stores encrypted model updates as JSON envelopes:
          Canonical format:  {"agent": str, "context": str,
                              "nonce": base64, "ct": base64}
          Legacy format:     {"nonce": base64, "ct": base64}

        Key derivation mirrors SecureStore._derive_key():
          HKDF-SHA256(master_key, info=b"{agent}:{context}")

        The master key is read from _GLOBAL_KEY_PATH, which must be the same
        key used by the uploading clients.

        Raises ValueError with a diagnostic message on any failure.
        NEVER falls back to returning ciphertext bytes as model parameters.
        """
        # ── Step 1: parse JSON envelope ───────────────────────────────────────
        try:
            envelope = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"GridFS payload is not a valid SecureStore JSON envelope "
                f"(parse error: {exc}). "
                "The data may be corrupted or was uploaded without SecureStore "
                "encryption. Refusing to treat raw bytes as model parameters."
            ) from exc

        missing = [f for f in ("nonce", "ct") if f not in envelope]
        if missing:
            raise ValueError(
                f"SecureStore envelope is missing required fields {missing}. "
                f"Found keys: {sorted(envelope.keys())}. "
                "Cannot decrypt — upload is malformed."
            )

        nonce_bytes = base64.b64decode(envelope["nonce"])
        ct_bytes    = base64.b64decode(envelope["ct"])

        # ── Step 2: load the canonical master key ─────────────────────────────
        if not _GLOBAL_KEY_PATH.exists():
            raise RuntimeError(
                f"Aggregator master key not found at {_GLOBAL_KEY_PATH}. "
                "The server must share the same master.key used by the uploading "
                "clients. In production this key is distributed by the KMS during "
                "device enrollment."
            )
        master_key_b64 = _GLOBAL_KEY_PATH.read_text().strip()
        try:
            master_key = base64.b64decode(master_key_b64)
        except Exception:
            master_key = _GLOBAL_KEY_PATH.read_bytes()

        # ── Step 3: build candidate (agent, context) pairs ───────────────────
        # The canonical SecureStore format embeds agent+context in the envelope,
        # so we can reproduce the exact HKDF key used at write time.
        # The legacy root-level format omits them; we try all known pipeline
        # agent names in priority order.
        if "agent" in envelope and "context" in envelope:
            candidates = [(envelope["agent"], envelope["context"])]
        else:
            # Legacy format: try every agent that the client pipeline may have used.
            candidates = [
                ("trainer",  ""),   # pipeline.py: SecureStore(agent="trainer", ...)
                ("dp-agent", ""),   # standalone DPAgent default
                ("enc-agent", ""),  # standalone EncryptionAgent default
                ("generic",  ""),   # SecureStore() default
            ]

        # ── Step 4: attempt HKDF key derivation and AES-GCM decryption ───────
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes as _hashes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        last_exc: Optional[Exception] = None
        for agent_name, context in candidates:
            info   = f"{agent_name}:{context}".encode()
            hkdf   = HKDF(
                algorithm=_hashes.SHA256(),
                length=32,
                salt=None,
                info=info,
                backend=default_backend(),
            )
            key    = hkdf.derive(master_key)
            aesgcm = AESGCM(key)
            try:
                return aesgcm.decrypt(nonce_bytes, ct_bytes, None)
            except Exception as exc:
                last_exc = exc
                continue

        tried = [(a, c) for a, c in candidates]
        raise ValueError(
            f"AES-GCM decryption failed for all {len(candidates)} candidate key(s). "
            f"Agents/contexts tried: {tried}. "
            f"master_key_path={_GLOBAL_KEY_PATH}. "
            f"Last error: {last_exc}. "
            "Ensure the aggregator and uploading clients share the same master.key "
            "and that the HKDF agent/context labels match."
        )

    def _decrypt_from_gridfs(self, gridfs_id: str) -> np.ndarray:
        """
        Fetch and decrypt a SecureStore-encrypted update from MongoDB GridFS.

        The bytes stored in GridFS are the SecureStore JSON envelope produced by
        the client's SecureStore.encrypt_write() call (not raw tensor bytes).
        We must decrypt with _decrypt_secure_store_envelope() before the payload
        can be deserialized as a model state dict.

        Raises ValueError/RuntimeError with a clear diagnostic on any failure.
        Does NOT fall back to interpreting ciphertext as model parameters.
        """
        from pymongo import MongoClient
        from bson.objectid import ObjectId
        import gridfs

        client = MongoClient(_MONGO_URI)
        db     = client[_MONGO_DATABASE]
        fs     = gridfs.GridFS(db)

        try:
            oid = ObjectId(gridfs_id)
        except Exception:
            raise ValueError(
                f"Invalid GridFS ObjectId: {gridfs_id!r}. "
                "enc_handle must be the server_handle returned by UploadAck, "
                "not a file path."
            )

        grid_out = fs.get(oid)
        raw      = grid_out.read()
        client.close()

        if not raw:
            raise ValueError(
                f"GridFS object {gridfs_id!r} is empty — "
                "the upload may have been corrupted."
            )

        # FIX-AGG-4: decrypt the SecureStore AES-GCM envelope.
        # raw = JSON bytes: {"agent":..., "context":..., "nonce":..., "ct":...}
        # _decrypt_secure_store_envelope() reproduces the HKDF key used at
        # write time and returns the plaintext model bytes.
        # Raises ValueError with a diagnostic message on failure — never falls
        # back to treating ciphertext as model parameters.
        plaintext = self._decrypt_secure_store_envelope(raw)

        buf = io.BytesIO(plaintext)
        try:
            tensor = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(
                f"Decryption succeeded but torch.load() failed on the plaintext "
                f"for GridFS object {gridfs_id!r}. Cause: {exc}. "
                "The update may have been serialized in an incompatible format."
            ) from exc

        if isinstance(tensor, torch.Tensor):
            # Raw 1-D tensor — returned as numpy for the legacy flat-vector path.
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, dict):
            # FIX-AGG-5 (GAP 3): return the full state dict with key names and
            # shapes intact. Flattening is no longer done here — key-by-key
            # FedAvg happens in _aggregate_state_dicts().
            state_dict_cpu: Dict[str, torch.Tensor] = {
                k: v.detach().cpu()
                for k, v in tensor.items()
                if isinstance(v, torch.Tensor)
            }
            if not state_dict_cpu:
                raise ValueError(
                    f"State dict from GridFS object {gridfs_id!r} contains no "
                    "tensor values — cannot aggregate an empty update."
                )
            return state_dict_cpu
        else:
            raise TypeError(
                f"Unexpected object type in GridFS object {gridfs_id!r}: "
                f"{type(tensor).__name__}. Expected torch.Tensor or state dict."
            )

    def _decrypt_from_store(self, enc_path: str, scheme: str) -> np.ndarray:
        """
        Decrypt from SecureStore using canonical root.

        FIX-AGG-1: Uses _CANONICAL_ROOT so the same master.key is used as
        all other agents. Previously used "./secure_store" (relative CWD).

        FIX-AGG-2: Validates enc_path is inside _CANONICAL_ROOT to prevent
        path traversal.

        FIX-AGG-4: Removed the silent np.frombuffer fallback — torch.load()
        failures now raise ValueError with a diagnostic message instead of
        returning ciphertext bytes interpreted as float32 values.
        """
        from server.aggregator_agent.core.centralized_secure_store import SecureStore

        if enc_path.startswith("file://"):
            enc_path = enc_path[len("file://"):]

        # Resolve and validate — reject anything outside canonical root
        resolved = Path(enc_path).resolve()
        if not str(resolved).startswith(str(_CANONICAL_ROOT.resolve())):
            raise ValueError(
                f"Path traversal rejected: {resolved}\n"
                f"Paths must be inside {_CANONICAL_ROOT}"
            )

        # FIX-AGG-1: canonical root
        store = SecureStore(agent="aggregator", root=_CANONICAL_ROOT)

        if scheme.lower().startswith("aes") or scheme.lower().startswith("kms"):
            raw = store.decrypt_read("file://" + str(resolved))
        else:
            raise ValueError(f"Unsupported scheme for path-based decrypt: {scheme}")

        buf = io.BytesIO(raw)
        try:
            obj = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(
                f"SecureStore decryption succeeded but torch.load() failed on the "
                f"plaintext from {resolved}. Cause: {exc}. "
                "The update may be in an incompatible serialization format."
            ) from exc

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            # FIX-AGG-5 (GAP 3): return full state dict — do not flatten.
            state_dict_cpu: Dict[str, torch.Tensor] = {
                k: v.detach().cpu()
                for k, v in obj.items()
                if isinstance(v, torch.Tensor)
            }
            if not state_dict_cpu:
                raise ValueError(
                    f"State dict from {resolved} contains no tensor values — "
                    "cannot aggregate an empty update."
                )
            return state_dict_cpu
        else:
            raise TypeError(
                f"Unexpected object type from {resolved}: {type(obj).__name__}. "
                "Expected torch.Tensor or state dict."
            )

    def aggregate_updates(
        self, updates: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Decrypt all client updates and aggregate them into a single state dict.

        FIX-AGG-5 (GAP 3): Each update must decrypt to a full state dict
        (Dict[str, torch.Tensor]).  Raw flat tensors are rejected — the
        standard pipeline always produces structured state dicts via
        dp_agent.unflatten_state_dict() followed by torch.save(state_dict).

        Raises ValueError if decryption fails, if key sets differ across
        clients, or if per-key tensor shapes are inconsistent.  Does NOT
        silently truncate or pad mismatched updates.
        """
        state_dicts: List[Dict[str, torch.Tensor]] = []

        rpt.header("FEDERATED SERVER — AGGREGATION")
        rpt.kv("Updates received", len(updates))
        rpt.subheader("CLIENT UPDATES")

        for idx, u in enumerate(updates):
            # FIX-AGG-3: prefer gridfs_id over local enc_uri
            gridfs_id = u.get("gridfs_id")
            enc_path  = u.get("enc_uri")
            scheme    = u.get("scheme", "AES-GCM-SecureStore")
            nonce     = u.get("nonce")

            rpt.line(f"Client {idx}")
            rpt.kv("gridfs_id", gridfs_id or "(none)", indent=2)
            rpt.kv("scheme", scheme, indent=2)

            try:
                result = self._decrypt_cb(gridfs_id, enc_path, scheme, nonce)
                rpt.kv("Hash/decrypt", "PASS", indent=2)
            except Exception as e:
                rpt.kv("Hash/decrypt", f"FAIL ({e})", indent=2)
                raise

            if isinstance(result, dict):
                state_dicts.append(result)
                n_tensors = len(result)
                total_numel = sum(v.numel() for v in result.values())
                rpt.kv("Accepted", "YES", indent=2)
                rpt.kv("Parameter tensors", n_tensors, indent=2)
                rpt.kv("Total parameters", f"{total_numel:,}", indent=2)
            elif isinstance(result, (np.ndarray, torch.Tensor)):
                shape = tuple(getattr(result, "shape", ()))
                raise TypeError(
                    f"Update {idx} (gridfs_id={gridfs_id!r}) decrypted to a raw "
                    f"tensor of shape {shape} instead of a state dict. "
                    "The standard pipeline always produces structured state dicts. "
                    "Ensure dp_agent.process_local_update() ran before upload and "
                    "that the model delta was saved with torch.save(state_dict, ...)."
                )
            else:
                raise TypeError(
                    f"Update {idx} (gridfs_id={gridfs_id!r}) decrypted to "
                    f"unexpected type {type(result).__name__!r}. "
                    "Expected Dict[str, torch.Tensor]."
                )

        if not state_dicts:
            raise ValueError("No updates to aggregate — updates list is empty.")

        return self._aggregate_state_dicts(state_dicts)

    # ── FIX-AGG-5: structure-preserving key-by-key FedAvg ────────────────────

    def _aggregate_state_dicts(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameter-wise federated averaging over a list of client state dicts.

        Validation (fail-loud):
          - All clients must have identical sorted key sets.
          - All clients must have identical tensor shapes for every key.
          - Non-floating buffers (e.g. position_ids int64) must be byte-identical
            across clients; the first client's value is used in the output.

        Aggregation:
          - Floating-point parameters are aggregated with _aggregate_tensor()
            (mean / trimmed_mean / coordinate_median) in float32 then cast
            back to the original dtype.
          - No numpy involved — avoids the float64 promotion bug in np.mean().
        """
        n_clients = len(state_dicts)
        if n_clients == 0:
            raise ValueError("_aggregate_state_dicts requires at least one state dict.")

        # ── 1. Key-set validation ─────────────────────────────────────────────
        reference_keys = sorted(state_dicts[0].keys())

        for i, sd in enumerate(state_dicts[1:], start=1):
            client_keys = sorted(sd.keys())
            if client_keys != reference_keys:
                extra   = sorted(set(client_keys)   - set(reference_keys))
                missing = sorted(set(reference_keys) - set(client_keys))
                raise ValueError(
                    f"Client {i} state dict has incompatible keys. "
                    f"Extra:   {extra}. "
                    f"Missing: {missing}. "
                    "All clients must use an identical model architecture "
                    "(same bert_name, audio_dim, vision_dim)."
                )

        rpt.subheader("AGGREGATION")
        algo_names = {
            "mean": "Unweighted federated averaging (mean)",
            "trimmed_mean": "Coordinate-wise trimmed mean (robust FedAvg)",
            "median": "Coordinate-wise median (robust)",
            "coordinate_median": "Coordinate-wise median (robust)",
        }
        rpt.kv("Algorithm", algo_names.get(self.mode, self.mode))
        rpt.kv("Number of clients", n_clients)
        if self.mode == "trimmed_mean":
            rpt.kv("Trim ratio", self.trim_ratio)
        rpt.kv("Parameter tensors (keys)", len(reference_keys))

        # Pick up to 2 small representative parameters to show a concrete,
        # per-client -> aggregated worked example (never the full model).
        _shown = 0
        _MAX_SHOWN = 2

        # ── 2. Per-key shape validation and aggregation ───────────────────────
        aggregated: Dict[str, torch.Tensor] = {}

        for key in reference_keys:
            ref = state_dicts[0][key].detach().cpu()

            # Shape check — reject mismatches before attempting aggregation
            for i, sd in enumerate(state_dicts[1:], start=1):
                t = sd[key].detach().cpu()
                if t.shape != ref.shape:
                    raise ValueError(
                        f"Shape mismatch for parameter '{key}': "
                        f"client 0 has {tuple(ref.shape)}, "
                        f"client {i} has {tuple(t.shape)}. "
                        "Clients must share identical audio_dim, vision_dim, "
                        "and BERT hidden_size."
                    )

            if ref.dtype.is_floating_point:
                # ── Float parameters ─────────────────────────────────────────
                # Upcast to float32 for numerically stable aggregation,
                # then cast back to the original dtype (e.g. float16, bfloat16).
                tensors = [
                    sd[key].detach().cpu().to(torch.float32)
                    for sd in state_dicts
                ]
                stacked  = torch.stack(tensors, dim=0)   # (N, *shape)
                agg_f32  = self._aggregate_tensor(stacked)
                aggregated[key] = agg_f32.to(ref.dtype)

                if _shown < _MAX_SHOWN:
                    rpt.line(f"Parameter '{key}':")
                    for ci, t in enumerate(tensors):
                        rpt.tensor_summary(f"  Client {ci}", t, indent=2)
                    rpt.tensor_summary("  Aggregated", agg_f32, indent=2)
                    _shown += 1

            else:
                # ── Non-floating buffers (int64, bool, …) ────────────────────
                # Averaging integer indices is semantically meaningless.
                # All clients must agree on the buffer's value; any discrepancy
                # signals a structural inconsistency between the model instances.
                for i, sd in enumerate(state_dicts[1:], start=1):
                    other = sd[key].detach().cpu()
                    if not torch.equal(ref, other):
                        raise ValueError(
                            f"Non-floating buffer '{key}' (dtype={ref.dtype}) "
                            f"differs between client 0 and client {i}. "
                            "This indicates a structural inconsistency between "
                            "the client model instances."
                        )
                aggregated[key] = ref.clone()

        total_params = sum(t.numel() for t in aggregated.values())
        rpt.line()
        rpt.kv("Total parameter tensors processed", len(aggregated))
        rpt.kv("Total scalar parameters aggregated", f"{total_params:,}")
        rpt.kv("Failed", 0)  # fail-loud design: reaching here means zero failures
        rpt.ok("Round aggregation complete")

        return aggregated

    def _aggregate_tensor(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Aggregate a (N, *param_shape) float32 tensor along the client axis (dim 0).

        Returns a tensor of shape *param_shape in float32.
        All operations run in PyTorch on CPU — no numpy, no float64 promotion.
        """
        N = stacked.shape[0]

        if self.mode == "mean":
            return stacked.mean(dim=0)

        elif self.mode == "trimmed_mean":
            lower = max(1, int(self.trim_ratio * N))
            upper = N - lower
            if lower >= upper:
                raise ValueError(
                    f"trim_ratio {self.trim_ratio} too large for {N} client(s) "
                    f"(lower={lower}, upper={upper}). "
                    "Reduce trim_ratio or collect more client updates before "
                    "triggering aggregation."
                )
            # Flatten to (N, D), sort per-coordinate across clients, trim, mean.
            orig_shape  = stacked.shape[1:]
            flat        = stacked.reshape(N, -1)             # (N, D)
            sorted_flat, _ = torch.sort(flat, dim=0)         # ascending per coord
            trimmed     = sorted_flat[lower:upper, :]        # (upper-lower, D)
            return trimmed.mean(dim=0).reshape(orig_shape)

        elif self.mode in ("median", "coordinate_median"):
            # torch.median with dim= returns namedtuple (values, indices).
            # Flatten to (N, D) first so the median is coordinate-wise.
            orig_shape = stacked.shape[1:]
            flat       = stacked.reshape(N, -1)              # (N, D)
            return flat.median(dim=0).values.reshape(orig_shape)

        else:
            raise NotImplementedError(
                f"Unknown aggregation mode: {self.mode!r}. "
                "Supported: 'mean', 'trimmed_mean', 'coordinate_median'/'median'."
            )

    def _apply_aggregation(self, arr: np.ndarray) -> np.ndarray:
        """
        Legacy numpy-based aggregation used only by custom decrypt_callback
        overrides that still return np.ndarray.  Not called by the standard
        aggregate_updates() → _aggregate_state_dicts() path.
        """
        if self.mode == "mean":
            return np.mean(arr, axis=0)

        elif self.mode == "trimmed_mean":
            n     = arr.shape[0]
            lower = max(1, int(self.trim_ratio * n))
            upper = n - lower
            if lower >= upper:
                raise ValueError(
                    f"trim_ratio {self.trim_ratio} too large for {n} updates"
                )
            sorted_arr = np.sort(arr, axis=0)
            return np.mean(sorted_arr[lower:upper], axis=0)

        elif self.mode in ("median", "coordinate_median"):
            return np.median(arr, axis=0)

        else:
            raise NotImplementedError(f"Unknown aggregation mode: {self.mode}")

    def run_job(self, job: Dict) -> Dict:
        import hashlib
        import gridfs as _gridfs
        from pymongo import MongoClient as _MongoClient

        self.mode       = job.get("mode",       self.mode)
        self.trim_ratio = job.get("trim_ratio", self.trim_ratio)

        # FIX-AGG-5 (GAP 3): aggregate_updates() now returns a full state dict.
        aggregated: Dict[str, torch.Tensor] = self.aggregate_updates(job["updates"])

        # Save the aggregated state dict as a local diagnostic artifact.
        # .pt (not .npy) because the result is a Dict[str, Tensor], not an array.
        out_path = f"./aggregated_round_{job['round_id']}.pt"
        torch.save(aggregated, out_path)

        # Serialise the full state dict for GridFS upload.
        # torch.save(state_dict, buf) produces a standard PyTorch pickle that
        # clients can load with:
        #   global_sd = torch.load(path)          # returns Dict[str, Tensor]
        #   model.load_state_dict(global_sd)       # works correctly
        model_buf   = io.BytesIO()
        torch.save(aggregated, model_buf)
        model_bytes = model_buf.getvalue()
        model_hash  = hashlib.sha256(model_bytes).hexdigest()

        # Upload to MongoDB GridFS so the Rust orchestrator can serve the model
        # via DownloadGlobalModel without touching the local filesystem.
        _client = _MongoClient(_MONGO_URI)
        _db     = _client[_MONGO_DATABASE]
        _fs     = _gridfs.GridFS(_db)
        file_id = _fs.put(
            model_bytes,
            filename=f"global_model_round_{job['round_id']}.pt",
        )

        rpt.subheader("GLOBAL MODEL")
        rpt.kv("Round ID", job["round_id"])
        rpt.kv("Aggregation mode", self.mode)
        rpt.kv("Model size", f"{len(model_bytes)} bytes")
        rpt.kv("Model hash (SHA-256)", model_hash)
        rpt.kv("Parameter tensors (keys)", len(aggregated))
        rpt.kv("Stored in GridFS", f"YES (file_id={file_id})")

        try:
            coll_names = _db.list_collection_names()
            rpt.subheader("MONGODB")
            for cname in coll_names:
                rpt.kv(cname, _db[cname].count_documents({}), indent=2)
        except Exception as e:
            rpt.warn(f"Could not enumerate MongoDB collections: {e}")

        _client.close()
        rpt.ok("Global model generated")

        return {
            "round_id":         job["round_id"],
            "aggregated_uri":   "file://" + os.path.abspath(out_path),
            "num_updates":      len(job["updates"]),
            "mode":             self.mode,
            "gridfs_file_id":   str(file_id),
            "model_hash":       model_hash,
            "num_keys":         len(aggregated),
            "total_parameters": sum(t.numel() for t in aggregated.values()),
        }


if __name__ == "__main__":
    import sys
    job    = json.load(sys.stdin)
    agent  = AggregatorAgent(
        mode=job.get("mode", "trimmed_mean"),
        trim_ratio=job.get("trim_ratio", 0.1),
    )
    result = agent.run_job(job)
    print(json.dumps(result))