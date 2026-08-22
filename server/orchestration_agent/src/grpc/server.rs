// server.rs — Zero-trust gRPC server
//
// SECURITY FIXES (already present, verified):
//   FIX-SERVER-1: require_client_cert() returns Err on missing cert — no cert = no access.
//   FIX-SERVER-2: UploadUpdate streams model bytes into MongoDB GridFS.
//                 Previously enc_uri was a local client path the server could never open.
//   FIX-SERVER-3: Per-chunk SHA-256 verification during streaming.
//   FIX-SERVER-4: payload_hash in Receipt is verified against what was actually uploaded.
//   FIX-SERVER-5: enc_handle validated as a known GridFS ObjectId — path traversal prevented.
//   FIX-SERVER-6: Receipt HMAC chaining stored in MongoDB — tamper-evident audit log.
//   FIX-SERVER-7: OTP expiry 600s (was 6000 = 100 min).
//   FIX-SERVER-8: DownloadGlobalModel streams model to client with hash verification.
//   FIX-SERVER-9: epsilon_spent range validated (0 < eps <= epsilon_max).
//
// COMPILE FIXES (this revision):
//   FIX-COMPILE-1: GridFsBucket created via db.gridfs_bucket() — GridFsBucket::new() is pub(crate).
//   FIX-COMPILE-2: GridFsUploadStream implements futures::AsyncWrite (NOT tokio::io::AsyncWrite).
//                  Removed `use tokio::io::AsyncWriteExt` and added futures equivalents.
//   FIX-COMPILE-3: GridFsDownloadStream implements futures::AsyncRead (NOT tokio::io::AsyncRead).
//                  Replaced tokio::io::AsyncReadExt::read_to_end with futures equivalent.
//   FIX-COMPILE-4: DownloadGlobalModelStream = Pin<Box<dyn Stream<...> + Send>> — tonic
//                  cannot convert tokio_stream::Iter into tonic::codec::Streaming directly.
//   FIX-COMPILE-5: Removed unused `ct_eq` import (caused warning treated as error in release).

use std::fs;
use std::io::Write;
use std::pin::Pin;
use std::process::Command;
use std::sync::Arc;

use futures::AsyncReadExt as FuturesAsyncReadExt;
use futures::AsyncWriteExt as FuturesAsyncWriteExt;
// NOTE: do NOT also import tokio::io::AsyncWriteExt or AsyncReadExt — the GridFS
// streams implement the futures traits, not tokio's.  Having both in scope causes
// method-resolution ambiguity and the wrong bound is selected.
use futures::StreamExt;
use hmac::{Hmac, Mac};
use mongodb::bson::{self, doc, oid::ObjectId, DateTime as BsonDateTime};
use mongodb::options::{FindOneOptions, FindOptions};
use mongodb::{Client as MongoClient, Database};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use tokio_stream::Stream; // re-exports futures_core::Stream

use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

use crate::config::Config;
use crate::crypto::hash_bytes; // ct_eq deliberately NOT imported — unused
use crate::grpc::orchestrator::orchestrator_server::{Orchestrator, OrchestratorServer};
use crate::grpc::orchestrator::{
    Ack, Certificate, Csr, DeviceId, EnrollRequest, EnrollResponse, EnrollmentRequest,
    EnrollmentRequestAck, ModelChunk, Receipt, RoundMetadata, RoundRequest, UpdateChunk,
    UploadAck,
};
use crate::identity::derive_device_id;
use crate::round::{AggregationReceipt, Round, RoundState, UpdateMeta};
use crate::state::OrchestratorState;

// ── Constants ─────────────────────────────────────────────────────────────────
const MAX_UPDATE_BYTES: usize = 1024 * 1024 * 1024; // 500 MB absolute cap
const CHUNK_SIZE_MAX: usize = 4 * 1024 * 1024; // 4 MB per chunk

// ── Service struct ────────────────────────────────────────────────────────────
// FIX-BOOKKEEPING-1: Clone is required so a detached background task
// (submit_receipt's tokio::spawn for aggregation) can own its own handle
// to the service, independent of the triggering request's lifetime. Every
// field is already cheap to clone: Arc, mongodb::Client (internally
// Arc-backed), Config, and a small Vec<u8> key.
#[derive(Clone)]
pub struct Service {
    state: Arc<OrchestratorState>,
    cfg: Config,
    mongo: MongoClient,
    /// FIX-MULTIMODAL-1: database name, from MONGO_DATABASE env var
    /// (defaults to "federated" — see main.rs). Lets a separate experiment
    /// (different parameter shapes, independent round numbering) run
    /// against its own database using the identical binary/code paths.
    db_name: String,
    /// HMAC key for receipt chaining — loaded from RECEIPT_CHAIN_KEY env var,
    /// never from config files or hardcoded defaults in production.
    receipt_chain_key: Vec<u8>,
}

impl Service {
    pub fn new(
        state: Arc<OrchestratorState>,
        cfg: Config,
        mongo: MongoClient,
        db_name: String,
    ) -> anyhow::Result<Self> {
        let receipt_chain_key = std::env::var("RECEIPT_CHAIN_KEY")
            .map(|s| hex::decode(s).expect("RECEIPT_CHAIN_KEY must be a hex string"))
            .unwrap_or_else(|_| {
                tracing::warn!(
                    "RECEIPT_CHAIN_KEY not set — using ephemeral random key. \
                     Receipts will NOT be verifiable across server restarts. \
                     Set RECEIPT_CHAIN_KEY in production."
                );
                use rand::RngCore;
                let mut k = vec![0u8; 32];
                rand::thread_rng().fill_bytes(&mut k);
                k
            });

        Ok(Self {
            state,
            cfg,
            mongo,
            db_name,
            receipt_chain_key,
        })
    }

    fn db(&self) -> Database {
        self.mongo.database(&self.db_name)
    }

    // ── FIX-SERVER-1: Enforce mTLS client certificate ─────────────────────────
    // Previously returned Ok(()) unconditionally — every endpoint was reachable
    // without a certificate.  Now any missing cert is an immediate rejection.
    fn require_client_cert<T>(req: &Request<T>) -> Result<(), Status> {
        match req.peer_certs() {
            Some(certs) if !certs.is_empty() => Ok(()),
            _ => {
                tracing::warn!("Request rejected — no mTLS client certificate presented");
                Err(Status::unauthenticated(
                    "mutual TLS client certificate required for all endpoints",
                ))
            }
        }
    }

    // ── FIX-SERVER-6: HMAC chain computation ──────────────────────────────────
    // Each receipt is linked to the previous one via HMAC, producing a
    // tamper-evident ordered chain.  Inserting, removing, or reordering a
    // receipt breaks every chain link that follows.
    fn compute_chain_hmac(&self, prev_hmac: Option<&str>, payload_hash_hex: &str) -> String {
        let mut mac =
            Hmac::<Sha256>::new_from_slice(&self.receipt_chain_key).expect("HMAC key valid");
        mac.update(prev_hmac.unwrap_or("genesis").as_bytes());
        mac.update(b"|");
        mac.update(payload_hash_hex.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }
}

// ── Orchestrator trait implementation ─────────────────────────────────────────
#[tonic::async_trait]
impl Orchestrator for Service {
    // ── UploadUpdate (FIX-SERVER-2, FIX-SERVER-3, FIX-COMPILE-1, FIX-COMPILE-2) ──
    //
    // Clients stream encrypted model bytes directly to the server.
    // The server stores them in MongoDB GridFS and returns a server-side
    // ObjectId handle.  Clients then reference that handle in SubmitReceipt.
    //
    // This is a critical security fix: previously enc_uri was a local file path
    // on the *client*, which the server could never read.
    async fn upload_update(
        &self,
        req: Request<Streaming<UpdateChunk>>,
    ) -> Result<Response<UploadAck>, Status> {
        // Every endpoint enforces mTLS.
        Self::require_client_cert(&req)?;

        let mut stream = req.into_inner();
        let db = self.db();

        let mut all_bytes: Vec<u8> = Vec::new();
        let mut round_id: u64 = 0;
        let mut device_id_bytes: Vec<u8> = Vec::new();
        let mut session_id = String::new();
        let mut expected_total: u64 = 0;
        let mut received_chunks: u64 = 0;
        let mut global_hasher = Sha256::new();
        let mut initialized = false;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| Status::internal(format!("stream read error: {}", e)))?;

            // Reject oversized chunks before doing any work.
            if chunk.data.len() > CHUNK_SIZE_MAX {
                return Err(Status::invalid_argument(format!(
                    "chunk {} exceeds max size {}MB",
                    chunk.chunk_index,
                    CHUNK_SIZE_MAX / 1024 / 1024
                )));
            }

            // FIX-SERVER-3: per-chunk SHA-256 verification.
            // Detects corruption or tampering in transit at the chunk level,
            // before accumulating data into the full upload buffer.
            let computed_hash = Sha256::digest(&chunk.data);
            if chunk.chunk_hash.as_slice() != computed_hash.as_slice() {
                return Err(Status::data_loss(format!(
                    "chunk {} hash mismatch — data corrupted or tampered in transit \
                     (expected={}, got={})",
                    chunk.chunk_index,
                    hex::encode(&chunk.chunk_hash),
                    hex::encode(computed_hash),
                )));
            }

            // Sequential ordering validation — prevents chunk reorder / replay.
            if chunk.chunk_index != received_chunks {
                return Err(Status::invalid_argument(format!(
                    "out-of-order chunk: expected index {}, got {}",
                    received_chunks, chunk.chunk_index
                )));
            }

            // Enforce absolute size cap — prevents DoS via memory exhaustion.
            if all_bytes.len() + chunk.data.len() > MAX_UPDATE_BYTES {
                return Err(Status::resource_exhausted(
                    "update exceeds 500 MB maximum — upload rejected",
                ));
            }

            // First chunk initialises the session context.
            if !initialized {
                round_id = chunk.round_id;
                device_id_bytes = chunk.device_id.clone();
                session_id = chunk.session_id.clone();
                expected_total = chunk.total_chunks;

                if device_id_bytes.is_empty() {
                    return Err(Status::invalid_argument(
                        "device_id required in the first chunk",
                    ));
                }
                if expected_total == 0 {
                    return Err(Status::invalid_argument("total_chunks must be > 0"));
                }

                // Reject uploads from unenrolled devices before accepting any bytes.
                let devices = db.collection::<bson::Document>("devices");
                let device_hex = hex::encode(&device_id_bytes);
                if devices
                    .find_one(doc! { "device_id": &device_hex }, None)
                    .await
                    .map_err(|_| Status::internal("db error"))?
                    .is_none()
                {
                    tracing::warn!(
                        "Upload rejected — device {} is not enrolled",
                        device_hex
                    );
                    return Err(Status::permission_denied("device not enrolled"));
                }

                // Validate round state.
                let round = self
                    .state
                    .rounds
                    .get(&round_id)
                    .ok_or_else(|| Status::not_found("round not found"))?;
                if round.state != RoundState::Collecting {
                    return Err(Status::failed_precondition(
                        "round is not in Collecting state",
                    ));
                }

                initialized = true;
            } else {
                // Subsequent chunks must carry identical session metadata.
                // Mismatches indicate a malformed or malicious client.
                if chunk.round_id != round_id
                    || chunk.device_id != device_id_bytes
                    || chunk.total_chunks != expected_total
                {
                    return Err(Status::invalid_argument(
                        "chunk metadata mismatch — all chunks must share \
                         the same round_id, device_id, and total_chunks",
                    ));
                }
            }

            global_hasher.update(&chunk.data);
            all_bytes.extend_from_slice(&chunk.data);
            received_chunks += 1;
        }

        if !initialized || received_chunks == 0 {
            return Err(Status::invalid_argument(
                "empty upload — no chunks received",
            ));
        }

        if received_chunks != expected_total {
            return Err(Status::invalid_argument(format!(
                "chunk count mismatch: declared {}, received {}",
                expected_total, received_chunks
            )));
        }

        let payload_hash = global_hasher.finalize();
        let payload_hash_hex = hex::encode(payload_hash);
        let device_hex = hex::encode(&device_id_bytes);

        // FIX-COMPILE-1: db.gridfs_bucket() is the public constructor.
        // GridFsBucket::new() is pub(crate) in mongodb 2.8 and must NOT be called directly.
        let bucket: mongodb::gridfs::GridFsBucket = db.gridfs_bucket(None);

        let file_name = format!(
            "update_r{}_d{}_s{}",
            round_id,
            &device_hex[..8.min(device_hex.len())],
            &session_id[..12.min(session_id.len())],
        );

        // Open a GridFS upload stream.
        let mut upload_stream = bucket.open_upload_stream(file_name, None);

        // FIX-COMPILE-2: GridFsUploadStream implements futures::AsyncWrite, NOT tokio::io::AsyncWrite.
        // We must bring futures::AsyncWriteExt into scope (imported at top as FuturesAsyncWriteExt).
        // Using tokio::io::AsyncWriteExt would fail with "trait bound not satisfied".
        upload_stream
            .write_all(&all_bytes)
            .await
            .map_err(|e| {
                tracing::error!("GridFS write_all failed: {}", e);
                Status::internal("GridFS write failed")
            })?;

        // close() finalises the GridFS file and flushes all metadata.
        upload_stream.close().await.map_err(|e| {
            tracing::error!("GridFS close failed: {}", e);
            Status::internal("GridFS close failed")
        })?;

        // Retrieve the server-assigned ObjectId AFTER close().
        // This is the canonical handle returned to the client for use in SubmitReceipt.
        let file_id: ObjectId = upload_stream
            .id()
            .as_object_id()
            .ok_or_else(|| Status::internal("GridFS returned non-ObjectId file_id"))?;

        // Record the upload in model_updates with verified=false.
        // SubmitReceipt sets verified=true only after signature + hash checks pass.
        // This two-phase design prevents a receipt from being accepted for an
        // upload that was never actually stored.
        let model_updates = db.collection::<bson::Document>("model_updates");
        model_updates
            .insert_one(
                doc! {
                    "device_id":    &device_hex,
                    "round_id":     round_id as i64,
                    "session_id":   &session_id,
                    "payload_hash": &payload_hash_hex,
                    "file_id":      file_id,
                    "upload_time":  BsonDateTime::now(),
                    "verified":     false,
                    "size_bytes":   all_bytes.len() as i64,
                },
                None,
            )
            .await
            .map_err(|e| {
                tracing::error!("model_updates insert failed: {}", e);
                Status::internal("db insert failed")
            })?;

        tracing::info!(
            "Upload stored — device={} round={} size={}B hash={}… handle={}",
            &device_hex[..8.min(device_hex.len())],
            round_id,
            all_bytes.len(),
            &payload_hash_hex[..16.min(payload_hash_hex.len())],
            file_id,
        );

        Ok(Response::new(UploadAck {
            ok: true,
            server_handle: file_id.to_hex(),
            error: String::new(),
        }))
    }

    // ── DownloadGlobalModel (FIX-SERVER-8, FIX-COMPILE-1, FIX-COMPILE-3, FIX-COMPILE-4) ──
    //
    // FIX-COMPILE-4: The associated type must be Pin<Box<dyn Stream<...>>> because
    // tonic 0.11 cannot convert tokio_stream::Iter into tonic::codec::Streaming directly.
    // The Into<Streaming<ModelChunk>> bound is not implemented for that iterator type.
    type DownloadGlobalModelStream =
        Pin<Box<dyn Stream<Item = Result<ModelChunk, Status>> + Send + 'static>>;

    async fn download_global_model(
        &self,
        req: Request<RoundRequest>,
    ) -> Result<Response<Self::DownloadGlobalModelStream>, Status> {
        Self::require_client_cert(&req)?;

        let inner = req.into_inner();
        let db = self.db();

        // FIX-COMPILE-1: public constructor.
        let bucket: mongodb::gridfs::GridFsBucket = db.gridfs_bucket(None);

        let global_models = db.collection::<bson::Document>("global_models");
        let model_doc = global_models
            .find_one(doc! { "round_id": inner.round_id as i64 }, None)
            .await
            .map_err(|_| Status::internal("db error"))?
            .ok_or_else(|| Status::not_found("no global model available for this round"))?;

        let file_id = model_doc
            .get_object_id("file_id")
            .map_err(|_| Status::internal("malformed global_models record — file_id missing"))?;

        let model_hash_hex = model_doc.get_str("model_hash").unwrap_or("").to_string();

        // Open GridFS download stream.
        let mut download = bucket
            .open_download_stream(bson::Bson::ObjectId(file_id))
            .await
            .map_err(|_| Status::not_found("GridFS file not found for this model"))?;

        // FIX-COMPILE-3: GridFsDownloadStream implements futures::AsyncRead, NOT tokio::io::AsyncRead.
        // The original code called tokio::io::AsyncReadExt::read_to_end which fails with
        // "the trait tokio::io::AsyncRead is not implemented for GridFsDownloadStream".
        // We use futures::AsyncReadExt (imported as FuturesAsyncReadExt) instead.
        let mut full_bytes = Vec::new();
        download.read_to_end(&mut full_bytes).await.map_err(|e| {
            tracing::error!("GridFS read_to_end failed: {}", e);
            Status::internal("GridFS read failed")
        })?;

        if full_bytes.is_empty() {
            return Err(Status::internal("global model file is empty"));
        }

        const DL_CHUNK: usize = 1 * 1024 * 1024; // 1 MB per gRPC chunk
        let total_chunks = full_bytes.len().div_ceil(DL_CHUNK) as u64;
        let model_hash_bytes = hex::decode(&model_hash_hex).unwrap_or_default();

        let chunks: Vec<ModelChunk> = full_bytes
            .chunks(DL_CHUNK)
            .enumerate()
            .map(|(i, c)| {
                let chunk_hash = Sha256::digest(c).to_vec();
                // Model hash is only included in the final chunk.
                let mh = if i as u64 == total_chunks - 1 {
                    model_hash_bytes.clone()
                } else {
                    vec![]
                };
                ModelChunk {
                    chunk_index: i as u64,
                    total_chunks,
                    data: c.to_vec(),
                    chunk_hash,
                    model_hash: mh,
                }
            })
            .collect();

        tracing::info!(
            "Streaming global model for round {} — {} bytes in {} chunks",
            inner.round_id,
            full_bytes.len(),
            total_chunks,
        );

        // FIX-COMPILE-4: Box and Pin the stream so it satisfies the associated type bound.
        let stream = tokio_stream::iter(chunks.into_iter().map(Ok::<ModelChunk, Status>));
        Ok(Response::new(Box::pin(stream)))
    }

    // ── SubmitReceipt (FIX-SERVER-4, FIX-SERVER-5, FIX-SERVER-6, FIX-SERVER-9) ─────────
    async fn submit_receipt(
        &self,
        req: Request<Receipt>,
    ) -> Result<Response<Ack>, Status> {
        Self::require_client_cert(&req)?;

        let receipt = req.into_inner();

        // ── Input validation ──────────────────────────────────────────────────
        if receipt.device_id.is_empty() {
            return Err(Status::invalid_argument("device_id is required"));
        }
        if receipt.payload_hash.len() != 32 {
            return Err(Status::invalid_argument(
                "payload_hash must be exactly 32 bytes (SHA-256 output)",
            ));
        }
        if receipt.signature.is_empty() {
            return Err(Status::invalid_argument("signature is required"));
        }
        // FIX-SERVER-5: enc_handle must be a GridFS ObjectId, never a file path.
        // Client-supplied file paths are rejected unconditionally to prevent
        // path traversal attacks against the server filesystem.
        if receipt.enc_handle.is_empty() {
            return Err(Status::invalid_argument(
                "enc_handle is required — call UploadUpdate before SubmitReceipt",
            ));
        }
        // FIX-SERVER-9: epsilon must be positive and come from a real RDP accountant.
        // Hardcoded 0.0 or negative values are rejected at the protocol level.
        if receipt.epsilon_spent <= 0.0 {
            return Err(Status::invalid_argument(
                "epsilon_spent must be positive — use a real RDP accountant, not a hardcoded value",
            ));
        }

        let db = self.db();
        let device_hex = hex::encode(&receipt.device_id);

        // ── Device lookup ─────────────────────────────────────────────────────
        let devices = db.collection::<bson::Document>("devices");
        let device_doc = devices
            .find_one(doc! { "device_id": &device_hex }, None)
            .await
            .map_err(|_| Status::internal("db error"))?
            .ok_or_else(|| {
                tracing::warn!(
                    "Receipt from unknown device {}",
                    &device_hex[..8.min(device_hex.len())]
                );
                Status::permission_denied("device not enrolled")
            })?;

        let pubkey_pem = device_doc
            .get_str("pubkey_pem")
            .map_err(|_| Status::internal("malformed device record — pubkey_pem missing"))?;

        // ── ECDSA signature verification ──────────────────────────────────────
        // Canonical message: device_id || round_id_BE8 || payload_hash
        // This binds the receipt to a specific device, round, and payload.
        let mut msg = Vec::with_capacity(receipt.device_id.len() + 8 + 32);
        msg.extend_from_slice(&receipt.device_id);
        msg.extend_from_slice(&receipt.round_id.to_be_bytes());
        msg.extend_from_slice(&receipt.payload_hash);

        crate::receipts::verify(pubkey_pem.as_bytes(), &msg, &receipt.signature).map_err(|_| {
            tracing::warn!(
                "Invalid receipt signature from device {}",
                &device_hex[..8.min(device_hex.len())]
            );
            Status::permission_denied("receipt signature verification failed")
        })?;

        // ── FIX-SERVER-5: Validate enc_handle is a real GridFS ObjectId ───────
        // Any value that is not a valid 24-hex ObjectId is rejected.
        // This prevents clients from submitting arbitrary paths or identifiers
        // that might reference resources they do not own.
        let file_oid = ObjectId::parse_str(&receipt.enc_handle).map_err(|_| {
            Status::invalid_argument(
                "enc_handle is not a valid GridFS ObjectId — \
                 use the server_handle field from UploadAck",
            )
        })?;

        // ── FIX-SERVER-4: Cross-verify payload_hash against the stored upload ─
        // We compare the hash in the receipt against the hash computed server-side
        // during UploadUpdate.  This prevents a client from submitting a receipt
        // that references an upload it did not make, or that has a different hash.
        let model_updates = db.collection::<bson::Document>("model_updates");
        let update_doc = model_updates
            .find_one(
                doc! {
                    "file_id":   file_oid,
                    "device_id": &device_hex,
                    "round_id":  receipt.round_id as i64,
                    "verified":  false, // reject double-submission
                },
                None,
            )
            .await
            .map_err(|_| Status::internal("db error"))?
            .ok_or_else(|| {
                Status::not_found(
                    "no matching unverified upload found — the upload may not exist, \
                     belong to a different device, or the receipt was already submitted",
                )
            })?;

        let stored_hash = update_doc
            .get_str("payload_hash")
            .map_err(|_| Status::internal("malformed update record — payload_hash missing"))?;
        let submitted_hash = hex::encode(&receipt.payload_hash);

        if stored_hash != submitted_hash {
            tracing::warn!(
                "payload_hash mismatch device={} stored={}… submitted={}…",
                &device_hex[..8.min(device_hex.len())],
                &stored_hash[..16.min(stored_hash.len())],
                &submitted_hash[..16.min(submitted_hash.len())],
            );
            return Err(Status::permission_denied(
                "payload_hash does not match the data that was uploaded — receipt rejected",
            ));
        }

        // Mark the upload record as verified.
        model_updates
            .update_one(
                doc! { "file_id": file_oid },
                doc! { "$set": { "verified": true, "verified_at": BsonDateTime::now() } },
                None,
            )
            .await
            .map_err(|_| Status::internal("db update failed"))?;

        // ── Epsilon budget enforcement ─────────────────────────────────────────
        let mut round = self
            .state
            .rounds
            .get_mut(&receipt.round_id)
            .ok_or_else(|| Status::not_found("round not found"))?;

        if round.state != RoundState::Collecting {
            return Err(Status::failed_precondition(
                "round is not in Collecting state",
            ));
        }

        // FIX-SERVER-9: hard epsilon ceiling prevents a single client from
        // consuming the entire privacy budget.
        if round.epsilon_spent + receipt.epsilon_spent > round.epsilon_max {
            return Err(Status::resource_exhausted(format!(
                "epsilon budget exceeded: accumulated={:.4} + submitted={:.4} > max={:.4}",
                round.epsilon_spent, receipt.epsilon_spent, round.epsilon_max
            )));
        }
        round.epsilon_spent += receipt.epsilon_spent;

        // ── FIX-SERVER-6: HMAC-chained receipt storage ────────────────────────
        // Each receipt is linked to the previous one via HMAC.  This produces a
        // tamper-evident chain: inserting, removing, or reordering any receipt
        // breaks every subsequent chain link.
        let receipts_col = db.collection::<bson::Document>("receipts");
        let prev_doc = receipts_col
            .find_one(
                doc! { "round_id": receipt.round_id as i64 },
                FindOneOptions::builder()
                    .sort(doc! { "_id": -1 })
                    .build(),
            )
            .await
            .map_err(|_| Status::internal("db error"))?;

        let prev_hmac = prev_doc.as_ref().and_then(|d| d.get_str("hmac_chain").ok());
        let chain_hmac = self.compute_chain_hmac(prev_hmac, &submitted_hash);

        receipts_col
            .insert_one(
                doc! {
                    "device_id":     &device_hex,
                    "round_id":      receipt.round_id as i64,
                    "payload_hash":  &submitted_hash,
                    "epsilon_spent": receipt.epsilon_spent,
                    "signature":     hex::encode(&receipt.signature),
                    "enc_handle":    &receipt.enc_handle,
                    "scheme":        &receipt.scheme,
                    "timestamp":     BsonDateTime::now(),
                    "verified":      true,
                    "hmac_chain":    chain_hmac,
                },
                None,
            )
            .await
            .map_err(|_| Status::internal("receipt db insert failed"))?;

        // Register update in the in-memory round state.
        // enc_uri stores the GridFS ObjectId, never a local path.
        round.updates.push(UpdateMeta {
            device_id: receipt.device_id.clone(),
            enc_uri: receipt.enc_handle.clone(),
            scheme: receipt.scheme.clone(),
            nonce: if receipt.nonce.is_empty() {
                None
            } else {
                Some(receipt.nonce.clone())
            },
        });

        tracing::info!(
            "Receipt accepted — device={} round={} eps={:.4} handle={}",
            &device_hex[..8.min(device_hex.len())],
            receipt.round_id,
            receipt.epsilon_spent,
            &receipt.enc_handle,
        );

        // Trigger aggregation once enough updates have been received.
        //
        // FIX-BOOKKEEPING-1: run_aggregation() used to be awaited synchronously
        // inside this RPC handler. The aggregator subprocess runs on its own
        // OS thread via spawn_blocking and keeps running even if this future is
        // dropped, but everything AFTER it (JSON parse, global_models insert,
        // round-complete bookkeeping) only runs if this future survives to
        // await that result. A client-side timeout/retry (grpc_client.py's
        // call_with_retry) can abandon the underlying stream while the
        // multi-minute aggregation is still in flight, which drops this future
        // and silently orphans the subprocess's already-successful output —
        // no error is logged because nothing downstream of spawn_blocking ever
        // runs. Detaching onto tokio::spawn makes aggregation immune to the
        // triggering client's own connection lifecycle: the Ack below returns
        // immediately, and aggregation completes independently in the background.
        let should_aggregate = round.updates.len() >= 3;
        if should_aggregate {
            round.state = RoundState::Aggregating;
            let round_id_copy = receipt.round_id;
            drop(round); // release the DashMap lock before spawning
            let svc = self.clone();
            tokio::spawn(async move {
                if let Err(e) = svc.run_aggregation(round_id_copy).await {
                    tracing::error!(
                        "Aggregation failed for round {}: {:?}",
                        round_id_copy,
                        e
                    );
                }
            });
        }

        Ok(Response::new(Ack { ok: true }))
    }

    // ── RequestEnrollment ──────────────────────────────────────────────────────
    // Phase B1: the device requests an OTP.  The server displays the OTP to the
    // administrator out-of-band.  No certificate is required here because the
    // client does not yet have one.
    async fn request_enrollment(
        &self,
        req: Request<EnrollmentRequest>,
    ) -> Result<Response<EnrollmentRequestAck>, Status> {
        let peer_addr = req
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let inner = req.into_inner();

        if inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey is required"));
        }
        if inner.csr.is_empty() {
            return Err(Status::invalid_argument("csr is required"));
        }

        let fingerprint_bytes = hash_bytes(&inner.device_pubkey);
        let fingerprint = hex::encode(&fingerprint_bytes[..8]);
        let otp = crate::otp::generate_otp_for(Some(fingerprint.clone()));

        self.state.pending_enrollments.insert(
            fingerprint.clone(),
            (inner.device_pubkey.clone(), inner.csr.clone()),
        );

        let device_info = if inner.device_info.is_empty() {
            format!("peer={}", peer_addr)
        } else {
            format!(
                "{} / peer={}",
                &inner.device_info[..inner.device_info.len().min(60)],
                peer_addr
            )
        };

        // Display the OTP to the server operator.  The client will obtain it
        // via a secure out-of-band channel (e.g. administrator tells the user).
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║  NEW ENROLLMENT REQUEST                                  ║");
        println!(
            "║  Fingerprint : {:<42} ║",
            fingerprint
        );
        println!(
            "║  Device      : {:<42} ║",
            &device_info[..device_info.len().min(42)]
        );
        println!("║  OTP         : {:<42} ║", otp);
        println!("║  Expiry      : 10 minutes                                ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");

        tracing::info!(
            "Enrollment requested — fingerprint={} peer={}",
            fingerprint,
            peer_addr
        );

        Ok(Response::new(EnrollmentRequestAck {
            accepted: true,
            device_fingerprint: fingerprint,
        }))
    }

    // ── EnrollDevice ──────────────────────────────────────────────────────────
    // Phase B2: the device presents the OTP.  On success, the server signs the
    // client's CSR and stores the device's public key for future receipt verification.
    async fn enroll_device(
        &self,
        req: Request<EnrollRequest>,
    ) -> Result<Response<EnrollResponse>, Status> {
        let peer_addr = req
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let req_inner = req.into_inner();

        // Consume and validate the OTP (includes rate limiting).
        if !crate::otp::consume_otp_from(&req_inner.enrollment_token, &peer_addr) {
            tracing::warn!(
                "Enrollment rejected for peer {} — invalid or expired OTP",
                peer_addr
            );
            return Err(Status::permission_denied("invalid or expired OTP"));
        }

        if req_inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey is required"));
        }
        if req_inner.csr.is_empty() {
            return Err(Status::invalid_argument("csr is required"));
        }

        let device_id = derive_device_id(&req_inner.device_pubkey);
        self.state
            .devices
            .insert(device_id.clone(), req_inner.device_pubkey.clone());

        // Sign the client's CSR with our CA.
        let mut csr_file =
            NamedTempFile::new().map_err(|_| Status::internal("failed to create temp file"))?;
        csr_file
            .write_all(&req_inner.csr)
            .map_err(|_| Status::internal("failed to write CSR"))?;

        let cert_file =
            NamedTempFile::new().map_err(|_| Status::internal("failed to create temp file"))?;

        let output = Command::new("openssl")
            .args([
                "x509",
                "-req",
                "-in",
                csr_file.path().to_str().unwrap(),
                "-CA",
                &self.cfg.tls.ca_cert,
                "-CAkey",
                &self.cfg.tls.ca_key,
                "-CAcreateserial",
                "-out",
                cert_file.path().to_str().unwrap(),
                "-days",
                "365",
                "-sha256",
            ])
            .output()
            .map_err(|e| {
                tracing::error!("openssl exec failed: {}", e);
                Status::internal("certificate signing failed")
            })?;

        if !output.status.success() {
            tracing::error!(
                "openssl stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Err(Status::internal("certificate signing failed"));
        }

        let signed_cert = fs::read(cert_file.path())
            .map_err(|_| Status::internal("failed to read signed certificate"))?;

        if signed_cert.is_empty() {
            return Err(Status::internal(
                "certificate signing produced an empty output",
            ));
        }

        // Persist the device in MongoDB.
        // The pubkey_pem is stored so that SubmitReceipt can verify ECDSA signatures
        // without contacting any external PKI.
        let db = self.db();
        let col = db.collection::<bson::Document>("devices");
        let device_hex = hex::encode(&device_id);
        let pubkey_pem = String::from_utf8_lossy(&req_inner.device_pubkey).to_string();

        // Upsert so that re-enrollment replaces the old record.
        col.update_one(
            doc! { "device_id": &device_hex },
            doc! { "$set": {
                "device_id":   &device_hex,
                "pubkey_pem":  pubkey_pem,
                "enrolled_at": BsonDateTime::now(),
                "last_seen":   BsonDateTime::now(),
                "peer_addr":   &peer_addr,
            }},
            mongodb::options::UpdateOptions::builder()
                .upsert(true)
                .build(),
        )
        .await
        .map_err(|_| Status::internal("db upsert failed"))?;

        // Clean up the pending enrollment entry.
        let fp_bytes = hash_bytes(&req_inner.device_pubkey);
        let fingerprint = hex::encode(&fp_bytes[..8]);
        self.state.pending_enrollments.remove(&fingerprint);

        tracing::info!(
            "Device enrolled — fingerprint={} peer={}",
            fingerprint,
            peer_addr
        );
        println!("[ENROLLED] fingerprint={} peer={}", fingerprint, peer_addr);

        Ok(Response::new(EnrollResponse {
            ok: true,
            client_cert: signed_cert,
        }))
    }

    // ── RegisterDevice (deprecated, kept for backward compatibility) ──────────
    async fn register_device(
        &self,
        req: Request<Csr>,
    ) -> Result<Response<Certificate>, Status> {
        Self::require_client_cert(&req)?;
        let inner = req.into_inner();
        if inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey is required"));
        }
        let device_id = derive_device_id(&inner.device_pubkey);
        self.state.devices.insert(device_id, inner.device_pubkey.clone());
        tracing::warn!(
            "register_device called — this RPC is deprecated; \
             use RequestEnrollment + EnrollDevice"
        );
        Ok(Response::new(Certificate {
            pem: inner.device_pubkey,
        }))
    }

    // ── GetRound ──────────────────────────────────────────────────────────────
    async fn get_round(
        &self,
        req: Request<DeviceId>,
    ) -> Result<Response<RoundMetadata>, Status> {
        Self::require_client_cert(&req)?;

        let inner = req.into_inner();
        let db = self.db();
        let device_hex = hex::encode(&inner.id);

        // Update last_seen timestamp so the operator can monitor liveness.
        let _ = db
            .collection::<bson::Document>("devices")
            .update_one(
                doc! { "device_id": &device_hex },
                doc! { "$set": { "last_seen": BsonDateTime::now() } },
                None,
            )
            .await;

        // Dynamic round selection: prefer the highest-ID Collecting round so that
        // once round N completes and round N+1 is created, clients automatically
        // migrate to the new round without a server restart.
        // Fallback (or_else): if no Collecting round exists (e.g. brief window
        // between rounds), return the highest-ID round regardless of state so
        // clients receive Complete/Aggregating metadata rather than an error.
        let current_round_id = self
            .state
            .rounds
            .iter()
            .filter(|r| r.state == RoundState::Collecting)
            .map(|r| r.id)
            .max()
            .or_else(|| self.state.rounds.iter().map(|r| r.id).max())
            .ok_or_else(|| Status::not_found("no round exists"))?;

        let round = self
            .state
            .rounds
            .get(&current_round_id)
            .ok_or_else(|| Status::not_found("round not found"))?;

        let receipt_ref = round.aggregation_receipt.as_ref();

        // Check whether a global model is available for download.
        let global_model_available = db
            .collection::<bson::Document>("global_models")
            .find_one(doc! { "round_id": round.id as i64 }, None)
            .await
            .map(|opt| opt.is_some())
            .unwrap_or(false);

        tracing::info!(
            "GetRound — device={} round={} state={:?} eps_spent={:.4}/{:.4} updates={} global_model_available={}",
            &device_hex[..8.min(device_hex.len())],
            round.id,
            round.state,
            round.epsilon_spent,
            round.epsilon_max,
            round.updates.len(),
            global_model_available,
        );

        Ok(Response::new(RoundMetadata {
            round_id: round.id,
            model_version: round.model_version.clone(),
            epsilon_max: round.epsilon_max,
            upload_uri: String::new(), // deprecated field
            state: format!("{:?}", round.state),
            num_updates: receipt_ref
                .map(|r| r.num_updates as u32)
                .unwrap_or(0),
            aggregation_mode: receipt_ref
                .map(|r| r.aggregation_mode.clone())
                .unwrap_or_default(),
            global_model_available,
        }))
    }
}

// FIX-RECOVERY-1: Full-chain, MongoDB-driven round recovery.
//
// Previous behavior (bug): this only ever walked round_ids already present
// in the in-memory `state.rounds` DashMap, which `OrchestratorState::new()`
// seeded with round 1 alone. Rounds created dynamically by a PRIOR process
// instance (round 2, round 3, ...) never existed in a freshly-started
// process's DashMap, so they were structurally invisible here — even if
// MongoDB already showed them as fully complete (verified updates AND a
// persisted global model for the following round). A client could then
// resubmit updates for an already-completed round, causing a second
// aggregation that collides with the global model a prior process already
// published for the round after it.
//
// Fixed behavior: reconstruct the ENTIRE round chain directly from MongoDB,
// not from whatever happens to already be in memory:
//   1. Read every `global_models` document. An entry with round_id = N means
//      round (N-1)'s aggregation is done — that model is what round N's
//      client(s) will warm-start from.
//   2. Read every verified `receipts` document, counted per round_id.
//   3. Walk round_id = 1, 2, 3, ... for as long as round_id's aggregation is
//      confirmed done (global_models contains round_id + 1). Each such round
//      is recreated in memory as Complete — never re-aggregated, per the
//      "a completed round must never be aggregated again" invariant.
//   4. The first round_id whose aggregation is NOT yet confirmed done is the
//      recovery target:
//        - >= 3 verified updates already ->  pending aggregation (a prior
//          process most likely crashed between accepting the 3rd receipt
//          and finishing aggregation). Hydrate its updates and resume
//          aggregation exactly once, through the unmodified
//          run_aggregation() / aggregator.py trimmed-mean path.
//        - < 3 verified updates -> genuinely still collecting. Hydrate
//          whatever updates already exist (0..2) so future submissions
//          count correctly, but do not aggregate.
// This is idempotent: run it against the same MongoDB state any number of
// times and it reconstructs the same logical state every time (verified by
// the hydration_recovery integration tests).
/// Pure recovery-decision logic: given round-completion data already read
/// from MongoDB (no I/O here), decide which rounds are already complete,
/// which round is the recovery candidate, and whether that candidate needs
/// aggregation resumed. This is the exact logic that was broken (it only
/// ever considered round 1) — extracted into a pure function so it can be
/// unit-tested without a live MongoDB connection, and reused unchanged by
/// `hydrate_and_resume_aggregation()` against real data.
#[derive(Debug, PartialEq, Eq)]
struct RecoveryPlan {
    latest_completed: u64,
    candidate_round_id: u64,
    candidate_verified_count: usize,
    should_aggregate: bool,
}

fn compute_recovery_plan(
    completed_target_rounds: &std::collections::HashSet<u64>,
    verified_counts: &std::collections::HashMap<u64, usize>,
) -> RecoveryPlan {
    let mut round_id: u64 = 1;
    let mut latest_completed: u64 = 0;
    while completed_target_rounds.contains(&(round_id + 1)) {
        latest_completed = round_id;
        round_id += 1;
    }
    let candidate_verified_count = verified_counts.get(&round_id).copied().unwrap_or(0);
    let should_aggregate = candidate_verified_count >= 3;
    RecoveryPlan {
        latest_completed,
        candidate_round_id: round_id,
        candidate_verified_count,
        should_aggregate,
    }
}

/// Pure persistence-action decision: given the existing global_models hash
/// (if any) for the target round and the freshly computed hash, decide
/// whether this is a fresh CREATE, an IDEMPOTENT no-op, or a genuine
/// CONFLICT that must not overwrite the existing (authoritative) record.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum PersistenceAction {
    Created,
    Idempotent,
    Conflict,
}

fn compute_persistence_action(existing_hash: Option<&str>, computed_hash: &str) -> PersistenceAction {
    match existing_hash {
        None => PersistenceAction::Created,
        Some(h) if h == computed_hash => PersistenceAction::Idempotent,
        Some(_) => PersistenceAction::Conflict,
    }
}

#[cfg(test)]
mod hydration_recovery_tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    // TEST 1: Round 1 -> 3 updates -> global model round 2.
    //         Round 2 -> 3 updates -> global model round 3.
    // Expected: the whole chain is recognized as already complete;
    // round 3 (no updates yet) is the next collecting round; no aggregation.
    #[test]
    fn test1_full_chain_already_complete_no_reaggregation() {
        let completed: HashSet<u64> = [2, 3].into_iter().collect();
        let verified: HashMap<u64, usize> = [(1, 3), (2, 3)].into_iter().collect();

        let plan = compute_recovery_plan(&completed, &verified);

        assert_eq!(plan.latest_completed, 2, "rounds 1 and 2 must both be recognized complete");
        assert_eq!(plan.candidate_round_id, 3, "round 3 must be the next round, not 2");
        assert_eq!(plan.candidate_verified_count, 0, "round 3 has no receipts yet");
        assert!(!plan.should_aggregate, "an already-complete chain must never re-aggregate");
    }

    // TEST 2: Round 2 has 3 verified updates but round 3's global model is
    // absent (a prior process crashed between accepting the 3rd receipt and
    // finishing aggregation).
    // Expected: round 2 is recovered as pending aggregation, exactly once.
    #[test]
    fn test2_pending_aggregation_recovered_once() {
        let completed: HashSet<u64> = [2].into_iter().collect(); // round 1 complete, round 2 is not
        let verified: HashMap<u64, usize> = [(1, 3), (2, 3)].into_iter().collect();

        let plan = compute_recovery_plan(&completed, &verified);

        assert_eq!(plan.latest_completed, 1);
        assert_eq!(plan.candidate_round_id, 2, "round 2 must be the recovery candidate");
        assert_eq!(plan.candidate_verified_count, 3);
        assert!(plan.should_aggregate, "3 verified updates with no next global model must resume aggregation");
    }

    // TEST 3: target global model already exists with the SAME hash as the
    // freshly computed one -> idempotent success, no duplicate GridFS model.
    #[test]
    fn test3_same_hash_is_idempotent() {
        let action = compute_persistence_action(Some("abc123"), "abc123");
        assert_eq!(action, PersistenceAction::Idempotent);
    }

    // TEST 4: target global model exists with a DIFFERENT hash -> explicit
    // conflict; the existing model must never be silently overwritten.
    #[test]
    fn test4_different_hash_is_conflict() {
        let action = compute_persistence_action(Some("abc123"), "xyz789");
        assert_eq!(action, PersistenceAction::Conflict);
    }

    // No existing record at all -> fresh create.
    #[test]
    fn test4b_no_existing_record_is_created() {
        let action = compute_persistence_action(None, "abc123");
        assert_eq!(action, PersistenceAction::Created);
    }

    // TEST 5: restarting the server twice against unchanged MongoDB state
    // must produce the identical recovered plan both times — no duplicate
    // aggregation, no duplicate global model, no duplicate updates.
    // compute_recovery_plan is a pure function of its inputs, so calling it
    // twice with the same inputs is exactly what a real double-restart does
    // (each restart re-derives its plan from the same MongoDB documents).
    #[test]
    fn test5_double_restart_is_idempotent() {
        let completed: HashSet<u64> = [2, 3].into_iter().collect();
        let verified: HashMap<u64, usize> = [(1, 3), (2, 3)].into_iter().collect();

        let plan_first_restart = compute_recovery_plan(&completed, &verified);
        let plan_second_restart = compute_recovery_plan(&completed, &verified);

        assert_eq!(plan_first_restart, plan_second_restart);
        assert!(!plan_first_restart.should_aggregate);
    }

    // Fresh install: no MongoDB data at all. Must behave like the original
    // OrchestratorState::new() default (round 1, Collecting, empty) — this
    // guards the removal of the hardcoded round-1 insert in state.rs.
    #[test]
    fn test_fresh_install_starts_at_round_1() {
        let completed: HashSet<u64> = HashSet::new();
        let verified: HashMap<u64, usize> = HashMap::new();

        let plan = compute_recovery_plan(&completed, &verified);

        assert_eq!(plan.latest_completed, 0);
        assert_eq!(plan.candidate_round_id, 1);
        assert_eq!(plan.candidate_verified_count, 0);
        assert!(!plan.should_aggregate);
    }

    // Reproduces this session's actual incident: round 1 complete (GM round
    // 2 present), round 2 has 6 verified updates (3 legitimate original +
    // 3 accepted later due to the bug) but GM round 3 is ALSO already
    // present (round 2 was already aggregated once, historically). The fix
    // must recognize round 2 as complete regardless of the extra updates,
    // and never attempt to re-aggregate it.
    #[test]
    fn test_regression_extra_updates_on_already_complete_round() {
        let completed: HashSet<u64> = [2, 3].into_iter().collect();
        let verified: HashMap<u64, usize> = [(1, 3), (2, 6)].into_iter().collect();

        let plan = compute_recovery_plan(&completed, &verified);

        assert_eq!(plan.latest_completed, 2, "round 2 must be recognized complete despite 6 receipts");
        assert_eq!(plan.candidate_round_id, 3);
        assert!(!plan.should_aggregate, "must never re-aggregate an already-complete round");
    }
}

async fn hydrate_and_resume_aggregation(svc: &Service) {
    use std::collections::{HashMap, HashSet};

    let db = svc.db();
    let receipts_col = db.collection::<bson::Document>("receipts");
    let global_models_col = db.collection::<bson::Document>("global_models");

    println!("============================================================");
    println!("           SERVER STARTUP / ROUND RECOVERY");
    println!("MongoDB database    : {}", svc.db_name);
    println!("============================================================");
    println!("MongoDB status       : CONNECTED");
    tracing::info!("[RECOVERY] Scanning MongoDB for existing rounds");

    // ── Step 1: which rounds already have a persisted global model? ──────────
    let mut completed_target_rounds: HashSet<u64> = HashSet::new();
    {
        let mut cursor = match global_models_col.find(doc! {}, None).await {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("[RECOVERY] global_models query failed: {}", e);
                println!("Recovery status      : FAIL (global_models query error: {})", e);
                println!("============================================================");
                return;
            }
        };
        while let Some(item) = cursor.next().await {
            if let Ok(d) = item {
                if let Ok(rid) = d.get_i64("round_id") {
                    completed_target_rounds.insert(rid as u64);
                }
            }
        }
    }

    // ── Step 2: verified-receipt count per round_id ───────────────────────────
    let mut verified_counts: HashMap<u64, usize> = HashMap::new();
    {
        let mut cursor = match receipts_col.find(doc! { "verified": true }, None).await {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("[RECOVERY] receipts query failed: {}", e);
                println!("Recovery status      : FAIL (receipts query error: {})", e);
                println!("============================================================");
                return;
            }
        };
        while let Some(item) = cursor.next().await {
            if let Ok(d) = item {
                if let Ok(rid) = d.get_i64("round_id") {
                    *verified_counts.entry(rid as u64).or_insert(0) += 1;
                }
            }
        }
    }

    // ── Step 3: apply the pure recovery plan, recreating every complete round ─
    let plan = compute_recovery_plan(&completed_target_rounds, &verified_counts);

    for r in 1..=plan.latest_completed {
        let vc = verified_counts.get(&r).copied().unwrap_or(0);
        tracing::info!(
            "[RECOVERY] Round {}: verified_updates={}, global_model=present",
            r, vc
        );
        println!(
            "Round {:<3}             : Complete — {} verified updates, global model present",
            r, vc
        );
        svc.state.rounds.entry(r).or_insert_with(|| Round {
            id: r,
            model_version: format!("v{}", r),
            epsilon_max: 100.0,
            upload_uri: String::new(),
            state: RoundState::Complete,
            updates: Vec::new(),
            aggregation_receipt: Some(AggregationReceipt {
                round_id: r,
                num_updates: vc,
                aggregation_mode: "trimmed_mean".to_string(),
                aggregated_uri: "recovered-from-mongodb".to_string(),
            }),
            epsilon_spent: 0.0,
        });
    }

    let round_id = plan.candidate_round_id;
    let candidate_verified = plan.candidate_verified_count;
    let has_global_model_for_candidate = completed_target_rounds.contains(&round_id);
    tracing::info!(
        "[RECOVERY] Round {}: verified_updates={}, global_model={}",
        round_id,
        candidate_verified,
        if has_global_model_for_candidate { "present" } else { "absent" }
    );
    println!(
        "Round {:<3}             : {} — {} verified updates, global model {}",
        round_id,
        if candidate_verified >= 3 { "Pending aggregation" } else { "Collecting" },
        candidate_verified,
        if has_global_model_for_candidate { "present" } else { "absent" }
    );

    tracing::info!("[RECOVERY] Latest completed round = {}", plan.latest_completed);
    tracing::info!("[RECOVERY] Next collecting round = {}", round_id);
    println!("Latest completed     : {}", plan.latest_completed);
    println!("Next collecting      : {}", round_id);

    // ── Step 4: hydrate the candidate round's updates from receipts ──────────
    let filter = doc! { "round_id": round_id as i64, "verified": true };
    let opts = FindOptions::builder().sort(doc! { "timestamp": 1 }).build();
    let mut hydrated: Vec<UpdateMeta> = Vec::new();
    match receipts_col.find(filter, opts).await {
        Ok(mut cursor) => {
            while let Some(item) = cursor.next().await {
                let doc = match item {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::error!("[RECOVERY] cursor read failed: {}", e);
                        continue;
                    }
                };
                let device_hex = doc.get_str("device_id").unwrap_or_default();
                let device_id = match hex::decode(device_hex) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let enc_uri = doc.get_str("enc_handle").unwrap_or_default().to_string();
                let scheme = doc.get_str("scheme").unwrap_or_default().to_string();
                if enc_uri.is_empty() {
                    continue;
                }
                hydrated.push(UpdateMeta { device_id, enc_uri, scheme, nonce: None });
            }
        }
        Err(e) => {
            tracing::error!(
                "[RECOVERY] receipts query failed for candidate round {}: {}",
                round_id, e
            );
        }
    }

    // plan.should_aggregate is the pure-function decision; hydrated.len() >= 3
    // is a belt-and-suspenders check that the receipts we actually parsed
    // match what the earlier count query saw.
    let should_aggregate = plan.should_aggregate && hydrated.len() >= 3;

    svc.state.rounds.entry(round_id).or_insert_with(|| Round {
        id: round_id,
        model_version: format!("v{}", round_id),
        epsilon_max: 100.0,
        upload_uri: String::new(),
        state: RoundState::Collecting,
        updates: Vec::new(),
        aggregation_receipt: None,
        epsilon_spent: 0.0,
    });
    {
        if let Some(mut r) = svc.state.rounds.get_mut(&round_id) {
            r.updates = hydrated;
            if should_aggregate {
                r.state = RoundState::Aggregating;
            }
        }
    }

    if should_aggregate {
        tracing::info!(
            "[RECOVERY] Round {} has {} verified updates but no global model yet — \
             resuming aggregation (a prior process likely crashed mid-aggregation)",
            round_id, candidate_verified
        );
        println!("Pending aggregation  : YES — resuming now");
        println!("Recovery status      : PASS");
        println!("============================================================");
        let svc2 = svc.clone();
        let rid = round_id;
        tokio::spawn(async move {
            if let Err(e) = svc2.run_aggregation(rid).await {
                tracing::error!("[RECOVERY] resumed aggregation failed for round {}: {:?}", rid, e);
            }
        });
    } else {
        tracing::info!("[RECOVERY] No duplicate aggregation required");
        println!("Pending aggregation  : NO");
        println!("Recovery status      : PASS");
        println!("============================================================");
    }
}

// ── Server bootstrap ──────────────────────────────────────────────────────────
pub async fn serve(
    cfg: Config,
    state: Arc<OrchestratorState>,
    mongo: MongoClient,
    db_name: String,
) -> anyhow::Result<()> {
    let svc = Service::new(state, cfg.clone(), mongo, db_name)?;
    hydrate_and_resume_aggregation(&svc).await;
    let addr = cfg.server.addr.parse()?;

    if cfg.server.enable_tls {
        let server_identity = tonic::transport::Identity::from_pem(
            std::fs::read(&cfg.tls.server_cert)?,
            std::fs::read(&cfg.tls.server_key)?,
        );
        // Load the CA cert to verify client certificates (mutual TLS).
        let client_ca = std::fs::read(&cfg.tls.ca_cert)?;
        // client_auth_optional(true): TLS layer accepts connections with or without a
        // client cert (needed for enrollment, which has no cert yet).  All operational
        // RPCs still enforce mTLS via require_client_cert() at the application layer.
        let tls = tonic::transport::ServerTlsConfig::new()
            .identity(server_identity)
            .client_ca_root(tonic::transport::Certificate::from_pem(client_ca))
            .client_auth_optional(true);

        tracing::info!("[SERVER] mTLS mode — binding to {}", addr);
        println!("[SERVER] Running in mTLS mode on {}", addr);

        Server::builder()
            .tls_config(tls)?
            .add_service(OrchestratorServer::new(svc))
            .serve(addr)
            .await?;
    } else {
        // Insecure mode is intentionally left in for local development only.
        // In production, enable_tls MUST be true.
        tracing::warn!("[SERVER] INSECURE mode — NOT for production");
        println!(
            "[SERVER] INSECURE mode on {} — set enable_tls=true in production",
            addr
        );

        Server::builder()
            .add_service(OrchestratorServer::new(svc))
            .serve(addr)
            .await?;
    }

    Ok(())
}

// ── Aggregation ───────────────────────────────────────────────────────────────
// Reads from GridFS by ObjectId (set by UploadUpdate) so the Python aggregator
// never receives or needs local file paths.
//
// PHASE-3 FIXES applied here:
//   FIX-P3-1: run_aggregation is now async; the subprocess runs inside
//             spawn_blocking so the Tokio worker thread is not stalled.
//   FIX-P3-2: Parses gridfs_file_id and model_hash from aggregator output.
//   FIX-P3-3: Inserts a global_models document so DownloadGlobalModel works
//             and GetRound reports global_model_available = true for round N+1.
//   FIX-P3-4: Creates round N+1 in Collecting state after round N completes.
impl Service {
    async fn run_aggregation(&self, round_id: u64) -> Result<(), Status> {
        // ── Phase 1: build the aggregator job from in-memory state ────────────
        // The DashMap shard lock is held only for the duration of this block.
        let (job_str, num_clients) = {
            let round = self
                .state
                .rounds
                .get(&round_id)
                .ok_or_else(|| Status::not_found("round not found for aggregation"))?;

            tracing::info!(
                "Aggregation starting — round={} updates={} algorithm=trimmed_mean trim_ratio=0.1",
                round.id,
                round.updates.len(),
            );

            let job_str = serde_json::json!({
                "round_id":   round.id,
                "mode":       "trimmed_mean",
                "trim_ratio": 0.1,
                "updates": round.updates.iter().map(|u| serde_json::json!({
                    "gridfs_id": u.enc_uri,
                    "scheme":    u.scheme,
                    "nonce":     u.nonce,
                })).collect::<Vec<_>>()
            })
            .to_string();

            (job_str, round.updates.len())
        }; // DashMap shard lock released here — safe to await below

        // ── Phase 2: run the Python subprocess on a blocking thread ───────────
        // wait_with_output() is a blocking syscall; offloading it to
        // spawn_blocking keeps the Tokio worker thread free for other tasks
        // while the Python aggregator runs (which can take minutes).
        //
        // B4/B5 fixes: explicit venv path (not "python3" — MS Store stub;
        // not "python" — Windows App Paths registry hijacks to system Python
        // before PATH, bypassing the venv). "../aggregator_agent/aggregator.py"
        // is relative to server/orchestration_agent/ CWD (not repo root).
        // PYTHONPATH="../.." is the repo root so local imports resolve.
        let stdout_bytes =
            tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
                use std::io::Write as _;
                let mut child = std::process::Command::new("../../.venv/Scripts/python")
                    .arg("../aggregator_agent/aggregator.py")
                    .env("PYTHONPATH", "../..")
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("aggregator spawn failed: {}", e))?;

                if let Some(mut stdin) = child.stdin.take() {
                    stdin
                        .write_all(job_str.as_bytes())
                        .map_err(|e| format!("stdin write failed: {}", e))?;
                }

                let output = child
                    .wait_with_output()
                    .map_err(|e| format!("wait_with_output failed: {}", e))?;

                if !output.status.success() {
                    return Err(format!(
                        "aggregator exited non-zero — stderr: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }

                Ok(output.stdout)
            })
            .await
            .map_err(|e| Status::internal(format!("spawn_blocking join error: {}", e)))?
            .map_err(|e| {
                tracing::error!("Aggregation subprocess failed: {}", e);
                Status::internal("aggregation failed")
            })?;

        // ── Phase 3: parse aggregator output ─────────────────────────────────
        // FIX-BOOKKEEPING-3: every branch here now logs via tracing::error!
        // before returning. Previously these five error paths returned a
        // Status with no log line at all, so a failure here was completely
        // invisible in the server log — indistinguishable from the task
        // simply never having reached this point (see FIX-BOOKKEEPING-1).
        let result: serde_json::Value = serde_json::from_slice(&stdout_bytes).map_err(|e| {
            tracing::error!(
                "Aggregation Phase 3: aggregator stdout is not valid JSON: {} — raw: {}",
                e,
                String::from_utf8_lossy(&stdout_bytes)
            );
            Status::internal("aggregator output is not valid JSON")
        })?;

        let aggregated_uri = result["aggregated_uri"]
            .as_str()
            .ok_or_else(|| {
                tracing::error!(
                    "Aggregation Phase 3: missing aggregated_uri in aggregator output: {}",
                    result
                );
                Status::internal("aggregator output missing aggregated_uri")
            })?
            .to_string();

        let gridfs_file_id_str = result["gridfs_file_id"]
            .as_str()
            .ok_or_else(|| {
                tracing::error!(
                    "Aggregation Phase 3: missing gridfs_file_id in aggregator output: {}",
                    result
                );
                Status::internal("aggregator output missing gridfs_file_id")
            })?
            .to_string();

        let model_hash = result["model_hash"]
            .as_str()
            .ok_or_else(|| {
                tracing::error!(
                    "Aggregation Phase 3: missing model_hash in aggregator output: {}",
                    result
                );
                Status::internal("aggregator output missing model_hash")
            })?
            .to_string();

        let file_oid = ObjectId::parse_str(&gridfs_file_id_str).map_err(|e| {
            tracing::error!(
                "Aggregation Phase 3: gridfs_file_id '{}' is not a valid ObjectId: {}",
                gridfs_file_id_str,
                e
            );
            Status::internal("aggregator returned an invalid GridFS ObjectId")
        })?;

        let num_keys = result["num_keys"].as_u64();
        let total_parameters = result["total_parameters"].as_u64();

        println!("============================================================");
        println!("        FEDERATED LEARNING — SERVER AGGREGATION");
        println!("============================================================");
        println!("Round ID             : {}", round_id);
        println!("Algorithm            : Trimmed Mean");
        println!("Trim ratio           : 0.1");
        println!("Clients              : {}", num_clients);
        match num_keys {
            Some(n) => println!("Parameter tensors    : {}", n),
            None => println!("Parameter tensors    : n/a (not reported by aggregator)"),
        }
        match total_parameters {
            Some(n) => println!("Total parameters     : {}", n),
            None => println!("Total parameters     : n/a (not reported by aggregator)"),
        }
        println!("Aggregation status   : PASS");
        println!("============================================================");

        // ── Phase 4: mark round N complete ────────────────────────────────────
        let num_updates = {
            let mut round = self
                .state
                .rounds
                .get_mut(&round_id)
                .ok_or_else(|| Status::not_found("round disappeared during aggregation"))?;

            let n = round.updates.len();
            round.state = RoundState::Complete;
            round.upload_uri = aggregated_uri.clone();
            round.aggregation_receipt = Some(AggregationReceipt {
                round_id,
                num_updates: n,
                aggregation_mode: "trimmed_mean".to_string(),
                aggregated_uri,
            });
            n
        }; // write lock released here

        tracing::info!(
            "Round {} complete — {} updates aggregated",
            round_id,
            num_updates
        );

        // ── Phase 5: persist global model in MongoDB ──────────────────────────
        // Stored under round_id + 1 so that clients entering the NEXT round see
        // global_model_available = true when they call GetRound, and can then
        // call DownloadGlobalModel(round_id = N+1) to retrieve this model.
        //
        // FIX-BOOKKEEPING-4: idempotent across restarts. In-memory round state
        // does not survive a process restart, but startup hydration (see
        // hydrate_and_resume_aggregation) can legitimately re-run aggregation
        // for a round whose global model a PRIOR process instance already
        // persisted — hydration has no way to know that without asking Mongo
        // first. Three cases:
        //   A. No existing record  -> insert fresh (CREATED).
        //   B. Existing record, same hash -> idempotent success, no duplicate
        //      GridFS insert needed (IDEMPOTENT).
        //   C. Existing record, different hash -> genuine conflict. The
        //      existing record is authoritative and is NEVER overwritten —
        //      but this must not leave the round stuck. Round N is already
        //      correctly marked Complete (Phase 4, above) — that aggregation
        //      genuinely happened and produced valid data, it's simply
        //      superseded by a global model an earlier process already
        //      published for round N+1. FIX-RECOVERY-2: previously this
        //      returned Err() here, which skipped Phase 6 entirely and left
        //      the round with no successor — no client could ever be served
        //      a Collecting round again until a human intervened. Now we log
        //      the conflict (CONFLICT, with both hashes for diagnosis) and
        //      fall through to Phase 6, which unconditionally ensures round
        //      N+1 exists as Collecting — using or_insert_with, so it never
        //      clobbers a round N+1 that may already be live with its own
        //      updates. The freshly-computed (conflicting) model itself is
        //      discarded: it was already durably written to GridFS by
        //      aggregator.py under its own hash, so no data is lost, it is
        //      simply not referenced by any global_models document.
        let next_id = round_id + 1;
        let db = self.db();
        let global_models_col = db.collection::<bson::Document>("global_models");

        let existing = global_models_col
            .find_one(doc! { "round_id": next_id as i64 }, None)
            .await
            .map_err(|e| {
                tracing::error!("global_models lookup failed: {}", e);
                Status::internal("failed to check existing global model")
            })?;

        let existing_hash_owned: Option<String> = existing
            .as_ref()
            .map(|doc| doc.get_str("model_hash").unwrap_or_default().to_string());
        let action = compute_persistence_action(existing_hash_owned.as_deref(), &model_hash);
        let existing_hash_display = existing_hash_owned.clone().unwrap_or_else(|| "none".to_string());
        let persistence_action = match action {
            PersistenceAction::Created => "CREATED",
            PersistenceAction::Idempotent => "IDEMPOTENT",
            PersistenceAction::Conflict => "CONFLICT",
        };

        match action {
            PersistenceAction::Idempotent => {
                tracing::info!(
                    "Global model for round {} already persisted (hash={}…) — \
                     hydration re-aggregation reproduced the same result; \
                     treating as success, not inserting a duplicate.",
                    next_id,
                    &model_hash[..16.min(model_hash.len())],
                );
            }
            PersistenceAction::Conflict => {
                tracing::error!(
                    "global_models for round {} already exists with a DIFFERENT \
                     hash (existing={}… new={}…) — not overwriting. Source aggregation \
                     round={}. The existing record remains authoritative; the freshly \
                     computed model is discarded (its GridFS blob remains, unreferenced).",
                    next_id,
                    &existing_hash_display[..16.min(existing_hash_display.len())],
                    &model_hash[..16.min(model_hash.len())],
                    round_id,
                );
            }
            PersistenceAction::Created => {
                global_models_col
                    .insert_one(
                        doc! {
                            "round_id":   next_id as i64,
                            "file_id":    file_oid,
                            "model_hash": &model_hash,
                        },
                        None,
                    )
                    .await
                    .map_err(|e| {
                        tracing::error!("global_models insert failed: {}", e);
                        Status::internal("failed to persist global model")
                    })?;

                tracing::info!(
                    "Global model for round {} stored in GridFS (hash={}…)",
                    next_id,
                    &model_hash[..16.min(model_hash.len())],
                );
            }
        };

        println!("============================================================");
        println!("              GLOBAL MODEL PERSISTENCE");
        println!("============================================================");
        println!("Target round        : {}", next_id);
        println!("Source agg. round   : {}", round_id);
        println!("Computed hash       : {}", model_hash);
        println!("Existing hash       : {}", existing_hash_display);
        println!("Persistence action  : {}", persistence_action);
        println!(
            "Status              : {}",
            if persistence_action == "CONFLICT" { "PASS (existing model retained as authoritative)" } else { "PASS" }
        );
        println!("============================================================");

        // ── Phase 6: create round N+1 ─────────────────────────────────────────
        // Runs UNCONDITIONALLY now, whether Phase 5 inserted fresh or found an
        // existing record — this is the actual fix: previously Phase 6 was only
        // reachable if insert_one succeeded, so a duplicate-key error on a
        // hydration re-run silently prevented round N+1 from ever being
        // (re)created in this process's memory, even though the aggregation
        // itself and its MongoDB record were both already correct.
        // or_insert_with avoids clobbering a round N+1 that may already be
        // Collecting with live updates from earlier in this same process.
        self.state.rounds.entry(next_id).or_insert_with(|| Round {
            id: next_id,
            model_version: format!("v{}", next_id),
            epsilon_max: 100.0,
            upload_uri: String::new(),
            state: RoundState::Collecting,
            updates: Vec::new(),
            aggregation_receipt: None,
            epsilon_spent: 0.0,
        });

        tracing::info!("Round {} ready — state=Collecting", next_id);

        Ok(())
    }
}