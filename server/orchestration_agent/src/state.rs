use dashmap::DashMap;
use std::sync::Arc;
use crate::round::Round;

pub type DeviceId = Vec<u8>;

/// device_fingerprint -> (device_pubkey_bytes, csr_bytes)
pub type PendingEnrollment = (Vec<u8>, Vec<u8>);

pub struct OrchestratorState {
    /// Registered devices: device_id (SHA-256 of pubkey) → pubkey bytes
    pub devices: DashMap<DeviceId, Vec<u8>>,
    /// Active federated rounds
    pub rounds: DashMap<u64, Round>,
    /// Legacy: single enrollment tokens (kept for backward compat)
    pub enrollment_tokens: DashMap<String, ()>,
    /// Multi-device pending enrollments: fingerprint (8-byte hex) → (pubkey, csr)
    pub pending_enrollments: DashMap<String, PendingEnrollment>,
}

impl OrchestratorState {
    /// FIX-RECOVERY-1: `rounds` starts EMPTY. Previously this hardcoded round 1
    /// into the map at construction time, which meant `hydrate_and_resume_aggregation()`
    /// — which only ever walks round_ids already present in this map — could
    /// never discover round 2, round 3, etc. that a prior process instance had
    /// already created and possibly completed. The map is now populated
    /// entirely by `hydrate_and_resume_aggregation()` at startup, which
    /// reconstructs the full round chain from MongoDB (receipts + global_models),
    /// including the fresh-install case (no MongoDB data yet) where it creates
    /// round 1 itself. See server.rs for the recovery logic.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            devices: DashMap::new(),
            rounds: DashMap::new(),
            enrollment_tokens: DashMap::new(),
            pending_enrollments: DashMap::new(),
        })
    }
}