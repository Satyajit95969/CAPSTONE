// clear_fed.js — operational cleanup for Clean-DB Option A (Phase 4)
// Drops ONLY the federated-round collections, preserving `devices`.
// Indexes are recreated automatically by the orchestrator's _ensure_indexes()
// on the next startup. This script makes no source/architecture changes.

const d = db.getSiblingDB("federated");

["model_updates", "receipts", "global_models", "fs.files", "fs.chunks"].forEach(function (c) {
  print(c + " dropped=" + d.getCollection(c).drop());
});

print("devices kept=" + d.devices.countDocuments({}));
