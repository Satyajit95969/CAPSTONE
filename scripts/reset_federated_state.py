#!/usr/bin/env python3
"""
scripts/reset_federated_state.py

Resets ONLY the four project collections in the federated_multimodal
MongoDB database to a clean, empty state:

    model_updates
    receipts
    global_models
    devices

Does NOT touch:
  - certificates, keys, TPM material (~/.federated/keys, ~/.federated/tpm)
  - dataset files (dataset_build/*)
  - pretrained models (~/.federated/models)
  - source code
  - any other MongoDB database (federated, admin, config, local, libraryDB, ...)
  - GridFS chunks/files from OTHER databases (fs.files/fs.chunks are also
    cleared, but only within federated_multimodal — see note below)

Usage:
    .venv\\Scripts\\python.exe scripts\\reset_federated_state.py

Exits 0 only if all four collections report zero documents after the reset.
Exits 1 on any connection error or if any collection is non-empty afterward.
"""
import sys
import pymongo

MONGO_URI = "mongodb://localhost:27017"
DATABASE = "federated_multimodal"
COLLECTIONS = ["model_updates", "receipts", "global_models", "devices"]


def main() -> int:
    print("=" * 60)
    print("FEDERATED STATE RESET")
    print("=" * 60)
    print(f"MongoDB URI : {MONGO_URI}")
    print(f"Database    : {DATABASE}")
    print()

    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except Exception as e:
        print(f"[FAIL] Could not connect to MongoDB: {e}")
        return 1

    db = client[DATABASE]

    print("BEFORE reset:")
    before = {}
    for name in COLLECTIONS:
        before[name] = db[name].count_documents({})
        print(f"  {name:<15s}: {before[name]}")

    # Also clear this database's own GridFS (fs.files/fs.chunks hold the
    # aggregated global model bytes referenced by global_models.file_id —
    # clearing global_models without clearing GridFS would leave orphaned
    # blobs, not corrupt anything, but a "fresh" reset should remove them
    # too since they're only ever referenced from this same database).
    gridfs_files_before = db["fs.files"].count_documents({})
    gridfs_chunks_before = db["fs.chunks"].count_documents({})

    print()
    print("Deleting documents...")
    for name in COLLECTIONS:
        result = db[name].delete_many({})
        print(f"  {name:<15s}: deleted {result.deleted_count}")

    if gridfs_files_before or gridfs_chunks_before:
        gf_files = db["fs.files"].delete_many({})
        gf_chunks = db["fs.chunks"].delete_many({})
        print(f"  {'fs.files':<15s}: deleted {gf_files.deleted_count}")
        print(f"  {'fs.chunks':<15s}: deleted {gf_chunks.deleted_count}")

    print()
    print("AFTER reset:")
    after = {}
    all_zero = True
    for name in COLLECTIONS:
        after[name] = db[name].count_documents({})
        print(f"  {name:<15s}: {after[name]}")
        if after[name] != 0:
            all_zero = False

    print("=" * 60)
    if all_zero:
        print("[OK] All four collections are empty. Reset successful.")
        print("=" * 60)
        return 0
    else:
        print("[FAIL] One or more collections still contain documents.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
