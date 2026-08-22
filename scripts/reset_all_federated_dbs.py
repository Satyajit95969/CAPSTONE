#!/usr/bin/env python3
"""
scripts/reset_all_federated_dbs.py

Resets BOTH federated-learning project databases to a clean, empty state:
    federated              (PIPELINE_MODE=text default)
    federated_multimodal   (PIPELINE_MODE=multimodal)

For each: clears model_updates, receipts, global_models, devices, and this
database's own GridFS (fs.files/fs.chunks).

Does NOT touch: any other database on this MongoDB instance (e.g. libraryDB,
admin, config, local), certificates/keys/TPM material, dataset files, or
pretrained models.

Usage:
    .venv\\Scripts\\python.exe scripts\\reset_all_federated_dbs.py

Exits 0 only if every collection in both databases reports zero documents
after the reset.
"""
import sys
import pymongo

MONGO_URI = "mongodb://localhost:27017"
DATABASES = ["federated", "federated_multimodal"]
COLLECTIONS = ["model_updates", "receipts", "global_models", "devices"]


def reset_db(client, db_name: str) -> bool:
    db = client[db_name]
    print(f"--- {db_name} ---")

    before = {name: db[name].count_documents({}) for name in COLLECTIONS}
    gridfs_files_before = db["fs.files"].count_documents({})
    gridfs_chunks_before = db["fs.chunks"].count_documents({})
    print(f"  before: {before}  fs.files={gridfs_files_before} fs.chunks={gridfs_chunks_before}")

    for name in COLLECTIONS:
        result = db[name].delete_many({})
        print(f"  {name:<15s}: deleted {result.deleted_count}")

    if gridfs_files_before or gridfs_chunks_before:
        gf_files = db["fs.files"].delete_many({})
        gf_chunks = db["fs.chunks"].delete_many({})
        print(f"  {'fs.files':<15s}: deleted {gf_files.deleted_count}")
        print(f"  {'fs.chunks':<15s}: deleted {gf_chunks.deleted_count}")

    after = {name: db[name].count_documents({}) for name in COLLECTIONS}
    print(f"  after:  {after}")
    ok = all(v == 0 for v in after.values())
    print(f"  [{'OK' if ok else 'FAIL'}]")
    print()
    return ok


def main() -> int:
    print("=" * 60)
    print("FEDERATED STATE RESET (all project databases)")
    print("=" * 60)
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except Exception as e:
        print(f"[FAIL] Could not connect to MongoDB: {e}")
        return 1

    all_ok = True
    for db_name in DATABASES:
        all_ok = reset_db(client, db_name) and all_ok

    print("=" * 60)
    if all_ok:
        print("[OK] All collections in all project databases are empty.")
    else:
        print("[FAIL] One or more collections still contain documents.")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
