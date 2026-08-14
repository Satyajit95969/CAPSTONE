import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path.home() / ".federated"

def _secure_delete(path: Path):
    try:
        if path.is_file():
            size = path.stat().st_size
            with open(path, "wb") as f:
                f.write(os.urandom(size))
            path.unlink()
    except Exception:
        pass

def trigger_self_destruct(reason: str):
    """
    DEBUG MODE
    Disable actual destruction temporarily.
    """

    print(f"[SECURITY] SELF-DESTRUCT: {reason}")

    # ===== TEMPORARILY DISABLED =====
    # if BASE_DIR.exists():
    #     for root, dirs, files in os.walk(BASE_DIR, topdown=False):
    #         for name in files:
    #             _secure_delete(Path(root) / name)
    #         for name in dirs:
    #             try:
    #                 shutil.rmtree(Path(root) / name)
    #             except Exception:
    #                 pass
    #
    #     try:
    #         shutil.rmtree(BASE_DIR)
    #     except Exception:
    #         pass

    raise RuntimeError(reason)
