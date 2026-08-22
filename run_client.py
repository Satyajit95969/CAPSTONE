import sys
import os

REPO_ROOT = r"D:\Download D\BE PIPELINE\Capstone-"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.path.append(os.path.join(REPO_ROOT, "installer", "runtime"))

os.environ.setdefault("FED_SERVER", "127.0.0.1:50051")

sys.argv = ["federated_client", "run-once"]

import runpy
runpy.run_module("runtime.federated_client", run_name="__main__")
