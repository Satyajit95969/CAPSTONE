import sys
import os

REPO_ROOT = r"D:\Download D\BE PIPELINE\Capstone-"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.path.append(os.path.join(REPO_ROOT, "installer", "runtime"))

os.environ.setdefault("FED_SERVER", "127.0.0.1:50051")
os.environ["PIPELINE_MODE"] = "multimodal"
# Fix D: "0" = no limit, load the full local partition (186 participants).
# Override with a positive int (e.g. MULTIMODAL_MAX_SAMPLES=1) for a quick
# smoke test - truncation is opt-in, not the default.
os.environ.setdefault("MULTIMODAL_MAX_SAMPLES", "0")

sys.argv = ["federated_client", "run-once"]

import runpy
runpy.run_module("runtime.federated_client", run_name="__main__")
