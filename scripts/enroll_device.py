#!/usr/bin/env python3
"""
scripts/enroll_device.py

Thin, discoverable wrapper around the project's EXISTING enrollment
mechanism (enroll_step5.py at the repo root) — no enrollment logic is
duplicated here. This performs the real RequestEnrollment -> OTP ->
EnrollDevice RPC sequence against the running orchestrator; it does not
bypass or weaken the "device not enrolled" authorization check in any way.

Usage:
    .venv\\Scripts\\python.exe scripts\\enroll_device.py <orchestrator_stdout_log_file>

Example:
    .venv\\Scripts\\python.exe scripts\\enroll_device.py trainer_outputs\\orchestrator_multimodal.log

Requirements:
  - The orchestrator must already be running (its OTP is printed once at
    startup and is valid for only 10 minutes — run this promptly after
    starting it).
  - <orchestrator_stdout_log_file> must be the file the orchestrator's
    stdout was redirected to (the OTP line is parsed from it).
"""
import sys
import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if len(sys.argv) < 2:
    print("Usage: enroll_device.py <orchestrator_stdout_log_file>")
    sys.exit(1)

# Re-exec the existing enroll_step5.py logic in-place — zero duplication.
sys.argv = ["enroll_step5.py", sys.argv[1]]
runpy.run_path(str(REPO_ROOT / "enroll_step5.py"), run_name="__main__")
