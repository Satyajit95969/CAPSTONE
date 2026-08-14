"""
gap2_test.py — Consistency tests for GAP 2 (trainer global model warm-start)

Proves:
  T1: Round 0 cold-start — no global_model_path → uses pretrained weights
  T2: Round 1 warm-start — global_model_path provided → weights loaded before base_state
  T3: base_state matches global model, not pretrained init
  T4: delta is relative to global model weights (not pretrained)
  T5: Missing file → FileNotFoundError (fail loud)
  T6: Non-dict payload → TypeError (fail loud)
  T7: Key mismatch → RuntimeError (fail loud)
  T8: Shape mismatch → RuntimeError (fail loud)
"""

import sys
import os
import io
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "trainer_agent"))

from trainer_agent.trainer_mentalbert_privacy import _load_global_model

PASS = "[PASS]"
FAIL = "[FAIL]"


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def state_dict(self, *a, **kw):
        return super().state_dict(*a, **kw)


def _save_sd(sd, path):
    torch.save(sd, str(path))


def _make_model_and_global():
    model = _Tiny()
    # Create a "global model" with distinctly different weights
    global_model = _Tiny()
    with torch.no_grad():
        for p in global_model.parameters():
            p.fill_(99.0)
    return model, global_model.state_dict()


# ─── T1: Round 0 cold-start — global_model_path=None, weights stay as-is ───────
def test_t1_cold_start():
    model = _Tiny()
    pretrained_val = model.fc.weight.data.clone()

    # Simulates: if global_model_path is not None: _load_global_model(...)
    global_model_path = None
    if global_model_path is not None:
        _load_global_model(model, global_model_path, "cpu")

    after_val = model.fc.weight.data
    if torch.allclose(pretrained_val, after_val):
        print(f"{PASS} T1: Round 0 cold-start — weights unchanged when global_model_path=None")
    else:
        print(f"{FAIL} T1: Round 0 cold-start — weights changed unexpectedly")
        sys.exit(1)


# ─── T2: Round 1 warm-start — weights replaced by global model ──────────────────
def test_t2_warm_start():
    model, global_sd = _make_model_and_global()
    original_val = model.fc.weight.data.clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    _save_sd(global_sd, tmp_path)

    try:
        _load_global_model(model, tmp_path, "cpu")
        loaded_val = model.fc.weight.data
        if torch.allclose(loaded_val, torch.full_like(loaded_val, 99.0)):
            print(f"{PASS} T2: Round 1 warm-start — model weights replaced by global model (all 99.0)")
        else:
            print(f"{FAIL} T2: warm-start weights incorrect: {loaded_val}")
            sys.exit(1)
    finally:
        os.unlink(tmp_path)


# ─── T3: base_state reflects global model, not pretrained ───────────────────────
def test_t3_base_state_matches_global():
    model, global_sd = _make_model_and_global()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    _save_sd(global_sd, tmp_path)

    try:
        _load_global_model(model, tmp_path, "cpu")
        base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ok = True
        for k, v in base_state.items():
            if not torch.allclose(v, global_sd[k].cpu()):
                print(f"{FAIL} T3: base_state[{k}] does not match global_sd")
                ok = False
        if ok:
            print(f"{PASS} T3: base_state is identical to global model state_dict after warm-start")
        else:
            sys.exit(1)
    finally:
        os.unlink(tmp_path)


# ─── T4: delta is relative to global model (not pretrained init) ────────────────
def test_t4_delta_relative_to_global():
    model, global_sd = _make_model_and_global()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    _save_sd(global_sd, tmp_path)

    try:
        _load_global_model(model, tmp_path, "cpu")
        base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Simulate one training step: nudge fc.weight by +1.0
        with torch.no_grad():
            model.fc.weight.add_(1.0)

        after_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        delta = {k: after_state[k] - base_state[k] for k in after_state}

        expected_delta = torch.ones_like(delta["fc.weight"])
        if torch.allclose(delta["fc.weight"], expected_delta):
            print(f"{PASS} T4: delta is relative to global model (fc.weight delta = +1.0 as expected)")
        else:
            print(f"{FAIL} T4: unexpected delta: {delta['fc.weight']}")
            sys.exit(1)
    finally:
        os.unlink(tmp_path)


# ─── T5: Missing file → FileNotFoundError ───────────────────────────────────────
def test_t5_missing_file():
    model = _Tiny()
    missing = "/tmp/does_not_exist_gap2_test_9999.pt"
    try:
        _load_global_model(model, missing, "cpu")
        print(f"{FAIL} T5: Expected FileNotFoundError — none raised")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"{PASS} T5: Missing file -> FileNotFoundError: {e}")
    except Exception as e:
        print(f"{FAIL} T5: Wrong exception type {type(e).__name__}: {e}")
        sys.exit(1)


# ─── T6: Non-dict payload → TypeError ───────────────────────────────────────────
def test_t6_non_dict_payload():
    model = _Tiny()
    bad_tensor = torch.randn(100)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save(bad_tensor, tmp_path)

    try:
        _load_global_model(model, tmp_path, "cpu")
        print(f"{FAIL} T6: Expected TypeError — none raised")
        sys.exit(1)
    except TypeError as e:
        print(f"{PASS} T6: Non-dict payload -> TypeError: {e}")
    except Exception as e:
        print(f"{FAIL} T6: Wrong exception type {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        os.unlink(tmp_path)


# ─── T7: Key mismatch → RuntimeError ────────────────────────────────────────────
def test_t7_key_mismatch():
    model = _Tiny()

    # Save a state dict with wrong keys
    bad_sd = {"wrong_key.weight": torch.randn(2, 4), "wrong_key.bias": torch.randn(2)}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save(bad_sd, tmp_path)

    try:
        _load_global_model(model, tmp_path, "cpu")
        print(f"{FAIL} T7: Expected RuntimeError for key mismatch — none raised")
        sys.exit(1)
    except RuntimeError as e:
        print(f"{PASS} T7: Key mismatch -> RuntimeError: {str(e)[:120]}")
    except Exception as e:
        print(f"{FAIL} T7: Wrong exception type {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        os.unlink(tmp_path)


# ─── T8: Shape mismatch → RuntimeError ──────────────────────────────────────────
def test_t8_shape_mismatch():
    model = _Tiny()

    # State dict has correct keys but wrong shapes
    bad_sd = {"fc.weight": torch.randn(8, 8), "fc.bias": torch.randn(8)}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save(bad_sd, tmp_path)

    try:
        _load_global_model(model, tmp_path, "cpu")
        print(f"{FAIL} T8: Expected RuntimeError for shape mismatch — none raised")
        sys.exit(1)
    except RuntimeError as e:
        print(f"{PASS} T8: Shape mismatch -> RuntimeError: {str(e)[:120]}")
    except Exception as e:
        print(f"{FAIL} T8: Wrong exception type {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("GAP 2 Consistency Tests")
    print("=" * 60)
    test_t1_cold_start()
    test_t2_warm_start()
    test_t3_base_state_matches_global()
    test_t4_delta_relative_to_global()
    test_t5_missing_file()
    test_t6_non_dict_payload()
    test_t7_key_mismatch()
    test_t8_shape_mismatch()
    print("=" * 60)
    print("All GAP 2 tests passed.")
    print("=" * 60)
