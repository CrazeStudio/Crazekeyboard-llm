"""Numerical parity and compact-size tests for CrazeLM's FP16 ONNX export."""
import os
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_onnx import LogitsOnly, load_model


@pytest.fixture(scope="module")
def exported_model():
    checkpoint_value = os.environ.get("CRAZELM_CHECKPOINT")
    onnx_value = os.environ.get("CRAZELM_ONNX")
    if not checkpoint_value or not onnx_value:
        pytest.skip("Set CRAZELM_CHECKPOINT and CRAZELM_ONNX to run release parity test.")

    checkpoint_path = Path(checkpoint_value)
    onnx_path = Path(onnx_value)
    assert checkpoint_path.is_file(), f"Missing .pt artifact: {checkpoint_path}"
    assert onnx_path.is_file(), f"Missing ONNX model: {onnx_path}"

    model, config, _ = load_model(checkpoint_path, torch.device("cpu"))
    wrapper = LogitsOnly(model).eval()
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return config, wrapper, session, onnx_path


@pytest.mark.parametrize("batch,sequence", [(1, 1), (1, 7), (2, 16)])
def test_fp16_onnx_logits_match_pytorch(exported_model, batch, sequence):
    config, model, session, _ = exported_model
    assert sequence <= config.max_seq_len
    rng = np.random.default_rng(2026 + batch + sequence)
    input_ids = rng.integers(
        0, config.vocab_size, size=(batch, sequence), dtype=np.int64
    )

    with torch.inference_mode():
        expected = model(torch.from_numpy(input_ids))
    actual = session.run(["logits"], {"input_ids": input_ids})[0]

    assert actual.shape == (batch, sequence, config.vocab_size)
    assert np.isfinite(actual).all()
    # FP16 internal arithmetic has a larger, but bounded, tolerance than FP32.
    np.testing.assert_allclose(
        actual, expected.cpu().numpy(), rtol=2e-2, atol=2e-2
    )


def test_onnx_has_expected_io(exported_model):
    config, _, session, _ = exported_model
    assert [x.name for x in session.get_inputs()] == ["input_ids"]
    assert [x.name for x in session.get_outputs()] == ["logits"]
    assert session.get_outputs()[0].shape[-1] == config.vocab_size


def test_export_is_smaller_than_release_checkpoint(exported_model):
    _, _, _, onnx_path = exported_model
    # The released model.pt is ~110 MB. FP16 weights plus shared tied embeddings
    # should keep the export comfortably below the original checkpoint size.
    assert onnx_path.stat().st_size < 100 * 1024 * 1024
