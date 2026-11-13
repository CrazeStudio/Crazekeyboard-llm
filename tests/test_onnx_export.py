"""Numerical parity test for the exported CrazeLM ONNX graph.

The release workflow sets CRAZELM_CHECKPOINT and CRAZELM_ONNX after exporting.
"""
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

from crazelm.model import CrazeLM, CrazeLMConfig


@pytest.fixture(scope="module")
def exported_model():
    checkpoint_value = os.environ.get("CRAZELM_CHECKPOINT")
    onnx_value = os.environ.get("CRAZELM_ONNX")
    if not checkpoint_value or not onnx_value:
        pytest.skip("Set CRAZELM_CHECKPOINT and CRAZELM_ONNX to run release parity test.")

    checkpoint_path = Path(checkpoint_value)
    onnx_path = Path(onnx_value)
    assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
    assert onnx_path.is_file(), f"Missing ONNX model: {onnx_path}"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = CrazeLMConfig(**checkpoint["config"])
    model = CrazeLM(config).cpu().eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    return config, model, session


@pytest.mark.parametrize("batch,sequence", [(1, 1), (1, 7), (2, 16)])
def test_onnx_logits_match_pytorch(exported_model, batch, sequence):
    config, model, session = exported_model
    assert sequence <= config.max_seq_len
    rng = np.random.default_rng(2026 + batch + sequence)
    input_ids = rng.integers(
        0, config.vocab_size, size=(batch, sequence), dtype=np.int64
    )

    with torch.inference_mode():
        expected, _ = model(torch.from_numpy(input_ids))
    actual = session.run(
        ["logits"], {"input_ids": input_ids.astype(np.int64)}
    )[0]

    assert actual.shape == (batch, sequence, config.vocab_size)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual, expected.cpu().numpy(), rtol=2e-3, atol=2e-3
    )


def test_onnx_has_expected_io(exported_model):
    config, _, session = exported_model
    assert [x.name for x in session.get_inputs()] == ["input_ids"]
    assert [x.name for x in session.get_outputs()] == ["logits"]
    output_shape = session.get_outputs()[0].shape
    assert output_shape[-1] == config.vocab_size
