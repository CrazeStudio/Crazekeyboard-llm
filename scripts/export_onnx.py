#!/usr/bin/env python3
"""Export a CrazeLM PyTorch checkpoint to ONNX for Android inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import onnx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crazelm.model import CrazeLM, CrazeLMConfig


class LogitsOnly(torch.nn.Module):
    """Expose only next-token logits; the training loss is not an inference output."""

    def __init__(self, model: CrazeLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        return logits


def load_model(checkpoint_path: Path, device: torch.device):
    # The release checkpoint is a trusted project artifact and contains optimizer
    # metadata in addition to the model state_dict.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            "Unsupported checkpoint: expected a dict containing 'model' and 'config'."
        )
    config = CrazeLMConfig(**checkpoint["config"])
    model = CrazeLM(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, checkpoint


def export(checkpoint_path: Path, output_path: Path, opset: int = 18):
    device = torch.device("cpu")
    model, config, checkpoint = load_model(checkpoint_path, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use non-special token IDs and a short sequence for tracing. Both batch and
    # sequence dimensions remain dynamic; the model still enforces max_seq_len.
    seq_len = min(8, config.max_seq_len)
    sample = torch.randint(
        low=0, high=config.vocab_size, size=(1, seq_len), dtype=torch.long
    )
    wrapper = LogitsOnly(model).eval()

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (sample,),
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    metadata = {
        "model_name": "CrazeLM-30M",
        "format": "ONNX",
        "task": "causal-language-model-next-token-logits",
        "vocab_size": str(config.vocab_size),
        "max_seq_len": str(config.max_seq_len),
        "parameter_count": str(model.parameter_count()),
        "checkpoint_step": str(checkpoint.get("step", "unknown")),
        "checkpoint_val_loss": str(checkpoint.get("val_loss", "unknown")),
    }
    del exported.metadata_props[:]
    for key, value in metadata.items():
        prop = exported.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(exported, str(output_path))

    print(f"Exported: {output_path}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Vocabulary: {config.vocab_size:,}")
    print(f"Maximum sequence length: {config.max_seq_len}")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("artifacts/CrazeLM-30M-v1.0.onnx"), type=Path)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.opset < 17:
        parser.error("Use ONNX opset 17 or newer.")
    export(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
