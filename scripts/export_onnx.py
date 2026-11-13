#!/usr/bin/env python3
"""Export a CrazeLM PyTorch artifact (checkpoint, state dict, module, or TorchScript) to ONNX."""
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
    """Normalize CrazeLM's (logits, loss) or logits-only forward to logits."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids)
        if isinstance(output, (tuple, list)):
            return output[0]
        if isinstance(output, dict):
            if "logits" not in output:
                raise RuntimeError("Model returned a dict without a 'logits' field.")
            return output["logits"]
        return output


def _config_from_object(model: torch.nn.Module):
    cfg = getattr(model, "cfg", None) or getattr(model, "config", None)
    if isinstance(cfg, CrazeLMConfig):
        return cfg
    if isinstance(cfg, dict):
        allowed = CrazeLMConfig.__dataclass_fields__
        return CrazeLMConfig(**{k: v for k, v in cfg.items() if k in allowed})
    # The repository's documented 30M architecture is the fallback for a bare
    # state_dict, where no config metadata was saved.
    return CrazeLMConfig()


def _load_torchscript(path: Path, device: torch.device):
    try:
        module = torch.jit.load(str(path), map_location=device)
        module.eval()
        return module
    except Exception:
        return None


def load_model(checkpoint_path: Path, device: torch.device):
    # TorchScript is a serialized module and is not a Python dict.
    scripted = _load_torchscript(checkpoint_path, device)
    if scripted is not None:
        return scripted, _config_from_object(scripted), {}

    artifact = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    # Some training pipelines serialize the complete nn.Module with torch.save.
    if isinstance(artifact, torch.nn.Module):
        artifact.eval()
        return artifact.to(device), _config_from_object(artifact), {}

    if not isinstance(artifact, dict):
        raise TypeError(
            "Unsupported .pt payload type: "
            f"{type(artifact).__name__}. Expected a TorchScript module, "
            "a serialized torch.nn.Module, a checkpoint dict, or a state_dict."
        )

    # Standard training checkpoint: {'model': state_dict, 'config': {...}, ...}
    if "model" in artifact and isinstance(artifact["model"], dict):
        config_data = artifact.get("config")
        config = (
            CrazeLMConfig(**config_data)
            if isinstance(config_data, dict)
            else CrazeLMConfig()
        )
        model = CrazeLM(config).to(device)
        model.load_state_dict(artifact["model"], strict=True)
        model.eval()
        return model, config, artifact

    # Raw state_dict: {'token_embedding.weight': Tensor, ...}
    state_dict = artifact.get("state_dict", artifact)
    if isinstance(state_dict, dict) and state_dict and all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        config = CrazeLMConfig()
        model = CrazeLM(config).to(device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "The .pt file is a raw state_dict, but it does not match the "
                "repository's default CrazeLMConfig (100k vocab, width 256, "
                "5 layers, 8 heads, context 128). The archive must include "
                "model_config.json or the exporter must be given its exact config. "
                f"Original load error: {exc}"
            ) from exc
        model.eval()
        return model, config, {}

    raise TypeError(
        "The .pt file loaded as a dict, but it is neither a CrazeLM checkpoint "
        "with a 'model' state_dict nor a raw tensor state_dict. "
        f"Top-level keys: {list(artifact)[:20]}"
    )


def export(checkpoint_path: Path, output_path: Path, opset: int = 18):
    device = torch.device("cpu")
    model, config, checkpoint = load_model(checkpoint_path, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    params = sum(p.numel() for p in model.parameters())
    metadata = {
        "model_name": "CrazeLM-30M",
        "format": "ONNX",
        "task": "causal-language-model-next-token-logits",
        "vocab_size": str(config.vocab_size),
        "max_seq_len": str(config.max_seq_len),
        "parameter_count": str(params),
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
    print(f"Parameters: {params:,}")
    print(f"Vocabulary: {config.vocab_size:,}")
    print(f"Maximum sequence length: {config.max_seq_len}")
    print(f"Loaded artifact type: {type(model).__name__}")


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
