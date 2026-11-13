#!/usr/bin/env python3
"""Export a CrazeLM PyTorch artifact to a compact FP16 ONNX model."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import onnx
from onnx import numpy_helper
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crazelm.model import CrazeLM, CrazeLMConfig


class LogitsOnly(torch.nn.Module):
    """Normalize tuple, dict, or tensor model outputs to logits."""

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
    return CrazeLMConfig()


def _load_torchscript(path: Path, device: torch.device):
    try:
        module = torch.jit.load(str(path), map_location=device)
        module.eval()
        return module
    except Exception:
        return None


def load_model(checkpoint_path: Path, device: torch.device):
    scripted = _load_torchscript(checkpoint_path, device)
    if scripted is not None:
        return scripted, _config_from_object(scripted), {}

    artifact = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(artifact, torch.nn.Module):
        artifact.eval()
        return artifact.to(device), _config_from_object(artifact), {}

    if not isinstance(artifact, dict):
        raise TypeError(
            f"Unsupported .pt payload type: {type(artifact).__name__}; expected "
            "TorchScript, nn.Module, checkpoint dict, or tensor state_dict."
        )

    if "model" in artifact and isinstance(artifact["model"], dict):
        config_data = artifact.get("config")
        config = CrazeLMConfig(**config_data) if isinstance(config_data, dict) else CrazeLMConfig()
        model = CrazeLM(config).to(device)
        model.load_state_dict(artifact["model"], strict=True)
        model.eval()
        return model, config, artifact

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
                "Raw state_dict does not match the repository's default "
                "CrazeLMConfig. Supply the exact model config. " + str(exc)
            ) from exc
        model.eval()
        return model, config, {}

    raise TypeError(
        "The .pt file is a dict but is neither a model checkpoint nor a raw "
        f"tensor state_dict. Top-level keys: {list(artifact)[:20]}"
    )


def deduplicate_initializers(model: onnx.ModelProto) -> int:
    """Share identical or transposed initializers, including tied embedding/head weights."""
    seen_exact: dict[tuple, str] = {}
    seen_transpose: dict[tuple, str] = {}
    replacements: dict[str, str] = {}
    transpose_nodes = []
    kept = []
    removed = 0

    for tensor in model.graph.initializer:
        array = numpy_helper.to_array(tensor)
        raw = tensor.raw_data or array.tobytes(order="C")
        exact_key = (tensor.data_type, tuple(tensor.dims), hashlib.sha256(raw).digest())

        if exact_key in seen_exact:
            replacements[tensor.name] = seen_exact[exact_key]
            removed += 1
            continue

        transpose_key = None
        if array.ndim == 2:
            transpose_key = (
                tensor.data_type,
                tuple(reversed(tensor.dims)),
                hashlib.sha256(array.T.tobytes(order="C")).digest(),
            )
            source_name = seen_transpose.get(
                (tensor.data_type, tuple(tensor.dims), hashlib.sha256(raw).digest())
            )
            if source_name is not None:
                alias = f"{tensor.name}__shared_transpose"
                transpose_nodes.append(
                    onnx.helper.make_node(
                        "Transpose", [source_name], [alias], perm=[1, 0],
                        name=f"share_tied_weight_{len(transpose_nodes)}",
                    )
                )
                replacements[tensor.name] = alias
                removed += 1
                continue

        seen_exact[exact_key] = tensor.name
        kept.append(tensor)
        if transpose_key is not None:
            seen_transpose[transpose_key] = tensor.name

    if replacements:
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name in replacements:
                    node.input[index] = replacements[name]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        # Place generated transpose nodes before their consumers to preserve topological order.
        existing_nodes = list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(transpose_nodes)
        model.graph.node.extend(existing_nodes)
    return removed

def export(checkpoint_path: Path, output_path: Path, opset: int = 18):
    device = torch.device("cpu")
    model, config, checkpoint = load_model(checkpoint_path, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seq_len = min(8, config.max_seq_len)
    sample = torch.randint(0, config.vocab_size, (1, seq_len), dtype=torch.long)
    wrapper = LogitsOnly(model).eval()
    fp32_path = output_path.with_suffix(".fp32.tmp.onnx")

    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper, (sample,), str(fp32_path),
                input_names=["input_ids"], output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=opset, do_constant_folding=True,
                export_params=True, dynamo=False,
            )

        graph = onnx.load(str(fp32_path))
        onnx.checker.check_model(graph)
        duplicate_count = deduplicate_initializers(graph)

        # Keep int64 token IDs and float32 logits at the public interface while
        # storing and computing supported internal tensors in FP16.
        compact = convert_float_to_float16(
            graph, keep_io_types=True, disable_shape_infer=False
        )
        params = sum(p.numel() for p in model.parameters())
        metadata = {
            "model_name": "CrazeLM-30M",
            "format": "ONNX-FP16",
            "task": "causal-language-model-next-token-logits",
            "vocab_size": str(config.vocab_size),
            "max_seq_len": str(config.max_seq_len),
            "parameter_count": str(params),
            "checkpoint_step": str(checkpoint.get("step", "unknown")),
            "checkpoint_val_loss": str(checkpoint.get("val_loss", "unknown")),
            "precision": "FP16 weights and internal compute; FP32 IO",
            "deduplicated_initializers": str(duplicate_count),
        }
        del compact.metadata_props[:]
        for key, value in metadata.items():
            prop = compact.metadata_props.add()
            prop.key, prop.value = key, value
        onnx.checker.check_model(compact)
        onnx.save(compact, str(output_path))
    finally:
        fp32_path.unlink(missing_ok=True)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported compact FP16 ONNX: {output_path}")
    print(f"File size: {size_mb:.2f} MiB")
    print(f"Parameters: {params:,}; duplicate initializers removed: {duplicate_count}")
    print(f"Vocabulary: {config.vocab_size:,}; context: {config.max_seq_len}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("artifacts/CrazeLM-30M-v1.0-fp16.onnx"), type=Path)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.opset < 17:
        parser.error("Use ONNX opset 17 or newer.")
    export(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
