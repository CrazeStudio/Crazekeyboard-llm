#!/usr/bin/env python3
"""Export the CrazeLM checkpoint directly to a LiteRT/TFLite model.

This deliberately does not use the ONNX release. The wrapper keeps the input
shape fixed and emits only the final valid token's logits, reducing activation
memory compared with the full [batch, sequence, vocabulary] graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
import litert_torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crazelm.model import CrazeLM, CrazeLMConfig


SEQUENCE_LENGTH = 48


class LiteRTNextToken(nn.Module):
    """Export-only inference graph using primitive causal-attention ops."""

    def __init__(self, model: CrazeLM, sequence_length: int):
        super().__init__()
        self.model = model
        self.sequence_length = sequence_length
        head_dim = model.cfg.d_model // model.cfg.n_heads
        # Additive causal mask: later positions cannot affect earlier positions.
        mask = torch.triu(
            torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.float32),
            diagonal=1,
        ) * -10000.0
        self.register_buffer("causal_bias", mask, persistent=False)
        self.scale = float(head_dim) ** -0.5

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        ids = input_ids.to(torch.int64)
        batch = ids.shape[0]
        positions = torch.arange(self.sequence_length, device=ids.device).unsqueeze(0)
        x = self.model.token_embedding(ids) + self.model.position_embedding(positions)

        for block in self.model.blocks:
            normalized = block.ln1(x)
            qkv = block.attn.qkv(normalized)
            q, k, v = qkv.chunk(3, dim=-1)
            shape = (
                batch,
                self.sequence_length,
                self.model.cfg.n_heads,
                self.model.cfg.d_model // self.model.cfg.n_heads,
            )
            q = q.reshape(shape).transpose(1, 2)
            k = k.reshape(shape).transpose(1, 2)
            v = v.reshape(shape).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            weights = torch.softmax(scores + self.causal_bias, dim=-1)
            attended = torch.matmul(weights, v)
            attended = attended.transpose(1, 2).contiguous().reshape(
                batch, self.sequence_length, self.model.cfg.d_model
            )
            x = x + block.attn.proj(attended)
            x = x + block.ff(block.ln2(x))

        # LayerNorm acts independently on each sequence position. Select the
        # last real token before the vocabulary projection to avoid [1,48,100k].
        last_index = (lengths.to(torch.int64) - 1).reshape(batch, 1, 1)
        gather_index = last_index.expand(batch, 1, self.model.cfg.d_model)
        last_hidden = torch.gather(x, dim=1, index=gather_index).squeeze(1)
        last_hidden = self.model.final_ln(last_hidden)
        return F.linear(last_hidden, self.model.token_embedding.weight)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_release(archive: Path, destination: Path) -> None:
    """Extract ZIP safely; reject path traversal and symlinks."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (root / item.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {item.filename}")
            # ZIP symlinks are not expected in model bundles; reject them.
            mode = (item.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise RuntimeError(f"Symlink in model archive: {item.filename}")
        bundle.extractall(root)


def find_artifact(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} was not found in the release archive")
    return sorted(matches, key=lambda p: (len(p.parts), str(p)))[0]


def load_release_model(root: Path):
    # The v1.0 release archive is a deployment bundle: model.pt is an FP16
    # state_dict, while config.json contains the checkpoint metadata.
    weights_path = find_artifact(root, "model.pt")
    config_path = find_artifact(root, "config.json")
    release_config = json.loads(config_path.read_text(encoding="utf-8"))
    config_fields = {
        "vocab_size", "max_seq_len", "d_model", "n_layers",
        "n_heads", "d_ff", "dropout",
    }
    model_config = CrazeLMConfig(**{
        key: release_config[key] for key in config_fields if key in release_config
    })
    model = CrazeLM(model_config)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("The release model.pt is not a PyTorch state_dict.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metadata = {
        "step": release_config.get("checkpoint_step"),
        "val_loss": release_config.get("validation_loss"),
        "weight_dtype": release_config.get("weight_dtype"),
    }
    return model, metadata


def encode_prompt(tokenizer, prompt: str, length: int) -> tuple[np.ndarray, np.ndarray]:
    ids = tokenizer.encode(prompt, out_type=int)[-length:]
    if not ids:
        raise RuntimeError(f"Tokenizer returned no IDs for: {prompt!r}")
    padded = np.zeros((1, length), dtype=np.int32)
    padded[0, :len(ids)] = np.asarray(ids, dtype=np.int32)
    return padded, np.asarray([len(ids)], dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path,
                        help="CrazeLM-30M-v1.0.zip release archive containing best.pt and tokenizer.model")
    parser.add_argument("--output", default=Path("CrazeLM-30M-v1.0.tflite"), type=Path)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"Release archive not found: {args.archive}")
    if not 8 <= args.sequence_length <= 128:
        parser.error("--sequence-length must be between 8 and 128")

    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="crazelm-tflite-") as temp:
        extracted = Path(temp)
        extract_release(args.archive, extracted)
        tokenizer_path = find_artifact(extracted, "tokenizer.model")
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        model, checkpoint = load_release_model(extracted)

        if tokenizer.get_piece_size() != model.cfg.vocab_size:
            raise RuntimeError(
                f"Tokenizer/model vocabulary mismatch: {tokenizer.get_piece_size()} vs {model.cfg.vocab_size}"
            )
        if model.cfg.vocab_size != 100_000 or model.cfg.max_seq_len != 128:
            raise RuntimeError(
                f"Unexpected CrazeLM config: vocab={model.cfg.vocab_size}, context={model.cfg.max_seq_len}"
            )

        sequence_length = args.sequence_length
        wrapper = LiteRTNextToken(model, sequence_length).eval()
        test_prompts = [
            "I am going to",
            "The weather is",
            "Thank you for",
            "I would like to",
        ]

        sample_ids, sample_lengths = encode_prompt(tokenizer, test_prompts[0], sequence_length)
        sample_inputs = (
            torch.from_numpy(sample_ids),
            torch.from_numpy(sample_lengths),
        )

        with torch.inference_mode():
            expected = wrapper(*sample_inputs).cpu().numpy()
            if expected.shape != (1, model.cfg.vocab_size) or not np.isfinite(expected).all():
                raise RuntimeError(f"Invalid PyTorch export output shape/values: {expected.shape}")

        print("Converting PyTorch CrazeLM directly with litert_torch...")
        edge_model = litert_torch.convert(wrapper, sample_inputs)

        # First validate the in-memory converted graph on several prompts.
        for prompt in test_prompts:
            ids_np, lengths_np = encode_prompt(tokenizer, prompt, sequence_length)
            torch_inputs = (torch.from_numpy(ids_np), torch.from_numpy(lengths_np))
            with torch.inference_mode():
                torch_logits = wrapper(*torch_inputs).cpu().numpy()
            edge_logits = edge_model(*torch_inputs)
            edge_logits = np.asarray(edge_logits)
            if edge_logits.shape != torch_logits.shape:
                raise RuntimeError(
                    f"Converted output shape mismatch for {prompt!r}: "
                    f"{edge_logits.shape} vs {torch_logits.shape}"
                )
            if not np.isfinite(edge_logits).all():
                raise RuntimeError(f"Non-finite LiteRT output for prompt {prompt!r}")
            max_error = float(np.max(np.abs(edge_logits - torch_logits)))
            if not np.allclose(edge_logits, torch_logits, rtol=2e-3, atol=2e-3):
                torch_top = int(torch_logits[0].argmax())
                edge_top = int(edge_logits[0].argmax())
                raise RuntimeError(
                    f"LiteRT parity failed for {prompt!r}: max_abs_error={max_error}, "
                    f"torch_top={torch_top}, edge_top={edge_top}"
                )
            print(f"Parity PASS: {prompt!r} (max_abs_error={max_error:.6g})")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        edge_model.export(str(args.output))

    manifest = {
        "model": "CrazeLM-30M-v1.0",
        "format": "LiteRT/TFLite",
        "input_ids": {"dtype": "int32", "shape": [1, args.sequence_length], "padding": "right"},
        "lengths": {"dtype": "int32", "shape": [1]},
        "output": {"dtype": "float32", "shape": [1, 100000], "meaning": "last valid token logits"},
        "vocabulary_size": 100000,
        "context_length": args.sequence_length,
        "checkpoint_step": checkpoint.get("step"),
        "validation_loss": checkpoint.get("val_loss"),
        "conversion": "litert_torch direct from PyTorch checkpoint; no ONNX intermediate",
        "size_bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "parity_prompts": test_prompts,
        "parity_tolerance": {"rtol": 2e-3, "atol": 2e-3},
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
