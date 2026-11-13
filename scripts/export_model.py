#!/usr/bin/env python3
"""Validate a trained CrazeLM checkpoint and export a portable inference bundle."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crazelm.model import CrazeLM, CrazeLMConfig


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output_dir", default="CrazeLM-30M-export")
    p.add_argument("--dtype", choices=("fp32", "fp16"), default="fp16")
    args = p.parse_args()

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint_path.is_file():
        p.error(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        p.error(f"Tokenizer not found: {tokenizer_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise SystemExit("Checkpoint must contain 'model' and 'config' entries.")

    cfg = CrazeLMConfig(**checkpoint["config"])
    if tokenizer.get_piece_size() != cfg.vocab_size:
        raise SystemExit(
            f"Tokenizer/model mismatch: tokenizer={tokenizer.get_piece_size():,}, "
            f"model={cfg.vocab_size:,}"
        )

    model = CrazeLM(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    params = model.parameter_count()
    if not 29_000_000 <= params <= 31_000_000:
        raise SystemExit(f"Expected ~30M parameters, got {params:,}")

    # Smoke test: finite logits, valid vocabulary dimension, and a usable prediction.
    prompt = "I am going to"
    ids = tokenizer.encode(prompt, out_type=int)[-cfg.max_seq_len:]
    if not ids:
        raise SystemExit("Tokenizer returned no IDs for the smoke-test prompt.")
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(x)
        if logits.shape != (1, len(ids), cfg.vocab_size):
            raise SystemExit(f"Unexpected logits shape: {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise SystemExit("Smoke test failed: model produced non-finite logits.")
        top_ids = logits[0, -1].float().topk(8).indices.tolist()
        top_pieces = [tokenizer.decode([i]) for i in top_ids]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    export_state = {
        key: value.detach().cpu().to(dtype=dtype) if value.is_floating_point()
        else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    weights_path = out / "model.pt"
    torch.save(export_state, weights_path)
    shutil.copy2(tokenizer_path, out / "tokenizer.model")

    config = {
        **model.config_dict(),
        "parameter_count": params,
        "language": "English",
        "format": "PyTorch state_dict",
        "weight_dtype": args.dtype,
        "checkpoint_step": checkpoint.get("step"),
        "validation_loss": checkpoint.get("val_loss"),
        "tokenizer_file": "tokenizer.model",
        "weights_file": "model.pt",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Reload the exported weights and run the same smoke test again.
    exported_model = CrazeLM(cfg)
    exported_model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    exported_model.eval()
    with torch.inference_mode():
        exported_logits, _ = exported_model(x)
    if not torch.isfinite(exported_logits).all():
        raise SystemExit("Export verification failed: exported model produced non-finite logits.")
    if args.dtype == "fp32" and not torch.allclose(logits, exported_logits, atol=1e-5, rtol=1e-5):
        raise SystemExit("FP32 export verification failed: logits changed after reload.")

    manifest = {
        "model": "CrazeLM",
        "architecture": "decoder-only causal Transformer",
        "parameter_count": params,
        "vocabulary_size": tokenizer.get_piece_size(),
        "context_length": cfg.max_seq_len,
        "checkpoint_step": checkpoint.get("step"),
        "validation_loss": checkpoint.get("val_loss"),
        "smoke_test": "passed",
        "device_used_for_test": device,
        "top_8_next_pieces": top_pieces,
        "files": {
            "model.pt": {"bytes": weights_path.stat().st_size, "sha256": sha256(weights_path)},
            "tokenizer.model": {"bytes": (out / "tokenizer.model").stat().st_size,
                                "sha256": sha256(out / "tokenizer.model")},
            "config.json": {"bytes": (out / "config.json").stat().st_size,
                            "sha256": sha256(out / "config.json")},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"Export verification: PASSED ({args.dtype})")
    print(f"Parameters: {params:,}")
    print(f"Checkpoint step: {checkpoint.get('step')}")
    print(f"Validation loss: {checkpoint.get('val_loss')}")
    print("Top next pieces:", top_pieces)
    print(f"Bundle directory: {out.resolve()}")
    print(f"ZIP archive: {archive}")
    print(f"ZIP size: {Path(archive).stat().st_size / 1_000_000:.2f} MB")


if __name__ == "__main__":
    main()
