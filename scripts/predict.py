#!/usr/bin/env python3
"""Run next-token predictions from a trained CrazeLM checkpoint."""
import argparse
import torch
import sentencepiece as spm
from pathlib import Path
import sys

# Allow direct execution as `python scripts/predict.py` from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crazelm.model import CrazeLM, CrazeLMConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()
    if args.top_k < 1 or args.temperature <= 0:
        p.error("--top_k must be positive and --temperature must be greater than zero")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = CrazeLMConfig(**checkpoint["config"])
    model = CrazeLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ids = tokenizer.encode(args.text, out_type=int)[-config.max_seq_len:]
    if not ids:
        raise SystemExit("Prompt produced no tokens.")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(x)
        probs = torch.softmax(logits[0, -1].float() / args.temperature, dim=-1)
        count = min(args.top_k, probs.numel())
        values, indices = torch.topk(probs, count)
    print(f"Prompt: {args.text}")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')} | validation loss: {checkpoint.get('val_loss')}")
    for token_id, score in zip(indices.tolist(), values.tolist()):
        piece = tokenizer.decode([token_id])
        print(f"{piece!r}  {score:.5f}")


if __name__ == "__main__":
    main()
