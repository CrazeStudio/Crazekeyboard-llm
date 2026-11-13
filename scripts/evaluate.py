#!/usr/bin/env python3
"""Evaluate a CrazeLM checkpoint on a held-out text directory."""
import argparse
import math
import numpy as np
import sentencepiece as spm
import torch
from pathlib import Path
import sys

# Allow direct execution as `python scripts/evaluate.py` from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crazelm.model import CrazeLM, CrazeLMConfig


def encode(folder, tokenizer):
    files = sorted(Path(folder).rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {folder}")
    ids = []
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = " ".join(line.split())
            if line:
                ids.extend(tokenizer.encode(line, out_type=int))
                ids.append(tokenizer.eos_id())
    return np.asarray(ids, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=200)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = CrazeLMConfig(**ckpt["config"])
    model = CrazeLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    data = encode(args.test_dir, tokenizer)
    length, seq = len(data), cfg.max_seq_len
    if length <= seq:
        raise SystemExit("Test token stream is too short.")
    losses, correct1, correct5, tokens = [], 0, 0, 0
    # Deterministic, non-overlapping batches over the held-out stream.
    batch_tokens = args.batch_size * seq
    starts = list(range(0, length - seq - 1, batch_tokens))[:args.max_batches]
    with torch.inference_mode():
        for start in starts:
            xs, ys = [], []
            for b in range(args.batch_size):
                s = start + b * seq
                if s + seq + 1 >= length:
                    break
                xs.append(data[s:s+seq])
                ys.append(data[s+1:s+seq+1])
            if not xs:
                continue
            x = torch.tensor(np.stack(xs), dtype=torch.long, device=device)
            y = torch.tensor(np.stack(ys), dtype=torch.long, device=device)
            logits, loss = model(x, y)
            losses.append(loss.item())
            top5 = logits.topk(5, dim=-1).indices
            correct1 += (logits.argmax(-1) == y).sum().item()
            correct5 += (top5 == y.unsqueeze(-1)).any(-1).sum().item()
            tokens += y.numel()
    if not tokens:
        raise SystemExit("No test batches evaluated.")
    mean_loss = sum(losses) / len(losses)
    print(f"Test loss: {mean_loss:.4f}")
    print(f"Perplexity: {math.exp(min(mean_loss, 20)):.2f}")
    print(f"Top-1 token accuracy: {100*correct1/tokens:.2f}%")
    print(f"Top-5 token accuracy: {100*correct5/tokens:.2f}%")
    print(f"Evaluated tokens: {tokens:,}")


if __name__ == "__main__":
    main()
