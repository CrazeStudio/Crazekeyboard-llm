import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm

from crazelm.model import CrazeLM, CrazeLMConfig


def encode_dir(folder, tokenizer, output):
    paths = sorted(Path(folder).rglob("*.txt"))
    if not paths:
        raise RuntimeError(f"No .txt files found in {folder}")
    total = 0
    with open(output, "wb") as dst:
        for path in tqdm(paths, desc=f"Tokenizing {folder}"):
            with path.open("r", encoding="utf-8", errors="ignore") as src:
                for line in src:
                    ids = tokenizer.encode(line.strip(), out_type=int)
                    if len(ids) > 1:
                        arr = np.asarray(ids, dtype=np.uint16)
                        arr.tofile(dst)
                        total += len(arr)
    if total < 1000:
        raise RuntimeError(f"Only {total} tokens found; provide a larger corpus")
    return total


def sample_batch(data, batch_size, seq_len, device):
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(data[int(i):int(i)+seq_len].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[int(i)+1:int(i)+seq_len+1].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, batch_size, seq_len, steps, device):
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = sample_batch(data, batch_size, seq_len, device)
        _, loss = model(x, y)
        losses.append(float(loss))
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output_dir", default="artifacts/crazelm")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--steps_per_epoch", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    train_bin = out / "train_tokens.bin"
    train_count = encode_dir(args.train_dir, tokenizer, train_bin)
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = None
    if args.val_dir:
        val_bin = out / "val_tokens.bin"
        encode_dir(args.val_dir, tokenizer, val_bin)
        val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    cfg = CrazeLMConfig(vocab_size=tokenizer.get_piece_size(), max_seq_len=args.seq_len)
    model = CrazeLM(cfg).to(device)
    count = model.parameter_count()
    print(f"Device: {device}; parameters: {count:,}; train tokens: {train_count:,}")
    print(f"FP32 weights: {count * 4 / 1e6:.2f} MB")
    with (out / "model_config.json").open("w") as f:
        json.dump({**model.config_dict(), "parameter_count": count}, f, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best = float("inf")
    metrics = out / "metrics.jsonl"
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        started = time.time()
        for step in tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = sample_batch(train_data, args.batch_size, args.seq_len, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            global_step += 1
            if global_step % 100 == 0:
                print(f"step={global_step} train_loss={running/100:.4f}")
                running = 0.0
            if global_step % 500 == 0:
                val_loss = evaluate(model, val_data, args.batch_size, args.seq_len, args.eval_steps, device) if val_data is not None else None
                record = {"step": global_step, "epoch": epoch + 1, "val_loss": val_loss}
                with metrics.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                checkpoint = {"model": model.state_dict(), "config": model.config_dict(), "step": global_step, "val_loss": val_loss}
                torch.save(checkpoint, out / "last.pt")
                if val_loss is not None and val_loss < best:
                    best = val_loss
                    torch.save(checkpoint, out / "best.pt")
        print(f"Epoch elapsed: {(time.time()-started)/60:.1f} min")
    if not (out / "best.pt").exists():
        torch.save({"model": model.state_dict(), "config": model.config_dict(), "step": global_step}, out / "best.pt")
    print(f"Training complete. Checkpoints: {out}")

if __name__ == "__main__":
    main()
