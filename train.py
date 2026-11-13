#!/usr/bin/env python3
import argparse
import json
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
    eos_id = tokenizer.eos_id()
    with open(output, "wb") as dst:
        for path in tqdm(paths, desc=f"Tokenizing {folder}"):
            with path.open("r", encoding="utf-8", errors="ignore") as src:
                for line in src:
                    text = " ".join(line.split())
                    if not text:
                        continue
                    ids = tokenizer.encode(text, out_type=int)
                    if ids:
                        ids.append(eos_id)  # prevent examples from running together
                        np.asarray(ids, dtype=np.uint16).tofile(dst)
                        total += len(ids)
    if total < 1000:
        raise RuntimeError(f"Only {total} tokens found in {folder}; provide a larger corpus.")
    return total


def sample_batch(data, batch_size, seq_len, device):
    max_start = len(data) - seq_len - 1
    if max_start < 0:
        raise RuntimeError(f"Token stream ({len(data)}) is shorter than seq_len+1 ({seq_len + 1}).")
    starts = torch.randint(0, max_start + 1, (batch_size,)).tolist()
    x_np = np.stack([np.asarray(data[i:i + seq_len], dtype=np.int64) for i in starts])
    y_np = np.stack([np.asarray(data[i + 1:i + seq_len + 1], dtype=np.int64) for i in starts])
    return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)


@torch.no_grad()
def evaluate(model, data, batch_size, seq_len, steps, device):
    if data is None:
        return None
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = sample_batch(data, batch_size, seq_len, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, optimizer, scaler, step, epoch, val_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": model.config_dict(),
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train the English-only CrazeLM next-token model.")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", default="artifacts/crazelm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.seq_len, args.steps_per_epoch, args.eval_steps, args.save_every) <= 0:
        parser.error("epochs, batch_size, seq_len, steps_per_epoch, eval_steps and save_every must be positive")
    if args.seq_len > 64:
        parser.error("This model configuration supports a maximum seq_len of 64 unless model config is changed.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        raise SystemExit(f"Tokenizer model not found: {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    vocab_size = tokenizer.get_piece_size()
    if vocab_size > 65535:
        raise SystemExit("Tokenizer vocabulary exceeds uint16 token storage capacity.")

    train_bin = out / "train_tokens.bin"
    train_count = encode_dir(args.train_dir, tokenizer, train_bin)
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_bin = out / "val_tokens.bin"
    val_count = encode_dir(args.val_dir, tokenizer, val_bin)
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    cfg = CrazeLMConfig(vocab_size=vocab_size, max_seq_len=args.seq_len)
    model = CrazeLM(cfg).to(device)
    count = model.parameter_count()
    print(f"Device: {device}; parameters: {count:,}; train tokens: {train_count:,}; validation tokens: {val_count:,}")
    print(f"FP32 weights: {count * 4 / 1e6:.2f} MB")
    with (out / "model_config.json").open("w", encoding="utf-8") as stream:
        json.dump({**model.config_dict(), "parameter_count": count, "language": "English"}, stream, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    best = float("inf")
    metrics = out / "metrics.jsonl"
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        started = time.time()
        for _ in tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y = sample_batch(train_data, args.batch_size, args.seq_len, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach().item()
            running_count += 1
            global_step += 1
            if global_step % 50 == 0:
                print(f"step={global_step} train_loss={running_loss / running_count:.4f}")
                running_loss = 0.0
                running_count = 0

            if global_step % args.save_every == 0 or global_step == args.epochs * args.steps_per_epoch:
                val_loss = evaluate(model, val_data, args.batch_size, args.seq_len, args.eval_steps, device)
                record = {"step": global_step, "epoch": epoch + 1, "train_loss_window": running_loss / running_count if running_count else None, "val_loss": val_loss}
                with metrics.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                save_checkpoint(out / "last.pt", model, optimizer, scaler, global_step, epoch + 1, val_loss)
                if val_loss < best:
                    best = val_loss
                    save_checkpoint(out / "best.pt", model, optimizer, scaler, global_step, epoch + 1, val_loss)
                print(f"step={global_step} validation_loss={val_loss:.4f} best={best:.4f}")

        print(f"Epoch elapsed: {(time.time() - started) / 60:.1f} min")

    print(f"Training finished. Best validation loss: {best:.4f}. Checkpoints: {out}")


if __name__ == "__main__":
    main()
