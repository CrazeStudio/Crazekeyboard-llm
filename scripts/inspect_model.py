#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crazelm.model import CrazeLM, CrazeLMConfig


def main():
    p = argparse.ArgumentParser(description="Inspect the final CrazeLM 30M architecture.")
    p.add_argument("--vocab_size", type=int, default=100_000)
    p.add_argument("--seq_len", type=int, default=128)
    args = p.parse_args()
    model = CrazeLM(CrazeLMConfig(vocab_size=args.vocab_size, max_seq_len=args.seq_len))
    count = model.parameter_count()
    print(f"Trainable parameters: {count:,}")
    print(f"FP32 weights: {count * 4 / 1_000_000:.2f} MB")
    print(f"FP16 weights: {count * 2 / 1_000_000:.2f} MB")
    print(f"INT8 weights (ideal): {count / 1_000_000:.2f} MB")
    print(f"Architecture: {model.cfg.n_layers} layers, width {model.cfg.d_model}, "
          f"{model.cfg.n_heads} heads, FFN {model.cfg.d_ff}, context {model.cfg.max_seq_len}")


if __name__ == "__main__":
    main()
