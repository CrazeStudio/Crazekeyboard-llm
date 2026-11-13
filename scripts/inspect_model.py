import argparse
from crazelm.model import CrazeLM, CrazeLMConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--seq_len", type=int, default=64)
    args = p.parse_args()
    model = CrazeLM(CrazeLMConfig(vocab_size=args.vocab_size, max_seq_len=args.seq_len))
    count = model.parameter_count()
    print(f"Trainable parameters: {count:,}")
    print(f"FP32 weights: {count * 4 / 1_000_000:.2f} MB")
    print(f"FP16 weights: {count * 2 / 1_000_000:.2f} MB")
    print(f"INT8 weights (ideal): {count / 1_000_000:.2f} MB")

if __name__ == "__main__":
    main()
