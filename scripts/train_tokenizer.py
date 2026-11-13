#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser(
        description="Train the final 100k-piece English CrazeLM SentencePiece tokenizer."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab_size", type=int, default=100_000)
    args = parser.parse_args()
    if args.vocab_size != 100_000:
        parser.error("The final CrazeLM tokenizer target is exactly 100,000 pieces.")
    root = Path(args.input_dir)
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found under {root}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = out / "tokenizer_corpus.txt"
    line_count = 0
    char_count = 0
    with corpus.open("w", encoding="utf-8") as dst:
        for path in files:
            with path.open("r", encoding="utf-8", errors="ignore") as src:
                for line in src:
                    line = " ".join(line.split())
                    if line:
                        dst.write(line + "\n")
                        line_count += 1
                        char_count += len(line)
    # 100k pieces cannot be learned from a tiny conversational sample.
    if line_count < 100_000 or char_count < 100_000_000:
        corpus.unlink(missing_ok=True)
        raise SystemExit(
            f"Corpus is too small for the 100k tokenizer target: "
            f"{line_count:,} lines and {char_count:,} characters. "
            "Add a large, diverse, properly licensed English corpus; do not duplicate lines."
        )

    try:
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(out / "tokenizer"),
            vocab_size=args.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            byte_fallback=True,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            hard_vocab_limit=True,
            shuffle_input_sentence=True,
            input_sentence_size=0,
            num_threads=max(1, min(8, os.cpu_count() or 1)),
        )
    finally:
        corpus.unlink(missing_ok=True)

    tokenizer = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
    actual_size = tokenizer.get_piece_size()
    if actual_size != args.vocab_size:
        raise SystemExit(
            f"Tokenizer has {actual_size:,} pieces; expected exactly {args.vocab_size:,}."
        )
    print(
        f"Tokenizer saved to {out / 'tokenizer.model'}; "
        f"lines={line_count:,}; characters={char_count:,}; vocab={actual_size:,}"
    )


if __name__ == "__main__":
    main()
