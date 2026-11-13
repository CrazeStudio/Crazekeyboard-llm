#!/usr/bin/env python3
import argparse
from pathlib import Path
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser(description="Train the English-only CrazeLM SentencePiece tokenizer.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab_size", type=int, default=8192)
    args = parser.parse_args()
    if args.vocab_size < 256 or args.vocab_size > 65535:
        parser.error("--vocab_size must be between 256 and 65535 (token storage uses uint16)")

    root = Path(args.input_dir)
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found under {root}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = out / "tokenizer_corpus.txt"
    line_count = 0
    with corpus.open("w", encoding="utf-8") as dst:
        for path in files:
            with path.open("r", encoding="utf-8", errors="ignore") as src:
                for line in src:
                    line = " ".join(line.split())
                    if line:
                        dst.write(line + "\n")
                        line_count += 1
    if line_count < 20:
        corpus.unlink(missing_ok=True)
        raise SystemExit(f"Only {line_count} non-empty lines; prepare a larger English corpus first.")

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
            hard_vocab_limit=False,
            shuffle_input_sentence=True,
            input_sentence_size=0,
            num_threads= max(1, min(8, __import__("os").cpu_count() or 1)),
        )
    finally:
        corpus.unlink(missing_ok=True)
    print(f"Tokenizer saved to {out / 'tokenizer.model'}; lines={line_count:,}; vocab={spm.SentencePieceProcessor(model_file=str(out / 'tokenizer.model')).get_piece_size():,}")


if __name__ == "__main__":
    main()
