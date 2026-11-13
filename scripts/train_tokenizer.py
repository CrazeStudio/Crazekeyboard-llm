import argparse
from pathlib import Path
import sentencepiece as spm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vocab_size", type=int, default=8192)
    args = p.parse_args()
    root = Path(args.input_dir)
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found under {root}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = out / "tokenizer_corpus.txt"
    with corpus.open("w", encoding="utf-8") as dst:
        for path in files:
            with path.open("r", encoding="utf-8", errors="ignore") as src:
                for line in src:
                    line = " ".join(line.split())
                    if line:
                        dst.write(line + "\n")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(out / "tokenizer"),
        vocab_size=args.vocab_size, model_type="bpe",
        character_coverage=1.0, byte_fallback=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        hard_vocab_limit=False, shuffle_input_sentence=True
    )
    corpus.unlink(missing_ok=True)
    print(f"Tokenizer saved to {out / 'tokenizer.model'}")

if __name__ == "__main__":
    main()
