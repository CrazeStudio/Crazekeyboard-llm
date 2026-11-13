# CrazeLM

Experimental compact causal Transformer for CrazeKeyboard next-token prediction. Training is intended for Kaggle GPU; inference/export is intended for Android later.

## Status

**Training scaffold only. No trained weights are included yet.** Do not integrate or describe this as a trained model until training and evaluation complete.

## Initial architecture

- Decoder-only causal Transformer
- 4 layers, width 384, 6 attention heads, FFN width 1536
- 8,192-token SentencePiece tokenizer with byte fallback
- Tied token embedding/output projection
- Maximum context: 64 tokens
- Approximately 10.27M parameters; exact count is printed by the model code
- English, Bengali and Hindi are tokenizer/training targets; actual quality depends on balanced, licensed data

## Kaggle workflow

1. Create a Kaggle notebook and enable a GPU.
2. Add a dataset containing UTF-8 text files that you have permission to use.
3. Clone/upload this repository and install dependencies with `pip install -r requirements.txt`.
4. Train tokenizer, then model:

```bash
python scripts/train_tokenizer.py --input_dir /kaggle/input/YOUR_DATASET --output_dir artifacts/tokenizer --vocab_size 8192
python train.py --train_dir /kaggle/input/YOUR_DATASET --tokenizer artifacts/tokenizer/tokenizer.model --output_dir /kaggle/working/crazelm --epochs 3 --batch_size 32
```

Start with a small subset to validate the pipeline. Adjust batch size to GPU memory. Checkpoints are saved periodically because Kaggle sessions can stop.

## Data policy

Use clean, deduplicated datasets with licenses that permit model training. Avoid private messages, passwords, personal data, and unlicensed scraped books. Keep validation data separate from training data.

## Planned outputs

- tokenizer.model and tokenizer.vocab
- best.pt and periodic checkpoints
- metrics.jsonl
- model_config.json

## Limitations

This first version is a next-token model, not a complete autocorrect engine. Quantization, Android integration, latency/RAM profiling, and language-specific evaluation are later stages. FP32 weights alone for 10.27M parameters are about 41 MB; FP16 about 20.5 MB; INT8 about 10.3 MB before overhead.