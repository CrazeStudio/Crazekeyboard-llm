# CrazeLM

CrazeLM is a compact English causal Transformer intended to provide next-token candidates for CrazeKeyboard. This repository contains the model, corpus preparation, tokenizer training, Kaggle training, checkpoint evaluation, and a prediction CLI.

> **Training status:** Source code is maintained here; trained weights are produced in the user's Kaggle session and are not committed to Git. A successful smoke test only verifies the pipeline. Do not treat a checkpoint as keyboard-ready until held-out evaluation and real typing tests pass.

## Model

- Decoder-only causal Transformer
- 4 layers, width 384, 6 attention heads, FFN width 1536
- SentencePiece BPE tokenizer (target vocabulary 8,192, byte fallback)
- Tied token embedding/output projection
- Context length 64 tokens
- Approximately 10.27M parameters
- English-only training pipeline

This is a small language model, not a general-purpose ChatGPT replacement.

## 1. Kaggle setup

Create a Kaggle notebook, enable a GPU (T4/P100 if available), and enable Internet.

```python
!git clone https://github.com/CrazeStudio/Crazekeyboard-llm.git
%cd Crazekeyboard-llm
!pip install -q -r requirements.txt
```

For an existing clone, use `%cd /kaggle/working/Crazekeyboard-llm` and `!git pull origin main`.

Check the runtime:

```python
import torch, sentencepiece, numpy, pandas, pyarrow
print("PyTorch:", torch.__version__)
print("SentencePiece:", sentencepiece.__version__)
print("CUDA:", torch.cuda.is_available())
print("PyArrow:", pyarrow.__version__)
!nvidia-smi
```

## 2. Prepare a conversational English corpus

The default pipeline downloads the train, validation, and test Parquet splits from the Hugging Face mirror `anezatra/dailydialog`. The mirror describes DailyDialog as human-written everyday multi-turn conversations and labels it Apache-2.0. Review the dataset card and upstream rights before redistribution.

```python
!python scripts/prepare_dataset.py \
  --output_dir /kaggle/working/crazelm_data \
  --min_train_lines 50000
```

The script discovers the available Parquet shard names through the Hugging Face dataset API, validates downloaded files, extracts utterances, removes empty/obvious markup and exact duplicate lines, and preserves the dataset's train/validation/test split. It refuses to continue if fewer than 50,000 unique training utterances are extracted. It writes:

- `crazelm_data/train/english.txt`
- `crazelm_data/validation/english.txt`
- `crazelm_data/test/english.txt`
- `crazelm_data/DATASET_MANIFEST.json`

If the script reports fewer than 50,000 lines, inspect the schema/error rather than fabricating or duplicating examples. Dataset size alone does not guarantee good keyboard predictions.

## 3. Train a fresh tokenizer

Train the tokenizer on training text only; validation and test data must not leak into tokenizer training.

```python
!python scripts/train_tokenizer.py \
  --input_dir /kaggle/working/crazelm_data/train \
  --output_dir /kaggle/working/tokenizer \
  --vocab_size 8192
```

Verify the artifacts:

```python
!ls -lh /kaggle/working/tokenizer/
!python scripts/inspect_model.py
```

## 4. Smoke test training

```python
!python train.py \
  --train_dir /kaggle/working/crazelm_data/train \
  --val_dir /kaggle/working/crazelm_data/validation \
  --tokenizer /kaggle/working/tokenizer/tokenizer.model \
  --output_dir /kaggle/working/crazelm_smoke \
  --epochs 1 --batch_size 4 --seq_len 64 \
  --steps_per_epoch 10 --eval_steps 2 --save_every 10
```

Confirm `best.pt` exists before moving on. This is only a pipeline check, not a meaningful training run.

## 5. Train CrazeLM

```python
!python train.py \
  --train_dir /kaggle/working/crazelm_data/train \
  --val_dir /kaggle/working/crazelm_data/validation \
  --tokenizer /kaggle/working/tokenizer/tokenizer.model \
  --output_dir /kaggle/working/crazelm \
  --epochs 5 --batch_size 16 --seq_len 64 \
  --steps_per_epoch 1000 --eval_steps 50 --save_every 250
```

If CUDA runs out of memory, reduce `--batch_size` to 8 or 4. Monitor validation loss; stop and investigate if it rises consistently while training loss falls. Kaggle sessions can time out, so download/save checkpoints and dataset outputs between sessions.

## 6. Evaluate on held-out test data

```python
!python scripts/evaluate.py \
  --checkpoint /kaggle/working/crazelm/best.pt \
  --tokenizer /kaggle/working/tokenizer/tokenizer.model \
  --test_dir /kaggle/working/crazelm_data/test
```

This reports test loss, perplexity, and top-1/top-5 token accuracy. These are language-model metrics, not keyboard word-suggestion accuracy.

## 7. Test next-token predictions

```python
!python scripts/predict.py \
  --checkpoint /kaggle/working/crazelm/best.pt \
  --tokenizer /kaggle/working/tokenizer/tokenizer.model \
  --text "I am going to" \
  --top_k 8
```

Try multiple prompts and inspect whether candidates change appropriately with context. SentencePiece can return subword fragments; this CLI exposes raw next-token pieces, not a final user-facing keyboard candidate ranker.

## 8. Package the trained artifacts

```python
!cd /kaggle/working && zip -r CrazeLM-English.zip \
  crazelm/best.pt crazelm/last.pt crazelm/model_config.json \
  tokenizer/tokenizer.model tokenizer/tokenizer.vocab \
  crazelm_data/DATASET_MANIFEST.json

from IPython.display import FileLink, display
display(FileLink("/kaggle/working/CrazeLM-English.zip"))
```

Keep the dataset files and manifest with the model so its provenance is traceable. Kaggle working files are temporary until downloaded or saved as notebook output.

## Current scope and remaining engineering

Implemented in this repository:
- English conversational corpus download and preparation
- Train/validation/test separation and basic cleaning/deduplication
- SentencePiece tokenizer training
- Causal Transformer training and checkpoint saving
- Held-out language-model evaluation
- CLI for inspecting next-token candidates

Still required before production keyboard integration:
- Larger and more diverse appropriately licensed English typing corpus
- Word-level candidate assembly and ranking (subword tokens are not complete words)
- Autocorrect and typo correction
- Keyboard-specific top-k evaluation on a separate real typing benchmark
- Personalization with privacy-preserving local adaptation
- Quantization/export and Android runtime integration
- On-device latency, memory, battery, and crash testing
- Filtering/guardrails for unwanted suggestions

No trained weights, measured keyboard accuracy, or Android integration are included by default. The training run must be completed and evaluated in Kaggle.
