# CrazeLM

Compact, English-only causal Transformer prototype for CrazeKeyboard next-token suggestions. Training is intended for Kaggle GPU; Android inference/export is a later stage.

## Status

**This repository contains training code, not trained weights.** No accuracy, keyboard quality, or Android performance is claimed until a training run and evaluation are completed.

## Model

- Decoder-only causal Transformer
- 4 layers, width 384, 6 attention heads, FFN width 1536
- SentencePiece BPE tokenizer, target vocabulary 8,192 with byte fallback
- Tied token embedding/output projection
- Context length: 64 tokens
- Approximately 10.27M parameters (the exact count is printed by `scripts/inspect_model.py`)
- Training language: English only

## Kaggle setup

Create a Kaggle notebook, enable a GPU, and run each cell in order. Internet must be enabled for the initial dataset download.

### 1. Clone and install

```python
!git clone https://github.com/CrazeStudio/Crazekeyboard-llm.git
%cd Crazekeyboard-llm
!pip install -q -r requirements.txt
```

The requirements deliberately do not install Hugging Face `datasets` or upgrade `pyarrow`/Pillow. Dataset preparation reads the small Parquet split files with Kaggle's preinstalled pandas/pyarrow and does not use the retired S3 archive URL. If you already cloned the repository in this session, use `%cd /kaggle/working/Crazekeyboard-llm` instead of cloning again.

### 2. Check GPU and imports

```python
import torch, sentencepiece, numpy
print("PyTorch:", torch.__version__)
print("SentencePiece:", sentencepiece.__version__)
print("NumPy:", numpy.__version__)
print("CUDA available:", torch.cuda.is_available())
!nvidia-smi
```

### 3. Prepare English-only data

```python
!python scripts/prepare_dataset.py \
  --output_dir /kaggle/working/crazelm_data
```

This downloads the train and validation Parquet files from the Hugging Face-hosted WikiText-2 repository using redirect-aware downloads, validates their Parquet signatures, and creates:
- `crazelm_data/train/english.txt`
- `crazelm_data/validation/english.txt`
- `crazelm_data/DATASET_MANIFEST.json`

WikiText-2 is a small Wikipedia-derived benchmark corpus. It is suitable for checking that the pipeline runs, but it is **not sufficient by itself for a high-quality keyboard suggestion model**. Before release or commercial use, verify the dataset's current license and attribution obligations. For stronger keyboard predictions, add a larger, appropriately licensed English corpus representative of natural typing. You can prepare your own .txt corpus with `--input_dir /path/to/english_text_folder`.

### 4. Train the English tokenizer

```python
!python scripts/train_tokenizer.py \
  --input_dir /kaggle/working/crazelm_data/train \
  --output_dir /kaggle/working/tokenizer \
  --vocab_size 8192
```

### 5. Check model parameter count

```python
!python scripts/inspect_model.py
```

### 6. Run a short smoke training first

Use a short run to catch environment or data problems before committing to a longer session:

```python
!python train.py \
  --train_dir /kaggle/working/crazelm_data/train \
  --val_dir /kaggle/working/crazelm_data/validation \
  --tokenizer /kaggle/working/tokenizer/tokenizer.model \
  --output_dir /kaggle/working/crazelm_smoke \
  --epochs 1 \
  --batch_size 4 \
  --seq_len 64 \
  --steps_per_epoch 10 \
  --eval_steps 2 \
  --save_every 10
```

If that finishes and creates `best.pt`, run the longer training job:

```python
!python train.py \
  --train_dir /kaggle/working/crazelm_data/train \
  --val_dir /kaggle/working/crazelm_data/validation \
  --tokenizer /kaggle/working/tokenizer/tokenizer.model \
  --output_dir /kaggle/working/crazelm \
  --epochs 3 \
  --batch_size 16 \
  --seq_len 64 \
  --steps_per_epoch 1000 \
  --eval_steps 50 \
  --save_every 250
```

If CUDA runs out of memory, reduce `--batch_size` to 8 or 4. The WikiText-2 corpus is small, so a longer run repeatedly samples from it; more steps do not replace the need for a larger and more suitable dataset.

### 7. Inspect outputs

```python
!ls -lh /kaggle/working/crazelm
!cat /kaggle/working/crazelm/metrics.jsonl
```

### 8. Package the checkpoint

```python
!cd /kaggle/working && zip -r CrazeLM-English-checkpoint.zip crazelm tokenizer crazelm_data/DATASET_MANIFEST.json
from IPython.display import FileLink
FileLink("/kaggle/working/CrazeLM-English-checkpoint.zip")
```

Download the zip from the notebook output. Kaggle working files are temporary unless downloaded or saved as a Kaggle output/dataset.

## Training implementation notes

- Each text line is encoded independently and terminated with SentencePiece EOS so separate examples do not silently merge.
- Token IDs are stored as uint16; the tokenizer script enforces a vocabulary limit compatible with that storage.
- Training and validation streams are checked for minimum length.
- Batch sampling includes the final valid starting position.
- Validation loss is evaluated during checkpointing; both `last.pt` and best-validation `best.pt` are saved with optimizer/scaler state and model configuration.
- No Hugging Face `datasets` package is needed for the default data preparation path.

## What this model is and is not

CrazeLM is currently a small next-token language model. It is not yet a complete keyboard engine: autocorrect, typo correction, candidate ranking, personalization, phrase completion, quantization, Android runtime integration, and on-device latency/RAM profiling remain separate work. A 10M-parameter count does not guarantee useful predictions; evaluate perplexity and real keyboard top-k suggestions on held-out English text before integrating it into CrazeKeyboard.
