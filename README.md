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
- English, Bengali and Hindi are tokenizer/training targets; actual quality depends on balanced training data

## Direct Kaggle setup

Create a Kaggle notebook, enable a GPU, then run these cells in order.

### 1. Clone the repository and install dependencies

```python
!git clone https://github.com/CrazeStudio/Crazekeyboard-llm.git
%cd Crazekeyboard-llm
!pip install -q -r requirements.txt
```

### 2. Download and prepare training data

This uses AI4Bharat Samanantar, which provides English-Bengali and English-Hindi parallel text. The script streams a limited number of pairs and writes separate training and validation directories; it does not download the full multi-gigabyte corpus.

**License warning:** Samanantar is licensed CC BY-NC 4.0 (Attribution-NonCommercial). This is not suitable for commercial model training/distribution without separate permission. Review the license and attribution requirements before proceeding. The explicit flag below acknowledges this restriction; it does not grant additional rights.

```python
!python scripts/prepare_dataset.py \
  --output_dir /kaggle/working/crazelm_data \
  --languages bn hi \
  --max_pairs_per_language 100000 \
  --accept_cc_by_nc
```

To prepare only Bengali or only Hindi, change `--languages bn hi` to `--languages bn` or `--languages hi`.

### 3. Train the tokenizer

```python
!python scripts/train_tokenizer.py \
  --input_dir /kaggle/working/crazelm_data/train \
  --output_dir /kaggle/working/tokenizer \
  --vocab_size 8192
```

### 4. Inspect the model size

```python
!python scripts/inspect_model.py
```

### 5. Train CrazeLM

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
  --eval_steps 50
```

If the GPU runs out of memory, lower `--batch_size` to 8 or 4. The first run should be treated as a pipeline test, not a quality-trained release. Inspect validation loss and evaluate English, Bengali and Hindi predictions separately.

### 6. Download the checkpoint

```python
!cd /kaggle/working && zip -r CrazeLM-checkpoint.zip crazelm tokenizer crazelm_data/DATASET_MANIFEST.json
from IPython.display import FileLink
FileLink("/kaggle/working/CrazeLM-checkpoint.zip")
```

Kaggle output files are temporary unless you save/download them or commit the notebook output as a Kaggle dataset.

## Dataset details and attribution

- Dataset: [AI4Bharat Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar)
- License: CC BY-NC 4.0
- Paper: Ramesh et al., *Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages*, TACL 2022.
- DOI: https://doi.org/10.1162/tacl_a_00452

The dataset is parallel translation text, not a representative sample of everyday keyboard typing. It is useful for an initial multilingual language-model experiment, but additional appropriately licensed conversational, editorial and general-domain text is needed for strong keyboard suggestions.

## Planned outputs

- tokenizer.model and tokenizer.vocab
- best.pt and periodic checkpoints
- metrics.jsonl
- model_config.json
- DATASET_MANIFEST.json

## Limitations

This first version is a next-token model, not a complete autocorrect engine. Quantization, Android integration, latency/RAM profiling, and language-specific evaluation are later stages. FP32 weights alone for 10.27M parameters are about 41 MB; FP16 about 20.5 MB; INT8 about 10.3 MB before overhead.
