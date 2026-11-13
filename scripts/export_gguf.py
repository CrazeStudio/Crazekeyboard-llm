#!/usr/bin/env python3
"""Convert CrazeLM's custom PyTorch checkpoint to a llama.cpp GPT-2 GGUF.

CrazeLM's block layout matches GPT-2's sequential pre-norm decoder: learned
absolute positions, fused QKV, GELU MLP, LayerNorm, and tied token embeddings.
The tokenizer remains the original SentencePiece BPE vocabulary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sentencepiece as spm
import torch
import gguf

from crazelm.model import CrazeLM, CrazeLMConfig


def load_checkpoint(path: Path):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("model"), dict):
        raise TypeError("Expected a training checkpoint containing a 'model' state_dict.")
    config_data = artifact.get("config")
    if not isinstance(config_data, dict):
        raise ValueError("Checkpoint is missing its exact 'config' metadata.")
    cfg = CrazeLMConfig(**config_data)
    model = CrazeLM(cfg)
    model.load_state_dict(artifact["model"], strict=True)
    model.eval()
    return model, cfg, artifact


def tensor_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().to(torch.float16).numpy()


def export(checkpoint: Path, tokenizer_path: Path, output: Path):
    model, cfg, artifact = load_checkpoint(checkpoint)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    if tokenizer.get_piece_size() != cfg.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: {tokenizer.get_piece_size()} != {cfg.vocab_size}"
        )
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if cfg.d_ff != 4 * cfg.d_model:
        raise ValueError("GPT-2 GGUF mapping currently requires d_ff == 4 * d_model")
    if not (tokenizer.bos_id() >= 0 and tokenizer.eos_id() >= 0 and tokenizer.unk_id() >= 0):
        raise ValueError("SentencePiece tokenizer must define BOS, EOS, and UNK tokens")

    arch = gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.GPT2]
    writer = gguf.GGUFWriter(str(output), arch)
    writer.add_name("CrazeLM-30M")
    writer.add_description("CrazeLM 30M converted from its native PyTorch checkpoint.")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_context_length(cfg.max_seq_len)
    writer.add_embedding_length(cfg.d_model)
    writer.add_block_count(cfg.n_layers)
    writer.add_feed_forward_length(cfg.d_ff)
    writer.add_head_count(cfg.n_heads)
    writer.add_layer_norm_eps(1e-5)
    writer.add_vocab_size(cfg.vocab_size)
    writer.add_causal_attention(True)

    # Preserve SentencePiece IDs and scores; do not rebuild or reorder the vocab.
    pieces = [tokenizer.id_to_piece(i) for i in range(tokenizer.get_piece_size())]
    scores = [float(tokenizer.get_score(i)) for i in range(tokenizer.get_piece_size())]
    token_types = []
    for i in range(tokenizer.get_piece_size()):
        if i == tokenizer.unk_id():
            token_types.append(gguf.TokenType.UNKNOWN)
        elif tokenizer.is_byte(i):
            token_types.append(gguf.TokenType.BYTE)
        elif tokenizer.is_control(i):
            token_types.append(gguf.TokenType.CONTROL)
        elif tokenizer.is_unused(i):
            token_types.append(gguf.TokenType.UNUSED)
        else:
            token_types.append(gguf.TokenType.NORMAL)

    writer.add_tokenizer_model("llama")
    writer.add_token_list(pieces)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(tokenizer.bos_id())
    writer.add_eos_token_id(tokenizer.eos_id())
    writer.add_unk_token_id(tokenizer.unk_id())
    if tokenizer.pad_id() >= 0:
        writer.add_pad_token_id(tokenizer.pad_id())
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    writer.add_add_space_prefix(True)
    writer.add_remove_extra_whitespaces(True)

    state = model.state_dict()
    tensors: list[tuple[str, torch.Tensor]] = [
        ("token_embd.weight", state["token_embedding.weight"]),
        ("position_embd.weight", state["position_embedding.weight"]),
        ("output_norm.weight", state["final_ln.weight"]),
        ("output_norm.bias", state["final_ln.bias"]),
    ]

    for i in range(cfg.n_layers):
        prefix = f"blocks.{i}."
        gg = f"blk.{i}."
        tensors.extend([
            (gg + "attn_norm.weight", state[prefix + "ln1.weight"]),
            (gg + "attn_norm.bias", state[prefix + "ln1.bias"]),
            (gg + "attn_qkv.weight", state[prefix + "attn.qkv.weight"]),
            (gg + "attn_qkv.bias", state[prefix + "attn.qkv.bias"]),
            (gg + "attn_output.weight", state[prefix + "attn.proj.weight"]),
            (gg + "attn_output.bias", state[prefix + "attn.proj.bias"]),
            (gg + "ffn_norm.weight", state[prefix + "ln2.weight"]),
            (gg + "ffn_norm.bias", state[prefix + "ln2.bias"]),
            (gg + "ffn_up.weight", state[prefix + "ff.0.weight"]),
            (gg + "ffn_up.bias", state[prefix + "ff.0.bias"]),
            (gg + "ffn_down.weight", state[prefix + "ff.2.weight"]),
            (gg + "ffn_down.bias", state[prefix + "ff.2.bias"]),
        ])

    # The output projection is tied to token_embd.weight; GPT-2 GGUF loaders
    # use the token embedding as the output matrix when no separate head exists.
    for name, tensor in tensors:
        writer.add_tensor(name, tensor_array(tensor))

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
    finally:
        writer.close()

    # Structural assertions: expected tensors and metadata must be present.
    expected_tensors = 4 + cfg.n_layers * 12
    if len(tensors) != expected_tensors:
        raise AssertionError(f"Unexpected tensor count: {len(tensors)} != {expected_tensors}")
    if not output.is_file() or output.stat().st_size < 10_000_000:
        raise AssertionError("GGUF output is missing or unexpectedly small")

    print(f"GGUF export: {output}")
    print(f"Architecture: {arch}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Vocabulary: {tokenizer.get_piece_size():,}")
    print(f"GGUF tensors: {expected_tensors}")
    print(f"Checkpoint step: {artifact.get('step', 'unknown')}")
    print(f"Size: {output.stat().st_size / (1024 * 1024):.2f} MiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", default=Path("artifacts/CrazeLM-30M-F16.gguf"), type=Path)
    args = parser.parse_args()
    for path in (args.checkpoint, args.tokenizer):
        if not path.is_file():
            parser.error(f"Required file does not exist: {path}")
    export(args.checkpoint, args.tokenizer, args.output)


if __name__ == "__main__":
    main()
