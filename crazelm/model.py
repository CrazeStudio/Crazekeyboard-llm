from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CrazeLMConfig:
    # Final CrazeLM target: 100k tokenizer vocabulary, approximately 30M params.
    vocab_size: int = 100_000
    max_seq_len: int = 128
    d_model: int = 256
    n_layers: int = 5
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        batch, seq, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, seq, self.n_heads, self.head_dim)
        q, k, v = [t.view(shape).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq, width)
        return self.dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class CrazeLM(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or CrazeLMConfig()
        cfg = self.cfg
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying reduces parameters and keeps input/output token spaces aligned.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        _, seq = input_ids.shape
        if seq > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        x = self.drop(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_ln(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self):
        return asdict(self.cfg)
