"""Transformer attention decoder (pre-LN) with teacher forcing and greedy decoding (SPEC section 10)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .fusion import SinusoidalPositionalEncoding


class AttentionDecoder(nn.Module):
    """Token embedding + sinusoidal positions -> ``nn.TransformerDecoder`` (causal self-attention, cross-attention
    with an encoder key padding mask) -> final LayerNorm -> Linear(vocab)."""

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, ffn_dim: int,
                 dropout: float, *, blank_id: int, pad_id: int, sos_id: int, eos_id: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by decoder heads {num_heads}")
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embed.weight, std=d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[pad_id].zero_()
        self.scale = math.sqrt(d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(d_model, num_heads, ffn_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, vocab_size)
        self.blank_id, self.pad_id, self.sos_id, self.eos_id = blank_id, pad_id, sos_id, eos_id

    def teacher_forcing(self, tokens: torch.Tensor, token_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``tokens [B, L]`` (pad-padded) -> decoder input ``[sos] + tokens`` and target ``tokens + [eos]``,
        both ``[B, L + 1]`` and padded with ``pad_id``."""
        b, l = tokens.shape
        pos = torch.arange(l + 1, device=tokens.device).unsqueeze(0)
        lens = token_lengths.to(tokens.device).unsqueeze(1)
        body = torch.cat([tokens, tokens.new_full((b, 1), self.pad_id)], dim=1)
        body = body.masked_fill(pos >= lens, self.pad_id)
        target = body.masked_fill(pos == lens, self.eos_id)
        inp = torch.cat([tokens.new_full((b, 1), self.sos_id), body[:, :-1]], dim=1)
        return inp, target

    def forward(self, tokens_in: torch.Tensor, memory: torch.Tensor, memory_pad: torch.Tensor) -> torch.Tensor:
        """``tokens_in [B, U]``, ``memory [B, T, D]``, ``memory_pad [B, T]`` (True = padded) -> logits ``[B, U, V]``.
        Causal masking alone keeps real positions from seeing trailing pads, so no target padding mask is needed."""
        u = tokens_in.size(1)
        causal = torch.ones(u, u, dtype=torch.bool, device=tokens_in.device).triu(1)
        x = self.dropout(self.pos(self.embed(tokens_in) * self.scale))
        h = self.layers(x, memory, tgt_mask=causal, memory_key_padding_mask=memory_pad, tgt_is_causal=True)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, memory: torch.Tensor, memory_pad: torch.Tensor, max_len: int) -> list[list[int]]:
        """Batched autoregressive argmax decoding until every hypothesis emitted ``eos`` or ``max_len`` tokens.
        Returns token ids without sos/eos/pad/blank."""
        b = memory.size(0)
        ys = torch.full((b, 1), self.sos_id, dtype=torch.long, device=memory.device)
        done = torch.zeros(b, dtype=torch.bool, device=memory.device)
        banned = [self.blank_id, self.pad_id, self.sos_id]
        for _ in range(max(int(max_len), 0)):
            logits = self.forward(ys, memory, memory_pad)[:, -1].float()
            logits[:, banned] = float("-inf")
            nxt = logits.argmax(dim=-1).masked_fill(done, self.pad_id)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt == self.eos_id
            if bool(done.all()):
                break
        hyps: list[list[int]] = []
        for row in ys[:, 1:].tolist():
            ids: list[int] = []
            for i in row:
                if i == self.eos_id:
                    break
                if i != self.pad_id:
                    ids.append(i)
            hyps.append(ids)
        return hyps


__all__ = ["AttentionDecoder"]
