"""Conformer encoder: a thin, padding-aware wrapper around ``torchaudio.models.Conformer`` (SPEC section 10).

torchaudio's ``Conformer.forward`` masks padded keys in self-attention but not in the depthwise convolution, so the
last ``kernel // 2`` real frames of a padded sequence would see padding content. This wrapper keeps torchaudio's
modules and parameters (state-dict layout ``conformer.conformer_layers.N.*``) and re-runs each layer with padded
frames zeroed right before the depthwise convolution, which makes the output of a sequence independent of how much
padding the batch adds (up to BatchNorm batch statistics in training mode). No subsampling: ``T`` is unchanged.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
from torchaudio.models import Conformer
from torchaudio.models.conformer import ConformerLayer


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Bool ``[B, max_len]``, True on padded positions."""
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def _check_layer(layer: ConformerLayer) -> None:
    """Fail loudly if the torchaudio layer structure this wrapper relies on ever changes."""
    seq = layer.conv_module.sequential
    ok = (isinstance(seq[0], nn.Conv1d) and isinstance(seq[1], nn.GLU) and isinstance(seq[2], nn.Conv1d)
          and seq[2].groups == seq[2].in_channels)
    if not ok:
        raise RuntimeError("unexpected torchaudio ConformerLayer layout (pointwise conv, GLU, depthwise conv)")


def _conv_block(layer: ConformerLayer, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Conformer convolution module on ``[T, B, D]`` with padded frames zeroed before the depthwise conv."""
    mod = layer.conv_module
    y = mod.layer_norm(x.transpose(0, 1)).transpose(1, 2)    # [B, D, T]
    y = mod.sequential[1](mod.sequential[0](y))              # pointwise conv + GLU
    y = y.masked_fill(pad.unsqueeze(1), 0.0)
    for m in mod.sequential[2:]:                             # depthwise conv, norm, SiLU, pointwise conv, dropout
        y = m(y)
    return x + y.permute(2, 0, 1)


def _layer_forward(layer: ConformerLayer, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Same computation as ``ConformerLayer.forward`` (``x``: ``[T, B, D]``) with the masked convolution."""
    x = x + 0.5 * layer.ffn1(x)
    if layer.convolution_first:
        x = _conv_block(layer, x, pad)
    y = layer.self_attn_layer_norm(x)
    y, _ = layer.self_attn(query=y, key=y, value=y, key_padding_mask=pad, need_weights=False)
    x = x + layer.self_attn_dropout(y)
    if not layer.convolution_first:
        x = _conv_block(layer, x, pad)
    x = x + 0.5 * layer.ffn2(x)
    return layer.final_layer_norm(x)


class ConformerEncoder(nn.Module):
    """``[B, T, d_model]`` + lengths -> ``([B, T, d_model], lengths)``; padded output frames are zero."""

    def __init__(self, d_model: int, num_layers: int, num_heads: int, ffn_dim: int, conv_kernel: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by encoder heads {num_heads}")
        self.conformer = Conformer(input_dim=d_model, num_heads=num_heads, ffn_dim=ffn_dim, num_layers=num_layers,
                                   depthwise_conv_kernel_size=conv_kernel, dropout=dropout)
        for layer in self.conformer.conformer_layers:
            _check_layer(layer)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = lengths_to_padding_mask(lengths, x.size(1))
        h = x.transpose(0, 1)
        for layer in self.conformer.conformer_layers:
            h = _layer_forward(layer, h, pad)
        return h.transpose(0, 1).masked_fill(pad.unsqueeze(-1), 0.0), lengths


_LEGACY_LSTM_KEY = re.compile(r"^(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)(_reverse)?$")


class BLSTMEncoder(nn.Module):
    """Bidirectional LSTM stack: ``[B, T, d_model]`` + lengths -> ``([B, T, d_model], lengths)``; padded frames zero.

    Robust choice for small data: on ~10 h of Korean audio a plain BiLSTM-CTC leaves the CTC blank plateau after
    ~1.5-2k steps. The LSTMs run in float32 (cuDNN LSTM under bf16 autocast is not reliable); packing keeps padding
    out of the recurrence in both directions.

    Built as ``num_layers`` single-layer LSTMs with ``nn.Dropout`` in between, which computes exactly what
    ``nn.LSTM(num_layers=N, dropout=p)`` computes: on this Windows/torch 2.11 setup a multi-layer cuDNN LSTM with
    inter-layer dropout, once run in training mode, makes the process crash at exit (0xC0000409), so training scripts
    reported failure after finishing. Checkpoints saved with the old single ``rnn`` module load unchanged (keys are
    remapped in ``_load_from_state_dict``)."""

    def __init__(self, d_model: int, num_layers: int = 4, hidden: int = 320, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.LSTM(d_model if i == 0 else 2 * hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
            for i in range(num_layers))
        self.dropout = nn.Dropout(dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(2 * hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        for key in [k for k in list(state_dict) if k.startswith(prefix + "rnn.")]:
            m = _LEGACY_LSTM_KEY.match(key[len(prefix) + 4:])
            if m:  # rnn.weight_ih_l2_reverse -> layers.2.weight_ih_l0_reverse
                state_dict[f"{prefix}layers.{m.group(2)}.{m.group(1)}_l0{m.group(3) or ''}"] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = lengths_to_padding_mask(lengths, x.size(1))
        cpu_lengths = lengths.cpu()
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = x.float()
            for i, rnn in enumerate(self.layers):
                if i > 0:
                    h = self.dropout(h)
                packed = nn.utils.rnn.pack_padded_sequence(h, cpu_lengths, batch_first=True, enforce_sorted=False)
                h, _ = rnn(packed)
                h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        out = self.norm(self.proj(h))
        return out.masked_fill(pad.unsqueeze(-1), 0.0), lengths


ENCODER_TYPES = ("conformer", "blstm")

__all__ = ["BLSTMEncoder", "ConformerEncoder", "ENCODER_TYPES", "lengths_to_padding_mask"]
