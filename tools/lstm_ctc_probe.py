"""Learnability probe: a plain BiLSTM-CTC on the audio features of the real data (read-only; nothing is saved).

If this simple, plateau-robust model's CTC loss drops within ~1000 steps while the Conformer's does not, the problem is
the Conformer setup (input scaling / positions / optimisation), not the data or the targets.
Usage: $py -m tools.lstm_ctc_probe [--steps 1500] [--upsample 2] [--lr 1e-3]
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from avsr.dataset import AVSRDataset, DurationBatchSampler, assign_split, collate_fn, load_manifests
from avsr.models.avsr_model import ctc_collapse
from avsr.text import Tokenizer
from avsr.utils import cer, load_config


class Probe(nn.Module):
    def __init__(self, vocab: int, in_dim: int = 320, hidden: int = 256, layers: int = 3, up: int = 2) -> None:
        super().__init__()
        self.up = up
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * hidden, vocab * up)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        b, t, _ = h.shape
        return self.out(h).reshape(b, t * self.up, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--upsample", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    cfg = load_config("configs/base.yaml", ["audio.noise_prob=0", "audio.specaug.freq_mask=0",
                                            "audio.specaug.time_mask=0"])
    tok = Tokenizer()
    sp = assign_split(load_manifests(cfg.work_dir), cfg)
    train = [r for r in sp["train"] if r["angle"] == "A"]
    val = [r for r in sp["val"] if r["angle"] == "A"][:200]
    ds = AVSRDataset(train, cfg, train=False, tokenizer=tok)
    sampler = DurationBatchSampler(train, max_frames=3200, shuffle=True, seed=0)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=8, persistent_workers=True)
    dev = torch.device("cuda")
    model = Probe(tok.vocab_size, up=args.upsample).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    step, t0 = 0, time.time()
    model.train()
    while step < args.steps:
        for batch in dl:
            x, lengths = batch["audio"].to(dev), batch["lengths"].to(dev)
            toks, tl = batch["tokens"].to(dev), batch["token_lengths"].to(dev)
            logits = model(x, lengths)
            lp = logits.float().log_softmax(-1).transpose(0, 1)
            loss = F.ctc_loss(lp, toks, lengths * args.upsample, tl, blank=0, reduction="sum",
                              zero_infinity=True) / x.size(0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step:5d} ctc {loss.item():7.1f} gnorm {float(gn):7.1f} ({time.time() - t0:.0f}s)", flush=True)
            if step >= args.steps:
                break
    model.eval()
    vds = AVSRDataset(val, cfg, train=False, tokenizer=tok)
    refs, hyps = [], []
    with torch.no_grad():
        for i in range(0, len(vds), 16):
            b = collate_fn([vds[j] for j in range(i, min(i + 16, len(vds)))])
            logits = model(b["audio"].to(dev), b["lengths"].to(dev))
            mask = (torch.arange(logits.size(1), device=dev)[None] < (b["lengths"].to(dev) * args.upsample)[:, None])
            for h, t in zip(ctc_collapse(logits.argmax(-1), mask, (0, 1, 2, 3)), b["texts"]):
                hyps.append(tok.decode(h))
                refs.append(tok.decode(tok.encode(t)))
    print(f"val CER (unseen speaker, {len(refs)} utts): {100 * cer(refs, hyps):.1f}%")
    for r, h in list(zip(refs, hyps))[:3]:
        print("  REF", r, "\n  HYP", h)


if __name__ == "__main__":
    main()
