"""Can the model memorise a handful of REAL utterances? Separates a code bug from an optimisation-recipe problem.

Usage: $py tools/overfit_check.py [--n 32] [--steps 400] [--mode audio|av|video|train] [--lr 1e-3] [--set k=v ...]
Prints the CTC loss curve and the greedy-CTC CER on the memorised utterances (should approach 0 if learning works).
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from avsr.dataset import AVSRDataset, assign_split, collate_fn, load_manifests
from avsr.models import build_model
from avsr.text import Tokenizer
from avsr.utils import cer, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--mode", default="audio")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    torch.manual_seed(0)
    tok = Tokenizer()
    rows = [r for r in assign_split(load_manifests(cfg.work_dir), cfg)["train"] if r["angle"] == "A"]
    rows = sorted(rows, key=lambda r: r["utt_id"])[:: max(1, len(rows) // args.n)][: args.n]
    ds = AVSRDataset(rows, cfg, train=False, tokenizer=tok)
    items = [ds[i] for i in range(len(ds))]
    batches = [collate_fn(items[i:i + args.batch]) for i in range(0, len(items), args.batch)]
    dev = torch.device("cuda")
    model = build_model(cfg, tok.vocab_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    to = lambda b: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}  # noqa: E731
    gb = [to(b) for b in batches]
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        b = gb[step % len(gb)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b, mode=args.mode)
            loss, parts = model.compute_loss(out, b, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        opt.step()
        sched.step()
        if step % 25 == 0 or step == 1:
            print(f"step {step:4d} ctc {parts['ctc']:8.2f} att_tok {parts['att_tok']:.3f} gnorm {float(gn):8.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    refs, hyps = [], []
    eval_mode = "av" if args.mode == "train" else args.mode
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in gb:
            for h, t in zip(model.decode(b, eval_mode, method="ctc_greedy"), b["texts"]):
                hyps.append(tok.decode(h))
                refs.append(tok.decode(tok.encode(t)))
    print(f"memorised CER ({eval_mode}): {100 * cer(refs, hyps):.1f}%")
    for r, h in list(zip(refs, hyps))[:3]:
        print("  REF", r, "\n  HYP", h)


if __name__ == "__main__":
    main()
