"""Tests for avsr.models (SPEC section 10). Run from the project root: $py -m tests.test_models

Needs no dataset. CUDA parts (bf16 autocast, timing, overfit check) run when a GPU is available.
"""
from __future__ import annotations

import copy
import inspect
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchaudio.models import Conformer  # noqa: E402

from avsr.models import AVSRModel, build_model  # noqa: E402
from avsr.models.avsr_model import DECODE_METHODS, ctc_collapse  # noqa: E402
from avsr.models.encoder import ConformerEncoder  # noqa: E402
from avsr.models.fusion import modality_flags  # noqa: E402
from avsr.text import Tokenizer  # noqa: E402

VOCAB = 77
PAD = Tokenizer.pad_id
HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda") if HAS_CUDA else None
CPU = torch.device("cpu")
# ~5 s of speech (29 syllables -> 80 jamo/space/punct tokens): the realistic CTC-feasibility case.
LONG_SENTENCE = "최근에 저희 팀장님께서, 시간을 돌리면 어떤 걸 하고 싶냐고 물었어요."


def make_cfg(small: bool = True) -> dict[str, Any]:
    """Config dict mirroring SPEC section 9 (only the sections the model reads, plus the loss weights)."""
    if small:
        model = {"d_model": 64, "visual": {"resnet": "resnet18", "out_dim": 64},
                 "skeleton": {"hidden": 32, "out_dim": 32}, "audio": {"out_dim": 48},
                 "encoder": {"layers": 1, "heads": 4, "ffn": 128, "conv_kernel": 7, "dropout": 0.1},
                 "decoder": {"layers": 1, "heads": 4, "ffn": 128, "dropout": 0.1}}
    else:
        model = {"d_model": 512, "visual": {"resnet": "resnet18", "out_dim": 512},
                 "skeleton": {"hidden": 256, "out_dim": 256}, "audio": {"out_dim": 512},
                 "encoder": {"layers": 8, "heads": 8, "ffn": 2048, "conv_kernel": 31, "dropout": 0.1},
                 "decoder": {"layers": 4, "heads": 8, "ffn": 2048, "dropout": 0.1}}
    model.update({"fusion": {"p_av": 0.5, "p_audio_only": 0.25, "p_video_only": 0.25},
                  "ctc_weight": 0.7, "snr_head_weight": 0.1, "label_smoothing": 0.1})
    return {"work_dir": "work", "seed": 1234,
            "video": {"channels": 1, "size": 96, "crop": 88, "fps_in": 30, "fps_out": 25, "mean": 0.421, "std": 0.165},
            "audio": {"sr": 16000, "n_mels": 80, "stack": 4},
            "model": model}


def make_batch(lengths: list[int], token_lengths: list[int], device: torch.device, seed: int = 0,
               channels: int = 1) -> dict[str, Any]:
    """Random collate_fn-style batch (SPEC section 8): zero padding, tokens padded with pad_id."""
    g = torch.Generator().manual_seed(seed)
    b, t, l = len(lengths), max(lengths), max(token_lengths)
    mask = (torch.arange(t)[None] < torch.tensor(lengths)[:, None]).float()
    valid = (torch.rand(b, t, generator=g) > 0.1).float() * mask
    tokens = torch.full((b, l), PAD, dtype=torch.long)
    for i, n in enumerate(token_lengths):
        tokens[i, :n] = torch.randint(5, VOCAB, (n,), generator=g)
    batch = {
        "video": torch.randn(b, t, channels, 88, 88, generator=g) * mask[:, :, None, None, None],
        "lm": torch.randn(b, t, 80, generator=g) * 0.3 * valid[..., None],
        "cue": torch.rand(b, t, 8, generator=g) * valid[..., None],
        "valid": valid,
        "audio": torch.randn(b, t, 320, generator=g) * mask[..., None],
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "tokens": tokens,
        "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
        "snr_bucket": torch.randint(0, 4, (b,), generator=g),
        "texts": ["" for _ in lengths], "utt_ids": [f"u{i}" for i in range(b)], "metas": [{} for _ in lengths],
    }
    return to_device(batch, device)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def grad_norm(module: torch.nn.Module) -> float:
    grads = [p.grad.float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(grads).norm()) if grads else 0.0


# ------------------------------------------------------------------------------------------------------ tests
def test_conformer_signature_and_encoder() -> None:
    """torchaudio Conformer API as assumed; the padding-aware wrapper equals torchaudio without padding and makes
    a sequence's output independent of batch padding."""
    params = inspect.signature(Conformer.__init__).parameters
    for name in ("input_dim", "num_heads", "ffn_dim", "num_layers", "depthwise_conv_kernel_size", "dropout"):
        assert name in params, name
    assert list(inspect.signature(Conformer.forward).parameters) == ["self", "input", "lengths"]

    torch.manual_seed(0)
    enc = ConformerEncoder(d_model=32, num_layers=2, num_heads=4, ffn_dim=64, conv_kernel=7, dropout=0.1).eval()
    x = torch.randn(2, 20, 32)
    full = torch.tensor([20, 20])
    with torch.no_grad():
        ours, _ = enc(x, full)
        ref, _ = enc.conformer(x, full)
        assert torch.allclose(ours, ref, atol=1e-5), float((ours - ref).abs().max())
        # sample 1 has 13 real frames; garbage in its padding must not change its output
        lengths = torch.tensor([20, 13])
        x2 = x.clone()
        x2[1, 13:] = torch.randn(7, 32) * 100
        a, _ = enc(x, lengths)
        b, _ = enc(x2, lengths)
        alone, _ = enc(x[1:2, :13], torch.tensor([13]))
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a[1, :13], alone[0], atol=1e-5), float((a[1, :13] - alone[0]).abs().max())
    assert float(a[1, 13:].abs().max()) == 0.0
    # torchaudio alone leaks padding through the depthwise conv (documents why the wrapper exists)
    with torch.no_grad():
        leak = (enc.conformer(x, lengths)[0][1, :13] - enc.conformer(x2, lengths)[0][1, :13]).abs().max()
    print(f"  encoder: wrapper == torchaudio (no padding); padding leak torchaudio={float(leak):.3g}, wrapper=0")


def test_modality_flags() -> None:
    probs = (0.5, 0.25, 0.25)
    for mode, (ea, ev) in {"av": (1, 1), "audio": (1, 0), "video": (0, 1)}.items():
        a, v = modality_flags(mode, 3, probs, CPU)
        assert a.tolist() == [bool(ea)] * 3 and v.tolist() == [bool(ev)] * 3
    torch.manual_seed(0)
    a, v = modality_flags("train", 20000, probs, CPU)
    frac = [float((a & v).float().mean()), float((a & ~v).float().mean()), float((~a & v).float().mean())]
    assert not bool((~a & ~v).any())
    assert all(abs(f - p) < 0.02 for f, p in zip(frac, probs)), frac
    try:
        modality_flags("bogus", 2, probs, CPU)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode must raise")


def test_teacher_forcing_and_ctc_collapse() -> None:
    torch.manual_seed(0)
    model = build_model(make_cfg(), VOCAB)
    tokens = torch.tensor([[10, 11, 12], [20, PAD, PAD]])
    inp, tgt = model.decoder.teacher_forcing(tokens, torch.tensor([3, 1]))
    assert inp.tolist() == [[2, 10, 11, 12], [2, 20, 1, 1]], inp.tolist()
    assert tgt.tolist() == [[10, 11, 12, 3], [20, 3, 1, 1]], tgt.tolist()

    best = torch.tensor([[0, 7, 7, 0, 7, 8, 8, 5, 0, 9], [3, 7, 2, 1, 7, 7, 0, 9, 9, 9]])
    mask = torch.tensor([[True] * 10, [True] * 8 + [False] * 2])
    assert ctc_collapse(best, mask, (0, 1, 2, 3)) == [[7, 7, 8, 5, 9], [7, 7, 9]]


def test_forward_modes_and_invariance() -> None:
    """Shapes for every mode on CPU (fp32); audio-mode output ignores video/skeleton, video-mode ignores audio."""
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build_model(cfg, VOCAB).eval()
    batch = make_batch([40, 31], [12, 7], CPU)
    with torch.no_grad():
        for mode in ("av", "audio", "video", "train"):
            out = model(batch, mode=mode)
            assert out["ctc_logits"].shape == (2, 40, VOCAB)
            assert out["enc_out"].shape == (2, 40, 64)
            assert out["enc_lengths"].tolist() == [40, 31]
            assert out["snr_logits"].shape == (2, 4)
            assert out["att_logits"].shape == (2, 13, VOCAB)
            assert all(bool(torch.isfinite(v).all()) for v in out.values())
        no_tokens = {k: v for k, v in batch.items() if k not in ("tokens", "token_lengths")}
        assert "att_logits" not in model(no_tokens, mode="av")

        other_visual = dict(batch, video=torch.randn_like(batch["video"]) * 5, lm=torch.randn_like(batch["lm"]),
                            cue=torch.rand_like(batch["cue"]), valid=torch.ones_like(batch["valid"]))
        other_audio = dict(batch, audio=torch.randn_like(batch["audio"]) * 5)
        ref_a, alt_a = model(batch, mode="audio"), model(other_visual, mode="audio")
        ref_v, alt_v = model(batch, mode="video"), model(other_audio, mode="video")
        for key in ("ctc_logits", "enc_out", "att_logits"):
            assert torch.equal(ref_a[key], alt_a[key]), f"audio mode depends on video ({key})"
            assert torch.equal(ref_v[key], alt_v[key]), f"video mode depends on audio ({key})"
        # and the streams are actually used in the other modes
        assert not torch.allclose(model(batch, mode="av")["ctc_logits"], model(other_visual, mode="av")["ctc_logits"])
        assert not torch.allclose(model(batch, mode="av")["ctc_logits"], model(other_audio, mode="av")["ctc_logits"])
        # a sequence's output does not depend on how much the batch pads it (eval mode)
        single = {k: (v[1:2, :31] if k in ("video", "lm", "cue", "valid", "audio") else v[1:2])
                  for k, v in batch.items() if isinstance(v, torch.Tensor)}
        single["tokens"] = single["tokens"][:, :7]
        full, alone = model(batch, mode="av"), model(single, mode="av")
        diff = (full["ctc_logits"][1, :31] - alone["ctc_logits"][0]).abs().max()
        assert float(diff) < 1e-4, float(diff)
        assert float((full["att_logits"][1, :8] - alone["att_logits"][0]).abs().max()) < 1e-4
    print("  forward: all modes OK; audio/video invariance OK; padding invariance max diff", f"{float(diff):.2e}")


def test_train_mode_matches_eval_modes() -> None:
    """Per-sample modality dropout in ``train`` mode computes, for each sample, exactly what the ``av`` / ``audio`` /
    ``video`` modes compute (no train/eval mismatch for the SNR-gated modality switch)."""
    torch.manual_seed(0)
    model = build_model(make_cfg(), VOCAB).eval()   # eval: no dropout, BatchNorm running stats
    batch = make_batch([24, 18, 12], [8, 6, 4], CPU)
    names = {(True, True): "av", (True, False): "audio", (False, True): "video"}
    seen: set[str] = set()
    with torch.no_grad():
        refs = {m: model(batch, mode=m)["ctc_logits"] for m in names.values()}
        for seed in range(4):
            torch.manual_seed(seed)
            audio_on, visual_on = modality_flags("train", 3, model.fusion_probs, CPU)
            torch.manual_seed(seed)                 # same draw inside forward
            out = model(batch, mode="train")["ctc_logits"]
            for i in range(3):
                mode = names[(bool(audio_on[i]), bool(visual_on[i]))]
                seen.add(mode)
                assert torch.allclose(out[i], refs[mode][i], rtol=0.0, atol=1e-6), (seed, i, mode)
    assert seen == set(names.values()), seen
    print(f"  train-mode dropout == eval modes for {sorted(seen)}")


def test_short_and_empty_edge_cases_cpu() -> None:
    """Utterances shorter than every temporal kernel (down to 1 frame) and empty transcripts: finite loss and
    gradients, and both decoders run. (A batch holding a single 1-frame utterance is only valid in eval mode:
    training BatchNorm needs more than one value per channel; data.min_duration keeps such items out.)"""
    torch.manual_seed(0)
    cfg = make_cfg()                                # encoder conv_kernel 7, skeleton kernel 5, stem kernel 5
    model = build_model(cfg, VOCAB).train()
    batch = make_batch([3, 1], [2, 0], CPU)
    loss, parts = model.compute_loss(model(batch, mode="train"), batch, cfg)
    loss.backward()
    assert torch.isfinite(loss) and grad_norm(model) > 0, parts
    empty = dict(batch, tokens=batch["tokens"][:, :0], token_lengths=torch.zeros(2, dtype=torch.long))
    loss, parts = model.compute_loss(model(empty, mode="av"), empty, cfg)
    assert torch.isfinite(loss), parts
    model.eval()
    one = make_batch([1], [1], CPU)
    for mode in ("av", "audio", "video"):
        for method in DECODE_METHODS:
            hyps = model.decode(one, mode, method=method, max_len=3)
            assert len(hyps) == 1 and len(hyps[0]) <= 3, hyps


def test_loss_backward_cpu() -> None:
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build_model(cfg, VOCAB).train()
    batch = make_batch([40, 31], [12, 7], CPU)
    out = model(batch, mode="train")
    loss, parts = model.compute_loss(out, batch, cfg)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"loss", "ctc", "att", "att_tok", "snr"}
    assert all(isinstance(v, float) for v in parts.values())
    assert abs(parts["loss"] - (0.7 * parts["ctc"] + 0.3 * parts["att"] + 0.1 * parts["snr"])) < 1e-3 * parts["loss"]
    # attention CE is per utterance (sum over the 12+1 and 7+1 eos-terminated targets / B), like CTC
    assert abs(parts["att"] - parts["att_tok"] * (13 + 8) / 2) < 1e-3 * parts["att"]
    # reference values computed independently
    with torch.no_grad():
        ref_ctc = F.ctc_loss(out["ctc_logits"].float().log_softmax(-1).transpose(0, 1), batch["tokens"],
                             batch["lengths"], batch["token_lengths"], reduction="sum", zero_infinity=True) / 2
        ref_snr = F.cross_entropy(out["snr_logits"].float(), batch["snr_bucket"])
    assert abs(parts["ctc"] - float(ref_ctc)) < 1e-3 * parts["ctc"] and abs(parts["snr"] - float(ref_snr)) < 1e-5
    print("  cpu loss:", {k: round(v, 3) for k, v in parts.items()})


def test_ctc_realistic_length() -> None:
    """T = 125 frames (5 s at 25 fps) with L = 80 jamo tokens: CTC must be feasible, i.e. finite without
    zero_infinity (which would otherwise silently zero the loss)."""
    tok = Tokenizer()
    ids = tok.encode(LONG_SENTENCE)
    assert len(ids) == 80, len(ids)
    repeats = sum(int(a == b) for a, b in zip(ids, ids[1:]))
    assert 125 >= len(ids) + repeats
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build_model(cfg, VOCAB).train()
    batch = make_batch([125, 110], [80, 60], CPU)
    batch["tokens"][0, :80] = torch.tensor(ids)
    out = model(batch, mode="av")
    loss, parts = model.compute_loss(out, batch, cfg)
    raw = F.ctc_loss(out["ctc_logits"].float().log_softmax(-1).transpose(0, 1), batch["tokens"], out["enc_lengths"],
                     batch["token_lengths"], blank=0, reduction="none", zero_infinity=False).detach()
    assert bool(torch.isfinite(raw).all()) and bool((raw > 0).all()), raw
    assert parts["ctc"] > 0 and torch.isfinite(loss)
    print(f"  ctc feasibility: T=125 L=80 repeats={repeats} -> per-utt ctc {[round(float(r), 1) for r in raw]}")


def test_decode_cpu() -> None:
    torch.manual_seed(0)
    model = build_model(make_cfg(), VOCAB).eval()
    batch = make_batch([40, 31], [12, 7], CPU)
    specials = {0, 1, 2, 3}
    for mode in ("av", "audio", "video"):
        for method in ("ctc_greedy", "attn_greedy"):
            hyps = model.decode(batch, mode=mode, method=method, max_len=15)
            assert isinstance(hyps, list) and len(hyps) == 2
            assert all(isinstance(h, list) and all(isinstance(i, int) for i in h) for h in hyps)
            assert all(not (set(h) & specials) for h in hyps), hyps
            if method == "attn_greedy":
                assert all(len(h) <= 15 for h in hyps)
            else:
                assert len(hyps[0]) <= 40 and len(hyps[1]) <= 31
    for bad in (("train", "ctc_greedy"), ("av", "beam")):
        try:
            model.decode(batch, mode=bad[0], method=bad[1])
        except ValueError:
            continue
        raise AssertionError(f"decode{bad} must raise")


def test_cuda_bf16_backward_and_decode() -> None:
    """Small model on CUDA under bf16 autocast: finite loss, gradients in every frontend, both decoders."""
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build_model(cfg, VOCAB).to(CUDA).train()
    batch = make_batch([40, 31], [12, 7], CUDA)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch, mode="av")
        loss, parts = model.compute_loss(out, batch, cfg)
    assert out["ctc_logits"].dtype == torch.bfloat16
    loss.backward()
    assert torch.isfinite(loss), parts
    for name in ("visual_frontend", "skeleton_frontend", "audio_frontend", "fusion", "encoder", "decoder",
                 "ctc_head", "snr_head"):
        module = getattr(model, name)
        missing = [n for n, p in module.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"{name}: no grad for {missing[:5]}"
        gn = grad_norm(module)
        assert gn > 0 and gn == gn and gn != float("inf"), (name, gn)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch, mode="train")
        loss, _ = model.compute_loss(out, batch, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for method in ("ctc_greedy", "attn_greedy"):
            hyps = model.decode(batch, mode="av", method=method, max_len=20)
            assert len(hyps) == 2 and all(isinstance(h, list) for h in hyps)
    print("  cuda bf16 loss:", {k: round(v, 3) for k, v in parts.items()})


def test_overfit_one_batch_cuda() -> None:
    """End-to-end wiring check: a small model memorises two utterances; both decoders recover the targets."""
    torch.manual_seed(0)
    cfg = copy.deepcopy(make_cfg())
    for section in ("encoder", "decoder"):
        cfg["model"][section]["dropout"] = 0.0
    cfg["model"]["label_smoothing"] = 0.0
    cfg["model"]["ctc_weight"] = 0.5
    model = build_model(cfg, VOCAB).to(CUDA).train()
    batch = make_batch([40, 31], [12, 7], CUDA, seed=3)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    first = last = 0.0
    for step in range(300):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = model.compute_loss(model(batch, mode="av"), batch, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        first = parts["loss"] if step == 0 else first
        last = parts["loss"]
    model.eval()
    targets = [batch["tokens"][i, :n].tolist() for i, n in enumerate(batch["token_lengths"].tolist())]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ctc = model.decode(batch, mode="av", method="ctc_greedy")
        att = model.decode(batch, mode="av", method="attn_greedy", max_len=30)
    print(f"  overfit: loss {first:.2f} -> {last:.3f}; ctc exact={ctc == targets}; attn exact={att == targets}")
    assert last < 0.1 * first, (first, last)
    assert ctc == targets, (ctc, targets)
    assert att == targets, (att, targets)


def test_blstm_ctc_upsample_no_pos() -> None:
    """encoder.type=blstm + ctc_upsample=2 + pos_enc=none (the configs/base.yaml setting): shapes, padding
    invariance, train-mode == eval-mode per sample, skipped visual frontend on all-audio batches, and memorisation."""
    torch.manual_seed(0)
    dev = CUDA if HAS_CUDA else torch.device("cpu")
    cfg = copy.deepcopy(make_cfg())
    cfg["model"]["encoder"].update({"type": "blstm", "rnn_layers": 2, "rnn_hidden": 48, "dropout": 0.0})
    cfg["model"].update({"ctc_upsample": 2, "pos_enc": "none", "label_smoothing": 0.0, "ctc_weight": 0.5})
    cfg["model"]["decoder"]["dropout"] = 0.0
    model = build_model(cfg, VOCAB).to(dev)
    assert type(model.encoder).__name__ == "BLSTMEncoder" and model.ctc_upsample == 2
    assert isinstance(model.fusion.pos, torch.nn.Identity)
    batch = make_batch([40, 31], [12, 7], dev, seed=5)
    model.eval()
    with torch.no_grad():
        out = model(batch, mode="av")
        assert out["ctc_logits"].shape == (2, 80, VOCAB) and out["ctc_lengths"].tolist() == [80, 62]
        alone = model({k: (v[1:2, :31] if isinstance(v, torch.Tensor) and v.dim() > 1 and k != "tokens" else
                           (v[1:2] if isinstance(v, torch.Tensor) else v)) for k, v in batch.items()}, mode="av")
        diff = float((out["ctc_logits"][1, :62] - alone["ctc_logits"][0]).abs().max())
        assert diff < 1e-3, diff                        # no leakage (leaks are ~0.1); cuDNN/TF32 noise is ~1e-4
        hyp = model.decode(batch, mode="audio", method="ctc_greedy")
        assert len(hyp) == 2
    # all-audio 'train' batch: visual frontend is skipped and output equals 'audio' mode
    model.fusion_probs = (0.0, 1.0, 0.0)
    with torch.no_grad():
        assert torch.allclose(model(batch, mode="train")["ctc_logits"], model(batch, mode="audio")["ctc_logits"])
    model.fusion_probs = (0.5, 0.25, 0.25)
    # memorise two utterances through the upsampled CTC
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(250):
        loss, parts = model.compute_loss(model(batch, mode="audio"), batch, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    targets = [batch["tokens"][i, :n].tolist() for i, n in enumerate(batch["token_lengths"].tolist())]
    with torch.no_grad():
        ctc = model.decode(batch, mode="audio", method="ctc_greedy")
    print(f"  blstm+upsample: padding diff {diff:.1e}, final ctc {parts['ctc']:.3f}, exact={ctc == targets}")
    assert ctc == targets, (ctc, targets)


def test_blstm_legacy_checkpoint_keys() -> None:
    """BLSTMEncoder = stacked single-layer LSTMs; weights saved from the old single multi-layer ``rnn`` module load
    (key remap) and give exactly the old output (eval mode)."""
    from avsr.models.encoder import BLSTMEncoder
    torch.manual_seed(0)
    old = torch.nn.LSTM(32, 16, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1).eval()
    enc = BLSTMEncoder(32, num_layers=3, hidden=16, dropout=0.1).eval()
    sd = {"rnn." + k: v for k, v in old.state_dict().items()}
    sd.update({"proj." + k: v for k, v in enc.proj.state_dict().items()})
    sd.update({"norm." + k: v for k, v in enc.norm.state_dict().items()})
    enc.load_state_dict(sd, strict=True)
    x, lengths = torch.randn(2, 20, 32), torch.tensor([20, 13])
    with torch.no_grad():
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = old(packed)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=20)
        pad = (torch.arange(20)[None] >= lengths[:, None]).unsqueeze(-1)
        ref = enc.norm(enc.proj(h)).masked_fill(pad, 0.0)
        out, _ = enc(x, lengths)
    diff = float((out - ref).abs().max())
    print(f"  blstm legacy keys: max diff {diff:.1e}")
    assert diff < 1e-5, diff


def test_default_size_and_speed_cuda() -> None:
    """Default (SPEC) model: parameter count and forward+backward time for B=8, T=150 under bf16 autocast."""
    torch.manual_seed(0)
    cfg = make_cfg(small=False)
    model: AVSRModel = build_model(cfg, VOCAB)
    total = n_params(model)
    parts = {n: n_params(getattr(model, n)) for n in ("visual_frontend", "skeleton_frontend", "audio_frontend",
                                                      "fusion", "encoder", "decoder", "ctc_head", "snr_head")}
    print(f"  params: {total / 1e6:.1f} M  " + ", ".join(f"{k}={v / 1e6:.2f}M" for k, v in parts.items()))
    assert 60e6 <= total <= 90e6, total
    if not HAS_CUDA:
        return
    model = model.to(CUDA).train()
    lengths = [150, 150, 145, 140, 138, 135, 130, 128]
    batch = make_batch(lengths, [100, 96, 95, 90, 90, 88, 85, 80], CUDA, seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(6):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, mode="train")
            loss, _ = model.compute_loss(out, batch, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        assert torch.isfinite(loss)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.decode(batch, mode="av", method="ctc_greedy")
        torch.cuda.synchronize()
        t_dec = time.perf_counter() - t0
    print(f"  speed B=8 T=150 bf16: fwd+bwd+step {min(times[2:]) * 1000:.0f} ms (first {times[0] * 1000:.0f} ms), "
          f"peak mem {peak:.2f} GiB, ctc decode {t_dec * 1000:.0f} ms")


def main() -> None:
    tests = [test_conformer_signature_and_encoder, test_modality_flags, test_teacher_forcing_and_ctc_collapse,
             test_forward_modes_and_invariance, test_train_mode_matches_eval_modes,
             test_short_and_empty_edge_cases_cpu, test_loss_backward_cpu, test_ctc_realistic_length, test_decode_cpu,
             test_blstm_ctc_upsample_no_pos, test_blstm_legacy_checkpoint_keys]
    if HAS_CUDA:
        tests += [test_cuda_bf16_backward_and_decode, test_overfit_one_batch_cuda]
    else:
        print("CUDA not available: skipping CUDA/bf16 tests")
    tests.append(test_default_size_and_speed_cuda)
    for fn in tests:
        t0 = time.perf_counter()
        fn()
        print(f"{fn.__name__}: OK ({time.perf_counter() - t0:.1f} s)")
    print("test_models: ALL OK")


if __name__ == "__main__":
    main()
