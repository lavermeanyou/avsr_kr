# avsr_kr — Korean Audio-Visual Speech Recognition (noise-gated lip reading) — SYSTEM SPEC

This document is the contract every module must follow. Read it fully before writing code.
Project root: `C:\Users\user\Desktop\avsr_kr` (Windows 11, PowerShell). Python 3.12 at
`C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe` (call it `$py`; the bare `python` on PATH is a
Microsoft Store stub — do NOT use it). ffmpeg/ffprobe 9.0 are on PATH (after refreshing PATH from registry in a new
shell: `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`).
GPU: RTX 5080 16 GB (CUDA 12.8, bf16 OK). CPU: 20 cores. RAM: 63 GB. Free disk: ~490 GB.
Installed: torch 2.11.0+cu128, torchaudio 2.11.0, torchvision 0.26, opencv-python 5.0, mediapipe 1.0.1 (tasks API only,
`mp.solutions` does NOT exist), numpy 2.5, soundfile, scipy, jiwer 4.0, tqdm, pyyaml, pandas, matplotlib.

## 1. Goal

Train a model that produces Korean subtitles from a talking-face video using **audio + lip video**. Audio is usually
more accurate, so the product differentiator is: **estimate the acoustic noise level; when the estimated SNR is below a
threshold, rely on the lip (visual) stream**. The visual stream uses (a) the mouth-crop image sequence (CNN), (b) a lip
"skeleton" (MediaPipe lip landmarks, normalized), and (c) hand-crafted colour cues from inside the mouth (dark cavity
fraction, tongue/red fraction, etc.). The system must be reusable on any dataset that follows the AI-Hub
"립리딩(입모양) 음성인식 데이터" format.

## 2. Dataset facts (AI-Hub 립리딩(입모양) 음성인식 데이터) — verified on the real files

Root given by user: `C:\Users\user\Downloads\009.립리딩(입모양) 음성인식 데이터\01.데이터\2.Validation\`
(a `1.Training\` sibling may exist later — same layout). Under a split folder:
- `라벨링데이터\VL{n}.tar` — label archives (JSON only). Here: VL5..VL9, 960 JSON files total.
- `원천데이터\VS{n}.tar` — source archives (mp4 + wav). Here: VS5..VS9, 313 GB total. DO NOT extract everything
  (won't fit comfortably); stream members with Python `tarfile` and extract one media file at a time to a temp dir.
- Inside a tar, paths look like (UTF-8, Korean dir names):
  `VL7/소음환경1/E(전문가)/F(여성)/F(여성)_1/lip_J_1_F_03_E220_A_001.json`
  `VS7/소음환경1/E(전문가)/F(여성)/F(여성)_1/lip_J_1_F_03_E220_A_001.mp4` (+ `.wav` only for angle A).
  The same label/source may also be given already extracted as plain directories; support both (tar files or dirs).
- File stem pattern: `lip_J_{group}_{gender F|M}_{age}_{speakerID}_{angle A..I}_{session 001..}`.
  `speakerID` like `E220`, `C013` (E = 전문가/expert, C = 일반인). Angle letters A–I are **different cameras of the same
  recording session**; files with the same stem except the angle letter have IDENTICAL `Sentence_info` (verified).
  Angle A is frontal. 16 speakers × 60 files here; per speaker: 12 sessions × 5 angles (A + four of B..I).
- Video: 1920×1080, 30 fps, H.264, ~5 min (≈9245 frames), with an embedded AAC 48 kHz stereo audio track.
  **The embedded mp4 audio is identical across angles and sample-exact aligned to the video; the separate `.wav`
  (angle A only) is the same audio with a 21 ms offset. => USE THE MP4 EMBEDDED AUDIO (`ffmpeg -i x.mp4 -vn -ac 1 -ar 16000 -f s16le -`).**
- JSON label = a list with one dict:
  ```
  [{"dataSet": {...},
    "Video_info": {"video_Name": "lip_J_1_F_03_E220_A_001.mp4", "video_Format": "MP4", "video_Duration": "0:05:08", "FPS": 30, "Resolution": "1920*1080"},
    "Audio_info": {"Audio_Name": "...wav", "Audio_Format": "wav", "Audio_Duration": "0:05:08", "Sampling_rate": "48Khz", "Channel(s)": 2},
    "Audio_env": {"Noise": 1},                       # 1 or 2 (noise environment id)
    "Video_env": {"env": "indoor_light", "Angle": "A"},
    "Sentence_info": [{"ID": 1, "topic": "financial", "sentence_text": "최근에 ...", "start_time": 0.938666, "end_time": 7.808}, ...],  # ~40 per file
    "speaker_info": {"speaker_ID": "E220", "Specificity": "E", "Gender": "F", "Age": 3, "Accent": "N"},
    "Bounding_box_info": {"Face_bounding_box": {"xtl_ytl_xbr_ybr": [[y0,x0,y1,x1], ...]},   # one entry per video frame (9245)
                          "Lip_bounding_box":  {"xtl_ytl_xbr_ybr": [[y0,x0,y1,x1], ...]}}}]
  ```
  **IMPORTANT QUIRK: despite the key name `xtl_ytl_xbr_ybr`, the 4 numbers are actually `[y_top, x_left, y_bottom, x_right]`
  in 1920×1080 pixel coordinates (verified visually).** The lip box is a square ≈ 230 px (nose tip → chin, ≈2× mouth width).
  Bounding-box list length == number of video frames (verified for all 960 files, no malformed boxes).
- Sentences: 41,350 total (≈8,270 unique audio utterances × 5 angles). Duration min 0.64 s, median 4.65 s, mean 5.1 s,
  max 22.4 s. Sentences never overlap; median gap 0.45 s, min gap 0.05 s.
- Text characters: Hangul syllables (1,126 distinct), space, `.` `?` `!` `,` and rare `(` `)` `/` `1`, NBSP `\xa0`, `\n`.
  `X` (925 occurrences in 790 sentences) marks unintelligible/censored words → such sentences are EXCLUDED from
  training/eval (`has_unk = true` in manifest).
- Speakers (id: gender, noise env): C013 F 2, C044 F 2, C069 F 2, C159 F 2, C230 F 2, C084 M 1, C085 M 1, C272 M 1,
  C313 M 1, C332 M 1, C341 M 1, C464 M 1, C473 M 1, E014 M 1, E205 M 1, E220 F 1.
  Default speaker-independent split: **val = [C313], test = [E014, C159], train = the other 13.** If a `1.Training`
  folder is preprocessed too, everything from it is train and the Validation folder supplies val/test (config-driven).

## 3. Repository layout

```
avsr_kr/
  README.md                 (Korean, user-facing: setup, preprocess, train, evaluate, infer, how to adapt to new data)
  requirements.txt
  configs/base.yaml         (single source of hyper-parameters; see §9)
  assets/face_landmarker.task   (MediaPipe model, already downloaded)
  docs/SPEC.md              (this file)
  avsr/__init__.py
  avsr/aihub.py             (dataset-format layer: discover tars/dirs, parse labels, pair label↔media)   [core, written by lead]
  avsr/text.py              (Hangul jamo tokenizer)                                                     [core, written by lead]
  avsr/video_feats.py       (MediaPipe landmarks, mouth crop, skeleton + colour cues)                   [core, written by lead]
  avsr/preprocess.py        (CLI: dataset → work_dir features + manifest shards)                        [core, written by lead]
  avsr/audio_feats.py       (audio loading, fbank, noise augmentation, DSP SNR estimate)                [agent A]
  avsr/dataset.py           (torch Dataset, augmentation, collate, duration-bucketed batch sampler)     [agent A]
  avsr/models/__init__.py, frontends.py, encoder.py, fusion.py, decoder.py, avsr_model.py              [agent B]
  avsr/utils.py             (config loading/merging, seeding, checkpoint io, logging, CER/WER)          [agent C]
  avsr/train.py             (CLI training loop)                                                         [agent C]
  avsr/evaluate.py          (CLI: CER/WER per condition, SNR sweep → suggested threshold)               [agent C]
  avsr/infer.py             (CLI: video → .srt/.json subtitles with SNR-gated modality selection)       [agent D]
  scripts/preprocess.ps1, train.ps1, evaluate.ps1, infer.ps1                                             [agent D]
  tests/test_text.py, test_video_feats.py (lead), test_audio_dataset.py (A), test_models.py (B), test_train_smoke.py (C), test_infer.py (D)
  work/                     (default work_dir: features, manifests, checkpoints, logs) — gitignored
```
Everything is run as modules from the project root: `& $py -m avsr.preprocess ...`, `& $py -m avsr.train ...`.
All CLIs use `argparse`; `--config configs/base.yaml` plus `--set key.sub=value` overrides (see `avsr/utils.py`).
Use only the installed packages (no new pip installs). Windows: use `pathlib`, open text files with `encoding="utf-8"`,
never rely on the console encoding; DataLoader workers must be Windows-safe (top-level functions, no lambdas,
`if __name__ == "__main__":` guards, spawn start method).

## 4. Preprocessed feature format (produced by `avsr/preprocess.py`, consumed by `avsr/dataset.py`)

`work_dir/` (default `work/`):
```
work/feats/{video_stem}/{utt_id}.mp4     mouth-crop video: 96×96, RGB (yuv420p H.264, crf 18), 30 fps, T frames
work/feats/{video_stem}/{utt_id}.npz     see below
work/manifests/{video_stem}.jsonl        one manifest line per utterance of that video (shard); written atomically
work/manifests/{video_stem}.done         marker: video fully processed (preprocess skips videos with a .done marker)
work/preprocess_log.jsonl                per-video summary lines (timing, landmark success ratio, errors)
```
`utt_id = f"{video_stem}__{sentence_ID:03d}"`, e.g. `lip_J_1_F_03_E220_A_001__007`.

`.npz` keys (np.savez_compressed):
- `lm`    float16 [T, 40, 2] — normalized lip skeleton per frame at 30 fps (see §6). Zeros where `valid == 0`.
- `cue`   float16 [T, 8]     — per-frame cues: [mouth_width, inner_height, outer_height, inner_area, dark_frac, red_frac, mean_v, mean_s] (see §6). Zeros where invalid.
- `valid` uint8   [T]        — 1 if landmarks were detected on that frame.
- `audio` int16   [N]        — 16 kHz mono PCM of the utterance, N = round(duration × 16000).
- `fps`   float64 scalar (30.0); `sr` int64 scalar (16000).
T = number of video frames of the utterance (= frames in the mp4). Audio and video cover exactly the same time span:
video frames `[round(start×30), round(end×30))`, audio samples `[round(start×16000), round(end×16000))`, where
`start/end` are the label's times padded by `pad = min(0.10 s, gap_to_neighbour/2)` on each side (clamped to the video).

Manifest line (JSON, one per utterance):
```
{"utt_id": "...", "video_stem": "lip_J_1_F_03_E220_A_001", "split_dir": "2.Validation", "speaker": "E220", "gender": "F",
 "age": 3, "specificity": "E", "angle": "A", "session": "001", "noise_env": 1, "topic": "financial",
 "sentence_id": 7, "start": 40.1, "end": 46.9, "duration": 6.8, "n_frames": 204, "n_samples": 108800,
 "text": "정규화된 문장", "text_raw": "원문", "has_unk": false, "lm_valid_ratio": 0.995,
 "mouth_mp4": "feats/lip_J_1_F_03_E220_A_001/lip_J_1_F_03_E220_A_001__007.mp4",
 "npz": "feats/lip_J_1_F_03_E220_A_001/lip_J_1_F_03_E220_A_001__007.npz"}
```
Paths are relative to `work_dir`. The dataset layer loads ALL `work/manifests/*.jsonl` (glob) — no merge step.
Split assignment is done at load time by `dataset.py` from config (`split.val_speakers`, `split.test_speakers`,
`split.train_split_dirs`), never baked into the manifest.

## 5. Tokenizer (`avsr/text.py`) — API

```python
from avsr.text import Tokenizer, normalize_text
normalize_text(s) -> str      # NFC; \xa0/\n/\t → space; remove ( ) / ; collapse spaces; strip
tok = Tokenizer()             # fixed vocabulary, no files needed
tok.vocab_size -> int  (77)
tok.blank_id == 0; tok.pad_id == 1; tok.sos_id == 2; tok.eos_id == 3; tok.unk_id == 4; tok.space_id == 5
tok.encode(text: str) -> list[int]   # normalizes first; Hangul syllable → 2-3 jamo ids (초성, 중성[, 종성] with
                                      # DISTINCT id ranges for 초성 vs 종성); ' ' → space_id; . ? ! , → own ids; else unk_id
tok.decode(ids: list[int], strip_special=True) -> str   # recomposes syllables; drops blank/pad/sos/eos; unk → 'X'
tok.has_unk(text) -> bool
```
Vocabulary order: `<blank> <pad> <sos> <eos> <unk> <space>`, 19 초성 (ᄀ..ᄒ order as in Unicode L index), 21 중성,
27 종성, then `. ? ! ,`. CTC uses blank=0; attention decoder uses sos/eos/pad.

## 6. Visual features (`avsr/video_feats.py`) — API (used by preprocess and by infer)

```python
from avsr.video_feats import LIP_IDX, FaceLandmarker, skeleton_features, color_cues, mouth_box, crop_square, BoxSmoother
LIP_IDX: list[int]  # 40 MediaPipe indices: outer(20) + inner(20) lips
lmk = FaceLandmarker(model_path="assets/face_landmarker.task")      # MediaPipe tasks FaceLandmarker, VIDEO mode, 1 face
pts = lmk.detect(frame_bgr: np.ndarray[H,W,3] uint8, timestamp_ms: int) -> np.ndarray[478,2] float32 (pixel coords in
      the given frame) | None if no face. Timestamps must be strictly increasing per FaceLandmarker instance.
lm, geom = skeleton_features(pts)   # lm: [40,2] float32 lip points translated to mouth centre, rotated so the eye line
                                    # (33→263) is horizontal, scaled by inter-ocular distance; geom: [4] float32 =
                                    # [mouth_width(61-291)/IOD, inner_height(13-14)/IOD, outer_height(0-17)/IOD, inner_lip_polygon_area/IOD²]
col = color_cues(frame_bgr, pts)    # [4] float32 = [dark_frac, red_frac, mean_v, mean_s] computed on the pixels inside the
                                    # inner-lip polygon (HSV, OpenCV ranges): dark = V<60; red = (H<12 or H>168) & S>90 & V>60;
                                    # mean_v, mean_s in [0,1]; all zeros if polygon area < 4 px
cx, cy, side = mouth_box(pts, scale=1.0)   # square box centred on mean of LIP_IDX points, side = scale × max(IOD, 1.25 × eye-mid→mouth distance) (px)
                                           # (face-size based, so the crop does NOT zoom with mouth width/articulation)
crop = crop_square(frame_bgr, cx, cy, side, out_size=96)  # [96,96,3] uint8 BGR; pads black outside the frame; INTER_AREA
sm = BoxSmoother();  box = sm.update((cx, cy, side) or None)  # running median over last 9 boxes (default window); takes ONE
                                                             # tuple argument; on None (no face) returns the last smoothed box (or None)
face_input_from_box(frame, box_xyxy, expand=1.3, target=320) -> (crop, scale, x0, y0)  # point in crop p → p/scale + (x0,y0)
downscale_for_detection(frame, max_side=640) -> (small, scale)                         # point in small p → p/scale
box_from_xyxy(x0, y0, x1, y1) -> (cx, cy, side)
```
`cue` in the npz = concat(geom[4], col[4]). Preprocess resizes the label face box (expanded 1.3×, square, clipped) to
320 px before calling `detect`, then maps the 478 points back to full-frame pixels; on detection failure the crop box
falls back to the label lip box (converted to (cx, cy, side)). At inference (no labels) `infer.py` runs `detect` on the
frame downscaled so that max(H, W) ≤ 640, maps the points back, and uses `BoxSmoother` hold-last on failure.

## 7. Audio features (`avsr/audio_feats.py`) — API (agent A)

```python
from avsr.audio_feats import load_audio_16k, compute_fbank, stack_frames, add_noise, make_noise, estimate_snr_db
wave = load_audio_16k(path_or_video: str, start: float | None = None, end: float | None = None) -> np.int16 [N]  # ffmpeg pipe, mono 16 kHz
fb = compute_fbank(wave_int16_or_float: np.ndarray | torch.Tensor, sr=16000) -> torch.FloatTensor [n_frames, 80]
     # torchaudio.compliance.kaldi.fbank(num_mel_bins=80, frame_length=25, frame_shift=10, dither=0 at eval) on float waveform scaled to [-1,1]*32768 as kaldi expects;
     # then per-utterance mean/variance normalisation over time.
x = stack_frames(fb, stack=4) -> torch.FloatTensor [n_frames//4, 320]     # 100 Hz → 25 Hz (drop the remainder)
noisy, snr = add_noise(wave_float, noise_float, snr_db) -> (np.float32, float)   # scales noise to the requested SNR (RMS over the whole utterance)
noise = make_noise(kind: "white"|"pink"|"babble", n_samples, rng, babble_pool: list[np.ndarray] | None) -> np.float32
snr_db = estimate_snr_db(wave_float, sr=16000) -> float    # DSP estimate: frame RMS (25 ms / 10 ms), noise floor = 10th
     # percentile of frame energies (dB), speech level = mean of frames above the 70th percentile; returns speech − noise (dB), clipped to [-10, 40]
```

## 8. Dataset (`avsr/dataset.py`) — API (agent A)

```python
from avsr.dataset import AVSRDataset, DurationBatchSampler, collate_fn, load_manifests, assign_split
rows = load_manifests(work_dir) -> list[dict]           # all work/manifests/*.jsonl
split_rows = assign_split(rows, cfg) -> dict[str, list[dict]]   # keys train/val/test; drops has_unk rows; applies cfg.split.*,
                                                                # cfg.data.min_duration / max_duration, cfg.data.angles (list or "all")
ds = AVSRDataset(rows, cfg, train: bool, tokenizer)
item = ds[i] -> dict:
   video   FloatTensor [Tv, C, 88, 88]  C = cfg.video.channels (1 = grayscale, 3 = RGB), values normalised to [0,1] then (x-0.421)/0.165 style (mean/std from cfg)
           frames resampled 30 → 25 fps via index round(k*30/25); train: random crop 88 from 96 + horizontal flip p=0.5 + time-mask; eval: centre crop
   lm      FloatTensor [Tv, 80]   (40×2 flattened, same frame indices; horizontally flipped consistently when video is flipped: x → -x and swap left/right point order is NOT required — just negate x)
   cue     FloatTensor [Tv, 8]
   valid   FloatTensor [Tv]
   audio   FloatTensor [Ta, 320]  (stacked fbank at 25 Hz after optional noise augmentation)
   snr_bucket LongTensor scalar  (0 = clean/≥20 dB, 1 = 10–20, 2 = 0–10, 3 = <0 dB)  — from the augmentation actually applied
   tokens  LongTensor [L] (tokenizer.encode(text)), text str, utt_id str, and meta (speaker, angle, noise_env)
Tv and Ta are trimmed to the same length min(Tv, Ta) (they differ by ≤ 2 frames).
Noise augmentation (train only): with prob cfg.audio.noise_prob apply add_noise with kind sampled from cfg.audio.noise_kinds
and snr_db ~ U(cfg.audio.snr_min, cfg.audio.snr_max); babble pool = 2–4 other random training utterances' audio (load their npz).
Eval datasets accept `fixed_noise=(kind, snr_db)` to build the SNR sweep.
batch = collate_fn(list[item]) -> dict with padded tensors: video [B,T,C,88,88], lm [B,T,80], cue [B,T,8], valid [B,T],
   audio [B,T,320], lengths LongTensor [B] (shared T for audio & video), tokens [B,L] padded with pad_id, token_lengths [B],
   snr_bucket [B], texts list[str], utt_ids list[str], metas list[dict]
sampler = DurationBatchSampler(rows, max_frames=cfg.train.max_frames_per_batch, shuffle=True, seed)  # buckets by n_frames (25 fps count = n_frames*25/30), yields lists of indices
```
Mouth mp4 decoding: `cv2.VideoCapture` (read all frames → np.uint8 [T,96,96,3]); fall back to zeros + a logged warning if
the file is unreadable. Video pixel normalisation constants live in config.

## 9. Config (`configs/base.yaml`) — keys (agent C writes the file; everyone reads these names)

```yaml
work_dir: work
seed: 1234
split: {val_speakers: [C313], test_speakers: [E014, C159], train_split_dirs: [2.Validation, 1.Training]}
data: {angles: all, min_duration: 0.5, max_duration: 16.0, num_workers: 6}
video: {channels: 1, size: 96, crop: 88, fps_in: 30, fps_out: 25, mean: 0.421, std: 0.165, flip_prob: 0.5, time_mask_prob: 0.3, time_mask_max: 8}
audio: {sr: 16000, n_mels: 80, stack: 4, noise_prob: 0.6, noise_kinds: [babble, white, pink], snr_min: -5, snr_max: 20, specaug: {freq_mask: 2, freq_width: 10, time_mask: 2, time_width: 10}}
model:
  d_model: 512
  visual: {resnet: resnet18, out_dim: 512}
  skeleton: {hidden: 256, out_dim: 256}
  audio: {out_dim: 512}
  fusion: {p_av: 0.5, p_audio_only: 0.25, p_video_only: 0.25}
  encoder: {layers: 8, heads: 8, ffn: 2048, conv_kernel: 31, dropout: 0.1}
  decoder: {layers: 4, heads: 8, ffn: 2048, dropout: 0.1}
  ctc_weight: 0.7
  snr_head_weight: 0.1
  label_smoothing: 0.1
train: {max_frames_per_batch: 3200, epochs: 60, lr: 5.0e-4, warmup_steps: 4000, weight_decay: 0.01, grad_clip: 5.0, amp: bf16, ckpt_dir: work/checkpoints, log_every: 50, eval_every_epoch: 1, keep_last: 3, resume: auto}
eval: {decode: ctc_greedy, snr_sweep: [20, 10, 5, 0, -5], noise_kind: babble, conditions: [av, audio, video]}
infer: {snr_threshold: 5.0, snr_estimator: hybrid, segment_max_sec: 8.0, segment_min_sec: 1.0, chunk_pad_sec: 0.2}
```

## 10. Model (`avsr/models/`) — API (agent B)

```python
from avsr.models import AVSRModel, build_model
model = build_model(cfg, vocab_size)  -> AVSRModel (nn.Module)
out = model(batch, mode="av"|"audio"|"video"|"train")   # "train": per-sample modality dropout per cfg.model.fusion probs
   # batch keys as in §8 (tensors already on device). Returns dict:
   #   ctc_logits [B,T,V], enc_out [B,T,D], enc_lengths [B], snr_logits [B,4], att_logits [B,L,V] (teacher-forced with sos-prefixed tokens; only when tokens given)
loss, parts = model.compute_loss(out, batch, cfg)   # ctc (F.ctc_loss, blank=0, zero_infinity=True, log_softmax float32) +
                                                     # attention CE (label smoothing, eos-terminated targets, ignore pad) + snr CE; parts = dict of floats
hyps = model.decode(batch, mode, method="ctc_greedy"|"attn_greedy", max_len=…) -> list[list[int]]  # token ids without blanks/specials
```
Architecture (defaults):
- VisualFrontend: Conv3d(C,64,(5,7,7),stride(1,2,2),pad(2,3,3)) → BN → ReLU → MaxPool3d((1,3,3),(1,2,2),(0,1,1)) →
  per-frame ResNet-18 trunk (torchvision resnet18 layers 1–4, conv1/maxpool removed, weights from scratch) → global avg pool → 512.
- SkeletonFrontend: LayerNorm(88) → Linear(88,256) → GELU → Conv1d(256,256,k=5,pad=2) over time → GELU → Linear(256,256).
  Input = concat(lm[80], cue[8]) × valid mask.
- AudioFrontend: Linear(320,512) → LayerNorm → GELU → Linear(512,512).
- Fusion: modality dropout (zero a stream's features), concat [512+256+512] + learned presence embedding (2 flags) →
  Linear → d_model → LayerNorm. Sinusoidal positional encoding added.
- Encoder: `torchaudio.models.Conformer(input_dim=d_model, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size, dropout)`, forward(x, lengths).
- CTC head: Linear(d_model, vocab). Attention decoder: nn.TransformerDecoder (post-LN fine), token embedding + sinusoidal pos,
  causal mask, cross-attention key padding mask from enc_lengths. SNR head: mean-pool of audio-frontend features (masked) → Linear(512,4).
- Everything must run with `torch.autocast("cuda", dtype=torch.bfloat16)`; compute CTC loss in float32.

## 11. Training / evaluation (agent C)

`& $py -m avsr.train --config configs/base.yaml [--set train.epochs=1 ...] [--limit N]`
- Builds tokenizer, loads manifests, splits, DataLoader (spawn-safe, persistent workers, pin_memory), model, AdamW,
  warmup+cosine LR, bf16 autocast, grad clipping, per-step logging to console + `work/logs/train.log` + `work/logs/metrics.jsonl`.
- Each epoch (or every `eval_every_epoch`): greedy CTC decode on the val set under conditions `av` (clean), `audio` (clean),
  `video`; log CER/WER; save `last.pt`, and `best.pt` on best val CER (av). Checkpoint = {model, optimizer, scheduler, step, epoch, cfg, tokenizer vocab, best}.
  `train.resume: auto` resumes from `last.pt` if present.
- `--limit N` uses only N train / N val utterances (smoke tests). Must survive an empty val set gracefully.
`& $py -m avsr.evaluate --config configs/base.yaml --ckpt work/checkpoints/best.pt --split test`
- Reports CER/WER for conditions × SNR sweep (audio corrupted with `eval.noise_kind` at each SNR; `video` condition once),
  prints a table + writes `work/eval/{split}_results.json`, and suggests the SNR threshold = the highest SNR at which
  CER(av) > CER(video) (or "video never better" / "always better").
CER/WER via jiwer on `tokenizer.decode` outputs; CER on characters without spaces, WER on space-split words.

## 12. Inference (`avsr/infer.py`, agent D)

`& $py -m avsr.infer --config configs/base.yaml --ckpt work/checkpoints/best.pt --video input.mp4 [--audio other.wav] [--out out.srt] [--mode auto|av|audio|video] [--snr-threshold 5]`
1. Load audio (mp4 track or `--audio`) → 16 kHz mono. 2. Decode video frames (cv2.VideoCapture; if fps ≠ 30 resample by time to
   30 fps), run FaceLandmarker (downscaled frame), skeleton/cue features, mouth crops (BoxSmoother). 3. Segment the timeline
   into utterances: energy VAD on audio when est. SNR ≥ threshold, otherwise "visual VAD" on `inner_height` (mouth activity)
   — pauses split segments; enforce `segment_min_sec`/`segment_max_sec`; always produce at least one segment.
4. Per segment: est. SNR (`infer.snr_estimator`: `dsp` = estimate_snr_db, `model` = SNR head argmax bucket midpoint {25,15,5,-5},
   `hybrid` = mean of both); mode = `av` if snr ≥ threshold else `video` (unless `--mode` forces). Run model.decode.
5. Write `.srt` (index, `HH:MM:SS,mmm --> ...`, text) and `.json` sidecar (segments with start, end, text, mode, snr_est).
Prints a summary table (segment, mode, SNR, text).

## 13. Quality bar

- Every module has a runnable test under `tests/` that does not need the real dataset (synthetic npz/mp4 written in a temp dir).
- No silent failures: raise on schema mismatch; log and skip a single bad utterance in the dataset with a counter.
- Deterministic given `seed`. Windows-safe multiprocessing. Type hints and short docstrings. No unused code.

## 14. Amendments after implementation (these override the sections above; `configs/base.yaml` is authoritative)

- **Preprocessing (§4):** time→frame mapping uses the video's real fps from ffprobe (29.97 fps videos exist); manifest
  rows add `fps` and `time_shift`. A per-video label offset (`time_shift`, ±0.8 s search; observed −0.34…+0.4 s and more for speaker C085) is estimated from speech-energy
  contrast at sentence boundaries and applied only when it improves the contrast by ≥ 1 dB (`--no-align` disables).
  Padding is `min(0.20 s, gap/2)`. Frames are decoded restricted to the union of the label face boxes (+12 %) and
  downscaled so the face is ~360 px, by NVDEC (`h264_cuvid -crop -resize`) with an OpenCV fallback (`--decoder`).
  Audio↔video sync was verified on all 16 speakers (lip-opening vs loudness cross-correlation: lag 0–3 frames, i.e.
  normal visual lead; `tools/av_sync_check.py`), so no A/V shift is applied.
- **Normalisation (§8, §9):** `video.norm: global|utterance` and `video.cue_norm: none|utterance`
  (`avsr.dataset.normalize_pixels` / `normalize_cue`, shared with `infer.build_segment_batch`). Default `utterance`:
  expert-speaker crops are ~2x brighter than the others (two lighting setups), so per-utterance z-scoring stops lighting
  from acting as a speaker cue. Global constants measured on the data: mean 0.4355, std 0.1897.
- **Model (§10):** `att_logits` is `[B, L+1, V]` (eos-terminated targets need L+1 positions); `parts` also has `att_tok`.
  New options: `model.encoder.type: blstm|conformer` (+ `rnn_layers`, `rnn_hidden`), `model.ctc_upsample: U`
  (each encoder frame → U CTC sub-frames; `ctc_logits [B, U·T, V]`, `ctc_lengths = U·enc_lengths`),
  `model.pos_enc: none|sinusoidal`. The visual frontends are skipped for batches in which no sample uses them.
  **Default = BiLSTM encoder (4×320), U = 2, no absolute positions (30.7 M parameters).** Evidence (all on the full
  train set, audio-only, clean): the Conformer-S (d 256, 12 blocks) stayed on the CTC blank plateau (CTC ≈ 200/utt,
  input-independent hypotheses) for 3k+ steps in every variant tried (lr 3e-4…1e-3, batch 3200/9600 frames, U = 1/2,
  with/without positions, hybrid or CTC-only), although it memorises 32 utterances in 100 steps; a BiLSTM encoder in the
  same model left the plateau within ~500 steps (val CER on an unseen speaker 43 % after epoch 1, 36 % after epoch 2).
  Data/labels were verified independently (clean spectrograms; syllable count vs duration r = 0.89, 0.16 when shifted).
- **Training (§11):** `train.fusion_curriculum` (per-epoch modality-dropout stages: audio-only → AV + audio →
  `model.fusion`), `train.aug_start_epoch` (clean audio before it), `train.cuda_mem_fraction` (keeps PyTorch's CUDA cache
  inside dedicated VRAM; on Windows an overgrown cache spills into system RAM and training slows ~2.5×).
- **Evaluation (§11):** noisy eval sets draw babble from the TRAIN speakers (as in training). For every sweep level the
  evaluation also records the DSP / SNR-head / hybrid SNR *estimates* inference would compute and converts the
  true-SNR threshold into estimator units: `suggested_snr_threshold.infer_threshold.{hybrid,dsp,model}` (midpoint of the
  estimates at the threshold level and the next cleaner level). That value is what `infer.snr_threshold` compares against.
- **Inference (§12):** adaptive energy-VAD margin `clip(0.35·(p90−p10), 2, 6) dB`; lips visible on < 20 % of frames →
  energy VAD and `audio` mode; SRT written as UTF-8 without BOM, CRLF; `infer.segment_max_sec` default 12.
