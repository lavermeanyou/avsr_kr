# Speaker–lip matching (audio-visual sync) model + 4-PC tooling — SPEC

Binding contract for the new components. Read `docs/SPEC.md` (sections 2, 4, 6, 8, 14) and `README.md` first; reuse
the existing, tested code instead of re-implementing it.

## 0. Product scenario (why)

A video shows several people. The audio team separates the mixed soundtrack into one waveform per speaker and
transcribes each waveform. **Our job:** decide which on-screen face produced each separated waveform (and when), so
the subtitle of that waveform can be attributed to that person. The lips do NOT have to read words; they only have to
tell whether a mouth moves in sync with a given audio stream. When no visible face matches a stream well enough, the
stream is an off-screen speaker.

## 1. Data (already preprocessed on this PC; nothing new to extract)

Use the existing preprocessed utterances (`work/manifests/*.jsonl`, `work/feats/<stem>/<utt>.npz|.mp4`, SPEC §4):
96×96 mouth crops at 30/29.97 fps, lip landmarks `lm [T,40,2]`, cues `cue [T,8]`, `valid [T]`, 16 kHz audio of the
same time span. Speaker split from `configs/*.yaml` `split.*` via `avsr.dataset.assign_split` (val C313, test E014 +
C159, train = the other 13). All 5–9 camera angles of one sentence share ONE audio track: their `(speaker, session,
sentence_id)` is identical (`avsr.dataset._audio_key(row)`). Such pairs are NOT negatives of each other.

Reuse from `avsr.dataset`: `load_manifests`, `assign_split`, `resample_indices`, `resampled_length`, `read_frames`,
`normalize_pixels`, `normalize_cue`, `cfg_get`, `_audio_key`; from `avsr.audio_feats`: `compute_fbank`, `stack_frames`,
`add_noise`, `make_noise`, `to_float_wave`; from `avsr.models.frontends`: `VisualFrontend`, `SkeletonFrontend`,
`AudioFrontend`. Everything runs at the model rate of 25 fps (video resampled 30→25 as in the AVSR dataset; audio =
kaldi fbank 100 Hz stacked ×4 = 25 Hz, 320 dims). One 25-fps frame = 40 ms.

## 2. `avsr/sync/data.py` — windows of synchronised mouth video + audio

```python
from avsr.sync.data import SyncWindowDataset, sync_collate, audio_window_features
ds = SyncWindowDataset(rows, cfg, train: bool, window: int | None = None, seed: int = 0)
```
- One item per row (an epoch sees every utterance once). Rows whose 25-fps length is `< window + 2` are dropped (count
  logged). `window` defaults to `cfg.sync.window_frames` (25 = 1 s).
- `train=True`: random window start; video augmentation as the AVSR dataset (random 88 crop from 96, horizontal flip
  with lm x negation, NO time masking); audio augmentation: with prob `sync.leak_prob` add ONE other utterance of a
  DIFFERENT speaker (imperfect separation leakage) at SNR ~ U(`sync.leak_snr_min`, `sync.leak_snr_max`) dB, and with
  prob `sync.noise_prob` add white/pink noise at U(`sync.noise_snr_min`, `sync.noise_snr_max`) dB; light SpecAugment
  (1 freq mask ≤ 8 bins, 1 time mask ≤ 4 frames at 100 Hz). `train=False`: centre window, no augmentation, deterministic.
- Video/cue normalisation exactly as the AVSR dataset with the same config keys (`video.norm`, `video.cue_norm`, mean/std).
- Audio features are computed from the waveform of the window only (plus a 0.1 s context on each side that is cut
  away after the fbank, so edge frames are not degraded), then per-window CMVN is applied by `compute_fbank`; returns
  exactly `window` frames of `[320]`.
- Shifted-audio hard negative (train and eval): the same utterance's audio window shifted by `s` frames,
  `|s| ~ U{sync.shift_min .. sync.shift_max}` (5..15 frames = 0.2..0.6 s), random sign, only if it stays inside the
  utterance; otherwise `has_shift = False` and zeros.
- Item dict: `video [W,C,88,88] float32`, `lm [W,80]`, `cue [W,8]`, `valid [W]`, `audio [W,320]`, `audio_shift [W,320]`,
  `has_shift` bool, `shift` int, `speaker` str, `audio_key` tuple/str, `utt_id` str, `start_frame` int.
- `sync_collate(items) -> dict` stacks the tensors (all windows have the same W) and keeps lists for strings; adds
  `audio_key_ids LongTensor [B]` (equal ids = same underlying audio) and `speaker_ids LongTensor [B]`.
- `audio_window_features(wave_int16_or_float, start_frame, window, sr=16000) -> FloatTensor [window, 320]`: the pure
  function used by the dataset and by evaluation (so eval can build distractor/shifted/leaky audio windows).
- Windows-safe (spawn workers, 1 thread per worker via the same helper the AVSR dataset uses), bad items skipped with a
  counter, deterministic eval.

## 3. `avsr/sync/model.py` — dual encoder

```python
from avsr.sync.model import SyncModel, build_sync_model
model = build_sync_model(cfg)                    # reads cfg.sync.*, cfg.video.channels, cfg.audio.n_mels*stack
v = model.embed_video(video, lm, cue, valid)     # [B,W,E], L2-normalised per frame
a = model.embed_audio(audio)                     # [B',W,E], L2-normalised per frame
s = model.pair_scores(v, a)                      # [B,B'] = mean over t of cos(v[b,t], a[b',t]) (same time index)
f = model.frame_scores(v, a)                     # [B,W] per-frame cosine for matched pairs (b with b)
loss, parts = model.compute_loss(batch_out...)   # see §4 (or a free function in avsr/sync/losses.py)
n = model.init_from_avsr(ckpt_path) -> dict      # copy frontend weights from an AVSR checkpoint (report copied/skipped)
```
- Video branch: `VisualFrontend(channels, 512, 'resnet18')` (if `sync.use_cnn`) and `SkeletonFrontend(88, 256, 256)`
  on `concat(lm, cue) * valid` (if `sync.use_skeleton`); at least one must be on. Concat → Linear → `sync.hidden`
  (256) → temporal encoder → Linear → `sync.emb_dim` (256) → L2 norm.
- Audio branch: `AudioFrontend(320, 256)` → Linear → `sync.hidden` → temporal encoder → Linear → `emb_dim` → L2 norm.
- Temporal encoder (`sync.temporal`): `conv` = 4 residual blocks of [Conv1d(k=5, dilation 1,2,4,1, same padding) →
  LayerNorm → GELU], or `gru` = 2-layer bidirectional GRU. Receptive field must stay local (≤ ~1 s) so the per-frame
  embedding reflects that moment's articulation.
- Learnable logit scale (init 10, clamp ≤ 100). bf16-autocast safe; cosine and losses in float32.
- `init_from_avsr`: maps `visual_frontend.*`, `skeleton_frontend.*`, `audio_frontend.*` from the AVSR checkpoint
  (`work/checkpoints/best.pt`, keys under `ckpt['model']`) when shapes match; lip features learnt for recognition are a
  strong start for sync. Configurable by `sync.init_from` (empty = from scratch).

## 4. Loss

For a batch of B windows: `S = scale * pair_scores(v, a)` `[B,B]`. Mask off-diagonal entries whose audio_key equals the
row's audio_key (other camera angle of the same audio) with −inf. Video→audio: cross-entropy over the row, with the
row's shifted-audio score appended as one extra negative column when `has_shift` (hard negative: same voice, same
content, wrong time). Audio→video: cross-entropy over the column (masked the same way). `loss = (l_va + l_av)/2`.
`parts`: loss, l_va, l_av, acc_va (argmax = diagonal), pos_cos (mean diagonal cosine), neg_cos, shift_cos.

## 5. `avsr/sync/train.py` — CLI

`& $py -m avsr.sync.train --config configs/sync.yaml [--set k=v ...] [--limit N] [--work-dir DIR]`
- Mirrors `avsr.train` conventions and reuses `avsr.utils` (load_config/overrides, set_seed, get_logger, metrics
  writer, checkpoint save/load, make_loader, autocast_context, cuda memory cap `train.cuda_mem_fraction`, the
  PYTORCH_CUDA_ALLOC_CONF default). AdamW, linear warmup + cosine, bf16, grad clip, graceful Ctrl+C, resume auto
  (refuse to resume a `--limit` run into a full one), non-finite step skip.
- Batches: `sync.batch_size` windows (default 64) from a shuffled `RandomSampler` over train rows (plain batch_size
  loader, drop_last=True). Logs to `work/logs/sync_train.log` + `work/logs/sync_metrics.jsonl`; checkpoints in
  `train.ckpt_dir` (default `work/checkpoints_sync`): `last.pt`, `best.pt`, rotating `epoch_XXX.pt`.
- Per-epoch validation on the val speaker: 4-way selection accuracy at 1 s and offset accuracy (§6), best.pt by the
  4-way accuracy. Print a few example numbers.

## 6. `avsr/sync/evaluate.py` — scenario metrics

`& $py -m avsr.sync.evaluate --ckpt work/checkpoints_sync/best.pt --split test [--max-utts N]`
Held-out speakers only. Fixed seed, deterministic. Writes `work/eval/sync_<split>_results.json` and prints tables:
1. **N-way selection** (the core question "which of these audio streams belongs to this face?"): for each video
   window, candidates = its own audio window + (N−1) windows from utterances of OTHER held-out speakers (different
   speakers only — as in the scenario, where separated streams belong to different people). Accuracy for
   N ∈ `sync.eval_n_way` ([2,3,4]) × window ∈ `sync.eval_windows` ([13, 25, 50] frames = 0.5/1/2 s). Also report the
   reverse direction (given an audio stream, which face?) at N=2..4.
2. **Leakage robustness:** the same 2- and 4-way test when every candidate stream contains the other candidates'
   speech leaked in at `sync.eval_leak_db` ([20, 10, 5] dB below) — imperfect separation.
3. **Sync offset:** for each window, scores for audio offsets −15..+15 frames; accuracy = argmax within ±1 frame of 0;
   also median absolute error in ms.
4. **Scene assignment:** K ∈ `sync.scene_k` ([2, 3]) distinct held-out speakers, each with one utterance, cropped to
   the common length (≥ 2 s); streams = each speaker's audio (+ leakage variants); score matrix K×K over the whole
   overlap and over 1-s sub-windows; Hungarian assignment (`scipy.optimize.linear_sum_assignment`); report scene
   accuracy (all correct) and per-window accuracy. ≥ 300 scenes per K.
5. **Match / no-match verification (off-screen speaker detection):** window scores of true pairs vs pairs with a
   different speaker's audio: ROC-AUC, EER and the score threshold at EER per window length — this threshold decides
   "no visible face matches → off-screen speaker". Save it in the JSON as `offscreen_threshold`.
Held-out speakers: val = C313; test = E014, C159 (only 2 speakers → for K=3 scenes and N=4 the distractors may come
from different SESSIONS of the same test speakers only if not enough speakers; say so in the output). Evaluate
`--split val` and `--split test`; `--split heldout` = val+test together (3 speakers) is the default for scenes.

## 7. `configs/sync.yaml`

Complete, standalone (contains every section the reused code reads: work_dir, seed, split, data, video, audio,
train) plus:
```yaml
sync:
  window_frames: 25
  shift_min: 5
  shift_max: 15
  hidden: 256
  emb_dim: 256
  temporal: conv
  use_cnn: true
  use_skeleton: true
  leak_prob: 0.5
  leak_snr_min: 5
  leak_snr_max: 30
  noise_prob: 0.3
  noise_snr_min: 5
  noise_snr_max: 30
  batch_size: 64
  init_from: work/checkpoints/best.pt
  eval_windows: [13, 25, 50]
  eval_n_way: [2, 3, 4]
  eval_leak_db: [20, 10, 5]
  scene_k: [2, 3]
  eval_max_utts: 600
```
`train:` epochs 30, lr 3.0e-4, warmup_steps 500, weight_decay 0.01, grad_clip 5, amp bf16, cuda_mem_fraction 0.8,
ckpt_dir work/checkpoints_sync, log_every 50, eval_every_epoch 1, keep_last 3, resume auto. `data.num_workers` 10.
`video` and `audio` sections identical to `configs/base.yaml` (norm: utterance, cue_norm: utterance, ...).

## 8. 4-PC tooling (all PCs hold the same raw dataset; preprocessing is split, training runs on one PC)

- `avsr/preprocess.py`: new option `--shard K/N` (e.g. `2/4`): the job list is sorted by stem and PC K keeps the jobs
  with `index % N == K-1`. Deterministic; independent of `--order`. `--dry-run` shows the shard's list.
- `tools/shards.py` (CLI, standard library + existing code only):
  - `pack --work-dir work_shard2 --dest <folder>`: copy finished videos (manifest `.jsonl` + `.done` + their
    `feats/<stem>/` directory) and `preprocess_log.jsonl` into `<folder>` (USB drive or `\\PC1\share\...`), skip what
    is already there with the same size, verify counts, print total size.
  - `merge --src <folder> [--src ...] --work-dir work`: copy shards into the main work dir; skip videos already
    `.done` there unless `--overwrite`; validate that every manifest row's npz/mp4 exists; summary table.
  - `status --work-dir work`: done videos, utterances, missing files, per-speaker counts.
- `scripts/setup.ps1` (fresh PC): winget Python 3.12 (user scope) + FFmpeg (Gyan.FFmpeg), pip install
  `requirements.txt` (torch/torchaudio/torchvision from the cu128 index), `msvc-runtime` (needed for torch DLLs on
  this Windows image), verify `torch.cuda.is_available()`, ffmpeg, mediapipe and the face_landmarker asset. Idempotent;
  logs to `setup.log`; clear messages.
- `scripts/make_dist.ps1`: zip the code for other PCs (`avsr/ tools/ scripts/ configs/ assets/ tests/ docs/
  README.md requirements.txt *.bat`), excluding `work*/`, `__pycache__`, checkpoints.
- `scripts/preprocess_shard.ps1 -Shard K -NumShards 4 -DataRoot <path> [-WorkDir work_shardK] [-Dest <transfer>]`,
  `scripts/merge_shards.ps1 -Src <a>,<b>,...`, `scripts/train_sync.ps1`, `scripts/evaluate_sync.ps1`: thin wrappers
  (same conventions as the existing `scripts/*.ps1`: refresh PATH, `$env:AVSR_PYTHON` override, Push/Pop-Location,
  exit code propagated).
- Double-click launchers at the project root, ASCII-only CONTENT (cmd cannot parse UTF-8 Korean reliably):
  `1_setup.bat`, `2_preprocess_part.bat` (asks: PC number 1-4, data folder, transfer folder), `3_merge_parts.bat`
  (asks for the transfer folders), `4_train_sync.bat`, `5_evaluate_sync.bat`. Each calls
  `powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\...ps1" ...` and pauses at the end.
  Any `.ps1` that contains Korean text must be saved as UTF-8 WITH BOM (Windows PowerShell 5.1 reads BOM-less files as
  the ANSI code page); otherwise keep scripts ASCII.
- README (Korean): new sections "화자-입술 매칭 모델" and "PC 4대로 나눠서 전처리하기" with exact commands.

## 9. Quality bar

Runnable tests without the real dataset (synthetic npz/mp4 in a temp dir) under `tests/test_sync.py` and
`tests/test_shards.py`; existing tests (`tests/test_*.py`) must keep passing. No new pip installs. No silent failures.
Deterministic eval. Windows-safe multiprocessing.
