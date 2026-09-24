# avsr_kr — 한국어 음성+입술(AVSR) 자막 생성기

말하는 사람의 얼굴이 나오는 영상에서 **음성과 입술 움직임을 함께 사용해 한국어 자막(.srt)** 을 만듭니다.
보통은 음성이 더 정확하므로 음성+입술(`av`)로 인식하고, **추정한 잡음 수준(SNR)이 임계값보다 낮으면
입술(`video`)만으로 인식**하도록 자동 전환하는 것이 이 프로젝트의 핵심입니다.

- 입술(시각) 정보는 세 가지를 씁니다.
  1. 입 주변 96×96 크롭 영상 → 3D-Conv + ResNet-18 (CNN)
  2. 입술 "스켈레톤": MediaPipe 얼굴 랜드마크 중 입술 40점(바깥 20 + 안쪽 20)을 입 중심·눈 수평·눈 사이 거리로 정규화
  3. 입 안 색상 단서: 입 벌림 폭/높이/면적, 어두운 구강 비율, 혀(붉은색) 비율, 평균 명도·채도
- 음성은 80차 fbank(4프레임 묶음, 25 Hz)입니다. 세 흐름을 합쳐 BiLSTM 인코더(설정으로 Conformer 선택 가능) →
  CTC(50 Hz, +어텐션 디코더)로 한글 자모 단위(초성/중성/종성, 어휘 77개)를 출력하고, 보조 SNR 헤드가 잡음 수준(4구간)을
  함께 예측합니다. 학습 결과는 [10절](#10-학습-결과-이-데이터-2026-09-24)에 있습니다.
- 학습 때 모달리티 드롭아웃(av / 음성만 / 입술만)과 잡음 합성(babble·white·pink, −5~20 dB)을 사용하므로
  하나의 모델이 `av`, `audio`, `video` 세 모드를 모두 지원합니다.
- AI-Hub **「립리딩(입모양) 음성인식 데이터」 형식**을 따르는 데이터라면 그대로 전처리·학습할 수 있습니다.

상세 설계 계약서는 [`docs/SPEC.md`](docs/SPEC.md)에 있습니다.

---

## 1. 폴더 구조

```
avsr_kr/
  README.md, requirements.txt
  1_setup.bat ~ 5_evaluate_sync.bat  더블클릭 실행 파일: 설치 → 나눠서 전처리 → 합치기 → 화자-입술 매칭 학습 → 평가 (11·12절)
  configs/base.yaml          모든 하이퍼파라미터 (아래 7절)
  assets/face_landmarker.task  MediaPipe 얼굴 랜드마크 모델
  docs/SPEC.md               시스템 명세
  avsr/                      파이썬 패키지
    aihub.py        데이터셋 형식 처리 (tar/폴더 탐색, 라벨 파싱, bbox 순서 보정)
    text.py         한글 자모 토크나이저
    video_feats.py  랜드마크·입 크롭·스켈레톤·색상 단서
    audio_feats.py  음성 로딩(ffmpeg), fbank, 잡음 합성, DSP SNR 추정
    preprocess.py   (CLI) 데이터셋 → 특징 + 매니페스트
    dataset.py      학습용 Dataset / 배치 구성
    models/         모델 (frontends, encoder, fusion, decoder, avsr_model)
    utils.py        설정 로딩, 체크포인트, CER/WER 등
    train.py        (CLI) 학습
    evaluate.py     (CLI) 평가 + SNR 임계값 제안
    infer.py        (CLI) 영상 → 자막(.srt/.json)
  scripts/*.ps1     위 CLI를 감싼 PowerShell 스크립트
  tools/            보조 분석 스크립트 (data_report.py: 분할/화자/각도별 문장 수·시간, av_sync_check.py: 음성-입술 싱크 점검)
  tests/            합성 데이터로 도는 테스트
  work/             기본 작업 폴더 (전처리 결과, 체크포인트, 로그) — 4절
```

---

## 2. 데이터 형식 요약 (AI-Hub 립리딩 데이터)

분할 폴더(예: `2.Validation`, `1.Training`) 아래 구조:

```
<데이터 루트>\01.데이터\2.Validation\
    라벨링데이터\VL5.tar ...   ← 라벨(JSON)
    원천데이터\VS5.tar ...     ← 영상(mp4) + 음성(wav, 정면 A 각도만)
```

- tar 파일 그대로 두어도 되고, 이미 풀어 둔 폴더여도 됩니다(둘 다 자동 인식). 전처리는 tar 안의 영상을
  **한 개씩만** 임시 폴더로 꺼내 처리한 뒤 지웁니다(313 GB를 전부 풀 필요 없음).
- tar 내부 경로 예: `VS7/소음환경1/E(전문가)/F(여성)/F(여성)_1/lip_J_1_F_03_E220_A_001.mp4` (+ 같은 이름의 `.json` 라벨)
- 파일 이름: `lip_J_{그룹}_{성별 F|M}_{연령}_{화자ID}_{각도 A~I}_{세션 001~}`
  - 화자ID 예: `E220`(E=전문가), `C013`(C=일반인). 각도 A~I는 **같은 녹화 세션의 서로 다른 카메라**이며 A가 정면입니다.
  - 각도만 다른 파일들은 문장(Sentence_info)과 음성이 동일합니다.
- 영상: 1920×1080, H.264, 약 5분, 오디오 트랙(AAC 48 kHz) 포함.
  **mp4에 들어 있는 음성을 사용합니다**(별도 wav는 21 ms 어긋나 있음).
  라벨에는 30 fps라고 되어 있지만 일부 영상은 실제로 29.97 fps이므로 전처리는 실제 fps를 측정해 사용합니다.
- 라벨 JSON(요소 1개짜리 리스트):
  ```
  [{"Video_info": {"video_Name": "...mp4", "FPS": 30, "Resolution": "1920*1080", ...},
    "Audio_env": {"Noise": 1}, "Video_env": {"env": "indoor_light", "Angle": "A"},
    "Sentence_info": [{"ID": 1, "topic": "financial", "sentence_text": "최근에 ...",
                       "start_time": 0.938666, "end_time": 7.808}, ...],        ← 영상당 약 40문장
    "speaker_info": {"speaker_ID": "E220", "Gender": "F", "Age": 3, ...},
    "Bounding_box_info": {"Face_bounding_box": {"xtl_ytl_xbr_ybr": [[...], ...]},  ← 영상 프레임마다 1개
                          "Lip_bounding_box":  {"xtl_ytl_xbr_ybr": [[...], ...]}}}]
  ```
- **bbox 순서 주의:** 키 이름은 `xtl_ytl_xbr_ybr`이지만 실제 값은 **`[y_top, x_left, y_bottom, x_right]`** 입니다
  (1920×1080 픽셀 좌표). `avsr/aihub.py`가 화면 밖으로 벗어나는 좌표 수를 세어 순서를 자동 판별하므로,
  올바른 `[x, y, x, y]` 순서의 데이터가 들어와도 그대로 동작합니다.
- 문장 텍스트의 `X`는 알아들을 수 없거나 가려진 단어 표시입니다. 이런 문장은 학습·평가에서 제외됩니다(`has_unk`).
- 일부 영상은 라벨 시각이 0.1~0.3 초 이르게 기록되어 있어, 전처리가 문장 경계의 음성 에너지로 영상별
  시간 오프셋을 추정해 (개선이 확실할 때만) 보정합니다(`--no-align`으로 끌 수 있음).
- 화자와 기본 분할(화자 독립): **val = C313, test = E014·C159, train = 나머지 13명.**

| 화자 | 성별 | 소음환경 | | 화자 | 성별 | 소음환경 |
|---|---|---|---|---|---|---|
| C013, C044, C069, C159, C230 | F | 2 | | C084, C085, C272, C313, C332 | M | 1 |
| E220 | F | 1 | | C341, C464, C473, E014, E205 | M | 1 |

---

## 3. 설치

### 3.1 이 PC (이미 설치됨)

- Python 3.12: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`
  (PATH의 `python`은 Microsoft Store 바로가기라서 **사용하면 안 됩니다**. 항상 전체 경로를 쓰세요.)
- torch 2.11(cu128)·torchaudio·torchvision, opencv-python 5.0, mediapipe 1.0.1, numpy, soundfile, scipy, jiwer, tqdm,
  pyyaml, pandas, matplotlib, ffmpeg/ffprobe, GPU RTX 5080(bf16).

새 PowerShell 창에서는 먼저 아래를 실행해 두면 편합니다(ffmpeg가 PATH에 잡히도록 새로고침 포함).
`scripts\*.ps1` 스크립트는 이 작업을 스스로 하므로, 스크립트만 쓸 때는 생략해도 됩니다.

```powershell
$py = 'C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe'
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
cd C:\Users\user\Desktop\avsr_kr
```

### 3.2 다른 PC에 설치할 때

1. Python 3.12(64비트)와 ffmpeg(ffmpeg/ffprobe가 PATH에 있어야 함)를 설치합니다.
2. 패키지 설치(버전은 `requirements.txt`에 고정):
   ```powershell
   & $py -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   & $py -m pip install -r requirements.txt
   ```
3. MediaPipe 모델을 `assets\face_landmarker.task`로 저장합니다
   (배포 주소: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`).
4. 스크립트는 기본적으로 위 3.1의 python 경로를 씁니다. 다른 위치라면 `$env:AVSR_PYTHON = 'D:\...\python.exe'`로 지정하세요.

설치 확인(합성 데이터 테스트, 실제 데이터 불필요):

```powershell
& $py -m tests.test_text
& $py -m tests.test_video_feats
& $py -m tests.test_audio_dataset
& $py -m tests.test_models
& $py -m tests.test_train_smoke
& $py -m tests.test_infer
```

---

## 4. 사용법

모든 명령은 **프로젝트 루트에서** 실행합니다. 방법은 두 가지이며 결과는 같습니다.

- `scripts\*.ps1`: PATH 새로고침, 프로젝트 루트로 이동, UTF-8 출력 설정 후 모듈을 실행합니다. 인자는 그대로 전달됩니다.
- `& $py -m avsr.<모듈> ...`: 직접 실행.

> PowerShell 주의: 쉼표가 들어간 값은 **따옴표로 감싸야** 합니다(`--angles "A,C"`, `--set "eval.snr_sweep=[10,0]"`).
> 스크립트는 프로젝트 루트에서 실행되므로, 프로젝트 밖의 파일은 **절대 경로**로 지정하세요.

설정은 `configs/base.yaml`이 기본이며, 어떤 키든 `--set 섹션.키=값`으로 덮어쓸 수 있습니다(여러 번 사용 가능).

### 4.1 전처리 (데이터셋 → 특징)

```powershell
.\scripts\preprocess.ps1 --data-root "C:\Users\user\Downloads\009.립리딩(입모양) 음성인식 데이터"
# 동일:
& $py -m avsr.preprocess --data-root "C:\Users\user\Downloads\009.립리딩(입모양) 음성인식 데이터" --work-dir work --workers 14
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--data-root` | (필수) | `라벨링데이터`/`원천데이터`가 있는 폴더 또는 그 상위 폴더(여러 분할 폴더 자동 탐색) |
| `--work-dir` | `work` | 결과 폴더 |
| `--tmp-dir` | `<work>/tmp` | 영상 임시 추출 위치 |
| `--workers` | CPU 코어 수 − 6 | 동시에 처리할 영상 수(프로세스) |
| `--angles` | `all` | 처리할 각도, 예: `"A"`, `"A,C,E"` |
| `--speakers` | (전체) | 처리할 화자, 예: `"C313,E014"` |
| `--limit-videos` | 0(무제한) | 이번 실행에서 처리할 최대 영상 수 |
| `--order` | `a-first` | `a-first`: 모든 화자의 정면(A) 영상부터 처리, `natural`: 원래 순서 |
| `--decoder` | `auto` | 영상 디코딩: `nvdec`(NVIDIA 하드웨어), `cpu`, `auto`(nvdec 실패 시 cpu) |
| `--no-align` | 꺼짐 | 라벨 시간 오프셋 자동 보정 끄기 |
| `--keep-media` | 꺼짐 | 추출한 mp4를 지우지 않음 |
| `--model` | `assets/face_landmarker.task` | MediaPipe 모델 경로 |
| `--dry-run` | 꺼짐 | 처리할 목록만 출력(매체 색인 `work/media_index.json`은 만들어짐) |

- 영상 하나가 끝날 때마다 `work/manifests/<영상>.jsonl`과 `.done` 표시가 생깁니다. **중단 후 같은 명령을 다시 실행하면
  끝난 영상은 건너뛰고 이어서** 처리합니다. 실패한 영상도 재실행하면 다시 시도합니다.
- 처리 방식: 라벨 얼굴 박스로 얼굴을 잘라 320 px로 MediaPipe에 넣고 → 입술 40점, 스켈레톤, 색상 단서 계산 →
  얼굴 크기 기준 입 박스(최근 9프레임 중앙값으로 안정화) → 96×96 입 영상(H.264) 저장. 얼굴 검출에 실패한 프레임은
  라벨 입술 박스로 자르고 `valid=0`으로 표시합니다. 문장은 앞뒤로 최대 0.2 초(이웃 문장과의 간격 절반 이내) 여유를 둡니다.

### 4.2 학습

```powershell
.\scripts\train.ps1
# 동일:
& $py -m avsr.train --config configs/base.yaml
# 빠른 동작 확인(1 epoch, 64문장) — 체크포인트를 별도 폴더에 두어 본 학습과 섞이지 않게:
.\scripts\train.ps1 --set train.epochs=1 --set data.num_workers=2 --set train.ckpt_dir=work/checkpoints_smoke --limit 64
```
(`--limit` 실행의 체크포인트가 본 학습 폴더에 있으면 본 학습이 이어받지 않고 오류로 알려 줍니다.)

| 옵션 | 설명 |
|---|---|
| `--config` | 설정 파일(기본 `configs/base.yaml`) |
| `--set` | 설정 덮어쓰기(반복 가능) |
| `--limit N` | 학습/검증 문장을 N개만 사용(동작 확인용) |
| `--work-dir` | 작업 폴더 변경(체크포인트도 그 아래로 이동) |

- 매 epoch 끝에 검증 세트를 `av`/`audio`/`video`(깨끗한 음성) 및 `av`/`audio`(0 dB 잡음)로 디코딩해 CER/WER을 기록합니다.
- `work/checkpoints/last.pt`(매 epoch), `best.pt`(검증 CER(av) 최저), `epoch_XXX.pt`(최근 `train.keep_last`개)가 저장됩니다.
- `train.resume: auto`이면 `last.pt`에서 자동으로 이어서 학습합니다. **Ctrl+C 한 번**이면 현재 스텝을 마치고
  `last.pt`를 저장한 뒤 종료합니다(두 번 누르면 즉시 중단). 처음부터 다시 하려면 `--set train.resume=none`.
- 로그: 콘솔, `work/logs/train.log`, `work/logs/metrics.jsonl`.

### 4.3 평가 (조건별 CER/WER + SNR 임계값 제안)

```powershell
.\scripts\evaluate.ps1 --ckpt work\checkpoints\best.pt --split test
# 동일:
& $py -m avsr.evaluate --config configs/base.yaml --ckpt work/checkpoints/best.pt --split test
# 검증 세트 300문장, 더 촘촘한 SNR 스윕:
& $py -m avsr.evaluate --ckpt work/checkpoints/best.pt --split val --max-utts 300 --set "eval.snr_sweep=[20,15,10,5,2,0,-2,-5]"
```

| 옵션 | 설명 |
|---|---|
| `--ckpt` | 평가할 체크포인트(필수) |
| `--split` | `val` 또는 `test`(기본 `test`) |
| `--config`, `--set` | 설정 파일/덮어쓰기. 모델 구조와 입력 특징(`model`, `video`, `audio`)은 체크포인트의 것을 사용 |
| `--max-utts N` | 고르게 뽑은 N문장만 평가 |
| `--work-dir` | 작업 폴더 변경 |

- 각 문장의 음성에 `eval.noise_kind` 잡음을 `eval.snr_sweep`의 각 SNR로 섞어 `av`/`audio`를 디코딩하고,
  `video`는(음성을 보지 않으므로) 한 번만 디코딩합니다. 표를 출력하고 `work/eval/<split>_results.json`에 저장합니다.
- CER은 공백을 뺀 글자 단위, WER은 띄어쓰기 단위입니다.

### 4.4 추론 (영상 → 자막)

```powershell
.\scripts\infer.ps1 --ckpt work\checkpoints\best.pt --video D:\videos\talk.mp4
# 동일:
& $py -m avsr.infer --config configs/base.yaml --ckpt work/checkpoints/best.pt --video D:\videos\talk.mp4 --out D:\videos\talk.srt
# 입술만 사용(강제), 임계값 변경, 다른 음성 파일 사용:
.\scripts\infer.ps1 --ckpt work\checkpoints\best.pt --video D:\videos\talk.mp4 --mode video
.\scripts\infer.ps1 --ckpt work\checkpoints\best.pt --video D:\videos\talk.mp4 --snr-threshold 8
.\scripts\infer.ps1 --ckpt work\checkpoints\best.pt --video D:\videos\talk.mp4 --audio D:\videos\talk_mic.wav
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--ckpt` | (필수) | 체크포인트 |
| `--video` | (필수) | 입력 영상(OpenCV/ffmpeg가 읽을 수 있는 형식, 한글 경로 가능) |
| `--audio` | 영상의 오디오 | 다른 음성 파일 사용(읽을 수 없는 파일이면 오류로 중단) |
| `--out` | `<영상>.srt` | 출력 .srt 경로 또는 폴더(아직 없는 폴더는 끝에 `/`를 붙여 지정, 예: `"D:\my subs/"`). `.json`은 .srt 옆에 같은 이름으로 생성 |
| `--mode` | `auto` | `auto`(SNR로 자동 선택) / `av` / `audio` / `video` 강제 |
| `--snr-threshold` | `infer.snr_threshold` | 임계값(dB) 덮어쓰기 |
| `--vad` | `auto` | 구간 분할: `auto`, `energy`(음성 에너지), `visual`(입 움직임) |
| `--device` | GPU 있으면 `cuda` | `cuda` 또는 `cpu` |
| `--config`, `--set` | `configs/base.yaml` | 추론 설정(`infer.*`, `eval.decode`) 등. 모델 구조·입력 특징은 체크포인트의 것을 사용 |
| `--landmarker` | `assets/face_landmarker.task` | MediaPipe 모델 경로 |
| `--no-progress` | | 진행 막대 끄기 |

처리 순서:

1. 음성을 16 kHz 모노로 읽습니다(오디오 트랙이 없거나 무음이면 입술만으로 진행).
2. 영상 프레임을 **시간 기준으로 30 fps로 맞춥니다**(25/29.97/60 fps나 가변 프레임레이트도 각 30 fps 시점에 가장 가까운
   프레임 사용). 프레임을 긴 변 640 px 이하로 줄여 MediaPipe로 얼굴 랜드마크를 찾고, 원래 해상도 좌표로 되돌려
   전처리와 **같은 방식**으로 스켈레톤·색상 단서·96×96 입 크롭(9프레임 중앙값 박스 안정화)을 만듭니다.
   얼굴을 못 찾은 프레임은 직전 박스를 유지하고 `valid=0`이 됩니다.
3. 전체 음성의 SNR을 추정해 임계값 이상이면 **음성 에너지 VAD**, 미만이면 **입 움직임 VAD**(안쪽 입술 높이의 0.5 초
   이동 표준편차)로 발화 구간을 나눕니다. 에너지 VAD의 기준은 잡음 바닥(프레임 음량 하위 10 %) + 6 dB이며, 잡음이 커서
   음량 범위가 좁을 때는 작은 음절을 놓치지 않도록 여유를 최소 2 dB까지 줄입니다. 입술이 영상 프레임의 20 % 미만에서만
   보이면 입 움직임으로 나눌 수 없으므로 SNR과 관계없이 에너지 VAD를 씁니다. 짧은 쉼으로 끊긴 구간은 합치고,
   `segment_max_sec`보다 긴 구간은 가장 조용한(입이 가장 덜 움직이는) 지점에서 나누며, 너무 짧은 구간은
   `segment_min_sec`까지 늘립니다. 최소 1개 구간은 항상 만듭니다.
4. 구간마다 SNR을 다시 추정하고(`infer.snr_estimator`), `SNR ≥ 임계값`이면 `av`, 아니면 `video`로 디코딩합니다.
   단, `auto`에서 구간 프레임의 20 % 미만에서만 입술이 보이면(옆모습·가림) 입술 인식이 불가능하므로 `audio`를 씁니다.
5. `.srt`(번호, `HH:MM:SS,mmm --> ...`, 텍스트)와 `.json`을 쓰고, 콘솔에 구간/모드/SNR/텍스트 표를 출력합니다.

`.json` 내용: 상단에 실행 정보(`video`, `audio`, `ckpt`, `mode`, `snr_threshold`, `snr_estimator`, `decode`, `vad`,
`global_snr_db`, `video_src_fps`, `landmark_valid_ratio`, `modes_used`, 처리 시간 등), `segments`에 구간별
`start`, `end`, `text`, `mode`, `snr_est`(판단에 쓴 SNR), `snr_dsp`, `snr_model`, `visual_ratio`(입술 검출 비율).

파이썬에서 호출:

```python
from avsr.infer import load_model_bundle, transcribe_video, write_srt
from avsr.utils import load_config

segs = transcribe_video("D:/videos/talk.mp4", "work/checkpoints/best.pt", "configs/base.yaml")
for s in segs:
    print(s.start, s.end, s.mode, s.snr_est, s.text)
write_srt("D:/videos/talk.srt", segs)

# 여러 영상: 모델을 한 번만 읽기
bundle = load_model_bundle("work/checkpoints/best.pt", load_config("configs/base.yaml"))
for video in ["a.mp4", "b.mp4"]:
    segs = transcribe_video(video, model_bundle=bundle, mode="auto")
```

---

## 5. 산출물 구조 (`work/`)

```
work/
  media_index.json                  원천 tar 안의 mp4/wav 위치 색인(캐시)
  tmp/                              전처리 중 임시로 꺼낸 영상(처리 후 삭제)
  feats/<영상>/<영상>__<문장ID 3자리>.mp4   입 크롭 영상 96×96, 원본 fps
  feats/<영상>/<영상>__<문장ID 3자리>.npz   lm [T,40,2], cue [T,8], valid [T], audio(16 kHz int16), fps, sr
  manifests/<영상>.jsonl            문장별 1줄: utt_id, 화자·성별·각도·세션, start/end/duration, n_frames, fps,
                                    text(정규화), text_raw, has_unk, lm_valid_ratio, mouth_mp4, npz 경로 등
  manifests/<영상>.done             영상 처리 완료 표시(재실행 시 건너뜀)
  preprocess_log.jsonl              영상별 처리 요약(시간, 랜드마크 성공률, 시간 오프셋, 오류)
  checkpoints/last.pt, best.pt, epoch_XXX.pt
  logs/train.log, logs/metrics.jsonl, logs/evaluate.log
  eval/val_results.json, eval/test_results.json
```

- `cue`의 8개 값: `[입 너비, 안쪽 입술 높이, 바깥 입술 높이, 안쪽 입술 면적, 어두운 비율, 붉은 비율, 평균 명도, 평균 채도]`
  (길이는 눈 사이 거리로 정규화).
- 학습/검증/테스트 분할은 매니페스트에 저장되지 않고 **읽을 때 설정으로** 정해집니다. 따라서 분할을 바꿔도 전처리를 다시 할 필요가 없습니다.

---

## 6. 결과 해석과 SNR 임계값 정하기

`avsr.evaluate` 출력 예(형식):

```
     SNR |    av CER/WER   |  audio CER/WER  |  video CER/WER  | AV beats audio
   clean |  ...            |  ...            |  ...            | yes (+x.x pt)
   20 dB |  ...
    ...
   -5 dB |  ...
SNR threshold: suggested infer.snr_threshold = 5 dB (highest SNR where CER(av) > CER(video)).
```

- `video` 열은 음성을 쓰지 않으므로 모든 행에서 같습니다(입술만의 성능).
- `av`는 SNR이 낮아질수록 나빠집니다. **`av`의 CER이 `video`보다 나빠지기 시작하는 가장 높은 SNR**이 제안 임계값이며,
  `work/eval/<split>_results.json`의 `suggested_snr_threshold`에도 기록됩니다.
  - `video_never_better`: 시험한 범위에서 입술만 쓰는 편이 한 번도 낫지 않음 → 임계값을 가장 낮은 SNR보다 더 낮게.
  - `video_always_better`: 가장 높은 SNR에서도 입술이 나음 → 음성 쪽 학습이 부족하거나 음성 품질 문제를 의심.
- `AV beats audio`는 입술을 더했을 때 음성만보다 좋아지는지(융합 효과)를 보여 줍니다.

임계값 정하는 순서:

1. **검증 세트**(test가 아닌 `--split val`)에서 실제 사용 환경과 비슷한 잡음 종류로 촘촘하게 스윕합니다.
   ```powershell
   & $py -m avsr.evaluate --ckpt work/checkpoints/best.pt --split val --set "eval.snr_sweep=[20,15,10,7,5,3,0,-3,-5]" --set eval.noise_kind=babble
   ```
2. **추론에 넣을 값은 "추정치 단위" 임계값입니다.** 평가의 SNR은 **실제로 섞은 SNR**이지만 추론은 SNR을 **추정**해서
   비교합니다. 그래서 평가는 각 SNR 단계에서 추론과 똑같은 두 추정치(DSP, 모델 SNR 헤드)를 함께 계산해 표로 보여 주고,
   실제-SNR 임계값을 추정치 단위로 바꾼 값을 출력합니다:
   ```
   SNR estimates inference would see (median dsp / model / hybrid):
      clean:  38.2 / 25.0 / 31.6
      20   :  ...
   -> in estimator units (what infer compares): hybrid 9.8, dsp 11.2, model 10.0  (e.g. infer --snr-threshold 9.8 ...)
   ```
   이 값(`work/eval/<split>_results.json`의 `suggested_snr_threshold.infer_threshold.hybrid`)을
   `configs/base.yaml`의 `infer.snr_threshold`에 적거나 추론 때 `--snr-threshold`로 줍니다. (실제 임계 SNR 단계와 그보다
   한 단계 깨끗한 단계의 추정치 중간값입니다.)
3. 추정 방식(`infer.snr_estimator`):
   - `dsp`: 프레임 에너지 통계로 계산(하위 10 % = 잡음, 상위 30 % 평균 = 음성). 잡음이 심할수록 실제보다 높게 나오는 경향이
     있습니다. 예) 이 데이터의 한 영상에 white 잡음을 섞었을 때 실제 −5/0/5/10 dB → 추정 약 2.5/5.1/8.8/12.9 dB.
     위 2번의 변환이 이 치우침을 보정합니다.
   - `model`: 모델 SNR 헤드의 구간(≥20, 10~20, 0~10, <0 dB)을 대표값 25/15/5/−5 dB로 변환(거칠지만 실제 SNR 기준으로 학습됨).
   - `hybrid`(기본): 두 값의 평균.
   실제 사용 환경의 잡음 종류(`eval.noise_kind`)로 평가해야 변환이 정확합니다. 모드가 기대와 다르면 `--mode av|video`로
   강제해 비교해 볼 수 있고, 추론 `.json`의 `snr_dsp`, `snr_model`, `snr_est`로 실제 추정치를 확인할 수 있습니다.

---

## 7. 설정 키 (`configs/base.yaml`)

| 키 | 기본값 | 의미 |
|---|---|---|
| `work_dir` | `work` | 작업 폴더 |
| `seed` | 1234 | 난수 시드 |
| `split.val_speakers` / `split.test_speakers` | `[C313]` / `[E014, C159]` | 검증/테스트 화자(이 화자들은 학습에 쓰이지 않음) |
| `split.train_split_dirs` | `[2.Validation, 1.Training]` | 나머지 화자 중 학습에 쓸 분할 폴더 이름 |
| `data.angles` | `all` | 사용할 각도(`all` 또는 `[A, C]`) |
| `data.min_duration` / `max_duration` | 0.5 / 16.0 | 사용할 문장 길이(초) |
| `data.num_workers` | 10 | DataLoader 프로세스 수 |
| `video.channels` | 1 | 1 = 흑백, 3 = RGB |
| `video.size` / `crop` | 96 / 88 | 저장 크기 / 모델 입력 크기(학습 시 무작위, 평가·추론 시 중앙 자르기) |
| `video.fps_in` / `fps_out` | 30 / 25 | 입 영상 fps / 모델 fps |
| `video.norm` | `utterance` | 입 영상 픽셀 정규화. `utterance` = 문장(구간)마다 평균 0·표준편차 1로 맞춤. 이 데이터는 전문가(E) 화자 영상이 일반인(C) 화자보다 약 2배 밝아서, 조명 차이가 화자 단서가 되지 않도록 기본으로 사용. `global` = 아래 상수로 정규화 |
| `video.cue_norm` | `utterance` | 8개 cue(입 크기·색 비율)도 문장마다 표준화(조명·입 크기 차이 제거, 움직임만 남김). `none` = 원래 값 |
| `video.mean` / `std` | 0.4355 / 0.1897 | `video.norm=global`일 때의 픽셀 정규화 상수(이 데이터에서 측정) |
| `video.flip_prob`, `time_mask_prob`, `time_mask_max` | 0.5, 0.3, 8 | 학습용 영상 증강 |
| `audio.sr`, `n_mels`, `stack` | 16000, 80, 4 | 샘플레이트, 멜 필터 수, 프레임 묶음(100 Hz → 25 Hz) |
| `audio.noise_prob`, `noise_kinds`, `snr_min`, `snr_max` | 0.6, `[babble, white, pink]`, −5, 20 | 학습 잡음 합성 |
| `audio.specaug` | | SpecAugment(주파수/시간 마스크) |
| `model.encoder.type` | `blstm` | 인코더 종류. `blstm`(4층, 방향당 320) = 기본. `conformer`도 선택 가능하지만 이 데이터(약 10시간)로 처음부터 학습하면 CTC가 초기 정체 구간(모든 입력에 같은 출력)을 벗어나지 못했음(아래 FAQ) |
| `model.ctc_upsample` | 2 | 인코더 한 프레임(40 ms)을 CTC 2칸으로 나눔 → CTC가 50 Hz로 동작. 한글 자모는 초당 약 14개라 25 Hz에서는 너무 빽빽함 |
| `model.pos_enc` | `none` | 인코더 입력의 절대 위치 인코딩(`sinusoidal`)을 쓸지 여부 |
| `model.*` (그 외) | | d_model 256, 디코더 6층(약 30.7 M 파라미터), `fusion`(모달리티 드롭아웃 확률 0.5/0.25/0.25), `ctc_weight` 0.7, `snr_head_weight` 0.1, `label_smoothing` 0.1 |
| `train.max_frames_per_batch` | 3200 | 배치 크기(25 fps 프레임 수 합). GPU 메모리가 부족하면 줄이세요 |
| `train.epochs`, `lr`, `warmup_steps`, `weight_decay`, `grad_clip` | 30, 1e-3, 1000, 0.01, 5.0 | 최적화(한 에포크에 같은 음성이 카메라 각도 5개만큼 반복되므로 30 에포크) |
| `train.fusion_curriculum` | 1~3 에포크 음성만 → 4~6 에포크 음성+영상/음성 → 7~ `model.fusion` | 모달리티 커리큘럼. 처음부터 입술-only 샘플을 섞으면 출력이 입력과 무관한 문장으로 붕괴함 |
| `train.aug_start_epoch` | 2 | 이 에포크 전까지는 잡음 합성·SpecAugment 없이(정렬을 먼저 학습) |
| `train.cuda_mem_fraction` | 0.8 | PyTorch가 쓸 GPU 메모리 비율 상한. 없으면 캐시가 16 GB를 넘어 Windows가 시스템 RAM으로 넘겨 학습이 약 2.5배 느려짐 |
| `train.amp` | `bf16` | 혼합 정밀도(`none`이면 fp32) |
| `train.ckpt_dir`, `keep_last`, `resume` | `work/checkpoints`, 3, `auto` | 체크포인트 위치/보관 수/이어하기(`auto`, `none`, 또는 파일 경로) |
| `train.log_every`, `eval_every_epoch` | 50, 1 | 로그/검증 주기 |
| `eval.decode` | `ctc_greedy` | 디코딩 방식(`ctc_greedy` 또는 `attn_greedy`), 추론에도 사용 |
| `eval.snr_sweep`, `noise_kind`, `conditions` | `[20,10,5,0,-5]`, `babble`, `[av, audio, video]` | 평가 스윕 |
| `eval.max_val_utts` | 400 | 학습 중 검증에 쓸 최대 문장 수 |
| `infer.snr_threshold` | 5.0 | **추정** SNR(dB)이 이 값 이상이면 `av`, 미만이면 `video`. 학습 후 평가가 출력하는 "추정치 단위" 값으로 바꾸세요(6절) |
| `infer.snr_estimator` | `hybrid` | `dsp` / `model` / `hybrid` (6절) |
| `infer.segment_max_sec` / `segment_min_sec` | 12.0 / 1.0 | 자막 구간 최대/최소 길이(초). 학습 문장이 최대 16초라 12초까지 한 구간으로 둠 |
| `infer.chunk_pad_sec` | 0.2 | 구간 앞뒤 여유(초, 이웃 구간과 겹치지 않게) |

추론 시 `model`, `video`, `audio` 섹션은 항상 **체크포인트에 저장된 값**을 씁니다(가중치가 그 구조에만 맞기 때문).

---

## 8. 새 데이터셋(같은 라벨 형식)에 적용하기

1. 새 데이터가 `라벨링데이터`/`원천데이터` 구조(tar 또는 풀린 폴더)인지 확인합니다. 각 라벨 JSON의
   `video_Name`과 같은 이름의 mp4가 원천데이터 어딘가에 있으면 됩니다(하위 폴더 구조는 상관없음).
2. 전처리합니다. 같은 `work` 폴더에 넣으면 기존 결과에 **추가**됩니다(매니페스트는 영상별 파일이므로 합치는 과정 불필요).
   ```powershell
   .\scripts\preprocess.ps1 --data-root "D:\new_data\009.립리딩(입모양) 음성인식 데이터" --dry-run   # 목록 확인
   .\scripts\preprocess.ps1 --data-root "D:\new_data\009.립리딩(입모양) 음성인식 데이터"
   ```
   섞지 않으려면 `--work-dir D:\work_new`로 따로 만들고, 학습/평가 때도 `--work-dir D:\work_new`를 줍니다.
3. 분할을 정합니다. 분할은 매니페스트의 `speaker`와 `split_dir`(데이터가 있던 분할 폴더 이름, 예 `1.Training`)로 결정됩니다.
   - `1.Training`을 추가하고 `2.Validation`은 검증/테스트에만 쓰려면:
     ```powershell
     .\scripts\train.ps1 --set "split.train_split_dirs=[1.Training]" --set "split.val_speakers=[C313]" --set "split.test_speakers=[E014,C159]"
     ```
     (이때 `2.Validation`의 나머지 화자는 사용되지 않습니다. 모두 쓰려면 검증/테스트 화자 목록에 넣으세요.)
   - 화자 ID가 다른 데이터라면 `split.val_speakers`/`test_speakers`를 그 데이터의 화자로 바꿉니다. 화자 목록과
     분할 결과(분할/화자/각도별 문장 수·시간)는 `& $py tools/data_report.py --config configs/base.yaml`로 확인할 수 있습니다.
   - 음성과 입술의 시간 정렬이 의심되면 `& $py tools/av_sync_check.py`로 영상별 지연(프레임)을 점검하세요.
   - 자주 쓰는 조합은 `configs/base.yaml`을 복사한 파일(예: `configs/full.yaml`)에 적고 `--config configs/full.yaml`로 사용하세요.
4. fps가 30이 아니거나(29.97 등) 라벨 시각이 조금 어긋난 영상도 전처리가 처리합니다. 각도를 제한하려면
   `--angles "A"`(전처리) 또는 `--set "data.angles=[A]"`(학습)을 씁니다.
5. 학습 → 평가(6절) → 임계값 설정 → 추론 순서로 진행합니다.

라벨이 없는 일반 영상은 전처리·학습 없이 **추론(4.4절)만** 하면 됩니다. 정면에 가깝고 입이 잘 보일수록 입술 인식이 정확합니다.

---

## 9. 문제 해결 (FAQ)

**`python`을 실행하면 Microsoft Store가 열려요.**
PATH의 `python`은 Store 바로가기입니다. `$py`(3.1절)의 전체 경로나 `scripts\*.ps1`을 쓰세요.

**`ffmpeg`를 찾을 수 없다는 오류가 나요.**
새 창에서 PATH가 갱신되지 않은 경우입니다. 3.1절의 `$env:Path = ...` 줄을 실행하거나 스크립트를 쓰세요
(`avsr.audio_feats`도 레지스트리 PATH에서 ffmpeg를 찾아봅니다). 설치 여부는 `ffmpeg -version`으로 확인합니다.

**콘솔에서 한글이 깨져요.**
스크립트는 UTF-8 출력으로 설정합니다. 직접 실행하면서 파이프/리디렉션을 쓸 때는 먼저
`$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding=[Text.Encoding]::UTF8`를 실행하세요.
`.srt`(줄바꿈 CRLF)와 `.json`은 항상 UTF-8(BOM 없음)입니다. 자막이 깨져 보이면 플레이어의 자막 인코딩을 UTF-8로 지정하세요.

**`--angles A,C`나 `--set eval.snr_sweep=[10,0]`이 이상하게 동작해요.**
PowerShell이 쉼표를 배열로 해석합니다. `--angles "A,C"`, `--set "eval.snr_sweep=[10,0]"`처럼 따옴표로 감싸세요.

**`invalid output path ... contains one of <>:"|?*` 오류가 나요.**
Windows PowerShell 5.1에서는 따옴표로 감싼 경로가 `\`로 끝나면(`--out "D:\my subs\"`) 마지막 `\`가 닫는 따옴표를
무효로 만들어 뒤의 인자까지 경로에 붙습니다. 끝의 `\`를 빼거나 `/`로 바꾸세요(`--out "D:\my subs/"`).
`--data-root` 등 다른 경로 인자도 마찬가지입니다.

**스크립트에서 상대 경로 파일을 못 찾아요.**
스크립트는 프로젝트 루트로 이동한 뒤 실행하므로 상대 경로는 프로젝트 루트 기준입니다. 절대 경로를 쓰세요.

**전처리가 중간에 멈췄어요 / 일부 영상이 실패했어요.**
같은 명령을 다시 실행하면 이어서 처리하고 실패한 영상을 재시도합니다. 실패 원인은 `work/preprocess_log.jsonl`의
`error`에 있습니다. NVDEC 관련 오류가 반복되면 `--decoder cpu`를 쓰세요. 아무것도 실행 중이지 않을 때라면
`work/tmp`의 남은 파일은 지워도 됩니다.

**학습 중 GPU 메모리 부족(CUDA out of memory).**
`--set train.max_frames_per_batch=2000`처럼 배치를 줄이세요. Windows에서 DataLoader 문제가 의심되면
`--set data.num_workers=0`으로 확인할 수 있습니다.

**체크포인트와 설정이 맞지 않는다는 오류.**
다른 `model` 설정으로 만든 `last.pt`에서 이어 하려는 경우입니다. 다른 `--work-dir`/`train.ckpt_dir`을 쓰거나
`--set train.resume=none`으로 새로 시작하세요. 평가·추론은 모델 구조를 체크포인트에서 읽으므로 신경 쓸 필요가 없습니다.

**추론 로그에 "lips were found on only X% of the frames"가 나와요.**
얼굴이 너무 작거나, 옆모습이거나, 가려진 경우입니다. 이런 구간은 `auto` 모드에서 입술 대신 음성(`audio`)으로 인식됩니다.
얼굴이 크게 나오는 정면 영상일수록 좋습니다.

**자막 구간이 너무 길거나 짧아요 / 문장 중간에서 끊겨요.**
`--set infer.segment_max_sec=12`(학습 문장은 최대 16초), `--set infer.segment_min_sec=1.5`, `--set infer.chunk_pad_sec=0.3`
등으로 조정하세요. 구간 나누기를 고정하려면 `--vad energy` 또는 `--vad visual`을 씁니다. 긴 문장은 최대 길이를
넘지 않도록 가장 조용한 지점에서 나뉘므로 문장 중간에서 끊길 수 있습니다.

**영상에 소리가 없어요 / 소리를 따로 녹음했어요.**
오디오 트랙이 없으면 자동으로 입술만 사용합니다. 따로 녹음한 음성은 `--audio 파일`로 지정하세요
(영상과 시작 시점이 맞아야 합니다).

**25 fps, 60 fps, 가변 프레임레이트(휴대폰) 영상도 되나요?**
네. 프레임 시각(타임스탬프) 기준으로 30 fps에 맞춘 뒤 처리합니다.

**`--mode`는 언제 쓰나요?**
`auto`가 기본입니다. `av`/`audio`/`video`는 비교·확인용으로 모든 구간을 한 모드로 강제합니다.

**학습 초반 CER이 90 % 이상에서 안 내려가고, 모든 입력에 같은 문장이 나와요.**
CTC의 초기 정체(모든 프레임을 blank/사전 분포로 예측) 상태입니다. 이 프로젝트에서 확인된 원인과 기본 설정의 대책:
1. 인코더: 약 10시간 데이터로 처음부터 학습할 때 Conformer는 3천 스텝 넘게 정체를 못 벗어났고 BiLSTM은 수백 스텝 만에
   벗어났습니다 → `model.encoder.type: blstm`(기본).
2. 처음부터 입술-only 샘플을 섞으면 붕괴 → `train.fusion_curriculum`으로 음성부터 학습(기본).
3. 한글 자모는 25 Hz에서 너무 빽빽함 → `model.ctc_upsample: 2`(기본).
진단 도구: `tools/overfit_check.py`(몇 문장을 외울 수 있는지 = 코드 결함 여부), `tools/lstm_ctc_probe.py`
(간단한 BiLSTM-CTC로 데이터 자체가 학습 가능한지), `tools/ctc_trend.py`(로그의 CTC 손실 추이).

**학습 속도가 도중에 2~3배 느려졌어요 (Windows).**
작업 관리자의 "공유 GPU 메모리" 사용량이 크게 늘었다면 PyTorch 캐시가 전용 메모리를 넘어 시스템 RAM으로 넘어간 것입니다.
`train.cuda_mem_fraction`(기본 0.8)을 낮추거나 다른 GPU 프로그램을 종료하세요. 학습 중에 다른 GPU 작업(평가 등)을 같이
돌리면 같은 현상이 생깁니다.

**진행 상황을 한눈에 보고 싶어요.**
`& $py tools/train_status.py` — 에포크별 검증 CER(깨끗한 음성의 av/audio/video, 0 dB 잡음의 av/audio)과 "입술 이득"
(잡음에서 음성만 대비 음성+입술이 좋아진 정도)을 표로 보여 줍니다. `& $py tools/data_report.py`는 전처리된 데이터 요약입니다.

---

## 10. 학습 결과 (이 데이터, 2026-09-24)

**데이터:** `2.Validation` 폴더만 사용(960개 영상, 화자 16명, 문장 41,350개). 학습 13명 31,835문장(각도 5개 포함 48.8시간,
고유 음성 9.8시간), 검증 C313, 테스트 E014·C159. 전처리 109분(RTX 5080 NVDEC + CPU 14프로세스), 랜드마크 검출률 99.97 %.
**모델:** `work/checkpoints/best.pt`(30.7 M 파라미터, BiLSTM 인코더, 30 에포크 중 26 에포크, 학습 약 3시간).
수치는 CER(글자 오류율, 공백 제외)이며 babble 잡음(다른 화자 음성)을 섞은 결과입니다.

검증 화자 C313(학습에 없는 화자, 1,200문장):

| 잡음 | av (음성+입술) | audio (음성만) | video (입술만) | 입술 이득 |
|---|---|---|---|---|
| 깨끗함 | 15.3 % | 18.9 % | 83.6 % | +3.6 pt |
| 10 dB | 31.9 % | 42.4 % | 83.6 % | +10.5 pt |
| 5 dB | 52.5 % | 66.6 % | 83.6 % | +14.1 pt |
| 0 dB | 74.9 % | 86.5 % | 83.6 % | +11.6 pt |
| −5 dB | 88.6 % | 93.6 % | 83.6 % | +5.0 pt |

테스트 화자(깨끗한 음성): C159 av 26.7 % / audio 33.6 % / video 77.5 %, E014 av 56.8 % / audio 56.6 % / video 88.9 %.
E014는 녹음 SNR이 매우 높고(약 38 dB, 다른 화자 11~14 dB) 말이 가장 빠르며 5개 각도 중 4개가 측면이라 가장 어렵습니다.

- **입술은 모든 잡음 수준에서 도움이 됩니다**(av가 항상 audio보다 좋음). 그래서 기본 동작은 "항상 av로 융합",
  잡음이 극심해 음성이 오히려 방해가 될 때(약 −2~−5 dB 이하)만 입술 전용(`video`)으로 전환합니다.
  이 경계를 추정치 단위로 옮긴 값 **6.1**이 `infer.snr_threshold`에 들어 있습니다(검증 화자로 보정).
- 데모(테스트 화자 C159, 앞 25초 깨끗 / 뒤 −5 dB babble, `work/demo/`): 자동 전환 결과 앞부분 av 40.0 %,
  뒷부분 video 77.0 %로 각 구간에서 가장 좋은 모드를 골랐습니다(뒷부분 강제 av 86.9 %, 강제 audio 121.3 %).
  ```powershell
  .\scripts\infer.ps1 --ckpt work\checkpoints\best.pt --video work\demo\lip_J_2_F_04_C159_A_001_60s.mp4 `
      --audio work\demo\lip_J_2_F_04_C159_A_001_60s_clean-then-babble-5dB.wav --out work\demo\out.srt
  ```

**한계와 다음 단계(효과가 큰 순서):**
1. **데이터 양:** 현재는 Validation 분할(화자 16명)뿐입니다. AI-Hub의 `1.Training` 분할을 같은 폴더 구조로 받아
   전처리하면(8절) 화자·시간이 크게 늘어 음성·입술 모두 크게 좋아집니다. 특히 입술 전용(현재 CER 약 84 %)은
   데이터 양에 가장 민감합니다.
2. **사전학습 음성 인코더:** torchaudio의 `WAV2VEC2_XLSR_300M`(한국어 포함 128개 언어 사전학습, 이미 내려받아 둠)을
   음성 프런트엔드로 쓰면 적은 데이터에서도 음성 인식이 크게 좋아집니다.
3. **입술 학습 보강:** 음성 모델의 CTC 출력을 선생으로 입술 인코더를 증류 학습(ASR→VSR distillation)하면 입술 전용
   성능을 끌어올릴 수 있습니다.
4. **SNR 추정기:** 현재 추정치는 0 dB 이하에서 값이 잘 구분되지 않습니다(0 / −2 / −5 dB가 6.5 / 6.2 / 6.1).
   SNR 헤드를 4구간 분류 대신 회귀로 바꾸면 전환 판단이 더 안정적입니다.
5. 실제 서비스 환경의 잡음(카페, 거리 등)으로 `eval.noise_kind`를 바꿔 임계값을 다시 보정하세요.

---

## 11. 화자-입술 매칭 모델

### 11.1 무엇을 하나요?

영상에 여러 사람이 나오고, 음성 팀이 섞인 소리를 **사람별 음성(분리된 음성)** 으로 나눠 각각 자막을 만든다고 합시다.
이 모델은 **"분리된 음성 하나하나가 화면의 어느 얼굴에서 나온 소리인가"** 를 정해서, 그 음성의 자막을 그 사람에게
붙일 수 있게 합니다.

- 입술로 단어를 읽는 것이 아니라 **입 움직임과 소리가 같은 박자로 움직이는지(싱크)** 만 봅니다. 그래서 입술만으로
  글자를 읽는 것(10절, CER 약 84 %)보다 훨씬 쉬운 문제입니다.
- 화면의 어떤 얼굴과도 충분히 맞지 않는 음성은 **화면 밖 화자(off-screen)** 로 판단합니다. 그 기준값이
  `offscreen_threshold`입니다(11.4).
- 구조: 입 영상 + 입술 스켈레톤·색상 단서 → 영상 임베딩, 음성 fbank → 음성 임베딩을 **40 ms 프레임마다** 만들고,
  같은 시각끼리의 코사인 유사도를 창(예: 1초) 동안 평균한 값이 "이 얼굴–이 음성" 점수입니다. 시작 가중치는 AVSR
  체크포인트(`work/checkpoints/best.pt`)의 입·음성 프런트엔드입니다(`sync.init_from`).
- 학습: 1초 창 64개로 이루어진 배치에서 "자기 음성 고르기"를 배웁니다. 같은 문장을 다른 카메라 각도로 찍은 창은 음성이
  같으므로 오답으로 쓰지 않고, **같은 음성을 0.2~0.6초 밀어 놓은 것**을 어려운 오답으로 넣습니다(내용이 같아도 시각이
  틀리면 오답). 음성에는 다른 화자 목소리(분리가 불완전한 상황, 5~30 dB 아래)와 잡음을 섞어 학습합니다.
- 데이터: **이미 전처리된 `work` 폴더를 그대로 씁니다**(새로 추출할 것 없음). 화자 분할은 AVSR과 같습니다
  (검증 C313, 테스트 E014·C159, 나머지 13명 학습).
- 설정 파일: `configs/sync.yaml`(단독 설정. `sync:` 섹션에 창 길이, 모델, 증강, 평가 설정).

### 11.2 학습 (메인 PC)

**더블클릭:** `4_train_sync.bat` → Enter.

```powershell
.\scripts\train_sync.ps1                  # configs/sync.yaml, 중단된 곳부터 자동으로 이어서 학습
# 동일:
& $py -m avsr.sync.train --config configs/sync.yaml
# 빠른 동작 확인(1 에포크, 256문장) - 체크포인트를 별도 폴더에:
.\scripts\train_sync.ps1 --set train.epochs=1 --set data.num_workers=2 --set train.ckpt_dir=work/checkpoints_sync_smoke --limit 256
```

- 결과: `work/checkpoints_sync/last.pt`(매 에포크), `best.pt`(검증 화자에서 1초 창 4지선다 정확도가 가장 높은 것),
  `epoch_XXX.pt`(최근 3개). 로그: `work/logs/sync_train.log`, `work/logs/sync_metrics.jsonl`.
- **Ctrl+C 한 번**이면 현재 스텝을 마치고 저장한 뒤 끝납니다(창에 "일괄 작업을 끝내시겠습니까 (Y/N)?"가 나오면 `N`).
  `4_train_sync.bat`를 다시 실행하면 이어서 학습합니다. 처음부터 하려면 `--set train.resume=none`.
- 학습은 **GPU 한 대**로 합니다. 다른 GPU 작업(AVSR 학습·평가)과 동시에 돌리면 둘 다 느려집니다(9절).
  PC 여러 대로 할 수 있는 일은 12.1절을 보세요.

### 11.3 평가

**더블클릭:** `5_evaluate_sync.bat` → Enter(= `val`, `test`, `heldout` 세 가지 모두) 또는 셋 중 하나를 입력.

```powershell
.\scripts\evaluate_sync.ps1 --split heldout       # --ckpt 기본값: work/checkpoints_sync/best.pt
# 동일:
& $py -m avsr.sync.evaluate --ckpt work/checkpoints_sync/best.pt --split heldout
# 검증 화자만, 300문장:
& $py -m avsr.sync.evaluate --ckpt work/checkpoints_sync/best.pt --split val --max-utts 300
```

| 옵션 | 설명 |
|---|---|
| `--ckpt` | 매칭 모델 체크포인트(`evaluate_sync.ps1`/`5_evaluate_sync.bat`는 기본 `work/checkpoints_sync/best.pt`) |
| `--split` | `val`(C313), `test`(E014·C159), `heldout`(세 명 모두, 기본값. 서로 다른 세 사람의 3인 장면은 여기서만 가능) |
| `--max-utts N` | 고르게 뽑은 N문장만 사용(기본 `sync.eval_max_utts` = 600, 0 = 전부) |
| `--scenes N` | K인 장면 수(기본 300) |
| `--out` | 결과 JSON 경로(기본 `work/eval/sync_<split>_results.json`) |
| `--config`, `--set`, `--work-dir` | 설정 파일/덮어쓰기/작업 폴더. 모델 구조와 입력 특징은 체크포인트의 것을 사용 |

- 학습에 쓰지 않은 화자만 쓰며, 시드가 고정되어 같은 명령은 항상 같은 결과를 냅니다. 표를 출력하고 JSON에 저장합니다.
- `test`는 화자가 2명뿐이라 N = 3·4 후보나 3인 장면에 필요한 "다른 사람"이 모자랍니다. 이때는 같은 화자의 **다른
  녹화 세션** 음성을 후보로 쓰고, 출력과 JSON의 `notes`에 그렇게 적습니다.

### 11.4 결과 읽는 법 (실제 사용 상황으로)

창(window) 길이 0.52 s / 1 s / 2 s(13 / 25 / 50 프레임)는 **판단에 쓰는 시간**입니다. 길수록 정확하지만, 말하는 사람이
바뀌는 순간은 늦게 알아챕니다.

| 표 | 무엇을 재나 | 실제 상황에서의 의미 |
|---|---|---|
| 1) N-way selection | 얼굴 하나에 후보 음성 N개(자기 음성 + 다른 사람 음성 N−1개) 중 점수가 가장 높은 것이 자기 음성인 비율. 우연히 맞힐 확률은 1/N(50 / 33 / 25 %). `reverse`는 반대로 음성 하나에 얼굴 N개 | "분리된 음성 N개 중 이 얼굴의 것은?" / "이 음성은 화면의 N명 중 누구 입?" |
| 2) Leakage robustness | 1)과 같지만 후보 음성마다 다른 후보들의 목소리가 20 / 10 / 5 dB 낮게 섞여 있음 | 음성 분리가 완벽하지 않아 다른 사람 목소리가 남아 있어도 맞히는지(5 dB = 꽤 크게 남은 경우) |
| 3) Sync offset | 음성을 −15 ~ +15 프레임(±0.6 s) 밀어 가며 점수가 가장 높은 위치가 0 근처(±1 프레임 = ±40 ms)인 비율과 오차 중앙값(ms) | 모델이 정말 "소리와 입이 같은 순간에 움직이는지"를 보는지. 음성-영상이 어긋난 파일을 찾을 때도 쓸 수 있음 |
| 4) Scene assignment | K명(2·3명)이 나오는 장면: 얼굴 K × 음성 K 점수표를 헝가리안 알고리즘으로 1:1 배정. `scene acc` = 장면의 배정이 모두 맞은 비율, `stream acc` = 음성별 정답 비율. 전체 겹침 구간 / 1초 창마다(시간에 따라 자막을 붙일 때), 누설 조건 포함 | **실제 제품 동작에 가장 가까운 수치** |
| 5) Match / no-match | 진짜 쌍(얼굴 + 자기 음성)과 가짜 쌍(얼굴 + 다른 사람 음성)을 점수로 구분하는 능력. AUC(1 = 완벽, 0.5 = 무작위), EER(잘못 받아들이는 비율과 잘못 거절하는 비율이 같아지는 지점의 오류율), 그 지점의 점수 = `threshold` | **화면 밖 화자 판정 기준** |

**`offscreen_threshold`**(출력 마지막 줄, JSON의 `offscreen_threshold.value` = 1초 창 기준, 창 길이별 값은 `by_window`):
어떤 음성에 대해 화면에 보이는 **모든 얼굴의 점수가 이 값보다 낮으면, 그 음성은 화면 밖 화자**로 봅니다.
**`--split heldout` 결과의 값을 쓰세요.** 검증 화자는 C313 한 명뿐이라 `--split val`의 "가짜 쌍"은 같은 사람의 다른
녹화분이 되어, 제품 상황(다른 사람의 목소리)과 분포가 다릅니다. `heldout`(C313·E014·C159)은 가짜 쌍이 모두 다른 사람입니다.

제품에 붙이는 순서(요약):

1. 화면의 얼굴마다 입 크롭·입술 랜드마크를 전처리와 같은 방식으로 만들고(`avsr/infer.py`의 영상 처리와 같음),
   분리된 음성마다 fbank를 계산합니다.
2. 1초 창마다 `model.embed_video(...)`, `model.embed_audio(...)`로 임베딩을 만들고 `pair_scores`로 얼굴 × 음성
   점수표를 만듭니다(`avsr/sync/model.py`. 창을 만드는 방법은 `avsr/sync/evaluate.py`와 똑같이).
3. 헝가리안 알고리즘(`scipy.optimize.linear_sum_assignment`)으로 얼굴과 음성을 1:1로 짝짓고, 짝의 점수가
   `offscreen_threshold`보다 낮은 음성은 화면 밖 화자로 표시합니다. 여러 창의 점수를 평균하면 더 안정적입니다(표 1:
   창이 길수록 정확).

### 11.5 학습 결과 (이 데이터, 2026-09-24)

`work/checkpoints_sync/best.pt`(14.8 M 파라미터, 30 에포크 중 22 에포크, 이 PC에서 학습 약 27분). 입술 쪽은 자막 모델
(`work/checkpoints/best.pt`)의 입술·음성 앞단에서 시작했습니다. 평가는 학습에 없던 화자만 사용합니다.
`heldout` = C313·E014·C159, `test` = E014·C159(체크포인트 선택에 쓰지 않은 화자).

| 항목 (1초 창) | heldout | test |
|---|---|---|
| 장면 배정, 2명 (전체 구간) | 100 % | 100 % |
| 장면 배정, 3명 (전체 구간) | 100 % | 99.3~100 % |
| 장면 배정, 3명, 1초마다 (다른 목소리가 5 dB 남은 분리 결과) | 96.5 % | 92.7 % |
| 음성 2개 중 이 얼굴의 것 고르기 | 94.2 % | 93.0 % |
| 음성 4개 중 고르기 | 89.3 % | 85.3 % |
| 진짜 쌍 / 다른 사람 음성 구분 AUC (화면 밖 화자 판정) | 95.2 % | 93.7 % |
| `offscreen_threshold` | **0.2193** | 0.2024 |

- 제품에 가까운 "장면 배정"(여러 창의 점수를 모아 얼굴과 음성을 1:1로 짝짓기)은 거의 완벽합니다. 한 창만 보고 N개
  중 하나를 고르는 것보다 훨씬 정확하므로, 실제 적용 때도 1초 이상 모아서 판단하세요.
- 화자 E014(측면 각도가 많고 녹음 특성이 다른 화자)가 가장 어렵습니다. 진짜 쌍 점수가 다른 화자보다 낮습니다.
- 3명 장면은 테스트 화자가 2명뿐이라 일부 상대가 같은 화자의 다른 녹화분입니다(같은 얼굴·목소리라 실제보다 어려운 조건).
- 결과 파일: `work/eval/sync_heldout_results.json`, `work/eval/sync_test_results.json`.

---

## 12. PC 4대로 나눠서 전처리하기

### 12.1 먼저 알아 두세요

- **여러 PC로 나눌 수 있는 것은 전처리(영상 → 특징)입니다. 학습은 메인 PC 한 대에서 합니다.** 학습은 GPU 한 대가
  전체 데이터를 여러 번 반복해서 봐야 하고, 여러 PC로 나눠 학습하려면 고속 네트워크와 분산 학습 설정이 필요합니다.
  이 데이터 규모(특징 8.3 GB)에서는 이득이 없습니다.
- **지금 데이터(960개 영상)는 메인 PC에서 이미 전부 전처리되어 있습니다**(`work` 폴더, 41,350문장.
  확인: `& $py tools/shards.py status`). 다른 PC들이 **같은 데이터**를 가지고 있다면, 나눠서 다시 전처리해도 메인 PC에
  이미 있는 것과 같은 결과가 나올 뿐이고, 합치기(12.4)에서도 "이미 있음(already)"으로 모두 건너뜁니다.
  (실제로 확인: 이 PC에서 2/4 조각 중 82개 영상을 다시 처리해 비교 → 매니페스트 3,459줄 모두 동일, 음성과 검출 여부
  100 % 동일, 입술 랜드마크는 프레임의 91.8 %가 비트 단위로 동일하고 나머지는 MediaPipe의 실행마다 생기는 작은 차이.)
  → **다른 PC들은 새 데이터가 생겼을 때 필요합니다**(AI-Hub `1.Training` 분할, 새로 녹화한 영상 등 — 8절).
- 다른 PC들을 지금 활용하는 방법: 메인 PC의 `work` 폴더 중 `manifests`, `feats`, `checkpoints\best.pt`(합계 약 9 GB)를
  다른 PC의 `avsr_kr\work` 폴더로 복사한 뒤, **PC마다 다른 설정으로 매칭 모델 학습을 동시에** 돌려 비교할 수 있습니다.
  ```powershell
  .\scripts\train_sync.ps1 --set sync.temporal=gru --set train.ckpt_dir=work/checkpoints_sync_gru        # PC 2
  .\scripts\train_sync.ps1 --set sync.window_frames=50 --set train.ckpt_dir=work/checkpoints_sync_w50     # PC 3
  ```
  가장 좋은 `best.pt`를 메인 PC로 가져와 11.3절처럼 같은 조건으로 평가합니다.

### 12.2 준비 (PC마다 처음 한 번)

1. **메인 PC**에서 코드를 묶습니다. 탐색기에서 `avsr_kr\scripts\make_dist.ps1`을 오른쪽 클릭 → "PowerShell에서 실행"
   (Windows 11에서는 "추가 옵션 표시" 안에 있음. 또는 PowerShell 창에서 `.\scripts\make_dist.ps1`). `dist\avsr_kr_code.zip`이 생깁니다(약 3.5 MB, 파일 약 70개 — 코드,
   설정, MediaPipe 모델, 테스트, 문서. 데이터·체크포인트·`work` 폴더는 들어가지 않음).
2. zip을 USB 등으로 다른 PC에 옮겨 압축을 풀고(오른쪽 클릭 → 모두 압축 풀기), 안의 `avsr_kr` 폴더를 둘 곳에 둡니다.
   **경로에 한글이 없는 곳**을 권장합니다(예: `C:\avsr_kr`). MediaPipe(얼굴 랜드마크)가 한글 경로의 모델 파일을 열지
   못하기 때문입니다. `2_preprocess_part.bat`는 이 문제를 스스로 피해 가지만(모델 파일 사본을 `C:\ProgramData\avsr_kr`에
   두고 사용), 추론 등 다른 명령은 영향을 받습니다(12.6).
3. `avsr_kr` 폴더의 **`1_setup.bat`를 더블클릭** → Enter.
   Python 3.12(winget, 사용자 설치), FFmpeg(winget `Gyan.FFmpeg`), 파이썬 패키지(`requirements.txt`, torch는 CUDA 12.8판,
   처음에는 약 4 GB 다운로드), `msvc-runtime`(없으면 torch가 `WinError 126 ... c10.dll` 오류), MediaPipe 모델 파일을 확인해
   없는 것만 설치하고, GPU(CUDA·bf16)·MediaPipe·FFmpeg(NVDEC 포함)가 실제로 동작하는지 검사합니다. 이미 설치된 것은
   건너뛰므로 여러 번 실행해도 됩니다. 기록은 `setup.log`. `C`를 입력하면 아무것도 설치하지 않고 확인만 합니다
   (메인 PC에서 실행한 결과: 모든 항목 OK, 약 5초).
4. 각 PC에 데이터셋 폴더(`009.립리딩(입모양) 음성인식 데이터`, tar 파일 그대로)가 있어야 합니다. 위치는 PC마다 달라도 됩니다.

### 12.3 전처리 — 4대에서 동시에

각 PC에서 **`2_preprocess_part.bat`를 더블클릭**하고 질문에 답합니다(질문은 영어로 나옵니다).

| 질문 | PC 1 (메인 PC) | PC 2~4 |
|---|---|---|
| `Which PC is this?` | `1` | `2`, `3`, `4` (**PC마다 다른 번호**) |
| `Same videos on all PCs?` | Enter(= 예, 4등분) | Enter |
| `Data folder` | Enter = `다운로드` 폴더에서 이름이 `009.`로 시작하는 폴더. 다른 곳이면 경로를 붙여 넣기: 탐색기에서 폴더를 오른쪽 클릭 → "경로로 복사" → 검은 창에서 오른쪽 클릭 | 같음 |
| `Transfer folder` | (묻지 않음: 결과가 바로 `work` 폴더에 들어감) | USB 드라이브(예: `E:\`) 또는 메인 PC의 공유 폴더(예: `\\메인PC이름\share`). 그 안의 `avsr_part2` 같은 폴더로 복사됨. Enter = 복사하지 않고 `avsr_kr\work_shard2`에 남김 |

- **걸리는 시간:** 한 PC가 960개를 모두 처리하면 약 110분(10절) → 4대로 나누면 **PC당 240개, 약 30분**입니다(이 PC에서 2/4
  조각을 돌려 본 결과 첫 79개가 10.3분 → 240개 약 31분). 그 뒤 전송 폴더로 복사하는 시간이 더해집니다(조각 하나 약
  2.1 GB, 파일 약 2만 개. 로컬 디스크에서는 수 초지만, 작은 파일이 많아 느린 USB 메모리에서는 수 분 이상 걸릴 수 있습니다.
  외장 SSD나 네트워크 폴더가 빠릅니다).
- 나누는 규칙: 영상 이름순으로 정렬해 PC K가 K번째, K+4번째, K+8번째 … 영상을 맡습니다. 그래서 **모든 PC의 데이터가
  같아야** 겹치거나 빠지는 영상이 없습니다. 시작하면 `[preprocess] shard 2/4: 240 of 960 videos`처럼 나오는데,
  **모든 PC에서 "of 960"(전체 영상 수)이 같아야 합니다.** (이 데이터로 확인: 네 조각 모두 240개, 서로 겹치지 않고 합치면
  960개, 조각마다 화자 16명이 모두 들어 있음.)
- 중간에 멈추거나 창을 닫아도, 다시 실행하면 끝난 영상은 건너뛰고 이어서 처리합니다. 복사도 같은 크기로 이미 복사된
  파일은 건너뜁니다.
- PC마다 **서로 다른 영상**을 가지고 있다면 두 번째 질문에 `N` → 그 PC가 가진 영상을 모두 처리합니다(합치기는 같음).
- 전송 폴더를 비워 둔 경우: `avsr_kr\work_shard2` 폴더를 통째로 USB 등에 복사해 메인 PC로 가져가면 됩니다.

### 12.4 합치기 — 메인 PC

PC 2~4가 끝나면 USB를 메인 PC에 꽂고(또는 공유 폴더를 쓰고) **`3_merge_parts.bat`를 더블클릭**합니다.

- 조각이 들어 있는 폴더를 입력합니다. 예: `E:\` (그 안의 `avsr_part2`, `avsr_part3`, …를 자동으로 찾음).
  여러 곳이면 `;`로 구분합니다: `E:\;F:\;\\PC3\share`. `avsr_kr` 폴더 안의 `work_shard1`~`work_shard4` 폴더는
  자동으로 포함됩니다.
- 동작: 끝난 영상(`.done` 표시가 있는 것)만 `work`로 복사하고, 복사한 파일의 크기를 검증한 뒤, 조각별 요약 표와
  화자별 상태를 출력합니다.
  - `already`: `work`에 이미 있는 영상 → 건너뜀(덮어쓰지 않음). 같은 데이터를 다시 처리했다면 전부 여기에 해당합니다.
    단, `work`에서 끝난 것으로 표시되어 있어도 파일이 빠졌거나 비어 있는 영상은 조각의 완전한 사본으로 교체하고
    "replaced from a part"로 알려 줍니다. 복사 후 크기 검증에 실패한 영상은 끝난 표시(`.done`)를 지우므로, 다시 실행하면
    그 영상만 다시 복사합니다.
  - `broken`: 조각 안에 파일이 빠진 영상 → 복사하지 않음. 그 PC에서 전처리를 다시 실행하고 다시 복사하세요.
  - `unfin.`: 그 PC에서 아직 끝나지 않은 영상 → 복사하지 않음.
  - 조각 폴더(원본)는 수정하거나 지우지 않습니다. 다시 실행해도 안전합니다(이미 복사된 것은 건너뜀).
- 끝나면 `4_train_sync.bat`로 학습합니다(11.2). 새 데이터를 AVSR 모델에도 쓰려면 4.2절 학습도 다시 하세요.

### 12.5 명령으로 하기

```powershell
# 메인 PC: 다른 PC로 가져갈 코드 묶기 -> dist\avsr_kr_code.zip
.\scripts\make_dist.ps1
# 설치 / 확인만
.\scripts\setup.ps1
.\scripts\setup.ps1 -CheckOnly
# PC 2: 2/4 조각 전처리 -> E:\avsr_part2 로 복사 (다시 실행하면 이어서)
.\scripts\preprocess_shard.ps1 -Shard 2 -NumShards 4 -DataRoot "D:\data\009.립리딩(입모양) 음성인식 데이터" -Dest E:\
# 이 PC가 맡을 영상 목록만 보기 / 메인 PC는 바로 work 폴더로 / 추가 인자는 avsr.preprocess로 전달
.\scripts\preprocess_shard.ps1 -Shard 2 -DataRoot "D:\data\009.립리딩(입모양) 음성인식 데이터" -DryRun
.\scripts\preprocess_shard.ps1 -Shard 1 -WorkDir work --workers 10
# 메인 PC: 합치기(avsr_kr\work_shard* 포함) / 복사할 것만 미리 보기
.\scripts\merge_shards.ps1 -Src E:\ -IncludeLocal
.\scripts\merge_shards.ps1 -Src "E:\;F:\parts" -DryRun
# 같은 일을 모듈로:
& $py -m avsr.preprocess --data-root "D:\data\009.립리딩(입모양) 음성인식 데이터" --work-dir work_shard2 --shard 2/4
& $py tools/shards.py pack   --work-dir work_shard2 --dest E:\avsr_part2
& $py tools/shards.py merge  --src E:\ --work-dir work
& $py tools/shards.py status --work-dir work
```

| 명령 | 옵션 | 설명 |
|---|---|---|
| `avsr.preprocess` | `--shard K/N` | 영상 이름순 목록에서 `번호 % N == K−1`인 영상만 처리(`--order`와 무관). `--dry-run`과 함께 쓰면 그 조각의 목록 |
| `tools/shards.py pack` | `--work-dir`, `--dest` | 조각 작업 폴더의 끝난 영상(매니페스트 + `.done` + 그 영상의 `feats` 파일)과 `preprocess_log.jsonl`을 전송 폴더로 복사 |
| `tools/shards.py merge` | `--src`(여러 번 또는 `;`로 구분), `--work-dir`(기본 `work`) | 조각들을 작업 폴더로 복사. `--src`는 조각 폴더 자체이거나 조각 폴더를 담은 폴더(두 단계 아래까지 찾음) |
|  | `--overwrite` | `work`에 이미 있는 영상도 조각의 것으로 교체 |
|  | `--dry-run` | 복사하지 않고 계획만 출력 |
| `pack`, `merge` | `--checksum` | 크기뿐 아니라 SHA-1까지 비교(느림) |
| `tools/shards.py status` | `--work-dir`, `--deep` | 끝난 영상·문장 수, 빠진 파일, 화자별 수, 실패 기록. `--deep`: npz/mp4 내용까지 검사 |

- 종료 코드: 0 = 정상, 1 = 문제 있음(목록 출력), 2 = 잘못된 폴더 등.
- 안전장치: 파일은 `<이름>.part`로 복사 → 크기 확인 → 이름 변경. `.done` 표시는 마지막에 복사하므로 중간에 끊기면
  "끝나지 않은 영상"으로 남아 다음 실행에서 다시 복사됩니다. USB 빠짐·네트워크 끊김은 몇 번 재시도하고, 260자가 넘는
  긴 경로와 `\\서버\공유` 경로도 지원하며, 복사 전에 남은 공간을 확인합니다.
- 이 PC의 `status` 결과(2026-09-24): 완료 영상 960, 문장 41,350(모든 각도 합 63.4시간), 특징 파일 82,700개(8.3 GB),
  빠진 파일 0, 화자 16명 각 60개 영상.

### 12.6 문제 해결

- **PC마다 "of 960"의 숫자가 달라요.** 데이터가 다릅니다(빠진 tar 등). 이대로 나누면 영상이 겹치거나 빠집니다. 겹친 것은
  합치기가 건너뛰고, 빠진 것은 메인 PC에서 `2_preprocess_part.bat`를 `1`, `N`(나누지 않음)으로 실행하면 채워집니다
  (이미 있는 영상은 건너뜀).
- **PowerShell에서 경로 뒤에 `\`를 붙였더니 이상하게 동작해요.** Windows PowerShell 5.1은 `"D:\my dir\"`처럼 따옴표 안의
  경로가 `\`로 끝나면 뒤의 인자까지 경로에 붙여 버립니다. `.bat` 파일은 이를 자동으로 처리하고, `preprocess_shard.ps1`과
  `merge_shards.ps1`은 이런 경로를 발견하면 오류로 알려 줍니다. 끝의 `\`를 빼세요(드라이브 루트는 따옴표 없이 `E:\`).
- **`avsr_kr`를 한글이 들어간 폴더에 두었더니 `Unable to open file ... face_landmarker.task` 오류가 나요.** MediaPipe가
  한글 경로의 파일을 열지 못합니다. `2_preprocess_part.bat`(`preprocess_shard.ps1`)는 자동으로 우회합니다. 직접 명령을 쓸
  때는 `--model C:\ProgramData\avsr_kr\face_landmarker.task`(전처리) 또는 `--landmarker C:\ProgramData\avsr_kr\face_landmarker.task`
  (추론)를 주거나(사본은 `1_setup.bat`가 만들어 둠), `avsr_kr` 폴더를 `C:\avsr_kr`처럼 영문 경로로 옮기세요. 학습·평가는
  영향을 받지 않습니다.
- **USB를 뽑았거나 네트워크가 끊겼어요.** 같은 것을 다시 실행하면 이어서 복사합니다. 전송 폴더의 드라이브나 공유
  폴더가 없으면(USB를 아직 꽂지 않음 등) `2_preprocess_part.bat`는 전처리를 시작하기 전에 `transfer folder not reachable`
  오류로 알려 줍니다. USB를 꽂고 다시 실행하거나, 전송 폴더를 비워 두세요.
- **디스크 공간:** 조각 하나의 결과는 약 2.1 GB이고, 처리 중에는 영상을 tar에서 하나씩 임시로 꺼내므로(동시에 최대 14개 ×
  약 0.4 GB) 약 10 GB의 여유가 필요합니다. 메인 PC의 `work` 폴더에는 전체 약 8.3 GB가 필요합니다.
- **`status`에 "interrupted videos"가 보여요.** 처리 도중 멈춘 영상의 흔적입니다. 사용되지도 복사되지도 않으며, 전처리를
  다시 실행하면 새로 처리합니다.
