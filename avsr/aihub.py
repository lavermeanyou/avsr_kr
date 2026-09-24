"""Dataset-format layer for the AI-Hub "립리딩(입모양) 음성인식 데이터" (SPEC section 2).

Responsibilities
- discover label / media sources under a data root (tar archives or plain directories, any split folders)
- build an index of media members (stem+ext -> location) without extracting anything
- parse a label JSON into a typed LabelDoc (fixing the bounding-box coordinate order quirk)
- extract a single media file on demand
"""
from __future__ import annotations

import json
import re
import shutil
import tarfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

LABEL_DIR_NAME = "라벨링데이터"
MEDIA_DIR_NAME = "원천데이터"
STEM_RE = re.compile(r"^lip_([A-Z])_(\d+)_([FM])_(\d+)_([A-Z]\d+)_([A-Z])_(\d+)$")


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------
@dataclass
class Source:
    """A label or media source: either a .tar archive or a directory tree."""

    kind: str          # "tar" | "dir"
    path: str          # absolute path of the tar file or directory
    split_dir: str     # e.g. "2.Validation" (name of the split folder; "" if none)
    role: str          # "label" | "media"


def _split_dirs(root: Path) -> List[Path]:
    """Return candidate split folders: root itself if it directly holds 라벨링데이터/원천데이터,
    otherwise every (grand)child folder that does (e.g. 01.데이터/1.Training, 01.데이터/2.Validation)."""
    if (root / LABEL_DIR_NAME).exists() or (root / MEDIA_DIR_NAME).exists():
        return [root]
    found: List[Path] = []
    for p in sorted(root.rglob(LABEL_DIR_NAME)):
        if p.is_dir():
            found.append(p.parent)
    for p in sorted(root.rglob(MEDIA_DIR_NAME)):
        if p.is_dir() and p.parent not in found:
            found.append(p.parent)
    return found


def discover_sources(data_root: str | Path) -> List[Source]:
    """Find every label/media tar or directory under data_root."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(root)
    sources: List[Source] = []
    for split in _split_dirs(root):
        split_name = split.name if split != root else ""
        for role, dname in (("label", LABEL_DIR_NAME), ("media", MEDIA_DIR_NAME)):
            d = split / dname
            if not d.exists():
                continue
            tars = sorted(d.glob("*.tar"))
            for t in tars:
                sources.append(Source("tar", str(t.resolve()), split_name, role))
            # plain directories (already extracted)
            for sub in sorted(p for p in d.iterdir() if p.is_dir()):
                sources.append(Source("dir", str(sub.resolve()), split_name, role))
    if not sources:
        raise FileNotFoundError(f"no {LABEL_DIR_NAME}/{MEDIA_DIR_NAME} found under {root}")
    return sources


# --------------------------------------------------------------------------------------
# Media index
# --------------------------------------------------------------------------------------
@dataclass
class MediaRef:
    stem: str
    ext: str            # ".mp4" | ".wav"
    source_kind: str    # "tar" | "dir"
    source_path: str    # tar path or directory root
    member: str         # member name inside tar, or absolute file path for dir
    size: int
    split_dir: str
    offset: int = -1    # byte offset of the member data inside the tar (-1 = unknown / dir source)


def build_media_index(sources: List[Source], cache_path: Optional[Path] = None, log=print) -> Dict[str, MediaRef]:
    """Index every mp4/wav member of the media sources: key = f"{stem}{ext}". Cached as JSON keyed by source mtimes."""
    media = [s for s in sources if s.role == "media"]
    sig = {s.path: (Path(s.path).stat().st_mtime if s.kind == "tar" else 0.0) for s in media}
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == {k: v for k, v in sig.items()}:
                return {k: MediaRef(**v) for k, v in cached["index"].items()}
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    index: Dict[str, MediaRef] = {}
    for s in media:
        t0 = time.time()
        n = 0
        if s.kind == "tar":
            with tarfile.open(s.path, "r") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    name = Path(m.name)
                    if name.suffix.lower() not in (".mp4", ".wav"):
                        continue
                    key = name.stem + name.suffix.lower()
                    index[key] = MediaRef(name.stem, name.suffix.lower(), "tar", s.path, m.name, m.size, s.split_dir,
                                          int(m.offset_data) if not m.sparse else -1)
                    n += 1
        else:
            for f in Path(s.path).rglob("*"):
                if f.is_file() and f.suffix.lower() in (".mp4", ".wav"):
                    key = f.stem + f.suffix.lower()
                    index[key] = MediaRef(f.stem, f.suffix.lower(), "dir", s.path, str(f.resolve()), f.stat().st_size, s.split_dir)
                    n += 1
        log(f"[aihub] indexed {n} media files in {Path(s.path).name} ({time.time() - t0:.1f}s)")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"signature": sig, "index": {k: asdict(v) for k, v in index.items()}}, ensure_ascii=False), encoding="utf-8")
    return index


def extract_media(ref: MediaRef, dest_dir: Path) -> Path:
    """Materialise one media file. For tar sources copies the member to dest_dir; for dir sources returns the existing path."""
    if ref.source_kind == "dir":
        return Path(ref.member)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{ref.stem}{ref.ext}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    if ref.offset >= 0:
        # fast path: seek straight to the member data
        chunk = 16 * 1024 * 1024
        with open(ref.source_path, "rb") as f, open(tmp, "wb") as out:
            f.seek(ref.offset)
            left = ref.size
            while left > 0:
                buf = f.read(min(chunk, left))
                if not buf:
                    raise IOError(f"truncated tar member {ref.member} in {ref.source_path}")
                out.write(buf)
                left -= len(buf)
    else:
        with tarfile.open(ref.source_path, "r") as tf:
            m = tf.getmember(ref.member)
            src = tf.extractfile(m)
            if src is None:
                raise IOError(f"cannot extract {ref.member} from {ref.source_path}")
            with open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
    if tmp.stat().st_size != ref.size:
        raise IOError(f"size mismatch extracting {ref.member}: {tmp.stat().st_size} != {ref.size}")
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
@dataclass
class Sentence:
    id: int
    topic: str
    text: str
    start: float
    end: float


@dataclass
class LabelDoc:
    stem: str                    # e.g. lip_J_1_F_03_E220_A_001
    video_name: str
    fps: float
    width: int
    height: int
    duration_s: float
    noise_env: int
    angle: str
    video_env: str
    speaker: str
    specificity: str
    gender: str
    age: int
    accent: str
    group: str
    session: str
    sentences: List[Sentence]
    face_boxes: np.ndarray       # int32 [F, 4] as (x0, y0, x1, y1)
    lip_boxes: np.ndarray        # int32 [F, 4] as (x0, y0, x1, y1)
    bbox_order: str              # "yx" (dataset quirk, verified) or "xy"
    split_dir: str = ""
    label_path: str = ""
    audio_name: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.lip_boxes.shape[0])


def _parse_duration(s: str) -> float:
    parts = [float(x) for x in str(s).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts[-3:]
    return h * 3600 + m * 60 + sec


def infer_bbox_order(boxes: np.ndarray, width: int, height: int, default: str = "yx") -> str:
    """Decide whether rows are (x0,y0,x1,y1) ["xy"] or (y0,x0,y1,x1) ["yx"] by counting out-of-frame violations."""
    if boxes.size == 0:
        return default
    b = boxes.astype(np.int64)
    viol_xy = int((b[:, 2] > width).sum() + (b[:, 3] > height).sum())
    viol_yx = int((b[:, 3] > width).sum() + (b[:, 2] > height).sum())
    if viol_xy == viol_yx:
        return default
    return "xy" if viol_xy < viol_yx else "yx"


def parse_label(data: bytes | str | dict | list, label_path: str = "", split_dir: str = "", bbox_order: str = "auto") -> LabelDoc:
    """Parse one AI-Hub label JSON (bytes/str/parsed) into a LabelDoc with boxes as (x0, y0, x1, y1)."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8-sig")
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"label list has {len(data)} entries (expected 1): {label_path}")
        data = data[0]
    vi, ai = data["Video_info"], data.get("Audio_info", {})
    res = str(vi.get("Resolution", "1920*1080")).replace("x", "*").replace("X", "*")
    w, h = (int(v) for v in res.split("*"))
    stem = Path(vi["video_Name"]).stem
    m = STEM_RE.match(stem)
    if m:
        group, _age_tok, gender_tok, _, speaker_tok, angle_tok, session = m.groups()
    else:
        group, gender_tok, speaker_tok, angle_tok, session = "", "", "", "", ""
    sp = data.get("speaker_info", {})
    bb = data["Bounding_box_info"]
    face = np.asarray(bb["Face_bounding_box"]["xtl_ytl_xbr_ybr"], dtype=np.int32).reshape(-1, 4)
    lip = np.asarray(bb["Lip_bounding_box"]["xtl_ytl_xbr_ybr"], dtype=np.int32).reshape(-1, 4)
    if face.shape[0] != lip.shape[0]:
        raise ValueError(f"face/lip box count mismatch {face.shape[0]} vs {lip.shape[0]} in {label_path}")
    order = bbox_order if bbox_order in ("xy", "yx") else infer_bbox_order(np.concatenate([face, lip]), w, h)
    if order == "yx":  # (y0, x0, y1, x1) -> (x0, y0, x1, y1)
        face = face[:, [1, 0, 3, 2]]
        lip = lip[:, [1, 0, 3, 2]]
    sentences = [
        Sentence(int(s["ID"]), str(s.get("topic", "")), str(s["sentence_text"]), float(s["start_time"]), float(s["end_time"]))
        for s in data.get("Sentence_info", [])
    ]
    sentences.sort(key=lambda s: s.start)
    return LabelDoc(
        stem=stem,
        video_name=vi["video_Name"],
        fps=float(vi.get("FPS", 30)),
        width=w,
        height=h,
        duration_s=_parse_duration(vi.get("video_Duration", "0")),
        noise_env=int(data.get("Audio_env", {}).get("Noise", 0)),
        angle=str(data.get("Video_env", {}).get("Angle", angle_tok)),
        video_env=str(data.get("Video_env", {}).get("env", "")),
        speaker=str(sp.get("speaker_ID", speaker_tok)),
        specificity=str(sp.get("Specificity", "")),
        gender=str(sp.get("Gender", gender_tok)),
        age=int(sp.get("Age", 0) or 0),
        accent=str(sp.get("Accent", "")),
        group=group,
        session=session,
        sentences=sentences,
        face_boxes=face,
        lip_boxes=lip,
        bbox_order=order,
        split_dir=split_dir,
        label_path=label_path,
        audio_name=str(ai.get("Audio_Name", "")),
    )


@dataclass
class LabelRef:
    stem: str
    source_kind: str    # "tar" | "dir"
    source_path: str
    member: str         # member name in tar or absolute path
    split_dir: str

    def read_bytes(self) -> bytes:
        if self.source_kind == "dir":
            return Path(self.member).read_bytes()
        with tarfile.open(self.source_path, "r") as tf:
            f = tf.extractfile(tf.getmember(self.member))
            if f is None:
                raise IOError(f"cannot read {self.member} from {self.source_path}")
            return f.read()

    def load(self, bbox_order: str = "auto") -> LabelDoc:
        return parse_label(self.read_bytes(), label_path=f"{self.source_path}::{self.member}", split_dir=self.split_dir, bbox_order=bbox_order)


def iter_label_refs(sources: List[Source]) -> Iterator[LabelRef]:
    """Enumerate every label JSON (without parsing) across the label sources."""
    for s in sources:
        if s.role != "label":
            continue
        if s.kind == "tar":
            with tarfile.open(s.path, "r") as tf:
                for m in tf:
                    if m.isfile() and m.name.lower().endswith(".json"):
                        yield LabelRef(Path(m.name).stem, "tar", s.path, m.name, s.split_dir)
        else:
            for f in sorted(Path(s.path).rglob("*.json")):
                yield LabelRef(f.stem, "dir", s.path, str(f.resolve()), s.split_dir)


def parse_stem(stem: str) -> Optional[dict]:
    """Split lip_J_1_F_03_E220_A_001 into its fields (None if it does not match)."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    group, _, gender, age, speaker, angle, session = m.groups()
    return {"group": group, "gender": gender, "age": int(age), "speaker": speaker, "angle": angle, "session": session}


__all__ = [
    "Source", "MediaRef", "LabelRef", "LabelDoc", "Sentence",
    "discover_sources", "build_media_index", "extract_media", "iter_label_refs", "parse_label", "parse_stem", "infer_bbox_order",
]
