"""Move preprocessed parts between PCs (docs/SYNC_SPEC.md section 8). Standard library only.

  $py tools/shards.py pack   --work-dir work_shard2 --dest E:\\avsr_parts\\avsr_part2
  $py tools/shards.py merge  --src E:\\avsr_parts [--src \\\\PC3\\share\\avsr_part3 ...] [--work-dir work] [--overwrite]
  $py tools/shards.py status [--work-dir work] [--deep]

A "part" folder has the same layout as a work dir (manifests/<stem>.jsonl + .done, feats/<stem>/*.npz|*.mp4,
preprocess_log.jsonl), so a work_shardK folder itself can also be given to merge. `merge --src` accepts such a folder
or a folder whose (grand)child folders are parts (e.g. a USB drive holding avsr_part2, avsr_part3, avsr_part4).

Safety and robustness:
- Only finished videos (manifest + .done marker) are copied, and only the files their manifest rows reference.
- Every file goes to "<name>.part" first, its size (and SHA-1 with --checksum) is checked, then it is renamed. The
  .done marker is copied last, so an interrupted copy never looks finished. Files that already exist with the same
  size are skipped, so re-running the same command resumes. Transient I/O errors (USB hiccup, network drop) are
  retried; long paths (> 240 characters) use the Windows extended-length form, also on \\\\server\\share paths.
- Source data is never modified or deleted. merge never replaces a video that is already finished in the work dir
  unless --overwrite is given, or its files there are missing/empty (then a complete copy from a part replaces it).
  A video that fails the size verification after copying loses its .done marker, so re-running copies it again.
- An unreachable destination (USB drive not plugged in, share offline) is refused before anything is copied.
Exit code: 0 = OK, 1 = finished with problems (listed), 2 = usage error / nothing to do.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MANIFESTS = "manifests"
LOG_NAME = "preprocess_log.jsonl"
PART_SUFFIX = ".part"
FILE_KEYS = ("npz", "mouth_mp4")
NPZ_MEMBERS = {"lm.npy", "cue.npy", "valid.npy", "audio.npy"}
RETRIES = 3
RETRY_WAIT_S = 2.0
FREE_SPACE_MARGIN = 256 * 1024 * 1024
MAX_EXAMPLES = 10


class ShardError(Exception):
    """Fatal, user-facing error (wrong folder, refused operation, device gone)."""


# ----------------------------------------------------------------------------------------------------------------------
# file-system helpers (all path access goes through _ext so long USB/UNC paths work without LongPathsEnabled)
# ----------------------------------------------------------------------------------------------------------------------
def _ext(path: Path | str) -> str:
    """Absolute path; on Windows, long paths get the extended-length prefix (\\\\?\\C:\\... or \\\\?\\UNC\\server\\...)."""
    s = os.path.abspath(str(path))
    if os.name != "nt" or len(s) < 240 or s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def _size(path: Path) -> Optional[int]:
    """File size in bytes, or None when it does not exist (or is not a regular file)."""
    try:
        st = os.stat(_ext(path))
    except OSError:
        return None
    return st.st_size if stat.S_ISREG(st.st_mode) else None


def _is_dir(path: Path) -> bool:
    return os.path.isdir(_ext(path))


def _same_dir(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(_ext(a), _ext(b))
    except OSError:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _disk_full(e: OSError) -> bool:
    return e.errno == errno.ENOSPC or getattr(e, "winerror", None) in (39, 112)


def _retry(what: str, fn: Callable[[], None]) -> None:
    """Run fn, retrying transient OSErrors (unplugged/remounted USB, dropped network share) with backoff."""
    wait = RETRY_WAIT_S
    for attempt in range(RETRIES + 1):
        try:
            fn()
            return
        except OSError as e:
            if _disk_full(e):
                raise ShardError(f"{what} failed: the destination disk is full ({e})") from e
            if attempt == RETRIES:
                raise ShardError(f"{what} failed after {RETRIES + 1} attempts: {e}\n  Is the USB drive / network "
                                 f"folder still connected? Run the same command again to resume.") from e
            print(f"    ! {what}: {e} - retrying in {wait:.0f} s", flush=True)
        time.sleep(wait)
        wait *= 2


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(_ext(path), "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def needs_copy(src: Path, dst: Path, checksum: bool, src_size: Optional[int] = None) -> bool:
    """True unless dst already holds the same file (same size; same SHA-1 too with checksum)."""
    s = _size(src) if src_size is None else src_size
    if s is None:
        raise ShardError(f"source file disappeared: {src}")
    if _size(dst) != s:
        return True
    return checksum and _sha1(src) != _sha1(dst)


def copy_file(src: Path, dst: Path, checksum: bool = False) -> int:
    """Copy src -> dst through dst.part with size (and optional SHA-1) verification; returns the bytes copied."""
    size = _size(src)
    if size is None:
        raise ShardError(f"source file disappeared: {src}")
    tmp = dst.with_name(dst.name + PART_SUFFIX)

    def once() -> None:
        os.makedirs(_ext(dst.parent), exist_ok=True)
        shutil.copyfile(_ext(src), _ext(tmp))
        got = os.stat(_ext(tmp)).st_size
        if got != size:
            raise OSError(f"size check failed ({got} of {size} bytes arrived)")
        if checksum and _sha1(src) != _sha1(tmp):
            raise OSError("SHA-1 of the copy differs from the source")
        os.replace(_ext(tmp), _ext(dst))

    _retry(f"copy {src} -> {dst}", once)
    return size


def _remove(path: Path) -> None:
    try:
        os.remove(_ext(path))
    except FileNotFoundError:
        pass


# ----------------------------------------------------------------------------------------------------------------------
# work-dir model
# ----------------------------------------------------------------------------------------------------------------------
def read_rows(path: Path) -> List[dict]:
    """Manifest rows of one video (ValueError on invalid JSON)."""
    rows: List[dict] = []
    with open(_ext(path), "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path.name} line {i}: invalid JSON ({e.msg})") from None
    return rows


def row_file(row: dict, key: str) -> str:
    """The work-dir-relative feature path of a manifest row (posix form); ValueError if absent or outside the dir."""
    rel = str(row.get(key) or "").replace("\\", "/")
    p = PurePosixPath(rel)
    if not rel or p.is_absolute() or ".." in p.parts or ":" in rel:
        raise ValueError(f"utterance {row.get('utt_id', '?')}: bad {key} path {rel!r}")
    return rel


def scan_manifests(root: Path) -> Tuple[List[str], List[str], List[str]]:
    """(finished stems [manifest + .done], unfinished stems [manifest only], stems with a .done but no manifest)."""
    man = root / MANIFESTS
    if not _is_dir(man):
        return [], [], []
    names = os.listdir(_ext(man))
    jsonl = {n[: -len(".jsonl")] for n in names if n.endswith(".jsonl")}
    done = {n[: -len(".done")] for n in names if n.endswith(".done")}
    return sorted(jsonl & done), sorted(jsonl - done), sorted(done - jsonl)


@dataclass
class Video:
    stem: str
    rows: List[dict]
    files: List[str]                      # work-dir-relative feature files referenced by the rows
    sizes: Dict[str, int] = field(default_factory=dict)   # source sizes of files (+ manifest, marker)
    problems: List[str] = field(default_factory=list)

    @property
    def manifest_file(self) -> str:
        return f"{MANIFESTS}/{self.stem}.jsonl"

    @property
    def done_file(self) -> str:
        return f"{MANIFESTS}/{self.stem}.done"

    @property
    def all_files(self) -> List[str]:
        return self.files + [self.manifest_file, self.done_file]


def load_video(root: Path, stem: str) -> Video:
    """Read a finished video's manifest and check that every referenced file exists and is not empty."""
    v = Video(stem, [], [])
    try:
        v.rows = read_rows(root / v.manifest_file)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        v.problems.append(f"unreadable manifest: {e}")
        return v
    if not v.rows:
        v.problems.append("manifest has no utterances")
    for r in v.rows:
        for key in FILE_KEYS:
            try:
                rel = row_file(r, key)
            except ValueError as e:
                v.problems.append(str(e))
                continue
            size = _size(root / rel)
            if not size:
                v.problems.append(f"{rel} {'missing' if size is None else 'is empty'}")
                continue
            v.files.append(rel)
            v.sizes[rel] = size
    for rel in (v.manifest_file, v.done_file):
        size = _size(root / rel)
        if size is None:
            v.problems.append(f"{rel} missing")
        else:
            v.sizes[rel] = size
    return v


def _subdirs(d: Path) -> List[Path]:
    try:
        names = os.listdir(_ext(d))
    except OSError:
        return []
    return sorted((d / n for n in names if _is_dir(d / n)), key=lambda p: p.name.lower())


def shard_roots(src: Path) -> List[Path]:
    """src itself if it is a part/work folder (has manifests/), else its child (or grandchild) folders that are."""
    if not _is_dir(src):
        raise ShardError(f"folder not found: {src}")
    if _is_dir(src / MANIFESTS):
        return [src]
    children = _subdirs(src)
    found = [c for c in children if _is_dir(c / MANIFESTS)]
    if not found:
        found = [g for c in children for g in _subdirs(c) if _is_dir(g / MANIFESTS)]
    if not found:
        raise ShardError(f"no preprocessed part found in {src} (expected a folder containing '{MANIFESTS}', or "
                         f"folders like avsr_part2 that contain it)")
    return found


def read_log_lines(root: Path) -> List[str]:
    p = root / LOG_NAME
    if _size(p) is None:
        return []
    with open(_ext(p), "r", encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\r\n") for ln in f if ln.strip()]


def _log_record(line: str) -> Optional[dict]:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    return rec if isinstance(rec, dict) else None


def append_log_lines(root: Path, lines: Sequence[str]) -> None:
    """Append lines to root/preprocess_log.jsonl (adds a missing trailing newline first)."""
    if not lines:
        return
    p = root / LOG_NAME
    size = _size(p)
    lead = ""
    if size:
        with open(_ext(p), "rb") as f:
            f.seek(size - 1)
            lead = "" if f.read(1) == b"\n" else "\n"
    with open(_ext(p), "a", encoding="utf-8", newline="\n") as f:
        f.write(lead + "".join(ln + "\n" for ln in lines))


# ----------------------------------------------------------------------------------------------------------------------
# transfer of one video (shared by pack and merge)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Plan:
    root: Path
    video: Video
    todo: List[str]                       # relative files to copy; the .done marker, when listed, is always last
    nbytes: int
    log_lines: List[str] = field(default_factory=list)


def plan_video(src_root: Path, dst_root: Path, video: Video, force: bool, checksum: bool) -> Plan:
    """Which files of a video must be copied (all with force; else those not already identical at dst)."""
    todo = [rel for rel in video.files + [video.manifest_file]
            if force or needs_copy(src_root / rel, dst_root / rel, checksum, video.sizes.get(rel))]
    if todo or needs_copy(src_root / video.done_file, dst_root / video.done_file, checksum,
                          video.sizes.get(video.done_file)):
        todo.append(video.done_file)
    return Plan(src_root, video, todo, sum(video.sizes.get(rel, 0) for rel in todo))


def transfer(plan: Plan, dst_root: Path, checksum: bool) -> Tuple[int, int]:
    """Copy the planned files; the log lines and then the .done marker come last, so an interrupted transfer never
    looks finished. Returns (files, bytes) copied."""
    if not plan.todo:
        return 0, 0
    v = plan.video
    _remove(dst_root / v.done_file)  # never leave a finished-looking video while its files are being replaced
    n = nbytes = 0
    for rel in plan.todo:
        if rel == v.done_file:
            append_log_lines(dst_root, plan.log_lines)
        nbytes += copy_file(plan.root / rel, dst_root / rel, checksum)
        n += 1
    return n, nbytes


def verify_video(dst_root: Path, video: Video) -> List[str]:
    """Files of the video whose destination copy is missing or differs in size from the source."""
    return [rel for rel in video.all_files if _size(dst_root / rel) != video.sizes.get(rel)]


def unmark_failed(dst_root: Path, video: Video, failed: List[str]) -> List[str]:
    """A video that failed verification must not look finished at the destination: remove its .done marker so that
    running the same command again copies it again (merge skips finished videos). Returns `failed`."""
    try:
        _remove(dst_root / video.done_file)
    except OSError as e:
        failed = failed + [f"could not remove {video.done_file} ({e}); delete it by hand"]
    return failed


def check_reachable(dst: Path, what: str) -> None:
    """Refuse early (clear message, no traceback) when the drive / network share of dst does not exist."""
    anchor = Path(os.path.abspath(str(dst))).anchor
    if not anchor or not _is_dir(Path(anchor)):
        raise ShardError(f"{what} {dst} is not reachable: the drive or network share {anchor or dst} was not found "
                         f"(is the USB drive plugged in / the network folder shared?)")


def check_free_space(dst: Path, need: int) -> None:
    probe = dst
    while not _is_dir(probe) and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(_ext(probe)).free
    except OSError as e:
        print(f"  (could not read the free space of {probe}: {e}; continuing)")
        return
    if need + FREE_SPACE_MARGIN > free:
        raise ShardError(f"not enough free space on {probe}: need {fmt_bytes(need)} (+{fmt_bytes(FREE_SPACE_MARGIN)} "
                         f"margin), free {fmt_bytes(free)}")


def _print_problems(title: str, items: Sequence[Tuple[str, Sequence[str]]]) -> None:
    if not items:
        return
    print(f"  {title}: {len(items)}")
    for stem, probs in items[:MAX_EXAMPLES]:
        print(f"    - {stem}: {'; '.join(list(probs)[:3])}{' ...' if len(probs) > 3 else ''}")
    if len(items) > MAX_EXAMPLES:
        print(f"    ... and {len(items) - MAX_EXAMPLES} more")


# ----------------------------------------------------------------------------------------------------------------------
# pack
# ----------------------------------------------------------------------------------------------------------------------
def cmd_pack(work_dir: str, dest: str, checksum: bool = False) -> int:
    src, dst = Path(work_dir), Path(dest)
    if not _is_dir(src / MANIFESTS):
        raise ShardError(f"{src} is not a preprocessed work folder (no '{MANIFESTS}' folder inside)")
    if _is_dir(dst) and _same_dir(src, dst):
        raise ShardError("--dest must be a different folder than --work-dir")
    check_reachable(dst, "--dest")
    t0 = time.time()
    done, unfinished, orphan = scan_manifests(src)
    print(f"[pack] {os.path.abspath(src)} -> {os.path.abspath(dst)}")
    print(f"[pack] finished videos: {len(done)}, unfinished (no .done yet): {len(unfinished)}")
    plans: List[Plan] = []
    broken: List[Tuple[str, List[str]]] = []
    for stem in done:
        v = load_video(src, stem)
        if v.problems:
            broken.append((stem, v.problems))
            continue
        plans.append(plan_video(src, dst, v, False, checksum))
    need = sum(p.nbytes for p in plans)
    total = sum(sum(p.video.sizes.values()) for p in plans)
    print(f"[pack] to copy: {sum(len(p.todo) for p in plans)} files, {fmt_bytes(need)} "
          f"(part total {fmt_bytes(total)}; files already there with the same size are skipped)")
    if need:
        check_free_space(dst, need)
    os.makedirs(_ext(dst / MANIFESTS), exist_ok=True)

    n_files = n_bytes = n_new = 0
    for i, p in enumerate(plans, 1):
        f, b = transfer(p, dst, checksum)
        n_files += f
        n_bytes += b
        if f:
            n_new += 1
            el = max(time.time() - t0, 1e-6)
            print(f"  [{i}/{len(plans)}] {p.video.stem}: {len(p.video.rows)} utts, {f} files, {fmt_bytes(b)} "
                  f"| {fmt_bytes(n_bytes / el)}/s", flush=True)
    src_log = read_log_lines(src)
    have = set(read_log_lines(dst))
    new_log = [ln for ln in dict.fromkeys(src_log) if ln not in have]
    append_log_lines(dst, new_log)

    bad = [(p.video, verify_video(dst, p.video)) for p in plans]
    bad = [(v.stem, unmark_failed(dst, v, b)) for v, b in bad if b]
    n_ok = len(plans) - len(bad)
    n_feat = sum(len(p.video.files) for p in plans)
    n_dst_done = len(set(scan_manifests(dst)[0]) & {p.video.stem for p in plans})
    print(f"[pack] copied {n_files} files ({fmt_bytes(n_bytes)}) for {n_new} videos, "
          f"{len(plans) - n_new} videos were already there, log lines added: {len(new_log)}, "
          f"{time.time() - t0:.0f} s")
    print(f"[pack] verify: {n_ok}/{len(plans)} videos complete with the right sizes, {n_dst_done}/{len(plans)} "
          f"finished markers ({n_feat} feature files, {fmt_bytes(total)})")
    _print_problems("videos with missing/empty files in the SOURCE (not packed; re-run preprocessing)", broken)
    _print_problems("videos that failed verification in the destination (run pack again)", bad)
    if unfinished:
        print(f"  note: {len(unfinished)} unfinished videos (e.g. {unfinished[0]}) were not packed - run the "
              f"preprocessing again to finish them, then pack again")
    if orphan:
        _print_problems(".done markers without a manifest (not packed)", [(s, ["no manifest"]) for s in orphan])
    ok = not broken and not bad and not orphan
    print(f"[pack] {'OK' if ok else 'finished WITH PROBLEMS (see above)'}: {dst}")
    if ok:
        print("[pack] next: bring this folder to the main PC and run 3_merge_parts.bat there "
              "(or: tools/shards.py merge --src <this folder>)")
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class SourceStats:
    root: Path
    videos: int = 0
    merged: int = 0
    already: int = 0
    duplicate: int = 0
    broken: int = 0
    unfinished: int = 0
    files: int = 0
    nbytes: int = 0


def split_src_args(values: Sequence[str]) -> List[str]:
    """--src values; each may also hold several folders separated by ';' (quotes and blanks are stripped)."""
    out: List[str] = []
    for v in values:
        for part in str(v).split(";"):
            part = part.strip().strip('"').strip()
            if part:
                out.append(part)
    return out


def cmd_merge(srcs: Sequence[str], work_dir: str, overwrite: bool = False, checksum: bool = False,
              dry_run: bool = False) -> int:
    dst = Path(work_dir)
    t0 = time.time()
    roots: List[Path] = []
    for s in split_src_args(srcs):
        for r in shard_roots(Path(s)):
            if _is_dir(dst) and _same_dir(r, dst):
                print(f"[merge] skipping {r}: it is the destination work folder itself")
            elif not any(_same_dir(r, q) for q in roots):
                roots.append(r)
    if not roots:
        raise ShardError("nothing to merge (every source folder is the destination itself)")
    check_reachable(dst, "--work-dir")
    dst_done = set(scan_manifests(dst)[0])
    print(f"[merge] destination: {dst.resolve()} ({len(dst_done)} finished videos already there)")
    print("[merge] parts: " + ", ".join(str(r) for r in roots))

    stats: List[SourceStats] = []
    plans: List[Tuple[SourceStats, Plan]] = []
    broken: List[Tuple[str, List[str]]] = []
    taken: Dict[str, Path] = {}
    dst_health: Dict[str, List[str]] = {}   # finished videos of the work dir -> their missing/empty files
    repaired: List[Tuple[str, List[str]]] = []
    for root in roots:
        st = SourceStats(root)
        stats.append(st)
        done, unfinished, _orphan = scan_manifests(root)
        st.videos, st.unfinished = len(done), len(unfinished)
        logs: Dict[str, List[str]] = defaultdict(list)
        for ln in read_log_lines(root):
            rec = _log_record(ln)
            if rec and rec.get("stem"):
                logs[str(rec["stem"])].append(ln)
        for stem in done:
            if stem in taken:
                st.duplicate += 1
                continue
            repair = False
            if stem in dst_done and not overwrite:
                if stem not in dst_health:
                    dst_health[stem] = load_video(dst, stem).problems
                if not dst_health[stem]:
                    st.already += 1
                    continue
                repair = True  # finished in the work dir but files are missing/empty there: replace it from this part
            v = load_video(root, stem)
            if v.problems:
                st.broken += 1
                broken.append((f"{stem} (in {root})", v.problems))
                continue
            taken[stem] = root
            if repair:
                repaired.append((stem, dst_health[stem]))
            plan = plan_video(root, dst, v, overwrite or repair, checksum)
            plan.log_lines = logs.get(stem, [])
            plans.append((st, plan))
    need = sum(p.nbytes for _, p in plans)
    print(f"[merge] videos to merge: {len(plans)}, files to copy: {sum(len(p.todo) for _, p in plans)}, "
          f"{fmt_bytes(need)}")
    if dry_run:
        _merge_table(stats, planned={id(st): sum(1 for s, _ in plans if s is st) for st in stats})
        _print_problems("finished videos of the work folder with missing/empty files (would be replaced from a part)",
                        repaired)
        _print_problems("videos with missing/empty files in the part (would be skipped)", broken)
        print("[merge] dry run: nothing copied")
        return 0
    if need:
        check_free_space(dst, need)
    os.makedirs(_ext(dst / MANIFESTS), exist_ok=True)

    bad: List[Tuple[str, List[str]]] = []
    for i, (st, p) in enumerate(plans, 1):
        f, b = transfer(p, dst, checksum)
        st.files += f
        st.nbytes += b
        st.merged += 1
        miss = verify_video(dst, p.video)
        if miss:
            bad.append((p.video.stem, unmark_failed(dst, p.video, miss)))
        if f:
            el = max(time.time() - t0, 1e-6)
            print(f"  [{i}/{len(plans)}] {p.video.stem}: {len(p.video.rows)} utts, {f} files, {fmt_bytes(b)} "
                  f"| {fmt_bytes(sum(s.nbytes for s in stats) / el)}/s", flush=True)
    _merge_table(stats)
    _print_problems("finished videos of the work folder that had missing/empty files (replaced from a part)", repaired)
    _print_problems("videos with missing/empty files in the part (NOT merged; re-run that PC's preprocessing and "
                    "pack again)", broken)
    _print_problems("videos that failed verification after copying (run merge again)", bad)
    print(f"[merge] {time.time() - t0:.0f} s")
    print()
    status_ok = print_status(collect_status(dst))
    ok = not broken and not bad and status_ok
    print(f"[merge] {'OK' if ok else 'finished WITH PROBLEMS (see above)'}")
    return 0 if ok else 1


def _merge_table(stats: Sequence[SourceStats], planned: Optional[Dict[int, int]] = None) -> None:
    head = "to merge" if planned is not None else "merged"
    print(f"\n  {'part folder':<48} {'videos':>7} {head:>8} {'already':>8} {'dupl.':>6} {'broken':>7} "
          f"{'unfin.':>7} {'copied':>10}")
    for st in stats:
        name = str(st.root)
        name = name if len(name) <= 48 else "..." + name[-45:]
        n = planned.get(id(st), 0) if planned is not None else st.merged
        print(f"  {name:<48} {st.videos:>7} {n:>8} {st.already:>8} {st.duplicate:>6} {st.broken:>7} "
              f"{st.unfinished:>7} {fmt_bytes(st.nbytes):>10}")
    print("  (already = finished in the work folder before, skipped; dupl. = same video in an earlier part; "
          "unfin. = not finished on that PC, not copied)\n")


# ----------------------------------------------------------------------------------------------------------------------
# status
# ----------------------------------------------------------------------------------------------------------------------
def _deep_problem(path: Path, kind: str) -> Optional[str]:
    """Content check: npz = valid zip (CRC) with the expected arrays; mp4 = has an ftyp box."""
    try:
        if kind == "npz":
            with zipfile.ZipFile(_ext(path)) as z:
                if z.testzip() is not None:
                    return "npz CRC error"
                if not NPZ_MEMBERS <= set(z.namelist()):
                    return "npz lacks arrays " + ",".join(sorted(NPZ_MEMBERS - set(z.namelist())))
        else:
            with open(_ext(path), "rb") as f:
                if f.read(12)[4:8] != b"ftyp":
                    return "not an mp4 (no ftyp box)"
    except (OSError, zipfile.BadZipFile, EOFError) as e:
        return f"unreadable ({type(e).__name__})"
    return None


def collect_status(root: Path, deep: bool = False) -> dict:
    """Read-only summary of a work/part folder."""
    root = Path(root)
    if not _is_dir(root / MANIFESTS):
        raise ShardError(f"{root} is not a preprocessed work folder (no '{MANIFESTS}' folder inside)")
    done, unfinished, orphan = scan_manifests(root)
    spk: Dict[str, dict] = defaultdict(lambda: {"videos": 0, "utts": 0, "seconds": 0.0, "angles": set(), "has_unk": 0})
    split_dirs: Counter = Counter()
    missing: List[Tuple[str, str]] = []
    bad_manifests: List[Tuple[str, List[str]]] = []
    deep_bad: List[Tuple[str, str]] = []
    n_utts = n_files = n_unk = 0
    nbytes = 0
    seconds = 0.0
    for stem in done:
        try:
            rows = read_rows(root / MANIFESTS / f"{stem}.jsonl")
        except (OSError, ValueError, UnicodeDecodeError) as e:
            bad_manifests.append((stem, [str(e)]))
            continue
        if not rows:
            bad_manifests.append((stem, ["manifest has no utterances"]))
            continue
        speaker = str(rows[0].get("speaker", "?"))
        spk[speaker]["videos"] += 1
        for r in rows:
            n_utts += 1
            s = spk[str(r.get("speaker", speaker))]
            s["utts"] += 1
            dur = float(r.get("duration", 0.0) or 0.0)
            s["seconds"] += dur
            seconds += dur
            s["angles"].add(str(r.get("angle", "?")))
            unk = bool(r.get("has_unk", False))
            s["has_unk"] += int(unk)
            n_unk += int(unk)
            split_dirs[str(r.get("split_dir", ""))] += 1
            uid = str(r.get("utt_id", stem))
            for key in FILE_KEYS:
                try:
                    rel = row_file(r, key)
                except ValueError as e:
                    missing.append((uid, str(e)))
                    continue
                size = _size(root / rel)
                if not size:
                    missing.append((uid, f"{rel} {'missing' if size is None else 'is empty'}"))
                    continue
                n_files += 1
                nbytes += size
                if deep:
                    prob = _deep_problem(root / rel, "npz" if key == "npz" else "mp4")
                    if prob:
                        deep_bad.append((uid, f"{rel}: {prob}"))
    last: Dict[str, dict] = {}
    n_log = 0
    for ln in read_log_lines(root):
        rec = _log_record(ln)
        if rec and rec.get("stem"):
            n_log += 1
            last[str(rec["stem"])] = rec
    done_set = set(done)
    failed = sorted((s, str(r.get("error", "?"))[:160]) for s, r in last.items() if not r.get("ok") and s not in done_set)
    known = done_set | set(unfinished)
    partial = sorted(d.name for d in _subdirs(root / "feats") if d.name not in known)
    return {
        "root": str(root), "done": len(done), "unfinished": unfinished, "orphan_done": orphan, "partial": partial,
        "utterances": n_utts, "has_unk": n_unk, "hours": seconds / 3600.0, "files": n_files, "bytes": nbytes,
        "missing": missing, "bad_manifests": bad_manifests, "deep": deep, "deep_bad": deep_bad,
        "speakers": {k: {**v, "angles": sorted(v["angles"])} for k, v in sorted(spk.items())},
        "split_dirs": dict(split_dirs), "log_entries": n_log, "failed_not_done": failed,
    }


def print_status(st: dict, brief: bool = False) -> bool:
    """Print a status summary; returns True when nothing is missing or broken."""
    print(f"[status] {os.path.abspath(st['root'])}")
    print(f"  finished videos: {st['done']}   unfinished (manifest without .done): {len(st['unfinished'])}   "
          f".done without manifest: {len(st['orphan_done'])}")
    print(f"  utterances: {st['utterances']} (has_unk {st['has_unk']}), {st['hours']:.1f} h over all camera angles, "
          f"feature files: {st['files']} ({fmt_bytes(st['bytes'])})")
    print(f"  missing/empty feature files: {len(st['missing'])}")
    for uid, what in st["missing"][:MAX_EXAMPLES]:
        print(f"    - {uid}: {what}")
    if len(st["missing"]) > MAX_EXAMPLES:
        print(f"    ... and {len(st['missing']) - MAX_EXAMPLES} more")
    _print_problems("unreadable manifests", st["bad_manifests"])
    if st["deep"]:
        print(f"  deep content check: {len(st['deep_bad'])} bad files")
        for uid, what in st["deep_bad"][:MAX_EXAMPLES]:
            print(f"    - {uid}: {what}")
    if st["orphan_done"]:
        _print_problems(".done markers without a manifest", [(s, ["no manifest"]) for s in st["orphan_done"]])
    if st["unfinished"]:
        print(f"  unfinished videos (e.g. {st['unfinished'][0]}): still running, or interrupted - run the "
              f"preprocessing again to finish them")
    if st["partial"]:
        print(f"  interrupted videos (feature folder without manifest, e.g. {st['partial'][0]}): {len(st['partial'])} "
              f"- not used and not copied; the next preprocessing run redoes them")
    if not brief:
        print(f"\n  {'speaker':<8} {'videos':>6} {'utts':>6} {'hours':>6} {'unk':>5}  angles")
        for name, s in st["speakers"].items():
            print(f"  {name:<8} {s['videos']:>6} {s['utts']:>6} {s['seconds'] / 3600:>6.2f} {s['has_unk']:>5}  "
                  f"{','.join(s['angles'])}")
        print(f"\n  split folders: " + ", ".join(f"{k or '(none)'} {v}" for k, v in sorted(st["split_dirs"].items())))
        print(f"  preprocess log: {st['log_entries']} entries, failed and not finished: {len(st['failed_not_done'])}")
        for stem, err in st["failed_not_done"][:MAX_EXAMPLES]:
            print(f"    - {stem}: {err}")
    ok = not st["missing"] and not st["bad_manifests"] and not st["orphan_done"] and not st["deep_bad"]
    print(f"  status: {'OK' if ok else 'PROBLEMS FOUND'}")
    return ok


def cmd_status(work_dir: str, deep: bool = False) -> int:
    return 0 if print_status(collect_status(Path(work_dir), deep)) else 1


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    ap = argparse.ArgumentParser(description="pack / merge / check preprocessed parts made on several PCs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="copy the finished videos of a work folder into a transfer folder (USB / share)")
    p.add_argument("--work-dir", required=True, help="the part's work folder, e.g. work_shard2")
    p.add_argument("--dest", required=True, help="transfer folder, e.g. E:\\avsr_parts\\avsr_part2")
    p.add_argument("--checksum", action="store_true", help="also compare SHA-1 of every file (slower)")
    m = sub.add_parser("merge", help="copy parts into the main work folder")
    m.add_argument("--src", action="append", required=True,
                   help="part folder, or a folder containing part folders (repeatable; ';' separates several)")
    m.add_argument("--work-dir", default="work", help="main work folder (default: work)")
    m.add_argument("--overwrite", action="store_true", help="replace videos that are already finished in the work folder")
    m.add_argument("--checksum", action="store_true", help="also compare SHA-1 of every file (slower)")
    m.add_argument("--dry-run", action="store_true", help="only show what would be copied")
    s = sub.add_parser("status", help="finished videos, utterances, missing files, per-speaker counts")
    s.add_argument("--work-dir", default="work")
    s.add_argument("--deep", action="store_true", help="also check file contents (npz CRC, mp4 header; slower)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "pack":
            return cmd_pack(args.work_dir, args.dest, args.checksum)
        if args.cmd == "merge":
            return cmd_merge(args.src, args.work_dir, args.overwrite, args.checksum, args.dry_run)
        return cmd_status(args.work_dir, args.deep)
    except ShardError as e:
        print(f"[{args.cmd}] ERROR: {e}", flush=True)
        return 2
    except OSError as e:  # e.g. permission denied, device removed while listing: clear message, resumable
        print(f"[{args.cmd}] ERROR: {type(e).__name__}: {e}\n  Fix the cause and run the same command again to resume "
              f"(finished files are skipped).", flush=True)
        return 1
    except KeyboardInterrupt:
        print(f"\n[{args.cmd}] interrupted - run the same command again to resume (finished files are skipped)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
