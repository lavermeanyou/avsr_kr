"""Tests for tools/shards.py (pack / merge / status) and avsr.preprocess --shard. Synthetic data in temp dirs only.

Run from the project root:  $py -m tests.test_shards   (or  $py tests\\test_shards.py,  $py -m unittest tests.test_shards)
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("avsr_tools_shards", ROOT / "tools" / "shards.py")
shards = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = shards
_spec.loader.exec_module(shards)

from avsr.preprocess import parse_shard, select_shard  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------------
# synthetic parts
# ----------------------------------------------------------------------------------------------------------------------
def make_video(work: Path, stem: str, n_utts: int = 3, seed: int = 0, done: bool = True, ok_log: bool = True) -> None:
    """One preprocessed video in the real layout: feats/<stem>/<utt>.npz|.mp4, manifests/<stem>.jsonl (+ .done)."""
    rng = np.random.default_rng(seed)
    speaker, angle = stem.split("_")[5], stem.split("_")[6]
    (work / "feats" / stem).mkdir(parents=True, exist_ok=True)
    (work / "manifests").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n_utts + 1):
        uid = f"{stem}__{i:03d}"
        t = 10 + i
        np.savez_compressed(work / "feats" / stem / f"{uid}.npz",
                            lm=rng.standard_normal((t, 40, 2)).astype(np.float16),
                            cue=rng.standard_normal((t, 8)).astype(np.float16), valid=np.ones(t, np.uint8),
                            audio=rng.integers(-3000, 3000, t * 533).astype(np.int16), fps=np.float64(30.0),
                            sr=np.int64(16000))
        (work / "feats" / stem / f"{uid}.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + rng.bytes(200 + 17 * i))
        rows.append({"utt_id": uid, "video_stem": stem, "split_dir": "2.Validation", "speaker": speaker, "gender": "F",
                     "age": 3, "specificity": speaker[0], "angle": angle, "session": stem.split("_")[7],
                     "noise_env": 1, "topic": "t", "sentence_id": i, "start": 1.0 * i, "end": 1.0 * i + 1.5,
                     "duration": 1.5, "n_frames": t, "n_samples": t * 533, "fps": 30.0, "time_shift": 0.0,
                     "text": "안녕 하세요", "text_raw": "안녕 하세요", "has_unk": i == 2, "lm_valid_ratio": 1.0,
                     "mouth_mp4": f"feats/{stem}/{uid}.mp4", "npz": f"feats/{stem}/{uid}.npz"})
    with open(work / "manifests" / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if done:
        (work / "manifests" / f"{stem}.done").write_text(json.dumps({"n_utts": n_utts}), encoding="utf-8")
    with open(work / "preprocess_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"stem": stem, "ok": False, "error": "RuntimeError: first try failed"}) + "\n")
        if ok_log:
            f.write(json.dumps({"stem": stem, "ok": True, "n_utts": n_utts}) + "\n")


def snapshot(root: Path) -> Dict[str, int]:
    """relative path -> mtime_ns of every file (to prove that re-runs do not rewrite anything)."""
    out = {}
    for dirpath, _dirs, files in os.walk(shards._ext(root)):
        for name in files:
            p = os.path.join(dirpath, name)
            out[os.path.relpath(p, shards._ext(root))] = os.stat(p).st_mtime_ns
    return out


def run(fn, *args, **kwargs):
    """Call fn with stdout captured; returns (result, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(*args, **kwargs)
    return res, buf.getvalue()


A1, A2, U1 = "lip_J_1_F_03_E220_A_001", "lip_J_1_F_03_E220_C_001", "lip_J_1_F_03_E220_E_001"
B1, B2 = "lip_J_2_M_02_C313_A_001", "lip_J_2_M_02_C313_B_002"


class ShardsToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="avsr_shards_"))
        # spaces, Korean and parentheses in every path, as on the real PCs
        self.pc2 = self.tmp / "PC2 작업 폴더" / "work_shard2"
        self.pc3 = self.tmp / "PC3" / "work_shard3"
        self.usb = self.tmp / "USB 전송 (E)"
        for stem, seed in ((A1, 1), (A2, 2)):
            make_video(self.pc2, stem, seed=seed)
        make_video(self.pc2, U1, seed=3, done=False, ok_log=False)  # still running / interrupted: must not travel
        for stem, seed in ((B1, 4), (B2, 5)):
            make_video(self.pc3, stem, seed=seed)

    def tearDown(self) -> None:
        path = os.path.abspath(self.tmp)
        shutil.rmtree("\\\\?\\" + path if os.name == "nt" else path, ignore_errors=True)  # also removes > 260-char paths

    def pack_both(self) -> None:
        rc, out = run(shards.cmd_pack, str(self.pc2), str(self.usb / "avsr_part2"))
        self.assertEqual(rc, 0, out)
        self.assertIn("verify: 2/2 videos complete", out)
        rc, out = run(shards.cmd_pack, str(self.pc3), str(self.usb / "avsr_part3"))
        self.assertEqual(rc, 0, out)

    def test_pack_is_complete_resumable_and_skips_identical(self) -> None:
        self.pack_both()
        dest = self.usb / "avsr_part2"
        self.assertTrue((dest / "manifests" / f"{A1}.done").is_file())
        self.assertFalse((dest / "manifests" / f"{U1}.jsonl").exists(), "unfinished video must not be packed")
        self.assertEqual(len(list((dest / "feats").rglob("*.npz"))), 6)
        self.assertEqual(len(list((dest / "feats").rglob("*.mp4"))), 6)
        src_log = (self.pc2 / "preprocess_log.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual((dest / "preprocess_log.jsonl").read_text(encoding="utf-8").splitlines(), src_log)
        for rel in ("manifests/%s.jsonl" % A1, "feats/%s/%s__001.npz" % (A2, A2)):
            self.assertEqual((dest / rel).read_bytes(), (self.pc2 / rel).read_bytes())

        before = snapshot(dest)
        rc, out = run(shards.cmd_pack, str(self.pc2), str(dest))
        self.assertEqual(rc, 0, out)
        self.assertIn("copied 0 files", out)
        self.assertEqual(snapshot(dest), before, "identical files must be skipped, log lines not duplicated")

        # an interrupted/damaged copy on the USB stick: only that file is copied again, the marker is rewritten last
        bad = dest / "feats" / A1 / f"{A1}__002.npz"
        bad.write_bytes(b"truncated")
        (dest / "feats" / A1 / f"{A1}__003.mp4.part").write_bytes(b"leftover of an interrupted copy")
        rc, out = run(shards.cmd_pack, str(self.pc2), str(dest))
        self.assertEqual(rc, 0, out)
        self.assertIn("copied 2 files", out)  # the npz + the .done marker
        self.assertEqual(bad.read_bytes(), (self.pc2 / "feats" / A1 / f"{A1}__002.npz").read_bytes())
        after = snapshot(dest)
        changed = sorted(k for k in after if before.get(k) != after[k])
        self.assertEqual({Path(k).name for k in changed},
                         {f"{A1}__002.npz", f"{A1}.done", f"{A1}__003.mp4.part"}, changed)

    def test_merge_into_empty_work_dir_and_status(self) -> None:
        self.pack_both()
        work = self.tmp / "main pc" / "work"
        rc, out = run(shards.cmd_merge, [str(self.usb)], str(work))  # parent folder: finds avsr_part2 + avsr_part3
        self.assertEqual(rc, 0, out)
        self.assertIn("status: OK", out)
        st = shards.collect_status(work)
        self.assertEqual(st["done"], 4)
        self.assertEqual(st["utterances"], 12)
        self.assertEqual(st["files"], 24)
        self.assertEqual(st["has_unk"], 4)
        self.assertEqual(st["missing"], [])
        self.assertEqual(st["unfinished"], [])
        self.assertEqual({k: v["videos"] for k, v in st["speakers"].items()}, {"C313": 2, "E220": 2})
        self.assertEqual(st["speakers"]["E220"]["angles"], ["A", "C"])
        for stem in (A1, A2, B1, B2):
            self.assertEqual((work / "manifests" / f"{stem}.jsonl").read_bytes(),
                             ((self.pc2 if stem in (A1, A2) else self.pc3) / "manifests" / f"{stem}.jsonl").read_bytes())
        log = [json.loads(x) for x in (work / "preprocess_log.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted({r["stem"] for r in log}), sorted([A1, A2, B1, B2]))
        self.assertEqual(len(log), 8)
        self.assertEqual(st["failed_not_done"], [])

        # the merged folder is readable by the training code
        from avsr.dataset import load_manifests
        rows = load_manifests(work)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all((Path(r["work_dir"]) / r["npz"]).is_file() for r in rows))

        before = snapshot(work)
        rc, out = run(shards.cmd_merge, [str(self.usb)], str(work))
        self.assertEqual(rc, 0, out)
        self.assertIn("videos to merge: 0", out)
        self.assertEqual(snapshot(work), before)

    def test_merge_into_work_dir_that_already_has_a_video(self) -> None:
        self.pack_both()
        work = self.tmp / "work"
        make_video(work, A1, n_utts=2, seed=99)  # the main PC processed this video itself (different result)
        own = (work / "manifests" / f"{A1}.jsonl").read_bytes()
        rc, out = run(shards.cmd_merge, [str(self.usb / "avsr_part2"), str(self.usb / "avsr_part3")], str(work))
        self.assertEqual(rc, 0, out)
        self.assertEqual((work / "manifests" / f"{A1}.jsonl").read_bytes(), own, "finished video must be kept")
        st = shards.collect_status(work)
        self.assertEqual((st["done"], st["utterances"]), (4, 2 + 3 * 3))
        rows = re_table(out)
        self.assertEqual(rows["avsr_part2"][:4], [2, 1, 1, 0])  # videos, merged, already, duplicate
        self.assertEqual(rows["avsr_part3"][:4], [2, 2, 0, 0])

        rc, out = run(shards.cmd_merge, [f"{self.usb / 'avsr_part2'};{self.pc2}"], str(work), overwrite=True)
        self.assertEqual(rc, 0, out)
        self.assertEqual((work / "manifests" / f"{A1}.jsonl").read_bytes(),
                         (self.pc2 / "manifests" / f"{A1}.jsonl").read_bytes())
        self.assertEqual(re_table(out)[self.pc2.name][3], 2, "same videos in a later part are duplicates")
        self.assertEqual(shards.collect_status(work)["utterances"], 12)

    def test_merge_repairs_damaged_finished_video_and_reruns_failed_verification(self) -> None:
        self.pack_both()
        work = self.tmp / "work"
        rc, out = run(shards.cmd_merge, [str(self.usb)], str(work))
        self.assertEqual(rc, 0, out)
        # damaged after merging (deleted / emptied by hand, disk error): .done is there but files are missing/empty
        gone = work / "feats" / A1 / f"{A1}__002.npz"
        gone.unlink()
        (work / "feats" / A1 / f"{A1}__003.mp4").write_bytes(b"")
        before = snapshot(work)
        rc, out = run(shards.cmd_merge, [str(self.usb)], str(work))
        self.assertEqual(rc, 0, out)
        self.assertIn("had missing/empty files (replaced from a part): 1", out)
        self.assertEqual(gone.read_bytes(), (self.pc2 / "feats" / A1 / f"{A1}__002.npz").read_bytes())
        self.assertEqual(re_table(out)["avsr_part2"][:3], [2, 1, 1])  # videos, merged (the repaired one), already
        ok, status_out = run(shards.print_status, shards.collect_status(work, deep=True), True)
        self.assertTrue(ok, status_out)
        untouched = [k for k in before if A1 not in k and k != "preprocess_log.jsonl"]
        after = snapshot(work)
        self.assertEqual({k: after[k] for k in untouched}, {k: before[k] for k in untouched})

        # a video that fails the verification after copying must not stay "finished": the next run copies it again
        real_verify = shards.verify_video
        work2 = self.tmp / "work2"
        shards.verify_video = lambda root, v: ["feats/x.npz"] if v.stem == B1 else real_verify(root, v)
        try:
            rc, out = run(shards.cmd_merge, [str(self.usb)], str(work2))
        finally:
            shards.verify_video = real_verify
        self.assertEqual(rc, 1, out)
        self.assertNotIn(B1, shards.scan_manifests(work2)[0])
        self.assertEqual(len(shards.scan_manifests(work2)[0]), 3)
        rc, out = run(shards.cmd_merge, [str(self.usb)], str(work2))
        self.assertEqual(rc, 0, out)
        self.assertEqual(re_table(out)["avsr_part3"][:3], [2, 1, 1])
        self.assertEqual(len(shards.scan_manifests(work2)[0]), 4)

    @unittest.skipUnless(os.name == "nt", "drive letters are Windows specific")
    def test_unreachable_destination_is_refused_before_copying(self) -> None:
        free = [c for c in "QRSTUVWXYZ" if not os.path.exists(f"{c}:\\")]
        if not free:
            self.skipTest("no unused drive letter")
        rc, out = run(shards.main, ["pack", "--work-dir", str(self.pc2), "--dest", f"{free[0]}:\\avsr_part2"])
        self.assertEqual(rc, 2, out)
        self.assertIn("not reachable", out)
        rc, out = run(shards.main, ["merge", "--src", str(self.pc2), "--work-dir", f"{free[0]}:\\work"])
        self.assertEqual(rc, 2, out)
        self.assertIn("not reachable", out)

    def test_status_reports_missing_and_corrupt_files(self) -> None:
        work = self.pc3
        st = shards.collect_status(work, deep=True)
        self.assertEqual((st["missing"], st["deep_bad"], st["partial"]), ([], [], []))
        (work / "feats" / "lip_J_2_M_02_C313_C_003").mkdir()  # preprocessing killed mid-video: no manifest yet
        self.assertEqual(shards.collect_status(work)["partial"], ["lip_J_2_M_02_C313_C_003"])
        (work / "feats" / B1 / f"{B1}__002.npz").unlink()
        (work / "feats" / B2 / f"{B2}__001.npz").write_bytes(b"not a zip file at all")
        st = shards.collect_status(work, deep=True)
        self.assertEqual(len(st["missing"]), 1)
        self.assertEqual(st["missing"][0][0], f"{B1}__002")
        self.assertIn(f"feats/{B1}/{B1}__002.npz missing", st["missing"][0][1])
        self.assertEqual([u for u, _ in st["deep_bad"]], [f"{B2}__001"])
        rc, out = run(shards.main, ["status", "--work-dir", str(work)])
        self.assertEqual(rc, 1)
        self.assertIn(f"{B1}__002: feats/{B1}/{B1}__002.npz missing", out)
        self.assertIn("PROBLEMS FOUND", out)
        # a broken video is not packed (and not merged), the healthy one still is
        rc, out = run(shards.cmd_pack, str(work), str(self.usb / "p3"))
        self.assertEqual(rc, 1, out)
        self.assertFalse((self.usb / "p3" / "manifests" / f"{B1}.done").exists())
        self.assertTrue((self.usb / "p3" / "manifests" / f"{B2}.done").exists())
        rc, out = run(shards.cmd_merge, [str(work)], str(self.tmp / "w"))
        self.assertEqual(rc, 1, out)
        self.assertEqual(shards.scan_manifests(self.tmp / "w")[0], [B2])

    def test_refusals_and_errors(self) -> None:
        rc, out = run(shards.main, ["merge", "--src", str(self.pc2), "--work-dir", str(self.pc2)])
        self.assertEqual(rc, 2, out)
        rc, out = run(shards.main, ["merge", "--src", str(self.tmp / "nowhere"), "--work-dir", str(self.tmp / "w")])
        self.assertEqual(rc, 2, out)
        self.assertIn("folder not found", out)
        (self.tmp / "empty").mkdir()
        rc, out = run(shards.main, ["merge", "--src", str(self.tmp / "empty"), "--work-dir", str(self.tmp / "w")])
        self.assertEqual(rc, 2, out)
        self.assertIn("no preprocessed part found", out)
        rc, out = run(shards.main, ["pack", "--work-dir", str(self.tmp / "empty"), "--dest", str(self.usb)])
        self.assertEqual(rc, 2, out)
        rc, out = run(shards.main, ["merge", "--src", str(self.pc2), "--work-dir", str(self.tmp / "w"), "--dry-run"])
        self.assertEqual(rc, 0, out)
        self.assertFalse((self.tmp / "w" / "manifests").exists(), "dry run must not write")
        self.assertEqual(shards.split_src_args(['E:\\a; "F:\\b c" ;', " "]), ["E:\\a", "F:\\b c"])

    @unittest.skipUnless(os.name == "nt", "extended-length paths are Windows specific")
    def test_long_paths(self) -> None:
        deep = self.tmp.joinpath(*(["긴 폴더 이름 " + "x" * 40] * 6))  # > 260 characters
        self.assertGreater(len(str(deep / "avsr_part2" / "feats" / A1 / f"{A1}__001.npz")), 300)
        rc, out = run(shards.cmd_pack, str(self.pc2), str(deep / "avsr_part2"))
        self.assertEqual(rc, 0, out)
        rc, out = run(shards.cmd_merge, [str(deep)], str(self.tmp / "work"), checksum=True)
        self.assertEqual(rc, 0, out)
        self.assertEqual(shards.collect_status(self.tmp / "work", deep=True)["utterances"], 6)


def re_table(out: str) -> Dict[str, List[int]]:
    """Parse the merge summary table: part folder name -> [videos, merged, already, duplicate, broken, unfinished]."""
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 9 and all(p.isdigit() for p in parts[-8:-2]):
            rows[Path(" ".join(parts[:-8])).name] = [int(p) for p in parts[-8:-2]]
    return rows


class ShardSelectionTest(unittest.TestCase):
    def test_parse_shard(self) -> None:
        self.assertEqual(parse_shard("2/4"), (2, 4))
        self.assertEqual(parse_shard(" 1/1 "), (1, 1))
        for bad in ("0/4", "5/4", "2-4", "a/b", "2/4/1", ""):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                parse_shard(bad)

    def test_partition_is_disjoint_complete_and_order_independent(self) -> None:
        stems = [f"lip_J_1_F_03_E{i:03d}_{a}_001" for i in range(5) for a in "ACEGI"] + ["lip_J_9_M_01_C999_A_012"]
        jobs = [(SimpleNamespace(stem=s), None) for s in stems]
        parts = {k: [j[0].stem for j in select_shard(jobs, k, 4)] for k in range(1, 5)}
        self.assertEqual(sorted(len(v) for v in parts.values()), [6, 6, 7, 7])
        self.assertEqual(sorted(sum(parts.values(), [])), sorted(stems))
        shuffled = list(jobs)
        random.Random(1).shuffle(shuffled)
        self.assertEqual({k: [j[0].stem for j in select_shard(shuffled, k, 4)] for k in range(1, 5)}, parts)
        self.assertEqual(parts[1][:2], [sorted(stems)[0], sorted(stems)[4]])
        self.assertEqual([j[0].stem for j in select_shard(jobs, 1, 1)], sorted(stems))


if __name__ == "__main__":
    unittest.main(verbosity=2)
