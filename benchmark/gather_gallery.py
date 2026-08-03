#!/usr/bin/env python3
"""Gather t2i/edit/video samples into one classified gallery folder.
Layout: gallery-<date>/<job>/<system>/<model>__<config>.<png|gif>
avi videos -> gif (imageio, no ffmpeg needed). Only server paths reported; no upload.
"""
import os
import re
import shutil
from pathlib import Path

BENCH = Path("/home/zhangyichen/work/edge-dit.cpp/benchmark")
OUT = BENCH / "gallery-20260803"
JOBS = ["t2i", "edit", "video"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VID_EXT = {".avi", ".mp4"}

# results/<job>/<system>/<model-task>/<config>/samples/<system>/output_NNN.ext
RUN_RE = re.compile(r"results/(?P<job>[^/]+)/(?P<system>[^/]+)/(?P<modeltask>[^/]+)/(?P<config>[^/]+)/samples/")


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "_", s)


def classify(path: Path):
    m = RUN_RE.search(str(path))
    if not m:
        return None
    job = m.group("job")
    system = m.group("system")
    modeltask = m.group("modeltask")
    config = m.group("config")
    # strip trailing -<task> suffix from modeltask for a cleaner model name
    model = re.sub(r"-(text-to-image|image-editing|text-to-video)$", "", modeltask)
    stem = path.stem  # output_000
    idx = stem.replace("output_", "p") if stem.startswith("output_") else stem
    base = f"{sanitize(model)}__{sanitize(config)}__{sanitize(idx)}"
    return job, system, base


def avi_to_gif(src: Path, dst: Path) -> bool:
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    try:
        import cv2
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return False
        duration = max(1.0 / float(fps), 0.02)
        imageio.mimsave(str(dst), frames, duration=duration, loop=0)
        return True
    except Exception as e:
        print(f"  [gif-fail] {src}: {e}")
        return False


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    n_img = 0
    n_gif = 0
    n_fail = 0
    per_job = {}
    for job in JOBS:
        root = BENCH / "results" / job
        if not root.exists():
            continue
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext not in IMG_EXT and ext not in VID_EXT:
                continue
            info = classify(f)
            if not info:
                continue
            j, system, base = info
            dst_dir = OUT / j / system
            dst_dir.mkdir(parents=True, exist_ok=True)
            per_job.setdefault(j, {}).setdefault(system, 0)
            if ext in IMG_EXT:
                dst = dst_dir / f"{base}{ext}"
                shutil.copy2(f, dst)
                n_img += 1
                per_job[j][system] += 1
            else:
                dst = dst_dir / f"{base}.gif"
                if avi_to_gif(f, dst):
                    n_gif += 1
                    per_job[j][system] += 1
                else:
                    n_fail += 1
    print("=" * 60)
    print(f"gallery -> {OUT}")
    print(f"images copied: {n_img}   videos->gif: {n_gif}   gif-fail: {n_fail}")
    print("-" * 60)
    for j in sorted(per_job):
        total = sum(per_job[j].values())
        systems = ", ".join(f"{s}:{c}" for s, c in sorted(per_job[j].items()))
        print(f"  {j}: {total}   ({systems})")
    print("=" * 60)


if __name__ == "__main__":
    main()
