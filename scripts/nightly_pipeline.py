#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


NEEDED_FILES = [
    "PlayByPlay.parquet",
    "Games.csv",
    "Players.csv",
]

def now():
    return time.perf_counter()

def run(cmd, cwd=None):
    t0 = now()
    print("\n$", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, cwd=str(cwd) if cwd else None)
    dt_s = now() - t0
    print(f"[OK] step took {dt_s/60:.1f} min")

def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

def kaggle_download_files(dataset: str, work: Path):
    api = KaggleApi()
    api.authenticate()

    # Download just the files we need (avoid pulling the entire dataset zip)
    for fname in NEEDED_FILES:
        print(f"\nDownloading {fname} -> {work / fname}")
        api.dataset_download_file(
            dataset=dataset,
            file_name=fname,
            path=str(work),
            force=True,
            quiet=False,
        )

    # Kaggle API sometimes downloads as "<file>.zip" for single-file downloads; unzip if so.
    # If it already downloaded the raw file, this is a no-op.
    for fname in NEEDED_FILES:
        z = work / f"{fname}.zip"
        if z.exists():
            print(f"Unzipping {z.name} ...")
            shutil.unpack_archive(str(z), str(work))
            z.unlink(missing_ok=True)

    # Sanity check
    for fname in NEEDED_FILES:
        p = work / fname
        if not p.exists():
            raise FileNotFoundError(f"Expected {p} after Kaggle download, but it is missing.")
        print(f"  ✅ {fname:18} {p.stat().st_size/1_048_576:.1f} MB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="owner/slug")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--force", action="store_true", help="Delete and recreate workdir before running")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    work = (repo_root / args.workdir).resolve()

    print("=== nightly_pipeline.py ===")
    print("Repo:    ", repo_root)
    print("Workdir: ", work)
    print("Dataset: ", args.dataset)
    print("Force:   ", args.force)

    if args.force:
        ensure_empty_dir(work)
    else:
        work.mkdir(parents=True, exist_ok=True)

    t_all = now()

    # 1) Download Kaggle inputs
    t0 = now()
    kaggle_download_files(args.dataset, work)
    print(f"[OK] Kaggle download total {((now()-t0)/60):.1f} min")

    pbp = work / "PlayByPlay.parquet"
    games = work / "Games.csv"
    players = work / "Players.csv"

    # 2) Run your build.py
    # IMPORTANT: build.py expects --cache_dir (NOT --hex_cache_dir)
    clean_out = work / "nba_shot_data_clean.parquet"
    final_out = work / "nba_shot_data_final.parquet"

    player_index_out = repo_root / "player_index.parquet"
    cache_dir = repo_root / "hex_cache"

    cmd = [
        sys.executable, str(repo_root / "build.py"),
        "--pbp", str(pbp),
        "--games", str(games),
        "--players", str(players),
        "--clean_out", str(clean_out),
        "--final_out", str(final_out),
        "--player_index", str(player_index_out),
        "--cache_dir", str(cache_dir),
        "--force_cache",
    ]
    if args.debug:
        cmd.append("--debug")

    run(cmd, cwd=repo_root)

    # 3) Optional: remove big intermediates to keep repo clean (we don't commit work/)
    # (hex_cache + player_index.parquet remain)
    try:
        shutil.rmtree(work, ignore_errors=True)
        print(f"[OK] cleaned workdir {work}")
    except Exception as e:
        print("[WARN] failed to clean workdir:", e)

    print(f"\nALL DONE in {((now()-t_all)/60):.1f} min")
    print(f"Artifacts:\n  - {cache_dir}\n  - {player_index_out}")

if __name__ == "__main__":
    main()
