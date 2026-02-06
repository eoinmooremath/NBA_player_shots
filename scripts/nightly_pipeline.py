#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

import kagglehub  # uses KAGGLE_API_TOKEN or ~/.kaggle/access_token


DATASET_DEFAULT = "eoinamoore/historical-nba-data-and-player-box-scores"
NEEDED_FILES = ["PlayByPlay.parquet", "Games.csv", "Players.csv"]


def _ensure_file(path_or_dir: Path, filename: str) -> Path:
    """
    kagglehub.dataset_download sometimes returns a file path, sometimes a directory.
    This makes it robust.
    """
    if path_or_dir.is_file():
        return path_or_dir
    return path_or_dir / filename


def download_kaggle_files(dataset: str, workdir: Path, force: bool = True) -> dict[str, Path]:
    """
    Downloads only the needed files into workdir using Kaggle API Token auth.
    """
    workdir.mkdir(parents=True, exist_ok=True)

    # kagglehub supports KAGGLE_API_TOKEN (recommended) or ~/.kaggle/access_token
    if not os.environ.get("KAGGLE_API_TOKEN") and not (Path.home() / ".kaggle" / "access_token").exists():
        raise SystemExit(
            "Missing Kaggle auth. Set KAGGLE_API_TOKEN (recommended) or provide ~/.kaggle/access_token."
        )

    out: dict[str, Path] = {}
    for fname in NEEDED_FILES:
        print(f"Downloading to {workdir / fname}...")
        p = kagglehub.dataset_download(
            dataset,
            path=fname,                 # download a specific file from the dataset
            output_dir=str(workdir),    # place it in workdir
            force_download=force,
        )
        p = _ensure_file(Path(p), fname)
        if not p.exists():
            raise SystemExit(f"Download failed: expected {p} to exist for {fname}")
        out[fname] = p

    return out


def run_build(repo_root: Path, workdir: Path, debug: bool = False) -> None:
    """
    Calls build.py to produce:
      - cache_dir (hex_cache/)
      - player_index.parquet
      - clean/final parquet (optional, but you already wire these)
    """
    build_py = repo_root / "build.py"
    if not build_py.exists():
        raise SystemExit(f"Missing {build_py}. Put build.py at repo root or update this script.")

    # These match your build.py flags from the usage string in your log
    cache_dir = repo_root / "hex_cache"
    player_index = repo_root / "player_index.parquet"
    final_out = repo_root / "nba_shot_data_final.parquet"
    clean_out = repo_root / "nba_shot_data_clean.parquet"

    # Clean old cache to avoid stale partitions/files
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        str(build_py),

        "--pbp", str(workdir / "PlayByPlay.parquet"),
        "--games", str(workdir / "Games.csv"),
        "--players", str(workdir / "Players.csv"),

        "--clean_out", str(clean_out),
        "--final_out", str(final_out),
        "--player_index", str(player_index),

        "--cache_dir", str(cache_dir),

        "--force_cache",
    ]
    if debug:
        cmd.append("--debug")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root))

    # Sanity checks
    if not cache_dir.exists():
        raise SystemExit("Build failed: hex_cache/ not created.")
    if not player_index.exists():
        raise SystemExit("Build failed: player_index.parquet not created.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--force-download", action="store_true", help="Force re-download from Kaggle")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    workdir = (repo_root / args.workdir).resolve()

    print("Repo:", repo_root)
    print("Work:", workdir)
    print("Dataset:", args.dataset)

    # 1) Download raw files into work/ (do not commit these)
    download_kaggle_files(args.dataset, workdir, force=args.force_download)

    # 2) Build derived artifacts in repo root (commit these)
    run_build(repo_root, workdir, debug=args.debug)

    print("Nightly pipeline complete.")


if __name__ == "__main__":
    main()
