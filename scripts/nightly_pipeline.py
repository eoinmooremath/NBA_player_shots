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
    candidate = path_or_dir / filename
    return candidate


def download_kaggle_files(dataset: str, workdir: Path, force: bool = True) -> dict[str, Path]:
    """
    Downloads only the needed files into workdir using Kaggle API Token auth.
    """
    workdir.mkdir(parents=True, exist_ok=True)

    # Fail fast if token isn't present (recommended path for GitHub Actions).
    # kagglehub also supports ~/.kaggle/access_token. :contentReference[oaicite:2]{index=2}
    if not os.environ.get("KAGGLE_API_TOKEN") and not (Path.home() / ".kaggle" / "access_token").exists():
        raise SystemExit(
            "Missing Kaggle auth. Set KAGGLE_API_TOKEN (recommended) or provide ~/.kaggle/access_token."
        )

    out = {}
    for fname in NEEDED_FILES:
        # kagglehub supports downloading a dataset or a specific file via `path=...`. :contentReference[oaicite:3]{index=3}
        p = kagglehub.dataset_download(
            dataset,
            path=fname,
            output_dir=str(workdir),
            force_download=force,
        )
        p = _ensure_file(Path(p), fname)
        if not p.exists():
            raise SystemExit(f"Download failed: expected {p} to exist for {fname}")
        out[fname] = p

    return out


def run_build(repo_root: Path, workdir: Path, debug: bool = False) -> None:
    """
    Calls your repo's build.py to produce:
      - hex_cache/
      - player_index.parquet
      - (optionally) nba_shot_data_final.parquet
    Adjust CLI flags here to match your build.py.
    """
    build_py = repo_root / "build.py"
    if not build_py.exists():
        raise SystemExit(f"Missing {build_py}. Put your orchestrator at repo root or update this script.")

    # Common outputs (what your Streamlit app expects)
    hex_cache_dir = repo_root / "hex_cache"
    player_index = repo_root / "player_index.parquet"
    final_out = repo_root / "nba_shot_data_final.parquet"
    clean_out = repo_root / "nba_shot_data_clean.parquet"

    # Ensure we start clean so stale bins don't linger
    if hex_cache_dir.exists():
        shutil.rmtree(hex_cache_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        str(build_py),

        # Inputs downloaded from Kaggle into workdir
        "--pbp", str(workdir / "PlayByPlay.parquet"),
        "--games", str(workdir / "Games.csv"),
        "--players", str(workdir / "Players.csv"),

        # Outputs committed to repo
        "--clean_out", str(clean_out),
        "--final_out", str(final_out),
        "--player_index", str(player_index),
        "--hex_cache_dir", str(hex_cache_dir),

        "--force",
    ]
    if debug:
        cmd.append("--debug")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root))

    # Sanity checks
    if not hex_cache_dir.exists():
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

    # Download required raw files (do NOT commit these)
    download_kaggle_files(args.dataset, workdir, force=args.force_download)

    # Build derived artifacts (DO commit these)
    run_build(repo_root, workdir, debug=args.debug)

    print("Nightly pipeline complete.")


if __name__ == "__main__":
    main()
