#!/usr/bin/env python3
# scripts/nightly_pipeline.py
#
# Nightly:
#   - downloads PlayByPlay.parquet, Games.csv, Players.csv from Kaggle dataset
#   - runs build.py to generate hex_cache/ and player_index.parquet
#   - removes large intermediates (nba_shot_data_clean.parquet, nba_shot_data_final.parquet)
#
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = ["PlayByPlay.parquet", "Games.csv", "Players.csv"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def ensure_kaggle_json() -> Path:
    """
    Kaggle tooling (both CLI + KaggleApi) looks for:
      - env vars KAGGLE_USERNAME / KAGGLE_KEY, OR
      - ~/.kaggle/kaggle.json with {"username": "...", "key": "..."}
    """
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    if not username or not key:
        raise SystemExit(
            "Missing Kaggle credentials. Set GitHub Actions secrets:\n"
            "  - KAGGLE_USERNAME\n"
            "  - KAGGLE_KEY  (you can put a Kaggle API Token here)\n"
        )

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")

    # Kaggle requires strict perms; ignore failures on Windows.
    try:
        os.chmod(kaggle_json, 0o600)
    except Exception:
        pass

    print(f"kaggle.json written to: {kaggle_json}")
    return kaggle_json


def download_from_kaggle(dataset: str, workdir: Path, fresh: bool) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    if fresh and workdir.exists():
        print(f"Cleaning workdir: {workdir}")
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)

    # Use the Kaggle CLI (most reliable across versions, supports --unzip easily)
    # Equivalent to: kaggle datasets download -d <dataset> -f <file> -p <workdir> --unzip --force
    for fn in REQUIRED_FILES:
        run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-f",
                fn,
                "-p",
                str(workdir),
                "--unzip",
                "--force",
            ]
        )

    # Sanity check
    missing = [fn for fn in REQUIRED_FILES if not (workdir / fn).exists()]
    if missing:
        raise SystemExit(f"Download finished but missing files in workdir: {missing}")

    for fn in REQUIRED_FILES:
        p = workdir / fn
        mb = p.stat().st_size / 1_048_576
        print(f"Downloaded: {p.name:20} {mb:8.1f} MB")


def run_build(repo_root: Path, workdir: Path, force_cache: bool, debug: bool) -> None:
    pbp = workdir / "PlayByPlay.parquet"
    games = workdir / "Games.csv"
    players = workdir / "Players.csv"

    clean_out = repo_root / "nba_shot_data_clean.parquet"
    final_out = repo_root / "nba_shot_data_final.parquet"
    player_index = repo_root / "player_index.parquet"
    cache_dir = repo_root / "hex_cache"

    cmd = [
        sys.executable,
        str(repo_root / "build.py"),
        "--pbp",
        str(pbp),
        "--games",
        str(games),
        "--players",
        str(players),
        "--clean_out",
        str(clean_out),
        "--final_out",
        str(final_out),
        "--player_index",
        str(player_index),
        "--cache_dir",
        str(cache_dir),
    ]
    if force_cache:
        cmd.append("--force_cache")
    if debug:
        cmd.append("--debug")

    run(cmd, cwd=repo_root)

    if not cache_dir.exists():
        raise SystemExit("build.py finished, but hex_cache/ is missing.")
    if not player_index.exists():
        raise SystemExit("build.py finished, but player_index.parquet is missing.")

    # DO NOT keep huge intermediates in git
    for p in [clean_out, final_out]:
        if p.exists():
            print(f"Removing intermediate: {p}")
            p.unlink()

    print("Build complete: hex_cache/ and player_index.parquet ready.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        help="Kaggle dataset slug, e.g. eoinamoore/historical-nba-data-and-player-box-scores",
    )
    ap.add_argument("--workdir", default="work", help="Scratch folder for Kaggle downloads")
    ap.add_argument("--fresh", action="store_true", help="Delete workdir before download")
    ap.add_argument("--force_cache", action="store_true", help="Force rebuild hex_cache/")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    workdir = (repo_root / args.workdir).resolve()

    print("=== nightly_pipeline.py ===")
    print("repo_root:", repo_root)
    print("workdir  :", workdir)
    print("dataset  :", args.dataset)

    ensure_kaggle_json()
    download_from_kaggle(args.dataset, workdir, fresh=args.fresh)
    run_build(repo_root, workdir, force_cache=args.force_cache, debug=args.debug)

    print("Done.")


if __name__ == "__main__":
    main()
