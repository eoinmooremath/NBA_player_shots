#!/usr/bin/env python3
# scripts/nightly_build.py
#
# Nightly pipeline:
#   1) Download specific files from Kaggle dataset into workdir/
#   2) Run build.py to produce:
#        - player_index.parquet  (repo root)
#        - hex_cache/            (repo root)
#      while keeping large intermediate parquet outputs inside workdir/
#
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


NEEDED_FILES = [
    "PlayByPlay.parquet",
    "Games.csv",
    "Players.csv",
]


def _print(*a):
    print(*a, flush=True)


def ensure_clean_dir(p: Path, fresh: bool):
    p.mkdir(parents=True, exist_ok=True)
    if fresh:
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)


def kaggle_auth() -> KaggleApi:
    # KaggleApi.authenticate() looks for ~/.kaggle/kaggle.json by default
    api = KaggleApi()
    api.authenticate()
    return api


def download_one(api: KaggleApi, dataset: str, file_name: str, work: Path, force: bool) -> Path:
    """
    Kaggle API downloads a ZIP for a single file: <file_name>.zip
    We'll unzip it into work/ and delete the zip.
    """
    _print(f"Downloading {file_name} from {dataset} ...")
    api.dataset_download_file(
        dataset=dataset,
        file_name=file_name,
        path=str(work),
        force=force,
        quiet=False,
    )

    zip_path = work / f"{file_name}.zip"
    if not zip_path.exists():
        # Sometimes Kaggle may name it slightly differently; fall back to searching.
        zips = sorted(work.glob("*.zip"))
        raise FileNotFoundError(f"Expected {zip_path} but not found. Found zips: {[z.name for z in zips]}")

    # Unzip into work/
    shutil.unpack_archive(str(zip_path), str(work))
    zip_path.unlink(missing_ok=True)

    out_path = work / file_name
    if not out_path.exists():
        # Some zips contain nested folders; search.
        hits = list(work.rglob(file_name))
        if not hits:
            raise FileNotFoundError(f"Unzipped but did not find {file_name} under {work}")
        # Move first hit to expected location
        shutil.move(str(hits[0]), str(out_path))

    _print(f"  -> OK: {out_path} ({out_path.stat().st_size/1_048_576:.1f} MB)")
    return out_path


def run_build(repo_root: Path, work: Path, pbp: Path, games: Path, players: Path, force_cache: bool):
    """
    Runs your repo's build.py from repo root so imports work (nba_shots package, etc).
    Writes large intermediates into work/ and the small runtime artifacts into repo root.
    """
    clean_out = work / "nba_shot_data_cleaned.parquet"
    final_out = work / "nba_shot_data_final.parquet"

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
        str(repo_root / "player_index.parquet"),
        "--cache_dir",
        str(repo_root / "hex_cache"),
    ]
    if force_cache:
        cmd.append("--force_cache")

    _print("\nRunning build.py:")
    _print("  " + " ".join(cmd))

    subprocess.run(cmd, cwd=str(repo_root), check=True)

    # Sanity checks
    meta = repo_root / "hex_cache" / "_meta.json"
    grid = repo_root / "hex_cache" / "_grid.parquet"
    pidx = repo_root / "player_index.parquet"
    if not meta.exists():
        raise FileNotFoundError(f"Expected {meta} to exist after build.")
    if not grid.exists():
        raise FileNotFoundError(f"Expected {grid} to exist after build.")
    if not pidx.exists():
        raise FileNotFoundError(f"Expected {pidx} to exist after build.")

    _print("\nBuild outputs:")
    _print(f"  player_index.parquet: {pidx.stat().st_size/1_048_576:.1f} MB")
    _print(f"  hex_cache/:           {sum(p.stat().st_size for p in (repo_root/'hex_cache').rglob('*') if p.is_file())/1_048_576:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="owner/slug on Kaggle")
    ap.add_argument("--workdir", default="work", help="temp working directory (NOT committed)")
    ap.add_argument("--fresh", action="store_true", help="delete workdir before starting")
    ap.add_argument("--force_download", action="store_true", help="force Kaggle to re-download files")
    ap.add_argument("--force_cache", action="store_true", help="force rebuild of hex_cache/")
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    work = (repo_root / args.workdir).resolve()

    _print("=== nightly_build.py ===")
    _print("Repo root:", repo_root)
    _print("Workdir:  ", work)
    _print("Dataset:  ", args.dataset)
    _print(f"Options: fresh={args.fresh} force_download={args.force_download} force_cache={args.force_cache}")

    ensure_clean_dir(work, fresh=args.fresh)

    api = kaggle_auth()

    paths = {}
    for fn in NEEDED_FILES:
        paths[fn] = download_one(api, args.dataset, fn, work, force=args.force_download)

    run_build(
        repo_root=repo_root,
        work=work,
        pbp=paths["PlayByPlay.parquet"],
        games=paths["Games.csv"],
        players=paths["Players.csv"],
        force_cache=args.force_cache,
    )

    _print("\nDone.")


if __name__ == "__main__":
    main()
