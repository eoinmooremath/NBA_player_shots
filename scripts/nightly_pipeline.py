#!/usr/bin/env python3
# scripts/nightly_pipeline.py

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

LA_TZ = ZoneInfo("America/Los_Angeles")

FILES_TO_DOWNLOAD = [
    "PlayByPlay.parquet",
    "Games.csv",
    "Players.csv",
]

def log(msg: str) -> None:
    print(msg, flush=True)

def run(cmd, cwd: Path | None = None) -> None:
    log(f"$ {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

def ensure_kaggle_token_present() -> None:
    tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not tok:
        raise SystemExit(
            "Missing KAGGLE_API_TOKEN. "
            "Set it as a GitHub Actions secret named KAGGLE_API_TOKEN, "
            "or write it to ~/.kaggle/access_token."
        )

def maybe_skip_if_not_local_time(hhmm: str | None) -> None:
    """
    If hhmm is provided (e.g. '03:30'), exit 0 unless current LA time matches.
    Useful if you schedule multiple UTC cron times to cover DST changes.
    """
    if not hhmm:
        return
    now_la = datetime.now(LA_TZ).strftime("%H:%M")
    if now_la != hhmm:
        log(f"Skipping run: LA time is {now_la}, not {hhmm}.")
        sys.exit(0)

def download_files_from_kagglehub(dataset: str, workdir: Path, fresh: bool) -> None:
    ensure_kaggle_token_present()

    # Import here so dependency errors are clearer in logs
    import kagglehub  # type: ignore

    if fresh and workdir.exists():
        log(f"Cleaning workdir: {workdir}")
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)

    log(f"Downloading from Kaggle dataset: {dataset}")
    for fname in FILES_TO_DOWNLOAD:
        log(f"Downloading {fname} ...")
        # kagglehub downloads to its cache and returns a local path
        src_path = Path(kagglehub.dataset_download(dataset, path=fname))
        if not src_path.exists():
            raise SystemExit(f"Download failed: {fname} not found at {src_path}")

        dst_path = workdir / fname
        shutil.copy2(src_path, dst_path)
        log(f"Saved -> {dst_path} ({dst_path.stat().st_size/1_048_576:.1f} MB)")

def run_build(repo_root: Path, workdir: Path, force_cache: bool, debug: bool) -> None:
    pbp = workdir / "PlayByPlay.parquet"
    games = workdir / "Games.csv"
    players = workdir / "Players.csv"

    if not pbp.exists():
        raise SystemExit(f"Missing {pbp}")
    if not games.exists():
        raise SystemExit(f"Missing {games}")
    if not players.exists():
        raise SystemExit(f"Missing {players}")

    # IMPORTANT: build.py uses --cache_dir (NOT --hex_cache_dir)
    cmd = [
        sys.executable, str(repo_root / "build.py"),
        "--pbp", str(pbp),
        "--games", str(games),
        "--players", str(players),
        "--clean_out", str(repo_root / "nba_shot_data_clean.parquet"),
        "--final_out", str(repo_root / "nba_shot_data_final.parquet"),
        "--player_index", str(repo_root / "player_index.parquet"),
        "--cache_dir", str(repo_root / "hex_cache"),
    ]
    if force_cache:
        cmd.append("--force_cache")
    if debug:
        cmd.append("--debug")

    run(cmd, cwd=repo_root)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="owner/slug, e.g. eoinamoore/historical-nba-data-and-player-box-scores")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--fresh", action="store_true", help="Delete workdir before downloading")
    ap.add_argument("--force_cache", action="store_true", help="Force rebuild hex_cache")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--run_only_at_local", default=None, help="Run only if LA time == HH:MM (e.g. 03:30).")
    args = ap.parse_args()

    maybe_skip_if_not_local_time(args.run_only_at_local)

    repo_root = Path(__file__).resolve().parents[1]
    workdir = (repo_root / args.workdir).resolve()

    log("=== nightly_pipeline.py ===")
    log(f"Repo root: {repo_root}")
    log(f"Workdir  : {workdir}")
    log(f"Dataset  : {args.dataset}")

    download_files_from_kagglehub(args.dataset, workdir, fresh=args.fresh)
    run_build(repo_root, workdir, force_cache=args.force_cache, debug=args.debug)

    log("Done.")

if __name__ == "__main__":
    main()
