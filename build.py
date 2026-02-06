# build.py
#
# One-command pipeline for the demo:
#   1) Clean shots from raw PBP parquet
#   2) Enrich with Games.csv + Players.csv
#   3) Precompute hex cache for the Streamlit app
#
# Example:
#   python build.py --pbp nba_pbp_v3_backup.parquet --games Games.csv --players Players.csv --force_cache
#
import argparse
import os

from nba_shots.shot_data_cleaning import clean_shot_data
from nba_shots.shot_data_enrichment import normalize_with_local_data
from nba_shots.hex_cache_builder import build_hex_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp", default="nba_pbp_v3_backup.parquet", help="Raw PBP parquet")
    ap.add_argument("--games", default="Games.csv")
    ap.add_argument("--players", default="Players.csv")

    ap.add_argument("--clean_out", default="nba_shot_data_cleaned.parquet")
    ap.add_argument("--final_out", default="nba_shot_data_final.parquet")
    ap.add_argument("--player_index", default="player_index.parquet")

    ap.add_argument("--cache_dir", default="hex_cache")
    ap.add_argument("--gridsize", type=int, default=30)
    ap.add_argument("--legacy_to_plot_x0", type=float, default=887.5)

    ap.add_argument("--no_bounds_filter", action="store_true")
    ap.add_argument("--force_cache", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        print("---- Step 1/3: clean_data ----")
    clean_shot_data(
        input_path=args.pbp,
        output_path=args.clean_out,
        bounds_filter=(not args.no_bounds_filter),
        debug=args.debug,
    )

    if args.debug:
        print("---- Step 2/3: normalize_with_local_data ----")
    normalize_with_local_data(
        shot_data=args.clean_out,
        games_csv=args.games,
        players_csv=args.players,
        output_file=args.final_out,
        player_index_file=args.player_index,
        debug=args.debug,
    )

    if args.debug:
        print("---- Step 3/3: precompute_hex_cache ----")
    build_hex_cache(
        input_path=args.final_out,
        output_dir=args.cache_dir,
        gridsize=args.gridsize,
        legacy_to_plot_x0=args.legacy_to_plot_x0,
        force=args.force_cache,
        debug=args.debug,
    )

    print("✅ Pipeline complete.")
    print(f" - Clean shots: {args.clean_out}")
    print(f" - Final shots: {args.final_out}")
    print(f" - Player index: {args.player_index}")
    print(f" - Hex cache: {args.cache_dir}")


if __name__ == "__main__":
    main()
