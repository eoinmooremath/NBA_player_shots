# shot_data_enrichment.py
import argparse
import os
import polars as pl
import re

def norm_str_expr(col_name: str) -> pl.Expr:
    """Polars expression to normalize string columns (lowercase, strip extra spaces)."""
    return (
        pl.col(col_name)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[\s\-_]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def map_game_type_to_season_phase_expr(col_name: str) -> pl.Expr:
    """
    Polars expression to map gameType to SeasonPhase.
    """
    gt = norm_str_expr(col_name)
    
    return (
        pl.when(gt.is_in(["regular season", "regular"])).then(pl.lit("Regular Season"))
          .when(gt.is_in(["preseason", "pre season"])).then(pl.lit("Preseason"))
          .when(gt.is_in(["playoffs", "playoff"])).then(pl.lit("Playoffs"))
          .when(gt.is_in(["play in tournament", "play in"])).then(pl.lit("Play-In Tournament"))
          # In-Season Tournament / Cup logic
          .when(gt.str.contains("in season") | gt.str.contains("cup") | gt.str.contains("emirates"))
          .then(pl.lit("Regular Season"))
          # Catch-all
          .otherwise(pl.lit("Other"))
    )

def write_player_index(df: pl.DataFrame, player_index_file: str, debug: bool = False) -> None:
    log = print if debug else (lambda *a, **k: None)
    
    # Polars is strict; ensure types before aggregation
    pi = (
        df.lazy()
          .select([
              pl.col("personId").cast(pl.Int64),
              pl.col("Season").cast(pl.Int32),
              pl.col("firstName"),
              pl.col("lastName"),
              pl.col("playerName"),
              pl.col("playerTeamName")
          ])
          .drop_nulls(subset=["personId", "Season"])
    )

    # 1. Calculate Bounds (Min/Max Season)
    bounds = pi.group_by("personId").agg([
        pl.min("Season").alias("minSeason"),
        pl.max("Season").alias("maxSeason")
    ])

    # 2. Get Latest Team
    # Sort by Season desc, take first
    latest_team = (
        pi.sort(["personId", "Season"], descending=[False, True])
          .filter(pl.col("playerTeamName").is_not_null())
          .unique(subset=["personId"], keep="first")
          .select(["personId", pl.col("playerTeamName").alias("latestTeam")])
    )

    # 3. Get Name info (latest)
    name_df = (
        pi.sort(["personId", "Season"], descending=[False, True])
          .unique(subset=["personId"], keep="first")
          .select(["personId", "firstName", "lastName", "playerName"])
          .with_columns(
             (pl.col("firstName").fill_null("") + " " + pl.col("lastName").fill_null(""))
             .str.strip_chars()
             .alias("Full_Name")
          )
    )
    # Fix empty full names
    name_df = name_df.with_columns(
        pl.when(pl.col("Full_Name") == "")
          .then(pl.col("playerName").cast(pl.String))
          .otherwise(pl.col("Full_Name"))
          .alias("Full_Name")
    )

    # 4. Teams Played List
    # Group by person, collect unique teams, join them
    teams_played = (
        pi.filter(pl.col("playerTeamName").is_not_null())
          .sort("Season") # Ensure order
          .group_by("personId")
          # maintain order of appearance (unique_stable)
          .agg(pl.col("playerTeamName").unique(maintain_order=True).alias("teams_list"))
          .with_columns(
              pl.col("teams_list").map_elements(lambda x: " · ".join(x), return_dtype=pl.String).alias("teamsPlayed")
          )
          .select(["personId", "teamsPlayed"])
    )

    # Join everything
    final_index = (
        name_df
        .join(bounds, on="personId", how="left")
        .join(latest_team, on="personId", how="left")
        .join(teams_played, on="personId", how="left")
        .with_columns([
            pl.col("teamsPlayed").fill_null(""),
            pl.col("personId").cast(pl.Int64)
        ])
        .collect()
    )

    final_index.write_parquet(player_index_file)
    log(f"Wrote {player_index_file}: rows={len(final_index):,}")


def normalize_with_local_data(
    shot_data: str,
    games_csv: str,
    players_csv: str,
    output_file: str,
    player_index_file: str,
    debug: bool = False,
) -> None:
    log = print if debug else (lambda *a, **k: None)

    if not os.path.exists(shot_data):
        raise FileNotFoundError(f"Missing {shot_data}")

    log("Scanning datasets (Lazy)...")
    
    # LazyFrames for efficiency
    lf_shots = pl.scan_parquet(shot_data)
    
    # CSVs might have type issues, so we infer or explicitly cast if needed
    # We use infer_schema_length=10000 or 0 (read all) to be safe
    lf_games = pl.scan_csv(games_csv, infer_schema_length=10000)
    lf_players = pl.scan_csv(players_csv, infer_schema_length=10000)

    # ---- Clean / Type IDs ----
    # Ensure join keys are consistent (Strings vs Ints). 
    # Usually GameID is string "00...", personId is Int.
    
    lf_shots = lf_shots.with_columns([
        pl.col("GameID").cast(pl.String).str.strip_chars().str.pad_start(10, "0"),
        pl.col("personId").cast(pl.Int64, strict=False)
    ])
    
    lf_games = lf_games.with_columns([
        pl.col("gameId").cast(pl.String).str.strip_chars().str.pad_start(10, "0").alias("gameId_join")
    ])

    lf_players = lf_players.with_columns([
        pl.col("personId").cast(pl.Int64, strict=False)
    ])

    # ---- Filter Drops ----
    lf_shots = lf_shots.filter(
        pl.col("personId").is_not_null() & pl.col("GameID").is_not_null()
    )

    # ---- Merge Players ----
    # Select only what we need from players
    lf_players_small = lf_players.select(["personId", "firstName", "lastName"])
    
    # Left Join
    lf_merged = lf_shots.join(lf_players_small, on="personId", how="left")
    
    # Drop rows where player mapping failed
    lf_merged = lf_merged.filter(
        pl.col("firstName").is_not_null() & pl.col("lastName").is_not_null()
    )
    
    # Create Full Name
    lf_merged = lf_merged.with_columns(
        (pl.col("firstName") + " " + pl.col("lastName")).str.strip_chars().alias("Full_Name")
    )

    # ---- Merge Games ----
    # Select needed game cols
    game_cols = ["gameId_join", "hometeamId", "awayteamId", "hometeamCity", "hometeamName", 
                 "awayteamCity", "awayteamName", "gameType"]
    # Filter only if they exist in CSV
    valid_game_cols = [c for c in game_cols if c in lf_games.columns]
    
    lf_merged = lf_merged.join(
        lf_games.select(valid_game_cols), 
        left_on="GameID", 
        right_on="gameId_join", 
        how="left"
    )

    # ---- Player Team Logic ----
    # Vectorized logic for home/away
    # Polars `when().then().otherwise()` is perfect here.
    
    lf_merged = lf_merged.with_columns([
        pl.when(pl.col("teamId") == pl.col("hometeamId"))
          .then(pl.col("hometeamCity"))
          .otherwise(pl.col("awayteamCity"))
          .alias("playerTeamCity"),
          
        pl.when(pl.col("teamId") == pl.col("hometeamId"))
          .then(pl.col("hometeamName"))
          .otherwise(pl.col("awayteamName"))
          .alias("playerTeamName"),
          
        pl.when(pl.col("teamId") == pl.col("hometeamId"))
          .then(pl.col("awayteamCity"))
          .otherwise(pl.col("hometeamCity"))
          .alias("opponentTeamCity"),

        pl.when(pl.col("teamId") == pl.col("hometeamId"))
          .then(pl.col("awayteamName"))
          .otherwise(pl.col("hometeamName"))
          .alias("opponentTeamName"),
    ])

    # ---- Season Phase ----
    if "gameType" in lf_merged.columns:
        lf_merged = lf_merged.with_columns(
            map_game_type_to_season_phase_expr("gameType").alias("SeasonPhase")
        )
    else:
        lf_merged = lf_merged.with_columns(pl.lit("Unknown").alias("SeasonPhase"))

    lf_merged = lf_merged.with_columns(
        pl.col("SeasonPhase").is_in(["Playoffs", "Play-In Tournament"]).cast(pl.Int8).alias("IsPostseason")
    )

    # ---- Execute and Save ----
    # We use streaming to handle memory safety
    log("collecting and writing final parquet (streaming)...")
    
    # We can write directly from the LazyFrame without collecting to RAM first if supported,
    # but `write_parquet` usually wants a DataFrame. 
    # `sink_parquet` is the memory-safe streaming writer in Polars.
    lf_merged.sink_parquet(output_file)
    
    log(f"Wrote {output_file}")

    # ---- Create Index ----
    # Re-read the file we just wrote (it's safe and optimized now) to build the index
    # (Or branch the LazyFrame if you prefer, but re-reading is often safer for memory peak)
    df_final = pl.read_parquet(output_file)
    write_player_index(df_final, player_index_file, debug=debug)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="nba_shot_data_cleaned.parquet")
    ap.add_argument("--games", default="Games.csv")
    ap.add_argument("--players", default="Players.csv")
    ap.add_argument("--out", default="nba_shot_data_final.parquet")
    ap.add_argument("--player_index", default="player_index.parquet")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    normalize_with_local_data(
        shot_data=args.shots,
        games_csv=args.games,
        players_csv=args.players,
        output_file=args.out,
        player_index_file=args.player_index,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
