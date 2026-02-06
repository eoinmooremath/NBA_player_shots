# shot_data_enrichment.py
#
# Enriches the cleaned shots parquet by merging:
#   - Players.csv (names)
#   - Games.csv (home/away teams, gameType)
#
# Outputs:
#   - nba_shot_data_final.parquet (enriched shots)
#   - player_index.parquet (small index used by the Streamlit app)
#
# Debug logging is optional via --debug.
#
import argparse
import os
import re
import pandas as pd
import numpy as np


def _norm_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def map_game_type_to_season_phase(game_type_raw: str) -> str:
    """
    Canonical SeasonPhase values written to nba_shot_data_final.parquet:

      - Regular Season
      - Preseason
      - Playoffs
      - Play-In Tournament
      - Other
      - Unknown

    UI policy (app-side):
      - In-Season Tournament / Cup games are treated as Regular Season.
      - Postseason is Play-In + Playoffs.
      - Unknown/Other appear only when phase filter is "All".
    """
    gt = _norm_str(game_type_raw)
    if gt == "":
        return "Unknown"

    if gt in {"regular season", "regular"}:
        return "Regular Season"
    if gt in {"preseason", "pre season"}:
        return "Preseason"
    if gt in {"playoffs", "playoff"}:
        return "Playoffs"
    if gt in {"play in tournament", "play in"}:
        return "Play-In Tournament"

    # In-season tournament / cup variants -> Regular Season
    if ("in season" in gt) or ("cup" in gt) or ("emirates" in gt):
        return "Regular Season"

    if "global" in gt:
        return "Other"

    return "Other"


def write_player_index(df_merged: pd.DataFrame, player_index_file: str, debug: bool = False) -> None:
    log = print if debug else (lambda *a, **k: None)

    cols_for_index = ["personId", "Season", "firstName", "lastName", "playerName", "playerTeamName"]
    cols_for_index = [c for c in cols_for_index if c in df_merged.columns]
    pi = df_merged[cols_for_index].copy()

    pi["personId"] = pd.to_numeric(pi["personId"], errors="coerce")
    pi["Season"] = pd.to_numeric(pi["Season"], errors="coerce")
    pi = pi.dropna(subset=["personId", "Season"])
    pi["personId"] = pi["personId"].astype("int64")
    pi["Season"] = pi["Season"].astype("int32")

    bounds = (
        pi.groupby("personId")["Season"]
          .agg(minSeason="min", maxSeason="max")
          .reset_index()
    )

    latest_team = (
        pi.sort_values(["personId", "Season"], ascending=[True, False])
          .dropna(subset=["playerTeamName"])
          .drop_duplicates("personId")[["personId", "playerTeamName"]]
          .rename(columns={"playerTeamName": "latestTeam"})
    )

    name_df = (
        pi.sort_values(["personId", "Season"], ascending=[True, False])
          .drop_duplicates("personId")[["personId", "firstName", "lastName", "playerName"]]
    )
    name_df["Full_Name"] = (
        (name_df["firstName"].fillna("").astype(str) + " " + name_df["lastName"].fillna("").astype(str)).str.strip()
    )
    name_df.loc[name_df["Full_Name"].eq(""), "Full_Name"] = name_df["playerName"].astype(str)

    teams = pi.dropna(subset=["playerTeamName"]).copy()
    if len(teams):
        first_seen = (
            teams.groupby(["personId", "playerTeamName"])["Season"]
                 .min()
                 .reset_index()
                 .sort_values(["personId", "Season", "playerTeamName"])
        )
        teams_played = (
            first_seen.groupby("personId")["playerTeamName"]
                      .apply(lambda s: list(dict.fromkeys(s.tolist())))
                      .reset_index()
                      .rename(columns={"playerTeamName": "teamsPlayedList"})
        )
        teams_played["teamsPlayed"] = teams_played["teamsPlayedList"].apply(
            lambda xs: " · ".join([str(x) for x in xs])
        )
        teams_played = teams_played[["personId", "teamsPlayed"]]
    else:
        teams_played = pd.DataFrame({"personId": bounds["personId"], "teamsPlayed": ""})

    out = (
        name_df[["personId", "Full_Name", "firstName", "lastName"]]
        .merge(bounds, on="personId", how="left")
        .merge(latest_team, on="personId", how="left")
        .merge(teams_played, on="personId", how="left")
    )

    out["teamsPlayed"] = out["teamsPlayed"].fillna("").astype(str)
    out["personId"] = out["personId"].astype("int64")
    out["minSeason"] = out["minSeason"].astype("int32")
    out["maxSeason"] = out["maxSeason"].astype("int32")

    out.to_parquet(player_index_file, index=False)
    log(f"Wrote {player_index_file}: rows={len(out):,}")


def normalize_with_local_data(
    shot_data: str = "nba_shot_data_cleaned.parquet",
    games_csv: str = "Games.csv",
    players_csv: str = "Players.csv",
    output_file: str = "nba_shot_data_final.parquet",
    player_index_file: str = "player_index.parquet",
    debug: bool = False,
) -> None:
    log = print if debug else (lambda *a, **k: None)

    if not os.path.exists(shot_data):
        raise FileNotFoundError(f"Missing {shot_data}. Run clean_data.py first.")
    if not os.path.exists(games_csv) or not os.path.exists(players_csv):
        raise FileNotFoundError("Missing Games.csv or Players.csv.")

    log("Loading datasets...")
    df_shots = pd.read_parquet(shot_data)
    df_games = pd.read_csv(games_csv, low_memory=False)
    df_players = pd.read_csv(players_csv, low_memory=False)

    log(f"Shots loaded:   n={len(df_shots):,}, cols={len(df_shots.columns)}")
    log(f"Games loaded:   n={len(df_games):,}, cols={len(df_games.columns)}")
    log(f"Players loaded: n={len(df_players):,}, cols={len(df_players.columns)}")

    # ---- IDs / keys ----
    for req in ["GameID", "personId"]:
        if req not in df_shots.columns:
            raise KeyError(f"Shots file missing '{req}'")
    for req in ["gameId"]:
        if req not in df_games.columns:
            raise KeyError(f"{games_csv} missing '{req}'")
    for req in ["personId"]:
        if req not in df_players.columns:
            raise KeyError(f"{players_csv} missing '{req}'")

    df_shots["GameID"] = df_shots["GameID"].astype(str).str.strip().str.zfill(10)
    df_games["gameId"] = df_games["gameId"].astype(str).str.strip().str.zfill(10)

    df_shots["personId"] = pd.to_numeric(df_shots["personId"], errors="coerce")
    df_players["personId"] = pd.to_numeric(df_players["personId"], errors="coerce")

    if "teamId" in df_shots.columns:
        df_shots["teamId"] = pd.to_numeric(df_shots["teamId"], errors="coerce")
    for c in ["hometeamId", "awayteamId"]:
        if c in df_games.columns:
            df_games[c] = pd.to_numeric(df_games[c], errors="coerce")

    subset_drop = ["personId", "GameID"]
    if "teamId" in df_shots.columns:
        subset_drop.append("teamId")
    before = len(df_shots)
    df_shots = df_shots.dropna(subset=subset_drop)
    log(f"Dropped {before - len(df_shots):,} shots missing {subset_drop}")

    # ---- Merge Players ----
    required_player_cols = ["personId", "firstName", "lastName"]
    for c in required_player_cols:
        if c not in df_players.columns:
            raise KeyError(f"{players_csv} missing '{c}'")

    df_players_small = df_players[required_player_cols].copy()
    df_merged = df_shots.merge(df_players_small, on="personId", how="left")
    before = len(df_merged)
    df_merged = df_merged.dropna(subset=["firstName", "lastName"])
    log(f"Dropped {before - len(df_merged):,} shots with unmapped personId in {players_csv}")

    df_merged["firstName"] = df_merged["firstName"].astype(str)
    df_merged["lastName"] = df_merged["lastName"].astype(str)
    df_merged["Full_Name"] = (df_merged["firstName"] + " " + df_merged["lastName"]).str.strip()

    # ---- Merge Games ----
    required_game_cols = [
        "gameId",
        "hometeamId", "awayteamId",
        "hometeamCity", "hometeamName",
        "awayteamCity", "awayteamName",
    ]
    for c in required_game_cols:
        if c not in df_games.columns:
            raise KeyError(f"{games_csv} missing '{c}'")

    game_cols = required_game_cols.copy()
    if "gameType" in df_games.columns:
        game_cols.append("gameType")
    df_merged = df_merged.merge(df_games[game_cols], left_on="GameID", right_on="gameId", how="left")
    log(f"Games merge coverage (hometeamId present): {df_merged['hometeamId'].notna().mean():.4f}")

    # ---- Determine player/opponent teams ----
    df_merged["playerTeamCity"] = pd.NA
    df_merged["playerTeamName"] = pd.NA
    df_merged["opponentTeamCity"] = pd.NA
    df_merged["opponentTeamName"] = pd.NA

    if "teamId" in df_merged.columns:
        is_home = df_merged["teamId"] == df_merged["hometeamId"]
        is_away = df_merged["teamId"] == df_merged["awayteamId"]

        df_merged.loc[is_home, "playerTeamCity"] = df_merged.loc[is_home, "hometeamCity"]
        df_merged.loc[is_home, "playerTeamName"] = df_merged.loc[is_home, "hometeamName"]
        df_merged.loc[is_home, "opponentTeamCity"] = df_merged.loc[is_home, "awayteamCity"]
        df_merged.loc[is_home, "opponentTeamName"] = df_merged.loc[is_home, "awayteamName"]

        df_merged.loc[is_away, "playerTeamCity"] = df_merged.loc[is_away, "awayteamCity"]
        df_merged.loc[is_away, "playerTeamName"] = df_merged.loc[is_away, "awayteamName"]
        df_merged.loc[is_away, "opponentTeamCity"] = df_merged.loc[is_away, "hometeamCity"]
        df_merged.loc[is_away, "opponentTeamName"] = df_merged.loc[is_away, "hometeamName"]

    # ---- SeasonPhase ----
    if "gameType" in df_merged.columns:
        df_merged["SeasonPhase"] = df_merged["gameType"].apply(map_game_type_to_season_phase).astype(str)
    else:
        df_merged["SeasonPhase"] = "Unknown"
    df_merged["IsPostseason"] = df_merged["SeasonPhase"].isin(["Playoffs", "Play-In Tournament"]).astype(np.int8)

    # ---- Save final ----
    df_merged.drop(columns=["gameId"], inplace=True, errors="ignore")
    df_merged.to_parquet(output_file, index=False)
    log(f"Wrote {output_file}: rows={len(df_merged):,}")

    # ---- Player index ----
    write_player_index(df_merged, player_index_file, debug=debug)


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
