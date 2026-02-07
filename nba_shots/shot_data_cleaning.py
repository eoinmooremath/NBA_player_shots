# shot_data_cleaning.py
#
# Loads raw NBA PBP parquet and outputs a shot-only dataset with:
# - Season (derived from GameID if missing)
# - ShotType_Simple, ShotOutcome
# - Canonical viz coords computed from xLegacy/yLegacy (but legacy coords are kept too).
#
# Notes:
# - Shot filtering is era-aware:
#     <=2019 often uses actionType in {"made shot","missed shot"}
#     >=2020 often uses actionType in {"2pt","3pt"}
# - We do NOT clip to half court; we only drop rows outside FULL court bounds (optional).
# - Debug logging is optional via --debug.
#
import argparse
import numpy as np
import pandas as pd

from .shot_type_bucketing import shot_type_simple_from_subtype

import pyarrow.parquet as pq

# Only read columns we actually use downstream.
# CRITICAL: exclude list-typed columns like `qualifiers`, `personIdsFilter`, etc.
PBP_COLS_NEEDED = [
    # ids / keys
    "GameID", "gameId",
    "personId",
    "teamId", "teamTricode",

    # season / time context
    "Season",
    "period", "periodType",
    "clock", "timeActual",

    # shot classification
    "actionType",
    "subType",
    "descriptor",
    "description",

    # shot result / points
    "shotResult",
    "ShotOutcome",
    "shotValue",
    "pointsTotal",
    "isFieldGoal",

    # geometry
    "xLegacy",
    "yLegacy",
    "shotDistance",

    # misc that some older data has
    "actionNumber",
    "actionId",
    "side",
    "location",
]

def read_pbp_parquet_safe(path: str) -> pd.DataFrame:
    """
    Read PlayByPlay.parquet without triggering pyarrow->pandas conversion failures
    from list<string_view> columns. We do that by ONLY loading scalar columns we need.
    Also normalizes GameID/gameId into GameID.
    """
    pf = pq.ParquetFile(path)
    available = set(pf.schema.names)

    # pick columns that exist (and are safe)
    cols = [c for c in PBP_COLS_NEEDED if c in available]

    # If neither GameID nor gameId exists, fail loudly
    if "GameID" not in cols and "gameId" not in cols:
        raise ValueError(f"Parquet is missing GameID/gameId. Available columns: {sorted(list(available))[:50]} ...")

    table = pq.read_table(path, columns=cols)  # avoids reading list columns entirely
    df = table.to_pandas()

    # normalize key column name
    if "GameID" not in df.columns and "gameId" in df.columns:
        df = df.rename(columns={"gameId": "GameID"})

    return df

# ---- Court constants (feet) ----
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
RIM_OFFSET_FT = 5.25

RIGHT_RIM_X_FT = COURT_LENGTH_FT - RIM_OFFSET_FT  # 88.75
RIM_Y_FT = COURT_WIDTH_FT / 2.0                   # 25.0

# Viz units are tenths of feet
XMIN_VIZ, XMAX_VIZ = 0.0, 940.0
YMIN_VIZ, YMAX_VIZ = 0.0, 500.0


def _season_debug(df: pd.DataFrame) -> str:
    if "Season" not in df.columns:
        return "(no Season column)"
    s = pd.to_numeric(df["Season"], errors="coerce").dropna()
    if s.empty:
        return "Season all null"
    return f"Season {int(s.min())}..{int(s.max())} (unique={s.astype(int).nunique()}, n={len(df):,})"


def derive_season_from_gameid(df: pd.DataFrame) -> pd.DataFrame:
    """
    NBA GameID examples:
      0029401101 -> "94" -> 1994 (1994-95 season)
      0022500658 -> "25" -> 2025 (2025-26 season)
    We store Season as the season-start year (int), e.g., 2025 for 2025-26.
    """
    gid = df["GameID"].astype(str).str.strip().str.zfill(10)
    yy = pd.to_numeric(gid.str[3:5], errors="coerce")
    season = np.where(yy >= 50, 1900 + yy, 2000 + yy)
    df["Season"] = pd.Series(season, index=df.index).astype("Int64")
    return df


def normalize_from_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy coords are hoop-relative (offensive hoop):
      w_ft = xLegacy/10 : lateral offset from hoop centerline (feet)
      d_ft = yLegacy/10 : depth away from hoop (feet), ~0 at the rim

    Embed into a single RIGHT-BASKET full-court frame:
      Right rim is at (88.75, 25) feet.
      x_ft = 88.75 - d_ft
      y_ft = 25 + w_ft
    """
    w_ft = pd.to_numeric(df["xLegacy"], errors="coerce") / 10.0
    d_ft = pd.to_numeric(df["yLegacy"], errors="coerce") / 10.0

    x_ft = RIGHT_RIM_X_FT - d_ft
    y_ft = RIM_Y_FT + w_ft

    df["x_viz"] = (x_ft * 10.0).astype("float32")
    df["y_viz"] = (y_ft * 10.0).astype("float32")
    df["y_viz_centered"] = (df["y_viz"] - (RIM_Y_FT * 10.0)).astype("float32")  # subtract 250

    return df


def is_shot_row(df: pd.DataFrame) -> pd.Series:
    """
    Shot (field-goal attempt) detection.

    Why this exists: actionType values differ across eras/providers.
    Using only a small hard-coded list (e.g. {"made shot","missed shot","2pt","3pt"})
    can silently drop a large fraction of legitimate FGAs.

    Preferred signal (when available):
      - isFieldGoal == 1/True

    Fallbacks (when isFieldGoal is absent/unreliable):
      - shotResult in {Made, Missed}
      - numeric shotDistance present
      - actionType contains "shot" (excluding "free")
    """

    # 1) isFieldGoal is the best indicator for FGAs
    if "isFieldGoal" in df.columns:
        fg = df["isFieldGoal"]
        if fg.dtype == bool:
            return fg.fillna(False)
        fg_num = pd.to_numeric(fg, errors="coerce").fillna(0).astype(int)
        return fg_num.eq(1)

    m = pd.Series(False, index=df.index)

    # 2) shotResult (when present)
    if "shotResult" in df.columns:
        sr = df["shotResult"].astype(str).str.strip().str.lower()
        m |= sr.isin(["made", "missed"])

    # 3) shotDistance (when present)
    if "shotDistance" in df.columns:
        sd = pd.to_numeric(df["shotDistance"], errors="coerce")
        m |= sd.notna()

    # 4) actionType hints (last resort)
    if "actionType" in df.columns:
        at = df["actionType"].astype(str).str.strip().str.lower()
        m |= at.str.contains("shot", na=False) & ~at.str.contains("free", na=False)
        m |= at.isin(["made shot", "missed shot", "2pt", "3pt"])  # keep legacy behavior

    return m


def build_shot_outcome(df: pd.DataFrame) -> pd.Series:
    """
    Made/miss logic (preferred first):
      1) shotResult == "Made"
      2) actionType contains "made" vs "missed"
      3) description contains "MISS"
    Returns an int series (1=made, 0=miss).
    """
    if "shotResult" in df.columns:
        sr = df["shotResult"].astype(str).str.strip().str.lower()
        made_sr = sr.eq("made")
    else:
        made_sr = pd.Series(False, index=df.index)

    if "actionType" in df.columns:
        at = df["actionType"].astype(str).str.lower()
        made_at = at.str.contains("made", na=False)
        miss_at = at.str.contains("miss", na=False)
    else:
        made_at = pd.Series(False, index=df.index)
        miss_at = pd.Series(False, index=df.index)

    if "description" in df.columns:
        desc = df["description"].astype(str)
        miss_desc = desc.str.contains("MISS", case=False, na=False)
    else:
        miss_desc = pd.Series(False, index=df.index)

    made = made_sr | made_at
    miss = miss_at | miss_desc

    out = np.where(made, 1, np.where(miss, 0, 0)).astype("int8")
    return pd.Series(out, index=df.index)


def clean_shot_data(
    input_path: str = "nba_pbp_v3_backup.parquet",
    output_path: str = "nba_shot_data_cleaned.parquet",
    bounds_filter: bool = True,
    debug: bool = False,
) -> None:
    log = print if debug else (lambda *a, **k: None)

    log(f"Loading {input_path} ...")
    df = read_pbp_parquet_safe(input_path)
    log(f"Raw loaded: n={len(df):,}, cols={len(df.columns)}")

    # Keep only shot rows
    m_shot = is_shot_row(df)
    df = df.loc[m_shot].copy()
    log(f"After is_shot_row: n={len(df):,} ({m_shot.mean():.4f} kept). {_season_debug(df)}")

    # Ensure Season
    if "Season" not in df.columns or df["Season"].isna().all():
        if "GameID" not in df.columns:
            raise KeyError("Missing Season and GameID; cannot derive Season.")
        df = derive_season_from_gameid(df)
        log(f"Derived Season from GameID. {_season_debug(df)}")

    # Numeric IDs
    if "personId" in df.columns:
        df["personId"] = pd.to_numeric(df["personId"], errors="coerce")
    if "teamId" in df.columns:
        df["teamId"] = pd.to_numeric(df["teamId"], errors="coerce")

    # ShotType_Simple (vectorized; Hook merged into Jump)
    if "subType" in df.columns:
        df["ShotType_Simple"] = shot_type_simple_from_subtype(df["subType"]).astype(str)
    else:
        df["ShotType_Simple"] = "Other"

    # ShotOutcome
    df["ShotOutcome"] = build_shot_outcome(df)

    # Coords
    if "xLegacy" not in df.columns or "yLegacy" not in df.columns:
        raise KeyError("Expected xLegacy and yLegacy columns in the input parquet.")
    xL = pd.to_numeric(df["xLegacy"], errors="coerce")
    yL = pd.to_numeric(df["yLegacy"], errors="coerce")
    coord_ok = xL.notna() & yL.notna()
    df = df.loc[coord_ok].copy()
    log(f"After legacy coord non-null: n={len(df):,}. {_season_debug(df)}")

    df = normalize_from_legacy(df)

    # Bounds filter: full court
    if bounds_filter:
        in_bounds = df["x_viz"].between(XMIN_VIZ, XMAX_VIZ) & df["y_viz"].between(YMIN_VIZ, YMAX_VIZ)
        df = df.loc[in_bounds].copy()
        log(f"After bounds filter: n={len(df):,}. {_season_debug(df)}")

    # Keep columns (preserve anything useful; keep stable ones first)
    cols_to_keep = [
        "actionNumber", "clock", "timeActual", "period", "periodType",
        "teamTricode", "teamId", "GameID", "Season",
        "personId", "playerName", "playerNameI",
        "actionType", "subType", "descriptor", "shotDistance", "shotResult", "pointsTotal",
        "area", "areaDetail",
        "ShotType_Simple", "ShotOutcome",
        "x_viz", "y_viz", "y_viz_centered",
        "xLegacy", "yLegacy",
        "side",
        "description",
        "shotValue",
        "isFieldGoal",
        "scoreHome", "scoreAway",
        "location",
        "actionId",
        "firstName", "lastName", "Full_Name",
        "hometeamId", "awayteamId",
        "hometeamCity", "hometeamName", "awayteamCity", "awayteamName",
        "gameType",
        "playerTeamCity", "playerTeamName", "opponentTeamCity", "opponentTeamName",
        "SeasonPhase", "IsPostseason",
    ]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]

    log(f"Saving {len(df):,} rows -> {output_path}")
    df[cols_to_keep].to_parquet(output_path, index=False)
    log("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="nba_pbp_v3_backup.parquet")
    ap.add_argument("--output", default="nba_shot_data_cleaned.parquet")
    ap.add_argument("--no_bounds_filter", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    clean_shot_data(
        input_path=args.input,
        output_path=args.output,
        bounds_filter=(not args.no_bounds_filter),
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
