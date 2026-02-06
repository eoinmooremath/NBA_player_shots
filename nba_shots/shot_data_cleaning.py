# shot_data_cleaning.py
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .shot_type_bucketing import shot_type_simple_from_subtype

# ---- Court constants (feet) ----
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
RIM_OFFSET_FT = 5.25

RIGHT_RIM_X_FT = COURT_LENGTH_FT - RIM_OFFSET_FT  # 88.75
RIM_Y_FT = COURT_WIDTH_FT / 2.0                   # 25.0

# Viz units are tenths of feet
XMIN_VIZ, XMAX_VIZ = 0.0, 940.0
YMIN_VIZ, YMAX_VIZ = 0.0, 500.0


# CRITICAL: exclude list-typed columns like `qualifiers` to prevent crashes.
PBP_COLS_NEEDED = [
    # ids / keys
    "GameID", "gameId", "personId", "teamId", "teamTricode",
    # season / time context
    "Season", "period", "periodType", "clock", "timeActual",
    # shot classification
    "actionType", "subType", "descriptor", "description",
    # shot result / points
    "shotResult", "ShotOutcome", "shotValue", "pointsTotal", "isFieldGoal",
    # geometry
    "xLegacy", "yLegacy", "shotDistance",
    # misc
    "actionNumber", "actionId", "side", "location",
    # Metadata for filtering/joins
    "firstName", "lastName", "Full_Name",
    "hometeamId", "awayteamId", "hometeamCity", "hometeamName", 
    "awayteamCity", "awayteamName", "gameType",
    "playerTeamCity", "playerTeamName", "opponentTeamCity", "opponentTeamName",
    "SeasonPhase", "IsPostseason", "area", "areaDetail"
]


def read_pbp_parquet_safe(path: str) -> pd.DataFrame:
    """
    1. Identifies safe columns (skipping complex nested types).
    2. Loads ONLY those columns using the 'pyarrow' backend for maximum memory efficiency.
    """
    # 1. Inspect the file to see what columns exist
    pf = pq.ParquetFile(path)
    available = set(pf.schema.names)

    # 2. Intersect with our needed list
    cols = [c for c in PBP_COLS_NEEDED if c in available]

    if "GameID" not in cols and "gameId" not in cols:
        raise ValueError(f"Parquet missing GameID. Available: {list(available)[:10]}...")

    # 3. Read using Pandas with PyArrow backend (Memory Efficient + Fast)
    df = pd.read_parquet(
        path, 
        columns=cols, 
        dtype_backend="pyarrow"  # <--- The Magic Memory Saver
    )

    # Normalize GameID
    if "GameID" not in df.columns and "gameId" in df.columns:
        df = df.rename(columns={"gameId": "GameID"})

    return df


def _season_debug(df: pd.DataFrame) -> str:
    if "Season" not in df.columns:
        return "(no Season column)"
    s = pd.to_numeric(df["Season"], errors="coerce").dropna()
    if s.empty: return "Season all null"
    return f"Season {int(s.min())}..{int(s.max())} (unique={s.astype(int).nunique()}, n={len(df):,})"


def derive_season_from_gameid(df: pd.DataFrame) -> pd.DataFrame:
    gid = df["GameID"].astype(str).str.strip().str.zfill(10)
    yy = pd.to_numeric(gid.str[3:5], errors="coerce")
    season = np.where(yy >= 50, 1900 + yy, 2000 + yy)
    df["Season"] = pd.Series(season, index=df.index).astype("Int64")
    return df


def normalize_from_legacy(df: pd.DataFrame) -> pd.DataFrame:
    w_ft = pd.to_numeric(df["xLegacy"], errors="coerce") / 10.0
    d_ft = pd.to_numeric(df["yLegacy"], errors="coerce") / 10.0
    x_ft = RIGHT_RIM_X_FT - d_ft
    y_ft = RIM_Y_FT + w_ft
    df["x_viz"] = (x_ft * 10.0).astype("float32")
    df["y_viz"] = (y_ft * 10.0).astype("float32")
    df["y_viz_centered"] = (df["y_viz"] - (RIM_Y_FT * 10.0)).astype("float32")
    return df


def is_shot_row(df: pd.DataFrame) -> pd.Series:
    if "isFieldGoal" in df.columns:
        fg = df["isFieldGoal"]
        # Handle PyArrow boolean type or standard object type
        if pd.api.types.is_bool_dtype(fg.dtype):
            return fg.fillna(False)
        fg_num = pd.to_numeric(fg, errors="coerce").fillna(0).astype(int)
        return fg_num.eq(1)

    m = pd.Series(False, index=df.index)
    if "shotResult" in df.columns:
        sr = df["shotResult"].astype(str).str.strip().str.lower()
        m |= sr.isin(["made", "missed"])
    if "shotDistance" in df.columns:
        sd = pd.to_numeric(df["shotDistance"], errors="coerce")
        m |= sd.notna()
    if "actionType" in df.columns:
        at = df["actionType"].astype(str).str.strip().str.lower()
        m |= at.str.contains("shot", na=False) & ~at.str.contains("free", na=False)
        m |= at.isin(["made shot", "missed shot", "2pt", "3pt"])
    return m


def build_shot_outcome(df: pd.DataFrame) -> pd.Series:
    if "shotResult" in df.columns:
        sr = df["shotResult"].astype(str).str.strip().str.lower()
        made_sr = sr.eq("made")
    else: made_sr = pd.Series(False, index=df.index)

    if "actionType" in df.columns:
        at = df["actionType"].astype(str).str.lower()
        made_at = at.str.contains("made", na=False)
        miss_at = at.str.contains("miss", na=False)
    else: made_at, miss_at = pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    if "description" in df.columns:
        desc = df["description"].astype(str)
        miss_desc = desc.str.contains("MISS", case=False, na=False)
    else: miss_desc = pd.Series(False, index=df.index)

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

    # --- USE COMBINED SAFE READER ---
    df = read_pbp_parquet_safe(input_path)
    # --------------------------------
    
    log(f"Loaded: n={len(df):,}, cols={len(df.columns)}")

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
    for col in ["personId", "teamId"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")

    # ShotType_Simple
    if "subType" in df.columns:
        df["ShotType_Simple"] = shot_type_simple_from_subtype(df["subType"]).astype(str)
    else: df["ShotType_Simple"] = "Other"

    # ShotOutcome
    df["ShotOutcome"] = build_shot_outcome(df)

    # Coords
    if "xLegacy" not in df.columns or "yLegacy" not in df.columns:
        raise KeyError("Expected xLegacy and yLegacy columns.")
    xL = pd.to_numeric(df["xLegacy"], errors="coerce")
    yL = pd.to_numeric(df["yLegacy"], errors="coerce")
    coord_ok = xL.notna() & yL.notna()
    df = df.loc[coord_ok].copy()
    
    df = normalize_from_legacy(df)

    # Bounds filter
    if bounds_filter:
        in_bounds = df["x_viz"].between(XMIN_VIZ, XMAX_VIZ) & df["y_viz"].between(YMIN_VIZ, YMAX_VIZ)
        df = df.loc[in_bounds].copy()
        log(f"After bounds filter: n={len(df):,}")

    # Final Column Selection
    cols_to_keep = PBP_COLS_NEEDED + [
        "ShotType_Simple", "ShotOutcome",
        "x_viz", "y_viz", "y_viz_centered",
        "hometeamId", "awayteamId",
    ]
    final_cols = list(dict.fromkeys([c for c in cols_to_keep if c in df.columns]))

    log(f"Saving {len(df):,} rows -> {output_path}")
    # Write back to parquet (will write using pyarrow automatically)
    df[final_cols].to_parquet(output_path, index=False)
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
