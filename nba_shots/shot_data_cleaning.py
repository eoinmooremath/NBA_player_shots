# shot_data_cleaning.py
import argparse
import polars as pl
import numpy as np
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

# Columns to select (Polars optimizes this automatically, but good to be explicit)
PBP_COLS_NEEDED = [
    "GameID", "gameId", "personId", "teamId", "teamTricode",
    "Season", "period", "periodType", "clock", "timeActual",
    "actionType", "subType", "descriptor", "description",
    "shotResult", "ShotOutcome", "shotValue", "pointsTotal", "isFieldGoal",
    "xLegacy", "yLegacy", "shotDistance",
    "actionNumber", "actionId", "side", "location",
    "firstName", "lastName", "Full_Name",
    "hometeamId", "awayteamId", "hometeamCity", "hometeamName", 
    "awayteamCity", "awayteamName", "gameType",
    "playerTeamCity", "playerTeamName", "opponentTeamCity", "opponentTeamName",
    "SeasonPhase", "IsPostseason", "area", "areaDetail"
]

def clean_shot_data(
    input_path: str = "nba_pbp_v3_backup.parquet",
    output_path: str = "nba_shot_data_cleaned.parquet",
    bounds_filter: bool = True,
    debug: bool = False,
) -> None:
    log = print if debug else (lambda *a, **k: None)
    log(f"Scanning {input_path} via Polars...")

    # 1. Scan Parquet (Lazy Frame) - Instantaneous, no data loaded yet
    # infer_schema_length=0 forces it to read the file schema directly
    lf = pl.scan_parquet(input_path)
    
    # 2. Filter Columns (Handle GameID/gameId mismatch lazily)
    available_cols = lf.columns
    cols_to_select = [c for c in PBP_COLS_NEEDED if c in available_cols]
    
    lf = lf.select(cols_to_select)
    
    # Normalize GameID
    if "GameID" not in lf.columns and "gameId" in lf.columns:
        lf = lf.rename({"gameId": "GameID"})
    
    # 3. Create Basic Filters (Shot Detection)
    # logic: isFieldGoal == 1 OR shotResult in [made, missed] OR ...
    
    # Pre-calculate boolean expressions
    # Note: Polars handles nulls in boolean logic differently (null != False), so we use fill_null
    
    # isFieldGoal check
    has_ifg = "isFieldGoal" in lf.columns
    expr_ifg = pl.col("isFieldGoal").fill_null(0).cast(pl.Int8).eq(1) if has_ifg else pl.lit(False)
    
    # shotResult check
    has_sr = "shotResult" in lf.columns
    expr_sr = pl.col("shotResult").str.to_lowercase().is_in(["made", "missed"]) if has_sr else pl.lit(False)

    # shotDistance check
    has_sd = "shotDistance" in lf.columns
    expr_sd = pl.col("shotDistance").is_not_null() if has_sd else pl.lit(False)

    # actionType check
    has_at = "actionType" in lf.columns
    if has_at:
        at_clean = pl.col("actionType").str.strip_chars().str.to_lowercase()
        expr_at = (
            (at_clean.str.contains("shot") & ~at_clean.str.contains("free")) |
            at_clean.is_in(["made shot", "missed shot", "2pt", "3pt"])
        )
    else:
        expr_at = pl.lit(False)

    # Apply Shot Filter
    lf = lf.filter(expr_ifg | expr_sr | expr_sd | expr_at)

    # 4. Transformations (Season, Coords, Outcomes)
    lf = lf.with_columns([
        # --- Season Derivation ---
        # "00223..." -> slice(3,2) -> "23" -> cast int -> logic
        pl.when(pl.col("Season").is_null())
          .then(
              pl.when(pl.col("GameID").str.slice(3, 2).cast(pl.Int32) >= 50)
                .then(pl.col("GameID").str.slice(3, 2).cast(pl.Int32) + 1900)
                .otherwise(pl.col("GameID").str.slice(3, 2).cast(pl.Int32) + 2000)
          )
          .otherwise(pl.col("Season"))
          .cast(pl.Int64)
          .alias("Season"),
          
        # --- Numeric Casting ---
        pl.col("personId").cast(pl.Float64, strict=False),
        pl.col("teamId").cast(pl.Float64, strict=False),
        
        # --- Normalize Legacy Coords ---
        (pl.col("xLegacy").cast(pl.Float64, strict=False) / 10.0).alias("_w_ft"),
        (pl.col("yLegacy").cast(pl.Float64, strict=False) / 10.0).alias("_d_ft"),
    ])

    # Calculate Visual Coords based on previous step
    # Polars runs expressions in parallel, but dependent columns need a second context 
    # or just chained expressions. Chaining is cleaner here.
    lf = lf.with_columns([
        ((pl.lit(RIGHT_RIM_X_FT) - pl.col("_d_ft")) * 10.0).cast(pl.Float32).alias("x_viz"),
        ((pl.lit(RIM_Y_FT) + pl.col("_w_ft")) * 10.0).cast(pl.Float32).alias("y_viz"),
    ])
    
    lf = lf.with_columns([
        (pl.col("y_viz") - (RIM_Y_FT * 10.0)).cast(pl.Float32).alias("y_viz_centered")
    ])

    # --- Shot Outcome Logic ---
    # Made if: shotResult='made' OR actionType has 'made'
    # Miss if: actionType has 'miss' OR description has 'MISS'
    
    is_made = pl.lit(False)
    is_miss = pl.lit(False)
    
    if "shotResult" in lf.columns:
        is_made = is_made | pl.col("shotResult").str.to_lowercase().eq("made")
        
    if "actionType" in lf.columns:
        at_lower = pl.col("actionType").str.to_lowercase()
        is_made = is_made | at_lower.str.contains("made")
        is_miss = is_miss | at_lower.str.contains("miss")
        
    if "description" in lf.columns:
        is_miss = is_miss | pl.col("description").str.contains("(?i)MISS") # case insensitive regex
        
    lf = lf.with_columns(
        pl.when(is_made).then(1)
          .when(is_miss).then(0)
          .otherwise(0)
          .cast(pl.Int8)
          .alias("ShotOutcome")
    )

    # 5. Bounds Filter
    if bounds_filter:
        lf = lf.filter(
            pl.col("x_viz").is_between(XMIN_VIZ, XMAX_VIZ) & 
            pl.col("y_viz").is_between(YMIN_VIZ, YMAX_VIZ)
        )

    # 6. ShotType_Simple (User-Defined Function)
    # Since `shot_type_simple_from_subtype` is Python code, we must execute it.
    # We collect strictly necessary columns, map, and join back OR just map if dataset is small enough now.
    # Note: Because this uses an external Python function, we can't stream this specific step easily 
    # without map_elements (which is slow) or rewriting that function in Polars expressions.
    # STRATEGY: We will collect the dataset now (it's filtered and safe), convert to pandas for 
    # just this one column if needed, or stick to Polars map_elements.
    
    log("Executing plan (Streaming)...")
    
    # STREAMING: This is the magic that prevents OOM
    df = lf.collect(streaming=True)
    
    log(f"Filtered to {len(df):,} rows.")

    # 7. Apply the Python UDF for ShotType
    # We do this post-collection on the smaller dataset
    if "subType" in df.columns:
        # Convert to pandas for the UDF to ensure 100% compatibility with your existing helper
        # or use map_elements if you prefer to stay in Polars.
        # Given the existing helper likely expects a Series, we'll leverage the helper.
        
        # Fast way: Extract column, to pandas, map, back to polars
        subtypes_pd = df["subType"].to_pandas()
        simple_types = shot_type_simple_from_subtype(subtypes_pd).astype(str)
        df = df.with_columns(pl.Series("ShotType_Simple", simple_types))
    else:
        df = df.with_columns(pl.lit("Other").alias("ShotType_Simple"))

    # 8. Save
    log(f"Saving to {output_path}...")
    df.write_parquet(output_path)
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
