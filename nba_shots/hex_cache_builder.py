# hex_cache_builder.py
import argparse
import json
import os
import shutil

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the new Polars expression from the updated bucketing file
from nba_shots.shot_type_bucketing import shot_type_simple_from_subtype_expr

# Plot-space constants
XMIN, XMAX = 0.0, 940.0
YMIN, YMAX = -250.0, 250.0
CLIP_X0, CLIP_W = 0.0, 940.0
CLIP_Y0, CLIP_H = -250.0, 500.0

def make_matplotlib_template_grid(gridsize: int, extent):
    """
    Generates the reference grid dataframe using matplotlib's logic.
    We keep this in NumPy/Pandas because it's tiny (only generates template bins once).
    """
    import pandas as pd # Local import just for this small dataframe creation
    xmin, xmax, ymin, ymax = extent
    nx = gridsize
    ny = gridsize
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny

    x1 = xmin + (np.arange(nx) + 0.5) * sx
    y1 = ymin + (np.arange(ny) + 0.5) * sy
    X1, Y1 = np.meshgrid(x1, y1)

    x2 = xmin + (np.arange(nx + 1)) * sx
    y2 = ymin + (np.arange(ny + 1)) * sy
    X2, Y2 = np.meshgrid(x2, y2)

    xs = np.concatenate([X1.ravel(), X2.ravel()]).astype(np.float64)
    ys = np.concatenate([Y1.ravel(), Y2.ravel()]).astype(np.float64)

    fig, ax = plt.subplots(figsize=(6, 3))
    hb = ax.hexbin(xs, ys, gridsize=gridsize, extent=extent, mincnt=1, linewidths=0.0, edgecolors="none")
    offsets = hb.get_offsets()
    plt.close(fig)

    grid = pd.DataFrame({
        "bin_template_index": np.arange(offsets.shape[0], dtype=np.int32),
        "bin_x": offsets[:, 0].astype(np.float32),
        "bin_y": offsets[:, 1].astype(np.float32),
    })

    meta = {
        "cache_version": 2,
        "cache_layout": "per_season_file",
        "cache_data_file": "hex.parquet",
        "partition_cols": ["Season"],
        "phase_col": "PhaseGroup",
        "phase_groups_all": ["regular", "preseason", "postseason", "unknown"],
        "extent": list(extent),
        "gridsize": int(gridsize),
        "nx": int(nx),
        "ny": int(ny),
        "sx": float(sx),
        "sy": float(sy),
        "n_total_bins": int(offsets.shape[0]),
        "court_clip": [float(CLIP_X0), float(CLIP_W), float(CLIP_Y0), float(CLIP_H)],
    }
    return grid, meta

def assign_bins_polars_expr(x_col: str, y_col: str, gridsize: int, extent) -> pl.Expr:
    """
    Pure Polars implementation of matplotlib's hexbin logic.
    This replaces the NumPy function, allowing parallel/streaming execution.
    """
    xmin, xmax, ymin, ymax = extent
    nx = int(gridsize)
    ny = int(gridsize)
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny

    # 1. Grid coordinates (floored)
    ix1 = ((pl.col(x_col) - xmin) / sx - 0.5).floor().cast(pl.Int64)
    iy1 = ((pl.col(y_col) - ymin) / sy - 0.5).floor().cast(pl.Int64)
    
    # Centers 1
    cx1 = xmin + (ix1 + 0.5) * sx
    cy1 = ymin + (iy1 + 0.5) * sy

    # 2. Secondary grid
    ix2 = ((pl.col(x_col) - xmin) / sx).floor().cast(pl.Int64)
    iy2 = ((pl.col(y_col) - ymin) / sy).floor().cast(pl.Int64)
    
    # Centers 2
    cx2 = xmin + ix2 * sx
    cy2 = ymin + iy2 * sy

    # 3. Distances
    dx1 = pl.col(x_col) - cx1
    dy1 = pl.col(y_col) - cy1
    dx2 = pl.col(x_col) - cx2
    dy2 = pl.col(y_col) - cy2

    d1 = (dx1 * dx1) + (3.0 * dy1 * dy1)
    d2 = (dx2 * dx2) + (3.0 * dy2 * dy2)

    # 4. Selection
    # Clip indices to grid bounds
    ix1c = ix1.clip(0, nx - 1)
    iy1c = iy1.clip(0, ny - 1)
    ix2c = ix2.clip(0, nx)
    iy2c = iy2.clip(0, ny)

    primary_id = (iy1c * nx + ix1c)
    secondary_id = (nx * ny) + (iy2c * (nx + 1) + ix2c)

    return (
        pl.when(d1 <= d2)
          .then(primary_id)
          .otherwise(secondary_id)
          .cast(pl.Int32)
          .alias("bin_id")
    )

def ensure_points_made_expr() -> pl.Expr:
    """Polars expression to determine points made."""
    # Logic: shotValue -> actionType -> description -> distance -> default(2)
    
    # Base points (float32)
    pts = pl.lit(None, dtype=pl.Float32)

    # 1. shotValue (if exists)
    # We use coalesce later, but for precedence we chain `when/then`
    
    # Helper for "Is 3PT" in description: "3PT" but not "PTS"
    desc_is_3pt = pl.col("description").str.contains(r"3PT(?!S)")

    points_val = (
        pl.when(pl.col("shotValue").is_in([2, 3]))
          .then(pl.col("shotValue"))
          
          .when(pl.col("actionType").str.to_lowercase() == "2pt").then(2.0)
          .when(pl.col("actionType").str.to_lowercase() == "3pt").then(3.0)
          
          .when(desc_is_3pt).then(3.0)
          
          .when(pl.col("shotDistance") > 22).then(3.0)
          .when(pl.col("shotDistance") <= 22).then(2.0)
          
          .otherwise(2.0) # Default
          .cast(pl.Float32)
    )
    
    # Multiply by outcome (1 or 0)
    return (points_val * pl.col("ShotOutcome").cast(pl.Float32)).alias("PointsMade")

def build_hex_cache(
    input_path: str,
    output_dir: str,
    gridsize: int = 30,
    legacy_to_plot_x0: float = 887.5,
    force: bool = False,
    debug: bool = False,
):
    log = print if debug else (lambda *a, **k: None)
    extent = (XMIN, XMAX, YMIN, YMAX)

    if force and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Template Grid (Metadata)
    grid_df_pd, meta = make_matplotlib_template_grid(gridsize, extent)
    meta["legacy_to_plot_x0"] = float(legacy_to_plot_x0)
    
    grid_path = os.path.join(output_dir, "_grid.parquet")
    meta_path = os.path.join(output_dir, "_meta.json")
    
    # Write grid using pandas (it's small and already in pandas from the helper)
    grid_df_pd.to_parquet(grid_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    log(f"Scanning {input_path} (Lazy)...")
    lf = pl.scan_parquet(input_path)

    # 2. Data transformations (Lazy)
    
    # Ensure types
    lf = lf.with_columns([
        pl.col("Season").cast(pl.Int32),
        pl.col("personId").cast(pl.Int32)
    ])

    # Shot Type (using new Polars helper)
    if "ShotType_Simple" not in lf.columns:
        if "subType" in lf.columns:
            lf = lf.with_columns(shot_type_simple_from_subtype_expr("subType"))
        else:
            lf = lf.with_columns(pl.lit("Other").alias("ShotType_Simple"))
    
    # Replace "Hook" with "Jump"
    lf = lf.with_columns(
        pl.col("ShotType_Simple").replace("Hook", "Jump")
    )

    # Season Phase / PhaseGroup
    if "SeasonPhase" not in lf.columns:
        lf = lf.with_columns(pl.lit("Unknown").alias("SeasonPhase"))
        
    sp = pl.col("SeasonPhase").str.to_lowercase()
    lf = lf.with_columns(
        pl.when(sp == "preseason").then(pl.lit("preseason"))
          .when((sp == "regular season") | sp.str.contains("in-season") | sp.str.contains("cup"))
          .then(pl.lit("regular"))
          .when(sp.is_in(["playoffs", "play-in tournament"]) | sp.str.contains("play-in") | sp.str.contains("playoff"))
          .then(pl.lit("postseason"))
          .otherwise(pl.lit("unknown"))
          .alias("PhaseGroup")
    )

    # Shot Outcome (Ensure 0/1)
    # If already computed in enrichment, great. If not, simple fallback logic:
    if "ShotOutcome" not in lf.columns:
        # Minimal fallback if enrichment didn't happen
        lf = lf.with_columns(pl.lit(0).alias("ShotOutcome")) # Simplified for cache builder

    lf = lf.with_columns(pl.col("ShotOutcome").fill_null(0).cast(pl.Int16))

    # Points Made
    lf = lf.with_columns(ensure_points_made_expr())

    # 3. Coordinate Transform & Filter (Lazy)
    # x_plot = legacy_to_plot_x0 - yLegacy
    # y_plot = xLegacy
    lf = lf.with_columns([
        (legacy_to_plot_x0 - pl.col("yLegacy").cast(pl.Float32)).alias("x_plot"),
        (pl.col("xLegacy").cast(pl.Float32)).alias("y_plot")
    ])

    # Filter Clip
    lf = lf.filter(
        (pl.col("x_plot") >= CLIP_X0) & (pl.col("x_plot") <= CLIP_X0 + CLIP_W) &
        (pl.col("y_plot") >= CLIP_Y0) & (pl.col("y_plot") <= CLIP_Y0 + CLIP_H)
    )

    # 4. Assign Hex Bins (The Math!)
    lf = lf.with_columns(
        assign_bins_polars_expr("x_plot", "y_plot", gridsize, extent)
    )

    # 5. Aggregate (GroupBy)
    # We aggregate by: Season, personId, PhaseGroup, ShotType_Simple, bin_id
    agg_lf = (
        lf.group_by(["Season", "personId", "PhaseGroup", "ShotType_Simple", "bin_id"])
          .agg([
              pl.len().cast(pl.Int32).alias("attempts"),
              pl.sum("ShotOutcome").cast(pl.Int32).alias("makes"),
              pl.sum("PointsMade").cast(pl.Float32).alias("points")
          ])
    )

    # 6. Execute & Partition Write
    # We collect the aggregated result. 
    # Since we aggregated millions of shots into bins, this result is small enough for RAM.
    log("Aggregating data (Streaming)...")
    df_agg = agg_lf.collect(streaming=True)
    
    log(f"Aggregated into {len(df_agg):,} bin-rows. Writing partitions...")

    # Write partitions per season
    seasons = sorted(df_agg["Season"].unique().to_list())
    
    for season in seasons:
        out_dir = os.path.join(output_dir, f"Season={int(season)}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "hex.parquet")
        
        # Filter and sort
        df_s = (
            df_agg.filter(pl.col("Season") == season)
                  .drop("Season") # Drop partition col
                  .sort(["personId", "PhaseGroup", "ShotType_Simple", "bin_id"])
        )
        
        df_s.write_parquet(out_path, compression="snappy", row_group_size=200_000)
        log(f"Wrote {out_path}")

    log("Done.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="nba_shot_data_final.parquet")
    ap.add_argument("--output_dir", default="hex_cache")
    ap.add_argument("--gridsize", type=int, default=30)
    ap.add_argument("--legacy_to_plot_x0", type=float, default=887.5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    build_hex_cache(
        input_path=args.input,
        output_dir=args.output_dir,
        gridsize=args.gridsize,
        legacy_to_plot_x0=args.legacy_to_plot_x0,
        force=args.force,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
