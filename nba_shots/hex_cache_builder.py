# hex_cache_builder.py
#
# Builds a pre-aggregated hex cache for fast Streamlit rendering.
#
# Cache version 2 (per-season files):
#   hex_cache/
#     _meta.json
#     _grid.parquet
#     Season=2024/hex.parquet
#     Season=2025/hex.parquet
#     ...
#
# Each Season=YYYY/hex.parquet contains rows:
#   personId, PhaseGroup, ShotType_Simple, bin_id, attempts, makes, points
#
# Notes:
# - We do NOT partition by ShotType_Simple.
# - Hook is merged into Jump (taxonomy centralized in shot_type_bucketing.py).
# - PhaseGroup is a safe slug (regular/preseason/postseason/unknown) stored AS A COLUMN
#   (fewer files -> much faster app loads).
#
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.parquet as pq

from nba_shots.shot_type_bucketing import shot_type_simple_from_subtype


# Plot-space constants (centered court, right basket)
XMIN, XMAX = 0.0, 940.0
YMIN, YMAX = -250.0, 250.0

# Clip region (full court)
CLIP_X0 = 0.0
CLIP_W = 940.0
CLIP_Y0 = -250.0
CLIP_H = 500.0


def ensure_shot_type_simple(df: pd.DataFrame) -> pd.DataFrame:
    if "ShotType_Simple" in df.columns:
        st = df["ShotType_Simple"].fillna("").astype(str).str.strip()
        st = st.replace({"Hook": "Jump"})
        df["ShotType_Simple"] = st
        return df

    if "subType" in df.columns:
        df["ShotType_Simple"] = shot_type_simple_from_subtype(df["subType"]).astype(str)
    else:
        df["ShotType_Simple"] = "Other"
    df["ShotType_Simple"] = df["ShotType_Simple"].astype(str).replace({"Hook": "Jump"})
    return df


def ensure_season_phase(df: pd.DataFrame) -> pd.DataFrame:
    if "SeasonPhase" in df.columns:
        df["SeasonPhase"] = df["SeasonPhase"].fillna("Unknown").astype(str)
        return df
    df["SeasonPhase"] = "Unknown"
    return df


def ensure_shot_outcome(df: pd.DataFrame) -> pd.DataFrame:
    if "ShotOutcome" in df.columns:
        df["ShotOutcome"] = pd.to_numeric(df["ShotOutcome"], errors="coerce").fillna(0).astype(np.int16)
        return df

    if "shotResult" in df.columns:
        sr = df["shotResult"].astype(str).str.strip().str.lower()
        made = sr.eq("made")
        missed = sr.eq("missed")
        out = np.where(made, 1, np.where(missed, 0, np.nan))
        df["ShotOutcome"] = pd.Series(out, index=df.index)
    else:
        df["ShotOutcome"] = np.nan

    if "actionType" in df.columns:
        at = df["actionType"].astype(str).str.strip().str.lower()
        df.loc[at.str.contains("made", na=False), "ShotOutcome"] = 1
        df.loc[at.str.contains("miss", na=False), "ShotOutcome"] = 0

    if "pointsTotal" in df.columns:
        pt = pd.to_numeric(df["pointsTotal"], errors="coerce").fillna(0)
        df.loc[pt > 0, "ShotOutcome"] = 1
        df.loc[pt == 0, "ShotOutcome"] = 0

    df["ShotOutcome"] = pd.to_numeric(df["ShotOutcome"], errors="coerce").fillna(0).astype(np.int16)
    return df


def ensure_points_made(df: pd.DataFrame) -> pd.DataFrame:
    """
    Points made on the play:
      - Prefer shotValue in {2,3}
      - Else: actionType when explicitly 2pt/3pt
      - Else: description contains "3PT" where T is NOT followed by S (avoid "PTS")
      - Else: distance heuristic
      - Else: default 2
    """
    out = pd.Series(np.nan, index=df.index, dtype="float32")

    # 1) shotValue is best (when valid)
    if "shotValue" in df.columns:
        sv = pd.to_numeric(df["shotValue"], errors="coerce")
        ok = sv.isin([2, 3])
        out.loc[ok] = sv.loc[ok].astype("float32")

    # 2) actionType explicitly 2pt/3pt (strong)
    if "actionType" in df.columns:
        at = df["actionType"].astype(str).str.strip().str.lower()
        m2 = at.eq("2pt")
        m3 = at.eq("3pt")
        out.loc[out.isna() & m2] = 2.0
        out.loc[out.isna() & m3] = 3.0

    # 3) description "3PT" but not "PTS"
    # pattern: 3PT where T is not immediately followed by S
    if "description" in df.columns:
        desc = df["description"].astype(str)
        is_3pt = desc.str.contains(r"3PT(?!S)", regex=True, na=False)
        out.loc[out.isna() & is_3pt] = 3.0

    # 4) distance heuristic (weak)
    dist = pd.to_numeric(df.get("shotDistance", pd.Series(np.nan, index=df.index)), errors="coerce")
    out.loc[out.isna() & dist.notna() & (dist > 22)] = 3.0
    out.loc[out.isna() & dist.notna() & (dist <= 22)] = 2.0

    # 5) default
    out = out.fillna(2.0).astype("float32")

    df["PointsMade"] = (out * df["ShotOutcome"].astype("float32")).astype("float32")
    return df


def make_matplotlib_template_grid(gridsize: int, extent):
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

    grid = pd.DataFrame(
        {
            "bin_template_index": np.arange(offsets.shape[0], dtype=np.int32),
            "bin_x": offsets[:, 0].astype(np.float32),
            "bin_y": offsets[:, 1].astype(np.float32),
        }
    )

    meta = {
        "cache_version": 2,
        "cache_layout": "per_season_file",
        "cache_data_file": "hex.parquet",
        "partition_cols": ["Season"],
        "phase_col": "PhaseGroup",
        "phase_groups_all": ["regular", "preseason", "postseason", "unknown"],
        "phase_group_labels": {
            "regular": "Regular Season",
            "preseason": "Preseason",
            "postseason": "Postseason",
            "unknown": "Unknown/Other",
        },
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


def assign_bins_matplotlib_compatible(x: np.ndarray, y: np.ndarray, gridsize: int, extent) -> np.ndarray:
    xmin, xmax, ymin, ymax = extent
    nx = int(gridsize)
    ny = int(gridsize)

    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny

    ix1 = np.floor((x - xmin) / sx - 0.5).astype(np.int64)
    iy1 = np.floor((y - ymin) / sy - 0.5).astype(np.int64)
    cx1 = xmin + (ix1 + 0.5) * sx
    cy1 = ymin + (iy1 + 0.5) * sy

    ix2 = np.floor((x - xmin) / sx).astype(np.int64)
    iy2 = np.floor((y - ymin) / sy).astype(np.int64)
    cx2 = xmin + ix2 * sx
    cy2 = ymin + iy2 * sy

    dx1 = x - cx1
    dy1 = y - cy1
    dx2 = x - cx2
    dy2 = y - cy2

    d1 = dx1 * dx1 + 3.0 * dy1 * dy1
    d2 = dx2 * dx2 + 3.0 * dy2 * dy2
    use_primary = d1 <= d2

    ix1c = np.clip(ix1, 0, nx - 1)
    iy1c = np.clip(iy1, 0, ny - 1)

    ix2c = np.clip(ix2, 0, nx)
    iy2c = np.clip(iy2, 0, ny)

    primary_id = (iy1c.astype(np.int64) * nx + ix1c.astype(np.int64))
    secondary_id = (nx * ny) + (iy2c.astype(np.int64) * (nx + 1) + ix2c.astype(np.int64))
    return np.where(use_primary, primary_id, secondary_id).astype(np.int32)


def legacy_to_plot_xy(df: pd.DataFrame, legacy_to_plot_x0: float = 887.5):
    xL = pd.to_numeric(df["xLegacy"], errors="coerce").to_numpy(np.float32)
    yL = pd.to_numeric(df["yLegacy"], errors="coerce").to_numpy(np.float32)
    x_plot = (legacy_to_plot_x0 - yL).astype(np.float32)
    y_plot = xL.astype(np.float32)
    return x_plot, y_plot


def phase_group_from_season_phase(season_phase: pd.Series) -> pd.Series:
    sp = season_phase.fillna("Unknown").astype(str).str.strip().str.lower()

    # Treat in-season / cup as regular
    is_pre = sp.eq("preseason")
    is_reg = sp.eq("regular season") | sp.str.contains("in-season", na=False) | sp.str.contains("cup", na=False)
    is_post = sp.isin(["playoffs", "play-in tournament"]) | sp.str.contains("play-in", na=False) | sp.str.contains("playoff", na=False)

    out = np.where(is_pre, "preseason", np.where(is_reg, "regular", np.where(is_post, "postseason", "unknown")))
    return pd.Series(out, index=season_phase.index, dtype="string")


def _write_parquet_with_schema(df: pd.DataFrame, out_path: str, row_group_size: int = 200_000):
    schema = pa.schema(
        [
            ("personId", pa.int32()),
            ("PhaseGroup", pa.string()),
            ("ShotType_Simple", pa.string()),
            ("bin_id", pa.int32()),
            ("attempts", pa.int32()),
            ("makes", pa.int32()),
            ("points", pa.float32()),
        ]
    )

    # Ensure correct column order and types before Arrow conversion
    df = df[["personId", "PhaseGroup", "ShotType_Simple", "bin_id", "attempts", "makes", "points"]].copy()
    df["personId"] = pd.to_numeric(df["personId"], errors="coerce").fillna(-1).astype("int32")
    df["bin_id"] = pd.to_numeric(df["bin_id"], errors="coerce").fillna(-1).astype("int32")
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0).astype("int32")
    df["makes"] = pd.to_numeric(df["makes"], errors="coerce").fillna(0).astype("int32")
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype("float32")
    df["PhaseGroup"] = df["PhaseGroup"].fillna("unknown").astype(str)
    df["ShotType_Simple"] = df["ShotType_Simple"].fillna("Other").astype(str)

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(
        table,
        out_path,
        compression="snappy",
        use_dictionary=False,          # avoid mixed dictionary encoding surprises
        data_page_size=1 << 20,
        row_group_size=row_group_size,
    )


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

    # Template grid + meta
    grid_df, meta = make_matplotlib_template_grid(gridsize, extent)
    meta["legacy_to_plot_x0"] = float(legacy_to_plot_x0)
    meta["coords_source"] = "xLegacy/yLegacy"
    meta["coords_transform"] = "x_plot = legacy_to_plot_x0 - yLegacy ; y_plot_centered = xLegacy"

    grid_path = os.path.join(output_dir, "_grid.parquet")
    meta_path = os.path.join(output_dir, "_meta.json")
    grid_df.to_parquet(grid_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    log(f"Wrote {grid_path}")
    log(f"Wrote {meta_path}")

    log(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)

    # Required columns
    for c in ["Season", "personId", "xLegacy", "yLegacy"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("int32")
    df["personId"] = pd.to_numeric(df["personId"], errors="coerce").astype("int32")

    df = ensure_shot_type_simple(df)
    df = ensure_season_phase(df)
    df = ensure_shot_outcome(df)
    df = ensure_points_made(df)

    # coords -> plot space
    x_plot, y_plot = legacy_to_plot_xy(df, legacy_to_plot_x0=float(legacy_to_plot_x0))
    m = np.isfinite(x_plot) & np.isfinite(y_plot)

    df = df.loc[m, ["personId", "Season", "SeasonPhase", "ShotType_Simple", "ShotOutcome", "PointsMade"]].copy()
    x_plot = x_plot[m]
    y_plot = y_plot[m]

    # court clip
    in_clip = (
        (x_plot >= CLIP_X0) & (x_plot <= CLIP_X0 + CLIP_W) &
        (y_plot >= CLIP_Y0) & (y_plot <= CLIP_Y0 + CLIP_H)
    )
    df = df.loc[in_clip].copy()
    x_plot = x_plot[in_clip]
    y_plot = y_plot[in_clip]

    # phase group
    df["PhaseGroup"] = phase_group_from_season_phase(df["SeasonPhase"]).astype(str)

    # assign bins
    df["bin_id"] = assign_bins_matplotlib_compatible(x_plot, y_plot, gridsize, extent)

    # aggregate (keep PhaseGroup column inside)
    agg = (
        df.groupby(["Season", "personId", "PhaseGroup", "ShotType_Simple", "bin_id"], sort=False)
          .agg(
              attempts=("bin_id", "size"),
              makes=("ShotOutcome", "sum"),
              points=("PointsMade", "sum"),
          )
          .reset_index()
    )

    # compact dtypes
    agg["Season"] = agg["Season"].astype("int16")
    agg["personId"] = agg["personId"].astype("int32")
    agg["bin_id"] = agg["bin_id"].astype("int32")
    agg["attempts"] = agg["attempts"].astype("int32")
    agg["makes"] = agg["makes"].astype("int32")
    agg["points"] = agg["points"].astype("float32")
    agg["PhaseGroup"] = agg["PhaseGroup"].astype("string")
    agg["ShotType_Simple"] = agg["ShotType_Simple"].astype("string")

    # Write one parquet per Season (PhaseGroup stays as a column)
    data_file = meta["cache_data_file"]
    seasons = sorted(agg["Season"].unique().tolist())

    for season in seasons:
        df_s = agg[agg["Season"] == season].copy()
        if df_s.empty:
            continue

        # sort for better row-group filtering by personId
        df_s = df_s.sort_values(["personId", "PhaseGroup", "ShotType_Simple", "bin_id"], kind="mergesort")

        out_dir = os.path.join(output_dir, f"Season={int(season)}")
        out_path = os.path.join(out_dir, data_file)
        _write_parquet_with_schema(df_s.drop(columns=["Season"]), out_path, row_group_size=200_000)

        log(f"Wrote {out_path} ({len(df_s):,} rows)")

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
