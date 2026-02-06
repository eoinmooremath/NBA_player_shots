# hex_cache_builder.py
#
# Builds a pre-aggregated hex cache for fast Streamlit rendering.
#
# Cache version 2:
#   hex_cache/
#     _meta.json
#     _grid.parquet
#     Season=2024/
#       PhaseGroup=regular/hex.parquet
#       PhaseGroup=postseason/hex.parquet
#       ...
#
# IMPORTANT (schema + dataset compatibility):
# - Each hex.parquet intentionally stores ONLY:
#     personId, ShotType_Simple, bin_id, attempts, makes, points
# - We do NOT store Season or PhaseGroup inside hex.parquet because they are
#   already encoded in the hive-style directory partitions. Keeping them
#   inside the files can cause pyarrow dataset schema conflicts (e.g. int16 vs
#   dictionary-encoded partition fields) when reading the whole directory.
# - We write all hex.parquet with a fixed PyArrow schema and with
#   use_dictionary=False to avoid dictionary-encoding type drift across files.
#
# Notes:
# - Hook is merged into Jump (taxonomy is centralized in shot_type_bucketing.py).
# - PhaseGroup is normalized to safe slugs to avoid space/%20 issues.
#
import argparse
import io
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

# Import works both as a package module and as a standalone script.
try:
    from .shot_type_bucketing import shot_type_simple_from_subtype
except Exception:
    from nba_shots.shot_type_bucketing import shot_type_simple_from_subtype


# Plot-space constants (centered court, right basket)
XMIN, XMAX = 0.0, 940.0
YMIN, YMAX = -250.0, 250.0

# Clip region (full court)
CLIP_X0 = 0.0
CLIP_W = 940.0
CLIP_Y0 = -250.0
CLIP_H = 500.0

# Fixed schema for all hex.parquet files (no Season/PhaseGroup inside)
HEX_SCHEMA = pa.schema(
    [
        pa.field("personId", pa.int32()),
        pa.field("ShotType_Simple", pa.string()),
        pa.field("bin_id", pa.int32()),
        pa.field("attempts", pa.int32()),
        pa.field("makes", pa.int32()),
        pa.field("points", pa.float32()),
    ]
)


def ensure_shot_type_simple(df: pd.DataFrame) -> pd.DataFrame:
    if "ShotType_Simple" in df.columns:
        st = df["ShotType_Simple"].fillna("").astype(str).str.strip()
        # Back-compat: merge old Hook into Jump.
        st = st.replace({"Hook": "Jump"})
        df["ShotType_Simple"] = st
        return df

    if "subType" in df.columns:
        df["ShotType_Simple"] = shot_type_simple_from_subtype(df["subType"]).astype(str)
    else:
        df["ShotType_Simple"] = "Other"
    return df


def ensure_season_phase(df: pd.DataFrame) -> pd.DataFrame:
    if "SeasonPhase" in df.columns:
        df["SeasonPhase"] = df["SeasonPhase"].fillna("Unknown").astype(str)
        return df
    df["SeasonPhase"] = "Unknown"
    return df


def ensure_shot_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a 0/1 made indicator into ShotOutcome.
    """
    if "ShotOutcome" in df.columns:
        df["ShotOutcome"] = pd.to_numeric(df["ShotOutcome"], errors="coerce").fillna(0).astype(np.int16)
        return df

    # Prefer shotResult, else actionType, else pointsTotal
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
    Compute PointsMade (points scored on the play) row-by-row.

    Rule of thumb:
      1) shotValue (if 2/3) is best.
      2) If missing: actionType explicitly 2pt/3pt (very strong).
      3) If missing: description contains "3PT" using regex r"\\b3PT(?!S)\\b" (case-insensitive).
      4) If still missing: distance heuristic (weak, but better than 0).
      5) If even distance missing: default to 2 (safest).

    PointsMade = ShotOutcome * inferred_value
    """
    # Made/missed indicator
    made = pd.to_numeric(df["ShotOutcome"], errors="coerce").fillna(0).astype(np.int8)

    # 1) Trusted shotValue where valid
    if "shotValue" in df.columns:
        sv = pd.to_numeric(df["shotValue"], errors="coerce")
        ok_sv = sv.isin([2, 3])
    else:
        sv = pd.Series(np.nan, index=df.index)
        ok_sv = pd.Series(False, index=df.index)

    # Start with NaN and fill in confidence order
    val = pd.Series(np.nan, index=df.index, dtype="float32")

    # Use shotValue when it's 2 or 3
    if ok_sv.any():
        val.loc[ok_sv] = sv.loc[ok_sv].astype(np.float32)

    # 2) actionType explicitly 2pt/3pt
    action = df.get("actionType", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    need = val.isna()

    if need.any():
        is_3pt = action.eq("3pt") | action.str.contains("3pt", na=False)
        is_2pt = action.eq("2pt") | action.str.contains("2pt", na=False)

        idx = need & is_3pt
        if idx.any():
            val.loc[idx] = 3.0

        idx = need & is_2pt
        if idx.any():
            val.loc[idx] = 2.0

    # 3) description contains 3PT token (NOT 3PTS), regex: r"\b3PT(?!S)\b"
    need = val.isna()
    if need.any():
        desc = df.get("description", pd.Series("", index=df.index)).astype(str)
        desc_has_3pt = desc.str.contains(r"\b3PT(?!S)\b", case=False, regex=True, na=False)

        idx = need & desc_has_3pt
        if idx.any():
            val.loc[idx] = 3.0

    # 4) distance heuristic (weak): if still unknown, use >=22 => 3 else 2
    need = val.isna()
    if need.any():
        dist = pd.to_numeric(df.get("shotDistance", pd.Series(np.nan, index=df.index)), errors="coerce")
        idx3 = need & dist.notna() & (dist >= 22.0)
        idx2 = need & dist.notna() & (dist < 22.0)

        if idx3.any():
            val.loc[idx3] = 3.0
        if idx2.any():
            val.loc[idx2] = 2.0

    # 5) final fallback: default to 2
    val = val.fillna(2.0).astype(np.float32)

    df["PointsMade"] = (made.to_numpy(np.float32) * val.to_numpy(np.float32)).astype(np.float32)
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
    hb = ax.hexbin(
        xs, ys,
        gridsize=gridsize,
        extent=extent,
        mincnt=1,
        linewidths=0.0,
        edgecolors="none",
    )
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
        "cache_data_file": "hex.parquet",
        "partition_cols": ["Season", "PhaseGroup"],
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
    """
    Canonical transform (matches the Streamlit app's centered-y court):
      x_plot = legacy_to_plot_x0 - yLegacy
      y_plot_centered = xLegacy
    """
    xL = pd.to_numeric(df["xLegacy"], errors="coerce").to_numpy(np.float32)
    yL = pd.to_numeric(df["yLegacy"], errors="coerce").to_numpy(np.float32)

    x_plot = (legacy_to_plot_x0 - yL).astype(np.float32)
    y_plot = xL.astype(np.float32)

    return x_plot, y_plot


def phase_group_from_season_phase(season_phase: pd.Series) -> pd.Series:
    """
    Normalize SeasonPhase into safe slugs:
      - regular: includes regular season + in-season tournament / cup variants
      - preseason
      - postseason: playoffs + play-in variants
      - unknown
    """
    sp = season_phase.fillna("Unknown").astype(str).str.strip().str.lower()

    is_pre = sp.eq("preseason")

    is_reg = sp.eq("regular season") | sp.isin(
        [
            "in-season tournament",
            "in season tournament",
            "nba cup",
            "in-season tournament finals",
            "in season tournament finals",
        ]
    )

    is_post = sp.isin(
        [
            "playoffs",
            "play-in tournament",
            "play in tournament",
            "play-in",
            "play in",
        ]
    )

    out = np.where(is_pre, "preseason", np.where(is_reg, "regular", np.where(is_post, "postseason", "unknown")))
    return pd.Series(out, index=season_phase.index, dtype="string")


def _write_hex_parquet_consistent(path: str, df_hex: pd.DataFrame, row_group_size: int = 100_000):
    """
    Write a parquet file with a fixed schema and no dictionary encoding,
    so all files are dataset-merge compatible.
    """
    # Ensure exactly the desired columns, in order
    df_hex = df_hex[["personId", "ShotType_Simple", "bin_id", "attempts", "makes", "points"]].copy()

    # Enforce stable pandas dtypes before Arrow conversion
    df_hex["personId"] = pd.to_numeric(df_hex["personId"], errors="coerce").fillna(0).astype(np.int32)
    df_hex["bin_id"] = pd.to_numeric(df_hex["bin_id"], errors="coerce").fillna(0).astype(np.int32)
    df_hex["attempts"] = pd.to_numeric(df_hex["attempts"], errors="coerce").fillna(0).astype(np.int32)
    df_hex["makes"] = pd.to_numeric(df_hex["makes"], errors="coerce").fillna(0).astype(np.int32)
    df_hex["points"] = pd.to_numeric(df_hex["points"], errors="coerce").fillna(0).astype(np.float32)
    df_hex["ShotType_Simple"] = df_hex["ShotType_Simple"].fillna("").astype(str)

    table = pa.Table.from_pandas(df_hex, schema=HEX_SCHEMA, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="snappy",
        use_dictionary=False,      # critical for consistent Arrow types across files
        row_group_size=row_group_size,
        write_statistics=True,
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
    meta["hex_schema"] = {f.name: str(f.type) for f in HEX_SCHEMA}

    grid_path = os.path.join(output_dir, "_grid.parquet")
    meta_path = os.path.join(output_dir, "_meta.json")
    grid_df.to_parquet(grid_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    log(f"Wrote {grid_path}")
    log(f"Wrote {meta_path}")

    log(f"Loading {input_path} ...")

    # Read only required columns (saves time/memory for huge input)
    needed_cols = [
        "Season", "personId", "xLegacy", "yLegacy",
        "SeasonPhase", "ShotType_Simple", "subType",
        "ShotOutcome", "shotResult", "actionType", "pointsTotal",
        "shotValue", "description", "shotDistance",
    ]
    df = pd.read_parquet(input_path, columns=[c for c in needed_cols if c in pd.read_parquet(input_path, engine="pyarrow").columns])

    # Ensure required columns exist
    for c in ["Season", "personId", "xLegacy", "yLegacy"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Canonical numeric types for processing
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

    # phase group (safe slug)
    df["PhaseGroup"] = phase_group_from_season_phase(df["SeasonPhase"]).astype(str)

    # assign bins
    df["bin_id"] = assign_bins_matplotlib_compatible(x_plot, y_plot, gridsize, extent)

    # aggregate
    agg = (
        df.groupby(["Season", "PhaseGroup", "personId", "ShotType_Simple", "bin_id"], sort=False)
          .agg(
              attempts=("bin_id", "size"),
              makes=("ShotOutcome", "sum"),
              points=("PointsMade", "sum"),
          )
          .reset_index()
    )

    # Stable dtypes (processing)
    agg["Season"] = pd.to_numeric(agg["Season"], errors="coerce").fillna(0).astype(np.int32)
    agg["personId"] = pd.to_numeric(agg["personId"], errors="coerce").fillna(0).astype(np.int32)
    agg["bin_id"] = pd.to_numeric(agg["bin_id"], errors="coerce").fillna(0).astype(np.int32)
    agg["attempts"] = pd.to_numeric(agg["attempts"], errors="coerce").fillna(0).astype(np.int32)
    agg["makes"] = pd.to_numeric(agg["makes"], errors="coerce").fillna(0).astype(np.int32)
    agg["points"] = pd.to_numeric(agg["points"], errors="coerce").fillna(0).astype(np.float32)
    agg["PhaseGroup"] = agg["PhaseGroup"].fillna("unknown").astype(str)
    agg["ShotType_Simple"] = agg["ShotType_Simple"].fillna("Other").astype(str).replace({"Hook": "Jump"})

    # Write one parquet per (Season, PhaseGroup)
    data_file = meta["cache_data_file"]
    seasons = agg["Season"].unique().tolist()

    for season in sorted(seasons):
        df_s = agg[agg["Season"] == int(season)]
        for pg in meta["phase_groups_all"]:
            df_sp = df_s[df_s["PhaseGroup"] == pg]
            if df_sp.empty:
                continue

            out_dir = os.path.join(output_dir, f"Season={int(season)}", f"PhaseGroup={pg}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, data_file)

            # Only write the canonical file schema (no Season/PhaseGroup columns inside)
            df_out = df_sp[["personId", "ShotType_Simple", "bin_id", "attempts", "makes", "points"]].copy()

            # Sorting helps parquet row-group filtering by personId.
            df_out = df_out.sort_values(["personId", "ShotType_Simple", "bin_id"], kind="mergesort")

            _write_hex_parquet_consistent(out_path, df_out, row_group_size=100_000)
            log(f"Wrote {out_path}  rows={len(df_out):,}")

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
