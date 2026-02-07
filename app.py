# app.py
import os
import json
import base64
from io import BytesIO
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

from nba_shots import court_drawing as cu

HEX_CACHE_DIR = "hex_cache"
META_PATH = os.path.join(HEX_CACHE_DIR, "_meta.json")
PLAYER_INDEX_PATH = "player_index.parquet"

# -------------------------
# Figure sizes (control pixel geometry)
# -------------------------
MAP_FIG_W_IN = 12.0
MAP_FIG_H_IN = 5.25
BAR_FIG_W_IN = 6.0
BAR_FIG_H_IN = 5.25

# Make plots taller without changing width
TALLER = 1.25
MAP_FIG_H_IN *= TALLER
BAR_FIG_H_IN *= TALLER

MAP_ASPECT = MAP_FIG_W_IN / MAP_FIG_H_IN
BAR_ASPECT = BAR_FIG_W_IN / BAR_FIG_H_IN

MAP_DPI = 110
BAR_DPI = 110

# IMPORTANT: use identical layout rectangles for BOTH the cached background and the overlay.
# These are figure-relative [left, bottom, width, height].
MAP_AX_RECT = [0.04, 0.24, 0.92, 0.70]  # court + hexes
CB_AX_RECT = [0.12, 0.09, 0.76, 0.07]   # colorbar strip (wider -> avoids label cutoff)

st.set_page_config(page_title="NBA Player Shots", layout="wide", page_icon="🏀")

# -------------------------
# Styling (UI + alignment)
# -------------------------
st.markdown(
    f"""
    <style>
    html, body {{ margin: 0; padding: 0; }}
    .stApp {{ background-color: #0e1117; color: #ffffff; }}

    /* ---- STICKY SIDEBAR FOOTER FIX ---- */
    
    /* 1. Target the vertical stack of widgets inside the sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        min-height: 92vh;     /* Force the widget stack to be tall */
        display: flex;        /* Turn it into a flex container */
        flex-direction: column;
    }}

    /* 2. Push the LAST widget (your 'How to Use' div) to the bottom */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:last-child {{
        margin-top: auto;     /* Auto margin pushes it down */
        padding-bottom: 20px; /* Add breathing room at the bottom */
    }}
    
    /* ----------------------------------- */

    header[data-testid="stHeader"] {{ display: none; }}
    div[data-testid="stToolbar"] {{ display: none; }}
    #MainMenu {{ display: none; }}
    footer {{ display: none; }}

    .block-container {{
        padding-top: 0.8rem !important;
        padding-bottom: 0.10rem !important;
    }}
    [data-testid="stAppViewContainer"] {{ padding-bottom: 0 !important; }}
    [data-testid="stAppViewContainer"] > .main {{ padding-bottom: 0 !important; }}

    [data-testid="stSidebar"] {{ background-color: #161b22; }}
    h1, h2, h3, p, span, label {{ color: #e6e6e6 !important; }}

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {{
        font-size: 18px;
        font-weight: 650;
        color: #e6e6e6 !important;
        margin-bottom: 0.1rem !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: #0d1117 !important;
        color: #ffffff !important;
        border-color: #30363d !important;
    }}
    div[data-baseweb="select"] input {{
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }}
    div[data-baseweb="popover"] div {{
        background-color: #0d1117 !important;
        color: #ffffff !important;
    }}

    div[role="radiogroup"] label {{ color: #ffffff !important; }}

    [data-testid="stSidebar"] h2 {{ font-size: 22px; font-weight: 800; color: #ffffff !important; }}
    [data-testid="stSidebar"] h3 {{ font-size: 18px; font-weight: 800; color: #ffffff !important; }}

    div[data-testid="stHorizontalBlock"] {{ gap: 1rem; }}

    a {{
        color: #58a6ff !important;
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}

    .app-title {{
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #c9d1d9;
        margin: 0.25rem 0 0.35rem 0;
        width: 100%;
        text-align: center;
    }}

    .header-row {{
        display: flex;
        align-items: baseline;
        gap: 16px;
        margin-top: 2px;
        flex-wrap: nowrap;
        min-width: 0;
    }}
    .player-name {{
        font-size: 42px;
        font-weight: 850;
        color: #ffffff;
        line-height: 1.1;
        white-space: nowrap;
    }}
    .teams-played {{
        font-size: 18px;
        color: #c9d1d9;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 60vw;
        min-width: 0;
    }}
    .date-range {{
        margin-top: 4px;
        font-size: 22px;
        color: #e6e6e6;
    }}

    .panel-title {{
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        margin: 0.25rem 0 0.35rem 0;
        line-height: 1.1;
    }}
    .control-label {{
        font-size: 18px;
        font-weight: 800;
        color: #ffffff;
        margin: 0.15rem 0 0.15rem 0;
        line-height: 1.1;
    }}

    .kpi-divider {{
        height: 1px;
        background: rgba(255,255,255,0.10);
        margin: 0.55rem 0 0.45rem 0;
    }}
    .kpi-header-row {{
        display: flex;
        align-items: baseline;
        gap: 10px;
        flex-wrap: wrap;
        margin: 0.05rem 0 0.25rem 0;
    }}
    .kpi-title {{
        font-size: 22px;
        font-weight: 900;
        color: #c9d1d9;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0;
    }}
    .kpi-context {{
        font-size: 18px;
        font-weight: 650;
        color: #e6e6e6;
        margin: 0;
        opacity: 0.92;
    }}

    .plot-wrap {{
        position: relative;
        width: 100%;
        background: #0e1117;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 12px;
        overflow: hidden;
    }}
    .plot-wrap.map {{ aspect-ratio: {MAP_ASPECT:.6f}; }}
    .plot-wrap.bar {{ aspect-ratio: {BAR_ASPECT:.6f}; }}

    .plot-wrap .skel {{
        position: absolute;
        inset: 0;
        background: #0e1117;
        z-index: 0;
    }}

    .plot-wrap img {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }}
    .plot-wrap img.bg {{ z-index: 1; }}
    .plot-wrap img.ov {{ z-index: 2; }}

    .overlay-msg {{
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 3;
        color: #c9d1d9;
        font-size: 18px;
        font-weight: 650;
        padding: 0 12px;
        text-align: center;
        pointer-events: none;
        opacity: 0.95;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(2px); }}
        to   {{ opacity: 1; transform: translateY(0px); }}
    }}
    .fade-in {{
        animation: fadeIn 180ms ease-out;
    }}
    .no-anim {{
        animation: none !important;
    }}

    .howto {{
        margin-top: 10px;
        padding: 10px 12px;
        border-radius: 12px;
        background: rgba(255,255,255,0.03);
    }}
    .howto-title {{
        font-size: 22px;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #c9d1d9;
        margin: 0 0 6px 0;
    }}
    .howto-body {{
        font-size: 16px;
        color: #e6e6e6;
        opacity: 0.92;
        line-height: 1.35;
        margin: 0;
    }}
    .howto-body ul {{
        margin: 6px 0 6px 1.1rem;
        padding: 0;
    }}
    .howto-body li {{
        margin: 2px 0;
    }}
    .howto-meta {{
        margin-top: 8px;
        font-size: 16px;
        color: #e6e6e6;
        opacity: 0.85;
        line-height: 1.35;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
# -------------------------
# Utilities
# -------------------------
def _png_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def fig_to_png_bytes(fig: plt.Figure, transparent: bool) -> bytes:
    buf = BytesIO()
    # IMPORTANT: no tight bbox; fixed canvas -> stable pixel sizes and no layout drift
    fig.savefig(
        buf,
        format="png",
        dpi=fig.dpi,
        bbox_inches=None,
        pad_inches=0.0,
        transparent=transparent,
    )
    buf.seek(0)
    return buf.getvalue()


def map_layered_html(
    bg_png: bytes,
    overlay_png: Optional[bytes],
    animate_overlay: bool,
    message: str = "",
) -> str:
    bg64 = _png_to_b64(bg_png)
    cls = "fade-in" if animate_overlay else "no-anim"
    parts = [
        '<div class="plot-wrap map">',
        '  <div class="skel"></div>',
        f'  <img class="bg" src="data:image/png;base64,{bg64}" />',
    ]
    if overlay_png is not None:
        ov64 = _png_to_b64(overlay_png)
        parts.append(f'  <img class="ov {cls}" src="data:image/png;base64,{ov64}" />')
    if message:
        parts.append(f'  <div class="overlay-msg">{message}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def bar_wrapper_html(png_bytes: Optional[bytes], animate: bool) -> str:
    cls = "fade-in" if animate else "no-anim"
    if png_bytes is None:
        return '<div class="plot-wrap bar"><div class="skel"></div></div>'
    b64 = _png_to_b64(png_bytes)
    return (
        f'<div class="plot-wrap bar">'
        f'  <div class="skel"></div>'
        f'  <img class="{cls}" src="data:image/png;base64,{b64}" />'
        f"</div>"
    )

# -------------------------
# Data loading helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    if not os.path.exists(META_PATH):
        raise FileNotFoundError("Missing hex_cache/_meta.json. Run hex_cache_builder.py.")
    with open(META_PATH, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_centers_from_meta(meta: dict) -> pd.DataFrame:
    xmin, xmax, ymin, ymax = meta["extent"]
    nx = int(meta["nx"])
    ny = int(meta["ny"])
    sx = float(meta["sx"])
    sy = float(meta["sy"])

    i1 = np.arange(nx, dtype=np.int32)
    j1 = np.arange(ny, dtype=np.int32)
    I1, J1 = np.meshgrid(i1, j1)
    x1 = (xmin + (I1.ravel() + 0.5) * sx).astype(np.float32)
    y1 = (ymin + (J1.ravel() + 0.5) * sy).astype(np.float32)
    id1 = (J1.ravel().astype(np.int64) * nx + I1.ravel().astype(np.int64)).astype(np.int32)

    i2 = np.arange(nx + 1, dtype=np.int32)
    j2 = np.arange(ny + 1, dtype=np.int32)
    I2, J2 = np.meshgrid(i2, j2)
    x2 = (xmin + I2.ravel() * sx).astype(np.float32)
    y2 = (ymin + J2.ravel() * sy).astype(np.float32)
    base = nx * ny
    id2 = (base + J2.ravel().astype(np.int64) * (nx + 1) + I2.ravel().astype(np.int64)).astype(np.int32)

    return pd.DataFrame(
        {
            "bin_id": np.concatenate([id1, id2]).astype(np.int32),
            "bin_x": np.concatenate([x1, x2]).astype(np.float32),
            "bin_y": np.concatenate([y1, y2]).astype(np.float32),
        }
    )


@st.cache_data(show_spinner=False)
def load_player_index():
    if not os.path.exists(PLAYER_INDEX_PATH):
        raise FileNotFoundError("Missing player_index.parquet. Run normalize_with_local_data.py first.")

    idx = pd.read_parquet(PLAYER_INDEX_PATH)

    idx["personId"] = pd.to_numeric(idx["personId"], errors="coerce").astype("int64")
    idx["minSeason"] = pd.to_numeric(idx["minSeason"], errors="coerce").astype("int32")
    idx["maxSeason"] = pd.to_numeric(idx["maxSeason"], errors="coerce").astype("int32")
    if "teamsPlayed" not in idx.columns:
        idx["teamsPlayed"] = ""
    idx["teamsPlayed"] = idx["teamsPlayed"].fillna("").astype(str)
    idx["Full_Name"] = idx["Full_Name"].astype(str)

    idx = idx.sort_values(["Full_Name"], kind="mergesort").reset_index(drop=True)

    player_list = idx["Full_Name"].tolist()
    name_to_pid = dict(zip(idx["Full_Name"], idx["personId"]))
    pid_to_bounds = dict(zip(idx["personId"], zip(idx["minSeason"], idx["maxSeason"])))
    pid_to_teams = dict(zip(idx["personId"], idx["teamsPlayed"]))

    return player_list, name_to_pid, pid_to_bounds, pid_to_teams


def season_phase_to_groups(single_phase: str):
    if not single_phase or single_phase.startswith("All"):
        return ["__ALL__"]
    if single_phase == "Regular Season":
        return ["regular"]
    if single_phase == "Preseason":
        return ["preseason"]
    if single_phase == "Postseason":
        return ["postseason"]
    return ["__ALL__"]


@st.cache_data(show_spinner=False)
def load_hex_base(person_id: int, season_start: int, season_end: int, phases_key: tuple) -> pd.DataFrame:
    """
    Cache layout expected:
      hex_cache/Season=YYYY/hex.parquet
    Columns inside each file:
      personId, PhaseGroup, ShotType_Simple, bin_id, attempts, makes, points
    """
    meta = load_meta()
    data_file = str(meta.get("cache_data_file") or meta.get("data_file") or "hex.parquet")

    cols = ["personId", "PhaseGroup", "ShotType_Simple", "bin_id", "attempts", "makes", "points"]
    rows = []
    phases = list(phases_key)

    for season in range(int(season_start), int(season_end) + 1):
        part_file = os.path.join(HEX_CACHE_DIR, f"Season={int(season)}", data_file)
        if not os.path.exists(part_file):
            continue
        try:
            d = pd.read_parquet(
                part_file,
                columns=cols,
                engine="pyarrow",
                filters=[("personId", "==", int(person_id))],
            )
        except Exception:
            continue
        if len(d):
            rows.append(d)

    if not rows:
        return pd.DataFrame(columns=["ShotType_Simple", "PhaseGroup", "bin_id", "attempts", "makes", "points"])

    df = pd.concat(rows, ignore_index=True)

    df["PhaseGroup"] = df["PhaseGroup"].fillna("unknown").astype(str)
    if phases != ["__ALL__"]:
        df = df[df["PhaseGroup"].isin(list(phases))]

    df["ShotType_Simple"] = df["ShotType_Simple"].fillna("Other").astype(str).replace({"Hook": "Jump"})
    df = df.drop(columns=["personId"], errors="ignore")

    out = (
        df.groupby(["ShotType_Simple", "bin_id"], sort=False)[["attempts", "makes", "points"]]
        .sum()
        .reset_index()
    )
    return out


@st.cache_data(show_spinner=False)
def load_hex_agg(person_id: int, season_start: int, season_end: int, shot_type: str, phases_key: tuple) -> pd.DataFrame:
    base = load_hex_base(person_id, season_start, season_end, phases_key)
    if len(base) == 0:
        return pd.DataFrame(columns=["bin_id", "attempts", "makes", "points"])

    if shot_type and shot_type != "All":
        base = base[base["ShotType_Simple"].astype(str) == str(shot_type)]

    out = (
        base.groupby("bin_id", sort=False)[["attempts", "makes", "points"]]
        .sum()
        .reset_index()
    )
    out["bin_id"] = pd.to_numeric(out["bin_id"], errors="coerce").astype("int32")
    out["attempts"] = pd.to_numeric(out["attempts"], errors="coerce").fillna(0).astype("int32")
    out["makes"] = pd.to_numeric(out["makes"], errors="coerce").fillna(0).astype("int32")
    out["points"] = pd.to_numeric(out["points"], errors="coerce").fillna(0).astype("float32")
    return out


def compute_kpis_from_hex(hex_df: pd.DataFrame):
    total_shots = int(hex_df["attempts"].sum()) if len(hex_df) else 0
    made_shots = int(hex_df["makes"].sum()) if total_shots > 0 else 0
    total_points = float(hex_df["points"].sum()) if total_shots > 0 else 0.0
    fg_pct = (made_shots / total_shots * 100.0) if total_shots > 0 else 0.0
    pps = (total_points / total_shots) if total_shots > 0 else 0.0
    return total_shots, made_shots, fg_pct, total_points, pps

# -------------------------
# Cached "empty court" background PNG
# -------------------------
@st.cache_data(show_spinner=False)
def render_empty_court_bg_png(extent: Tuple[float, float, float, float], gridsize: int) -> bytes:
    xmin, xmax, ymin, ymax = extent

    fig = plt.figure(figsize=(MAP_FIG_W_IN, MAP_FIG_H_IN), dpi=MAP_DPI)
    fig.patch.set_facecolor("#0e1117")

    ax = fig.add_axes(MAP_AX_RECT)
    cax = fig.add_axes(CB_AX_RECT)

    ax.set_facecolor("#0e1117")
    cax.set_facecolor("#0e1117")

    # Draw the court once (cached)
    cu.draw_court(ax, centered_y=True)

    # Lock geometry so the court never changes size across views
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.axis("off")

    # Keep colorbar region reserved but empty
    cax.axis("off")

    png = fig_to_png_bytes(fig, transparent=False)
    plt.close(fig)
    return png

# -------------------------
# Hex overlay (transparent) + colorbar (also drawn here)
# -------------------------
def _lock_map_axes(ax, extent: Tuple[float, float, float, float]):
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.axis("off")

def _hexbin_only(ax, merged: pd.DataFrame, view: str, extent, gridsize: int):
    merged = merged[merged["attempts"] > 0].copy()
    if merged.empty:
        return None, "No data", None

    if view == "Frequency":
        C = merged["attempts"].to_numpy(float)
        reduce_fn = np.sum
        cmap = "viridis"
        bins = "log"
        norm = None
        label = "Shot Volume (Log Scale)"

    elif view == "Points Added":
        C = merged["points"].to_numpy(dtype=float)
        reduce_fn = np.sum
        cmap = "plasma"
        bins = None
        vmax = float(np.nanmax(C)) if np.isfinite(C).any() else 1.0
        norm = SymLogNorm(linthresh=1.0, linscale=1.0, base=10, vmin=0, vmax=vmax)
        label = "Total Points Scored (Symlog Scale)"

    else:  # Efficiency
        att = merged["attempts"].to_numpy(float)
        mk = merged["makes"].to_numpy(float)
        fg = np.divide(mk, att, out=np.zeros_like(mk), where=att > 0)
        C = fg
        reduce_fn = np.mean
        cmap = "summer"
        bins = None
        norm = None
        label = "FG% (Low→High)"

    hb = ax.hexbin(
        merged["bin_x"].to_numpy(float),
        merged["bin_y"].to_numpy(float),
        C=C,
        reduce_C_function=reduce_fn,
        gridsize=int(gridsize),
        extent=extent,
        cmap=cmap,
        bins=bins,
        norm=norm,
        mincnt=1,
        linewidths=0.0,
        edgecolors="none",
        alpha=0.95,
        zorder=2,
    )

    # Clip to full court
    clip_box = Rectangle((extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2], transform=ax.transData)
    hb.set_clip_path(clip_box)

    return hb, label, (norm is not None)


@st.cache_data(show_spinner=False)
def render_map_overlay_png(
    person_id: int,
    y0: int,
    y1: int,
    shot_type: str,
    phases_key: tuple,
    view: str,
) -> Tuple[Optional[bytes], str, int, int, float]:
    """
    Returns:
      overlay_png (transparent), message, total_attempts, total_makes, total_points
    """
    meta = load_meta()
    centers_df = load_centers_from_meta(meta)

    extent = tuple(meta["extent"])
    gridsize = int(meta["gridsize"])

    hex_agg = load_hex_agg(person_id, y0, y1, shot_type, phases_key)
    if hex_agg.empty or int(hex_agg["attempts"].sum()) <= 0:
        return None, "No shots found for this selection.", 0, 0, 0.0

    merged = hex_agg.merge(centers_df, on="bin_id", how="left")
    if merged.empty:
        return None, "No shots found for this selection.", 0, 0, 0.0

    # Transparent overlay: only hexes + colorbar (no court)
    fig = plt.figure(figsize=(MAP_FIG_W_IN, MAP_FIG_H_IN), dpi=MAP_DPI)
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes(MAP_AX_RECT)
    cax = fig.add_axes(CB_AX_RECT)

    ax.set_facecolor((0, 0, 0, 0))
    cax.set_facecolor((0, 0, 0, 0))

    _lock_map_axes(ax, extent)

    hb, label, _ = _hexbin_only(ax, merged, view, extent, gridsize)
    if hb is None:
        plt.close(fig)
        return None, "No shots found for this selection.", 0, 0, 0.0

    # Colorbar on transparent canvas; underlying bg is black
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_label(label, color="white", fontsize=10, labelpad=6)
    cb.ax.tick_params(color="white", labelcolor="white", pad=2)
    cb.outline.set_edgecolor("white")
    for spine in cax.spines.values():
        spine.set_color("white")

    overlay_png = fig_to_png_bytes(fig, transparent=True)
    plt.close(fig)

    total_attempts = int(hex_agg["attempts"].sum())
    total_makes = int(hex_agg["makes"].sum())
    total_points = float(hex_agg["points"].sum())
    return overlay_png, "", total_attempts, total_makes, total_points


# -------------------------
# Shot type breakdown plot
# -------------------------
def make_type_breakdown(person_id: int, season_start: int, season_end: int, phases_key: tuple, view: str):
    types = ["Putback", "Dunk", "Layup", "Jump", "Other"]
    base = load_hex_base(person_id, season_start, season_end, phases_key)

    if len(base) == 0:
        df = pd.DataFrame({"Type": types, "attempts": 0, "makes": 0, "points": 0.0, "Value": 0.0})
        xlab = "Shot Volume" if view == "Frequency" else ("Total Points" if view == "Points Added" else "FG% (by type)")
        return df, xlab

    g = (
        base.groupby("ShotType_Simple", sort=False)[["attempts", "makes", "points"]]
        .sum()
        .reindex(types, fill_value=0)
        .reset_index()
        .rename(columns={"ShotType_Simple": "Type"})
    )

    if view == "Efficiency":
        g["Value"] = np.where(g["attempts"] > 0, (g["makes"] / g["attempts"]) * 100.0, 0.0)
        xlab = "FG% (by type)"
    elif view == "Points Added":
        g["Value"] = g["points"]
        xlab = "Total Points"
    else:
        g["Value"] = g["attempts"]
        xlab = "Shot Volume"

    g = g.sort_values("Value", ascending=False)
    return g, xlab


def plot_type_breakdown_bar(bd_df: pd.DataFrame, xlab: str) -> bytes:
    fig, ax = plt.subplots(figsize=(BAR_FIG_W_IN, BAR_FIG_H_IN), dpi=BAR_DPI)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    vals = bd_df["Value"].to_numpy(float)
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0
    if vmax > 0:
        v = vals / vmax
    else:
        v = np.zeros_like(vals)

    # Your chosen scheme (visually distinct from map colormaps)
    cmap = plt.cm.Oranges
    colors = cmap(0.5 - 0.25 * v)

    ax.barh(bd_df["Type"].astype(str).tolist(), vals, color=colors, alpha=0.95)

    ax.set_xlabel(xlab, color="white")
    ax.set_ylabel("", color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    fig.subplots_adjust(left=0.28, right=0.98, top=0.98, bottom=0.14)
    png = fig_to_png_bytes(fig, transparent=False)
    plt.close(fig)
    return png


@st.cache_data(show_spinner=False)
def render_breakdown_png(
    person_id: int,
    y0: int,
    y1: int,
    phases_key: tuple,
    view: str,
) -> bytes:
    bd_df, xlab = make_type_breakdown(person_id, y0, y1, phases_key, view)
    return plot_type_breakdown_bar(bd_df, xlab)

# -------------------------
# Startup checks
# -------------------------
if not os.path.exists(HEX_CACHE_DIR) or not os.path.exists(META_PATH):
    st.error("❌ Missing hex_cache or meta. Run hex_cache_builder.py first.")
    st.stop()

if not os.path.exists(PLAYER_INDEX_PATH):
    st.error("❌ Missing player_index.parquet. Run normalize_with_local_data.py first.")
    st.stop()

meta = load_meta()
extent = tuple(meta["extent"])
gridsize = int(meta["gridsize"])

player_list, name_to_pid, pid_to_bounds, pid_to_teams = load_player_index()

# Cached background (empty court) PNG
bg_png = render_empty_court_bg_png(extent, gridsize)

# -------------------------
# Sidebar: Global filters
# -------------------------
st.sidebar.markdown("## 🏀 Global Filters")

default_player = "LeBron James" if "LeBron James" in player_list else player_list[0]
selected_player = st.sidebar.selectbox("Player", player_list, index=player_list.index(default_player))

person_id = int(name_to_pid.get(selected_player, -1))
if person_id == -1:
    st.error("No personId found for selected player.")
    st.stop()

min_year, max_year = pid_to_bounds.get(person_id, (None, None))
if min_year is None:
    st.error("No season bounds found for selected player.")
    st.stop()

if int(min_year) == int(max_year):
    selected_years = (int(min_year), int(max_year))
    st.sidebar.info(f"Season: {min_year}")
else:
    selected_years = st.sidebar.slider("Season Range", int(min_year), int(max_year), (int(min_year), int(max_year)))

st.sidebar.markdown("### Season Phases")
phase_opts = ["All (includes Unknown)", "Regular Season", "Preseason", "Postseason"]
phase_choice = st.sidebar.radio(
    "Season Phases",
    options=phase_opts,
    index=0,
    label_visibility="collapsed",
    key="season_phase",
)
phases_key = tuple(season_phase_to_groups(phase_choice))
st.sidebar.markdown(
    """
    <div class="howto">
      <div class="howto-title">How to use</div>
        <div class="howto-body">
        Explore NBA shot charts for any player since 1996, when the NBA began collecting modern shot-location data.<br/>
        Choose a view, then filter by Shot Type, Season Phase, and Season Range.
        </div>
      <div class="howto-meta">
        Created <b>February 2026</b>. Updated daily with the </br>
        <a href="https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores?select=PlayByPlay.parquet" target="_blank">latest shot information</a>.
        <br/>
        Inquiries: <a href="mailto:eoinmooremath@gmail.com">eoinmooremath@gmail.com</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# -------------------------
# Main header
# -------------------------
st.markdown('<div class="app-title">NBA PLAYER SHOTS</div>', unsafe_allow_html=True)

teams_played = str(pid_to_teams.get(person_id, "")).strip()
teams_display = teams_played if teams_played and teams_played.lower() != "nan" else ""

st.markdown(
    f"""
    <div class="header-row">
      <div class="player-name">{selected_player}</div>
      <div class="teams-played" title="{teams_display}">{teams_display}</div>
    </div>
    <div class="date-range">Data Range: {selected_years[0]} – {selected_years[1]}</div>
    """,
    unsafe_allow_html=True,
)

# Controls
control_left, control_right = st.columns([2, 1], gap="small")

shot_type_options = ["All", "Putback", "Dunk", "Layup", "Jump", "Other"]
with control_left:
    st.markdown('<div class="control-label">Shot Type</div>', unsafe_allow_html=True)
    selected_shot_type = st.radio(
        "Shot Type",
        shot_type_options,
        horizontal=True,
        index=0,
        key="shot_type",
        label_visibility="collapsed",
    )

with control_right:
    st.markdown('<div class="control-label">View</div>', unsafe_allow_html=True)
    view = st.radio(
        "View",
        ["Efficiency", "Frequency", "Points Added"],
        horizontal=True,
        index=0,
        key="view",
        label_visibility="collapsed",
    )

yr0, yr1 = selected_years

# -------------------------
# Keys (control recompute + animation)
# -------------------------
map_key = (person_id, yr0, yr1, phases_key, view, selected_shot_type)  # map updates with shot type
bar_key = (person_id, yr0, yr1, phases_key, view)                      # breakdown DOES NOT depend on shot type

prev_map_key = st.session_state.get("_map_key")
prev_bar_key = st.session_state.get("_bar_key")

map_changed = (map_key != prev_map_key)
bar_changed = (bar_key != prev_bar_key)

# -------------------------
# Plot region (fixed wrappers)
# -------------------------
col_map, col_bar = st.columns([2, 1], gap="large")

with col_map:
    st.markdown('<div class="panel-title">Shot Map</div>', unsafe_allow_html=True)
    map_ph = st.empty()

with col_bar:
    st.markdown('<div class="panel-title">Shot Type Breakdown</div>', unsafe_allow_html=True)
    bar_ph = st.empty()

# ---- Pre-paint placeholders immediately:
# Map: always show empty court; ONLY show overlay if it matches current key.
current_overlay = st.session_state.get("_map_overlay_png") if (st.session_state.get("_map_key") == map_key) else None
map_ph.markdown(
    map_layered_html(
        bg_png=bg_png,
        overlay_png=current_overlay,
        animate_overlay=False,
        message="",
    ),
    unsafe_allow_html=True,
)

# Bar: show previous bar only if key matches; else show skeleton
current_bar = st.session_state.get("_bar_png") if (st.session_state.get("_bar_key") == bar_key) else None
bar_ph.markdown(bar_wrapper_html(current_bar, animate=False), unsafe_allow_html=True)

# -------------------------
# KPI header (never moves because wrappers are fixed height)
# -------------------------
st.markdown('<div class="kpi-divider"></div>', unsafe_allow_html=True)

season_txt = f"{yr0}–{yr1}" if yr0 != yr1 else f"{yr0}"
phase_txt = "All phases" if phase_choice.startswith("All") else phase_choice
shot_txt = "All shots" if selected_shot_type == "All" else f"{selected_shot_type} shots"
kpi_context = f"{selected_player} · {season_txt} · {phase_txt} · {shot_txt}"

st.markdown(
    f"""
    <div class="kpi-header-row">
      <div class="kpi-title">KPI</div>
      <div class="kpi-context">{kpi_context}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)

# -------------------------
# Compute / update MAP (overlay) only when needed
# -------------------------
if map_changed:
    overlay_png, msg, total_att, total_mk, total_pts = render_map_overlay_png(
        person_id=person_id,
        y0=yr0,
        y1=yr1,
        shot_type=selected_shot_type,
        phases_key=phases_key,
        view=view,
    )
    st.session_state["_map_overlay_png"] = overlay_png
    st.session_state["_map_msg"] = msg
    st.session_state["_map_totals"] = (total_att, total_mk, total_pts)
    st.session_state["_map_key"] = map_key
else:
    overlay_png = st.session_state.get("_map_overlay_png")
    msg = st.session_state.get("_map_msg", "")
    total_att, total_mk, total_pts = st.session_state.get("_map_totals", (0, 0, 0.0))

# Show layered map: empty court background ALWAYS + overlay (transparent) when available
map_ph.markdown(
    map_layered_html(
        bg_png=bg_png,
        overlay_png=overlay_png,
        animate_overlay=map_changed,
        message=msg,
    ),
    unsafe_allow_html=True,
)

# -------------------------
# Compute / update BREAKDOWN only when needed (NOT on shot type changes)
# -------------------------
if bar_changed:
    bar_png = render_breakdown_png(
        person_id=person_id,
        y0=yr0,
        y1=yr1,
        phases_key=phases_key,
        view=view,
    )
    st.session_state["_bar_png"] = bar_png
    st.session_state["_bar_key"] = bar_key
else:
    bar_png = st.session_state.get("_bar_png")

bar_ph.markdown(bar_wrapper_html(bar_png, animate=bar_changed), unsafe_allow_html=True)

# -------------------------
# KPIs (must update with shot type)
# Use cached hex totals directly to avoid re-reading if we already computed map overlay.
# -------------------------
if map_changed:
    total_shots = int(total_att)
    made_shots = int(total_mk)
    total_points = float(total_pts)
    fg_pct = (made_shots / total_shots * 100.0) if total_shots > 0 else 0.0
    pps = (total_points / total_shots) if total_shots > 0 else 0.0
else:
    # safe fallback if no totals stored yet
    hex_agg_for_kpi = load_hex_agg(person_id, yr0, yr1, selected_shot_type, phases_key)
    total_shots, made_shots, fg_pct, total_points, pps = compute_kpis_from_hex(hex_agg_for_kpi)

k1.metric("Total Shots", f"{total_shots:,}")
k2.metric("FG%", f"{fg_pct:.1f}%")
k3.metric("Points", f"{total_points:,.0f}")
k4.metric("Points / Shot", f"{pps:.2f}")

# -------------------------
# Left-column footer/help text (under Shot Map)
# -------------------------

