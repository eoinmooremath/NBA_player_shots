# court_drawing.py
#
# Full-court NBA drawing + hexbin helpers for your shot chart app.
#
# Coordinate conventions ("viz units"):
#   - 1 viz unit = 0.1 ft (tenths of feet)
#   - Court: 94ft x 50ft -> 940 x 500 viz units
#   - Two y modes:
#       centered_y=True  : y in [-250..250], hoop centerline at y=0  (use y_viz_centered)
#       centered_y=False : y in [0..500],  hoop centerline at y=250 (use y_viz)
#
# Data expectations:
#   - x_viz always in viz units (0..940 typically)
#   - y_viz in viz units (0..500)
#   - y_viz_centered = y_viz - 250  (in [-250..250])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.colors import LogNorm


# -----------------------------
# Dark mode styling
# -----------------------------
COURT_BG_COLOR = "#000000"
LINE_COLOR = "#FFFFFF"
LINE_WIDTH = 1.5


# -----------------------------
# Court geometry in viz units
# -----------------------------
COURT_L_VIZ = 940.0
COURT_W_VIZ = 500.0
HALF_W_VIZ = COURT_W_VIZ / 2.0  # 250

RIM_OFFSET_VIZ = 52.5           # 5.25 ft * 10
RIM_RADIUS_VIZ = 7.5            # 0.75 ft * 10
BACKBOARD_OFFSET_VIZ = 40.0     # 4.0 ft * 10

LEFT_RIM_X = RIM_OFFSET_VIZ
RIGHT_RIM_X = COURT_L_VIZ - RIM_OFFSET_VIZ


def _y0(centered_y: bool) -> float:
    """Return the y coordinate of the hoop centerline in the chosen coordinate mode."""
    return 0.0 if centered_y else HALF_W_VIZ


def draw_court(ax=None, centered_y: bool = True):
    """
    Draw a FULL NBA court (horizontal) in viz units.

    centered_y=True:
        y runs from [-250..250], rim is at y=0 (matches y_viz_centered)
    centered_y=False:
        y runs from [0..500], rim is at y=250 (matches y_viz)

    Typical use:
        draw_court(ax, centered_y=True)
        ax.hexbin(df.x_viz, df.y_viz_centered, ...)
    """
    if ax is None:
        ax = plt.gca()

    ax.set_facecolor(COURT_BG_COLOR)
    y0 = _y0(centered_y)

    # --- Court boundary ---
    if centered_y:
        ax.add_patch(
            Rectangle((0, -HALF_W_VIZ), COURT_L_VIZ, COURT_W_VIZ,
                      linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor="none")
        )
    else:
        ax.add_patch(
            Rectangle((0, 0), COURT_L_VIZ, COURT_W_VIZ,
                      linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor="none")
        )

    # --- Hoops & backboards ---
    # Left hoop
    ax.add_patch(
        Circle((LEFT_RIM_X, y0), radius=RIM_RADIUS_VIZ,
               linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor="none")
    )
    ax.plot([BACKBOARD_OFFSET_VIZ, BACKBOARD_OFFSET_VIZ],
            [y0 - 30, y0 + 30], color=LINE_COLOR, linewidth=LINE_WIDTH)

    # Right hoop
    ax.add_patch(
        Circle((RIGHT_RIM_X, y0), radius=RIM_RADIUS_VIZ,
               linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor="none")
    )
    ax.plot([COURT_L_VIZ - BACKBOARD_OFFSET_VIZ, COURT_L_VIZ - BACKBOARD_OFFSET_VIZ],
            [y0 - 30, y0 + 30], color=LINE_COLOR, linewidth=LINE_WIDTH)

    # --- The paint (lane) ---
    # Lane: 16ft wide -> 160 units, half=80
    # Lane depth: 19ft -> 190 units
    lane_half = 80.0
    lane_depth = 190.0

    # Left lane rectangle
    ax.add_patch(
        Rectangle((0, y0 - lane_half), lane_depth, 2 * lane_half,
                  linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor="none")
    )
    # Free throw circle arcs (6ft radius -> 60 units, diameter 120)
    ax.add_patch(
        Arc((lane_depth, y0), 120, 120, theta1=270, theta2=90,
            linewidth=LINE_WIDTH, color=LINE_COLOR)
    )
    ax.add_patch(
        Arc((lane_depth, y0), 120, 120, theta1=90, theta2=270,
            linewidth=LINE_WIDTH, color=LINE_COLOR, linestyle="--")
    )

    # Right lane rectangle
    rx0 = COURT_L_VIZ - lane_depth
    ax.add_patch(
        Rectangle((rx0, y0 - lane_half), lane_depth, 2 * lane_half,
                  linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor="none")
    )
    ax.add_patch(
        Arc((COURT_L_VIZ - lane_depth, y0), 120, 120, theta1=90, theta2=270,
            linewidth=LINE_WIDTH, color=LINE_COLOR)
    )
    ax.add_patch(
        Arc((COURT_L_VIZ - lane_depth, y0), 120, 120, theta1=270, theta2=90,
            linewidth=LINE_WIDTH, color=LINE_COLOR, linestyle="--")
    )

    # --- Restricted area (4ft radius -> 40 units, diameter 80) centered on rim ---
    ax.add_patch(
        Arc((LEFT_RIM_X, y0), 80, 80, theta1=270, theta2=90,
            linewidth=LINE_WIDTH, color=LINE_COLOR)
    )
    ax.add_patch(
        Arc((RIGHT_RIM_X, y0), 80, 80, theta1=90, theta2=270,
            linewidth=LINE_WIDTH, color=LINE_COLOR)
    )

    # --- Three point lines ---
    # 23.75ft radius => 237.5 units, diameter 475
    # Corner 3 is 22ft from hoop centerline -> used for arc angle
    angle = np.degrees(np.arcsin(22.0 / 23.75))
    arc_diam = 475.0

    # Corner 3 horizontal ticks at y=±22ft (±220 units from hoop centerline)
    # Baseline to corner intersection x ~ 14ft; you used 140 which is fine aesthetically.
    corner_x = 140.0
    corner_y = 220.0

    # Left corners (short baseline segment)
    ax.plot([0, corner_x], [y0 - corner_y, y0 - corner_y], color=LINE_COLOR, linewidth=LINE_WIDTH)
    ax.plot([0, corner_x], [y0 + corner_y, y0 + corner_y], color=LINE_COLOR, linewidth=LINE_WIDTH)
    ax.add_patch(
        Arc((LEFT_RIM_X, y0), arc_diam, arc_diam, theta1=-angle, theta2=angle,
            linewidth=LINE_WIDTH, color=LINE_COLOR)
    )

    # Right corners
    ax.plot([COURT_L_VIZ, COURT_L_VIZ - corner_x], [y0 - corner_y, y0 - corner_y],
            color=LINE_COLOR, linewidth=LINE_WIDTH)
    ax.plot([COURT_L_VIZ, COURT_L_VIZ - corner_x], [y0 + corner_y, y0 + corner_y],
            color=LINE_COLOR, linewidth=LINE_WIDTH)
    ax.add_patch(
        Arc((RIGHT_RIM_X, y0), arc_diam, arc_diam, theta1=180 - angle, theta2=180 + angle,
            linewidth=LINE_WIDTH, color=LINE_COLOR)
    )

    # --- Center court ---
    mid_x = COURT_L_VIZ / 2.0
    if centered_y:
        ax.plot([mid_x, mid_x], [-HALF_W_VIZ, HALF_W_VIZ], color=LINE_COLOR, linewidth=LINE_WIDTH)
    else:
        ax.plot([mid_x, mid_x], [0, COURT_W_VIZ], color=LINE_COLOR, linewidth=LINE_WIDTH)

    ax.add_patch(Circle((mid_x, y0), radius=60, linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor="none"))
    ax.add_patch(Circle((mid_x, y0), radius=20, linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor="none"))

    # --- Hash marks ---
    def draw_lane_hash(base_x, is_left_side):
        direction = 1 if is_left_side else -1
        marks_dist = [70, 80, 110, 140]
        for dist in marks_dist:
            x_pos = base_x + (dist * direction)
            ax.plot([x_pos, x_pos], [y0 + 80, y0 + 85], color=LINE_COLOR, linewidth=LINE_WIDTH)
            ax.plot([x_pos, x_pos], [y0 - 80, y0 - 85], color=LINE_COLOR, linewidth=LINE_WIDTH)

    draw_lane_hash(0, True)
    draw_lane_hash(COURT_L_VIZ, False)

    # Sideline hash marks (approx positions)
    ax.plot([280, 280], [y0 - HALF_W_VIZ, y0 - (HALF_W_VIZ - 30)], color=LINE_COLOR, linewidth=LINE_WIDTH)
    ax.plot([280, 280], [y0 + HALF_W_VIZ, y0 + (HALF_W_VIZ - 30)], color=LINE_COLOR, linewidth=LINE_WIDTH)
    ax.plot([COURT_L_VIZ - 280, COURT_L_VIZ - 280], [y0 - HALF_W_VIZ, y0 - (HALF_W_VIZ - 30)],
            color=LINE_COLOR, linewidth=LINE_WIDTH)
    ax.plot([COURT_L_VIZ - 280, COURT_L_VIZ - 280], [y0 + HALF_W_VIZ, y0 + (HALF_W_VIZ - 30)],
            color=LINE_COLOR, linewidth=LINE_WIDTH)

    # Limits / aspect
    ax.set_xlim(-50, COURT_L_VIZ + 50)
    if centered_y:
        ax.set_ylim(-260, 260)
    else:
        ax.set_ylim(-10, COURT_W_VIZ + 10)

    ax.set_aspect(1)
    ax.axis("off")
    return ax


def generate_hexbin_chart(
    data: "pd.DataFrame",
    ax,
    map_type: str = "Efficiency",
    gridsize: int = 30,
    centered_y: bool = True,
):
    """
    Hexbin overlay.

    Assumes:
      - x_viz exists (viz units)
      - if centered_y=True: y_viz_centered exists ([-250..250])
      - if centered_y=False: y_viz exists ([0..500])

    map_type:
      - "Efficiency": mean ShotOutcome per bin
      - "Frequency": count per bin (log)
      - "Points Added": sum PointsGenerated over made shots (log color norm)
    """
    draw_court(ax, centered_y=centered_y)

    if "x_viz" not in data.columns:
        raise KeyError("Missing column 'x_viz' in data.")

    if centered_y:
        if "y_viz_centered" not in data.columns:
            raise KeyError("centered_y=True requires column 'y_viz_centered'.")
        y_col = "y_viz_centered"
        extent = [-50, 990, -260, 260]
        clip_box = Rectangle((0, -250), 940, 500, transform=ax.transData)
    else:
        if "y_viz" not in data.columns:
            raise KeyError("centered_y=False requires column 'y_viz'.")
        y_col = "y_viz"
        extent = [-50, 990, -10, 510]
        clip_box = Rectangle((0, 0), 940, 500, transform=ax.transData)

    plot_data = data
    bins_arg = None
    norm_arg = None

    if map_type == "Efficiency":
        if "ShotOutcome" not in plot_data.columns:
            raise KeyError("Efficiency mode requires column 'ShotOutcome'.")
        c_metric = plot_data["ShotOutcome"]
        reduce_fn = np.mean
        cmap = "RdYlBu_r"
        lbl = "FG% (Blue=Cold, Red=Hot)"
        min_cnt = 1

    elif map_type == "Frequency":
        c_metric = None
        reduce_fn = None
        cmap = "viridis"
        lbl = "Shot Volume (Log Scale)"
        min_cnt = 1
        bins_arg = "log"

    elif map_type == "Points Added":
        if "PointsGenerated" not in plot_data.columns:
            raise KeyError("Points Added mode requires column 'PointsGenerated'.")
        if "ShotOutcome" not in plot_data.columns:
            raise KeyError("Points Added mode requires column 'ShotOutcome'.")
        plot_data = plot_data[plot_data["ShotOutcome"] == 1].copy()
        c_metric = plot_data["PointsGenerated"]
        reduce_fn = np.sum
        cmap = "spring"
        lbl = "Total Points Scored (Log Scale)"
        min_cnt = 1
        norm_arg = LogNorm()

    else:
        raise ValueError(f"Unknown map_type: {map_type}")

    xvals = plot_data["x_viz"]
    yvals = plot_data[y_col]

    hb = ax.hexbin(
        xvals,
        yvals,
        C=c_metric,
        gridsize=gridsize,
        extent=extent,
        cmap=cmap,
        mincnt=min_cnt,
        reduce_C_function=reduce_fn,
        bins=bins_arg,
        norm=norm_arg,
        edgecolors=COURT_BG_COLOR,
        linewidths=0.5,
        alpha=0.9,
        zorder=2,
    )

    hb.set_clip_path(clip_box)
    return hb, lbl


def sanity_overlay(ax=None, centered_y: bool = True):
    """
    Optional helper to draw reference circles around the RIGHT rim:
      - restricted area (4ft)
      - 3pt arc (23.75ft)
    Useful for debugging coordinate alignment.
    """
    if ax is None:
        ax = plt.gca()
    y0 = _y0(centered_y)

    # restricted area radius 4ft => 40 units
    ax.add_patch(Arc((RIGHT_RIM_X, y0), 80, 80, theta1=0, theta2=360,
                     linewidth=1.0, color="#888888", linestyle=":"))

    # 3pt radius 23.75ft => 237.5 units, diameter 475
    ax.add_patch(Arc((RIGHT_RIM_X, y0), 475, 475, theta1=0, theta2=360,
                     linewidth=1.0, color="#888888", linestyle=":"))

    return ax
