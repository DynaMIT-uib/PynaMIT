"""Geometry illustration."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Wedge

RADIAL = False  # True for radial field lines, False for dipole

RE = 6371.2

fig, ax = plt.subplots(figsize=(8, 8))

rh = 2.0 * RE
RI = RE + 110
textsize = 14

x = y = np.linspace(0, rh, 2000)
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx**2 + yy**2)
r = r.flatten()

# This assumes you have 'conductivity.csv' in the same directory
try:
    cond = pd.read_csv("conductivity.csv")
    cond.index = cond[cond.columns[0]] + RE
    cond = cond[["hall", "pedersen"]]
    ss = np.interp(r, cond.index, cond["pedersen"]).reshape(xx.shape)

    ccc = ax.contourf(
        xx,
        yy,
        ss.reshape(xx.shape),
        cmap=plt.cm.Blues,
        levels=np.linspace(0, 1.5e-4, 100),
        zorder=0,
    )
except FileNotFoundError:
    print("Warning: 'conductivity.csv' not found. Skipping contour plot.")


a = np.linspace(0, np.pi / 2, 100)
ax.fill_between(RE * np.sin(a), np.zeros_like(a), RE * np.cos(a), color="lightgrey")


# Plot magnetic field lines.
B0 = 1
dth0 = np.deg2rad(4)

th = [np.deg2rad(1)]
while True:
    th.append(th[-1] + dth0 * B0 / np.sqrt(4 - 3 * np.sin(th[-1]) ** 2))

    if th[-1] > np.pi / 2:
        break

# Variable to store the t-value of the HIGHEST APEX purple line
highest_apex_t = None

for t in th:
    req = 1 / np.sin(t) ** 2
    th_max = np.pi / 2
    theta = np.linspace(t, th_max, 100)

    r_line = req * np.sin(theta) ** 2 * RE + 110
    x_line, y_line = r_line * np.sin(theta), r_line * np.cos(theta)

    if t > np.pi / 4:
        ax.plot(x_line, y_line, color="C4", linewidth=0.5)
        if highest_apex_t is None:
            highest_apex_t = t
    else:
        ax.plot(x_line, y_line, color="C3", linewidth=0.5)


ax.set_xlim(0, rh)
ax.set_ylim(0, rh)
ax.set_aspect("equal")

# Remove the ticks.
ax.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)

# Set the spine colors to black.
ax.spines["top"].set_color("black")
ax.spines["bottom"].set_color("black")
ax.spines["left"].set_color("black")
ax.spines["right"].set_color("black")

# Increase width of frame.
ax.spines["top"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)


# Code to add the concentric magnetosphere boundary
if highest_apex_t is not None:
    req_highest_apex = 1 / np.sin(highest_apex_t) ** 2
    magnetosphere_radius = req_highest_apex * RE + 110

    # Define an outer radius large enough to cover the corner of the plot
    outer_radius = rh * np.sqrt(2)

    # Add light red background outside the magnetosphere boundary.
    outer_region = Wedge(
        center=(0, 0),
        r=outer_radius,
        theta1=0,
        theta2=90,
        width=outer_radius - magnetosphere_radius,
        color="red",
        alpha=0.2,
        zorder=0.5,
    )
    ax.add_patch(outer_region)

    # Magnetosphere boundary
    magnetosphere_circle = Circle(
        (0, 0),
        magnetosphere_radius,
        color="black",
        fill=False,
        linewidth=2,
        linestyle="-",
        zorder=2,
    )
    ax.add_patch(magnetosphere_circle)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
label_bbox = dict(facecolor="white", edgecolor="none", pad=0.15, alpha=0.65)

def add_rotated_label(xp, yp, text, size=textsize):
    ax.text(
        xp,
        yp,
        text,
        rotation=-45,
        rotation_mode="anchor",
        ha="center",
        va="center",
        size=size,
        bbox=label_bbox,
        zorder=5,
    )

def xy_from_rtheta(radius, angle):
    """Coordinates using x = r sin(theta), y = r cos(theta)."""
    return radius * np.sin(angle), radius * np.cos(angle)


# ---------------------------------------------------------------------
# Region labels
# ---------------------------------------------------------------------
add_rotated_label(0.38 * RE, 0.38 * RE, "Neutral atmosphere")
add_rotated_label(1.05 * RE, 1.05 * RE, "Gap region")
add_rotated_label(1.65 * RE, 1.65 * RE, "Magnetosphere")

# Ionosphere label
add_rotated_label(
    RI / np.sqrt(2),
    RI / np.sqrt(2),
    "Ionosphere",
)


# ---------------------------------------------------------------------
# B+, B-, square-brace connector, and ΔB -> J_S -> E chain
# ---------------------------------------------------------------------

# Rotate the whole construction a bit further counterclockwise.
# Smaller angle = more counterclockwise in the convention
# x = r sin(theta), y = r cos(theta).
B_LABEL_ANGLE = np.deg2rad(7.5)

# Equal radial distance from the ionosphere line.
B_OFFSET = 0.055 * RE

# Put B+ and B- at the same visual x-position, while keeping both
# exactly B_OFFSET away from the ionosphere.
b_label_x = RI * np.sin(B_LABEL_ANGLE)

theta_plus = np.arcsin(b_label_x / (RI + B_OFFSET))
theta_minus = np.arcsin(b_label_x / (RI - B_OFFSET))

b_plus_xy = np.array(xy_from_rtheta(RI + B_OFFSET, theta_plus))
b_minus_xy = np.array(xy_from_rtheta(RI - B_OFFSET, theta_minus))

# B labels
ax.text(
    *b_plus_xy,
    r"$\mathbf{B}^{+}$",
    ha="center",
    va="center",
    size=textsize,
    zorder=7,
)

ax.text(
    *b_minus_xy,
    r"$\mathbf{B}^{-}$",
    ha="center",
    va="center",
    size=textsize,
    zorder=7,
)

# Square-brace geometry:
#
# B+ -|
#     |- ΔB -> J_S -> E
# B- -|

b_plus_y = b_plus_xy[1]
b_minus_y = b_minus_xy[1]
brace_mid_y = 0.5 * (b_plus_y + b_minus_y)

# Bring the brace / ΔB chain a bit closer to the ionosphere surface.
line_start_x = b_label_x + 0.035 * RE
brace_x = b_label_x + 0.11 * RE
chain_start_x = brace_x + 0.055 * RE

brace_lw = 1.4

# Top and bottom horizontal parts: B+ -| and B- -|
ax.plot(
    [line_start_x, brace_x],
    [b_plus_y, b_plus_y],
    color="black",
    lw=brace_lw,
    zorder=6,
)

ax.plot(
    [line_start_x, brace_x],
    [b_minus_y, b_minus_y],
    color="black",
    lw=brace_lw,
    zorder=6,
)

# Vertical square-brace part
ax.plot(
    [brace_x, brace_x],
    [b_minus_y, b_plus_y],
    color="black",
    lw=brace_lw,
    zorder=6,
)

# Middle connector: |-
ax.plot(
    [brace_x, chain_start_x - 0.02 * RE],
    [brace_mid_y, brace_mid_y],
    color="black",
    lw=brace_lw,
    zorder=6,
)

# All math, including arrows, in LaTeX/mathtext
ax.text(
    chain_start_x,
    brace_mid_y,
    r"$\Delta\mathbf{B}\;\rightarrow\;\mathbf{J}_{\mathrm{S}}\;\rightarrow\;\mathbf{E}_{\mathrm{S}}$",
    ha="left",
    va="center",
    size=textsize,
    bbox=dict(facecolor="white", edgecolor="none", pad=0.08, alpha=0.8),
    zorder=7,
)


# ---------------------------------------------------------------------
# Neutral-wind contribution on the opposite (clockwise) side
# ---------------------------------------------------------------------

# Mirror the B-label angle around the ionosphere label angle (~45 deg),
# then nudge the text a bit to the right.
U_CHAIN_ANGLE = np.deg2rad(75)
U_CHAIN_RADIUS = RI# - 0.05 * RE

u_chain_x, u_chain_y = xy_from_rtheta(U_CHAIN_RADIUS, U_CHAIN_ANGLE)

# Small manual shift to the right.
#u_chain_x += 0.06 * RE

ax.text(
    u_chain_x,
    u_chain_y,
    r"$\mathbf{u}\;\rightarrow\;\mathbf{E}_{\mathrm{S}}$",
    ha="center",
    va="center",
    size=textsize,
    bbox=dict(facecolor="white", edgecolor="none", pad=0.08, alpha=0.8),
    zorder=7,
)


plt.tight_layout()
plt.show()
plt.close()