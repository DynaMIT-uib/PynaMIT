"""SimulationGeometry illustration."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Wedge  # Import Wedge

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
    # The contourf plot has zorder=0
    filled_contour = ax.contourf(
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

    r = req * np.sin(theta) ** 2 * RE + 110

    x, y = r * np.sin(theta), r * np.cos(theta)

    if t > np.pi / 4:
        ax.plot(x, y, color="C4", linewidth=0.5)
        # Capture the t of the FIRST purple line encountered
        if highest_apex_t is None:
            highest_apex_t = t
    else:
        ax.plot(x, y, color="C3", linewidth=0.5)


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


plt.tight_layout()


ax.text(
    1.65 * RE,
    1.65 * RE,
    "$\\Delta\\mathbf{B} =\\mathbf{B}_\\mathrm{mag}$",
    ha="center",
    va="center",
    size=textsize,
    bbox=dict(facecolor="white", edgecolor="none", pad=0, alpha=0.5),
)

ax.text(
    1.05 * RE,
    1.05 * RE,
    "$\\Delta\\mathbf{B} = \\mathbf{B}_\\mathrm{ind} + \\mathbf{B}_\\mathrm{imp}$",
    ha="center",
    va="center",
    size=textsize,
    bbox=dict(facecolor="white", edgecolor="none", pad=0, alpha=0.5),
)
ax.text(
    0.375 * RE,
    0.375 * RE,
    "$\\Delta\\mathbf{B} = \\mathbf{B}_\\mathrm{ind}$",
    ha="center",
    va="center",
    size=textsize,
)

# Changed "r = R" to "Ionosphere"
ax.text(
    RI / np.sqrt(2),
    RI / np.sqrt(2),
    "Ionosphere",
    rotation=-45,
    ha="center",
    va="center",
    size=textsize,
    bbox=dict(facecolor="white", edgecolor="none", pad=0),
)

# Code to add the concentric magnetosphere boundary and label

# Calculate the radius of the magnetosphere boundary
if highest_apex_t is not None:
    req_highest_apex = 1 / np.sin(highest_apex_t) ** 2
    magnetosphere_radius = req_highest_apex * RE + 110

    # Define an outer radius large covering the corner of the plot
    outer_radius = rh * np.sqrt(2)

    # Add light red background outside the inner magnetosphere boundary.
    outer_region = Wedge(
        center=(0, 0),
        r=outer_radius,  # <-- Use the larger radius to cover the corner
        theta1=0,
        theta2=90,
        width=outer_radius - magnetosphere_radius,  # <-- Adjust width accordingly
        color="red",
        alpha=0.2,
        zorder=0.5,
    )
    ax.add_patch(outer_region)

    # Create a solid Circle patch for the magnetosphere boundary
    magnetosphere_circle = Circle(
        (0, 0), magnetosphere_radius, color="black", fill=False, linewidth=2, linestyle="-"
    )
    ax.add_patch(magnetosphere_circle)

    # Add the "Inner magnetosphere" label at a 45-degree angle on
    # the new circle
    ax.text(
        magnetosphere_radius / np.sqrt(2),
        magnetosphere_radius / np.sqrt(2),
        "Inner magnetosphere",
        rotation=-45,
        ha="center",
        va="center",
        size=textsize,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.1),
    )

plt.show()
plt.close()
