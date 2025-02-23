"""Geometry illustration."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

RADIAL = False  # True for radial field lines, False for dipole

RE = 6371.2

fig, ax = plt.subplots(figsize=(8, 8))

rh = 2.0 * RE
RI = RE + 110
# d = 10 # Step used to illustrate radii just above and below ionosphere
textsize = 14

x = y = np.linspace(0, rh, 2000)
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx**2 + yy**2)
r = r.flatten()

cond = pd.read_csv("conductivity.csv")
cond.index = cond[cond.columns[0]] + RE
cond = cond[["hall", "pedersen"]]

# cond.index = cond.index + RE - 15
# rs = np.linspace(RI, rh, 400)
# cond_ = pd.DataFrame(index = rs, columns = cond.columns)
# cond_new = (
#    pd.concat(
#        [cond, cond_]
#    ).sort_index().interpolate(limit=10).fillna(0)
# )

ss = np.interp(r, cond.index, cond["pedersen"]).reshape(xx.shape)

ccc = ax.contourf(
    xx,
    yy,
    ss.reshape(xx.shape),
    cmap=plt.cm.Blues,
    levels=np.linspace(0, 1.5e-4, 100),
    zorder=0,
)

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

# rmax = rh / RI * 20
for t in th:
    req = 1 / np.sin(t) ** 2
    th_max = np.pi / 2
    theta = np.linspace(t, th_max, 100)

    r = req * np.sin(theta) ** 2 * RE + 110

    x, y = r * np.sin(theta), r * np.cos(theta)

    if t > np.pi / 4:
        ax.plot(x, y, color="C4", linewidth=0.5)
    else:
        ax.plot(x, y, color="C3", linewidth=0.5)


ax.set_xlim(0, rh)
ax.set_ylim(0, rh)

ax.set_aspect("equal")

# Remove the ticks.
ax.tick_params(
    axis="both",  # Apply to both x and y axes
    which="both",  # Apply to both major and minor ticks
    bottom=False,  # Turn off bottom ticks
    top=False,  # Turn off top ticks
    left=False,  # Turn off left ticks
    right=False,  # Turn off right ticks
    labelbottom=False,  # Turn off bottom tick labels
    labelleft=False,  # Turn off left tick labels
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

# plt.savefig('equations_fig00.png', dpi = 250)


ax.text(
    1.0 * RE,
    1.0 * RE,
    "$\\mathbf{B}\\times\\nabla\\times\\mathbf{B} = 0$\n"
    "$\\mathbf{E} + \\mathbf{v}\\times \\mathbf{B} = 0$\n"
    "$\\mathbf{B} = -\\nabla V^i + \\mathbf{P} + \\mathbf{r}\\times\\nabla T$",
    ha="center",
    va="center",
    size=textsize,
    bbox=dict(facecolor="white", edgecolor="none", pad=0, alpha=0.5),
)
ax.text(
    0.5 * RE,
    0.5 * RE,
    "$\\nabla\\times\\mathbf{B} = 0$\n$\\mathbf{B} = -\\nabla V^e$",
    ha="center",
    va="center",
    size=textsize,
)
ax.text(
    RI / np.sqrt(2),
    RI / np.sqrt(2),
    "$r = R$",
    rotation=-45,
    ha="center",
    va="center",
    size=textsize,
    bbox=dict(facecolor="white", edgecolor="none", pad=0),
)
# ax.text(
#    0.5 * RE,
#    0.5 * RE,
#    (
#        "$r < R$\n$\\mathbf{B}\\cdot\\nabla\\times\\mathbf{B} = 0$\n"
#        "$\\mathbf{E} + \\mathbf{v}\\times \\mathbf{B} = 0$"
#    ),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )


# for cc in ccc.collections:
#     cc.remove()

# ax.plot(RI * x, RI * np.cos(a), linewidth=5, color="C0")
# ax.text(
#    (RI + d) * x[-5],
#    (RI) * np.cos(a[-5]),
#    r"$R  $     ",
#    rotation=-np.rad2deg(a[-5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    color="C0",
#    bbox=dict(facecolor="white", edgecolor="none", pad=1),
# )
# ax.text(
#    (RI + d) * x[-5],
#    (RI + d) * np.cos(a[-5]),
#    r"$R^+$   ",
#    rotation=-np.rad2deg(a[-5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    color="C0",
#    bbox=dict(facecolor="white", edgecolor="none", pad=1),
# )
# ax.text(
#    (RI + d) * x[-5],
#    (RI - d) * np.cos(a[-5]),
#    r"$R^-$   ",
#    rotation=-np.rad2deg(a[-5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    color="C0",
#    bbox=dict(facecolor="white", edgecolor="none", pad=1),
# )

# plt.savefig("equations_fig01.png", dpi=250)


# ax.text(
#    x[5] * (RE - 10),
#    (RE - 10) * np.cos(a[5]),
#    r"ground",
#    rotation=-np.rad2deg(a[5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# ax.text(
#    x[5] * (RE + 50),
#    (RE + 50) * np.cos(a[5]),
#    r"atmosphere:"
#    + "\n"
#    + r"$\mathbf{j} = 0$"
#    + "\n"
#    + r"$\mathbf{B} = -\nabla V_B$",
#    rotation=-np.rad2deg(a[5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# ax.text(
#    x[5] * (RE + 50),
#    (RE + 50) * np.cos(a[5]),
#    r"atmosphere, $\mathbf{j} = 0$",
#    rotation=-np.rad2deg(a[5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# ax.text(
#    x[5] * (RI + 50),
#    (RI + 50) * np.cos(a[5]),
#    r"magnetosphere:"
#    + "\n"
#    + r"$\mathbf{j}_{\mathrm{FAC}} = \nabla\times\mathbf{B}/\mu_0$",
#    rotation=-np.rad2deg(a[5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# ax.text(
#    x[5] * (RI + 50),
#    (RI + 50) * np.cos(a[5]),
#    r'"gap region" $\mathbf{j}\times\mathbf{B} = 0$',
#    rotation=-np.rad2deg(a[5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# ax.text(
#    x[5] * (RI + 1),
#    (RI + 1) * np.cos(a[5]),
#    r"ionosphere",
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )


# plt.savefig("equations_fig02.png", dpi=250)

# ax.scatter(
#    (RI + d) * x[::2],
#    (RI + d) * np.cos(a[::2]),
#    marker="+",
#    color="C0"
# )
# ax.scatter(
#    (RI - d) * x[::2],
#    (RI - d) * np.cos(a[::2]),
#    marker="_", color="C0"
# )


# ax.text(
#    x[15] * (RI + 1),
#    (RI + 1) * np.cos(a[15]),
#    r"$\partial B_r / \partial t = -\nabla \times \mathbf{E}$",
#    rotation=-np.rad2deg(a[15]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )


# ax.text(
#    x[5] * (RE - 10),
#    (RE - 10) * np.cos(a[5]),
#    r"ground",
#    rotation=-np.rad2deg(a[5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# ax.text(
#    x[5] * (RE + 50),
#    (RE + 50) * np.cos(a[5]),
#    r"atmosphere:"
#    + "\n"
#    + r"$\mathbf{j} = 0$"
#    + "\n"
#    + r"$\mathbf{B} = -\nabla V_B$",
#    rotation=-np.rad2deg(a[5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# ax.text(
#    x[5] * (RI + 50),
#    (RI + 50) * np.cos(a[5]),
#    r"magnetosphere:"
#    + "\n"
#    + r"$\mathbf{E} = -\mathbf{v} \times \mathbf{B}$"
#    + "\n"
#    + r"$\mathbf{j} = \nabla\times\mathbf{B}/\mu_0$",
#    rotation=-np.rad2deg(a[5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# plt.savefig("equations_fig03.png", dpi=250)

# ax.text(
#    x[5] * (RI + 1),
#    (RI + 1) * np.cos(a[5]),
#    r"ionosphere:",
#    rotation=-np.rad2deg(a[5]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# ax.text(
#    x[40] * (RI + 1),
#    (RI + 1) * np.cos(a[40]),
#    (
#        r"$\mathbf{E} = (\Sigma_P\mathbf{j}\times\hat{\mathbf{r}} + "
#        r"\Sigma_H\mathbf{j})/(\Sigma_P^2 +\Sigma_H^2)$"
#    ),
#    rotation=-np.rad2deg(a[40]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# ax.text(
#    x[-35] * (RI + 1),
#    (RI + 1) * np.cos(a[-35]),
#    (
#        r"$\mathbf{j}_S = \hat{\mathbf{r}} \times (\mathbf{B}^+ - "
#        r"\mathbf{B}^-)/\mu_0$"
#    ),
#    rotation=-np.rad2deg(a[-35]),
#    ha="left",
#    va="center",
#    size=textsize,
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
# )
# plt.savefig("equations_fig04.png", dpi=250)

# ax.text(
#    x[-35] * (RI),
#    (RI) * np.cos(a[-35]),
#    r"$\mathbf{j}_S = \hat{\mathbf{r}} \times V_J - \nabla_S T_J$",
#    size=textsize,
#    va="center",
#    bbox=dict(facecolor="white", edgecolor="none", pad=0),
#    rotation=-np.rad2deg(a[-35]),
# )
# ax.text(
#    x[-35] * (RI + 50),
#    (RI + 50) * np.cos(a[-35]),
#    r"$\mathbf{B} = -\nabla V_B + \hat{\mathbf{r}} \times \nabla T_B$"
#    + "\n"
#    + r"$\mathbf{v}_S = -\nabla W_v + "
#    + r"\hat{\mathbf{r}}\times\nabla_S\Phi_V$"
#    + "\n"
#    + r"$\mathbf{E}_S = -\nabla \Phi_E + "
#    + r"\hat{\mathbf{r}}\times\nabla_SW_E$"
#    + "\n"
#    + r"$\mathbf{j} = T_{J_r}(R/r)^2\hat{\mathbf{r}}$",
#    size=textsize,
#    va="center",
#    rotation=-np.rad2deg(a[-35]),
# )
# ax.text(
#    x[-35] * (RI + 50),
#    (RI + 20) * np.cos(a[-35]),
#    r"$\mathbf{B} = -\nabla V_B + \hat{\mathbf{r}} \times \nabla T_B$",
#    size=textsize,
#    va="center",
#    rotation=-np.rad2deg(a[-35]),
# )

# plt.savefig("./figures/geometry.png", dpi=250)
# plt.savefig("./figures/geometry.pdf")


plt.show()
plt.close()
