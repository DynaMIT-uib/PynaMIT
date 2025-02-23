"""Visualize the simulation case."""

import numpy as np
import pynamit
import matplotlib.pyplot as plt
import datetime
from polplot import Polarplot
from string import ascii_lowercase as abc

path = "../simulation/data/brn"  # Where the save files are located
path = (
    "/Users/laundal/Dropbox/git/dynamit/PynaMIT/scripts/simulation/data/"
    "pynamit_paper_simulation"
)

print(datetime.datetime.now(), "making PynamEye object")
a = pynamit.PynamEye(path, steady_state=True)

phin = a.evaluator["north"].basis_to_grid(a.m_Phi)
phis = a.evaluator["south"].basis_to_grid(a.m_Phi)
print(
    datetime.datetime.now(),
    "CPCP in the North is {:.1f} kV".format((phin.max() - phin.min()) * 1e-3),
)
print(
    datetime.datetime.now(),
    "CPCP in the South is {:.1f} kV".format((phis.max() - phis.min()) * 1e-3),
)
conductance_levels = np.linspace(0, 20, 22)

fig = plt.figure(figsize=(12, 12))

gax1 = plt.subplot2grid(
    (4, 62), (0, 2 + 20 * 2), colspan=20, projection=a.get_global_projection()
)
gax2 = plt.subplot2grid(
    (4, 62), (1, 2 + 20 * 2), colspan=20, projection=a.get_global_projection()
)
gax3 = plt.subplot2grid(
    (4, 62), (2, 2 + 20 * 2), colspan=20, projection=a.get_global_projection()
)
gax4 = plt.subplot2grid(
    (4, 62), (3, 2 + 20 * 2), colspan=20, projection=a.get_global_projection()
)


paxn1 = Polarplot(
    plt.subplot2grid(
        (4, 62), (0, 2 + 20 * 0), colspan=20, projection=a.get_global_projection()
    )
)
paxs1 = Polarplot(
    plt.subplot2grid(
        (4, 62), (0, 2 + 20 * 1), colspan=20, projection=a.get_global_projection()
    )
)
paxn2 = Polarplot(
    plt.subplot2grid(
        (4, 62), (1, 2 + 20 * 0), colspan=20, projection=a.get_global_projection()
    )
)
paxs2 = Polarplot(
    plt.subplot2grid(
        (4, 62), (1, 2 + 20 * 1), colspan=20, projection=a.get_global_projection()
    )
)
paxn3 = Polarplot(
    plt.subplot2grid(
        (4, 62), (2, 2 + 20 * 0), colspan=20, projection=a.get_global_projection()
    )
)
paxs3 = Polarplot(
    plt.subplot2grid(
        (4, 62), (2, 2 + 20 * 1), colspan=20, projection=a.get_global_projection()
    )
)
paxn4 = Polarplot(
    plt.subplot2grid(
        (4, 62), (3, 2 + 20 * 0), colspan=20, projection=a.get_global_projection()
    )
)
paxs4 = Polarplot(
    plt.subplot2grid(
        (4, 62), (3, 2 + 20 * 1), colspan=20, projection=a.get_global_projection()
    )
)

cbar_axes = [plt.subplot2grid((4, 62), (i, 0)) for i in range(3)]


for ax in [gax1, gax2, gax3, gax4]:
    a.jazz_global_plot(ax)

for i, ax in enumerate(
    [
        paxn1.ax,
        paxs1.ax,
        gax1,
        paxn2.ax,
        paxs2.ax,
        gax2,
        paxn3.ax,
        paxs3.ax,
        gax3,
        paxn4.ax,
        paxs4.ax,
        gax4,
    ]
):
    ax.text(
        0.01,
        0.99,
        abc[i] + ")",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="none"),
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
    )

a.plot_wind(gax1, color="black")
a.plot_conductance(
    gax2, region="global", levels=conductance_levels, extend="both"
).set_edgecolor("face")
a.plot_Br(gax3, region="global", levels=a.Br_defaults["levels"] * 2).set_edgecolor(
    "face"
)

a.plot_equivalent_current(gax3, region="global")
a.plot_electric_field_stream_function(gax3, region="global")
a.plot_electric_potential(
    gax4, region="global", colors="black", levels=np.r_[-201.25:202:0.5] * 1e3
)


a.plot_jr(paxn1, region="north").set_edgecolor("face")
a.plot_conductance(
    paxn2, region="north", levels=conductance_levels, extend="both"
).set_edgecolor("face")
a.plot_Br(paxn3, region="north", levels=a.Br_defaults["levels"] * 2).set_edgecolor(
    "face"
)


a.plot_equivalent_current(paxn3, region="north")
a.plot_electric_potential(
    paxn4, region="north", colors="black", levels=np.r_[-201.5:202:3] * 1e3
)
a.plot_electric_field_stream_function(paxn4, region="north")


a.plot_jr(paxs1, region="south").set_edgecolor("face")
a.plot_conductance(paxs2, region="south", levels=conductance_levels).set_edgecolor(
    "face"
)
a.plot_Br(paxs3, region="south", levels=a.Br_defaults["levels"] * 2).set_edgecolor(
    "face"
)
a.plot_equivalent_current(paxs3, region="south")
a.plot_electric_potential(
    paxs4, region="south", colors="black", levels=np.r_[-201.5:202:3] * 1e3
)
a.plot_electric_field_stream_function(paxs4, region="south", extend="both")

levels = a.jr_defaults["levels"] * 1e6
xx, zz = (
    np.vstack((np.zeros(levels.size), np.ones(levels.size))).T,
    np.vstack((levels, levels)).T,
)
cbar_axes[0].contourf(xx, zz, zz, levels=levels, cmap=a.jr_defaults["cmap"])
cbar_axes[0].set_xticks([])
cbar_axes[0].set_ylabel(r"$\mu$A/m$^2$")

levels = conductance_levels
xx, zz = (
    np.vstack((np.zeros(levels.size), np.ones(levels.size))).T,
    np.vstack((levels, levels)).T,
)
cbar_axes[1].contourf(xx, zz, zz, levels=levels, cmap=a.conductance_defaults["cmap"])
cbar_axes[1].set_xticks([])
cbar_axes[1].set_ylabel("mho")

levels = a.Br_defaults["levels"] * 2 * 1e9
xx, zz = (
    np.vstack((np.zeros(levels.size), np.ones(levels.size))).T,
    np.vstack((levels, levels)).T,
)
cbar_axes[2].contourf(xx, zz, zz, levels=levels, cmap=a.Br_defaults["cmap"])
cbar_axes[2].set_xticks([])
cbar_axes[2].set_ylabel("nT")


paxn1.ax.set_title("Input $j_r$ North")
paxs1.ax.set_title("Input $j_r$ South")
paxn2.ax.set_title("Input Hall conductance North")
paxs2.ax.set_title("Input Hall conductance South")

paxn1.writeLATlabels(color="black", backgroundcolor=(0, 0, 0, 0))
paxn1.writeLTlabels()
paxs1.writeLATlabels(color="black", backgroundcolor=(0, 0, 0, 0), north=False)
paxs1.writeLTlabels()

gax1.set_title("Input horizontal wind")
gax2.set_title("Input Hall conductance")

paxn3.ax.set_title(r"Steady state $B_r(r=R)$ and $\Psi$")
paxs3.ax.set_title(r"Steady state $B_r(r=R)$ and $\Psi$")
gax3.set_title(r"Steady state $B_r(r=R)$ and $\Psi$")

paxn4.ax.set_title("Steady state electric potential")
paxs4.ax.set_title("Steady state electric potential")
gax4.set_title("Steady state electric potential")


plt.subplots_adjust(
    top=0.96, bottom=0.015, left=0.095, right=0.985, hspace=0.2, wspace=0.145
)

plt.savefig("simulation_case_illustration.png", dpi=250)
plt.savefig("simulation_case_illustration.pdf")

plt.show()
