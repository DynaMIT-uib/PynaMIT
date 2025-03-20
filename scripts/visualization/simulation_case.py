"""Visualize the simulation case."""

import numpy as np
import pynamit
import matplotlib.pyplot as plt
import datetime
from polplot import Polarplot
from string import ascii_lowercase as abc

path = "../simulation/data/pynamit_paper_simulation"  # Where the save files are located

print(datetime.datetime.now(), "making PynamEye object")
a = pynamit.PynamEye(path, steady_state=True)

for plot_num, simulation_time in enumerate([0, 480]):
    a.set_time(simulation_time, steady_state = True)

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

    gax_wind = plt.subplot2grid((4, 62), (0, 2  + 20 * 2), colspan=20, projection=a.get_global_projection())
    gax_hall = plt.subplot2grid((4, 62), (1, 8  + 20 * 0), colspan=20, projection=a.get_global_projection())
    gax_pede = plt.subplot2grid((4, 62), (1, 18 + 20 * 1), colspan=20, projection=a.get_global_projection())
    gax_Brad = plt.subplot2grid((4, 62), (2, 2  + 20 * 2), colspan=20, projection=a.get_global_projection())
    gax_Elec = plt.subplot2grid((4, 62), (3, 2  + 20 * 2), colspan=20, projection=a.get_global_projection())


    pax_fac_n = Polarplot(
        plt.subplot2grid((4, 62), (0, 2 + 20 * 0), colspan=20, projection=a.get_global_projection())
    )
    pax_fac_s = Polarplot(
        plt.subplot2grid((4, 62), (0, 2 + 20 * 1), colspan=20, projection=a.get_global_projection())
    )
    pax_Bra_n = Polarplot(
        plt.subplot2grid((4, 62), (2, 2 + 20 * 0), colspan=20, projection=a.get_global_projection())
    )
    pax_Bra_s = Polarplot(
        plt.subplot2grid((4, 62), (2, 2 + 20 * 1), colspan=20, projection=a.get_global_projection())
    )
    pax_Ele_n = Polarplot(
        plt.subplot2grid((4, 62), (3, 2 + 20 * 0), colspan=20, projection=a.get_global_projection())
    )
    pax_Ele_s = Polarplot(
        plt.subplot2grid((4, 62), (3, 2 + 20 * 1), colspan=20, projection=a.get_global_projection())
    )

    cbar_axes = [plt.subplot2grid((4, 62), (i, 0)) for i in range(3)]


    for ax in [gax_wind, gax_hall, gax_pede, gax_Brad, gax_Elec]:
        a.jazz_global_plot(ax)

    for i, ax in enumerate(
        [
            pax_fac_n.ax,
            pax_fac_s.ax,
            gax_wind,
            gax_hall,
            gax_pede,
            pax_Bra_n.ax,
            pax_Bra_s.ax,
            gax_Brad,
            pax_Ele_n.ax,
            pax_Ele_s.ax,
            gax_Elec,
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

    a.plot_wind(gax_wind, color="black")
    a.plot_conductance(gax_hall, region="global", hp = 'h', levels=conductance_levels, extend="both").set_edgecolor(
        "face"
    )
    a.plot_conductance(gax_pede, region="global", hp = 'p', levels=conductance_levels, extend="both").set_edgecolor(
        "face"
    )
    a.plot_Br(gax_Brad, region="global", levels=a.Br_defaults["levels"]).set_edgecolor("face")

    a.plot_equivalent_current(gax_Brad, region="global", levels = a.eqJ_defaults['levels'], linewidths = .7)
    a.plot_electric_field_stream_function(gax_Brad, region="global")
    a.plot_electric_potential(
        gax_Elec, region="global", colors="black", levels=np.r_[-201.25:202:3] * 1e3
    )


    a.plot_jr(pax_fac_n, region="north").set_edgecolor("face")
    a.plot_Br(pax_Bra_n, region="north", levels=a.Br_defaults["levels"]).set_edgecolor("face")


    a.plot_equivalent_current(pax_Bra_n, region="north", levels = a.eqJ_defaults['levels'], linewidths = .7)
    a.plot_electric_potential(pax_Ele_n, region="north", colors="black", levels=a.Phi_defaults['levels'])
    a.plot_electric_field_stream_function(pax_Ele_n, region="north")


    a.plot_jr(pax_fac_s, region="south").set_edgecolor("face")
    a.plot_Br(pax_Bra_s, region="south", levels=a.Br_defaults["levels"]).set_edgecolor("face")
    a.plot_equivalent_current(pax_Bra_s, region="south", levels = a.eqJ_defaults['levels'], linewidths = .7)
    a.plot_electric_potential(pax_Ele_s, region="south", colors="black", levels=a.Phi_defaults['levels'])
    a.plot_electric_field_stream_function(pax_Ele_s, region="south", extend="both")

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

    levels = a.Br_defaults["levels"] * 1e9
    xx, zz = (
        np.vstack((np.zeros(levels.size), np.ones(levels.size))).T,
        np.vstack((levels, levels)).T,
    )
    cbar_axes[2].contourf(xx, zz, zz, levels=levels, cmap=a.Br_defaults["cmap"])
    cbar_axes[2].set_xticks([])
    cbar_axes[2].set_ylabel("nT")


    pax_fac_n.ax.set_title("Input $j_r$ North")
    pax_fac_s.ax.set_title("Input $j_r$ South")

    pax_fac_n.writeLATlabels(color="black", backgroundcolor=(0, 0, 0, 0))
    pax_fac_n.writeLTlabels()
    pax_fac_s.writeLATlabels(color="black", backgroundcolor=(0, 0, 0, 0), north=False)
    pax_fac_s.writeLTlabels()

    gax_wind.set_title("Input horizontal wind")
    gax_hall.set_title("Input Hall conductance")
    gax_pede.set_title("Input Pedersen conductance")

    pax_Bra_n.ax.set_title(r"Steady state $B_r(r=R)$ and $\Psi$")
    pax_Bra_s.ax.set_title(r"Steady state $B_r(r=R)$ and $\Psi$")
    gax_Brad.set_title(r"Steady state $B_r(r=R)$ and $\Psi$")

    pax_Ele_n.ax.set_title("Steady state electric potential")
    pax_Ele_s.ax.set_title("Steady state electric potential")
    gax_Elec.set_title("Steady state electric potential")


    plt.subplots_adjust(top=0.96, bottom=0.015, left=0.095, right=0.985, hspace=0.2, wspace=0.145)

    fn = "simulation_case_illustration" + str(plot_num+1)
    plt.savefig(fn + ".png", dpi=250)
    plt.savefig(fn + ".pdf")
    print("Saved {}".format(fn + ".x"))

plt.show()
