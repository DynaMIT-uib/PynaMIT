"""Make (among other things) figures for graphical abstract."""

import pynamit
import polplot
import numpy as np
import matplotlib.pyplot as plt

path = "../simulation/data/pynamit_paper_simulation"  # Where the save files are located
a = pynamit.PynamEye(path)


TIMESERIES = False
CONDUCTANCE_AND_WIND = True
FAC = True
TS_ILLUSTRATION = True
SS_ILLUSTRATION = True
LONG_TS = False
LONG_TS_POLAR = False
LONG_TS_POLAR_E = False
COLORBAR = True

if CONDUCTANCE_AND_WIND:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=a.get_global_projection())
    a.plot_conductance(ax, region="global", extend="both")
    a.plot_wind(ax, color="lightgrey")
    a.jazz_global_plot(ax)

    ax.set_title("Hall conductance (EUV + Hardy)\nand wind (HWM2014)", size=18)

    plt.tight_layout()
    plt.savefig("figures/conditions.png", dpi=200)
    plt.show()


if FAC:
    ts = [0, 481]
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    paxes = [polplot.Polarplot(ax) for ax in axes]

    for t, pax in zip(ts, paxes[:2]):
        a.set_time(t)
        a.plot_jr(pax, region="north")
        pax.writeLATlabels(backgroundcolor=(0, 0, 0, 0))
        pax.writeLTlabels()

    a.plot_jr(paxes[2], region="south")

    axes[0].set_title("FAC at\n$t<0$ s", size=18)
    axes[1].set_title("FAC at\n$t\\ge 0$ s, north", size=18)
    axes[2].set_title("FAC at\n$t\\ge 0$ s, south", size=18)

    plt.tight_layout()
    plt.savefig("figures/FAC_step.png", dpi=200)
    plt.show()


if TS_ILLUSTRATION:
    ts = [0, 3 + 480, 10 + 480, 20 + 480]
    fig, axes = plt.subplots(ncols=4, figsize=(15, 4))
    paxes = [polplot.Polarplot(ax) for ax in axes]

    for t, pax in zip(ts, paxes):
        a.set_time(t, steady_state=True if t == 0 else False)
        a.plot_Br(pax, region="north", levels=np.linspace(-300, 300, 22) * 1e-9)
        a.plot_electric_potential(pax, region="north")
        a.plot_electric_field_stream_function(pax, region="north")
        pax.ax.set_title(
            r"t" + ("=" if t > 0 else "$<$") + "{} s".format(t if t < 1 else t - 480),
            size=22,
            pad=-10,
        )

        pax.writeLATlabels(backgroundcolor=(0, 0, 0, 0))
        pax.writeLTlabels()

    plt.tight_layout()
    plt.savefig("figures/TS_illustration.png", dpi=200)
    plt.show()

if SS_ILLUSTRATION:
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    paxes = [polplot.Polarplot(ax) for ax in axes]

    for i, pax in enumerate(paxes):
        if i == 0:
            a.set_time(0, steady_state=True)
        else:
            a.set_time(481, steady_state=True)
        a.plot_Br(pax, region="north", levels=np.linspace(-300, 300, 22) * 1e-9)
        a.plot_electric_potential(pax, region="north")
        a.plot_electric_field_stream_function(pax, region="north")
        pax.ax.set_title(r"$t<0$ s" if i == 0 else r"$t\ge 0$ s", size=22, pad=-10)

        pax.writeLATlabels(backgroundcolor=(0, 0, 0, 0))
        pax.writeLTlabels()

    plt.tight_layout()
    plt.savefig("figures/steady_state_illustration.png", dpi=200)
    plt.show()

if LONG_TS:
    ts = [0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 120, 180, 240, 300, 600, 900, 1200]
    fig = plt.figure(figsize=(14, 14))

    for i, t in enumerate(ts):
        a.set_time(t)
        ax = fig.add_subplot(5, 4, i + 1, projection=a.get_global_projection())
        a.plot_Br(ax, region="global")
        a.plot_equivalent_current(ax, region="global")
        a.jazz_global_plot(ax, draw_labels=True if i == 0 else False)
        ax.set_title("t={} s".format(t))

    plt.tight_layout()
    plt.savefig("figures/long_ts.png", dpi=200)
    plt.show()


if LONG_TS_POLAR:
    ts = [0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 120, 180, 240, 300, 600, 900, 1200]
    fig = plt.figure(figsize=(14, 14))

    for i, t in enumerate(ts):
        a.set_time(t)
        ax = fig.add_subplot(5, 4, i + 1)
        ax = polplot.Polarplot(ax)
        a.plot_Br(ax, region=LONG_TS_POLAR)
        a.plot_equivalent_current(ax, region=LONG_TS_POLAR)
        ax.ax.set_title("t={} s".format(t))

    plt.tight_layout()
    plt.savefig("figures/long_ts" + LONG_TS_POLAR + ".png", dpi=200)
    plt.show()


if LONG_TS_POLAR_E:
    ts = [0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 120, 180, 240, 300, 600, 900, 1200]
    fig = plt.figure(figsize=(14, 14))

    for i, t in enumerate(ts):
        a.set_time(t)
        ax = fig.add_subplot(5, 4, i + 1)
        ax = polplot.Polarplot(ax)
        a.plot_electric_potential(ax, region=LONG_TS_POLAR_E)
        a.plot_electric_field_stream_function(ax, region=LONG_TS_POLAR_E)
        ax.ax.set_title("t={} s".format(t))

    plt.tight_layout()
    plt.savefig("figures/long_ts_electric_field_" + LONG_TS_POLAR_E + ".png", dpi=200)
    plt.show()


if TIMESERIES:
    ts = np.linspace(0, 100, 101)

    for t in ts:
        a.set_time(t)
        a.make_multipanel_output_figure(label="t = {:.1f} s".format(t))

        plt.savefig("tmp/fig{:.2f}.png".format(t))

if COLORBAR:
    fig, ax = plt.subplots(figsize=(3, 10))
    faclevels = a.jr_defaults["levels"] * 1e6
    Brlevels = levels = np.linspace(-300, 300, 22)

    y = np.vstack((faclevels, faclevels)).T
    x = np.vstack((np.zeros(faclevels.size), np.ones(faclevels.size))).T

    ax.contourf(x, y, y, cmap=a.jr_defaults["cmap"], levels=faclevels)
    ax2 = ax.twinx()
    ax2.set_ylim(Brlevels[0], Brlevels[-1])
    ax.set_xticks([])
    ax2.set_xticks([])
    ax.set_ylabel(r"Field-aligned current [$\mu$A/m$^2$]", size=22)
    ax2.set_ylabel("Radial magnetic field at ionosphere radius [nT]", size=22)
    ax.tick_params(axis="y", labelsize=16)
    ax2.tick_params(axis="y", labelsize=16)

    plt.subplots_adjust(left=0.43, right=0.57, bottom=0.02, top=0.98)
    plt.savefig("figures/colorbar.png", dpi=200)

    plt.show()
