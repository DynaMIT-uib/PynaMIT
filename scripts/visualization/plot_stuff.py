"""Plot stuff."""

import pynamit
import polplot
import numpy as np
import matplotlib.pyplot as plt

# ts = np.linspace(1,240, 40)
# path = (
#    "/Users/laundal/Dropbox/git/dynamit/PynaMIT/scripts/simulation/"
#    "wind_step"
# )
path = "/Users/laundal/Dropbox/git/dynamit/PynaMIT/scripts/simulation/long_step"
a = pynamit.PynamEye(path)


TIMESERIES = True
CONDUCTANCE_AND_WIND = False
FAC = False
TS_ILLUSTRATION = False
SS_ILLUSTRATION = False
LONG_TS = False
LONG_TS_POLAR = False
LONG_TS_POLAR_E = False

if CONDUCTANCE_AND_WIND:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=a.get_global_projection())
    a.plot_conductance(ax, region="global", extend="both")
    a.plot_wind(ax, color="lightgrey")
    a.jazz_global_plot(ax)

    ax.set_title("Hall conductance (EUV + Hardy)\nand wind (HWM2014)")

    plt.tight_layout()
    plt.savefig("figures/conditions.png", dpi=200)
    plt.show()


if FAC:
    ts = [0, 1]
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    paxes = [polplot.Polarplot(ax) for ax in axes]

    for t, pax in zip(ts, paxes[:2]):
        a.set_time(t)
        a.plot_jr(pax, region="north")
        pax.writeLATlabels(backgroundcolor=(0, 0, 0, 0))
        pax.writeLTlabels()

    a.plot_jr(paxes[2], region="south")

    axes[0].set_title("FAC at\n$t=0$ s", size=14)
    axes[1].set_title("FAC at\n$t>0$ s", size=14)
    axes[2].set_title("FAC at\n$t>0$ s, south", size=14)

    plt.tight_layout()
    plt.savefig("figures/FAC_step.png", dpi=200)
    plt.show()


if TS_ILLUSTRATION:
    ts = [0, 3, 21, 601]
    fig, axes = plt.subplots(ncols=4, figsize=(15, 4))
    paxes = [polplot.Polarplot(ax) for ax in axes]

    for t, pax in zip(ts, paxes):
        a.set_time(t)
        a.plot_Br(pax, region="north", levels=np.linspace(-300, 300, 22) * 1e-9)
        a.plot_electric_potential(pax, region="north")
        a.plot_electric_field_stream_function(pax, region="north")
        pax.ax.set_title("t={} s".format(t if t < 1 else t - 1), size=22, pad=-10)

        pax.writeLATlabels(backgroundcolor=(0, 0, 0, 0))
        pax.writeLTlabels()

    plt.tight_layout()
    plt.savefig("figures/TS_illustration.png", dpi=200)
    plt.show()

if SS_ILLUSTRATION:
    ts = [0, 600]
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    paxes = [polplot.Polarplot(ax) for ax in axes]

    for t, pax in zip(ts, paxes):
        a.set_time(t)
        a.plot_Br(pax, region="north", levels=np.linspace(-300, 300, 22) * 1e-9)
        a.plot_electric_potential(pax, region="north")
        a.plot_electric_field_stream_function(pax, region="north")
        pax.ax.set_title("t=0 s" if t == 0 else "$t>0$ s", size=22, pad=-10)

        pax.writeLATlabels(backgroundcolor=(0, 0, 0, 0))
        pax.writeLTlabels()

    plt.tight_layout()
    plt.savefig("figures/steady_state_illustration.png", dpi=200)
    plt.show()

if LONG_TS:
    ts = [
        0,
        1,
        2,
        3,
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        120,
        180,
        240,
        300,
        600,
        900,
        1200,
    ]
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
    ts = [
        0,
        1,
        2,
        3,
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        120,
        180,
        240,
        300,
        600,
        900,
        1200,
    ]
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
    ts = [
        0,
        1,
        2,
        3,
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        120,
        180,
        240,
        300,
        600,
        900,
        1200,
    ]
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
