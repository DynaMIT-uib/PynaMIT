"""Script to check if the PFAC calculation gives a reasonable result."""

from importlib import reload
import pynamit
import dipole
import numpy as np
import datetime
import secsy
import pyamps
import matplotlib.pyplot as plt
from lompe import conductance
import os
from pynamit.visualization.results import plot_global_polar_map

COMPARE_TO_SECS = True
SIMULATE_DYNAMIC_RESPONSE = False

reload(pynamit)
RE = 6371.2e3
RI = RE + 110e3

run_directory = "PFAC_test"

Nmax, Mmax, Ncs = 25, 20, 30  # Model resolution

# Define plotting parameters.
fig_directory = "figs/"
Blevels = np.linspace(-5, 5, 22) * 1e-9  # color levels for Br
levels = np.linspace(-0.9, 0.9, 22)  # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100)  # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:5]

# Set up simulation object.
simulation = pynamit.Simulation(
    run_directory=run_directory,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    main_field_kind="dipole",
    fac_integration_radii=np.logspace(np.log10(RI), np.log10(7 * RE), 11),
    enable_pfac_coupling=True,
)

# Get and set conductance input.
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date)  # noon longitude

conductance_lat = simulation.geometry.model_grid.lat
conductance_lon = simulation.geometry.model_grid.lon
hall, pedersen = conductance.hardy_EUV(
    conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=True
)
simulation.set_conductance(hall, pedersen, lat=conductance_lat, lon=conductance_lon)

# Get and set jr input.
jr_lat = simulation.geometry.model_grid.lat
jr_lon = simulation.geometry.model_grid.lon
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
jr = a.get_upward_current(mlat=jr_lat, mlt=d.mlon2mlt(jr_lon, date)) * 1e-6
jr[np.abs(jr_lat) < 50] = 0  # filter low latitude jr
simulation.set_jr(jr, lat=jr_lat, lon=jr_lon)

simulation.update_conductance()
simulation.update_jr()
simulation.response.update_m_imp()
simulation.response.update_E()

# Set up plotting grid and evaluators.
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat=lat, lon=lon)
plt_state_evaluator = pynamit.SphericalTransform(simulation.geometry.horizontal_basis, plt_grid)

G_Br = plt_state_evaluator.contract_scalar_coeffs_to_grid(simulation.response.m_ind_to_Br)
Br = G_Br.dot(simulation.geometry.pfac_coupling_matrix.dot(simulation.response.m_imp.array))


if SIMULATE_DYNAMIC_RESPONSE:
    fig, paxn, paxs, axg = plot_global_polar_map(
        plt_grid.lon,
        plt_grid.lat,
        Br.reshape(plt_grid.lat.shape),
        returnplot=True,
        levels=Blevels,
        cmap="bwr",
        noon_longitude=lon0,
        extend="both",
    )

    plt.savefig("figs/PFAC_steady_state.png")
    plt.close()

    # Manipulate GTB to remove the r x grad(T) part.
    GrxgradT = -simulation.geometry.horizontal_transform.G_rxgrad * RI
    simulation.response.GTB = simulation.response.GTB - GrxgradT  # Subtract GrxgradT off

    # Run the simulation.
    plotsteps = 400
    fig_directory = "figs/"
    dt = 1e-3
    totalsteps = 2001
    coeffs = []
    count = 0
    filecount = 1
    time = 0
    while True:
        simulation.response.evolve_Br(dt)
        time = time + dt
        coeffs.append(simulation.response.m_ind.array)
        count += 1
        # print(
        #     count,
        #     time,
        #     (simulation.response.m_ind.array
        #      * simulation.response.m_ind_to_Br)[:3],
        # )

        if count % plotsteps == 0:
            print(
                count,
                time,
                (simulation.response.m_ind.array * simulation.response.m_ind_to_Br)[:3],
            )
            fn = os.path.join(fig_directory, "PFAC_" + str(filecount).zfill(3) + ".png")
            filecount += 1
            title = "t = {:.3} s".format(time)
            Br = simulation.response.get_Br(plt_state_evaluator)
            fig, paxn, paxs, axg = plot_global_polar_map(
                plt_grid.lon,
                plt_grid.lat,
                Br.reshape(plt_grid.lat.shape),
                title=title,
                returnplot=True,
                levels=Blevels,
                cmap="bwr",
                noon_longitude=lon0,
                extend="both",
            )

            W = simulation.response.get_W(plt_state_evaluator) * 1e-3

            simulation.response.update_E()
            Phi = simulation.response.get_Phi(plt_state_evaluator) * 1e-3

            plt.savefig(fn)
            plt.close()


if COMPARE_TO_SECS:
    print("Building SECS matrices. This takes some time (and memory) because of global grids...")
    secsI = (
        -jr * simulation.run_data.schema.cs_basis.unit_area * RI**2
    )  # SECS amplitudes are downward current density times area
    lat, lon = plt_grid.lat.flatten(), plt_grid.lon.flatten()
    r = np.full(lat.size, RI - 1)
    lat_secs, lon_secs = simulation.geometry.model_grid.lat, simulation.geometry.model_grid.lon
    field_evaluation = pynamit.MagneticFieldEvaluation(
        simulation.geometry.main_field, pynamit.Grid(lat=lat_secs, lon=lon_secs), RI
    )
    Be, Bn, Br = (
        field_evaluation.unit_bphi,
        -field_evaluation.unit_btheta,
        field_evaluation.unit_br,
    )
    Ge, Gn, Gu = secsy.get_CF_SECS_B_G_matrices_for_inclined_field(
        lat, lon, r, lat_secs, lon_secs, Be, Bn, Br, RI=RI
    )

    Br_SECS = Gu.dot(secsI)

    fig, paxn, paxs, axg = plot_global_polar_map(
        plt_grid.lon,
        plt_grid.lat,
        Br_SECS.reshape(plt_grid.lat.shape),
        returnplot=True,
        levels=Blevels,
        cmap="bwr",
        noon_longitude=lon0,
        extend="both",
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(Br_SECS, Br)
    brm = np.max(np.abs(Br))
    ax.plot([-brm, brm], [-brm, brm], "r-")
    ax.set_aspect("equal")
    ax.set_xlim(-brm, brm)
    ax.set_xlabel("straight tilted SECS")
    ax.set_ylabel("Spherical harmonics")

    plt.show()
    plt.close()
