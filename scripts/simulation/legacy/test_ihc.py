"""Script to check if the interhemispheric connection works."""

from importlib import reload
import pynamit
import dipole
import numpy as np
import datetime
import pyamps
import matplotlib.pyplot as plt
from lompe import conductance
import pyhwm2014  # https://github.com/rilma/pyHWM14
import cartopy.crs as ccrs
import os
from pynamit.visualization.results import plot_global_polar_map

PLOT_WIND = False  # True to make a plot of the wind field
SIMULATE = True

reload(pynamit)
RE = 6371.2e3
RI = RE + 110e3
interhemispheric_coupling_latitude = 40

run_directory = "ihc_test"

Nmax, Mmax, Ncs = 25, 15, 50  # Model resolution
print(
    "we need a check that the poloidal field calculation is high enough resoultion compared to SH "
)


rk = RI / np.cos(np.deg2rad(np.linspace(0, 70, int(360 / (Nmax + 0.5)) + 1))) ** 2
# rk = np.hstack(
#     rk,
#     np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11)[5:] / RE
# )


# Define parameters for empirical models.
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date)  # noon longitude


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
    fac_integration_radii=rk,
    enable_pfac_coupling=True,
    enable_interhemispheric_coupling=True,
    interhemispheric_coupling_latitude=interhemispheric_coupling_latitude,
)

# Get and set conductance input.
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

# Get and set wind input.
hwm14Obj = pyhwm2014.HWM142D(
    alt=110.0,
    ap=[35, 35],
    glatlim=[-89.0, 88.0],
    glatstp=3.0,
    glonlim=[-180.0, 180.0],
    glonstp=8.0,
    option=6,
    verbose=False,
    ut=date.hour,
    day=date.timetuple().tm_yday,
)

u_theta, u_phi = (-hwm14Obj.Vwind.flatten(), hwm14Obj.Uwind.flatten())
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing="ij")
simulation.set_neutral_wind(
    u_theta=u_theta,
    u_phi=u_phi,
    lat=u_lat,
    lon=u_lon,
    sqrt_weights=np.tile(np.sqrt(np.sin(np.deg2rad(90 - u_lat.flatten()))), (2, 1)),
)

simulation.update_conductance()
simulation.update_u()
simulation.update_jr()
simulation.response.update_m_imp()
simulation.response.update_E()

# Set up plotting grid and evaluators.
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat=lat, lon=lon)
state_field_space = pynamit.FieldSpace.from_representation(
    simulation.geometry.horizontal_basis, field_type="scalar"
)
plt_state_evaluator = pynamit.SphericalTransform(state_field_space.representation, plt_grid)

G_Br = plt_state_evaluator.contract_scalar_coeffs_to_grid(simulation.response.m_ind_to_Br)
Br = G_Br.dot(simulation.geometry.pfac_coupling_matrix.dot(simulation.response.m_imp.array))


if PLOT_WIND:
    u_spherical_transform = pynamit.SphericalTransform(
        state_field_space.representation, pynamit.Grid(lat=u_lat, lon=u_lon)
    )
    scalar_state_space = pynamit.FieldSpace(simulation.geometry.horizontal_basis, field_type="scalar")

    u_theta_sh = pynamit.FieldCoefficients(
        scalar_state_space, u_spherical_transform.analyze_scalar(u_theta)
    )
    u_phi_sh = pynamit.FieldCoefficients(
        scalar_state_space, u_spherical_transform.analyze_scalar(u_phi)
    )

    u_theta_int = simulation.geometry.horizontal_transform.synthesize_scalar(u_theta_sh)
    u_phi_int = simulation.geometry.horizontal_transform.synthesize_scalar(u_phi_sh)

    fig, ax = plt.subplots(
        figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=lon0)}
    )
    ax.coastlines()
    Q = ax.quiver(
        u_lon.flatten(),
        u_lat.flatten(),
        u_phi.flatten(),
        -u_theta.flatten(),
        color="blue",
        transform=ccrs.PlateCarree(),
    )
    ax.quiver(
        simulation.geometry.model_grid.lon,
        simulation.geometry.model_grid.lat,
        u_phi_int,
        -u_theta_int,
        color="red",
        scale=Q.scale,
        transform=ccrs.PlateCarree(),
    )


if SIMULATE:
    dt = 5e-4
    totalsteps = 200001
    # Define plotting parameters.
    plotsteps = 500
    fig_directory = "figs/"
    Blevels = np.linspace(-50, 50, 22) * 1e-9  # color levels for Br
    levels = np.linspace(-0.9, 0.9, 22)  # color levels for FAC muA/m^2
    c_levels = np.linspace(0, 20, 100)  # color levels for conductance
    Wlevels = np.r_[-512.5:512.5:5]
    Philevels = np.r_[-212.5:212.5:2.5]

    # Run the simulation.
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
        #     (
        #         simulation.response.m_ind.array
        #         * simulation.response.m_ind_to_Br)[:3]
        #     ),
        # )

        if count % plotsteps == 0:
            print(
                count,
                time,
                (simulation.response.m_ind.array * simulation.response.m_ind_to_Br)[:3],
            )
            fn = os.path.join(fig_directory, "new_" + str(filecount).zfill(3) + ".png")
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

        if count > totalsteps:
            break

else:
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

    plt.show()
    plt.close()
