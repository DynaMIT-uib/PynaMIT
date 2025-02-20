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

PLOT_WIND = False  # True to make a plot of the wind field
SIMULATE = True

reload(pynamit)
RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40

dataset_filename_prefix = "ihc_test"

# MODEL PARAMETERS
Nmax, Mmax, Ncs = 25, 15, 50
print(
    "we need a check that the poloidal field calculation is high enough "
    "resoultion compared to SH "
)


rk = (
    RI
    / np.cos(np.deg2rad(np.linspace(0, 70, int(360 / (Nmax + 0.5)) + 1))) ** 2
)
# rk = np.hstack(
#     rk,
#     np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11)[5:] / RE
# )


# PARAMETERS FOR EMPIRICAL MODELS:
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date)  # noon longitude


## PLOT PARAMETERS
fig_directory = "figs/"
Blevels = np.linspace(-5, 5, 22) * 1e-9  # color levels for Br
levels = np.linspace(-0.9, 0.9, 22)  # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100)  # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:5]

## SET UP SIMULATION OBJECT
dynamics = pynamit.Dynamics(
    dataset_filename_prefix=dataset_filename_prefix,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    mainfield_kind="dipole",
    FAC_integration_steps=rk,
    ignore_PFAC=False,
    connect_hemispheres=True,
    latitude_boundary=latitude_boundary,
)

## CONDUCTANCE INPUT
conductance_lat = dynamics.state_grid.lat
conductance_lon = dynamics.state_grid.lon
hall, pedersen = conductance.hardy_EUV(
    conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=True
)
dynamics.set_conductance(
    hall, pedersen, lat=conductance_lat, lon=conductance_lon
)

## jr INPUT
jr_lat = dynamics.state_grid.lat
jr_lon = dynamics.state_grid.lon
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
jr = a.get_upward_current(mlat=jr_lat, mlt=d.mlon2mlt(jr_lon, date)) * 1e-6
jr[np.abs(jr_lat) < 50] = 0  # filter low latitude jr
dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon)

## WIND INPUT
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
dynamics.set_u(
    u_theta=u_theta,
    u_phi=u_phi,
    lat=u_lat,
    lon=u_lon,
    weights=np.tile(np.sin(np.deg2rad(90 - u_lat.flatten())), (2, 1)),
)

dynamics.update_conductance()
dynamics.update_u()
dynamics.update_jr()
dynamics.state.update_m_imp()
dynamics.state.update_E()

## SET UP PLOTTING GRID AND EVALUATORS
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat=lat, lon=lon)
plt_state_evaluator = pynamit.BasisEvaluator(dynamics.state_basis, plt_grid)

G_Br = plt_state_evaluator.scaled_G(dynamics.state_basis.n / RI)
Br = G_Br.dot(dynamics.state.m_imp_to_B_pol.dot(dynamics.state.m_imp.coeffs))


if PLOT_WIND:
    u_basis_evaluator = pynamit.BasisEvaluator(
        dynamics.state_basis, pynamit.Grid(lat=u_lat, lon=u_lon)
    )

    u_theta_sh = pynamit.FieldExpansion(
        dynamics.state_basis,
        basis_evaluator=u_basis_evaluator,
        grid_values=u_theta,
        field_type="scalar",
    )
    u_phi_sh = pynamit.FieldExpansion(
        dynamics.state_basis,
        basis_evaluator=u_basis_evaluator,
        grid_values=u_phi,
        field_type="scalar",
    )

    u_theta_int = u_theta_sh.to_grid(dynamics.state_basis_evaluator)
    u_phi_int = u_phi_sh.to_grid(dynamics.state_basis_evaluator)

    fig, ax = plt.subplots(
        figsize=(10, 7),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=lon0)},
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
        dynamics.state_grid.lon,
        dynamics.state_grid.lat,
        u_phi_int,
        -u_theta_int,
        color="red",
        scale=Q.scale,
        transform=ccrs.PlateCarree(),
    )


if SIMULATE:
    dt = 5e-4
    totalsteps = 200001
    ## PLOT PARAMETERS
    plotsteps = 500
    fig_directory = "figs/"
    Blevels = np.linspace(-50, 50, 22) * 1e-9  # color levels for Br
    levels = np.linspace(-0.9, 0.9, 22)  # color levels for FAC muA/m^2
    c_levels = np.linspace(0, 20, 100)  # color levels for conductance
    Wlevels = np.r_[-512.5:512.5:5]
    Philevels = np.r_[-212.5:212.5:2.5]

    ## RUN SIMULATION
    coeffs = []
    count = 0
    filecount = 1
    time = 0
    while True:

        dynamics.state.evolve_Br(dt)
        time = time + dt
        coeffs.append(dynamics.state.m_ind.coeffs)
        count += 1
        # print(
        #     count,
        #     time,
        #     (
        #         dynamics.state.m_ind.coeffs
        #         * dynamics.state.m_ind_to_Br)[:3]
        #     ),
        # )

        if count % plotsteps == 0:
            print(
                count,
                time,
                (dynamics.state.m_ind.coeffs * dynamics.state.m_ind_to_Br)[:3],
            )
            fn = os.path.join(
                fig_directory, "new_" + str(filecount).zfill(3) + ".png"
            )
            filecount += 1
            title = "t = {:.3} s".format(time)
            Br = dynamics.state.get_Br(plt_state_evaluator)
            fig, paxn, paxs, axg = pynamit.globalplot(
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

            W = dynamics.state.get_W(plt_state_evaluator) * 1e-3

            dynamics.state.update_E()
            Phi = dynamics.state.get_Phi(plt_state_evaluator) * 1e-3

            # paxn.contour(
            #     dynamics.state_grid.lat.flatten()[nnn],
            #     (dynamics.state_grid.lon.flatten() - lon0)[nnn] / 15,
            #     W[nnn],
            #     colors="black",
            #     levels=Wlevels,
            #     linewidths=0.5,
            # )
            # paxs.contour(
            #     dynamics.state_grid.lat.flatten()[sss],
            #     (dynamics.state_grid.lon.flatten() - lon0)[sss] / 15,
            #     W[sss],
            #     colors="black",
            #     levels=Wlevels,
            #     linewidths=0.5,
            # )
            # paxn.contour(
            #     plt_grid.lat.flatten()[nnn],
            #     (plt_grid.lon.flatten() - lon0)[nnn] / 15,
            #     Phi[nnn],
            #     colors="black",
            #     levels=Philevels,
            #     linewidths=0.5,
            # )
            # paxs.contour(
            #     plt_grid.lat.flatten()[sss],
            #     (plt_grid.lon.flatten() - lon0)[sss] / 15,
            #     Phi[sss],
            #     colors="black",
            #     levels=Philevels,
            #     linewidths=0.5,
            # )
            plt.savefig(fn)
            plt.close()

        if count > totalsteps:
            break

else:
    fig, paxn, paxs, axg = pynamit.globalplot(
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
