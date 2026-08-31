"""Script for testing various things with spherical harmonics."""

import datetime

import kompe
import numpy as np
import ppigrf
import pyhwm2014  # https://github.com/rilma/pyHWM14

from pynamit.external_inputs import get_conductance_inputs
from pynamit.external_inputs.coordinates import ExternalInputCoordinates

Nmax, Mmax = 20, 20
Ncs = 30
RI = (6371.2 + 110) * 1e3

Kp = 3
ubasis = kompe.SHBasis(Nmax, Mmax)


# Fit wind pattern using a spherical harmonic Helmholtz representation.
date = datetime.datetime(2000, 5, 12, 21, 45)

### CONDUCTANCE EXPERIMENT
cbasis = kompe.SHBasis(Nmax, Mmax, min_degree=0)

cs_basis = kompe.GlobalCSBasis(Ncs)
conductance_grid = cs_basis.native_grid
conductance_lat = conductance_grid.lat
conductance_lon = conductance_grid.lon
coordinates = ExternalInputCoordinates.from_geocentric_geo(
    conductance_lat, conductance_lon, grid_id="sh-experiment-grid"
)
pedersen, hall, _, _ = get_conductance_inputs(date, coordinates=coordinates, kp=Kp, starlight=1.0)

etaH, etaP = hall / (hall**2 + pedersen**2), pedersen / (hall**2 + pedersen**2)

G = cbasis.scalar_evaluation_array(conductance_grid)
d = etaH

m_plain = np.linalg.lstsq(G, d, rcond=0)[0]

GTG = G.T.dot(G)
R = np.eye(cbasis.index_length) * cbasis.n * (cbasis.n + 1)
alpha = 1
GTd = G.T.dot(d)
m_regularized = np.linalg.lstsq(GTG + alpha * R, GTd, rcond=0)[0]

# alpha = np.logspace(-5, 5, 11)
# ms = []
# for a in alpha:
#    ms.append(np.linalg.lstsq(GTG + a * R, GTd, rcond = 0)[0])
#
# misfits = [np.sqrt(np.sum((G.dot(m) - d)**2)) for m in ms]
# norms = [np.linalg.norm(m) for m in ms]


if False:
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
    u_phi = hwm14Obj.Uwind
    u_theta = -hwm14Obj.Vwind
    u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing="ij")

    ugrid = kompe.SphericalGrid(lat=u_lat.flatten(), lon=u_lon.flatten())

    Gphi, Gtheta = (
        ubasis.scalar_evaluation_array(ugrid, derivative="phi"),
        ubasis.scalar_evaluation_array(ugrid, derivative="theta"),
    )
    divergence_free_potential_matrix = np.vstack((-Gphi, Gtheta))
    curl_free_potential_matrix = np.vstack((Gtheta, Gphi))

    d = np.hstack((u_theta.flatten(), u_phi.flatten()))
    potential_coefficients = np.linalg.lstsq(
        np.hstack((curl_free_potential_matrix, divergence_free_potential_matrix)), d, rcond=0
    )[0]
    curl_free_coefficients, divergence_free_coefficients = np.split(potential_coefficients, 2)

    misfit = (
        divergence_free_potential_matrix.dot(divergence_free_coefficients)
        + curl_free_potential_matrix.dot(curl_free_coefficients)
        - d
    )
    print(f"rms misfit for fitted wind field is {np.sqrt(np.mean(misfit**2)):.5f} m/s")

    # Get IGRF gauss coefficients.
    igrf_date = datetime.datetime(date.year, 1, 1)
    g, h = ppigrf.ppigrf.read_shc()
    _n, _m = np.array([k for k in g.columns]).T  # n and m
    g, h = (g.loc[igrf_date, :].values, h.loc[igrf_date, :].values)  # Gauss coefficients

    igrf_basis = kompe.SHBasis(int(_n.max()), int(_m.max()))
    igrf_keys = igrf_basis.cnm

    # Calculate u x B numerically on grid (we evaluate on the ground).
    ph = np.deg2rad(u_lon).reshape((-1, 1))
    P = igrf_basis.legendre(np.deg2rad(90 - u_lat).reshape(-1))[:, igrf_basis.cnm_filter]
    G_Br = np.hstack(
        (
            (igrf_keys.n + 1) * P * np.cos(igrf_keys.m * ph),
            (igrf_keys.n + 1) * P * np.sin(igrf_keys.m * ph),
        )
    )
    Br = G_Br.dot(np.hstack((g, h))) * 1e-9

    uxB_theta = Br * u_phi.flatten()
    uxB_phi = -Br * u_theta.flatten()


# Now u x B is calculated from SH coefficients without grid evaluation.
