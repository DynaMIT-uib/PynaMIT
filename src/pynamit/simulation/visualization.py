"""Visualization utilities for simulation results.

This module provides plotting functions for visualizing ionospheric simulation
results, including global maps, debug plots, and time series visualizations.

Functions
---------
cs_interpolate
    Interpolate data from cubed sphere to regular grid
globalplot
    Create global map visualization with optional parameters
debugplot
    Generate diagnostic plots of simulation state
compare_AMPS_jr_and_CF_currents
    Compare AMPERE and curl-free current distributions
plot_AMPS_Br
    Visualize AMPERE radial magnetic field
show_jr_and_conductance
    Display radial current and conductance distributions
make_colorbars
    Generate standalone colorbars for figure templates
time_dependent_plot
    Create time series visualizations of simulation evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator

def cs_interpolate(data, cs_grid, new_grid):
    """Interpolate data from cubed sphere to regular grid.

    Parameters
    ----------
    data : array-like
        Values on cubed sphere grid points
    cs_grid : Grid
        Source cubed sphere grid
    new_grid : Grid
        Target regular grid for interpolation

    Returns
    -------
    ndarray
        Interpolated values on new_grid points with same shape as new_grid
    
    Notes
    -----
    Uses linear interpolation via scipy.interpolate.griddata
    """
    interpolated_data = griddata((cs_grid.lat.flatten(), cs_grid.lon.flatten()), data.flatten(), (new_grid.lat.flatten(), new_grid.lon.flatten()), method='linear')
    return interpolated_data.reshape(new_grid.lat.shape)

def globalplot(lon, lat, data, title=None, returnplot=False, levels=None, 
              cmap='viridis', noon_longitude=0, extend='both'):
    """Create global map visualization of field data.

    Parameters
    ----------
    lon : array-like
        Longitude coordinates in degrees
    lat : array-like
        Latitude coordinates in degrees  
    data : array-like
        Field values to plot, must broadcast with lon/lat
    title : str, optional
        Plot title
    returnplot : bool, optional
        If True, return figure and axes objects, by default False
    levels : array-like, optional
        Contour level boundaries, by default None for automatic
    cmap : str, optional
        Matplotlib colormap name, by default 'viridis'
    noon_longitude : float, optional
        Longitude of local noon meridian, by default 0
    extend : {'neither', 'both', 'min', 'max'}, optional
        How to extend colormap at ends, by default 'both'

    Returns
    -------
    tuple, optional
        (figure, axes) if returnplot=True
    """
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.set_global()
    contour = ax.contourf(lon, lat, data, levels=levels, cmap=cmap, extend=extend)
    plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05)
    if title:
        plt.title(title)
    if returnplot:
        return fig, ax

def debugplot(dynamics, title=None, filename=None, noon_longitude=0):
    """Generate diagnostic plots of simulation state.

    Creates visualizations of radial magnetic field, field-aligned currents,
    and equivalent current function for debugging.

    Parameters
    ----------
    dynamics : Dynamics
        Simulation dynamics object containing current state
    title : str, optional
        Plot title
    filename : str, optional
        If provided, save plot to this file
    noon_longitude : float, optional
        Longitude of local noon meridian, by default 0

    Notes
    -----
    Generates plots on a 50x90 lat-lon grid interpolated from simulation grid.
    Shows:
    - Radial magnetic field (Br)
    - Field-aligned currents normalized by radial field
    - Equivalent current function
    """
    NLA, NLO = 50, 90
    lat, lon = np.linspace(-89.9, 89.9, NLA), np.linspace(-180, 180, NLO)
    lat, lon = map(np.ravel, np.meshgrid(lat, lon))
    plt_grid = Grid(lat=lat, lon=lon)
    plt_state_evaluator = BasisEvaluator(dynamics.state.basis, plt_grid)
    plt_b_evaluator = FieldEvaluator(dynamics.state.mainfield, plt_grid, dynamics.state.RI)
    Br = dynamics.state.get_Br(plt_state_evaluator)
    #FAC = plt_state_evaluator.G.dot(dynamics.state.m_imp.coeffs * dynamics.state.m_imp_to_jr) / plt_b_evaluator.br
    #eq_current_function = dynamics.state.get_Jeq(plt_state_evaluator)
    _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.set_global()
    contour = ax.contourf(plt_grid.lon, plt_grid.lat, Br, levels=np.linspace(-300, 300, 22) * 1e-9, cmap='bwr', extend='both')
    plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05)
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.show()

def compare_AMPS_jr_and_CF_currents(amps_jr, cf_currents, lat, lon):
    """
    Compare AMPS jr and curl-free currents.

    Parameters
    ----------
    amps_jr : array-like
        AMPS jr values.
    cf_currents : array-like
        Curl-free currents.
    lat : array-like
        Latitudes of the data points.
    lon : array-like
        Longitudes of the data points.
    """
    fig, ax = plt.subplots()
    ax.scatter(amps_jr, cf_currents)
    ax.set_xlabel('AMPS jr')
    ax.set_ylabel('Curl-free currents')
    plt.show()

def plot_AMPS_Br(amps_Br, lat, lon):
    """
    Plot AMPS Br.

    Parameters
    ----------
    amps_Br : array-like
        AMPS Br values.
    lat : array-like
        Latitudes of the data points.
    lon : array-like
        Longitudes of the data points.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.set_global()
    contour = ax.contourf(lon, lat, amps_Br, levels=np.linspace(-300, 300, 22) * 1e-9, cmap='bwr', extend='both')
    plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05)
    plt.show()

def show_jr_and_conductance(jr, conductance, lat, lon):
    """
    Show jr and conductance.

    Parameters
    ----------
    jr : array-like
        jr values.
    conductance : array-like
        Conductance values.
    lat : array-like
        Latitudes of the data points.
    lon : array-like
        Longitudes of the data points.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.set_global()
    contour = ax.contourf(lon, lat, jr, levels=np.linspace(-.95, .95, 22) * 1e-6, cmap='bwr', extend='both')
    plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05)
    plt.show()
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.set_global()
    contour = ax.contourf(lon, lat, conductance, levels=np.linspace(0, 20, 22), cmap='viridis', extend='max')
    plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05)
    plt.show()

def make_colorbars():
    """
    Create colorbars for the plots.
    """
    fig, ax = plt.subplots()
    norm = plt.Normalize(vmin=0, vmax=20)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
    plt.show()

def time_dependent_plot(dynamics, fig_directory, filecount, lon0, plt_grid, 
                       pltshape, plt_state_evaluator):
    """Create time series visualization frame.

    Generates and saves a single frame for time-dependent visualization of
    simulation evolution.

    Parameters
    ----------
    dynamics : Dynamics
        Simulation dynamics object with current state
    fig_directory : str
        Directory to save output frames
    filecount : int
        Frame number for filename
    lon0 : float
        Reference longitude for local time
    plt_grid : Grid
        Grid for visualization interpolation
    pltshape : tuple
        Shape of plotting grid (nlat, nlon)
    plt_state_evaluator : BasisEvaluator
        Evaluator for computing fields on plot grid

    Notes
    -----
    Saves frame as PNG with radial field colored contours and
    electric potential contour lines in both hemispheres.
    """
    import os
    Blevels = np.linspace(-300, 300, 22) * 1e-9 # color levels for Br
    Philevels = np.r_[-212.5:212.5:5] # color levels for Phi
    fn = os.path.join(fig_directory, 'new_' + str(filecount).zfill(3) + '.png')
    title = 't = {:.3} s'.format(dynamics.current_time)
    Br = dynamics.state.get_Br(plt_state_evaluator)
    _, paxn, paxs, _ = globalplot(plt_grid.lon.reshape(pltshape), plt_grid.lat.reshape(pltshape), Br.reshape(pltshape), title=title, returnplot=True, levels=Blevels, cmap='bwr', noon_longitude=lon0, extend='both')
    Phi = dynamics.state.get_Phi(plt_state_evaluator) * 1e-3
    nnn = plt_grid.lat.flatten() > 50
    sss = plt_grid.lat.flatten() < -50
    paxn.contour(plt_grid.lat[nnn], (plt_grid.lon - lon0)[nnn] / 15, Phi[nnn], colors='black', levels=Philevels, linewidths=.5)
    paxs.contour(plt_grid.lat[sss], (plt_grid.lon - lon0)[sss] / 15, Phi[sss], colors='black', levels=Philevels, linewidths=.5)
    plt.savefig(fn)
    plt.close()