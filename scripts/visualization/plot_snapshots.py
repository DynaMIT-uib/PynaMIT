# visualization

import pynamit
import polplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import dipole

ts = [0, .5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 90, 120, 150, 180, 240, 300, 420]
DT = 480  #an offset to apply to all the ts
filename_prefix = 'fac'
shape = (5, 4) # layout of the figure (rows x columns)
assert len(ts) == np.product(shape)
path = '/Users/laundal/Dropbox/git/dynamit/PynaMIT/scripts/simulation/data/pynamit_paper_simulation'
#path = '/Users/laundal/Dropbox/git/dynamit/PynaMIT/scripts/simulation/data/steady_state'

a = pynamit.PynamEye(path)

GLOBAL_TIMESERIES = True
POLAR_TIMESERIES  = False
EQUATORIAL_EFIELD = False


if GLOBAL_TIMESERIES:

    fig_E = plt.figure(figsize = (14, 10))
    fig_B = plt.figure(figsize = (14, 10))
    fig_Q = plt.figure(figsize = (14, 10))
    
    for i, t in enumerate(ts):
        a.set_time(t + DT)
        ax_E = fig_E.add_subplot(shape[0], shape[1], i + 1, projection = a.get_global_projection())
        ax_B = fig_B.add_subplot(shape[0], shape[1], i + 1, projection = a.get_global_projection())
        ax_Q = fig_Q.add_subplot(shape[0], shape[1], i + 1, projection = a.get_global_projection())

        a.plot_electric_potential(ax_E, region = 'global')
        a.plot_electric_field_stream_function(ax_E, region = 'global')

        a.plot_Br(ax_B, region = 'global').set_edgecolor('face')
        a.plot_equivalent_current(ax_B, region = 'global')

        a.plot_joule(ax_Q, region = 'global', levels = np.linspace(-100, 100, 22)*1e-6).set_edgecolor('face')

        for ax in [ax_E, ax_B, ax_Q]:
            a.jazz_global_plot(ax, draw_labels = True if i == 0 else False, draw_coastlines = True if i == 0 else False)
            ax.set_title('t={} s'.format(t))

    for fig in [fig_E, fig_B, fig_Q]:
        fig.tight_layout()

    fig_E.savefig('figures/global_ts_' + filename_prefix + '_efield.png', dpi = 200)
    fig_E.savefig('figures/global_ts_' + filename_prefix + '_efield.pdf')

    fig_B.savefig('figures/global_ts_' + filename_prefix + '_bfield.png', dpi = 200)
    fig_B.savefig('figures/global_ts_' + filename_prefix + '_bfield.pdf')

    fig_Q.savefig('figures/global_ts_' + filename_prefix + '_joule.png', dpi = 200)
    fig_Q.savefig('figures/global_ts_' + filename_prefix + '_joule.pdf')


    plt.show()


if POLAR_TIMESERIES:
    minlat = 50
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(shape[0], shape[1]*2, figure=fig, wspace=0.1, hspace=0.3, left = 0.01, right = .99, bottom = 0.01, top = .95)  # Small wspace

    for i in range(len(ts)):
        row, col = divmod(i, shape[1])

        a.set_time(ts[i])

        
        # Define a subplot for the pair in two adjacent columns
        ax_left = polplot.Polarplot(fig.add_subplot(gs[row, col * 2])     ) # First axis (left)
        ax_right = polplot.Polarplot(fig.add_subplot(gs[row, col * 2 + 1]) )# Second axis (right)

        if EFIELD:
            a.plot_electric_potential(ax_left, region = 'north')
            a.plot_electric_field_stream_function(ax_left, region = 'north')
            a.plot_electric_potential(ax_right, region = 'south')
            a.plot_electric_field_stream_function(ax_right, region = 'south')
        
        if BR:
            a.plot_Br(ax_left, region = 'north').set_edgecolor('face')
            a.plot_equivalent_current(ax_left, region = 'north')
            a.plot_Br(ax_right, region = 'south').set_edgecolor('face')
            a.plot_equivalent_current(ax_right, region = 'south')

        if JOULE:
            a.plot_joule(ax_left, region = 'north', levels = np.linspace(-100, 100, 22)*1e-6).set_edgecolor('face')
            a.plot_joule(ax_right, region = 'south', levels = np.linspace(-100, 100, 22)*1e-6).set_edgecolor('face')
        
        ax_left.ax.set_title('t={} s'.format(ts[i]), loc = 'right')

        if i == 0: # write some labels
            ax_left.writeLATlabels(backgroundcolor = (0, 0, 0, 0), color = 'black')
            ax_right.writeLATlabels(backgroundcolor = (0, 0, 0, 0), north = False, color = 'black')
            ax_left.write(minlat, 12, '12', verticalalignment = 'bottom', horizontalalignment = 'center', ignore_plot_limits=True)        
            ax_left.write(minlat, 18, '18', verticalalignment = 'center', horizontalalignment = 'right', ignore_plot_limits=True)        
            ax_left.write(minlat, 0,  '00', verticalalignment = 'top', horizontalalignment = 'center', ignore_plot_limits=True)        
            ax_right.write(minlat, 12, '12', verticalalignment = 'bottom', horizontalalignment = 'center', ignore_plot_limits=True)        
            ax_right.write(minlat, 6, '06', verticalalignment = 'center', horizontalalignment = 'left', ignore_plot_limits=True)        
            ax_right.write(minlat, 0,  '00', verticalalignment = 'top', horizontalalignment = 'center', ignore_plot_limits=True)        

    # Manually adjust the position of the pairs
    for ax in fig.get_axes():
        pos = ax.get_position()  # Get current axis position
        if ax in fig.axes[0::2]:  # Check if it's a left plot
            pos.x0 += 0.019  # Shift left plots slightly to the right
            pos.x1 += 0.019  # Adjust width to maintain size
        ax.set_position(pos)  # Set the new position


    if EFIELD:
        plt.savefig('figures/polar_ts_ef.png', dpi = 250)
        plt.savefig('figures/polar_ts_ef.pdf')
    if BR:
        plt.savefig('figures/polar_ts_mag.png', dpi = 250)
        plt.savefig('figures/polar_ts_mag.pdf')
    if JOULE:
        plt.savefig('figures/polar_ts_joule.png', dpi = 250)
        plt.savefig('figures/polar_ts_joule.pdf')

    plt.show()





if EQUATORIAL_EFIELD:
    mlt  = np.linspace(8, 24 + 8, 361) % 24 
    dl = np.diff(mlt)[0] * 15 * np.pi / 180 * a.RI
    mlat = np.full_like(mlt, 0)
    d = dipole.Dipole(a.time.year)

    fig, ax = plt.subplots()
    for t in ts:
        if t == 0:
            a.set_time(t, steady_state = True)
        else:
            a.set_time(t)

        mlon = d.mlt2mlon(mlt, a.time)

        glat, glon, error = a.apx.apex2geo(mlat, mlon, 110)

        grid = pynamit.Grid(lat = glat, lon = glon)

        evaluator = pynamit.BasisEvaluator(a.basis, grid)

        phi = evaluator.basis_to_grid(a.m_Phi)

        Br, Btheta, Bphi = a.mainfield.get_B(a.RI, grid.theta, grid.lon)
        Bh = np.sqrt(Btheta**2 + Bphi**2).flatten()

        vr = (np.diff(phi) / dl) / Bh[:-1]

        if t == ts[-1]:
            ax.plot(vr, label = 'steady state', color = 'black', linewidth = 3)
        else:
            ax.plot(vr, label = 't={} s'.format(t))
    ax.set_title('not entirely accurate -- see dl')

    ii = list(map(int, np.linspace(0, len(mlt), 7)))
    ax.set_xticks(ii)
    ax.set_xticklabels(['{:.0f}'.format(x) for x in np.hstack((mlt[ii[:-1]], mlt[0]))])

    plt.show()



