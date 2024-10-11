# visualization

import pynamit
import polplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ts = [0, .5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 90, 120, 150, 180, 240, 300, 420]
shape = (5, 4) # layout of the figure (rows x columns)
assert len(ts) == np.product(shape)
path = '/Users/laundal/Dropbox/git/dynamit/PynaMIT/scripts/data/brn_wind'
a = pynamit.PynamEye(path)

GLOBAL_TIMESERIES = True
POLAR_TIMESERIES = False

EFIELD = False

if GLOBAL_TIMESERIES:
    fig = plt.figure(figsize = (14, 12))

    for i, t in enumerate(ts):
        a.set_time(t)
        ax = fig.add_subplot(shape[0], shape[1], i + 1, projection = a.get_global_projection())
        if EFIELD:
            a.plot_electric_potential(ax, region = 'global')
            a.plot_electric_field_stream_function(ax, region = 'global')
        else:            
            a.plot_Br(ax, region = 'global').set_edgecolor('face')
            a.plot_equivalent_current(ax, region = 'global')
        a.jazz_global_plot(ax, draw_labels = True if i == 0 else False)
        ax.set_title('t={} s'.format(t))


    plt.tight_layout()
    if EFIELD:
        plt.savefig('figures/global_ts_efield.png', dpi = 200)
        plt.savefig('figures/global_ts_efield.pdf')
    else:
        plt.savefig('figures/global_ts.png', dpi = 200)
        plt.savefig('figures/global_ts.pdf')
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
        else:
            a.plot_Br(ax_left, region = 'north').set_edgecolor('face')
            a.plot_equivalent_current(ax_left, region = 'north')
            a.plot_Br(ax_right, region = 'south').set_edgecolor('face')
            a.plot_equivalent_current(ax_right, region = 'south')
        
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
    else:
        plt.savefig('figures/polar_ts_mag.png', dpi = 250)
        plt.savefig('figures/polar_ts_mag.pdf')

    plt.show()



