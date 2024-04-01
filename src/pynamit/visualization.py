import numpy as np
import matplotlib.pyplot as plt
import polplot
import cartopy.crs as ccrs
from scipy.interpolate import griddata

def cs_interpolate(projection, inlat, inlon, values, outlat, outlon, **kwargs):
    """ interpolate from cubed sphere grid to new points lon, lat 

    Parameters
    ----------
    projection: cubed sphere projection object
    inlat: latitudes of input
    inon : longitudes of input
    etc.

    """
    inlat, inlon, values = map(np.ravel, np.broadcast_arrays(inlat, inlon, values))
    in_r = np.vstack((np.cos(np.deg2rad(inlat)) * np.cos(np.deg2rad(inlon)),
                      np.cos(np.deg2rad(inlat)) * np.sin(np.deg2rad(inlon)),
                      np.sin(np.deg2rad(inlat))  
                    ))

    outlon, outlat = np.broadcast_arrays(outlon, outlat)
    shape = outlon.shape # get the shape so we can reshape the result in the end
    outlon, outlat = outlon.flatten(), outlat.flatten()

    result = np.zeros_like(outlon) -1 

    xi_o, eta_o, block_o = projection.geo2cube(outlon, outlat)

    # go through each block:
    for i in range(6):
        jjj = (block_o == i) # these are the points we want to specify

        # find the points that are on the right side:
        _, th0, ph0 = projection.cube2spherical(0, 0, i)
        r0 = np.array([np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)])
        iii = np.sum(r0.reshape((-1, 1)) * in_r, axis = 0) > 0
        xi_i, eta_i, _ = projection.geo2cube(inlon[iii], inlat[iii], block = i)
        result[jjj] = griddata(np.vstack((xi_i, eta_i)).T, values[iii], np.vstack((xi_o[jjj], eta_o[jjj])).T, **kwargs)

    return(result.reshape(shape))



def globalplot(lon, lat, data, noon_longitude = 0, scatter = False, **kwargs):
    fig = plt.figure(figsize=(10, 10))
    
    if 'title' in kwargs.keys():
        title = kwargs.pop('title')
    else:
        title = None
    if 'save' in kwargs.keys():
        save = kwargs.pop('save')
    else:
        save = None

    # global plot:
    global_projection = ccrs.PlateCarree(central_longitude = noon_longitude)
    ax = fig.add_subplot(2, 1, 2, projection = global_projection)    
    ax.coastlines(zorder = 2, color = 'grey')
    if scatter:
        ax.scatter(lon, lat, c = data, transform = ccrs.PlateCarree(), **kwargs)
    else:
        ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), **kwargs)
    
    if title != None:
        ax.set_title(title)

    pax1 = polplot.Polarplot(fig.add_subplot(2, 2, 1), minlat = 50)
    pax2 = polplot.Polarplot(fig.add_subplot(2, 2, 2), minlat = 50)

    lon = lon - noon_longitude + 180

    iii = lat > 50
    if scatter:
        pax1.scatter(lat[iii],  lon[iii] / 15, c = data[iii], **kwargs)
    else:
        pax1.contourf(lat[iii], lon[iii] / 15, data[iii], **kwargs)
    pax1.ax.set_title('North')

    iii = lat < -50
    if scatter:
        pax2.scatter(lat[iii], lon[iii] / 15, c = data[iii], **kwargs)
    else:
        pax2.contourf(lat[iii], lon[iii] / 15, data[iii], **kwargs)
    pax2.ax.set_title('South')


    plt.tight_layout()

    if save != None:
        plt.savefig(save)
    else:
        plt.show()


if __name__ == "__main__":

    # import cubedsphere submodule
    import os, sys
    cs_path = os.path.join(os.path.dirname(__file__), 'cubedsphere')
    sys.path.insert(0, cs_path)
    import cubedsphere
    csp = cubedsphere.CSprojection() # cubed sphere projection object

    Ncs = 30
    k, i, j = csp.get_gridpoints(Ncs)
    xi, eta = csp.xi(i, Ncs), csp.eta(j, Ncs)
    _, theta, phi = csp.cube2spherical(xi, eta, k, deg = True)

    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)

    import pyamps
    from visualization import globalplot
    import matplotlib.pyplot as plt
    from lompe import conductance
    import dipole
    import datetime
    import polplot

    # specify a time and Kp (for conductance):
    date = datetime.datetime(2001, 5, 12, 21, 45)
    Kp   = 5
    d = dipole.Dipole(date.year)

    # noon longitude
    lon0 = d.mlt2mlon(12, date)

    hall, pedersen = conductance.hardy_EUV(phi, 90 - theta, Kp, date, starlight = 1, dipole = True)

    hall_plt = cs_interpolate(csp, 90 - theta, phi, hall, lat, lon)
    pede_plt = cs_interpolate(csp, 90 - theta, phi, pedersen, lat, lon)

    globalplot(lon, lat, hall_plt, noon_longitude = lon0, levels = np.linspace(0, 20, 100))

