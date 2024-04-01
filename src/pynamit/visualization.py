import numpy as np
import matplotlib.pyplot as plt
import polplot
import cartopy.crs as ccrs

def globalplot(lon, lat, data, noon_longitude = 0, scatter = False, **kwargs):
    fig = plt.figure(figsize=(10, 10))
    
    title = kwargs.pop('title')
    save = kwargs.pop('save')

    # global plot:
    global_projection = ccrs.Mollweide(   central_longitude = noon_longitude)
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
    # Example data generation for 2D arrays
    lon = np.linspace(-180, 180, 60)
    lat = np.linspace(-90, 90, 30)
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    data_2d = np.sin(lon_2d * np.pi  / 180) * lat_2d

    # Example usage with a specified central longitude
    central_longitude = 90  # For example, to center the maps on 90 degrees East
    globalplot(lon_2d, lat_2d, data_2d, central_longitude=central_longitude)
