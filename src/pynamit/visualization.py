import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def globalplot(lon, lat, data, central_longitude = 0, scatter = False, **kwargs):
    fig = plt.figure(figsize=(10, 10))
    
    projections = [
        ccrs.Orthographic(central_longitude = central_longitude, central_latitude=-90),  # North Pole
        ccrs.Orthographic(central_longitude = central_longitude, central_latitude=-90),  # South Pole
        ccrs.Mollweide(   central_longitude = central_longitude)  # Global
    ]
    
    titles = ['North Pole View', 'South Pole View', 'Global View (Mollweide Projection)']
    
    for i, (projection, title) in enumerate(zip(projections, titles)):
        if i < 2:  # For the polar views, set_global() to limit the map to the polar regions
            ax = fig.add_subplot(2, 2, i+1, projection = projections[i])
            ax.set_global()
        else:
            ax = fig.add_subplot(2, 1, 2, projection = projections[2])
        ax.coastlines(zorder = 2, color = 'grey')
        if scatter:
            ax.scatter(lon, lat, c = data, transform = ccrs.PlateCarree(), **kwargs)
        else:
            ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), **kwargs)
        ax.set_title(title)

    plt.tight_layout()
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
