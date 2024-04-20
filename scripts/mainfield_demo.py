import polplot
import matplotlib.pyplot as plt
from pynamit import CSprojection
from pynamit import Mainfield
import numpy as np

RE = 6371.2e3

### PLOT A GRID OF COORDS AT 3RE AND THE SAME COORDS MAPPED TO 1RE:
r = 3 * RE 
r_dest = RE
fig, axes = plt.subplots(ncols = 3, figsize = (15, 5))
paxes = [polplot.Polarplot(ax) for ax in axes]

csp = CSprojection(N = 20)

lat, lon = 90 - csp.arr_theta, csp.arr_phi
mask = lat > 50
lat, lon = lat[mask], lon[mask]

for pax, kind in zip(paxes, ['radial', 'dipole', 'igrf']):
    pax.scatter(lat, lon / 15, marker = 'o', s = 30)

    mf = Mainfield(kind = kind)
    th_I, ph_I = mf.map_coords(r_dest, r, 90 - lat, lon)

    pax.scatter(90 - th_I, ph_I/15, marker = 'o', s = 15)
    pax.write(50, 12, kind, size = 14, ha = 'center', va = 'bottom')

plt.tight_layout()
plt.show()


