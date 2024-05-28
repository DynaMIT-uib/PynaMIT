" B field geometry module "

import numpy as np

class BGeometry:
    " B field geometry class "

    def __init__(self, mainfield, grid, r):
        self.mainfield = mainfield
        self.grid = grid
        self.r = r


    @property
    def B(self):
        " Magnetic field vector."
        if not hasattr(self, '_B'):
            self._B = np.vstack(self.mainfield.get_B(self.r.flatten(), self.grid.theta, self.grid.lon))
        return self._B


    @property
    def B_magnitude(self):
        " Magnitude of the magnetic field vector."
        if not hasattr(self, '_B_magnitude'):
            self._B_magnitude = np.linalg.norm(self.B, axis = 0)
        return self._B_magnitude


    @property
    def br(self):
        " Radial component of the magnetic field unit vector. "
        if not hasattr(self, '_br'):
            self._br = self.B[0] / self.B_magnitude
        return self._br


    @property
    def btheta(self):
        " Theta component of the magnetic field unit vector."
        if not hasattr(self, '_btheta'):
            self._btheta = self.B[1] / self.B_magnitude
        return self._btheta


    @property
    def bphi(self):
        " Phi component of the magnetic field unit vector."
        if not hasattr(self, '_bphi'):
            self._bphi = self.B[2] / self.B_magnitude
        return self._bphi


    @property
    def sinI(self):
        " Sin of the inclination angle."
        if not hasattr(self, '_sinI'):
            self._sinI = -self.br / np.sqrt(self.btheta**2 + self.bphi**2 + self.br**2)
        return self._sinI