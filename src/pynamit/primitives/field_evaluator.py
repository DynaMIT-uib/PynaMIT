""" B field geometry module. """

import numpy as np

class FieldEvaluator:
    """ B field geometry class. """

    def __init__(self, mainfield, grid, r):
        self.mainfield = mainfield
        self.grid = grid
        self.r = r

    @property
    def B(self):
        """ Magnetic field vector. """

        if not hasattr(self, '_B'):
            self._B = np.vstack(self.mainfield.get_B(self.r, self.grid.theta, self.grid.lon))
        return self._B

    @property
    def Br(self):
        """ Radial component of the magnetic field. """

        return self.B[0]

    @property
    def Btheta(self):
        """ Theta component of the magnetic field. """

        return self.B[1]

    @property
    def Bphi(self):
        """ Phi component of the magnetic field. """

        return self.B[2]

    @property
    def B_magnitude(self):
        """ Magnitude of the magnetic field vector. """

        if not hasattr(self, '_B_magnitude'):
            self._B_magnitude = np.linalg.norm(self.B, axis = 0)
        return self._B_magnitude

    @property
    def br(self):
        """ Radial component of the magnetic field unit vector. """

        if not hasattr(self, '_br'):
            self._br = self.Br / self.B_magnitude
        return self._br

    @property
    def btheta(self):
        """ Theta component of the magnetic field unit vector. """

        if not hasattr(self, '_btheta'):
            self._btheta = self.Btheta / self.B_magnitude
        return self._btheta

    @property
    def bphi(self):
        """ Phi component of the magnetic field unit vector. """

        if not hasattr(self, '_bphi'):
            self._bphi = self.Bphi / self.B_magnitude
        return self._bphi

    @property
    def basevectors(self):
        """ Base vectors of the magnetic field. """

        if not hasattr(self, '_basevectors'):
            self._basevectors = self.mainfield.basevectors(self.r, self.grid.theta, self.grid.lon)
        return self._basevectors

    @property
    def e1r(self):
        """ Radial component of the e1 basis vector. """

        return self.basevectors[3][0]

    @property
    def e1t(self):
        """ Theta component of the e1 basis vector. """

        return self.basevectors[3][1]

    @property
    def e1p(self):
        """ Phi component of the e1 basis vector. """

        return self.basevectors[3][2]

    @property
    def e2r(self):
        """ Radial component of the e2 basis vector. """

        return self.basevectors[4][0]

    @property
    def e2t(self):
        """ Theta component of the e2 basis vector. """

        return self.basevectors[4][1]

    @property
    def e2p(self):
        """ Phi component of the e2 basis vector. """

        return self.basevectors[4][2]

    @property
    def aeP(self):
        """ a e P matrix. """

        if not hasattr(self, '_aeP'):
            alpha11_eP = self.bphi**2*self.e1t - self.bphi*self.btheta*self.e1p + self.br**2*self.e1t - self.br*self.btheta*self.e1r
            alpha12_eP = -self.bphi*self.br*self.e1r - self.bphi*self.btheta*self.e1t + self.br**2*self.e1p + self.btheta**2*self.e1p
            alpha21_eP = self.bphi**2*self.e2t - self.bphi*self.btheta*self.e2p + self.br**2*self.e2t - self.br*self.btheta*self.e2r
            alpha22_eP = -self.bphi*self.br*self.e2r - self.bphi*self.btheta*self.e2t + self.br**2*self.e2p + self.btheta**2*self.e2p
            self._aeP = np.vstack((np.hstack((np.diag(alpha11_eP), np.diag(alpha12_eP))),
                                   np.hstack((np.diag(alpha21_eP), np.diag(alpha22_eP)))))
        return self._aeP

    @property
    def aeH(self):
        """ a e H matrix. """

        if not hasattr(self, '_aeH'):
            alpha11_eH = self.bphi*self.e1r - self.br*self.e1p
            alpha12_eH = self.br*self.e1t - self.btheta*self.e1r
            alpha21_eH = self.bphi*self.e2r - self.br*self.e2p
            alpha22_eH = self.br*self.e2t - self.btheta*self.e2r
            self._aeH = np.vstack((np.hstack((np.diag(alpha11_eH), np.diag(alpha12_eH))),
                                   np.hstack((np.diag(alpha21_eH), np.diag(alpha22_eH)))))
        return self._aeH

    @property
    def aut(self):
        """ a u t matrix. """

        if not hasattr(self, '_aut'):
            alpha13_ut = -self.Br*self.bphi*self.e1r/self.br + self.Br*self.e1p
            alpha23_ut = -self.Br*self.bphi*self.e2r/self.br + self.Br*self.e2p
            self._aut = np.hstack((alpha13_ut, alpha23_ut))
        return self._aut

    @property
    def aup(self):
        """ a u p matrix. """

        if not hasattr(self, '_aup'):
            alpha13_up = -self.Br*self.e1t + self.Br*self.btheta*self.e1r/self.br
            alpha23_up = -self.Br*self.e2t + self.Br*self.btheta*self.e2r/self.br
            self._aup = np.hstack((alpha13_up, alpha23_up))
        return self._aup