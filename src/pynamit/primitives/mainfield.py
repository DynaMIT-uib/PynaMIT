"""Mainfield module."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
from datetime import datetime

import numpy as np
import ppigrf
import apexpy
import dipole
from pynamit.math.constants import RE


# New import
from pynamit.primitives.field import Field

class MainfieldImplementation(ABC):
    """Abstract base class for main field model implementations."""

    @abstractmethod
    def evaluate(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate magnetic field components."""
        pass

    @abstractmethod
    def map_coords(self, r_dest, r, theta, phi) -> Tuple[np.ndarray, np.ndarray]:
        """Map coordinates along field lines."""
        pass

    @abstractmethod
    def conjugate_coordinates(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray]:
        """Find magnetically conjugate points."""
        pass

    @abstractmethod
    def basis_vectors(
        self, r, theta, phi
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate apex coordinate basis vectors."""
        pass

    @abstractmethod
    def dip_equator(self, phi, theta) -> np.ndarray:
        """Calculate colatitude of given magnetic latitude at phi."""
        pass


class DipoleImplementation(MainfieldImplementation):
    """Dipole magnetic field implementation."""

    def __init__(self, epoch: int):
        self.dpl = dipole.Dipole(epoch)

    def evaluate(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Bn, Br = self.dpl.B(90 - theta, r * 1e-3)
        return (Br * 1e-9, -Bn * 1e-9, np.zeros_like(Br))

    def map_coords(self, r_dest, r, theta, phi) -> Tuple[np.ndarray, np.ndarray]:
        r, theta, phi = np.broadcast_arrays(r, theta, phi)
        hemisphere = np.sign(90 - theta)
        la_ = 90 - np.rad2deg(np.arcsin(np.sin(np.deg2rad(theta)) * np.sqrt(r_dest / r)))
        theta_out = 90 - hemisphere * la_
        return theta_out, phi

    def conjugate_coordinates(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray]:
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))
        return 180 - theta, phi

    def basis_vectors(
        self, r, theta, phi
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))
        _d1, _d2, _d3, _e1, _e2, _e3 = self.dpl.get_apex_base_vectors(
            90 - theta, r * 1e-3, R=RE * 1e-3
        )
        return self._transform_vectors(r.size, _d1, _d2, _d3, _e1, _e2, _e3)

    def dip_equator(self, phi, theta) -> np.ndarray:
        phi = np.array(phi) % 360
        return np.zeros_like(phi) + theta

    @staticmethod
    def _transform_vectors(size, _d1, _d2, _d3, _e1, _e2, _e3):
        # Transform vectors from east north up to r, theta phi.
        d1 = np.empty((3, size))
        d2 = np.empty((3, size))
        d3 = np.empty((3, size))
        e1 = np.empty((3, size))
        e2 = np.empty((3, size))
        e3 = np.empty((3, size))

        d1[0], d1[1], d1[2] = _d1[2], -_d1[1], _d1[0]
        d2[0], d2[1], d2[2] = _d2[2], -_d2[1], _d2[0]
        d3[0], d3[1], d3[2] = _d3[2], -_d3[1], _d3[0]

        e1[0], e1[1], e1[2] = _e1[2], -_e1[1], _e1[0]
        e2[0], e2[1], e2[2] = _e2[2], -_e2[1], _e2[0]
        e3[0], e3[1], e3[2] = _e3[2], -_e3[1], _e3[0]

        return d1, d2, d3, e1, e2, e3


class IGRFImplementation(MainfieldImplementation):
    """IGRF magnetic field implementation."""

    def __init__(self, epoch: int, hI: float):
        self.apx = apexpy.Apex(epoch, refh=hI)
        self.epoch_dt = datetime(epoch, 1, 1, 0, 0)

    def evaluate(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Br, Btheta, Bphi = ppigrf.igrf_gc(r * 1e-3, theta, phi, self.epoch_dt)
        return (Br * 1e-9, Btheta * 1e-9, Bphi * 1e-9)

    def map_coords(self, r_dest, r, theta, phi) -> Tuple[np.ndarray, np.ndarray]:
        r, theta, phi = np.broadcast_arrays(r, theta, phi)
        mlat, mlon = self.apx.geo2apex(90 - theta, phi, (r - RE) * 1e-3)
        lat_out, phi_out, _ = self.apx.apex2geo(mlat, mlon, (r_dest - RE) * 1e-3)
        return 90 - lat_out, phi_out

    def conjugate_coordinates(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray]:
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))
        h = (r - RE) * 1e-3
        mlat, mlon = self.apx.geo2apex(90 - theta, phi, h)
        glat, phi_conj, _ = self.apx.apex2geo(-mlat, mlon, h)
        return 90 - glat, phi_conj

    def basis_vectors(
        self, r, theta, phi
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))
        _, _, _, _, _, _, _d1, _d2, _d3, _e1, _e2, _e3 = self.apx.basevectors_apex(
            90 - theta, phi, (r - RE) * 1e-3, coords="geo"
        )
        return DipoleImplementation._transform_vectors(r.size, _d1, _d2, _d3, _e1, _e2, _e3)

    def dip_equator(self, phi, theta) -> np.ndarray:
        phi = np.array(phi) % 360
        mlon = np.linspace(0, 360, 360)
        # Calculate latitude of evenly spaced points.
        lat, lon, _ = self.apx.apex2geo(90 - theta, mlon, self.apx.refh)
        # Interpolate to phi.
        return (np.interp(phi.flatten(), lon % 360, 90 - lat, period=360)).reshape(phi.shape)


class RadialImplementation(MainfieldImplementation):
    """Radial magnetic field implementation."""

    def __init__(self, epoch: int, B0: Optional[float]):
        self.B0 = dipole.Dipole(epoch).B0 if B0 is None else B0

    def evaluate(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, theta, phi = np.broadcast_arrays(r, theta, phi)
        return ((RE / r) ** 2 * self.B0, r * 0, r * 0)

    def map_coords(self, r_dest, r, theta, phi) -> Tuple[np.ndarray, np.ndarray]:
        r, theta, phi = np.broadcast_arrays(r, theta, phi)
        return theta, phi

    def conjugate_coordinates(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray]:
        raise ValueError("Conjugate coordinates do not exist with radial field lines")

    def basis_vectors(
        self, r, theta, phi
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))
        size = r.size
        e = np.vstack((np.ones(size), np.zeros(size), np.zeros(size)))
        n = np.vstack((np.zeros(size), np.ones(size), np.zeros(size)))
        u = np.vstack((np.zeros(size), np.zeros(size), np.ones(size)))
        
        # Calculate sign from evaluate
        # We need a dummy call to evaluate at RE
        b_at_re = self.evaluate(np.array([RE]), np.array([0]), np.array([0]))[0][0]
        sign = np.sign(b_at_re)
        
        d1, e1 = e
        d2, e2 = n * sign * (-1)
        d3, e3 = u * sign
        
        return d1, d2, d3, e1, e2, e3

    def dip_equator(self, phi, theta) -> np.ndarray:
        print('dip_equator: Not defined for mainfield.kind=="radial"')
        return np.full_like(phi, np.nan)


class Mainfield(Field):
    """Class for representing the main magnetic field.

    Delegates to concrete implementations for Dipole, IGRF, or Radial fields.
    """

    def __init__(self, kind="dipole", epoch=2020, hI=0.0, B0=None):
        super().__init__()
        self.kind = kind.lower()
        if self.kind == "dipole":
            self._impl = DipoleImplementation(epoch)
        elif self.kind == "igrf":
            self._impl = IGRFImplementation(epoch, hI)
        elif self.kind == "radial":
            self._impl = RadialImplementation(epoch, B0)
        else:
            raise ValueError("kind must be either radial, dipole or igrf")

    @property
    def dpl(self) -> Optional[dipole.Dipole]:
        """Dipole instance if active."""
        return getattr(self._impl, "dpl", None)

    @property
    def apx(self) -> Optional[apexpy.Apex]:
        """Apex instance if active."""
        return getattr(self._impl, "apx", None)

    def evaluate(self, r, theta, phi):
        """Calculate magnetic field components."""
        return self._impl.evaluate(r, theta, phi)

    def get_sinI(self, r, theta, phi):
        """Calculate sine of the inclination angle."""
        B = np.vstack(self.evaluate(r, theta, phi))
        return -B[0] / np.linalg.norm(B, axis=0)

    def map_coords(self, r_dest, r, theta, phi):
        """Map coordinates along field lines."""
        return self._impl.map_coords(r_dest, r, theta, phi)

    def conjugate_coordinates(self, r, theta, phi):
        """Find magnetically conjugate points."""
        return self._impl.conjugate_coordinates(r, theta, phi)

    def basis_vectors(self, r, theta, phi):
        """Calculate apex coordinate basis vectors."""
        return self._impl.basis_vectors(r, theta, phi)

    def dip_equator(self, phi, theta=90):
        """Calculate colatitude of given magnetic latitude at phi."""
        return self._impl.dip_equator(phi, theta)
