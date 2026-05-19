"""Basis interface utilities."""

from abc import ABC, abstractmethod


def basis_kind(basis):
    """Return the normalized kind tag for a basis-like object."""
    kind = getattr(basis, "kind", None)
    return None if kind is None else str(kind).upper()


def is_basis_kind(basis, *kinds):
    """Return whether ``basis`` advertises one of ``kinds``."""
    kind = basis_kind(basis)
    return kind is not None and kind in {str(item).upper() for item in kinds}


def is_sh_basis(basis):
    """Return whether ``basis`` is a spherical-harmonic basis."""
    return is_basis_kind(basis, "SH")


def is_cs_basis(basis):
    """Return whether ``basis`` is a cubed-sphere basis."""
    return is_basis_kind(basis, "CS")


def is_grid_basis(basis):
    """Return whether ``basis`` stores values directly on a grid."""
    return isinstance(basis, GridBasis) or is_basis_kind(basis, "CS", "GRID")


def normalize_solution_basis_kind(kind):
    """Normalize a user supplied solution-basis kind."""
    normalized = str(kind).strip().upper()
    if normalized not in {"SH", "CS"}:
        raise ValueError("solution_basis_kind must be one of ['CS', 'SH'].")
    return normalized


class Basis(ABC):
    """Abstract metadata interface for basis representations.

    Concrete basis classes may expose these fields as regular instance
    attributes. Class-level placeholders satisfy the abstract contract,
    while ``validate_metadata`` checks initialized instances.
    """

    required_attributes = (
        "kind",
        "index_names",
        "index_length",
        "index_arrays",
        "minimum_phi_sampling",
        "caching",
    )

    @property
    def signature(self):
        """Return a stable cache signature for this basis instance."""
        parts = [type(self).__module__, type(self).__qualname__, self.kind]
        for name in ("Nmax", "Mmax", "Nmin", "mean_free", "backend", "is_normalized", "N"):
            if hasattr(self, name):
                parts.append((name, getattr(self, name)))
        return tuple(parts)

    @property
    @abstractmethod
    def kind(self):
        """Short identifier for the basis."""
        pass

    @property
    @abstractmethod
    def index_names(self):
        """Names of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def index_length(self):
        """Total number of basis functions."""
        pass

    @property
    @abstractmethod
    def index_arrays(self):
        """Arrays of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def minimum_phi_sampling(self):
        """Minimum required sampling in phi direction."""
        pass

    @property
    @abstractmethod
    def caching(self):
        """Whether basis evaluations can be cached."""
        pass

    def validate_metadata(self) -> None:
        """Validate that required basis metadata is initialized."""
        missing = [name for name in self.required_attributes if getattr(self, name, None) is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{type(self).__name__} is missing basis metadata: {joined}.")


class GridBasis(Basis):
    """Basis whose coefficients are values on a native grid."""

    def __init__(self):
        """Initialize default grid-basis metadata."""
        self._kind = "GRID"
        self._index_names = None
        self._index_length = None
        self._index_arrays = None
        self._minimum_phi_sampling = 1
        self._caching = False

    @property
    def kind(self):
        """Short identifier for the grid basis."""
        return self._kind

    @kind.setter
    def kind(self, value):
        self._kind = value

    @property
    def index_names(self):
        """Names of indices used in the basis."""
        return self._index_names

    @index_names.setter
    def index_names(self, value):
        self._index_names = value

    @property
    def index_length(self):
        """Total number of grid coefficients."""
        return self._index_length

    @index_length.setter
    def index_length(self, value):
        self._index_length = value

    @property
    def index_arrays(self):
        """Arrays of grid-coordinate indices used in the basis."""
        return self._index_arrays

    @index_arrays.setter
    def index_arrays(self, value):
        self._index_arrays = value

    @property
    def minimum_phi_sampling(self):
        """Minimum required sampling in phi direction."""
        return self._minimum_phi_sampling

    @minimum_phi_sampling.setter
    def minimum_phi_sampling(self, value):
        self._minimum_phi_sampling = value

    @property
    def caching(self):
        """Whether basis evaluations can be cached."""
        return self._caching

    @caching.setter
    def caching(self, value):
        self._caching = bool(value)


class EvaluableBasis(Basis):
    """Basis whose functions can be evaluated on a grid."""

    @abstractmethod
    def get_G(self, grid, derivative=None, cache_in=None, cache_out=False):
        """Evaluate basis functions or derivatives on ``grid``."""
        pass
