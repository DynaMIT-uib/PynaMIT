"""Reference checks against Kaiju's Fortran Geopack implementation."""

import datetime as dt
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pynamit.simulation.kaiju_dipole import kaiju_geopack_sm


KAIJU_REPO = Path(os.environ.get("KAIJU_REPO", "~/Repos/kaiju")).expanduser()
GEOPACK_SOURCES = (
    KAIJU_REPO / "src/base/kdefs.F90",
    KAIJU_REPO / "src/base/dates.F90",
    KAIJU_REPO / "src/base/3rd-party/geopack.F90",
)


def _require_fortran_geopack():
    """Return gfortran and Kaiju paths, or skip the test."""
    gfortran = shutil.which("gfortran")
    if gfortran is None:
        pytest.skip("gfortran is not available on PATH")
    missing = [str(path) for path in GEOPACK_SOURCES if not path.exists()]
    if missing:
        pytest.skip(f"Kaiju Geopack source files are not available: {missing}")
    return gfortran


def test_kaiju_geopack_sm_matches_compiled_geo2sm(tmp_path):
    """Python GEO->SM matches Kaiju's compiled GEO2SM basis."""
    gfortran = _require_fortran_geopack()
    program = tmp_path / "test_geo2sm.F90"
    program.write_text(
        """
program test_geo2sm
  use geopack
  implicit none
  real(kind=8) :: xsm, ysm, zsm

  call RECALC(2011, 297, 18, 0, 10)
  call GEO2SM(1.0d0, 0.0d0, 0.0d0, xsm, ysm, zsm)
  write(*,'(3ES26.17)') xsm, ysm, zsm
  call GEO2SM(0.0d0, 1.0d0, 0.0d0, xsm, ysm, zsm)
  write(*,'(3ES26.17)') xsm, ysm, zsm
  call GEO2SM(0.0d0, 0.0d0, 1.0d0, xsm, ysm, zsm)
  write(*,'(3ES26.17)') xsm, ysm, zsm
end program test_geo2sm
""".strip()
        + "\n",
        encoding="ascii",
    )
    executable = tmp_path / "test_geo2sm"
    subprocess.run(
        [
            gfortran,
            "-J",
            str(tmp_path),
            "-I",
            str(tmp_path),
            *(str(source) for source in GEOPACK_SOURCES),
            str(program),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = subprocess.run([str(executable)], check=True, capture_output=True, text=True)

    geo_basis_in_sm = np.array(
        [[float(value) for value in line.split()] for line in output.stdout.splitlines()]
    )
    observed_matrix = np.column_stack(geo_basis_in_sm)
    expected_matrix = kaiju_geopack_sm(dt.datetime(2011, 10, 24, 18, 0, 10)).geo_to_sm_matrix

    np.testing.assert_allclose(observed_matrix, expected_matrix, atol=5e-12)
