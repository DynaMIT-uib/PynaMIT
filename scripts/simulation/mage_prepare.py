"""Prepare resolution-independent forcing for the configured MAGE case.

This is the first of three MAGE workflow entry points:

1. ``mage_prepare.py`` reads GAMERA, ReMIX, and TIEGCM.
2. ``mage_project.py`` projects each configured resolution.
3. ``mage_run.py`` evolves PynaMIT from the projected inputs.

Edit ``SETTINGS`` below for the source and output paths. The preparation
physics and file-format implementation live in
``pynamit.workflows.mage.preparation``.
"""

from __future__ import annotations

from pathlib import Path

from pynamit.workflows.mage.preparation import ForcingSettings, prepare_forcing

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIRECTORY = SCRIPT_DIR / "mage_output" / "2011-10-24"
DEFAULT_GAMERA_DIRECTORY = Path("/disk/Gamera_Dong")
DEFAULT_OUTPUT_PATH = CASE_DIRECTORY / "forcing.h5"

SETTINGS = ForcingSettings(
    gamera_directory=DEFAULT_GAMERA_DIRECTORY, output_path=DEFAULT_OUTPUT_PATH
)


def main(settings: ForcingSettings = SETTINGS) -> None:
    """Prepare forcing from the configured source files."""
    output_path = prepare_forcing(settings)
    print(f"Prepared forcing written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
