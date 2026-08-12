"""Pure Panel launcher for the PynaMIT plotting app.

Run from a saved simulation directory with:

    panel serve /path/to/pynamit_panel.py --show

Set ``PYNAMIT_SIMULATION_DIR`` to open a different simulation
directory without editing this file.
"""

from __future__ import annotations

import os

from pynamit.gui.panel_app import servable

SIMULATION_DIRECTORY = os.environ.get("PYNAMIT_SIMULATION_DIR") or None
TITLE = "PynaMIT Plot"

app = servable(simulation_directory=SIMULATION_DIRECTORY, title=TITLE)
