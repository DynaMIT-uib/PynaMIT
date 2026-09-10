"""Pure Panel launcher for the PynaMIT plotting app.

Run from a saved simulation directory with:

    panel serve /path/to/pynamit_panel.py --show

Set ``PYNAMIT_SIMULATION_DIR`` to open a different simulation
directory without editing this file.
"""

from __future__ import annotations

from pynamit.gui.panel_app import servable

TITLE = "PynaMIT Plot"

app = servable(title=TITLE)
