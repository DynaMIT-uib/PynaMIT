"""Pure Panel launcher for the PynaMIT plotting app.

Run from a saved simulation directory with:

    panel serve /path/to/pynamit_panel.py --show

Set ``PYNAMIT_RUN_DIR`` to open a different saved-run directory without
editing this file.
"""

from __future__ import annotations

import os

from pynamit.visualization.panel_app import servable

RUN_DIRECTORY = os.environ.get("PYNAMIT_RUN_DIR") or None
TITLE = "PynaMIT Plot"

app = servable(run_directory=RUN_DIRECTORY, title=TITLE)
