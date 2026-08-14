# Visualization scripts

The maintained entry points are the library APIs:

```python
from pynamit.results import evaluate_projected_input, evaluate_simulation_output
from pynamit.plotting import FigureSettings, render_figure
from pynamit.gui import build_gui
```

`pynamit_panel.py` is the maintained standalone Panel launcher. The other
files in this directory are historical, paper-specific figure recipes. They
remain useful as records of particular figures, but they are not general API
examples and may require their original optional data and plotting packages.

For a new analysis, evaluate fields into ordinary arrays with
`pynamit.results`, then plot them directly or use `render_figure`. This keeps
the scientific calculation independent of a particular plotting frontend.
