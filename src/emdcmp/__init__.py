"""
The project-root __init__ module does things that we want to either ensure are
always done when the project package is imported, or done exactly once.

Things to ensure are always done:

- Initialize logging
- Set defaults for the Sumatra records viewer
- Define the `footer` variable for use in notebooks
"""

# from pathlib import Path
# import logging
from .config import Config, config
from . import utils

def __getattr__(attr):
    if attr in { "interp1d", "make_empirical_risk_ppf", "draw_R_samples", "Bemd"}:
        from . import emd
        return getattr(emd, attr)
    # elif attr == "Calibrate":
    #     from . import tasks
    #     return tasks.Calibrate
    raise AttributeError(f"Module `emdcmp` does not define '{attr}'.")

# Include this variable at the bottom of notebooks to display the branch name & git commit used to execute it
# from .utils import GitSHA;
# footer = GitSHA()
