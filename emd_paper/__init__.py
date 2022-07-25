"""
The project-root __init__ module does things that we want to either ensure are
always done when the project package is imported, or done exactly once.

Things to ensure are always done:

- Initialize logging
- Set defaults for the Sumatra records viewer
- Define the `footer` variable for use in notebooks
"""

from pathlib import Path
import logging
from .config import config
import smttask
from smttask.view import RecordStoreView

logging.basicConfig()
logger = logging.getLogger(__name__)

# Configure Sumatra records viewer
RecordStoreView.default_project_dir = config.PATHS.projectdir
smttask.config.load_project(config.PATHS.smtproject)
smttask.config.safe_packages.add('emd-paper')

# Include this variable at the bottom of notebooks to display the branch name & git commit used to execute it
from mackelab_toolbox.utils import GitSHA;
footer = GitSHA()
