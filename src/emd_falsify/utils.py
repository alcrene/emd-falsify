# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---

# # Utilities for the EMD-falsify library

import math
import numpy as np
import dataclasses
import logging
from inspect import unwrap
try:
    import pandas as pd
except ModuleNotFoundError:
    class pd:
        DataFrame = dict


from typing import ClassVar, Optional, Union, Tuple, NamedTuple, Literal, Dict
from scityping.numpy import NPValue, Array
from numpy.typing import ArrayLike

# -

logger = logging.getLogger(__name__)


# ## `get_bin_sizes`

def get_bin_sizes(total_points, target_bin_size=None) -> Array[int,1]:
    """
    Return an array of bin sizes, each approximately equal to `target_bin_size`
    and such that their sum is exactly equal to `total_points`.
    This can be used to construct histograms with bars of different width but
    comparable statistical power.

    .. Note: "Bin size" refers to the number of points in a bin, rather than
       their width on an axis.
    
    - If `target_bin_size` does not divide `total_points` exactly, some bins
      will be larger by 1.
    - All returned bins have at least the value specified by 
      `target_bin_size`.
    - The subset of bins which are larger, if any, is distributed roughly
      uniformly throughout the list.
    - The distribution of larger bins is deterministic: calling the function
      twice with the same arguments will always return the same list.

    .. rubric:: Auto-determined bin sizes
       When `target_bin_size` is `None`, a heuristic is used to choose a value.
       This heuristic tries to get 16 bins, but will keep bin sizes between 16 and 64.
       (So if there are too few samples, we will get fewer curve points but each
        will be the average of at least 16 samples)
       The target of 16 bins was chosen based on a plot of tanh between -2 and +2:
       with 16 points, kinks are barely noticeable.
       (Choosing the smallest number of bins gives each bin the most statistical power.)
       >>> xarr = np.linspace(-2, 2, 16)
       >>> hv.Curve(zip(xarr, np.tanh(xarr))).opts(fig_inches=5)
       The bin size limits of 16 and 64 are more arbitrary.
      
    Example
    -------
    >>> for tp, tbw in [(243, 30), (30, 30), (243, 20), (239, 12), (100, 30)]:
    ...   bw = get_bin_sizes(tp, tbw)
    ...   print(tp, tbw, bw.sum(), bw, np.unique(bw).size)
    243 30 243 [30 30 31 30 30 31 30 31] 2
    30 30 30 [30] 1
    243 20 243 [20 20 20 21 20 20 20 21 20 20 20 21] 2
    239 12 239 [12 13 12 13 12 13 13 12 13 12 13 12 13 13 12 13 12 13 13] 2
    """
    if target_bin_size is None:
        # See comment in docstring
        target_bin_size = np.clip(total_points//16, 16, 64).astype(int)

    report_err_msg = "This is most likely an unaccounted for corner case with `get_bin_sizes`; please report it along with the values for `total_points` and `target_bin_size` you used."
    nbins = total_points // target_bin_size
    total_extra = total_points % target_bin_size
    extra_per_bin_all = total_extra // nbins
    extra_to_distribute = total_extra % nbins
    if extra_to_distribute:
        dist_rate = extra_to_distribute / nbins
        distributed_ones = []
        c = 0
        for _ in range(nbins):
            new_c = c + dist_rate
            distributed_ones.append(int(int(c) != int(new_c)))
            c = new_c
        # Correct off-by-one errors by either removing or adding to the last bin
        tot_distributed = sum(distributed_ones)
        if tot_distributed < extra_to_distribute:
            assert tot_distributed == extra_to_distribute - 1, "Bin size arithmetic is off by more than one. " + report_err_msg
            last_0_idx = nbins - 1 - next(i for i, n in enumerate(reversed(distributed_ones)) if n == 0)  # Finds the rightmost index with a 0 value in `distributed_ones`
            distributed_ones[last_0_idx] = 1
        if tot_distributed > extra_to_distribute:
            assert tot_distributed == extra_to_distribute + 1, "Bin size arithmetic is off by more than one. " + report_err_msg
            last_1_idx = nbins - 1 - next(i for i, n in enumerate(reversed(distributed_ones)) if n == 1)  # Finds the rightmost index with a 1 value in `distributed_ones`
            distributed_ones[last_1_idx] = 0

    else:
        distributed_ones = np.zeros(nbins, dtype=int)
            
    bin_sizes = target_bin_size + extra_per_bin_all + np.array(distributed_ones)
    assert bin_sizes.sum() == total_points, f"Bins sum to {bin_sizes.sum()}, when they should sum to {total_points}. {report_err_msg}"
    assert np.unique(bin_sizes).size <= 2, f"Bin sizes should not differ by more than one. {report_err_msg}"
    return bin_sizes

# ## Bemd comparison matrix

def compare_matrix(R_samples: Dict[str, ArrayLike]) -> pd.DataFrame:
    """Return the Bemd probabilities as a square DataFrame.

    Probabilities are computed from samples of the expected risk.
    They are the EMD estimate of ``P(R_a < R_b)``, with ``a`` given along
    rows and ``b`` given along columns.

    Example usage:

        >>> import emd_falsify as emd
        >>> [define models A, B, C]
        >>> R_samples = {"A": emd.draw_R_samples(model A),
                         "B": emd.draw_R_samples(model B),
                         "C": emd.draw_R_samples(model C)}
        >>> emd.utils.compare_matrix(R_samples)
                   A      B       C   
            A   0.500   0.468   0.868
            B   0.532   0.500   0.880
            C   0.132   0.120   0.500

    :param:R_samples: A dictionary of samples of the expected risk.
       Each entry is a list of samples for a model. The dictionary keys
       determine the headings of the returned DataFrame.

    .. Note:: If `pandas` is not installed, the values are returned
       as a dictionary.
    """
    R_keys = list(R_samples)
    compare_data = {k: {} for k in R_keys}
    for i, a in enumerate(R_keys):
        for j, b in enumerate(R_keys):
            if i == j:
                assert a == b
                compare_data[b][a] = 0.5
            elif j < i:
                compare_data[b][a] = 1 - compare_data[a][b]
            else:
                compare_data[b][a] = np.less.outer(R_samples[a], R_samples[b]).mean()
    return pd.DataFrame(compare_data)

# ## Pretty-print Git version
# (Ported from *mackelab_toolbox.utils*)

import contextlib
from typing import Union
from pathlib import Path
from datetime import datetime
from socket import gethostname
from importlib.metadata import version
try:
    import git
except ModuleNotFoundError:
    git = None

class GitSHA:
    """
    Return an object that nicely prints the SHA hash of the current git commit.
    Displays as formatted HTML in a Jupyter Notebook, otherwise a simple string.

    .. Hint:: This is especially useful for including a git hash in a report
       produced with Jupyter Book. Adding a cell `GitSHA() at the bottom of
       notebook with the tag 'remove-input' will print the hash with no visible
       code, as though it was part of the report footer.

    Usage:
    >>> GitSHA()
    myproject main #3b09572a
    """
    pcss: str= "color: grey; text-align: right; margin-bottom: 2px"
    hrcss: str= "border-top: 5px grey; margin-top: 2px; margin-bottom: 2px;"
    divcss: str= "width: {width}ex; margin-left: auto; margin-right: 0; padding: 0"
    # Default values used when a git repository can’t be loaded
    path  : str="No git repo found"
    branch: str=""
    sha   : str=""
    hostname: str=""
    timestamp: str=None
    def __init__(self, path: Union[None,str,Path]=None, nchars: int=8,
                 sha_prefix: str='#', show_path: str='stem',
                 show_branch: bool=True, show_hostname: bool=False,
                 datefmt: str="%Y-%m-%d", packages: list[str]=()):
        """
        :param:path: Path to the git repository. Defaults to CWD.
        :param:nchars: Numbers of SHA hash characters to display. Default: 8.
        :param:sha_prefix: Character used to indicate the SHA hash. Default: '#'.
        :param:show_path: How much of the repository path to display.
            'full': Display the full path.
            'stem': (Default) Only display the directory name (which often
                    corresponds to the implied repository name)
            'none': Don't display the path at all.
        :param:datefmt: The format string to pass to ``datetime.strftime``.
            To not display any time at all, use an empty string.
            Default format is ``2000-12-31``.
        :param:packages: A list of package names for which we also want track
            the version. Each package will add a line to the output, formatted
            as "{packagename} : {version}". The package’s version is retrieved
            with `importlib.metadata.version`.

        .. Caution: The version numbers retrieved by `importlib` are set when
           when the package is installed. Therefore if the packages are installed
           with an editable install, the version reported may be different from
           the version actually used. In such a case, rerun `pip install -e`
           to refresh the version.
        """
        ## Set attributes that should always work (don't depend on repo)
        self.timestamp = datetime.now().strftime(datefmt) if datefmt else ""
        self.hostname = gethostname() if show_hostname else ""
        self.packages = {pkgname : version(pkgname) for pkgname in packages}
        ## Set attributes that depend on repository
        # Try to load repository
        if git is None:
            # TODO?: Add to GitSHA a message saying that git python package is not installed ?
            return
        try:
            repo = git.Repo(search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            # Skip initialization of repo attributes and use defaults
            return
        self.repo = repo
        self.sha = sha_prefix+repo.head.commit.hexsha[:nchars]
        if show_path.lower() == 'full':
            self.path = repo.working_dir
        elif show_path.lower() == 'stem':
            self.path = Path(repo.working_dir).stem
        elif show_path.lower() == 'none':
            self.path = ""
        else:
            raise ValueError("Argument `show_path` should be one of "
                             "'full', 'stem', or 'none'")
        self.branch = ""
        if show_branch:
            with contextlib.suppress(TypeError):
                self.branch = repo.active_branch.name

    def __str__(self):
        watermark = " ".join((s for s in (self.timestamp, self.hostname, self.path, self.branch, self.sha)
                                if s))
        if self.packages:
            lwidth = max(len(pkg) for pkg in self.packages.keys())
            rwidth = max(len(ver) for ver in self.packages.values())
            watermark += "\n" + "-"*(3+lwidth+rwidth)
            # Left align both the the package names and their versions. Also align the colons
            watermark += "\n" + "\n".join(
                f"{name: <{lwidth}} : {ver: <{rwidth}}"
                for name, ver in self.packages.items())
        return watermark
    def __repr__(self):
        return self.__str__()
    def _repr_html_(self):
        hoststr = f"&nbsp;&nbsp;&nbsp;host: {self.hostname}" if self.hostname else ""
        mainline_content = f"{self.timestamp}{hoststr}&nbsp;&nbsp;&nbsp;git: {self.path} {self.branch} {self.sha}"
        mainline = f"<p style=\"{self.pcss}\">{mainline_content}</p>"
        if self.packages:
            lwidth = max(len(pkg) for pkg in self.packages.keys())
            rwidth = max(len(ver) for ver in self.packages.values())
            packwidth = 3 + lwidth + rwidth
            packstr = f"<hr style=\"{self.hrcss}\"><p style=\"{self.pcss}\">" \
                      + "<br>".join(f"{name} : {ver}" for name, ver in self.packages.items()) \
                      + "</p>"
        else:
            packwidth = 0
            packstr = ""
        width = max(len(mainline_content), packwidth)
        divcss = self.divcss.format(width=int(1*width))
        return f"<div style=\"{divcss}\">{mainline}{packstr}</div>"


