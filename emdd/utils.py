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

# # Utilities for the EMD-distance library

import math
import numpy as np
import dataclasses
import logging
from inspect import unwrap

import IPython

from typing import ClassVar, Optional, Union, Tuple, NamedTuple, Literal
from scityping.numpy import NPValue, Array

# ```python
# import holoviews as hv
# ```

# + tags=["remove-cell"]
try:
    import holoviews as hv
except ModuleNotFoundError:
    # One could want to use this package without Holoviews: it is only required by the function `calibration_plot`
    class HoloviewsNotFound:
        def __getattr__(self, attr):
            raise ModuleNotFoundError("Unable to import Holoviews; perhaps it is not installed ?")
    hv = HoloviewsNotFound()
# -

import myst_nb

from .digitize import make_int_superscript

logger = logging.getLogger(__name__)


# ## glue

def glue(name, variable, display=True, print_name=True,
         backend: Optional[Literal['bokeh', 'matplotlib']]=None):
    """Use either `myst_nb.glue` or `myst_nb_bokeh.glue`.
    
    Which function to use is determined by inspecting the argument.
    If `print_name` is True (default), the glue name is also printed; this can 
    useful to find the correct name to use refer to the figure from the
    rendered document.
    
    Supports: Anything 'myst_nb.glue' supports, Bokeh, Holoviews
    """
    # TODO: Return a more nicely formatted object, with _repr_html_,
    #       which combines returned fig object and prints the name below

    if backend is None:
        backend = glue.default_holoviews_backend
    if backend is None:
        raise TypeError("Backend not specified. Either provide as argument, "
                        "or set `glue.default_holoviews_backend`")
    elif backend not in {"bokeh", "matplotlib"}:
        raise ValueError("config.backend should be either 'bokeh' or 'matplotlib'")
    
    if print_name:   
        if IPython.get_ipython():
            IPython.display.display(name)  # Should look nicer in Jupyter Book, especially when there are multiple glue statements
        else:
            print(name)
    mrostr = str(type(variable).mro()).lower()
    if "bokeh" in mrostr:
        # Normal Bokeh object (since HoloViews objects are renderer-agnostic, they should never have "bokeh" in their class inheritance)
        from myst_nb_bokeh import glue_bokeh
        return glue_bokeh(name, bokeh_obj, display)

    if "holoviews" in mrostr:
        import holoviews as hv
        if backend == "bokeh":
            from myst_nb_bokeh import glue_bokeh
            # Render Holoviews object to get a normal Bokeh plot
            bokeh_obj = hv.render(variable, backend="bokeh")
            return glue_bokeh(name, bokeh_obj, display)
        else:
            # Render Holoviews object to get a normal Matplotlib plot
            mpl_obj = hv.render(variable, backend="matplotlib")
            return myst_nb.glue(name, mpl_obj, display)

    else:
        return myst_nb.glue(name, variable, display)

glue.default_holoviews_backend = None

# ## format_scientific

def format_scientific(a: Union[int,float], sig_digits=3, tex=False) -> str:
    """
    Format a number in scientific notation, with the requested number of
    significant digits.

    :param:tex: If True, format the result as a TeX string enclosed with `$`.
    """
    # First deal with the sign, since log10 requires strictly positive values
    if a < 0:
        sgn = "-"
        a = -a
    elif a == 0:
        return "0." + "0"*(sig_digits-1)
    else:
        sgn = ""
    # First round the number to avoid issues with things like 0.99999
    ## vvv EARLY EXITS vvv ##
    if not math.isfinite(a):
        if a == math.inf:
            return "$\\infty$" if tex else "∞"
        elif a == -math.inf:
            return "-$\\infty$" if tex else "-∞"
        else:
            return str(a)
    ## ^^^ EARLY EXITS ^^^ ##
    p = int(math.log10(a))
    if p >= 0:
        a = round(a / 10**p, sig_digits-1)
    else:
        a = round(a / 10**p, sig_digits)  # Need one digit more, because we use the first one to replace the initial '0.'
    # Now we have a good value a with non-pathological loading on the first digit
    # Since a has changed though, we should check again that p is correct
    p2 = int(math.log10(a))  # Most of the time this will be 0
    p += p2
    i, f = str(float(a/10**p2)).split(".")
    if i == "0":
        i, f = f[0], f[1:]
        p -= 1
    f = (f+"0"*sig_digits)[:sig_digits-1]
    #i = int(a // 10**p2)
    #f = str(a % 10**p2).replace('.','')[:sig_digits-1]
    if p == 0:
        s = f"{sgn}{i}.{f}"
    else:
        if tex:
            s = f"{sgn}{i}.{f} \\times 10^{{{make_int_superscript(p)}}}"
        else:
            s = f"{sgn}{i}.{f}×10{make_int_superscript(p)}"

    return f"${s}$" if tex else s

# ### Test

# + tags=["active-ipynb"]
# if __name__ == "__main__":
#     assert format_scientific(0.9999999999999716) == "1.00"
#     assert format_scientific(1.0000000000000284) == "1.00"
#     assert format_scientific(9999.999999999716) == "1.00×10⁴"
#     assert format_scientific(1000.0000000000284) == "1.00×10³"
#     assert format_scientific(5.34) == "5.34"
#     assert format_scientific(32175254) == "3.22×10⁷"
#     assert format_scientific(0.000002789) == "2.79×10⁻⁶"
#     assert format_scientific(0.000002781) == "2.78×10⁻⁶"
#
#     assert format_scientific(0) == "0.00"
#
#     assert format_scientific(-0.9999999999999716) == "-1.00"
#     assert format_scientific(-1.0000000000000284) == "-1.00"
#     assert format_scientific(-9999.999999999716) == "-1.00×10⁴"
#     assert format_scientific(-1000.0000000000284) == "-1.00×10³"
#     assert format_scientific(-5.34) == "-5.34"
#     assert format_scientific(-32175254) == "-3.22×10⁷"
#     assert format_scientific(-0.000002789) == "-2.79×10⁻⁶"
#     assert format_scientific(-0.000002781) == "-2.78×10⁻⁶"
# -

# ## Plotting

from smttask.workflows import ParamColl, ExpandableRV
# Scipy.stats does not provide a public name for the frozen dist types
try:
    from scipy import stats
except ModuleNotFoundError:
    class RVFrozen:  # This is only used in isinstance() checks, so an empty
        pass         # class suffices to avoid those tests failing and simply return `False`
else:
    # This way of finding RVFrozen should be robust across SciPy version, even when the module name changes (as it has)
    RVFrozen = next(C for C in type(stats.norm()).mro() if "frozen" in C.__name__.lower())

# ### `get_bounds`

def get_bounds(*arrs, lower_margin=0.05, upper_margin=0.05) -> Tuple[float, float]:
    """
    Return bounds that include all values of the given arrays, plus a little more.
    How much more is determined by the margins; 0.05 increases the range
    by about 5%.
    Intended for recomputing the bounds in figures.
    
    Returns a tuple of ``(low, high)`` bounds.
    """
    low = min(arr.min() for arr in arrs); high = max(arr.max() for arr in arrs)
    width = high-low
    return (low-lower_margin*width, high+upper_margin*width)


# ### `plot_secondary`

# +
from bokeh.models import Range1d, LinearAxis
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting.figure import Figure

def plot_secondary(plot, element):
    """
    Hook to plot data on a secondary (twin) axis on a Holoviews Plot with Bokeh backend.
    More info:
    - http://holoviews.org/user_guide/Customizing_Plots.html#plot-hooks
    - https://docs.bokeh.org/en/latest/docs/user_guide/plotting.html#twin-axes
    
    Source: https://github.com/holoviz/holoviews/issues/396#issuecomment-1231249233
    """
    fig: Figure = plot.state
    glyph_first: GlyphRenderer = fig.renderers[0]  # will be the original plot
    glyph_last: GlyphRenderer = fig.renderers[-1] # will be the new plot
    right_axis_name = "twiny"
    # Create both axes if right axis does not exist
    if right_axis_name not in fig.extra_y_ranges.keys():
        # Recreate primary axis (left)
        y_first_name = glyph_first.glyph.y
        # If the figure is an overlay, there can be multiple plots associated to the left
        # axis. Assumption: The first renderer determines the name of the left axis
        # (consistent with Holoviews). Any other renderer with the same name for its
        # y axis may contribute to the bounds calculation.
        y_first_min = min(renderer.data_source.data[y_first_name].min()
                          for renderer in fig.renderers
                          if renderer.glyph.y == y_first_name)
        y_first_max = max(renderer.data_source.data[y_first_name].max()
                          for renderer in fig.renderers
                          if renderer.glyph.y == y_first_name)
        y_first_offset = (y_first_max - y_first_min) * 0.1
        y_first_offset = (y_first_max - y_first_min) * 0.1
        fig.y_range = Range1d(
            start=y_first_min - y_first_offset,
            end=y_first_max + y_first_offset
        )
        fig.y_range.name = glyph_first.y_range_name
        # Create secondary axis (right)
        y_last_name = glyph_last.glyph.y
        y_last_min = glyph_last.data_source.data[y_last_name].min()
        y_last_max = glyph_last.data_source.data[y_last_name].max()
        y_last_offset = (y_last_max - y_last_min) * 0.1
        fig.extra_y_ranges = {right_axis_name: Range1d(
            start=y_last_min - y_last_offset,
            end=y_last_max + y_last_offset
        )}
        fig.add_layout(LinearAxis(y_range_name=right_axis_name, axis_label=glyph_last.glyph.y), "right")
    # Set right axis for the last glyph added to the figure
    glyph_last.y_range_name = right_axis_name

# -

# ### `get_bin_widths`

def get_bin_widths(total_points, target_bin_width) -> Array[int,1]:
    """
    Return an array of bin widths, each approximately equal to `target_bin_width`
    and such that their sum is exactly equal to `total_points`.
    This can be used to construct histograms with bars of different width but
    comparable statistical power.
    
    - If `target_bin_width` does not divide `total_points` exactly, some bins
      will be larger by 1.
    - All returned bins have at least the value specified by 
      `target_bin_width`.
    - The subset of bins which are larger, if any, is distributed roughly
      uniformly throughout the list.
    - The distribution of larger bins is deterministic: calling the function
      twice with the same arguments will always return the same list.
      
    Example
    -------
    >>> for tp, tbw in [(243, 30), (30, 30), (243, 20), (239, 12), (100, 30)]:
    ...   bw = get_bin_widths(tp, tbw)
    ...   print(tp, tbw, bw.sum(), bw, np.unique(bw).size)
    243 30 243 [30 30 31 30 30 31 30 31] 2
    30 30 30 [30] 1
    243 20 243 [20 20 20 21 20 20 20 21 20 20 20 21] 2
    239 12 239 [12 13 12 13 12 13 13 12 13 12 13 12 13 13 12 13 12 13 13] 2
    """
    nbins = total_points // target_bin_width
    total_extra = total_points % target_bin_width
    extra_per_bin_all = total_extra // nbins
    extra_to_distribute = total_extra % nbins
    if extra_to_distribute:
        dist_rate = extra_to_distribute / nbins
        distributed_ones = []
        c = 0
        for i in range(nbins):
            new_c = c + dist_rate
            distributed_ones.append(int((int(c) != int(new_c)) and sum(distributed_ones) < extra_to_distribute))
            c = new_c
    else:
        distributed_ones = np.zeros(nbins, dtype=int)
            
    bin_widths = target_bin_width + extra_per_bin_all + np.array(distributed_ones)
    assert bin_widths.sum() == total_points  # Correct number of points
    assert np.unique(bin_widths).size <= 2   # Widths differ by at most 1
    return bin_widths


# ### `plot_param_space`

def _plot_param_hist_dict(Θcoll: ParamColl, θdim="param"):
    rv_params = [name for name, val in Θcoll.items() if isinstance(val, ExpandableRV)]
    kdims = {name: Θcoll.dims.get(name, name) for name in rv_params}
    kdims = {name: dim if isinstance(dim, hv.Dimension) else hv.Dimension(name, label=dim) for name, dim in kdims.items()}
    pdims = {name: hv.Dimension(f"p{name}", label=f"p({kdims[name].label})") for name in rv_params}
    hists = {name: hv.Histogram(np.histogram(Θcoll[name].rvs(1000), bins="auto", density=True),
                                kdims=kdims[name], vdims=pdims[name])
             for name in rv_params}
    # Augment with histograms from nested ParamColls
    nested_paramcolls = [(name, val) for name, val in Θcoll.items() if isinstance(val, ParamColl)]
    for name, paramcoll in nested_paramcolls:
        new_hists = {f"{name}.{pname}": phist
                     for pname, phist in _plot_param_hist_dict(paramcoll, θdim).items()}
        hists = {**hists, **new_hists}
    return hists

def plot_param_space(Θcoll: ParamColl, θdim="param"):
    hists = _plot_param_hist_dict(Θcoll, θdim)
    return hv.HoloMap(hists, kdims=[θdim]).layout().cols(3)

# ## Pretty-print Git version
# (Ported from *mackelab_toolbox.utils*)

from typing import Union
from pathlib import Path
from datetime import datetime
from socket import gethostname
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
    css: str= "color: grey; text-align: right"
    # Default values used when a git repository can’t be loaded
    path  : str="No git repo found"
    branch: str=""
    sha   : str=""
    hostname: str=""
    timestamp: str=None
    def __init__(self, path: Union[None,str,Path]=None, nchars: int=8,
                 sha_prefix: str='#', show_path: str='stem',
                 show_branch: bool=True, show_hostname: bool=False,
                 datefmt: str="%Y-%m-%d"):
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
        """
        ## Set attributes that should always work (don't depend on repo)
        if datefmt:
            self.timestamp = datetime.now().strftime(datefmt)
        else:
            self.timestamp = ""
        if show_hostname:
            self.hostname = gethostname()
        else:
            self.hostname = ""
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
            try:
                self.branch = repo.active_branch.name
            except TypeError:  # Can happen if on a detached head
                pass

    def __str__(self):
        return " ".join((s for s in (self.timestamp, self.hostname, self.path, self.branch, self.sha)
                         if s))
    def __repr__(self):
        return self.__str__()
    def _repr_html_(self):
        hoststr = f"&nbsp;&nbsp;&nbsp;host: {self.hostname}" if self.hostname else ""
        return f"<p style=\"{self.css}\">{self.timestamp}{hoststr}&nbsp;&nbsp;&nbsp;git: {self.path} {self.branch} {self.sha}</p>"

