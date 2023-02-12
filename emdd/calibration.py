# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,md:myst
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python (EMD-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown]
# # EMD calibration
#
# $\newcommand{\nN}{\mathcal{N}}
# \newcommand{\Unif}{\mathop{\mathrm{Unif}}}$

# %%
import os
import logging
import multiprocessing as mp
from collections.abc import Callable
from functools import partial, cache
from itertools import repeat, product, tee
from bisect import bisect
from typing import Optional, Union, Any, Dict, List, Tuple, Literal
from dataclasses import dataclass, field
import numpy as np
from tqdm.auto import tqdm

from emdd import Config
from emdd.emd import Bconf#compute_stats_m1
from emdd.path_sampling import generate_quantile_paths
from emdd.memoize import memoize

config = Config()
logger = logging.getLogger(__name__)

# %% [markdown]
# from emdd.emd import Bconf

# %% tags=["remove-cell"]
try:
    import holoviews as hv
except ModuleNotFoundError:
    # One could want to use this package without Holoviews: it is only required by the function `calibration_plot`
    class HoloviewsNotFound:
        @property
        def Dimension(self):
            def holoviews_not_loaded(*args, **kwargs):
                raise ModuleNotFoundError("Unable to import Holoviews; perhaps it is not installed ?")
            return holoviews_not_loaded
        def __getattr__(self, attr):
            raise ModuleNotFoundError("Unable to import Holoviews; perhaps it is not installed ?")
    hv = HoloviewsNotFound()


# %% [markdown] tags=["remove-cell"]
# Notebook only imports

# %% tags=["active-ipynb", "remove-cell"]
# from math import log10
# from numpy.random import default_rng
# from scipy import stats
# import pandas as pd
# from emdd.utils import ParamColl, expand
# logging.basicConfig(level=logging.WARNING)
# logger.setLevel(logging.ERROR)

# %% tags=["active-ipynb", "remove-cell", "skip-execution"]
# logger.setLevel(logging.DEBUG)

# %% tags=["remove-cell", "active-ipynb"]
# # Use Bokeh backend, unless we are running within Jupyter Book with a LaTeX target
# page_backend = "bokeh"
# try:
#     with open(config.rootdir/"md-source/outut-format.build-msg", 'r') as f:
#         msg = f.read()
# except FileNotFoundError:
#     pass
# else:
#     if msg.strip() == "latex":
#         #config.figures.backend = "matplotlib"
#         page_backend = "matplotlib"
#
# hv.extension(page_backend)

# %% [markdown] tags=[]
# ## Helper class for accumulating calibration statistics

# %%
@dataclass
class CalibrationStats:
    data_param_names: List[Union[hv.Dimension,str]]
    modelA_param_names: List[Union[hv.Dimension,str]]
    modelB_param_names: List[Union[hv.Dimension,str]]
    params     : List[tuple]=field(default_factory=lambda: [])
        # We expect relatively few `add_result` calls to create new keys, so the
        # overhead of searching and appending to `keys` should not be noticeable
        # NB: Must not be an ndarray, to avoid unpacking keys.
    # MC accumulator 
    N         : np.ndarray=field(default_factory=lambda: np.array([], dtype=int))
    _counts_A : np.ndarray=field(default_factory=lambda: np.array([], dtype=float))
    # EMD accumulator
    c         : np.ndarray=field(default_factory=lambda: np.array([], dtype=float))
    Bconf_emd : np.ndarray=field(default_factory=lambda: np.empty((0,0), dtype=float))
    # Stat table output
    # (name, label, attribute)   <-- 'attribute' must match an attribute or @property of `self`
    stat_cols_mc : List[Tuple[str, str, str]] = (
        ("N"        , "Sampled data sets", "N"),
        ("accept"   , "accept rate (A)"  , "accept_rate"),
        ("Bconf_mc" , "B_conf (MC)"      , "Bconf_mc")  # <-- Label must match hard-coded value in `Bconf_mc_dim`
    )
    stat_cols_emd : List[Tuple[str, str, str]] = (
        ("Bconf_emd", "B_conf (EMD)"     , "Bconf_emd"),  # <-- Label must match hard-coded value in `Bconf_emd_dim`
    )
    """
    Attributes:
        param_names: Concatenation of `data_param_names`, `modelA_param_names`, `modelB_param_names`.
           Used as parameter dimension labels.
        params: Ordered list of parameters, for which we have matching data.
        N: For each parameter key, number of MC samples we have accumulated.
           1d array of shape (params).
        _counts_A: For each parameter key, number of MC samples where logpA > logpB
           1d array of shape (params).
        c: Array (1d) of c values for which we have computed an EMD approximation of the bconf.
           This array is kept sorted: when a new c value is added, a column is inserted at
           the correct position in `Bconf_emd`.
        accept_rate: ratio of samples where logpA > logpB for each parameter set
        reject_rate: ratio of samples where logpA < logpB for each parameter set
        bconf_mc: bconf vector, as estimated from MC samples for each parameter set
        Bconf_emd: Evaluations of the bconf using the EMD approximation, for different
           parameter and c values.
           2d array of shape (params x c)
           
    Dimension attributes:
        param_dims        ( = data_param_dims + model_param_dims)
        model_param_dims  ( = modelA_param_dims + modelB_param_dims)
        modelA_param_dims
        modelB_param_dims
        c_dim
        stat_dims_mc
        stat_dims_emd
    """
    
    def __len__(self):
        """Return the number of different recorded parameters."""
        return len(self.params)
    
    def get_param_index(self, params: tuple, if_not_found: Literal["raise","create"]="raise"):
        """
        Return the index of the entry in `self.params` matching `params`.
        
        Args:
            - params: The param key for which we want the index.
            - if_not_found: What to do if not match is found.
              + "raise" (default): Raise `KeyError`.
              + "create": Add the param key to our list and extend the data structures.
        
        Raises:
            - TypeError if `params` is a list or NumPy array.
            - ValueError if the length `params` does not match the length of `self.param_names`.
            - KeyError if `params` looks valid but no matching entry was found and 
        """
        if isinstance(params, (list, np.ndarray)):
            raise TypeError("`params` should be a tuple of parameter values. "
                            "If there is only one parameter and it consists of "
                            "a list or array, wrap it in a tuple.")
        elif not isinstance(params, tuple):
            params = (params,)
        if len(params) != len(self.param_names):
            raise ValueError(f"Received {len(params)} parameters when "
                             f"{len(self.param_names)} were expected.")

        try:
            i = self.params.index(params)
        except ValueError:
            if if_not_found == "raise":
                raise KeyError(f"No param key was found matching: {params}.")
            else:
                i = len(self.params)
                self.params.append(params)
                self._counts_A = np.concatenate((self._counts_A, [0]))
                self.N = np.concatenate((self.N, [0]))
                self.Bconf_emd = np.vstack((self.Bconf_emd,
                                            np.empty_like(self.c)*np.nan))
        
        return i
    
    def get_c_index(self, c: float, if_not_found: Literal["raise","create"]="raise"):
        """
        Return the index of the entry in `self.c` matching `c`.
        
        If a matching c value is not found, and `if_not_found` is "create", a column
        is inserted in `self.Bconf_emd` in order to keep the c values sorted.
        
        Args:
            - c: The c value for which we want the index.
            - if_not_found: What to do if not match is found.
              + "raise" (default): Raise `KeyError`.
              + "create": Add the c value to our list and reshape the data structures.
        
        Raises:
            - KeyError if `params` looks valid but no matching entry was found and 
        """
        i = np.searchsorted(self.c, c)
        if i == self.c.size or self.c[i] != c:
            # No matching c was found: we need to create one
            if if_not_found == "raise":
                raise KeyError(f"No c value was found matching: {params}.")
            else:
                self.Bconf_emd = np.hstack((self.Bconf_emd[:,:i],
                                            np.empty((len(self.params), 1))*np.nan,
                                            self.Bconf_emd[:,i:]))
                self.c = np.concatenate((self.c[:i], [c], self.c[i:]))
        return i
    
    def add_mc_result(self, params: tuple, logpA: float, logpB: float):
        """
        Args:
            params: Tuple key containing the model parameters used to
               generate simulated data.
            logpA: Log likelihood of model A on the simulated data.
            logpB: Log likelihood of model B on the simulated data.
        """
        # We need the special case when logpA == logpB, otherwise when models are
        # the same, the mc result predicts model B with 100% certainty.
        result = 0.5 if logpA == logpB else int(logpA > logpB)
        
        i = self.get_param_index(params, if_not_found="create")
        self._counts_A[i] += result
        self.N[i] += 1
        
    def add_emd_result(self, params: tuple, c: float, bconf: float):
        i = self.get_param_index(params, if_not_found="create")
        j = self.get_c_index(c, if_not_found="create")
        self.Bconf_emd[i,j] = bconf
        
    def get_counts(self, params) -> Tuple[int, int]:
        """
        Return the (N, counts_A) tuple corresponding to `key`.
        Return (0, 0) if no entry matches the given key.
        """
        try:
            i = self.params.index(params)
        except ValueError:
            return 0, 0
        else:
            return self.N[i], self._counts_A[i]
    @property
    def param_names(self):
        return self.data_param_names + self.model_param_names
    @property
    def model_param_names(self):
        return self.modelA_param_names + self.modelB_param_names
    @property
    def param_dims(self):
        return self.data_param_dims + self.model_param_dims
    @property
    def data_param_dims(self):  # Ensures Dimension instances are created only once
        try:
            return self._data_param_dims
        except AttributeError:
            self._data_param_dims = [θ if isinstance(θ, hv.Dimension) else hv.Dimension(θ)
                                     for θ in self.data_param_names]
            return self._data_param_dims
    @property
    def model_param_dims(self):
        return self.modelA_param_dims + self.modelB_param_dims
    @property
    def modelA_param_dims(self):  # Idem
        try:
            return self._modelA_param_dims
        except AttributeError:
            self._modelA_param_dims = [θ if isinstance(θ, hv.Dimension) else hv.Dimension(θ)
                                        for θ in self.modelA_param_names]
            return self._modelA_param_dims
    @property
    def modelB_param_dims(self):  # Idem
        try:
            return self._modelB_param_dims
        except AttributeError:
            self._modelB_param_dims = [θ if isinstance(θ, hv.Dimension) else hv.Dimension(θ)
                                       for θ in self.modelB_param_names]
            return self._modelB_param_dims
    @property
    def c_dim(self):
        try:
            return self._c_dim
        except AttributeError:
            self._c_dim = hv.Dimension("c", label="EMD constant")
            return self._c_dim
    @property
    def Bconf_mc_dim(self):
        return self.stat_dims_mc[self.stat_names_mc.index("Bconf_mc")]
    @property
    def Bconf_emd_dim(self):
        return self.stat_dims_emd[self.stat_names_emd.index("Bconf_emd")]
    @property
    def stat_names_mc(self):
        return [stat[0] for stat in self.stat_cols_mc]
    @property
    def stat_names_emd(self):
        return [stat[0] for stat in self.stat_cols_emd]
    @property
    def stat_labels_mc(self):
        return [stat[1] for stat in self.stat_cols_mc]
    @property
    def stat_labels_emd(self):
        return [stat[1] for stat in self.stat_cols_emd]
    @property
    def stat_dims_mc(self):
        try:
            return self._stat_dims_mc
        except AttributeError:
            self._stat_dims_mc = [hv.Dimension(stat[0], label=stat[1]) for stat in self.stat_cols_mc]
            return self._stat_dims_mc
    @property
    def stat_dims_emd(self):
        try:
            return self._stat_dims_emd
        except AttributeError:
            self._stat_dims_emd = [hv.Dimension(stat[0], label=stat[1]) for stat in self.stat_cols_emd]
            return self._stat_dims_emd
        
    @property
    def accept_rate(self):
        return self._counts_A / N
    @property
    def reject_rate(self):
        return 1 - self._counts_A / N
    @property
    def Bconf_mc(self):
        return self.accept_rate
        #with np.errstate(divide="ignore"):  # No need to warn when reject=0: inf is the result we want in that case
        #    return self.accept_rate / self.reject_rate
    # @property
    # def bconf_mc(self):
    #     with np.errstate(divide="ignore"):  # No need to warn when reject=0: inf is the result we want in that case
    #         return np.log(self.Bconf_mc)
    @property
    def statvals_mc(self):
        return [getattr(self, stat[2]) for stat in self.stat_cols_mc]
    
    @property
    def statvals_emd(self):
        return [getattr(self, stat[2]) for stat in self.stat_cols_emd]

    # TODO: stats() which returns an xarray with both mc and emd data (emd data has extra c dimension)
    def stats_mc(self):
        "Return Pandas DataFrame compatible dict."
        statvals = [getattr(self, stat[2]) for stat in self.stat_cols]
        return {label: {Θ: v for Θ, v in zip(self.params, statcol)}
                for label, statcol in zip(self.statlabels_mc, statvals)}
    def statsdf_mc(self) -> "pandas.DataFrame":
        import pandas as pd
        param_names = [getattr(θdim, "label", θdim) for θdim in self.param_names]  # Replace holoviews Dimensions by their label
        df = pd.DataFrame(self.stats_mc())
        df.index.set_names(param_names, inplace=True)
        return df
    def statshv_mc(self) -> "holoviews.Table":
        tab = hv.Table([(*Θ, *rowstats) for Θ, rowstats in zip(self.params, zip(*self.statvals_mc))],
                       kdims=self.param_dims,
                       vdims=self.stat_dims_mc
                      )
        return tab

    def statshv_emd(self) -> "holoview.Table":
        emd_keys = ( (*Θ, c) for Θ in self.params for c in self.c )
        tab = hv.Table([(*key, Bconf)
                        for (key, Bconf) in zip(emd_keys, self.Bconf_emd.flat)],
                       kdims=self.param_dims + [self.c_dim],
                       vdims=self.stat_dims_emd)
        return tab


# %% [markdown]
# ## Figures

# %%
σparam_dim = hv.Dimension("σparam", label="sorted Θ")


# %%
def make_fig_Bconf_mc(calib: CalibrationStats, metric: Literal["Bconf","bconf"]="Bconf") -> "holoviews.HoloMap":
    """
    Construct a Holoviews figure for the b_conf curves computed with Monte
    Carlo, as part of an EMD calibration experiment. A full calibration
    experiment combines these results with the b_conf curves using the EMD
    approximation.
    
    Args:
        calib (CalibrationStats): Should be provided as a `CalibrationStats`
            object, which serves to accumulate results.
        metric ("Bconf" or "bconf"): Which metric to plot.
            - "Bconf" is a probability.
              It takes values between 0 and 1, centered on 0.5.
            - "bconf" is a logistic transformation of "Bconf".
              It takes values between -∞ and +∞, centered on 0.
              
    Returns:
        Instance of `holoviews.HoloMap`: this allows the caller to modify
        figure display styling, and to combine it with further
        figure elements (most usefully those of `make_fig_Bconf_emd`).
    
    .. Note:: The returned figure shows only one data set at a time, and
       requires interactivity to view other dataset.
       To convert to a static grid showing all panels, use `flatten_Bconf_panels`.
    """
    pass
    #### <-- Split here into cells in a Jupyter notebook, for easier testing/development

    mc_stats = calib.statshv_mc()
    if metric != "Bconf":
        raise NotImplementedError(f"`metric`={metric}")
    metric_dim = (calib.Bconf_mc_dim if metric == "Bconf" else
                  calib.bconf_mc_dim)
    pruned_param_dims = [dim for dim in calib.param_dims if len(set(mc_stats[dim])) > 1]
    pruned_data_param_dims = [dim for dim in calib.data_param_dims if len(set(mc_stats[dim])) > 1]
    pruned_model_param_dims = [dim for dim in calib.model_param_dims if len(set(mc_stats[dim])) > 1]

    grouped_stats = mc_stats.reindex(pruned_param_dims).groupby(pruned_data_param_dims)
    ####

    panels = {}
    for key, _stats in grouped_stats.items():  # NB: Don’t use 'stats', which is a module name
        σ = np.argsort(_stats[metric_dim])[::-1]

        # σparam_dim defined globally in module
        sorted_stats = _stats.iloc[σ].add_dimension(
            σparam_dim, 0, np.arange(len(_stats))
        )
        
        points = sorted_stats.to.scatter(kdims=[σparam_dim], vdims=[metric_dim] + pruned_model_param_dims,
                                         label=f"{metric} (Monte Carlo)")
        curve = sorted_stats.to.curve(kdims=[σparam_dim], vdims=[metric_dim] + pruned_model_param_dims,
                                     label=f"{metric} (Monte Carlo)")

        if metric == "bconf":
            # Replace ±∞ by a large finite number, so the curve is plotted, and fix the ranges so those infinity points are excluded.
            bconf = _stats[metric_dim]
            if len(np.isfinite(bconf)) < 2:  # Cannot rescale axis with less than 2 finite points
                high = bconf[np.isfinite(bconf)].max()
                low  = bconf[np.isfinite(bconf)].min()
                height = high - low
                _range = (low-0.1*height, high+0.1*height)
                curve.data.replace([np.inf, -np.inf], [1e5, -1e5], inplace=True)
                points.data.replace([np.inf, -np.inf], [1e5, -1e5], inplace=True)
                curve = curve.redim.range(**{metric_dim.name: _range})
                points = points.redim.range(**{metric_dim.name: _range})
        
        panel = curve * points

        panels[key] = panel
    ###

    hmap = hv.HoloMap(panels,
                      kdims=grouped_stats.kdims)
    hmap.opts(hv.opts.Scatter(size=4, toolbar="above", backend="bokeh"));
    
    # Larger, interactive, single-panel holomap: users can select a particular data set
    hmap.opts(hv.opts.Overlay(show_legend=True, legend_position="right")) \
        .opts(height=300, width=700, backend="bokeh")
    hmap
    ####

    return hmap


# %%
def make_fig_Bconf_emd(calib: CalibrationStats, metric: Literal["Bconf","bconf"]="Bconf") -> "holoviews.HoloMap":
    """
    Construct a Holoviews figure for the b_conf curves computed with the
    EMD approximation, as part of an EMD calibration experiment. A full
    calibration experiment combines these results with the b_conf curves 
    using Monte Carlo.
    
    Calibration data should be provided as a `CalibrationStats` object,
    which serves to accumulate results.
    The returned value is an instance of `holoviews.HoloMap`: this allows the
    user to easily modify its display styling, and to combine it with further
    figure elements (most usefully those of `make_fig_Bconf_emd`).
    
    .. Note:: The returned figure shows only one data set at a time, and
       requires interactivity to view other dataset.
       To convert to a static grid showing all panels, use `flatten_Bconf_panels`.
    """
    pass
    #### <-- Split here into cells in a Jupyter notebook, for easier testing/development

    emd_stats = calib.statshv_emd()
    if metric != "Bconf":
        raise NotImplementedError(f"`metric`={metric}")
    metric_dim = (calib.Bconf_emd_dim if metric == "Bconf" else
                  calib.bconf_emd_dim)
    c_dim     = calib.c_dim
    pruned_param_dims = [dim for dim in calib.param_dims if len(set(emd_stats[dim])) > 1]
    pruned_data_param_dims = [dim for dim in calib.data_param_dims if len(set(emd_stats[dim])) > 1]
    pruned_model_param_dims = [dim for dim in calib.model_param_dims if len(set(emd_stats[dim])) > 1]

    grouped_stats = emd_stats.reindex(pruned_param_dims + [calib.c_dim]).groupby(pruned_data_param_dims)
    ###


    panels = {}
    for key, _stats_all_c in grouped_stats.items():  # NB: Don’t use 'stats', which is a module name       
        plot_elements = []
        for c, _stats in _stats_all_c.groupby(calib.c_dim).items():
            σ = np.argsort(_stats[metric_dim])[::-1]

            # σparam_dim defined globally in module
            sorted_stats = _stats.iloc[σ].add_dimension(
                σparam_dim, 0, np.arange(len(_stats))
            )
            
            # TODO: Use groupby arg of `to.curve` to remove inner loop
            curve = sorted_stats.to.curve(kdims=[σparam_dim], vdims=[metric_dim] + pruned_model_param_dims,
                                         label=f"b_conf (EMD), c={c}")
            points = sorted_stats.to.scatter(kdims=[σparam_dim], vdims=[metric_dim] + pruned_model_param_dims,
                                             label=f"b_conf (EMD), c={c}")
            plot_elements.append(curve)
            plot_elements.append(points)
        
            # Replace ±inf by a large finite number, so the curve is plotted, and fix the ranges so those infinity points are excluded.
            if metric == "bconf":
                bconf = _stats[bconf_dim]
                if len(np.isfinite(bconf)) < 2:  # Cannot rescale axis with less than 2 finite points
                    high = bconf[np.isfinite(bconf)].max()
                    low  = bconf[np.isfinite(bconf)].min()
                    height = high - low
                    _range = (low-0.1*height, high+0.1*height)
                    curve.data.replace([np.inf, -np.inf], [1e5, -1e5], inplace=True)
                    points.data.replace([np.inf, -np.inf], [1e5, -1e5], inplace=True)
                    curve = curve.redim.range(**{metric_dim.name: _range})
                    points = points.redim.range(**{metric_dim.name: _range})
                
        panel = hv.Overlay(plot_elements)
        
        panels[key] = panel
    ###
    color = hv.Palette("copper")
    hmap = hv.HoloMap(panels,
                      kdims=grouped_stats.kdims)
    hmap.opts(hv.opts.Scatter(size=4, toolbar="above", backend="bokeh"),
              hv.opts.Curve(color=color), hv.opts.Scatter(color=color));
    
    # Larger, interactive, single-panel holomap: users can select a particular data set
    hmap.opts(hv.opts.Overlay(show_legend=True, legend_position="right")) \
        .opts(height=300, width=700, backend="bokeh")
    hmap
    ####
    

    return hmap


# %%
def flatten_panels(fig: "holoviews.HoloMap", clone: bool=True) -> "holoviews.Layout":
    """
    Take a HoloMap returned by `make_fig_Bconf_mc`, and flatten it so all
    panels can be viewed simultaneously.
    This is useful for getting an overview, and producing a static figure for
    a report.
    Specifically, the following style changes are applied:
    - Panel size is reduced.
    - Panels are arranged on an n x 3 grid (3 columns).
    - Legends are remove, except for the upper right panel.
    - (Bokeh) Total figure width is 1030 px (300px for figs, ~130px for legend)
    """
    pass
    ####

    # Panels for all data sets laid out in a grid
    if clone:
        # First clone so we don't break the styling of the single panel figure
        #     NB: Simply doing fig.clone() doesn't work: it doesn't clone the contained Overlay instances
        fig = hv.HoloMap({key: ov.clone() for key, ov in fig.items()},
                          kdims=fig.kdims,
                          group=fig.group, label=fig.label)
    
    # Create layout
    layout = fig.layout().cols(3)
    
    # Apply styling for smaller plots
    layout.opts(hv.opts.Overlay(height=200, width=300, backend="bokeh"))
        
    legend_idx = min(len(layout)-1, 2)
    print(legend_idx)
    for i, panel in enumerate(layout.values()):
        if i != legend_idx:
            panel.opts(hv.opts.Overlay(show_legend=False))
    panel_with_legend = layout.values()[legend_idx]
    panel_with_legend.opts(legend_position="right").opts(width=430, backend="bokeh")
    layout
    ####

    return layout


# %% [markdown]
# ## Test

# %% [markdown]
# $$\begin{aligned}
# x &\sim \Unif(0, 3) \\
# y &\sim e^{-λx} + ξ
# \end{aligned}$$
#
# Theory model: $λ=1$, $ξ \sim \nN(0, 1)$.  
# True model: $λ=1$, $ξ \sim \nN(-0.03, 1)$.

# %% tags=["active-ipynb"]
# @dataclass
# class DataParamset(ParamColl):
#     L: int
#     λ: float
#     σ: float
#     δy: float
#
# @dataclass
# class ModelParamset(ParamColl):
#     λ: float
#     σ: float
#     #μ: float

# %% tags=["active-ipynb"]
# λ = 1
# δy = -0.03
# L = 400
#
# def true_gen(L, λ, σ, δy, seed=None):
#     rng = default_rng(seed)
#     x = rng.uniform(0, 3, L)
#     y = np.exp(-λ*x) + rng.normal(δy, σ, L)
#     return x, y
# def theory_gen(L, λ, σ, seed=None):
#     rng = default_rng(seed)
#     x = rng.uniform(0, 3, L)
#     y = np.exp(-λ*x) + rng.normal(0, σ, L)
#     return x, y
# def theory_logp(xy, λ, σ):
#     x, y = xy
#     return stats.norm(0, σ).logpdf(y - np.exp(-λ*x))

# %% tags=["active-ipynb"]
# λA = 1
# λB = 1.5
# rngA = np.random.default_rng(1)
# rngB = np.random.default_rng(2)
# # theory_genA = partial(theory_gen, λ=λA, seed=rngA)
# # theory_genB = partial(theory_gen, λ=λB, seed=rngB)
# # logpA = partial(theory_logp, λ=λA)
# # logpB = partial(theory_logp, λ=λB)

# %% tags=["active-ipynb"]
# data_params  = DataParamset( L=400,
#                              λ=1,
#                              σ=1,
#                              δy=expand([-1, -0.3, 0, 0.3, 1])
#                            )
# theoA_params = ModelParamset(λ=1,
#                              σ=1
#                             )
# theoB_params = ModelParamset(λ=expand(np.logspace(-1, 1, 40)),
#                              #λ=expand(np.logspace(-1, 1, 3)),
#                              σ=1
#                             )

# %% [markdown]
# c_list = [0.01, 0.05, 0.1, 0.5, 0.8, 1, 1.3, 2, 5, 10, 20]

# %% tags=["active-ipynb"]
# calib_data = CalibrationStats(
#     data_param_names =   [hv.Dimension("L", label="data set size (L)"),
#                           hv.Dimension("λ_data", label="λ (data)"),
#                           hv.Dimension("σ_data", label="σ (data)"),
#                           hv.Dimension("δy_data", label="δy (data)")],
#     modelA_param_names = [hv.Dimension("λA", label="λ (A)"),
#                           hv.Dimension("σA", label="σ (A)")],
#     modelB_param_names = [hv.Dimension("λB", label="λ (B)"),
#                           hv.Dimension("σB", label="σ (B)")]
# )

# %% [markdown]
# EMD experiment

# %% [markdown]
# ```python
# data_params_iter = tqdm(data_params.outer(), desc="data params", total=data_params.outer_len)
# theo_params_progbar = tqdm(desc="theo params",
#                            total=theoA_params.outer_len*theoB_params.outer_len)
# c_progbar = tqdm(desc="c values", total=len(c_list))
# progbarA = tqdm(desc="sampling quantile fns (A)")
# progbarB = tqdm(desc="sampling quantile fns (B)")
#
# bconf_emd_list = []
#
# for Θdata in data_params_iter:
#     observed_data = true_gen(**Θdata, seed=1)
#     theo_params_progbar.reset()
#     for ΘtheoA, ΘtheoB in product(theoA_params.outer(), theoB_params.outer()):
#         logpA = partial(theory_logp, **ΘtheoA)
#         logpB = partial(theory_logp, **ΘtheoB)
#         Θkey = (*Θdata.values(), *ΘtheoA.values(), *ΘtheoB.values())
#         # Until here, same loop as for computing Monte Carlo estimates
#         c_progbar.reset()
#         for c in c_list:
#             bconf_emd = Bconf(
#                 observed_data, logpA, logpB,
#                 model_samplesA=theory_gen(4000, **ΘtheoA, seed=2),
#                 model_samplesB=theory_gen(4000, **ΘtheoB, seed=3),
#                 c=c,
#                 progbarA=progbarA, progbarB=progbarB
#             )
#             calib_data.add_emd_result((*Θdata.values(), *ΘtheoA.values(), *ΘtheoB.values()), c, bconf_emd)
#         
#             c_progbar.update(1)
#
#         theo_params_progbar.update(1)
# ```

# %% [markdown]
# EMD experiment (multiprocessing)

# %% tags=["active-ipynb"]
# def remove_seed(params):
#     p = params.copy()
#     p.pop("seed", None)
#     return p

# %% tags=["active-ipynb", "skip-execution"]
# def generator_of_paramsdata_paramsA_paramsB():
#     for Θdata in data_params.outer():
#         for ΘtheoA, ΘtheoB in product(theoA_params.outer(), theoB_params.outer()):
#             yield {**Θdata, 'seed':1}, {**ΘtheoA, 'seed': 2}, {**ΘtheoB, 'seed': 3}
#
# theory_genA = theory_genB = theory_gen
# theory_logpA = theory_logpB = theory_logp
# theory_L = 4000
# ncores = 12
# total = data_params.outer_len * theoA_params.outer_len * theoB_params.outer_len  # `None` is valid but won't show the total in the progress bar
#
# logging.getLogger("emdd.emd").setLevel(logging.ERROR)
#
# """
# Args:
#     total (int): Total number of data-parameter combinations (i.e. length of
#        `generator_of_paramsdata_paramsA_paramsB`). Used to display in the progress bar.
#
# .. Important:: It is assumed that `true_gen` is deterministic: it is called
#    in multiple child processes and expected to return the same dataset.
#    This is best achieved by fixing its randomness with a seed, and including
#    that seed in its parameters.
#    
#    For best reproducibility, `theory_genA` and `theory_genB` should probably
#    also be deterministic, although this is less critical.
# """
#
# ###
#
# def worker(args):
#     """
#     Args:
#         (Θdata, ΘtheoA, ΘtheoB), c  :  Where
#             Θdata : mapping of arguments for `true_gen`
#             ΘtheoA: mapping of arguments for `theory_genA` and `theory_logpA`
#             ΘtheoB: mapping of arguments for `theory_genB` and `theory_logpB`
#             c     : float
#     """
#     (Θdata, ΘtheoA, ΘtheoB), c = args
#
#     observed_data = true_gen(**Θdata)
#     samplesA = theory_gen(theory_L, **ΘtheoA)
#     samplesB = theory_gen(theory_L, **ΘtheoB)
#     logpA = partial(theory_logp, **remove_seed(ΘtheoA))
#     logpB = partial(theory_logp, **remove_seed(ΘtheoB))
#
#     return Bconf(
#         observed_data, logpA, logpB,
#         model_samplesA=samplesA,
#         model_samplesB=samplesB,
#         c=c,
#         progbarA=None, progbarB=None, use_multiprocessing=False
#     )
#
# progbar = tqdm(desc="Data-parameter combinations", total=total*len(c_list))
#
# if ncores is None:
#     ncores = os.cpu_count()
# arg_it1, arg_it2 = tee(generator_of_paramsdata_paramsA_paramsB())
# with mp.Pool(ncores) as pool:
#     # Chunk size calculated following Pool's algorithm (See https://stackoverflow.com/questions/53751050/multiprocessing-understanding-logic-behind-chunksize/54813527#54813527)
#     # (Naive approach would be total/ncores. This is most efficient if all taskels take the same time. Smaller chunks: more flexible job allocation, but more overhead)
#     chunksize, extra = divmod(total, ncores*6)
#     if extra:
#         chunksize += 1
#     Bconf_it = pool.imap(worker, product(arg_it1, c_list), chunksize=chunksize)
#     for (args, c), _Bconf in zip(product(arg_it2, c_list), Bconf_it):
#         Θdata, ΘtheoA, ΘtheoB = args
#         calib_data.add_emd_result((*noseed(Θdata).values(), *noseed(ΘtheoA).values(), *noseed(ΘtheoB).values()), c, _Bconf)
#         progbar.update(1)

# %% [markdown]
# Monte Carlo experiment

# %% tags=["active-ipynb", "skip-execution"]
# N = 1000
#
# data_params_iter = tqdm(data_params.outer(), desc="data params", total=data_params.outer_len)
# theo_params_progbar = tqdm(desc="theo params",
#                            total=theoA_params.outer_len*theoB_params.outer_len)
#
# for Θdata in data_params_iter:
#     observed_data = true_gen(**Θdata)
#     theo_params_progbar.reset()
#     for ΘtheoA, ΘtheoB in product(theoA_params.outer(), theoB_params.outer()):
#         logpA = partial(theory_logp, **ΘtheoA)
#         logpB = partial(theory_logp, **ΘtheoB)
#         Θkey = (*Θdata.values(), *ΘtheoA.values(), *ΘtheoB.values())
#         n, counts_A = calib_data.get_counts(Θkey)
#         for _ in range(N-n):
#             sim_data = true_gen(**Θdata)
#             calib_data.add_mc_result(Θkey, logpA(sim_data).mean(), logpB(sim_data).mean())
#         theo_params_progbar.update(1)

# %% [markdown]
# Figures

# %% tags=["active-ipynb", "skip-execution"]
# # The set of unique parameters (TODO: move into CalibData)
# tab = calib_data.statshv_mc()
# cols = (tab[dim] for dim in calib_data.model_param_dims)
# Ω = set(zip(*cols))
#
# # TODO: Make this a CalibData property
# pruned_model_param_dims = [dim for dim in calib_data.model_param_dims if len(set(calib_data.statshv_mc()[dim])) > 1]

# %% tags=["skip-execution", "active-ipynb"]
# # Use the bconf dimension for both MC and EMD
# metric_dim = hv.Dimension("Bconf", label="B_conf")
#
# mc_fig = make_fig_Bconf_mc(calib_data).redim(Bconf_mc=metric_dim)
# emd_fig = make_fig_Bconf_emd(calib_data).redim(Bconf_emd=metric_dim)
#
# chance_level = hv.Curve(zip(range(len(Ω)), repeat(0.5)), kdims=[σparam_dim], vdims=[metric_dim], label="chance level")
# chance_level.opts(color="#888888").opts(line_dash="dashed", line_width=2, backend="bokeh")
#
# fig = chance_level * mc_fig * emd_fig
#
# # Add a hover tool to display the parameter values
# if page_backend == "bokeh":
#     from bokeh.models import HoverTool
#     hover = HoverTool(
#         tooltips=[(calib_data.Bconf_mc_dim.label, f"@{calib_data.Bconf_mc_dim.name}"),
#                   (calib_data.Bconf_emd_dim.label, f"@{calib_data.Bconf_emd_dim.name}")]
#                  + [(dim.label, "@{"+dim.name+"}") for dim in pruned_model_param_dims])
#     fig.opts(tools=[hover])
#
# # Workaround: For some reason, after applying the overlay the plot sizes are lost
# fig.opts(hv.opts.Overlay(show_legend=True, legend_position="right")) \
#    .opts(height=300, width=700, backend="bokeh")
# fig

# %% [markdown]
# ---

# %% [markdown]
#     counts = CalibrationStats(
#         data_param_names =   [hv.Dimension("L", label="data set size (L)"),
#                               hv.Dimension("λ_data", label="λ (data)"),
#                               hv.Dimension("σ_data", label="σ (data)"),
#                               hv.Dimension("δy_data", label="δy (data)")],
#         modelA_param_names = [hv.Dimension("λA", label="λ (A)"),
#                               hv.Dimension("σA", label="σ (A)")],
#         modelB_param_names = [hv.Dimension("λB", label="λ (B)"),
#                               hv.Dimension("σB", label="σ (B)")]
#     )

# %% [markdown]
#     data_params  = DataParamset( L=400,
#                                  λ=1,
#                                  σ=1,
#                                  δy=expand([-1, -0.3])#, 0, 0.3, 1]
#                                )
#     theoA_params = ModelParamset(λ=1,
#                                  σ=1
#                                 )
#     theoB_params = ModelParamset(#λ=np.logspace(-1, 1, 40),
#                                  λ=expand(np.logspace(-1, 1, 3)),
#                                  σ=1
#                                 )

# %% tags=["skip-execution", "active-ipynb"]
# N = 1000
#
# data_params_iter = tqdm(data_params.outer(), desc="data params", total=data_params.outer_len)
# theo_params_progbar = tqdm(desc="theo params",
#                            total=theoA_params.outer_len*theoB_params.outer_len)
#
# for Θdata in data_params_iter:
#     observed_data = true_gen(**Θdata)
#     theo_params_progbar.reset()
#     for ΘtheoA, ΘtheoB in product(theoA_params.outer(), theoB_params.outer()):
#         logpA = partial(theory_logp, **ΘtheoA)
#         logpB = partial(theory_logp, **ΘtheoB)
#         Θkey = (*Θdata.values(), *ΘtheoA.values(), *ΘtheoB.values())
#         n, counts_A = calib_data.get_counts(Θkey)
#         for _ in range(N-n):
#             sim_data = true_gen(**Θdata)
#             # _lA, _lB = logpA(sim_data).mean(), logpB(sim_data).mean()
#             # calib_data.add_result(Θkey, 0.5 if _lA == _lB else _lA > _lB)
#             calib_data.add_mc_result(Θkey, logpA(sim_data).mean(), logpB(sim_data).mean())
#             # calib_data.add_result(Θkey, _lA := logpA(sim_data).mean() > _lB := logpB(sim_data).mean()
#             #                         if _lA != _lA else 0.5)
#         theo_params_progbar.update(1)

# %% [markdown]
#     counts.mc_stats()
#     counts.mc_statsdf()
#     counts.statshv()

# %% tags=["skip-execution", "active-ipynb"]
# bconf_mc = make_fig_bconf_mc(counts)
# bconf_mc_grid = flatten_bconf_panels(bconf_mc)
# display(bconf_mc_grid)
# display(bconf_mc)

# %% tags=["skip-execution", "active-ipynb"]
# print(bconf_mc_grid.layout().cols(3))

# %% tags=["skip-execution", "active-ipynb"]
# c_list = [0.5, 0.8, 1, 1.3]

# %% tags=["skip-execution", "active-ipynb"]
# c_dim = hv.Dimension("c", label="EMD prop. constant")
# bconf_dim = hv.Dimension("bconf", label="b_conf")

# %% tags=["skip-execution", "active-ipynb"]
# data_params_iter = tqdm(data_params.outer(), desc="data params", total=data_params.outer_len)
# theo_params_progbar = tqdm(desc="theo params",
#                            total=theoA_params.outer_len*theoB_params.outer_len)
# c_progbar = tqdm(desc="c values", total=len(c_list))
# progbarA = tqdm(desc="sampling quantile fns (A)")
# progbarB = tqdm(desc="sampling quantile fns (B)")
#
# bconf_emd_list = []
#
# for Θdata in data_params_iter:
#     observed_data = true_gen(**Θdata, seed=1)
#     theo_params_progbar.reset()
#     for ΘtheoA, ΘtheoB in product(theoA_params.outer(), islice(theoB_params.outer(1, 2))):
#         logpA = partial(theory_logp, **ΘtheoA)
#         logpB = partial(theory_logp, **ΘtheoB)
#         Θkey = (*Θdata.values(), *ΘtheoA.values(), *ΘtheoB.values())
#         # Until here, same loop as for computing Monte Carlo estimates
#         c_progbar.reset()
#         for c in c_list:
#             bconf_emd = Bconf(
#                 observed_data, logpA, logpB,
#                 model_samplesA=theory_gen(4000, **ΘtheoA, seed=2),
#                 model_samplesB=theory_gen(4000, **ΘtheoB, seed=3),
#                 c=c,
#                 progbarA=progbarA, progbarB=progbarB
#             )
#             bconf_emd_list.append((*Θdata.values(), *ΘtheoA.values(), *ΘtheoB.values(), c, bconf_emd))
#         
#             c_progbar.update(1)
#
#         theo_params_progbar.update(1)

# %% tags=["skip-execution", "active-ipynb"]
# emd_vals = hv.Table(bconf_emd_list, kdims=counts.param_dims+[c_dim], vdims=[bconf_dim])

# %%

# %% [markdown]
# ---

# %% tags=["remove-input"]
from emdd.utils import GitSHA
GitSHA()
