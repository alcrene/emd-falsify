# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light,md:myst
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python (EMD-paper)
#     language: python
#     name: emd-paper
# ---

# # EMD implementation
#
# $\renewcommand{\RR}{\mathbb{R}}
# \renewcommand{\nN}{\mathcal N}
# \renewcommand{\D}[1][]{\mathcal{D}^{#1}}
# \renewcommand{\l}[1][]{l_{#1}}
# \renewcommand{\Me}[1][]{\mathcal{M}^ε_{#1}}
# \renewcommand{\Unif}{\mathop{\mathrm{Unif}}}
# \renewcommand{\Philt}[2][]{\widetilde{Φ}_{#1|#2}}
# \renewcommand{\Elmu}[2][1]{μ_{{#2}}^{(#1)}}
# \renewcommand{\Elsig}[2][1]{Σ_{{#2}}^{(#1)}}
# \renewcommand{\Bconf}[1][]{B_{\mathrm{conf}#1}}
# $

# + tags=["hide-input"]
import logging
import multiprocessing as mp
from collections.abc import Callable, Mapping
from math import sqrt, ceil, isclose
from itertools import product  # Only used in calibration_plot
from functools import partial
from more_itertools import all_equal
import numpy as np
from numpy.random import default_rng
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from scipy.special import erf
from tqdm.auto import tqdm

from typing import Optional, Union, Any, Literal, Tuple, List, Dict, NamedTuple
from scityping.numpy import Array

from emd_paper import Config
from emd_paper.path_sampling import generate_quantile_paths

config = Config()
logger = logging.getLogger(__name__)

# + tags=["remove-cell"]
try:
    import holoviews as hv
except ModuleNotFoundError:
    # One could want to use this package without Holoviews: it is only required by the function `calibration_plot`
    class HoloviewsNotFound:
        def __getattr__(self, attr):
            raise ModuleNotFoundError("Unable to import Holoviews; perhaps it is not installed ?")
    hv = HoloviewsNotFound()

# + [markdown] tags=["remove-cell"]
# Notebook only imports

# + tags=["active-ipynb", "remove-cell"]
# from scipy import stats
# hv.extension(config.figures.backend)
# logging.basicConfig(level=logging.WARNING)
# logger.setLevel(logging.ERROR)

# + tags=["active-ipynb", "remove-cell", "skip-execution"]
# logger.setLevel(logging.DEBUG)
# -

from emd_paper.memoize import memoize

# + [markdown] tags=["remove-cell"]
#     Set up caching according to configuration. Since the `lru_cache` cache only accepts hashable arguments, we wrap it with `nofail_functools_cache` which skips the caching (but at least doesn't fail) on non-hashable args.
#
#     For the moment we don't offer similar similar functionality for the on-disk joblib cache, for two reasons:
#     - It is opt-in: since the user must explicitly request it, ostensibly they can be expected to use valid arguments.
#     - Any pickleable argument is supported, with special support for NumPy arrays, so most typical values should be supported.

# + [markdown] tags=["remove-cell"]
#     if config.caching.use_disk_cache:
#         from joblib import Memory
#         from emd_paper.utils import nofail_joblib_cache
#         memory = Memory(**config.caching.joblib.dict(exclude={"rootdir"}))
#         def cache(func=None, **kwargs):
#             "Combine @nofail_joblib_cache and @memory.cache into one decorator"
#             if func is None:
#                 # Support adding arguments on the decorator line: @cache(warn=False)
#                 return partial(cache, **kwargs)
#
#             warn = kwargs.pop("warn", None)
#             nofail_kws = {} if warn is None else {"warn": warn}
#             return nofail_joblib_cache(**nofail_kws)(memory.cache(func, **kwargs))
#
#     else:
#         from functools import lru_cache
#         from emd_paper.utils import nofail_functools_cache
#         def cache(func=None, **kwargs):
#             "Combine @nofail_functools_cache and @lru_cache into one decorator"
#             warn = kwargs.pop("warn", None)
#             nofail_kws = {} if warn is None else {"warn": warn}
#             if func is None:
#                 # Gobble up kwargs – they are meant for joblib
#                 assert "@cache only accepts keyword arguments, or no arguments at all"
#                 return partial(cache, **nofail_kws)
#             else:
#                 assert (len(kwargs) == 0 and isinstance(func, Callable)), \
#                         "@cache only accepts keyword arguments, or no arguments at all"
#                 return nofail_functools_cache(**nofail_kws)(lru_cache(func))
# -

# Default plot configuration
#
# Methods to change plot style:
# - A user can modify the defaults below by modifying the `*_opts` lists in place.  
#   This is a bit hacky but should be safe.
# - Different options can simply be applied to the returned object.

# +
colors = config.figures.matplotlib.colors["medium-contrast"]

# Option lists used in calibration_plot
try:
    true_opts = [hv.opts.Curve(color="#666666", logx=False),
                 hv.opts.Area(alpha=1, logx=False),
                 hv.opts.Area(edgecolor="none", facecolor="#cfcfcf", backend="matplotlib")]
                 # cfcfcf is the greyscale equivalent of "light yellow"

    high_opts = [hv.opts.Curve(color=colors["dark blue"], logx=False),
                 hv.opts.Area(alpha=0.4, logx=False),
                 hv.opts.Area(facecolor=colors["light blue"], edgecolor="none", backend="matplotlib")]

    low_opts  = [hv.opts.Curve(color=colors["dark red"], logx=False),
                 hv.opts.Area(alpha=0.4, logx=False),
                 hv.opts.Area(facecolor=colors["light red"], edgecolor="none", backend="matplotlib")]
except ModuleNotFoundError:
    pass


# -

# ### Path sampler test
#
# $$\begin{aligned}
# \tilde{\l} &= \log Φ\,, & Φ &\in [0, 1]\\
# \tilde{σ} &= c \sin π Φ \,, & c &\in \mathbb{R}_+
# \end{aligned}$$
#
# The upper part of the yellow region is never sampled, because monotonicity prevents paths from exceeding $\log 1$ at any point. The constant $c$ is determined by a calibration experiment, and controls the variability of paths. Here we use $c=1$.

# + tags=["active-ipynb", "hide-input"]
# res = 7
# Φarr = np.arange(1, 2**res) / 2**res
# ltilde = np.log(Φarr)
# σtilde = np.sin(Φarr * np.pi)
# R = 20
#
# lcurve = hv.Curve(zip(Φarr, ltilde), kdims=["Φ"], vdims=["l"], label=r"$\langle\tilde{l}\rangle$")
# σarea = hv.Area((Φarr, ltilde - σtilde, ltilde + σtilde),
#                 kdims=["Φ"], vdims=["l-σ", "l+σ"])
# GP_fig = σarea.opts(edgecolor="none", facecolor="#EEEEBB", backend="matplotlib") * lcurve
#
# lhat_gen = generate_quantile_paths(R, Φarr, ltilde, σtilde, res=res)
# random_colors = default_rng().uniform(0.65, 0.85, R).reshape(-1,1) * np.ones(3)  # Random light grey for each curve
# lhat_curves = [hv.Curve(zip(Φhat, lhat), kdims=["Φ"], vdims=["l"], label=r"$\hat{l}$")
#                .opts(color=color)               
#                for (Φhat, lhat), color in zip(lhat_gen, random_colors)]
#
# Φ_fig = hv.Overlay(lhat_curves)
#
# GP_fig * Φ_fig
# -

# Similar test, but we allow variability in the end point. Note that now samples samples can cover all of the yellow region.
#
# $$\begin{aligned}
# \tilde{\l} &= \log Φ\,, & Φ &\in [0, 1]\\
# \tilde{σ} &= c \sin \frac{3 π Φ}{4} \,,  & c &\in \mathbb{R}_+ \,.
# \end{aligned}$$

# + tags=["active-ipynb", "hide-input"]
# res = 7
# Φarr = np.arange(1, 2**res) / 2**res
# ltilde = np.log(Φarr)
# σtilde = np.sin(Φarr * 0.75*np.pi)
# R = 20
#
# lcurve = hv.Curve(zip(Φarr, ltilde), kdims=["Φ"], vdims=["l"], label=r"$\langle\tilde{l}\rangle$")
# σarea = hv.Area((Φarr, ltilde - σtilde, ltilde + σtilde),
#                 kdims=["Φ"], vdims=["l-σ", "l+σ"])
# GP_fig = σarea.opts(edgecolor="none", facecolor="#EEEEBB", backend="matplotlib") * lcurve
#
# lhat_gen = generate_quantile_paths(R, Φarr, ltilde, σtilde, res=res)
# random_colors = default_rng().uniform(0.65, 0.85, R).reshape(-1,1) * np.ones(3)  # Random light grey for each curve
# lhat_curves = [hv.Curve(zip(Φarr, lhat), kdims=["Φ"], vdims=["l"], label=r"$\hat{l}$")
#                .opts(color=color)               
#                for (Φhat, lhat), color in zip(lhat_gen, random_colors)]
#
# Φ_fig = hv.Overlay(lhat_curves)
#
# GP_fig * Φ_fig
# -

# ## Statistics of the first moment
#
# **Given**
#
# - $\D[L]$: A data set of $L$ samples.
# - $\l[A;\Me]: X \times Y \to \RR$: Function returning the log likelihood of a sample.
# - $c$: Proportionality constant between the EMD and the metric variance when sampling increments.
# - $r$: Resolution of the $\Philt{\Me,A}$ paths. Paths will discretized into $2^r$ steps. (And therefore contain $2^r + 1$ points.)
# - $R$: The number of paths over which to average.
#
# **Return**
#
# - $\Elmu{A}$
# - $\Elsig{A}$
#
# See {prf:ref}`alg-estimator-statistics`.

# :::{margin}  
# The rule for computing `new_R` comes from the following ($ε$: `stderr`, $ε_t$: `stderr_tol`, $R'$: `new_R`)
# $$\begin{aligned}
# ε &= σ/\sqrt{R} \\
# ε' &= σ/\sqrt{R'} \\
# \frac{ε^2}{ε'^2} &= \frac{R'}{R}
# \end{aligned}$$
# :::

def compute_stats_m1(data: "array_like", logp: Callable,
                     model_samples: "array_like"=None,
                     *, model_ppf: Callable=None,
                     c: float=None,
                     res: int=7, N: int=100, R: int=30, max_R: int=1000,
                     relstderr_tol: float=4e-3,
                     path_progbar: Union[Literal["auto"],None,tqdm,mp.queues.Queue]=None,
                    ) -> Tuple[float, float]:
    """
        
    .. Note:: When using multiprocessing to call this function multiple times,
       use either a `multiprocessing.Queue` or `None` for the `progbar` argument.
    
    Parameters
    ----------
    data: The observed data.
    logp: Log likelihood function, as given by the model we want to characterize.
       Must support the vectorized operation ``logp(data)``.
    model_samples: A data set drawn from the model we want to characterize.
       Must have the same format as `data`, but need not have the same number of samples.
       This must be equivalent to samples drawn from the distribution specified by `logp`.
       (So in theory a Monte Carlo sampler for `logp` could be used to generate them.)
       This is used to estimate the model's quantile function (aka PPF).
       Exactly one of `model_samples` and `model_ppf` must be specified.
    model_ppf: Instead of specifying a model sampler, if a closed form for the
       PPF (point probability function, aka quantile function) is known,
       that can be given directly as a callable.
       It must support vectorized operations on NumPy arrays.
       Exactly one of `model_samples` and `model_ppf` must be specified.
    c: (Required; keyword only) Proportionality constant between EMD and path sampling variance.
    res: Controls the resolution of the random quantile paths generated to compute statistics.
       Paths have length ``2**res + 1``; typical values of `res` are 6, 7 and 8, corresponding
       to paths of length 64, 128 and 256. Smaller may be useful to accelerate debugging,
       but larger values are unlikely to be useful.
    R: The minimum number of paths over which to average.
       Actual number may be more, to achieve the specified standard error.
    max_R: The maximum number of paths over which to average.
       This serves to prevent runaway computation in case the specified
       standard error is too low.
    relstderr_tol: The maximum relative standard error on the moments we want to allow.
       (i.e. ``stderr / |μ1|``). If this is exceeded after taking `R` path samples,
       the number of path samples is increased until we are under tolerance, or we have
       drawn 1000 samples. A warning is displayed if 1000 paths does not achieve tolerance.
    progbar: Control whether to create progress bar or use an existing one.
       - With the default value 'auto', a new tqdm progress is created.
         This is convenient, however it can lead to many bars being created &
         destroyed if this function is called within a loop.
       - To prevent this, a tqdm progress bar can be created externally (e.g. with
         ``tqdm(desc="Generating paths")``) and passed as argument.
         Its counter will be reset to zero, and its set total to `R` + `previous_R`.
       - (Multiprocessing): To support updating progress bars within child processes,
         a `multiprocessing.Queue` object can be passed, in which case no
         progress bar is created or updated. Instead, each time a quantile path
         is sampled, a value is added to the queue with ``put``. This way, the
         parent process can update a progress by consuming the queue; e.g.
         ``while not q.empty(): progbar.update()``.
         The value added to the queue is `R`+`previous_R`, which can be
         used to update the total value of the progress bar.
       - A value of `None` prevents displaying any progress bar.
       
    Returns
    -------
    μ1
    Σ1
    
    Todo
    ----
    - Support a `res=None` argument, which automatically selects an appropriate resolution
      based on the number of data points. Make this the default.
    """
    
    if (model_samples is None) + (model_ppf is None) != 1:
        raise TypeError("Exactly one of `model_samples` and `model_cdf` must be specified.")
    if c is None:
        raise TypeError("`c` argument to `compute_stats_m1` is required.")

    # Get the log likelihood CDF for observed samples
    l = np.sort(logp(data))
    L = len(l)
    Φarr = np.arange(1, L+1) / (L+1)  # Exclude 0 and 1 from Φarr, since we have only finite samples

    # Get the log likelihood CDF for the model
    if model_ppf:
        ltilde = np.log(model_ppf(Φarr))
    else:
        _ltilde = np.sort(logp(model_samples))
        # NB: _ltilde is always a 1d array, even though model_samples could be (x,y), or (x, (y1, y2)), or (x, y)^T
        if len(_ltilde) == L:
            ltilde = _ltilde
        else:
            # Align the model CDF to the data CDF using linear interpolation
            _L = len(_ltilde)
            _Φarr = np.arange(1, _L+1) / (_L+1)
            ltilde = interp1d(_Φarr, _ltilde)(Φarr)

    # Compute EMD and σtilde
    emd = abs(l - ltilde)
    σtilde = c * emd

    # Compute m1 for enough sample paths to reach relstderr_tol
    m1 = []
    def extend_m(R, previous_R=0, Φarr=Φarr, ltilde=ltilde, σtilde=σtilde):
        for Φhat, lhat in generate_quantile_paths(R, Φarr, ltilde, σtilde, res, progbar=path_progbar, previous_R=previous_R):
            m1.append(simpson(lhat, Φhat))  # Generated paths always have an odd number of steps, which is good for Simpson's rule

    extend_m(R)
    μ1 = np.mean(m1)
    Σ1 = np.var(m1)
    relstderr = sqrt(Σ1) / max(abs(μ1), 1e-8) / sqrt(R)  # TODO?: Allow setting abs tol instead of hardcoding 1e-8 ?
    while relstderr > relstderr_tol and R < max_R:
        # With small R, we don’t want to put too much trust in the
        # initial estimate of relstderr. So we cap increases to doubling R.
        new_R = min(ceil( (relstderr/relstderr_tol)**2 * R ), 2*R)
        logger.debug(f"Increased number of sampled paths (R) to {new_R}. "
                     f"Previous rel std err: {relstderr}")
        if new_R > max_R:
            new_R = max_R
            logger.warning(f"Capped the number of sample paths to {max_R} "
                           "to avoid undue computation time.")
        elif new_R == R:
            # Can happen due to rounding
            break
        extend_m(new_R - R, R)
        R = new_R
        μ1 = np.mean(m1)
        Σ1 = np.var(m1)
        relstderr = sqrt(Σ1) / max(abs(μ1), 1e-8) / sqrt(R)  # TODO?: Allow setting abs tol instead of hardcoding 1e-8 ?
        
    if relstderr > relstderr_tol:
        logger.warning("Requested std err tolerance was not achieved. "
                       f"std err: {relstderr}\nRequested max std err: {relstderr_tol}")
    return μ1, Σ1


# ### Test compute stats
#
# $$\begin{aligned}
# x &\sim \Unif(0, 3) \\
# y &\sim e^{-λx} + ξ
# \end{aligned}$$
#
# Theory model: $λ=1$, $ξ \sim \nN(0, 1)$.  
# True model: $λ=1$, $ξ \sim \nN(-0.03, 1)$.

# + tags=["active-ipynb"]
# λ = 1
# δy = -0.03
# L = 400
#
# def true_gen(L, seed=None):
#     rng = default_rng(seed)
#     x = rng.uniform(0, 3, L)
#     y = np.exp(-λ*x) + rng.normal(-0.03, 1, L)
#     return x, y
# def theory_gen(L, seed=None):
#     rng = default_rng(seed)
#     x = rng.uniform(0, 3, L)
#     y = np.exp(-λ*x) + rng.normal(0, 1, L)
#     return x, y
# def theory_logp(xy):
#     x, y = xy
#     return stats.norm(0, 1).logpdf(y - np.exp(-λ*x))
#
# data = true_gen(L)

# + tags=["active-ipynb"]
# compute_stats_m1(data, theory_logp, theory_gen(1000),
#                  c=1, N=50, R=100)
# -

# ## $\Bconf$
#
# **Given**
#
# - $\D[L]$: A data set of $L$ samples.
# - $\l[A;\Me]: X \times Y \to \RR$: Function returning the log likelihood of a sample given model A.
# - $\l[B;\Me]: X \times Y \to \RR$: Function returning the log likelihood of a sample given model B.
# - $c$: Proportionality constant between the EMD and the metric variance when sampling increments.
# - $r$: Resolution of the $\Philt{\Me,A}$ paths. Paths will discretized into $2^r$ steps. (And therefore contain $2^r + 1$ points.)
# - $R$: The number of paths over which to average.
#
# **Return**
#
# - $\Bconf\bigl(\Elmu{A}, \Elsig{A}, \Elmu{B}, \Elsig{B}\bigr)$, where the moments are given by `compute_stats_m1`.

# +
def mp_wrapper(f: Callable, *args, out: "mp.queues.Queue", **kwargs):
    "Wrap a function by putting its return value in a Queue. Used for multiprocessing."
    out.put(f(*args, **kwargs))
    
CallableLike = Union[Callable, Tuple[Callable, Mapping]]


# -

@memoize(ignore=["progbarA", "progbarB"])
def Bconf(data: "array_like",
          logpA: CallableLike, logpB: CallableLike,
          model_samplesA: "array_like"=None, model_samplesB: "array_like"=None,
          *, model_ppfA: Callable=None, model_ppfB: Callable=None,
          c: float=None,
          res: int=7, N: int=100, R: int=30, relstderr_tol: float=4e-3,
          progbarA: Union[tqdm,Literal['auto'],None]='auto',
          progbarB: Union[tqdm,Literal['auto'],None]='auto',
          use_multiprocessing: bool=True
         ) -> float:
    # NB: Most of this function is just managing mp processes and progress bars
    if isinstance(progbarA, tqdm):
        close_progbarA = False  # Closing a progbar prevents it from being reused
    elif progbarA == 'auto':  # NB: This works because we already excluded tqdm (tqdm types raise AttributeError on ==)
        progbarA = tqdm(desc="sampling quantile fns (A)")
        close_progbarA = True
    if isinstance(progbarB, tqdm):
        close_progbarB = False
    elif progbarB == 'auto':
        progbarB = tqdm(desc="sampling quantile fns (B)")
        close_progbarB = True

    # If logp functions were passed with separate arguments, reconstruct them
    if not isinstance(logpA, Callable):
        logpA, kwds = logpA
        logpA = partial(logpA, **kwds)
    if not isinstance(logpB, Callable):
        logpB, kwds = logpB
        logpB = partial(logpB, **kwds)
    
    if not use_multiprocessing:
        μA, ΣA = compute_stats_m1(
            data, logpA, model_samplesA, model_ppf=model_ppfA,
            c=c, res=res, N=N, R=R, relstderr_tol=relstderr_tol,
            path_progbar=progbarA)
        μB, ΣB = compute_stats_m1(
            data, logpB, model_samplesB, model_ppf=model_ppfA,
            c=c, res=res, N=N, R=R, relstderr_tol=relstderr_tol,
            path_progbar=progbarB)
        
    else:
        progqA = mp.Queue() if progbarA is not None else None  # We manage the progbar ourselves. The Queue is used for receiving
        progqB = mp.Queue() if progbarA is not None else None  # progress updates from the function
        outqA = mp.Queue()   # Function output values are returned via a Queue
        outqB = mp.Queue()
        _compute_stats_m1_A = partial(
            compute_stats_m1,
            data, logpA, model_samplesA, model_ppf=model_ppfA,
            c=c, res=res, N=N, R=R, relstderr_tol=relstderr_tol,
            path_progbar=progqA)
        _compute_stats_m1_B = partial(
            compute_stats_m1,
            data, logpB, model_samplesB, model_ppf=model_ppfB,
            c=c, res=res, N=N, R=R, relstderr_tol=relstderr_tol,
            path_progbar=progqB)
        pA = mp.Process(target=mp_wrapper, args=(_compute_stats_m1_A,),
                        kwargs={'path_progbar': progqA, 'out': outqA})
        pB = mp.Process(target=mp_wrapper, args=(_compute_stats_m1_B,),
                        kwargs={'path_progbar': progqB, 'out': outqB})
        pA.start()
        pB.start()
        progbar_handles = ( ([(progqA, progbarA)] if progbarA is not None else [])
                           +([(progqB, progbarB)] if progbarB is not None else []) )
        if progbar_handles:
            for _, progbar in progbar_handles:
                progbar.reset()  # We could reset the total here, but already reset it below
            while pA.is_alive() or pB.is_alive():
                for (progq, progbar) in progbar_handles:
                    if not progq.empty():
                        n = 0
                        while not progq.empty():  # Flush the progress queue, then update the progress bar.
                            total = progq.get()   # Otherwise the progress bar may not keep up
                            n += 1
                        if total != progbar.total:
                            progq.dynamic_miniters = False  # Dynamic miniters doesn’t work well when we mess around with the total
                            # Reset the max for the progress bar
                            progbar.total = total
                            if "notebook" in str(progbar.__class__.mro()):  # Specific to tqdm_notebook
                                progbar.container.children[1].max = total  
                                progbar.container.children[1].layout.width = None  # Reset width; c.f. progbar.reset()
                        progbar.update(n)

        pA.join()
        pB.join()
        pA.close()
        pB.close()
        # NB: Don't close progress bars unless we created them ourselves
        if close_progbarA: progbarA.close()
        if close_progbarB: progbarB.close()
        μA, ΣA = outqA.get()
        μB, ΣB = outqB.get()
    
    return 0.5 + 0.5 * erf( (μA-μB)/sqrt(2*(ΣA + ΣB)) )


# ## Calibration
#
# :::{NOTE}  
# To make interactive use more responsive, we memoize (cache) the computed statistics for each value of `c`. This despite the fact that calling the function again gives slightly different results, due to the use of random numbers.
#
# We anticipate that for most applications, this would be the preferred trade-off, but if completely reproducible results are desired, then the calibration utility functions should probably not be used.  
# :::

# +
class Stats(NamedTuple):
    mean: float
    std : float

class StatIdx(NamedTuple):
    hightop: int
    highbot: int
    lowtop: int
    lowbot: int

# To avoid passing uncacheable parameters to `get_estimated_stats`,
# progbars can be passed as module variables
dataset_progbar = None
path_progbar = None

# NB: ignore only works with joblib; it is ignored by lru_cache
@memoize(ignore=["progbars"])
def get_estimated_stats(c: float, n_data_sets: int,
                        data_L: int, theory_L: int, theory_logp: Callable,
                        true_gen: Callable, theory_gen: Callable,
                        N: int, R: int, relstderr_tol: float=2e-2,
                        progbars: Optional[Tuple[tqdm,tqdm]]=(None,None)
                       ) -> Tuple[float, float, float]:
    """
    Return the statistics which would be estimated
    Returns (low, mean, high)
    """
    global dataset_progbar, path_progbar
    
    stats = np.zeros((n_data_sets, 2), dtype=float)
    
    _dataset_progbar, _path_progbar = progbars
    if _dataset_progbar is not None:
        dataset_progbar = tqdm(desc="Generating synthetic data sets", leave=False)
    if _path_progbar is not None:
        path_progbar = tqdm(desc="Sampling quantile paths", leave=False)
        
    dataset_progbar.reset(total=n_data_sets)
    path_progbar.reset(total=R)
    
    for i in range(n_data_sets):
        μ1, Σ1 = compute_stats_m1(
            true_gen(data_L, seed=i), theory_logp, theory_gen(theory_L, seed=i),
            c=c, N=N, R=R, relstderr_tol=relstderr_tol, path_progbar=path_progbar)
        stats[i] = μ1, sqrt(Σ1)
        dataset_progbar.update(1)

    return Stats(mean=stats[:,0], std=stats[:,1])

@memoize
def get_true_stats(n_data_sets,
                   L: int, theory_logp: Callable, true_gen: Callable
                   ) -> Tuple[float, float]:
    """
    Estimate the baseline variability of the observed data.
    Generates multiple data sets and returns the (mean, std) of the
    expected likelihood over “true” data sets.
    """
    true_El_samples = np.fromiter(
        (theory_logp(true_gen(L, seed=i)).mean() for i in range(n_data_sets)),
        float, n_data_sets)
    μ, σ = true_El_samples.mean(), true_El_samples.std()
    # NB: stderr on the mean is σ / sqrt(N)
    if n_data_sets < 400:  # Threshold for determining mean determined to within 5% of the std.dev. "tube" width
        logger.warning("You may want to increase the number of data sets to at least 400;\n"
                       "for error 5% of tube width: 400 data sets;\n"
                       "for error 1% of tube width: 10000 data sets.")
    return Stats(μ, σ)


# -

def calibration_plot(c_list: List[float], data_L: int, theory_L: int,
                     theory_logp: Union[Callable, Dict[tuple, Callable]],
                     theory_gen : Union[Callable, Dict[tuple, Callable]],
                     true_gen   : Union[Callable, Dict[tuple, Callable]],
                     n_true_data_sets=400, n_estimated_data_sets=5,
                     sampled_path_length=64, R=30, relstderr_tol=2e-2,
                     *, progbars: Optional[Tuple[tqdm,tqdm,tqdm,tqdm]]=None
                     ) -> hv.HoloMap:
    """
    Run simulated model comparisons with a known true model and different
    EMD proportionality constants `c`, to determine which value is most
    appropriate.
    For best results, other parameters (data distribution, data set size, etc.)
    should be as close as possible to application conditions.
    
    To facilitate interactive use, results are cached, so the function can be
    re-run with a different `c_list`, and only the new values of `c` will be
    computed. Note that this only works if all other function parameters are
    the same.
    
    This is essentially a wrapper around `compute_stats_m1`. See also that function
    for more information on the parameters.
    
    Returns a collection of HoloViews plots (a `HoloMap`). To display the
    result, the calling code should take care of things like selecting a backend.
    HoloMap keys are all the possible combinations from the keys of
    `theory_logp`, `theory_gen` and `true_gen`.
    
    Parameters
    ----------
    c_list: The values of `c` to test.
    data_L  : Size of the observed data set. This is used to estimate true
       statistics from synthetic data sets generated with the *true* model.
       This value should match the size of the actual data against which the
       models were fitted.
    theory_L: Size of data sets to use to estimate the quantile function of
       the *theoretical* model. It is not necessary that this number be the same
       as the dataset; it should be large enough to estimate the quantile
       function accurately.
    theory_logp: Log likelihood function of the *theoretical* model.
    theory_gen : A function which generates datasets from the *theoretical* model.
       It must take one argument, the data set size `L`, and return a tuple,
       `x` and `y` of the independent variable and observations.
    true_gen   : A function which generates datasets from the *true* model.
       It must take one argument, the data set size `L`, and return a tuple,
       `x` and `y` of the independent variable and observations.
    n_true_data_sets: The number of true data sets to generate to estimate
       the *true* statistics. For each data set, this just involves evaluating
       logp – this is normally pretty cheap, so a fairly large number can be used.
       The default (400) should determine the true mean to within 5%
       of relative standard error.
    n_estimated_data_sets: The number of *theoretical* data sets to generate.
       For each, the full bootstrapped EMD analysis (with `R` sampled paths in logp space)
       is performed to estimate the variability of the statistics. This is pretty
       expensive, so a small number (~5) is recommended. We are most interested in
       the variability of intervals, more than the spread.
    sampled_path_length: For each path sampled in the EMD analysis, determines
       the number of stops in Φ.
    R: Number of paths to sample in the EMD analysis.
    relstderr_tol: Tolerance for the EMD analysis (see `compute_stats_m1`).
       This should normally be set to a higher value than would normally be
       used for `compute_stats_m1`: higher tolerance allows to sample more
       data sets, to get a better idea of the spread.
       It also adds noise to the estimate from each data set, but modest
       noise in this case might actually be beneficial, serving to offset
       the small number of data sets.
    progbars: If provided, must be a tuple of three tqdm progress bars:
       ``(tqdm(desc="Iterating over parameter sets"), tqdm(desc="Iterating over c values"),
          tqdm(desc="Generating synthetic data sets"), tqdm(desc="Sampling quantile paths"))``
       This allows the same progress bars to be used in multiple calls to
       `calibration_plot`. If progress bars are not passed, new ones are created
       with ``leave=False``, such that they are removed after completion, modulo
       a few pixels of vertical whitespace. Therefore passing `progbars` is 
       especially useful when there are multiple calls to `calibration_plot`,
       to avoid unsightly accumulation of white space.
       
    Returns
    -------
    HoloMap:
        Each frame consists of a plot of the variability of the likelihood
        due to the finite data set. The EMD estimate is plotted against the
        true spread, for different values of `c`.
        One frame is produced for each combination of `theory_logp`,
        `theory_gen` and `true_gen`.
        
    Hint
    ----
    The hash of a normal Python functions is just its identity. Therefore
    to ensure caching, it is important that the same functions are reused.
    In other words, instead of doing
    
    .. code:: python
       from functools import partial
       from itertools import product
       for Θdata, Θmodel in product(Θdata_list, Θmodel_list):
         calibration_plot(..., theory_gen=partial(theory_gen,Θ=Θmodel))
         
    do
    
    .. code:: python
       theory_gens = {tuple(Θmodel): partial(theory_gen,Θ=Θmodel)}
       for Θdata, Θmodel in product(Θdata_list, Θmodel_list):
         calibration_plot(..., theory_gen=theory_gens[Θmodel])
    
    """
    global dataset_progbar, path_progbar
    # Normalize inputs: all dicts with tuples as keys
    if not isinstance(theory_logp, dict):
        theory_logp = {(): theory_logp}
    else:
        theory_logp = {(k,) if not isinstance(k, tuple) else k: v
                       for k,v in theory_logp.items()}
    if not isinstance(theory_gen, dict):
        theory_gen = {(): theory_gen}
    else:
        theory_gen = {(k,) if not isinstance(k, tuple) else k: v
                      for k,v in theory_gen.items()}
    if not isinstance(true_gen, dict):
        true_gen = {(): true_gen}
    else:
        true_gen = {(k,) if not isinstance(k, tuple) else k: v
                    for k,v in true_gen.items()}
        
    # Construct key dimensions
    k0 = next(iter(theory_logp))
    if not all_equal(len(k) for k in theory_logp):
        raise ValueError("All keys to `theory_logp` must have the same length.")
    elif k0 == ():
        logp_keydims = []  # Only one element => no need for a dimension
    elif len(k0) == 1:
        logp_keydims = ["logp"]
    else:
        logp_keydims = [f"logp{i}" for i in range(len(k0))]
    k0 = next(iter(theory_gen))
    if not all_equal(len(k) for k in theory_gen):
        raise ValueError("All keys to `theory_gen` must have the same length.")
    elif k0 == ():
        theory_keydims = []  # Only one element => no need for a dimension
    elif len(k0) == 1:
        theory_keydims = ["theory"]
    else:
        theory_keydims = [f"theory{i}" for i in range(len(k0))]
    k0 = next(iter(true_gen))
    if not all_equal(len(k) for k in true_gen):
        raise ValueError("All keys to `true_gen` must have the same length.")
    elif k0 == ():
        true_keydims = []  # Only one element => no need for a dimension
    elif len(k0) == 1:
        true_keydims = ["true"]
    else:
        true_keydims = [f"true{i}" for i in range(len(k0))]
        
    keydims = logp_keydims + theory_keydims + true_keydims
        
    # Set up progress bars
    if progbars:
        param_progbar, c_progbar, dataset_progbar, path_progbar = progbars
    else:
        param_progbar   = tqdm(desc="Iterating over parameter sets", leave=False)
        c_progbar       = tqdm(desc="Iterating over c values", leave=False)
        dataset_progbar = tqdm(desc="Theory data set", leave=False)
        path_progbar    = tqdm(desc="Sampling quantile paths", leave=False)
    param_progbar.reset(total=len(theory_logp)*len(theory_gen)*len(true_gen))
    c_progbar.reset(total=len(c_list))
    dataset_progbar.reset(total=n_estimated_data_sets)
    path_progbar.reset(total=R)
    
    def c_generator():
        for c in c_list:
            yield c
            c_progbar.update(1)
        
    # OK let’s go
    frames = {}
    for (logp_key, logp), (theory_key, theory), (true_key, true) \
        in product(theory_logp.items(), theory_gen.items(), true_gen.items()):

        # Compute true and estimated statistics
        true_stats = get_true_stats(n_true_data_sets,
                                    data_L, logp, theory)
        est_stats = [get_estimated_stats(c, n_estimated_data_sets,
                                         data_L, theory_L, logp,
                                         true, theory,
                                         N=sampled_path_length,
                                         R=R, relstderr_tol=relstderr_tol,
                                         progbars=(None, None))  # Pass None to allow caching
                     for c in c_generator()]
        
        # For each value of c, identify which data set generated the highest & lowest logp
        idcs = [ StatIdx((s := es.mean + es.std).argmax(),
                         (d := es.mean - es.std).argmax(),
                         s.argmin(),
                         d.argmin())
                 for es in est_stats ]
        
        # Construct plot elements
        #highmean1 = (es.mean[si.hightop] for es, si in zip(est_stats, idcs))
        #highmean2 = (es.mean[si.highbot] for es, si in zip(est_stats, idcs))
        #lowmean1  = (es.mean[si.lowtop] for es, si in zip(est_stats, idcs))
        #lowmean2  = (es.mean[si.lowbot] for es, si in zip(est_stats, idcs))
        highmean = (es.mean.max() for es, si in zip(est_stats, idcs))
        lowmean = (es.mean.min() for es, si in zip(est_stats, idcs))

        highspreadtop = (es.mean[si.hightop] + es.std[si.hightop] for es, si in zip(est_stats, idcs))
        highspreadbot = (es.mean[si.highbot] - es.std[si.highbot] for es, si in zip(est_stats, idcs))
        lowspreadtop  = (es.mean[si.lowtop]  + es.std[si.lowtop]  for es, si in zip(est_stats, idcs))
        lowspreadbot  = (es.mean[si.lowbot]  - es.std[si.lowbot]  for es, si in zip(est_stats, idcs))

        true_curve = hv.Curve([(c, true_stats.mean) for c in c_list],
                             label="True mean", kdims=["c"], vdims=["l"],
                            ).opts(*true_opts)
        true_area  = hv.Area([(c, true_stats.mean-true_stats.std, true_stats.mean+true_stats.std)
                                    for c in c_list],
                                   label="True std dev", kdims=["c"], vdims=["l", "l_high"],
                                  ).opts(*true_opts)

        highcurve = hv.Curve([(c, m) for c, m in zip(c_list, highmean)],
                             kdims=["c"], vdims=["l"], label="Estimated mean (high)"
                             ).opts(*high_opts)
        #highcurve2 = hv.Curve([(c, m) for c, m in zip(c_list, highmean2)],
        #                      kdims=["c"], vdims=["l"], label="Estimated mean (high)"
        #                      ).opts(*high_opts)
        lowcurve  = hv.Curve([(c, m) for c, m in zip(c_list, lowmean)],
                             kdims=["c"], vdims=["l"], label="Estimated mean (low)"
                             ).opts(*low_opts)
        #lowcurve2  = hv.Curve([(c, m) for c, m in zip(c_list, lowmean2)],
        #                      kdims=["c"], vdims=["l"], label="Estimated mean (low)",
        #                      ).opts(*low_opts)

        higharea   = hv.Area([(c, b, t) for c, b, t in zip(c_list, highspreadbot, highspreadtop)],
                             kdims=["c"], vdims=["l", "l'"], label="Estimated mean (high)") \
                             .opts(*high_opts)
        lowarea    = hv.Area([(c, b, t) for c, b, t in zip(c_list, lowspreadbot, lowspreadtop)],
                             kdims=["c"], vdims=["l", "l'"], label="Estimated mean (low)") \
                             .opts(*low_opts)
        
        # Combine the plot
        frame = (true_area * higharea * lowarea
                 * true_curve * highcurve * lowcurve)
        frame.opts(fig_inches=3.5, aspect=2., backend="matplotlib") \
             .opts(legend_position="bottom_left") \
             .redim.range(l=(true_stats.mean-6*true_stats.std, true_stats.mean+6*true_stats.std))
        if frames:
            # We already have a frame: no need to add another legend
            # (this assumes frames will be placed in a Layout)
            frame.opts(show_legend=False)
        
        # Merge the keys and add to the dict of frames
        key = logp_key + theory_key + true_key
        frames[key] = frame
        
        # Increment the progress bar by 1
        param_progbar.update(1)
        
    # Remove progress bars from module variables
    param_progbar = None
    dataset_progbar = None
    # Finally, package the dictionary into a HoloMap
    return hv.HoloMap(frames, kdims=keydims)

# + tags=["active-ipynb"]
# calibration_plot(
#     c_list=[0.2, 0.5, 1, 1.6, 4],
#     data_L=400, theory_L=400,
#     theory_logp=theory_logp, theory_gen=theory_gen,
#     true_gen=true_gen) \
# .redim.range(l=(-1.6,-1.3)) \
# .opts(legend_position="right") \
# .opts(fig_inches=6, aspect=2, backend="matplotlib")

# + tags=["remove-input", "active-ipynb"]
# from emd_paper.utils import GitSHA
# GitSHA()
#
