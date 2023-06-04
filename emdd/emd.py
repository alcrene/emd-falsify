# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light,md:myst
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python (emd-paper)
#     language: python
#     name: emd-paper
# ---

# (supp_emd-implementation)=
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
# \renewcommand{\Bemd}[1][]{B_{#1}^{\mathrm{EMD}}}
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

from emdd import Config
from emdd.path_sampling import generate_quantile_paths
from emdd.utils import hv  # Holoviews is imported with a guard in case it is not installed

config = Config()
logger = logging.getLogger(__name__)

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

from emdd.memoize import memoize

# + [markdown] tags=["remove-cell"]
#     Set up caching according to configuration. Since the `lru_cache` cache only accepts hashable arguments, we wrap it with `nofail_functools_cache` which skips the caching (but at least doesn't fail) on non-hashable args.
#
#     For the moment we don't offer similar similar functionality for the on-disk joblib cache, for two reasons:
#     - It is opt-in: since the user must explicitly request it, ostensibly they can be expected to use valid arguments.
#     - Any pickleable argument is supported, with special support for NumPy arrays, so most typical values should be supported.

# + [markdown] tags=["remove-cell"]
#     if config.caching.use_disk_cache:
#         from joblib import Memory
#         from emdd.utils import nofail_joblib_cache
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
#         from emdd.utils import nofail_functools_cache
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

# (supp_emd-implementation_example-sampling-paths)=
# ## Path sampler test
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

# + tags=["active-ipynb", "hide-input"] editable=true slideshow={"slide_type": ""}
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

# + editable=true slideshow={"slide_type": ""}
@memoize(ignore=["path_progbar"])
def draw_elppd_samples(data: "array_like", logp: Callable,
                     model_samples: "array_like"=None,
                     *, model_ppf: Callable=None,
                     c: float=None,
                     res: int=7, N: int=100, R: int=30, max_R: int=1000,
                     relstderr_tol: float=4e-3,
                     path_progbar: Union[Literal["auto"],None,tqdm,mp.queues.Queue]=None,
                    ) -> Array[float, 1]:
    """
    Compute statistics of the expected log probability on unseen data with unknown
    observation noise. In other words, we treat ``logp(x)`` as a random variable,
    and want to describe the distribution of ``E[logp(x)]``.
    This computes the first two cumulants (the mean and variance) of that distribution.
        
    .. Note:: When using multiprocessing to call this function multiple times,
       use either a `multiprocessing.Queue` or `None` for the `progbar` argument.
    
    Parameters
    ----------
    data: The observed data.
    logp: Log probability function, as given by the model we want to characterize.
       Must support the vectorized operation ``logp(data)``.
       Function should typically correspond to either the likelihood, or likelihood x priors.
    model_samples: A data set drawn from the model we want to characterize.
       Must have the same format as `data`, but need not have the same number of samples.
       This must be equivalent to samples drawn from the distribution specified by `logp`.
       (So in theory a Monte Carlo sampler for `logp` could be used to generate them.)
       This is used to estimate the model's quantile function (aka PPF).
       Exactly one of `model_samples` and `model_ppf` must be specified.
    model_ppf: Instead of specifying model samples, if a closed form for the
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
    path_progbar: Control whether to create progress bar or use an existing one.
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
    μ: Mean of ``E[logp(x)]``.
    Σ: Variance of ``E[logp(x)]``.
    
    Todo
    ----
    - Support a `res=None` argument, which automatically selects an appropriate resolution
      based on the number of data points. Make this the default.
    """
    
    if (model_samples is None) + (model_ppf is None) != 1:
        raise TypeError("Exactly one of `model_samples` and `model_cdf` must be specified.")
    if c is None:
        raise TypeError("`c` argument to `compute_elppd_stats` is required.")

    # Get the log likelihood CDF for observed samples
    l_empirical = np.sort(logp(data))
    L = len(l_empirical)
    Φarr = np.arange(1, L+1) / (L+1)  # Exclude 0 and 1 from Φarr, since we have only finite samples

    # Get the log likelihood CDF for the model
    if model_ppf:
        l_theory = np.log(model_ppf(Φarr))
    else:
        _l_theory = np.sort(logp(model_samples))
        # NB: _l_theory is always a 1d array, even though model_samples could be (x,y), or (x, (y1, y2)), or (x, y)^T
        if len(_l_theory) == L:
            l_theory = _l_theory
        else:
            # Align the model CDF to the data CDF using linear interpolation
            _L = len(_l_theory)
            _Φarr = np.arange(1, _L+1) / (_L+1)
            l_theory = interp1d(_Φarr, _l_theory, fill_value="extrapolate")(Φarr)

    # Compute EMD and σtilde
    emd = abs(l_empirical - l_theory)
    σtilde = c * emd

    # Compute m1 for enough sample paths to reach relstderr_tol
    m1 = []
    def extend_m(R, previous_R=0, Φarr=Φarr, ltilde=l_empirical, σtilde=σtilde):
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
    return np.array(m1)


# -

draw_Elogp_samples = draw_elppd_samples


# + editable=true slideshow={"slide_type": ""}
def compute_elppd_stats(*args, **kwargs) -> Tuple[float, float]:
    """
    Wrapper around `draw_Elogp_samples` which returns the statistics of the
    samples instead of the samples themselves. Partly provided for compatibility
    with scripts which expect this, but also because this function skips the
    memoization of `draw_Elogp_samples`. This can be useful if we expect it to
    be called many times (as in a calibration experiment), to avoid swamping
    the cache directory.

    .. Note:: This function is NOT cached. The expectation is that it will be
       used as part of a wider loop which may call it many times.
    """
    # Calling the underlying __wrapped__ function avoids caching
    Elogp_samples = draw_Elogp_samples.__wrapped__(*args, **kwargs)
    return Elogp_samples.mean(), Elogp_samples.var()


# + [markdown] editable=true slideshow={"slide_type": ""}
# ### Test computation of first moment statistics
#
# $$\begin{aligned}
# x &\sim \Unif(0, 3) \\
# y &\sim e^{-λx} + ξ
# \end{aligned}$$
#
# Theory model: $λ=1$, $ξ \sim \nN(0, 1)$.  
# True model: $λ=1$, $ξ \sim \nN(-0.03, 1)$.

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# :::{hint}
# :class: margin
#
# Here we use plain custom functions to illustrate that `Elogp_stats` can be used with arbitrary callables, but for our own use we find the provided [FullModel class](./models.py) helpful: it packages physical and observation models into a standardized object.  
# Higher level functions, like those in the [tasks module](./tasks.py), use this to define pipeline which don’t need to know anything about the particular physical and observation model.
# :::

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# λ = 1
# δy = -0.03
# L = 400
#
# def true_gen(L, seed=None):
#     rng = default_rng(seed)
#     x = rng.uniform(0, 3, L)
#     y = np.exp(-λ*x) + rng.normal(-0.03, 1, L)
#     return y, x
# def theory_gen(L, seed=None):
#     rng = default_rng(seed)
#     x = rng.uniform(0, 3, L)
#     y = np.exp(-λ*x) + rng.normal(0, 1, L)
#     return y, x
# def theory_logp(xy):
#     x, y = xy
#     return stats.norm(0, 1).logpdf(y - np.exp(-λ*x))  # z = exp(-λ*x)
#
# data = true_gen(L)

# + tags=["active-ipynb"] editable=true slideshow={"slide_type": ""}
# compute_elppd_stats(data, theory_logp, theory_gen(1000),
#                  c=1, N=50, R=100)

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# (supp_emd-implementation_Bemd)=
# ## Implementation of $\Bemd$
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
# - $\Bemd\bigl(\Elmu{A}, \Elsig{A}, \Elmu{B}, \Elsig{B}\bigr)$, where the moments are given by `compute_elppd_stats`.

# +
def mp_wrapper(f: Callable, *args, out: "mp.queues.Queue", **kwargs):
    "Wrap a function by putting its return value in a Queue. Used for multiprocessing."
    out.put(f(*args, **kwargs))
    
LazyPartial = Union[Callable, Tuple[Callable, Mapping]]


# -

@memoize(ignore=["progbarA", "progbarB", "use_multiprocessing"])
def Bemd(data: "array_like",
         logpA: Union[Callable,LazyPartial], logpB: Union[Callable,LazyPartial],
         model_samplesA: "array_like"=None , model_samplesB: "array_like"=None,
         *, model_ppfA: Callable=None, model_ppfB: Callable=None,
         c: float=None,
         res: int=7, N: int=100, R: int=30, relstderr_tol: float=4e-3,
         progbarA: Union[tqdm,Literal['auto'],None]='auto',
         progbarB: Union[tqdm,Literal['auto'],None]='auto',
         use_multiprocessing: bool=True
        ) -> float:

    """
    Parameters
    ----------
    data: The observed data.
    logpA, logpB: Log probability functions, as given by the model we want to characterize.
       Must support the vectorized operation ``logp(data)``.
       Functions should typically correspond to either the likelihood, or likelihood x priors.
       It is also allowed to pass a tuple ``(logp, kwds)``, composed of the likelihood
       function and keyword parameters. This is then reconstructed with
       ``functools.partial(logp, **kwds)``
    model_samplesA, model_samplesB: Data sets drawn from the models we want to characterize.
       Must have the same format as `data`, but need not have the same number of samples.
       This must be equivalent to samples drawn from the distribution specified by `logp`.
       (So in theory a Monte Carlo sampler for `logp` could be used to generate them.)
       This is used to estimate the model's quantile function (aka PPF).
       Exactly one of `model_samples` and `model_ppf` must be specified.
    model_ppfA, model_ppfB: Instead of specifying model samples, if a closed form for the
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
    progbarA, probgbarB: Control whether to create progress bar or use an existing one.
       These progress bars track the number of generated quantile paths.
       - With the default value 'auto', a new tqdm progress is created.
         This is convenient, however it can lead to many bars being created &
         destroyed if this function is called within a loop.
       - To prevent this, a tqdm progress bar can be created externally (e.g. with
         ``tqdm(desc="Generating paths")``) and passed as argument.
         Its counter will be reset to zero, and its set total to `R` + `previous_R`.
       - A value of `None` prevents displaying any progress bar.
    use_multiprocessing: If `True`, the statistics for models A and B are
       computed simultaneously; otherwise they are computed sequentially.
       Default is `True`.
       One reason not to use multiprocessing is if this call is part of a
       higher-level loop with is itself parallelized: child multiprocessing
       processes can’t spawn their own child processes.

    TODO
    ----
    - Use separate threads to update progress bars. This should minimize their
      tendency to lag behind the actual number of sampled paths.
    """


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
        μA, ΣA = compute_elppd_stats(
            data, logpA, model_samplesA, model_ppf=model_ppfA,
            c=c, res=res, N=N, R=R, relstderr_tol=relstderr_tol,
            path_progbar=progbarA)
        μB, ΣB = compute_elppd_stats(
            data, logpB, model_samplesB, model_ppf=model_ppfA,
            c=c, res=res, N=N, R=R, relstderr_tol=relstderr_tol,
            path_progbar=progbarB)
        
    else:
        progqA = mp.Queue() if progbarA is not None else None  # We manage the progbar ourselves. The Queue is used for receiving
        progqB = mp.Queue() if progbarA is not None else None  # progress updates from the function
        outqA = mp.Queue()   # Function output values are returned via a Queue
        outqB = mp.Queue()
        _compute_elppd_stats_A = partial(
            compute_elppd_stats,
            data, logpA, model_samplesA, model_ppf=model_ppfA,
            c=c, res=res, N=N, R=R, relstderr_tol=relstderr_tol,
            path_progbar=progqA)
        _compute_elppd_stats_B = partial(
            compute_elppd_stats,
            data, logpB, model_samplesB, model_ppf=model_ppfB,
            c=c, res=res, N=N, R=R, relstderr_tol=relstderr_tol,
            path_progbar=progqB)
        pA = mp.Process(target=mp_wrapper, args=(_compute_elppd_stats_A,),
                        kwargs={'path_progbar': progqA, 'out': outqA})
        pB = mp.Process(target=mp_wrapper, args=(_compute_elppd_stats_B,),
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

# + tags=["remove-input", "active-ipynb"]
# from emdd.utils import GitSHA
# GitSHA()
