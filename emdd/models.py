# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,md:myst
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (EMD-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown]
# # Models
# $\renewcommand{\EE}{\mathbb{E}}
# \renewcommand{\RR}{\mathbb{R}}
# \renewcommand{\Me}{\mathcal{M}^ε}
# \renewcommand{\Image}{\mathop{\mathrm{Image}}}
# \renewcommand{\sgn}{\mathop{\mathrm{sgn}}}
# \renewcommand{\Exponential}{\mathop{\mathrm{Exponential}}}
# \renewcommand{\SkewNormal}{\mathop{\mathrm{SkewNormal}}}
# \renewcommand{\exGaussian}{\mathop{\mathrm{exGaussian}}}
# \renewcommand{\IG}{\mathop{\mathrm{IG}}}
# \renewcommand{\NIG}{\mathop{\mathrm{NIG}}}
# \renewcommand{\Poisson}{\mathop{\mathrm{Poisson}}}
# \renewcommand{\Gammadist}{\mathop{\mathrm{Gamma}}}
# \renewcommand{\Lognormal}{\mathop{\mathrm{Lognormal}}}$
#
# :::{margin}
# **TODO**: Add link to paper.
# :::  
# This module provides a few simple models for use with the inferred model distance functions defined in [emd.py](./emd.py). To use these functions, a model must be divided into three parts:
# - An independent value generator, typically $x$ or $t$. ($x \sim p_x$)
# - A physical model. ($z \sim p_z(x)$)
# - An observation model. ($y \sim p_y(x, z)$)
#
# We provide simple models for each of these. In some cases they may be used as-is, and are used for demonstrations and tests, but more generally they are intended as templates to help users implement models compatible with the distance functions of this package.
#
# For the most part, our implementations wrap a distribution in `numpy.random` or `scipy.stats` with the required interface.
#
# **Interface requirements**  
# Models must have the following signatures:
#
# - *Independent model*: $L \mathtt{: int}, \mathtt{[seed]} \rightarrow x = \{x_1, \dotsc, x_L\}$
# - *Physical model*: $x, \mathtt{[seed]} \rightarrow z$
# - *Observation model*: $x, z, \mathtt{[seed]} \rightarrow y$
#
# The `seed` argument *must* be accepted, and *must* be optional. For stochastic models it should set state of the employed random number generator; for deterministic models it should be ignored.
#
# The main thing is that the distance functions will *only* use the parameters above (some combination of $L$, $x$, $z$ and `seed`). Any additional parameter must therefore already be associated to the model. This can be achieved in a number of ways, for example by hard-coding them in the functions, by wrapping functions with `functools.partial`, or by defining them as callable classes (i.e. classes with a `__call__` method).
#
# **Implementation**
# The model implementations we provide use callable [dataclasses](https://docs.python.org/3/library/dataclasses.html); class instances store parameters (as well as a possible random seed) as attributes. Users may inherit from them to use their seed management, or simply use them as inspiration for their own implementations.
#
# *Seed management feature*:
# A seed may be provided both when instantiating a model, *and* when calling for samples. The idea is that we may want different “sources of unicity”: a big unique random seed for the whole project, different seeds for different experiments, different seeds for data sets within an experiment, etc. Seeds can be passed as ints or tuples, and both the instantiation and calling seeds are combined to produce a unique random state for the random number generator.

# %% tags=["hide-input"]
import emdd

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from dataclasses import dataclass, InitVar
from typing import Union, Tuple

from more_itertools import always_iterable
import numpy as np
from numpy.random import Generator, PCG64
from scipy import stats
from scipy.optimize import root_scalar, minimize

# %% tags=["remove-cell"]
SeedType = Union[None,int,Tuple[int]]


# %% [markdown]
# Plotting configuration (notebook only).

# %% tags=["active-ipynb", "hide-input"]
# import holoviews as hv
# config = emdd.Config()
#
# backend = config.figures.backend
# colors  = config.figures.colors
# hv.extension(backend)
#
# if backend == "matplotlib":
#     hv.opts.defaults(
#         hv.opts.Histogram(edgecolor="none", facecolor=colors["bright"]["blue"]
#                           ),
#         hv.opts.Curve(color=colors["bright"]["red"]),
#         hv.opts.Scatter(color=colors["bright"]["grey"], s=16),
#         hv.opts.Layout(sublabel_format="")
#     )
# elif backend == "bokeh":
#     hv.opts.defaults(
#         hv.opts.Histogram(color=colors["bright"]["blue"]),
#         hv.opts.Curve(color=colors["bright"]["red"]),
#         hv.opts.Scatter(color=colors["bright"]["grey"], size=4),
#     )
#
# def show_dist(model, varname="y", L=10000):
#     """
#     Plot overlaid histogram and pdf.
#     If the model includes a transformation which was specified manually,
#     this can serve to check that the inverse map and inverse jacobian
#     are defined correctly
#     """
#     if isinstance(model.dist, stats._distn_infrastructure.rv_discrete):
#         # Discrete distribution
#         counts = Counter(model.rvs(L))
#         centers = np.sort(list(counts))
#         if len(centers) == 1:
#             edges = np.array([centers[0], centers[0]+1])
#         else:
#             widths = centers[1:] - centers[:-1]
#             edges = np.concatenate(([centers[0] - widths[0]/2],
#                                     centers[:-1] + widths/2,
#                                     [centers[-1] + widths[-1]/2]))
#         freqs = [counts[x]/L for x in centers]
#         hist = hv.Histogram(
#             (freqs, edges),
#             kdims=[varname], vdims=["Frequency"], label="samples")
#         theory = hv.Scatter(zip(centers, model.pmf(centers)), label="pmf",
#                             kdims=[varname], vdims=["Frequency"]
#                  ).opts(color=colors["bright"]["red"]
#                  ).opts(size=4, backend="bokeh")
#     else:
#         # Continuous distribution
#         hist = hv.Histogram(
#             np.histogram(model.rvs(L), bins='auto', density=True),
#             kdims=[varname], vdims=["Frequency"], label="samples"
#         ).opts(line_width=0, backend="bokeh")
#         xarr = np.linspace(hist.data[varname].min(), hist.data[varname].max())
#         theory = hv.Curve(zip(xarr, model.pdf(xarr)), label="pdf",
#                           kdims=[varname], vdims=["Frequency"])
#     return hist * theory
#
# def get_text_pos(panel):
#     "Utility function, used to place Text elements in a plot."
#     # Imperfect workaround; Bokeh doesn't really allow this for now (see https://discourse.holoviz.org/t/annotations-in-screen-coordinates)
#     data = panel.data[('Histogram', 'Samples')].data
#     dims = [dim.name for dim in panel.dimensions()]
#     x0 = data[dims[0]].min(); width = data[dims[0]].max() - x0
#     y0 = data[dims[1]].min(); height = data[dims[1]].max() - y0
#     #return x0 + 0.7*width, y0 + 0.65*height
#     return x0 + 0.1*width, y0 + 0.9*height

# %% [markdown]
# :::{Note}
# The PDF of the natural model is defined on the set of events of random variable $(X,Y)$. For example, if $Ω(X) = \RR$ and $Y = e^X$, we do not define $p(x,y) = p(x)δ(y - e^{x})$, since then the density $p(x,y)$ is either zero or infinite – which does not allow calculating the log likelihood. Instead we restrict ourselves to the 1D manifold $\{(x,y) \in Ω(X \times Y)\} \subset \RR^2$, on which the PDF $p(x,y) = p(x)$ is well behaved.
#
# Note also that the restriction to the event set of $(X,Y)$ works whether $Y$ is deterministic or random; in the latter case, $δ(y-g(x))$ is replaced by $p(y|x)$
# :::

# %% [markdown]
# :::{NOTE}  
# All models are defined such that they are hashable, so that functions which take a model as argument can be cached. This is used in by some calibration utility functions to avoid recomputing expensive functions.
# - ~~*Stochastic models* are hashed according to their identity.~~
# - ~~*Deterministic models*~~ Models are hashed according to their fields. This allows memoization to occur also between different instances of the same model.
#
# :::

# %% [markdown]
# ## Independent var models
#
# Models generating values of the independent variable ($x \in X$ in our notation).
# Generally expressions are conditioned on $X$, sometimes implicitely.
#
# $$\{\} \to X$$
#
# ### Typical usage
#
# Instantiate:
# ```python
# MX = MyIndepModel(*params, seed=seed)
# ```
#
# Create $L$ data points:
# ```python
# x = MX(L)
# ```
#
# As an infinite iterator (not available for all models):
# ```python
# for x in MX:
#     do something
# ```

# %%
@dataclass(frozen=True)
class StatXModel(ABC):
    """
    Base class for an independent model wrapping a distribution from
    `scipy.stats`.
    Subclasses must add distribution parameters as dataclass fields and implement
    `get_rv`, which takes those fields and returns a distribution.
    
    The main feature provided by this class is management of the `seed`:
    unless the seed is changed, multiple draws always return the same samples.
    To generate different samples, an additional `seed` argument can be provided.
    This seed is combined with the one set at instantiation.
    """
    seed: SeedType=None

    @abstractmethod
    def get_rv(self, seed: SeedType=None) -> stats._distn_infrastructure.rv_generic:
        raise NotImplementedError
    
    def get_rng(self, seed: SeedType=None) -> Generator:
        # Combine self.seed and seed
        if self.seed is not None and seed is not None:
            seed = (*always_iterable(self.seed),
                    *always_iterable(seed))
        elif self.seed is not None:
            seed = self.seed
        return Generator(PCG64(seed))
    def __iter__(self):
        if self.seed is None:
            raise RuntimeError("`seed` was not specified: cannot draw random samples.")
        rv = self.get_rv(seed=self.seed)
        while True:
            yield rv.rvs()
    def __call__(self, L, seed=None):
        if seed is None and self.seed is None:
            raise RuntimeError("`seed` was not specified: cannot draw random samples.")
        # Call L times
        return self.get_rv(seed=seed).rvs(size=L)


# %% [markdown]
# ### Uniformly distributed $X$
#
# Parameters: $a$, $b$
#
# $$x \sim \mathcal{U}([a, b])$$
#
# The use of a uniform distribution for $x$ can be seen as a mathematical convenience, to allow all pairs $(x,y)$ to be treated as samples from the same distributions. For a process with no particular temporal structure, this produces almost equal statistics for $y$ than using `LinspaceX` to sample $y$ at regular intervals of $x$.
#
# Stochastic: Even with the same `L`, each new call will return new $x$ values.

# %%
@dataclass(frozen=True)
class UniformX(StatXModel):
    low : float=0.
    high: float=1.
    
    def get_rv(self, seed=None):
        rv = stats.uniform(self.low, self.high)
        rv.random_state = self.get_rng(seed)
        return rv


# %% [markdown]
# ### Sequentially generated $X$
#
# Parameters: $x_0$, $Δx$
# $$\begin{aligned}
# x_i &= x_{i-1} + Δx \,,& i \geq 1
# \end{aligned}$$
#
# Deterministic: For a given `L`, the same values are always returned.
#
# **Hashing behaviour** (relevant for caching):
# Two models will hash to the same (and be considered equal) *iff both `x0` and `Δx` are equal*.

# %%
@dataclass(frozen=True)
class SequentialX:
    x0: float=0.
    Δx: float=1.
    def __iter__(self):
        x = self.x0
        i = 0
        while True:
            yield self.x0 + i*self.Δx  # More numerically stable than always adding Δx
            i += 1
    def __call__(self, L):
        return np.array([x for x, _ in zip(self, range(L))])


# %% [markdown]
# ### Linearly spaced $X$
#
# Parameters: `low`, `high`
# $$\begin{aligned}
# x_i &= \mathrm{low} + i \cdot \frac{\mathrm{high} - \mathrm{low}}{L-1} \,,& i = 0,\dotsc,L-1
# \end{aligned}$$
#
# Equivalent to NumPy’s `linspace`: values span the given interval uniformly. Deterministic: For a given `L`, the same values are always returned.
#
# **Hashing behaviour** (relevant for caching):
# Two models will hash to the same (and be considered equal) *iff both `low` and `high` are equal*.
#
# **Restriction**: Since the step size depends on $L$, it is not possible to provide an iterator interface.

# %%
@dataclass(frozen=True)
class LinspaceX:
    low: float=0.
    high: float=1.
    def __iter__(self):
        raise NotImplementedError("Since the step size depends on L, it is not "
                                  "possible to provide an iterator interface.")
    def __call__(self, L):
        return np.linspace(self.low, self.high, L)


# %% [markdown]
# ## Natural models
#
# $$X \to Z$$

# %% [markdown]
# ### Deterministic Exponential
#
# Deterministic
#
# This is our standard model for a natural process following an exponential decay.
#
# $$z(x) = e^{-λx}$$
#
# Possible physical models:
# - Voltage across a discharging capacitor.
#   + $x$: measurement time; $y$: voltage
# - Electric field penetration in a conductor.
#   + $x$: depth from surface; $y$: electric field intensity
# - Radioactive decay.
#   + $x$: measurement time; $y$: emission rate

# %%
@dataclass(frozen=True)
class DeterministicExpon:
    λ: float=1.
    seed: None=None  # Only for consistency with probabilistic models
    def __call__(self, x):
        return np.exp(-self.λ*x)


# %% tags=["active-ipynb"]
# MX = UniformX(0, 6)
# MN = DeterministicExpon(λ=0.3)
# xarr = MX(60)
# hv.Scatter(zip(xarr, MN(xarr)))

# %% [markdown]
# ## Observation (noise) models
#
# $$\begin{alignedat}{2}
# \Me&:\;& X \times Z &\to Y \\
# &\;& (x,z) &\mapsto y
# \end{alignedat}$$
#
# To facilitate comparisons, all noise models are parameterized by their normalized moments:
#
# - mean (always zero): $\EE[Y \mid X] = Z \,.$
# - standard deviation $σ = \sqrt{\EE[(Y-Z)^2 \mid X]}$
# - skew (when applicable): $γ = \frac{\EE[(Y-Z)^3 \mid X]}{σ^3}$
# - excess kurtosis (when applicable): $κ = \frac{\EE[(Y-Z)^4 \mid X]}{σ^4}$
#
# The definitions for *skew* and *kurtosis* seen in the literature are not always consistent. We use the same ones as the summary tables of Wikipedia articles for statistical distributions. In particular, $κ = 0$ for a normal distribution (the “excess” kurtosis is relative to the normal distribution).
#
# All distributions are constructed using a variation on the method of moments, using the *second* and higher moments: we invert the closed form expressions for those moments in terms of parameters. We then match the *first* moment by *translation*. For a distribution with unbounded support, this is usually the same as inverting the closed-form expression for the mean, but for distributions with bounded support this results in moving the support, typically so that it contains some negative numbers.

# %% [markdown]
# ### Generative vs inference noise models
#
# For inference,  we restrict ourselves to distributions that have support over $(-\infty, \infty)$. Otherwise, if even a single data point falls outside the support (which is often occurs when the model does not match the data perfectly), the likelihood diverges to $-\infty$, preventing any comparison of models.
# When generating synthetic data we do not have this constraint (the model is by definition always exact). This allows generating data using distributions which have support on $(0, \infty)$ for the noise model, such as exponential, Poisson or lognormal distributions.
#
# Note that unsupported data samples are only the most salient examples of this issue. More generally, bounded distributions are very sensitive[^safe-bounded] to the location of their upper or lower bound, to the point of making fits unstable. In some cases it may be possible to remedy this by fixing the bound(s), but that constitutes a strong inductive bias which must be warranted.
#
# :::{admonition} Remark
# The requirement that inference noise model have support on $\RR$ means that if the true noise model as bounded support, then for any practical inference model there will always be some mismatch between model and data.
# :::
#
# [^safe-bounded]: This is because there is a sharp decrease in probability density at the bound. Counter examples exist, like unbounded distributions which truncated far into their tails, but are not what one usually intends as bounded distributions.

# %%
@dataclass(frozen=True)
class AdditiveRVError:
    """
    Base class for experimental models consisting of adding noise from a
    scipy.stats random variable. If this random is denoted RV, then the
    resulting Y is given by
    
       Y|Z ~ Z + rv
    """
    seed: int=None
    
    @abstractmethod
    def get_rv(self, seed: SeedType=None) -> stats._distn_infrastructure.rv_generic:
        raise NotImplementedError
    def get_rng(self, seed: SeedType=None) -> Generator:
        # Combine self.seed and seed
        if self.seed is not None and seed is not None:
            seed = (*always_iterable(self.seed),
                    *always_iterable(seed))
        elif self.seed is not None:
            seed = self.seed
        return Generator(PCG64(seed))
    def __call__(self, x, z, seed=None):
        if seed is None and self.seed is None:
            raise RuntimeError("`seed` was not specified: cannot draw random samples.")
        return z + self.get_rv(seed).rvs(size=z.shape)
    def logpdf(self, y, z):
        return self.get_rv().logpdf(y - z)


# %% [markdown]
# ### Gaussian error
#
# Model for **symmetrically** distributed errors with **light** tails ($\sim \exp(-x^2)$)
#
# **Parameters**  
# - $μ$ – Mean
# - $σ$ – Standard deviation
#
# Centered normal distribution. 
#
# $$y \sim \mathcal{N}(0, σ)$$

# %%
@dataclass(frozen=True)
class GaussianError(AdditiveRVError):
    μ: float=0.
    σ: float=1.
    def get_rv(self, seed: SeedType=None):
        rv = stats.norm(self.μ, self.σ)
        rv.random_state = self.get_rng(seed)
        return rv


# %% tags=["active-ipynb", "hide-input"]
# show_dist(GaussianError(σ=0.5).get_rv())

# %% [markdown]
# ### Skew-normal distribution
#
# Model for **asymmetrically** distributed errors with **light** tails ($\sim \exp(-x^2)$).  
# A light tail means that deviations away from the mean on that side are **strongly** penalized.
#
# $γ = 0$ recovers the Gaussian distribution.
#
# **Parameters**  
# - $μ$ – Mean
# - $σ$ – Standard deviation  
# - $γ \in (-1, 1)$ – Skewness  (These bounds are not perfectly tight, but close.)
#
# $$y \sim \SkewNormal(0, σ, γ)$$

# %%
@dataclass(frozen=True)
class SkewNormalError(AdditiveRVError):
    μ: float=0.
    γ: float=0.  # Gaussian
    σ: float=1.
    
    @lru_cache(None)
    def _get_rv_args(self):
        # See below for explanation
        x0 = self.xroot(self.γ)
        δ = np.sign(self.γ)*np.sqrt(x0*np.pi/2)
        α = δ / np.sqrt((1-δ)*(1+δ))
        ω = self.σ / np.sqrt(1 - 2*δ**2/np.pi)
        ξ = self.μ - ω*δ*np.sqrt(2/np.pi)
        return α, ξ, ω
    
    def get_rv(self, seed: SeedType=None):
        rv = stats.skewnorm(*self._get_rv_args())
        rv.random_state = self.get_rng(seed)
        return rv
      
    @staticmethod
    def f(x, γ): return 4*γ*(1-x)**3 - (4-np.pi)**2*x**3
    @staticmethod
    def fp(x, γ): return -12*γ*(1-x)**2 - 3*(4-np.pi)**2*x**2
    @classmethod
    def xroot(cls, γ): return root_scalar(cls.f, (γ**2,), bracket=(0, 2),
                                     fprime=cls.fp, x0=0).root


# %% tags=["active-ipynb", "hide-input"]
# %%opts Overlay [legend_position="top_left"]
#
# γlst = [-.99, -0.5, 0, 0.5]
# σlst = [1, 2]
# dists = {(γ, σ): SkewNormalError(γ=γ, σ=σ).get_rv() for γ in γlst for σ in σlst}
# panels = [show_dist(d).opts(title=f"γ={γ}, σ={σ}") for (γ,σ), d in dists.items()]
# hv.Layout([(p * hv.Text(*get_text_pos(p), f"μ={d.mean():.2f}\nσ={d.std():.2f}\nmedian={d.median():.2f}")).opts(title=f"γ={γ}, σ={σ}")
#            for (γ, σ), p, d in zip(dists.keys(), panels, dists.values())]
#           + [(show_dist(stats.norm()) * hv.Text(-3, 0.25, "μ=0.00\nσ=1.00")).opts(title="Normal")]) \
#   .opts(shared_axes=False).cols(3)

# %% [markdown]
# #### Conversion to scipy arguments
#
# If $ξ$, $ω$, and $α$ are the location, scale and skew (shape) parameters passed to `scipy.stats.skewnorm`, then the moments are given by
#
# $$\begin{aligned}
# δ &:= \frac{α}{\sqrt{1 + α^2}} \\
# 0 \stackrel{!}{=} \EE[y] &= ξ + ωδ\sqrt{{2}/{π}}\\
# σ &= ω \sqrt{\left( 1 - \frac{1δ^2}{π} \right)} \\
# γ &= \frac{4 - π}{2} \frac{(δ\sqrt{2/π})^3}{(1 - 2δ^2/π)^{3/2}}
# \end{aligned}$$
#
# Let $x := 2δ^2/π \geq 0$. Then the equation for $γ$ can be rewritten
#
# $$\underbrace{4γ^2(1-x)^3 - (4-π)^2 x^3}_{f(x)} = 0 \,, \quad x \geq 0 \,.$$
#
# For any value of $γ$ the l.h.s. is a monotone decreasing function, as can be seen in the figure below. (We can check this analytically by searching for values of $x$ where the $f(x)$ changes sign: differentiating w.r.t. $x$ and setting to 0 yields always complex solutions when $δ > 0$: $x_{00} = \frac{24 \pm 6(4-π)γ \mathrm{i}}{12γ + 3(4-π)}$.) Exploring the parameter space for $γ^2$, we also see that the solution $x_0$ is always smaller than 2. Therefore a robust numerical solution is to use a bracketed solver and look for a root within $[0, 2)$.
#
# Since $α$, $δ$ and $γ$ must all have the same sign, we get
#
# $$δ = \sgn(γ) \sqrt{\frac{x_0 π}{2}}$$
#
# The expressions for $ξ$, $ω$ and $α$ are then easily obtained by inverting the equations:
#
# $$\begin{aligned}
# α &= \frac{δ}{\sqrt{(1-δ)(1+δ)}} \\
# ω &= \frac{σ}{\sqrt{1 - 2δ^2/π}} \\
# ξ &= \underbrace{0}_{\EE[y]} - ωδ\sqrt{2/π}
# \end{aligned}$$

# %% tags=["active-ipynb", "hide-input"]
# %%opts Scatter (color=colors["bright"]["red"])
# %%opts Curve (color=colors["muted"]["cyan"])
# %%opts HLine (color="grey", line_width=1)
#
# xarr = np.linspace(-2, 2, 100)
# xdim = hv.Dimension("x", soft_range=(xarr.min(), xarr.max()))
# ydim = hv.Dimension("fx", label="f(x)")
# γdim = hv.Dimension("γ", default=0)
# hv.HoloMap({γ: hv.Curve(zip(xarr, SkewNormalError.f(xarr, γ)), kdims=[xdim], vdims=[ydim])
#                * hv.Scatter([(SkewNormalError.xroot(γ), 0)], label="solved x")
#                * hv.HLine(0)
#             for γ in [0, 0.1, 0.5, 1]},  # γ limited to interval (-1, 1)
#            kdims=[γdim]).opts(
#     ylim=(-4, 2),
#     title="Cubic for intermediate value δ")

# %% [markdown]
# ### Exponentially modified Gaussian distribution
#
# Model for **asymmetrically** distributed errors with a **moderately heavy** long tail ($\sim \exp(-|x|)$).  
# A heavy tail means that deviations from the mean on that side are **moderately** penalized.
#
# An $\exGaussian$ random variable may be expressed as the sum of a Gaussian and an exponential distribution.
# The skew is thefore strictly positive, and in fact constrained to $γ \in [0, 2)$.
#
# $γ \to 0$ recovers the Gaussian distribution in the limit. This corresponds the rate $λ$ of the exponential distribution to $\infty$.
#
# $γ \to 2$ corresponds to the rate $λ$ going to 0. This is an improper distribution, with PDF proportional to the Heaviside function.
#
# TODO: Extend to negative skew, by flipping the distribution.
#
# **Parameters**  
# - $μ$ – Mean
# - $σ$ – Standard deviation  
# - $γ$ – Skewness
#
# $$y \sim \exGaussian(0, σ, γ)$$

# %%
@dataclass(frozen=True)
class exGaussianError(AdditiveRVError):
    μ: float=0.
    σ: float=1.
    γ: float=0  # Gaussian
    
    @lru_cache(None)
    def _get_rv_args(self):
        # NB: All special cases, which might break xroot, are dealt with in `get_rv`
        x0 = self.xroot(self.γ)
        K = np.sqrt(x0)
        ω = self.σ / np.sqrt(1 + K**2)
        return K, self.μ-ω*K, ω
        
    
    def get_rv(self, seed: SeedType=None):
        # See below for explanation
        if self.γ < 0:
            raise ValueError(f"Exponentially modified Gaussian cannot have negative skew; received γ={self.γ}")
        if self.γ > 2:
            raise ValueError(f"Exponentially modified Gaussian cannot have skew ≥2; received γ={self.γ}")
        elif self.γ == 0:
            # Special case for Gaussian limit
            rv = stats.norm(scale=self.σ)
        else:
            rv = stats.exponnorm(*self._get_rv_args())
        
        rv.random_state = self.get_rng(seed)
        return rv
       
    @staticmethod
    def f(x, γ): return γ**2 * (1+x)**3 - 4*x**3
    @staticmethod
    def fp(x, γ): return 3*γ**2 * (1+x)**2 - 12*x**2
    @staticmethod
    def xguess(γ): return np.nan if γ == 0 or γ == 2 else 1.5 * γ/(2-γ)
    @classmethod
    def xroot(cls, γ):
        if γ <= 0 or γ >= 2:
            return np.nan
        return root_scalar(cls.f, (γ,),# bracket=(x_i, np.inf),
                           fprime=cls.fp, x0=cls.xguess(γ)).root


# %% tags=["active-ipynb", "hide-input"]
# %%opts Overlay [legend_position="top_right"]
# %%opts Layout [fig_inches=3.5]
#
# γlst = [0, 0.1, 0.5, 1, 1.5, 1.75, 1.9]
# σlst = [1, 2]
# dists = {(γ, σ): exGaussianError(γ=γ, σ=σ).get_rv() for γ in γlst for σ in σlst}
# panels = [show_dist(d).opts(title=f"γ={γ}, σ={σ}", show_legend=False) for (γ,σ), d in dists.items()]
# panels = [(show_dist(stats.norm()) * hv.Text(-3, 0.25, "μ=0.00\nσ=1.00")).opts(title="Normal")] \
#          + [(p * hv.Text(*get_text_pos(p), f"μ={d.mean():.2f}\nσ={d.std():.2f}\nmedian={d.median():.2f}"))
#             .opts(title=f"γ={γ}, σ={σ}", show_legend=False)
#             for (γ, σ), p, d in zip(dists.keys(), panels, dists.values())]
# panels[2].opts(show_legend=True)
# hv.Layout(panels) \
#     .opts(shared_axes=False) \
#     .opts(hv.opts.Overlay(aspect=2, backend="matplotlib")) \
#     .cols(3)

# %% [markdown]
# #### Conversion to scipy arguments
#
# If $ξ$, $ω$, and $K$ are the location, scale and shape (inverse rate) parameters passed to `scipy.stats.skewnorm`, then the moments are given by
#
# $$\begin{aligned}
# 0 \stackrel{!}{=} \EE[y] &= ξ + ωK \\
# σ &= ω \sqrt{1 + K^2} \\
# γ &= 2 K^{3} \left( 1 + K^2 \right)^{-3/2} \,. \\
# \end{aligned}$$
#
# We see that when $K \to \pm \infty$, the last equation reduces to $γ = \pm 2$. This is why the skewness cannot exceed 2.

# %% [markdown]
# Let $x := K^{2} \geq 0$. Then the equation for $γ$ can be rewritten
#
# $$\underbrace{γ^2 (1 + x)^3 - 4 x^3}_{f(x)} = 0 \,, \quad x \geq 0 \,.$$

# %% [markdown]
# Differentiating once, we find can find the inflection points by solving the quadratic. This gives us
#
# $$f'(x) = 0 \;\;\text{iff}\;\; x = \frac{γ(-γ \pm 2)}{(γ + 2)(γ - 2)}$$
#
# We also see (either by inspecting the graph or the expression for $f'(x)$) that $f$ has negative slope for large $x$ and the inflection points are always above the abscissa. Therefore we want the root which is to the right of the rightmost inflection point. We can enforce this with the brackets
#
# $$x_0 \in \left(\frac{γ}{2-γ}, \infty\right)$$
#
# and the initial guess
#
# $$x_g = \frac{2γ}{2-γ} \,.$$
#
# In practice we found that just using the initial guess is already reliable, and avoids having to guess an appropriate upper bound.

# %% [markdown]
# Since we have ensured that $x_0$ is positive, and we know that $K > 0$, we then recover scipy arguments as
#
# $$\begin{aligned}
# K &= \sqrt{x_0} \\
# ω &= \frac{σ}{\sqrt{1 + K^2}}\\
# ξ &= \underbrace{0}_{\EE[y]} - ωK
# \end{aligned}$$

# %% tags=["active-ipynb", "hide-input"]
# %%opts Scatter [width=600] (size=4)
# %%opts Curve [width=600] (color=colors["muted"]["cyan"])
# %%opts HLine (color="grey", line_width=1)
# %%opts Overlay [width=600, legend_position="right"]
#
# xarr = np.linspace(-2, 10, 400)
# xdim = hv.Dimension("x", range=(xarr.min(), xarr.max()))
# ydim = hv.Dimension("fx", label="f(x)")
# γdim = hv.Dimension("γ", default=0.001)
# hm = hv.HoloMap({γ: hv.Curve(zip(xarr, exGaussianError.f(xarr, γ)), kdims=[xdim], vdims=[ydim])
#                * hv.Scatter([(exGaussianError.xguess(γ), 0)], label="initial guess x_0").opts(color=colors["bright"]["red"])
#                * hv.Scatter([(exGaussianError.xroot(γ), 0)], label="solved x").opts(color=colors["bright"]["green"])
#                * hv.HLine(0)
#             for γ in [0, 0.001, 0.01, 0.1, 0.5, 1, 1.7, 2]},  # γ limited to interval (-1, 1)
#            kdims=[γdim])
# hm.opts(ylim=(-4, 2),
#         title="Cubic for intermediate value δ") \
#        .redim.range(fx=(-4, 4))

# %% [markdown]
# ### Normal-inverse Gaussian error
#
# Denoted $\NIG$.
#
# :::{note}
# This is not the same as the *Inverse Gaussian* distribution, denoted $\IG$.
# :::
#
# The normal-inverse Gaussian distribution has properties that make it an attractive trade-off between flexibility and tractability [[wikipedia]](https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution):
# - Support on $(-\infty, \infty)$.
# - Simple expressions for all moments.
# - Includes fat-tailed and skewed distributions.
# - Includes the normal distribution as a limiting case.
#
# The main limitation is that the kurtosis cannot be less than a Gaussian, so very light-tailed distributions are outside this family's range.
#
# **Parameters**  
# - $μ \in (-\infty, \infty)$ – Mean
# - $σ \in [0, \infty)$ – Standard deviation  
# - $γ \in (-\infty, \infty)$ – Skewness
# - $κ \in \left(\frac{5}{3} γ^2, \infty\right)$ – Excess kurtosis
#
# Note that the lower bound on the excess kurtosis implies in particular that it cannot be negative, i.e. a $\NIG$ cannot have less kurtosis than a Gaussian.

# %% [markdown]
# $$\begin{aligned}
# α &\geq β \\
# κ &> \frac{5}{3} γ^2
# \end{aligned}$$

# %%
@dataclass(frozen=True)
class NormInvGaussError(AdditiveRVError):
    μ: float=0.
    σ: float=1.
    γ: float=0.   # Defaults to Gaussian
    κ: float=0.
    
    @lru_cache(None)
    def _get_rv_args(self):
        # NB: Only the cases where we return norminvgauss call this function
        μ, σ, γ, κ = self.μ, self.σ, self.γ, self.κ
        if γ == 0:  # Special case: Symmetric distribution
            a = 3*κ
            b = 0
            ω = σ/np.sqrt(a)
            ξ = 0
        else:  # General case
            x = γ/np.sqrt(3*κ - 4*γ**2)
            c = 9*x**2 / γ**2
            a = c/np.sqrt(1-x**2)
            b = np.sign(γ) * a * x
            ω = σ * c**(3/2) / a
            ξ = μ-ω*b/c
        return a, b, ξ, ω
    
    def get_rv(self, seed: SeedType=None):
        μ, σ, γ, κ = self.μ, self.σ, self.γ, self.κ
        if γ == κ == 0:  # Special case: Gaussian distribution
            rv = stats.norm(0, σ)
        elif γ == 0:  # Special case: Symmetric distribution
            rv = stats.norminvgauss(*self._get_rv_args())
        elif κ <= 5/3 * γ**2:
            raise ValueError("Kurtosis (κ) and skewness (γ) must satisfy κ > 5/3 γ². Provided values:\n"
                             f"κ     : {κ}\n5/3 γ²: {5/3*γ**2}")
        else:  # General case
            rv = stats.norminvgauss(*self._get_rv_args())
        
        rv.random_state = self.get_rng(seed)
        return rv


# %% tags=["active-ipynb", "hide-input"]
# %%opts Overlay [xlabel=""]
# %%opts Text (color="#888888")
#
# γlst = np.array([-5, -1, 0., 0.3, 0.6, 1, 5])
# κγlst = np.array([1.01, 1.5, 2, 5, 20])
#
# frames = {}
# for κγ in κγlst:
#     dists = {γ: NormInvGaussError(σ=1, γ=γ, κ=κγ*5/3*γ**2).get_rv() for γ in γlst}
#     panels = [show_dist(d, f"y_{γ}").redim.range(Frequency=(0,0.5)) for γ, d in dists.items()]
#     layout = hv.Layout([(p * hv.Text(get_text_pos(p)[0], 0.4,
#                                      f"μ={d.mean():.2f}, median={d.median():.2f}\nσ={d.std():.2f}\nγ={d.stats('s'):.2f}\nκ={d.stats('k'):.2f}",
#                                     halign="left"))
#                         .opts(show_legend=False, title=f"γ={γ}")
#                         for p, (γ, d) in zip(panels, dists.items())]
#                        + [(show_dist(stats.norm()) * hv.Text(-3, 0.25, "μ=0.00\nσ=1.00\nγ=0.00\nκ=0.00")).opts(title="Normal")]
#              )
#     layout.opts(shared_axes=True)
#     frames[κγ] = layout
# hv.HoloMap(frames,
#            kdims=[hv.Dimension("κγ", label="κ/κ_min")]
#           ).collate()

# %% [markdown]
# #### Conversion to scipy arguments
#
# If $ξ$, $ω$, $a$ and $b$ are the location, scale and shape (tail heaviness and asymmetry) parameters passed to `scipy.stats.norminvgauss`, then the moments are given by
#
# ::::{margin}
# :::{admonition} Conversion to [Wikipedia](https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution) parameterization
# $$\begin{aligned}
# μ &= ξ \\
# δ &= ω \\
# α &= a/ω \\
# β &= b/ω
# \end{aligned}$$
# :::
# ::::
#
# $$\begin{aligned}
# 0 \stackrel{!}{=} \EE[y] &= ξ + ω \frac{b}{\sqrt{a^2 - b^2}} &&= ξ + \frac{ωb}{c}\\
# σ &= aω (a^2 - b^2)^{-3/4} &&= \frac{aω}{c^{3/2}}\\
# γ &= \frac{3b}{a} (a^2 - b^2)^{-1/4} &&= \frac{3x}{\sqrt{c}}\\
# κ &= \frac{3\left(1 + 4\frac{b^2}{a^2}\right)}{\sqrt{a^2 - b^2}} &&= \frac{3(1 + 4x^2)}{c}
# \,, \\
# \end{aligned}$$
#
# ::::{margin}
# :::{warning}
# The *scipy* documentation incorrectly makes the inequality non-strict: $|b| \leq a$.
# :::
# ::::
# where for the r.h.s. equations we defined $x := b/a$ and $c := \sqrt{a^2-b^2}$.
# As documented in *scipy*, parameters must satisfy $a > 0$ and $|b| < a$.

# %% [markdown]
# These equations are easily solved for $|x|$. Matching the sign of the asymmetry parameter ($b$) to the skewness ($γ$), we get all the function parameters:
#
# \begin{alignat}{2}
# x &= \frac{b}{a} &&= \pm \frac{γ}{\sqrt{3κ - 4γ^2}} \\
# c &= \sqrt{a^2- b^2} &&= \frac{9x^2}{γ^2} \\
# a &= \frac{c}{\sqrt{1-x^2}} \\
# b &= \sgn(γ) \, \frac{a}{|x|} \\
# ω &= \frac{σc^{3/2}}{a} \\
# ξ &= - \frac{ωb}{c}
# \end{alignat}
#
# From the first equation and the requirement that $|x| < 1$ we get the constraint that $κ > \frac{5}{3} γ^2$ . The equality $κ = \frac{5}{3} γ^2$ corresponds to $a = b$, which has infinite moments.
#
# **Special case: $γ = 0$**  
# We need to treat the case of a symmetric distribution specially, since $γ = b = 0$ implies $x \equiv 0$. However in this case the equations become trivially solvable:
#
# $$\begin{aligned}
# 0 &= ξ \\
# σ &= \frac{ω}{\sqrt{a}} \\
# γ &= 0 \\
# κ &= \frac{3}{a},
# \end{aligned}$$
#
# which have the solution $a = 3κ$, $b = 0$, $ω = σ \sqrt{a}$, $ξ = 0$.
# In the limit $a \to \infty$ this recovers the Gaussian distribution.
#
# **Special case: $γ = κ = 0$**  
# This is the Gaussian distribution with standard deviation $σ$.

# %% [markdown]
# ---
#
# :::{caution}
# The distributions below have bounded support. [As noted above](#generative-vs-inference-noise-models), they should not be use for inference, but can be used to generate synthetic data.
# :::
#
# ### Exponential error
#
# Denoted $\Exponential$.
#
# - Support on $(-σ, \infty)$.
#
# **Parameters**  
# - $μ \in (-\infty, \infty)$: Mean  *(translated)*
# - $σ \in [0, \infty)$: Standard deviation  
#
# **Higher moments**  
# - $γ = 2$
# - $κ = 6$

# %%
@dataclass(frozen=True)
class ExponentialError(AdditiveRVError):
    μ: float=0.
    σ: float=1.
    
    def get_rv(self, seed: SeedType=None):
        σ = self.σ
        rv = stats.expon(self.μ-σ, σ)
        rv.random_state = self.get_rng(seed)
        return rv


# %% tags=["remove-cell", "skip-execution", "active-ipynb"]
# for σ in [0.1, 1, 4, 8.3]:
#     assert ExponentialError(σ=σ).get_rv().stats("mvsk") == (0, σ**2, 2, 6)

# %% [markdown]
# ### Poisson error
#
# Denoted $\Poisson$.
#
# - Support on $(-σ^2, \infty)$.
#
# **Parameters**  
# - $μ \in (-\infty, \infty)$: Mean  *(translated)*
# - $σ \in [0, \infty)$: Standard deviation  
#
# **Higher moments**  
# - $γ = σ^{-1}$
# - $κ = σ^{-2}$
#
# ::::{margin}
# :::{table} Parameter conversions for the Poisson distribution
#
# | Wikipedia | Scipy |
# |-----------|-------|
# | $λ$       | $μ$   |
#
# :::
# ::::

# %%
@dataclass(frozen=True)
class PoissonError(AdditiveRVError):
    μ: float=0.
    σ: float=1.
    
    def get_rv(self, seed: SeedType=None):
        μ, σ = self.μ, self.σ
        λ = σ**2
        rv = stats.poisson(λ, loc=μ-λ)
        rv.random_state = self.get_rng(seed)
        return rv


# %% tags=["remove-cell", "skip-execution", "active-ipynb"]
# for σ in [0.1, 1, 4, 8.3]:
#     assert PoissonError(σ=σ).get_rv().stats("mvsk") == (0, σ**2, 1/σ, 1/σ**2)

# %% tags=["active-ipynb", "hide-input"]
# %%opts Overlay [legend_position="top_right", show_legend=False]
# %%opts Layout [fig_inches=3.5]
#
# σlst = [0.1, 1, 2, 4, 10]
# dists = {σ: PoissonError(σ=σ).get_rv() for σ in σlst}
# panels = [show_dist(d).opts(title=f"σ={σ}") for σ, d in dists.items()]
# hv.Layout(
#     [(p * hv.Text(*get_text_pos(p), f"μ={d.mean():.2f}, median={d.median():.2f}\nσ={d.std():.2f}\nγ={d.stats('s'):.2f}\nκ={d.stats('k'):.2f}"))
#      .opts(title=f"σ={σ}")
#         for σ, p, d in zip(dists.keys(), panels, dists.values())]
#     ) \
#   .opts(hv.opts.Layout(shared_axes=False),
#         hv.opts.Scatter(size=4, backend="bokeh")) \
#   .cols(3)

# %% [markdown]
# ### Gamma error
#
# Denoted $\Gammadist$.
#
# - Support on $\left(-\frac{2σ}{γ}, \infty\right)$.
#
# **Parameters**  
# - $μ \in (-\infty, \infty)$: Mean  *(translated)*
# - $σ \in [0, \infty)$: Standard deviation  
# - $γ \in (0, \infty)$: Skewness
#
# **Higher moments**
# - $κ = \frac{3}{2}γ^2$
#
# ::::{margin}
# :::{table} Parameter conversions for the Gamma distribution
#
# | Wikipedia    | Scipy |
# |--------------|-------|
# | $a = k$      | $a$   |
# | $θ = β^{-1}$ | `scale` |
#
# :::
# ::::

# %%
@dataclass(frozen=True)
class GammaError(AdditiveRVError):
    μ: float=0.
    σ: float=1.
    γ: float=np.sqrt(2)
    
    def get_rv(self, seed: SeedType=None):
        μ, σ, γ = self.μ, self.σ, self.γ
        k = 4 / γ**2
        θ = σ*γ/2
        rv = stats.gamma(k, loc=μ-k*θ, scale=θ)

        rv.random_state = self.get_rng(seed)
        return rv

# %% tags=["remove-cell", "skip-execution", "active-ipynb"]
# for σ in [0.1, 1, 4, 8.3]:
#     for γ in [0.1, 1, 4, 8.3]:
#         assert np.allclose(GammaError(σ=σ, γ=γ).get_rv().stats("mvsk"),
#                            (0, σ**2, γ, 1.5*γ**2))

# %% tags=["active-ipynb", "hide-input"]
# %%opts Overlay [xlabel=""]
# %%opts Text (color="#888888")
# %%opts Layout [fig_inches=2.5]
#
# σlst = np.array([0.1, 1, 4, 8])
# γlst = np.array([0.1, 1, 4, 8])
#
# frames = {}
# for σ in σlst:
#     dists = {γ: GammaError(σ=σ, γ=γ).get_rv() for γ in γlst}
#     panels = [show_dist(d, f"y_{γ}").redim.range(Frequency=(0,0.5)) for γ, d in dists.items()]
#     layout = hv.Layout([(p * hv.Text(get_text_pos(p)[0], 0.4,
#                                      f"μ={d.mean():.2f}, median={d.median():.2f}\nσ={d.std():.2f}\nγ={d.stats('s'):.2f}\nκ={d.stats('k'):.2f}",
#                                     halign="left"))
#                         .opts(show_legend=False, title=f"γ={γ}")
#                         for p, (γ, d) in zip(panels, dists.items())]
#                        #+ [(show_dist(stats.norm()) * hv.Text(-3, 0.25, "μ=0.00\nσ=1.00\nγ=0.00\nκ=0.00")).opts(title="Normal")]
#              )
#     layout.opts(shared_axes=True).cols(3)
#     frames[σ] = layout
# hv.HoloMap(frames,
#            kdims=[hv.Dimension("σ", label="σ")]
#           ).collate()

# %% tags=["active-ipynb", "remove-input"]
# from emdd.utils import GitSHA
# GitSHA()
