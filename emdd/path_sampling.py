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
# (supp_path-sampling)=
# # Sampling quantile paths

# %% [markdown]
# $\renewcommand{\lnLtt}{\tilde{l}}
# \renewcommand{\lnLh}[1][]{\hat{l}_{#1}}
# \renewcommand{\EE}{\mathbb{E}}
# \renewcommand{\VV}{\mathbb{V}}
# \renewcommand{\nN}{\mathcal N}
# \renewcommand{\emdstd}[1][]{\tilde{σ}_{{#1}}}
# \renewcommand{\emdstd}[1][]{{\mathrm{EMD}}_{#1}}
# \renewcommand{\Mvar}{\mathop{\mathrm{Mvar}}}
# \renewcommand{\EMD}[1][]{{\mathrm{EMD}}_{#1}}
# \renewcommand{\Beta}{\mathop{\mathrm{Beta}}}
# \renewcommand{\pathP}{\mathop{\mathcal{P}}}
# $

# %% tags=["hide-input"]
import logging
import warnings
import math
import time
#import multiprocessing as mp
import numpy as np
from scipy.special import digamma, polygamma
from scipy.optimize import root, brentq
from tqdm.auto import tqdm

from typing import Optional, Union, Literal, Tuple, Generator
from scityping import Real
from scityping.numpy import Array, Generator as RNGenerator

from emdd.digitize import digitize  # Used to improve numerical stability when finding Beta parameters

# %% [markdown] tags=["remove-cell"] user_expressions=[]
# Notebook-only imports

# %% tags=["active-ipynb", "hide-input"]
# import itertools
# import scipy.stats
# import holoviews as hv
# from myst_nb import glue
#
# from emdd.utils import GitSHA
# from config import config  # Uses config from CWD
#
# hv.extension(config.figures.backend)

# %% tags=["remove-cell"]
logger = logging.getLogger(__name__)


# %% [markdown] user_expressions=[]
# We want to generate paths $\lnLh$ for the quantile function $l(Φ)$, with $Φ \in [0, 1]$, from a stochastic process $\pathP$ determined by $\lnLtt(Φ)$ and $\emdstd(Φ)$. This process must satisfy the following requirements:
# - It must generate only monotone paths, since quantile functions are monotone.
# - The process must be heteroscedastic, with variability at $Φ$ given by $\emdstd(Φ)$.
# - Generated paths should track the function $\lnLtt(Φ)$, and track it more tightly when variability is low.
# - Paths should be “temporally uncorrelated”: each stop $Φ$ corresponds to a different ensemble of data points, which can be drawn in any order. So we don't expect any special correlation between $\lnLh(Φ)$ and $\lnLh(Φ')$, beyond the requirement of monotonicity.
#   + In particular, we want to avoid defining a stochastic process which starts at one point and accumulates variance, like the $\sqrt{t}$ envelope characteristic of a Gaussian white noise.
#   + Concretely, we require the process to be “$Φ$-symmetric”: replacing $\lnLtt(Φ) \to \lnLtt(-Φ)$ and $\emdstd(Φ) \to \emdstd(-Φ)$ should define the same process, just inverted along the $Φ$ axis.

# %% [markdown] user_expressions=[]
# (supp_path-sampling_hierarchical-beta)=
# ## Hierarchical beta process

# %% [markdown] user_expressions=[]
# Because the joint requirements of monotonicity, non-stationarity and $Φ$-symmetry are uncommon for a stochastic process, some care is required to define an appropriate $\pathP$. The approach we choose here is to first select the end points $\lnLh(0)$ and $\lnLh(1)$, then fill the interval by successive binary partitions: first $\bigl\{\lnLh\bigl(\tfrac{1}{2}\bigl)\bigr\}$, then $\bigl\{\lnLh\bigl(\tfrac{1}{4}\bigr), \lnLh\bigl(\tfrac{3}{4}\bigr)\bigr\}$, $\bigl\{\lnLh\bigl(\tfrac{1}{8}\bigr), \lnLh\bigl(\tfrac{3}{8}\bigr), \lnLh\bigl(\tfrac{5}{8}\bigr), \lnLh\bigl(\tfrac{7}{8}\bigr)\bigr\}$, etc. (Below we will denote these ensembles $\{\lnLh\}^{(1)}$, $\{\lnLh\}^{(2)}$, $\{\lnLh\}^{(3)}$, etc.) Thus integrating over paths becomes akin to a path integral with variable end points.
# Moreover, instead of drawing quantile values, we draw increments
# $$Δ l_{ΔΦ}(Φ) := \lnLh(Φ+ΔΦ) - \lnLh(Φ) \,.$$ (eq_def-quantile-increment)
# Given two initial end points $\lnLh(0)$ and $\lnLh(1)$, we therefore first we draw the pair $\bigl\{Δ l_{2^{-1}}(0),\; Δ l_{2^{-1}}\bigl(2^{-1}\bigr)\}$, which gives us
# $$\lnLh\bigl(2^{-1}\bigr) = \lnLh(0) + Δ l_{2^{-1}}(0) = \lnLh(1) - Δ l_{2^{-1}}\bigl(2^{-1}\bigr)\,.$$
# Then $\bigl\{\lnLh(0), \lnLh\bigl(\frac{1}{2}\bigr) \bigr\}$ and $\bigl\{ \lnLh\bigl(\frac{1}{2}\bigr), \lnLh(1) \bigr\}$ serve as end points to draw $\bigl\{Δ l_{2^{-2}}\bigl(0\bigr),\; Δ l_{2^{-2}}\bigl(2^{-2}\bigr) \bigr\}$ and $\bigl\{Δ l_{2^{-2}}\bigl(2^{-1}\bigr),\; Δ l_{2^{-2}}\bigl(2^{-1} + 2^{-2}\bigr) \bigr\}$. We repeat the procedure as needed, sampling smaller and smaller incremenents, until the path has the desired resolution. As the increments are constrained:
# $$Δ l_{2^{-n}}(Φ) \in \bigl( 0, \lnLh(Φ+2^{-n+1}) - \lnLh(Φ)\,\bigr)\,, $$
# the path thus sampled is always monotone. Note also that increments must be drawn in pairs (or more generally as a *combination*) of values constrained by their sum:
# $$Δ l_{2^{-n}}\bigl(Φ\bigr) + Δ l_{2^{-n}}\bigl(Φ + 2^{-n} \bigr) \stackrel{!}{=} \lnLh(Φ+2^{-n+1}) - \lnLh(Φ) \,.$$ (eq_sum-constraint)
# The possible increments therefore lie on a 1-simplex, for which a natural choice is to use a beta distribution[^1], with the random variable corresponding to the first increment $Δ l_{2^{-n}}(Φ)$. The density function of a beta random variable has the form
# $$p(x_1) \propto x^{α-1} (1-x)^{β-1}\,,$$ (eq_beta-pdf)
# with $α$ and $β$ parameters to be determined.

# %% [markdown] user_expressions=[]
# :::{IMPORTANT}  
# An essential property of a stochastic process is *consistency*: it must not matter exactly how we discretize the interval {cite:p}`gillespieMathematicsBrownianMotion1996`. Let $\{\lnLh\}^{(n)}$ denote the steps which are added when we refine the discretization from steps of $2^{-n+1}$ to steps of $2^{-n}$:
# $$\{\lnLh\}^{(n)} := \bigl\{\lnLh(k\cdot 2^{-n}) \,\big|\, k=1,3,\dotsc,2^n \bigr\} \,.$$ (eq_added-steps)
# A necessary condition for consistency is that coarsening the discretization from steps of $2^{-n}$ to steps of $2^{-n+1}$ (i.e. marginalizing over the points at $\{\lnLh\}^{(n)}$) does not substantially change the probability law:
# $$p\bigl(\{\lnLh\}^{(n)}\bigr)\bigr) \stackrel{!}{=} \int p\bigl(\{\lnLh\}^{(n)} \,\big|\, \{\lnLh\}^{(n+1)}\bigr) \,d\{\lnLh\}^{(n+1)} \;+\; ε\,,$$ (eq_consistency-condition)
# with $ε$ vanishing as $n$ is increased to infinity.
#
# We have found that failure to satisfy this requirement leads to unsatisfactory sampling of quantile paths. In particular, naive procedures tend to perform worse as $ΔΦ$ is reduced, making accurate integration impossible.
# :::

# %% [markdown] user_expressions=[]
# [^1]: One could conceivably draw all increments at once, with a [*shifted scaled Dirichlet distribution*](https://doi.org/10.1007/978-3-030-71175-7_4) instead of a beta, if it can be shown that also in this case coarsening the distribution still results in the same probability law.

# %% [markdown] user_expressions=[]
# (supp_path-sampling_conditions-beta-param)=
# ### Conditions for choosing the beta parameters
#
# To draw an increment $Δ l_{2^{-n}}$, we need to convert $\lnLtt(Φ)$ and $\emdstd(Φ)$ (obtained from the model discrepancy analysis) into beta distribution parameters $α$ and $β$. If $x_1$ follows a beta distribution, then its first two cumulants are given by
# $$\begin{aligned}
# x_1 &\sim \Beta(α, β) \,, \\
# \EE[x_1] &= \frac{α}{α+β} \,, \\
# \VV[x_1] &= \frac{αβ}{(α+β)^2(α+β+1)} \,. \\
# \end{aligned}$$
# However, as discussed by Mateu-Figueras et al. (2021, 2011), to properly account for the geometry of a simplex, one should consider instead statistics of with respect to the Aitchison measure, sometimes referred to as the *center* and *metric variance*. Defining $x_2 = 1-x_1$, these can be written (Mateu-Figueras et al., 2021)
# $$\begin{align}
# \EE_a[(x_1, x_2)] &= \frac{1}{e^{ψ(α)} + e^{ψ(β)}} \bigl[e^{ψ(α)}, e^{ψ(β)}\bigr] \,, \label{eq_Aitchison-moments__EE} \\
# \Mvar[(x_1, x_2)] &= \frac{1}{2} \bigl(ψ_1(α) + ψ_1(β)\bigr) \,. \label{eq_Aitchison-moments__Mvar}
# \end{align}$$ (eq_Aitchison-moments)
# Here $ψ$ and $ψ_1$ are the digamma and trigamma functions respectively.
# (In addition to ontological considerations, it is much more straightforward to define a consistent stochastic process using the center and metric variance. For example, since the metric variance is unbounded, we can easily scale it with $\emdstd(Φ)$.)

# %% [markdown] user_expressions=[]
# Since we want the sum to be $d := \lnLh(Φ+2^{-n+1}) - \lnLh(Φ)$, we define
# $$\bigl[Δ l_{2^{-n}}\bigl(Φ\bigr),\, Δ l_{2^{-n}}\bigl(Φ+2^{-n})\bigr)\bigr] = d \cdot \bigl[x_1, x_2\bigr] \,.$$  (eq_relation-beta-increment)
# Then

# %% [markdown] user_expressions=[]
# $$\begin{aligned}
# \EE_a\Bigl[\bigl[Δ l_{2^{-n}}\bigl(Φ\bigr),\, Δ l_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\Bigr] &= \frac{d}{e^{ψ(α)} + e^{ψ(β)}} \bigl[e^{ψ(α)}, e^{ψ(β)}\bigr] \,, \\
# \Mvar\Bigl[\bigl[Δ l_{2^{-n}}\bigl(Φ\bigr),\, Δ l_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\Bigr] &= \frac{1}{2} \bigl(ψ_1(α) + ψ_1(β)\bigr) \,.
# \end{aligned}$$

# %% [markdown] user_expressions=[]
# We now choose to define the parameters $α$ and $β$ via the following relations:
# $$\begin{aligned}
# \EE_a\Bigl[\bigl[Δ l_{2^{-n}}\bigl(Φ\bigr),\, Δ l_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\Bigr] &=^* \bigl[\, \lnLtt\bigl(Φ+2^{-n}\bigr) - \lnLtt\bigl(Φ\bigr),\,\lnLtt\bigl(Φ+2^{-n+1}\bigr) - \lnLtt\bigl(Φ+2^{-n}\bigr) \,\bigr]\,, \\
# \Mvar\Bigl[\bigl[Δ l_{2^{-n}}\bigl(Φ\bigr),\, Δ l_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\Bigr] &\stackrel{!}{=} \emdstd\bigl(Φ+2^{-n}\bigr)^2 \,.
# \end{aligned}$$ (eq_defining-conditions-a)

# %% [markdown] user_expressions=[]
# These follow from interpretating $\lnLtt$ and $\emdstd$ as estimators for the mean and square root of the metric variance.
# We use $=^*$ to indicate equality in spirit rather than true equality, since strictly speaking, these are 3 equations for 2 unknown. To reduce the $\EE_a$ equations to one, we use instead
# $$\frac{\EE_a\bigl[Δ l_{2^{-n}}\bigl(Φ\bigr)\bigr]}{\EE_a \bigl[Δ l_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]} \stackrel{!}{=} \frac{\lnLtt\bigl(Φ+2^{-n}\bigr) - \lnLtt\bigl(Φ\bigr)}{\lnLtt\bigl(Φ+2^{-n+1}\bigr) - \lnLtt\bigl(Φ+2^{-n}\bigr)} \,.$$ (eq_defining_conditions-b)

# %% [markdown] tags=["remove-cell"] user_expressions=[]
# ---
#
# TO REMOVE: I originally put a $2^{-n}$ coefficient in front of $\Mvar$. I don't think it's warranted anymore, since we already scale with $d$.
#
# In the second, the scaling proportional to the step size ensures that as increments get smaller, the variance also decreases. More specifically, that $\Mvar\Bigl[\bigl[Δ l_{2^{-n+1}}\bigl(Φ\bigr),\, Δ l_{2^{-n+1}}\bigl(Φ+2^{-n+1})\bigr)\bigr]\Bigr] \approx 2\,\Mvar\Bigl[\bigl[Δ l_{2^{-n}}\bigl(Φ\bigr),\, Δ l_{2^{-n}}\bigl(Φ+2^{-n})\bigr)\bigr]\Bigr]$.
#
# ---

# %% [markdown] user_expressions=[]
# **Remarks**
# - We satisfy the necessary condition for consistency by construction:
#   $$p\bigl(\{l\}^{(n)}\bigr)\bigr) = \int p\bigl(\{l\}^{(n)} \,\big|\, \{l\}^{(n+1)}\bigr) \,d\{l\}^{(n+1)}\,.$$
# - The stochastic process is not Markovian, so successive increments are not independent. The variance of a larger increment therefore need not equal the sum of the variance of constituent smaller ones; in other words,
#   $$Δ l_{2^{-n+1}}\bigl(Φ\bigr) = Δ l_{2^{-n}}\bigl(Φ\bigr) + Δ l_{2^{-n}}\bigl(Φ+2^{-n}\bigr)$$
#   does *not* imply
#   $$\VV\bigl[Δ l_{2^{-n+1}}\bigl(Φ\bigr)\bigr] = \VV\bigl[Δ l_{2^{-n}}\bigl(Φ\bigr)\bigr] + \VV\bigl[Δ l_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\,.$$
# - Our defining equations make equivalent use of the pre ($Δ l_{2^{-n}}(Φ)$) and post ($Δ l_{2^{-n}}(Φ+2^{-n})$) increments, thus preserving symmetry in $Φ$.
# - Step sizes of the form $2^{-n}$ have exact representations in binary. Thus even small step sizes should not introduce additional numerical errors.

# %% [markdown] user_expressions=[]
# (supp_path-sampling_beta-param-algorithm)=
# ### Formulation of the parameter equations as a root-finding problem

# %% [markdown] user_expressions=[]
# Define
# $$\begin{align}
# r &:= \frac{\lnLtt(Φ+2^{-n}) - \lnLtt(Φ)}{\lnLtt(Φ+2^{-n+1}) - \lnLtt(Φ+2^{-n})} \,; \\
# v &:= \frac{1}{2} \emdstd\bigl(Φ + 2^{-n}\bigr)^2 \,.
# \end{align}$$ (eq_def-r-v_supp) 
# (Definitions repeated from Eqs. \labelcref{eq_def-r-v__r,eq_def-r-v__v}.) The first value, $r$, is the ratio of two subincrements within $Δ l_{2^{-n+1}}(Φ)$.
# Setting $\frac{e^{ψ(α)}}{e^{ψ(β)}} = r$, the two equations we need to solve for $α$ and $β$ can be written
# $$\begin{align}
# ψ(α) - ψ(β) &= \ln r \,; \\
# \ln\bigl[ ψ_1(α) + ψ_1(β) \bigr] &= \ln v \,.
# \end{align}$$ (eq_root-finding-problem_supp)
# (Definitions repeated from Eqs. \labelcref{eq_root-finding-problem__r,eq_root-finding-problem__v}.) Note that these equations are symmetric in $Φ$: replacing $Φ$ by $-Φ$ simply changes the sign on both sides of the first. The use of the logarithm in the equation for $v$ helps to stabilize the numerics.

# %% [markdown] tags=["remove-cell"] user_expressions=[]
# :::{margin}
# ~~To allow solving for multiple $α$, $β$ simultaneously, we solve for a flat vector $\left(\begin{smallmatrix}α\\β\end{smallmatrix}\right)$. Despite the much higher-dimensional space, this reliable enough for our use.~~
# :::

# %%
def f(lnαβ, lnr_v, _array=np.array, _exp=np.exp, digamma=digamma, polygamma=polygamma):
    α, β = _exp(lnαβ).reshape(2, -1)  # scipy's `root` always flattens inputs
    lnr, v = lnr_v
    return _array((
        digamma(α) - digamma(β) - lnr,
        np.log(polygamma(1, α) + polygamma(1, β)) - np.log(v)
    )).flat

def f_mid(lnα, v, _exp=np.exp, polygamma=polygamma):
    "1-d equation, for the special case α=β (equiv to r=1)"
    return np.log(2*polygamma(1, _exp(lnα))) - np.log(v)


# %% [markdown] user_expressions=[]
# :::{margin}  
# This implementation of the Jacobian is tested for both scalar and vector inputs, but fits turned out to be both faster and more numerically stable when they don't use it.
# Therefore we keep it only for reference and illustration purposes.  
# :::

# %% tags=["active-ipynb"]
# def jac(lnαβ, lnr_v):
#     lnα, lnβ = lnαβ.reshape(2, -1)      # scipy's `root` always flattens inputs
#     α, β = np.exp(lnαβ).reshape(2, -1)
#     j = np.block([[np.diagflat(digamma(α)*lnα), np.diagflat(-digamma(β)*lnβ)],
#                      [np.diagflat(polygamma(1,α)*lnα), np.diagflat(polygamma(1,β)*lnβ)]])
#     return j

# %% [markdown] user_expressions=[]
# The functions $ψ$ and $ψ_1$ diverge at zero, so $α$ and $β$ should remain positive. Therefore it makes sense to fit their logarithm: this enforces the lower bound, and improves the resolution where the derivative is highest. The two objective functions (up to scalar shift) are plotted below: the region for low $\ln α$ and $\ln β$ shows sharp variation around $α=β$, suggesting that this area may be most challenging for a numerical optimizer. In practice this is indeed what we observed.
#
# We found however that we can make fits much more reliable by first choosing a suitable initialization point along the $\ln α = \ln β$ diagonal. In practice this means setting $α_0 = α = β$ and solving the 1d problem of Eq. \labelcref{eq_root-finding-problem__v} for $α_0$. (We use the implementation of Brent’s method in SciPy.) Then we can solve the full 2d problem of Eqs. \labelcref{eq_root-finding-problem__r,eq_root-finding-problem__v}, with $(α_0, α_0)$ as initial value. This procedure was successful for all values of $r$ and $v$ we encountered in our experiments.

# %% tags=["active-ipynb", "hide-input"]
# α = np.logspace(-2, 1.2)
# β = np.logspace(-2, 1.2).reshape(-1, 1)
#
# EEa = digamma(α) / (digamma(α) + digamma(β))
# Mvar = 0.5*(polygamma(1, α) + polygamma(1, β))
# domλ = np.array([[(ReJ:=np.real(np.linalg.eigvals(jac(np.stack((lnα, lnβ)), 0))))[abs(ReJ).argmax()]
#                     for lnβ in np.log(β.flat)]
#                    for lnα in np.log(α.flat)])

# %% tags=["active-ipynb", "hide-input"]
# dim_lnα = hv.Dimension("lnα", label=r"$\ln α$")
# dim_lnβ = hv.Dimension("lnβ", label=r"$\ln β$")
# dim_ψα  = hv.Dimension("ψα",  label=r"$ψ(α)$")
# dim_ψ1α = hv.Dimension("ψ1α", label=r"$ψ_1(α)$")
# dim_Eobj = hv.Dimension("Eobj", label=r"$ψ(α)-ψ(β)$")
# dim_lnMvar = hv.Dimension("Mvar", label=r"$\ln \mathrm{{Mvar}}[x_1, x_2]$")  # Doubled {{ because Holoviews applies .format to title
# dim_Reλ = hv.Dimension("Reλ", label=r"$\mathrm{{Re}}(λ)$")
# fig = hv.Curve(zip(np.log(α.flat), digamma(α.flat)), kdims=[dim_lnα], vdims=[dim_ψα], label=dim_ψα.name).opts(title=dim_ψα.label) \
#       + hv.Curve(zip(np.log(α.flat), polygamma(1, α.flat)), kdims=[dim_lnα], vdims=[dim_ψ1α], label=dim_ψα.name).opts(title=dim_ψ1α.label) \
#       + hv.QuadMesh((np.log(α.flat), np.log(β.flat), digamma(α)-digamma(β)),
#                     kdims=[dim_lnα, dim_lnβ],
#                     vdims=[dim_Eobj], label=dim_Eobj.name).opts(title=dim_Eobj.label) \
#       + hv.QuadMesh((np.log(α.flat), np.log(β.flat), np.log(Mvar)),
#                     kdims=[dim_lnα, dim_lnβ],
#                     vdims=[dim_lnMvar], label=dim_lnMvar.name).opts(title=dim_lnMvar.label)
# fig.opts(hv.opts.QuadMesh(colorbar=True, clabel=""))
# fig.opts(fig_inches=3, sublabel_format="", vspace=0.4, backend="matplotlib")
# fig.cols(2);
#
# #glue("fig_polygamma", fig, display=None)

# %% tags=["remove-cell", "active-ipynb"]
# path = config.paths.figuresdir/f"path-sampling_polygamma"
# hv.save(fig, path.with_suffix(".svg"), backend="matplotlib")
# hv.save(fig, path.with_suffix(".pdf"), backend="matplotlib")

# %% [markdown] user_expressions=[]
# :::{figure} ../figures/path-sampling_polygamma.svg
# :name: fig_polygamma
#
# Characterization of the digamma ($ψ$) and trigamma ($ψ_1$) functions, and of the metric variance $\Mvar$.  
# :::

# %% [markdown] user_expressions=[]
# Plotting the eigenvalues of the Jacobian (specifically, the real part of its dominant eigenvalue) in fact highlights three regions with a center at roughly $(\ln α, \ln β) = (0, 0)$. (The Jacobian does not depend on $r$ or $v$, so this is true for all fit conditions). Empirically we found that initializing fits at $(0, 0)$ resulted in robust fits for a large number of $(r,v)$ tuples, even when $r > 100$. We hypothesize that this is because it is difficult for the fit to move from one region to another; by initializing where the Jacobian is small, fits are able to find the desired values before getting stuck in the wrong region.
#
# Note that the color scale is clipped, to better resolve values near zero. Eigenvalues quickly increase by multiple orders of magnitude away from $(0,0)$.
#
# It turns out that the only region where $(0, 0)$ is *not* a good initial vector for the root solver is when $\boldsymbol{r \approx 1}$. This can be resolved by choosing a better initial value along the $(α_0, α_0)$ diagonal, as described above. In practice we found no detriment to always using the 1d problem to select an initial vector, so we use that approach in all cases.

# %% tags=["active-ipynb", "hide-input"]
# fig = hv.QuadMesh((np.log(α.flat), np.log(β.flat), domλ),
#             kdims=[dim_lnα, dim_lnβ], vdims=[dim_Reλ],
#             label="Real part of dom. eig val"
#            ).opts(clim=(-1, 1), cmap="gwv",
#                   colorbar=True)
# #glue("fig_Jac-spectrum", fig, display=False)

# %% tags=["remove-cell", "active-ipynb"]
# path = config.paths.figuresdir/f"path-sampling_jac-spectrum"
# hv.save(fig, path.with_suffix(".svg"), backend="matplotlib")
# hv.save(fig, path.with_suffix(".pdf"), backend="matplotlib")

# %% [markdown] user_expressions=[]
# :::{figure} ../figures/path-sampling_jac-spectrum.svg
# :name: fig_Jac-spectrum
#
# **Objective function has a saddle-point around (0,0)**
# After rewriting Eqs. {eq}`eq_root-finding-problem` in terms of $\ln α$ and $\ln β$, we compute the Jacobian $J$. Plotted is the real part of the eigenvalue $λ_i$ of $J$ for which $\lvert\mathop{\mathrm{Re}}(λ)\rvert$ is largest; this gives an indication of how quickly the fit moves away from a given point.
# In most cases, a root finding algorithm initialized at (0,0) will find a solution.
# :::

# %% [markdown] user_expressions=[]
# (supp_path-sampling_beta-param-special-cases)=
# ### Special cases for extreme values
#
# For extreme values of $r$ or $v$, the beta distribution becomes degenerate and numerical optimization may break. We identify four cases requiring special treatment.
#
# :::::{div} full-width
# ::::{grid}
# :gutter: 3
#
# :::{grid-item-card}
#
# $\boldsymbol{r = 0}$
# ^^^
#
# The corresponds to stating that $Δ l_{2^{-n}}(Φ)$ is infinitely smaller than $Δ l_{2^{-n}}(Φ+2^{-n})$. Thus we set $x_1 = 1$, which is equivalent to setting
#
# $$\begin{aligned}
# Δ l_{2^{-n}}(Φ) &= 0 \,, \\
# Δ l_{2^{-n}}(Φ+2^{-n}) &= \lnLtt(Φ+2^{-n+1}) - \lnLtt(Φ) \,.
# \end{aligned}$$
#
# :::
#
# :::{grid-item-card}
#
# $\boldsymbol{r = 0}$
# ^^^
#
# The converse of the previous case: $Δ l_{2^{-n}}(Φ)$ is infinitely larger than $Δ l_{2^{-n}}(Φ+2^{-n})$. We set $x_1 = 0$.
#
# :::
#
# :::{grid-item-card}
#
# $\boldsymbol{v \to 0}$
# ^^^
#
# As $v$ vanishes, the distribution for $x_1$ approaches a Dirac delta centered on $\tfrac{1}{r+1}$.
# In our implementation, we replace $x_1$ by a constant when $v < 10^{-8}$.
#
# :::
#
# :::{grid-item-card}
#
# $\boldsymbol{v \to \infty}$
# ^^^
#
# Having $v$ go to infinity requires that $α$ and/or $β$ go to $0$ (see Eq. {eq}`eq_root-finding-problem` and {numref}`fig_polygamma`). The probability density of $x_1$ is then a Dirac delta: placed at $x_1=0$ if $α \to 0$, or placed at $x_1 = 1$ if $β \to 0$. If both $α$ and $β$ go to $0$, the PDF must be the sum of two weighted deltas:
# $$p(x_1) = w_0 δ(x_1 - 0) + w_1 δ(x_1 - 1) \,.$$
# The weights $w_i$ can be determined by requiring that
# $$\EE[x_1] = r \,,$$
# which yields
# $$\begin{aligned}
# w_1 &= \frac{r}{r+1}\,, & w_2 &= \frac{1}{r+1} \,.
# \end{aligned}$$
# (For this special case, we revert to writing the condition in terms of a standard (Lebesgue) expectation, since the center (Eq. \labelcref{eq_Aitchison-moments__EE}) is undefined when $α, β \to 0$.)
#
# Since we have already considered the special cases $r = 0$ and $r \to \infty$, we can assume $0 < r < \infty$. Then both $α$ and $β$ are zero, and $x_1$ should be a Bernoulli random variable with success probability $p = w_2 = \frac{1}{r+1}$.
#
# :::
#
# ::::
# :::::

# %% tags=["active-ipynb"]
# def get_beta_rv(r: Real, v: Real) -> Tuple[float]:
#     """
#     Return α and β corresponding to `r` and `v`.
#     This function is not exported: it is used only for illustrative purposes
#     within the notebook.
#     """
#     # Special cases for extreme values of r
#     if r == 0:
#         return scipy.stats.bernouilli(0)  # Dirac delta at 0
#     elif r > 1e12:
#         return scipy.stats.bernouilli(1)  # Dirac delta at 1
#     # Special cases for extreme values of v
#     elif v < 1e-8:
#         return get_beta_rv(r, 1e-8)
#     elif v > 1e4:
#         # (Actual draw function replaces beta by a Bernoulli in this case)
#         return scipy.stats.bernoulli(1/(r+1))
#     
#     # if v < 1e-6:
#     #     # Some unlucky values, like r=2.2715995006941436, v=6.278153793994013e-08,
#     #     # seem to be particularly pathological for the root solver.
#     #     # At least in the case above, the function is continuous at those values
#     #     # (±ε returns very similar values for a and b).
#     #     # Replacing these by nearby values which are more friendly to binary representation
#     #     # seems to help.
#     #     v = digitize(v, rtol=1e-5, show=False)
#     
#     # if 0.25 < r < 4:
#     #     # Special case for r ≈ 1: improve initialization by first solving r=1 <=> α=β
#     # Improve initialization by first solving r=1 <=> α=β
#     x0 = brentq(f_mid, -5, 20, args=(v,))
#     x0 = (x0, x0)
#     # else:
#     #     # Normal case: Initialize fit at (α, β) = (1, 1)
#     #     x0 = (0, 0)
#     res = root(f, x0, args=[math.log(r), v])
#     if not res.success:
#         logger.error("Failed to determine α & β parameters for beta distribution. "
#                      f"Conditions were:\n  {r=}\n{v=}")
#     α, β = np.exp(res.x)
#     return scipy.stats.beta(α, β)

# %% tags=["hide-input"]
def _draw_from_beta_scalar(r: Real, v: Real, rng: RNGenerator, n_samples: int=1,
                           *, _log=math.log, _exp=np.exp, _shape=np.shape
                          ) -> Tuple[float]:
    rng = np.random.default_rng(rng)  # No-op if `rng` is already a Generator
    size = None if n_samples == 1 else (*_shape(r), n_samples)
    # Special cases for extreme values of r
    if r == 0:
        special_val = 1           # EXIT AT END
    elif r > 1e12:
        special_val = 0           # EXIT AT END
    # Special cases for extreme values of v
    elif v < 1e-8:
        special_val = 1 / (1+r)   # EXIT AT END
    elif v > 1e4:
        # Replace beta by a Bernouilli distribution
        return rng.binomial(1, 1/(r+1), size=size)
    
    # Normal case
    else:

        # if v < 1e-6:
        #     # Some unlucky values, like r=2.2715995006941436, v=6.278153793994013e-08,
        #     # seem to be particularly pathological for the root solver.
        #     # At least in the case above, the function is continuous at those values
        #     # (±ε returns very similar values for a and b).
        #     # Replacing these by nearby values which are more friendly to binary representation
        #     # seems to help.
        #     v = digitize(v, rtol=1e-5, show=False)
        
        # if 0.25 < r < 4:
        #     # Special case for r ≈ 1 and 1 < v < 1e5: Initialize on the α=β line
        #     # In this case the initialization (0,0) is unstable, so we 
        # First find a better initialization by solving for the 1d case
        # where r=1 and therefore α=β.
        # (The limits where the normal case fails are around (r=1/3, v=1e4) and (r=3, v=1e4)
        # NB: The values -5 and 20 are slightly beyond the special case limits 5e-8 < v < 1e4 set above;
        #     since also the trigamma function is monotone, this should always find a solution.
        x0 = brentq(f_mid, -5, 20, args=(v,))
        x0 = (x0, x0)
        # else:
        #     # Normal case: Initialize fit at log(α, β) = (1, 1)
        #     x0 = (0., 0.)
        res = root(f, x0, args=[_log(r), v])
        try:
            assert res.success
        except AssertionError:
            logger.error("Failed to determine α & β parameters for beta distribution. "
                         f"Conditions were:\n  {r=}\n  {v=}")
        α, β = _exp(res.x)
        return rng.beta(α, β, size=size)
    
    # Finally, if `size` was passed, ensure result has the right shape
    # NB: We only reach this point if we want through one of the 3 first special cases
    if size:
        return np.array(special_val)[...,None].repeat(n_samples, axis=-1)
    else:
        return special_val

def draw_from_beta(r: Union[Real,Array[float,1]],
                   v: Union[Real,Array[float,1]],
                   rng: Optional[RNGenerator]=None,
                   n_samples: int=1
                  ) -> Tuple[float]:
    """
    Return α, β for a beta distribution with an metric variance `v` and center
    biased by `r`. More precisely, `r` is the ratio of the lengths ``c`` and
    ``1-c``, where ``c`` is the center.
    
    `r` and `v` may either be scalars or arrays
    
    FIXME: Special cases not vectorized
    """
    rng = np.random.default_rng(rng)  # No-op if `rng` is already a Generator
    
    if hasattr(r, "__iter__"):
        return np.array([_draw_from_beta_scalar(_r, _v, rng, n_samples)
                          for _r, _v in zip(r, v)])
    else:
        return _draw_from_beta_scalar(r, v, rng, n_samples)


# %% [markdown] user_expressions=[]
# (supp_path-sampling_example-fitted-beta)=
# ### Examples of different fitted beta distributions

# %% [markdown] user_expressions=[]
# Plotted below are the beta distributions for different values of $r$ and $v$.

# %% tags=["active-ipynb", "hide-input", "full-width"]
# %%opts Curve [title="Fitted beta distributions", ylim=(None,7)]
# %%opts Table [title="Empirical statistics (4000 samples)"]
# %%opts Layout [sublabel_format=""]
#
# curves = {}
# stats = {}
# xarr = np.linspace(0, 1, 400)
# for (r, v), c in zip(
#             [(0.2, 1e-32), (0.2, 1e-16), (0.2, 1e-8), (0.2, 1e-4), (0.2, 1e-2), (0.2, 0.5),
#              (0.5, 0.5), (0.5, 0.1), (0.5, 1),
#              (1, 0.5), (1, 1e1), (1, 30), (1, 50), (1, 70), (1, 1e2), (1, 1e3), (1, 1e5), (1, 1e6),
#              (5, 0.5), (5, 8), (5, 1e4), (5, 2e4), (5, 4e4), (5, 1e6), (5, 1e8), (5, 1e16), (5, 1e32),
#              (6.24122778821756, 414.7130462762959),
#              (2.2715995006941436, 6.278153793994013e-08),
#              (2.271457193328191, 6.075242708902806e-08),
#              (2.269182419251242, 6.794061846449025e-08),
#             ],
#         itertools.cycle(config.figures.colors.bright.cycle)):
#     rv = get_beta_rv(r, v)
#     if isinstance(rv.dist, scipy.stats.rv_discrete):
#         # Dirac delta distribution
#         p, = rv.args
#         if p == 0:
#             α, β = np.inf, 0
#         elif p == 1:
#             α, β = 0, np.inf
#         else:
#             α, β = 0, 0
#     else:
#         # rv is a beta random variable
#         α, β = rv.args
#     # α, β = get_beta_α_β(r, v)
#     x = draw_from_beta(r, v, n_samples=4000)
#     # rv = beta(α, β)
#     # x = rv.rvs(4000)
#     pdf = rv.pmf(xarr) if isinstance(rv.dist, scipy.stats.rv_discrete) else rv.pdf(xarr)
#     curves[(r,v)] = hv.Curve(zip(xarr, pdf), label=f"{r=}, {v=}",
#                              kdims=[hv.Dimension("x1", label="$x_1$")],
#                              vdims=[hv.Dimension("px1", label="p($x_1$)")]  # non-TeX brackets to avoid legend issue 
#                             ).opts(color=c)
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", "invalid value encountered", category=RuntimeWarning)
#         stats[(r,v)] = tuple(f"{y:.3f}" for y in
#                               (α, β, x.mean(), x.std(),
#                                np.exp(digamma(α)) / (np.exp(digamma(α)) + np.exp(digamma(β))),
#                                0.5 * (polygamma(1, α) + polygamma(1, β))
#                               ))
#
# hmap = hv.HoloMap(curves, kdims=["r", "v"])
# dists = hmap.overlay()
# dists.opts(legend_position="right")
# dists.opts(width=500, backend="bokeh")
# dists.opts(fig_inches=7, aspect=2.5, legend_cols=2, backend="matplotlib")

# %% [markdown] tags=["remove-cell", "skip-execution"] user_expressions=[]
# Extra test, for a point that is especially sensitive to numerical issues: despite the very tight distribution, only 10-25% points raise a warning.

# %% tags=["remove-cell", "skip-execution"]
r=2.2691824192512438;    Δr = 0.000000000000001
v=6.79406184644904e-08;  Δv = 1e-22
s=2
rng = np.random.default_rng(45)
for r,v in rng.uniform([r-s*Δr, v-s*Δv], [r+s*Δr, v+s*Δv], size=(40,2)):
    draw_from_beta(r,v)


# %% [markdown] user_expressions=[]
# Statistics for the fitted beta distributions. $\mathbb{E}[x_1]$ and $\mathrm{std}[x_1]$ are computed from 4000 samples. $\mathbb{E}_a[x_1]$ and $\mathrm{Mvar}[x_1,x_2]$ are computed using the expressions above.

# %% tags=["active-ipynb", "hide-input", "full-width"]
# def clean_table_mpl(plot, element):
#     "TODO: Modify table to remove vertical lines"
#     table = plot.handles["artist"]
#
# flat_stats = [k + v for k,v in stats.items()]
# dim_Ex1 = hv.Dimension("Ex1", label="$\mathbb{E}[x_1]$")
# dim_stdx1 = hv.Dimension("stdx1", label="$\mathrm{std}[x_1]$")
# dim_Eax1 = hv.Dimension("Eax1", label="$\mathbb{E}_a[x_1]$")
# dim_Mvar = hv.Dimension("Mvar", label="$\mathrm{Mvar}[x_1,x_2]$")
# stattable = hv.Table(flat_stats, kdims=["r", "v"],
#                      vdims=["α", "β", dim_Ex1, dim_stdx1, dim_Eax1, dim_Mvar])
# # We need to increase max_value_len because Holoviews uses the unformatted
# # length to decide when to truncate
# stattable.opts(max_rows=len(stattable)+1)  # Ensure all rows are shown
# stattable.opts(fig_inches=18, aspect=2.5, max_value_len=30, hooks=[clean_table_mpl], backend="matplotlib")

# %% [markdown] tags=["remove-cell"] user_expressions=[]
# `draw_from_beta` also supports passing `r` and `v` as vectors. This is mostly a convenience: internally the vectors are unpacked and $(α,β)$ are solved for individually.

# %% tags=["active-ipynb", "skip-execution", "remove-cell", "full-width"]
# r_vals, v_vals = np.array(
#     [(0.2, 1e-32), (0.2, 1e-16), (0.2, 1e-8), (0.2, 1e-4), (0.2, 1e-2), (0.2, 0.5),
#      (0.5, 0.5), (0.5, 0.1), (0.5, 1),
#      (1, 0.5), (1, 1e1), (1, 1e2), (1, 1e3), (1, 1e5), (1, 1e6),
#      (5, 0.5), (5, 8), (5, 1e4), (5, 2e4), (5, 4e4), (5, 1e6), (5, 1e8), (5, 1e16), (5, 1e32)]
# ).T
#
# # α, β = get_α_β(r_vals, v_vals)
# # rv = beta(α, β)
# # x = rv.rvs((4000, len(r_vals)))
# x = draw_from_beta(r_vals, v_vals, n_samples=4000)
# flat_stats = np.stack((r_vals, v_vals, x.mean(axis=-1), x.std(axis=-1)
#                        #np.exp(digamma(α)) / (np.exp(digamma(α)) + np.exp(digamma(β))),
#                        #0.5 * (polygamma(1, α) + polygamma(1, β))
#                       )).T
# stattable = hv.Table(flat_stats, kdims=["r", "v"],
#                      vdims=[dim_Ex1, dim_stdx1])
# stattable.opts(max_rows=len(stattable)+1)
# stattable.opts(fig_inches=14, aspect=1.8, max_value_len=30, hooks=[clean_table_mpl], backend="matplotlib")

# %% [markdown] user_expressions=[] tags=["remove-cell"]
# ### Timings for the root solver

# %% [markdown] tags=["remove-cell"]
# Reported below are timings for different numbers of fits, comparing a “loop” approach where `get_α_β` is called for each pair `(r, v)`, and a “vectorized” approach where `r` and `v` are passed as vectors.
#
# At the moment there is no clear benefit to using the vectorized form; this is likely because it performs the fit in a much higher dimensional space, and it must continue calculations with this large vector until all equations are solved.
#
# NB: Timings were done for an older version, where the function returned the $α, β$ parameters rather than a beta random variable. This function also performed vectorized operations by enlarging the fit dimension, rather than the current approach of looping over $(r, v)$ pairs. The observation that this approach was in general no faster than looping partly motivated the change, so we keep these timings results as documentation.

# %% [markdown] tags=["remove-cell"]
# ```python
# time_results = []
# for L in tqdm([1, 7, 49, 343]):
#     r_vals = np.random.uniform(low=0, high=1, size=L)
#     v_vals = np.random.exponential(3, size=L)
#     [get_α_β(r, v) for r, v in zip(r_vals, v_vals)]
#     res_loop = %timeit -q -o [get_α_β(r, v) for r, v in zip(r_vals, v_vals)]
#     res_vec = %timeit -q -o get_α_β(r_vals, v_vals)
#     time_results.append((L, res_loop, res_vec))
#
# def time_str(time_res): s = str(time_res); return s[:s.index(" per loop")]
# time_table = hv.Table([(L, time_str(res_loop), time_str(res_vec))
#                        for L, res_loop, res_vec in time_results],
#                       kdims=["# fits"], vdims=["loop", "vectorized"])
# time_table.opts(aspect=4, fig_inches=7)
# ```

# %% [markdown] tags=["remove-cell"]
# | # fits |             loop |        vectorized |
# |-------:|-----------------:|------------------:|
# |      1 | 555 μs ± 14.2 μs |  515 μs ± 20.2 μs |
# |      7 | 4.06 ms ± 108 μs |    1.8 ms ± 20 μs |
# |     49 | 28.5 ms ± 589 μs |  17.1 ms ± 146 μs |
# |    343 | 187 ms ± 2.15 ms | 3.95 ms ± 33.1 ms |

# %% [markdown] tags=["remove-cell"]
# The test above samples $r$ from the entire interval $[0, 1]$, but we get similar results when restricting values to the “easy” region $[0.4, 0.6]$. Reducing the values of $v$ (by sampling from a distribution with lighter tail) does bring down the execution time of the vectorized approach. This is consistent with the hypothesis that a few especially difficult $(r,v)$ combinations are slowing down the computations.

# %% [markdown] tags=["remove-cell"]
# ```python
# time_results2 = []
# for L in tqdm([1, 7, 49, 343]):
#     r_vals = np.random.uniform(low=0.4, high=.6, size=L)
#     v_vals = np.random.exponential(1, size=L)
#     [get_α_β(r, v) for r, v in zip(r_vals, v_vals)]
#     res_loop = %timeit -q -o [get_α_β(r, v) for r, v in zip(r_vals, v_vals)]
#     res_vec = %timeit -q -o get_α_β(r_vals, v_vals)
#     time_results2.append((L, res_loop, res_vec))
#
# def time_str(time_res): s = str(time_res); return s[:s.index(" per loop")]
# time_table = hv.Table([(L, time_str(res_loop), time_str(res_vec))
#                        for L, res_loop, res_vec in time_results2],
#                       kdims=["# fits"], vdims=["loop", "vectorized"])
# time_table.opts(aspect=4, fig_inches=7)
# ```

# %% [markdown] tags=["remove-cell"]
# | # fits |              loop |        vectorized |
# |-------:|------------------:|------------------:|
# |      1 |  474 μs ± 12.3 μs |  452 μs ± 2.62 μs |
# |      7 | 3.92 ms ± 43.7 μs | 1.65 ms ± 22.9 μs |
# |     49 |  29.2 ms ± 167 μs | 16.6 ms ± 84.8 μs |
# |    343 |  214 ms ± 4.27 ms | 1.37 ms ± 6.24 ms |

# %% [markdown]
# (supp_path-sampling_implementation)=
# ## Implementation

# %% [markdown]
# ### Generate a single path
#
# Now that we know how to construct an sampling distribution for the increments, sampling an entire path is just a matter of repeating the process recursively until we reach the desired resolution.

# %%
def generate_path_binary_partition(
        Phi: Array[float,1], ltilde: Array[float,1], sigmatilde: Array[float,1],
        lstart: float, lend: float, res: int=7, rng=None
    ) -> Tuple[Array[float,1], Array[float,1]]:
    """
    Returned path has length ``2**res + 1``.
    If `ltilde` and`sigmatilde` have a different length, they are linearly-
    interpolated to align with the returned array `Φhat`.
    Typical values of `res` are 6, 7 and 8, corresponding to paths of length
    64, 128 and 256. Smaller values may be useful to accelerate debugging. Larger values
    increase the computation cost with (typically) negligible improvements in accuracy.
    
    Parameters
    ----------
    Phi: Values of Φ at which `ltilde` and `sigmatilde` are evaluated.
    ltilde: Vector of means for the Gaussian process.
    sigmatilde: Vector of standard deviations for the Gaussian process.
    res: Returned paths have length ``2**res``.
       Typical values of `res` are 6, 7 and 8, corresponding to paths of length
       64, 128 and 256. Smaller may be useful to accelerate debugging, but larger
       values are unlikely to be useful.
    seed: Any argument accepted by `numpy.random.default_rng` to initialize an RNG.
    
    Returns
    -------
    The pair Φhat, lhat.
        Φhat: Array of equally spaced values between 0 and 1, with step size ``2**-res``.
        lhat: The generated path, evaluated at the values listed in `Φhat`.
    """
    # Validation
    res = int(res)
    if not (len(Phi) == len(ltilde) == len(sigmatilde)):
        raise ValueError("`Phi`, `ltilde` and `sigmatilde` must all have "
                         "the same shape. Values received have the respective shapes "
                         f"{np.shape(Phi)}, {np.shape(ltilde)}, {np.shape(sigmatilde)}")
    if res <= 1:
        raise ValueError("`res` must be greater than 1.")
    rng = np.random.default_rng(rng)  # No-op if `rng` is already a Generator
    # Interpolation
    N  = 2**res + 1
    Φhat = np.linspace(Phi[0], Phi[-1], N)
    if not np.array_equal(Phi, Φhat):  # NB: This condition doesn't depend on Phi being sorted or regularly spaced
        ltilde = np.interp(Φhat, Phi, ltilde)
        sigmatilde = np.interp(Φhat, Phi, sigmatilde)
    # Pre-computations
    sigma2tilde = sigmatilde**2
    # Algorithm
    lhat = np.empty(N)
    lhat[0] = lstart
    lhat[-1] = lend
    for n in range(1, res+1):
        Δi = 2**(res-n)
        i = np.arange(Δi, N, 2*Δi)  # Indices for the new values to insert
        d = lhat[i+Δi] - lhat[i-Δi] # Each pair of increments must sum to `d`
        r = (ltilde[i] - ltilde[i-Δi]) / (ltilde[i+Δi]-ltilde[i])  # Ratio of first/second increments
        v = 0.5*sigma2tilde[i]
        x1 = draw_from_beta(r, v, rng=rng)
        lhat[i] = lhat[i-Δi] + d * x1
    return Φhat, lhat


# %% tags=["active-ipynb", "hide-input"]
# Φtilde = np.linspace(0.01, 1, 20)
# ltilde = np.log(Φtilde)
# σtilde = 0.5*np.ones_like(Φtilde)
# Φhat, lhat = generate_path_binary_partition(
#     Φtilde, ltilde, 3*σtilde, lstart=np.log(Φtilde[0]), lend=np.log(Φtilde[-1]),
#     res=8, rng=None)
#
# cycle = config.figures.colors.bright.cycle
# curve_ltilde = hv.Curve(zip(Φtilde, ltilde), label=r"$\tilde{l}$", kdims=["Φ"])
# curve_lhat = hv.Curve(zip(Φhat, lhat), label=r"$\hat{l}$", kdims=["Φ"])
# fig = curve_ltilde.opts(color=cycle[0]) * curve_lhat.opts(color=cycle[1])
# fig.opts(ylabel="")

# %% [markdown]
# ### Generate ensemble of sample paths
#
# :::{Note} This is the only public function exposed by this module
# :::
#
# To generate $R$ paths, we repeat the following $R$ times:
# 1. Select start and end points by sampling $\nN(\tilde{Φ}[0], \lnLtt{}[0])$ and $\nN(\tilde{Φ}[-1], \lnLtt{}[-1])$.
# 2. Call `generate_path_binary_partition`.

# %%
def generate_quantile_paths(R: int, Phi: Array[float,1],
                            ltilde: Array[float,1], sigmatilde: Array[float,1],
                            res: int=7, seed=None,
                            *, progbar: Union[Literal["auto"],None,tqdm,"mp.queues.Queue"]="auto",
                            previous_R: int=0
                           ) -> Generator[Tuple[Array[float,1], Array[float,1]], None, None]:
    """
    Generate `R` distinct quantile paths, with trajectory and variability determined
    by `ltilde` and `sigmatilde`.
    Paths are generated using the binary partition algorithm, with normal distributions
    for the end points and beta distributions for the increments.
    
    Returned paths have length ``2**res + 1``.
    If `ltilde` and`sigmatilde` have a different length, they are linearly-
    interpolated to align with the returned array `Φhat`.
    Typical values of `res` are 6, 7 and 8, corresponding to paths of length
    64, 128 and 256. Smaller values may be useful to accelerate debugging. Larger values
    increase the computation cost with (typically) negligible improvements in accuracy.
    
    .. Note:: When using multiprocessing to call this function multiple times,
       use either a `multiprocessing.Queue` or `None` for the `progbar` argument.
    
    Parameters
    ----------
    R: Number of paths to generate.
    Phi: Values of Φ at which `ltilde` and `sigmatilde` are evaluated.
    ltilde: Vector of means for the Gaussian process.
    sigmatilde: Vector of standard deviations for the Gaussian process.
    res: Returned paths have length ``2**res + 1``.
       Typical values of `res` are 6, 7 and 8, corresponding to paths of length
       64, 128 and 256. Smaller may be useful to accelerate debugging, but larger
       values are unlikely to be useful.
    seed: Any argument accepted by `numpy.random.default_rng` to initialize an RNG.
    progbar: Control whether to create a progress bar or use an existing one.
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

    previous_R: Used only to improve the display of the progress bar:
       the indicated total on the progress bar will be `R` + `previous_R`.
       Use this when adding paths to an 

    Yields
    ------
    Tuples of two 1-d arrays: (Φhat, lhat).
    
    Notes
    -----
    Returned paths always have an odd number of steps, which as a side benefit is
    beneficial for integration with Simpson's rule.
    """
    rng = np.random.default_rng(seed)
    total = R + previous_R
    progbar_is_queue = ("multiprocessing.queues.Queue" in str(type(progbar).mro()))  # Not using `isinstance` avoids having to import multiprocessing & multiprocessing.queues
    if isinstance(progbar, str) and progbar == "auto":
        progbar = tqdm(desc="Sampling quantile paths", leave=False,
                       total=total)
    elif progbar is not None and not progbar_is_queue:
        progbar.reset(total)
        if previous_R:
            # Dynamic miniters don’t work well with a restarted prog bar: use whatever miniter was determined on the first run (or 1)
            progbar.miniters = max(progbar.miniters, 1)
            progbar.dynamic_miniters = False
            progbar.n = previous_R
            progbar.refresh()
    for r in range(R):
        for _ in range(100):  # In practice, this should almost always work on the first try; 100 failures would mean a really pathological probability
            lstart  = rng.normal(ltilde[0] , sigmatilde[0])
            lend = rng.normal(ltilde[-1], sigmatilde[-1])
            if lstart < lend:
                break
        else:
            raise RuntimeError("Unable to generate start and end points such that "
                               "start < end. Are you sure `ltilde` is compatible "
                               "with monotone paths ?")
        Φhat, lhat = generate_path_binary_partition(
            Phi, ltilde, sigmatilde, lstart=lstart, lend=lend,
            res=res, rng=rng)
        
        yield Φhat, lhat
        
        if progbar_is_queue:
            progbar.put(total)
        elif progbar is not None:
            progbar.update()
            time.sleep(0.05)  # Without a small wait, the progbar might not update

# %% [markdown]
# ### Usage example

# %% tags=["active-ipynb"]
# Φtilde = np.linspace(0.01, 1, 20)
# ltilde = np.log(Φtilde)
# σtilde = 0.5*np.ones_like(Φtilde)
#
# colors = cycle = config.figures.colors.bright
#
# curves_lhat = []
# for Φhat, lhat in generate_quantile_paths(10, Φtilde, ltilde, 3*σtilde,
#                                  res=8, seed=None):
#     curves_lhat.append(hv.Curve(zip(Φhat, lhat), label=r"$\hat{l}$", kdims=["Φ"])
#                        .opts(color=colors.grey))
# curve_ltilde = hv.Curve(zip(Φtilde, ltilde), label=r"$\tilde{l}$", kdims=["Φ"]) \
#                         .opts(color=colors.blue)
#
# hv.Overlay((*curves_lhat, curve_ltilde)).opts(ylabel="")

# %% tags=["remove-input", "active-ipynb"]
# GitSHA()
