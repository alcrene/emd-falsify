# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
# # Old EMD implementation
#
# $\renewcommand{\EE}{\mathbb{E} \,}
# \renewcommand{\VV}{\mathbb{V} \,}
# \renewcommand{\nN}{\mathcal N}
# \renewcommand{\T}{{\displaystyle\stackrel{\intercal}{\phantom{.}}}}
# \renewcommand{\SE}{\mathop{\mathrm{SE}}}  % Standard Error
# \renewcommand{\MNA}{\mathcal{M}^{\mathcal{N}}_A}
# \renewcommand{\lasz}{l_{\smash{{}^{*}_a},z}}
# \renewcommand{\laez}{l_{\smash{{}^{e}_a},z}}
# \renewcommand{\lasez}{l_{\smash{{}^{*e}_a},z}}
# \renewcommand{\Philsaz}{Φ_{\smash{{}^{*}_a},z}} % Φ_{\lasz}
# \renewcommand{\Phileaz}{Φ_{\smash{{}^{e}_a},z}} % Φ_{\laez}
# \renewcommand{\Philseaz}{Φ_{\smash{{}^{*e}_a},z}} % Φ_{\laez}
# \renewcommand{\paez}{p_{\smash{{}^{e}_a},z}}
# \renewcommand{\lhatasz}{\hat{l}_{\smash{{}^{*}_a},z}}
# $

# %% tags=["remove-input"]
import math
import logging
import numpy as np
#from scipy.special import erf
from scipy import stats
from scipy.integrate import simpson
from metalogistic import MetaLogistic
from metalogistic.algebra import QuantileSum

# %% tags=["remove-input"]
from collections.abc import Callable, Iterator, Iterable
from typing import Optional, Union, Any, List, Tuple
from scityping.numpy import Array
FloatArray = Array[np.inexact]

# %% tags=["remove-input"]
logger = logging.getLogger("emdd.emd")

# %% tags=["active-ipynb", "remove-input"]
# from emdd import models, config
# from emdd.utils import glue, SeedGenerator, get_bounds
# from emdd.digitize import digitize
# from dataclasses import replace

# %% tags=["active-py"]
from .digitize import digitize


# %% tags=["active-ipynb", "hide-input"]
# import holoviews as hv
# hv.extension("bokeh")

# %% [markdown]
#
#
# To illustrate the algorithm for computing EMD, we recreate the two exponentials example and evaluate the EMD on the correct model (model $\MNA$). These conditions we select are challenging: since the model is correct, any differences should be due to either finite samples or numerical errors. Accuracy is especially important in correct model condition, since any residual errors in this situation sets a floor for the resolution in the comparison to any other models.
#
# Two regimes are of interest:
# - In the low data regime (e.g. 40 points), fitting errors are substantial, and we are mostly concerned with keeping calculations valid (avoiding `NaN` and `inf` values).
# - In the high data regime (e.g. 4000 points), we want to ensure that the fitted CDF converges to the true CDF with very high accuracy.
#
# We apply different numerical tricks to ensure good results in both regimes. The examples below use a datasets of {glue:text}`emd-implementation-L` points.

# %% tags=["active-ipynb", "hide-input"]
# L_lst   = [40, 4000]; glue("emd-implementation-L", L_lst, display=False, print_name=False)  # Number of samples
# R_Φ = 6   # Number of c.d.f. path realizations to average over -- this small number is for illustration only
# c   = 1   # Global uncertainty scaling parameter
#
# MX_model = models.UniformX
# xrange = (0, 3)
#
# MN_model = models.PerfectExpon
# λ_true = 1
# λ_alt = 1.1
#
# Me_model = models.GaussianError
# σ = 0.2
#
# class SeedGen(SeedGenerator):
#     MX_data : int
#     MX_synth: int
#     Me_data: int
#     Me_true: int
#     Me_alt : int    
# seeds = SeedGen(entropy=config.random.entropy)(0)
#
# MX_data = MX_model(*xrange, seed=seeds.MX_data)
# MX_synth = MX_model(*xrange, seed=seeds.MX_synth)
# MNA = MN_model(λ=λ_true)
# MNB = MN_model(λ=λ_alt)
# Me_data = Me_model(σ=σ, seed=seeds.Me_data)
# MeA = replace(Me_data, seed=seeds.Me_true)
# MeB = replace(Me_data, seed=seeds.Me_alt)
#
# xdata = {L: MX_data(L) for L in L_lst}
# ydata = {L: Me_data(xdata, MNA(xdata[L])) for L in L_lst}

# %% [markdown]
# ## Fitting the synthetic CDF $Φ_{\smash{{}^{e}_a},z}(l)$
#
# For some simple models, it is possible to simply write down $\Phileaz$. However, since we already have a generative model, a general (and presumably cheap) solution is simply to sample data points and fit a metalog, in the same way we do for the mixed CDF $\Philseaz$. We can increase the number of data points until a given tolerance is reached.
# (Note however that perfect reproduction can generally be achieved only with infinite number of terms in the metalog and infinite data samples, and that increasing both shows diminishing returns. Increasing the number of terms may also *increase* artifacts in the analytic derivatives.)
#
# :::{important}
# In most cases of interest, the log likelihood is not bounded from below. This means a) that the quantile function goes to $-\infty$ as $Φ \to 0$; and b) that no matter how many points we sample, values at the low end will be sparse. (As more samples are drawn, the lowest computed $\lasz$ gets ever lower.) Since we use ranking to estimate the CDF, sparsely distributed values contribute more noise than information – they make the fits less consistent (i.e. sensitive to the particular draw of samples), and sometimes contribute to them failing altogether.
#
# Our current approach to mitigate this is to discard the lowest 0.5% of values, and let the metalog extrapolate the lower tail.
# :::

# %% [markdown]
# :::{prf:algorithm} fit_Φeaz
# :label: alg-emp-CDF
#
# **Inputs**  
# ~ `MX`  – Generative model for the independent variable.  
# ~ `MN`  – Generative natural model. If $Z$ is the natural variable and $X$ the independent, then `MN` samples from $p(Z | X)$.  
# ~ `Me`  – Generative experimental model: $p(Y | Z, X)$.  
# ~ `l`  – Function $(y, z) \mapsto \log p(y | z)$.  
# ~ `nterms`  – Number of terms to use in the metalog.  
# ~ `tol_synth`  – The data set size is increased until this tolerance is reached.
# ~ `fit_attempts`  – Number of times to increase the data set size by sampling more data points from `MX` and `MN`.  
# ~ `dataset_size_step` ($L$) – Number of samples to add each time we increase data set size.  
# ~ `never_fail`  – If true and the tolerance `tol_synth` is not achieved, print a warning instead of raising `RuntimeError`.
#
# **Outputs**   
# ~ $\paez$  – Random variable for $\laez$ (`Metalogistic` instance). Provides methods `pdf`, `logpdf`, `cdf`, etc.
#
# 1. $\begin{aligned}[t]
#     \mathtt{err} &\leftarrow \infty \\
#     \vec{l} &= ()
#     \end{aligned}$
# 2. While $\mathtt{err} > \mathtt{tol-synth}$:
#     1. $\begin{aligned}[t]
#         \vec{x} &\stackrel{\text{unif}}{\sim} \mathtt{MX} & \text{(sample }L\text{ times)} \\
#         \vec{z} &\leftarrow \mathtt{MN}(\vec{x}) & \text{(may involve sampling if }\mathtt{MN}\text{ is stochastic)} \\
#         \vec{y} &\stackrel{\text{unif}}{\sim} \mathtt{Me}(\vec{y}, \vec{x}) \\
#        \end{aligned}$
#     2. Append $\vec{l}$ with $l(\vec{y}, \vec{z})$.
#     2. (Re)sort $\vec{l}$: $\quad\vec{l} \leftarrow \mathtt{sort}(\vec{l})$
#     3. Since we sampled uniformly, we can approximate the cumulative probabilities of the $\vec{l}$ by their rank:  
#     $\tilde{\vec{Φ}} \leftarrow \left( \frac{1}{L_{\mathrm{synth}}+1},  \frac{2}{L_{\mathrm{synth}}+1}, \dotsc,  \frac{L}{L+1} \right)^\T$
#     4. Obtain $\Phileaz$, $\paez$ by fitting an upper-bounded metalog distribution to the set of points $\left\{({l}_i, \tilde{{Φ}}_i)\right\}_{i=1}^{L}$.  
#     5. Evaluate the mean squared error between the fitted CDF and the data:  
#        $\displaystyle \mathtt{err} \leftarrow \sqrt{\int \bigl(\Phileaz(l) - \tilde{Φ}(l)\bigr)^2 \, dl}$  
#        where the integral is approximated from the discrete $\tilde{Φ}_i$ with the trapezoidal rule.
# 3. Return $\paez$  
# :::

# %% tags=["hide-input"]
def fit_Φeaz(MX, MN, Me, l,
             nterms: int, tol_synth: float, fit_attempts=10,
             dataset_size_step=4000, never_fail: bool=False):
    # The fitted quantile function will be evaluated on the CDF values given by Φarr
    # Using a 'sane' dt is almost equivalent to doing
    #    Φarr = np.linspace(0.001,0.999, 1000)
    # except that dt is guaranteed to have an exact base-2 representation,
    # and therefore may lead to fewer rounding errors.
    dt = digitize(0.001, show=False)
    Φarr = np.arange(1, 1000)*dt
    # Estimate synthetic Φe from empirical CDF by fitting a metalogistic.
    le_arr = []
    #ze_arr = []
    err_lst = []
    
    for fit_attempt in range(fit_attempts):
        # Generate new data points and compute their l
        xe_arr = MX(dataset_size_step)
        _ze_arr = MN(xe_arr)
        _le_arr = l(Me(xe_arr, _ze_arr), _ze_arr)
        # Insert them into le_arr, and ensure the whole thing is sorted
        idxsort = np.argsort(_le_arr)
        _le_arr = _le_arr[idxsort]
        #_ze_arr = _ze_arr[idxsort]
        idxinsert = np.searchsorted(le_arr, _le_arr)
        le_arr = np.insert(le_arr, idxinsert, _le_arr)
        # ze_arr = np.insert(ze_arr, idxinsert, _ze_arr)
        # Fit the metalog
        Le = len(le_arr); cdf_ps = np.arange(1, Le+1)/(Le+1)
        # Discard the first 0.5%: log likelihood is unbounded from below, so the lowest values never grow dense
        imin = max(1, int(0.005*Le))
        pe = MetaLogistic(cdf_xs=le_arr[imin:], cdf_ps=cdf_ps[imin:],
                          term=nterms, fit_method="Linear least squares",
                          ubound=True)           
        # Next check that the mean error is below the prescribed tolerance
        err = np.sqrt(np.trapz((cdf_ps - pe.cdf(le_arr))**2) / Le)
        # If fit error is below tolerance, terminate
        if err < tol_synth:
            if fit_attempt > 0:
                logger.info(f"Required {Le} samples to achieve target accuracy of the `le` fit. "
                            f"(Attempt #{fit_attempt+1})")
            break
        err_lst.append((Le, err))
    else:
        err_lst_title = "  No. synthetic terms      Fit error\n"
        err_lst_str   = '\n'.join(f"  {n:^19}      {e:.8f}"
                                  for n, e in err_lst)
        # if np.diff(pe.quantile(Φarr)).min() <= 0:
        #     monotone = "\nFitted CDF was not monotone."
        #     import pdb; pdb.set_trace()
        # else:
        #     monotone = ""
        msg = ("Increasing the number of synthetic points does "
               "not seem enough to bring the error on Φe below "
               "the requested threshold.\nSometimes this can be "
               "solved by increasing the number of terms in "
               "the metalog approximation; currently "
               f"this is set to use {nterms} terms.\n"
               f"  Tolerance: {tol_synth}\n"
               f"{err_lst_title}{err_lst_str}"
               #f"{monotone}"
              )
        if never_fail:
            logger.warning(msg)
        else:
            raise RuntimeError(msg)
        
    return pe


# %% [markdown]
# ## Typical shape of a likelihood CDF
#
# A Gaussian likelihood has a maximum but no minimum; therefore we can expect $l(Φ)$ to be unbounded below but bounded above. The likelihood also becomes very peaked around its maximum as the number of points $L$ is increased, so we further expect the highest values of $l$ to be very rare – this translates to a very flat $l(Φ)$ curve near $Φ=1$. Empirically this is indeed what we observe:

# %% tags=["active-ipynb", "hide-input"]
# larr, cdf_ps, pe, quantsum_e, quantsum_se, lep, lsep, lepp, lsepp = {}, {}, {}, {}, {}, {}, {}, {}, {}
# for key in xdata.keys():
#     _zarr = MNA(xdata[key])
#     _larr = MeA.logpdf(ydata[key], _zarr); σ = np.argsort(_larr);
#     _larr = _larr[σ];
#
#     _pe = fit_Φeaz(MX_synth, MNA, MeA, MeA.logpdf,
#                   nterms=9, tol_synth=0.05)#tol_synth=0.008)
#     L = len(_larr);
#     pe[key] = _pe  # We only need to keep `_pe` for the tests at the very end of this notebook
#     cdf_ps[key] = np.arange(1, L+1)/(L+1)
#
#     pse = MetaLogistic(cdf_xs=_larr, cdf_ps=cdf_ps[key],
#                        term=9, fit_method="Linear least squares",
#                        ubound=True)  # See comments below for why we use ubound=True
#
#     larr[key] = _larr
#     quantsum_e[key] = QuantileSum.from_metalogistic(_pe)
#     quantsum_se[key] = QuantileSum.from_metalogistic(pse)
#     lep[key]  = quantsum_e[key].diff()  # 'p' for 'prime' => 1st derivative
#     lsep[key] = quantsum_se[key].diff()
#     lepp[key]  = lep[key].diff()
#     lsepp[key] = lepp[key].diff()

# %% tags=["active-ipynb", "remove-input"]
# Φarr, le_, lep_, lepp_, lse_, lsep_, lsepp_ = {}, {}, {}, {}, {}, {}, {}
# for key in xdata.keys():
#     # Using a 'sane' dt is almost equivalent to doing
#     #    Φarr = np.linspace(0.01,0.99, 1000)
#     # except that dt is guaranteed to have an exact base-2 representation,
#     # and therefore may lead to fewer rounding errors.
#     dt = digitize(0.01, show=False)
#     _Φarr = np.arange(1, 100)*dt
#
#     Φarr[key] = _Φarr
#     le_[key] = quantsum_e[key](_Φarr)
#     lep_[key] = lep[key](_Φarr)
#     lepp_[key] = lepp[key](_Φarr)
#
#     lse_[key] = quantsum_se[key](_Φarr)
#     lsep_[key] = lsep[key](_Φarr)
#     lsepp_[key] = lsepp[key](_Φarr)

# %% [markdown]
# :::{note}  
# Since the metalog distribution is defined by its quantile function (the reciprocal of the CDF), this is what we report. This is also easier to work with in the vicinity $Φ \to 1$, where the quantile function becomes nearly flat.  
# :::

# %% [markdown]
# The figure below shows the quantile functions for both the synthetic ($\laez$) and mixed ($\lasez$) log likelihood. The mixed log likelihood CDF is fitted to a finite number of samples, making it much more prone to artifacts, especially at the edges.

# %% tags=["active-ipynb", "remove-input"]
# frames = {}
# for key in xdata.keys():
#     panels = {}
#
#     dims=dict(kdims=["Φ"], vdims=["l"])
#     panels["l"] = hv.Curve(zip(Φarr[key], le_[key]), **dims, label="le") \
#     *hv.Curve(zip(Φarr[key], lse_[key]), **dims, label="l*e")
#     panels["l"].opts(legend_position="bottom_right")
#
#     dims=dict(kdims=["Φ"], vdims=["l'"])
#     panels["l'"] = hv.Curve(zip(Φarr[key], lep_[key]), **dims, label="l'e") \
#     *hv.Curve(zip(Φarr[key], lsep_[key]), **dims, label="l'*e")
#
#     dims=dict(kdims=["Φ"], vdims=["l''"])
#     panels["l''"] = hv.Curve(zip(Φarr[key], lepp_[key]), **dims, label="l''e") \
#     *hv.Curve(zip(Φarr[key], lsepp_[key]), **dims, label="l''*e")
#
#     frames[key] = panels["l"] + panels["l'"] + panels["l''"]
# hv.HoloMap(frames, kdims=["L"]).collate()

# %% [markdown]
# :::{NOTE}  
# Despite the fact that derivatives are analytic, there are large deviations between the second derivatives of $\laez$ and $\lasez$, even in ideal cases where we have many (10’000) good quality data points. Thus the second derivative $\lasez''(Φ)$ must not be used in calculations, as it cannot be estimated reliably with this method. 
# Partly for this reason, our approach only uses the first derivative.  
# :::

# %%

# %%

# %% [markdown]
# ## Removing artifacts in the CDF
#
# The reliability of the metalog fit drops at the edges near 0 and 1 – in particular, the quantile function may not be monotone. This should be avoidable for the estimate of $l_e$, since it describes a proper likelihood, and one can always get a generate more data points from the model. However, for the estimate of $l_{*e}$, which mixes data and model probabilities, and only has access to finite data points, pathologies are easily encountered.
# Moreover, as is often the case with polynomial fits, these pathologies can be amplified in the analytic derivatives. Derivatives obtained by finite differences may be less precise analytic ones, but they do not suffer from polynomial ringing – therefore we apply the following rules to remove egregious artifacts:
#
# - We use the upper-bounded form of the metalog, a.k.a. the log-metalog. This is justified by the fact that the likelihood should have a global maximum, and dramatically reduces polynomial artifacts around 1. (Examples above already do this.)
# - The log-metalog is fit by minimizing the mean squared error over all given points. Therefore, by fitting a second log-metalog on only the points in the interval $[0.8, 1)$, we can mitigate edge artifacts since the function can go where it likes for $Φ < 0.8$. We then stitch this upper-interval function with the original one (setting the switchover point at 0.9), so we get a good fit over the whole range $(0, 1)$.
#   + This effect matters only when we have a lot of data points (≥2000). With fewer than ~2000 data points, a split-fit is actually detrimental.
#   + Our understanding is that with a lot of points in the bulk region $(0.2, 0.8)$, minimizing the squared error sacrifices some fit accuracy in the tails in order to better fit the bulk. Fundamentally, this is due to the limited flexibility of a 9-term metalog, which forces the least-squares fitting to make a compromise.
#   + Another solution may be to give more weight to the points in the upper tail; we have not investigated whether this would have advantages over the split-fit approach. Re-weighting would still be limited by the flexibility of a 9-term metalog.
#   + Even with this correction, CDF fitting errors remain one of the principal limitations on the method.
# - After being fit to data, $l(Φ)$ is made monotone:
#   + It is not clear whether a single best interpretation exists for a decreasing CDF, which is mainly a numerical artifact. For a well-defined CDF, $ΔΦ(l) = Φ(l)-Φ(l-Δl)$ is the probability mass associated to the interval $[l-Δl,l)$. We generalize this interpretation, associating each discrete value $l_i$ the probability mass $ΔΦ(l_i)$.
#   + In this view, a decreasing CDF simply corresponds an incorrect sorting of the values $\{l_i\}$. A monotone $l(Φ)$ is therefore recovered by sorting the pairs $(l_i, ΔΦ(l_i))$ in ascending order of $l_i$, and then accumulating the $ΔΦ$ to create a new $Φ$ array.
#   + This sorting approach seems to work better than clamping $l(Φ)$ to make it monotone: having $l'(Φ) = 0$ over some interval is problematic, since the equation for $δ$ involves $\log l'(Φ)$.
#   + Worries that this approach is heavy-handed can be somewhat mitigated by that fact that as the number of data points increases and the CDF fit improves, non-monoticity tends to disappear.
# - For $l'$ and $l''$, we compare the analytical values to those obtained by centered differences. If they are close, the analytical value is taken (more precise); otherwise, the numerical derivative is taken (more robust).
#
# Thus we use two postprocessing functions to remove artifacts in the CDF and its derivatives:
# - `make_monotone`, which sorts the ordinate values;
# - `robustify`, which pointwise selects between accurate and robust functions.

# %% [markdown]
# :::{hint}  
# Especially with regards to identifying and fixing artifacts, we found working with the quantile function somewhat easier than with the CDF, and much easier than with the PDF.  
# :::

# %% [markdown]
# ### Re-fitting the CDF on just the upper region
#
# If there are enough values that at least 5 values have CDF ≥ 0.8, we use a specialized fit for the high CDF region. We typically found that 9 terms was the best compromise between accuracy and polynomial artifacts, but the since the metalog cannot have more terms than CDF values, we may use as little as 5 terms.
# (The threshold of 5 terms was not particularly investigated; other values may be better, although with so few data points, the error on this method will always be quite high.)

# %% tags=["active-ipynb", "remove-input"]
# quantsum_se_up, lsep_up, lsepp_up, lse_up_, lsep_up_, lsepp_up_ = {}, {}, {}, {}, {}, {}
# for key in xdata:
#     nterms_up = min((cdf_ps[key]>0.8).sum(), 9)
#     if len(cdf_ps[key]) >= 2000:
#         err_settings = np.seterr(over="ignore", under="ignore", divide="ignore")
#         pse_up = MetaLogistic(cdf_xs=larr[key][cdf_ps[key]>0.8], cdf_ps=cdf_ps[key][cdf_ps[key]>0.8],
#                               term=nterms_up, fit_method="Linear least squares",
#                               ubound=True)
#         np.seterr(**err_settings)
#
#         quantsum_se_up[key] = QuantileSum.from_metalogistic(pse_up)
#         lsep_up[key] = quantsum_se_up[key].diff()
#         lsepp_up[key] = lsep_up[key].diff()
#
#         lse_up_[key] = quantsum_se_up[key](Φarr[key])
#         lsep_up_[key] = lsep_up[key](Φarr[key])
#         lsepp_up_[key] = lsepp_up[key](Φarr[key])

# %% tags=["remove-input", "active-ipynb"]
# frames = {}
# for key in xdata:
#     if len(cdf_ps[key]) >= 2000:
#         panels = {}
#
#         dims=dict(kdims=["Φ"], vdims=["l"])
#         panels["l"] = hv.Curve(zip(Φarr[key], le_[key]), **dims, label="le") \
#           * hv.Curve(zip(Φarr[key], lse_[key]), **dims, label="l*e") \
#           * hv.Curve(zip(Φarr[key][abs(lse_up_[key])<100], lse_up_[key][abs(lse_up_[key])<100]), **dims, label="l*e (upper)")
#
#         dims=dict(kdims=["Φ"], vdims=["l'"])
#         panels["l'"] = hv.Curve(zip(Φarr[key], lep_[key]), **dims, label="l'e") \
#           * hv.Curve(zip(Φarr[key], lsep_[key]), **dims, label="l'*e") \
#           * hv.Curve(zip(Φarr[key][abs(lsep_up_[key])<100], lsep_up_[key][abs(lsep_up_[key])<100]), **dims, label="l*e (upper)")
#
#         dims=dict(kdims=["Φ"], vdims=["l''"])
#         panels["l''"] = hv.Curve(zip(Φarr[key], lepp_[key]), **dims, label="l''e") \
#           * hv.Curve(zip(Φarr[key], lsepp_[key]), **dims, label="l''*e") \
#           * hv.Curve(zip(Φarr[key][abs(lsepp_up_[key])<1000], lsepp_up_[key][abs(lsepp_up_[key])<1000]), **dims, label="l''*e (upper)")
#
#         panels["l"].opts(legend_position="top_left")
#         panels["l'"].opts(legend_position="bottom_left")
#         panels["l''"].opts(legend_position="top_left")
#         layout = panels["l"] + panels["l'"] + panels["l''"]
#         frames[key] = layout.redim.range(
#             **{"Φ"  : (0.7, 1),
#                "l"  : get_bounds(le_[key][Φarr[key]>0.8], lse_[key][Φarr[key]>0.8], lse_up_[key][Φarr[key]>0.8]),
#                "l'" : get_bounds(lep_[key][Φarr[key]>0.8], lsep_[key][Φarr[key]>0.8], lsep_up_[key][Φarr[key]>0.8]),
#                "l''" : get_bounds(lepp_[key][Φarr[key]>0.8], lsepp_[key][Φarr[key]>0.8], lsepp_up_[key][Φarr[key]>0.8]),
#               }).opts(title=f"L = {key}")
# hv.HoloMap(frames, kdims=["L"]).collate()

# %% tags=["active-ipynb", "remove-input"]
# for key in xdata:
#     if len(cdf_ps[key]) > 2000:
#         # Stitch the upper l*e, starting at Φ=0.9
#         lse_[key] = np.where(Φarr[key]<0.9, lse_[key], lse_up_[key])
#         lsep_[key] = np.where(Φarr[key]<0.9, lsep_[key], lsep_up_[key])
#         lsepp_[key] = np.where(Φarr[key]<0.9, lsepp_[key], lsepp_up_[key])

# %% [markdown]
# ### Make the CDF monotone

# %% tags=["hide-input"]
def make_monotone(direction: str, arr: Array, Φarr: Array,
                  other_arrs: List[Array]=(), strict: bool=True) -> Array:
    """
    Return a copy `arr` which is made monotone by sorting its values.
    Values need not be equally spaced; if they are not equally spaced, linear
    interpolation is used to resample the values to Φarr.
    
    :param:direction: One of "increasing", "decreasing".
    :param:arr: The array to make monotone.
    :param:Φarr: The array of abscissa. Must be of the same length as `arr`.
       arr[i] is interpreted as the average value between Φarr[i-1] and Φarr[i].
    :param:other_arrs: Other arrays to sort alongside `arr`, for example
       an array of error values.
       NOTE: Arrays are modified in-place.
    :param:strict: Whether to check that the array is strictly monotone
       (no value is repeated).
    """
    assert direction in {"increasing", "decreasing"}
    
    σ = np.argsort(arr)
    # l_i is assigned the prob mass Φ_i - Φ_{i-1}
    # Using the preceding mass ensures that if the l_i are already sorted,
    # cumsum(Φarr[σ]) recreates Φarr exactly
    ΔΦ = np.diff(Φarr, prepend=0.)
    # We want to keep a single shared set of Φarr stops, so interpolate back into them
    arr = np.interp(Φarr, np.cumsum(ΔΦ[σ]), arr[σ])
    
    for _arr in other_arrs:
        _arr[:] = _arr[σ]
    
    for _arr in (arr, *other_arrs):
        if direction == "decreasing":
            _arr[:] = _arr[::-1]
        if strict:
            assert (np.diff(_arr) != 0).all(), "The returned array is not strictly monotone."
    return arr


# %% [markdown]
# ### Use numerical derivatives where more realiable

# %% tags=["hide-input"]
def robustify(f, fapprox) -> Array:
    """
    Robustify an array of function values using an approximation `fapprox`
    that may be less precise but more numerically stable. For example,
    `f` can be an analytic derivative, and `fapprox` a 2nd order approximation
    by finite differences.
    """
    return np.where(np.isclose(fapprox, f, atol=1e-8, rtol=5e-2),
                    f, fapprox)


# %% tags=["remove-input"]
def is_robust(f, fapprox) -> bool:
    """
    Return True if `robustify` would leave `f` unchanged.
    Return False if at least one element would be replaced by `fapprox`.
    """
    return np.all(np.isclose(fapprox, f, atol=1e-8, rtol=5e-2))


# %% tags=["active-ipynb", "remove-input"]
# lep_r, lepp_r, lse_r, lsep_r, lsepp_r = {}, {}, {}, {}, {}
# for key in xdata:
#     _Φarr = Φarr[key]
#     #assert (np.diff(le_) > 0).all()                     # le_ is already monotone
#     if not (np.diff(le_[key]) > 0).all():
#         logger.warning("The CDF fitted to the purely theoretical model (CDF synth) was not monotone. "
#                        "This may limit the model resolution.")
#         le_[key] = make_monotone("increasing", le_[key], _Φarr)   # le_ should already be monotone, but even with many samples sometimes small artifacts remain
#     lep_r[key] = robustify(lep_[key], np.gradient(le_[key], _Φarr))     # Even with monotone le,
#     lepp_r[key] = robustify(lepp_[key], np.gradient(lep_r[key], _Φarr)) # derivatives often still show artifacts
#
#     lse_r[key] = make_monotone("increasing", lse_[key], _Φarr)
#     lsep_r[key] = robustify(lsep_[key], np.gradient(lse_r[key], _Φarr))
#     lsepp_r[key] = robustify(lsepp_[key], np.gradient(lsep_r[key], _Φarr))

# %% [markdown]
# The figure below illustrates artifact removal for the quantile function $l_{*e}(Φ)$ and its derivatives.

# %% tags=["active-ipynb", "remove-input"]
# frames = {}
# for key in xdata:
#     Φarr_ = Φarr[key]
#     
#     dims=dict(kdims=["Φ"], vdims=["l"])
#     panel_lse = hv.Curve(zip(Φarr_, lse_[key]), **dims, label="original l*e") \
#     * hv.Curve(zip(Φarr_, lse_r[key]), **dims, label="monotone")
#     panel_lse.opts(legend_position="bottom_right")
#
#     _lsep_num = np.gradient(lse_r[key], Φarr_)
#     _lsep_ana = lsep_[key]
#     dims=dict(kdims=["Φ"], vdims=["l'"])
#     panel_lsep = hv.Curve(zip(Φarr_, _lsep_ana), **dims, label="original l'*e (analytic diff)") \
#     * hv.Curve(zip(Φarr_, _lsep_num), **dims, label="finite diff") \
#     * hv.Curve(zip(Φarr_, lsep_r[key]), **dims, label="robustified")
#     #panel_lsep.opts(legend_position="bottom_right")
#
#     _lsepp_num = np.gradient(lsep_r[key], Φarr_)
#     _lsepp_ana = lsepp_[key]
#     dims=dict(kdims=["Φ"], vdims=["l''"])
#     panel_lsepp = hv.Curve(zip(Φarr_, _lsepp_ana), **dims, label="original l''*e (analytic diff)") \
#     * hv.Curve(zip(Φarr_, _lsepp_num), **dims, label="finite diff") \
#     * hv.Curve(zip(Φarr_, lsepp_r[key]), **dims, label="robustified")
#
#     frames[key] = panel_lse + panel_lsep + panel_lsepp
# hv.HoloMap(frames, kdims=["L"]).collate()

# %% [markdown]
# ## Estimating δ
#
# We use a variant of {eq}`eq:data-rep-prob` to estimate the correction function $δ$:
# $$\hat{δ}(l) = \left[\log\frac{d\laez}{dΦ} - \log \frac{d\lasez}{dΦ}\right]{\laez=l} $$
#
# :::{caution}  
# The correction function is defined in terms of likelihoods for the *synthetic* model. Thus when inverting the derivatives in {eq}`eq:data-rep-prob`, we need to make sure to evaluate them at the value of $Φ$ corresponding to the same value of $l$.
#
# For an implementation like ours where all functions are discretized consistently, this can be accounted for implicitely by matching discretized indices.  
# :::

# %% [markdown]
# **TODO**: Compute the discretized $\Philseaz$ on the correct bins from the start, so we don’t have to use linear interpolation.

# %% tags=["active-ipynb"]
# δ = {key: np.log(lep_r[key]) - np.interp(le_[key], lse_r[key], np.log(lsep_r[key])) for key in xdata}

# %% [markdown]
# The estimate of $\lasz$ is then obtain directly with {eq}`eq:relation-logl-rvs`: (In the figure, the shaded region corresponds to the range $\pm δ$.)

# %% tags=["active-ipynb"]
# ls_ = {key: le_[key] + δ[key] for key in xdata}

# %% tags=["active-ipynb", "remove-input"]
# frames = {}
# for key in xdata:
#     dims=dict(kdims=["Φ"], vdims=["l"])
#     panels["l"] = hv.Spread((Φarr[key], ls_[key], c*δ[key]), kdims=["Φ"], vdims=["l", "Δl"]
#                            ).opts(line_color=None, fill_color="#efce88") \
#     *hv.Curve(zip(Φarr[key], le_[key]), **dims, label="le") \
#     *hv.Curve(zip(Φarr[key], lse_r[key]), **dims, label="l*e") \
#     *hv.Curve(zip(Φarr[key], ls_[key]), **dims, label="l*")
#     frames[key] = panels["l"].opts(legend_position="bottom_right")
# hv.HoloMap(frames, kdims=["L"]).collate()

# %% [markdown]
# ## Averaging over data-reproducing CDFs
#
# Our goal is to describe the statistics of $\lasz$ (the data-reproducing likelihood). Since we expect this random variable to be Gaussian, we therefore need its first two cumulants. We can do this by viewing the CDF as a random path from $\Philsaz=0$ to $\Philsaz=1$. The randomness comes from the uncertainty on $δ(l)$, which we assume to be proportional to $δ(l)$ itself:
# \begin{equation*}
# \lasz \sim \mathcal{N}\bigl(\laez + δ(\laez),\, c^2δ(\laez)^2\bigr)\,.
# \end{equation*}
# (Intuitively, one can view the values $\lhatasz \pm c δ(\laez)$ as a set of soft signposts, through which paths should pass.) Thus we can express the cumulants as a path integral, which we can estimate this expectation by averaging over realizations: (in the expressions below, $\mathcal{D} Φ(l)$ denotes the measure over CDF paths)
# \begin{align*}
# \EE[l^n] &= \bigl\langle\, \bigl\langle\, l^n \,\bigr\rangle_{\text{samples}} \,\bigr\rangle_{\text{fit uncertainty}}
#   & \EE[(l-\langle l \rangle)^n] &= \bigl\langle\, \bigl\langle\, (l-\langle l \rangle)^n \,\bigr\rangle_{\text{samples}} \,\bigr\rangle_{\text{fit uncertainty}} \\
# &= \int \mathcal{D} Φ(l)\underbrace{\int l^n dΦ(l)}_{= \EE[(l-\langle l \rangle)^n \mid Φ]}
#   & &= \int \mathcal{D} Φ(l)\underbrace{\int (l-\langle l \rangle)^n dΦ(l)}_{= \EE[(l-\langle l \rangle)^n \mid Φ]} \\
# &\approx \frac{1}{R} \sum_{r=1}^R \sum_i (l_i^{(r)})^n \, \bigl(Φ(l_{i}) - Φ(l_{i-1})\bigr)\,.
#   & &\approx \frac{1}{R} \sum_{r=1}^R \sum_i (l_i^{(r)}-\langle l \rangle)^n \, \bigl(Φ(l_{i}) - Φ(l_{i-1})\bigr)\,.
# \end{align*}
# (One advantage of this approach is that once we have an ensemble of realizations of the quantile paths, we can quickly compute any expectation over those realizations.)

# %% [markdown]
# :::{note}  
# We write the inner sum over $i$ without bounds because it is only illustrative: it uses left Riemann sums for simplicity, but in practice a higher order integration scheme would be preferred. Below we use the Simpson rule.  
# :::

# %% [markdown]
# Recall that $μ^*$ and $Σ^*$ are the statistics of the _estimate_ of the first moment; i.e. $\widehat{\EE[l]} \sim \nN(μ^*, Σ^*)$. Therefore we have
# \begin{align}
# μ^* &= \widehat{\EE[l]} \notag \\
# &= \frac{1}{R} \sum_{r=1}^R \sum_i l_i^{(r)} \, \bigl(Φ(l_{i}) - Φ(l_{i-1})\bigr) \\
# Σ^* &= \bigl\langle\, \bigl\langle\, l \,\bigr\rangle_{\text{samples}} - μ^* \,\bigr\rangle_{\text{fit uncertainty}} \notag \\
# &= \frac{1}{R}  \sum_{r=1}^R \left[ \Bigl( \sum_i l_i^{(r)} \, \bigl(Φ(l_{i}) - Φ(l_{i-1})\bigr) \Bigr) - μ^*
#  \right] \end{align}

# %% [markdown]
# :::{note}  
# One could be tempted to define a variance *density*, and thereby arrive at a stochastic differential equation for the cumulants, which could then be integrated. However, this approach is fraught because in order to obtain sensible results, the CDF *must* be monotone. This puts highly non-trivial constraints on the autocorrelation of $δ$ – in particular, it cannot be described by white noise. Consequently, the usual half-order scaling is not guaranteed to be consistent: results may diverge as the step size goes to zero. (See {cite:t}`gillespieMathematicsBrownianMotion1996` for discussion on how consistency requirements determine the exponent of differentials.)
#
# Moreover, integrating a stochastic equation would accumulate variance, leading to strongly asymmetric assignment of uncertainty (almost no uncertainty at the end where integration starts). This asymmetry, and its dependence on the choice of starting point, contradicts our intuition that paths should at each point show variance proportional to $δ^2$.
#
# In contrast, the “signpost” interpretation is not only simpler, but also has a well-defined finite limit when the integration steps are reduced to zero.  
# :::

# %% [markdown]
# To generate random CDF paths, we first
# 1. sample from the normal distribution at each discretized point $\laez$: $\lasz \sim \mathcal{N}\bigl( \laez + δ(\laez),\, c^2 δ(\laez)^2 \bigr)$.
# Since this does not produce a monotone increasing path, we then
# 2. make the path monotone by sorting the likelihood values.
#
# This approach has a few important advantages:
# - Variance is everywhere proportional to $δ(l)$ — there is no bias towards higher variance for low or high values of $l$.
# - The result is consistent when the integration step is changed.
#
# Also note that this approach tends to depress lower likelihood values, and increase the probability of high likelihood values, which is consistent with our idea that the data-reproducing model should be the best fitting model. (This can be understood as a statistical effect due to sorting.)

# %% [markdown]
# In the figure below, we have sorted the values for $\lhatasz$. This is why std. dev. ranges appear jagged.

# %% tags=["active-ipynb", "remove-input"]
# ls_lst, ls_m, δ_m = {}, {}, {}
# frames = {}
# for key in xdata:
#     Φarr_ = Φarr[key]
#     δ_m[key] = δ[key].copy()
#     ls_m[key] = make_monotone("increasing", ls_[key], Φarr_, other_arrs=[δ_m[key]])
#
#     ls_lst[key] = np.array(
#         [make_monotone("increasing", np.random.normal(ls_m[key], c*abs(δ_m[key])), Φarr_)
#          for _ in range(R_Φ)])
#
#     dims=dict(kdims=["Φ"], vdims=["l"])
#     panels["l"] = hv.Spread((Φarr_, ls_m[key], abs(δ_m[key])), kdims=["Φ"], vdims=["l", "σl"]).opts(line_color=None, fill_color="#efce88") \
#     *hv.Overlay([hv.Curve(zip(Φarr_, _ls), **dims, label="l").opts(line_color="#AAA", line_width=1.5) for _ls in ls_lst[key]]) \
#     *hv.Curve(zip(Φarr_, le_[key]), **dims, label="le") \
#     *hv.Curve(zip(Φarr_, lse_r[key]), **dims, label="l*e") \
#     *hv.Curve(zip(Φarr_, ls_m[key]), **dims, label="^l*") \
#
#     frames[key] = panels["l"].opts(legend_position="bottom_right").opts(hv.opts.Curve(line_width=1.5))
# hv.HoloMap(frames, kdims=["L"]).collate()

# %% [markdown]
# Finally, $μ$ and $Σ$ are obtained by averaging over realizations (in brackets are shown the values for each realization). `std_err` is the standard error on the estimate of $μ^* = \EE[\lasz]$.

# %% tags=["active-ipynb", "remove-input"]
# stderr = {}
# μ = {}  ; μ_nb = μ
# Σ = {}  ; Σ_nb = Σ
# # For code which will use this later, we create a var with the 'nb' suffix to indicate this
# # was computed in the notebook. (Version without suffix is closer to the code in the exported function)
# for key in xdata:
#     μ_lst = simpson(ls_lst[key], Φarr[key], even="last")  # even="last" b/c accuracy is most important for large Φ
#     μ[key] = μ_lst.mean()
#
#     # ls_var = np.trapz((ls_lst-final_ls_mean)**2, Φarr)
#     Σ_lst = ((μ_lst - μ[key])**2)
#     Σ[key] = Σ_lst.sum() / (len(Σ_lst) - 1)  # -1 for unbiased sample estimator
#
#     stderr[key] = math.sqrt(Σ[key]/R_Φ)
#
#     print(f"---------- L = {key:>4} ----------")
#     print(f"μ*        : {μ[key]:>6.4f}  {μ_lst.round(4)}")
#     print(f"√Σ*        : {math.sqrt(Σ[key]):>6.4f}  {np.sqrt(Σ_lst).round(4)}")
#     print(f"stderr(μ*): {stderr[key]:>6.4f}")
#     print()

# %% [markdown]
# To ensure the accuracy of our results is not unnecessarily reduced by the finite number of realizations, we increase the number of data points as needed to achieve the tolerance `cdf_tol`. For $R_Φ$ data points, we compute this as
#
# $$\SE^{(R_Φ)}\nolimits(μ^*) = \sqrt{\frac{\sum_{r=1}^{R_Φ} Σ^{*}_{(r)}}{R_Φ}} \,.$$
#
# Therefore, from an initial sample, we can directly compute the number of samples $R_Φ'$ required to reduce the standard error to some prescribed value $ε_{\scriptscriptstyle\mathrm{CDF}}$:
#
# $$R_Φ' = \left(\frac{\SE^{(R_Φ)}}{ε_{\scriptscriptstyle\mathrm{CDF}}}\right)^2 R_Φ \,.$$

# %% [markdown]
# Since expectation is linear, if $μ^{(R_Φ)}$ is the original estimated statistic obtained by averaging over $L$ data points, and $μ^{(R_Φ'-R_Φ)}$ is the new estimate obtained with $R_Φ'-R_Φ$ different data points, then an estimate from $R_Φ$ data points can be obtained with an online formula:
#
# $$μ^{(R_Φ)} = \frac{R_Φ}{R_Φ'} μ^{(R_Φ)} + \left(1 - \frac{R_Φ}{R_Φ'}\right) μ^{(R_Φ'-R_Φ)}\,.$$
#
# This can be used to update the estimates of the mean and variance without having to concatenate the arrays $μ^{(R_Φ)}$ and $μ^{(R_Φ'-R_Φ)}$, which may be large. (Although for the variance we still need to recompute the first term with the newly updated mean.)
#
# In practice the updated $R_Φ'$ may not be exactly enough, especially if the initial small sample lead to an underestimate of the variance $Σ$. So we repeat this a few times until the measured standard error achieves the desired tolerance.

# %% tags=["active-ipynb", "remove-input"]
# cdf_tol = 0.01
# max_Rcdf = int(3e4)  # Safety factor, in case tolerance is too low and would require very many CDF realizations

# %% tags=["active-ipynb", "remove-input"]
# R_Φ_final = {key: R_Φ for key in xdata}
# for key in xdata:
#     old_R_Φ = R_Φ
#     while stderr[key] > cdf_tol and old_R_Φ < max_Rcdf:
#         new_RΦ = int((stderr[key]/cdf_tol)**2 * old_R_Φ)
#         if new_RΦ <= old_R_Φ:
#             # We may be above the requested tolerance, but by so little that
#             # it amounts to rounding errors.
#             break
#         else:
#             if new_RΦ > max_Rcdf:
#                 projected_tol = stderr[key] * math.sqrt(old_R_Φ/max_Rcdf)
#                 logger.warning(f"Achieving a tolerance of {cdf_tol:.2g} would require "
#                                f"approximately {new_RΦ} CDF realizations. Instead we "
#                                f"will limit ourselves to {max_Rcdf} realizations, to prevent "
#                                "excess time and memory consumption. This should achieve "
#                                f"a tolerance of about {projected_tol:.2g}.")
#                 new_RΦ = max_Rcdf
#             new_ls_lst = np.array(
#                 [make_monotone("increasing", np.random.normal(ls_m[key], c*abs(δ_m[key])), Φarr[key])
#                  for _ in range(new_RΦ - old_R_Φ)])
#
#             new_μ_lst = simpson(new_ls_lst, Φarr[key], even="last")
#             μ[key] = (old_R_Φ/new_RΦ)*μ[key] + (1 - old_R_Φ/new_RΦ)*new_μ_lst.mean()
#             # We try to avoid concenating ls_var and new_ls_var, which may both be large arrays
#             Σ[key] = ((old_R_Φ/new_RΦ)*((μ_lst - μ[key])**2).sum()
#                       + (1 - old_R_Φ/new_RΦ)*((new_μ_lst - μ[key])**2).sum()
#                      ) / (new_RΦ - 1)  # -1 for unbiased estimator of the variance
#
#             stderr[key] = math.sqrt(Σ[key]/new_RΦ)
#             
#             old_R_Φ = new_RΦ
#             μ_lst = np.concatenate((μ_lst, new_μ_lst))  #OPTIM: Avoid concatenating potentially large arrays (e.g. keep a short list of arrays)
#
#     R_Φ_final[key] = old_R_Φ
#             
#     print(f"---------- L = {key:>4} ----------")
#     print(f"RΦ: {old_R_Φ:>3}       stderr: {stderr[key]:>.4f}")
#     print()

# %% tags=["hide-input"]
def average_moments_over_cdfs(
    draw_cdfs: Callable[[int], Array[np.inexact,2]], Φarr,
    tol=0.05, R_init=400, R_max=int(3e4)
    ) -> Tuple[float, float]:
    """
    Return the mean and variance computed averaged over a distribution of CDFs.
    
    Parameters
    ----------
    draw_cdfs: Function taking an integer and returning a collection of that
       many CDFs, as a 2D array. Array dimensions must be (CDF index, Φ index).
       Array values are the abscissa of the CDF (the random var), or
       equivalently the ordinate of the quantile function.
    Φarr: Cumulative probabilities at which points are sampled, i.e.
       the ordinate of the CDF.
    tol: The number of CDF will be increased such that the standard error
       on μ and sqrt(Σ) is approximately equal to `tol`.
       (Note that the standard error on sqrt(Σ) scales as R^¼).
    
    """
    R = R_init
    cdf_lst = draw_cdfs(R)
    _μ_lst = simpson(cdf_lst, Φarr, even="last")  # even="last" b/c accuracy is most important for large Φ
    _μ = _μ_lst.mean()
    _Σ = ((_μ_lst - _μ)**2).sum() / (R -1)  # -1 for unbiased sample estimator
    stderr = _Σ / math.sqrt(R)
    
    if stderr > tol:
        while stderr > cdf_tol and R < max_Rcdf:
            new_R = int((stderr/tol)**2 * R)
            if new_R <= R:
                    # We may be above the requested tolerance, but by so little that
                    # it amounts to rounding errors.
                    break
            elif new_R > R_max:
                global warned_that_Rcdf_too_big
                if "warned_that_Rcdf_too_big" not in globals():
                    projected_tol = stderr * math.sqrt(R/R_max)
                    if not math.isclose(projected_tol, tol, rel_tol=0.4):  # Don't show a warning if the adjusted tolerance rounds up to roughly the same
                        logger.warning(f"Achieving a tolerance of {tol:.2g} would require "
                                       f"approximately {new_R} CDF realizations. Instead we "
                                       f"will limit ourselves to {R_max} realizations, to prevent "
                                       "excess time and memory consumption. This should achieve "
                                       f"a tolerance of about {projected_tol:.2g}.")
                        warned_that_Rcdf_too_big = True
                new_R = R_max

            new_cdf_lst = draw_cdfs(new_R - R)
            # NB: We try to avoid concenating cdf_lst and new_cdf_lst, which may both be large arrays
            new_μ_lst = np.trapz(new_cdf_lst, Φarr)
            _μ = (R/new_R)*_μ + (1 - R/new_R)*new_μ_lst.mean()
            _Σ = ((R/new_R)*((_μ_lst - _μ)**2).sum()  # Recompute variance on original traces with updated mean
                 + (1 - R/new_R)*((new_μ_lst - _μ)**2).sum()
                 ) / (new_R - 1)  # -1 for unbiased sample estimator

            stderr = math.sqrt(_Σ/new_R)
            
            R = new_R
            _μ_lst = np.concatenate((_μ_lst, new_μ_lst))  #OPTIM: Avoid concatenating potentially large arrays (e.g. keep a short list of arrays)
    
    # print(R, _μ, _Σ, _μ_lst[:5])  # DEBUG
    return _μ, _Σ


# %% [markdown]
# ### Averaging over fits of the theoretical distribution $\paez$
#
# In practice we found that there is a limit to the precision with which we can fit $\paez$ when we fit it to samples of $\laez$.
# Some amount of numerical error is always present, no matter the number of samples, and some random sample draws of $\{\laez\}$ are more fortuitous than others.
# For this reason, one should always strive to provide an analytical expression for $\paez$.
# For cases where this is not possible, we account for this uncertainty by repeating the entire calculation with multiple fits of $\paez$. This gives muliple values for $μ^*_r$ and $Σ^*_r$, each representating a Gaussian distribution for true $l^*$. Since the fits are independently generated, we assume that each has the same probability of being the true distribution; the best estimate for the distribution of $\laez$ is then as a mixture distribution:
# \begin{equation*}
# \laez \sim \frac{1}{R} \sum_{r=1}^R \nN\left(μ^*_r, Σ^*_r\right) \,.
# \end{equation*}
# If the mixture components are close (which they should be, since these are differences between fits to the same distribution), the mixture model is well-approximated by a Gaussian. Our function `compute_μ_Σ_m1` can therefore still return $μ^*$ and $Σ^*$, but they should be those of the mixture:
# \begin{align}
# μ^* &= \frac{1}{R} \sum_{r=1}^R μ^*_r \\
# Σ^* &= \underbrace{\frac{1}{R} \sum_{r=1}^R \left(Σ^*_r + (μ^*_r)^2  \right)}_{\approx\, \langle\laez^2\rangle} - \left(μ^*\right)^2 \,.
# \end{align}
# In these expressions we have assumed that components can be weighted equally. (In line with our assumption that they are fitted independently.)
# One obtains the expression for $Σ^*$ by noting that for any mixture distribution, the non-central moments are just weighted averages of the moments of the components.
#
# **TODO**: Check whether the draws of $\paez$ need to be conditioned on $z$.

# %% [markdown]
# ---

# %% [markdown]
# ## Complete EMD function

# %% [markdown]
# :::{prf:algorithm} compute_μ_Σ_m1
# :label: alg-compute-mu-Sigma-m1
#
# **Inputs**  
# ~ - MX  
#   Generative model for the independent variable.  
# ~ - MN  
#   Generative natural model. If $Z$ is the natural variable and $X$ the independent, then `MN` samples from $p(z | X)$.  
# ~ - Me  
# ~ - xdata, ydata  
#   Observed data points  
# ~ - c  
# ~ - R_z  
# ~ - nterms  
#   Number of terms in the metalog.  
# ~ tol_synth  
#   Tolerance to achieve for $\Phileaz$.  
# ~ - fit_attempts  
# ~ - dataset_size_step  
#   See `fit_Φeaz` ({prf:ref}`alg-emp-CDF`).
#
# **Outputs**
#   Mean and variance of $l*$
#
# :::

# %% tags=["hide-input"]
def compute_μ_Σ_m1(MX: Callable[[int], FloatArray], MN: Callable[[Array], FloatArray],
                   Me: Callable[[FloatArray], FloatArray], l: Callable[[FloatArray], float],
                   xdata: FloatArray, ydata: FloatArray,
                   *,
                   c: float, R_z: int, nterms: int=9,
                   p_synth: Union[None,stats.rv_continuous,List[stats.rv_continuous]]=None,
                   tol_synth: float=None, num_synth_fits: int=8,
                   synth_fit_stepsize: int=4000, synth_fit_attempts: int=6,
                   cdf_tol=0.05, Rcdf_init=400, Rcdf_max=int(3e4),
                   Φarr: Optional[FloatArray]=None, zdata: Optional[FloatArray]=None
    ) -> Tuple[float, float]:
    """
    Return μ and Σ for the first moment of l*.
    
    Parameters
    ----------
    ...
    ...
    p_synth: An object compatible with the API of random variables defined in scipy.stats,
        describing the 1D distribution of likelihoods `l` for model distributions.
        In other words, if a synthetic sample ``z`` is drawn from `Me` ○ `MN` ○ `MX`, this is the
        probability distribution for the value of ``l(z)``. If `p_synth` is not provided,
        it is estimated exactly in this way.
        If a list is provided, it is assumed that these are different estimates of
        `p_synth`. `compute_μ_Σ_m1` is called on each, and the mean and variance of the
        resulting Gaussian mixture is returned.
    tol_synth: Ignored if `p_synth` is provided, required otherwise.
        The tolerance used to determine the number of samples when estimating `p_synth`.
        Tolerance is defined as the mean squared difference between the empirical CDF
        of samples, and the fitted CDF.
    num_synth_fits: Fitting `p_synth` introduces additional uncertainty. To account for
        this, we fit it multiple times. The `compute_μ_Σ_m1` call is repeated for each,
        resulting in multiple μ and Σ values. We treat these as a Gaussian mixture, and
        return the mean and variance of the mixture.
    synth_fit_stepsize:
    synth_fit_attempts:
        When fitting `p_synth` to samples, we first try with a number of samples equal
        to `synth_fit_stepsize`. If that doesn’t achiev `tol_synth`, we try again
        with twice, then thrice, etc. up to `synth_fit_attempts` ⨉ `synth_fit_stepsize`.
        If that is still not enough, the fit fails, raising RuntimeError.
    Φarr: Set of points over which the CDF will be discretized.
        The default is ``linspace(0.01, 0.99)``.
    zdata: Mostly intended for diagnostics.
        In normal use, `zdata` is computed by applying `MN` to `xdata`; if `MN` is
        stochastic, the result is non-deterministic. When `zdata` is provided, it is
        used directly, and `xdata` and `MN` are ignored.
        
    .. Caution:: While the argument `p_synth` is optional, its estimation from samples is
       both computationally the most expensive step, and numerically the least stable step.
       The returned variance Σ is also inflated, to account for uncertainty on p_synth itself.
       Thus one should strive to provide an analytical expression whenever possible.

    Raises
    ------
    If `synth_fit_attempts`x`synth_fit_stepsize` is not enough
    samples to achieve tolerance `tol_synth, we raise `RuntimeError`.
    
    .. Note::
       The function `EMD` does not currently expose the parameters `synth_fit_attempts`
       and `synth_fit_stepsize`. If they are found to be insufficient, our
       preference is to adjust the fixed values (defined in the defaults here)
    """
    if len(xdata) != len(ydata):
        raise ValueError("The first dimensions of `xdata` and `ydata` must match, "
                         "since they correspond to the sample index. Received:\n"
                         f"  xdata shape: {np.shape(xdata)}\n  ydata shape: {np.shape(ydata)}")

    μ = -math.inf
    Σ = None

    #z_synth = MN(xdata)
    last_zarr = None

    # Estimate synthetic Φ(l)
    if p_synth is None or isinstance(p_synth, Iterable):
        # Recursive branch: construct GMM (one component per p_synth) and return the μ, Σ of the mixture
        if p_synth is None:
            if tol_synth is None:
                raise ValueError("Either `p_synth` or `tol_synth` must be specified")
            pe_lst = (fit_Φeaz(MX, MN, Me, l, nterms, tol_synth,
                               synth_fit_attempts, synth_fit_stepsize,
                               never_fail=True)
                      for _ in range(num_synth_fits))
        else:
            pe_lst = p_synth
        # Create the list of (μ, Σ) tuples for the GMM components
        comp_stats = [compute_μ_Σ_m1(
                          MX, MN, Me, l, xdata, ydata,
                          c=c, R_z=R_z, nterms=nterms, p_synth=pe,
                          cdf_tol=cdf_tol, Rcdf_init=Rcdf_init, Rcdf_max=Rcdf_max,
                          Φarr=Φarr, zdata=zdata)
                      for pe in pe_lst]
        # Compute mean and variance of the GMM
        # NB: We assume the pe’s are generated independently and that we can assume the components have equal weights 
        μ = sum(μc for μc, Σc in comp_stats)/len(comp_stats)
        m2 = sum(Σc + μc**2 for μc, Σc in comp_stats)/len(comp_stats)  # Second bare (non-central) moment
        Σ = m2 - μ**2
        return μ, Σ   # EARLY EXIT
    else:
        # Standard branch: Estimate the μ and Σ of a Gaussian
        pe = p_synth
    
    for r in range(R_z):
        zarr = MN(xdata) if zdata is None else zdata
        if r == 1 and np.all(last_zarr == zarr):
            logger.warning("Using R>1 for a non-stochastic natural model is "
                           "useless, since `MN(xdata)` always returns the "
                           f"same values for `z`. (Received R={R}.)")
            break
        larr = l(ydata, zarr)
        idxsort = np.argsort(larr)
        #zarr = zarr[idxsort]  # If we sort zarr, we should probably sort xdata and ydata as well
        larr = larr[idxsort]

        # Estimate mixed Φ from empirical CDF by fitting a metalogistic.
        # Since there are likely larger/smaller points we haven't sampled,
        # we choose the largest/smallest cdf probs to be strictly within (0, 1).
        L = len(larr); cdf_ps = np.arange(1, L+1)/(L+1)
        pse = MetaLogistic(cdf_xs=larr, cdf_ps=cdf_ps,
                           term=nterms, fit_method="Linear least squares",
                           ubound=True)
        # # Estimate fit error as L⁻¹ ∑ (Φ_emp(l) - Φse(l))²,
        # # which is the expected squared error on the cdf.
        # εs = np.sqrt(np.trapz((cdf_ps - pse.cdf(larr))**2) / L)
            
        # Construct functions to evaluate 1st & 2nd derivatives of l(Φe), l(Φ*e)
        quantsum_e = QuantileSum.from_metalogistic(pe)
        quantsum_se = QuantileSum.from_metalogistic(pse)
        lep  = quantsum_e.diff()  # 'p' for 'prime' => 1st derivative
        lsep = quantsum_se.diff()
        # lepp  = lep.diff()
        # lsepp = lepp.diff()
        
        # Re-fit on upper region to improve fit for high CDF values
        if L > 2000:
            err_settings = np.seterr(over="ignore", under="ignore", divide="ignore")
            pse_up = MetaLogistic(cdf_xs=larr[cdf_ps>0.8], cdf_ps=cdf_ps[cdf_ps>0.8],
                                  term=nterms, fit_method="Linear least squares",
                                  ubound=True)
            np.seterr(**err_settings)

            quantsum_se_up = QuantileSum.from_metalogistic(pse_up)
            lsep_up = quantsum_se_up.diff()
        else:
            quantsum_se_up = quantsum_se
            lsep_up = lsep
        
        # Evaluate all functions at the points given by Φarr
        if Φarr is None:
            Φarr = np.linspace(0.01, 0.99)
        le = quantsum_e(Φarr)
        lep = lep(Φarr)
        lse = np.piecewise(Φarr, [Φarr<=0.9, Φarr>0.9], [quantsum_se, quantsum_se_up])
        lsep = np.piecewise(Φarr, [Φarr<=0.9, Φarr>0.9], [lsep, lsep_up])
        
        # Remove artifacts / make monotone
        #assert (np.diff(le) > 0).all(), "The CDF of the purely theoretical model (CDF synth) should already be monotone"
        if not (np.diff(le) > 0).all():
            global warned_that_CDF_is_non_monotone
            if "warned_that_CDF_is_non_monotone" not in globals():
                logger.warning("The CDF fitted to the purely theoretical model (CDF synth) was not monotone. "
                               "This may limit the EMD resolution.")
                warned_that_CDF_is_non_monotone = True
            le = make_monotone("increasing", le, Φarr)  # le_ should already be monotone, but even with many samples sometimes small artifacts remain
        lep = robustify(lep, np.gradient(le, Φarr))     # Even with a monotone le, the gradient may have artifacts
        lse = make_monotone("increasing", lse, Φarr)
        lsep = robustify(lsep, np.gradient(lse, Φarr))
        
        
        δ = np.log(lep) - np.interp(le, lse, np.log(lsep))
        ls = le + δ
        ls = make_monotone("increasing", ls, Φarr)  # Because δ is random, we need to make ls random again
        
        def draw_cdfs(R):
            return np.array(
                [make_monotone("increasing", np.random.normal(ls, c*abs(δ)), Φarr)
                 for _ in range(R)])
        _μ, _Σ = average_moments_over_cdfs(draw_cdfs, Φarr, cdf_tol, Rcdf_init, Rcdf_max)

        if np.any(np.isnan(_μ)) or np.any(np.isnan(_Σ)):
            raise RuntimeError

        # Taking the max l over R repeats is equivalent to taking the highest μ, since l ~ N(μ, Σ)
        if μ < _μ:
            μ = _μ
            Σ = _Σ

    return μ, Σ


# %% tags=["hide-input"]
def EMD(MX: Optional[Callable[[int], FloatArray]],
        MNA: Callable[[FloatArray],Iterator[FloatArray]], MeA: Callable[[FloatArray,FloatArray], FloatArray,FloatArray], lA: Callable[[FloatArray,FloatArray], FloatArray],
        MNB: Callable[[FloatArray],Iterator[FloatArray]], MeB: Callable[[FloatArray,FloatArray], FloatArray,FloatArray], lB: Callable[[FloatArray,FloatArray], FloatArray],
        xdata: FloatArray, ydata: FloatArray, c: float, R_z: int,
        nterms: int=9, p_synth=None, tol_synth: float=0.008, num_synth_fits: int=8):
    """
    
    (For brevity, we use the convention of x, z, y denoting independent,
    latent and observed variables respectively.)
    
    .. Hint:: If the likelihood probability for the purely theoretic model is
       not given, it is automatically fit by generating samples. This typically
       adds a few seconds to the evaluation. If `EMD` is to be called multiple
       times, it is usually beneficial to pre-fit this function with  `fit_Φeaz`
       and pass it as the `p_synth` argument.
    
    Parameters
    ----------
    MX: Function which generates the requested number of points for the
       independent variable (e.g. time or position).
       To generate ``L`` data points, we do ``x=MX(L); Me(x, MN(x))``.
       Can be `None` if `p_synth` is provided.
    MNA, MNB: Functions taking x and returning z.
    MeA, MeB: Functions taking (x, z) pairs and returning y.
    lA, lB  : log likelihood functions for MeA and MeB.
       In other words, log likelihood of the data conditioned on z.
       Functions take a pair (y, z) and return a real number.
    xdata: Observed independent variable.
       For example time, position...
    ydata: Observed dependent variable.
       For example voltage, emission rate, voltage, speed...

    c: Noise scaling parameter.
    R_z: Number of latent variables to sample. Must be ⩾ 1.
       Use 1 if `MNA` and `MNB` are not stochastic, larger values if they are.
    R_cdf: For each sampled latent, number of CDF paths to generate.
    nterms: Number of terms in the metalogs.
    tol_synth: tolerance for the fitting error of the theoretical c.d.f.
       In theory, with enough data and enough terms, it should be possible to
       make this arbitrarily low, although in practice improvoments below 0.005
       come with high additional computational cost.
    num_synth_fits: Fitting `p_synth` introduces additional uncertainty. To account for
        this, we fit it multiple times. The `compute_μ_Σ_m1` call is repeated for each,
        resulting in multiple μ and Σ values. We treat these as a Gaussian mixture, and
        return the mean and variance of the mixture.
    """
    # Note: Currently the parameters `synthetic_fit_attempts` and `synthetic_fit_stepsize` are not exposed
    μA1, ΣA1 = compute_μ_Σ_m1(MX, MNA, MeA, lA, xdata, ydata,
                              c=c, R_z=R_z, nterms=nterms, p_synth=p_synth, tol_synth=tol_synth, num_synth_fits=num_synth_fits)
    μB1, ΣB1 = compute_μ_Σ_m1(MX, MNB, MeB, lB, xdata, ydata,
                              c=c, R_z=R_z, nterms=nterms, p_synth=p_synth, tol_synth=tol_synth, num_synth_fits=num_synth_fits)

    ## The two forms below are equivalent, but the second more clearly
    ## expresses that the calculation is based on the probability that l* < 0
    #x = erf((μA1 - μB1) / math.sqrt(2*(ΣA1 + ΣB1)) )
    #return math.log(1+x) - math.log(1-x)
    Φ0 = stats.norm.cdf(0, loc=μA1-μB1, scale=np.sqrt(ΣA1+ΣB1))
    return np.log(1-Φ0) - np.log(Φ0)

# %% tags=["hide-input"]
def dEMD(*args, **kwargs):
    "Return abs(EMD(*args, **kwargs))"
    return abs(EMD(*args, **kwargs))

# %% [markdown]
# ## Tests

# %% [markdown]
# Smoke test

# %% tags=["active-ipynb", "remove-stderr"]
# for x_, y_ in zip(xdata.values(), ydata.values()):
#     EMD(MX=MX_synth,
#         MNA=MNA, MeA=MeA, lA=MeA.logpdf,
#         MNB=MNB, MeB=MeB, lB=MeB.logpdf,
#         xdata=x_, ydata=y_,
#         c=1, R_z=1, nterms=9)

# %% [markdown]
# Unit test: compare the moments computed within the notebook examples (`μ_nb`, `Σ_nb`) with those computed in by the function we provide (`μ_fn`, `Σ_fn`). Basically this tests that the function provided by this module is consistent with its documented example. Remarks:
# - The standard error measures our uncertainty on the estimate of $μ^*$. Thus the criterion
#   
#   $$\left\lvert μ_{\text{function}}^* - μ_{\text{notebook}}^*\right\rvert < 8 \times SE(μ^*)$$
#   
#   checks 1) that our function is consistent with the notebook documentation, and 2) that our certainty on that estimate is not too high.
#
# - For simplicity we use the same test for the variance, rather than separately compute the standard error on $Σ^*$:
#
#   $$\left\lvert Σ_{\text{function}}^* - Σ_{\text{notebook}}^* \right\rvert < 8 \times SE(μ^*)$$
#   
# - We make the test excessively permissive ($8 \times SE(μ^*)$) because we need to guarantee the success of the assertions.
#   Still, the fact that $5 \times SE(μ^*)$ wasn’t 100% reliable suggests we might be underestimating the error.

# %% tags=["active-ipynb"]
# for key, x_, y_ in zip(xdata, xdata.values(), ydata.values()):
#     μ_fn, Σ_fn = compute_μ_Σ_m1(MX_synth, MNA, MeA, MeA.logpdf, x_, y_, c=1, R_z=1,
#                           p_synth=pe[key], zdata=MNA(x_))
#     #assert math.isclose(μ, ls_mean.mean(), rel_tol=1.5e-1, abs_tol=3e-2)
#     #assert math.isclose(Σ, ls_var.mean(), rel_tol=1.5e-1, abs_tol=3e-2)
#     assert abs(μ_fn - μ_nb[key]) < 8*stderr[key]
#     assert abs(Σ_fn - Σ_nb[key]) < 8*stderr[key]

# %% [markdown] tags=["remove-cell"]
# ## Activation of diagnostics
#
# When present, `_emd_internals` stores intermediate values computed with EMD.
# It is filled automatically when defined; as a side-effect, this means
# all intermediate values will also remain in memory.

# %% tags=["remove-cell", "active-ipynb"]
# _emd_internals = {}
