# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python (EMD-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown]
# # Validation of the EMD approximation
#
# $ % Caution: Preamble is not inserted into the .ipynb file, hence the definitions below.
# \renewcommand{\EE}{\mathbb{E}}
# \renewcommand{\RR}{\mathbb{R}}
# \renewcommand{\nN}{\mathcal{N}}
# \renewcommand{\M}{\mathcal{M}}
# \renewcommand{\MNA}{\mathcal{M}^{\mathcal{N}}_A}
# \renewcommand{\MNB}{\mathcal{M}^{\mathcal{N}}_B}
# \renewcommand{\MeA}{\mathcal{M}^e_A}
# \renewcommand{\MeB}{\mathcal{M}^e_B}
# \renewcommand{\MsA}{\mathcal{M}^*_A}
# \renewcommand{\MsB}{\mathcal{M}^*_B}
# \renewcommand{\dconf}{d_{\mathrm{conf}}}
# \renewcommand{\Bconf}{B_{\mathrm{conf}}}
# \renewcommand{\bconf}{b_{\mathrm{conf}}}  % log(Bconf)
# \renewcommand{\EMD}{\mathop{\mathrm{EMD}}}
# \renewcommand{\laez}{l_{\smash{{}^{e}_a},z}}
# \renewcommand{\lAez}{l_{\smash{{}^{e}_A},z}}
# \renewcommand{\lBez}{l_{\smash{{}^{e}_B},z}}
# \renewcommand{\lAszo}{l_{\smash{{}^{*}_A},z^0}}
# \renewcommand{\lBszo}{l_{\smash{{}^{*}_B},z^0}}
# \renewcommand{\musA}[1][(1)]{μ^{#1}_{\smash{{}^{\,*}_A}}}
# \renewcommand{\musB}[1][(1)]{μ^{#1}_{\smash{{}^{\,*}_B}}}
# \renewcommand{\SigsA}[1][(1)]{Σ^{#1}_{\smash{{}^{\,*}_A}}}
# \renewcommand{\SigsB}[1][(1)]{Σ^{#1}_{\smash{{}^{\,*}_B}}}
# $
#
# We have proposed that a good metric for comparing fitted models is the confusion ratio, given by  Eq. {eq}`eq-def-bdconf`:
# \begin{equation}
# \bconf(\MNA, \MNB) = \log P(Δ_{AB} > 0) - \log P(Δ_{AB} < 0)\,.
# \end{equation}
# Here $Δ_{AB}$ is the Bayes ratio divided by the number of samples $L$; if $Δ_{AB} > 0$, then model A is better supported by the data, and vice-versa if $Δ_{AB} < 0$. The variance on the random variable $Δ_{AB}$ comes from our *uncertainty on the true experimental noise*: Deviations between model predictions and observations need to be accounted for by experimental noise; the greater these deviations, the less confident we are in the experimental model. Of course, if we have too little data, this will also increase our uncertainty on $Δ_{AB}$. Thus $Δ_{AB}$ should have the following features
# - As the number of samples $L$ increases, the variance on $Δ_{AB}$ converges to a finite value. The closer the model predictions align with observations, the smaller the variance of $Δ_{AB}$.
# - The variance of $Δ_{AB}$ depends on the accuracy of both models $\MNA$ and $\MNB$.
#
# :::{admonition} Reminder
# A value $\bconf$ = 5 (-5) can be interpreted as model $\MNA$ ($\MNB$) being five times more likely than $\MNB$ ($\MNA$) given the data. Crucially, this is a probability ratio is over all experimental models. Therefore, if we find that $\bconf = 5$, then we can conclude that there exists an experimental model $\MeA$ under which $\MNA$ is *at least* five times more likely than $\MNB$, for *any* experimental $\MeB$.
# :::
#
# We further proposed the [empirical model discrepancy (EMD)](sec-definition-EMD) as a method to estimate $\bconf$; the result of the derivation is to write (Eqs. {eq}`eq:bayes-diff-lszo` and {eq}`eq:Elsz-is-Gaussian`)
# \begin{align*}
# Δ_{AB} &\approx \EE[\lAszo] - \EE[\lBszo] \\
#        &\sim \nN\bigl(\musA[], \SigsA[]\bigr) + \nN\bigl( \musB[], \SigsB[] \bigr) \\
#        &\sim \nN\bigl(\musA[]-\musB[], \SigsA[]+\SigsB[] \bigr) \,,
# \end{align*}
# where the $μ_*$ and $Σ_*$ are values that can be computed using only the data and generative models for $\MNA$ and $\MNB$. High absolute values of $\bconf$ are obtained when the overlap between these distributions is small. We provide an [implementation](./emd_paper/emd) of the EMD, which we now want to validate.
#
# Recall that the distributions $\EE[\lAszo] \sim \nN\bigl(\musA[], \SigsA[]\bigr)$ and $\EE[\lBszo] \sim \nN\bigl( \musB[], \SigsB[] \bigr)$ reflect our uncertainty on the data-reproducing models $\MsA$ and $\MsB$ respectively. In our approach, this uncertainty is estimated via the *empirical mismatch* between data and model predictions; this involves a scaling constant $c$, for model converting mismatch to uncertainty. Thus we should have
# \begin{equation}
# \EMD(\MNA, \MNB, c) \approx \bconf(\MNA, \MNB)
# \end{equation}
# for some $c \in \RR_+$.

# %% tags=["hide-input"]
from emd_paper.emd import compute_μ_Σ_m1, fit_Φeaz
from emd_paper import models, config
from emd_paper.utils import glue, format_scientific, SeedGenerator, plot_secondary

# %% tags=["remove-input"]
import numpy as np
from scipy import stats
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, replace
from tqdm.auto import tqdm

# %% tags=["active-ipynb", "remove-input"]
# import logging
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("emd_paper.emd").setLevel(logging.INFO)

# %% tags=["remove-input"]
import holoviews as hv
hv.extension("bokeh")

# %% [markdown]
# ```{div}
# .
# ```
# Our first test involves the distributions $\EE[\lAszo]$ and $\EE[\lBszo]$ for different data set sizes $L$. We use the exponential decay example, with $\MNA$ the true model. In the figure, we represent the Gaussian distributions as solid lines ($μ_*$, mean) and shading ($\sqrt{Σ_*}$, standard deviation). We also compute the $\EMD$ with $c=1$ and plot it for comparison ($\EMD$ scale is on the right). We observe the following, which are essential features for the $\EMD$ to work as advertised:
# - As $L$ increases, the expected likelihood of the true model ($\EE[\lAszo]$) converges to a value higher than the incorrect model.
# - The uncertainty on $\EE[\lAszo]$ goes almost to zero, while that on $\EE[\lBszo]$ remains relatively high. This is because $\MNA$ perfectly predicts the data, and therefore we have high confidence in its likelihood. In contrast, $\MNB$ does not predict the data as well, and so we have less confidence in the likelihood it produces.
#
# For given $L$, we repeat the calculations for each model {glue:text}`R_L` times. This allows us to check:
# - That (for moderately large $L$) the values of statistics $μ_*$ and $Σ_*$ do not depend much on the particular draw of samples (i.e. $\mathrm{std}(μ_*)$ is low).
# - That while errors on the statistics ($\mathrm{std}(\musA[])$ and $\mathrm{std}(\musB[])$) reduce similarly, the estimates of $\SigsB[]$ converge to much higher values than $\SigsA[]$. Therefore $Σ_*$ is not simply a reflection of the estimation error of $μ_*$, but a fundamentally different quantity, determined by the mismatch between model and data.

# %% tags=["hide-input"]
R_L = 4; glue("R_L", R_L, display=False, print_name=False)  # Number of times to repeat experiment for a given data size L
L_lst = [40] + [i*10**p for p in range(2, 6) for i in [1, 2, 4]]  # [40, 100, 200, ..., 100000, 200000, 400000]

MX_model = models.UniformX
xrange = (0, 3)

MN_model = models.PerfectExpon
λ_true = 1
λ_alt = 1.5

Me_model = models.GaussianError
σ = 0.2


# %% tags=["remove-input"]
class SeedGen(SeedGenerator):
    MX_data : int
    MX_synth: int
    Me_data: int
    Me_true: int
    Me_alt : int
        
seedgen = SeedGen(entropy=config.random.entropy)
seeds = seedgen(1)

# %% tags=["remove-input"]
MX_data = MX_model(*xrange, seed=seeds.MX_data)
MX_synth = MX_model(*xrange, seed=seeds.MX_synth)
MN_data = MN_model(λ=λ_true)
MNA = MN_model(λ=λ_true)
MNB = MN_model(λ=λ_alt)
Me_data = Me_model(σ=σ, seed=seeds.Me_data)
MeA = replace(Me_data, seed=seeds.Me_true)
MeB = replace(Me_data, seed=seeds.Me_alt)

# %% [markdown] tags=["remove-cell"]
# Pre-fit the function $p(\lAez)$, since $\MNA$ is deterministic and therefore $p(\lAez)$ is always the same for given model $\MNA$.

# %% tags=["remove-input"]
pleA = fit_Φeaz(MX_data, MNA, MeA, MeA.logpdf,
                nterms=9, tol_synth=0.008)

# %% tags=["remove-input"]
dataA = defaultdict(lambda: [])

# %% tags=["remove-cell"]
MX_data = MX_model(*xrange, seed=seeds.MX_data)  # Reset RNG to initial state
for L in tqdm(L_lst, desc="Data set size"):
    #data[L] = {"μ": [], "Σ": []}
    for _ in tqdm(range(R_L - len(dataA[(L, "μ")])),
                  desc="Resampled data sets", leave=False):
        xdata = MX_data(L)
        ydata = Me_data(xdata, MN_data(xdata))
        
        
        μ, Σ = compute_μ_Σ_m1(MX_data, MNA, MeA, MeA.logpdf, xdata, ydata,
                            c=1, R_z=1, nterms=9, p_synth=pleA)
        dataA[(L, "μ")].append(μ)
        dataA[(L, "Σ")].append(Σ)

# %% tags=["remove-input"]
dfA = pd.DataFrame(dataA, columns=pd.MultiIndex.from_tuples(dataA, names=["L", "stat"]))

μ     = dfA.loc[:,(slice(None), "μ")].mean(axis=0)
stdμ  = dfA.loc[:,(slice(None), "μ")].std(axis=0)
sqrtΣ = np.sqrt(dfA.loc[:,(slice(None), "Σ")]).mean(axis=0)

sqrtΣ.index = pd.MultiIndex.from_tuples(
    [(L, "√Σ") for (L, _) in sqrtΣ.index], names=sqrtΣ.index.names)
stdμ.index = pd.MultiIndex.from_tuples(
    [(L, "std(μ)") for (L, _) in stdμ.index], names=stdμ.index.names)

statsA = pd.concat([μ, sqrtΣ, stdμ]).unstack().reindex(columns=["μ", "√Σ", "std(μ)"])

# %% [markdown] tags=["remove-cell"]
# Pre-fit the function $p(\lBez)$, since $\MNB$ is deterministic and therefore $p(\lBez)$ is always the same for given model $\MNB$.

# %% tags=["remove-input"]
pleB = fit_Φeaz(MX_data, MNB, MeB, MeB.logpdf,
                nterms=9, tol_synth=0.008)

# %% tags=["remove-input"]
dataB = defaultdict(lambda: [])

# %% tags=["remove-cell"]
MX_data = MX_model(*xrange, seed=seeds.MX_data)  # Reset RNG to initial state
for L in tqdm(L_lst, desc="Data set size"):
    #data[L] = {"μ": [], "Σ": []}
    for _ in tqdm(range(R_L - len(dataB[(L, "μ")])),
                  desc="Resampled data sets", leave=False):
        xdata = MX_data(L)
        ydata = Me_data(xdata, MN_data(xdata))
        
        
        μ, Σ = compute_μ_Σ_m1(MX_data, MNB, MeB, MeB.logpdf, xdata, ydata,
                            c=1, R_z=1, nterms=9, p_synth=pleB)
        dataB[(L, "μ")].append(μ)
        dataB[(L, "Σ")].append(Σ)

# %% tags=["remove-input"]
dfB = pd.DataFrame(dataB, columns=pd.MultiIndex.from_tuples(dataB, names=["L", "stat"]))

μ     = dfB.loc[:,(slice(None), "μ")].mean(axis=0)
stdμ  = dfB.loc[:,(slice(None), "μ")].std(axis=0)
sqrtΣ = np.sqrt(dfB.loc[:,(slice(None), "Σ")]).mean(axis=0)

sqrtΣ.index = pd.MultiIndex.from_tuples(
    [(L, "√Σ") for (L, _) in sqrtΣ.index], names=sqrtΣ.index.names)
stdμ.index = pd.MultiIndex.from_tuples(
    [(L, "std(μ)") for (L, _) in stdμ.index], names=stdμ.index.names)

statsB = pd.concat([μ, sqrtΣ, stdμ]).unstack().reindex(columns=["μ", "√Σ", "std(μ)"])

# %% tags=["remove-input"]
pd.concat([statsA, statsB], axis="columns", keys=["model A", "model B"])

# %% tags=["hide-input"]
dfA = dfA.sort_index(axis=1,level="L")
dfB = dfB.sort_index(axis=1,level="L")

μA1 = dfA.loc[:,(slice(None), "μ")].mean(axis=0).to_numpy()
μB1 = dfB.loc[:,(slice(None), "μ")].mean(axis=0).to_numpy()
ΣA1 = dfA.loc[:,(slice(None), "Σ")].mean(axis=0).to_numpy()
ΣB1 = dfB.loc[:,(slice(None), "Σ")].mean(axis=0).to_numpy()

Φ0 = stats.norm.cdf(0, loc=μA1-μB1, scale=np.sqrt(ΣA1+ΣB1))
EMD = np.log((1-Φ0)/Φ0)
# # Equivalent calculation
#x = erf((μA1 - μB1) / np.sqrt(2*(ΣA1 + ΣB1)) )
#EMD = np.log(1-x) - np.log(1+x)

# %% tags=["hide-input"]
# %%opts Overlay [logx=True, legend_position="bottom_right", width=450]
# %%opts Spread (line_color=None)

μA_curve = \
  hv.Curve(statsA.μ).redim.label(μ="E[l*]") \
  * hv.Spread((statsA.index, statsA.loc[:,"μ"].to_numpy(), statsA.loc[:, "√Σ"].to_numpy()),
              kdims=["L"], vdims=["μ*", "σ"], label=f"Model A (λ={λ_true}, true)")

μB_curve = \
  hv.Curve(statsB.μ).redim.label(μ="E[l*]") \
  * hv.Spread((statsB.index, statsB.loc[:,"μ"].to_numpy(), statsB.loc[:, "√Σ"].to_numpy()),
              kdims=["L"], vdims=["μ*", "σ"], label=f"Model B (λ={λ_alt})")

emd_curve = hv.Curve(zip(L_lst, EMD), kdims=["L"], vdims=["EMD"], label="EMD(A, B)").opts(color="grey", hooks=[plot_secondary])

μ_curves = μA_curve * μB_curve
(μ_curves * emd_curve).opts(ylim=(-1, .3))

# %% tags=["active-ipynb", "remove-cell"]
# (μ_curves * emd_curve).opts(ylim=(-1, .3)) \
# .opts(hv.opts.Curve(logx=True, width=4500, height=3000, line_width=10, fontscale=12),
#       hv.opts.Overlay(legend_position="bottom_right"))
#       #hv.opts.Overlay(show_legend=False))

# %%
