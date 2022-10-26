# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Example: Exponential decay

# %% tags=["hide-input"]
import emd_paper

from typing import List,Tuple,Dict
from numbers import Number
from itertools import product
from functools import partial
import math
import numpy as np
import holoviews as hv
# Local imports
from emd_paper import models, config
from emd_paper.utils import glue, format_scientific, SeedGenerator
    # Extended glue which works with calls myst_nb_bokeh.glue when needed

# %% tags=["remove-input"]
hv.extension(config.figures.backend)  # Inject javascript for rendering figures in notebook

# %% tags=["hide-input"]
colors = config.figures.colors
hv.opts.defaults(
    hv.opts.Scatter(alpha=0.7, color=colors["bright"]["grey"]),
    hv.opts.Curve(alpha=0.5),
    hv.opts.Overlay(legend_limit=75)
)

# %% [markdown] tags=["remove-cell"]
# **NOTE**: Holoviews (at least with the bokeh backend) only displays the legend for HoloMaps with less than 45 frames (the limit might be as low as 41). So we want to keep #L x #σ x #μ ⩽ 40.

# %%
#L_lst = [10, 100, 1000, 4000]
L_lst = [25, 1000]

MX_model = models.UniformX
xrange = (0, 3)

MN_model = models.PerfectExpon
λ_true = 1
λ_alt_lst = [1.1, 1.5]

Me_model = models.GaussianError
σ_lst = [1, 0.4, 0.1, 0.04, 0.01]  # Std dev
#γ_lst = [-0.9, -0.5, -0.1, 0, .1] # Skew; only used for asym model

data_label = "Data (sym. error)"
true_label = "Model {M_label} (λ={λ}, true)\nB: 1"
alt_label = "Model {M_label} (λ={λ})\nB: {rbayes}"


# %%
class SeedGen(SeedGenerator):
    MX: int
    Me_data: int
    Me_true: int
    Me_alt : int
        
seedgen = SeedGen(entropy=config.random.entropy)

# %%
#MX = MX_model(*xrange, seed=seeds.MX)
MN_data = MN_model(λ=λ_true)
# MNA = MN_model(λ=λ_true)
# MN_alts = [MN_model(λ=λ_alt) for λ_alt in λ_alt_lst]
# Me_data = Me_model(σ=σ, seed=seeds.Me_data)
#MeA = replace(Me_data, seed=seeds.Me_true)
#MeB = replace(Me_data, seed=seeds.Me_alt)

# %% [markdown]
# **NOTE:** We assume additive noise.

# %%
def get_frames(σ_lst=σ_lst, shape_lst: List[Dict[str, Number]]=[{}], L_lst=L_lst, λ_alt_lst=λ_alt_lst,
               Me_model_data=Me_model, Me_model_logp=Me_model, data_label=data_label):
    # NB: At present `shape` is only passed to Me_data – the idea is that Me_log
    #     is the approximate, and therefore simpler/non-skewed error distribution
    frames = {}
    for σ, shape, L in product(σ_lst, shape_lst, L_lst):
        # Start a fresh color cycle
        color_cycle = iter(colors["bright"]["cycle"].split())  # TODO: config parser should create an MPL cycle
        
        MX = MX_model(*xrange, seed=seedgen.MX(σ, L))
        Me_data = Me_model_data(σ=σ, **shape, seed=seedgen.Me_data(σ, L))
        Me_logp = Me_model_logp(σ=σ, seed=seedgen.Me_data(σ, L))

        x = MX(L)
        z = MN_data(x)
        y = Me_data(x, z)

        σsort = np.argsort(x)
        x = x[σsort]; y = y[σsort]; z = z[σsort]
        
        logL_true = np.sum(Me_logp.logpdf(y, z))  # + np.sum(_MN.logpdf(x,y))  # Assume additive noise
            # We don't add contribution from the data X, since it gets substracted out
        
        data = hv.Scatter(zip(x, y), label=data_label)
        # Allow omitting 5% of data points on each end, so extreme points don't warp axes too much
        data.redim.soft_range(y=(y[int(0.05*len(y))], y[int(0.95*len(y))]))  

        curve_true = hv.Curve(zip(x, z), label=true_label.format(M_label="A", λ=λ_true)).opts(color=next(color_cycle))
        curve_alts = []
        for λ_alt in λ_alt_lst:
            z_alt = MN_model(λ=λ_alt)(x)
            # _MN = MN(*xrange, λ_alt)
            # z_alt = _MN.y_given_x(x)

            logL_alt = np.sum(Me_logp.logpdf(y, z_alt))  # + np.sum(_MN.logpdf(x,y))  # Assume additive noise
                # We don't add contribution from the data X, since it just cancels with logL_true
            try:
                rbayes = math.exp(logL_alt - logL_true)
            except OverflowError:
                rbayes = math.inf

            curve_alts.append(hv.Curve(zip(x, z_alt),
                                       label=alt_label.format(
                                           M_label='BC'[λ_alt_lst.index(λ_alt)],
                                           λ=λ_alt,
                                           rbayes=format_scientific(rbayes))
                                      ).opts(color=next(color_cycle)))

        frames[(σ,L,*shape.values())] = hv.Overlay([data, curve_true, *curve_alts])
    return frames


# %% tags=[]
fig_sym = hv.HoloMap(get_frames(),
                     kdims=[hv.Dimension("σ", label="σ data"),
                            hv.Dimension("L", label="no. data points")])

# %% [markdown] tags=[]
# :::{admonition} Previous explorations of zero-mean noise models
# :class: dropdown
#
# Some of the model/parameter combinations below do achieve higher Bayes factor for the $λ=1.1$ alternative, but they look a bit contrived.
#
# ```python
# fig_asym = hv.HoloMap(get_frames(Me_model_data=models.NormInvGaussError,
#                                  shape_lst=[{"γ":γ} for γ in [-0.9, -0.5, -0.1, 0, .1]],
#                                  ...
# fig_asym = hv.HoloMap(get_frames(Me_model_data=models.SkewNormalError,
#                                  shape_lst=[{"γ":γ} for γ in [-0.9, -0.5, -0.1, 0, .1]],
#                                  ...
# fig_asym = hv.HoloMap(get_frames(Me_model_data=models.GammaError,
#                                  #σ_lst=[0.7, 0.8, 0.9, 1, 1.2],
#                                  #shape_lst=[{"γ":γ} for γ in [12, 16, 22, 26]],
#                                  #σ_lst=[0.1, 0.3, 0.7, 1.2],
#                                  σ_lst=[0.7, 1.2, 1.6, 2, 3, 4, 6, 8, 12],
#                                  shape_lst=[{"γ":γ} for γ in [1, 2, 4, 8, 12, 16]],
#                                  ...
# fig_asym = hv.HoloMap(get_frames(Me_model_data=models.PoissonError,
# fig_asym = hv.HoloMap(get_frames(Me_model_data=models.ExponentialError,
# ```
#
# (NB: The best results were obtained from the Gamma distribution, after adding noise to the $x$ samples (so the reported value is not exactly the true one) and uniformly scaling down the noise to 5% of its original size.)
#
# :::

# %% tags=[]
fig_asym = hv.HoloMap(get_frames(Me_model_data=models.GaussianError,
                                 shape_lst=[{"μ":μ} for μ in [-0.1, -0.03, -0.01]],
                                 data_label="Data (asym. error)"),
                      kdims=[hv.Dimension("σ", label="σ data"),
                             hv.Dimension("L", label="no. data points"),
                             hv.Dimension("μ", label="asym bias")
                            ])

# %% [markdown]
# (sec-fig-exp-decay-example)=
# ## Figure

# %%
defaults = dict(σ=0.1, μ=-0.03, L=1000)

# %%
fig = fig_sym + fig_asym
fig.redim.range(y=(-0.25, 1.5)).redim.default(**defauts)#.opts(hv.opts.Scatter(s=10))
    # Redim b/c a few extreme data points expand the axis, making the decay harder to see

# %% [markdown] tags=["hide-input"]
# Remove interactivity before creating the glue object, otherwise plots don't show up on the pages where they are glued.

# %% tags=["hide-input"]
fig = fig.select(**defaults)

# %% tags=["hide-cell"]
glue("param_exp-decay-λ_true", λ_true)
for i, λ_alt in enumerate(λ_alt_lst, start=1):
    glue(f"param_exp-decay-λ_alt{i}", λ_alt)
glue("param_exp-decay-σdata", defaults["σ"])
glue("param_exp-decay-L", defaults["L"])
glue("fig_exp-decay-example", fig, display=False)

# %% tags=["skip-execution"]
hv.save(fig, "Thought experiment - different exp models - sym and asym.svg", backend="matplotlib")

# %% tags=["remove-input"]
emd_paper.footer
