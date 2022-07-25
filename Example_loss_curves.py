# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python (wcml)
#     language: python
#     name: wcml
# ---

# %% [markdown]
# # Figure: Comparison of log posterior curves
#
# :::{table} Hyperparameters
#
# |Number of fits|$α$|Number of fit steps|
# |---|---|---|
# |65 | 0.9 | 13558 |
# :::
#
# :::{table} Data
#
# |Regime|Input|Noise|
# |---|---|---|
# |Pre-ictal|low $I_0$| very low $D$ |
# |Ictal|high $I_0$| very low $D$ |
# :::

# %% tags=["remove-input"]
import wcml
wcml.setup('numpy', view_only=True)

# %%
import holoviews as hv
from utils import glue  # Extended glue which works with calls myst_nb_bokeh.glue when needed
from wcml.viz import RSView, config, BokehOpts, ColorEvolCurvesByMaxL

# %%
bokeh = True

# %%
if bokeh:
    from myst_nb_bokeh import glue_bokeh
    hv.extension("bokeh")
else:
    hv.extension("matplotlib")


# %% tags=["remove-input"]
def get_logpcurves(rsview):

    color_fn = ColorEvolCurvesByMaxL(rsview.logL_curves(),
                                     quantile=.3, window=1000)

    logpcurves = rsview.logL_curves(color=color_fn)
    logpcurves.opts(width=400,
                    ylim=(-30, 45),#ylim=(None, None)
                    )
    for fit in rsview.fitcoll:  # NB: This relies on FitData caching curves
        if fit.oracle:
            fit.logL_curve.opts(hv.opts.Curve(color="darkgreen", line_width=2))

    if True:
        # For synthetic data, add a line showing the likelihood of the ground truth data
        # FIXME: This only works if the prior is 100% portable between the optimizer and data models
        # logp_gt = get_ground_truth_logp(rsview.last,
        #                                 hist_tags={"latent": "input.ξ",
        #                                            "observed": "observations.ubar"})
        logp_gt = None

        x = logpcurves.dimension_values('step').max()
        y = logpcurves.dimension_values('log L').min()
        if logp_gt:
            gtline = hv.HLine(logp_gt)
            gtlabel = hv.Points([(x,y)], label=f"ground truth: {logp_gt:.1f}")  # B/c HLine can't create legend
            gtlabel.opts(size=0, color="#6d5799")
            gtline.opts(line_dash="dashed", line_width=1, color="#6d5799")
            ov = logpcurves * gtline * gtlabel  # NB: `ov` might actually be a Layout, if logpcurves is a Layout
            logpcurves = ov
        else:
            logpcurves.opts(hv.opts.NdOverlay(show_legend=False), hv.opts.Overlay(show_legend=False))

        logpcurves.opts(hv.opts.Overlay(legend_position="bottom_left"))

    logpcurves.opts(ylim=(-24, -10));
    
    return logpcurves


# %% [markdown]
# ## Pre-ictal curves

# %%
rsview_preictal = RSView().filter.tags({'finished', "<ξtilde>", "ξbar init zero", "beta prior", "normal", "fixed σ", "fixed Σbar"}) \
               .filter.reason("Fixed σ – α=0.9 only – more fits") \
               .filter.after(2022,3,4,22).filter.before(2022,3,6,3)

# %%
logp_preictal = get_logpcurves(rsview_preictal)

# %% [markdown]
# ## Ictal curves

# %%
rsview_ictal = RSView().filter.tags({'finished', "<ξtilde>", "ξbar init zero", "beta prior", "ictal", "fixed σ", "fixed Σbar"}) \
                 .filter.reason("Fixed σ – α=0.9 only – more fits") \
                 .filter.after(2022,2,24,16).filter.before(2022,2,25,21)

# %%
logp_ictal = get_logpcurves(rsview_ictal)

# %% [markdown]
# ## Figure

# %%
fig = logp_preictal + logp_ictal
figname = "fig:logp_curves_compare_fixed_threshold"

if bokeh:
    # Bokeh doesn't support subfigure labels, so we substitute with titles
    logp_preictal.opts(title="A")
    logp_ictal.opts(title="B")

glue(figname, fig)

# %%
wcml.footer
