from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import holoviews as hv

from typing import NamedTuple, Optional

from .config import config
from . import utils

dash_patterns = ["dotted", "dashed", "solid"]

sanitize = hv.core.util.sanitize_identifier_fn

## Utility functions ##

def despine_hook(plot, element):
    """Apply seaborn.despine. Matplotlib hook."""
    sns.despine(ax=plot.handles["axis"], trim=True, offset=5)

def no_spine_hook(*sides):
    """Remove the specified spine(s) completely. Matplotlib hook."""
    def hook(plot, element):
        ax = plot.handles["axis"]
        for side in sides:
            ax.spines[side].set_visible(False)
    return hook

# def transparent_hook(plot, element):
#     """Make the axes transparent. Matplotlib hook."""
#     ax = plot.handles["axis"]
#     ax.patch.set_alpha(0.)

def lighten(c, factor):
    """
    Return a hex color with same hue and saturation, but new lightness value.
    Positive values for `factor` make the color lighter, negative values make it darker.
    """
    h,s,v = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(c))
    vsig = np.log(v/(1-v))
    vsig += np.log(factor)
    v = 1 / (1 + np.exp(-vsig))
    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))

## Plotting functions ##

class CalibrationPlotElements(NamedTuple):
    calibration_curves: hv.Overlay
    prohibited_area   : hv.Area
    discouraged_areas : hv.Overlay

    def _repr_mimebundle_(self, *args, **kwds):
        return (prohibited_area * discouraged_areas * calibration_curves)._repr_mimebundle_(*args, **kwds)

def calibration_plot(calib_results: CalibrateResult,
                     target_bin_size: Optional[int]=None
                    ) -> CalibrationPlotElements:
    """Create a calibration plot from the results of calibration experiments.

    Parameters
    ----------
    calib_results: The calibration results to plot. The typical way to obtain
       these is to create and run `Calibrate` task:
       >>> task = emd_falsify.tasks.Calibrate(...)
       >>> calib_results = task.unpack_results(task.run())
    target_bin_size: Each point on the calibration curve is an average over
       some number of calibration experiments; this parameter sets that number.
       (The actual number may vary a bit, if `target_bin_size` does not exactly
       divide the total number of samples.)
       Larger bin sizes result in fewer but more accurate curve points.
       The default is to aim for the largest bin size possible which results
       in 16 curve points, with some limits in case the number of results is
       very small or very large.
    """

    ## Calibration curves ##
    calib_curves = {}
    for c, data in calib_results.items():
        data.sort(order="Bemd")

        N = len(data)
        if target_bin_size is None:
            # Try to get 16 points for the curve, but keep bin sizes between 16 and 64.
            # (So if there are too few samples, we will get fewer curve points but each
            #  will be the average of at least 16 samples)
            # The target of 16 points for the curve was chosen based on a plot of tanh
            # between -2 and +2: kinks are barely noticeable. (We want the smallest
            # number of points since then each point has more statistical power.)
            # The bin size limits 16 and 64 are more arbitrary.
            # >>> xarr = np.linspace(-2, 2, 16)
            # >>> hv.Curve(zip(xarr, np.tanh(xarr))).opts(fig_inches=5)
            target_bin_size = np.clip(N//16, 16, 64).astype(int)

        curve_data = []
        i = 0
        for j, w in enumerate(utils.get_bin_sizes(N, target_bin_size)):
            curve_data.append((data[i:i+w]["Bemd"].mean(),
                               data[i:i+w]["Bconf"].mean()))
            i += w

        curve = hv.Curve(curve_data, kdims="Bemd", vdims="Bconf", label=f"{c=}")
        curve = curve.redim.range(Bemd=(0,1), Bconf=(0,1))
        curve.opts(hv.opts.Curve(**config.viz.calibration_curves))
        calib_curves[c] = curve

    calib_hmap = hv.HoloMap(calib_curves, kdims=["c"])

    ## Prohibited & discouraged areas ##
    # Prohibited area
    prohibited_area = hv.Area([(x, x, 1-x) for x in np.linspace(0, 1, 32)],
                              kdims=["Bemd"], vdims=["Bconf", "Bconf2"],
                              group="overconfident area")

    # Discouraged areas
    discouraged_area_1 = hv.Area([(x, 1-x, 1) for x in np.linspace(0, 0.5, 16)],
                         kdims=["Bemd"], vdims=["Bconf", "Bconf2"],
                         group="undershoot area")
    discouraged_area_2 = hv.Area([(x, 0, 1-x) for x in np.linspace(0.5, 1, 16)],
                         kdims=["Bemd"], vdims=["Bconf", "Bconf2"],
                         group="undershoot area")

    prohibited_area = prohibited_area.redim.range(Bemd=(0,1), Bconf=(0,1))
    discouraged_area_1 = discouraged_area_1.redim.range(Bemd=(0,1), Bconf=(0,1))
    discouraged_area_2 = discouraged_area_2.redim.range(Bemd=(0,1), Bconf=(0,1))

    prohibited_area.opts(hv.opts.Area(**config.viz.prohibited_area))
    discouraged_area_1.opts(hv.opts.Area(**config.viz.discouraged_area))
    discouraged_area_2.opts(hv.opts.Area(**config.viz.discouraged_area))

    ## Combine & return ##
    return CalibrationPlotElements(
        calib_hmap, prohibited_area, discouraged_area_1*discouraged_area_2)


def plot_R_dists(df_emd_cond: pd.DataFrame, colors: list[str], xformatter=None
    ) -> hv.Overlay:
    """
    Plot distributions of the expected risk for different models
    
    `colors` must be a list at least as long as the number of rows in `df_emd_cond`.
    """
    model_list = df_emd_cond.columns.unique("Model")
    # NB: Passing a `group=` argument does not seem to work for Distributions
    emd_dists = [hv.Distribution(df_emd_cond.loc[size, model_lbl], label=f"model: {model_lbl}",
                                 kdims= hv.Dimension("R", label="expected risk"))
                 for size in df_emd_cond.index
                 for model_lbl in model_list]
                 
    ov = hv.Overlay(emd_dists)
    ov = ov.redim(logL=hv.Dimension("R", label="expected Risk"))

    #group = sanitize(size)  # What we would like
    group = "Distribution"  # What we need to do because Distribution doesn’t accept group arg
    ov.opts(
        hv.opts.Distribution(labelled=["x"]),
        hv.opts.Distribution(line_alpha=0, fill_alpha=0.35,
                             muted_line_alpha=0, muted_fill_alpha=0.05, muted=True,
                             backend="bokeh"),
        *(hv.opts.Distribution(f"{group}.A", muted=False, backend="bokeh")
          for size in df_emd_cond.index ),
        *(hv.opts.Distribution(f"{group}.{sanitize('model: true')}", muted=False, backend="bokeh")
          for size in df_emd_cond.index ),
        *(hv.opts.Distribution(f"{group}.{model_lbl.title()}",
                       color=c,# hatch_pattern=hatch_pattern,
                       backend="bokeh")
          for model_lbl, c in zip(model_list, colors)
          for size in df_emd_cond.index ),
        hv.opts.Overlay(legend_position="bottom", legend_cols=len(model_list)),
        hv.opts.Overlay(width=600, height=200,
                        legend_spacing=30, backend="bokeh"),
    )
    if xformatter:
        ov.opts(xformatter=xformatter, backend="bokeh")
    
    return ov

def plot_R_bars(df_emd_cond: pd.DataFrame, data_label: str,
                   colors: list[str], size_dim: str, xformatter=None
    ) -> hv.Overlay:
    """
    Create a plot of vertical marks, with each mark corresponding to a value
    in `df_emd_cond`.
    The span of marks corresponding to the same dataset size is show by a
    horizontal bar above them. The values of dataset sizes should be given in
    of the columns of the DataFrame `df_emd_cond`; `size_dim` indicates the
    name of this column.

    `colors` must be a list at least as long as the number of rows in `df_emd_cond`.
    `size_dim` must match the name of the index level in the DataFrame
    used to indicate the dataset size.
    """
    size_labels = pd.Series(df_emd_cond.index.get_level_values(size_dim), index=df_emd_cond.index)
    size_marker_heights = pd.Series((1.3 + np.arange(len(size_labels))*0.7)[::-1],  # [::-1] places markers for smaller data sets higher
                                    index=df_emd_cond.index)
    logL_dim = hv.Dimension("logL", label=data_label)
    y_dim = hv.Dimension("y", label=" ", range=(-0.5, max(size_marker_heights)))
    size_dim = hv.Dimension("data_size", label=size_dim)
    
    ## Construct the actual lines marking the log likelihood ##
    vlines = [hv.Path([[(logL, -0.5), (logL, 1)]], kdims=[logL_dim, y_dim],
                      group=size, label=model_lbl)
                  .opts(color=c)
                  .opts(line_dash=dash_pattern, backend="bokeh")
                  .opts(linestyle=dash_pattern, backend="matplotlib")
              for (size, dash_pattern) in zip(df_emd_cond.index, dash_patterns)
              for (model_lbl, logL), c in zip(df_emd_cond.loc[size].items(), colors)]
    
    # Legend proxies (Required because Path elements are not included in the legend)
    legend_proxies = [hv.Curve([(0,0)], label=f"model: {model_lbl}")
                          .opts(color=c)
                          .opts(linewidth=3, backend="matplotlib")
                      for model_lbl, c in zip(df_emd_cond.columns, colors)]

    ## Construct the markers indicating data set sizes ##
    # These are composed of a horizontal segment above the log L markers, and a label describing the data set size
    logp_centers = (df_emd_cond.max(axis="columns") + df_emd_cond.min(axis="columns"))/2
    df_size_labels = pd.DataFrame(
        (logp_centers, size_marker_heights, size_labels),
        index=["x", "y", size_dim.name]
    ).T
    
    size_labels_labels = hv.Labels(df_size_labels, kdims=["x", "y"], vdims=size_dim)
    
    size_markers = hv.Segments(dict(logL=df_emd_cond.min(axis="columns"),
                                    logL_right=df_emd_cond.max(axis="columns"),
                                    y0=size_marker_heights.to_numpy(),
                                    y1=size_marker_heights.to_numpy()),
                               [logL_dim, "y0", "logL_right", "y1"]) 

    ## Assemble and configure options ##
    ov = hv.Overlay([*vlines, *legend_proxies, size_markers, size_labels_labels])
    
    # For some reason, applying these opts separately is more reliable with matplotlib backend
    size_markers.opts( 
        hv.opts.Segments(color="black"),
        hv.opts.Segments(line_width=1, backend="bokeh"),
        hv.opts.Segments(linewidth=1, backend="matplotlib")
    )
    size_labels_labels.opts(
        hv.opts.Labels(yoffset=0.2),
        hv.opts.Labels(yoffset=0.2, text_font_size="8pt", text_align="center", backend="bokeh"),
        hv.opts.Labels(yoffset=0.4, size=10, verticalalignment="top", backend="matplotlib")
    )
    ov.opts(
        hv.opts.Path(yaxis="bare", show_frame=False),
        # *(hv.opts.Path(f"{sanitize(size)}.{model_lbl}", color=c)
        #   for model_lbl, c in zip(df_emd_cond.columns, colors.curve) for size in size_labels),
        # *(hv.opts.Path(sanitize(size), line_dash=dashpattern, backend="bokeh")
        #   for size, dashpattern in zip(
        #       size_labels, dash_patterns)),
        hv.opts.Path(line_width=3, backend="bokeh"),
        hv.opts.Path(linewidth=3, backend="matplotlib"),
        hv.opts.Overlay(yaxis="bare", show_frame=False, padding=0.175,
                        show_legend=True, legend_position="bottom"),
        hv.opts.Overlay(width=600, height=200,
                        legend_spacing=30, backend="bokeh")
    )
    if xformatter:
        ov.opts(xformatter=xformatter, backend="bokeh")
    ov.opts(hooks=[no_spine_hook("left")], backend="matplotlib")
    
    return ov

def Bemd_matrix(R_samples: pd.Series) -> pd.DataFrame:
    """
    Return the Bemd probabilities as a square DataFrame.
    `R_samples` is a set of samples of the expected risks for each model:
    - index values are the model labels
    - values are arrays of R evaluations
    """
    P_compare = {}
    for model1 in R_samples.index:
        for model2 in R_samples.index:
            if (model1, model2) in P_compare:
                continue
            elif (model2, model1) in P_compare:
                P_compare[(model1, model2)] = 1-P_compare[(model2, model1)]
            else:
                samples1, samples2 = R_samples.loc[[model1, model2]]
                P_compare[(model1, model2)] = np.less.outer(samples1, samples2).mean()
                # P_compare[(model1, model2)] = (samples1[:,None] > samples2).mean()
    # NB: model1 is placed along the columns. I think rows are easier to compare, so a transpose the dataframe
    df_cmp = pd.DataFrame.from_dict({model1: {model2: P_compare[model1, model2] for model2 in R_samples.index} for model1 in R_samples.index}).T
    return df_cmp.style.format(formatter=lambda x: "{:.1f} %".format(100*x))