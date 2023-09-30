import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import holoviews as hv

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

def plot_R_dists(df_emd_cond: pd.DataFrame, colors: list[str], xformatter=None
    ) -> hv.Overlay:
    """
    Plot distributions for the elppd (the expected log pointwise likelihood).
    
    `colors` must be a list at least as long as the number of rows in `df_logp_cond`.
    """
    Θ_list = df_emd_cond.columns.unique("Model")
    # NB: Passing a `group=` argument does not seem to work for Distributions
    emd_dists = [hv.Distribution(df_emd_cond.loc[size, Θlbl], label=f"Θ: {Θlbl}",
                                 kdims= hv.Dimension("logL", label="expected log likelihood per data point"))
                 for size in df_emd_cond.index
                 for Θlbl in Θ_list]
                 
    ov = hv.Overlay(emd_dists)
    ov = ov.redim(logL=hv.Dimension("logL", label="expected log likelihood per data point"))

    #group = sanitize(size)  # What we would like
    group = "Distribution"  # What we need to do because Distribution doesn’t accept group arg
    ov.opts(
        hv.opts.Distribution(labelled=["x"]),
        hv.opts.Distribution(line_alpha=0, fill_alpha=0.35,
                             muted_line_alpha=0, muted_fill_alpha=0.05, muted=True,
                             backend="bokeh"),
        *(hv.opts.Distribution(f"{group}.A", muted=False, backend="bokeh")
          for size in df_emd_cond.index ),
        *(hv.opts.Distribution(f"{group}.{sanitize('Θ: true')}", muted=False, backend="bokeh")
          for size in df_emd_cond.index ),
        *(hv.opts.Distribution(f"{group}.{Θlbl.title()}",
                       color=c,# hatch_pattern=hatch_pattern,
                       backend="bokeh")
          for Θlbl, c in zip(Θ_list, colors)
          for size in df_emd_cond.index ),
        hv.opts.Overlay(legend_position="bottom", legend_cols=len(Θ_list)),
        hv.opts.Overlay(width=600, height=200,
                        legend_spacing=30, backend="bokeh"),
    )
    if xformatter:
        ov.opts(xformatter=xformatter, backend="bokeh")
    
    return ov

def plot_logp_bars(df_logp_cond: pd.DataFrame, data_label: str,
                   colors: list[str], size_dim: str, xformatter=None
    ) -> hv.Overlay:
    """
    Create a plot of vertical marks, with each mark corresponding to a value
    in `def_logp_cond`.
    The span of marks corresponding to the same dataset size is show by a
    horizontal bar above them. The values of dataset sizes should be given in
    of the columns of the DataFrame `df_logp_cond`; `size_dim` indicates the
    name of this column.

    `colors` must be a list at least as long as the number of rows in `df_logp_cond`.
    `size_dim` must match the name of the index level in the DataFrame
    used to indicate the dataset size.
    """
    size_labels = pd.Series(df_logp_cond.index.get_level_values(size_dim), index=df_logp_cond.index)
    size_marker_heights = pd.Series((1.3 + np.arange(len(size_labels))*0.7)[::-1],  # [::-1] places markers for smaller data sets higher
                                    index=df_logp_cond.index)
    logL_dim = hv.Dimension("logL", label=data_label)
    y_dim = hv.Dimension("y", label=" ", range=(-0.5, max(size_marker_heights)))
    size_dim = hv.Dimension("data_size", label=size_dim)
    
    ## Construct the actual lines marking the log likelihood ##
    vlines = [hv.Path([[(logL, -0.5), (logL, 1)]], kdims=[logL_dim, y_dim],
                      group=size, label=Θlbl)
                  .opts(color=c)
                  .opts(line_dash=dash_pattern, backend="bokeh")
                  .opts(linestyle=dash_pattern, backend="matplotlib")
              for (size, dash_pattern) in zip(df_logp_cond.index, dash_patterns)
              for (Θlbl, logL), c in zip(df_logp_cond.loc[size].items(), colors)]
    
    # Legend proxies (Required because Path elements are not included in the legend)
    legend_proxies = [hv.Curve([(0,0)], label=f"Θ: {Θlbl}")
                          .opts(color=c)
                          .opts(linewidth=3, backend="matplotlib")
                      for Θlbl, c in zip(df_logp_cond.columns, colors)]

    ## Construct the markers indicating data set sizes ##
    # These are composed of a horizontal segment above the log L markers, and a label describing the data set size
    logp_centers = (df_logp_cond.max(axis="columns") + df_logp_cond.min(axis="columns"))/2
    df_size_labels = pd.DataFrame(
        (logp_centers, size_marker_heights, size_labels),
        index=["x", "y", size_dim.name]
    ).T
    
    size_labels_labels = hv.Labels(df_size_labels, kdims=["x", "y"], vdims=size_dim)
    
    size_markers = hv.Segments(dict(logL=df_logp_cond.min(axis="columns"),
                                    logL_right=df_logp_cond.max(axis="columns"),
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
        # *(hv.opts.Path(f"{sanitize(size)}.{Θlbl}", color=c)
        #   for Θlbl, c in zip(df_logp_cond.columns, colors.curve) for size in size_labels),
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

def compare_matrix(logp_samples: pd.Series) -> pd.DataFrame:
    """
    Return the Pemd probabilities as a square DataFrame.
    `logp_samples` is a set of evaluations of the log likelihood for each model:
    - index values are the model labels
    - values are arrays of logp evaluations
    """
    P_compare = {}
    for Θ1 in logp_samples.index:
        for Θ2 in logp_samples.index:
            if (Θ1, Θ2) in P_compare:
                continue
            elif (Θ2, Θ1) in P_compare:
                P_compare[(Θ1, Θ2)] = 1-P_compare[(Θ2, Θ1)]
            else:
                samples1, samples2 = logp_samples.loc[[Θ1, Θ2]]
                P_compare[(Θ1, Θ2)] = (samples1[:,None] > samples2).mean()
    # NB: Θ1 is placed along the columns. I think rows are easier to compare, so a transpose the dataframe
    df_cmp = pd.DataFrame.from_dict({Θ1: {Θ2: P_compare[Θ1, Θ2] for Θ2 in logp_samples.index} for Θ1 in logp_samples.index}).T
    return df_cmp.style.format(formatter=lambda x: "{:.1f} %".format(100*x))