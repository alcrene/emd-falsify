---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: py:percent,md:myst
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python (emd-falsify-dev)
  language: python
  name: emd-falsify-dev
---

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
from __future__ import annotations
```

```{code-cell}
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import holoviews as hv
```

```{code-cell}
from dataclasses import dataclass
from typing import Optional, Dict, List
```

```{code-cell}
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: ''
tags: [skip-execution]
---
from .config import config
from . import utils
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
from config import config
import utils
```

```{code-cell}
dash_patterns = ["dotted", "dashed", "solid"]
```

# Plotting functions

```{code-cell}
@dataclass
class CalibrationPlotElements:
    """
    `bin_idcs`: Dictionary indicating which experiment index were assigned to
       each bin. Use in conjunction with the EpistemicDistribution iterator
       to reconstruct specific experiments.
    """
    calibration_curves: hv.Overlay
    prohibited_areas   : hv.Area
    discouraged_areas : hv.Overlay
    bin_idcs: Dict[float,List[np.ndarray[int]]]

    def __iter__(self):
        yield self.calibration_curves
        yield self.prohibited_areas
        yield self.discouraged_areas

    def _repr_mimebundle_(self, *args, **kwds):
        return (self.prohibited_areas * self.discouraged_areas * self.calibration_curves)._repr_mimebundle_(*args, **kwds)
```

```{code-cell}
def calibration_bins(calib_results: CalibrateResult,
                     target_bin_size: Optional[int]=None):
    """Return the bin edges for the histograms produced by `calibration_plot`.
    
    .. Note:: These are generally *not* good bin edges for plotting a histogram
    of calibration results: by design, they will produce an almost
    flat histogram.
    """
    bin_edges = {}
    for c, data in calib_results.items():
        i = 0
        Bemd = np.sort(data["Bemd"])
        edges = [Bemd[0]]
        for w in utils.get_bin_sizes(len(Bemd), target_bin_size)[:-1]:
            i += w
            edges.append(Bemd[i:i+2].mean())
        edges.append(Bemd[-1])
        bin_edges[c] = edges
    return bin_edges
```

```{code-cell}
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
    calib_bins = {}
    for c, data in calib_results.items():
        # # We don’t do the following because it uses the Bconf data to break ties.
        # # If there are a lot equal values (typically happens with a too small c),
        # # then those will get sorted and we get an artificial jump from 0 to 1
        # data.sort(order="Bemd")
        σ = np.argsort(data["Bemd"])  # This will only use Bemd data; order within ties remains random
        Bemd = data["Bemd"][σ]    # NB: Don’t modify original data order: 
        Bconf = data["Bconf"][σ]  #     we may want to inspect it later.

        curve_data = []
        bin_idcs = []
        i = 0
        for w in utils.get_bin_sizes(len(data), target_bin_size):
            curve_data.append((Bemd[i:i+w].mean(),
                               Bconf[i:i+w].mean()))
            bin_idcs.append(σ[i:i+w])
            i += w

        curve = hv.Curve(curve_data, kdims="Bemd", vdims="Bconf", label=f"{c=}")
        curve = curve.redim.range(Bemd=(0,1), Bconf=(0,1))
        curve.opts(hv.opts.Curve(**config.viz.calibration_curves))
        calib_curves[c] = curve
        calib_bins[c] = bin_idcs

    calib_hmap = hv.HoloMap(calib_curves, kdims=["c"])

    ## Prohibited & discouraged areas ##
    # Prohibited area
    prohibited_areas = hv.Area([(x, x, 1-x) for x in np.linspace(0, 1, 32)],
                              kdims=["Bemd"], vdims=["Bconf", "Bconf2"],
                              group="overconfident area")

    # Discouraged areas
    discouraged_area_1 = hv.Area([(x, 1-x, 1) for x in np.linspace(0, 0.5, 16)],
                         kdims=["Bemd"], vdims=["Bconf", "Bconf2"],
                         group="undershoot area")
    discouraged_area_2 = hv.Area([(x, 0, 1-x) for x in np.linspace(0.5, 1, 16)],
                         kdims=["Bemd"], vdims=["Bconf", "Bconf2"],
                         group="undershoot area")

    prohibited_areas = prohibited_areas.redim.range(Bemd=(0,1), Bconf=(0,1))
    discouraged_area_1 = discouraged_area_1.redim.range(Bemd=(0,1), Bconf=(0,1))
    discouraged_area_2 = discouraged_area_2.redim.range(Bemd=(0,1), Bconf=(0,1))

    prohibited_areas.opts(hv.opts.Area(**config.viz.prohibited_area))
    discouraged_area_1.opts(hv.opts.Area(**config.viz.discouraged_area))
    discouraged_area_2.opts(hv.opts.Area(**config.viz.discouraged_area))

    ## Combine & return ##
    return CalibrationPlotElements(
        calib_hmap, prohibited_areas, discouraged_area_1*discouraged_area_2,
        calib_bins)
```

sanitize = hv.core.util.sanitize_identifier_fn

+++

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

+++

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

+++

    ## Construct the actual lines marking the log likelihood ##
    vlines = [hv.Path([[(logL, -0.5), (logL, 1)]], kdims=[logL_dim, y_dim],
                      group=size, label=model_lbl)
                  .opts(color=c)
                  .opts(line_dash=dash_pattern, backend="bokeh")
                  .opts(linestyle=dash_pattern, backend="matplotlib")
              for (size, dash_pattern) in zip(df_emd_cond.index, dash_patterns)
              for (model_lbl, logL), c in zip(df_emd_cond.loc[size].items(), colors)]

+++

    # Legend proxies (Required because Path elements are not included in the legend)
    legend_proxies = [hv.Curve([(0,0)], label=f"model: {model_lbl}")
                          .opts(color=c)
                          .opts(linewidth=3, backend="matplotlib")
                      for model_lbl, c in zip(df_emd_cond.columns, colors)]

+++

    ## Construct the markers indicating data set sizes ##
    # These are composed of a horizontal segment above the log L markers, and a label describing the data set size
    logp_centers = (df_emd_cond.max(axis="columns") + df_emd_cond.min(axis="columns"))/2
    df_size_labels = pd.DataFrame(
        (logp_centers, size_marker_heights, size_labels),
        index=["x", "y", size_dim.name]
    ).T

+++

    size_labels_labels = hv.Labels(df_size_labels, kdims=["x", "y"], vdims=size_dim)

+++

    size_markers = hv.Segments(dict(logL=df_emd_cond.min(axis="columns"),
                                    logL_right=df_emd_cond.max(axis="columns"),
                                    y0=size_marker_heights.to_numpy(),
                                    y1=size_marker_heights.to_numpy()),
                               [logL_dim, "y0", "logL_right", "y1"])

+++

    ## Assemble and configure options ##
    ov = hv.Overlay([*vlines, *legend_proxies, size_markers, size_labels_labels])

+++

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

+++

    return ov
