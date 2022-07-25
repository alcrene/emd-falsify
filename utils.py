def glue(name, variable, display=True, print_name=True):
    """Use either `myst_nb.glue` or `myst_nb_bokeh.glue`.
    
    Which function to use is determined by inspecting the argument.
    If `print_name` is True (default), the glue name is also printed; this can 
    useful to find the correct name to use refer to the figure from the
    rendered document.
    
    Supports: Anything 'myst_nb.glue' supports, Bokeh, Holoviews (Bokeh only)
    
    .. Todo:: Don't assume that Holoviews => Bokeh
    """
    if print_name:   # TODO: Return a more nicely formatted object, with _repr_html_,
        print(name)  # which combines returned fig object and prints the name below
    mrostr = str(type(variable).mro())
    bokeh_output = ("holoviews" in mrostr or "bokeh" in mrostr)
    if bokeh_output:
        from myst_nb_bokeh import glue_bokeh
        if "holoviews" in mrostr:
            import holoviews as hv
            # Convert Holoviews object to normal Bokeh plot
            bokeh_obj = hv.render(variable, backend="bokeh")
        else:
            bokeh_obj = variable
        return glue_bokeh(name, bokeh_obj, display)
    else:
        from myst_nb import glue
        return glue(name, variable, display)
