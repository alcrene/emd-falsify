# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---

import math
import numpy as np
import dataclasses
from collections import namedtuple
import logging
from inspect import unwrap

import IPython

from typing import ClassVar, Optional, Union, Tuple, NamedTuple, Literal
from scityping.numpy import NPValue, Array

import myst_nb

from .find_sane_dt import make_int_superscript

logger = logging.getLogger(__name__)


# ## glue

def glue(name, variable, display=True, print_name=True,
         backend: Optional[Literal['bokeh', 'matplotlib']]=None):
    """Use either `myst_nb.glue` or `myst_nb_bokeh.glue`.
    
    Which function to use is determined by inspecting the argument.
    If `print_name` is True (default), the glue name is also printed; this can 
    useful to find the correct name to use refer to the figure from the
    rendered document.
    
    Supports: Anything 'myst_nb.glue' supports, Bokeh, Holoviews
    """
    # TODO: Return a more nicely formatted object, with _repr_html_,
    #       which combines returned fig object and prints the name below

    if backend is None:
        backend = glue.default_holoviews_backend
    if backend is None:
        raise TypeError("Backend not specified. Either provide as argument, "
                        "or set `glue.default_holoviews_backend`")
    elif backend not in {"bokeh", "matplotlib"}:
        raise ValueError("config.backend should be either 'bokeh' or 'matplotlib'")
    
    if print_name:   
        if IPython.get_ipython():
            IPython.display.display(name)  # Should look nicer in Jupyter Book, especially when there are multiple glue statements
        else:
            print(name)
    mrostr = str(type(variable).mro()).lower()
    if "bokeh" in mrostr:
        # Normal Bokeh object (since HoloViews objects are renderer-agnostic, they should never have "bokeh" in their class inheritance)
        from myst_nb_bokeh import glue_bokeh
        return glue_bokeh(name, bokeh_obj, display)

    if "holoviews" in mrostr:
        import holoviews as hv
        if backend == "bokeh":
            from myst_nb_bokeh import glue_bokeh
            # Render Holoviews object to get a normal Bokeh plot
            bokeh_obj = hv.render(variable, backend="bokeh")
            return glue_bokeh(name, bokeh_obj, display)
        else:
            # Render Holoviews object to get a normal Matplotlib plot
            mpl_obj = hv.render(variable, backend="matplotlib")
            return myst_nb.glue(name, mpl_obj, display)

    else:
        return myst_nb.glue(name, variable, display)

glue.default_holoviews_backend = None

# ## format_scientific

def format_scientific(a: Union[int,float], sig_digits=3) -> str:
    """
    Format a number in scientific notation, with the requested number of
    significant digits.
    """
    # First deal with the sign, since log10 requires strictly positive values
    if a < 0:
        sgn = "-"
        a = -a
    elif a == 0:
        return "0." + "0"*(sig_digits-1)
    else:
        sgn = ""
    # First round the number to avoid issues with things like 0.99999
    if not math.isfinite(a):
        if a == math.inf:
            return "∞"
        elif a == -math.inf:
            return "-∞"
        else:
            return str(a)
    p = int(math.log10(a))
    if p >= 0:
        a = round(a / 10**p, sig_digits-1)
    else:
        a = round(a / 10**p, sig_digits)  # Need one digit more, because we use the first one to replace the initial '0.'
    # Now we have a good value a with non-pathological loading on the first digit
    # Since a has changed though, we should check again that p is correct
    p2 = int(math.log10(a))  # Most of the time this will be 0
    p += p2
    i, f = str(float(a/10**p2)).split(".")
    if i == "0":
        i, f = f[0], f[1:]
        p -= 1
    f = (f+"0"*sig_digits)[:sig_digits-1]
    #i = int(a // 10**p2)
    #f = str(a % 10**p2).replace('.','')[:sig_digits-1]
    if p == 0:
        return f"{sgn}{i}.{f}"
    else:
        return f"{sgn}{i}.{f}×10{make_int_superscript(p)}"

# ### Test

# + tags=["active-ipynb"]
# if __name__ == "__main__":
#     assert format_scientific(0.9999999999999716) == "1.00"
#     assert format_scientific(1.0000000000000284) == "1.00"
#     assert format_scientific(9999.999999999716) == "1.00×10⁴"
#     assert format_scientific(1000.0000000000284) == "1.00×10³"
#     assert format_scientific(5.34) == "5.34"
#     assert format_scientific(32175254) == "3.22×10⁷"
#     assert format_scientific(0.000002789) == "2.79×10⁻⁶"
#     assert format_scientific(0.000002781) == "2.78×10⁻⁶"
#
#     assert format_scientific(0) == "0.00"
#
#     assert format_scientific(-0.9999999999999716) == "-1.00"
#     assert format_scientific(-1.0000000000000284) == "-1.00"
#     assert format_scientific(-9999.999999999716) == "-1.00×10⁴"
#     assert format_scientific(-1000.0000000000284) == "-1.00×10³"
#     assert format_scientific(-5.34) == "-5.34"
#     assert format_scientific(-32175254) == "-3.22×10⁷"
#     assert format_scientific(-0.000002789) == "-2.79×10⁻⁶"
#     assert format_scientific(-0.000002781) == "-2.78×10⁻⁶"
# -

# ## SeedGenerator

import numbers


class SingleSeedGenerator(np.random.SeedSequence):
    """
    Make SeedSequence callable, allowing to create multiple high-quality seeds.
    On each call, the arguments are converted to integers (via hashing), then 
    used to produce a unique seed.
    By default, a single integer seed is returned per call.
    To make each call return multiple seeds, use the keyword argument `length`
    when creating (not calling) the seed generator. When `length` is greater
    than one, seeds are returned as NumPy vector.

    .. Important:: Multiple calls with the same arguments will return the same seed.
    .. Note:: Recommended usage is through `SeedGenerator`.
    """
    def __init__(self, *args, length=1, **kwargs):
        self.length = length  # The generated state will have this many integers
        super().__init__(*args, **kwargs)
    def __call__(self, *key,
        ) -> Union[NPValue[np.unsignedinteger], Array[np.unsignedinteger, 1]]:
        """
        If length=1, returns an integer. Otherwise returns an array.
        """
        # Convert keys to ints. stableintdigest also works with ints, but
        # returns a different value – it seems that this could be suprising,
        # so if we don't need to, we don't apply it.
        key = tuple(_key if isinstance(key, numbers.Integral)
                       else stableintdigest(_key)
                    for _key in key)
        seed = np.random.SeedSequence(self.entropy,
                                      spawn_key=self.spawn_key+key
               ).generate_state(self.length)
        if self.length == 1:
            seed = seed[0]
        return seed


@dataclasses.dataclass
class SeedGenerator:
    """
    Maintain multiple, mutually independent seed generators.
    
    For each named seed, the length of the required state vector is inferred
    from the annotations. For example,
    
    >>> class SeedGen(SeedGenerator):
    >>>     data: Tuple[int, int]
    >>>     noise: int
            
    will generator length-2 state vectors for ``data`` and scalars for ``noise``.
    (Length 1 arrays are automatically converted to scalars.)
    
    .. Important:: Only the types ``int`` and ``Tuple`` are supported.
       The ellipsis (``...``) argument to ``Tuple`` is not supported.
       Other types may work accidentally.
       
    **Usage**
    
    Initialize a seed generator with a base entropy value (for example obtained
    with ``np.random.SeedSequence().entropy``):
    
    >>> seedgen = SeedGen(entropy)
        
    Seeds are generated by providing a key, which can be of any length and contain
    either integers, floats or strings:
    
    >>> seedgen.noise(4)
    # 760562028
    >>> seedgen.noise("tom", 6.4)
    # 3375185240
    
    To get a named tuple storing a seed value for every attribute, pass the key
    to the whole generator
    
    >>> seedgen(4)
    # SeedValues(data=array([2596421399, 1856282581], dtype=uint32), noise=760562028)
    
    Note that the value generated for ``seedgen.noise`` is the same.
    """
    entropy: dataclasses.InitVar[int]
    SeedValues: ClassVar[NamedTuple]
    def __init__(self, entropy):
        seedseq = np.random.SeedSequence(entropy)
        seednames = [nm for nm, field in self.__dataclass_fields__.items()
                     if field._field_type == dataclasses._FIELD]
        # for nm, sseq in zip(seednames, seedseq.spawn(len(seednames))):
        #     setattr(self, nm, sseq.generate_state(1)[0])
        for i, nm in enumerate(seednames):
            # Get the length of the required state from the annotations
            seedT = self.__dataclass_fields__[nm].type
            length = len(getattr(seedT, "__args__", [None]))  # Defaults to length 1 if type does not define length
            # Set the seed attribute
            setattr(self, nm, SingleSeedGenerator(entropy, spawn_key=(i,), length=length))
    def __init_subclass__(cls):
        # Automatically decorate all subclasses with @dataclasses.dataclass
        # We want to use the __init__ of the parent, so we disable automatic creation of __init__
        dataclasses.dataclass(cls, init=False)
        seednames = [nm for nm, field in cls.__dataclass_fields__.items()
                     if field._field_type == dataclasses._FIELD]
        cls.SeedValues = namedtuple(f"SeedValues", seednames)
    def __call__(self, key):
        return self.SeedValues(**{nm: getattr(self, nm)(key)
                                  for nm in self.SeedValues._fields})

# ### Test

# + tags=["active-ipynb"]
# class SeedGen(SeedGenerator):
#     data: Tuple[int, int]
#     noise: int
# seedgen = SeedGen(123456789)
# seeds1a = seedgen(1)
# seeds1b = seedgen(1)
# seeds2 = seedgen(2)
# assert (seeds1a.data == seeds1b.data).all() and seeds1a.noise == seeds1b.noise
# assert (seeds1a.data != seeds2.data).all() and seeds1a.noise != seeds2.noise
# assert len(seeds2.data) == 2
# assert np.isscalar(seeds2.noise)
#
# # We also allow using non-integer values for the keys. This allows to easily
# # generate different keys for different parameter values.
# # assert seedgen.data("b", 4.3) != seedgen.data(4.3, "b") != seedgen.data(4.3, "a")
# #   (Unique both in values and order)
# -

# ## Parameter collections

# import logging
from collections.abc import Mapping, Sequence, Generator
from dataclasses import dataclass, field
try:
    from dataclasses import KW_ONLY
except ImportError:
    # With Python < 3.10, all parameters in subclasses will need to be specified, but at least the code won’t break
    KW_ONLY = None
from itertools import product, repeat, islice
from math import prod
from typing import List, Union
from numpy.typing import ArrayLike
try:
    from scipy import stats
except ModuleNotFoundError:
    stats = None

# Allow NumPy arrays to be recognized as sequences. Other Sequence-compatible types can be added is needed, if they don't already register themselves as virtual subclasses.

import numpy as np
Sequence.register(np.ndarray);

Seed = Union[int, ArrayLike, np.random.SeedSequence]
# Scipy.stats does not provide a public name for the frozen dist types
if stats:
    RVFrozen = type(stats.norm()).mro()[1]
    MultiRVFrozen = type(stats.multivariate_normal()).mro()[1]
else:
    class RVFrozen:  # These are only used in isinstance() checks, so an empty
        pass         # class suffices to avoid those tests failing and simply return `False`
    class MultiRVFrozen:
        pass

class NoArg:  # Sentinel value: used to identify when no argument was passed, if
    pass      # if `None` cannot be used for that purpose

class expand(Sequence):
    def __init__(self, seq: Sequence):
        if not isinstance(seq, Sequence):
            raise TypeError("`seq` must be a Sequence (i.e. a non-consuming iterable).\n"
                            "If you know your argument type is compatible with a Sequence, "
                            "you can indicate this by registering it as a virtual subclass:\n"
                            "    from collections.abc import Sequence\n"
                            "    Sequence.register(MyType)")
        self._seq = seq
    def __len__(self):
        return self._seq.__len__()
    def __getitem__(self, key):
        return self._seq.__getitem__(key)
    def __str__(self):
        return str(self._seq)
    def __repr__(self):
        return f"expand({repr(self._seq)})"
    def __eq__(self, other):
        return self._seq == other


@dataclass
class ParamColl(Mapping):
    """
    A container for parameter sets, which allows expanding lists parameters.
    Implemented as a dataclass, with an added Mapping API to facilitate use for
    keyword arguments.

    .. rubric:: Expandable parameters

       - `outer()` will expand every expandable parameter separately and
          return a new `ParamColl` for each possible combination.
          This is akin to itertools’s `product`, or a mathematical outer product.
       - `inner()` will expand every expandable parameter simultaneously and
          return a new `ParamColl` for each combination.
          This is akin to `zip`, or a mathematical inner product.
       Parameters are made expandable by wrapping an iterable with `emdd.utils.expand`.

    .. rubric:: Random parameters

       So called “frozen” random variables from `scipy.stats` may be used as
       parameters. To ensure reproducibility, in this case a seed *must* be
       specified. (If you really want different values on each call, pass
       `None` as the seed.)
       If there are only random or scalar parameters, the `ParamColl` is of
       infinite size.
       If there are also expandable parameters, the `ParamColl` has inner/outer
       size determined by the expandable parameters.

    .. rubric:: Use as keyword arguments

       The primary use case for `ParamColl` instances is as keyword arguments.
       To make this easier, instances provide a mapping interface:
       if ``params`` is a `ParamColl` instance, then ``f(**params)`` will pass
       all its public attributes as keyword arguments.

    .. rubric:: Private attributes

       Attributes whose name start with an underscore ``_`` are private:
       they are excluded from the values returned by
       `.keys()`, `.values()`, `.items()`, and `.kdims`.

    .. rubric:: Parameter dimensions
       Dimension instances (such as those created with `holoviews.Dimension`)
       can be assigned to parameters by updating the class’s `dims` dictionary.
       Keys in `dims` should correspond to parameter names.
       `kdims` will preferentially return the values in `dims` when a dimension
       matching a parameter name is found, otherwise it returns the parameter name.
    """
    dims = {}  # Can optionally expand this with hv.Dimension instances
               # Missing dimensions will use the default ``hv.Dimension(θname)``
    _       : KW_ONLY  # kw_only required, otherwise subclasses need to define defaults for all of their values
    seed    : Union[Seed,NoArg] = field(default=NoArg)  # NB: kw_only arg here would be cleaner, but would break for Python <3.10
    _lengths: List[int] = field(init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        # NB: We use object.__setattr__ to avoid triggering `self.__setattr__` (and thus recursion errors and other nasties)
        object.__setattr__(self, "_lengths",
            [len(v) if isinstance(v, expand)
             else math.inf if isinstance(v, RVFrozen)
             else 1 for k, v in self.items()])

        if math.inf in self._lengths:
            if self.seed is NoArg:
                raise TypeError("A seed is required when some of the parameters "
                                "are specified as random variables.")
        else:
            if self.seed is not NoArg:
                logger.info("A seed is not necessary if none of the arguments "
                            "are random. It was set to `NoArg` to ensure "
                            "consistent hashes.")
                object.__setattr__(self, "seed", NoArg)

        object.__setattr__(self, "_initialized", True)

    # Rerun __post_init__ every time a parameter is changed
    def __setattr__(self, attr, val):
        super().__setattr__(attr, val)
        if self._initialized:
            self.__post_init__()
        
    ## Mapping API ##

    def __len__(self):
        return len(self._lengths)
                   
    def __iter__(self):
        yield from self.keys()
    
    @classmethod
    def keys(cls):  # TODO: Return something nicer, like a KeysView ?
        return [k for k in cls.__dataclass_fields__
                if not k.startswith("_") and k not in ParamColl.__dataclass_fields__]  # Exclude private fields and those in the base class
    def __getitem__(self, key):
        return getattr(self, key)

    ## Other descriptors ##

    @classmethod
    @property
    def kdims(cls):
        return [cls.dims.get(θname, θname) for θname in self.keys()]
    
    # Expansion API
                       
    @property
    def outer_len(self):
        """
        The length of the iterable created by `outer()`.
        
        If outer products are not supported the outer length is `None`.
        (This happens if the parameter iterables are all infinite)
        """
        L = prod(self._lengths)
        if L != math.inf:
            return L 
        else:
            # If there are only lengths 1 or inf, we return inf
            # Otherwise, the iterator will terminate once the finite expansians are done
            L = prod(l for l in self._lengths if l < math.inf)
            return L if L != 1 else None
    
    @property
    def inner_len(self):
        """
        The length of the iterable created by `inner()`.

        If inner products are not supported, the inner length is `None`.
        """
        diff_lengths = set(self._lengths) - {1}
        if len(diff_lengths - {math.inf}) > 1:
            return None
        elif diff_lengths == {math.inf}:
            return math.inf
        elif len(diff_lengths) == 0:
            # There are no parameters to exand
            return 1
        else:
            return next(iter(diff_lengths - {math.inf}))  # Length is that of the non-infinite iterator
    
    def inner(self, start=None, stop=None, step=None):
        if start is not None or stop is not None or step is not None:
            yield from islice(self.inner(), start, stop, step)
        else:
            for kw in self._get_kw_lst_inner():
                yield type(self)(**kw)
            
    def outer(self, start=None, stop=None, step=None):
        if start is not None or stop is not None or step is not None:
            yield from islice(self.outer(), start, stop, step)
        else:
            for kw in self._get_kw_lst_outer():
                yield type(self)(**kw)

    ## Private methods ##
   
    def get_rng(self):
        return np.random.Generator(np.random.PCG64(self.seed))

    @staticmethod
    def _make_rv_iterator(rv, random_state, size=None, max_chunksize=1024):
        """
        Return an amortized infinite iterator: each `rvs` call requests twice
        as many samples as the previous call, up to `max_chunksize`.
        """
        if size is None:
            # Size unknown: Return an amortized infinite iterator
            chunksize = 1
            while True:
                chunksize = min(chunksize, max_chunksize)
                yield from rv.rvs(chunksize, random_state=random_state)
                chunksize *= 2
        else:
            # Size known: draw that many samples immediately
            k = 0
            while k < size:
                chunksize = min(size-k, max_chunksize)
                yield from rv.rvs(chunksize, random_state=random_state)
                k += chunksize

    def _get_kw_lst_inner(self):
        if self.seed is not NoArg:
            rng = self.get_rng()
        inner_len = self.inner_len
        if inner_len is None:
            diff_lengths = set(self._lengths) - {1}
            raise ValueError("Expandable parameters do not all have the same lengths."
                 "`expand` parameters with the following lengths were found:\n"
                 f"{diff_lengths}")
        elif inner_len == 1:
            # There are no parameters to exand  (this implies in particular that there are no random parameters)
            return [{k: v[0] if isinstance(v, expand)
                        else v for k, v in self.items()}]
        else:
            kw = {k: v if isinstance(v, expand)
                  else self._make_rv_iterator(v, rng, inner_len)
                    if isinstance(v, (RVFrozen, MultiRVFrozen))
                  else repeat(v) for k,v in self.items()}
            for vlst in zip(*kw.values()):
                yield {k: v for k, v in zip(kw.keys(), vlst)}
    
    def _get_kw_lst_outer(self):
        if self.seed is not NoArg:
            rng = self.get_rng()
        outer_len = self.outer_len
        if outer_len is None:
            raise ValueError("An 'outer' product of only infinite iterators "
                             "does not really make sense. Use 'inner' to "
                             "create an infinite parameter iterator.")
        kw = {k: v if isinstance(v, expand)
                 else [self._make_rv_iterator(v, rng, outer_len)]  # NB: We don’t want `product`
                    if isinstance(v, (RVFrozen, MultiRVFrozen))    #     to expand the RV iterator
                 else [v] for k,v in self.items()}
        for vlst in product(*kw.values()):
            yield {k: next(v) if isinstance(v, Generator) else v   # `Generator` is for the RV iterator
                   for k, v in zip(kw.keys(), vlst)}               # Ostensibly we could support other generators ?


# ### Test

# + tags=["active-ipynb"]
# import numpy as np
# import pytest
# from dataclasses import asdict
#
# @dataclass
# class DataParamset(ParamColl):
#     L: int
#     λ: float
#     σ: float
#     δy: float
#
# @dataclass
# class ModelParamset(ParamColl):
#     λ: float
#     σ: float
#     #μ: float

# + tags=["active-ipynb"]
# data_params = DataParamset(
#     L=400,
#     λ=1,
#     σ=1,
#     δy=expand([-1, -0.3, 0, 0.3, 1])
# )
# model_params = ModelParamset(
#     λ=expand(np.logspace(-1, 1, 10)),
#     σ=expand(np.linspace(0.1, 3, 8))
# )
# model_params_aligned = ModelParamset(
#     λ=expand(np.logspace(-1, 1, 10)),
#     σ=expand(np.linspace(0.1, 3, 10))
# )

# + tags=["active-ipynb"]
# # Iterating over ParamColl returns the keys
# assert len(list(data_params)) == len(data_params.keys()) == 4
# assert list(data_params) == ["L", "λ", "σ", "δy"]
#
# # Expanding a list
# assert list(data_params.inner()) == list(data_params.outer())
# assert len(list(data_params.outer())) == len(data_params.δy) == data_params.outer_len == 5
#
# # Expanding an array + Non-aligned doesn't allow inner() iterator
# assert len(list(model_params.outer())) == 10*8
# with pytest.raises(ValueError):
#     next(model_params.inner())
#
# # Expanding an array + Aligned expanded params allows inner() iterator
# assert len(list(model_params_aligned.inner())) == len(model_params_aligned.λ) == model_params_aligned.inner_len == 10
# assert len(list(model_params_aligned.outer())) == model_params_aligned.outer_len == 10*10
#
# assert dict(**data_params) == {k: v for k,v in asdict(data_params).items() if not k.startswith("_")}
#
# # Slicing inner() and outer() works as advertised
# assert len(list(model_params_aligned.inner(2, 8))) == 6
# assert len(list(model_params_aligned.inner(2, 8, 2))) == 3
# assert len(list(model_params_aligned.outer(5,20)))   == 15
# assert len(list(model_params_aligned.outer(5,20,5))) == 3
# -

# ## Plotting

def get_bounds(*arrs, lower_margin=0.05, upper_margin=0.05) -> Tuple[float, float]:
    """
    Return bounds that include all values of the given arrays, plus a little more.
    How much more is determined by the margins; 0.05 increases the range
    by about 5%.
    Intended for recomputing the bounds in figures.
    
    Returns a tuple of ``(low, high)`` bounds.
    """
    low = min(arr.min() for arr in arrs); high = max(arr.max() for arr in arrs)
    width = high-low
    return (low-lower_margin*width, high+upper_margin*width)


# +
from bokeh.models import Range1d, LinearAxis
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting.figure import Figure

def plot_secondary(plot, element):
    """
    Hook to plot data on a secondary (twin) axis on a Holoviews Plot with Bokeh backend.
    More info:
    - http://holoviews.org/user_guide/Customizing_Plots.html#plot-hooks
    - https://docs.bokeh.org/en/latest/docs/user_guide/plotting.html#twin-axes
    
    Source: https://github.com/holoviz/holoviews/issues/396#issuecomment-1231249233
    """
    fig: Figure = plot.state
    glyph_first: GlyphRenderer = fig.renderers[0]  # will be the original plot
    glyph_last: GlyphRenderer = fig.renderers[-1] # will be the new plot
    right_axis_name = "twiny"
    # Create both axes if right axis does not exist
    if right_axis_name not in fig.extra_y_ranges.keys():
        # Recreate primary axis (left)
        y_first_name = glyph_first.glyph.y
        # If the figure is an overlay, there can be multiple plots associated to the left
        # axis. Assumption: The first renderer determines the name of the left axis
        # (consistent with Holoviews). Any other renderer with the same name for its
        # y axis may contribute to the bounds calculation.
        y_first_min = min(renderer.data_source.data[y_first_name].min()
                          for renderer in fig.renderers
                          if renderer.glyph.y == y_first_name)
        y_first_max = max(renderer.data_source.data[y_first_name].max()
                          for renderer in fig.renderers
                          if renderer.glyph.y == y_first_name)
        y_first_offset = (y_first_max - y_first_min) * 0.1
        y_first_offset = (y_first_max - y_first_min) * 0.1
        fig.y_range = Range1d(
            start=y_first_min - y_first_offset,
            end=y_first_max + y_first_offset
        )
        fig.y_range.name = glyph_first.y_range_name
        # Create secondary axis (right)
        y_last_name = glyph_last.glyph.y
        y_last_min = glyph_last.data_source.data[y_last_name].min()
        y_last_max = glyph_last.data_source.data[y_last_name].max()
        y_last_offset = (y_last_max - y_last_min) * 0.1
        fig.extra_y_ranges = {right_axis_name: Range1d(
            start=y_last_min - y_last_offset,
            end=y_last_max + y_last_offset
        )}
        fig.add_layout(LinearAxis(y_range_name=right_axis_name, axis_label=glyph_last.glyph.y), "right")
    # Set right axis for the last glyph added to the figure
    glyph_last.y_range_name = right_axis_name

# -

# ## Pretty-print Git version
# (Ported from *mackelab_toolbox.utils*)

from typing import Union
from pathlib import Path
from datetime import datetime
from socket import gethostname
try:
    import git
except ModuleNotFoundError:
    git = None

class GitSHA:
    """
    Return an object that nicely prints the SHA hash of the current git commit.
    Displays as formatted HTML in a Jupyter Notebook, otherwise a simple string.

    .. Hint:: This is especially useful for including a git hash in a report
       produced with Jupyter Book. Adding a cell `GitSHA() at the bottom of
       notebook with the tag 'remove-input' will print the hash with no visible
       code, as though it was part of the report footer.

    Usage:
    >>> GitSHA()
    myproject main #3b09572a
    """
    css: str= "color: grey; text-align: right"
    # Default values used when a git repository can’t be loaded
    path  : str="No git repo found"
    branch: str=""
    sha   : str=""
    hostname: str=""
    timestamp: str=None
    def __init__(self, path: Union[None,str,Path]=None, nchars: int=8,
                 sha_prefix: str='#', show_path: str='stem',
                 show_branch: bool=True, show_hostname: bool=False,
                 datefmt: str="%Y-%m-%d"):
        """
        :param:path: Path to the git repository. Defaults to CWD.
        :param:nchars: Numbers of SHA hash characters to display. Default: 8.
        :param:sha_prefix: Character used to indicate the SHA hash. Default: '#'.
        :param:show_path: How much of the repository path to display.
            'full': Display the full path.
            'stem': (Default) Only display the directory name (which often
                    corresponds to the implied repository name)
            'none': Don't display the path at all.
        :param:datefmt: The format string to pass to ``datetime.strftime``.
            To not display any time at all, use an empty string.
            Default format is ``2000-12-31``.
        """
        ## Set attributes that should always work (don't depend on repo)
        if datefmt:
            self.timestamp = datetime.now().strftime(datefmt)
        else:
            self.timestamp = ""
        if show_hostname:
            self.hostname = gethostname()
        else:
            self.hostname = ""
        ## Set attributes that depend on repository
        # Try to load repository
        if git is None:
            # TODO?: Add to GitSHA a message saying that git python package is not installed ?
            return
        try:
            repo = git.Repo(search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            # Skip initialization of repo attributes and use defaults
            return
        self.repo = repo
        self.sha = sha_prefix+repo.head.commit.hexsha[:nchars]
        if show_path.lower() == 'full':
            self.path = repo.working_dir
        elif show_path.lower() == 'stem':
            self.path = Path(repo.working_dir).stem
        elif show_path.lower() == 'none':
            self.path = ""
        else:
            raise ValueError("Argument `show_path` should be one of "
                             "'full', 'stem', or 'none'")
        self.branch = ""
        if show_branch:
            try:
                self.branch = repo.active_branch.name
            except TypeError:  # Can happen if on a detached head
                pass

    def __str__(self):
        return " ".join((s for s in (self.timestamp, self.hostname, self.path, self.branch, self.sha)
                         if s))
    def __repr__(self):
        return self.__str__()
    def _repr_html_(self):
        hoststr = f"&nbsp;&nbsp;&nbsp;host: {self.hostname}" if self.hostname else ""
        return f"<p style=\"{self.css}\">{self.timestamp}{hoststr}&nbsp;&nbsp;&nbsp;git: {self.path} {self.branch} {self.sha}</p>"

# -

# ## Hashing
# (Ported from *mackelab_toolbox.utils*; used in SeedGenerator)

import hashlib
from collections.abc import Iterable, Sequence, Collection, Mapping
from enum import Enum

terminating_types = (str, bytes)

def stablehash(o):
    """
    The builtin `hash` is not stable across sessions for security reasons.
    This `stablehash` can be used when consistency of a hash is required, e.g.
    for on-disk caches.

    For obtaining a usable digest, see the convenience functions
    `stablehexdigest`, `stablebytesdigest` and `stableintdigest`.
    `stabledigest` is a synonym for `stableintdigest` and is suitable as the
    return value of a `__hash__` method.

    .. Note:: For exactly the reason stated above, none of the hash functions
       in this module are cryptographically secure.

    .. Tip:: The following function can be used to calculate the likelihood
       of a hash collision::

           def p_coll(N, M):
             '''
             :param N: Number of distinct hashes. For a 6 character hex digest,
                this would be 16**6.
             :param M: Number of hashes we expect to create.
             '''
             logp = np.sum(np.log(N-np.arange(M))) - M*np.log(N)
             return 1-np.exp(logp)

    Returns
    -------
    HASH object
    """
    return hashlib.sha1(_tobytes(o))
def stablehexdigest(o) -> str:
    """
    Returns
    -------
    str
    """
    return stablehash(o).hexdigest()
def stablebytesdigest(o) -> bytes:
    """
    Returns
    -------
    bytes
    """
    return stablehash(o).digest()
def stableintdigest(o, byte_len=4) -> int:
    """
    Suitable as the return value of a `__hash__` method.

    .. Note:: Although this method is provided, note that the purpose of a
       digest (a unique fingerprint) is not the same as the intended use of
       the `__hash__` magic method (fast hash tables, in particular for
       dictionaries). In the latter case, a certain degree of hash collisions
       is in fact desired, since that is required for the most efficient tables.
       Because this function uses SHA1 to obtain almost surely unique digests,
       it is much slower than typical `__hash__` implementations. This can
       become noticeable if it is involved in a lot of dictionary lookups.

    Parameters
    ----------
    o : object to hash (see `stablehash`)
    byte_len : int, Optional (default: 4)
        Number of bytes to keep from the hash. A value of `b` provides at most
        `8**b` bits of entropy. With `b=4`, this is 4096 bits and 10 digit
        integers.

    Returns
    -------
    int
    """
    return int.from_bytes(stablebytesdigest(o)[:byte_len], 'little')
stabledigest = stableintdigest


def _tobytes(o) -> bytes:
    """
    Utility function for converting an object to bytes. This is used for the
    state digests, and thus is designed with the following considerations:

    1. Different inputs should, with high probability, return different byte
       sequences.
    2. The same inputs should always return the same byte sequence, even when
       executed in a new session (in order to satisfy the 'stable' description).
       Note that this precludes using an object's `id`, which is sometimes
       how `hash` is implemented.
    3. It is NOT necessary for the input to be reconstructable from
       the returned bytes.

    ..Note:: To avoid overly complicated byte sequences, the distinction
       guarantee is not preserved across types. So `_tobytes(b"A")`,
       `_tobytes("A")` and `_tobytes(65)` all return `b'A'`.
       So multiple inputs can return the same byte sequence, as long as they
       are unlikely to be used in the same location to mean different things.

    **Supported types**
    - None
    - bytes
    - str
    - int
    - float
    - Enum
    - type
    - Any object implementing a ``__bytes__`` method
    - Mapping
    - Sequence
    - Collection
    - Any object for which `bytes(o)` does not raise an exception

    Raises
    ------
    TypeError:
        - If `o` is a consumable Iterable.
        - If `o` is of a type for which `_to_bytes` is not implemented.
    """
    # byte converters for specific types
    if o is None:
        # TODO: Would another value more appropriately represent None ? E.g. \x00 ?
        return b""
    elif isinstance(o, bytes):
        return o
    elif isinstance(o, str):
        return o.encode('utf8')
    elif isinstance(o, int):
        l = ((o + (o<0)).bit_length() + 8) // 8  # Based on https://stackoverflow.com/a/54141411
        return o.to_bytes(length=l, byteorder='little', signed=True)
    elif isinstance(o, float):
        return o.hex().encode('utf8')
    elif isinstance(o, Enum):
        return _tobytes(o.value)
    elif isinstance(o, type):
        return _tobytes(f"{o.__module__}.{o.__qualname__}")
    # Generic byte encoders. These methods may not be ideal for each type, or
    # even work at all, so we first check if the type provides a __bytes__ method.
    elif hasattr(o, '__bytes__'):
        return bytes(o)
    elif isinstance(o, Mapping) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(k) + _tobytes(v) for k,v in o.items())
    elif isinstance(o, Sequence) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(oi) for oi in o)
    elif isinstance(o, Collection) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(oi) for oi in sorted(o))
    elif isinstance(o, Iterable) and not isinstance(o, terminating_types):
        raise ValueError("Cannot compute a stable hash for a consumable Iterable.")
    else:
        try:
            return bytes(o)
        except TypeError:
            # As an ultimate fallback, attempt to use the same decomposition
            # that pickle would
            try:
                state = o.__getstate__()
            except Exception:
                breakpoint()
                raise TypeError("mackelab_toolbox.utils._tobytes does not know how "
                                f"to convert values of type {type(o)} to bytes. "
                                "One way to solve this is may be to add a "
                                "`__bytes__` method to that type. If that is "
                                "not possible, you may also add a converter to "
                                "mackelab_toolbox.utils.byte_converters.")
            else:
                return _tobytes(state)
