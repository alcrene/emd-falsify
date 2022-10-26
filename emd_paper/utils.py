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

from emd_paper.find_sane_dt import make_int_superscript
from typing import ClassVar, Union, Tuple, NamedTuple
from scityping.numpy import NPValue, Array

# ## glue

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

# ### Testing

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
    Make SeedSequence callable, allowing to create different high-quality seeds
    simply by passing different integers.
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

# ### Testing

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

# ## Hashing
# (Ported from *mackelab_toolbox.utils*)

import hashlib
from collections.abc import Iterable, Sequence, Collection, Mapping
from enum import Enum


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
