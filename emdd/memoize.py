# # Caching decorator

# This module defines the `@cache` decorator according to configuration. There are two options:
# - On-disk caching uses *joblib.Memory*.
# - In-memory caching (the default) uses *functools.lru_cache*.
# On-disk caching is opt-in for two reasons:
# - It requires installing an additional packages (*joblib*).
# - An on-disk is prone to becoming stale, for example if code changes. Since it is likely impossible to completely prevent loading from a stale cache, it is important that the user be aware when one is used.
#   + *joblib* uses a basic check and will flush its cache whenever the code of the function wrapped with `@cache` changes.
#     Any change outside the cached function may lead to loading a stale cahe.
#   + *[SumatraTask](sumatratask.readthedocs.io/)* uses a stricter approach, which is less likely to produce accidental loads but is also less user-friendly.
#
# Both on-disk and in-memory caching are further wrapped with a `no_fail` decorator, which falls back to calling the function without memoization if its arguments are not hashable (for *lru_cache*) or pickleable (*joblib.Memory*).
# This helps keep memoization transparent: it should be used when possible, but not prevent the use of otherwise valid functions.

import logging
from collections.abc import Callable
from functools import partial, update_wrapper, WRAPPER_ASSIGNMENTS
from inspect import unwrap
from pickle import PicklingError
try:
    import joblib
except ModuleNotFoundError:
    joblib = None

from .config import Config
config = Config()
logger = logging.getLogger(__name__)

__all__ = ["cache"]

# ## `@memoize` decorator

# +
if config.caching.use_disk_cache:
    from joblib import Memory
    memory = Memory(**config.caching.joblib.dict(exclude={"rootdir"}))
    def memoize(func=None, **kwargs):
        "Combine @nofail_joblib_cache and @memory.cache into one decorator"
        if func is None:
            # Support adding arguments on the decorator line: @cache(warn=False)
            return partial(memoize, **kwargs)
        
        warn = kwargs.pop("warn", None)
        nofail_kws = {} if warn is None else {"warn": warn}
        return nofail_joblib_cache(**nofail_kws)(memory.cache(func, **kwargs))
        
else:
    from functools import lru_cache
    def memoize(func=None, **kwargs):
        "Combine @nofail_functools_cache and @lru_cache into one decorator"
        warn = kwargs.pop("warn", None)
        nofail_kws = {} if warn is None else {"warn": warn}
        if func is None:
            # Gobble up kwargs – they are meant for joblib
            assert "@cache only accepts keyword arguments, or no arguments at all"
            return partial(memoize, **nofail_kws)
        else:
            assert (len(kwargs) == 0 and isinstance(func, Callable)), \
                    "@cache only accepts keyword arguments, or no arguments at all"
            return nofail_functools_cache(**nofail_kws)(lru_cache(func))


# -

# ## `@nofail` decorators

def nofail_functools_cache(warn: bool=True):
    """
    Add a fallback to a memoized function, so that if the arguments
    are unhashable, the unmemoized form is called.
    This assumes that the original, unmemoized function is stored
    as ``cached_f.__wrapped__`` – as is the case if it is created
    with `functools.lru_cache`.

    Typical use:

    >>> from functools import cache
    >>> from .utils import nofail_functools_cache
    >>>
    >>> @nofail_functools_cache
    >>> @cache
    >>> def my_slow_function(x):
          return x*2
    >>>
    >>> my_slow_function(2)
    >>> my_slow_function(2)     # Uses cache
    >>> my_slow_function([2])   # Does not fail
    >>> my_slow_function([2])   # Does not use cache
    """
    def decorator(cached_f: "cached_function"):
        def wrapper(*args, **kwds):
            nonlocal warn

            try:
                return cached_f(*args, **kwds)
            except TypeError as e:
                # First ensure that something else didn't cause the TypeError;
                # in such a case, we don't want to catch the error.
                # (More robust would be to check first that all args and kwds
                #  are instances of collections.abc.hashable, but that would
                #  add a lot more compute cost to each call)
                if not e.args[0].startswith("unhashable type"):
                    raise e
                # Now try to call the bare, unmemoized function
                try:
                    f = unwrap(cached_f)
                except AttributeError:
                    # Either `cached_f` is not memoized, or it uses a different pattern than lru_cache
                    raise e
                else:
                    if warn:
                        logger.warning(f"Calling unmemoized form of {cached_f.__name__} "
                                       "because some arguments are unhashable. "
                                       "To avoid log spam, this message will not be repeated.")
                        warn = False
                    return f(*args, **kwds)
        update_wrapper(wrapper, cached_f,
                       WRAPPER_ASSIGNMENTS + ("cache_info", "cache_clear"))
        return wrapper

    # Allow decorator to be used with and without arguments
    if isinstance(warn, Callable):
        return decorator(warn)
    else:
        return decorator

if joblib:
    class MemorizedFuncNoFail(joblib.memory.MemorizedFunc):
        def __init__(self, warn, *args, **kwargs):
            self.warn = warn
            super().__init__(*args, **kwargs)
        def __call__(self, *args, **kwargs):
            try:
                return super().__call__(*args, **kwargs)
            except PicklingError as e:
                # Uncached branch
                if self.warn:
                    logger.warning(f"Calling unmemoized form of {self.__qualname__} "
                                   "because some arguments cannot be pickled:\n"
                                   f"{e.args[0]}\n"  # args[1] can be very long and drown the log message
                                   "To avoid log spam, this message will not be repeated.")
                    self.warn = False
                return self.func(*args, **kwargs)
        def check_call_in_cache(self, *args, **kwargs):
            try:
                return super().check_call_in_cache(*args, **kwargs)
            except PicklingError:
                return False


def nofail_joblib_cache(warn: bool=True):
    """
    Add a fallback to a function memoized with `joblib.Memory`, so that if the
    arguments are unhashable, the unmemoized form is called.
    This has no effect on functions not wrapped with `joblib.Memory.cache`.

    Caution
    -------
    Caching failures are detecting by catching `pickle.PicklingError`.
    Consequenly, if the wrapped function itself raises `pickle.PicklingError`,
    it will unnecessarily be run twice (the second time allowing the error
    to propagate up the call stack).

    Example
    -------

    >>> from joblib import Memory
    >>> from .utils import nofail_joblib_cache
    >>>
    >>> memory = Memory(cachedir, verbose=0)
    >>>
    >>> @nofail_joblib_cache
    >>> @memory.cache
    >>> def apply(f, x):
          return f(x)

    >>> def g(x):
          return x**2
    >>> def generator(p):
          def h(x):
            return x**p
          return h
    >>> h = generator(2)
    >>>
    >>> apply(g, 2)
    >>> apply(g, 2)   # Uses cache
    >>> apply(h, 2)   # Does not fail
    >>> apply(h, 2)   # Does not use cache
    >>>
    >>> apply.check_call_in_cache(g, 2)  # Returns True
    >>> apply.check_call_in_cache(f, 2)  # Returns False
    """
    def decorator(cached_f: "cached_function"):
        if joblib is None:
            # joblib unavailable: cannot have a joblib-cached function
            return cached_f
        elif isinstance(cached_f, joblib.memory.MemorizedFunc):
            # Decorator was applied on top of memory.cache: make it non-failing
            return MemorizedFuncNoFail(
                warn     =warn,
                func     =cached_f.func,
                location =cached_f.store_backend,
                backend  =None,
                ignore   =cached_f.ignore,
                mmap_mode=cached_f.mmap_mode,
                compress =cached_f.compress,
                verbose  =cached_f._verbose)
        else:
            # Not a function managed by joblib: do nothing
            return cached_f

    # Allow decorator to be used with and without arguments
    if isinstance(warn, Callable):
        return decorator(warn)
    else:
        return decorator
