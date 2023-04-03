# -*- coding: utf-8 -*-
# %% [markdown]
# # Tasks
#
# Running our experiments via [Tasks]() has two purposes:
# - Maintaining an electronic lab book: recording all input/code/output triplets, along with a bunch of metadata to ensure reproducibility (execution date, code versions, etc.)
# - Avoid re-running calculations, with hashes that are both portable and long-term stable.

# %%
from config import config

# %%
import psutil
import logging
import multiprocessing as mp
import numpy as np
from functools import partial
from itertools import repeat
from typing import Union, Dict, Tuple, List, Iterable
from dataclasses import dataclass, is_dataclass, replace
from scityping import Serializable, Dataclass, Type
from tqdm.auto import tqdm
from scityping.functions import PureFunction
# Make sure Array (numpy) and RV (scipy) serializers are loaded
import scityping.numpy
import scityping.scipy

from smttask import RecordedTask, TaskOutput
from smttask.workflows import ParamColl, SeedGenerator

from emdd import Bemd
from emdd.models import Model, FullModel


# %%
class SeedGen(SeedGenerator):
    MX   : int
    Mtrue: int
    MA   : int
    MB   : int


# %% [markdown]
# Since our models are just dataclasses with additional methods, they can also double as parameter collections (we just ignore the methods).

# %%
# Types needed ParamColl
from smttask.workflows import Literal, NOSEED, Seed, Union, Dimension, Dict, List

@dataclass(frozen=True)
class ModelColl(ParamColl, FullModel):
    pass


# %% [markdown]
# ## Calibration task
#
# **Current limitations**
# - Parameter collections are always combined with `inner` (hard-coded)
# - Physical model is assumed deterministic
# - `ncores` depends on `config.mp.max_cores`, but is determined automatically.
#   No way to control via parameter.

# %% [markdown]
# ### Utilities

# %% [markdown]
# #### *logp* wrapper

# %% [markdown]
# The function `Bemd` expects *logp* functions which takes a single combined parameter `xy`. Therefore we define a new generic function which unpacks the arguments and passes them to the model's *logpdf*.
# Within one $B^{\mathrm{EMD}}$ evaluation we need two *logp* functions, $\log p_A$ and $\log p_B$. Therefore the function below takes an additional parameter, `logp_obs`, which is bound later using `partial`.

# %% [markdown]
# ::::{sidebar}
# :::{hint}
# For stochastic physical models, the idea would be to take *two* additional parameters: `logp_phys` and `logp_obs`.
# :::
# ::::

# %% [markdown]
# :::{admonition} Important consideration for caching
# :class: important, dropdown
#
# On-disk caching will compute a hash based on the pickled value of the `logp_theo` function. Since standard functions are pickled with a reference to their module, this makes the hash non-portable: only the original notebook which created the data may reload it.
#
# In contrast, a `@PureFunction` is pickled by serializing its code (after removing comments and whitespace). Therefore as long as the function definition is the same, we can copy code to another notebook/script and the cache will still be found.
# :::

# %%
@PureFunction
def logp_theo(xy, logp_obs):
    x, y = xy
    return logp_obs(y, x)

# TODO: Version with support for stochastic physical model
# @PureFunction
# def logp_theo(xzy, logp_obs, logp_phys=None):
#     x, z, y = xzy
#     return logp_obs(y, x) + logp_phys(z, x)


# %% [markdown]
# #### Functions for the calibration experiment

# %% [markdown]
# Define the two functions to compute $B^{\mathrm{EMD}}$ and $B_{\mathrm{conf}}$; these will be the abscissa and ordinate in the calibration plot.
# Both functions take an argument `Θtup`, which is a tuple of all the parameter sets, for the *true*, *A* and *B* models for **one** *experiment*.
#
# $B^{\mathrm{EMD}}$ needs to be recomputed for each value of $c$, so we also pass $c$ as a parameter to allow dispatching to different MP processes.  
# This means we need to recreate the model from `Θtup` within each process, rather than create it once and then loop over the $c$ values. For the models used here creation and sampling takes negligible compute time, so this is not a problem.
#
# $B_{\mathrm{conf}}$ only needs to be computed once per parameter tuple. This is also very cheap, so is not worth dispatching to an MP subprocess.  
# This also means that within the function we again have to recreate the models from `Θtup`. As stated above, this is not a problem for the models used here.

# %%
def update_random_state(model, new_rngstate) -> "Model":
    """
    Supports a few different conventions for updating the RNG state.
    
    - Frozen dataclasses
      - Updates the first field which exists: "random_state", "seed".
      - If a random state attribute is found, returns a copy.
        Otherwise, returns `model` unchanged`
    - Non-frozen dataclasses
      - Updates the first field which exists: "random_state", "seed".
      - Updates in-place
    - Scipy distributions
      - Updates the "random_state" attribute.
      - Updates in-place
    - Any other type
      - Attempts to update the first field which exists, if present: "random_state", "seed".
      - Updates in-place
    """
    rngstate_attr = ("random_state" if hasattr(model, "random_state")
                     else "seed" if hasattr(model, "seed")
                     else None)
    if rngstate_attr is None:
        # Non-random model => return unchanged
        pass
    elif is_dataclass(model) and model.__dataclass_params__.frozen:
        model = replace(model, **{rngstate_attr: new_rngstate})
    else:
        setattr(model, rngstate_attr, new_rngstate)
    return model            


# %%
def compute_Bemd(models_c, Ldata, Ltheo, seeds):
    """
    Wrapper for `emdd.Bemd`:
    - Instantiates models using parameters in `Θtup_c`.
    - Constructs log-probability functions for `MtheoA` and `MtheoB`.
    - Generates synthetic observed data using `Mtrue`.
    - Calls `emdd.Bemd`

    `Mptrue`, `Metrue`, `Mptheo` and `Metheo` should be generic models:
    they are functions accepting parameters, and returning frozen models
    with fixed parameters.
    """
    (MX, Mtrue, MA, MB), c = models_c
    MX = update_random_state(MX, seeds.MX(MX))
    Mtrue_phys = update_random_state(Mtrue.phys, seeds.Mtrue("phys", Mtrue))
    Mtrue_obs  = update_random_state(Mtrue.obs , seeds.Mtrue("obs" , Mtrue))
    MA_phys    = update_random_state(MA.phys, seeds.MA("phys", MA))
    MA_obs     = update_random_state(MA.obs , seeds.MA("obs" , MA))
    MB_phys    = update_random_state(MB.phys, seeds.MB("phys", MB))
    MB_obs     = update_random_state(MB.obs , seeds.MB("obs" , MB))
    
    assert not any((hasattr(Mtrue.phys, "random_state"), hasattr(MA.phys, "random_state"), hasattr(MB.phys, "random_state"))), \
        "In order to support stochastic physical models, we need to update `logp_theo` (and do some testing)."
        
    # (Θtrue, ΘA, ΘB), c = Θtup_c
    # Mptrue = Mptrue(**Θptrue, seed=seeds.Mptrue(Θptrue.phys))
    # Metrue = Metrue(**Θetrue, seed=seeds.Metrue(Θetrue.obs))

    # MpA = Mptheo(**ΘA.phys, seed=seeds.MA(θA.phys))
    # MeA = Metheo(**ΘA.obs , seed=seeds.MA(θA.obs))
    # MpB = Mptheo(**ΘB.phys, seed=seeds.MB(ΘB.phys))
    # MeB = Metheo(**ΘB.obs , seed=seeds.MB(ΘB.obs))

    # Likelihood functions for models A and B.
    # If we want priors, they should be included here
    logpA = partial(logp_theo, logp_obs=MA_obs.logpdf)
    logpB = partial(logp_theo, logp_obs=MB_obs.logpdf)

    # Generate observed data
    xdata = MX(Ldata)
    zdata = Mtrue_phys(xdata)
    ydata = Mtrue_obs(xdata, zdata)
    data = np.vstack((ydata, ydata))

    # Generate samples to estimate model quantile functions
    x = MX(Ltheo)
    zA = MA_phys(x)    ; zB = MB_phys(x)
    yA = MA_obs(x, zA) ; yB = MB_obs(x, zB)
    samplesA = np.vstack((yA, x))
    samplesB = np.vstack((yB, x))

    # Silence sampling warnings: Calibration involves evaluating Bemd for models far from the data distribution, which require more
    # than 1000 path samples to evaluate the path integral within the default margin.
    # The further paths are from the most likely one, the more likely they are to trigger numerical warnings.
    # This is expected, so we turn off warnings to avoid spamming the console.
    logging.getLogger("emdd.emd").setLevel(logging.ERROR)
    
    # Compute the Bemd
    return Bemd(data, logpA, logpB, samplesA, samplesB, c=c,
                progbarA=None, progbarB=None, use_multiprocessing=False)


# %%
def compute_Bconf(MX, Mtrue, MA, MB, Linf, seeds):
    """Compute the true Bconf (using a quasi infinite number of samples)"""

    MX = update_random_state(MX, seeds.MX(MX))
    Mtrue_phys = update_random_state(Mtrue.phys, seeds.Mtrue("phys", Mtrue))
    Mtrue_obs  = update_random_state(Mtrue.obs , seeds.Mtrue("obs" , Mtrue))
    # Θtrue, ΘA, ΘB = Θtup
    # Mptrue = Mptrue(**Θptrue, seed=seeds.Mptrue(Θptrue.phys))
    # Metrue = Metrue(**Θetrue, seed=seeds.Metrue(Θetrue.obs))

    # Likelihood functions for models A and B.
    # If we want priors, they should be included within the model's `logpdf`
    logpA = partial(logp_theo, logp_obs=MA.obs.logpdf)
    logpB = partial(logp_theo, logp_obs=MB.obs.logpdf)
    
    # Generate samples
    x = MX(Linf)
    y = Mtrue_obs(x, Mtrue_phys(x))
    
    # Compute Bconf
    return logpA((x, y)).mean() > logpB((x, y)).mean()


# %% [markdown]
# ### Calibration task

# %% [markdown]
# Types: Since we use dataclasses to define all our models, we can use `scityping`'s built-in support for simple dataclasses to avoid having to define any additional serializable types.
# For extra self-documentation, we could define base clases for `Model` from which all our models would derive, and use those as type annotations.

# %% [markdown]
# ## Task parameters
#
# | Parameter | Value | Description |
# |-----------|---------|-------------|
# | `N` | {glue:text}`N` | Number of experiments per $c$ per parameter set distribution. |
# | `Linf` | {glue:text}`Linf` | Data set size considered equivalent to "infinite". |
# | `Ltheo` | {glue:text}`Ltheo` | Number of points to sample from the theoretical models to estimate their quantile function. |
# | `c_list` | [{glue:text}`c_list`]  | The values of $c$ we want to test. |
# | `ncores` | # physical cores | Number of CPU cores to use. |
#
# ### Effects on compute time
#
# The total number of experiments will be
# $$N \times \lvert\mathtt{c\_list}\rvert \times \text{(\# parameter set distributions)} \,.$$
# In the best scenario, one can expect compute times to be 2.5 minutes / experiment. So expect this to take a few hours.
#
# Results are cached on-disk with [joblib.Memory](https://joblib.readthedocs.io/en/latest/memory.html), so this notebook can be reexecuted without re-running the experiments. Loading from disk takes about 1 minute for 6000 experiments.
#
# ### Effects on caching
#
# Like any RecordedTask, `Calibrate` will record its output to disk. If executed again with exactly the same parameters, instead of evaluating the task again, the result is simply loaded from disk.
#
# In addition, `Calibrate` (or rather `Bemd`, which it calls internally) also uses a faster `joblib.Memory` cache to store intermediate results for each value of $c$ in `c_list`. Because `joblib.Memory` computes its hashes by first pickling its inputs, this cache is neither portable nor suitable for long-term storage: the output of `pickle.dump` may change depending on the machine, OS version, Python version, etc. Therefore this cache should be consider *local* and *short-term*. Nevertheless it is quite useful, because it means that `c_list` can be modified and only the new $c$ values will be computed.
#
# Changing any argument other than `c_list` will invalidate all caches and force all recomputations.

# %%
#ModelColl = ParamColl
CalibrateKey = Tuple[Model,ModelColl,ModelColl,ModelColl]  # MX, MTrue, MA, MB – Note that the latter three are hierarchical, hence the ModelColl type
EntropyVal = Union[int,str]  # In general, any type accepted by smttask.workflows._normalize_entropy should work
# ModelColl = ParamColl[Model]  # One day perhaps ParamColl will support this

class CalibrateOutput(TaskOutput):
    # NB: Conceptually these are dicts, but JSONEncoder doesn’t support tuples as dict keys
    Bemd : List[Tuple[Tuple[CalibrateKey, float], float]]
    Bconf: List[Tuple[CalibrateKey, float]]


# %%
@RecordedTask
def Calibrate(
    MX         : Model,
    Mtrue_coll : ModelColl,
    MtheoA_coll: ModelColl,
    MtheoB_coll: ModelColl,
    # Θcoll_true : ParamColl,
    # Θcoll_theoA: ParamColl,
    # Θcoll_theoB: ParamColl,
    c_list     : List[float],
    N          : int,
    Ldata      : int,
    Ltheo      : int,
    Linf       : int,
    entropy    : Union[EntropyVal,Tuple[EntropyVal,...]]
    ) -> CalibrateOutput:
    """
    Parameters
    ----------
    N: Number of different parameter sets to try when computing Bemd.
    Ldata: Number of data points from the true model to generate when computing Bemd.
        This should be chosen commensurate with the data will analyze, in order
        to accurately mimic data variability.
    Ltheo: Number of data points from the theoretical model to generate when computing `Bemd`.
        This does not need to be the same as `Ldata`, and should be large enough
        to make numerical errors negligible.
    Linf: Number of data points from the true model to generate when computing `Bconf`.
        This is to emulate an infinitely large data set, and so should be large
        enough that numerical variability is completely suppressed. The computational
        cost of `Bconf` is miniscule compare do `Bemd`, so taking very large
        values of `Linf` – on the order of 10^6 or more – is recommended.
    """
    pass


# %% [markdown]
# Convert `entropy` to the object that will generate all of our seeds.

    # %%
    seeds = SeedGen(entropy)

# %% [markdown]
# Bind arguments to the `Bemd` function, so it only take one argument (`models`) – this is required by `imap`.

    # %%
    compute_Bemd_partial = partial(compute_Bemd,
           Ldata=Ldata, Ltheo=Ltheo, seeds=SeedGen((entropy, "Bemd")))

# %% [markdown]
# Define dictionaries into which we will accumulate the results of the $B^{\mathrm{EMD}}$ and $B_{\mathrm{conf}}$ calculations.

    # %%
    Bemd_results = {}
    Bconf_results = {}

# %% [markdown]
# - Set the iterator over parameter combinations (we need two identical ones)
# - Set up progress bar.
# - Determine the number of multiprocessing cores we will use.

    # %%
    modelgen1 = zip(repeat(MX), Mtrue_coll.inner(N), MtheoA_coll.inner(N), MtheoB_coll.inner(N))
    modelgen2 = zip(repeat(MX), Mtrue_coll.inner(N), MtheoA_coll.inner(N), MtheoB_coll.inner(N))
    total = N*len(c_list)
    progbar = tqdm(desc="Calib. experiments", total=total)
    ncores = psutil.cpu_count(logical=False)
    ncores = min(ncores, total, config.mp.max_cores)

# %% [markdown]
# Run the experiments. Since there are a lot of them, and they each take a few minutes, we use multiprocessing to do this in reasonable time.  

    # %%
    with mp.Pool(ncores) as pool:
        # Chunk size calculated following Pool's algorithm (See https://stackoverflow.com/questions/53751050/multiprocessing-understanding-logic-behind-chunksize/54813527#54813527)
        # (Naive approach would be total/ncores. This is most efficient if all taskels take the same time. Smaller chunks == more flexible job allocation, but more overhead)
        chunksize, extra = divmod(N, ncores*6)
        def models_c_gen(modelgen: Iterable, c_list):
            for models in modelgen:
                for c in c_list:
                    if (models, c) not in Bemd_results:  # Skip results which are
                        yield (models, c)                # already loaded
                    else:
                        assert models in Bconf_results
        if extra:
            chunksize += 1
        Bemd_it = pool.imap(compute_Bemd_partial, models_c_gen(modelgen1, c_list),
                            chunksize=chunksize)
        for (models, c), Bemd_res in zip(models_c_gen(modelgen2, c_list), Bemd_it):  # NB: This will always yield the same
            progbar.update(1)        # Updating first more reliable w/ ssh           #  tuples as modelgen1, because we only
            Bemd_results[models, c] = Bemd_res                                       #  update the dict after drawing from modelgen2
            Bconf_results[models] = compute_Bconf(*models, Linf, seeds=SeedGen((entropy, "Bconf")))

    progbar.close()

# %% [markdown]
# Return the results
# Serialization does not support tuples for the dict keys (only str, int, float, bytes), so we convert the dicts to lists

    # %%
    return dict(Bemd =list(Bemd_results.items()),
                Bconf=list(Bconf_results.items()))
