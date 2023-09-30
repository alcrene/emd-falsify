# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python (emd-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ---
# math:
#   '\Bconf' : 'B^{\mathrm{conf}}_{#1}'
#   '\Bemd'  : 'B_{#1}^{\mathrm{EMD}}'
# ---

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
from __future__ import annotations

# %% [markdown]
# # Tasks
#
# Running our experiments via [Tasks]() has two purposes:
# - Maintaining an electronic lab book: recording all input/code/output triplets, along with a bunch of metadata to ensure reproducibility (execution date, code versions, etc.)
# - Avoid re-running calculations, with hashes that are both portable and long-term stable.

# %% tags=["active-ipynb"] editable=true slideshow={"slide_type": ""}
# from config import config   # Notebook

# %% tags=["active-py"] editable=true slideshow={"slide_type": ""} raw_mimetype=""
from .config import config  # Python script

# %% editable=true slideshow={"slide_type": ""}
import psutil
import logging
import time
import multiprocessing as mp
import numpy as np
from functools import partial
from itertools import repeat
from typing import Optional, Union, Any, Callable, Dict, Tuple, List, Iterable, NamedTuple
from dataclasses import dataclass, is_dataclass, replace
from scityping import Serializable, Dataclass, Type
from tqdm.auto import tqdm
from scityping.functions import PureFunction
# Make sure Array (numpy) and RV (scipy) serializers are loaded
import scityping.numpy
import scityping.scipy

from smttask import RecordedTask, TaskOutput
from smttask.workflows import ParamColl, SeedGenerator

import emd_falsify as emd

# %%
logger = logging.getLogger(__name__)

# %%
__all__ = ["Calibrate", "CalibrateKey", "CalibrateOutput"]


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
# Define the two functions to compute $\Bemd{}$ and $\Bconf{}$; these will be the abscissa and ordinate in the calibration plot.
# Both functions take an argument `Θtup`, which is a tuple of all the parameter sets, for the *true*, *A* and *B* models for **one** *experiment*.
#
# $\Bemd{}$ needs to be recomputed for each value of $c$, so we also pass $c$ as a parameter to allow dispatching to different MP processes.  
# This means we need to recreate the model from `Θtup` within each process, rather than create it once and then loop over the $c$ values.
#
# $\Bconf{}$ only needs to be computed once per parameter tuple. $\Bconf{}$ is also very cheap, so it is not worth dispatching to an MP subprocess.  
# This also means that within the function we again have to recreate the models from `Θtup`.

# %% editable=true slideshow={"slide_type": ""}
def compute_Bemd(datamodel_c, riskA, riskB, synth_ppfA, synth_ppfB,
                 Ldata):
    """
    Wrapper for `emd_falsify.Bemd`:
    - Instantiates models using parameters in `Θtup_c`.
    - Constructs log-probability functions for `MtheoA` and `MtheoB`.
    - Generates synthetic observed data using `Mtrue`.
    - Calls `emd_falsify.Bemd`

    `Mptrue`, `Metrue`, `Mptheo` and `Metheo` should be generic models:
    they are functions accepting parameters, and returning frozen models
    with fixed parameters.
    """
    ## Unpack arg 1 ##  (pool.imap requires iterating over one argument only)
    data_model, c = datamodel_c

    ## Generate observed data ##
    logger.debug(f"Compute Bemd - Generating {Ldata} data points."); t1 = time.perf_counter()
    data = data_model(Ldata)                                      ; t2 = time.perf_counter()
    logger.debug(f"Compute Bemd - Done generating {Ldata} data points. Took {t2-t1:.2f} s")

    ## Construct mixed quantile functions ##
    mixed_ppfA = emd.make_empirical_risk_ppf(riskA(data))
    mixed_ppfB = emd.make_empirical_risk_ppf(riskB(data))

    ## Draw sets of expected risk values (R) for each model ##
                     
    # Silence sampling warnings: Calibration involves evaluating Bemd for models far from the data distribution, which require more
    # than 1000 path samples to evaluate the path integral within the default margin.
    # The further paths are from the most likely one, the more likely they are to trigger numerical warnings.
    # This is expected, so we turn off warnings to avoid spamming the console.

    logger.debug("Compute Bemd - Generating R samples"); t1 = time.perf_counter()
    
    emdlogger = logging.getLogger("emd_falsify.emd")
    emdlogginglevel = emdlogger.level
    emdlogger.setLevel(logging.ERROR)

    RA_lst = emd.draw_R_samples(mixed_ppfA, synth_ppfA, c=c)
    RB_lst = emd.draw_R_samples(mixed_ppfB, synth_ppfB, c=c)

    # Reset logging level as it was before
    emdlogger.setLevel(emdlogginglevel)

    t2 = time.perf_counter()
    logger.debug(f"Compute Bemd - Done generating R samples. Took {t1-t2:.2f} s")
                     
    ## Compute the EMD criterion ##
    return np.less.outer(RA_lst, RB_lst).mean()

# %% editable=true slideshow={"slide_type": ""}
def compute_Bconf(data_model, riskA, riskB, Linf):
    """Compute the true Bconf (using a quasi infinite number of samples)"""
    
    # Generate samples
    data = data_model(Linf)
    
    # Compute Bconf
    return riskA(data).mean() > riskB(data).mean()


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Input and output types
#
# To be able to retrieve pasts results, tasks rely on their inputs being serializable. To leave users free to define their own model types, we define models very generically as any `Callable`, but this leaves the onus on the user to make sure that they are hashable.
#
# (We actually don’t need each individual model to be serializable: only the `data_models` collection needs to be serializable.)
#
# :::{hint}
# If models are defined as callable [dataclasses](https://docs.python.org/3/library/dataclasses.html) (and their fields don’t ues complicated data types), they are immediately serializable thanks to `scityping`'s built-in support for simple dataclasses. This avoids having to define custom serializable types.
# :::

# %% editable=true slideshow={"slide_type": ""}
class CalibrateOutput(TaskOutput):
    Bemd : List[float]
    Bconf: List[float]

DataModel    = Callable[[int], Any]  # Type also needs to be hashable
BemdResult   = Dict[Tuple[DataModel, float], float]
BconfResult  = Dict[DataModel, float]
class UnpackedCalibrateResult(NamedTuple):
    Bemd : BemdResult
    Bconf: BconfResult


# %% [markdown]
# ### Task definition

# %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
@RecordedTask
class Calibrate:
    pass


# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Model generators
#
# These yield the randomly generated models in a determininistic sequence. In addition to being used within the task itself, these methods can also be used to recreate the sequence of models, avoiding the need to save them with the output.
#
# The additional arguments to `models_c_gen` could eventually be used for a incremental task, which would allow increasing the number `N` of sampled experiments $ω$ and only computing $\Bemd{}$ and $\Bconf{}$ for the new ones. At present this is not supported, so it is always safe to pass empty dictionaries:  
# `task.models_c_gen(Bemd_results={}, Bconf_results={})`.

    # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
    def model_c_gen(self, Bemd_results: "dict|set", Bconf_results: "dict|set"):
        """Return an iterator over data models and c values.
        The two additional arguments `Bemd_results` and `Bconf_results` should be sets of
        *already computed* results. At the moment this is mostly a leftover from a previous
        implementation, before this function was made a *Task* — in the current 
        implementation, the task always behaves as though empty dictionaries were passed.
        """
        for data_model in self.taskinputs.data_models:    # Using taskinputs allows us to call `model_c_gen`
            for c in self.taskinputs.c_list:             # after running the task to recreate the keys.
                if (data_model, c) not in Bemd_results:
                    yield (data_model, c)
                else:                                    # Skip results which are already loaded
                    assert data_model in Bconf_results

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Task parameters
#
# | Parameter | Value | Description |
# |-----------|---------|-------------|
# | `c_list` | [{glue:text}`c_list`]  | The values of $c$ we want to test. |
# | `data_models` | | |
# | `riskA` | | |
# | `riskA` | | |
# | `synth_risk_ppfA` | | |
# | `synth_risk_ppfB` | | |
# | `Ldata` | | |
# | `Linf` | {glue:text}`Linf` | Data set size considered equivalent to "infinite". |
#
# Config values:
# | Parameter | Value | Description |
# | `ncores` | # physical cores | Number of CPU cores to use. |
#
# ##### Effects on compute time
#
# The total number of experiments will be
# $$N \times \lvert\mathtt{c\_list}\rvert \times \text{(\# parameter set distributions)} \,.$$
# In the best scenario, one can expect compute times to be 2.5 minutes / experiment. So expect this to take a few hours.
#
# Results are cached on-disk with [joblib.Memory](https://joblib.readthedocs.io/en/latest/memory.html), so this notebook can be reexecuted without re-running the experiments. Loading from disk takes about 1 minute for 6000 experiments.
#
# ##### Effects on caching
#
# Like any [RecordedTask](https://sumatratask.readthedocs.io/en/latest/basics.html), `Calibrate` will record its output to disk. If executed again with exactly the same parameters, instead of evaluating the task again, the result is simply loaded from disk.
#
# In addition, `Calibrate` (or rather `Bemd`, which it calls internally) also uses a faster `joblib.Memory` cache to store intermediate results for each value of $c$ in `c_list`. Because `joblib.Memory` computes its hashes by first pickling its inputs, this cache is neither portable nor suitable for long-term storage: the output of `pickle.dump` may change depending on the machine, OS version, Python version, etc. Therefore this cache should be consider *local* and *short-term*. Nevertheless it is quite useful, because it means that `c_list` can be modified and only the new $c$ values will be computed.
#
# Changing any argument other than `c_list` will invalidate all caches and force all recomputations.

    # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
    def __call__(
        self,
        c_list     : List[float],
        #data_models: Sequence[PureFunction],
        data_models: Any,
        riskA      : PureFunction,
        riskB      : PureFunction,
        synth_risk_ppfA  : Union[emd.interp1d, PureFunction],
        synth_risk_ppfB  : Union[emd.interp1d, PureFunction],
        Ldata      : int,
        Linf       : int,
        ) -> CalibrateOutput:
        """
        Run a calibration experiment using the models listed in `data_models`.
        Data models must be functions taking a single argument – an integer – and
        returning a dataset with that many samples. They should be “ready to use”;
        in particular, their random number generator should already be properly seeded
        to avoid correlations between different models in the list.
        
        Parameters
        ----------
        data_models: Iterable of data models to use for calibration. Each model yields
            one (Bconf, Bemd) pair. If this iterable is sized, progress bars will
            estimate the remaining compute time. 
            
        Ldata: Number of data points from the true model to generate when computing Bemd.
            This should be chosen commensurate with the size of the dataset that will be analyzed,
            in order to accurately mimic data variability.
        Linf: Number of data points from the true model to generate when computing `Bconf`.
            This is to emulate an infinitely large data set, and so should be large
            enough that numerical variability is completely suppressed.
            Choosing a too small value for `Linf` will add noise to the Bconf estimate,
            which would need to compensated by more calibration experiments.
            Since generating more samples is generally cheaper than performing more
            experiments, it is also generally preferable to choose rather large `Linf`
            values.

        .. Important:: An appropriate value of `Linf` will depend on the models and
           how difficult they are to differentiate; it needs to be determined empirically.
        """
        pass

# %% [markdown]
# Bind arguments to the `Bemd` function, so it only take one argument (`datamodel_c`) as required by `imap`.

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        compute_Bemd_partial = partial(compute_Bemd,
                                       riskA=riskA, riskB=riskB,
                                       synth_ppfA=synth_risk_ppfA, synth_ppfB=synth_risk_ppfB,
                                       Ldata=Ldata)

# %% [markdown]
# Define dictionaries into which we will accumulate the results of the $B^{\mathrm{EMD}}$ and $B_{\mathrm{conf}}$ calculations.

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        Bemd_results = {}
        Bconf_results = {}

# %% [markdown]
# - Set the iterator over parameter combinations (we need two identical ones)
# - Set up progress bar.
# - Determine the number of multiprocessing cores we will use.

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        try:
            N = len(data_models)
        except (TypeError, AttributeError):  # Typically TypeError, but AttributeError seems also plausible
            logger.info("Data model iterable has no length: it will not be possible to estimate the remaining computation time.")
            total = None
        else:
            total = N*len(c_list)
        progbar = tqdm(desc="Calib. experiments", total=total)
        ncores = psutil.cpu_count(logical=False)
        ncores = min(ncores, total, config.mp.max_cores)

# %% [markdown]
# Run the experiments. Since there are a lot of them, and they each take a few minutes, we use multiprocessing to run multiple of them at once.  

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        if ncores > 1:
            with mp.Pool(ncores) as pool:
                # Chunk size calculated following Pool's algorithm (See https://stackoverflow.com/questions/53751050/multiprocessing-understanding-logic-behind-chunksize/54813527#54813527)
                # (Naive approach would be total/ncores. This is most efficient if all taskels take the same time. Smaller chunks == more flexible job allocation, but more overhead)
                chunksize, extra = divmod(N, ncores*6)
                if extra:
                    chunksize += 1
                Bemd_it = pool.imap(compute_Bemd_partial,
                                    self.model_c_gen(Bemd_results, Bconf_results),
                                    chunksize=chunksize)
                for (data_model, c), Bemd_res in zip(                                     # NB: Both `models_c_gen` generators
                        self.model_c_gen(Bemd_results, Bconf_results),               # always yield the same tuples,
                        Bemd_it):                                                     # because we only update Bemd_results
                    progbar.update(1)        # Updating first more reliable w/ ssh    # after drawing from the second generator
                    Bemd_results[data_model, c] = Bemd_res
                    if data_model not in Bconf_results:
                        Bconf_results[data_model] = compute_Bconf(data_model, riskA, riskB, Linf)

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        else:
            Bemd_it = (compute_Bemd_partial(arg)
                       for arg in self.model_c_gen(Bemd_results, Bconf_results))
            for (data_model, c), Bemd_res in zip(                                     # NB: Both `model_c_gen` generators
                    self.model_c_gen(Bemd_results, Bconf_results),               # always yield the same tuples,
                    Bemd_it):                                                     # because we only update Bemd_results
                progbar.update(1)        # Updating first more reliable w/ ssh    # after drawing from the second generator
                Bemd_results[data_model, c] = Bemd_res
                if data_model not in Bconf_results:
                    Bconf_results[data_model] = compute_Bconf(data_model, riskA, riskB, Linf)

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        progbar.close()

# %% [markdown]
# #### Result format
#
# If we serialize the whole dict, most of the space is taken up by serializing the data_models in the keys. Not only is this wasteful – we can easily recreate them with `model_c_gen` – but it also makes deserializing the results quite slow.
# So instead we return just the values as a list, and provide an `unpack_result` method which reconstructs the result dictionary.

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        return dict(Bemd =list(Bemd_results.values()),
                    Bconf=list(Bconf_results.values()))

    # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
    def unpack_result(self, result: Calibrate.Outputs.result_type
                     ) -> UnpackedCalibrateResult:
        return UnpackedCalibrateResult(
            Bemd  = dict(zip(self.model_c_gen({},{}),
                             result.Bemd)),
            Bconf = dict(zip(self.model_gen(),
                             result.Bconf))
        )
