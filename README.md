# EMD falsification library

The EMD (empirical model discrepancy) criterion is used to compare models based on how well each describes the data.
It is described in [this publication](); it’s main features are:
- **Symmetric**: All models are treated the same. (There is no preferred null model.)
  A corollary is that the test works for any number of parameters.
- **Specific**: Models are compared for particular parameter sets. In particular, the different models may all be the same equations but with different parameters.
- **Dimension agnostic**: Models are compared based on their quantile function, which is always 1d. So the method scales well to high-dimensional problems.

## Problem requirements

The main requirement to be able to compute the EMD criterion are:

- Observation data.
- At least two models to compare.
- The ability to use the models to generate synthetic samples.

The models can take any form; they can even be blackbox deep neural networks.

## Installation

    pip install emd-falsify

## Usage

The short form for comparing two models is

```python
from emd_falsify import Bemd, make_empirical_risk

# Load data into `data`
# Define `modelA`, `modelB`
# Define `lossA`, `lossB` : functions
# Define `Lsynth` : int
# Define `c` : float

synth_ppfA = make_empirical_risk(lossA(modelA.generate(Lsynth)))
synth_ppfB = make_empirical_risk(lossB(modelB.generate(Lsynth)))
mixed_ppfA = make_empirical_risk(lossA(data))
mixed_ppfB = make_empirical_risk(lossB(data))

Bemd(mixed_ppfA, mixed_ppfB, synth_ppfA, synth_ppfB, c=c)
```

We also expose additional functions like the lower level `draw_R_samples`.
These may be more verbose, but especially for cases with multiple models to compare,
they may be more convenient.

Note that comparisons depend on choosing an appropriate value for `c`; a systematic way to do this is via a *calibration experiment*, as described in our publication.
This package provides `emd_falsify.tasks.Calibrate` to help run calibration experiments.

### Complete usage examples

The documentation contains a [simple example](https://alcrene.github.io/emd-falsify/src/emd_falsify/emd.html#test-sampling-of-expected-risk-r).
Moreover, all the [code for the paper’s figures] is available, in the form of Jupyter notebooks.
These are heavily commented with extra additional usage hints; they are highly recommended reading.


## Debugging

If computations are taking inordinately long, set the debug level to `DEBUG`:

    
    logging.getLogger("emd_falsify").setLevel("DEBUG")

This will print messages to your console reporting how much time each computation step is taking.

## Further work

The current implementation of the hierarchical beta process (used for sampling quantile paths) has seen quite a lot of testing for numerical stability, but little optimization effort. In particular it makes a lot of calls to functions in `scipy.optimize`, which makes the whole function quite slow: even with a relatively complicated data model like the [pyloric circuit](https://alcrene.github.io/pyloric-network-simulator/pyloric_simulator/prinz2004.html), drawing quantile paths can still take 10x longer than generating the data.

Substantial performance improvements to the sampling algorithm are almost certainly possible with dedicated computer science effort, which would proportionately reduce compute time for the whole EMD procedure.

The hierarchical beta process is also not the only possible process for stochastically generating quantile paths: it was chosen in part because it made proving self-consistency particularly simple. Other processes may provide other advantages, either with respect to statistical robustness or computation speed.
